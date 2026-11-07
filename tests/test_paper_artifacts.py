"""Integrity gates for paper artifact generation, without executing a solver."""
from pathlib import Path
import json
import hashlib

import pandas as pd
import pytest

from scripts.paper_tables import format_cell, generate
from scripts.build_planning import find_verified, validate_study_selection
from scripts.reproduction_manifest import guard_output_path

ROOT = Path(__file__).resolve().parents[1]


def test_accepted_common_rejects_reference_mixed_baselines():
    with pytest.raises(ValueError, match='Use render-paper'):
        validate_study_selection('common', Path('DATASET.json'))
    validate_study_selection('common', None)
    validate_study_selection('final', Path('DATASET.json'))


def test_format_preserves_actual_printed_precision_and_no_hits():
    scalar = {'kind':'number', 'precision':[0], 'layout':'scalar'}
    assert format_cell(dict(scalar, group_thousands=True), [1868]) == r'\(1{,}868\)'
    assert format_cell(dict(scalar, precision=[1]), [.1]) == r'\(0.1\)'
    assert format_cell(dict(scalar, precision=[5,5], layout='mean_se'), [4.543219, .076231]) == r'\(4.54322 \pm 0.07623\)'
    assert format_cell(dict(scalar, precision=[5], layout='scientific', exponent=-6), [9.98242e-6]) == r'\(9.98242\times10^{-6}\)'
    assert format_cell({'kind':'missing', 'display':'---'}, [None]) == '---'
    assert format_cell(scalar, [-1e-15]) == r'\(0\)'
    assert format_cell(dict(scalar, precision=[1,2,2], layout='estimate_interval'), [2.1, 1.23, 2.98]) == r'\(2.1 [1.23,\,2.98]\)'
    assert format_cell(dict(scalar, precision=[2,2], layout='interval'), [-.51, -.46]) == r'\([-0.51,\,-0.46]\)'


def test_table_reference_contains_structured_data_without_manuscript_source():
    reference = json.loads((ROOT/'docs/coverage/coverage.json').read_text())
    assert 'selection_authority' not in reference
    tables = reference['table_inventory']
    assert len(tables) == 11
    assert all(table['classification'] == 'empirical' for table in tables)
    for item in reference['figure_inventory'] + tables:
        assert not {'source', 'line', 'expected_table_latex'} & item.keys()
    for table in tables:
        for block in table['blocks']:
            assert all(len(row) == len(block['columns']) for row in block['rows'])
            assert all(cell['kind'] in {'number','missing','text'} for row in block['rows'] for cell in row)
    assert '\\' not in json.dumps(reference)


def test_coverage_identifies_published_figures_tables_and_designated_reference():
    reference = json.loads((ROOT/'docs/coverage/coverage.json').read_text())
    figures = reference['figure_inventory']
    tables = reference['table_inventory']
    assert {item['paper_id'] for item in figures} == {'Figure 1', 'Figure B.1'} | {f'Figure N.{i}' for i in range(1,18)}
    assert {item['paper_id'] for item in tables} == {f'Table N.{i}' for i in range(1,12)}
    assert len({item['paper_label'] for item in figures}) == len(figures) == 19
    assert len({item['paper_label'] for item in tables}) == len(tables) == 11
    assert next(item for item in figures if item['paper_id']=='Figure N.16')['paper_label'] == 'fig:final-horizon-scaling'
    assert [item['paper_label'] for item in tables[:2]] == ['tab:final-control-counts','tab:final-control-costs']
    assert [item['paper_label'] for item in tables[4:7]] == ['tab:final-horizon-fixed','tab:final-horizon-growing','tab:final-horizon-slopes']
    main = next(item for item in figures if item['paper_id']=='Figure 1')
    assert main['family_ids'] == ['cliffwalk_control','cliffwalk_rl']
    assert main['source_tasks'] == ['planning-05_threshold','selected-control-cliffwalk-gamma099']
    assert main['study']['methods'] == ['q','os','ie']
    assert all('asset_sha256' not in item for item in figures)
    for asset in [reference['paper'], main['reference_asset']]:
        assert hashlib.sha256((ROOT/asset['path']).read_bytes()).hexdigest() == asset['sha256']
    dataset = json.loads((ROOT/'datasets/reference/planning/DATASET.json').read_text())
    assert {task for item in figures+tables for task in item['source_tasks']} <= dataset['tasks'].keys()


def test_recomputed_lookup_keeps_current_identity_gate(tmp_path, monkeypatch):
    attempt = tmp_path / 'task' / 'attempt'
    attempt.mkdir(parents=True)
    record = attempt / 'execution.json'
    record.write_text('{}')
    calls = []
    monkeypatch.setattr('scripts.build_planning.identity', lambda task, root: ('current-identity', {}))
    monkeypatch.setattr('scripts.build_planning.verified_record', lambda path, fingerprint: calls.append(fingerprint) or False)
    with pytest.raises(ValueError, match='current content-verified'):
        find_verified({'task':{}}, tmp_path, 'task')
    assert calls == ['current-identity']


def test_no_output_can_enter_latex_or_frozen_inputs(tmp_path):
    with pytest.raises(ValueError):
        guard_output_path(ROOT.parent / 'latex' / 'new-figures', ROOT)
    with pytest.raises(ValueError):
        guard_output_path(ROOT / 'methods' / 'new-figures', ROOT)


def test_all_tables_recomputed_and_restoration_uses_recomputed_baselines(tmp_path, monkeypatch):
    dataset=ROOT/'datasets/reference/planning/DATASET.json'
    if not dataset.exists():
        pytest.skip('Accepted dataset not installed')
    from scripts import reference_dataset
    resolve_attempt = reference_dataset.resolve_attempt
    monkeypatch.setattr(reference_dataset, 'resolve_attempt',
                        lambda source, task: resolve_attempt(source, task, cache_root=tmp_path/'cache'))
    output=tmp_path/'tables'
    report=generate(dataset,output)
    assert sum(report['counts'].values()) == 628
    assert not (set(report['counts']) - {'match','new_timing_measurement'})
    assert sum(cell['comparison_kind']=='timing' for cell in report['cells']) == 18
    references = {table['paper_label']:table for table in json.loads((ROOT/'docs/coverage/coverage.json').read_text())['table_inventory']}
    for cell in report['cells']:
        spec = references[cell['table']]['blocks'][cell['block']]['rows'][cell['row']-1][cell['column']]
        expected = spec['values'] if spec['kind']=='number' else [None]
        assert cell['paper_values'] == expected
        actual = cell['recomputed_values']
        if expected == [None]:
            assert cell['absolute_tolerances'] == []
            matches = actual == [None]
        else:
            exponent = spec.get('exponent',0) if spec['layout']=='scientific' else 0
            tolerances = [0.0 if cell['comparison_kind']=='exact' else 1e-12 if value==0 else 10.0**(exponent-precision)/2
                          for value,precision in zip(expected,spec['precision'])]
            assert cell['absolute_tolerances'] == tolerances
            matches = len(actual) == len(expected) and all(
                value is not None and abs(value-reference) <= tolerance + 1e-12*max(abs(reference),1)
                for value,reference,tolerance in zip(actual,expected,tolerances))
        assert cell['matches_printed_value'] == matches
        assert cell['status'] == ('match' if matches else 'new_timing_measurement' if cell['comparison_kind']=='timing' else 'difference')
        if cell['comparison_kind'] != 'timing':
            assert matches
    assert report['generated_cells']==628 and report['all_generated_cells_match_aggregation']
    assert len(report['generated_tables'])==11
    assert all(x['generated_matches_aggregation'] for x in report['cells'])
    assert all(x['matches_recomputed_saved'] for x in report['slope_independent_recomputation'])
    primary=pd.read_csv(output/'primary-common-summary.csv')
    restoration=pd.read_csv(output/'restoration-common-summary.csv')
    baseline=primary[primary.algorithm!='IEVI_Waterfall_common_VI']
    pd.testing.assert_frame_equal(baseline.reset_index(drop=True),restoration[restoration.algorithm!='IEVI_Waterfall_common_VI'][primary.columns].reset_index(drop=True),check_dtype=False)
    assert len(baseline)==57 and len(restoration)==95
    assert len(restoration[restoration.algorithm=='IEVI_Waterfall_common_VI'])==38
    for label in ['final-horizon-fixed','final-horizon-growing']:
        assert (output/(label+'-block-0.tex')).exists()
        assert (output/(label+'-block-1.tex')).exists()
