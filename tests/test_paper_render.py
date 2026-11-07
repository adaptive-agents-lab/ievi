"""Scientific and visual gates for the paper main panel."""
from pathlib import Path
import shutil

import numpy as np
import pytest

import reproduce
from scripts.reference_dataset import DEFAULT_DATASET
from scripts.render_main_two_panel import cliffwalk_statistics, render_figure
from scripts.reproduction_manifest import sha256
from studies.cliffwalk.study import DATA, load_panel_data


def test_current_raw_statistics_preserve_reference_bootstrap_and_seed_axes():
    panel = load_panel_data()
    means, intervals = cliffwalk_statistics(panel)
    assert np.array_equal(means, panel['errors'].mean(axis=-1))
    assert np.array_equal(intervals, panel['intervals'])
    bad = dict(panel, seeds=np.arange(60))
    with pytest.raises(ValueError, match='held-out seeds'):
        cliffwalk_statistics(bad)


@pytest.mark.skipif(shutil.which('pdftoppm') is None, reason='Poppler required for exact PDF raster verification')
def test_current_render_matches_reference_and_preserves_existing_outputs(tmp_path):
    stem = tmp_path / 'figure1'
    before = sha256(DATA / 'figure-reference.pdf')
    metadata = render_figure(stem)
    assert metadata['pixel_identical_to_reference_at_300dpi']
    assert metadata['planning_panel_width_points'] == 198
    assert metadata['planning_native_content_only']
    from pypdf import PdfReader
    text = PdfReader(stem.with_suffix('.pdf')).pages[0].extract_text()
    for label in ('(a)', '(b)', 'environment samples', 'Q-learning', 'OS-Dyna', 'IE-Dyna', '40k', '80k'):
        assert text.count(label) == 1
    assert metadata['displayed_methods'] == ['q', 'os', 'ie']
    assert metadata['raw_data_methods'] == ['q', 'os', 'ie']
    assert metadata['bootstrap_unit'] == 'whole seed trajectory'
    assert metadata['metric'] == 'pol|all|L1'
    assert before == sha256(DATA / 'figure-reference.pdf')
    original = sha256(stem.with_suffix('.pdf'))
    with pytest.raises(ValueError, match='existing figure-panel artifacts'):
        render_figure(stem)
    assert sha256(stem.with_suffix('.pdf')) == original


def test_top_level_cli_routes_only_paper_main(monkeypatch, tmp_path):
    calls = []
    class Result:
        returncode = 0
    monkeypatch.setattr(reproduce.subprocess, 'run', lambda command, **kwargs: calls.append(command) or Result())
    output = tmp_path / 'paper'
    assert reproduce.main(['render-paper', '--output-root', str(output)]) == 0
    assert calls[-1][1:] == [str(reproduce.ROOT / 'scripts/render_paper.py'),
                             '--dataset', str(DEFAULT_DATASET), '--output-root', str(output)]
    assert reproduce.main(['cliffwalk', 'run', '--output', str(tmp_path / 'run'), '--workers', '2']) == 0
    assert calls[-1][1:] == ['-m', 'studies.cliffwalk', 'run', '--output', str(tmp_path / 'run'), '--workers', '2']


def test_preserved_transition_claim_still_requires_every_condition():
    import copy
    import json
    from scripts.check_planning_reproduction import transition_claim
    reference = json.loads((Path(__file__).resolve().parents[1] /
                            'configs/scientific-claims/cliffwalk-pe-transition.json').read_text())
    effective = dict(reference['experiment'])
    effective['env_params'] = {'gamma': effective.pop('gamma')}
    rows = [{'lambda': lam, 'criterion': 'stable', 'algorithm': 'waterfall_besp', 'hit': hit}
            for lam, hit in ((.625, 10), (.675, 20), (.725, None), (.775, None))]
    assert transition_claim(rows, effective, reference) == (True, True)
    wrong_boundary = copy.deepcopy(rows)
    wrong_boundary[2]['hit'] = 1000
    assert transition_claim(wrong_boundary, effective, reference) == (True, False)
    assert transition_claim(rows[:-1], effective, reference) == (True, False)
    assert transition_claim(rows, dict(effective, max_queries=10000), reference) == (False, False)
    altered_excerpt = copy.deepcopy(reference)
    altered_excerpt['reference_source']['excerpt'] += ' altered'
    assert transition_claim(rows, effective, altered_excerpt) == (False, False)
