"""Read frozen results; write a comparison table and its provenance."""
import csv
import hashlib
import json
import math
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
R = CODE / 'methods/planning'
S = CODE / 'methods/incumbent-restoration'
OUT = CODE / 'reference-assets/method-comparison'
inputs = [S / 'results/common-solver/all-95-row-comparison.csv', R / 'results/benchmarks/control_accuracy/raw/control_accuracy_summary.csv', S / 'results/common-solver/control_accuracy_ievi_summary.csv']
def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def key(r):
    return (float(r['lambda']), float(r['delta_solve']))
comparison = [r for r in read(inputs[0]) if r['algorithm'] == 'IEVI_Waterfall_common_VI']
primary = {key(r): r for r in read(inputs[1]) if r['algorithm'] == 'IEVI_Waterfall_common_VI'}
secondary = {key(r): r for r in read(inputs[2]) if r['algorithm'] == 'IEVI_Waterfall_common_VI'}
assert len(comparison) == len(primary) == len(secondary) == 38
grid = sorted({float(r['lambda']) for r in comparison})
assert len(grid) == 19
indexed = {key(r): r for r in comparison}
for r in comparison:
    for suffix, original in [('old', primary[key(r)]), ('new', secondary[key(r)])]:
        for field in ['stable_N_P', 'stable_N_Phat']:
            a, b = r[f'{field}_{suffix}'], original[field]
            assert a == b or (a and b and float(a) == float(b)), (key(r), field, suffix, a, b)
def value(s):
    if s == '' or not math.isfinite(float(s)):
        return 'no hit'
    n = float(s)
    assert n.is_integer()
    return str(int(n))
header = ['lambda', 'primary_N_P', 'primary_N_Phat_delta_1e-8', 'primary_N_Phat_delta_1e-2', 'restoration_N_P', 'restoration_N_Phat_delta_1e-8', 'restoration_N_Phat_delta_1e-2']
rows = []
for lam in grid:
    tight, loose = indexed[(lam, 1e-8)], indexed[(lam, .01)]
    for suffix in ['old', 'new']:
        assert value(tight[f'stable_N_P_{suffix}']) == value(loose[f'stable_N_P_{suffix}'])
    rows.append([format(lam, '.15g')] + [value(tight['stable_N_P_old']), value(tight['stable_N_Phat_old']), value(loose['stable_N_Phat_old']), value(tight['stable_N_P_new']), value(tight['stable_N_Phat_new']), value(loose['stable_N_Phat_new'])])
OUT.mkdir(parents=True, exist_ok=True)
targets = [OUT / 'common-solver-comparison.csv', OUT / 'common-solver-comparison-rows.tex', OUT / 'PROVENANCE.json']
assert not any(p.exists() for p in targets), 'Refusing to overwrite table/provenance'
with targets[0].open('w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(header); w.writerows(rows)
targets[1].write_text(''.join(' & '.join(r) + ' \\\\\n' for r in rows), encoding='utf-8')
def record(p):
    return {'path': p.relative_to(CODE).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
provenance = {'purpose': 'Stable-hit operator-cost comparison of common-solver IE-VI with and without incumbent restoration across 19 corruption levels.', 'inputs': [record(p) for p in inputs], 'generator': record(Path(__file__).resolve()), 'outputs': [record(p) for p in targets[:2]], 'checks': {'comparison_rows': 38, 'lambda_settings': 19, 'source_summary_fields_match': True, 'true_count_equality_across_tolerances_per_variant': True, 'no_hit_encoding': 'no hit; never converted to a finite budget', 'no_hit_cells': sum(x == 'no hit' for r in rows for x in r)}, 'columns': header, 'metric': 'stable policy-error hit; common-solver actual full operator applications including initial true application', 'delta_solve_order': [1e-8, .01], 'evidence': 'The primary rows use the supplied common-solver reference measurements; the restoration rows use measurements produced by the incumbent-restoration harness. Implementation identity qualifications are documented with each input.'}
targets[2].write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'rows': len(rows), 'outputs': [str(p) for p in targets], 'no_hit_cells': provenance['checks']['no_hit_cells']}, indent=2))
