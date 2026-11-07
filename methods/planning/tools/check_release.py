"""Check optional horizon aggregation and retained-data invariants; executes no solvers."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import tempfile
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_horizon_scaling', ROOT / 'benchmarks/src/ievi_experiments/horizon_scaling/run_horizon_scaling.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def check(input_root, report_path):
    raw = input_root / 'data/raw'
    records = pd.read_csv(raw / 'records.csv')
    timings = pd.read_csv(raw / 'timing_repetitions.csv')
    original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.csv')}
    results = []
    with tempfile.TemporaryDirectory(prefix='ievi-aggregation-') as tmp:
        for horizons, timed in [([10], False), ([640], True), ([10, 20], True), ([10, 20, 40, 80, 160, 320, 640], True)]:
            module.H_GRID = np.asarray(horizons, dtype=np.int64)
            selected = records[records.H.isin(horizons)].copy()
            selected_times = timings[timings.H.isin(horizons)].copy()
            if not timed:
                selected.loc[:, ['wall_time_median_seconds', 'wall_time_mad_seconds']] = np.nan
                selected_times = pd.DataFrame()
            dest = Path(tmp) / ('-'.join(map(str, horizons)) + ('-timed' if timed else '-skip'))
            result = module.aggregate_results(dest, selected, selected_times, timing_requested=timed)
            slopes = pd.read_csv(dest / 'data/derived/slope_summary.csv')
            assert result['endpoint_H'] == max(horizons)
            assert (dest / f'data/derived/endpoint_H{max(horizons)}.csv').is_file()
            assert (max(horizons) == 640) == ('endpoint_H640' in result)
            if len(horizons) == 1:
                assert slopes.slope.isna().all() and slopes.ci_low.isna().all() and slopes.ci_high.isna().all()
                assert slopes.status.eq('insufficient_distinct_horizons').all()
            if max(horizons) < 80:
                assert slopes.loc[slopes.grid == 'H_ge_80', 'slope'].isna().all()
            if not timed:
                assert result['timing_status'] == 'not_requested'
                assert pd.read_csv(dest / 'data/raw/timing_repetitions.csv').empty
            else:
                assert result['timing_status'] == 'available'
            assert not list(dest.rglob('*.png')) and not list(dest.rglob('*.pdf'))
            if len(horizons) == 7:
                expected = pd.read_csv(input_root / 'data/derived/slope_summary.csv')
                keys = ['schedule', 'quantity', 'grid']
                merged = slopes.merge(expected, on=keys, suffixes=('_new', '_old'))
                assert len(merged) == len(expected)
                for key in ['slope', 'ci_low', 'ci_high']:
                    np.testing.assert_allclose(merged[key + '_new'], merged[key + '_old'], rtol=1e-13, atol=1e-13)
                expected_summary = pd.read_csv(input_root / 'data/derived/summary_by_H.csv')
                observed_summary = pd.read_csv(dest / 'data/derived/summary_by_H.csv')
                pd.testing.assert_frame_equal(expected_summary, observed_summary, check_exact=False, rtol=1e-13, atol=1e-13)
            results.append(dict(horizons=horizons, timing_requested=timed, passed=True))
        # Requested timing with no observations must fail clearly, rather than silently skip.
        bad = records[records.H == 10].copy()
        bad['wall_time_median_seconds'] = np.nan
        try:
            module.aggregate_results(Path(tmp) / 'invalid-timing', bad, pd.DataFrame(), timing_requested=True)
        except ValueError as error:
            assert 'Timing was requested' in str(error)
        else:
            raise AssertionError('Invalid requested timing was silently accepted')
        assert np.isnan(module.loglog_slope([10, 10], [1, 2]))
    assert original_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.csv')}
    report = dict(passed=True, checks=results, missing_requested_timing_rejected=True,
                  duplicate_horizon_slope_unavailable=True, raw_csv_hashes_unchanged=True,
                  solver_experiments_run=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True, help='Recorded published horizon-study root with data/raw and data/derived.')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    check(args.input_root.resolve(), args.report)
