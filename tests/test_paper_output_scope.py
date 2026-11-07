"""Reported measurement builders preserve statistics without auxiliary charts."""
from pathlib import Path
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_horizon():
    path = ROOT / "methods/planning/benchmarks/src/ievi_experiments/horizon_scaling/run_horizon_scaling.py"
    spec = importlib.util.spec_from_file_location("paper_scope_horizon", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return "<NaN>"
    return value


class ReportedOutputScopeTests(unittest.TestCase):
    def assert_no_charts(self, directory):
        charts = [path for path in directory.rglob("*") if path.suffix.lower() in {".png", ".pdf", ".svg"}]
        self.assertEqual(charts, [])

    def test_horizon_measurement_tables_match_reference_without_auxiliary_charts(self):
        from scripts.reference_dataset import DEFAULT_DATASET, accessible, resolve_attempt
        with tempfile.TemporaryDirectory() as temporary:
            record, _ = resolve_attempt(DEFAULT_DATASET, "planning-horizon", Path(temporary) / "inputs")
            reference = accessible(record.parent / "horizon_scaling")
            records = pd.read_csv(reference / "data/raw/records.csv")
            timings = pd.read_csv(reference / "data/raw/timing_repetitions.csv")
            self.assertEqual((len(records), len(timings)), (1680, 8400))
            output = Path(temporary) / "aggregated"
            summary = load_horizon().aggregate_results(output, records, timings)
            self.assertEqual(summary["records"], 1680)
            for path in (reference / "data/derived").glob("*.csv"):
                expected = pd.read_csv(path)
                actual = pd.read_csv(output / "data/derived" / path.name)
                self.assertTrue(set(expected.columns).issubset(actual.columns), path.name)
                self.assertEqual(len(expected), len(actual), path.name)
                numeric = expected.select_dtypes(include=[np.number]).columns
                for column in expected.columns:
                    if column in numeric:
                        np.testing.assert_allclose(expected[column], actual[column], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f"{path.name}: {column}")
                    else:
                        self.assertEqual(expected[column].fillna("<NA>").tolist(), actual[column].fillna("<NA>").tolist())
            self.assert_no_charts(output)

    def test_restoration_common_tables_preserve_measurements_without_variant_charts(self):
        source = ROOT / "methods/incumbent-restoration/results/common-solver"
        expected = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            inputs = output / "inputs"
            shutil.copytree(source, inputs)
            artifacts = output / "tables"
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, str(ROOT / "methods/incumbent-restoration/common-solver/analyze.py"), "--archive", str(ROOT / "methods/planning"), "--results", str(inputs), "--artifacts", str(artifacts)]
            result = subprocess.run(command, text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads((inputs / "summary.json").read_text(encoding="utf-8"))
            # The independent restoration audit is produced by its separate verifier.
            # Every field produced by this analyzer must agree with the retained data.
            self.assertEqual(finite_json(actual), finite_json({key: expected[key] for key in actual}))
            self.assertEqual((actual["recomputed_runs"], actual["total_rows"]), (38, 95))
            for name in ("ARTIFACT-MANIFEST.json", "summary.json", "table-counts-rows.tex", "table-costs-rows.tex", "table-tolerance-ievi-rows.tex"):
                self.assertTrue((artifacts / name).is_file(), name)
            self.assert_no_charts(output)


if __name__ == "__main__":
    unittest.main()
