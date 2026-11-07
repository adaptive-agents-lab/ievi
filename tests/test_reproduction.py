"""Small integrity tests; these never run scientific solvers."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import reproduce
from scripts.compare_results import compare_pair, compare_task
from scripts.reference_dataset import DEFAULT_DATASET, resolve_attempt
from scripts.reproduction_manifest import identity, output_snapshot, sha256, verified_record
from scripts.build_planning import stage
from scripts.check_scientific_outputs import compare_arrays, read_csv, timing_invariants


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        for path in ["reproduce.py", "scripts/reproduction_manifest.py"]:
            (self.root / path).write_text("# fixture\n")
        (self.root / "input.txt").write_text("original")
        self.output_root = self.root / "runs"
        self.task = {"id": "fixture", "category": "test", "argv": ["{python}", "-c", "from pathlib import Path; Path(r'{output}/result.txt').write_text('ok')"],
                     "cwd": "{root}", "inputs": ["input.txt"], "code": [], "env": {}, "expected_outputs": ["result.txt"], "depends_on": []}

    def tearDown(self):
        self.temporary.cleanup()

    def test_resume_requires_identical_inputs_and_output_hashes(self):
        with patch.object(reproduce, "ROOT", self.root):
            self.assertTrue(reproduce.run_task(self.task, self.output_root, False))
            record = next(self.output_root.rglob("execution.json"))
            digest, _ = identity(self.task, self.root)
            self.assertTrue(verified_record(record, digest))
            self.assertTrue(reproduce.run_task(self.task, self.output_root, True))
            self.assertEqual(len(list(self.output_root.rglob("execution.json"))), 1)
            (record.parent / "result.txt").write_text("tampered")
            self.assertFalse(verified_record(record, digest))
            (self.root / "input.txt").write_text("modified")
            changed, _ = identity(self.task, self.root)
            self.assertNotEqual(digest, changed)

    def test_nonzero_exit_and_missing_outputs_never_complete(self):
        with patch.object(reproduce, "ROOT", self.root):
            self.task["argv"] = ["{python}", "-c", "raise SystemExit(7)"]
            self.assertFalse(reproduce.run_task(self.task, self.output_root, False))
            record = next(self.output_root.rglob("execution.json"))
            data = json.loads(record.read_text())
            self.assertEqual(data["exit_code"], 7)
            self.assertEqual(data["status"], "failed")
            self.assertFalse(verified_record(record))

    def test_csv_scientific_comparison_separates_timing(self):
        actual, reference = self.root / "actual.csv", self.root / "reference.csv"
        actual.write_text("id,calls,seconds\na,20,2\n")
        reference.write_text("id,calls,seconds\na,20,1\n")
        spec = {"mode": "csv", "key_columns": ["id"], "ignore_columns": ["seconds"], "report_columns": ["seconds"]}
        result = compare_pair(actual, reference, spec)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reported_variation"][0]["actual_to_reference_ratio"], 2)
        actual.write_text("id,calls,seconds\na,21,2\n")
        self.assertFalse(compare_pair(actual, reference, spec)["passed"])

    def test_dependency_cycle_rejected(self):
        tasks = {"a": {"id": "a", "category": "test", "depends_on": ["b"]}, "b": {"id": "b", "category": "test", "depends_on": ["a"]}}
        with self.assertRaisesRegex(ValueError, "cycle"):
            reproduce.selected_tasks(tasks, "all")

    def test_large_discrete_counts_never_use_relative_tolerance(self):
        actual, reference = self.root / "actual.csv", self.root / "reference.csv"
        actual.write_text("id,N_P,residual\na,10000000000000001,1e-12\n")
        reference.write_text("id,N_P,residual\na,10000000000000000,2e-12\n")
        result = compare_pair(actual, reference, {"mode": "csv", "key_columns": ["id"]})
        self.assertFalse(result["passed"])
        self.assertEqual(result["differences"][0]["column"], "N_P")

    def test_archived_inputs_require_exact_array_values(self):
        actual, reference = self.root / "actual.npz", self.root / "reference.npz"
        arrays = {"surrogate_models": np.zeros((30, 2, 7, 2, 2)), "seeds": np.arange(30),
                  "H_grid": np.arange(7), "schedules": np.array(["fixed80", "growing"])}
        np.savez_compressed(actual, **arrays)
        np.savez_compressed(reference, **arrays)
        self.assertTrue(compare_arrays(actual, reference)["passed"])
        arrays["surrogate_models"][0, 0, 0, 0, 0] = 1e-14
        np.savez_compressed(actual, **arrays)
        self.assertFalse(compare_arrays(actual, reference)["passed"])

    def reference_fixture(self):
        attempt = self.root / "runs/reference-fixture/attempt"
        study = attempt / "study"
        study.mkdir(parents=True)
        (study / "completed.json").write_text(json.dumps({"id": "reference-fixture", "completed": True, "prepare_only": False}))
        (self.root / "scripts/check_planning_reproduction.py").write_text("# checker fixture\n")
        return attempt, {"id": "reference-fixture", "scientific_checker":True}

    def test_reference_missing_scientific_report_cannot_pass(self):
        attempt, task = self.reference_fixture()
        checker = SimpleNamespace(check=lambda *args, **kwargs: {"passed": True, "rows": []})
        with patch("scripts.compare_results.load_reference_checker", return_value=(checker, sha256(self.root / "scripts/check_planning_reproduction.py"))):
            result = compare_task(task, self.root, attempt)
        self.assertFalse(result["passed"])
        self.assertIn("did not produce", result["comparisons"][0]["error"])

    def test_reference_scientific_failure_remains_failure_without_output_mutation(self):
        attempt, task = self.reference_fixture()
        before = output_snapshot(attempt)
        def check(study, report_root):
            report = {"passed": False, "rows": [{"algorithm": "osvi", "reference_hit": 19, "hit": 20, "matches": False}]}
            report_root.mkdir(parents=True)
            (report_root / "scientific-comparison.json").write_text(json.dumps(report))
            return report
        with patch("scripts.compare_results.load_reference_checker", return_value=(SimpleNamespace(check=check), sha256(self.root / "scripts/check_planning_reproduction.py"))):
            result = compare_task(task, self.root, attempt)
        self.assertFalse(result["passed"])
        self.assertEqual(result["comparisons"][0]["report"]["rows"][0]["hit"], 20)
        self.assertEqual(before, output_snapshot(attempt))

    def test_timing_cartesian_coverage_rejects_duplicate_missing_pair(self):
        attempt, _ = resolve_attempt(DEFAULT_DATASET, "planning-timing")
        reference = attempt.parent / "wallclock/data"
        rows = read_csv(reference / "raw/benchmark_runs.csv")
        pilot = read_csv(reference / "raw/calibration_runs.csv")
        selection = json.loads((reference / "derived/calibration_selection.json").read_text())
        self.assertTrue(all(timing_invariants(rows, pilot, selection).values()))
        duplicated = [rows[0], *rows[1:-1], rows[0]]
        self.assertEqual(len(duplicated), 50)
        self.assertFalse(timing_invariants(duplicated, pilot, selection)["exact_50_unique_method_repetition_keys"])

    def test_derivative_staging_preserves_completed_source(self):
        source = self.root / "solver-output"
        source.mkdir()
        (source / "summary.csv").write_text("id,count\na,49\n")
        destination = self.root / "derived-inputs"
        evidence = stage(source, destination)
        self.assertEqual((source / "summary.csv").read_bytes(), (destination / "summary.csv").read_bytes())
        self.assertIn("summary.csv", evidence["files"])
        (destination / "summary.csv").write_text("auxiliary analysis may change staging")
        self.assertEqual((source / "summary.csv").read_text(), "id,count\na,49\n")
        with self.assertRaisesRegex(ValueError, "existing"):
            stage(source, destination)


if __name__ == "__main__":
    unittest.main()
