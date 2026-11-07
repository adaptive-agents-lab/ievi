"""Relocation/tampering and scientific comparisons retain strict integrity."""
import hashlib, io, json, shutil, tarfile, tempfile, unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from scripts.reference_dataset import resolve_attempt, verify_dataset
from scripts.compare_results import compare_pair, compare_accepted_task, accepted_specs, timing_measurement_invariants
from scripts.reproduction_manifest import guard_output_path, identity, sha256, verified_record

class AcceptedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.dataset = self.root / "package/datasets/accepted/fixture/DATASET.json"
        self.dataset.parent.mkdir(parents=True)
        self.payload = {"result.csv": b"lambda,seed,N_P,residual,seconds\n0.5,7,10000000000000000,1e-12,1\n"}
        self.outputs = self.inventory()
        record = {"task_id": "fixture", "identity_sha256": "reference", "status": "completed", "exit_code": 0, "outputs": self.outputs,
                  "identity": {"runtime": {"python_executable": "C:/old/python.exe"}}, "output_directory": "C:/old/attempt"}
        self.payload["DATA.json"] = json.dumps(record).encode()
        self.archive = self.dataset.parent / "fixture.tar.gz"; self.write_archive(self.payload)
        files = self.inventory()
        task = {"task_id": "fixture", "attempt": "original", "category": "test", "identity_sha256": "reference", "record_sha256": files["DATA.json"]["sha256"],
                "files": files, "outputs": self.outputs, "archive": {"path": self.archive.name, "sha256": sha256(self.archive), "bytes": self.archive.stat().st_size}}
        index = self.dataset.parents[3] / "verification/data-index.json"; index.parent.mkdir(parents=True); index.write_text(json.dumps({"schema_version": 2, "tasks": [{"task_id":"fixture", "record_sha256":task["record_sha256"], "identity_sha256":task["identity_sha256"]}]}))
        self.manifest = {"selection_source":"verification/data-index.json", "selection_source_sha256":sha256(index), "schema_version": 2, "dataset_id": "fixture", "selection_count": 1, "tasks": {"fixture": task}}; self.save_manifest()
    def tearDown(self): self.temp.cleanup()
    def inventory(self): return {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)} for name, value in self.payload.items()}
    def write_archive(self, payload):
        with tarfile.open(self.archive, "w:gz") as tar:
            for name, raw in payload.items():
                info = tarfile.TarInfo(name); info.size = len(raw); tar.addfile(info, io.BytesIO(raw))
    def save_manifest(self): self.dataset.write_text(json.dumps(self.manifest))
    def test_relocation_and_roundtrip_independent_runtime(self):
        relocated = self.root / "elsewhere"; shutil.copytree(self.dataset.parents[3], relocated); manifest = relocated / "datasets/accepted/fixture/DATASET.json"
        self.assertTrue(verify_dataset(manifest)["passed"])
        path, record = resolve_attempt(manifest, "fixture", self.root / "cache")
        self.assertEqual((path.parent / "result.csv").read_bytes(), self.payload["result.csv"])
        self.assertEqual(record["identity"]["runtime"]["python_executable"], "C:/old/python.exe")
        self.assertFalse(verified_record(path, "current-runtime"))
        self.assertEqual(resolve_attempt(manifest, "fixture", self.root / "cache")[0], path)
    def test_archive_tamper_and_conflicting_cache_refused(self):
        path, _ = resolve_attempt(self.dataset, "fixture", self.root / "cache"); (path.parent / "result.csv").write_text("user data")
        with self.assertRaisesRegex(ValueError, "Conflicting"): resolve_attempt(self.dataset, "fixture", self.root / "cache")
        self.assertEqual((path.parent / "result.csv").read_text(), "user data")
        self.archive.write_bytes(self.archive.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "tampered"): verify_dataset(self.dataset)
    def test_tar_traversal_refused(self):
        self.write_archive({**self.payload, "../escape.txt": b"escape"})
        self.manifest["tasks"]["fixture"]["archive"].update(sha256=sha256(self.archive), bytes=self.archive.stat().st_size); self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe"): resolve_attempt(self.dataset, "fixture", self.root / "cache")
        self.assertFalse((self.root / "escape.txt").exists())
    def test_discrete_arrays_exact_continuous_tolerant(self):
        actual, reference = self.root / "a.npz", self.root / "b.npz"
        np.savez(actual, seeds=np.array([7]), values=np.array([1. + 1e-12])); np.savez(reference, seeds=np.array([7]), values=np.array([1.]))
        self.assertTrue(compare_pair(actual, reference, {"mode": "arrays"})["passed"])
        self.assertFalse(compare_pair(actual, reference, {"mode": "arrays", "exact_arrays": True})["passed"])
        np.savez(actual, seeds=np.array([8]), values=np.array([1.]))
        self.assertFalse(compare_pair(actual, reference, {"mode": "arrays"})["passed"])
    def test_accepted_compare_seconds_vary_counter_exact(self):
        output = self.root / "new"; output.mkdir()
        task = {"id": "fixture", "comparator": {"pairs": [{"actual": "result.csv", "mode": "csv", "key_columns": ["lambda", "seed"]}]}}
        (output / "result.csv").write_text(self.payload["result.csv"].decode().replace(",1\n", ",9\n"))
        self.assertTrue(compare_accepted_task(task, self.root, output, self.dataset)["passed"])
        (output / "result.csv").write_text(self.payload["result.csv"].decode().replace("10000000000000000", "10000000000000001"))
        self.assertFalse(compare_accepted_task(task, self.root, output, self.dataset)["passed"])
    def test_frozen_and_manuscript_output_guards(self):
        root = self.root / "project/code"
        for output in (root, root / "methods/new", root / "datasets/new", root.parent / "latex/new"):
            with self.assertRaises(ValueError): guard_output_path(output, root)
        self.assertEqual(guard_output_path(root / "runs/new", root), (root / "runs/new").resolve())

    def test_all_accounting_field_names_require_exact_csv_values(self):
        names = ["attempt", "outer", "direct_solves", "stable_direct_solves", "stable_hit_true_applications", "verification_true_applications", "restorations", "incumbent_attempt", "accepted_attempt", "stable_hit_surrogate_applications"]
        actual, reference = self.root / "a.csv", self.root / "b.csv"
        for name in names:
            with self.subTest(name=name):
                actual.write_text(name + "\n49.000000001\n"); reference.write_text(name + "\n49\n")
                self.assertFalse(compare_pair(actual, reference, {"mode": "csv"})["passed"])
    def test_timing_continuous_roundoff_and_exact_protocol(self):
        root = self.root / "timing"; root.mkdir()
        actual, reference = self.root / "a.csv", root / "b.csv"
        reference.write_text("final_relative_true_residual,sup_error_to_reference,outer_iterations\n1e-6,1e-8,49\n")
        task = {"id": "planning-timing", "comparator": {"pairs": [{"actual": "b.csv", "mode": "csv", "exact_columns": ["final_relative_true_residual", "sup_error_to_reference", "outer_iterations"]}]}}
        spec = accepted_specs(task, root)[0]
        actual.write_text("final_relative_true_residual,sup_error_to_reference,outer_iterations\n1.000000001e-6,1.000000001e-8,49\n")
        self.assertTrue(compare_pair(actual, reference, spec)["passed"])
        actual.write_text("final_relative_true_residual,sup_error_to_reference,outer_iterations\n1e-6,1e-8,49.000000001\n")
        self.assertFalse(compare_pair(actual, reference, spec)["passed"])

    def test_timing_setup_and_summary_invariants_reject_negative_nan(self):
        rows = [{"grid_side": "100", "method": "VI", "runtime_seconds": "2", "solver_seconds": "2", "true_setup_seconds": "0.2", "surrogate_setup_seconds": "0", "setup_plus_solver_seconds": "2.2"}]
        pilot = [{"runtime_seconds": "0.1"}]
        summary = [{"grid_side": "100", "method": "VI", "median_runtime_seconds": "2", "runtime_q25": "2", "runtime_q75": "2"}]
        self.assertTrue(all(timing_measurement_invariants(rows, pilot, summary).values()))
        for name in ("runtime_seconds", "solver_seconds", "true_setup_seconds", "surrogate_setup_seconds", "setup_plus_solver_seconds"):
            for value in ("-1", "nan"):
                with self.subTest(name=name, value=value):
                    changed = [dict(rows[0], **{name: value})]
                    self.assertFalse(all(timing_measurement_invariants(changed, pilot, summary).values()))
        self.assertFalse(all(timing_measurement_invariants(rows, [{"runtime_seconds": "nan"}], summary).values()))
        self.assertFalse(all(timing_measurement_invariants(rows, pilot, [dict(summary[0], runtime_q25="-1")]).values()))
        self.assertFalse(all(timing_measurement_invariants([dict(rows[0], setup_plus_solver_seconds="3")], pilot, summary).values()))

    def test_resume_identity_changes_with_runtime_and_location(self):
        root = self.root / "runtime-fixture"; (root / "scripts").mkdir(parents=True)
        (root / "reproduce.py").write_text("# fixture"); (root / "scripts/reproduction_manifest.py").write_text("# fixture")
        task = {"id": "fixture", "category": "test", "argv": ["{python}", "-c", "pass"], "cwd": "{root}", "inputs": [], "code": [], "env": {}, "expected_outputs": ["ok"], "depends_on": []}
        with patch("scripts.reproduction_manifest.runtime_metadata", return_value={"platform": "old"}): old, _ = identity(task, root)
        with patch("scripts.reproduction_manifest.runtime_metadata", return_value={"platform": "new"}): new, _ = identity(task, root)
        self.assertNotEqual(old, new)
        moved = self.root / "moved"; shutil.copytree(root, moved)
        with patch("scripts.reproduction_manifest.runtime_metadata", return_value={"platform": "old"}): relocated, _ = identity(task, moved)
        self.assertNotEqual(old, relocated)

    def test_relocated_target_manifest_metadata_keeps_scientific_checks(self):
        reference_root = self.root / "reference"; reference = reference_root / "study/solver/raw/control_traces.json"; reference.parent.mkdir(parents=True)
        baseline = {"seed": 7, "queries": [1, 2], "errors": [0.2, 0.1], "N": {"metadata": {"deterministic_target_manifest": "C:/original/target_manifest.json"}}}
        reference.write_text(json.dumps(baseline)); task = {"id": "selected-planning-fixture"}; spec = accepted_specs(task, reference_root)[0]
        actual = self.root / "actual.json"
        changed = json.loads(json.dumps(baseline)); changed["N"]["metadata"]["deterministic_target_manifest"] = "/relocated/target_manifest.json"; actual.write_text(json.dumps(changed))
        self.assertTrue(compare_pair(actual, reference, spec)["passed"])
        changed["seed"] = 7.000000001; actual.write_text(json.dumps(changed))
        self.assertFalse(compare_pair(actual, reference, spec)["passed"])
        changed["seed"] = 7; changed["errors"][0] = 0.3; actual.write_text(json.dumps(changed))
        self.assertFalse(compare_pair(actual, reference, spec)["passed"])
