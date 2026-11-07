"""Bounded checks of portable scope, reference parameters, and safe outputs."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import planning_reproduction as reference
import selected_planning as selected
import selected_planning_variants as variants
from check_planning_reproduction import stable, first, practical, trace_complete, reference_coverage, failed_floor_stop
import yaml


class ReferenceReproductionTests(unittest.TestCase):
    def test_current_selected_scope_omits_stale_and_preserves_science(self):
        data = reference.read_json(ROOT / "configs/planning/selected/catalog.json")
        self.assertEqual(len(data["studies"]), 23)
        ids = {e["source_catalog_id"] for e in data["studies"]}
        self.assertTrue(ids.isdisjoint({"source-control-cliffwalk-gamma09", "source-control-cliffwalk-gamma099-dense", "source-control-cliffwalk-gamma099-dense-practical"}))
        original = {e["id"]: e for e in reference.studies()}
        for entry in data["studies"]:
            cfg = yaml.safe_load((ROOT / entry["source_config"]).read_text())
            old = yaml.safe_load((ROOT / original[entry["source_catalog_id"]]["source_config"]).read_text())
            self.assertTrue(set(cfg["lambdas"]).issubset(old["lambdas"]))
            self.assertEqual({k: v for k, v in cfg.items() if k != "lambdas"}, {k: v for k, v in old.items() if k != "lambdas"})
        self.assertEqual(next(e for e in data["studies"] if e["id"] == "selected-control-cliffwalk-gamma099")["selected_grid"], selected.CURRENT_19)
        self.assertEqual(len(next(e for e in data["studies"] if e["id"] == "selected-evaluation-cliffwalk-self-loop-gamma099-budget1000")["selected_grid"]), 21)

    def test_frozen_roots_and_nonempty_outputs_refused(self):
        for path in (ROOT, reference.PLANNING_ROOT, reference.RESTORATION_ROOT):
            with self.assertRaises(ValueError):
                reference.empty_output(path)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "user-data").write_text("keep")
            with self.assertRaises(ValueError):
                reference.empty_output(path)
            self.assertEqual((path / "user-data").read_text(), "keep")

    def test_strict_stable_hit_after_last_failure(self):
        self.assertEqual(first([1, 2, 4, 8], [.1, .001, .0009, .002], .001), 4)
        self.assertIsNone(stable([1, 2, 4, 8], [.1, .001, .0009, .002], .001))
        self.assertEqual(stable([1, 2, 4, 8], [.1, .001, .0009, .0008], .001), 4)
        self.assertIsNone(stable([1, 2], [.001, float("nan")], .001))

    def test_trace_requires_full_dense_observation_budget(self):
        cfg = {"problem_type": "control", "max_queries": 4}
        self.assertTrue(trace_complete({"queries": [1, 2, 3, 4]}, [.1, .01, .001, 0], "vi", cfg, ROOT))
        self.assertFalse(trace_complete({"queries": [1, 2]}, [.1, 0], "vi", cfg, ROOT))
        self.assertFalse(trace_complete({"queries": [1, 2, 3, 4]}, [.1, 0], "vi", cfg, ROOT))
        cfg["problem_type"] = "pe"
        self.assertTrue(trace_complete({"queries": [0, 1, 2, 3, 4]}, [1, .1, .01, .001, 0], "osvi", cfg, ROOT))

    def test_empty_practical_summary_cannot_pass(self):
        cfg = {"algorithms": ["waterfall"], "lambdas": [0], "row_seeds": [0], "problem_type": "control", "max_queries": 2}
        with patch("check_planning_reproduction.planning.read_json", side_effect=[cfg, [], []]), patch("check_planning_reproduction.check_target", return_value={"passed": True}):
            result = practical(ROOT)
        self.assertFalse(result["passed"])
        self.assertFalse(result["complete_selected_coverage"])

    def test_reference_coverage_cannot_shrink_with_missing_metadata(self):
        expected = {(0., 0, "vi"), (.5, 500, "vi")}
        keys = {(lam, "vi", .001, criterion) for lam in (0., .5) for criterion in ("first", "stable")}
        self.assertTrue(reference_coverage(dict.fromkeys(keys), list(keys), expected, [.001], True, 1))
        reduced = keys - {(.5, "vi", .001, "stable")}
        self.assertFalse(reference_coverage(dict.fromkeys(reduced), list(reduced), expected, [.001], True, 1))
        self.assertFalse(reference_coverage({}, [], expected, [.001], True, 1))

    def test_declared_failed_floor_stop_rejects_earlier_truncation(self):
        import numpy as np
        cfg = {"max_queries": 10, "threshold": .001, "algorithms": [{"name": "waterfall", "h_init": 100., "h_min": 1., "decay_factor": .1, "stall_patience": 3}]}
        trace = {"queries": [1, 2, 3], "residuals": [2., 2., 2.], "value_errors": [2., 2., 2.]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            np.savez(directory / "environment-target.npz", r=np.array([[1.]]))
            self.assertTrue(failed_floor_stop(trace, [1., 1., 1.], cfg, directory))
            truncated = {key: value[:2] for key, value in trace.items()}
            self.assertFalse(failed_floor_stop(truncated, [1., 1.], cfg, directory))
            self.assertFalse(failed_floor_stop(trace, [1., 1., 0.], cfg, directory))

    def test_variant_reconstruction_rejects_wrong_reference_fingerprint(self):
        cfg = yaml.safe_load((ROOT / "configs/planning/selected/selected-control-four-rooms-gamma09.yaml").read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            manifest = path / "target.json"
            reference.write_json(manifest, {"true_mdp_fingerprint": "0" * 64})
            with patch.object(variants, "_certify_target", return_value=manifest), patch.object(variants, "_active_cfg", cfg), patch.object(variants, "_active_output", path):
                with self.assertRaisesRegex(ValueError, "reference target identity"):
                    variants.certify(cfg, path)

    def test_registry_excludes_unpublished_sampled_comparison(self):
        tasks = reference.read_json(ROOT / "configs/tasks.d/planning.json")["tasks"]
        self.assertFalse(any(t["id"] == "sampled-control" for t in tasks))
        with self.assertRaisesRegex(ValueError, "outside the paper task registry"):
            reference.run_study("source-control-cliffwalk-gamma099-dense", ROOT)


if __name__ == "__main__":
    unittest.main()
