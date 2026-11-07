"""Input integrity, complete reported-scope selection and curve comparison checks."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies.cliffwalk import study


def test_complete_inputs_and_seed_ranges():
    receipt = study.verify()
    assert receipt["tuning_files"] == 44
    assert receipt["heldout_files"] == 3
    assert receipt["nonfinite_tuning_files_retained"] == 0
    assert receipt["selection_rows"] == 3
    for row in json.loads((study.DATA / "tuning-manifest.json").read_text()):
        assert row["source"] == "results/cliffwalk/tuning/" + row["name"]
        assert study.sha256(study.DATA / row["source"]) == row["sha256"]


def test_full_reported_tuning_grid_and_ranking(tmp_path):
    output = tmp_path / "selection"
    receipt = study.select(output)
    assert receipt["candidate_rows"] == 1584
    assert receipt["method_candidate_counts"] == {"q": 36, "os": 108, "ie": 1440}
    assert receipt["selection_uses_heldout"] is False
    with np.load(output / "tuning-candidate-scores.npz") as scores:
        original = scores["area40"]
        penalized = scores["penalized_area40"]
        bad = ~np.isfinite(original)
        assert not bad.any()
        np.testing.assert_array_equal(scores["nonfinite_penalized"], bad)
        assert np.isposinf(penalized[bad]).all()
        np.testing.assert_array_equal(original[~bad], penalized[~bad])
        assert len(set(scores["source_file"].tolist())) == 44
        assert original.max() > 97
        assert scores["selected"].sum() == 3
        for method, count in receipt["method_candidate_counts"].items():
            np.testing.assert_array_equal(np.sort(scores["rank"][scores["method"] == method]), np.arange(1, count+1))
        with np.load(study.DATA / "tuning-candidate-scores.npz") as reference:
            for key in scores.files:
                np.testing.assert_array_equal(scores[key], reference[key])
    with pytest.raises(FileExistsError):
        study.select(output)


def test_panel_metric_axis_and_fixed_primary_settings():
    panel = study.load_panel_data()
    np.testing.assert_array_equal(panel["seeds"], np.arange(250, 310))
    np.testing.assert_array_equal(panel["nominal_samples"], np.arange(0, 80001, 2000))
    assert panel["actual_samples"][1, -1] == 79980
    settings = study.selected_settings()
    assert (settings["q"]["sched"], settings["q"]["B"]) == ("v:n^-0.8", 10)
    assert (settings["os"]["sched"], settings["os"]["B"]) == ("c0.01", 30)
    assert (settings["ie"]["sched"], settings["ie"]["B"], settings["ie"]["h"]) == ("v:3/(3+n)", 10, .03)
    np.testing.assert_allclose([study.area40(e).mean() for e in panel["errors"][:3]],
                               [6.208971515496572, 16.810612147057743, 12.762761556527908], rtol=0, atol=1e-12)
    np.testing.assert_allclose(study.area40(np.arange(41)[:, None] * 2), [20])


def test_comparison_detects_changed_primary_observation(tmp_path):
    setting = study.selected_settings()["q"]
    source = study.CW / "heldout" / study.raw_name(setting)
    destination = tmp_path / "runtime/results/cliffwalk/heldout" / source.name
    destination.parent.mkdir(parents=True)
    shutil.copy2(source, destination)
    assert study.compare(tmp_path, ["q"])["all_array_equal"]
    with np.load(destination, allow_pickle=False) as raw:
        arrays = {k: raw[k].copy() for k in raw.files}
    combo = list(zip(arrays["scheds"].tolist(), arrays["hs"].tolist())).index((setting["sched"], setting["h"]))
    cell = arrays["cell_names"].tolist().index(study.CELL)
    arrays["cells"][10, combo, 3, cell] += 1
    np.savez_compressed(destination, **arrays)
    receipt = study.compare(tmp_path, ["q"])
    assert not receipt["all_array_equal"]
    assert not receipt["all_within_tolerance"]
    assert receipt["methods"][0]["max_absolute_error_percent"] == 1


def test_heldout_exports_and_grouped_execution_metadata():
    assert study.METHODS == ("q", "os", "ie")
    groups = json.loads((study.CW / "execution-groups.json").read_text())
    assert len(groups) == 3 and sum(len(job[-1]) for job in groups) == 13
    settings = study.selected_settings()
    for job in groups:
        setting = settings[job[0]]
        assert [setting["sched"], setting["h"]] in job[-1]
        with np.load(study.CW / "heldout" / study.raw_name(setting), allow_pickle=False) as raw:
            assert set(raw.files) == {"cells", "used", "seeds", "cell_names", "scheds", "hs"}
            assert raw["cells"].shape == (41, 1, 60, 1)
            assert raw["cell_names"].tolist() == [study.CELL]
            assert raw["scheds"].tolist() == [setting["sched"]]
            assert raw["hs"].tolist() == [setting["h"]]
