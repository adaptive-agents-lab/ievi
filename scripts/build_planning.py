"""Assemble and render content-verified recomputed runs in a new derivative directory.

The reference renderers are read-only. Common-solver analysis writes auxiliary
tables, so it receives a staged copy rather than the completed solver attempt.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduction_manifest import guard_output_path, identity, load_tasks, output_snapshot, runtime_metadata, sha256, utc_now, verified_record, write_json

PLANNING_ROOT = ROOT / "methods/planning"
RESTORATION_ROOT = ROOT / "methods/incumbent-restoration"


def validate_study_selection(study, accepted_dataset):
    if study == "common" and accepted_dataset is not None:
        raise ValueError("--dataset cannot be combined with --study common: the reference common analyzer includes reference baseline rows. Use render-paper for the recomputed primary/restoration comparison.")


def find_verified(tasks, run_root, task_id):
    digest, _ = identity(tasks[task_id], ROOT)
    for path in sorted((run_root / task_id).glob("*/execution.json"), reverse=True):
        if verified_record(path, digest):
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"No completed current content-verified attempt for {task_id} in {run_root}")


def stage(source, destination):
    if destination.exists():
        raise ValueError(f"Refusing existing staged destination: {destination}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {"source": str(source), "destination": str(destination), "files": output_snapshot(destination)}


def compatibility_wrapper(path, inputs, derived):
    """Bridge the frozen builder/renderer path contract using recomputed CSVs only."""
    frozen = PLANNING_ROOT / "tools/build_primary.py"
    renderer = PLANNING_ROOT / "tools/render_primary.py"
    transfer_report = path.parent / "recomputed-statistics-staging.json"
    source = f'''"""Generated path compatibility adapter; scientific source remains unchanged."""
import hashlib,json,runpy,shutil,subprocess,sys
from pathlib import Path
FROZEN=Path({str(frozen)!r})
RENDERER=Path({str(renderer)!r})
INPUTS=Path({str(inputs)!r})
DERIVED=Path({str(derived)!r})
REPORT=Path({str(transfer_report)!r})
original_run=subprocess.run
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def compatible_run(command,*args,**kwargs):
    if isinstance(command,(list,tuple)) and len(command)>1 and Path(str(command[1])).resolve()==RENDERER.resolve():
        transfers=[]
        for name in ['runtime-statistics.csv','speedup-statistics.csv']:
            source=DERIVED/'data'/name
            destination=INPUTS/name
            if not source.is_file():raise ValueError('Recomputed derived statistic missing: '+str(source))
            if destination.exists():raise ValueError('Refusing to replace staged statistic: '+str(destination))
            shutil.copy2(source,destination)
            assert digest(source)==digest(destination)
            transfers.append({{'source':str(source),'destination':str(destination),'sha256':digest(source),'origin':'newly computed from recomputed solver measurements'}})
        REPORT.write_text(json.dumps({{'transfers':transfers,'renderer_argv':list(command),'reference_measurements_copied':False}},indent=2)+'\\n',encoding='utf-8')
        command=[command[0],str(RENDERER),*command[2:]]
    return original_run(command,*args,**kwargs)
subprocess.run=compatible_run
sys.argv=[str(FROZEN),*sys.argv[1:]]
try:
    builder_source=FROZEN.read_text(encoding='utf-8')
    builder_source=builder_source.replace('uses up to four processes', 'uses '+str(json.loads((INPUTS/'hardware-and-protocol.json').read_text())['control_accuracy_workers'])+' process')
    builder_source=builder_source.replace('results/benchmarks/', 'inputs/')
    exec(compile(builder_source,str(FROZEN),'exec'),{{'__name__':'__main__','__file__':str(FROZEN)}})
finally:subprocess.run=original_run
'''
    path.write_text(source, encoding="utf-8")
    return {"path": str(path), "sha256": sha256(path), "scientific_builder": str(frozen),
            "scientific_builder_sha256": sha256(frozen), "transfer_report": str(transfer_report),
            "scope": "Stage two statistics CSVs newly computed by the unchanged builder where the unchanged renderer expects them."}


def export_final_figures(output, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Choose a new figure export directory; existing exports are preserved")
    destination.mkdir(parents=True, exist_ok=False)
    exported = {}
    for source in sorted((output / "derived").rglob("*")):
        if source.is_file():
            relative = source.relative_to(output / "derived")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            exported[relative.as_posix()] = {"sha256": sha256(target), "source_sha256": sha256(source), "source": str(source)}
    for name in ["planning-build-report.json", "recomputed-statistics-staging.json", "compatibility-build.py", "render.log"]:
        source = output / name
        if source.is_file():
            target = destination / "provenance" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            exported[target.relative_to(destination).as_posix()] = {"sha256": sha256(target), "source_sha256": sha256(source), "source": str(source)}
    write_json(destination / "EXPORT-MANIFEST.json", {"source_build_report_sha256": sha256(output / "planning-build-report.json"),
        "exporter_sha256": sha256(__file__), "raw_solver_outputs_included": False, "staged_raw_directories_included": False,
        "scope": "Recomputed E1-E4 figures, newly computed tables, evidence/protocol and exact build provenance only.", "files": exported})
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/recomputed")
    parser.add_argument("--dataset", type=Path, help="Use exact accepted reference attempts; verify archive hashes independent of current runtime")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--study", choices=["final", "common"], default="final")
    parser.add_argument("--export-root", type=Path, help="New directory for final figures/tables/provenance only; no staged raw data")
    args = parser.parse_args()
    validate_study_selection(args.study, args.dataset)
    run_root, output = args.run_root.resolve(), args.output_root.resolve()
    if output.exists():
        raise ValueError("Choose a new nonexistent output directory")
    guard_output_path(output, ROOT)
    if output.is_relative_to(run_root) or run_root.is_relative_to(output):
        raise ValueError("Choose a derivative destination separate from the solver attempt tree")
    tasks = load_tasks(ROOT)
    ids = ["planning-control", "planning-horizon", "planning-timing"] if args.study == "final" else ["restoration-common"]
    if args.dataset:
        from scripts.reference_dataset import resolve_attempt
        sources = {key: resolve_attempt(args.dataset, key) for key in ids}
    else:
        sources = {key: find_verified(tasks, run_root, key) for key in ids}
    renderer_paths = [PLANNING_ROOT / "tools/build_primary.py", PLANNING_ROOT / "tools/render_primary.py"] if args.study == "final" else [RESTORATION_ROOT / "common-solver/analyze.py", RESTORATION_ROOT / "common-solver/control_accuracy_common.py"]
    renderer_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in renderer_paths}
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "study": args.study, "status": "preparing", "started_at": utc_now(),
              "argv": sys.argv, "runtime": runtime_metadata(), "builder_sha256": sha256(__file__), "renderer_code_sha256": renderer_hashes,
              "accepted_dataset": str(args.dataset) if args.dataset else None,
              "source_attempts": {key: {"execution_record": str(path), "execution_record_sha256": sha256(path),
                                        "identity_sha256": data["identity_sha256"], "outputs": data["outputs"]}
                                  for key, (path, data) in sources.items()}, "staging": [], "commands": []}
    report = output / "planning-build-report.json"
    write_json(report, record)
    env = os.environ.copy()
    env.update({key: "1" for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]})
    env.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    try:
        if args.study == "final":
            inputs = output / "inputs"
            inputs.mkdir()
            for key, name in [("planning-control", "control_accuracy"), ("planning-horizon", "horizon_scaling"), ("planning-timing", "wallclock")]:
                record["staging"].append(stage(sources[key][0].parent / name, inputs / name))
            timing_source = inputs / "wallclock/data/raw/benchmark_runs.csv"
            shutil.copy2(timing_source, inputs / "solver-timing-measurements.csv")
            if args.dataset:
                protocol = sources["planning-control"][1]["protocol"]
                runtime = sources["planning-timing"][1]["protocol"]["runtime"]
                workers = int(protocol["control_accuracy_workers"])
                thread_count = int(protocol["threads_per_process"])
            else:
                runtime = sources["planning-timing"][1]["identity"]["runtime"]
                command = sources["planning-control"][1]["argv"]
                workers = int(command[command.index("--workers") + 1])
                thread_count = int(sources["planning-control"][1]["identity"]["task"]["env"]["OMP_NUM_THREADS"])
            hardware = {"cpu": runtime.get("processor") or None, "physical_cores": None,
                        "logical_processors": runtime.get("logical_processors"), "installed_ram_GiB": None,
                        "os": runtime.get("platform"), "python": runtime.get("python_version"), "packages": runtime.get("packages"),
                        "gpu_used": False, "control_accuracy_workers": workers, "threads_per_process": thread_count,
                        "provenance": "Scientific data protocol; unrecorded hardware measurements are null.",
                        "timing_source": "wallclock/data/raw/benchmark_runs.csv"}
            write_json(inputs / "hardware-and-protocol.json", hardware)
            wrapper = output / "compatibility-build.py"
            record["compatibility_adapter"] = compatibility_wrapper(wrapper, inputs, output / "derived")
            cmd = [sys.executable, str(wrapper), "--input-root", str(inputs), "--output-root", str(output / "derived")]
            expected = ["derived/BUILD-CHECKS.json", "derived/data/horizon-statistics.csv", "derived/data/runtime-statistics.csv", "derived/data/speedup-statistics.csv",
                        *[f"derived/figures/{name}.pdf" for name in ["E1-weighted-cost", "E2-solve-tolerance", "E3-horizon-scaling", "E4-wallclock-fidelity"]],
                        "recomputed-statistics-staging.json"]
        else:
            inputs = output / "inputs/common"
            record["staging"].append(stage(sources["restoration-common"][0].parent / "common", inputs))
            cmd = [sys.executable, str(RESTORATION_ROOT / "common-solver/analyze.py"), "--archive", str(PLANNING_ROOT),
                   "--results", str(inputs), "--artifacts", str(output / "artifacts")]
            expected = ["artifacts/ARTIFACT-MANIFEST.json", "artifacts/summary.json", "artifacts/table-counts-rows.tex", "artifacts/table-costs-rows.tex", "artifacts/table-tolerance-ievi-rows.tex"]
        record["status"] = "running"
        write_json(report, record)
        started = time.monotonic()
        with (output / "render.log").open("wb") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["commands"].append({"argv": cmd, "cwd": str(ROOT), "exit_code": result.returncode,
                                   "duration_seconds": time.monotonic() - started, "log": "render.log"})
        missing = [path for path in expected if not (output / path).is_file()]
        record["missing_expected_outputs"] = missing
        record["source_attempts_still_verified"] = all(resolve_attempt(args.dataset, key)[1]["identity_sha256"] == data["identity_sha256"] for key, (path, data) in sources.items()) if args.dataset else all(verified_record(path, data["identity_sha256"]) for path, data in sources.values())
        record["renderer_code_unchanged"] = renderer_hashes == {str(path.relative_to(ROOT)): sha256(path) for path in renderer_paths}
        record["status"] = "completed" if result.returncode == 0 and not missing and record["source_attempts_still_verified"] and record["renderer_code_unchanged"] else "failed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    record["finished_at"] = utc_now()
    record["output_files"] = {key: value for key, value in output_snapshot(output).items() if key != report.name}
    write_json(report, record)
    exported = export_final_figures(output, args.export_root) if args.export_root and args.study == "final" and record["status"] == "completed" else None
    print(json.dumps({"status": record["status"], "study": args.study, "report": str(report), "exported_figures": exported, "error": record.get("error")}, indent=2))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
