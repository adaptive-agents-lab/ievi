"""Run the declared planning experiments.

All generated inputs and outputs live below an explicitly chosen output root.
The release is read-only; reconstructed targets/replay have explicit certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
def extended(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


PLANNING_ROOT = extended(ROOT / "methods/planning")
IMPLEMENTATION_ROOT = PLANNING_ROOT / "implementation"
FIGURES = IMPLEMENTATION_ROOT / "experiments/paper_figures"
RESTORATION_ROOT = ROOT / "methods/incumbent-restoration"
CONFIGS = ROOT / "configs/planning"


def read_json(path):
    return json.loads(extended(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(extended(path).read_bytes()).hexdigest()


def rel(path):
    normalized = str(Path(path).resolve()).removeprefix("\\\\?\\")
    return Path(normalized).relative_to(ROOT).as_posix()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_runtime():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    sys.path.insert(0, str(IMPLEMENTATION_ROOT / "src"))


def studies():
    return read_json(CONFIGS / "catalog.json")["studies"]


def entry_for(ident):
    return next(item for item in studies() if item["id"] == ident)


def empty_output(output):
    output = Path(output).resolve()
    try:
        from scripts.reproduction_manifest import guard_output_path
    except ModuleNotFoundError:
        from reproduction_manifest import guard_output_path
    guard_output_path(output, ROOT)
    if output == ROOT or ROOT.is_relative_to(output) or output.is_relative_to(PLANNING_ROOT) or output.is_relative_to(RESTORATION_ROOT):
        raise ValueError("Output must be a separate derivative directory outside frozen source/reference roots")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def certify_target(cfg, output):
    """Use the recorded explicit reconstruction, never silently drop target input."""
    # --primary selects the primary method dispatcher.
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(RESTORATION_ROOT / "harness/run_practical.py"), "--primary"]
        helper = load_module("reference_target_reconstruction", RESTORATION_ROOT / "harness/run_practical.py")
    finally:
        sys.argv = original_argv
    import jax
    from besp.environments.jax_native import create_environment_jax_native
    import numpy as np
    params = dict(cfg.get("env_params", {}))
    gamma = float(params.pop("gamma", 0.99))
    mdp = create_environment_jax_native(cfg["env"], gamma=gamma, seed=int(cfg.get("seed", 0)), **params)
    target = helper.build_target(mdp, cfg, output)
    arrays = np.load(output / "environment-target.npz", allow_pickle=False)
    np.savez_compressed(output / "target-arrays.npz", **{key: arrays[key] for key in ("pi_star", "v_star_control", "v_pi_star", "p_pi", "r_pi")})
    manifest = {"target_id": target.target_id, "lane": target.lane, "env_name": target.env_name,
                "env_params": target.env_params, "gamma": target.gamma, "true_mdp_fingerprint": target.mdp_fingerprint,
                "certified": bool(target.certification["policy_optimal_to_float64_precision"]),
                "certification": target.certification, "artifacts": {"target_arrays_npz": "target-arrays.npz"}}
    write_json(output / "target_manifest.json", manifest)
    if not manifest["certified"]:
        raise ValueError("Reconstructed target failed certification; solver run refused")
    return output / "target_manifest.json"


def run_baseline(entry, output, cores):
    import yaml
    source = extended(ROOT / entry["source_config"])
    if sha(source) != entry["source_sha256"]:
        raise ValueError("Frozen source configuration hash changed")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    target_root = output / "inputs"
    target_root.mkdir()
    target_cfg = dict(cfg)
    target_cfg["reference_target_manifest"] = cfg.get("deterministic_target_manifest")
    manifest = certify_target(target_cfg, target_root)
    cfg["deterministic_target_manifest"] = str(manifest)
    config_path = output / "portable-config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    command = [sys.executable, str(IMPLEMENTATION_ROOT / "scripts/run_experiment.py"), "--config", str(config_path),
               "--platform", "cpu", "--cores", str(cores), "--output_dir", str(output / "solver")]
    write_json(output / "input-provenance.json", {"source": rel(source), "sha256": sha(source),
               "changes": {"deterministic_target_manifest": {"from": target_cfg.get("reference_target_manifest"), "to": str(manifest)}},
               "scientific_parameters_preserved": True, "reference_target_recovered": False, "argv": command})
    subprocess.run(command, check=True, cwd=output)


def run_study(ident, output, cores=1):
    registered = read_json(ROOT / "configs/tasks.d/planning.json")["tasks"]
    if ident not in {task["id"] for task in registered if any(arg.endswith("scripts/planning_reproduction.py") for arg in task["argv"])}:
        raise ValueError(f"Study is outside the paper task registry: {ident}")
    entry = entry_for(ident)
    output = empty_output(output)
    configure_runtime()
    write_json(output / "study-provenance.json", {**entry, "argv": sys.argv, "python": sys.version,
                                                 "adapter_sha256": sha(__file__), "started_at_unix": time.time()})
    family = entry["family"]
    if family == "baseline":
        run_baseline(entry, output, cores)
    elif family == "practical":
        run_id = Path(entry["source_config"]).stem
        subprocess.run([sys.executable, str(RESTORATION_ROOT / "harness/run_practical.py"), run_id, "--primary",
                        "--output", str(output / "solver")], cwd=ROOT, check=True)
    elif family == "planning-profile":
        verifier = PLANNING_ROOT / "cliffwalk-planning"
        subprocess.run([sys.executable, str(verifier / "run_verification.py"), "--output", str(output / "solver")], cwd=verifier, check=True)
        subprocess.run([sys.executable, str(verifier / "check_verification.py"), str(output / "solver")], cwd=verifier, check=True)
    else:
        raise ValueError(f"Cannot run unavailable study: {ident}")
    write_json(output / "completed.json", {"id": ident, "completed": True,
                                           "completed_at_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "run"))
    parser.add_argument("--study")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()
    if args.action == "list":
        registered = {task["id"] for task in read_json(ROOT / "configs/tasks.d/planning.json")["tasks"] if any(arg.endswith("scripts/planning_reproduction.py") for arg in task["argv"])}
        for item in studies():
            if item["id"] not in registered:
                continue
            print(f"{item['id']}\t{item['family']}\t{item['description']}")
    else:
        if not args.study or not args.output_root:
            parser.error("--study and --output-root are required for run")
        run_study(args.study, args.output_root, args.cores)


if __name__ == "__main__":
    main()
