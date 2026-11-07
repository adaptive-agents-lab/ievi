"""Explicit seed-zero Four Rooms manifest reconstruction for selected reruns."""
from __future__ import annotations
import json
from pathlib import Path
import re
import yaml

import planning_reproduction as planning
import selected_planning as selected

_run_baseline = planning.run_baseline
_certify_target = planning.certify_target
_active_cfg = None
_active_output = None


def certify(cfg, output):
    manifest = _certify_target(cfg, output)
    target = planning.read_json(manifest)
    reference_path = str(_active_cfg["deterministic_target_manifest"])
    prefix = re.search(r"-([0-9a-f]{12})/target_manifest\.json$", reference_path)
    if prefix is None or not target["true_mdp_fingerprint"].startswith(prefix.group(1)):
        raise ValueError("Reconstructed seed-zero MDP fingerprint does not match reference reference target identity")
    gamma_suffix = {0.9: "09", 0.99: "099"}.get(float(_active_cfg["env_params"]["gamma"]))
    if gamma_suffix is None:
        raise ValueError("Missing full-fingerprint reconstruction evidence for Four Rooms manifest")
    evidence = planning.CONFIGS / "target-evidence" / f"four-rooms-gamma{gamma_suffix}.json"
    saved = planning.read_json(evidence)
    if saved["mdp_fingerprint"] != target["true_mdp_fingerprint"]:
        raise ValueError("Independent target full MDP fingerprint differs from verified reconstruction evidence")
    matches = [{"path": planning.rel(evidence), "sha256": planning.sha(evidence), "mdp_fingerprint": saved["mdp_fingerprint"]}]
    planning.write_json(_active_output / "environment-variant-certification.json", {
        "reference_variant_manifest": _active_cfg["env_variants_manifest"],
        "reference_manifest_recovered": False, "reconstruction": "One explicitly fixed Four Rooms environment at source seed zero.",
        "evidence": ["seed0 in reference environment-variant manifest name", "reference target MDP SHA256 prefix", "full verified reconstruction MDP fingerprint"],
        "reference_target_identity_prefix": prefix.group(1), "true_mdp_fingerprint": target["true_mdp_fingerprint"],
        "full_fingerprint_evidence": matches, "source_seed": 0, "certified_before_solver": True})
    return manifest


def run(entry, output, cores):
    global _active_cfg, _active_output
    cfg = yaml.safe_load((planning.ROOT / entry["source_config"]).read_text(encoding="utf-8"))
    if cfg["env"] != "four_rooms" or not re.fullmatch(r"configs/env_variants/four_rooms15_seed0_(?:pe_)?gamma0(?:9|99)_target\.yaml", cfg.get("env_variants_manifest", "")):
        raise ValueError("This explicit reconstruction supports only the reference seed0 Four Rooms variants")
    _active_cfg, _active_output = cfg, output
    manifest_path = output / cfg["env_variants_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target_manifest = output / "inputs/target_manifest.json"
    payload = {"env_name": "four_rooms", "variants": [{"variant_id": "four_rooms15_seed0_reconstruction",
        "env_params": {**cfg["env_params"], "seed": 0}, "deterministic_target_manifest": str(target_manifest),
        "metadata": {"source_seed": 0, "reference_variant_manifest_recovered": False}}]}
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    planning.run_baseline = _run_baseline
    planning.certify_target = certify
    _run_baseline(entry, output, cores)


if __name__ == "__main__":
    planning.run_baseline = run
    selected.main()
