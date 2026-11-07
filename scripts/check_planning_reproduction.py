"""Check recomputed reference numerical records against their stated reference evidence."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

try:
    from scripts import planning_reproduction as planning
except ModuleNotFoundError:
    import planning_reproduction as planning


def stable(queries, errors, threshold):
    failures = [i for i, value in enumerate(errors) if not math.isfinite(float(value)) or float(value) >= threshold]
    index = failures[-1] + 1 if failures else 0
    return int(queries[index]) if index < len(queries) else None


def first(queries, errors, threshold):
    return next((int(q) for q, e in zip(queries, errors) if math.isfinite(float(e)) and float(e) < threshold), None)


def transition_claim(rows, effective, reference):
    """Check the preserved prose boundary without requiring a full manuscript."""
    experiment = {'problem_type': 'pe', 'env': 'cliffwalk', 'gamma': .99,
                  'error_mode': 'self_loop', 'max_queries': 1000, 'threshold': .001,
                  'algorithm': 'waterfall_besp'}
    source = reference['reference_source']
    phrase = r'\AIEVI{}\) fails from \(\lambda=0.725\)'
    reference_valid = (reference.get('schema_version') == 1
                       and reference.get('source_catalog_id') == 'source-evaluation-cliffwalk-self-loop-gamma099-boundary'
                       and reference.get('claim_id') == 'cliffwalk-pe-self-loop-gamma099-transition'
                       and reference.get('experiment') == experiment
                       and reference.get('finite_stable_hit_lambdas') == [.625, .675]
                       and reference.get('no_stable_hit_lambdas') == [.725, .775]
                       and source.get('sha256') == '7685e34e6b3deb72edf7b12b4254e2ba72d02c9f7ec867d325d6aaf7c6d8897e'
                       and hashlib.sha256(source['excerpt'].encode('utf-8')).hexdigest() == source.get('excerpt_sha256')
                       and phrase in source['excerpt'])
    contract = (reference_valid and effective['problem_type'] == 'pe' and effective['env'] == 'cliffwalk'
                and float(effective['env_params']['gamma']) == .99 and effective['error_mode'] == 'self_loop'
                and int(effective['max_queries']) == 1000 and float(effective['threshold']) == .001)
    stable_rows = [row for row in rows if row['criterion'] == 'stable' and row['algorithm'] == 'waterfall_besp']
    by_lambda = {float(row['lambda']): row for row in stable_rows}
    assertion = (len(stable_rows) == 4 and set(by_lambda) == {.625, .675, .725, .775}
                 and all(by_lambda[lam]['hit'] is not None for lam in (.625, .675))
                 and all(by_lambda[lam]['hit'] is None for lam in (.725, .775)))
    return bool(contract), bool(contract and assertion)


def reference_coverage(references, compared, expected_traces, thresholds, control, seeds):
    required = {(lam, alg, float(threshold), criterion)
                for lam, _, alg in expected_traces for threshold in thresholds
                for criterion in (("stable", "first") if control else ("stable",))}
    available = {key for key in references if (key[0], key[1]) in {(lam, alg) for lam, _, alg in expected_traces}}
    return bool(required and required <= set(references) and set(compared) == available and len(compared) == len(available) * seeds)


def trace_complete(trace, metric, algorithm, cfg, target_directory, allow_terminal_inference=False):
    """Validate observation coverage, including AISP's declared early stopping."""
    queries = trace["queries"]
    valid = bool(queries) and len(queries) == len(metric)
    start = 0 if cfg["problem_type"] == "pe" else 1
    valid = valid and all(math.isfinite(float(q)) and int(q) == q and start <= q <= int(cfg["max_queries"]) for q in queries)
    valid = valid and all(a < b for a, b in zip(queries, queries[1:]))
    if algorithm != "aisp":
        dense = queries == list(range(start, int(cfg["max_queries"]) + 1))
        terminal = allow_terminal_inference and algorithm == "waterfall" and cfg["problem_type"] == "control" and failed_floor_stop(trace, metric, cfg, target_directory)
        return bool(valid and (dense or terminal))
    import numpy as np
    gamma = float(cfg["env_params"]["gamma"])
    cost = int(math.ceil(math.log2(8 / (1 - gamma)))) + 1
    valid = valid and queries == [1 + cost * i for i in range(1, len(queries) + 1)]
    parameters = next(a for a in cfg["algorithms"] if isinstance(a, dict) and a["name"] == "aisp")
    with np.load(target_directory / "environment-target.npz", allow_pickle=False) as arrays:
        initial_residual = float(np.max(np.abs(np.max(arrays["r"], axis=1))))
    tolerance = float(parameters.get("target_epsilon", 1e-12)) * (1 - gamma) / (1 + gamma) * initial_residual
    residuals = trace.get("residuals", [])
    valid = valid and len(residuals) == len(queries)
    stopped = bool(residuals) and (float(residuals[-1]) <= tolerance or queries[-1] + cost > int(cfg["max_queries"]))
    return bool(valid and stopped)


def failed_floor_stop(trace, metric, cfg, target_directory):
    """Recognize the reference Control waterfall's terminal rejected floor attempt.

    Rejected attempts preserve both residual and reference-value error. This is
    a source-derived interpretation, since the old trace has no rejection flag.
    Restrict it to terminal no-stable-hit traces and require the first inferred floor failure at
    the final dense observation; an arbitrary earlier truncation is rejected.
    """
    import numpy as np
    queries = trace["queries"]
    if not queries or queries[-1] >= int(cfg["max_queries"]) or queries != list(range(1, len(queries) + 1)):
        return False
    if not metric or any(not math.isfinite(float(e)) for e in metric) or float(metric[-1]) < float(cfg["threshold"]):
        return False
    residuals, values = trace.get("residuals", []), trace.get("value_errors", [])
    if len(residuals) != len(queries) or len(values) != len(queries):
        return False
    parameters = next(a for a in cfg["algorithms"] if isinstance(a, dict) and a["name"] == "waterfall")
    h, floor = float(parameters["h_init"]), float(parameters["h_min"])
    decay, patience = float(parameters["decay_factor"]), int(parameters["stall_patience"])
    with np.load(target_directory / "environment-target.npz", allow_pickle=False) as arrays:
        previous = float(np.max(np.abs(np.max(arrays["r"], axis=1))))
    best, previous_value, stalled = previous, 1., 0
    floor_failures = []
    for index, (residual, value) in enumerate(zip(residuals, values, strict=True)):
        if not math.isfinite(float(residual)) or not math.isfinite(float(value)):
            return False
        rejected = residual == previous and value == previous_value
        if rejected:
            h, stalled = max(floor, h * decay), 0
        elif residual < best:
            best, stalled = residual, 0
        else:
            stalled += 1
            if stalled >= patience:
                h, stalled = max(floor, h * decay), 0
        if rejected and h <= floor:
            floor_failures.append(index + 1)
        previous, previous_value = residual, value
    return floor_failures == [len(queries)]


def planning_profile(output):
    checks = planning.read_json(output / "solver/verification-checks.json")
    required_true = ("pass", "all_stable_hits_equal_to_reference", "all_query_indices_equal_to_reference",
                     "actual_query_counters_match_every_recorded_trajectory", "all_actual_run_counts_within_1000",
                     "source_ast_equivalent_after_removing_only_declared_hooks_and_budget_preflight")
    passed = all(checks.get(key) is True for key in required_true) and checks.get("algorithm_grid_points") == 57
    rows = planning.read_json(output / "solver/all-19-comparison.json")
    grid = [0, .1, .2, .3, .4, .5, .6, .625, .65, .675, .7, .725, .75, .775, .8, .85, .9, .95, 1]
    coverage = len(rows) == 19 and sorted(float(row["lambda"]) for row in rows) == grid
    passed = passed and coverage and all(row.get(alg + suffix) is True for row in rows for alg in ("vi", "osvi", "aisp") for suffix in ("_stable_hit_equal", "_query_indices_equal"))
    return {"kind": "planning-stable-hit-comparison", "passed": passed, "rows": rows, "checks": checks,
            "complete_selected_coverage": coverage,
            "scope": "All 57 stable hit results, every recorded query index, actual query counters, independent offline policy metric.",
            "reference_build_identity_claimed": False,
            "full_policy_trajectories_match": checks.get("all_policy_traces_equal_to_reference")}


def two_state(output):
    actual = output / "package/data/two_state_compact_inset.raw_data.json"
    reference = planning.FIGURES / "two_state_bellman_flow_stiffness/data/two_state_compact_inset.raw_data.json"
    left, right = planning.read_json(actual), planning.read_json(reference)
    differences = []
    def visit(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append({"path": path + "/" + key, "error": "missing key"})
                else:
                    visit(a[key], b[key], path + "/" + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append({"path": path, "lengths": [len(a), len(b)]})
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(i))
        elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
                differences.append({"path": path, "actual": a, "reference": b})
        elif a != b:
            differences.append({"path": path, "actual": a, "reference": b})
    visit(left, right)
    return {"kind": "two-state-analytic", "passed": not differences, "rows": differences[:100],
            "difference_count": len(differences), "scope": "Every computed trajectory, model parameter and declared illustrative display override.",
            "reference": planning.rel(reference), "reference_sha256": planning.sha(reference)}


def practical(output):
    cfg = planning.read_json(output / "solver/effective-config.json")
    actual = planning.read_json(output / "solver/summary.json")
    rows = []
    reference_rows = cfg.get("reference_rows", [])
    algorithms = [a if isinstance(a, str) else a["name"] for a in cfg["algorithms"]]
    thresholds = [.01, .001, .0001] if cfg.get("run_id") == "05_threshold" else [.001]
    expected_keys = {(float(lam), int(seed), algorithm, float(threshold))
                     for lam, seed in zip(cfg["lambdas"], cfg["row_seeds"], strict=True)
                     for algorithm in algorithms for threshold in thresholds}
    actual_keys = [(float(r["lambda"]), int(r["seed"]), r["algorithm"], float(r["threshold"])) for r in actual]
    coverage = bool(expected_keys) and set(actual_keys) == expected_keys and len(actual_keys) == len(expected_keys)
    raw = planning.read_json(output / "solver/raw-traces.json")
    trace_map = {(float(r["lambda"]), int(r["seed"]), algorithm): trace for r in raw for algorithm, trace in r["traces"].items()}
    expected_traces = {key[:3] for key in expected_keys}
    complete_traces = set(trace_map) == expected_traces and sum(len(r["traces"]) for r in raw) == len(expected_traces)
    short = any(key[2] == "waterfall" and trace["queries"] and trace["queries"][-1] < int(cfg["max_queries"]) for key, trace in trace_map.items())
    execution_verified = False
    if short:
        try:
            from scripts.reproduction_manifest import verified_record
        except ModuleNotFoundError:
            from reproduction_manifest import verified_record
        execution_path = output.parent / "execution.json"
        execution_verified = verified_record(execution_path) and planning.read_json(execution_path).get("inputs_and_code_unchanged") is True
    for key, trace in trace_map.items():
        metric = trace["policy_value_errors"] if cfg["problem_type"] == "control" else trace["errors"]
        complete_traces = complete_traces and trace_complete(trace, metric, key[2], cfg, output / "solver", allow_terminal_inference=execution_verified)
    terminal_stops = [{"lambda": key[0], "seed": key[1], "algorithm": key[2], "last_query": trace["queries"][-1]}
                      for key, trace in trace_map.items() if key[2] == "waterfall" and trace["queries"] and trace["queries"][-1] < int(cfg["max_queries"])]
    summary_bound = True
    for current in actual:
        trace = trace_map.get((float(current["lambda"]), int(current["seed"]), current["algorithm"]))
        if trace is None:
            summary_bound = False
        else:
            metric = trace["policy_value_errors"] if cfg["problem_type"] == "control" else trace["errors"]
            summary_bound = summary_bound and current["first_hit"] == first(trace["queries"], metric, float(current["threshold"])) and current["stable_hit"] == stable(trace["queries"], metric, float(current["threshold"]))
        candidates = [r for r in reference_rows if float(r["lambda"]) == float(current["lambda"])]
        reference = candidates[0] if candidates else None
        for criterion in ("stable", "first"):
            selected_reference = reference
            if "reference_panels" in cfg:
                selected_reference = None
                flat = [item for panel in cfg["reference_panels"] for item in (panel if isinstance(panel, list) else [panel])]
                for panel in flat:
                    if float(panel.get("threshold", -1)) == float(current["threshold"]) and panel.get("criterion") == criterion:
                        index = [float(x) for x in panel["lambdas"]].index(float(current["lambda"]))
                        selected_reference = {criterion + "_hit_query": panel["aievi_hits"][index]}
                        break
            required = criterion == "stable" or cfg["problem_type"] == "control"
            available = selected_reference is not None and any(key in selected_reference for key in ("aievi_" + criterion + "_hit", criterion + "_hit_query"))
            if not available:
                rows.append({**current, "criterion": criterion, "reference_available": False, "reference_required": required, "matches": None})
                continue
            hit = selected_reference.get("aievi_" + criterion + "_hit", selected_reference.get(criterion + "_hit_query"))
            if str(selected_reference.get(criterion + "_hit_dnf", "False")).lower() == "true" or (hit is not None and int(hit) > int(cfg["max_queries"])):
                hit = None
            else:
                hit = int(hit) if hit is not None else None
            rows.append({**current, "criterion": criterion, "reference_available": True, "reference_required": required, "reference_hit": hit,
                         "matches": current[criterion + "_hit"] == hit})
    target = check_target(output, cfg, practical=True)
    required_count = len(expected_keys) * (2 if cfg["problem_type"] == "control" else 1)
    required_rows = [row for row in rows if row["reference_required"] or row["reference_available"]]
    return {"kind": "primary-practical", "passed": bool(coverage and complete_traces and summary_bound and target["passed"] and len(rows) == len(expected_keys) * 2 and len(required_rows) >= required_count and all(r["matches"] is True for r in required_rows)), "rows": rows,
            "complete_selected_coverage": coverage, "expected_metric_comparisons": required_count,
            "complete_recorded_observation_budget": complete_traces, "summary_matches_raw_traces": summary_bound,
            "declared_terminal_stops": terminal_stops,
            "terminal_inference_content_verified_execution": execution_verified if terminal_stops else None,
            "terminal_stop_evidence": "Short Control waterfall traces are checked against the source-declared failed-floor protocol using reference residual/value-error state. Rejection flags were not saved; successful plateaus can be ambiguous, so this source-derived interpretation also relies on content-verified process completion and the unchanged solver stop contract. It does not claim unobserved later queries." if terminal_stops else None,
            "target_fingerprint_check": target,
            "scope": "Every requested lambda/seed/algorithm and threshold's strict first and stable hit; compare every reference reference metric. PE metadata retains stable hits only, so its recomputed first hits are informative.",
            "reference_target_recovered": False, "reference_build_identity_claimed": False}


def check_target(output, cfg, practical=False):
    """Rehash actual recomputed MDP arrays and compare their declared source identity."""
    import numpy as np
    directory = output / ("solver" if practical else "inputs")
    cert = planning.read_json(directory / "target-certification.json")
    with np.load(directory / "environment-target.npz", allow_pickle=False) as arrays:
        p, r = (np.ascontiguousarray(arrays[key], dtype=np.float64) for key in ("p", "r"))
        gamma = np.asarray([float(arrays["gamma"])], dtype=np.float64)
    digest = hashlib.sha256()
    for array in (p, r):
        digest.update(str(array.shape).encode("utf-8")); digest.update(array.tobytes())
    digest.update(gamma.tobytes())
    actual = digest.hexdigest()
    if practical:
        reference = cfg.get("reference_target_manifest")
    else:
        provenance = planning.read_json(output / "input-provenance.json")
        reference = provenance.get("changes", {}).get("deterministic_target_manifest", {}).get("from")
    match = re.search(r"-([0-9a-f]{12})/target_manifest\.json$", str(reference))
    prefix = match.group(1) if match else None
    finite_certified = cert.get("policy_optimal_to_float64_precision") is True and math.isfinite(float(cert["value_error_upper_bound"])) and float(cert["value_error_upper_bound"]) < 1e-7
    return {"passed": bool(prefix is not None and actual.startswith(prefix) and actual == cert.get("mdp_fingerprint") and finite_certified),
            "actual_mdp_sha256": actual, "certificate_mdp_sha256": cert.get("mdp_fingerprint"),
            "recomputed_arrays_match_certificate": actual == cert.get("mdp_fingerprint"),
            "reference_identity_prefix": prefix, "reference_identity_available": prefix is not None,
            "reference_prefix_matches": actual.startswith(prefix) if prefix else None,
            "recomputed_target_certified": finite_certified, "reference_target_arrays_recovered": False,
            "evidence_scope": "Reference target identifier retains only a 12-hex MDP fingerprint prefix; full recomputed array fingerprint and independent numerical certificate are checked. Reference target arrays/build identity remain unavailable."}


def numerical_summary(output):
    files = list((output / "solver/raw").glob("*_traces.json"))
    if len(files) != 1:
        raise ValueError(f"Expected one numerical trace file under {output}")
    cfg = planning.read_json(output / "study-provenance.json")
    import yaml
    effective = yaml.safe_load((output / "portable-config.yaml").read_text(encoding="utf-8"))
    source_id = cfg.get("source_catalog_id", cfg["id"])
    register = planning.read_json(planning.CONFIGS / "reference-metadata.json")
    selected = next((item for item in register["studies"] if item["id"] == source_id), None)
    metadata = None
    references = {}
    duplicate_reference = False
    def reference_value(key, value):
        nonlocal duplicate_reference
        duplicate_reference = duplicate_reference or key in references
        references[key] = value
    if selected is not None:
        evidence = selected["active_metadata"]
        metadata = planning.read_json(planning.ROOT / evidence["path"])
        selector = evidence["selector"]
        if selector == "panels[*][*]":
            for panels in metadata["panels"]:
                for panel in panels:
                    for index, lam in enumerate(panel["lambdas"]):
                        for alg, value in (("vi", panel["vi_hit"]), ("osvi", panel["osvi_hits"][index]),
                                           ("aisp", panel["aievi_hits"][index])):
                            reference_value((float(lam), alg, float(panel["threshold"]), panel["criterion"]), value)
        else:
            if selector == "rows":
                metadata_rows = metadata["rows"]
            else:
                match = re.fullmatch(r"(datasets|panels)\[(\d+)\]\.rows", selector)
                if match is None:
                    raise ValueError(f"Unsupported explicit metadata selector: {selector}")
                metadata_rows = metadata[match.group(1)][int(match.group(2))]["rows"]
            for reference in metadata_rows:
                for alg, prefix in (("vi", "vi"), ("osvi", "osvi"), ("aisp", "aievi")):
                    for criterion in ("stable", "first"):
                        key = prefix + "_" + criterion + "_hit"
                        if key in reference:
                            reference_value((float(reference["lambda"]), alg, float(effective["threshold"]), criterion), reference[key])
    raw = planning.read_json(files[0])
    algorithms = [a if isinstance(a, str) else a["name"] for a in effective["algorithms"]]
    expected_traces = {(float(lam), int(effective.get("seed", 0)) + int(lam * 1000) + i, alg)
                       for lam in effective["lambdas"] for i in range(int(effective["n_seeds"])) for alg in algorithms}
    actual_traces = [(float(row["lambda"]), int(row["seed"]), alg) for row in raw for alg in row["traces"]]
    coverage = bool(expected_traces) and set(actual_traces) == expected_traces and len(actual_traces) == len(expected_traces)
    complete_traces = True
    rows = []
    for row in raw:
        for algorithm, trace in row["traces"].items():
            metric = trace["policy_value_errors"] if effective["problem_type"] == "control" else trace["errors"]
            queries = trace["queries"]
            complete_traces = complete_traces and trace_complete(trace, metric, algorithm, effective, output / "inputs")
            thresholds = metadata["thresholds"] if metadata is not None and selected["active_metadata"]["selector"] == "panels[*][*]" else [float(effective["threshold"])]
            for threshold in thresholds:
                for criterion, compute in (("stable", stable), ("first", first)):
                    hit = compute(queries, metric, threshold)
                    key = (float(row["lambda"]), algorithm, float(threshold), criterion)
                    reference = references.get(key)
                    if reference is not None and int(reference) > effective["max_queries"]:
                        reference = None
                    available = key in references
                    rows.append({"lambda": row["lambda"], "seed": row["seed"], "algorithm": algorithm,
                                 "threshold": threshold, "criterion": criterion, "hit": hit,
                                 "reference_available": available, "reference_hit": reference,
                                 "matches": hit == reference if available else None,
                                 "last_query": int(queries[-1]) if queries else None, "final_error": metric[-1] if metric else None})
    target = check_target(output, effective)
    report = {"kind": "selected-baseline-summary", "rows": rows,
              "scope": cfg.get("scope", "reference baseline configuration"),
              "comparison_pending": selected is None, "reference_build_identity_claimed": False,
              "complete_selected_coverage": coverage, "complete_recorded_observation_budget": bool(complete_traces and raw),
              "target_fingerprint_check": target}
    if selected is not None:
        required = {key for key in references if key[0] in {float(lam) for lam in effective["lambdas"]} and key[1] in algorithms}
        compared = [(float(row["lambda"]), row["algorithm"], float(row["threshold"]), row["criterion"]) for row in rows if row["reference_available"]]
        thresholds = metadata["thresholds"] if selected["active_metadata"]["selector"] == "panels[*][*]" else [float(effective["threshold"])]
        metadata_coverage = not duplicate_reference and reference_coverage(references, compared, expected_traces, thresholds, effective["problem_type"] == "control", int(effective["n_seeds"]))
        report.update(passed=bool(coverage and complete_traces and target["passed"] and metadata_coverage and all(row["matches"] is True for row in rows if row["reference_available"])),
                      complete_active_metadata_coverage=metadata_coverage, expected_metric_comparisons=len(required) * int(effective["n_seeds"]),
                      reference_metadata=selected["active_metadata"], reference_sha256=planning.sha(planning.ROOT / selected["active_metadata"]["path"]))
    else:
        report["note"] = "Supplemental prose grid has no plotted metadata rows; numerical summary is preserved without a plotted-reference claim."
        if source_id == "source-evaluation-cliffwalk-self-loop-gamma099-boundary":
            source = planning.ROOT / 'configs/scientific-claims/cliffwalk-pe-transition.json'
            reference = planning.read_json(source)
            contract, claim = transition_claim(rows, effective, reference)
            report.update(kind="supplemental-prose-claim", comparison_pending=False,
                          passed=bool(coverage and complete_traces and target["passed"] and contract and claim),
                          claim_supported=bool(contract and claim),
                          reference_claim_contract=planning.rel(source), reference_claim_contract_sha256=planning.sha(source),
                          reference_prose=reference['reference_source']['path'],
                          reference_prose_sha256=reference['reference_source']['sha256'], reference_prose_packaged=False,
                          claim=reference['claim'],
                          reference_query_values_available=False,
                          note="Compared to the explicit manuscript no-hit claim, not to invented reference query counts. These four transition points supplement the separately checked plotted coarse grid; reference raw transition traces are unavailable.")
    return report


def check(output, report_root=None):
    output = Path(output).resolve()
    if (output / "solver/effective-config.json").exists():
        report = practical(output)
    elif (output / "portable-config.yaml").exists():
        report = numerical_summary(output)
    elif (output / "solver/verification-checks.json").exists():
        report = planning_profile(output)
    elif (output / "package/data/two_state_compact_inset.raw_data.json").exists():
        report = two_state(output)
    else:
        raise ValueError(f"Unsupported or incomplete reference output {output}")
    ident = planning.read_json(output / "study-provenance.json")["id"]
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    destination = Path(report_root) if report_root else planning.ROOT / "verification/reference-comparisons" / ident / suffix
    destination.mkdir(parents=True, exist_ok=True)
    report["recomputed_output_root"] = str(output)
    report["recomputed_output_hashes"] = {str(path.relative_to(output)).replace("\\", "/"): planning.sha(path)
        for pattern in ("curves/*.json", "solver/raw/*_traces.json", "solver/summary.json", "solver/raw-traces.json",
                        "solver/target-certification.json", "solver/environment-target.npz",
                        "inputs/target-certification.json", "inputs/environment-target.npz",
                        "solver/verification-checks.json", "solver/all-19-comparison.json", "package/data/*.raw_data.json") for path in output.glob(pattern)}
    planning.write_json(destination / "scientific-comparison.json", report)
    rows = report["rows"]
    if rows:
        columns = sorted(set().union(*(r.keys() for r in rows)))
        with (destination / "scientific-summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report-root", type=Path, help="Separate report destination; source attempt remains immutable")
    args = parser.parse_args()
    report = check(args.output, args.report_root)
    print(json.dumps({"kind": report["kind"], "rows": len(report["rows"]), "passed": report.get("passed"),
                      "comparison_pending": report.get("comparison_pending", False)}))
    if report.get("passed") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
