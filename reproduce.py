#!/usr/bin/env python3
"""Reproduce reference figures or new solver tasks without replacing evidence."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.reproduction_manifest import guard_output_path, expand, identity, load_tasks, output_snapshot, utc_now, verified_record, write_json
from scripts.compare_results import compare_task

ROOT = Path(__file__).resolve().parent


def selected_tasks(tasks, selector):
    selected = list(tasks) if selector == "all" else [key for key, task in tasks.items() if key == selector or task["category"] == selector]
    if not selected:
        raise ValueError(f"Unknown task or category: {selector}")
    ordered, visiting = [], set()
    def visit(key):
        if key in visiting:
            raise ValueError(f"Dependency cycle at {key}")
        if key in ordered:
            return
        visiting.add(key)
        for dependency in tasks[key]["depends_on"]:
            visit(dependency)
        visiting.remove(key)
        ordered.append(key)
    for key in selected:
        visit(key)
    return [tasks[key] for key in ordered]


def records(output_root, task_id):
    return sorted((output_root / task_id).glob("*/execution.json"), reverse=True)


def run_task(task, output_root, resume):
    digest, details = identity(task, ROOT)
    if resume:
        for path in records(output_root, task["id"]):
            if verified_record(path, digest):
                print(f"RESUMED {task['id']} {path.parent}", flush=True)
                return True
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output_root / task["id"] / attempt
    output.mkdir(parents=True, exist_ok=False)
    argv = [expand(x, ROOT, output, sys.executable) for x in task["argv"]]
    cwd = Path(expand(task["cwd"], ROOT, output, sys.executable)).resolve()
    env_overrides = {key: expand(str(value), ROOT, output, sys.executable) for key, value in task["env"].items()}
    environment = os.environ.copy()
    environment.update(env_overrides)
    record = {"schema_version": 1, "task_id": task["id"], "category": task["category"], "identity_sha256": digest,
              "identity": details, "argv": argv, "cwd": str(cwd), "env_overrides": env_overrides,
              "started_at": utc_now(), "status": "running", "exit_code": None, "duration_seconds": None,
              "log": "process.log", "output_directory": str(output)}
    write_json(output / "execution.json", record)
    print(f"START {task['id']} -> {output}", flush=True)
    start = time.monotonic()
    code, error = None, None
    try:
        with (output / "process.log").open("wb") as log:
            code = subprocess.run(argv, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
    except KeyboardInterrupt:
        error = "Interrupted by user"
    except OSError as exc:
        error = str(exc)
    missing = [item for item in task["expected_outputs"] if not list(output.glob(item))]
    try:
        post_digest, _ = identity(task, ROOT)
        unchanged = post_digest == digest
    except ValueError:
        unchanged = False
    record.update(finished_at=utc_now(), duration_seconds=time.monotonic() - start, exit_code=code,
                  missing_expected_outputs=missing, inputs_and_code_unchanged=unchanged, error=error,
                  status="completed" if code == 0 and not missing and unchanged else "failed")
    record["outputs"] = output_snapshot(output)
    write_json(output / "execution.json", record)
    print(f"{record['status'].upper()} {task['id']} exit={code} seconds={record['duration_seconds']:.3f}", flush=True)
    if error == "Interrupted by user":
        raise KeyboardInterrupt
    return record["status"] == "completed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Optional standalone task manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for command in ("verify-dataset", "materialize-dataset"):
        p = sub.add_parser(command)
        p.add_argument("selector", nargs="?", default="all")
        p.add_argument("--dataset", type=Path)
        p.add_argument("--cache-root", type=Path)
    render = sub.add_parser("render-paper")
    render.add_argument("--dataset", type=Path)
    render.add_argument("--output-root", required=True, type=Path)
    cliffwalk = sub.add_parser("cliffwalk", help="Verify, select, rerun, or render the current Cliffwalk study")
    cliffwalk.add_argument("study_command", choices=("verify", "select", "run", "compare", "render-data", "render"))
    cliffwalk.add_argument("study_args", nargs=argparse.REMAINDER)
    verify = sub.add_parser("verify-inputs")
    verify.add_argument("selector", nargs="?", default="all")
    for command in ["run", "compare", "status"]:
        p = sub.add_parser(command)
        p.add_argument("selector", nargs="?", default="all")
        p.add_argument("--output-root", type=Path, default=ROOT / "runs/current")
        if command == "compare":
            p.add_argument("--dataset", type=Path)
        if command == "run":
            p.add_argument("--resume", action="store_true", help="Skip only content-verified successful attempts with identical commands, inputs, code and runtime")
    args = parser.parse_args(argv)
    if args.command == "cliffwalk":
        if args.study_command == "render":
            from scripts.cliffwalk import main as cliffwalk_main
            return cliffwalk_main(args.study_args)
        command = [sys.executable, "-m", "studies.cliffwalk", args.study_command, *args.study_args]
        return subprocess.run(command, cwd=ROOT).returncode
    if args.command in ("verify-dataset", "materialize-dataset"):
        from scripts.reference_dataset import DEFAULT_DATASET, main as dataset_main
        command = "verify" if args.command == "verify-dataset" else "materialize"
        forwarded = [command, args.selector, "--dataset", str(args.dataset or DEFAULT_DATASET)]
        if args.cache_root:
            forwarded.extend(["--cache-root", str(args.cache_root)])
        return dataset_main(forwarded)
    if args.command == "render-paper":
        from scripts.reference_dataset import DEFAULT_DATASET
        command = [sys.executable, str(ROOT / "scripts/render_paper.py"), "--dataset", str(args.dataset or DEFAULT_DATASET), "--output-root", str(args.output_root)]
        return subprocess.run(command, cwd=ROOT).returncode
    tasks = load_tasks(ROOT, args.manifest)
    if args.command == "list":
        print("cliffwalk\tcliffwalk-study\tCLI: reproduce.py cliffwalk verify|select|run|compare|render-data|render --help")
        for task in tasks.values():
            print(f"{task['id']}\t{task['category']}\t{'expensive' if task.get('expensive') else 'ordinary'}\t{task.get('description', '')}")
        return 0
    chosen = selected_tasks(tasks, args.selector)
    if args.command == "verify-inputs":
        failed = False
        for task in chosen:
            try:
                digest, details = identity(task, ROOT)
                print(f"OK {task['id']} identity={digest} inputs={len(details['inputs']['files'])} code={len(details['code']['files'])}")
            except ValueError as exc:
                print(f"FAIL {exc}")
                failed = True
        return int(failed)
    output_root = guard_output_path(args.output_root, ROOT)
    failed = False
    for task in chosen:
        if args.command == "run":
            if not run_task(task, output_root, args.resume):
                return 1
        elif args.command == "status":
            paths = records(output_root, task["id"])
            if not paths:
                print(f"NOT_RUN {task['id']}")
                continue
            record = json.loads(paths[0].read_text())
            digest, _ = identity(task, ROOT)
            valid = verified_record(paths[0], digest)
            print(f"{record['status'].upper()} {task['id']} verified_current={valid} {paths[0].parent}")
            failed |= not valid
        else:
            paths = records(output_root, task["id"])
            digest, _ = identity(task, ROOT)
            path = next((p for p in paths if verified_record(p, digest)), None)
            if path is None:
                print(f"NO_VERIFIED_RUN {task['id']}")
                failed = True
                continue
            if args.manifest is None or args.dataset is not None:
                from scripts.reference_dataset import DEFAULT_DATASET
                from scripts.compare_results import compare_accepted_task
                result = compare_accepted_task(task, ROOT, path.parent, args.dataset or DEFAULT_DATASET)
            else:
                result = compare_task(task, ROOT, path.parent)
                result["reference_role"] = "caller-supplied manifest inputs"
            write_json(path.parent / "comparison.json", result)
            print(f"COMPARE {task['id']} status={result['status']} passed={result['passed']}")
            failed |= result["passed"] is not True
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
