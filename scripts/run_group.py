"""Run independent non-timing tasks with a bounded number of workers."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import selected_tasks
from scripts.reproduction_manifest import guard_output_path, load_tasks, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("category")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    p.add_argument("--output-root", type=Path, default=ROOT / "runs/current")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", nargs="+", help="Restrict this category to the listed task IDs")
    a = p.parse_args()
    tasks = selected_tasks(load_tasks(ROOT), a.category)
    if a.only:
        available = {t["id"] for t in tasks}
        if set(a.only) - available:
            p.error("--only contains a task outside the selected category")
        tasks = [t for t in tasks if t["id"] in a.only]
    if any(t["category"] == "timing" for t in tasks):
        p.error("Timing must use the sequential task driver with other jobs stopped.")
    if any(t["depends_on"] for t in tasks):
        p.error("Use the sequential driver for tasks with dependencies.")
    output = guard_output_path(a.output_root, ROOT)
    output.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    results = []

    def run(task):
        command = [sys.executable, str(ROOT / "reproduce.py"), "run", task["id"], "--output-root", str(output)]
        if a.resume:
            command.append("--resume")
        print("QUEUED", task["id"], flush=True)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="", flush=True)
        return {"task_id": task["id"], "argv": command, "exit_code": result.returncode}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    report = {"category": a.category, "workers": a.workers, "started_at": started,
              "finished_at": utc_now(), "results": results,
              "passed": all(r["exit_code"] == 0 for r in results)}
    name = started.replace(":", "").replace("+", "_")
    write_json(output / f"group-{a.category}-{name}.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
