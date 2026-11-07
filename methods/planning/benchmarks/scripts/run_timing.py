"""Prepare a run directory and execute the 10,000-state value-iteration runtime comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = CODE_ROOT / "src" / "ievi_experiments" / "wallclock_10k"
PYTHON = Path(sys.executable)


def pin_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    return env


def materialize(run_dir: Path, grid_sides: list[int], repetitions: dict[str, int]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "code").mkdir(parents=True, exist_ok=True)
    for src in (PACKAGE / "code").glob("*.py"):
        shutil.copy2(src, run_dir / "code" / src.name)
    config = json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))
    config["benchmark_grid_sides"] = [int(x) for x in grid_sides]
    config["benchmark_repetitions"] = {str(int(k)): int(v) for k, v in repetitions.items()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in ("data/raw", "data/derived", "environment", "validation"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir / "config.json"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=pin_env(), check=False, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-sides", type=int, nargs="+", default=[100])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")

    run_dir = args.output_dir.resolve()
    repetitions = {str(n): args.reps for n in args.grid_sides}
    materialize(run_dir, args.grid_sides, repetitions)

    run([str(PYTHON), "-u", "code/prepare_experiment.py"], run_dir)
    if args.prepare_only:
        return 0
    for n in args.grid_sides:
        for rep in range(args.reps):
            run(
                [str(PYTHON), "-u", "code/run_one_benchmark.py", "--grid-side", str(n), "--repetition", str(rep)],
                run_dir,
            )
    run([str(PYTHON), "-u", "code/aggregate_results.py"], run_dir)
    print(f"wrote {run_dir / 'data' / 'derived' / 'runtime_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
