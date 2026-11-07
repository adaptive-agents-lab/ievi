"""Run the common-solver control accuracy study with 1–4 CPU workers and pinned numerical threads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
CONTROL_ACCURACY = CODE_ROOT / "src" / "ievi_experiments" / "control_accuracy"
DEFAULT_TARGET = CONTROL_ACCURACY / "assets" / "target_arrays.npz"
PYTHON = Path(sys.executable)

LAMBDAS = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.625, 0.65, 0.675,
    0.7, 0.725, 0.75, 0.775, 0.8, 0.85, 0.9, 0.95, 1.0,
]


def pin_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    return env


def run_lambda(lam: float, target: Path, workers_root: Path) -> str:
    out = workers_root / f"lambda_{lam:g}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(CONTROL_ACCURACY / "run_control_accuracy_lambda.py"),
        "--target-arrays",
        str(target),
        "--lambda-value",
        f"{lam:.10g}",
        "--output-dir",
        str(out),
    ]
    proc = subprocess.run(cmd, env=pin_env(), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"lambda {lam} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-arrays", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("Choose 1-4 CPU workers")

    out = args.output_dir.resolve()
    workers_root = out / "workers"
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    workers_root.mkdir(parents=True, exist_ok=True)

    print(f"Controlled solve-accuracy comparison: {len(args.lambdas)} lambdas, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_lambda, lam, args.target_arrays, workers_root): lam
            for lam in args.lambdas
        }
        for fut in as_completed(futures):
            lam = futures[fut]
            line = fut.result()
            print(line or f"lambda {lam} done", flush=True)

    agg = subprocess.run(
        [
            str(PYTHON),
            str(CONTROL_ACCURACY / "aggregate_control_accuracy.py"),
            "--input-root",
            str(workers_root),
            "--output-dir",
            str(raw),
        ],
        env=pin_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    if agg.returncode != 0:
        raise RuntimeError(f"aggregate failed\n{agg.stdout}\n{agg.stderr}")
    print(agg.stdout.strip(), flush=True)
    print(f"wrote {raw / 'control_accuracy_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
