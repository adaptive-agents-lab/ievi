"""Run the policy-evaluation horizon study, optionally restricted by H or schedule."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
HORIZON_SCALING = CODE_ROOT / "src" / "ievi_experiments" / "horizon_scaling"


def pin_threads() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--schedules", nargs="+", default=None)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--timing-repetitions", type=int, default=5)
    args = parser.parse_args()

    pin_threads()
    sys.path.insert(0, str(HORIZON_SCALING))
    import run_horizon_scaling  # noqa: E402

    if args.horizons:
        run_horizon_scaling.H_GRID = np.asarray(args.horizons, dtype=np.int64)
    if args.schedules:
        run_horizon_scaling.SCHEDULE_ORDER = tuple(args.schedules)

    argv = [
        "run_horizon_scaling.py",
        "--output-root",
        str(args.output_root.resolve()),
        "--replications",
        str(args.replications),
        "--base-seed",
        str(args.base_seed),
        "--target",
        "1e-3",
        "--stable-tail",
        "50",
        "--timing-repetitions",
        str(args.timing_repetitions),
    ]
    if args.skip_timing:
        argv.append("--skip-timing")
    sys.argv = argv
    run_horizon_scaling.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
