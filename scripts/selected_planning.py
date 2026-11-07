"""Selected planning experiments from the declared environment catalog."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import yaml

import planning_reproduction as planning

ROOT = planning.ROOT
CONFIGS = ROOT / "configs/planning/selected"
CURRENT_19 = [0, .1, .2, .3, .4, .5, .6, .625, .65, .675, .7, .725, .75, .775, .8, .85, .9, .95, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--study")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()
    if not args.study or not args.output_root:
        parser.error("--study and --output-root are required")
    entry = next(e for e in planning.read_json(CONFIGS / "catalog.json")["studies"] if e["id"] == args.study)
    output = planning.empty_output(args.output_root)
    planning.configure_runtime()
    planning.write_json(output / "study-provenance.json", {**entry, "argv": sys.argv, "python": sys.version,
        "adapter_sha256": planning.sha(__file__), "reference_adapter_sha256": planning.sha(planning.__file__), "started_at_unix": time.time()})
    planning.run_baseline(entry, output, args.cores)
    planning.write_json(output / "completed.json", {"id": args.study, "completed": True, "completed_at_unix": time.time()})


if __name__ == "__main__":
    main()
