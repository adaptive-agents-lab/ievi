#!/usr/bin/env python3
"""Verify experiment artifacts against an experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifest.json.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_path: Path) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    exp_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for item in manifest.get("files", []):
        rel_path = item["path"]
        path = exp_dir / rel_path
        if not path.exists():
            failures.append(f"missing: {rel_path}")
            continue
        checked += 1
        actual_size = path.stat().st_size
        actual_sha = _sha256(path)
        if actual_size != int(item["bytes"]):
            failures.append(f"size mismatch: {rel_path}: {actual_size} != {item['bytes']}")
        if actual_sha != item["sha256"]:
            failures.append(f"sha256 mismatch: {rel_path}: {actual_sha} != {item['sha256']}")
    return {"checked": checked, "failures": failures}


def main() -> None:
    args = _parse_args()
    result = verify(args.manifest)
    print(json.dumps(result, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
