"""Hash-verified scientific dataset access, separate from strict live-run resume.

Archives are never extracted with tar.extractall. Every regular member is checked
against the frozen manifest, and a complete verified staging tree is atomically
renamed into an ignored cache. Existing conflicting data is never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduction_manifest import sha256
DEFAULT_DATASET = ROOT / "datasets/reference/planning/DATASET.json"


def accessible(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def iter_files(directory):
    """Enumerate all regular files with long-path access and surfaced I/O errors.

    Glob traversal can silently omit deep Windows paths. Enumeration is also an
    integrity check, so inaccessible directories and links must fail closed.
    """
    pending = [accessible(directory)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = Path(entry.path)
            if entry.is_symlink():
                raise ValueError("Symlink in materialized accepted dataset")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                yield path
            else:
                raise ValueError(f"Nonregular accepted dataset member: {path}")


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Unsafe dataset member path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")) or ":" in value:
        raise ValueError(f"Unsafe dataset member path: {value}")
    return Path(*path.parts)


def load_dataset(dataset_path=DEFAULT_DATASET):
    dataset_path = Path(dataset_path).resolve()
    data = json.loads(accessible(dataset_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 2 or not isinstance(data.get("tasks"), dict):
        raise ValueError("Unsupported dataset manifest")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", data.get("dataset_id", "")):
        raise ValueError("Unsafe dataset id")
    if data.get("selection_count") != len(data["tasks"]):
        raise ValueError("Dataset selection count mismatch")
    for key, task in data["tasks"].items():
        if key != task.get("task_id") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
            raise ValueError("Unsafe or mismatched dataset task id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task.get("attempt", "")):
            raise ValueError("Unsafe dataset attempt id")
        safe_relative(task["archive"]["path"])
        for name in task["files"]:
            safe_relative(name)
        if task["files"].get("DATA.json", {}).get("sha256") != task["record_sha256"]:
            raise ValueError("Data record hash and file inventory disagree")
        if set(task["outputs"]) != set(task["files"]) - {"DATA.json"}:
            raise ValueError("Data output inventory is incomplete")
        if any(task["files"][name] != info for name, info in task["outputs"].items()):
            raise ValueError("Data output inventory differs from member inventory")
    source = safe_relative(data["selection_source"])
    summary_path = dataset_path.parents[3] / source
    if sha256(accessible(summary_path)) != data["selection_source_sha256"]:
        raise ValueError("Dataset selection index changed")
    index = json.loads(accessible(summary_path).read_text(encoding="utf-8"))
    rows = index["tasks"]
    selected = {row["task_id"]: row for row in rows}
    if len(selected) != len(rows) or set(selected) != set(data["tasks"]):
        raise ValueError("Dataset and selection index membership differ")
    for key, task in data["tasks"].items():
        if task["record_sha256"] != selected[key]["record_sha256"] or task["identity_sha256"] != selected[key]["identity_sha256"]:
            raise ValueError("Data record differs from selection index")
    return data


def _verify_record(record, task):
    if record.get("task_id") != task["task_id"] or record.get("identity_sha256") != task["identity_sha256"]:
        raise ValueError("Reference execution identity mismatch")
    if record.get("status") != "completed" or record.get("exit_code") != 0 or record.get("outputs") != task["outputs"]:
        raise ValueError("Reference execution is not the accepted completed record")
    if set(task["outputs"]) != set(task["files"]) - {"DATA.json"}:
        raise ValueError("Data record output inventory incomplete")
    for name, info in task["outputs"].items():
        if task["files"].get(name) != info:
            raise ValueError("Data output inventory mismatch")
    return record


def _verify_tree(directory, task):
    directory = accessible(directory)
    actual = {}
    for path in iter_files(directory):
        actual[path.relative_to(directory).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if actual != task["files"]:
        raise ValueError(f"Conflicting or tampered materialized data: {directory}")
    record = json.loads((directory / "DATA.json").read_text(encoding="utf-8"))
    return _verify_record(record, task)


def _archive(dataset_path, task, destination=None):
    # Resolve the parent, preserving the member itself for the symlink check.
    path = accessible(Path(dataset_path).resolve().parent) / safe_relative(task["archive"]["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != task["archive"]["bytes"] or sha256(path) != task["archive"]["sha256"]:
        raise ValueError(f"Missing or tampered accepted archive: {path}")
    seen = set()
    execution_bytes = bytearray()
    with tarfile.open(path, mode="r|gz") as tar:
        for member in tar:
            relative = safe_relative(member.name)
            if not member.isfile() or member.name in seen or member.name not in task["files"]:
                raise ValueError(f"Unexpected, duplicate, or nonregular archive member: {member.name}")
            seen.add(member.name)
            info = task["files"][member.name]
            if member.size != info["bytes"]:
                raise ValueError(f"Archive member size mismatch: {member.name}")
            digest = hashlib.sha256()
            sink = None
            if destination is not None:
                target = accessible(destination / relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                sink = target.open("xb")
            try:
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        if member.name == "DATA.json":
                            execution_bytes.extend(block)
                        if sink is not None:
                            sink.write(block)
            finally:
                if sink is not None:
                    sink.close()
            if digest.hexdigest() != info["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {member.name}")
    if seen != set(task["files"]):
        raise ValueError("Archive member inventory incomplete")
    _verify_record(json.loads(execution_bytes), task)
    return path


def verify_dataset(dataset_path=DEFAULT_DATASET, task_ids=None):
    data = load_dataset(dataset_path)
    chosen = list(data["tasks"]) if task_ids is None else list(task_ids)
    checked = []
    for key in chosen:
        task = data["tasks"][key]
        _archive(dataset_path, task)
        checked.append(key)
    return {"passed": True, "dataset_id": data["dataset_id"], "manifest_sha256": sha256(dataset_path), "verified_tasks": checked,
            "verification": "archive SHA256 and every member SHA256/size; independent of current runtime and original absolute paths"}


def resolve_attempt(dataset_path, task_id, cache_root=None):
    """Return (execution_path, data_record), safely materializing if needed."""
    data = load_dataset(dataset_path)
    if task_id not in data["tasks"]:
        raise ValueError(f"Task is absent from accepted dataset: {task_id}")
    task = data["tasks"][task_id]
    cache = Path(cache_root).resolve() if cache_root is not None else Path(dataset_path).resolve().parents[3] / ".cache/accepted"
    frozen = [ROOT / x for x in ("methods", "datasets", "studies", "reference-assets")]
    if cache == ROOT or ROOT.is_relative_to(cache) or any(cache.is_relative_to(x) or x.is_relative_to(cache) for x in frozen):
        raise ValueError("Materialization cache conflicts with frozen inputs")
    destination = cache / data["dataset_id"] / task_id / task["attempt"]
    if not destination.resolve().is_relative_to(cache):
        raise ValueError("Materialization destination escapes cache")
    if destination.is_symlink():
        raise ValueError("Symlink materialization destination refused")
    if destination.exists():
        _archive(dataset_path, task)
        record = _verify_tree(destination, task)
        return destination / "DATA.json", record
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".accepted-", dir=destination.parent))
    try:
        _archive(dataset_path, task, temporary)
        record = _verify_tree(temporary, task)
        if destination.exists():
            # Another writer may have completed identical materialization.
            record = _verify_tree(destination, task)
        else:
            try:
                temporary.rename(destination)
            except FileExistsError:
                record = _verify_tree(destination, task)
        return destination / "DATA.json", record
    finally:
        if temporary.exists():
            shutil.rmtree(accessible(temporary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "materialize"))
    parser.add_argument("selector", nargs="?", default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args(argv)
    data = load_dataset(args.dataset)
    chosen = list(data["tasks"]) if args.selector == "all" else [k for k, x in data["tasks"].items() if k == args.selector or x["category"] == args.selector]
    if not chosen:
        raise ValueError(f"No accepted tasks match {args.selector}")
    if args.command == "verify":
        print(json.dumps(verify_dataset(args.dataset, chosen), indent=2))
    else:
        for key in chosen:
            path, _ = resolve_attempt(args.dataset, key, args.cache_root)
            print(f"MATERIALIZED {key} {path.parent}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
