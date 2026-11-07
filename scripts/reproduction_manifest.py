"""Portable task loading, content identities, and atomic execution records."""
from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
EXCLUDED_PARTS = {"__pycache__", ".git", ".venv", ".pytest_cache"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def accessible(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def sha256(path):
    h = hashlib.sha256()
    with accessible(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_tasks(root, manifest=None):
    root = Path(root).resolve()
    paths = [Path(manifest)] if manifest else sorted((root / "configs/tasks.d").glob("*.json"))
    if manifest is None and (root / "configs/tasks.json").exists():
        paths.insert(0, root / "configs/tasks.json")
    tasks = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported task schema: {path}")
        for task in data["tasks"]:
            ident = task["id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ident):
                raise ValueError(f"Unsafe task id: {ident}")
            if ident in tasks:
                raise ValueError(f"Duplicate task id: {ident}")
            if not task.get("argv") or not all(isinstance(x, str) for x in task["argv"]):
                raise ValueError(f"Task {ident}: argv must be a nonempty string list")
            if not task.get("expected_outputs"):
                raise ValueError(f"Task {ident}: expected_outputs must be explicit")
            task = dict(task)
            task.setdefault("category", "other")
            task.setdefault("cwd", "{root}")
            task.setdefault("env", {})
            task.setdefault("inputs", [])
            task.setdefault("code", [])
            task.setdefault("depends_on", [])
            tasks[ident] = task
    if not tasks:
        raise ValueError("No tasks found")
    for task in tasks.values():
        for dependency in task["depends_on"]:
            if dependency not in tasks:
                raise ValueError(f"Task {task['id']}: unknown dependency {dependency}")
    return tasks


def expand(value, root, output, python):
    return value.format(root=str(root), output=str(output), python=str(python))


def snapshot(root, patterns):
    root = Path(root).resolve()
    files = {}
    missing = []
    for pattern in patterns:
        path_pattern = expand(pattern, root, "{output}", "{python}")
        if not Path(path_pattern).is_absolute():
            path_pattern = str(root / path_pattern)
        matches = [Path(x) for x in glob.glob(str(accessible(path_pattern)), recursive=True)]
        found = []
        for match in matches:
            found.extend(match.rglob("*") if match.is_dir() else [match])
        found = [x for x in found if x.is_file() and not (set(x.parts) & EXCLUDED_PARTS)]
        if not found:
            missing.append(pattern)
        for path in found:
            resolved = Path(str(path.resolve()).removeprefix("\\\\?\\"))
            key = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
            files[key] = {"sha256": sha256(resolved), "bytes": accessible(resolved).stat().st_size}
    return {"files": dict(sorted(files.items())), "missing_patterns": missing}


def runtime_metadata():
    packages = {}
    for name in ["numpy", "scipy", "pandas", "numba", "matplotlib", "jax", "jaxlib", "PyYAML", "gymnasium"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python_executable": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "logical_processors": os.cpu_count(), "packages": packages}


def identity(task, root):
    inputs = snapshot(root, task["inputs"])
    code = snapshot(root, task["code"] + ["reproduce.py", "scripts/reproduction_manifest.py"])
    if inputs["missing_patterns"] or code["missing_patterns"]:
        raise ValueError(f"Task {task['id']} missing inputs/code: {inputs['missing_patterns'] + code['missing_patterns']}")
    runtime = runtime_metadata()
    inherited = {key: os.environ[key] for key in ["PYTHONPATH", "PYTHONHASHSEED", "XLA_FLAGS", "JAX_ENABLE_X64", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"]
                 if key in os.environ and key not in task["env"]}
    payload = {"schema_version": SCHEMA_VERSION, "task": task, "inputs": inputs, "code": code,
               "inherited_execution_environment": inherited,
               "runtime": runtime, "resolved_command_template": [expand(x, root, "{output}", sys.executable) for x in task["argv"]],
               "resolved_cwd": expand(task["cwd"], root, "{output}", sys.executable)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, payload


def output_snapshot(directory):
    directory = accessible(directory)
    return {p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.relative_to(directory).as_posix() not in {"execution.json", "execution.json.tmp", "comparison.json", "comparison.json.tmp"}
            and not (set(p.parts) & EXCLUDED_PARTS)}


def verified_record(path, identity_hash=None):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("exit_code") != 0:
            return False
        if identity_hash is not None and record.get("identity_sha256") != identity_hash:
            return False
        outputs = record.get("outputs", {})
        if not outputs or output_snapshot(Path(path).parent) != outputs:
            return False
        return True
    except (OSError, ValueError, TypeError):
        return False


FROZEN_DIRECTORIES = ("methods", "studies", "reference-assets", "paper", "configs", "scripts", "tests", "docs", "datasets", "verification")


def guard_output_path(output, root):
    """Allow new generated trees while protecting frozen inputs and their parents."""
    output, root = Path(output).resolve(), Path(root).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Output must not replace or contain the reproduction root")
    for name in FROZEN_DIRECTORIES:
        frozen = root / name
        if output.is_relative_to(frozen) or frozen.is_relative_to(output):
            raise ValueError(f"Output conflicts with frozen inputs: {name}")
    latex = root.parent / "latex"
    if output.is_relative_to(latex) or latex.is_relative_to(output):
        raise ValueError("Code workflows must not write into the manuscript workspace")
    # Artifacts may contain newly generated exports beside immutable imports.
    # Never reuse an existing nonempty artifact destination.
    if output.is_relative_to(root / "artifacts") and output.exists() and any(output.iterdir()):
        raise ValueError("Existing artifact output is preserved; choose a new destination")
    return output
