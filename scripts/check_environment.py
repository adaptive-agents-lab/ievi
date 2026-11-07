"""Check the installed numerical environment without running an experiment."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    packages = {
        "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
        "jax": "jax", "jaxlib": "jaxlib", "numba": "numba",
        "matplotlib": "matplotlib", "PyYAML": "yaml",
        "threadpoolctl": "threadpoolctl", "pypdf": "pypdf",
    }
    versions, errors = {}, []
    for distribution, module in packages.items():
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
        except Exception as exc:
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")
    devices, x64 = [], None
    if "jax" in versions:
        import jax
        devices = [str(device) for device in jax.devices()]
        x64 = bool(jax.config.x64_enabled)
        if not x64:
            errors.append("JAX float64 must be enabled for these experiments.")
    consistency = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if consistency.returncode:
        errors.append(consistency.stdout.strip() or consistency.stderr.strip())
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "executable": sys.executable, "virtual_environment": sys.prefix != sys.base_prefix,
        "versions": versions, "jax_devices": devices, "jax_float64": x64,
        "dependency_check": consistency.stdout.strip(), "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
