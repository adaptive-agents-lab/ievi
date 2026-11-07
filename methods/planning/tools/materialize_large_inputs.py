"""Materialize losslessly compressed scientific inputs in an extracted working copy."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def extended(path):
    path = path.resolve()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def materialize(root=ROOT):
    root = extended(root)
    entries = json.loads((root / 'LARGE-INPUTS.json').read_text(encoding='utf-8'))
    for entry in entries:
        compressed = root / entry['compressed_path']
        assert hashlib.sha256(compressed.read_bytes()).hexdigest() == entry['compressed_sha256']
        raw = gzip.decompress(compressed.read_bytes())
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
        dest = root / entry['path']
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Refusing to replace a different existing input: {entry["path"]}')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
    return entries


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    print(json.dumps({'materialized_inputs': len(materialize(args.root))}))
