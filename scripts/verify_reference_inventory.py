"""Verify the distributed scientific reference inventory."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def accessible(path):
    resolved = str(Path(path).resolve())
    return Path('\\\\?\\' + resolved) if os.name == 'nt' and not resolved.startswith('\\\\?\\') else Path(resolved)


def digest(path):
    result = hashlib.sha256()
    with accessible(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def reference_path(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe reference path: {name}')
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'External or symlinked reference path: {name}')
    return accessible(path)


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = root / 'verification/reference-inventory.json'
    inventory = json.loads(manifest.read_text(encoding='utf-8'))
    records = inventory['files']
    originals = {record['path']: record for record in records}
    if len(originals) != len(records):
        raise ValueError('Duplicate reference inventory paths')
    discrepancies = []
    total_bytes = 0
    verified = 0
    for record in records:
        path = reference_path(root, record['path'])
        if not path.is_file():
            discrepancies.append({'path': record['path'], 'issue': 'missing'})
            continue
        size = path.stat().st_size
        total_bytes += size
        actual = digest(path)
        matches = actual == record['sha256'] and size == record['bytes']
        if not matches:
            discrepancies.append({'path': record['path'], 'issue': 'changed', 'actual_sha256': actual, 'actual_bytes': size})
        else:
            verified += 1
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'repository': str(root),
              'manifest_sha256': digest(manifest), 'file_count': len(records), 'bytes': total_bytes,
              'inventory_scope': inventory.get('scope', 'Distributed reference inventory'),
              'verified_count': verified,
              'verification': 'Exact SHA-256 and size equality for every declared scientific reference file.',
              'passed': not discrepancies, 'discrepancies': discrepancies}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = verify()
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
