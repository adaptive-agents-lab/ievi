"""Reference inventory checks fail on changed, missing, unsafe or duplicate files."""
import hashlib
import json
import pytest
from scripts.verify_reference_inventory import verify


@pytest.fixture
def references(tmp_path):
    (tmp_path / 'verification').mkdir()
    for name, raw in [('source.py', b'VALUE = 1\n'), ('data.csv', b'value\n1\n')]:
        (tmp_path / name).write_bytes(raw)
    records = [{'path': name, 'sha256': hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
                'bytes': (tmp_path / name).stat().st_size} for name in ('source.py', 'data.csv')]
    manifest = tmp_path / 'verification/reference-inventory.json'
    manifest.write_text(json.dumps({'schema_version': 2, 'scope': 'Scientific references.', 'files': records}))
    return tmp_path, manifest, records


def test_every_reference_has_exact_hash_and_size(references):
    root, _, _ = references
    report = verify(root)
    assert report['passed'] and report['verified_count'] == 2
    (root / 'data.csv').write_bytes(b'value\n2\n')
    report = verify(root)
    assert not report['passed'] and report['verified_count'] == 1
    assert report['discrepancies'][0]['path'] == 'data.csv'


def test_missing_reference_never_passes(references):
    root, _, _ = references
    (root / 'source.py').unlink()
    report = verify(root)
    assert not report['passed'] and report['discrepancies'][0]['issue'] == 'missing'


@pytest.mark.parametrize('name', ['../source.py', '/source.py', 'C:/source.py', 'a\\b', 'a/./b'])
def test_unsafe_inventory_paths_refused(references, name):
    root, manifest, records = references
    records[0]['path'] = name
    manifest.write_text(json.dumps({'files': records}))
    with pytest.raises(ValueError, match='Unsafe'):
        verify(root)


def test_duplicate_inventory_paths_refused(references):
    root, manifest, records = references
    manifest.write_text(json.dumps({'files': records + [records[0]]}))
    with pytest.raises(ValueError, match='Duplicate'):
        verify(root)
