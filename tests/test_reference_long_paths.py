"""Accepted scientific membership must not shrink at Windows MAX_PATH."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from scripts.reference_dataset import accessible, resolve_attempt
from scripts.compare_results import accepted_specs, compare_accepted_task

ROOT = Path(__file__).resolve().parents[1]


def info(raw):
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


@pytest.fixture
def scientific_paths(tmp_path):
    names = [f'study/solver/arrays/{"long-path-component-" * 7}/measurement-{index:02d}.npy' for index in range(31)]
    array = io.BytesIO()
    np.save(array, np.array([1.25]))
    payload = {name: (array.getvalue() if name.endswith('.npy') else b'{"mean":[1.25],"x_axis":[0]}')
               for name in names}
    outputs = {name: info(raw) for name, raw in payload.items()}
    record = {'task_id': 'planning-fixture', 'identity_sha256': 'fixture',
              'status': 'completed', 'exit_code': 0, 'outputs': outputs}
    payload['DATA.json'] = json.dumps(record).encode()
    dataset = tmp_path / 'package/datasets/accepted/fixture/DATASET.json'
    dataset.parent.mkdir(parents=True)
    archive = dataset.parent / 'sampled.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, raw in payload.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    accepted = {'task_id': 'planning-fixture', 'attempt': 'original', 'category': 'test',
                'identity_sha256': 'fixture', 'record_sha256': info(payload['DATA.json'])['sha256'],
                'files': {name: info(raw) for name, raw in payload.items()}, 'outputs': outputs,
                'archive': dict(info(archive.read_bytes()), path=archive.name)}
    index = dataset.parents[3] / 'verification/data-index.json'
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({'schema_version':2,'tasks':[{'task_id':accepted['task_id'],'record_sha256':accepted['record_sha256'],'identity_sha256':accepted['identity_sha256']}]}))
    dataset.write_text(json.dumps({'selection_source':'verification/data-index.json','selection_source_sha256':info(index.read_bytes())['sha256'],'schema_version': 2, 'dataset_id': 'fixture', 'selection_count': 1,
                                   'tasks': {'planning-fixture': accepted}}))
    # Several accepted IE paths exceed 260 characters even under a short root.
    actual = tmp_path / ('deep-actual-' + 'x' * 100)
    for name, raw in payload.items():
        path = accessible(actual / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    assert max(len(str(actual / name)) for name in names) > 260
    return dataset, actual, accepted, payload


def test_all_31_streams_compare_on_deep_paths_without_glob_membership(scientific_paths, monkeypatch):
    dataset, actual, accepted, _ = scientific_paths
    monkeypatch.setattr(Path, 'rglob', lambda *args, **kwargs: pytest.fail('Glob must not determine accepted membership'))
    task = {'id': 'planning-fixture'}
    result = compare_accepted_task(task, ROOT, actual, dataset)
    assert result['passed'] and result['comparison_count'] == 31
    assert all(result['invariants'].values())
    execution, _ = resolve_attempt(dataset, task['id'])
    specs = accepted_specs(task, execution.parent, reference_files=accepted['files'])
    assert len(specs) == 31
    assert all(spec['rtol'] == 1e-9 and spec['atol'] == 1e-11 for spec in specs)


def test_missing_deep_reference_fails_closed_instead_of_shrinking_specs(scientific_paths):
    dataset, _, accepted, _ = scientific_paths
    execution, _ = resolve_attempt(dataset, 'planning-fixture')
    name = next(name for name in accepted['outputs'])
    accessible(execution.parent / name).unlink()
    with pytest.raises(ValueError, match='reference file missing'):
        accepted_specs({'id': 'planning-fixture'}, execution.parent, reference_files=accepted['files'])
    with pytest.raises(ValueError, match='Conflicting or tampered'):
        resolve_attempt(dataset, 'planning-fixture')


def test_missing_actual_scientific_array_never_passes(scientific_paths):
    dataset, actual, accepted, payload = scientific_paths
    name = next(name for name in accepted['outputs'])
    path = accessible(actual / name)
    path.unlink()
    task = {'id': 'planning-fixture'}
    result = compare_accepted_task(task, ROOT, actual, dataset)
    assert not result['passed'] and result['comparison_count'] == 31
    assert any(item.get('error') == 'Actual or reference file missing' for item in result['comparisons'])
    path.write_bytes(payload[name])
    assert compare_accepted_task(task, ROOT, actual, dataset)["passed"]
