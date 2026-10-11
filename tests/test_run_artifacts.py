from pathlib import Path
import json

import pytest

from common import run_artifacts as runs


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv('AG3S_STORAGE_ROOT', str(tmp_path))
    monkeypatch.delenv('AG3S_RUN_DIR', raising=False)
    return tmp_path


def test_run_records_baseline_and_collision(storage):
    path = runs.create_run(run_id='test', settings={'algorithm': 'V11AB'}, checkpoint='/input/29999')
    manifest = json.loads((path / 'run_manifest.json').read_text())
    assert path.parent == storage / 'results/benchmark'
    assert manifest['baseline_v2']['benchmark']['commit'] == '498f713b20bf9b75b4b065cd9bcdd508d7cdaecf'
    assert manifest['baseline_v2']['pi05_TO_hybrid']['commit'] == 'f62ab814e867912af3cecea52519d00c470d39e3'
    assert manifest['settings']['algorithm'] == 'V11AB'
    assert manifest['checkpoint'] == '/input/29999'
    assert not (path / 'policy_records').exists()
    # Force the exact destination to verify that existing evidence cannot be reused.
    original = path / 'run_manifest.json'
    before = original.read_bytes()
    with pytest.raises(FileExistsError):
        path.mkdir(exist_ok=False)
    assert original.read_bytes() == before


def test_input_and_symlink_escape_rejected(storage):
    inputs = storage / 'recordings/policy_records'
    inputs.mkdir(parents=True)
    with pytest.raises(ValueError):
        runs.output_path(inputs)
    base = storage / 'results/benchmark'
    base.mkdir(parents=True)
    (base / 'escape').symlink_to(inputs)
    with pytest.raises(ValueError):
        runs.output_path(base / 'escape/new')
    with pytest.raises(ValueError):
        runs.create_run(run_id='../input')


def test_manifest_reuse_and_foreign_checkout_rejected(storage, monkeypatch):
    path = runs.create_run(settings={'a': 1})
    monkeypatch.setenv('AG3S_RUN_DIR', str(path))
    assert runs.ensure_run(settings={'b': 2}) == path
    target = path / 'run_manifest.json'
    manifest = json.loads(target.read_text())
    assert manifest['executions'][-1]['settings'] == {'b': 2}
    manifest['checkout'] = '/other/checkout'
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runs.ensure_run(settings={})


def test_v11_server_records_actual_flags_without_gpu(storage, tmp_path):
    import os
    import subprocess
    path = runs.create_run(settings={})
    stub = tmp_path / 'python-stub'
    stub.write_text('#!/bin/sh\nexit 0\n')
    stub.chmod(0o755)
    environment = dict(os.environ, AG3S_STORAGE_ROOT=str(storage), AG3S_RUN_DIR=str(path),
                       AG3S_OPENPI_PYTHON=str(stub))
    subprocess.run(['bash', str(runs.PROJECT_ROOT / 'scripts/benchmark/v11ab/start_server.sh'),
                    'V11AB', '8286', str(path / 'runs/server')], env=environment, check=True)
    manifest = json.loads((path / 'run_manifest.json').read_text())
    execution = next(row for row in manifest['executions'] if row['settings'].get('role') == 'server')
    argv = execution['settings']['argv']
    assert argv[argv.index('--plan-horizon') + 1] == '16'
    assert argv[argv.index('--w-deflection-rate') + 1] == '20'
    assert '--servo-accel' not in argv
    assert str(path / 'runs/server/server_constraints') in argv
    assert execution['settings']['environment']['XLA_FLAGS'].endswith('--xla_gpu_autotune_level=0')
