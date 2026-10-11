"""Policy recording preserves existing evidence across restarts and concurrent servers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('flax')
from openpi.policies.policy import PolicyRecorder
from common.run_output import allocate_run_dir


class EchoPolicy:
    metadata = {'test': True}

    def infer(self, obs):
        return {'actions': np.array([[obs['value']]], dtype=np.float32)}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('AG3S_STORAGE_ROOT', str(tmp_path / 'storage'))
    monkeypatch.delenv('AG3S_RUN_DIR', raising=False)
    return allocate_run_dir()


def test_restarts_and_servers_preserve_step_zero(run_dir):
    first = PolicyRecorder(EchoPolicy(), 'policy_records', run_dir=run_dir)
    first.infer({'value': 1})
    original = (first.record_dir / 'step_0.npy').read_bytes()
    second = PolicyRecorder(EchoPolicy(), 'policy_records', run_dir=run_dir)
    second.infer({'value': 2})
    assert first.record_dir != second.record_dir
    assert (first.record_dir / 'step_0.npy').read_bytes() == original
    assert np.load(second.record_dir / 'step_0.npy', allow_pickle=True).item()['inputs/value'] == 2
    assert first.metadata == {'test': True}


def test_parallel_calls_keep_each_record(run_dir):
    recorder = PolicyRecorder(EchoPolicy(), run_dir=run_dir)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: recorder.infer({'value': i}), range(16)))
    records = list(recorder.record_dir.glob('step_*.npy'))
    assert len(records) == 16
    assert {np.load(p, allow_pickle=True).item()['inputs/value'] for p in records} == set(range(16))


def test_historical_directory_rejected_before_writes(run_dir, tmp_path):
    historical = tmp_path / 'historical'
    historical.mkdir()
    existing = historical / 'step_0.npy'
    existing.write_bytes(b'original evidence')
    with pytest.raises(ValueError, match='Output must be inside'):
        PolicyRecorder(EchoPolicy(), str(historical), run_dir=run_dir)
    assert existing.read_bytes() == b'original evidence'
    assert list(historical.iterdir()) == [existing]


def test_checkpoint_output_confined_to_run(run_dir):
    from openpi.training.config import TrainConfig
    config = TrainConfig(name='test', exp_name='new')
    assert config.checkpoint_dir == run_dir / 'checkpoints/test/new'
    with pytest.raises(ValueError, match='Output must be inside'):
        TrainConfig(name='test', exp_name='new', checkpoint_base_dir='/mnt/dev/work/checkpoints').checkpoint_dir
    with pytest.raises(ValueError, match='Output must be inside'):
        TrainConfig(name='../../../../recordings', exp_name='new').checkpoint_dir


def test_norm_stats_output_never_writes_input_assets(run_dir, monkeypatch):
    from types import SimpleNamespace
    from third_party.openpi.scripts import compute_norm_stats
    observed = {}
    dataset = SimpleNamespace(rlds_data_dir=None, repo_id='local/test')
    config = SimpleNamespace(name='test', assets_dirs=Path('/read-only-input-assets'),
                             model=SimpleNamespace(action_horizon=1), batch_size=1, num_workers=0,
                             data=SimpleNamespace(create=lambda *args: dataset))
    monkeypatch.setattr(compute_norm_stats._config, 'get_config', lambda name: config)
    monkeypatch.setattr(compute_norm_stats, 'create_torch_dataloader',
                        lambda *args: ([{'state': np.ones((2, 2)), 'actions': np.ones((2, 2))}], 1))
    monkeypatch.setattr(compute_norm_stats.normalize, 'save',
                        lambda path, stats: observed.update(path=path))
    compute_norm_stats.main('test')
    assert observed['path'] == run_dir / 'assets/test/local/test'


def test_norm_stats_input_destination_rejected_before_dataset(run_dir):
    from third_party.openpi.scripts import compute_norm_stats
    with pytest.raises(ValueError, match='Output must be inside'):
        compute_norm_stats.main('missing_config', output_dir='/mnt/dev/AG3S_VLA-storage/assets/openpi')


def test_serve_record_flag_routes_into_selected_run(run_dir, monkeypatch):
    from third_party.openpi.scripts import serve_policy
    observed = {}

    class Server:
        def __init__(self, *, policy, **kwargs):
            observed['policy'] = policy

        def serve_forever(self):
            observed['policy'].infer({'value': 3})

    monkeypatch.setattr(serve_policy, 'create_policy', lambda args: EchoPolicy())
    monkeypatch.setattr(serve_policy.websocket_policy_server, 'WebsocketPolicyServer', Server)
    monkeypatch.setattr(serve_policy.socket, 'gethostbyname', lambda name: '127.0.0.1')
    serve_policy.main(serve_policy.Args(record=True, run_dir=run_dir))
    policy = observed['policy']
    assert isinstance(policy, PolicyRecorder)
    assert policy.record_dir.parent == run_dir / 'policy_records'
    assert (policy.record_dir / 'step_0.npy').is_file()
    assert list((run_dir / 'logs').glob('serve_policy_*.log'))


def test_serve_rejects_input_record_path_before_model_loading(run_dir, monkeypatch):
    from third_party.openpi.scripts import serve_policy
    monkeypatch.setattr(serve_policy, 'create_policy', lambda args: pytest.fail('must reject before model loading'))
    with pytest.raises(ValueError, match='Output must be inside'):
        serve_policy.main(serve_policy.Args(record=True, run_dir=run_dir,
                                            record_dir=Path('/mnt/dev/AG3S_VLA-storage/recordings/policy_records')))
