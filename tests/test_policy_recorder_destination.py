"""Existing policy evidence is immutable; new records retain the OpenPI format."""
import numpy as np
import pytest

PolicyRecorder = pytest.importorskip("openpi.policies.policy", exc_type=ImportError).PolicyRecorder


class DummyPolicy:
    def infer(self, obs):
        return {'actions': np.array([[1, 2]], dtype=np.float32)}


def test_recorder_existing_directory_and_step_are_never_overwritten(tmp_path):
    directory = tmp_path / 'records'
    recorder = PolicyRecorder(DummyPolicy(), str(directory))
    output = recorder.infer({'state': np.zeros(2, dtype=np.float32)})
    assert output['actions'].shape == (1, 2)
    saved = np.load(directory / 'step_0.npy', allow_pickle=True).item()
    assert 'inputs/state' in saved and 'outputs/actions' in saved
    original = (directory / 'step_0.npy').read_bytes()
    with pytest.raises(FileExistsError):
        PolicyRecorder(DummyPolicy(), str(directory))
    (directory / 'step_1.npy').write_bytes(b'protected evidence')
    with pytest.raises(FileExistsError):
        recorder.infer({})
    assert (directory / 'step_1.npy').read_bytes() == b'protected evidence'
    assert (directory / 'step_0.npy').read_bytes() == original


def test_serve_record_uses_new_storage_run_and_preserves_input(tmp_path, monkeypatch):
    from third_party.openpi.scripts import serve_policy as serve
    storage = tmp_path / 'storage'
    legacy = storage / 'recordings/policy_records'
    legacy.mkdir(parents=True)
    (legacy / 'step_0.npy').write_bytes(b'original recording')
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    (cwd / 'policy_records').symlink_to(legacy)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv('AG3S_STORAGE_ROOT', str(storage))
    monkeypatch.delenv('AG3S_RUN_DIR', raising=False)
    monkeypatch.delenv('AG3S_V11_OUTPUT_DIR', raising=False)
    policy = DummyPolicy()
    policy.metadata = {}
    monkeypatch.setattr(serve, 'create_policy', lambda args: policy)
    received = []

    class Server:
        def __init__(self, **kwargs):
            received.append(kwargs['policy'])

        def serve_forever(self):
            received[-1].infer({'state': np.zeros(2, dtype=np.float32)})

    monkeypatch.setattr(serve.websocket_policy_server, 'WebsocketPolicyServer', Server)
    # main adds a file handler; close it so the test has no global logging side effects.
    before = list(serve.logging.getLogger().handlers)
    try:
        serve.main(serve.Args(record=True, port=8286))
    finally:
        for handler in list(serve.logging.getLogger().handlers):
            if handler not in before:
                serve.logging.getLogger().removeHandler(handler)
                handler.close()
    runs = list((storage / 'results/benchmark').iterdir())
    assert len(runs) == 1
    assert (runs[0] / 'policy_records/step_0.npy').is_file()
    assert (runs[0] / 'logs/serve_policy_8286.log').exists()
    assert (legacy / 'step_0.npy').read_bytes() == b'original recording'
    assert list(legacy.iterdir()) == [legacy / 'step_0.npy']
