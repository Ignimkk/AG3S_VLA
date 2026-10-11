"""SL0: pi0.5-only LIBERO client/launcher -- CPU checks against the official OpenPI example."""
import ast
import json
import math
from pathlib import Path
import socket

import numpy as np
import pytest

from benchmark.runners import libero_pi05_client as client
from benchmark.runners import launch_libero_pi05_only as launcher
from common.workspace import PROJECT_ROOT

OFFICIAL = PROJECT_ROOT / "third_party/openpi/examples/libero/main.py"


# --- Official example parsed by AST ---------------------------------------------------------------
@pytest.fixture(scope="module")
def official():
    tree = ast.parse(OFFICIAL.read_text())
    out = {"functions": {}, "args": {}, "max_steps": {}, "fps": None}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = eval(compile(ast.Expression(node.value), "main.py", "eval"), {"__builtins__": {}})
        elif isinstance(node, ast.FunctionDef):
            out["functions"][node.name] = node
        elif isinstance(node, ast.ClassDef) and node.name == "Args":
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    out["args"][item.target.id] = ast.literal_eval(item.value)
    for node in ast.walk(out["functions"]["eval_libero"]):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            suite = node.test.comparators[0]
            assign = node.body[0]
            if isinstance(suite, ast.Constant) and isinstance(assign, ast.Assign) and assign.targets[0].id == "max_steps":
                out["max_steps"][suite.value] = assign.value.value
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "mimwrite":
            out["fps"] = next(k.value.value for k in node.keywords if k.arg == "fps")
    return out


def test_protocol_constants_match_official(official):
    assert client.LIBERO_DUMMY_ACTION == official["LIBERO_DUMMY_ACTION"] == [0.0] * 6 + [-1.0]
    assert client.LIBERO_ENV_RESOLUTION == official["LIBERO_ENV_RESOLUTION"] == 256
    args = official["args"]
    assert client.DEFAULT_HOST == args["host"]
    assert client.DEFAULT_RESIZE_SIZE == args["resize_size"] == 224
    assert client.DEFAULT_REPLAN_STEPS == args["replan_steps"] == 5
    assert client.DEFAULT_TASK_SUITE == args["task_suite_name"] == "libero_spatial"
    assert client.DEFAULT_NUM_STEPS_WAIT == args["num_steps_wait"] == 10
    assert client.DEFAULT_SEED == args["seed"] == 7
    assert client.MAX_STEPS == official["max_steps"]
    assert client.MAX_STEPS["libero_spatial"] == 220
    assert client.VIDEO_FPS == official["fps"] == 10


def _strip_docstring(function):
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def test_quat2axisangle_source_and_values_match_official(official):
    ours = ast.parse(Path(client.__file__).read_text())
    ours_fn = next(n for n in ours.body if isinstance(n, ast.FunctionDef) and n.name == "_quat2axisangle")
    official_fn = official["functions"]["_quat2axisangle"]
    assert _strip_docstring(ours_fn) == _strip_docstring(official_fn)
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=[official_fn], type_ignores=[]), "main.py", "exec"), namespace)
    reference = namespace["_quat2axisangle"]
    rng = np.random.default_rng(0)
    quats = [q / np.linalg.norm(q) for q in rng.normal(size=(64, 4))]
    quats += [np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0000001]),
              np.array([0.0, 0.0, 1e-9, -1.0000001]), np.array([1.0, 0.0, 0.0, 0.0])]
    for quat in quats:
        a, b = quat.copy(), quat.copy()
        np.testing.assert_array_equal(client._quat2axisangle(a), reference(b))
        np.testing.assert_array_equal(a, b)  # same in-place clipping side effect


def test_get_libero_env_matches_official_apart_from_deferred_imports(official):
    ours = ast.parse(Path(client.__file__).read_text())
    ours_fn = next(n for n in ours.body if isinstance(n, ast.FunctionDef) and n.name == "_get_libero_env")
    ours_fn.body = [n for n in ours_fn.body if not isinstance(n, ast.ImportFrom)]
    assert _strip_docstring(ours_fn) == _strip_docstring(official["functions"]["_get_libero_env"])


# --- Run-dir rules --------------------------------------------------------------------------------
@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("AG3S_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    base = tmp_path / "storage/results/benchmark"
    base.mkdir(parents=True)
    return base


def test_storage_root_agrees_with_workspace(storage):
    from common.workspace import shared_storage_root
    assert client.storage_root() == shared_storage_root()
    assert client.results_root() == storage.resolve()


def test_run_dir_rejection_rules(storage, tmp_path):
    good = storage / "20261012T000000_abc_run"
    good.mkdir()
    (good / "run_manifest.json").write_text("{}")
    assert client.validate_run_dir(good) == good.resolve()
    no_manifest = storage / "no_manifest"
    no_manifest.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "run_manifest.json").write_text("{}")
    escape = storage / "escape"
    escape.symlink_to(outside)
    inputs = tmp_path / "storage/checkpoints/x"
    inputs.mkdir(parents=True)
    (inputs / "run_manifest.json").write_text("{}")
    for bad in [storage, no_manifest, outside, escape, inputs, storage / "missing", storage / "../benchmark"]:
        with pytest.raises(ValueError):
            client.validate_run_dir(bad)
    with pytest.raises(SystemExit):
        client.build_parser().parse_args(["--port", "1"])  # --run-dir is required


def test_libero_config_required(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)
    with pytest.raises(RuntimeError):
        client.require_libero_config()
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    with pytest.raises(RuntimeError):
        client.require_libero_config()
    (tmp_path / "config.yaml").write_text("bddl_files: x\n")
    assert client.require_libero_config() == tmp_path / "config.yaml"


def test_exclusive_publish_never_overwrites(tmp_path):
    final = tmp_path / "a.npz"
    final.write_bytes(b"evidence")
    tmp = client._tmp_sibling(final, ".npz")
    tmp.write_bytes(b"new")
    with pytest.raises(FileExistsError):
        client._exclusive_publish(tmp, final)
    assert final.read_bytes() == b"evidence"
    assert not tmp.exists()


# --- Episode loop with fake env / policy (no LIBERO, no server) ------------------------------------
class FakeEnv:
    def __init__(self, done_at=None, raise_at=None):
        self.done_at, self.raise_at, self.calls, self.actions = done_at, raise_at, 0, []

    def _obs(self):
        value = self.calls
        return {"agentview_image": np.full((256, 256, 3), value % 255, np.uint8),
                "robot0_eye_in_hand_image": np.full((256, 256, 3), 7, np.uint8),
                "robot0_eef_pos": np.array([0.1, 0.2, 0.3]) + value,
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_joint_pos": np.zeros(7) + value, "robot0_gripper_qpos": np.array([0.04, -0.04])}

    def reset(self):
        self.calls = 0

    def set_init_state(self, state):
        return self._obs()

    def step(self, action):
        self.calls += 1
        if self.raise_at is not None and self.calls == self.raise_at:
            raise RuntimeError("boom")
        self.actions.append(list(action))
        done = self.done_at is not None and self.calls == self.done_at
        return self._obs(), float(done), done, {}


class FakePolicy:
    def __init__(self, horizon=10):
        self.horizon, self.calls = horizon, []

    def infer(self, element):
        self.calls.append(element)
        return {"actions": np.tile(np.arange(7.0), (self.horizon, 1)) + len(self.calls),
                "server_timing": {"infer_ms": 1.0}}


def _args(**kw):
    args = client.build_parser().parse_args(["--run-dir", "x", "--port", "1"])
    for key, value in kw.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("done_at,raise_at,status,steps", [
    (10 + 12, None, "success", 22), (None, None, "max_steps", 10 + 30), (None, 10 + 3, "execution_error", 12)])
def test_run_episode_status_and_counts(done_at, raise_at, status, steps):
    pytest.importorskip("openpi_client.image_tools")
    env, policy = FakeEnv(done_at, raise_at), FakePolicy()
    summary, rec, frames = client.run_episode(env, [None], 0, "pick up the bowl", policy, _args(), max_steps=30)
    assert summary["status"] == status
    assert summary["official_success"] == (status == "success")
    assert (summary["error"] is not None) == (status == "execution_error")
    assert summary["env_steps_including_wait"] == steps
    assert summary["wait_steps"] == 10
    assert summary["env_steps_excluding_wait"] == steps - 10
    policy_steps = steps - 10 + (status == "execution_error")
    assert summary["policy_calls"] == len(policy.calls) == math.ceil(policy_steps / 5)
    assert len(summary["policy_latency_ms"]) == summary["policy_calls"]
    assert env.actions[:10] == [client.LIBERO_DUMMY_ACTION] * 10
    arrays = rec.arrays()
    assert arrays["action"].shape == (steps, 7)
    assert arrays["obs_agentview_image"].shape == (steps + 1, 256, 256, 3)
    assert arrays["action_chunks"].shape == (summary["policy_calls"], 10, 7)
    assert arrays["is_wait"].sum() == 10 and (arrays["chunk_index"][:10] == -1).all()
    # executed action == first replan_steps of the chunk that produced it
    for i in np.flatnonzero(~arrays["is_wait"]):
        chunk = arrays["action_chunks"][arrays["chunk_index"][i]]
        np.testing.assert_array_equal(arrays["action"][i], chunk[(i - 10) % 5])
    first = policy.calls[0]
    assert first["observation/image"].shape == (224, 224, 3) and first["observation/image"].dtype == np.uint8
    assert first["observation/state"].shape == (8,)
    assert first["prompt"] == "pick up the bowl"
    assert len(frames) == summary["replay_frames"]


# --- Port selection -------------------------------------------------------------------------------
def test_port_selection_excludes_forbidden_and_busy():
    with pytest.raises(ValueError):
        launcher.choose_port(8123)
    with socket.socket() as busy:
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        with pytest.raises(ValueError):
            launcher.choose_port(taken)
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", 0))
            free = probe.getsockname()[1]
        assert launcher.choose_port(candidates=[8123, taken, free]) == free
        with pytest.raises(RuntimeError):
            launcher.choose_port(candidates=[8123, taken])
    chosen = launcher.choose_port()
    assert chosen != 8123 and launcher.port_is_free(chosen)


# --- Checkpoint identity and dry-run --------------------------------------------------------------
@pytest.fixture
def checkpoint(storage):
    root = storage.parent.parent / "checkpoints/panda/pi05_libero"
    norm = root / launcher.NORM_STATS_RELATIVE
    norm.parent.mkdir(parents=True)
    norm.write_text('{"norm_stats": {}}')
    (root / "params").mkdir()
    (root / "params/x").write_bytes(b"1234")
    records = [{"path": launcher.NORM_STATS_RELATIVE, "bytes": norm.stat().st_size, "sha256": launcher.sha256_file(norm)},
               {"path": "params/x", "bytes": 4, "sha256": launcher.sha256_file(root / "params/x")}]
    lines = [f"{r['sha256']}  {r['path']}\n" for r in sorted(records, key=lambda r: r["path"])]
    aggregate = __import__("hashlib").sha256("".join(lines).encode()).hexdigest()
    (root / "CHECKSUMS.json").write_text(json.dumps({"files": 2, "aggregate_sha256": aggregate, "records": records}))
    return root


def test_aggregate_uses_path_order_of_producer():
    records = {"b": {"sha256": "0" * 64}, "a": {"sha256": "f" * 64}}
    by_path = "".join(f"{records[p]['sha256']}  {p}\n" for p in ["a", "b"])
    assert launcher.aggregate_sha256(records) == __import__("hashlib").sha256(by_path.encode()).hexdigest()
    by_line = "".join(sorted(f"{r['sha256']}  {p}\n" for p, r in records.items()))
    assert launcher.aggregate_sha256(records) != __import__("hashlib").sha256(by_line.encode()).hexdigest()


def _aggregate(root):
    return json.loads((root / "CHECKSUMS.json").read_text())["aggregate_sha256"]


def test_checkpoint_identity_rules(checkpoint):
    info = launcher.checkpoint_identity(checkpoint, rehash=True)
    assert info["aggregate_sha256"] == info["aggregate_recomputed_from_records"] == _aggregate(checkpoint)
    assert info["norm_stats_matches_checksums"] is True and info["size_or_hash_mismatch_or_missing"] == []
    (checkpoint / "params/x").write_bytes(b"12345")  # size differs
    with pytest.raises(ValueError):
        launcher.checkpoint_identity(checkpoint)
    (checkpoint / "params/x").write_bytes(b"4321")  # same size, different content: only a rehash sees it
    launcher.checkpoint_identity(checkpoint)
    with pytest.raises(ValueError):
        launcher.checkpoint_identity(checkpoint, rehash=True)
    (checkpoint / "params/x").write_bytes(b"1234")
    data = json.loads((checkpoint / "CHECKSUMS.json").read_text())
    (checkpoint / "CHECKSUMS.json").write_text(json.dumps(dict(data, aggregate_sha256="0" * 64)))
    with pytest.raises(ValueError):
        launcher.checkpoint_identity(checkpoint)
    (checkpoint / "CHECKSUMS.json").write_text(json.dumps(data))
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    partial.mkdir()
    with pytest.raises(ValueError):
        launcher.checkpoint_identity(checkpoint)
    partial.rmdir()
    (checkpoint / "CHECKSUMS.json").unlink()
    with pytest.raises(FileNotFoundError):
        launcher.checkpoint_identity(checkpoint)
    assert "error" in launcher.checkpoint_identity(checkpoint, require=False)


def test_dry_run_commands_and_settings(checkpoint, storage, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "interpreter_info", lambda python, packages: {"executable": str(python)})
    assert launcher.main(["--dry-run", "--checkpoint-dir", str(checkpoint)]) == 0
    settings = json.loads(capsys.readouterr().out)
    server, client_cmd = settings["commands"]["server"], settings["commands"]["client"]
    port = settings["port"]["selected"]
    assert port != 8123 and settings["port"]["forbidden"] == [8123]
    assert server[0].endswith(".venv-openpi-live/bin/python") and server[1].endswith("scripts/serve_policy.py")
    assert server[2:] == ["--port", str(port), "--record", "policy:checkpoint", "--policy.config", "pi05_libero",
                          "--policy.dir", str(checkpoint.resolve())]
    assert "--env" not in server and not any(arg.startswith("gs://") for arg in server)
    assert client_cmd[0] == str(PROJECT_ROOT / ".venv-libero/bin/python")
    assert client_cmd[1] == str(PROJECT_ROOT / "benchmark/runners/libero_pi05_client.py")
    flags = dict(zip(client_cmd[2::2], client_cmd[3::2]))
    assert flags["--port"] == str(port) and flags["--task-suite-name"] == "libero_spatial"
    assert flags["--task-ids"] == "0" and flags["--episode-ids"] == "0" and flags["--seed"] == "7"
    assert flags["--run-dir"].startswith(str(storage.resolve()))
    client.build_parser().parse_args(client_cmd[2:])  # the client accepts exactly these arguments
    assert settings["condition"] == "pi0.5-only" and settings["ag3s"]["enabled"] is False
    assert settings["privileged_inputs"] == "none"
    protocol = settings["protocol"]
    assert (protocol["num_steps_wait"], protocol["replan_steps"], protocol["resize_size"],
            protocol["render_resolution"], protocol["max_steps"], protocol["seed"]) == (10, 5, 224, 256, 220, 7)
    assert protocol["dummy_action"] == [0.0] * 6 + [-1.0]
    assert settings["checkpoint"]["aggregate_sha256"] == _aggregate(checkpoint)
    assert settings["checkpoint"]["norm_stats_sha256"] == launcher.sha256_file(checkpoint / launcher.NORM_STATS_RELATIVE)
    assert settings["environment"]["MUJOCO_GL"]
    assert list(storage.iterdir()) == []  # dry-run allocates no run


def test_client_environment_drops_checkout_pythonpath_and_foreign_torch_libs():
    own = str(PROJECT_ROOT / ".venv-libero/lib/python3.8/site-packages/x")
    env = launcher.client_environment({
        "PYTHONPATH": "/x", "MUJOCO_GL": "",
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/compat/lib:"
                           "/opt/py/lib/python3.11/site-packages/foo:" + own + ":/usr/local/nvidia/lib64"})
    assert "PYTHONPATH" not in env and env["MUJOCO_GL"] == "osmesa"
    assert env["LD_LIBRARY_PATH"] == "/usr/local/cuda/compat/lib:" + own + ":/usr/local/nvidia/lib64"
    assert all(env[key] == "4" for key in launcher.THREAD_LIMIT_KEYS)
    assert launcher.client_environment({"OMP_NUM_THREADS": "2"})["OMP_NUM_THREADS"] == "2"


def test_recorded_requirements_are_the_official_copies():
    record = PROJECT_ROOT / "environments/libero"
    if not (record / "requirements.txt").exists():
        pytest.skip("environments/libero not generated yet")
    assert (record / "requirements.txt").read_bytes() == (OFFICIAL.parent / "requirements.txt").read_bytes()
    assert (record / "libero-requirements.txt").read_bytes() == (
        PROJECT_ROOT / "benchmark/LIBERO/requirements.txt").read_bytes()
