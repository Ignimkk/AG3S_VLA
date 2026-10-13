"""SL1: LIBERO Panda camera adapter -> V11AB wire, replay tool, SL0-path preservation (CPU)."""
import ast
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmark.adapters.libero_panda import camera, wire_pack
from benchmark.runners import launch_libero_pi05_only as launcher
from benchmark.runners import launch_libero_replay as replay_launcher
from benchmark.runners import libero_pi05_client as client
from benchmark.runners import libero_replay
from common.workspace import PROJECT_ROOT

from tests.benchmark.test_libero_pi05_sl0 import FakeEnv, FakePolicy, _args, _strip_docstring, official  # noqa: F401

LIBERO_PYTHON = PROJECT_ROOT / ".venv-libero/bin/python"
PY38_CHECKS = PROJECT_ROOT / "tests/benchmark/libero_py38_checks.py"
SL0_RUN = "20261012T045129_6ba81024d243_sl0-pi05only-t0e0"


# --- wire_pack (3.8 replica) == AG3S.trajopt.wire --------------------------------------------------
def _frames(rng, cams=("head", "left_wrist")):
    frames = {}
    for k, cam in enumerate(cams):
        depth = rng.uniform(0.0, 3.0, size=(6, 5))
        depth[0, :4] = [-0.1, 0.0004999, 0.0005, 70.0]  # clip low, round down, half -> even, clip high
        frames[cam] = {"depth_m": depth, "K": rng.normal(size=(3, 3)), "T_base_cam": rng.normal(size=(4, 4)),
                       "robot_state": rng.normal(size=9), "stamp": 1.25 + k}
    return frames


def test_wire_pack_matches_ag3s_pack_request():
    from AG3S.trajopt import wire

    frames = _frames(np.random.default_rng(0))
    ours = wire_pack.pack_cameras(frames, stamp_mode="sim_frozen")
    cams = list(frames)
    theirs = wire.pack_request(
        {}, cameras=cams, depth={c: f["depth_m"] for c, f in frames.items()},
        intrinsics={c: f["K"] for c, f in frames.items()}, extrinsics={c: f["T_base_cam"] for c, f in frames.items()},
        robot_state={c: f["robot_state"] for c, f in frames.items()}, stamps={c: f["stamp"] for c, f in frames.items()},
        phase="transit", stamp_mode="sim_frozen")
    camera_keys = {k for k in theirs if k == wire.PREFIX + "cameras" or k == wire.PREFIX + wire.STAMP_MODE
                   or any(k.startswith(f"{wire.PREFIX}{field}/") for field in wire_pack.CAMERA_ARRAY_FIELDS)}
    assert set(ours) == camera_keys
    assert wire_pack.PREFIX == wire.PREFIX and wire_pack.DEPTH_SCALE_MM == wire.DEPTH_SCALE_MM
    assert wire_pack.STAMP_MODES == wire.STAMP_MODES
    for key in camera_keys:
        a, b = ours[key], theirs[key]
        assert type(a) is type(b), key
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), key
        else:
            assert a == b, key
    assert ours["ag3s/depth/head"][0, :4].tolist() == [0, 0, 0, 65535]
    with pytest.raises(ValueError):
        wire_pack.pack_cameras(frames, stamp_mode="bogus")


def test_wire_npz_roundtrip_unpacks_to_camera_observations(tmp_path):
    from AG3S.ag3s.types import CameraID
    from AG3S.trajopt import wire

    frames = _frames(np.random.default_rng(1))
    packed = wire_pack.pack_cameras(frames)
    path = tmp_path / "wire.npz"
    wire_pack.save_wire_npz(path, packed)
    with pytest.raises(FileExistsError):
        wire_pack.save_wire_npz(path, packed)
    loaded = wire_pack.load_wire_npz(path)
    assert loaded["ag3s/cameras"] == ["head", "left_wrist"] and loaded["ag3s/stamp_mode"] == "sim_frozen"
    _, scene = wire.strip_request(loaded)
    observations = wire.unpack_camera_observations(scene, robot_model=None)
    assert [o.camera_id for o in observations] == [CameraID.HEAD, CameraID.LEFT_WRIST]
    for obs, cam in zip(observations, frames):
        np.testing.assert_array_equal(obs.depth, packed[f"ag3s/depth/{cam}"].astype(np.float64) / 1000.0)
        np.testing.assert_array_equal(obs.camera_intrinsics, frames[cam]["K"])
        np.testing.assert_array_equal(obs.T_base_cam, frames[cam]["T_base_cam"])
        np.testing.assert_array_equal(obs.robot_state, frames[cam]["robot_state"])
        assert obs.timestamp == frames[cam]["stamp"]
    assert wire.unpack_stamps(scene)["mode"] == "sim_frozen"


def test_wire_from_geometry_npz_layout(tmp_path):
    rng = np.random.default_rng(2)
    n = 3
    geom = {"cameras": np.array(["head", "left_wrist"]), "stamp_mode": np.array("sim_frozen"),
            "obs_index": np.arange(n)}
    for cam in ("head", "left_wrist"):
        geom["depth_mm/" + cam] = rng.integers(0, 5000, size=(n, 4, 4)).astype(np.uint16)
        geom["K/" + cam] = rng.normal(size=(n, 3, 3))
        geom["T_base_cam/" + cam] = rng.normal(size=(n, 4, 4))
        geom["robot_state/" + cam] = rng.normal(size=(n, 9))
        geom["stamp/" + cam] = np.arange(n) * 0.05
    np.savez_compressed(tmp_path / "x_geom.npz", **geom)
    written = wire_pack.export(tmp_path / "x_geom.npz", tmp_path / "wire", "0,2")
    assert [Path(p).name for p in written] == ["x_geom_wire_0000.npz", "x_geom_wire_0002.npz"]
    loaded = wire_pack.load_wire_npz(written[1])
    np.testing.assert_array_equal(loaded["ag3s/depth/left_wrist"], geom["depth_mm/left_wrist"][2])
    assert loaded["ag3s/stamp/head"] == pytest.approx(0.1)
    with pytest.raises(FileExistsError):
        wire_pack.export(tmp_path / "x_geom.npz", tmp_path / "wire", "0")


# --- Policy image <-> OpenCV pixel ---------------------------------------------------------------
def test_policy_pixel_map_is_the_client_array_operation():
    H = W = 8
    v, u = np.mgrid[0:H, 0:W]
    opencv = np.stack([u, v], axis=-1)  # each pixel stores its own OpenCV (u, v)
    raw = opencv[::-1]  # robosuite 'opengl' rows
    policy = camera.raw_to_policy_array(raw)  # client: obs[...][::-1, ::-1]
    x, y = camera.opencv_to_policy_pixel(u, v, (H, W), (H, W))  # no resize
    assert np.array_equal(policy[y.astype(int), x.astype(int)], opencv)
    np.testing.assert_array_equal(policy, opencv[:, ::-1])  # i.e. a horizontal mirror of the OpenCV image
    uu, vv = camera.policy_to_opencv_pixel(x, y, (H, W), (H, W))
    np.testing.assert_array_equal(uu, u)
    np.testing.assert_array_equal(vv, v)
    ru, rv = camera.opencv_to_raw_pixel(u, v, H)
    assert np.array_equal(raw[rv.astype(int), ru.astype(int)], opencv)


@pytest.mark.parametrize("src,out", [((256, 256), (224, 224)), ((256, 320), (224, 224)), ((240, 180), (224, 224))])
def test_policy_pixel_map_matches_resize_with_pad(src, out):
    image_tools = pytest.importorskip("openpi_client.image_tools")
    from PIL import Image

    H, W = src
    rh, rw, ph, pw = camera.resize_with_pad_geometry(H, W, *out)
    # Coordinate ramps (x1000, int32 'I' image) through the client's resize_with_pad code. PIL's
    # antialiased bilinear filter reproduces a ramp to <= 0.064 px for these sizes; the alternative
    # "edge" convention x_out = x_in * scale would be off by 0.10-0.23 px, so 0.1 px separates them.
    u_ramp = np.tile(np.arange(W, dtype=np.int32) * 1000, (H, 1))
    v_ramp = np.tile(np.arange(H, dtype=np.int32)[:, None] * 1000, (1, W))
    for ramp, axis in ((u_ramp, "u"), (v_ramp, "v")):
        policy_src = camera.raw_to_policy_array(ramp[::-1])  # raw = opencv[::-1]
        # resize_with_pad's per-image function (its batch wrapper needs a channel axis PIL 'I' cannot take).
        resized = np.asarray(image_tools._resize_with_pad_pil(Image.fromarray(policy_src), out[0], out[1],
                                                              Image.BILINEAR), np.float64) / 1000
        ys, xs = np.mgrid[ph + 3:ph + rh - 3, pw + 3:pw + rw - 3]
        u_back, v_back = camera.policy_to_opencv_pixel(xs, ys, src, out)
        expected = u_back if axis == "u" else v_back
        np.testing.assert_allclose(resized[ys, xs], expected, atol=0.1)
    if (H, W) != out:
        x, y = camera.opencv_to_policy_pixel(np.array([0.0, W - 1.0]), np.array([0.0, H - 1.0]), src, out)
        uu, vv = camera.policy_to_opencv_pixel(x, y, src, out)
        np.testing.assert_allclose([uu, vv], [[0.0, W - 1.0], [0.0, H - 1.0]], atol=1e-9)


def test_projection_inverts_backprojection():
    from AG3S.ag3s.stages.reconstruction import backproject

    K = camera.intrinsic_matrix(45.0, 32, 32)
    R = np.array([[0.0, -1.0, 0.0], [-0.6, 0.0, -0.8], [0.8, 0.0, -0.6]])
    T = camera.pose([0.5, 0.1, 1.2], R)
    depth = np.full((32, 32), 1.3)
    cloud = backproject(depth, K, T)
    u, v, z = camera.project_base_points(cloud.points, K, T)
    np.testing.assert_allclose(np.stack([u, v], 1), cloud.uv, atol=1e-9)
    np.testing.assert_allclose(z, 1.3, atol=1e-12)
    np.testing.assert_allclose(camera.rigid_inverse(T), np.linalg.inv(T), atol=1e-12)


def test_conventions_and_camera_table():
    assert camera.CAMERA_TABLE == (("agentview", "head"), ("robot0_eye_in_hand", "left_wrist"))
    from AG3S.ag3s.types import CameraID

    for _, wire_name in camera.CAMERA_TABLE:
        CameraID.parse(wire_name)  # only names AG3S accepts
    json.dumps(camera.conventions())
    K = camera.intrinsic_matrix(45.0, 256, 256)  # robosuite restatement (reference)
    assert K[0, 0] == K[1, 1] == pytest.approx(128 / math.tan(math.radians(22.5)))
    assert (K[0, 2], K[1, 2]) == (128.0, 128.0)
    # SL2: the adapter writes the pixel-centre principal point.
    Kp = camera.pixel_centre_intrinsic_matrix(45.0, 256, 256)
    assert camera.K_CONVENTION == "pixel_centre" and camera.conventions()["K_convention"] == "pixel_centre"
    assert (Kp[0, 2], Kp[1, 2]) == (127.5, 127.5) and Kp[0, 0] == K[0, 0] and Kp[1, 1] == K[1, 1]
    Kr = camera.pixel_centre_intrinsic_matrix(60.0, 240, 320)
    assert (Kr[0, 2], Kr[1, 2]) == (159.5, 119.5)
    assert camera.MANIPULATOR_SIDE == "left" and camera.conventions()["manipulator_side"]["panda"] == "left"


# --- .venv-libero adapter checks (subprocess) + fixture round trip ---------------------------------
@pytest.fixture(scope="module")
def py38_checks(tmp_path_factory):
    if not LIBERO_PYTHON.exists():
        pytest.skip(".venv-libero not installed")
    out = tmp_path_factory.mktemp("sl1_py38")
    env = launcher.client_environment()
    completed = subprocess.run([str(LIBERO_PYTHON), str(PY38_CHECKS), "--out", str(out)], cwd=str(out), env=env,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    return out, json.loads((out / "checks.json").read_text())


@pytest.mark.parametrize("name", ["camera_table_and_keys", "K_matches_fovy_formula_and_robosuite",
                                  "T_world_cam_and_depth_match_robosuite", "T_base_cam_orthonormal_over_steps",
                                  "depth_row_order_table_plane", "client_episode_geometry_alignment",
                                  "robot_state_is_image_instant", "bddl_target_matches_libero_parser"])
def test_py38_adapter_checks(py38_checks, name):
    _, results = py38_checks
    assert results["_python"].startswith("3.8.")
    result = results[name]
    assert result["ok"], result.get("trace") or result


def test_py38_fixture_unpacks_in_ag3s(py38_checks):
    from AG3S.ag3s.types import CameraID
    from AG3S.trajopt import wire

    out, _ = py38_checks
    loaded = wire_pack.load_wire_npz(out / "wire_t0.npz")
    reference = dict(np.load(out / "reference_t0.npz"))
    _, scene = wire.strip_request(loaded)
    observations = wire.unpack_camera_observations(scene, robot_model=None)
    assert [o.camera_id for o in observations] == [CameraID.HEAD, CameraID.LEFT_WRIST]
    for obs, cam in zip(observations, ("head", "left_wrist")):
        depth_m = reference["depth_m/" + cam].astype(np.float64)
        assert obs.depth.shape == depth_m.shape == (256, 256)
        in_range = depth_m < 65.535
        assert np.max(np.abs(obs.depth[in_range] - depth_m[in_range])) <= 0.0005 + 1e-9
        np.testing.assert_array_equal(obs.camera_intrinsics, reference["K/" + cam])
        np.testing.assert_array_equal(obs.T_base_cam, reference["T_base_cam/" + cam])
        np.testing.assert_array_equal(obs.robot_state, reference["robot_state/" + cam])
    assert len({float(scene[f"stamp/{c}"]) for c in scene["cameras"]}) == 1


# --- Client: SL0 path preserved, geometry path plumbed -------------------------------------------
def test_get_libero_env_with_depth_differs_only_by_camera_depths(official):
    ours = ast.parse(Path(client.__file__).read_text())
    fn = next(n for n in ours.body if isinstance(n, ast.FunctionDef) and n.name == "_get_libero_env_with_depth")
    fn.body = [n for n in fn.body if not isinstance(n, ast.ImportFrom)]
    env_args = next(n for n in fn.body if isinstance(n, ast.Assign) and n.targets[0].id == "env_args")
    keys = [k.value for k in env_args.value.keys]
    assert keys[-1] == "camera_depths" and env_args.value.values[-1].value is True
    env_args.value.keys.pop()
    env_args.value.values.pop()
    assert _strip_docstring(fn) == _strip_docstring(official["functions"]["_get_libero_env"])


def test_geometry_flags_default_off_and_dependencies():
    args = client.build_parser().parse_args(["--run-dir", "x", "--port", "1"])
    assert (args.capture_geometry, args.capture_segmentation, args.capture_float_depth) == (False, False, False)
    client.check_geometry_arguments(args)
    for flag in ("--capture-segmentation", "--capture-float-depth"):
        with pytest.raises(ValueError):
            client.check_geometry_arguments(client.build_parser().parse_args(["--run-dir", "x", "--port", "1", flag]))


def test_run_episode_observer_sees_every_observation():
    pytest.importorskip("openpi_client.image_tools")
    seen = []
    env = FakeEnv(done_at=10 + 7)
    summary, rec, _ = client.run_episode(env, [None], 0, "t", FakePolicy(), _args(), max_steps=30,
                                         observer=lambda e, o: seen.append(int(o["robot0_joint_pos"][0])))
    assert summary["status"] == "success"
    assert seen == [int(x[0]) for x in rec.obs["robot0_joint_pos"]] == list(range(18))


class FakeSuite:
    def get_task(self, task_id):
        return type("Task", (), {"language": "pick up the bowl"})()

    def get_task_init_states(self, task_id):
        return [None, None]


class FakeGeometry:
    def __init__(self, segmentation=False, float_depth=False):
        self.frames, self.segmentation, self.float_depth = [], segmentation, float_depth
        self.camera = camera

    def observe(self, env, obs):
        self.frames.append(obs["robot0_joint_pos"][0])

    def arrays(self):
        return {"frames": np.asarray(self.frames)}


def _run_tasks(tmp_path, monkeypatch, *flags):
    pytest.importorskip("openpi_client.image_tools")
    run_dir = tmp_path / "run"
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "videos").mkdir()
    made = []
    monkeypatch.setattr(client, "_get_libero_env", lambda task, res, seed: (made.append("rgb") or FakeEnv(12), task.language))
    monkeypatch.setattr(client, "_get_libero_env_with_depth",
                        lambda task, res, seed: (made.append("depth") or FakeEnv(12), task.language))
    monkeypatch.setattr(client, "_save_video", lambda frames, final: final.write_bytes(b"mp4"))
    monkeypatch.setattr(client, "_GeometryRecorder", FakeGeometry)
    monkeypatch.setattr(client, "_geometry_adapter", lambda: (type("C", (), {"install_capture_observable": staticmethod(
        lambda env: made.append("observable"))}), wire_pack))
    args = client.build_parser().parse_args(["--run-dir", "x", "--port", "1", *flags])
    with open(run_dir / "episodes.jsonl", "x") as jsonl:
        errors = client.run_tasks(args, run_dir, jsonl, FakeSuite(), 30,
                                  lambda t, e: (type("P", (FakePolicy,), {"get_server_metadata": lambda self: {"m": 1}})(),
                                                "fake", {}))
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    return errors, rows, sorted(p.name for p in (run_dir / "episodes").iterdir()), made


SL0_ROW_KEYS = {"suite", "task_id", "task_description", "episode_idx", "init_state_index", "seed", "official_success",
                "status", "error", "env_steps_including_wait", "env_steps_excluding_wait", "wait_steps", "loop_t_final",
                "max_steps", "num_steps_wait", "policy_calls", "policy_latency_ms", "server_timing", "replay_frames",
                "started_at", "ended_at", "npz", "video", "server_metadata", "client_connector"}


def test_run_tasks_geometry_off_writes_exactly_the_sl0_outputs(tmp_path, monkeypatch):
    errors, rows, files, made = _run_tasks(tmp_path, monkeypatch)
    assert errors == 0 and made == ["rgb"]
    assert files == ["libero_spatial_t0_e0.npz"]
    assert set(rows[0]) == SL0_ROW_KEYS


def test_run_tasks_geometry_on_writes_geom_npz(tmp_path, monkeypatch):
    errors, rows, files, made = _run_tasks(tmp_path, monkeypatch, "--capture-geometry")
    assert errors == 0 and made == ["depth", "observable"]
    assert files == ["libero_spatial_t0_e0.npz", "libero_spatial_t0_e0_geom.npz"]
    assert set(rows[0]) == SL0_ROW_KEYS | {"geometry"}
    assert rows[0]["geometry"]["npz"] == "episodes/libero_spatial_t0_e0_geom.npz"
    assert rows[0]["geometry"]["frames"] == rows[0]["env_steps_including_wait"] + 1
    assert rows[0]["geometry"]["cameras"] == [{"robosuite": "agentview", "wire": "head"},
                                              {"robosuite": "robot0_eye_in_hand", "wire": "left_wrist"}]


# --- Replay ---------------------------------------------------------------------------------------
def _recorded_episode(done_at=10 + 12):
    pytest.importorskip("openpi_client.image_tools")
    env, policy = FakeEnv(done_at), FakePolicy()
    summary, rec, _ = client.run_episode(env, [None], 0, "t", policy, _args(), max_steps=30)
    return summary, rec.arrays(), env.actions


def test_replay_policy_reproduces_the_recorded_episode():
    summary, source, actions = _recorded_episode()
    libero_replay.check_replayable(source, 5)
    np.testing.assert_array_equal(libero_replay.executed_actions_from_chunks(source, 5), source["action"])
    env = FakeEnv(10 + 12)
    policy = libero_replay.ReplayPolicy(source["action_chunks"], {"replay_of": "x"})
    replay_summary, rec, _ = client.run_episode(env, [None], 0, "t", policy, _args(), max_steps=30)
    assert env.actions == actions  # identical python floats reach env.step
    assert replay_summary["status"] == summary["status"] and policy.calls == len(source["action_chunks"])
    for key, value in rec.arrays().items():
        np.testing.assert_array_equal(value, source[key], err_msg=key)


def test_replay_refuses_inconsistent_records_and_never_invents_actions():
    _, source, _ = _recorded_episode()
    tampered = dict(source, action=source["action"].copy())
    tampered["action"][13, 0] += 1e-12
    with pytest.raises(ValueError):
        libero_replay.check_replayable(tampered, 5)
    with pytest.raises(ValueError):
        libero_replay.check_replayable(source, 4)  # a chunk would be used for 5 actions
    env = FakeEnv(done_at=None)  # the replay does not finish where the source did
    policy = libero_replay.ReplayPolicy(source["action_chunks"], {})
    summary, _, _ = client.run_episode(env, [None], 0, "t", policy, _args(), max_steps=30)
    assert summary["status"] == "execution_error" and "replay exhausted" in summary["error"]


def test_sl0_record_is_replayable():
    from common.workspace import shared_storage_root

    npz = shared_storage_root() / "results/benchmark" / SL0_RUN / "episodes/libero_spatial_t0_e0.npz"
    if not npz.is_file():
        pytest.skip("SL0 run not present in this storage")
    with np.load(npz) as data:
        source = {key: data[key] for key in ("action", "action_chunks", "chunk_index", "is_wait", "done")}
    libero_replay.check_replayable(source, client.DEFAULT_REPLAN_STEPS)


@pytest.fixture
def fake_source_run(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("AG3S_STORAGE_ROOT", str(storage))
    source = storage / "results/benchmark/20261012T000000_abc_src"
    (source / "episodes").mkdir(parents=True)
    _, arrays, _ = _recorded_episode()
    np.savez_compressed(source / "episodes/libero_spatial_t0_e0.npz", **arrays)
    (source / "run_manifest.json").write_text(json.dumps({"commit": "abc", "settings": {
        "condition": "pi0.5-only", "protocol": {"suite": "libero_spatial", "seed": 7, "num_steps_wait": 10,
                                                  "task_ids": [0], "episode_ids": [0]}}}))
    row = {"suite": "libero_spatial", "task_id": 0, "episode_idx": 0, "seed": 7, "num_steps_wait": 10,
           "status": "success", "npz": "episodes/libero_spatial_t0_e0.npz", "policy_calls": 3}
    (source / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    return storage, source


def test_replay_launcher_dry_run(fake_source_run, monkeypatch, capsys):
    storage, source = fake_source_run
    monkeypatch.setattr(launcher, "interpreter_info", lambda python, packages: {"executable": str(python)})
    assert replay_launcher.main(["--source-run", source.name, "--capture-geometry", "--dry-run"]) == 0
    settings = json.loads(capsys.readouterr().out)
    assert settings["step"] == "SL1" and settings["ag3s"]["enabled"] is False
    rs = settings["replay_source"]
    assert rs["run_id"] == source.name and rs["run_dir"] == str(source.resolve())
    assert rs["episodes"][0]["npz_sha256"] == launcher.sha256_file(source / "episodes/libero_spatial_t0_e0.npz")
    cmd = settings["commands"]["replay"]
    assert cmd[0] == str(PROJECT_ROOT / ".venv-libero/bin/python") and cmd[1].endswith("benchmark/runners/libero_replay.py")
    parsed = libero_replay.build_parser().parse_args(cmd[2:])
    assert parsed.capture_geometry and not parsed.capture_segmentation and parsed.source_run_dir == str(source.resolve())
    assert settings["geometry_capture"]["robot_model"]["manipulator_side"] == "left"
    assert settings["geometry_capture"]["conventions"]["K_convention"] == "pixel_centre"
    assert settings["geometry_capture"]["conventions"]["camera_table"][1] == {"robosuite": "robot0_eye_in_hand",
                                                                              "wire": "left_wrist"}
    assert settings["runtime_schedule"]["control_freq_hz"] == 20
    assert sorted(p.name for p in (storage / "results/benchmark").iterdir()) == [source.name]  # nothing allocated
    with pytest.raises(ValueError):
        replay_launcher.main(["--source-run", source.name, "--seed", "8", "--dry-run"])
    with pytest.raises(ValueError):
        replay_launcher.main(["--source-run", source.name, "--episode-ids", "1", "--dry-run"])


def test_replay_validates_source_and_protocol(fake_source_run):
    storage, source = fake_source_run
    new = storage / "results/benchmark/20261012T000001_abc_new"
    new.mkdir()
    (new / "run_manifest.json").write_text("{}")
    args = libero_replay.build_parser().parse_args(["--run-dir", str(new), "--source-run-dir", str(source), "--seed", "8"])
    with pytest.raises(ValueError):
        libero_replay.replay(args)  # seed differs from the source episode
    args = libero_replay.build_parser().parse_args(["--run-dir", str(source), "--source-run-dir", str(source)])
    with pytest.raises(ValueError):
        libero_replay.replay(args)  # never write into the source run
    assert not (new / "episodes.jsonl").exists()


# --- Launcher manifest (D.9) ----------------------------------------------------------------------
def test_launcher_manifest_records_schedule_and_not_applicable(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "interpreter_info", lambda python, packages: {"executable": str(python)})
    args = launcher.build_parser().parse_args(["--capture-geometry", "--capture-segmentation"])
    cmd = launcher.client_command(Path("py"), "<run>", 1234, args)
    assert cmd[-2:] == ["--capture-geometry", "--capture-segmentation"]
    client.build_parser().parse_args(cmd[2:])
    settings = launcher.build_settings(args, 1234, {}, ["server"], cmd, probe=False)
    assert settings["step"] == "SL1" and settings["geometry_capture"]["capture_segmentation"] is True
    schedule = settings["runtime_schedule"]
    assert schedule["control_freq_hz"] == 20 and schedule["replan_steps"] == 5
    assert schedule["policy_call_period_sim_s"] == pytest.approx(0.25)
    items = settings["not_applicable"]["items"]
    assert {"protected_links", "authorized_links", "phase", "level"} <= set(items)
    assert set(items.values()) == {"n/a"}
    off = launcher.build_parser().parse_args([])
    assert launcher.client_command(Path("py"), "<run>", 1, off)[-2:] == ["--seed", "7"]
    assert launcher.build_settings(off, 1, {}, [], [], probe=False)["step"] == "SL0"
    wrapper = tmp_path / "env_wrapper.py"
    wrapper.write_text("class ControlEnv:\n    def __init__(self, a, control_freq=15, horizon=1):\n        pass\n")
    assert launcher.libero_control_freq(wrapper) == 15


# --- serve_policy manifest entry (D.9) ------------------------------------------------------------
def test_serve_policy_records_the_served_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("openpi.policies.policy", exc_type=ImportError)
    from third_party.openpi.scripts import serve_policy as serve

    storage = tmp_path / "storage"
    monkeypatch.setenv("AG3S_STORAGE_ROOT", str(storage))
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    policy = FakePolicy()
    policy.metadata = {}
    monkeypatch.setattr(serve, "create_policy", lambda args: policy)

    class Server:
        def __init__(self, **kwargs):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr(serve.websocket_policy_server, "WebsocketPolicyServer", Server)
    before = list(serve.logging.getLogger().handlers)
    try:
        serve.main(serve.Args(port=8287, policy=serve.Checkpoint(config="pi05_libero", dir="/ckpt/pi05_libero")))
        serve.main(serve.Args(port=8288, env=serve.EnvMode.LIBERO))
    finally:
        for handler in list(serve.logging.getLogger().handlers):
            if handler not in before:
                serve.logging.getLogger().removeHandler(handler)
                handler.close()
    manifests = sorted((storage / "results/benchmark").glob("*/run_manifest.json"))
    settings = [json.loads(p.read_text())["settings"] for p in manifests]
    by_port = {s["port"]: s for s in settings}
    assert by_port[8287]["env"] is None
    assert by_port[8287]["policy"] == {"source": "checkpoint", "config": "pi05_libero", "dir": "/ckpt/pi05_libero"}
    assert by_port[8288]["env"] == "libero"
    assert by_port[8288]["policy"] == {"source": "default", "config": "pi05_libero",
                                       "dir": "gs://openpi-assets/checkpoints/pi05_libero"}
