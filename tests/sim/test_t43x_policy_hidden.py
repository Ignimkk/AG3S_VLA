"""T43 X — policy-blind obstacle (`pi05_infer --obstacle-policy-hidden`) and the obstacle-scene statistic fix.

What this file pins:

1. **Statistic fix.** `model_transport_pick_place_obstacles.xml` / `model_pick_place_obstacles.xml` pin
   `<statistic extent center>` to the plain model's compiled values (bit for bit).  Nothing else of the model
   changes (every numeric `MjModel` array = the unpinned compile; `meaninertia` / `meanmass` / `meansize` too).
   With every slot parked the obstacle scene renders **pixel-identically** to the plain scene: the 3 policy
   cameras (`render_cam`, the 299x224 -> 224x224 path) and the 3 AG3S depth images (`TransportScene.capture`),
   transport (16-D ep1968) and block scene (teleop).  The unpinned compile does differ (the test can see it).
2. **Hidden mode, render level.** With a pick obstacle (`post_0` beside the apple) and the place hurdle placed:
   the visible policy RGB differs from the plain scene; the hidden policy RGB equals the plain scene pixel for
   pixel on all 3 cameras (so no shadow / reflection of the hidden geoms is left: they are not in the
   `mjvScene`); `geom_group` is restored (also on an exception); the AG3S depth taken after the hidden render is
   the obstacle-present depth bit for bit; the third-person record frame still shows the obstacle; contacts with
   the obstacle's collision geoms are the same inside and outside the hidden block.
3. **Hidden mode, main().**  `main()` with a fake safe server, 32 steps, 4 requests: flag off = the pre-T43X
   `pi05_infer.py` bit for bit (frames, manifest, npz, requests, video); flag on vs off differs only in the
   request `images` (and wall-clock fields); the on images = those of the no-obstacle run (`clear`); the on
   depth = the off depth (obstacle present) and differs from `clear`; the `--record` video on = off and differs
   from `clear`; the manifest / npz carry the mode only when on.

CPU only: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa`.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path
from common.workspace import bind_historical_client_inputs as _bind_legacy_inputs

import contextlib
import copy
import importlib.util
import io
import json
import os
import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")

HYBRID = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid'))
BRINGUP = HYBRID / "rby1_bringup"
MJCF = HYBRID / "rby1_description" / "models" / "rby1a" / "mujoco"
PI05_INFER = BRINGUP / "pi05_infer.py"
TRANSPORT_XML = MJCF / "model_transport.xml"
TRANSPORT_OBSTACLE_XML = MJCF / "model_transport_pick_place_obstacles.xml"
BLOCK_XML = MJCF / "model.xml"
BLOCK_OBSTACLE_XML = MJCF / "model_pick_place_obstacles.xml"
EPISODE = 1968  # apple, left arm
CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
POLICY_KEYS = {"zed_left": "cam_high", "wrist_cam_l": "cam_left_wrist", "wrist_cam_r": "cam_right_wrist"}
#: a pick obstacle beside the apple (in cam_high and the left wrist) + the T40 F ep1968 place hurdle
CONFIG = {
    "version": 1, "table_surface_z": 0.82,
    "profiles": {
        "clear": {"scene": "any", "description": "no obstacle", "obstacles": []},
        "t43x": {"scene": "fruit", "description": "T43 X: post beside the apple + the T40 F ep1968 hurdle",
                 "obstacles": [{"slot": "post_0", "position": [0.62, 0.24, 0.82], "yaw": 0.0},
                               {"slot": "hurdle_0", "position": [0.4939, 0.1778, 0.82],
                                "yaw": 0.311739, "bar_height": 0.24}]}},
}

if not TRANSPORT_OBSTACLE_XML.exists():  # pragma: no cover - asset-dependent
    pytest.skip("pi05_TO_hybrid scene assets not present", allow_module_level=True)
if str(HYBRID / "rby1_manipulation" / "src") not in sys.path:
    sys.path.insert(0, str(HYBRID / "rby1_manipulation" / "src"))

from rby1_manipulation.simulation import pick_place_obstacles as ppo  # noqa: E402


def _load_client(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        _bind_legacy_inputs(mod)
    except ImportError as exc:  # the client needs its own venv (mink, PIL): .venv-openpi-live has them
        pytest.skip(f"pi05_infer.py dependencies missing in this interpreter: {exc}")
    return mod


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("PIL")
    return _load_client(PI05_INFER, "_t43x_pi05_infer")


@pytest.fixture(scope="module")
def TransportScene():
    try:
        from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene as TS
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"benchmark not importable: {exc}")
    return TS


# ============================================================================ helpers


def _unpinned(xml: pathlib.Path) -> mujoco.MjModel:
    """The obstacle model as compiled before T43 X: the compiler computes extent / center (NaN = auto)."""
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.stat.extent = float("nan")
    spec.stat.center = [float("nan"), 0.0, 0.0]
    return spec.compile()


def _scene(client, xml=None, model=None, episode=EPISODE):
    m = model if model is not None else mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop"))
    client.hold_keyframe_pose(m, d)
    if episode is not None:
        client.reset_randomized_scene(m, d, client.load_randomized_episode(episode))
    mujoco.mj_forward(m, d)
    return m, d


def _policy(client, m, d, hide=()) -> dict:
    """The three policy images exactly as `build_obs` (16-D / rby1) makes them, optionally policy-blind."""
    r = mujoco.Renderer(m, client.POLICY_SOURCE_HEIGHT, client.POLICY_SOURCE_WIDTH)
    try:
        with client.hidden_from_policy_render(m, hide):
            return {c: client.render_cam(m, d, r, c, match_rby1_dataset=True) for c in CAMERAS}
    finally:
        r.close()


def _depth(TransportScene, m, d) -> dict:
    s = TransportScene.attach(m, d)
    try:
        return {c: s.capture(c).depth for c in CAMERAS}
    finally:
        s.close()


def _record_frame(client, m, d) -> np.ndarray:
    """One third-person `--record` frame (front view, 640x480), as the client renders it."""
    cam = mujoco.MjvCamera()
    client.configure_view_camera(cam, "front")
    r = mujoco.Renderer(m, height=480, width=640)
    try:
        r.update_scene(d, camera=cam)
        return r.render().copy()
    finally:
        r.close()


def _model_arrays(m) -> dict:
    out = {}
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            v = getattr(m, name)
        except Exception:  # pragma: no cover
            continue
        if isinstance(v, np.ndarray):
            out[name] = v.copy()
    return out


def _n_diff(a, b) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


# ============================================================================ 1. statistic fix


@pytest.mark.parametrize("plain, obstacle", [(TRANSPORT_XML, TRANSPORT_OBSTACLE_XML), (BLOCK_XML, BLOCK_OBSTACLE_XML)])
def test_statistic_is_pinned_to_the_plain_model_and_nothing_else_changes(plain, obstacle):
    p = mujoco.MjModel.from_xml_path(str(plain))
    o = mujoco.MjModel.from_xml_path(str(obstacle))
    u = _unpinned(obstacle)
    assert o.stat.extent == p.stat.extent and np.array_equal(o.stat.center, p.stat.center)
    # the unpinned compile is the pre-T43X model: its box is the one the parked slots widen
    assert u.stat.extent != p.stat.extent or not np.array_equal(u.stat.center, p.stat.center)
    for field in ("meaninertia", "meanmass", "meansize"):
        assert getattr(o.stat, field) == getattr(u.stat, field), field
    a, b = _model_arrays(o), _model_arrays(u)
    assert set(a) == set(b)
    assert [k for k in a if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")] == []


@pytest.mark.parametrize("scene", ["transport", "block"])
def test_empty_obstacle_scene_renders_like_the_plain_scene(client, TransportScene, scene):
    plain, obstacle, episode = ((TRANSPORT_XML, TRANSPORT_OBSTACLE_XML, EPISODE) if scene == "transport"
                                else (BLOCK_XML, BLOCK_OBSTACLE_XML, None))
    mp, dp = _scene(client, plain, episode=episode)
    mo, do = _scene(client, obstacle, episode=episode)
    # a manager on the empty scene ("clear" profile = every slot parked) changes nothing either
    mgr = ppo.PickPlaceObstacleManager(mo, do, CONFIG)
    mgr.activate("clear")
    assert np.array_equal(dp.qpos, do.qpos)
    P, O = _policy(client, mp, dp), _policy(client, mo, do)
    DP, DO = _depth(TransportScene, mp, dp), _depth(TransportScene, mo, do)
    for c in CAMERAS:
        assert P[c].shape == (224, 224, 3)
        assert np.array_equal(P[c], O[c]), (scene, c, _n_diff(P[c], O[c]))
        assert np.array_equal(DP[c], DO[c]), (scene, c, float(np.abs(DP[c] - DO[c]).max()))
    # sensitivity: the unpinned (pre-T43X) obstacle model does differ from the plain scene
    mu, du = _scene(client, model=_unpinned(obstacle), episode=episode)
    U, DU = _policy(client, mu, du), _depth(TransportScene, mu, du)
    assert sum(_n_diff(P[c], U[c]) for c in CAMERAS) > 0
    assert any(not np.array_equal(DP[c], DU[c]) for c in CAMERAS)


# ============================================================================ 2. hidden mode, render level


@pytest.fixture(scope="module")
def placed(client):
    mp, dp = _scene(client, TRANSPORT_XML)
    mo, do = _scene(client, TRANSPORT_OBSTACLE_XML)
    mgr = ppo.PickPlaceObstacleManager(mo, do, CONFIG)
    mgr.activate("t43x")
    return {"mp": mp, "dp": dp, "mo": mo, "do": do, "mgr": mgr}


def test_hidden_set_is_every_visual_geom_and_no_collision_geom(placed):
    m, mgr = placed["mo"], placed["mgr"]
    hidden = mgr.visual_geom_ids
    assert hidden and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in hidden)
    assert not hidden & mgr.collision_geom_ids
    assert all(int(m.geom_group[g]) == 2 for g in hidden)  # drawn like the rest of the scene
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in hidden}
    assert all(n.startswith("pick_obstacle_") for n in names)
    for slot in ("post_0", "hurdle_0"):
        assert any(n.startswith(f"pick_obstacle_{slot}") for n in names), slot
    # group 5 is free and off by default
    assert not np.any(m.geom_group == 5)
    assert mujoco.MjvOption().geomgroup[5] == 0


def test_hidden_policy_rgb_equals_the_plain_scene(client, placed):
    mp, dp, mo, do, mgr = (placed[k] for k in ("mp", "dp", "mo", "do", "mgr"))
    assert np.array_equal(dp.qpos, do.qpos)
    groups = mo.geom_group.copy()
    P = _policy(client, mp, dp)
    V = _policy(client, mo, do)
    H = _policy(client, mo, do, mgr.visual_geom_ids)
    assert np.array_equal(mo.geom_group, groups)
    # the obstacle is in the head and the left-wrist image when visible ...
    assert _n_diff(V["zed_left"], P["zed_left"]) > 500 and _n_diff(V["wrist_cam_l"], P["wrist_cam_l"]) > 5000
    # ... and policy-blind it is the plain scene, pixel for pixel (no shadow, no reflection left)
    for c in CAMERAS:
        assert np.array_equal(H[c], P[c]), (c, _n_diff(H[c], P[c]))


def test_visible_obstacle_casts_pixels_outside_its_silhouette_and_hidden_removes_them(client, placed):
    """Shadow evidence at the 299x224 source resolution: pixels that change with the obstacle although no
    obstacle geom covers them (shadow on the table / fruit; reflections are off for policy renders)."""
    mp, dp, mo, do, mgr = (placed[k] for k in ("mp", "dp", "mo", "do", "mgr"))
    hidden = np.asarray(sorted(mgr.visual_geom_ids))
    assert client.POLICY_RENDER_REFLECTIONS is False
    off_silhouette = {}
    for c in CAMERAS:
        imgs = {}
        for key, (m, d, hide) in {"plain": (mp, dp, ()), "vis": (mo, do, ()),
                                  "hid": (mo, do, hidden)}.items():
            r = mujoco.Renderer(m, client.POLICY_SOURCE_HEIGHT, client.POLICY_SOURCE_WIDTH)
            with client.hidden_from_policy_render(m, hide):
                r.update_scene(d, camera=c)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                imgs[key] = r.render().copy()
                if key == "vis":
                    r.enable_segmentation_rendering()
                    r.update_scene(d, camera=c)
                    seg = r.render().copy()
                    r.disable_segmentation_rendering()
            r.close()
        obstacle_px = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], hidden)
        changed = np.any(imgs["vis"] != imgs["plain"], axis=-1)
        off_silhouette[c] = int(np.count_nonzero(changed & ~obstacle_px))
        assert np.array_equal(imgs["hid"], imgs["plain"]), c
    # the visible obstacle does shade something outside its own pixels in at least one camera (shadow)
    assert sum(off_silhouette.values()) > 0, off_silhouette


def test_geom_group_is_restored_on_error_and_untouched_when_empty(client, placed):
    m, mgr = placed["mo"], placed["mgr"]
    before = m.geom_group.copy()
    with pytest.raises(RuntimeError):
        with client.hidden_from_policy_render(m, mgr.visual_geom_ids):
            assert np.all(m.geom_group[sorted(mgr.visual_geom_ids)] == client.POLICY_HIDDEN_GEOMGROUP)
            raise RuntimeError("boom")
    assert np.array_equal(m.geom_group, before)
    with client.hidden_from_policy_render(m, ()):
        assert np.array_equal(m.geom_group, before)


def test_depth_and_record_still_show_the_obstacle(client, TransportScene, placed):
    mp, dp, mo, do, mgr = (placed[k] for k in ("mp", "dp", "mo", "do", "mgr"))
    D_visible = _depth(TransportScene, mo, do)
    R_visible = _record_frame(client, mo, do)
    _policy(client, mo, do, mgr.visual_geom_ids)  # a policy-blind render in between
    D_after, R_after = _depth(TransportScene, mo, do), _record_frame(client, mo, do)
    D_plain, R_plain = _depth(TransportScene, mp, dp), _record_frame(client, mp, dp)
    for c in CAMERAS:
        assert np.array_equal(D_after[c], D_visible[c]), c
    assert _n_diff(D_after["zed_left"], D_plain["zed_left"]) > 500
    assert _n_diff(D_after["wrist_cam_l"], D_plain["wrist_cam_l"]) > 5000
    assert np.array_equal(R_after, R_visible)
    assert _n_diff(R_after, R_plain) > 1000


def test_collision_is_unchanged_inside_the_hidden_block(client):
    """post_0 dropped onto the apple: MuJoCo's contacts are the same inside and outside the block."""
    m, d = _scene(client, TRANSPORT_OBSTACLE_XML)
    apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    x, y = d.xpos[apple][:2]
    cfg = copy.deepcopy(CONFIG)
    cfg["profiles"]["onto_apple"] = {"scene": "fruit", "obstacles": [
        {"slot": "post_0", "position": [float(x), float(y), 0.82], "yaw": 0.0}]}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    mgr.activate("onto_apple")

    def contacts():
        mujoco.mj_forward(m, d)
        return sorted((int(c.geom1), int(c.geom2), round(float(c.dist), 12)) for c in d.contact[:d.ncon])

    outside = contacts()
    with client.hidden_from_policy_render(m, mgr.visual_geom_ids):
        inside = contacts()
    assert inside == outside
    assert any(g in mgr.collision_geom_ids for pair in outside for g in pair[:2])


# ============================================================================ 3. hidden mode, main()


class _FakeSafeServer:
    """`WebsocketClientPolicy` stand-in for `--safe-remote`: every chunk executes; actions depend on `state`."""

    def __init__(self):
        self.requests = []

    def infer(self, request):
        from AG3S.trajopt import wire

        self.requests.append(copy.deepcopy(request))
        state = np.asarray(request["state"], dtype=np.float64)
        ref = np.tile(state, (50, 1))
        ref[:, 1] += np.linspace(0.0, 0.02, 50)
        return {"actions": ref.astype(np.float32), "seq": int(request[wire.PREFIX + "seq"]),
                "safe": True, "ag3s_status": "ok", "geometry_certified": True,
                "trajopt_status": "feasible", "max_violation_m": 0.0, "timing_ms": {"total": 1.0},
                "notes": [], wire.SHADOW: False, wire.ACTIONS_REFERENCE: ref.astype(np.float32)}


def _run_main(client, out: pathlib.Path, cfg: pathlib.Path, profile: str, extra=()) -> dict:
    server = _FakeSafeServer()
    original = client.load_remote_policy
    client.load_remote_policy = lambda remote: server
    old_argv = sys.argv
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                "--episode-index", str(EPISODE), "--headless", "--max-steps", "32", "--warmup-steps", "20",
                "--view", "front", "--record", str(out / "third_person_front.mp4"),
                "--record-frames", str(out), "--trajectory-out", str(out / "traj.npz"),
                "--obstacle-config", str(cfg), "--obstacle-profile", profile,
                "--obstacle-stop-distance", "0",
                "--safe-remote", "--safe-timeout", "120", "--safe-manipulators", "left", *extra]
    np.random.seed(1234)
    random.seed(1234)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            client.main()
    except SystemExit as exc:
        raise AssertionError(f"main() exited ({exc.code}):\n{buf.getvalue()[-3000:]}") from exc
    finally:
        sys.argv = old_argv
        client.load_remote_policy = original
    return {"out": out, "requests": server.requests, "stdout": buf.getvalue()}


def _pre_t43x_source() -> bytes:
    """`rby1_bringup/pi05_infer.py` of the newest commit without the flag (HEAD until T43 X is committed)."""
    rel = "rby1_bringup/pi05_infer.py"
    try:
        commits = subprocess.run(["git", "-C", str(HYBRID), "log", "--format=%H", "-n", "30", "--", rel],
                                 check=True, capture_output=True, text=True).stdout.split()
        for commit in commits:
            src = subprocess.run(["git", "-C", str(HYBRID), "show", f"{commit}:{rel}"],
                                 check=True, capture_output=True).stdout
            if b"obstacle-policy-hidden" not in src:
                return src
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pass
    pytest.skip("no pre-T43X pi05_infer.py in git")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    try:
        import PIL  # noqa: F401
        import imageio  # noqa: F401
        from AG3S.trajopt import wire  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"missing: {exc}")
    tmp = tmp_path_factory.mktemp("t43x")
    cfg = tmp / "t43x.json"
    cfg.write_text(json.dumps(CONFIG))
    # HEAD's pi05_infer.py inside a mirror of pi05_TO_hybrid (REPO_ROOT = parents[1]); assets are the working
    # tree's (symlinks), so the head/off comparison isolates the client code.
    head_root = tmp / "mirror_of_head"
    (head_root / "rby1_bringup").mkdir(parents=True)
    for entry in HYBRID.iterdir():
        if entry.name not in ("rby1_bringup", ".git"):
            (head_root / entry.name).symlink_to(entry)
    (head_root / "rby1_bringup" / "pi05_infer_head.py").write_bytes(_pre_t43x_source())
    head = _load_client(head_root / "rby1_bringup" / "pi05_infer_head.py", "_t43x_pi05_infer_head")
    new = _load_client(PI05_INFER, "_t43x_pi05_infer_new")
    return {
        "head_root": head_root,
        "head": _run_main(head, tmp / "head", cfg, "t43x"),
        "off": _run_main(new, tmp / "off", cfg, "t43x"),
        "on": _run_main(new, tmp / "on", cfg, "t43x", ("--obstacle-policy-hidden",)),
        "clear": _run_main(new, tmp / "clear", cfg, "clear"),
    }


#: what two runs may differ in by wall clock alone (see test_t42_avoidance_arrow._volatile)
_CLOCK = ("stamp", "now", "applied_at", "render_spread", "timing_ms", "recorded_at", "inference_ms")


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}/{k}"))
        return out
    if isinstance(obj, list) and obj and all(isinstance(x, (dict, list)) for x in obj):
        out = {}
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def _jsonl(run, root, name):
    text = (run["out"] / name).read_text().replace(str(run["out"]), "<OUT>").replace(
        str(root.resolve()), "<ROOT>").replace(str(root), "<ROOT>")
    if name.endswith(".jsonl"):
        return [_flatten(json.loads(line), f"#{i}") for i, line in enumerate(text.splitlines())]
    return [_flatten(json.loads(text))]


def _diff_paths(a: list, b: list) -> set:
    assert len(a) == len(b)
    paths = set()
    for ra, rb in zip(a, b):
        for k in set(ra) | set(rb):
            if ra.get(k, "<absent>") != rb.get(k, "<absent>"):
                paths.add(k.split("/", 1)[-1] if k.startswith("#") else k)
    return paths


def _npz(run):
    with np.load(run["out"] / "traj.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _deep_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _request_diff(ra, rb) -> set:
    assert len(ra) == len(rb)
    out = set()
    for qa, qb in zip(ra, rb):
        assert list(qa) == list(qb)
        out |= {k for k in qa if not _deep_equal(qa[k], qb[k])}
    return out


def _video(run):
    import imageio.v2 as iio

    rd = iio.get_reader(run["out"] / "third_person_front.mp4")
    try:
        return [np.asarray(f) for f in rd]
    finally:
        rd.close()


def _only_clock(paths) -> bool:
    return all(any(t in p for t in _CLOCK) for p in paths)


def test_flag_off_is_bit_identical_to_head(runs):
    head, off = runs["head"], runs["off"]
    assert _only_clock(_diff_paths(_jsonl(head, runs["head_root"], "frames.jsonl"), _jsonl(off, HYBRID, "frames.jsonl")))
    assert _only_clock(_diff_paths(_jsonl(head, runs["head_root"], "manifest.json"),
                                   _jsonl(off, HYBRID, "manifest.json")) - {"/extra/argv"})
    a, b = _npz(head), _npz(off)
    assert set(a) == set(b) and "obstacle_policy_hidden" not in b
    assert {k for k in a if not np.array_equal(a[k], b[k])} <= {"inference_ms"}
    assert _only_clock(_request_diff(head["requests"], off["requests"]))
    vh, vo = _video(head), _video(off)
    assert len(vh) == len(vo) == 32 and all(np.array_equal(x, y) for x, y in zip(vh, vo))
    assert "policy-blind" not in off["stdout"]


def test_flag_on_hides_the_obstacle_from_the_policy_images_only(runs):
    on, off, clear = runs["on"], runs["off"], runs["clear"]
    assert len(on["requests"]) == len(off["requests"]) == len(clear["requests"]) == 4
    assert "policy-blind: " in on["stdout"]
    # physics and actions: identical across all three (the obstacle is never touched in 32 steps)
    a, b, c = _npz(on), _npz(off), _npz(clear)
    for k in ("executed_actions", "measured_qpos", "predicted_chunks", "inference_states"):
        assert np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k]), k
    # the request: on vs off differ in the policy images (and wall clock) only
    assert _only_clock(_request_diff(on["requests"], off["requests"]) - {"images"})
    for qa, qb, qc in zip(on["requests"], off["requests"], clear["requests"]):
        assert np.array_equal(qa["state"], qc["state"])
        for cam in CAMERAS:
            key = POLICY_KEYS[cam]
            # policy RGB on = no-obstacle RGB, pixel for pixel
            assert np.array_equal(qa["images"][key], qc["images"][key]), (cam, _n_diff(qa["images"][key], qc["images"][key]))
            # AG3S depth on = the obstacle-present depth
            assert np.array_equal(qa[f"ag3s/depth/{cam}"], qb[f"ag3s/depth/{cam}"]), cam
    q_on, q_off, q_clear = on["requests"][0], off["requests"][0], clear["requests"][0]
    assert _n_diff(q_off["images"]["cam_high"], q_clear["images"]["cam_high"]) > 100        # visible when off
    assert _n_diff(q_off["images"]["cam_left_wrist"], q_clear["images"]["cam_left_wrist"]) > 1000
    assert _n_diff(q_on["ag3s/depth/zed_left"], q_clear["ag3s/depth/zed_left"]) > 500     # AG3S sees it
    assert _n_diff(q_on["ag3s/depth/wrist_cam_l"], q_clear["ag3s/depth/wrist_cam_l"]) > 5000
    # the third-person recording shows the obstacle
    v_on, v_off, v_clear = _video(on), _video(off), _video(clear)
    assert len(v_on) == 32 and all(np.array_equal(x, y) for x, y in zip(v_on, v_off))
    assert all(_n_diff(x, y) > 1000 for x, y in zip(v_on, v_clear))


def test_the_mode_is_recorded_only_when_on(runs):
    man_on = json.loads((runs["on"]["out"] / "manifest.json").read_text())
    man_off = json.loads((runs["off"]["out"] / "manifest.json").read_text())
    assert man_on["scene"]["obstacle_policy_hidden"] is True
    names = man_on["scene"]["obstacle_policy_hidden_geoms"]
    assert names and all(n.startswith("pick_obstacle_") for n in names)
    assert any("post_0" in n for n in names) and any("hurdle_0" in n for n in names)
    assert "obstacle_policy_hidden" not in man_off["scene"]
    assert "obstacle_policy_hidden_geoms" not in man_off["scene"]
    # the rest of the manifest is the off one (argv aside)
    on_flat, off_flat = _flatten(man_on), _flatten(man_off)
    extra = {k for k in on_flat if k not in off_flat}
    assert extra == {"/scene/obstacle_policy_hidden", "/scene/obstacle_policy_hidden_geoms"}
    assert bool(_npz(runs["on"])["obstacle_policy_hidden"]) is True
    assert "obstacle_policy_hidden" not in _npz(runs["off"])
