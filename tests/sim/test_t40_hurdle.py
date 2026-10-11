"""T40 E — the hurdle obstacle (`hurdle_0`) that can be added and removed, in pi05_TO_hybrid.

What this file pins:

1. **clear = the old scene, bit for bit.** `--obstacle-profile clear` runs `pi05_infer.main()` on
   `model_transport.xml` with no obstacle manager; the model and the reset state (qpos/qvel/ctrl/mocap) are
   byte-identical to the pre-T40E client (`22d8dae`), and the qpos addresses are apple 38, crate 31, left
   finger 27 in both the plain and the obstacle XML.
2. **Config.** `bar_height` is a hurdle-only field in [0.20, 0.32] m (default 0.26); JSON round trip.
3. **Placement and height.** Position / yaw / bar height land where the config says; the sliding posts stay
   inside the sleeves over the whole range; `clear` parks everything.
4. **Physical collision** with the carried apple (drop test: contact force, no pass-through) and the gripper.
5. **16D + obstacle** goes through `main()` to the policy call on the transport obstacle XML
   (T40 A blocker: `apple_free missing`).
6. **Distances.** The reset clearance check and the run-time stop use `pair_distance`, which equals an
   independent GJK for every hurdle pair type (capsule/cylinder x mesh/box); the capsule-box case where
   MuJoCo 3.11 `mj_geomDistance` reports a positive distance for overlapping geoms is pinned.

CPU only: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa`.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path
from common.workspace import bind_historical_client_inputs as _bind_legacy_inputs

import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")

from tests.sim._gjk import gjk_dist, make_support  # noqa: E402

HYBRID = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid'))
MJCF = HYBRID / "rby1_description" / "models" / "rby1a" / "mujoco"
TRANSPORT_XML = MJCF / "model_transport.xml"
TRANSPORT_OBSTACLE_XML = MJCF / "model_transport_pick_place_obstacles.xml"
BLOCK_OBSTACLE_XML = MJCF / "model_pick_place_obstacles.xml"
PI05_INFER = HYBRID / "rby1_bringup" / "pi05_infer.py"
#: the client before T40 E (T39 head of pi05_TO_hybrid)
BASE_COMMIT = "22d8dae"
EPISODE = 1807

if not TRANSPORT_OBSTACLE_XML.exists():  # pragma: no cover - asset-dependent
    pytest.skip("pi05_TO_hybrid scene assets not present", allow_module_level=True)
if str(HYBRID / "rby1_manipulation" / "src") not in sys.path:
    sys.path.insert(0, str(HYBRID / "rby1_manipulation" / "src"))

from rby1_manipulation.simulation import pick_place_obstacles as ppo  # noqa: E402

HURDLE_COLLISION = (
    "pick_obstacle_hurdle_0_collision_foot_left",
    "pick_obstacle_hurdle_0_collision_foot_right",
    "pick_obstacle_hurdle_0_collision_sleeve_left",
    "pick_obstacle_hurdle_0_collision_sleeve_right",
    "pick_obstacle_hurdle_0_collision_bar",
    "pick_obstacle_hurdle_0_collision_post_left",
    "pick_obstacle_hurdle_0_collision_post_right",
)
TABLE_Z = 0.82


def _config(**hurdle) -> dict:
    # default: beyond the teleop-key crate, bar along y, clear of both teleop grippers
    obstacle = {"slot": "hurdle_0", "position": [0.75, 0.0, TABLE_Z], "yaw": math.pi / 2}
    obstacle.update(hurdle)
    return {
        "version": 1,
        "table_surface_z": TABLE_Z,
        "profiles": {
            "clear": {"scene": "any", "obstacles": []},
            "hurdle": {"scene": "fruit", "obstacles": [obstacle]},
        },
    }


def _gid(m, name):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert g >= 0, name
    return g


def _qadr(m, joint):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def _teleop_scene(xml=TRANSPORT_OBSTACLE_XML):
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop"))
    for a in range(m.nu):
        d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
    mujoco.mj_forward(m, d)
    return m, d


@pytest.fixture()
def scene():
    m, d = _teleop_scene()
    return m, d, ppo.PickPlaceObstacleManager(m, d, _config())


# ----------------------------------------------------------------------------- pi05_infer.main() harness


class _Reached(Exception):
    pass


def _load_client(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        _bind_legacy_inputs(mod)
    except ImportError as exc:  # the client needs its own venv (mink, PIL): .venv-openpi-live has them
        pytest.skip(f"pi05_infer.py dependencies missing in this interpreter: {exc}")
    return mod


def _run_until_policy(client, argv) -> dict:
    """Run `client.main()` up to `load_remote_policy` (no server) and return main()'s locals there."""
    captured: dict = {}

    def stub(remote):
        captured.update(inspect.currentframe().f_back.f_locals)
        raise _Reached

    original = client.load_remote_policy
    client.load_remote_policy = stub
    old_argv = sys.argv
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                "--episode-index", str(EPISODE), "--headless", *argv]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            client.main()
    except _Reached:
        pass
    except SystemExit as exc:  # argparse error: show it
        raise AssertionError(f"main() exited ({exc.code}):\n{out.getvalue()[-2000:]}") from exc
    finally:
        sys.argv = old_argv
        client.load_remote_policy = original
    assert captured, "main() never reached load_remote_policy"
    captured["_stdout"] = out.getvalue()
    return captured


@pytest.fixture(scope="module")
def client():
    try:
        import PIL  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("PIL missing")
    return _load_client(PI05_INFER, "_t40e_pi05_infer")


@pytest.fixture(scope="module")
def base_client(tmp_path_factory):
    """The pre-T40E client file, run against the same scene assets (REPO_ROOT = a tmp dir of symlinks)."""
    try:
        src = subprocess.run(["git", "-C", str(HYBRID), "show", f"{BASE_COMMIT}:rby1_bringup/pi05_infer.py"],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip(f"cannot read {BASE_COMMIT} from git")
    root = tmp_path_factory.mktemp("base_repo")
    (root / "rby1_bringup").mkdir()
    for sub in ("rby1_description", "rby1_manipulation"):
        (root / sub).symlink_to(HYBRID / sub)
    path = root / "rby1_bringup" / "pi05_infer_base.py"
    path.write_text(src)
    return _load_client(path, "_t40e_pi05_infer_base")


def _model_fingerprint(m) -> str:
    h = hashlib.sha256()
    for field in ("nq", "nv", "nu", "nbody", "ngeom", "nmocap"):
        h.update(str(getattr(m, field)).encode())
    for field in ("qpos0", "body_pos", "body_quat", "geom_type", "geom_size", "geom_pos", "geom_quat",
                  "geom_contype", "geom_conaffinity", "geom_rbound", "jnt_qposadr", "actuator_trnid",
                  "opt.timestep"):
        obj = m.opt if field.startswith("opt.") else m
        h.update(np.ascontiguousarray(getattr(obj, field.split(".")[-1])).tobytes())
    return h.hexdigest()


def _state_fingerprint(d) -> str:
    h = hashlib.sha256()
    for field in ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat", "time"):
        h.update(np.ascontiguousarray(getattr(d, field)).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- 1. clear = the old scene


def test_clear_profile_is_the_pre_t40e_scene_bit_for_bit(client, base_client):
    new = _run_until_policy(client, ["--obstacle-profile", "clear"])
    old = _run_until_policy(base_client, ["--obstacle-profile", "clear"])
    assert pathlib.Path(new["model_xml"]).resolve() == TRANSPORT_XML.resolve()
    assert pathlib.Path(old["model_xml"]).resolve() == TRANSPORT_XML.resolve()
    assert new["obstacle_manager"] is None and old["obstacle_manager"] is None
    m_new, d_new, m_old, d_old = new["m"], new["d"], old["m"], old["d"]
    assert mujoco.mj_name2id(m_new, mujoco.mjtObj.mjOBJ_BODY, "pick_obstacle_hurdle_0") == -1
    assert _model_fingerprint(m_new) == _model_fingerprint(m_old)
    assert _state_fingerprint(d_new) == _state_fingerprint(d_old)
    for field in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
        assert np.array_equal(getattr(d_new, field), getattr(d_old, field)), field


def test_qpos_addresses_and_dimensions_unchanged_by_the_obstacle_xml():
    for xml in (TRANSPORT_XML, TRANSPORT_OBSTACLE_XML):
        m = mujoco.MjModel.from_xml_path(str(xml))
        assert (m.nq, m.nv, m.nu) == (66, 61, 29), xml.name
        assert _qadr(m, "apple_free") == 38
        assert _qadr(m, "crate_free") == 31
        assert _qadr(m, "gripper_finger_l1") == 27


def test_obstacle_xml_with_everything_parked_resets_like_clear():
    """Same robot/object initial state from the teleop key: the parked slots add no DoF and touch nothing."""
    m0, d0 = _teleop_scene(TRANSPORT_XML)
    m1, d1 = _teleop_scene(TRANSPORT_OBSTACLE_XML)
    ppo.PickPlaceObstacleManager(m1, d1, _config()).activate("clear")
    assert np.array_equal(d0.qpos, d1.qpos)
    for _ in range(200):
        mujoco.mj_step(m0, d0)
        mujoco.mj_step(m1, d1)
    np.testing.assert_allclose(d1.qpos, d0.qpos, atol=1e-9)


# ----------------------------------------------------------------------------- 2. config


def test_bar_height_is_validated_and_defaults(tmp_path):
    cfg = ppo.validate_pick_place_obstacle_config(_config())
    assert cfg["profiles"]["hurdle"]["obstacles"][0]["bar_height"] == pytest.approx(0.26)
    for ok in (0.20, 0.25, 0.32):
        cfg = ppo.validate_pick_place_obstacle_config(_config(bar_height=ok))
        assert cfg["profiles"]["hurdle"]["obstacles"][0]["bar_height"] == pytest.approx(ok)
    for bad in (0.19, 0.33, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="bar_height"):
            ppo.validate_pick_place_obstacle_config(_config(bar_height=bad))
    raw = _config()
    raw["profiles"]["hurdle"]["obstacles"][0]["slot"] = "bollard_0"
    raw["profiles"]["hurdle"]["obstacles"][0]["bar_height"] = 0.25
    with pytest.raises(ValueError, match="only valid for slots"):
        ppo.validate_pick_place_obstacle_config(raw)
    path = tmp_path / "hurdle.json"
    path.write_text(json.dumps(_config(position=[0.6, 0.21, 0.82], yaw=0.3, bar_height=0.29)))
    loaded = ppo.load_pick_place_obstacle_config(path)
    entry = loaded["profiles"]["hurdle"]["obstacles"][0]
    assert entry == {"slot": "hurdle_0", "position": [0.6, 0.21, 0.82], "yaw": 0.3, "bar_height": 0.29}


def test_hurdle_is_capsules_and_cylinders_with_the_prop_conventions():
    m = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    hurdle = [g for g, n in enumerate(names) if n.startswith("pick_obstacle_hurdle_0")]
    collision = [g for g in hurdle if "_collision" in names[g]]
    visual = [g for g in hurdle if "_collision" not in names[g]]
    assert sorted(names[g] for g in collision) == sorted(HURDLE_COLLISION)
    for g in hurdle:
        assert int(m.geom_type[g]) in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER), names[g]
    for g in collision:  # class prop_collision: hidden group 3, contype/conaffinity 1
        assert (m.geom_group[g], m.geom_contype[g], m.geom_conaffinity[g]) == (3, 1, 1), names[g]
    for g in visual:  # class prop_visual: group 2, no contacts, no mass
        assert (m.geom_group[g], m.geom_contype[g], m.geom_conaffinity[g]) == (2, 0, 0), names[g]
    bar = _gid(m, "pick_obstacle_hurdle_0_collision_bar")
    radius, half = m.geom_size[bar][:2]
    assert 2 * (half + radius) == pytest.approx(0.264)  # end to end
    for body in ("pick_obstacle_hurdle_0", "pick_obstacle_hurdle_0_bar"):
        assert m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)] >= 0


# ----------------------------------------------------------------------------- 3. placement and height


@pytest.mark.parametrize("bar_height", [0.20, 0.26, 0.32])
@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.9])
def test_activate_places_the_bar_at_the_config_height_and_yaw(bar_height, yaw):
    m, d = _teleop_scene()
    pos = [0.60, 0.18, TABLE_Z]
    mgr = ppo.PickPlaceObstacleManager(m, d, _config(position=pos, yaw=yaw, bar_height=bar_height))
    mgr.activate("hurdle")
    assert mgr.active_slots == ("hurdle_0",)
    np.testing.assert_allclose(mgr.positions()["hurdle_0"], pos)
    bar = _gid(m, "pick_obstacle_hurdle_0_collision_bar")
    np.testing.assert_allclose(d.geom_xpos[bar], [pos[0], pos[1], TABLE_Z + bar_height], atol=1e-12)
    axis = d.geom_xmat[bar].reshape(3, 3)[:, 2]  # capsule axis
    assert abs(axis @ [math.cos(yaw), math.sin(yaw), 0.0]) == pytest.approx(1.0, abs=1e-12)
    for side, sign in (("left", -1.0), ("right", 1.0)):
        foot = _gid(m, f"pick_obstacle_hurdle_0_collision_foot_{side}")
        sleeve = _gid(m, f"pick_obstacle_hurdle_0_collision_sleeve_{side}")
        post = _gid(m, f"pick_obstacle_hurdle_0_collision_post_{side}")
        expect_xy = np.array(pos[:2]) + sign * 0.12 * np.array([math.cos(yaw), math.sin(yaw)])
        for g in (foot, sleeve, post):
            np.testing.assert_allclose(d.geom_xpos[g][:2], expect_xy, atol=1e-12)
        assert d.geom_xpos[foot][2] - m.geom_size[foot][1] == pytest.approx(TABLE_Z, abs=1e-12)  # on the table
        sleeve_top = d.geom_xpos[sleeve][2] + m.geom_size[sleeve][1]
        post_bottom = d.geom_xpos[post][2] - m.geom_size[post][1]
        post_top = d.geom_xpos[post][2] + m.geom_size[post][1]
        assert post_bottom < sleeve_top - 0.005  # the post never leaves the sleeve
        assert post_bottom > d.geom_xpos[foot][2] + m.geom_size[foot][1]  # nor goes through the foot
        assert post_top == pytest.approx(TABLE_Z + bar_height, abs=1e-12)  # ends at the bar centre


def test_clear_and_park_remove_the_hurdle(scene):
    m, d, mgr = scene
    mgr.activate("hurdle")
    mgr.activate("clear")
    assert mgr.active_slots == () and mgr.collision_geom_ids == set()
    assert math.isinf(mgr.robot_clearance())
    for name in HURDLE_COLLISION:
        assert np.linalg.norm(d.geom_xpos[_gid(m, name)][:2]) > 10.0  # parked off the floor plan


# ----------------------------------------------------------------------------- 4. physical collision


def _contact_force(m, d, g1, g2) -> float:
    total = 0.0
    f = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if {int(c.geom1), int(c.geom2)} == {g1, g2}:
            mujoco.mj_contactForce(m, d, i, f)
            total += f[0]
    return total


def test_contact_filters_let_the_hurdle_touch_every_robot_and_payload_geom(scene):
    m, d, mgr = scene
    hurdle = mgr._collision_geoms["hurdle_0"]
    assert len(hurdle) == len(HURDLE_COLLISION)
    for h in hurdle:
        for o in (*mgr.robot_geom_ids, *mgr.payload_geom_ids):
            assert (m.geom_contype[h] & m.geom_conaffinity[o]) or (m.geom_contype[o] & m.geom_conaffinity[h])


def test_dropped_apple_lands_on_the_bar(scene):
    m, d, mgr = scene
    mgr.activate("hurdle")  # bar centre (0.75, 0.0, 0.82 + 0.26), along y
    assert mgr.robot_clearance() > 0.05
    apple, bar = _gid(m, "apple_geom"), _gid(m, "pick_obstacle_hurdle_0_collision_bar")
    adr = _qadr(m, "apple_free")
    bar_z = d.geom_xpos[bar][2]
    d.qpos[adr:adr + 7] = [0.75, 0.0, bar_z + 0.08, 1, 0, 0, 0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    peak, lowest_while_touching = 0.0, np.inf
    for _ in range(int(0.4 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        mgr.observe_contacts()
        force = _contact_force(m, d, apple, bar)
        if force > 0:
            lowest_while_touching = min(lowest_while_touching, d.geom_xpos[apple][2])
        peak = max(peak, force)
    assert peak > 0.5  # N; the bar pushes back
    assert mgr.payload_collision and not mgr.robot_collision
    assert any("apple_geom" in pair and "pick_obstacle_hurdle_0_collision_bar" in pair for pair in mgr.contact_pairs)
    # the apple never sinks through: its centre stays above the bar centre while they touch
    assert lowest_while_touching > bar_z + 0.012
    assert d.geom_xpos[apple][2] > bar_z + 0.03  # and it ends up resting on the bar (measured 0.042 m)


def _body_geoms(m, mgr, *bodies):
    ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
    return [g for g in mgr.robot_geom_ids if int(m.geom_bodyid[g]) in ids]


def test_gripper_hitting_the_bar_is_a_contact_and_an_obstacle_stop(scene):
    m, d, mgr = scene
    left = _body_geoms(m, mgr, "EE_BODY_L", "ee_finger_l1", "ee_finger_l2")
    finger = max(_body_geoms(m, mgr, "ee_finger_l1"), key=lambda g: m.geom_rbound[g])
    bar = _gid(m, "pick_obstacle_hurdle_0_collision_bar")
    # bar centre on the finger geom centre: overlap by construction (bypasses the table-workspace validation)
    mgr.activate("hurdle")
    mgr._place("hurdle_0", d.geom_xpos[finger] - [0.0, 0.0, 0.26], 0.0, 0.26)
    mujoco.mj_forward(m, d)
    # overlap: <= 0 is all the checks use (`<= stop_distance`, `<= 0.0`); MuJoCo returns 0.0 for this deep
    # capsule-mesh overlap rather than a penetration depth
    assert _gjk(m, d, bar, finger) == 0.0
    assert mgr.pair_distance(bar, finger, 0.25) <= 0.0
    assert mgr.unsafe(0.02)
    assert mgr.robot_collision
    left_names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom#{g}" for g in left}
    assert any(set(pair) & left_names for pair in mgr.contact_pairs)
    # physically: stepping produces a contact force between the bar and the gripper
    mujoco.mj_step(m, d)
    assert sum(_contact_force(m, d, bar, g) for g in left) > 0.0


# ----------------------------------------------------------------------------- 5. 16D + obstacle via main()


def test_16d_with_a_hurdle_reaches_the_policy_on_the_transport_obstacle_xml(client, tmp_path):
    cfg = _config(position=[0.68, 0.23, TABLE_Z], yaw=0.0, bar_height=0.26)
    path = tmp_path / "hurdle.json"
    path.write_text(json.dumps(cfg))
    run = _run_until_policy(client, ["--obstacle-config", str(path), "--obstacle-profile", "hurdle"])
    assert pathlib.Path(run["model_xml"]).resolve() == TRANSPORT_OBSTACLE_XML.resolve()
    m, d, mgr = run["m"], run["d"], run["obstacle_manager"]
    assert _qadr(m, "apple_free") == 38
    assert mgr.active_slots == ("hurdle_0",)
    assert "obstacles  : hurdle ('hurdle_0',)" in run["_stdout"]
    assert mgr.robot_clearance() > 0.02 and mgr.object_clearance() > 0.0
    bar = _gid(m, "pick_obstacle_hurdle_0_collision_bar")
    np.testing.assert_allclose(d.geom_xpos[bar], [0.68, 0.23, TABLE_Z + 0.26], atol=1e-12)
    # the episode scene itself is the recorded one (apple where 1807 starts)
    apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    np.testing.assert_allclose(d.xpos[apple][:2], [0.5635, 0.3204], atol=2e-3)


def test_the_block_obstacle_xml_still_carries_every_slot():
    m = mujoco.MjModel.from_xml_path(str(BLOCK_OBSTACLE_XML))
    d = mujoco.MjData(m)
    mgr = ppo.PickPlaceObstacleManager(m, d, _config())
    assert set(mgr._collision_geoms) == set(ppo.SLOTS)


# ----------------------------------------------------------------------------- 6. distances


def _gjk(m, d, g1, g2) -> float:
    return gjk_dist(make_support(m, d, g1), make_support(m, d, g2), d.geom_xpos[g2] - d.geom_xpos[g1] + 1e-9)


def _check_pairs(m, d, mgr, others, cap=0.25):
    """Every hurdle pair within `cap`: pair_distance == GJK (separated) or <= 0 (overlap). Returns the GJK
    minimum (capped) and the count per type pair."""
    minimum, counts = cap, {}
    for h in mgr.collision_geom_ids:
        for o in others:
            gap = np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[o]) - m.geom_rbound[h] - m.geom_rbound[o]
            if gap > cap:
                continue
            ours = mgr.pair_distance(h, int(o), cap)
            ref = _gjk(m, d, h, int(o))
            key = (int(m.geom_type[h]), int(m.geom_type[o]))
            counts[key] = counts.get(key, 0) + 1
            if ref == 0.0:
                assert ours <= 0.0, (key, ours)
            else:
                assert ours == pytest.approx(min(ref, cap), abs=5e-5), (key, ours, ref)
            minimum = min(minimum, ref)
    return minimum, counts


def test_reset_and_runtime_clearance_equal_an_independent_gjk():
    """Random arm poses, hurdle placed against a random robot or payload geom; the manager's
    robot_clearance (run-time stop) and object_clearance (reset check) equal the GJK minimum."""
    m, d = _teleop_scene()
    mgr = ppo.PickPlaceObstacleManager(m, d, _config())
    mgr.activate("hurdle")
    rng = np.random.default_rng(40)
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_arm_{i}") for s in ("left", "right") for i in range(7)]
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    hurdle = sorted(mgr._collision_geoms["hurdle_0"])
    seen: dict = {}
    for trial in range(12):
        mujoco.mj_resetDataKeyframe(m, d, key)
        for j in arm:
            lo, hi = m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]] = 0.4 * d.qpos[m.jnt_qposadr[j]] + 0.6 * rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        pool = mgr.payload_geom_ids if trial % 2 else mgr.robot_geom_ids
        target = int(rng.choice(pool))
        part = int(rng.choice(hurdle))
        yaw, height = rng.uniform(-np.pi, np.pi), rng.uniform(0.20, 0.32)
        mgr._place("hurdle_0", np.zeros(3), yaw, height)
        mujoco.mj_forward(m, d)
        offset = d.geom_xpos[part] - d.mocap_pos[mgr._mocap_ids["hurdle_0"]]
        mgr._place("hurdle_0", d.geom_xpos[target] - offset + rng.normal(0.0, 0.03, 3), yaw, height)
        mujoco.mj_forward(m, d)
        robot_ref, c1 = _check_pairs(m, d, mgr, mgr.robot_geom_ids)
        payload_ref, c2 = _check_pairs(m, d, mgr, mgr.payload_geom_ids)
        for k, v in {**c1, **c2}.items():
            seen[k] = seen.get(k, 0) + v
        robot = mgr.robot_clearance(0.25)
        payload = mgr.object_clearance(0.25)
        if robot_ref > 0.0:
            assert robot == pytest.approx(robot_ref, abs=5e-5)
        else:
            assert robot <= 0.0
        if payload_ref > 0.0:
            assert payload == pytest.approx(payload_ref, abs=5e-5)
        else:
            assert payload <= 0.0
    G = mujoco.mjtGeom
    for pair in ((G.mjGEOM_CYLINDER, G.mjGEOM_MESH), (G.mjGEOM_CAPSULE, G.mjGEOM_MESH),
                 (G.mjGEOM_CYLINDER, G.mjGEOM_BOX), (G.mjGEOM_CAPSULE, G.mjGEOM_BOX)):
        assert seen.get((int(pair[0]), int(pair[1])), 0) > 0, f"pair type {pair} never exercised"


def _bar_through_crate_wall(m, d, mgr):
    """Crate at the teleop key pose; bar centre 0.10 m along x, 0.03 m inward, 0.04 m down from the crate_wall_px
    centre, yaw 0 — the bar runs through the wall (found by the T40 E survey)."""
    wall = _gid(m, "crate_wall_px")
    mgr.activate("hurdle")
    centre = d.geom_xpos[wall] + np.array([-0.10, -0.03, -0.04])
    mgr._place("hurdle_0", centre - [0.0, 0.0, 0.26], 0.0, 0.26)
    mujoco.mj_forward(m, d)
    return _gid(m, "pick_obstacle_hurdle_0_collision_bar"), wall


def test_capsule_box_overlap_is_reported_as_overlap(scene):
    m, d, mgr = scene
    bar, wall = _bar_through_crate_wall(m, d, mgr)
    assert _gjk(m, d, bar, wall) == 0.0  # reference: they overlap
    assert ppo.capsule_box_distance(m, d, bar, wall) < -0.01
    assert mgr.pair_distance(bar, wall, 0.25) < 0.0
    assert mgr.pair_distance(wall, bar, 0.25) < 0.0  # either order
    assert mgr.object_clearance() <= 0.0  # the reset check would reject this placement


@pytest.mark.skipif(mujoco.__version__ != "3.11.0", reason="pins the MuJoCo 3.11.0 behaviour")
def test_mujoco_3_11_mj_geomDistance_misses_this_capsule_box_overlap(scene):
    """Why capsule-box pairs bypass mj_geomDistance. If this starts failing on 3.11.0 the workaround is moot."""
    m, d, mgr = scene
    bar, wall = _bar_through_crate_wall(m, d, mgr)
    assert mujoco.mj_geomDistance(m, d, bar, wall, 0.25, np.zeros(6)) > 0.0


def test_capsule_box_distance_matches_gjk_when_separated(scene):
    m, d, mgr = scene
    bar = _gid(m, "pick_obstacle_hurdle_0_collision_bar")
    boxes = [g for g in mgr.payload_geom_ids if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    rng = np.random.default_rng(7)
    mgr.activate("hurdle")
    n_sep = n_overlap = 0
    for _ in range(300):
        box = int(rng.choice(boxes))
        yaw, height = rng.uniform(-np.pi, np.pi), rng.uniform(0.20, 0.32)
        centre = d.geom_xpos[box] + rng.normal(0.0, 0.04, 3)
        mgr._place("hurdle_0", centre - [0.0, 0.0, height], yaw, height)
        mujoco.mj_forward(m, d)
        ref = _gjk(m, d, bar, box)
        ours = ppo.capsule_box_distance(m, d, bar, box)
        if ref == 0.0:
            n_overlap += 1
            assert ours <= 0.0
        else:
            n_sep += 1
            assert ours == pytest.approx(ref, abs=1e-7)
    assert n_sep > 50 and n_overlap > 20


def _robot_gjk(m, d, mgr, near=0.08) -> float:
    best = np.inf
    for h in mgr.collision_geom_ids:
        for r in mgr.robot_geom_ids:
            if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[r]) - m.geom_rbound[h] - m.geom_rbound[r] > near:
                continue
            best = min(best, _gjk(m, d, h, r))
    return best


def test_stop_rule_threshold_on_the_hurdle(scene):
    """The bar 15 mm (GJK) below the gripper stops the run (<= 0.02 m); 30 mm does not."""
    m, d, mgr = scene
    lowest = min(_body_geoms(m, mgr, "ee_finger_l1", "ee_finger_l2"), key=lambda g: d.geom_xpos[g][2])
    x, y, top = d.geom_xpos[lowest]
    mgr.activate("hurdle")
    for gap, stops in ((0.015, True), (0.030, False)):
        lo, hi = top - 0.30, top  # bar centre z; GJK gap shrinks as the bar rises
        for _ in range(26):
            z = 0.5 * (lo + hi)
            mgr._place("hurdle_0", [x, y, z - 0.26], 0.3, 0.26)
            mujoco.mj_forward(m, d)
            lo, hi = (z, hi) if _robot_gjk(m, d, mgr) > gap else (lo, z)
        assert _robot_gjk(m, d, mgr) == pytest.approx(gap, abs=1e-4)
        mgr.reset_metrics()
        assert mgr.unsafe(0.02) is stops, gap
        assert mgr.min_robot_clearance == pytest.approx(gap, abs=1e-4)


def test_bounding_sphere_prefilter_does_not_change_the_minimum():
    """`_minimum_distance` skips pairs whose bounding spheres are > max_distance apart (13.5 -> 0.4 ms per
    stop check for the 7-geom hurdle). Same value as every pair through `pair_distance`."""
    m, d = _teleop_scene()
    cfg = _config()
    cfg["profiles"]["bollard"] = {"scene": "fruit",
                                  "obstacles": [{"slot": "bollard_0", "position": [0.60, 0.15, TABLE_Z], "yaw": 0.0}]}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    rng = np.random.default_rng(3)
    for profile in ("hurdle", "bollard"):
        mgr.activate(profile)
        for _ in range(6):
            slot = mgr.active_slots[0]
            mgr._place(slot, [rng.uniform(0.40, 0.75), rng.uniform(-0.3, 0.3), TABLE_Z + rng.uniform(0.0, 0.25)],
                       rng.uniform(-np.pi, np.pi), rng.uniform(0.20, 0.32))
            mujoco.mj_forward(m, d)
            for others in (mgr.robot_geom_ids, mgr.payload_geom_ids):
                for cap in (0.07, 0.25):
                    brute = min([cap] + [mgr.pair_distance(h, int(o), cap)
                                         for h in mgr.collision_geom_ids for o in others])
                    assert mgr._minimum_distance(others, cap) == brute
