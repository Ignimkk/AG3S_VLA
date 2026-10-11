"""T40 H — the client obstacle distance no longer returns a false 0.0 (pi05_TO_hybrid).

T40 G: 5 closed-loop runs stopped at ``--obstacle-stop-distance 0`` with clearance exactly 0.0 and
no contact (E3b C1995 · D1967 · D1982 · D1995, E0 D1967).  Cause: MuJoCo 3.11 ``mj_geomDistance``
returns 0.0 for a hurdle foot (cylinder) vs an arm/finger mesh that are 0.070-0.236 m apart
(GJK).  The manager now keeps a raw value only if :func:`convex_distance.certify` passes and
otherwise uses an exact GJK (:func:`convex_distance.gjk_distance`).

What this file pins:

1. **The 5 stop states** (``data/t40h_stop_states.json``, reproduced bit for bit by replay): the
   raw pair value is 0.0 (MuJoCo 3.11.0), and the manager's robot / object clearance and every
   obstacle pair equal the independent GJK (``tests/sim/_gjk.py``) within 0.1 mm.
2. **Replay sweep** (needs ``outputs/verify/T40``): ``main()`` reset + the recorded ``applied_ctrl``
   reproduces every recorded control row's qpos bit for bit and reaches the recorded stop
   sub-step; at every stop-check state (every 5th sub-step) and every recorded control row the
   robot clearance equals GJK within 0.1 mm (object clearance at the control rows), and no
   recorded state stops any more.
3. **The other slots**: bollard / divider (box) pairs vs GJK, and T40 A's divider reset case
   (object clearance 0.0 -> the GJK value; the reset check no longer rejects it).
4. **convex_distance** on its own: exact GJK vs the reference for every geom type pair; certify
   accepts MuJoCo's correct answers and refuses the false 0.0.

CPU only: ``JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa``.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")

from tests.sim._gjk import gjk_dist, make_support  # noqa: E402

HYBRID = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid'))
MJCF = HYBRID / "rby1_description" / "models" / "rby1a" / "mujoco"
TRANSPORT_OBSTACLE_XML = MJCF / "model_transport_pick_place_obstacles.xml"
PI05_INFER = HYBRID / "rby1_bringup" / "pi05_infer.py"
T40 = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T40'))
STOP_STATES = pathlib.Path(__file__).parent / "data" / "t40h_stop_states.json"
CAP = 0.25
TOL = 1e-4  # T40 H requirement: client distance = GJK within 0.1 mm

if not TRANSPORT_OBSTACLE_XML.exists():  # pragma: no cover - asset-dependent
    pytest.skip("pi05_TO_hybrid scene assets not present", allow_module_level=True)
if str(HYBRID / "rby1_manipulation" / "src") not in sys.path:
    sys.path.insert(0, str(HYBRID / "rby1_manipulation" / "src"))

from rby1_manipulation.simulation import convex_distance as cd  # noqa: E402
from rby1_manipulation.simulation import pick_place_obstacles as ppo  # noqa: E402

STOPS = json.loads(STOP_STATES.read_text())["runs"]
RUNS = sorted(STOPS)
G = mujoco.mjtGeom


def _gjk(m, d, g1, g2) -> float:
    return gjk_dist(make_support(m, d, g1), make_support(m, d, g2), d.geom_xpos[g2] - d.geom_xpos[g1] + 1e-9)


def _gjk_minimum(m, d, obstacles, others, cap=CAP) -> float:
    """Exact capped minimum by GJK (reference supports only). Pairs in bounding-sphere-gap order (a lower
    bound; stop past the best); a pair is also skipped when its support gap along the centre-to-centre axis
    (another lower bound) is past the best."""
    obstacles = np.array(sorted(obstacles))
    others = np.array(list(others))
    gap = (np.linalg.norm(d.geom_xpos[others][None] - d.geom_xpos[obstacles][:, None], axis=2)
           - m.geom_rbound[others][None] - m.geom_rbound[obstacles][:, None])
    best = cap
    for index in np.argsort(gap, axis=None):
        if gap.flat[index] > best:
            break
        i, j = divmod(int(index), len(others))
        g1, g2 = int(obstacles[i]), int(others[j])
        s1, s2 = make_support(m, d, g1), make_support(m, d, g2)
        axis = d.geom_xpos[g2] - d.geom_xpos[g1]
        norm = float(np.linalg.norm(axis))
        if norm > 0.0:
            axis = axis / norm
            if float(axis @ s2(-axis)) - float(axis @ s1(axis)) > best:
                continue
        best = min(best, gjk_dist(s1, s2, d.geom_xpos[g2] - d.geom_xpos[g1] + 1e-9))
    return best


@pytest.fixture(scope="module")
def model():
    m = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    return m, mujoco.MjData(m)


def _stop_scene(model, run):
    m, d = model
    entry = STOPS[run]
    mgr = ppo.PickPlaceObstacleManager(m, d, entry["config"])
    mgr.activate(entry["profile"])
    d.qpos[:] = entry["qpos_kin"]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)  # geom_xpos of the stop check bit for bit, and this pose's contacts
    return m, d, mgr, entry


# ----------------------------------------------------------------------------- 1. the 5 stop states


@pytest.mark.skipif(mujoco.__version__ != "3.11.0", reason="pins the MuJoCo 3.11.0 behaviour")
@pytest.mark.parametrize("run", RUNS)
def test_stop_state_raw_mj_geomDistance_is_a_false_zero(model, run):
    """The cause, at the exact stop state: foot cylinder vs mesh, 0.0 in both orders, GJK 7-24 cm, witness not 0."""
    m, d, mgr, entry = _stop_scene(model, run)
    assert entry["false_zero_pairs"], run
    for pair in entry["false_zero_pairs"]:
        foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, pair["obstacle_geom"])
        mesh = pair["robot_geom_id"]
        assert int(m.geom_type[foot]) == G.mjGEOM_CYLINDER and int(m.geom_type[mesh]) == G.mjGEOM_MESH
        from_to = np.zeros(6)
        assert mujoco.mj_geomDistance(m, d, foot, mesh, CAP, from_to) == 0.0
        assert np.linalg.norm(from_to[3:] - from_to[:3]) > 0.05  # not a contact
        assert mujoco.mj_geomDistance(m, d, mesh, foot, CAP, np.zeros(6)) == 0.0
        reference = _gjk(m, d, foot, mesh)
        assert 0.05 < reference < CAP
        # below the true gap MuJoCo's cut-off exit fires first and the capped answer is right
        assert mujoco.mj_geomDistance(m, d, foot, mesh, 0.99 * reference, np.zeros(6)) == 0.99 * reference
        assert mgr.pair_distance(foot, mesh, CAP) == pytest.approx(reference, abs=1e-9)


@pytest.mark.parametrize("run", RUNS)
def test_stop_state_clearances_equal_gjk(model, run):
    m, d, mgr, _ = _stop_scene(model, run)
    robot_ref = _gjk_minimum(m, d, mgr.collision_geom_ids, mgr.robot_geom_ids)
    object_ref = _gjk_minimum(m, d, mgr.collision_geom_ids, mgr.payload_geom_ids)
    assert robot_ref > 0.01 and object_ref > 0.0
    assert mgr.robot_clearance() == pytest.approx(robot_ref, abs=TOL)
    assert mgr.object_clearance() == pytest.approx(object_ref, abs=TOL)
    mgr.reset_metrics()
    assert mgr.unsafe(0.0) is False  # the run-time rule of T40 G (--obstacle-stop-distance 0)


@pytest.mark.parametrize("run", RUNS)
def test_stop_state_every_obstacle_pair_equals_gjk(model, run):
    """Every hurdle geom x (robot + payload) geom within the cap: pair_distance = GJK (capped)."""
    m, d, mgr, _ = _stop_scene(model, run)
    n = 0
    for h in mgr.collision_geom_ids:
        for o in (*mgr.robot_geom_ids, *mgr.payload_geom_ids):
            if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[o]) - m.geom_rbound[h] - m.geom_rbound[o] > CAP:
                continue
            reference = _gjk(m, d, h, int(o))
            ours = mgr.pair_distance(h, int(o), CAP)
            if reference == 0.0:
                assert ours <= 0.0
            else:
                assert ours == pytest.approx(min(reference, CAP), abs=TOL), (h, o)
            n += 1
    assert n > 1000


# ----------------------------------------------------------------------------- 2. replay of the recorded runs


class _Reached(Exception):
    pass


@pytest.fixture(scope="module")
def client():
    try:
        import PIL  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("PIL missing")
    spec = importlib.util.spec_from_file_location("_t40h_pi05_infer", PI05_INFER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:  # pragma: no cover - the client needs .venv-openpi-live (mink, PIL)
        pytest.skip(f"pi05_infer.py dependencies missing in this interpreter: {exc}")
    return mod


def _reset(client, episode, seed, config_path, profile) -> dict:
    """``main()`` up to ``load_remote_policy`` (reset, obstacle manager, reset checks); returns its locals."""
    captured: dict = {}

    def stub(remote):
        captured.update(inspect.currentframe().f_back.f_locals)
        raise _Reached

    original, old_argv = client.load_remote_policy, sys.argv
    client.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                "--episode-index", str(episode), "--headless", "--warmup-steps", "1000", "--policy-seed", str(seed),
                "--obstacle-stop-distance", "0", "--obstacle-config", str(config_path), "--obstacle-profile", profile]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            client.main()
    except _Reached:
        pass
    except SystemExit as exc:
        raise AssertionError(f"main() exited ({exc.code}):\n{out.getvalue()[-2000:]}") from exc
    finally:
        client.load_remote_policy, sys.argv = original, old_argv
    assert captured, "main() never reached load_remote_policy"
    return captured


@pytest.mark.parametrize("run", RUNS)
def test_replay_of_the_recorded_run_sweeps_clean(client, run):
    run_dir = T40 / "G" / "runs" / run
    config_path = T40 / "F" / "configs" / f"{STOPS[run]['profile']}.json"
    if not (run_dir / "frames.jsonl").exists() or not config_path.exists():
        pytest.skip(f"T40 G records not present: {run_dir}")
    entry = STOPS[run]
    rows = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] == "control"]
    local = _reset(client, entry["episode"], entry["policy_seed"], config_path, entry["profile"])
    m, d, mgr, act = local["m"], local["d"], local["obstacle_manager"], local["act"]
    steps_per_action = max(1, int(round(1.0 / (client.CTRL_HZ * m.opt.timestep))))
    for _ in range(1000):  # --warmup-steps 1000 (wait_before_inference)
        mujoco.mj_step(m, d)
    worst = 0.0
    n_checks = 0
    stop_state_seen = False
    for row in rows:
        assert np.array_equal(np.asarray(row["qpos"]), d.qpos), f"t={row['t_step']}: replay diverged"
        # the recorded control row itself (= d.qpos now; kinematics only, mj_step recomputes it):
        # robot (stop check) and payload (reset check) minimum
        mujoco.mj_kinematics(m, d)
        for others in (mgr.robot_geom_ids, mgr.payload_geom_ids):
            reference = _gjk_minimum(m, d, mgr.collision_geom_ids, others)
            worst = max(worst, abs(mgr._minimum_distance(others, CAP) - reference))
        applied = row["applied_ctrl"]
        for i in range(7):
            d.ctrl[act["left_a"][i]] = applied["arm"][i]
            d.ctrl[act["right_a"][i]] = applied["arm"][7 + i]
        d.ctrl[act["left_grip_a"]] = applied["gripper"][0] * client.RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = applied["gripper"][1] * client.RBY1_GRIPPER_OPEN
        for sub in range(steps_per_action):
            before = d.qpos.copy()
            mujoco.mj_step(m, d)
            mgr.observe_contacts()
            if sub % 5:
                continue
            clearance = mgr.robot_clearance()  # the client's stop check, same state
            reference = _gjk_minimum(m, d, mgr.collision_geom_ids, mgr.robot_geom_ids)
            worst = max(worst, abs(clearance - reference))
            n_checks += 1
            assert not mgr.robot_collision
            assert clearance > 0.0, (row["t_step"], sub, clearance, reference)
            if row["t_step"] == entry["stop_t_step"] and sub == entry["stop_substep"]:
                assert np.array_equal(before, np.asarray(entry["qpos_kin"]))
                stop_state_seen = True
    assert stop_state_seen
    assert n_checks >= 7 * (len(rows) - 1)
    assert worst <= TOL, worst


# ----------------------------------------------------------------------------- 3. the other slots


def _teleop_scene():
    m = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop"))
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize("slot", ["bollard_0", "bollard_1", "divider_0", "hurdle_0"])
def test_every_slot_pair_equals_gjk_near_the_robot_and_payload(slot):
    """Each slot placed with one of its geoms at a random robot / payload geom (+-3 cm), random yaw: every
    pair within the cap = GJK (separated) or <= 0 (overlap); robot / object clearance = GJK minimum."""
    m, d = _teleop_scene()
    cfg = {"version": 1, "table_surface_z": 0.82,
           "profiles": {"clear": {"scene": "any", "obstacles": []}, "one": {"scene": "fruit", "obstacles": [{"slot": slot, "position": [0.6, 0.0, 0.82], "yaw": 0.0}]}}}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    mgr.activate("one")
    rng = np.random.default_rng(41)
    arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_arm_{i}") for s in ("left", "right") for i in range(7)]
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    geoms = sorted(mgr._collision_geoms[slot])
    n_pairs = n_overlap = 0
    for trial in range(8):
        mujoco.mj_resetDataKeyframe(m, d, key)
        for j in arm:
            lo, hi = m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]] = 0.4 * d.qpos[m.jnt_qposadr[j]] + 0.6 * rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        target = int(rng.choice(mgr.payload_geom_ids if trial % 2 else mgr.robot_geom_ids))
        part = int(rng.choice(geoms))
        yaw = rng.uniform(-np.pi, np.pi)
        mgr._place(slot, np.zeros(3), yaw, ppo.HURDLE_DEFAULT_BAR_HEIGHT)
        mujoco.mj_kinematics(m, d)
        offset = d.geom_xpos[part] - d.mocap_pos[mgr._mocap_ids[slot]]
        mgr._place(slot, d.geom_xpos[target] - offset + rng.normal(0.0, 0.03, 3), yaw, ppo.HURDLE_DEFAULT_BAR_HEIGHT)
        mujoco.mj_kinematics(m, d)
        for others in (mgr.robot_geom_ids, mgr.payload_geom_ids):
            for h in geoms:
                for o in others:
                    if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[o]) - m.geom_rbound[h] - m.geom_rbound[o] > CAP:
                        continue
                    reference, ours = _gjk(m, d, h, int(o)), mgr.pair_distance(h, int(o), CAP)
                    n_pairs += 1
                    if reference == 0.0:
                        n_overlap += 1
                        assert ours <= 0.0, (h, o, ours)
                    else:
                        assert ours == pytest.approx(min(reference, CAP), abs=TOL), (h, o, ours, reference)
            reference = _gjk_minimum(m, d, geoms, others)
            ours = mgr._minimum_distance(others, CAP)
            if reference > 0.0:
                assert ours == pytest.approx(reference, abs=TOL)
            else:
                assert ours <= 0.0
    assert n_pairs > 500 and n_overlap > 0


def test_t40a_divider_reset_check_no_longer_sees_a_false_zero(client):
    """T40 A: ep 1967 + divider (box) — the reset object check read 0.0 (box-box / box-mesh, GJK 117 / 189 mm) and
    the client exited. Now main() reaches the policy call and the object clearance is the GJK value."""
    config_path = T40 / "A" / "configs" / "1967_19672_divider.json"
    if not config_path.exists():
        pytest.skip("T40 A divider config not present")
    local = _reset(client, 1967, 19672, config_path, "t40_ep1967_s19672_divider")
    m, d, mgr = local["m"], local["d"], local["obstacle_manager"]
    reference = _gjk_minimum(m, d, mgr.collision_geom_ids, mgr.payload_geom_ids)
    assert reference > 0.01
    assert local["initial_object_clearance"] == pytest.approx(reference, abs=TOL)
    assert local["initial_robot_clearance"] == pytest.approx(
        _gjk_minimum(m, d, mgr.collision_geom_ids, mgr.robot_geom_ids), abs=TOL)
    raw_zero = [(h, o) for h in mgr.collision_geom_ids for o in mgr.payload_geom_ids
                if mujoco.mj_geomDistance(m, d, h, o, CAP, np.zeros(6)) <= 0.0]
    if mujoco.__version__ == "3.11.0":
        assert raw_zero  # the box pairs MuJoCo gets wrong here
    for h, o in raw_zero:
        assert _gjk(m, d, h, o) > 0.1
        assert mgr.pair_distance(h, o, CAP) == pytest.approx(min(_gjk(m, d, h, o), CAP), abs=TOL)


# ----------------------------------------------------------------------------- 4. convex_distance on its own

_SHAPES = """
<mujoco>
  <asset><mesh name="hull" vertex="{verts}"/></asset>
  <worldbody>
    <body name="a" mocap="true"><geom name="a_box" type="box" size="0.03 0.05 0.02"/></body>
    <body name="b" mocap="true"><geom name="b_sphere" type="sphere" size="0.025"/></body>
    <body name="c" mocap="true"><geom name="c_capsule" type="capsule" size="0.012 0.13"/></body>
    <body name="e" mocap="true"><geom name="e_cylinder" type="cylinder" size="0.035 0.006"/></body>
    <body name="f" mocap="true"><geom name="f_cylinder" type="cylinder" size="0.01 0.08"/></body>
    <body name="g" mocap="true"><geom name="g_ellipsoid" type="ellipsoid" size="0.03 0.02 0.045"/></body>
    <body name="h" mocap="true"><geom name="h_mesh" type="mesh" mesh="hull"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def shapes():
    rng = np.random.default_rng(5)
    verts = rng.normal(0.0, 1.0, (40, 3)) * [0.04, 0.025, 0.06]
    m = mujoco.MjModel.from_xml_string(_SHAPES.format(verts=" ".join(f"{v:.6f}" for v in verts.ravel())))
    return m, mujoco.MjData(m)


def _random_pose(m, d, rng, spread):
    for k in range(m.nmocap):
        d.mocap_pos[k] = rng.uniform(-spread, spread, 3)
        q = rng.normal(size=4)
        d.mocap_quat[k] = q / np.linalg.norm(q)
    mujoco.mj_kinematics(m, d)


def test_exact_gjk_matches_the_reference_for_every_type_pair(shapes):
    m, d = shapes
    support = cd.GeomSupport(m, d)
    rng = np.random.default_rng(11)
    n_sep = n_overlap = 0
    for trial in range(60):
        _random_pose(m, d, rng, 0.12 if trial % 2 else 0.05)
        for g1 in range(m.ngeom):
            for g2 in range(g1 + 1, m.ngeom):
                reference = _gjk(m, d, g1, g2)
                ours = cd.gjk_distance(support, g1, g2)
                if reference == 0.0:
                    n_overlap += 1
                    assert ours <= 1e-9, (g1, g2, ours)
                else:
                    n_sep += 1
                    assert ours == pytest.approx(reference, abs=1e-7), (g1, g2, ours, reference)
    assert n_sep > 300 and n_overlap > 100


def test_exact_gjk_handles_a_flat_foot_far_below_a_mesh(shapes):
    """The T40 G shape: flat cylinder (r 35 mm x 12 mm, axis vertical) 7-25 cm under a mesh, many poses."""
    m, d = shapes
    support = cd.GeomSupport(m, d)
    foot, mesh = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("e_cylinder", "h_mesh"))
    rng = np.random.default_rng(12)
    for _ in range(300):
        _random_pose(m, d, rng, 0.0)
        d.mocap_quat[m.body_mocapid[m.geom_bodyid[foot]]] = [1.0, 0.0, 0.0, 0.0]
        d.mocap_pos[m.body_mocapid[m.geom_bodyid[mesh]]] = rng.uniform([-0.15, -0.15, 0.10], [0.15, 0.15, 0.25])
        mujoco.mj_kinematics(m, d)
        assert cd.gjk_distance(support, foot, mesh) == pytest.approx(_gjk(m, d, foot, mesh), abs=1e-7)


def test_certify_accepts_correct_answers_and_refuses_wrong_ones(shapes):
    m, d = shapes
    support = cd.GeomSupport(m, d)
    rng = np.random.default_rng(13)
    accepted = separated = 0
    for _ in range(60):
        _random_pose(m, d, rng, 0.12)
        for g1 in range(m.ngeom):
            for g2 in range(g1 + 1, m.ngeom):
                from_to = np.zeros(6)
                raw = mujoco.mj_geomDistance(m, d, g1, g2, 1.0, from_to)
                reference = _gjk(m, d, g1, g2)
                if reference < 1e-4:
                    continue
                separated += 1
                if cd.certify(support, g1, g2, raw, from_to):
                    accepted += 1
                    assert raw == pytest.approx(reference, abs=cd.CERTIFY_TOL)
                # the T40 G failure: a 0.0 with a non-zero witness; and a value that is too small
                assert not cd.certify(support, g1, g2, 0.0, from_to)
                assert not cd.certify(support, g1, g2, 0.5 * reference, from_to)
                shrunk = from_to.copy()
                shrunk[3:] = from_to[:3] + 0.5 * (from_to[3:] - from_to[:3])
                assert not cd.certify(support, g1, g2, 0.5 * raw, shrunk)
    assert accepted >= 0.95 * separated  # the cheap path is the common path


def test_manager_minimum_equals_brute_force_over_checked_pairs():
    """The sorted sweep with distmax = running minimum gives the same value as every pair through pair_distance."""
    m, d = _teleop_scene()
    cfg = {"version": 1, "table_surface_z": 0.82, "profiles": {
        "clear": {"scene": "any", "obstacles": []},
        "h": {"scene": "fruit", "obstacles": [{"slot": "hurdle_0", "position": [0.5, 0.2, 0.82], "yaw": 0.3}]}}}
    mgr = ppo.PickPlaceObstacleManager(m, d, cfg)
    mgr.activate("h")
    rng = np.random.default_rng(17)
    for _ in range(6):
        mgr._place("hurdle_0", [rng.uniform(0.40, 0.75), rng.uniform(-0.3, 0.3), 0.82 + rng.uniform(0.0, 0.2)],
                   rng.uniform(-math.pi, math.pi), rng.uniform(0.20, 0.32))
        mujoco.mj_kinematics(m, d)
        for others in (mgr.robot_geom_ids, mgr.payload_geom_ids):
            for cap in (0.07, 0.25):
                brute = min([cap] + [mgr.pair_distance(h, int(o), cap)
                                     for h in mgr.collision_geom_ids for o in others])
                assert mgr._minimum_distance(others, cap) == pytest.approx(brute, abs=1e-9)
