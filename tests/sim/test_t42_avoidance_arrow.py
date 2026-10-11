"""T42 — avoidance arrow overlay (`--viz-avoidance-arrow`, pi05_TO_hybrid `rby1_bringup/avoidance_arrow.py`).

What this file pins:

1. **The vector.** ``v = TCP(q_to) - TCP(q_reference)``, both by FK of the commanded arm targets on the same
   model at the same base ``qpos``. It equals an independent ``mj_forward`` on a fresh ``MjData``, bit for bit,
   and ``J_site @ dq`` to first order. Only the arm that moved deflects, and the 16-D and 14-D row layouts map
   like ``apply_action``.
2. **Threshold, scale, colours.** No arrow at ``|v| <= threshold``; above it, one arrow from the rendered TCP of
   length ``scale * |v|`` (capped), green on execute. HOLD gives a red arrow ``q_hold - reference``, or a red
   marker when that is below the threshold or there is no reference. No pair (E0) gives nothing.
3. **Drawing.** ``draw`` appends MuJoCo ``mjGEOM_ARROW`` / sphere decor geoms (pos, length, direction, rgba).
   Pixels change only where geoms were added, and ``maxgeom`` is respected.
4. **Read only.** ``step`` leaves every ``MjData`` array of the run, and the RNG, untouched. The optional
   obstacle normal equals the obstacle manager's checked distance.
5. **Purely visual, end to end.** ``pi05_infer.main()`` with ``--safe-remote`` against a fake server
   (execute / unsafe HOLD with a reference / comms HOLD / execute), 32 steps with ``--record``,
   ``--record-frames``, ``--trajectory-out`` and a hurdle obstacle:
   * flag off = the pre-arrow client (``pi05_infer.py`` of the newest commit without the flag, HEAD until
     T42-arrow is committed) on every field that two flag-off runs
     agree on: ``frames.jsonl``, ``traj.npz``, server requests, RNG, mp4 pixels;
   * flag on (+ obstacle normal) = flag off on the same fields. ``manifest.extra.argv`` is the only extra
     difference; the mp4 frames differ.

CPU only: ``JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa``.
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
TRANSPORT_XML = MJCF / "model_transport.xml"
TRANSPORT_OBSTACLE_XML = MJCF / "model_transport_pick_place_obstacles.xml"
PI05_INFER = BRINGUP / "pi05_infer.py"
EPISODE = 1968
#: the T40 F ep1968 hurdle (outputs/verify/T40/F/configs/t40f_ep1968_s19681_r1.json), inlined
HURDLE_CONFIG = {
    "version": 1, "table_surface_z": 0.82,
    "profiles": {
        "clear": {"scene": "any", "description": "no obstacle", "obstacles": []},
        "t42_arrow_hurdle": {
            "scene": "fruit", "description": "T42 arrow test: the T40 F ep1968 hurdle",
            "obstacles": [{"slot": "hurdle_0", "position": [0.4939, 0.1778, 0.82],
                           "yaw": 0.311739, "bar_height": 0.24}]}},
}

for _p in (str(BRINGUP), str(HYBRID / "rby1_manipulation" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

aa = pytest.importorskip("avoidance_arrow")


# ============================================================================ helpers


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(TRANSPORT_XML))


def _teleop_data(m):
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    return d


def _arm_adr(m, side):
    return [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in aa.ARM_JOINTS[side]]


def _row16(m, d):
    """The current arm pose as a 16-D action row (grippers open)."""
    return np.r_[d.qpos[_arm_adr(m, "left")], 1.0, d.qpos[_arm_adr(m, "right")], 1.0]


def _independent_tcp(m, base_qpos, side, q7):
    """Fresh MjData + mj_forward: an FK path that shares nothing with the module."""
    d = mujoco.MjData(m)
    d.qpos[:] = base_qpos
    d.qpos[_arm_adr(m, side)] = q7
    mujoco.mj_forward(m, d)
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, aa.TCP_SITES[side])].copy()


_DATA_FIELDS = ("qpos", "qvel", "qacc", "act", "ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos",
                "mocap_quat", "xpos", "xquat", "xmat", "site_xpos", "site_xmat", "geom_xpos",
                "geom_xmat", "qfrc_actuator", "cacc", "cfrc_int", "sensordata")


def _snapshot(d):
    return {k: np.array(getattr(d, k), copy=True) for k in _DATA_FIELDS} | {"time": d.time}


def _assert_same(a, b):
    for k in a:
        assert np.array_equal(a[k], b[k]), k


# ============================================================================ 1. the vector


def test_row_layouts_match_apply_action():
    row = np.arange(16, dtype=float)
    t = aa.arm_targets("rby1_16d", row)
    assert np.array_equal(t["left"], row[0:7]) and np.array_equal(t["right"], row[8:15])
    row14 = np.arange(14, dtype=float)
    t = aa.arm_targets("rby1", row14, arm6_hold={"left": -7.0, "right": -8.0})
    assert np.array_equal(t["left"], np.r_[row14[0:6], -7.0])
    assert np.array_equal(t["right"], np.r_[row14[7:13], -8.0])
    with pytest.raises(ValueError):
        aa.arm_targets("rby1", row14)
    with pytest.raises(ValueError):
        aa.arm_targets("droid", row)
    with pytest.raises(ValueError):
        aa.arm_targets("rby1_16d", row[:14])


@pytest.mark.parametrize("side", ["left", "right"])
def test_vector_equals_independent_fk_and_jacobian(model, side):
    m = model
    d = _teleop_data(m)
    ref = _row16(m, d)
    dq = np.array([0.03, -0.02, 0.04, -0.05, 0.02, 0.03, -0.01])
    refined = ref.copy()
    sl = slice(0, 7) if side == "left" else slice(8, 15)
    refined[sl] += dq
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left", "right"), threshold_m=0.0)
    ov = viz.step(d, executed=True, refined_row=refined, reference_row=ref)

    expect = (_independent_tcp(m, d.qpos, side, refined[sl])
              - _independent_tcp(m, d.qpos, side, ref[sl]))
    assert ov.side == side                                   # the arm that moved
    assert np.array_equal(ov.vector_m, expect)               # same FK, bit for bit
    assert ov.magnitude_m == pytest.approx(np.linalg.norm(expect), abs=0)
    assert ov.magnitude_m > 0.01                            # a real deflection (cm scale)
    other = "right" if side == "left" else "left"
    assert np.array_equal(viz.kin.displacement(d.qpos, other, ref[0:7] if other == "left" else ref[8:15],
                                               refined[0:7] if other == "left" else refined[8:15]),
                          np.zeros(3))

    # first order: v ~= J_site(q_ref) @ dq for a small dq
    eps = 1e-5
    small = ref.copy()
    small[sl] += eps * dq
    v_small = viz.step(d, executed=True, refined_row=small, reference_row=ref).vector_m
    dj = mujoco.MjData(m)
    dj.qpos[:] = d.qpos
    dj.qpos[_arm_adr(m, side)] = ref[sl]
    mujoco.mj_forward(m, dj)
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, dj, jacp, None, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, aa.TCP_SITES[side]))
    dofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in aa.ARM_JOINTS[side]]
    np.testing.assert_allclose(v_small, jacp[:, dofs] @ (eps * dq), rtol=1e-3, atol=1e-12)


def test_vector_is_taken_at_the_current_base_pose(model):
    """Torso / base move both TCPs; the vector is computed for the current one, and anchored at the rendered TCP."""
    m = model
    d = _teleop_data(m)
    ref = _row16(m, d)
    refined = ref.copy()
    refined[1] += 0.05
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",), threshold_m=0.0)
    v0 = viz.step(d, executed=True, refined_row=refined, reference_row=ref).vector_m
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "torso_0")
    assert torso >= 0
    d.qpos[m.jnt_qposadr[torso]] += 0.3
    mujoco.mj_forward(m, d)
    ov = viz.step(d, executed=True, refined_row=refined, reference_row=ref)
    expect = (_independent_tcp(m, d.qpos, "left", refined[0:7]) - _independent_tcp(m, d.qpos, "left", ref[0:7]))
    assert np.array_equal(ov.vector_m, expect)
    assert not np.allclose(ov.vector_m, v0)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    assert np.array_equal(ov.items[0].start, d.site_xpos[site])


# ============================================================================ 2. threshold, scale, colours


def _pair_with_magnitude(m, d, viz, target_m):
    """A refined row whose left TCP deflection is ~target_m (bisection on one joint)."""
    ref = _row16(m, d)
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = ref.copy()
        r[3] += mid
        mag = np.linalg.norm(viz.kin.displacement(d.qpos, "left", ref[0:7], r[0:7]))
        lo, hi = (mid, hi) if mag < target_m else (lo, mid)
    r = ref.copy()
    r[3] += hi
    return ref, r


def test_threshold_gates_the_arrow_and_scale_sets_its_length(model):
    m = model
    d = _teleop_data(m)
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",), threshold_m=0.005, scale=8.0)
    ref, below = _pair_with_magnitude(m, d, viz, 0.0049)
    ov = viz.step(d, executed=True, refined_row=below, reference_row=ref)
    assert ov.magnitude_m < 0.005 and not ov.drawn and ov.items == [] and ov.state == "execute"
    ref, above = _pair_with_magnitude(m, d, viz, 0.0051)
    ov = viz.step(d, executed=True, refined_row=above, reference_row=ref)
    assert ov.magnitude_m > 0.005 and ov.drawn and len(ov.items) == 1
    item = ov.items[0]
    assert item.kind == "arrow" and item.rgba == aa.GREEN
    np.testing.assert_allclose(item.end - item.start, 8.0 * ov.vector_m, atol=1e-15)
    # threshold configurable
    viz2 = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",), threshold_m=0.006, scale=3.0)
    assert viz2.step(d, executed=True, refined_row=above, reference_row=ref).items == []
    # length cap
    viz3 = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",), threshold_m=0.0, scale=1000.0)
    ov = viz3.step(d, executed=True, refined_row=above, reference_row=ref)
    assert ov.clipped
    assert np.linalg.norm(ov.items[0].end - ov.items[0].start) == pytest.approx(aa.MAX_DRAWN_LENGTH_M, rel=1e-12)
    np.testing.assert_allclose((ov.items[0].end - ov.items[0].start) / aa.MAX_DRAWN_LENGTH_M,
                               ov.vector_m / ov.magnitude_m, atol=1e-12)
    with pytest.raises(ValueError):
        aa.AvoidanceArrows(m, action_format="rby1_16d", threshold_m=-1.0)
    with pytest.raises(ValueError):
        aa.AvoidanceArrows(m, action_format="rby1_16d", scale=0.0)


def test_hold_is_red_q_hold_minus_reference_or_a_marker(model):
    m = model
    d = _teleop_data(m)
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",), threshold_m=0.005)
    ref = _row16(m, d)
    q_hold = ref.copy()
    q_hold[1] -= 0.06
    refined = ref.copy()
    refined[1] += 0.2                     # must NOT be used on HOLD
    ov = viz.step(d, executed=False, refined_row=refined, reference_row=ref, applied_row=q_hold)
    expect = _independent_tcp(m, d.qpos, "left", q_hold[0:7]) - _independent_tcp(m, d.qpos, "left", ref[0:7])
    assert ov.state == "hold" and ov.drawn
    assert np.array_equal(ov.vector_m, expect)
    assert [(i.kind, i.rgba) for i in ov.items] == [("arrow", aa.RED)]
    # applied_arm (replay form) = applied_row
    ov2 = viz.step(d, executed=False, reference_row=ref,
                   applied_arm={"left": q_hold[0:7], "right": q_hold[8:15]})
    assert np.array_equal(ov2.vector_m, ov.vector_m)
    # q_hold == reference: below threshold -> red marker at the TCP
    ov = viz.step(d, executed=False, refined_row=refined, reference_row=ref, applied_row=ref)
    assert ov.magnitude_m == 0.0 and not ov.drawn
    assert [(i.kind, i.rgba) for i in ov.items] == [("marker", aa.RED)]
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    assert np.array_equal(ov.items[0].start, d.site_xpos[site])
    # no reference (comms HOLD): marker only, no vector
    ov = viz.step(d, executed=False, refined_row=None, reference_row=None, applied_row=q_hold)
    assert ov.state == "hold" and ov.vector_m is None and ov.magnitude_m is None
    assert [(i.kind, i.rgba) for i in ov.items] == [("marker", aa.RED)]
    assert "no reference" in ov.caption(8.0, 0.005)[0]


def test_no_pair_draws_nothing(model):
    m = model
    d = _teleop_data(m)
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",))
    ref = _row16(m, d)
    for kw in ({"refined_row": ref, "reference_row": None}, {"refined_row": None, "reference_row": ref},
               {"refined_row": None, "reference_row": None}):
        ov = viz.step(d, executed=True, **kw)
        assert ov.state == "none" and ov.items == [] and ov.caption(8.0, 0.005) == []


# ============================================================================ 3. drawing


def test_draw_adds_arrow_and_marker_geoms_with_the_right_pose(model):
    m = model
    d = _teleop_data(m)
    r = mujoco.Renderer(m, height=120, width=160)
    try:
        r.update_scene(d)
        n0 = r.scene.ngeom
        start, end = np.array([0.4, 0.1, 1.0]), np.array([0.4, 0.2, 1.1])
        items = [aa.DrawItem("arrow", start, end, aa.GREEN, 0.012),
                 aa.DrawItem("marker", start, start, aa.RED, 0.015)]
        assert aa.draw(r.scene, items) == 2
        assert r.scene.ngeom == n0 + 2
        g = r.scene.geoms[n0]
        assert g.type == mujoco.mjtGeom.mjGEOM_ARROW
        np.testing.assert_allclose(g.pos, start, atol=1e-6)
        assert g.size[2] == pytest.approx(np.linalg.norm(end - start), rel=1e-6)
        assert g.size[0] == pytest.approx(0.012)
        np.testing.assert_allclose(np.asarray(g.mat).reshape(3, 3)[:, 2], (end - start) / np.linalg.norm(end - start),
                                   atol=1e-6)
        np.testing.assert_allclose(g.rgba, aa.GREEN, atol=1e-7)
        s = r.scene.geoms[n0 + 1]
        assert s.type == mujoco.mjtGeom.mjGEOM_SPHERE and s.size[0] == pytest.approx(0.015)
        np.testing.assert_allclose(s.rgba, aa.RED, atol=1e-7)
        # maxgeom respected
        r.scene.ngeom = r.scene.maxgeom - 1
        assert aa.draw(r.scene, items) == 1 and r.scene.ngeom == r.scene.maxgeom
        with pytest.raises(ValueError):
            r.update_scene(d)
            aa.draw(r.scene, [aa.DrawItem("cone", start, end, aa.GREEN)])
    finally:
        r.close()


def test_viewer_user_scene_is_replaced_each_step(model):
    """`mujoco.viewer` passive handle API: `lock()` + `user_scn`. Each call replaces the overlay."""
    import threading

    class _Handle:
        def __init__(self):
            self.user_scn = mujoco.MjvScene(model, maxgeom=10)
            self._lock = threading.Lock()

        def lock(self):
            return self._lock

    h = _Handle()
    a = aa.DrawItem("arrow", np.zeros(3), np.array([0.0, 0.0, 0.1]), aa.GREEN)
    m_ = aa.DrawItem("marker", np.zeros(3), np.zeros(3), aa.RED, 0.015)
    aa.draw_user_scene(h, [a, m_])
    assert h.user_scn.ngeom == 2
    aa.draw_user_scene(h, [m_])
    assert h.user_scn.ngeom == 1 and h.user_scn.geoms[0].type == mujoco.mjtGeom.mjGEOM_SPHERE
    aa.draw_user_scene(h, [])
    assert h.user_scn.ngeom == 0


def test_only_pixels_change_and_only_where_drawn(model):
    m = model
    d = _teleop_data(m)
    before = _snapshot(d)
    r = mujoco.Renderer(m, height=240, width=320)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.45, 0.0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 1.7, 180.0, -18.0
    try:
        r.update_scene(d, camera=cam)
        plain = r.render().copy()
        r.update_scene(d, camera=cam)
        plain2 = r.render().copy()
        viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",))
        ref = _row16(m, d)
        refined = ref.copy()
        refined[1] += 0.08
        ov = viz.step(d, executed=True, refined_row=refined, reference_row=ref)
        r.update_scene(d, camera=cam)
        aa.draw(r.scene, ov.items)
        drawn = r.render().copy()
    finally:
        r.close()
    assert np.array_equal(plain, plain2)
    changed = np.any(drawn != plain, axis=2)
    assert changed.sum() > 30                      # the arrow is visible
    assert changed.mean() < 0.05                   # and local
    _assert_same(before, _snapshot(d))


# ============================================================================ 4. read only + obstacle normal


def test_step_reads_data_only_and_uses_no_rng(model):
    m = model
    d = _teleop_data(m)
    for _ in range(5):
        mujoco.mj_step(m, d)
    np.random.seed(7)
    random.seed(7)
    np_state, py_state = np.random.get_state(), random.getstate()
    before = _snapshot(d)
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left", "right"))
    ref = _row16(m, d)
    refined = ref.copy()
    refined[[1, 9]] += 0.05
    for executed, kw in ((True, {}), (False, {"applied_row": refined}), (False, {"applied_row": None})):
        viz.step(d, executed=executed, refined_row=refined,
                 reference_row=ref if kw.get("applied_row", 1) is not None else None, **kw)
    _assert_same(before, _snapshot(d))
    s1 = np.random.get_state()
    assert s1[0] == np_state[0] and np.array_equal(s1[1], np_state[1]) and s1[2:] == np_state[2:]
    assert random.getstate() == py_state


def test_obstacle_normal_matches_the_manager_distance():
    from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager

    m = mujoco.MjModel.from_xml_path(str(TRANSPORT_OBSTACLE_XML))
    d = _teleop_data(m)
    mgr = PickPlaceObstacleManager(m, d, copy.deepcopy(HURDLE_CONFIG))
    mgr.activate("t42_arrow_hurdle")
    mujoco.mj_forward(m, d)
    before = _snapshot(d)
    viz = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",),
                             obstacle_geom_ids=sorted(mgr.collision_geom_ids), obstacle_normal=True,
                             normal_range_m=1.0)
    assert viz.gripper_geom_ids["left"], "no gripper collision geoms found"
    n = viz.closest_obstacle_normal(d, "left")
    assert n is not None
    _assert_same(before, _snapshot(d))
    expect = min(mgr.pair_distance(go, gg, 1.0) for go in sorted(mgr.collision_geom_ids)
                 for gg in viz.gripper_geom_ids["left"])
    assert n["distance_m"] == pytest.approx(expect, abs=1e-6)
    assert np.linalg.norm(n["normal"]) == pytest.approx(1.0, abs=1e-12)
    # the witness on the obstacle side is `point - distance * normal`
    fromto = np.zeros(6)
    mujoco.mj_geomDistance(m, d, n["obstacle_geom"], n["gripper_geom"], 1.0, fromto)
    np.testing.assert_allclose(fromto[:3], n["point"] - n["distance_m"] * n["normal"], atol=1e-9)
    # off by default, and out of range -> None
    assert aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",),
                              obstacle_geom_ids=sorted(mgr.collision_geom_ids)
                              ).closest_obstacle_normal(d, "left") is None
    near = aa.AvoidanceArrows(m, action_format="rby1_16d", arms=("left",), obstacle_normal=True,
                              obstacle_geom_ids=sorted(mgr.collision_geom_ids), normal_range_m=1e-4)
    assert near.closest_obstacle_normal(d, "left") is None


# ============================================================================ 5. main(): purely visual


class _FakeSafeServer:
    """`WebsocketClientPolicy` stand-in for `--safe-remote`: chunk 1 execute, 2 unsafe HOLD (with a reference),
    3 comms HOLD (raises), 4 execute. The refined chunk deflects the left arm."""

    DEFLECT = np.array([0.0, 0.03, 0.0, -0.04, 0.0, 0.02, 0.0])

    def __init__(self):
        self.requests = []

    def infer(self, request):
        from AG3S.trajopt import wire

        self.requests.append(copy.deepcopy(request))
        n = len(self.requests)
        if n == 3:
            raise RuntimeError("fake comms failure")
        state = np.asarray(request["state"], dtype=np.float64)
        ref = np.tile(state, (50, 1))
        ref[:, 1] += np.linspace(0.0, 0.02, 50)
        refined = ref.copy()
        refined[:, 0:7] += self.DEFLECT
        return {"actions": refined.astype(np.float32), "seq": int(request[wire.PREFIX + "seq"]),
                "safe": n != 2, "ag3s_status": "ok", "geometry_certified": True,
                "trajopt_status": "feasible", "max_violation_m": 0.0, "timing_ms": {"total": 1.0},
                "notes": [], wire.SHADOW: False, wire.ACTIONS_REFERENCE: ref.astype(np.float32)}


def _load_client(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        _bind_legacy_inputs(mod)
    except ImportError as exc:
        pytest.skip(f"pi05_infer.py dependencies missing in this interpreter: {exc}")
    return mod


def _run_main(client, out: pathlib.Path, cfg: pathlib.Path, extra=()) -> dict:
    server = _FakeSafeServer()
    original = client.load_remote_policy
    client.load_remote_policy = lambda remote: server
    old_argv = sys.argv
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1",
                "--episode-index", str(EPISODE), "--headless", "--max-steps", "32", "--warmup-steps", "20",
                "--view", "front", "--record", str(out / "third_person_front.mp4"),
                "--record-frames", str(out), "--trajectory-out", str(out / "traj.npz"),
                "--obstacle-config", str(cfg), "--obstacle-profile", "t42_arrow_hurdle",
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
    return {"out": out, "requests": server.requests, "stdout": buf.getvalue(),
            "np_rng": np.random.get_state(), "py_rng": random.getstate()}


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


def _frames(run, root: pathlib.Path):
    text = (run["out"] / "frames.jsonl").read_text().replace(str(run["out"]), "<OUT>").replace(str(root.resolve()), "<ROOT>").replace(str(root), "<ROOT>")
    return [_flatten(json.loads(line), f"#{i}") for i, line in enumerate(text.splitlines())]


def _manifest(run, root: pathlib.Path):
    text = (run["out"] / "manifest.json").read_text().replace(str(run["out"]), "<OUT>").replace(str(root.resolve()), "<ROOT>").replace(str(root), "<ROOT>")
    return _flatten(json.loads(text))


def _diff_paths(a: list | dict, b: list | dict) -> set:
    if isinstance(a, dict):
        a, b = [a], [b]
    assert len(a) == len(b)
    paths = set()
    for ra, rb in zip(a, b):
        assert set(ra) == set(rb), set(ra) ^ set(rb)
        for k in ra:
            if ra[k] != rb[k]:
                paths.add(k.split("/", 1)[-1] if k.startswith("#") else k)
    return paths


def _npz(run):
    with np.load(run["out"] / "traj.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _npz_diff(a, b) -> set:
    assert set(a) == set(b)
    return {k for k in a if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])}


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
    """Top-level request keys whose payload differs (arrays: dtype, shape and every bit)."""
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


def _pre_arrow_source() -> bytes:
    """`rby1_bringup/pi05_infer.py` of the newest commit that does not have the flag yet (HEAD until T42-arrow
    is committed, its parent afterwards)."""
    rel = "rby1_bringup/pi05_infer.py"
    try:
        commits = subprocess.run(["git", "-C", str(HYBRID), "log", "--format=%H", "-n", "30", "--", rel],
                                 check=True, capture_output=True, text=True).stdout.split()
        for commit in commits:
            src = subprocess.run(["git", "-C", str(HYBRID), "show", f"{commit}:{rel}"],
                                 check=True, capture_output=True).stdout
            if b"viz-avoidance-arrow" not in src:
                return src
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pass
    pytest.skip("no pre-T42-arrow pi05_infer.py in git")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    try:
        import PIL  # noqa: F401
        import imageio  # noqa: F401
        from AG3S.trajopt import wire  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"missing: {exc}")
    tmp = tmp_path_factory.mktemp("t42_arrow")
    cfg = tmp / "hurdle.json"
    cfg.write_text(json.dumps(HURDLE_CONFIG))
    # HEAD's pi05_infer.py, inside a mirror of pi05_TO_hybrid (REPO_ROOT = parents[1] of the file)
    head_root = tmp / "mirror_of_head"  # not a prefix-sibling of the run dirs (path normalisation)
    (head_root / "rby1_bringup").mkdir(parents=True)
    for entry in HYBRID.iterdir():
        if entry.name not in ("rby1_bringup", ".git"):
            (head_root / entry.name).symlink_to(entry)
    head_src = _pre_arrow_source()
    (head_root / "rby1_bringup" / "pi05_infer_head.py").write_bytes(head_src)
    head = _load_client(head_root / "rby1_bringup" / "pi05_infer_head.py", "_t42_arrow_pi05_infer_head")
    new = _load_client(PI05_INFER, "_t42_arrow_pi05_infer_new")
    return {
        "head_root": head_root,
        "head": _run_main(head, tmp / "head", cfg),
        "off": _run_main(new, tmp / "off", cfg),
        "off2": _run_main(new, tmp / "off2", cfg),
        "on": _run_main(new, tmp / "on", cfg, ("--viz-avoidance-arrow", "--viz-obstacle-normal")),
    }


#: tokens that must never appear in a volatile path (physics / actions / decisions)
_PHYSICAL = ("qpos", "applied_ctrl", "actions", "object_poses", "executed", "gate", "verdict_reasons",
             "q_hold", "gripper", "chunk_shape")


def _volatile(runs):
    vf = _diff_paths(_frames(runs["off"], HYBRID), _frames(runs["off2"], HYBRID))
    vm = _diff_paths(_manifest(runs["off"], HYBRID), _manifest(runs["off2"], HYBRID))
    vn = _npz_diff(_npz(runs["off"]), _npz(runs["off2"]))
    vr = _request_diff(runs["off"]["requests"], runs["off2"]["requests"])
    assert not [p for p in vf if any(tok in p for tok in _PHYSICAL)], vf
    # what two flag-off runs disagree on is wall clock only (2026-10-05: camera / render stamps, `now`,
    # `field.applied_at`, `timing_ms.policy_infer`, manifest `recorded_at_*`, npz `inference_ms`)
    clock = ("stamp", "now", "applied_at", "render_spread", "timing_ms", "recorded_at", "inference_ms")
    assert all(any(t in p for t in clock) for p in vf | vm | vn | vr), (vf, vm, vn, vr)
    assert vn <= {"inference_ms"}, vn
    assert "state" not in vr and "images" not in vr, vr
    assert not any(k.startswith(("ag3s/depth/", "ag3s/robot_state/", "ag3s/T_base_cam/", "ag3s/K/")) for k in vr), vr
    return vf, vm, vn, vr


def test_the_scenario_exercises_execute_hold_and_comms(runs):
    frames = [json.loads(line) for line in (runs["off"]["out"] / "frames.jsonl").read_text().splitlines()]
    control = [f for f in frames if f["kind"] == "control"]
    assert len(control) == 32
    assert [c["executed"] for c in control] == [True] * 8 + [False] * 16 + [True] * 8
    planning = [f for f in frames if f["kind"] == "planning"]
    # frame_record omits `actions_reference` when there is none (comms HOLD)
    assert planning[1].get("actions_reference") is not None and planning[2].get("actions_reference") is None
    assert "[viz] avoidance arrow on" in runs["on"]["stdout"]
    assert "[viz]" not in runs["off"]["stdout"]


def test_flag_off_is_bit_identical_to_head(runs):
    vf, vm, vn, vr = _volatile(runs)
    head, off = runs["head"], runs["off"]
    assert _diff_paths(_frames(head, runs["head_root"]), _frames(off, HYBRID)) <= vf
    assert _diff_paths(_manifest(head, runs["head_root"]), _manifest(off, HYBRID)) <= vm | {"/extra/argv"}
    # argv differs only by the script path (HEAD copy vs working tree)
    assert _npz_diff(_npz(head), _npz(off)) <= vn
    assert _request_diff(head["requests"], off["requests"]) <= vr
    assert head["np_rng"][1].tolist() == off["np_rng"][1].tolist() and head["py_rng"] == off["py_rng"]
    vh, vo = _video(head), _video(off)
    assert len(vh) == len(vo) == 32
    assert all(np.array_equal(a, b) for a, b in zip(vh, vo))


def test_flag_on_changes_pixels_only(runs):
    vf, vm, vn, vr = _volatile(runs)
    on, off = runs["on"], runs["off"]
    assert _diff_paths(_frames(on, HYBRID), _frames(off, HYBRID)) <= vf
    assert _diff_paths(_manifest(on, HYBRID), _manifest(off, HYBRID)) <= vm | {"/extra/argv"}
    argv_on = json.loads((on["out"] / "manifest.json").read_text())["extra"]["argv"]
    argv_off = json.loads((off["out"] / "manifest.json").read_text())["extra"]["argv"]
    assert [a for a in argv_on if not a.startswith("--viz")] == [a.replace("/off", "/on") for a in argv_off]
    a, b = _npz(on), _npz(off)
    assert _npz_diff(a, b) <= vn
    for k in ("executed_actions", "measured_qpos", "predicted_chunks", "obstacle_safety_json"):
        assert np.array_equal(a[k], b[k]), k
    assert _request_diff(on["requests"], off["requests"]) <= vr
    assert len(on["requests"]) == len(off["requests"]) == 4
    assert on["np_rng"][1].tolist() == off["np_rng"][1].tolist() and on["py_rng"] == off["py_rng"]
    v_on, v_off, v_off2 = _video(on), _video(off), _video(runs["off2"])
    assert len(v_on) == len(v_off) == 32
    assert all(np.array_equal(x, y) for x, y in zip(v_off, v_off2))          # the off render is deterministic
    differ = [not np.array_equal(x, y) for x, y in zip(v_on, v_off)]
    assert all(differ)                                                     # every safe-remote step has an overlay
    # below the caption band (rows >= 60) the arrow itself changed pixels on the execute steps
    assert all(np.any(np.abs(x[60:].astype(int) - y[60:].astype(int)) > 40)
               for x, y in zip(v_on[:8], v_off[:8]))
