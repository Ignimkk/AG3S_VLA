"""AG3S-GEO-GRASP S1 — gripper geometry, normals, candidates, IK, task state, memory, planner (CPU, synthetic)."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from AG3S.ag3s.grasp.candidates import (GeoGraspConfig, generate_candidates, grasp_frame, grasp_rot_angle,
                                        nearest_symmetric, pose_checks, target_shell)
from AG3S.ag3s.grasp.gripper import RBY1_LEFT, measure_from_mjcf
from AG3S.ag3s.grasp.memory import TargetPointMemory
from AG3S.ag3s.grasp.surface import estimate_surface, voxel_downsample
from AG3S.ag3s.grasp.task_state import GraspTaskState

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJCF = os.path.join(ROOT, "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
TABLE_Z = 0.815


def sphere_cloud(c=(0.5, 0.2, 0.85), r=0.034, cam=((0.0, 0.2, 1.3), (0.5, 0.65, 1.1)), n=6000, noise=0.0005, seed=0):
    """A sphere seen from the cameras `cam` (the union of their front hemispheres — the accumulated observation of a
    head and a wrist view), above the table. One camera: pass `cam=((x, y, z),)`."""
    rng = np.random.default_rng(seed)
    c = np.asarray(c, float)
    cams = np.asarray(cam, float).reshape(-1, 3)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    vis = np.zeros(n, bool)
    for k in cams:
        v = (c - k) / np.linalg.norm(c - k)
        vis |= d @ (-v) > 0.05
    P = c + r * d[vis] + rng.normal(scale=noise, size=(int(vis.sum()), 3))
    return P[P[:, 2] > TABLE_Z + 0.002]


def table(c=(0.5, 0.2), half=0.2, step=0.005):
    xs, ys = np.meshgrid(np.arange(c[0] - half, c[0] + half, step), np.arange(c[1] - half, c[1] + half, step))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, TABLE_Z)])


# --- gripper -----------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(MJCF), reason="MJCF not available")
def test_gripper_spec_matches_mjcf():
    pytest.importorskip("mujoco")
    meas = measure_from_mjcf(MJCF, "left")
    lo, hi = meas["pad"]
    # pad face extent (y, z) of the spec within 1 mm of the mesh, finger thickness covers the pad part
    assert abs(RBY1_LEFT.pad_half_y - hi[1]) < 1e-3 and abs(-RBY1_LEFT.pad_half_y - lo[1]) < 1e-3
    assert abs(RBY1_LEFT.pad_z[0] - lo[2]) < 1e-3 and abs(RBY1_LEFT.pad_z[1] - hi[2]) < 1e-3
    assert RBY1_LEFT.finger_thickness >= hi[0] - 0.0 - 1e-4 or RBY1_LEFT.finger_thickness >= -lo[0] - 1e-4
    plo, phi = meas["palm"]
    assert np.all(np.asarray(RBY1_LEFT.palm.lo) <= plo + 1e-4) and np.all(np.asarray(RBY1_LEFT.palm.hi) >= phi - 1e-4)


def test_finger_boxes_follow_opening():
    f1, f2 = RBY1_LEFT.finger_boxes(0.06)
    assert f1.hi[0] == pytest.approx(-0.03) and f2.lo[0] == pytest.approx(0.03)
    assert f1.contains(np.array([[-0.035, 0.0, -0.1]]))[0] and not f1.contains(np.array([[0.0, 0.0, -0.1]]))[0]


# --- surface -----------------------------------------------------------------------------------------------------
def test_normals_point_outward_on_a_sphere():
    P = voxel_downsample(sphere_cloud(), 0.004)
    s = estimate_surface(P, radius=0.012)
    ok = np.linalg.norm(s.normals, axis=1) > 0.5
    radial = (s.points - np.array([0.5, 0.2, 0.85]))
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    cosang = np.einsum("ij,ij->i", s.normals[ok], radial[ok])
    assert s.centre_method == "sphere" and abs(s.radius - 0.034) < 0.002
    assert np.median(cosang) > 0.99 and np.mean(cosang > 0.9) > 0.9


# --- candidates --------------------------------------------------------------------------------------------------
def test_grasp_frame_puts_pad_centre_on_contact_midpoint():
    c, u, a = np.array([0.5, 0.2, 0.85]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    T = grasp_frame(c, u, a, 0.104)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12) and np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(T[:3, 0], u) and np.allclose(-T[:3, 2], a)
    assert np.allclose(T[:3, 3] + T[:3, :3] @ np.array([0, 0, -0.104]), c)


def test_candidates_are_antipodal_observed_and_collision_free():
    cfg = GeoGraspConfig(time_budget_ms=0.0)   # no wall-clock budget: a loaded pod must not empty the set
    P = sphere_cloud()
    cs = generate_candidates(P, table(), RBY1_LEFT, cfg)
    assert 0 < len(cs.candidates) <= cfg.max_candidates
    obs = voxel_downsample(P, cfg.voxel)
    # one camera only: contacts can only be where that camera saw (fewer or no pairs; never the hidden back)
    one = sphere_cloud(cam=((0.0, 0.2, 1.3),))
    cs1 = generate_candidates(one, table(), RBY1_LEFT, cfg)
    seen = voxel_downsample(one, cfg.voxel)
    for g in cs1.candidates:
        assert min(np.linalg.norm(seen - g.contacts[0], axis=1)) < 1e-9 and min(np.linalg.norm(seen - g.contacts[1], axis=1)) < 1e-9
    for g in cs.candidates:
        u = (g.contacts[1] - g.contacts[0]) / np.linalg.norm(g.contacts[1] - g.contacts[0])
        assert np.allclose(g.T[:3, 0], u, atol=1e-9) or np.allclose(g.T[:3, 0], -u, atol=1e-9)
        # both contacts are observed (downsampled) points — never the hidden back
        assert min(np.linalg.norm(obs - g.contacts[0], axis=1)) < 1e-9 and min(np.linalg.norm(obs - g.contacts[1], axis=1)) < 1e-9
        # observed normals in the generation cone
        assert -g.normals[0] @ u >= cfg.generation_cone_cos - 1e-9 and g.normals[1] @ u >= cfg.generation_cone_cos - 1e-9
        assert g.open_width <= RBY1_LEFT.max_opening + 1e-12 and g.open_width == pytest.approx(min(g.width + cfg.opening_clearance, RBY1_LEFT.max_opening))
        pad_centre = g.T[:3, 3] + g.T[:3, :3] @ np.array([0, 0, -g.depth])
        assert np.allclose(pad_centre, g.contacts.mean(axis=0), atol=1e-9)
        ok, checks, why = pose_checks(g.T, g.T_pre, RBY1_LEFT.body_boxes(g.open_width), voxel_downsample(table(), 0.005),
                                      target_shell(cs.surface, cfg), cfg)
        assert ok, (checks, why)
        assert g.T[2, 3] > TABLE_Z           # nothing below the table


def test_too_wide_object_has_no_candidate_and_says_why():
    P = sphere_cloud(r=0.05)          # 100 mm > 71.5 mm opening
    cs = generate_candidates(P, table(), RBY1_LEFT, GeoGraspConfig())
    assert cs.candidates == [] and cs.stats["reason"] == "no antipodal pair"


def test_obstacle_blocks_the_side_it_occupies():
    cfg = GeoGraspConfig(time_budget_ms=0.0)   # full counts, no wall-clock cut
    P = sphere_cloud()
    wall = np.array([[0.5 + 0.06, y, z] for y in np.arange(0.1, 0.3, 0.005) for z in np.arange(0.82, 1.05, 0.005)])
    free = generate_candidates(P, table(), RBY1_LEFT, cfg)
    blocked = generate_candidates(P, np.vstack([table(), wall]), RBY1_LEFT, cfg)
    for g in blocked.candidates:
        boxes = RBY1_LEFT.body_boxes(g.open_width)
        ok, _, _ = pose_checks(g.T, g.T_pre, boxes, wall, np.zeros((0, 3)), cfg)
        assert ok
    assert blocked.stats["rejected"]["scene_collision"] > free.stats["rejected"]["scene_collision"]


def test_symmetric_twin_and_vla_order():
    R = grasp_frame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), 0.1)[:3, :3]
    twin = R @ np.diag([-1.0, -1.0, 1.0])
    assert grasp_rot_angle(R, twin) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(nearest_symmetric(twin, R), R)
    cfg = GeoGraspConfig()
    P = sphere_cloud()
    cs = generate_candidates(P, table(), RBY1_LEFT, cfg)
    vla = cs.candidates[-1].T.copy()
    cs2 = generate_candidates(P, table(), RBY1_LEFT, cfg, vla_pose=vla, selection="vla")
    assert cs2.candidates[0].key == pytest.approx(min(g.key for g in cs2.candidates))
    assert cs2.candidates[0].vla_pos_err < 0.01


# --- IK ----------------------------------------------------------------------------------------------------------
def test_ik_round_trip_and_rotation_jacobian():
    pytest.importorskip("casadi")
    from AG3S.ag3s.grasp.ik import LinkPoseIK, so3_log, so3_right_jacobian_inv
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    m = UrdfSphereChain(parse_urdf(os.path.join(ROOT, RBY1_URDF) if not os.path.isabs(RBY1_URDF) else RBY1_URDF))
    L = [i for i, n in enumerate(m.joint_names) if n.startswith("left_arm_")]
    ik = LinkPoseIK(m, "ee_left", L)
    lo, hi = m.joint_limits()
    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(10):
        q = np.zeros(m.nq)
        q[L] = rng.uniform(lo[L] * 0.6, hi[L] * 0.6)
        T = m.link_pose(q, "ee_left")
        qs, info = ik.solve(T, np.zeros(m.nq), q[L] + rng.normal(scale=0.2, size=7))
        if qs is not None:
            q2 = np.zeros(m.nq)
            q2[L] = qs
            T2 = m.link_pose(q2, "ee_left")
            assert np.linalg.norm(T2[:3, 3] - T[:3, 3]) <= ik.pos_tol + 1e-9
            assert math.degrees(grasp_rot_angle(T2[:3, :3], T[:3, :3])) <= math.degrees(ik.rot_tol) + 1e-6 or \
                np.linalg.norm(so3_log(T[:3, :3].T @ T2[:3, :3])) <= ik.rot_tol + 1e-9
            ok += 1
    assert ok >= 8
    q = np.zeros(m.nq)
    q[L] = rng.uniform(lo[L] * 0.5, hi[L] * 0.5)
    Rg = m.link_pose(q, "ee_left")[:3, :3] @ grasp_frame(np.zeros(3), np.array([1.0, 0.2, 0]), np.array([0, 0.3, -1.0]), 0)[:3, :3]
    T, Jp, Jw = ik.pose_and_jacobians(q)
    e = so3_log(Rg.T @ T[:3, :3])
    J = so3_right_jacobian_inv(e) @ T[:3, :3].T @ Jw
    Jn = np.zeros((3, 7))
    for c, i in enumerate(L):
        qq = q.copy()
        qq[i] += 1e-6
        Jn[:, c] = (so3_log(Rg.T @ m.link_pose(qq, "ee_left")[:3, :3]) - e) / 1e-6
    assert np.abs(J - Jn).max() < 1e-5


# --- task state --------------------------------------------------------------------------------------------------
def _cs(P=None, scene=None, **kw):
    return generate_candidates(sphere_cloud() if P is None else P, table() if scene is None else scene, RBY1_LEFT,
                               GeoGraspConfig(), **kw)


def test_state_keeps_grasp_follows_centre_and_invalidates():
    st = GraspTaskState(RBY1_LEFT)
    r = st.update(phase="latched", target_id=3, target_state="visible", candidates=_cs())
    assert r["valid"] and r["event"] == "selected"
    gid = st.selected.grasp_id
    r = st.update(phase="latched", target_id=3, target_state="visible", candidates=_cs())
    assert r["valid"] and st.selected.grasp_id == gid and r["n_switches"] == 0
    # target moved 1 cm: same grasp id, pose moved with it
    moved = _cs(P=sphere_cloud(c=(0.51, 0.2, 0.85)), scene=table())
    p_before = st.selected.T[:3, 3].copy()
    r = st.update(phase="latched", target_id=3, target_state="visible", candidates=moved)
    assert st.selected.grasp_id == gid and np.linalg.norm(st.selected.T[:3, 3] - p_before - [0.01, 0, 0]) < 0.006
    # identity change
    r = st.update(phase="latched", target_id=4, target_state="visible", candidates=moved)
    assert "target changed" in r.get("invalidated", []) and st.selected.grasp_id != gid
    # lost
    r = st.update(phase="latched", target_id=4, target_state="lost", candidates=None)
    assert not r["valid"] and r["reason"] == "target lost" and st.selected is None


def test_state_no_switch_while_closing_and_one_reselect():
    st = GraspTaskState(RBY1_LEFT)
    st.update(phase="latched", target_id=1, target_state="visible", candidates=_cs())
    gid = st.selected.grasp_id
    r = st.update(phase="closing", target_id=1, target_state="visible", candidates=_cs())
    assert st.selected.grasp_id == gid and r["event"] == "frozen while closing"
    # while closing the grasp is frozen: a pushed / occluded target (centre jump) does not drop it
    far = _cs(P=sphere_cloud(c=(0.56, 0.2, 0.85)), scene=table(c=(0.56, 0.2)))
    r = st.update(phase="closing", target_id=1, target_state="visible", candidates=far)
    assert r["valid"] and st.selected.grasp_id == gid
    # an identity change drops it: one re-selection while closing, then lost
    r = st.update(phase="closing", target_id=2, target_state="visible", candidates=far)
    assert r["valid"] and r.get("closing_reselect") == 1 and st.selected.grasp_id != gid
    far2 = _cs(P=sphere_cloud(c=(0.62, 0.2, 0.85)), scene=table(c=(0.62, 0.2)))
    r = st.update(phase="closing", target_id=3, target_state="visible", candidates=far2)
    assert not r["valid"] and "re-selection used" in r["reason"]


def test_state_held_and_placed():
    st = GraspTaskState(RBY1_LEFT)
    st.update(phase="latched", target_id=1, target_state="visible", candidates=_cs())
    r = st.update(phase="held", target_id=1, target_state="occluded", candidates=None)
    assert st.carry is not None and st.selected is None and r["event"] == "held"
    st.update(phase="placed", target_id=1, target_state="visible", candidates=None)
    assert st.carry is None


# --- memory / planner --------------------------------------------------------------------------------------------
def test_memory_accumulates_and_resets():
    m = TargetPointMemory()
    a = sphere_cloud(cam=((0.0, 0.2, 1.3),), seed=1)
    b = sphere_cloud(cam=((0.5, 0.6, 1.2),), seed=2)
    m.update(1, a[::20], a.mean(axis=0))
    n1 = len(m.points)
    r = m.update(1, b[::20], b.mean(axis=0))
    assert len(m.points) > n1 and r["cleared"] is None          # a new viewpoint is not a moved object
    r = m.update(2, a[::20], a.mean(axis=0))
    assert r["cleared"] == "target changed"
    m.update(2, a[::5], a.mean(axis=0))
    pushed = a[::5] + np.array([0.0, 0.02, 0.0])
    r = m.update(2, pushed, pushed.mean(axis=0))
    assert r["cleared"] == "surface moved"


def test_planner_rechecks_between_refreshes():
    from AG3S.ag3s.grasp.planner import GeoGraspPlanner

    pl = GeoGraspPlanner(RBY1_LEFT, GeoGraspConfig(refresh_every=3))
    P, S = sphere_cloud(), table()
    gens = []
    for _ in range(5):
        r = pl.step(phase="latched", target_id=1, target_state="visible", target_points=P, target_centroid=P.mean(axis=0),
                    scene_points=S, vla_pose=None, ik_seed=None)
        gens.append(r["generation"])
    assert gens == ["full", "recheck", "recheck", "full", "recheck"]
    assert pl.selected is not None and pl.state.n_switches == 0


def test_time_budget_stops_generation_and_says_so():
    import time as _t
    cfg = GeoGraspConfig(time_budget_ms=1.0)
    slow_ik = lambda T, seed: (_t.sleep(0.005), (None, {"converged": False}))[1]  # noqa: E731
    cs = generate_candidates(sphere_cloud(), table(), RBY1_LEFT, cfg, ik_fn=slow_ik)
    assert cs.stats["budget_hit"] is not None and cs.stats["compute_ms"] < 200
    assert cs.candidates == [] and "time budget" in cs.stats["reason"]


def test_close_pose_check_accepts_a_valid_pose_and_rejects_bad_ones():
    from AG3S.ag3s.grasp.candidates import close_pose_check

    cfg = GeoGraspConfig(time_budget_ms=0.0)
    cs = generate_candidates(sphere_cloud(), table(), RBY1_LEFT, cfg)
    g = cs.candidates[0]
    ok, rec = close_pose_check(g.T, RBY1_LEFT, cfg, cs.surface, cs.scene, cs.shell)
    assert ok is True and rec["reason"] == "valid", rec
    # 40 mm sideways across the pads (beyond the 34 mm radius): only the rim is under the pads, normals far off the axis
    # (25 mm still gives a valid ~30 deg pair inside the mu = 0.8 cone)
    T = g.T.copy(); T[:3, 3] += 0.040 * T[:3, 1]
    ok, rec = close_pose_check(T, RBY1_LEFT, cfg, cs.surface, cs.scene, cs.shell)
    assert ok is False, rec
    # no observation: nothing to judge with
    ok, rec = close_pose_check(g.T, RBY1_LEFT, cfg, None, None, None)
    assert ok is None and "no observed" in rec["reason"]
    # one-sided view along the closing axis: the far pad has no observed surface under it
    one = sphere_cloud(cam=((0.5, 0.2, 1.3),))      # seen from straight above only
    cs1 = generate_candidates(one, table(), RBY1_LEFT, cfg, recheck_only=True)
    Tz = grasp_frame(cs1.surface.centre, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.104)  # closing vertically
    ok, rec = close_pose_check(Tz, RBY1_LEFT, cfg, cs1.surface, cs1.scene, cs1.shell)
    assert ok is False
