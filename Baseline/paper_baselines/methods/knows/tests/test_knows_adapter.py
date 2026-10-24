"""PAPER-METHODS C: KNOWS CBF-QP behind the common SafetyFilter interface (``knows/adapter.py``).

    CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/knows/tests -q
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.methods.interface import SafetyFilter, SceneObservation, get_method, validate_filter_result
from Baseline.paper_baselines.methods.knows.adapter import GRIPPER_SEMI_AXES, KnowsSafetyFilter
from Baseline.paper_baselines.methods.knows.cbf.ellipsoid import Ellipsoid, optimal_normal

RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")
EE = np.array([0.45, 0.0, 0.30])


def synth_obs(obstacles):
    T = np.eye(4)
    T[:3, 3] = EE
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=T, gripper=0.08,
                            instruction="pick", extras={"obstacle_ellipsoids": obstacles})


def straight(d, T=10):
    A = np.zeros((T, 7))
    A[:, :3] = d
    A[:, 6] = 1.0
    return A


BALL = [{"name": "ball", "center": EE + [GRIPPER_SEMI_AXES[0] + 0.05 + 0.03, 0, 0], "semi_axes": [0.05] * 3}]


@pytest.mark.parametrize("normals", ["relaxed", "fixed"])
def test_violating_chunk_is_modified(normals):
    f = KnowsSafetyFilter(obstacle_source="ellipsoids", cbf_params={"normals": normals})
    assert isinstance(f, SafetyFilter)
    A = straight([1.0, 0, 0])
    res = f.filter(synth_obs(BALL), A)
    validate_filter_result(res, A)
    assert res.intervened.all() and np.all(res.actions[:, 0] < 1.0)
    assert not any(s["emergency_stop"] for s in res.info["steps"])
    assert np.array_equal(res.actions[:, 6], A[:, 6])
    # the true gap after the first filtered step stays >= (1 - gamma) h  (discrete CBF, Eq. 7) -- only with fixed
    # normals: the paper's relaxed normals (eps box) can satisfy the row by tilting the plane (knows docs/16: true
    # CBF broken on 26.5% of steps), which this synthetic case reproduces
    if normals == "relaxed":
        return
    robot0 = Ellipsoid(EE, np.diag(np.square(GRIPPER_SEMI_AXES)))
    obst = Ellipsoid(BALL[0]["center"], np.eye(3) * 0.05 ** 2)
    h0 = 0.03
    moved = Ellipsoid(EE + res.actions[0, :3] * 0.05, robot0.Q)
    from Baseline.paper_baselines.methods.knows.cbf.ellipsoid import barrier

    h1 = barrier(optimal_normal(moved, obst), moved, obst)
    assert h1 >= (1 - f.params.gamma_h) * h0 - 1e-4


def test_safe_chunk_is_identity_bitwise():
    f = KnowsSafetyFilter(obstacle_source="ellipsoids")
    A = straight([-1.0, 0.2, 0.1])
    res = f.filter(synth_obs(BALL), A)
    assert not res.intervened.any() and np.array_equal(res.actions, A)
    f.reset()
    res = f.filter(synth_obs([]), straight([1.0, 0, 0]))
    assert not res.intervened.any()
    assert all(s["status"] == "no_obstacles" for s in res.info["steps"])


def test_registry_and_provenance():
    f = get_method("knows", obstacle_source="ellipsoids")
    p = f.provenance()
    assert p["cbf_params"]["gamma_h"] == 0.5 and p["units"]["pos_scale"] == 0.05


@pytest.mark.skipif(not (RUN / "episodes").is_dir(), reason="SL3 dev recorded run not available")
def test_recorded_frame_gt_seg_and_centroid_tracking():
    from Baseline.paper_baselines.methods.safety_common import load_recorded_call

    f = KnowsSafetyFilter()
    obs0, A0, _ = load_recorded_call(RUN, "libero_spatial_t0_e0", 0)
    r0 = f.filter(obs0, A0)
    validate_filter_result(r0, A0)
    shapes = {k: v.Q.copy() for k, v in f.shapes.items()}
    obs1, A1, _ = load_recorded_call(RUN, "libero_spatial_t0_e0", 12)
    r1 = f.filter(obs1, A1)
    validate_filter_result(r1, A1)
    assert all(np.array_equal(shapes[k], f.shapes[k].Q) for k in shapes)  # shapes frozen at the first call
    assert r1.info["perception"]["privileged"] and "akita_black_bowl_1" not in r1.info["obstacles"]


# --- CLOSED-LOOP-FIX: track identity (paper §3.2) and per-step diagnostics ------------------------------------------------
def _cloud(c, semi, n=400, seed=0):
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return np.asarray(c, float) + np.asarray(semi, float) * v / np.linalg.norm(v, axis=1, keepdims=True)


def given_obs(clouds: dict, ee=EE):
    T = np.eye(4)
    T[:3, 3] = ee
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=T, gripper=0.08,
                            instruction="pick", extras={"obstacle_points": clouds})


A_C, B_C = np.array([0.70, 0.20, 0.10]), np.array([0.70, -0.25, 0.15])
A_SEMI, B_SEMI = [0.02, 0.02, 0.03], [0.06, 0.05, 0.08]


def test_track_identity_survives_a_name_swap():
    """Detector / depth-component names are re-ranked every call. Tracks follow the nearest centroid (paper §3.2:
    identity kept, shape frozen); matching by name would put A's shape on B's centroid."""
    A0, B0 = _cloud(A_C, A_SEMI), _cloud(B_C, B_SEMI, seed=1)
    A1 = A0 + [0.02, 0.0, 0.0]
    first, second = {"depth#0": A0, "depth#1": B0}, {"depth#0": B0, "depth#1": A1}
    act = straight([0.0, 0.0, 0.0])
    f = KnowsSafetyFilter(obstacle_source="given", robot_filter=False)
    assert f.associate == "centroid"
    f.filter(given_obs(first), act)
    r = f.filter(given_obs(second), act)
    c = {k: np.asarray(v["c"]) for k, v in r.info["obstacles"].items()}
    np.testing.assert_allclose(c["depth#0"] - f.shapes["depth#0"].c, A1.mean(0) - A0.mean(0), atol=1e-4)
    np.testing.assert_allclose(c["depth#1"] - f.shapes["depth#1"].c, 0.0, atol=1e-4)
    assert r.info["perception"]["association"]["matched"] == 2
    fn = KnowsSafetyFilter(obstacle_source="given", robot_filter=False, associate="name")
    fn.filter(given_obs(first), act)
    rn = fn.filter(given_obs(second), act)
    jump = np.linalg.norm(np.asarray(rn.info["obstacles"]["depth#0"]["c"]) - fn.shapes["depth#0"].c)
    assert jump > 0.3  # A's ellipsoid teleported to B (the closed-loop failure the association removes)


def test_unmatched_track_holds_its_last_centroid():
    A0, B0 = _cloud(A_C, A_SEMI), _cloud(B_C, B_SEMI, seed=1)
    act = straight([0.0, 0.0, 0.0])
    f = KnowsSafetyFilter(obstacle_source="given", robot_filter=False)
    f.filter(given_obs({"depth#0": A0, "depth#1": B0}), act)
    f.filter(given_obs({"depth#0": A0 + [0.02, 0, 0], "depth#1": B0}), act)
    r = f.filter(given_obs({"depth#0": B0}), act)  # A occluded
    c = np.asarray(r.info["obstacles"]["depth#0"]["c"])
    np.testing.assert_allclose(c - f.shapes["depth#0"].c, [0.02, 0, 0], atol=1e-4)
    assert r.info["perception"]["association"]["unmatched_tracks"] == 1


@pytest.mark.parametrize("normals", ["relaxed", "fixed"])
def test_diagnostics_are_read_only_and_complete(normals):
    """The diagnostics leave the served chunk and the filter state untouched, and report h(n_stored), h*, the CBF
    condition on h* and the Eq. 12 relaxation for every step."""
    obst = [{"name": "ball", "center": EE + [0.0, 0.10, 0.0], "semi_axes": [0.05] * 3}]
    A = straight([0.0, 1.0, 0.0])
    out = {}
    for diag in (False, True):
        f = KnowsSafetyFilter(obstacle_source="ellipsoids", cbf_params={"normals": normals}, diagnostics=diag)
        r1 = f.filter(synth_obs(obst), A)
        r2 = f.filter(synth_obs(obst), A)
        out[diag] = (r1, r2, f.cbf.normals())
    for i in (0, 1):
        assert np.array_equal(out[False][i].actions, out[True][i].actions)
    assert all(np.array_equal(out[False][2][k], out[True][2][k]) for k in out[False][2])
    steps = out[True][0].info["steps"]
    keys = {"min_h", "h_star", "h_star_next", "h_star_next_nominal", "cbf_ok", "relax_m", "need_m"}
    assert all(keys <= set(s) for s in steps)
    assert all(s["h_star"] >= s["min_h"] - 1e-9 for s in steps)  # h(n) <= max_n h(n) = h*
    if normals == "fixed":
        assert all(s["relax_m"] == 0.0 for s in steps)
