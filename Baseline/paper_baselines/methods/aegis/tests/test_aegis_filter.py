"""PAPER-METHODS C: VLSA / AEGIS safety layer (``methods/aegis``).

Synthetic scene: one spherical obstacle, a straight action chunk. A barrier-violating chunk is modified and keeps
the CBF condition; a safe chunk passes bit-identically. Plus: the official helpers really are the submodule's,
their gradients match finite differences, the QP equals the closed-form half-space projection, and the
perception path runs end to end on a recorded LIBERO frame.

    CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/aegis/tests -q
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.methods.aegis import official
from Baseline.paper_baselines.methods.aegis.adapter import (AegisSafetyFilter, EE_OFFSET_LOCAL, Q1_DEFAULT, ee_ellipsoid,
                                                             solve_cbf_qp)
from Baseline.paper_baselines.methods.aegis.perception import ObstacleEllipsoid, filtering_points, fit_obstacle
from Baseline.paper_baselines.methods.interface import SceneObservation, validate_filter_result
from Baseline.paper_baselines.methods.safety_common import axisangle_to_mat

RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")


def synth_obs(obstacles, ee=None, instruction="pick up the bowl"):
    if ee is None:
        ee = np.eye(4)
        ee[:3, 3] = [0.45, 0.0, 0.30]
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=ee, gripper=0.08,
                            instruction=instruction, extras={"obstacle_ellipsoids": obstacles})


def straight(direction, T=10, grip=-1.0):
    A = np.zeros((T, 7))
    A[:, :3] = direction
    A[:, 6] = grip
    return A


def sphere_ahead(gap_x):
    """Sphere (r = 5 cm) on +x of the EE ellipsoid, ``gap_x`` metres of clearance along x."""
    p1 = np.array([0.45, 0.0, 0.30]) + EE_OFFSET_LOCAL
    return [{"name": "ball", "center": p1 + [Q1_DEFAULT[0] + 0.05 + gap_x, 0, 0], "semi_axes": [0.05] * 3}]


def test_official_helpers_are_the_submodule_code():
    U = official.load_official_utils()
    assert pathlib.Path(inspect.getsourcefile(U.compute_h_coeffs_3d)).resolve() == official.OFFICIAL_UTILS.resolve()
    assert pathlib.Path(inspect.getsourcefile(U.fit_ellipse)).resolve() == official.OFFICIAL_UTILS.resolve()
    prov = official.provenance()
    assert prov["sha"].startswith("2457feed") and len(prov["utils_sha256"]) == 64


def test_official_barrier_gradients_match_finite_differences():
    """a_v / a_w are body-frame velocity coefficients; ``@ R^T`` gives the base-frame ones (used by multilink)."""
    U = official.load_official_utils()
    rng = np.random.default_rng(3)
    for _ in range(5):
        Ri, Rj = axisangle_to_mat(rng.normal(size=3)), axisangle_to_mat(rng.normal(size=3))
        pi, pj = rng.normal(size=3) * 0.05, np.array([0.3, 0.1, -0.05]) + rng.normal(size=3) * 0.05
        qi, qj = np.array([0.06, 0.12, 0.11]), rng.uniform(0.02, 0.08, 3)
        z = pj - pi + rng.normal(size=3) * 0.05
        z /= np.linalg.norm(z)
        a_v, a_w, a_uz, h, mu = U.compute_h_coeffs_3d(pi, qi, Ri, pj, qj, Rj, z)
        e = 1e-6
        E = np.eye(3)
        f = lambda p=pi, R=Ri, zz=z: U.compute_h_ij(p, qi, R, pj, qj, Rj, zz)  # noqa: E731
        gp = np.array([(f(p=pi + e * E[k]) - f(p=pi - e * E[k])) / (2 * e) for k in range(3)])
        gw = np.array([(f(R=axisangle_to_mat(e * E[k]) @ Ri) - f(R=axisangle_to_mat(-e * E[k]) @ Ri)) / (2 * e)
                       for k in range(3)])
        gz = np.array([(f(zz=z + e * E[k]) - f(zz=z - e * E[k])) / (2 * e) for k in range(3)])
        assert np.allclose(np.asarray(a_v) @ Ri.T, gp, atol=1e-6)
        assert np.allclose(np.asarray(a_w) @ Ri.T, gw, atol=1e-6)
        assert np.allclose(np.asarray(a_uz), gz @ (np.eye(3) - np.outer(z, z)), atol=1e-6)


def test_qp_equals_closed_form_projection():
    rng = np.random.default_rng(0)
    w = np.concatenate([np.full(6, 1 / 25), np.ones(3)])
    for _ in range(5):
        u_ref, a = rng.normal(size=9), rng.normal(size=9)
        b = -float(a @ u_ref) - 0.5  # violated by 0.5 at the reference
        u, status = solve_cbf_qp(u_ref, w, a[None], np.array([b]))
        lam = 0.5 / float(a @ (a / w))
        assert status == "optimal"
        assert np.allclose(u, u_ref + lam * a / w, atol=1e-4)


def test_violating_chunk_is_modified_and_keeps_the_barrier():
    f = AegisSafetyFilter(obstacle_source="ellipsoids")
    A = straight([1.0, 0, 0])
    res = f.filter(synth_obs(sphere_ahead(0.05)), A)
    validate_filter_result(res, A)
    assert res.intervened.all()
    assert np.all(res.actions[:, 0] < 1.0) and np.all(res.actions[:, 0] > 0.0)  # slowed, still advancing
    assert np.array_equal(res.actions[:, 6], A[:, 6])  # gripper untouched
    h = np.array([s["h"][0] for s in res.info["steps"]])
    assert np.all(h > 0) and np.all(np.diff(h) < 0)
    # CBF condition with the action as velocity (AEGIS units): dh/dt >= -10 h; here dh/dx = -1 so a_x <= 10 h
    assert np.all(res.actions[:, 0] <= 10.0 * h + 1e-3)


def test_safe_chunk_is_identity_bitwise():
    f = AegisSafetyFilter(obstacle_source="ellipsoids")
    for d in ([-1.0, 0, 0], [0, 0.5, 0], [0, 0, 1.0]):
        f.reset()
        A = straight(d)
        A[:, 3:6] = 0.01
        res = f.filter(synth_obs(sphere_ahead(0.05)), A)
        assert not res.intervened.any() and np.array_equal(res.actions, A)
    f.reset()
    far = straight([1.0, 0, 0])
    res = f.filter(synth_obs(sphere_ahead(0.5)), far)  # far obstacle: CBF inactive
    assert not res.intervened.any() and np.array_equal(res.actions, far)


def test_no_obstacle_is_identity():
    f = AegisSafetyFilter(obstacle_source="ellipsoids")
    A = straight([1.0, 0, 0])
    res = f.filter(synth_obs([]), A)
    assert res.info["flag_safety_control"] is False
    assert np.array_equal(res.actions, A) and not res.intervened.any()


def test_obstacles_frozen_until_reset():
    f = AegisSafetyFilter(obstacle_source="ellipsoids")
    f.filter(synth_obs(sphere_ahead(0.05)), straight([1.0, 0, 0], T=1))
    res = f.filter(synth_obs([]), straight([1.0, 0, 0], T=1))  # official: obstacle fitted once per episode
    assert res.info["flag_safety_control"] and res.intervened.all()
    f.reset()
    assert not f.filter(synth_obs([]), straight([1.0, 0, 0], T=1)).intervened.any()


def test_tall_object_q1():
    assert np.array_equal(ee_ellipsoid(np.eye(4), "pick up the milk")[2], [0.06, 0.12, 0.2])
    assert np.array_equal(ee_ellipsoid(np.eye(4), "pick up the bowl")[2], Q1_DEFAULT)


def test_filtering_and_fit_on_synthetic_box():
    rng = np.random.default_rng(0)
    T_world_base = np.eye(4)
    T_world_base[:3, 3] = [-0.66, 0.0, 0.912]
    box = rng.uniform([-0.03, -0.02, 0.0], [0.03, 0.02, 0.06], size=(800, 3)) + [0.6, 0.0, 0.02]
    table = np.c_[rng.uniform(0.3, 0.9, 400), rng.uniform(-0.3, 0.3, 400), np.full(400, -0.011)]
    pts, info = filtering_points(np.r_[box, table], "libero_spatial", T_world_base)
    assert info["workspace"] == "table" and info["n_workspace"] <= len(box)
    E = fit_obstacle("box", pts)
    assert isinstance(E, ObstacleEllipsoid)
    assert np.linalg.norm(E.center - [0.6, 0.0, 0.05]) < 0.02
    assert 0.02 < E.semi_axes.max() < 0.08


@pytest.mark.skipif(not (RUN / "episodes").is_dir(), reason="SL3 dev recorded run not available")
def test_recorded_frame_end_to_end_gt_seg():
    from Baseline.paper_baselines.methods.safety_common import load_recorded_call

    obs, A, _ = load_recorded_call(RUN, "libero_spatial_t0_e0", 12)
    f = AegisSafetyFilter()
    res = f.filter(obs, A)
    validate_filter_result(res, A)
    assert res.info["perception"]["privileged"] and len(res.info["obstacles"]) >= 3
    assert res.intervened.any()  # the approach between plate and ramekin is slowed
    assert np.array_equal(res.actions[:, 6], A[:, 6])
