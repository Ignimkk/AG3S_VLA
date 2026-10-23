"""PAPER-METHODS C: multi-link safety filter [1] (``methods/multilink_cbf``).

    CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/multilink_cbf/tests -q
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.methods.aegis import official
from Baseline.paper_baselines.methods.aegis.adapter import AegisSafetyFilter
from Baseline.paper_baselines.methods.aegis.perception import ObstacleEllipsoid
from Baseline.paper_baselines.methods.interface import SceneObservation, validate_filter_result
from Baseline.paper_baselines.methods.multilink_cbf.adapter import MultiLinkCBFFilter, barrier_rows_world
from Baseline.paper_baselines.methods.multilink_cbf.robot import PandaGuardedBodies, damped_pinv

RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")
Q0 = np.array([-1.38e-04, -1.767e-01, -3.656e-02, -2.457, 5.935e-03, 2.235, 7.99e-01])  # SL3 t0 call 0


@pytest.fixture(scope="module")
def pg():
    p = PandaGuardedBodies()
    p.set_fingers(gripper_width=0.04)
    return p


def synth_obs(pg, obstacles, q=Q0):
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=q, ee_pose=pg.ee_pose(q), gripper=0.04,
                            instruction="pick", extras={"obstacle_ellipsoids": obstacles})


def straight(d, T=10):
    A = np.zeros((T, 7))
    A[:, :3] = d
    A[:, 6] = -1.0
    return A


def test_five_guarded_bodies_cover_their_spheres(pg):
    rng = np.random.default_rng(0)
    for q in [Q0, Q0 + rng.normal(size=7) * 0.3]:
        bodies = pg.bodies(q)
        assert [b.name for b in bodies] == ["gripper", "wrist", "forearm_0", "forearm_1", "forearm_2"]
        for b in bodies:
            assert np.allclose(b.R.T @ b.R, np.eye(3), atol=1e-9) and np.all(b.semi_axes > 0.01)
        # every sphere centre of the guarded links is inside the union of the ellipsoids
        for g, (links, _, _) in pg.groups.items():
            for link in links:
                c, r = pg.model.local_spheres(link)
                T = pg.pose(q, link)
                for ci in c:
                    p = T[:3, :3] @ ci + T[:3, 3]
                    inside = [np.sum((((p - b.center) @ b.R) / b.semi_axes) ** 2) <= 1.0 + 1e-9
                              for b in bodies if b.name.startswith(g)]
                    assert any(inside), (link, ci)


def test_jacobians_match_fk(pg):
    rng = np.random.default_rng(1)
    dq = rng.normal(size=7) * 1e-4
    J = pg.ee_jacobian(Q0)
    T0, T1 = pg.ee_pose(Q0), pg.ee_pose(Q0 + dq)
    assert np.linalg.norm(T1[:3, 3] - T0[:3, 3] - J[:3] @ dq) < 1e-7  # second order in |dq| = 1e-4
    b0, b1 = pg.bodies(Q0)[2], pg.bodies(Q0 + dq, refit=False)[2]
    assert np.linalg.norm(b1.center - b0.center - b0.J[:3] @ dq) < 1e-7


def test_constraint_row_predicts_barrier_change(pg):
    """a_u G u dt  ==  h(q + J_ee^+ u dt) - h(q)  for every guarded body (frames + G_k = J_k J_ee^+ consistent)."""
    U = official.load_official_utils()
    rng = np.random.default_rng(2)
    E = ObstacleEllipsoid("o", np.array([0.55, 0.05, 0.45]), np.eye(3), np.array([0.04, 0.05, 0.06]))
    J_pinv = damped_pinv(pg.ee_jacobian(Q0), 0.0)
    u = rng.normal(size=6) * np.r_[0.1, 0.1, 0.1, 0.3, 0.3, 0.3]
    dt = 1e-5
    B0 = pg.bodies(Q0)
    B1 = pg.bodies(Q0 + J_pinv @ u * dt, refit=False)
    for b0, b1 in zip(B0, B1):
        z = E.center - b0.center
        z /= np.linalg.norm(z)
        a_u, _, h0, _ = barrier_rows_world(U, b0.center, b0.semi_axes, b0.R, E, z)
        _, _, h1, _ = barrier_rows_world(U, b1.center, b1.semi_axes, b1.R, E, z)
        pred = a_u @ (b0.J @ J_pinv) @ u * dt
        assert abs((h1 - h0) - pred) < 1e-3 * dt + 1e-9, b0.name


def test_violating_chunk_is_modified_safe_chunk_is_identity(pg):
    f = MultiLinkCBFFilter(obstacle_source="ellipsoids")
    gc = pg.bodies(Q0)[0].center
    obst = [{"name": "box", "center": gc + [0.2, 0, 0], "semi_axes": [0.04] * 3}]
    A = straight([1.0, 0, 0])
    res = f.filter(synth_obs(pg, obst), A)
    validate_filter_result(res, A)
    assert res.intervened.any() and np.all(res.actions[:, 0] <= 1.0 + 1e-9)
    assert min(s["min_h"] for s in res.info["steps"]) > 0
    assert np.array_equal(res.actions[:, 6], A[:, 6])
    f.reset()
    B = straight([-1.0, 0, 0])
    res = f.filter(synth_obs(pg, obst), B)
    assert not res.intervened.any() and np.array_equal(res.actions, B)


def test_forearm_hazard_needs_multiple_links(pg):
    """A hazard just above the forearm: the EE-only AEGIS layer does not react, the multi-link filter does."""
    fc = {b.name: b for b in pg.bodies(Q0)}["forearm_1"].center
    obst = [{"name": "lamp", "center": fc + [0, 0, 0.12], "semi_axes": [0.03] * 3}]
    A = straight([0, 0, 1.0])
    ml = MultiLinkCBFFilter(obstacle_source="ellipsoids").filter(synth_obs(pg, obst), A)
    ee = AegisSafetyFilter(obstacle_source="ellipsoids").filter(synth_obs(pg, obst), A)
    assert ml.intervened.any() and not ee.intervened.any()
    assert ml.info["steps"][0]["min_h_pair"].startswith("forearm")


@pytest.mark.skipif(not (RUN / "episodes").is_dir(), reason="SL3 dev recorded run not available")
def test_recorded_frame_end_to_end():
    from Baseline.paper_baselines.methods.safety_common import load_recorded_call

    obs, A, _ = load_recorded_call(RUN, "libero_spatial_t0_e0", 12)
    f = MultiLinkCBFFilter()
    res = f.filter(obs, A)
    validate_filter_result(res, A)
    assert len(res.info["bodies"]) == 5 and len(res.info["obstacles"]) >= 3
    assert set(res.info["steps"][0]["h_by_body"]) == {"gripper", "wrist", "forearm_0", "forearm_1", "forearm_2"}
