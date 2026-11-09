"""F1 (2026-11-09, user-approved fidelity fix): the KNOWS EE ellipsoid encloses the Panda hand + fingers.

Paper Sec. 3.1: the EE ellipsoid is "calibrated offline"; Sec. 5: only links upstream of the wrist are unprotected.
The port's original (0.04, 0.04, 0.07) m grip-site ellipsoid enclosed none of the hand / finger collision-mesh vertices
(audit outputs/2026-11-09_KNOWS_port_점검). ``ee_model="hand_mvee"`` (default) = their MVEE, centre behind the grip
site, moved by wrist rotation through the RobotBody lever arm; ``ee_model="grip_site_v0"`` = the original.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.interface import SceneObservation, validate_filter_result
from Baseline.paper_baselines.methods.knows.adapter import GRIP_SITE_TO_EE, HAND_MVEE_CENTER, KnowsSafetyFilter

M = np.asarray(GRIP_SITE_TO_EE)  # grip-site vector -> obs.ee_pose (robot0_right_hand) frame (frame fix 2026-11-10)

EE = np.array([0.45, 0.0, 0.30])
FIXED = {"normals": "fixed"}


def obs(obstacles):
    T = np.eye(4)
    T[:3, :3] = M.T  # ee_pose = robot0_right_hand body, so that the grip-site frame (R_hand @ M) = world frame
    T[:3, 3] = EE
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=T, gripper=0.08,
                            instruction="pick", extras={"obstacle_ellipsoids": obstacles})


def ball(offset, r=0.03):
    return [{"name": "ball", "center": EE + np.asarray(offset, float), "semi_axes": [r] * 3}]


def one(a):
    A = np.zeros((1, 7))
    A[0, :6] = a
    A[0, 6] = 1.0
    return A


def inside(f, p):
    d = M @ np.asarray(p, float) - f.ee_offset  # p in the grip-site frame
    return float(d @ np.linalg.solve(f.Q_local, d)) <= 1.0


def test_default_model_and_provenance():
    f = KnowsSafetyFilter(obstacle_source="ellipsoids")
    rob = f.provenance()["robot"]
    assert rob["ee_model"] == "hand_mvee"
    assert np.allclose(rob["center_offset_ee_frame_m"], M @ np.asarray(HAND_MVEE_CENTER))
    assert np.allclose(sorted(rob["semi_axes"]), [0.0332, 0.0884, 0.1287], atol=1e-4)
    v0 = KnowsSafetyFilter(obstacle_source="ellipsoids", ee_model="grip_site_v0").provenance()["robot"]
    assert v0["ee_model"] == "grip_site_v0" and np.allclose(sorted(v0["semi_axes"]), [0.04, 0.04, 0.07])
    assert KnowsSafetyFilter(obstacle_source="ellipsoids", gripper_semi_axes=(0.05, 0.05, 0.05)).ee_model == \
        "custom_grip_site"


def test_hand_and_finger_points_are_enclosed_only_by_the_new_model():
    # grip-site frame: hand body 3-12 cm behind the grip site (-z), +-10 cm along x; open fingers +-4-7 cm in x
    pts = [(0.09, 0.0, -0.07), (-0.09, 0.0, -0.07), (0.0, 0.03, -0.07), (0.0, 0.0, -0.12), (0.05, 0.0, 0.005)]
    new = KnowsSafetyFilter(obstacle_source="ellipsoids")
    old = KnowsSafetyFilter(obstacle_source="ellipsoids", ee_model="grip_site_v0")
    assert all(inside(new, p) for p in pts)
    assert not any(inside(old, p) for p in pts[:4])


def test_obstacle_beside_the_hand_is_guarded():
    # 3 cm ball ~1.5 cm beyond the hand's +x side at hand height; one 2 cm step toward it
    B, A = ball([0.166, 0.0, -0.07]), one([0.4, 0, 0, 0, 0, 0])
    new = KnowsSafetyFilter(obstacle_source="ellipsoids", cbf_params=FIXED).filter(obs(B), A)
    old = KnowsSafetyFilter(obstacle_source="ellipsoids", cbf_params=FIXED, ee_model="grip_site_v0").filter(obs(B), A)
    validate_filter_result(new, A)
    assert new.intervened[0] and new.actions[0, 0] < 0.4
    assert not old.intervened[0]
    # default (paper) relaxed normals also hold the hand back
    rel = KnowsSafetyFilter(obstacle_source="ellipsoids").filter(obs(B), A)
    assert rel.intervened[0]


def test_wrist_rotation_moves_the_offset_hand():
    # pure rotation about y swings the hand (7 cm behind the grip site) toward -x: w x r = (-0.07 wy, 0, ...)
    B, A = ball([-0.166, 0.0, -0.07]), one([0, 0, 0, 0, 0.3, 0])
    new = KnowsSafetyFilter(obstacle_source="ellipsoids", cbf_params=FIXED).filter(obs(B), A)
    old = KnowsSafetyFilter(obstacle_source="ellipsoids", cbf_params=FIXED, ee_model="grip_site_v0").filter(obs(B), A)
    assert new.intervened[0] and not old.intervened[0]
    st = new.info["steps"][0]
    assert st["h_star_next"] >= (1 - 0.5) * st["h_star"] - 1e-4  # Eq. 7 on the true barrier, lever arm included


def test_frame_fix_matches_the_robosuite_gripper_and_v1_reproduces_the_bug():
    # panda_gripper.xml:14  <body name="right_gripper" quat="0.707107 0 0 -0.707107">  (w, x, y, z) relative to right_hand
    w, x, y, z = 0.707107, 0.0, 0.0, -0.707107
    R_hand_site = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(R_hand_site, M, atol=1e-5)
    pts = [(0.09, 0.0, -0.07), (-0.09, 0.0, -0.07), (0.0, 0.0, -0.12)]
    v1 = KnowsSafetyFilter(obstacle_source="ellipsoids", ee_model="hand_mvee_v1")
    assert not all(inside(v1, p) for p in pts)  # the 2026-11-09 runs: hand tips outside (axes 90 deg off)
