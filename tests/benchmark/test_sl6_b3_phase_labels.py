"""SL6 B3 — Panda physical phase labels (`sl6_phase_labels`) on small synthetic tracks, and the
`phase_export --finger-contacts` helpers (per-finger contact, target-env contact count, destination xy box).

Each phase boundary of the fixed rule (SL6.task.md §3 B3) is pinned at its threshold: close onset (action f-1),
two-finger contact, lift by 2 cm rise (≥) or by lost support, transport at 5 cm travel (≥) from the lift point,
placement = inside the destination box after a lift, failed close / drop -> approach.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import sl6_phase_labels as SPL

A, G, L, T, P = range(5)


def make_track(n=20, *, z0=0.90):
    return {"frames": np.arange(n), "target_xpos": np.tile([0.0, 0.0, z0], (n, 1)),
            "contact_left_finger_target": np.zeros(n, bool), "contact_right_finger_target": np.zeros(n, bool),
            "n_contact_target_env": np.full(n, 20, np.int64),
            "destination_xy_box": np.tile([0.20, -0.05, 0.30, 0.05], (n, 1))}


def actions_closing(n, start, stop=None):
    """``action (n-1, 7)``: close (+1) for steps start..stop-1 (frames start+1..stop)."""
    a = np.zeros((n - 1, 7))
    a[:, -1] = -1.0
    a[start:stop, -1] = 1.0
    return a


def test_close_frames_are_shifted_by_one_and_runs():
    a = actions_closing(10, 3, 6)
    c = SPL.close_frames(a, 10)
    assert c.tolist() == [False, False, False, False, True, True, True, False, False, False]
    assert SPL.runs_of(c).tolist() == [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1]
    assert SPL.runs_of(np.array([1, 1, 0, 1], bool)).tolist() == [0, 0, -1, 1]


def test_full_pick_and_place_boundaries():
    n = 20
    t = make_track(n)
    acts = actions_closing(n, 4)                        # close from frame 5 to the end
    two = np.zeros(n, bool)
    two[7:] = True                                      # both fingers from frame 7
    t["contact_left_finger_target"], t["contact_right_finger_target"] = two.copy(), two.copy()
    z = np.full(n, 0.90)
    z[8] = 0.919                                        # 1.9 cm: not lifted (support still touched)
    z[9:] = 0.920 + 1e-9                                # >= 2 cm: lifted
    t["target_xpos"][:, 2] = z
    t["n_contact_target_env"][9:] = 0
    x = np.zeros(n)
    x[10] = 0.0499
    x[11] = 0.0501                                      # >= 5 cm from the lift point (x = 0 at frame 9)
    x[12:] = 0.25                                       # inside the destination box
    t["target_xpos"][:, 0] = x
    lab, info = SPL.request_phases(t, acts, np.arange(2, n))
    full = info["frame_labels"].tolist()
    assert full[:5] == [A] * 5 and full[5:9] == [G] * 4 and full[9:11] == [L, L] and full[11] == T
    assert full[12:] == [P] * (n - 12)
    assert info["first_close"] == 5 and info["first_two"] == 7 and info["first_lift"] == 9
    assert info["first_transport"] == 11 and info["first_placement"] == 12 and info["lost_hold_frames"] == 0
    assert lab.tolist() == full[2:]
    # placement persists after release inside the box (gripper opens, fingers leave)
    t2 = {k: v.copy() for k, v in t.items()}
    t2["contact_left_finger_target"][15:] = False
    acts2 = actions_closing(n, 4, 14)
    assert SPL.frame_phases(t2, acts2, [2])[0][15:].tolist() == [P] * (n - 15)


def test_lift_by_lost_support_without_2cm():
    n = 12
    t = make_track(n)
    acts = actions_closing(n, 2)
    t["contact_left_finger_target"][5:] = True
    t["contact_right_finger_target"][5:] = True
    t["target_xpos"][7:, 2] = 0.905                     # 5 mm only
    t["n_contact_target_env"][7:] = 0                   # but off the cookie box / ramekin
    lab, _ = SPL.frame_phases(t, acts, [0])
    assert lab[3:7].tolist() == [G] * 4 and lab[7:].tolist() == [L] * (n - 7)


def test_failed_close_is_approach_and_retry_is_grasp():
    n = 24
    t = make_track(n)
    a = np.zeros((n - 1, 7))
    a[:, -1] = -1
    a[3:8, -1] = 1                                      # run 0: frames 4-8, one finger only
    a[12:, -1] = 1                                      # run 1: frames 13-, both fingers at 15
    t["contact_right_finger_target"][5:8] = True
    t["contact_left_finger_target"][15:] = True
    t["contact_right_finger_target"][15:] = True
    t["target_xpos"][18:, 2] = 0.95
    lab, info = SPL.frame_phases(t, a, [0])
    assert lab[:13].tolist() == [A] * 13
    assert lab[13:18].tolist() == [G] * 5 and lab[18:].tolist() == [L] * (n - 18)
    assert info["close_runs"] == 2 and info["close_runs_with_two_finger"] == 1


def test_drop_after_lift_is_approach_and_counted():
    n = 16
    t = make_track(n)
    acts = actions_closing(n, 1)
    t["contact_left_finger_target"][3:10] = True
    t["contact_right_finger_target"][3:10] = True
    t["target_xpos"][5:10, 2] = 0.95                    # lifted 5-9
    lab, info = SPL.frame_phases(t, acts, [0])
    assert lab[2:5].tolist() == [G] * 3 and lab[5:10].tolist() == [L] * 5
    assert lab[10:].tolist() == [A] * (n - 10) and info["lost_hold_frames"] == n - 10


def test_held_back_on_the_support_is_grasp_and_box_before_lift_is_not_placement():
    n = 14
    t = make_track(n)
    t["target_xpos"][:, 0] = 0.25                       # starts inside the destination box
    acts = actions_closing(n, 1)
    t["contact_left_finger_target"][2:] = True
    t["contact_right_finger_target"][2:] = True
    lab, _ = SPL.frame_phases(t, acts, [0])
    assert lab[:2].tolist() == [A, A] and set(lab[2:].tolist()) == {G}    # never lifted: no placement
    t["target_xpos"][:, 0] = 0.0
    t["target_xpos"][6:9, 2] = 0.95                     # lifted 6-8, back on the support 9-
    lab, _ = SPL.frame_phases(t, acts, [0])
    assert lab[6:9].tolist() == [L] * 3 and lab[9:].tolist() == [G] * (n - 9)


def test_track_must_cover_every_frame():
    t = make_track(5)
    t["frames"] = np.array([0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="0 … N-1"):
        SPL.frame_phases(t, actions_closing(6, 1), [0])


# --- phase_export helpers ------------------------------------------------------------------------------------
def test_finger_flags_and_env_count():
    from benchmark.adapters.libero_panda import phase_export as pe

    target = {10, 11}
    finger = np.array(["", "left", "right", "", "", "", "", "", "", "", "", ""])
    robot = np.array([True, True, True, False, False, False, False, False, False, False, False, False])
    assert pe.finger_flags([(10, 1)], target, finger, robot) == (True, False, 0)
    assert pe.finger_flags([(2, 11), (10, 1)], target, finger, robot) == (True, True, 0)
    assert pe.finger_flags([(10, 3), (4, 11), (10, 0), (10, 11)], target, finger, robot) == (False, False, 2)
    assert pe.FINGER_BODIES["left"] == ("gripper0_leftfinger", "gripper0_finger_joint1_tip")


def test_geoms_xy_box_rotated_aabb():
    from benchmark.adapters.libero_panda import phase_export as pe

    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m = types.SimpleNamespace(geom_aabb=np.array([[0, 0, 0, 0.1, 0.1, 0.01], [0, 0, 0, 0.05, 0.05, 0.05]]))
    d = types.SimpleNamespace(geom_xpos=np.array([[1.0, 2.0, 0.0], [1.3, 2.0, 0.0]]),
                              geom_xmat=np.stack([R.reshape(-1), np.eye(3).reshape(-1)]))
    box = pe.geoms_xy_box(m, d, [0])
    h = 0.1 * np.sqrt(2)
    assert np.allclose(box, [1 - h, 2 - h, 1 + h, 2 + h])
    assert np.allclose(pe.geoms_xy_box(m, d, [0, 1]), [1 - h, 2 - h, 1.35, 2 + h])
