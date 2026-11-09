"""SL-PE8 (2026-11-09, dev-4a) — the held object = the points that moved with the hand (`stages.held_motion`).

Off by default; on: while held, cloud points that kept their hand-frame position while the hand moved (and are not explained
by 'at rest', within the fingertip reach) are dropped from the scene obstacle-object set. A static neighbour — even one that
was inside the pre-held snapshot — is never dropped.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.held_motion import HeldMotionTracker, hand_pose_and_tip


def _T(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _bowl(centre, seed=0, n=300, r=0.05):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return np.asarray(centre, float) + r * v / np.linalg.norm(v, axis=1, keepdims=True)


def _plane(z, x0=0.2, x1=0.8, y0=-0.3, y1=0.3, step=0.008):
    xs, ys = np.arange(x0, x1, step), np.arange(y0, y1, step)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]


def _top(P):
    return P[P[:, 2] >= np.median(P[:, 2])]


def _bottom(P):
    return P[P[:, 2] < np.median(P[:, 2])]


def test_defaults_off_and_validated():
    assert ClusteringConfig().scene_obstacle_held_motion is False
    with pytest.raises(AG3SConfigError):
        AG3SConfig().with_overrides({"clustering": {"scene_obstacle_held_motion": True}})   # needs scene_obstacle_objects
    with pytest.raises(AG3SConfigError):
        AG3SConfig().with_overrides({"clustering": {"scene_obstacle_held_motion": 1, "scene_obstacle_objects": True}})
    AG3SConfig().with_overrides({"clustering": {"scene_obstacle_held_motion": True, "scene_obstacle_objects": True}})


def test_tracker_drops_what_moves_with_the_hand_not_the_static_scene():
    tr = HeldMotionTracker()
    hand0, bowl0 = np.array([0.5, 0.0, 0.20]), np.array([0.5, 0.0, 0.10])
    book = _plane(0.02, 0.60, 0.75, -0.05, 0.05)          # a static neighbour inside the snapshot reach
    tip0 = hand0 - [0, 0, 0.05]
    snap = np.r_[_top(_bowl(bowl0)), book[:20]]          # the pre-held cut also caught a bit of the book
    m = tr.update(np.r_[_top(_bowl(bowl0)), book], _T(*hand0), tip0, held=False, preheld_geometry=snap)
    assert not m.any()
    rng = np.random.default_rng(1)
    removed_bowl, removed_book = [], []
    for k in range(1, 6):                                 # lift 4 cm per request; the camera now sees the bottom
        d = np.array([0.0, 0.0, 0.04 * k])
        bowl = _bottom(_bowl(bowl0 + d, seed=k)) + rng.normal(0, 0.001, (150, 3))
        P = np.r_[bowl, book]
        m = tr.update(P, _T(*(hand0 + d)), tip0 + d, held=True)
        removed_bowl.append(m[:len(bowl)].mean())
        removed_book.append(int(m[len(bowl):].sum()))
    assert removed_bowl[-1] > 0.5                         # the bottom (never in the snapshot) is found once it moved
    assert sum(removed_book) == 0                         # the static book is never dropped
    assert tr.last["n_verified"] > 0


def test_tracker_nothing_new_without_hand_motion():
    tr = HeldMotionTracker()
    hand, bowl0 = np.array([0.5, 0.0, 0.20]), np.array([0.5, 0.0, 0.10])
    tr.update(_top(_bowl(bowl0)), _T(*hand), hand - [0, 0, 0.05], held=False, preheld_geometry=_top(_bowl(bowl0)))
    for k in range(3):                                    # the hand does not move: the test cannot tell -> keep
        m = tr.update(_bottom(_bowl(bowl0, seed=k + 3)), _T(*hand), hand - [0, 0, 0.05], held=True)
        assert not m.any()
    tr.reset()
    assert tr.last == {} and not len(tr._verified)


def test_scene_obstacles_off_unchanged_on_drops_the_lifted_bowl():
    hand0, bowl0 = np.array([0.5, 0.0, 0.20]), np.array([0.5, 0.0, 0.10])
    mug = _bowl([0.30, 0.20, 0.05], seed=9, r=0.03)

    class Model:   # stand-in for the constraint robot model: one gripper, hand pose = translation
        sphere_link_names = ("panda_leftfinger", "panda_rightfinger")

        def __init__(self):
            self.q = None

        def gripper_joint_pairs(self):
            return [{"parent_link": "panda_hand", "children": ("panda_leftfinger", "panda_rightfinger")}]

        def link_pose(self, q, link):
            return _T(*np.asarray(q, float)[:3])

        def sphere_centers_numeric(self, q):
            h = np.asarray(q, float)[:3]
            return np.array([h + [0.0, 0.02, -0.05], h + [0.0, -0.02, -0.05]]), np.array([0.01, 0.01])

    fake = types.SimpleNamespace(_obstacle_history=[], _held_motion=None, constraint_robot_model=Model())
    fake._held_motion_mask = types.MethodType(AG3S._held_motion_mask, fake)
    snap = _top(_bowl(bowl0))
    AG3S._scene_obstacle_points(fake, np.r_[snap, mug], None, snap, None,
                                held_motion={"robot_state": hand0, "held": False, "preheld_geometry": snap})
    last = None
    for k in range(1, 5):
        d = np.array([0.0, 0.0, 0.04 * k])
        bowl = _bottom(_bowl(bowl0 + d, seed=k))
        out, rec = AG3S._scene_obstacle_points(fake, np.r_[bowl, mug], None, None, None,
                                               held_motion={"robot_state": hand0 + d, "held": True})
        last = (out, rec, bowl)
    out, rec, bowl = last
    assert rec["held_motion"]["n_removed"] > 0
    assert np.any(np.linalg.norm(out - mug.mean(0), axis=1) < 0.05)        # the mug stays an obstacle
    off = types.SimpleNamespace(_obstacle_history=[])
    out_off, _ = AG3S._scene_obstacle_points(off, np.r_[bowl, mug], None, None, None)
    assert np.any(out_off[:, 2] > 0.2)                                       # off: the lifted bowl is an obstacle
    assert not np.any(np.linalg.norm(fake._obstacle_history[-1] - bowl.mean(0), axis=1) < 0.04)   # on: this request has none


def test_hand_pose_and_tip_without_model():
    assert hand_pose_and_tip(None, np.zeros(9)) == (None, None, None)


def test_server_flag_and_version():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    pe12345, pe123458 = pf.PE_VERSIONS[2], pf.PE_VERSIONS[6]
    assert pe123458.endswith("-pe123458")
    assert list(pf.AG3S_VERSIONS[pe123458]) == list(pf.AG3S_VERSIONS[pe12345]) + ["--scene-obstacle-held-motion"]
    p = pf.parse_v11ab_args(argv=pf.version_server_argv(pe123458) + list(pf.AG3S_VERSIONS[pe123458]))
    assert ss.t43t_identity_section(p).get("scene_obstacle_held_motion") is True
    p5 = pf.parse_v11ab_args(argv=pf.version_server_argv(pe12345) + list(pf.AG3S_VERSIONS[pe12345]))
    assert "scene_obstacle_held_motion" not in ss.t43t_identity_section(p5)


def test_pe12345678_is_pe1234567_plus_pe8():
    from AG3S.adapters.panda import profile as pf

    pe1234567, pe12345678 = pf.PE_VERSIONS[5], pf.PE_VERSIONS[7]
    assert pe12345678.endswith("-pe12345678")
    assert list(pf.AG3S_VERSIONS[pe12345678]) == list(pf.AG3S_VERSIONS[pe1234567]) + ["--scene-obstacle-held-motion"]
