"""T43 T fix 3 — identity robustness at pick (all three off by default).

(a) `clustering.latched_identity_hold`: while the grasp latch has the pick target locked (LATCHED) or
    is closing, a challenger frame counts only on positive evidence of a change of object — the
    object gone (`association_rejected` with the hand clear of it) or the hand at the challenger.
    Unseen / out-scored-while-seen = a gap. Off = the T32b rule (pinned by
    `test_release_split_and_hand.py`, unchanged).
(b) `clustering.refused_identity_release_frames`: N consecutive destination-overlap refusals of the
    exclusion gate release the identity; `first` picks again from the admissible clusters.
(c) `clustering.supported_max_bottom`: an admissible cluster must rest on a support plane (lowest
    point ≤ the limit above it, highest ≥ `supported_min_top`) or be in the hand.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.admissibility import NOT_SUPPORTED, HandSpheres
from AG3S.ag3s.stages.target_grounding import (
    ASSOCIATION_REJECTED,
    TargetConfirm,
    ground_target,
)
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

OPENING = 0.07146
APPLE_C = np.array([0.50, 0.00, 0.80])
APPLE_R = 0.031
PEAR_C = APPLE_C + np.array([0.0, -0.150, 0.0])
TABLE = np.array([[0.0, 0.0, 1.0, APPLE_C[2] - APPLE_R]])       # the apple rests on it
BASE = ClusteringConfig()
REACH = BASE.hand_occlusion_reach


def _sphere(centre, radius, n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


APPLE = _sphere(APPLE_C, APPLE_R, seed=0)
PEAR = _sphere(PEAR_C, 0.030, seed=2)


def _bar(centre, length=0.14, radius=0.011, n=600, seed=3):
    """A horizontal bar (capsule-ish point cloud) along x, like the hurdle bar top."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-length / 2, length / 2, n)
    a = rng.uniform(0, 2 * np.pi, n)
    return np.asarray(centre) + np.stack([t, radius * np.cos(a), radius * np.sin(a)], axis=1)


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.broadcast_to(np.asarray(a, np.float32), (len(b),)) for b, a in blobs])
    att = att + np.linspace(0.0, 1e-4, len(att), dtype=np.float32)
    return AttentionPointCloud(PointCloud(pts), att, att)


def _cfg(**kw):
    return dataclasses.replace(BASE, **kw)


def _confirm(cfg, **kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("overlap_distance", 0.005)
    return TargetConfirm(cfg, **kw)


def _hand(centre, radius=0.02):
    return HandSpheres(np.asarray(centre, float).reshape(1, 3), np.array([radius]), ("ee_left",))


HAND_ON_APPLE = _hand(APPLE_C + [0.0, 0.0, APPLE_R + 0.02 + 0.010])   # 10 mm above the apple
HAND_AWAY = _hand(APPLE_C + [0.0, 0.0, 0.40])


def _step(confirm, apc, cfg, *, hand=None, latched=None, support=None):
    if latched is not None:
        confirm.note_latched(latched)
    confirm.note_hand(hand)
    if support is not None:
        confirm.note_support(support)
    ground_target(apc, cfg, seed_threshold=0.3, confirm=confirm)
    return confirm.last


def _adopt(cfg, **kw):
    c = _confirm(cfg, **kw)
    _step(c, _apc((APPLE, 0.9)), cfg)
    assert c.manipulated.id == 0
    return c


# ================================================================================ config

def test_defaults_are_off_and_validated():
    c = ClusteringConfig()
    assert c.latched_identity_hold is False
    assert c.refused_identity_release_frames is None
    assert c.supported_max_bottom is None and c.supported_min_top == pytest.approx(0.01)
    AG3SConfig().with_overrides({"clustering": {"latched_identity_hold": True,
                                                "refused_identity_release_frames": 3,
                                                "supported_max_bottom": 0.06}})
    for bad in ({"latched_identity_hold": 1}, {"refused_identity_release_frames": 0},
                {"refused_identity_release_frames": 2.5}, {"refused_identity_release_frames": True},
                {"supported_max_bottom": float("nan")}, {"supported_max_bottom": "x"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig().with_overrides({"clustering": bad})


# ================================================================================ (a)

def test_off_an_unseen_latched_apple_is_switched_away_on_frame_3():
    """The T41 c C/D ep1982 sequence, off: apple unobserved, the bar leads — switch on frame 3,
    whether or not the latch holds the target (off reads nothing)."""
    cfg = _cfg()
    c = _adopt(cfg)
    modes = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY, latched=True).mode for _ in range(3)]
    assert modes == ["occluded", "occluded", "switch"] and c.manipulated.id == 1


def test_on_an_unseen_latched_apple_is_kept():
    cfg = _cfg(latched_identity_hold=True)
    c = _adopt(cfg)
    decs = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY, latched=True) for _ in range(6)]
    assert [d.mode for d in decs] == ["occluded"] * 6
    assert all(d.latched_hold and d.streak == 0 for d in decs)
    assert c.manipulated.id == 0


def test_on_an_out_scored_but_visible_latched_apple_is_kept():
    """Seen in place, attention on something else: attention moving, not the object changing."""
    cfg = _cfg(latched_identity_hold=True)
    c = _adopt(cfg)
    frame = _apc((APPLE, 0.35), (PEAR, 0.95))
    decs = [_step(c, frame, cfg, hand=HAND_AWAY, latched=True) for _ in range(5)]
    assert [d.mode for d in decs] == ["hold"] * 5 and c.manipulated.id == 0
    assert all(d.latched_hold for d in decs)


def test_on_not_latched_the_rule_is_unchanged():
    """Before the latch locks a target (SEARCHING) the T26/T32b switch still runs."""
    cfg = _cfg(latched_identity_hold=True)
    c = _adopt(cfg)
    modes = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY, latched=False).mode for _ in range(3)]
    assert modes == ["occluded", "occluded", "switch"]
    assert not c.last.latched_hold


def test_on_the_hand_at_the_challenger_still_switches():
    """Positive evidence: the hand went to the challenger (T32b's genuine change)."""
    cfg = _cfg(latched_identity_hold=True)
    c = _adopt(cfg)
    hand_at_pear = _hand(PEAR_C + [0.0, 0.0, 0.030 + 0.004 + 0.010], radius=0.004)
    modes = [_step(c, _apc((PEAR, 0.9)), cfg, hand=hand_at_pear, latched=True).mode
             for _ in range(3)]
    assert modes == ["occluded", "occluded", "switch"] and c.manipulated.id == 1


def test_on_a_rejected_association_with_the_hand_clear_switches():
    """Positive evidence the object is gone: a cluster at its place that is not it, hand clear."""
    cfg = _cfg(latched_identity_hold=True, target_identity_tolerance=0.2)
    c = _adopt(cfg, tolerance=0.2)
    orange = _sphere(APPLE_C + [0.0, 0.105, 0.0], 0.030, seed=1)
    decs = [_step(c, _apc((orange, 0.5)), cfg, hand=HAND_AWAY, latched=True) for _ in range(3)]
    assert decs[0].reason == ASSOCIATION_REJECTED
    assert [d.mode for d in decs] == ["occluded", "occluded", "switch"]
    # the same with the hand on the apple (unknown why it looks different): kept
    c = _adopt(cfg, tolerance=0.2)
    decs = [_step(c, _apc((orange, 0.5)), cfg, hand=HAND_ON_APPLE, latched=True) for _ in range(5)]
    assert all(d.mode == "occluded" for d in decs) and c.manipulated.id == 0
    # and without any hand information: kept (the hand cannot be shown clear)
    c = _adopt(cfg, tolerance=0.2)
    decs = [_step(c, _apc((orange, 0.5)), cfg, hand=None, latched=True) for _ in range(5)]
    assert c.manipulated.id == 0


def test_on_frozen_is_unchanged():
    cfg = _cfg(latched_identity_hold=True)
    c = _adopt(cfg)
    c.freeze(True)
    decs = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY, latched=True) for _ in range(4)]
    assert all(d.mode == "occluded" and not d.latched_hold and d.frozen for d in decs)


def test_latched_flag_is_cleared_by_reset():
    cfg = _cfg(latched_identity_hold=True)
    c = _adopt(cfg)
    c.note_latched(True)
    c.reset()
    _step(c, _apc((APPLE, 0.9)), cfg)
    modes = [_step(c, _apc((PEAR, 0.9)), cfg, hand=HAND_AWAY).mode for _ in range(3)]
    assert modes[-1] == "switch"


# ================================================================================ (b)

def _with_destination(cfg):
    c = _adopt(cfg)
    return c


def test_off_refusals_only_count():
    cfg = _cfg()
    c = _adopt(cfg)
    for _ in range(10):
        c.note_exclusion_refused(True)
    assert c.refused_streak == 10
    _step(c, _apc((APPLE, 0.9)), cfg)
    assert c.manipulated.id == 0 and c.refused_released is None


def test_on_n_refusals_release_and_first_picks_again():
    cfg = _cfg(refused_identity_release_frames=3)
    c = _adopt(cfg)
    for _ in range(2):
        c.note_exclusion_refused(True)
    _step(c, _apc((APPLE, 0.9)), cfg)
    assert c.manipulated.id == 0 and c.refused_released is None
    c.note_exclusion_refused(True)                       # the 3rd in a row
    d = _step(c, _apc((PEAR, 0.9), (APPLE, 0.2)), cfg)
    assert c.refused_released == {"id": 0, "refused_frames": 3}
    assert d.mode == "first" and c.manipulated.id == 1   # a fresh id, the leader (pear)
    assert c.refused_streak == 0


def test_on_a_break_in_the_refusals_restarts_the_count():
    cfg = _cfg(refused_identity_release_frames=3)
    c = _adopt(cfg)
    for r in (True, True, False, True, True):
        c.note_exclusion_refused(r)
    _step(c, _apc((APPLE, 0.9)), cfg)
    assert c.manipulated.id == 0 and c.refused_released is None


def test_on_no_release_while_frozen():
    cfg = _cfg(refused_identity_release_frames=2)
    c = _adopt(cfg)
    c.freeze(True)
    for _ in range(5):
        c.note_exclusion_refused(True)
    _step(c, _apc((PEAR, 0.9)), cfg)
    assert c.manipulated.id == 0 and c.refused_released is None


def test_on_release_also_applies_on_a_frame_without_clusters():
    cfg = _cfg(refused_identity_release_frames=1)
    c = _adopt(cfg)
    c.note_exclusion_refused(True)
    assert c.unseen() is None and c.manipulated is None
    assert c.refused_released == {"id": 0, "refused_frames": 1}


# ================================================================================ (c)

def test_off_a_bar_top_is_admissible():
    """The hurdle bar top: narrowest 22 mm ≤ 71.46 mm — graspable by T26's rule."""
    cfg = _cfg()
    c = _confirm(cfg)
    bar = _bar(APPLE_C + [0.10, 0.0, 0.21])
    _step(c, _apc((bar, 0.9)), cfg, support=TABLE)
    assert c.last.mode == "first" and c.last_admissibility[0].admissible


def test_on_a_bar_top_is_not_supported_and_the_apple_is_taken():
    cfg = _cfg(supported_max_bottom=0.06)
    c = _confirm(cfg)
    bar = _bar(APPLE_C + [0.10, 0.0, 0.21])
    d = _step(c, _apc((bar, 0.9), (APPLE, 0.4)), cfg, support=TABLE)
    reasons = [a.reason for a in c.last_admissibility]
    assert NOT_SUPPORTED in reasons
    assert d.mode == "first" and np.linalg.norm(c.manipulated.centroid - APPLE_C) < 0.01
    ev = c.last_support
    assert [e["supported"] for e in ev] == [False, True]
    assert ev[0]["bottom_mm"] > 150 and ev[1]["bottom_mm"] < 5


def test_on_an_apple_on_the_table_is_untouched():
    cfg = _cfg(supported_max_bottom=0.06)
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg, support=TABLE)
    assert c.last.mode == "first" and c.last_admissibility[0].admissible


def test_on_a_lifted_apple_in_the_hand_stays_admissible():
    cfg = _cfg(supported_max_bottom=0.06)
    lifted = APPLE + [0.0, 0.0, 0.20]
    c = _confirm(cfg)
    hand = _hand(APPLE_C + [0.0, 0.0, 0.20 + APPLE_R + 0.02 + 0.010])
    _step(c, _apc((lifted, 0.9)), cfg, support=TABLE, hand=hand)
    assert c.last_admissibility[0].admissible
    c = _confirm(cfg)
    _step(c, _apc((lifted, 0.9)), cfg, support=TABLE, hand=HAND_AWAY)
    assert c.last_admissibility[0].reason == NOT_SUPPORTED


def test_on_something_under_the_table_is_not_supported():
    cfg = _cfg(supported_max_bottom=0.06)
    c = _confirm(cfg)
    flat = APPLE_C + np.c_[np.random.default_rng(4).uniform(-0.04, 0.04, (300, 2)),
                           np.zeros(300)] * [1, 1, 0] - [0, 0, APPLE_R + 0.022]
    _step(c, _apc((flat, 0.9)), cfg, support=TABLE)
    assert c.last_admissibility[0].reason == NOT_SUPPORTED


def test_on_no_planes_or_frozen_skips_the_test():
    cfg = _cfg(supported_max_bottom=0.06)
    bar = _bar(APPLE_C + [0.10, 0.0, 0.21])
    c = _confirm(cfg)
    _step(c, _apc((bar, 0.9)), cfg, support=None)
    assert c.last_admissibility[0].admissible and c.last_support is None
    c = _confirm(cfg)
    c.freeze(True)
    _step(c, _apc((bar, 0.9)), cfg, support=TABLE)
    assert c.last_admissibility[0].admissible


def test_support_record_is_json():
    cfg = _cfg(supported_max_bottom=0.06)
    c = _confirm(cfg)
    _step(c, _apc((APPLE, 0.9)), cfg, support=TABLE)
    json.dumps(c.last_support)


# ================================================================================ step 2 (i)
#: `clustering.closing_keep_geometry` — while closing (frozen, nothing attached) the pre-closing
#: geometry stands; off = T30 (frozen ⇒ the geometry follows the observation).

from AG3S.ag3s.stages.target_grounding import CLOSING_KEPT_GEOMETRY  # noqa: E402

APPLE_CAP = APPLE[APPLE[:, 2] > APPLE_C[2] + 0.5 * APPLE_R]      # what the fingers leave visible


def _closing(cfg, pre_attach=True):
    c = _adopt(cfg)
    full = c.manipulated.geometry.points.copy()
    c.freeze(True)
    c.note_pre_attach_grasp(pre_attach)
    d = _step(c, _apc((APPLE_CAP, 0.9)), cfg)
    return c, d, full


def test_off_the_closing_frame_replaces_the_geometry_with_the_cap():
    c, d, full = _closing(_cfg())
    assert d.reason != CLOSING_KEPT_GEOMETRY
    assert len(c.manipulated.geometry.points) < len(full)        # 114 → 33 in T42-1 ep1908


def test_on_the_closing_frame_keeps_the_pre_closing_geometry():
    cfg = _cfg(closing_keep_geometry=True)
    c, d, full = _closing(cfg)
    assert d.reason == CLOSING_KEPT_GEOMETRY and d.frozen
    g = c.manipulated.geometry
    assert np.array_equal(g.points, full)
    assert c.manipulated.state == "visible" and c.manipulated.id == 0
    assert np.allclose(c.manipulated.centroid, full.mean(axis=0), atol=1e-9)


def test_on_ground_target_hands_back_the_kept_geometry():
    cfg = _cfg(closing_keep_geometry=True)
    c = _adopt(cfg)
    full = c.manipulated.geometry.points.copy()
    c.freeze(True)
    c.note_pre_attach_grasp(True)
    res = ground_target(_apc((APPLE_CAP, 0.9)), cfg, seed_threshold=0.3, confirm=c)
    assert np.array_equal(res.target.points, full)
    assert res.target.metrics["manipulated_subset_kept_anchor"] == 1.0


def test_on_once_attached_the_observation_is_followed_again():
    c, d, full = _closing(_cfg(closing_keep_geometry=True), pre_attach=False)
    assert d.reason != CLOSING_KEPT_GEOMETRY
    assert len(c.manipulated.geometry.points) < len(full)


def test_on_not_frozen_is_the_t30_rule():
    cfg = _cfg(closing_keep_geometry=True)
    c = _adopt(cfg)
    c.note_pre_attach_grasp(True)                      # without a freeze it means nothing
    d = _step(c, _apc((APPLE_CAP, 0.9)), cfg)
    assert d.reason != CLOSING_KEPT_GEOMETRY


def test_on_reset_clears_the_closing_flag():
    cfg = _cfg(closing_keep_geometry=True)
    c = _adopt(cfg)
    c.note_pre_attach_grasp(True)
    c.reset()
    _step(c, _apc((APPLE, 0.9)), cfg)
    c.freeze(True)
    d = _step(c, _apc((APPLE_CAP, 0.9)), cfg)
    assert d.reason != CLOSING_KEPT_GEOMETRY


def test_closing_keep_geometry_config():
    assert ClusteringConfig().closing_keep_geometry is False
    with pytest.raises(AG3SConfigError):
        AG3SConfig().with_overrides({"clustering": {"closing_keep_geometry": "yes"}})


# ================================================================================ AG3S wiring

def test_ag3s_hands_the_grasp_state_to_target_confirm_before_grounding(monkeypatch):
    """`AG3S.process` gives `note_latched`, `note_pre_attach_grasp` (grasp active ∧ nothing attached)
    and `note_support` (this frame's planes) before the grounding call, and reports a destination-
    overlap refusal after it (`note_exclusion_refused`)."""
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.stages.target_grounding import TargetConfirm as TC
    from tests.ag3s import test_admissibility as TA
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, LABEL_SPHERE, PlanarTwoLinkArm, make_scene

    seen = []
    for name in ("note_latched", "note_pre_attach_grasp", "note_support", "note_exclusion_refused"):
        orig = getattr(TC, name)

        def spy(self, v, _orig=orig, _name=name):
            seen.append((_name, None if v is None else (v if isinstance(v, bool) else len(v))))
            return _orig(self, v)
        monkeypatch.setattr(TC, name, spy)
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=TA.APPLE_AND_CRATE,
                       target_label=LABEL_SPHERE)
    ag = AG3S(AG3SConfig(), robot_model=arm)
    ag.process(**TA._inputs(scene), active_manipulators=["left"])
    ag.set_grasp_latched(True)
    ag.set_grasp_active(True)
    ag.process(**TA._inputs(scene), active_manipulators=["left"])
    by = lambda n: [v for k, v in seen if k == n]  # noqa: E731
    assert by("note_latched") == [False, True]
    assert by("note_pre_attach_grasp") == [False, True]
    assert len(by("note_support")) == 2
    assert by("note_exclusion_refused") == [False, False]
    ag.reset()
    assert ag._grasp_latched is False                                          # noqa: SLF001
