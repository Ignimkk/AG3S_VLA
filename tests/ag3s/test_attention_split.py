"""T31b — a cluster that may be two objects is split by attention.

E3: the manipulated object adopted at t=0 was the apple **merged** with a neighbour fruit on 2 of
3 episodes (ep1808 apple+orange 54·67·144 mm, ep1800 apple+banana 59·65·164 mm). The narrowest
extent fits the gripper (71.46 mm), so the merge was admissible and the exclusion covered the
neighbour on 249 of 408 pre-grasp chunks. Attention separates the two: the apple holds 0.75–0.90 of
the cluster's attention and its mean attention is ≥ 2.2× the neighbour's.

Pinned here, on synthetic clouds with the E3 geometry (a 62 mm apple, a neighbour 20 mm away — within
the 30 mm grounding `eps`, so grounding merges them):
  - the merge is split into the apple (target candidate, largest attention mass) and the neighbour
    (its own cluster: an obstacle), and the exclusion ball then holds the apple and not the
    neighbour;
  - a **single** apple is never split — not with uneven attention, not with a finger-wide band
    hidden, not when it measures longer than the opening;
  - no split when attention does not tell the parts apart, while a grasp is in progress, for a
    cluster wider than the opening in every direction, or with the split off;
  - every decision is recorded (before/after extents, attention mass, method).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.admissibility import principal_extents
from AG3S.ag3s.stages.attention_split import (
    ATTENTION_NOT_DISCRIMINATIVE,
    ONE_COMPONENT,
    SPLIT,
    SPLIT_METHOD,
    split_by_attention,
    triggered,
)
from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

CFG = ClusteringConfig()                       # min_points 20, eps 30 mm, split radius 15 mm
OFF = dataclasses.replace(CFG, attention_split_radius=None)
OPENING = 0.07146                              # RB-Y1 (T26)
APPLE_C = np.array([0.50, 0.00, 0.80])
APPLE_R = 0.031                                # 62 mm across
GAP = 0.020                                    # surface-to-surface: < eps 30 mm, > split 15 mm
ORANGE_R = 0.030
ORANGE_C = APPLE_C + np.array([0.0, APPLE_R + GAP + ORANGE_R, 0.0])
PAD = 0.005                                    # `_target_ball`: + 1 fine voxel


def _sphere(centre, radius, n=400, seed=0, scale=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(centre) + radius * v * np.asarray(scale)


APPLE = _sphere(APPLE_C, APPLE_R, seed=0)
ORANGE = _sphere(ORANGE_C, ORANGE_R, seed=1)
# A banana: 164 mm long, 36 mm thick, along y, its end 20 mm from the apple's surface.
BANANA_C = APPLE_C + np.array([0.0, APPLE_R + GAP + 0.082, 0.0])
BANANA = _sphere(BANANA_C, 1.0, n=500, seed=2, scale=(0.018, 0.082, 0.018))


def _apc(*blobs):
    """`(points, attention)` blobs → one attention cloud (attention may be a scalar or per point)."""
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.broadcast_to(np.asarray(a, np.float32), (len(b),)) for b, a in blobs])
    att = att + np.linspace(0.0, 1e-4, len(att), dtype=np.float32)   # not flat
    return AttentionPointCloud(PointCloud(pts), att, att)


def _ground(apc, confirm=None, cfg=CFG):
    return ground_target(apc, cfg, seed_threshold=0.3, confirm=confirm)


def _confirm(cfg=CFG, **kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("overlap_distance", 0.005)
    return TargetConfirm(cfg, **kw)


def _ball(points):
    """`AG3S._target_ball` (quantile 1.0 + 1 fine voxel), without `z_min`."""
    c = points.mean(axis=0)
    return c, float(np.linalg.norm(points - c, axis=1).max()) + PAD


def _split(points, attention, **kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("radius", CFG.attention_split_radius)
    kw.setdefault("min_points", CFG.min_points)
    kw.setdefault("min_ratio", CFG.attention_split_ratio)
    pts = np.asarray(points, np.float64)
    return split_by_attention(pts, np.asarray(attention, np.float64), np.arange(len(pts)), **kw)


# ------------------------------------------------------------------------ the pure decision


def test_the_fixture_is_the_e3_merge():
    """Grounding at eps 30 mm merges the apple and the orange 20 mm away: admissible (narrowest ≤
    opening) but longer than the opening — the E3 ep1808 shape."""
    g = _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), cfg=OFF)
    assert len(g.clusters) == 1 and g.clusters[0].point_count == len(APPLE) + len(ORANGE)
    ext = g.clusters[0].principal_extents
    assert ext[0] <= OPENING < ext[2]
    assert triggered(ext, OPENING)


def test_apple_and_orange_split_into_the_apple_and_an_obstacle():
    pts = np.concatenate([APPLE, ORANGE])
    att = np.r_[np.full(len(APPLE), 0.883), np.full(len(ORANGE), 0.001)]   # E3 ep1808 t=0
    d = _split(pts, att)
    assert d.accepted and d.reason == SPLIT and d.method == SPLIT_METHOD
    assert len(d.components) == 2
    np.testing.assert_array_equal(np.sort(d.target.indices), np.arange(len(APPLE)))
    np.testing.assert_array_equal(np.sort(d.components[1].indices),
                                  np.arange(len(APPLE), len(pts)))
    assert d.ratio >= CFG.attention_split_ratio
    # the apple alone fits the gripper along every axis; the merge did not
    assert d.target.extents_m[-1] <= OPENING < d.parent_extents_m[-1]
    groups = d.groups()
    assert sum(len(g) for g in groups) == len(pts)


def test_apple_and_banana_split_even_with_attention_smeared_onto_the_banana():
    """ep1800 t=0: banana mean attention 0.40 (0.96 at its 90th percentile, the end nearest the
    apple) against the apple's 1.00 — still ≥ 2× apart on average."""
    pts = np.concatenate([APPLE, BANANA])
    y = BANANA[:, 1] - BANANA[:, 1].min()
    banana_att = np.clip(1.0 - y / 0.12, 0.0, 1.0) * 0.96    # smear, decaying along the banana
    att = np.r_[np.ones(len(APPLE)), banana_att]
    assert 0.25 <= banana_att.mean() <= 0.45
    d = _split(pts, att)
    assert d.accepted
    np.testing.assert_array_equal(np.sort(d.target.indices), np.arange(len(APPLE)))
    assert d.ratio >= 2.0


def test_record_has_before_after_extents_attention_mass_and_method():
    pts = np.concatenate([APPLE, ORANGE])
    att = np.r_[np.full(len(APPLE), 0.9), np.full(len(ORANGE), 0.01)]
    rec = _split(pts, att).record()
    json.dumps(rec)
    assert rec["method"] == "attention_recluster" and rec["accepted"] and rec["reason"] == "split"
    assert rec["before"]["n_points"] == len(pts) and rec["after"]["n_points"] == len(APPLE)
    assert rec["before"]["extents_mm"][-1] > 71.46 >= rec["after"]["extents_mm"][-1]
    assert rec["after"]["attention_share"] == pytest.approx(
        0.9 * len(APPLE) / (0.9 * len(APPLE) + 0.01 * len(ORANGE)), abs=1e-3)
    assert rec["radius_mm"] == 15.0 and rec["min_points"] == 20 and rec["min_ratio"] == 2.0
    assert [c["n_points"] for c in rec["components"]] == [len(APPLE), len(ORANGE)]


def test_attention_on_both_does_not_split():
    """Attention that does not tell the parts apart (late E3 ep1800: apple and banana both ~1)
    leaves the cluster whole."""
    pts = np.concatenate([APPLE, ORANGE])
    att = np.r_[np.full(len(APPLE), 0.9), np.full(len(ORANGE), 0.6)]
    d = _split(pts, att)
    assert not d.accepted and d.reason == ATTENTION_NOT_DISCRIMINATIVE
    assert d.ratio == pytest.approx(1.5, abs=1e-6)
    assert [len(g) for g in d.groups()] == [len(pts)]


def test_the_target_is_the_largest_attention_mass():
    """Attention on the orange this frame → the orange is the target candidate (attention decides;
    the T26 switch rule then decides whether the manipulated object changes)."""
    pts = np.concatenate([APPLE, ORANGE])
    att = np.r_[np.full(len(APPLE), 0.02), np.full(len(ORANGE), 0.8)]
    d = _split(pts, att)
    assert d.accepted
    np.testing.assert_array_equal(np.sort(d.target.indices), np.arange(len(APPLE), len(pts)))


# ----------------------------------------------------------------- a single apple stays one


def _long_apple(seed=3):
    """A single apple measured longer than the opening (E3 ep1807: 43·66·74, T17: 52·71·73 mm)."""
    return _sphere(APPLE_C, 1.0, n=450, seed=seed, scale=(0.030, 0.034, 0.038))


def test_single_apple_with_uneven_attention_is_not_split():
    """T14 t=80: half of the apple at attention ~0.01, the other half at 1 (partial occlusion).
    Triggered (largest > opening), but its surface is one component at 15 mm → not split."""
    apple = _long_apple()
    att = np.where(apple[:, 2] > APPLE_C[2], 1.0, 0.01)
    ext = principal_extents(apple)
    assert ext[0] <= OPENING < ext[2]
    d = _split(apple, att)
    assert d is not None and not d.accepted and d.reason == ONE_COMPONENT


def test_single_apple_with_a_finger_wide_band_hidden_is_not_split():
    """A 10 mm band hidden (a finger in front) with attention only on one side: the two sides are
    10 mm apart — under the 15 mm split radius — so it is still one object."""
    apple = _long_apple(seed=4)
    band = np.abs(apple[:, 1] - APPLE_C[1]) < 0.005
    apple = apple[~band]
    att = np.where(apple[:, 1] > APPLE_C[1], 1.0, 0.02)
    d = _split(apple, att)
    assert d is not None and not d.accepted and d.reason == ONE_COMPONENT


def test_grounding_a_single_uneven_apple_is_unchanged_by_the_split():
    apple = _long_apple(seed=5)
    att = np.where(apple[:, 0] < APPLE_C[0], 1.0, 0.01)
    new, old = _confirm(), _confirm(cfg=OFF)
    g_new = _ground(_apc((apple, att), (ORANGE + [0.0, 0.10, 0.0], 0.001)), new)
    g_old = _ground(_apc((apple, att), (ORANGE + [0.0, 0.10, 0.0], 0.001)), old, cfg=OFF)
    assert [c.point_count for c in g_new.clusters] == [c.point_count for c in g_old.clusters]
    np.testing.assert_array_equal(new.manipulated.points, old.manipulated.points)
    rec = new.manipulated.record()["split"]
    assert rec["ran"] and rec["n_split"] == 0
    assert [d["reason"] for d in rec["decisions"]] == [ONE_COMPONENT]
    assert new.manipulated.record()["anchor"]["from_split"] is False


# ---------------------------------------------------------------------- through grounding


def test_grounding_splits_the_merge_and_the_manipulated_object_is_the_apple():
    confirm = _confirm()
    g = _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
    assert len(g.splits) == 1 and g.splits[0].accepted
    assert len(g.clusters) == 2
    target = g.clusters[0]
    assert target.split["role"] == "target" and target.point_count == len(APPLE)
    obstacle = [c for c in g.clusters if c is not target][0]
    assert obstacle.split["role"] == "obstacle" and obstacle.point_count == len(ORANGE)
    assert g.target.metrics["attention_split"] == 1.0
    assert g.target.metrics["attention_split_target"] == 1.0
    man = confirm.manipulated
    assert len(man.points) == len(APPLE)
    assert man.admissibility.admissible
    rec = man.record()
    json.dumps(rec)
    assert rec["anchor"]["from_split"] is True and rec["anchor"]["source"] == "first"
    assert rec["split"]["manipulated_from_split"] is True
    assert rec["split"]["n_triggered"] == 1 and rec["split"]["n_split"] == 1


def test_the_exclusion_ball_holds_the_apple_and_no_longer_the_orange():
    """The side effect T30b measured and this step removes: with the merge adopted, the orange's
    centre was inside the exclusion ball (the target-free layer lost the orange)."""
    new, old = _confirm(), _confirm(cfg=OFF)
    _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), new)
    _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), old, cfg=OFF)
    c_new, r_new = _ball(new.manipulated.points)
    c_old, r_old = _ball(old.manipulated.points)
    assert np.linalg.norm(APPLE_C - c_old) <= r_old
    assert np.linalg.norm(ORANGE_C - c_old) <= r_old               # before: the orange excluded
    assert np.linalg.norm(APPLE_C - c_new) + APPLE_R <= r_new       # after: the whole apple ...
    assert np.linalg.norm(ORANGE_C - c_new) > r_new                 # ... and not the orange
    assert np.linalg.norm(ORANGE_C - c_new) - ORANGE_R > r_new - PAD  # not even its near surface


def test_anchor_is_the_split_apple_across_frames():
    """T30b anchor: adopted at `first` from the split — later frames keep associating to it."""
    confirm = _confirm()
    for _ in range(4):
        _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
    man = confirm.manipulated
    assert man.id == 0 and man.state == "visible" and len(man.points) == len(APPLE)
    assert confirm.last.mode == "keep"
    assert man.record()["anchor"]["from_split"] is True


def test_a_later_unsplittable_merge_does_not_replace_the_split_anchor():
    """E3b ep1808 r2 t=16: the apple and the orange came within 15 mm, the blob was one component,
    it covered the split apple anchor — and T30 F3b refined the geometry to the blob, bringing the
    orange back into the exclusion. The blob shows the anchor (it is the object) but must not
    replace it: `unsplit_kept_anchor`."""
    from AG3S.ag3s.stages.target_grounding import UNSPLIT_KEPT_ANCHOR

    # the orange 10 mm away, seen from one side (150 points): the blob's centroid stays within the
    # apple's radius, so it passes the T30 association test — as the E3 blobs did
    close = _sphere(APPLE_C + [0.0, APPLE_R + 0.010 + ORANGE_R, 0.0], ORANGE_R, n=150, seed=1)
    confirm = _confirm()
    _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)          # t=0: split, anchor = apple
    g = _ground(_apc((APPLE, 0.9), (close, 0.001)), confirm)        # gap 10 mm: one component
    assert [d.reason for d in g.splits] == [ONE_COMPONENT]
    assert g.clusters[0].split["role"] == "unsplit"
    assert confirm.last.reason == UNSPLIT_KEPT_ANCHOR and confirm.last.mode == "keep"
    man = confirm.manipulated
    assert man.state == "visible" and len(man.points) == len(APPLE)   # still the split apple
    assert g.target.metrics["manipulated_subset_kept_anchor"] == 1.0
    c, r = _ball(man.points)
    assert np.linalg.norm(c - close.mean(axis=0)) > r
    # split off: the same blob is the manipulated geometry, the orange inside the exclusion
    old = _confirm(cfg=OFF)
    _ground(_apc((APPLE, 0.9), (close, 0.001)), old, cfg=OFF)
    assert len(old.manipulated.points) == len(APPLE) + len(close)
    c_old, r_old = _ball(old.manipulated.points)
    assert np.linalg.norm(c_old - close.mean(axis=0)) <= r_old


def test_a_single_apple_anchor_is_refined_as_before():
    """ep1807/T14/T17: the anchor is not a split component, so a triggered-but-unsplit view of the
    apple refines it exactly as under T30."""
    first = _long_apple(seed=6)
    second = first + [0.0005, 0.0, 0.0]                          # the same apple, re-observed
    att = lambda p: np.where(p[:, 2] > APPLE_C[2], 1.0, 0.01)  # noqa: E731
    new, old = _confirm(), _confirm(cfg=OFF)
    for pts in (first, second):
        _ground(_apc((pts, att(pts))), new)
        _ground(_apc((pts, att(pts))), old, cfg=OFF)
        assert new.last.reason == old.last.reason
        np.testing.assert_array_equal(new.manipulated.points, old.manipulated.points)
    assert new.manipulated.record()["anchor"]["geometry_source"] == "refined"


def test_no_split_while_a_grasp_is_in_progress():
    confirm = _confirm()
    _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
    confirm.freeze(True)
    g = _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
    assert g.splits == () and len(g.clusters) == 1
    rec = confirm.manipulated.record()["split"]
    assert rec["ran"] is False and rec["skipped"] == "frozen" and rec["decisions"] == []


def test_split_off_is_the_t30_behaviour():
    confirm = _confirm(cfg=OFF)
    g = _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm, cfg=OFF)
    assert g.splits == () and len(g.clusters) == 1
    assert confirm.manipulated.record()["split"]["skipped"] == "off"


def test_offline_grounding_splits_only_with_a_known_opening():
    apc = _apc((APPLE, 0.9), (ORANGE, 0.001))
    assert len(_ground(apc).clusters) == 1                     # no confirm, no opening: no split
    known = dataclasses.replace(CFG, gripper_max_opening=OPENING)
    g = _ground(apc, cfg=known)
    assert len(g.clusters) == 2 and g.clusters[0].point_count == len(APPLE)
    unchecked = TargetConfirm(CFG)                              # UNCHECKED, no config opening
    _ground(apc, unchecked)
    assert unchecked.manipulated.record()["split"]["skipped"] == "max_opening_unknown"
    unknown = TargetConfirm(CFG, max_opening=None)              # fail-closed: nothing graspable
    g = _ground(apc, unknown)
    assert g.splits == ()


def test_a_cluster_wider_than_the_opening_everywhere_is_not_split():
    """Never split an inadmissible cluster (a crate with a hot spot): it could only manufacture a
    graspable fragment of it."""
    rng = np.random.default_rng(7)
    crate = APPLE_C + rng.uniform(-0.1, 0.1, size=(3000, 3)) * [1.0, 1.0, 0.5]
    att = np.where(crate[:, 0] < APPLE_C[0] - 0.06, 1.0, 0.0)
    assert principal_extents(crate)[0] > OPENING
    assert _split(crate, att) is None


def test_stragglers_join_the_nearest_object():
    """Points in a component under `min_points` are re-attached, never dropped."""
    stray = APPLE_C + np.array([[0.0, 0.0, APPLE_R + 0.018 + 0.001 * i] for i in range(5)])
    pts = np.concatenate([APPLE, ORANGE, stray])
    att = np.r_[np.full(len(APPLE), 0.9), np.full(len(ORANGE), 0.001), np.full(len(stray), 0.9)]
    d = _split(pts, att)
    assert d.accepted and sum(len(g) for g in d.groups()) == len(pts)
    assert set(range(len(APPLE) + len(ORANGE), len(pts))) <= set(d.target.indices.tolist())


def test_split_is_deterministic():
    pts = np.concatenate([APPLE, ORANGE])
    att = np.r_[np.full(len(APPLE), 0.9), np.full(len(ORANGE), 0.01)]
    a, b = _split(pts, att), _split(pts, att)
    assert a.record() == b.record()


# ---------------------------------------------------------------------------------- config


def test_config_defaults_and_validation():
    cfg = AG3SConfig()
    assert cfg.clustering.attention_split_radius == 0.015
    assert cfg.clustering.attention_split_ratio == 2.0
    over = AG3SConfig.from_dict({"clustering": {"attention_split_radius": None,
                                                "attention_split_ratio": 3.0}})
    assert over.clustering.attention_split_radius is None
    assert AG3SConfig.from_dict(over.to_dict()) == over
    for bad in ({"attention_split_radius": 0.0}, {"attention_split_radius": -0.01},
                {"attention_split_radius": float("nan")}, {"attention_split_ratio": 0.5},
                {"attention_split_ratio": float("inf")}, {"attention_split_radius": True}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"clustering": bad})
