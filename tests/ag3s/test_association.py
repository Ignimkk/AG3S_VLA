"""T30 F3 — before a grasp, the manipulated object stays the object it was adopted as.

E3 (T28 Phase 3): ep1808's manipulated id 0 ended up on the orange and ep1800's on the banana
before any grasp. The replay of the server records (900/900 chunks reproduce) showed how: the
cluster adopted at t=0 was already the apple **merged** with its neighbour (extents 54·67·144 and
59·65·164 mm; the apple alone is 61·69·69), and as the hand hid the apple the blob shrank to the
neighbour alone — each frame a few millimetres from the last, so the held centroid and the
exclusion geometry followed it off the apple. What is pinned here:

(a) association — the cluster at the held object's place is that object only if its centroid is
    within the object's own radius of the last admissible geometry and its narrowest / middle
    extents are within `clustering.manipulated_extent_ratio`. Otherwise `association_rejected`
    (occluded, last admissible geometry kept). The T26 switch rule is untouched.
(b) anchor — before a grasp, an associated observation replaces the geometry only if it covers at
    least `clustering.manipulated_anchor_coverage` of it. One that shows only part of it (the hand
    hides the rest) is `subset_kept_anchor`: seen, but the anchor stays the exclusion geometry.
Both are off while a grasp is in progress (the object moves with the hand then).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.admissibility import (
    ASSOC_MIDDLE,
    ASSOC_NARROWEST,
    ASSOC_SHIFT,
    anchor_cover,
    associate,
    principal_extents,
)
from AG3S.ag3s.stages.target_grounding import (
    ASSOCIATION_REJECTED,
    OCCLUDED,
    SUBSET_KEPT_ANCHOR,
    VISIBLE,
    ClusterInfo,
    TargetConfirm,
    ground_target,
)
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

CFG = ClusteringConfig(min_points=10)
OPENING = 0.07146                               # RB-Y1, T26
APPLE_C = np.array([0.50, 0.00, 0.80])
ORANGE_C = APPLE_C + np.array([0.0, 0.105, 0.0])   # the orange at 105 mm (task)
APPLE_R = 0.031                                 # 62 mm across


def _ball(centre, radius=APPLE_R, n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


APPLE = _ball(APPLE_C, seed=0)
ORANGE = _ball(ORANGE_C, seed=1)


def _hidden(points, keep_fraction, axis=1, centre=APPLE_C, sign=-1.0):
    """The part of `points` not hidden by a hand coming from `sign * axis` — `keep_fraction` of
    them, cut by a plane across `axis`."""
    proj = (points - centre)[:, axis] * -sign
    cut = np.quantile(proj, 1.0 - keep_fraction)
    return points[proj >= cut]


def _clusters(*items):
    """`(points, score)` → (score-sorted ClusterInfo list, stacked cloud)."""
    clouds, infos, start = [], [], 0
    for i, (pts, score) in enumerate(items):
        pts = np.asarray(pts, np.float64)
        idx = np.arange(start, start + len(pts))
        start += len(pts)
        clouds.append(pts)
        infos.append(ClusterInfo(
            id=i, point_indices=idx, centroid=pts.mean(axis=0), mean_attention=score,
            max_attention=score, attention_mass=score, point_count=len(pts),
            spatial_compactness=1.0, distance_from_attention_peak=0.0, target_score=score))
    infos.sort(key=lambda c: -c.target_score)
    return infos, np.concatenate(clouds)


def _confirm(**kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("overlap_distance", 0.005)
    return TargetConfirm(CFG, **kw)


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.full(len(b), a, np.float32) + np.linspace(0, 1e-3, len(b),
                                                                       dtype=np.float32)
                          for b, a in blobs])
    return AttentionPointCloud(PointCloud(pts), att, att)


def _ground(confirm, *blobs):
    return ground_target(_apc(*blobs), CFG, seed_threshold=0.5, confirm=confirm)


def _step(confirm, *items):
    """One frame through `select` + `observe`, the way `ground_target` drives it."""
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry

    clusters, pts = _clusters(*items)
    dec = confirm.select(clusters, points=pts)
    if dec.cluster is not None and dec.reason != SUBSET_KEPT_ANCHOR:
        cp = pts[dec.cluster.point_indices]
        confirm.observe(TargetGeometry(
            id=dec.cluster.id, points=cp, point_indices=dec.cluster.point_indices,
            centroid=dec.cluster.centroid, bounding_geometry=fit_sphere(cp, semantic_role="target"),
            attention_score=0.0, confidence=dec.cluster.target_score, timestamp=0.0))
    return dec


def _apple_inside(geometry):
    """Every apple point inside the geometry's bounding sphere (what the exclusion is built on)."""
    r = float(np.asarray(geometry.bounding_geometry.dimensions)[0])
    return bool(np.all(np.linalg.norm(APPLE - geometry.centroid, axis=1) <= r + 1e-9))


# --------------------------------------------------------------------------- the pure test


def test_associate_is_shift_within_radius_and_both_ratios_within_the_band():
    ext = principal_extents(APPLE)
    ok = associate(APPLE_C + [0.01, 0, 0], ext, reference_centroid=APPLE_C,
                   reference_radius=APPLE_R, reference_extents=ext, max_ratio=2.0)
    assert ok.accepted and ok.failed == () and ok.extent_ratio == pytest.approx((1.0, 1.0))
    far = associate(ORANGE_C, ext, reference_centroid=APPLE_C, reference_radius=APPLE_R,
                    reference_extents=ext, max_ratio=2.0)
    assert not far.accepted and far.failed == (ASSOC_SHIFT,)
    assert far.shift_m == pytest.approx(0.105)
    small = associate(APPLE_C, ext * [0.4, 1.0, 1.0], reference_centroid=APPLE_C,
                      reference_radius=APPLE_R, reference_extents=ext, max_ratio=2.0)
    assert small.failed == (ASSOC_NARROWEST,)
    big = associate(APPLE_C, ext * [1.0, 2.5, 3.0], reference_centroid=APPLE_C,
                    reference_radius=APPLE_R, reference_extents=ext, max_ratio=2.0)
    assert big.failed == (ASSOC_MIDDLE,)
    json.dumps(big.record())


def test_anchor_cover_measures_how_much_of_the_anchor_is_shown():
    part = _hidden(APPLE, 0.55)
    assoc = associate(part.mean(0), principal_extents(part), reference_centroid=APPLE_C,
                      reference_radius=APPLE_R, reference_extents=principal_extents(APPLE),
                      max_ratio=2.0)
    assert assoc.accepted
    got = anchor_cover(assoc, APPLE, part, distance=0.005, min_coverage=0.8)
    # the shown 55 % plus the anchor points within 5 mm of the cut
    assert len(part) / len(APPLE) <= got.coverage < 0.8
    assert got.containment == pytest.approx(1.0)
    assert got.subset and got.outcome == SUBSET_KEPT_ANCHOR
    whole = anchor_cover(associate(APPLE_C, principal_extents(APPLE), reference_centroid=APPLE_C,
                                   reference_radius=APPLE_R,
                                   reference_extents=principal_extents(APPLE), max_ratio=2.0),
                         APPLE, _ball(APPLE_C, seed=5, n=2000), distance=0.005, min_coverage=0.5)
    assert not whole.subset and whole.outcome == "accepted"


# ----------------------------------------------------------------------------- (a) association


def test_a_partially_occluded_apple_still_associates():
    """The hand hides part of the apple: its centroid moves less than its radius and it is still
    the apple — `keep`, same id, visible (not `association_rejected`)."""
    confirm = _confirm()
    assert _step(confirm, (APPLE, 0.9)).mode == "first"
    part = _hidden(APPLE, 0.55)
    assert np.linalg.norm(part.mean(0) - APPLE_C) < APPLE_R
    dec = _step(confirm, (part, 0.9))
    assert dec.mode == "keep" and dec.reason != ASSOCIATION_REJECTED
    assert dec.association is not None and dec.association.accepted
    assert confirm.manipulated.id == 0 and confirm.state == VISIBLE


def test_the_orange_at_105_mm_is_never_the_apple():
    """With the apple hidden and the orange 105 mm away leading: never associated as the apple.
    Default tolerance (60 mm): not even the nearest candidate. With the identity tolerance widened
    to 0.2 m it is the nearest, and the association test (i) rejects it — shift 105 mm > r."""
    confirm = _confirm()
    _step(confirm, (APPLE, 0.9))
    dec = _step(confirm, (ORANGE, 0.05))       # below the switch minimum: no challenger either
    assert dec.mode == OCCLUDED and confirm.manipulated.id == 0
    assert np.allclose(confirm.manipulated.centroid, APPLE.mean(0))

    wide = _confirm(tolerance=0.2)
    _step(wide, (APPLE, 0.9))
    dec = _step(wide, (ORANGE, 0.05))
    assert dec.mode == OCCLUDED and dec.reason == ASSOCIATION_REJECTED
    assert ASSOC_SHIFT in dec.association.failed
    assert dec.association.shift_m == pytest.approx(0.105, abs=0.005)
    assert wide.manipulated.id == 0 and _apple_inside(wide.manipulated.geometry)


def test_a_rejected_cluster_is_still_an_ordinary_challenger():
    """The T26 switch rule is unchanged: a rejected cluster that leads at ≥ 0.1 for 3 frames takes
    over under a new id."""
    confirm = _confirm(tolerance=0.2)
    _step(confirm, (APPLE, 0.9))
    modes = [_step(confirm, (ORANGE, 0.5)).mode for _ in range(3)]
    assert modes == [OCCLUDED, OCCLUDED, "switch"]
    assert confirm.manipulated.id == 1 and confirm.manipulated.switched_from == 0


def test_an_abrupt_jump_from_a_fragment_to_the_neighbour_is_rejected():
    """ep1800 r1 t=128→136: the associated cluster was an apple fragment (14·34·58 mm), then the
    banana-side blob 50 mm away (shift/r 1.50, narrowest ratio 2.2)."""
    confirm = _confirm(tolerance=0.2, anchor_coverage=None)
    frag = _hidden(APPLE, 0.25, axis=1, sign=+1.0)       # the far quarter, towards the orange
    _step(confirm, (frag, 0.9))
    dec = _step(confirm, (ORANGE, 0.9))
    assert dec.reason == ASSOCIATION_REJECTED and ASSOC_SHIFT in dec.association.failed


def test_a_merged_blob_is_not_the_apple_it_grew_from():
    """The apple, then the apple merged with the orange in one cluster: the centroid moves 52 mm,
    beyond the apple's own 31 mm radius — rejected by (i). (Two fruits side by side grow the
    *largest* extent 62 → 167 mm; narrowest and middle stay ~62, so (ii) alone would not see it.)"""
    confirm = _confirm(tolerance=0.2)
    _step(confirm, (APPLE, 0.9))
    dec = _step(confirm, (np.vstack([APPLE, ORANGE]), 0.9))
    assert dec.reason == ASSOCIATION_REJECTED
    assert dec.association.failed == (ASSOC_SHIFT,)
    assert dec.association.extent_ratio[1] == pytest.approx(1.0, abs=0.1)


def test_no_association_test_while_a_grasp_is_in_progress():
    """Frozen (closing … held): the apple moves with the hand, so it is tracked as T26 did."""
    confirm = _confirm(tolerance=0.2)
    _step(confirm, (APPLE, 0.9))
    confirm.freeze(True)
    lifted = APPLE + [0.0, 0.0, 0.05]
    dec = _step(confirm, (lifted, 0.9))
    assert dec.mode == "keep" and dec.association is None
    assert np.allclose(confirm.manipulated.geometry.centroid, lifted.mean(0))
    assert confirm.manipulated.anchor["source"] == "observed"


def test_the_test_is_off_when_the_ratio_is_none_or_unchecked():
    for confirm in (_confirm(extent_ratio=None), TargetConfirm(CFG)):   # T26 / T20 behaviour
        clusters, pts = _clusters((APPLE, 0.9))
        confirm.select(clusters, points=pts)
        clusters, pts = _clusters((np.vstack([APPLE, ORANGE]), 0.9))
        dec = confirm.select(clusters, points=pts)
        assert dec.association is None and dec.reason is None


# ---------------------------------------------------------------------------------- (b) anchor


def test_e3_merged_blob_shrinking_to_the_neighbour_keeps_the_apple_excluded():
    """Synthetic ep1808: the adopted cluster is the apple merged with the orange; the hand then hides
    the apple a slice at a time until only the orange is left in the blob. Without the anchor the
    geometry follows the blob onto the orange and the apple leaves the exclusion sphere; with it
    the adopted blob — apple included — stays the exclusion geometry, the object stays `visible`."""
    merged = np.vstack([APPLE, ORANGE])
    frames = [np.vstack([_hidden(APPLE, f, axis=1, sign=+1.0), ORANGE])
              for f in (0.8, 0.6, 0.4, 0.2)] + [ORANGE]

    old = _confirm(tolerance=0.2, extent_ratio=None, anchor_coverage=None)
    _step(old, (merged, 0.9))
    for f in frames:
        _step(old, (f, 0.9))
    assert not _apple_inside(old.manipulated.geometry)            # the E3 failure

    new = _confirm(tolerance=0.2)
    assert _step(new, (merged, 0.9)).mode == "first"
    reasons = [_step(new, (f, 0.9)).reason for f in frames]
    assert set(reasons) <= {None, SUBSET_KEPT_ANCHOR} and SUBSET_KEPT_ANCHOR in reasons
    man = new.manipulated
    assert man.id == 0 and man.state == VISIBLE and _apple_inside(man.geometry)
    assert man.anchor["source"] == "first" and man.anchor["kept_frames"] >= 1
    # the held centroid is the kept geometry's, not the orange-only blob's
    assert np.allclose(man.centroid, man.geometry.centroid)
    assert np.linalg.norm(man.centroid - ORANGE_C) > 0.03


def test_a_full_view_that_covers_the_anchor_refines_it():
    confirm = _confirm()
    _step(confirm, (APPLE, 0.9))
    denser = _ball(APPLE_C, seed=7, n=3000)
    dec = _step(confirm, (denser, 0.9))
    assert dec.reason is None and dec.association.outcome == "accepted"
    assert len(confirm.manipulated.points) == 3000
    anchor = confirm.manipulated.anchor
    assert anchor["geometry_source"] == "refined" and anchor["source"] == "first"
    assert anchor["n_points"] == len(APPLE)          # coverage is measured against the adopted one


def test_ground_target_returns_the_anchor_on_a_subset_frame():
    confirm = _confirm()
    _ground(confirm, (APPLE, 0.9))
    part = _hidden(APPLE, 0.55)
    res = _ground(confirm, (part, 0.9))
    m = res.target.metrics
    assert confirm.last.reason == SUBSET_KEPT_ANCHOR
    assert m["manipulated_subset_kept_anchor"] == 1.0 and m["manipulated_occluded"] == 0.0
    assert len(res.target.point_indices) == 0 and len(res.target.points) == len(APPLE)
    assert m["association_shift_m"] < APPLE_R
    rec = confirm.manipulated.record()
    json.dumps(rec)
    assert rec["association"]["outcome"] == SUBSET_KEPT_ANCHOR
    assert rec["anchor"]["kept_frames"] == 1 and rec["anchor"]["n_points"] == len(APPLE)


def test_ground_target_records_a_rejection():
    confirm = TargetConfirm(CFG, max_opening=OPENING, overlap_distance=0.005, tolerance=0.2)
    _ground(confirm, (APPLE, 0.9))
    res = _ground(confirm, (ORANGE, 0.9))     # leads (challenger 1/3), is not the apple
    m = res.target.metrics
    assert m["manipulated_association_rejected"] == 1.0 and m["manipulated_occluded"] == 1.0
    assert m["association_shift_m"] == pytest.approx(0.105, abs=0.005)
    assert confirm.manipulated.record()["association"]["outcome"] == "rejected"


def test_a_switch_starts_a_new_anchor():
    confirm = _confirm()
    _step(confirm, (APPLE, 0.9))
    for _ in range(3):
        _step(confirm, (APPLE, 0.05), (ORANGE, 0.9))
    assert confirm.manipulated.id == 1
    assert confirm.manipulated.anchor["source"] == "switch"
    assert np.allclose(confirm.manipulated.geometry.centroid, ORANGE.mean(0))


# ---------------------------------------------------------------------------------- config


def test_defaults_and_validation():
    c = AG3SConfig().clustering
    assert c.manipulated_extent_ratio is not None and c.manipulated_extent_ratio > 1.0
    assert c.manipulated_anchor_coverage is not None and 0.0 < c.manipulated_anchor_coverage <= 1.0
    confirm = TargetConfirm(c, max_opening=OPENING)
    assert confirm.extent_ratio == c.manipulated_extent_ratio
    assert confirm.anchor_coverage == c.manipulated_anchor_coverage
    for bad in ({"manipulated_extent_ratio": 1.0}, {"manipulated_extent_ratio": -2},
                {"manipulated_anchor_coverage": 0.0}, {"manipulated_anchor_coverage": 1.5}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"clustering": bad})
    off = AG3SConfig.from_dict({"clustering": {"manipulated_extent_ratio": None,
                                               "manipulated_anchor_coverage": None}}).clustering
    assert TargetConfirm(off, max_opening=OPENING).extent_ratio is None
