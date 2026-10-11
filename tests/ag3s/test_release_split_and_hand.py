"""T32b — two side effects of the T31b attention split, fixed in `TargetConfirm`.

S2 — **a released grasp re-splits its anchor.** While a grasp is in progress (`frozen`) the split is
off and the geometry follows the observation, so the geometry standing when the freeze is released
can be the apple merged with its neighbour again. After the release the split apple is then only a
subset of that merged anchor (`subset_kept_anchor`), and the merge never goes away: 196 of the 200
merged pre-grasp chunks T31b left came after a freeze. Pinned here: on the first grounding call
after the release, the standing geometry goes through the same attention split once, with the
attention it was observed with — the apple becomes the anchor, the neighbour stays out of the
exclusion; a single apple is untouched; nothing happens without a release, twice, or when off.

S3 — **hidden by the hand does not count toward a switch.** With the neighbour split out as its own
clean cluster, a pear/banana piece led for 3 chunks while the hand hid the apple, and the T26
switch to it came ~100 steps early on 4 E3 runs. Pinned here: while the held object is unobserved
(or seen only as a subset) *and* a hand sphere is within `clustering.hand_occlusion_reach` of it, a
challenger's frame does not count; a visible held object, a hand far away, or no hand information
switch exactly as before — so a genuine change of object still happens on schedule.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.admissibility import HandSpheres, hand_links, hand_spheres
from AG3S.ag3s.stages.target_grounding import (
    SUBSET_KEPT_ANCHOR,
    TargetConfirm,
    ground_target,
)
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

CFG = ClusteringConfig()                       # split 15 mm, reach from config
OPENING = 0.07146                              # RB-Y1 (T26)
APPLE_C = np.array([0.50, 0.00, 0.80])
APPLE_R = 0.031
GAP = 0.020                                    # apple–orange surface gap: merged at eps 30 mm
ORANGE_R = 0.030
ORANGE_C = APPLE_C + np.array([0.0, APPLE_R + GAP + ORANGE_R, 0.0])
PEAR_C = APPLE_C + np.array([0.0, -0.150, 0.0])  # far outside the 60 mm identity tolerance
REACH = CFG.hand_occlusion_reach


def _sphere(centre, radius, n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


APPLE = _sphere(APPLE_C, APPLE_R, seed=0)
ORANGE = _sphere(ORANGE_C, ORANGE_R, seed=1)
PEAR = _sphere(PEAR_C, 0.030, seed=2)
APPLE_TOP = APPLE[APPLE[:, 2] > APPLE_C[2]]     # the hand hides the lower half: a subset


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.broadcast_to(np.asarray(a, np.float32), (len(b),)) for b, a in blobs])
    att = att + np.linspace(0.0, 1e-4, len(att), dtype=np.float32)   # not flat
    return AttentionPointCloud(PointCloud(pts), att, att)


def _ground(apc, confirm, cfg=CFG):
    return ground_target(apc, cfg, seed_threshold=0.3, confirm=confirm)


def _confirm(cfg=CFG, **kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("overlap_distance", 0.005)
    return TargetConfirm(cfg, **kw)


def _hand(centre, radius=0.02):
    """One palm-sized sphere (the tests only need a distance)."""
    return HandSpheres(np.asarray(centre, float).reshape(1, 3), np.array([radius]), ("ee_left",))


HAND_ON_APPLE = _hand(APPLE_C + [0.0, 0.0, APPLE_R + 0.02 + 0.010])   # 10 mm above the apple
HAND_AWAY = _hand(APPLE_C + [0.0, 0.0, 0.40])                          # 35 cm above it


def _is(points, blob):
    return len(points) == len(blob) and np.allclose(np.sort(points, axis=0), np.sort(blob, axis=0))


# ============================================================================== S2 — release split


def _grasp_then_release(confirm, cfg=CFG, *, frozen_frames=2):
    """t=0 the merge is split and the apple adopted; a grasp attempt (frozen) observes the merge
    whole; then the freeze is released and the next frame is grounded."""
    merge = _apc((APPLE, 0.9), (ORANGE, 0.001))
    _ground(merge, confirm, cfg)
    assert _is(confirm.manipulated.points, APPLE)                     # T31b: the split apple
    confirm.freeze(True)
    for _ in range(frozen_frames):
        _ground(merge, confirm, cfg)
    frozen_geometry = confirm.manipulated.points
    confirm.freeze(False)
    g = _ground(merge, confirm, cfg)
    return g, frozen_geometry


def test_the_grasp_attempt_leaves_a_merged_anchor():
    """The failure T31b left: frozen → no split → the standing geometry is apple + orange."""
    off = dataclasses.replace(CFG, split_anchor_on_release=False)
    confirm = _confirm(off)
    _, frozen = _grasp_then_release(confirm, off)
    assert len(frozen) == len(APPLE) + len(ORANGE)
    assert confirm.manipulated.record()["anchor"]["source"] == "observed"


def test_without_the_release_split_the_merge_stays_the_anchor():
    """T31b behaviour (off): the split apple is a subset of the merged anchor, which is kept."""
    off = dataclasses.replace(CFG, split_anchor_on_release=False)
    confirm = _confirm(off)
    _grasp_then_release(confirm, off)
    assert confirm.last.reason == SUBSET_KEPT_ANCHOR
    assert len(confirm.manipulated.points) == len(APPLE) + len(ORANGE)


def test_the_release_splits_the_anchor_once_and_the_apple_is_the_anchor():
    confirm = _confirm()
    g, _ = _grasp_then_release(confirm)
    man = confirm.manipulated
    assert _is(man.points, APPLE)
    rec = man.record()
    assert rec["anchor"]["source"] in ("release_split", "refined")
    assert rec["anchor"]["from_split"] is True
    ev = rec["split"]["release"]
    assert ev["replaced"] is True and ev["skipped"] is None
    assert ev["decision"]["accepted"] is True and ev["decision"]["reason"] == "split"
    assert ev["n_points"] == [len(APPLE) + len(ORANGE), len(APPLE)]
    json.dumps(rec)
    # the orange is out of the exclusion ball again (`_target_ball`: max radius + 1 voxel)
    c = man.points.mean(axis=0)
    r = float(np.linalg.norm(man.points - c, axis=1).max()) + 0.005
    assert np.linalg.norm(ORANGE_C - c) > r
    # the next frame: no second release split
    _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
    assert "release" not in (confirm.manipulated.record()["split"] or {})
    assert _is(confirm.manipulated.points, APPLE)


def test_the_release_split_anchor_is_protected_like_a_split_anchor():
    """It carries `attention_split = 1`, so a later unsplittable merge keeps it (T31b)."""
    confirm = _confirm()
    merge = _apc((APPLE, 0.9), (ORANGE, 0.001))
    _ground(merge, confirm)
    confirm.freeze(True)
    _ground(merge, confirm)
    confirm.freeze(False)
    # the first frame after the release sees nothing (no seed): the split is still done once
    g = ground_target(_apc((PEAR, 0.1)), CFG, seed_threshold=0.99, confirm=confirm)
    assert g.target is None
    man = confirm.manipulated
    assert _is(man.points, APPLE) and man.record()["anchor"]["source"] == "release_split"
    assert man.geometry.metrics["attention_split"] == 1.0
    assert man.geometry.metrics["attention_split_on_release"] == 1.0
    assert man.record()["anchor"]["release_split"]["replaced"] is True


def test_attention_that_does_not_separate_them_leaves_the_anchor():
    confirm = _confirm()
    both = _apc((APPLE, 0.9), (ORANGE, 0.8))
    _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
    confirm.freeze(True)
    _ground(both, confirm)
    confirm.freeze(False)
    _ground(both, confirm)
    ev = confirm.manipulated.record()["split"]["release"]
    assert ev["replaced"] is False and ev["decision"]["reason"] == "attention_not_discriminative"


def test_a_single_apple_is_untouched_by_the_release():
    """ep1807 / T14 / T17: one apple through a grasp attempt and a release — same geometry as with
    the release split off, frame by frame."""
    rng = np.random.default_rng(7)
    frames = [APPLE + rng.normal(scale=0.0005, size=APPLE.shape) for _ in range(5)]
    on, off = _confirm(), _confirm(dataclasses.replace(CFG, split_anchor_on_release=False))
    for k, pts in enumerate(frames):
        for c in (on, off):
            c.freeze(k in (1, 2))
            _ground(_apc((pts, 0.9)), c)
        np.testing.assert_array_equal(on.manipulated.points, off.manipulated.points)
    ev = on.manipulated.record()
    assert ev["anchor"]["release_split"] is None


def test_refreezing_before_the_next_call_disarms_the_release():
    confirm = _confirm()
    merge = _apc((APPLE, 0.9), (ORANGE, 0.001))
    _ground(merge, confirm)
    confirm.freeze(True)
    _ground(merge, confirm)
    confirm.freeze(False)
    confirm.freeze(True)                                   # still grasping after all
    _ground(merge, confirm)
    assert len(confirm.manipulated.points) == len(APPLE) + len(ORANGE)
    assert "release" not in (confirm.manipulated.record()["split"] or {})


def test_no_release_without_a_freeze():
    confirm = _confirm()
    for _ in range(3):
        confirm.freeze(False)
        _ground(_apc((APPLE, 0.9), (ORANGE, 0.001)), confirm)
        assert "release" not in (confirm.manipulated.record()["split"] or {})


def test_release_split_off_by_config_and_by_argument():
    assert _confirm(dataclasses.replace(CFG, split_anchor_on_release=False)).split_on_release is False
    assert _confirm(split_on_release=False).split_on_release is False
    assert _confirm().split_on_release is True


def test_release_split_follows_the_split_switch():
    """`attention_split_radius: null` (split off) also means no release split."""
    off = dataclasses.replace(CFG, attention_split_radius=None)
    confirm = _confirm(off)
    merge = _apc((APPLE, 0.9), (ORANGE, 0.001))
    _ground(merge, confirm, off)
    confirm.freeze(True)
    _ground(merge, confirm, off)
    confirm.freeze(False)
    _ground(merge, confirm, off)
    ev = confirm.manipulated.record()["split"]["release"]
    assert ev["replaced"] is False and ev["skipped"] == "off"


# ============================================================================= S3 — hand occlusion


def _adopt_apple(cfg=CFG, **kw):
    confirm = _confirm(cfg, **kw)
    _ground(_apc((APPLE, 0.9)), confirm, cfg)
    assert confirm.manipulated.id == 0
    return confirm


def _run(confirm, frames, hands, cfg=CFG):
    out = []
    for apc, hand in zip(frames, hands):
        confirm.note_hand(hand)
        _ground(apc, confirm, cfg)
        d = confirm.last
        out.append((d.mode, d.streak, d.hand_occluded, confirm.manipulated.id))
    return out


def test_the_pear_takes_over_on_frame_3_without_hand_information():
    """T31b behaviour: the apple is hidden, the pear leads at score ≥ 0.1 — switch on frame 3."""
    confirm = _adopt_apple()
    modes = _run(confirm, [_apc((PEAR, 0.9))] * 3, [None] * 3)
    assert [m for m, *_ in modes] == ["occluded", "occluded", "switch"]
    assert confirm.manipulated.id == 1


def test_a_hand_on_the_hidden_apple_blocks_the_count():
    confirm = _adopt_apple()
    modes = _run(confirm, [_apc((PEAR, 0.9))] * 6, [HAND_ON_APPLE] * 6)
    assert all(m == "occluded" and s == 0 and h and i == 0 for m, s, h, i in modes)
    rec = confirm.manipulated.record()
    assert rec["hand"]["occluded_by_hand"] is True
    assert rec["hand"]["gap_mm"] == pytest.approx(10.0, abs=1.5)
    assert rec["hand"]["reach_mm"] == pytest.approx(REACH * 1000.0)
    json.dumps(rec)


def test_a_hand_far_away_does_not_block_a_real_switch():
    confirm = _adopt_apple()
    modes = _run(confirm, [_apc((PEAR, 0.9))] * 3, [HAND_AWAY] * 3)
    assert [m for m, *_ in modes] == ["occluded", "occluded", "switch"]
    assert not any(h for _, _, h, _ in modes)


def test_after_the_hand_leaves_the_count_starts_over():
    """3 frames with the hand on the apple, then it moves to the pear side: 3 more frames to switch
    — the frames under the hand did not count."""
    confirm = _adopt_apple()
    modes = _run(confirm, [_apc((PEAR, 0.9))] * 6, [HAND_ON_APPLE] * 3 + [HAND_AWAY] * 3)
    assert [m for m, *_ in modes] == ["occluded"] * 5 + ["switch"]
    assert [s for _, s, _, _ in modes][3:5] == [1, 2]


def test_a_subset_view_under_the_hand_blocks_the_count():
    """The apple half-hidden by the fingers (`subset_kept_anchor`), the pear leading."""
    confirm = _adopt_apple()
    frame = _apc((APPLE_TOP, 0.35), (PEAR, 0.95))
    modes = _run(confirm, [frame] * 5, [HAND_ON_APPLE] * 5)
    assert confirm.last.reason == SUBSET_KEPT_ANCHOR
    assert all(m == "hold" and s == 0 and h and i == 0 for m, s, h, i in modes)
    # the same frames without the hand: switch on frame 3
    confirm = _adopt_apple()
    modes = _run(confirm, [frame] * 3, [None] * 3)
    assert modes[-1][0] == "switch"


def test_a_visible_apple_does_not_block_even_with_the_hand_on_it():
    """Only a *hidden* object is protected: the whole apple seen and the pear leading — the T26
    rule runs unchanged (the policy looks at the pear while the hand is still at the apple)."""
    confirm = _adopt_apple()
    frame = _apc((APPLE, 0.35), (PEAR, 0.95))
    modes = _run(confirm, [frame] * 3, [HAND_ON_APPLE] * 3)
    assert [m for m, *_ in modes] == ["hold", "hold", "switch"]
    assert not any(h for _, _, h, _ in modes)


def test_a_hand_at_the_challenger_counts_even_within_reach_of_the_apple():
    """E3 T30 ep1800 r3 t=432: the hand 20 mm from the (partly hidden) apple but 14 mm from the
    banana beside it, which it then grasped — the policy is changing object: the frame counts."""
    near_pear = APPLE_C + [0.0, -0.058, 0.0]              # between them, nearer the pear side
    pear_close = _sphere(APPLE_C + [0.0, -0.110, 0.0], 0.030, seed=2)
    confirm = _adopt_apple()
    hand = _hand(near_pear, radius=0.004)
    gap_apple = hand.gap(APPLE)
    assert gap_apple <= REACH and hand.gap(pear_close) < gap_apple
    modes = _run(confirm, [_apc((pear_close, 0.9))] * 3, [hand] * 3)
    assert [m for m, *_ in modes] == ["occluded", "occluded", "switch"]
    rec = confirm.manipulated.record()
    assert rec["id"] == 1


def test_the_hand_record_has_the_challenger_gap():
    confirm = _adopt_apple()
    confirm.note_hand(HAND_ON_APPLE)
    _ground(_apc((PEAR, 0.9)), confirm)
    rec = confirm.manipulated.record()["hand"]
    assert rec["occluded_by_hand"] is True
    assert rec["challenger_gap_mm"] > rec["gap_mm"]


def test_the_hand_is_consumed_by_one_call():
    confirm = _adopt_apple()
    confirm.note_hand(HAND_ON_APPLE)
    _ground(_apc((PEAR, 0.9)), confirm)
    assert confirm.last.hand_occluded
    _ground(_apc((PEAR, 0.9)), confirm)                     # no hand given this call
    assert not confirm.last.hand_occluded and confirm.last.hand_gap_m is None
    assert confirm.manipulated.record()["hand"] is None


def test_reach_off_is_the_t31b_behaviour():
    off = dataclasses.replace(CFG, hand_occlusion_reach=None)
    confirm = _adopt_apple(off)
    modes = _run(confirm, [_apc((PEAR, 0.9))] * 3, [HAND_ON_APPLE] * 3, off)
    assert modes[-1][0] == "switch"
    # the gap is still recorded
    assert confirm.last.hand_gap_m == pytest.approx(0.010, abs=0.0015)


def test_the_reach_is_a_distance_threshold():
    just_in = _hand(APPLE_C + [0.0, 0.0, APPLE_R + 0.02 + REACH - 0.002])
    just_out = _hand(APPLE_C + [0.0, 0.0, APPLE_R + 0.02 + REACH + 0.004])
    for hand, blocked in ((just_in, True), (just_out, False)):
        confirm = _adopt_apple()
        modes = _run(confirm, [_apc((PEAR, 0.9))] * 3, [hand] * 3)
        assert (modes[-1][0] == "occluded") is blocked


def test_frozen_is_unchanged():
    confirm = _adopt_apple()
    confirm.freeze(True)
    modes = _run(confirm, [_apc((PEAR, 0.9))] * 4, [HAND_AWAY] * 4)
    assert all(m == "occluded" and s == 0 and not h for m, s, h, _ in modes)


def test_grounding_metrics_carry_the_hand():
    confirm = _adopt_apple()
    confirm.note_hand(HAND_ON_APPLE)
    g = _ground(_apc((PEAR, 0.9)), confirm)
    assert g.target.metrics["manipulated_hand_occluded"] == 1.0
    assert g.target.metrics["manipulated_hand_gap_m"] == pytest.approx(0.010, abs=0.0015)


def test_hand_spheres_gap_is_sphere_surface_to_nearest_point():
    h = HandSpheres(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([0.01, 0.05]))
    assert h.gap(np.array([[0.0, 0.0, 0.03], [1.0, 0.0, 0.2]])) == pytest.approx(0.02)
    assert h.gap(np.zeros((0, 3))) is None
    with pytest.raises(ValueError):
        HandSpheres(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(TypeError):
        TargetConfirm(CFG).note_hand(np.zeros((1, 4)))


def test_config_defaults_and_validation():
    c = AG3SConfig().clustering
    assert c.split_anchor_on_release is True
    assert c.hand_occlusion_reach is not None and c.hand_occlusion_reach > 0.0
    for bad in ({"hand_occlusion_reach": -0.01}, {"hand_occlusion_reach": float("nan")},
                {"hand_occlusion_reach": True}, {"split_anchor_on_release": 1}):
        with pytest.raises(AG3SConfigError):
            dataclasses.replace(c, **bad).validate()
    dataclasses.replace(c, hand_occlusion_reach=None).validate()


MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'))


def test_rby1_hand_spheres_are_the_palms_and_fingers():
    """`hand_spheres` finds the grippers structurally: RB-Y1 → `ee_left`/`ee_right` and their four
    finger links, with the self-filter's FK."""
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    robot = build_robot_model(TransportScene.attach(model, mujoco.MjData(model)))
    links = hand_links(robot)
    assert {"ee_left", "ee_right"} <= set(links) and len(links) == 6
    q = np.zeros(robot.nq)
    h = hand_spheres(robot, q)
    assert h is not None and set(h.links) == set(links)
    c, r = robot.sphere_centers_numeric(q)
    keep = np.array([n in set(links) for n in robot.sphere_link_names])
    np.testing.assert_allclose(h.centres, c[keep])
    np.testing.assert_allclose(h.radii, r[keep])
    assert hand_spheres(robot, None) is None and hand_spheres(None, q) is None
