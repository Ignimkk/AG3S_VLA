"""T5e — the target is the **rank-1 cluster**, and a challenger needs `confirm_frames` to take over.

Two changes are under test and they are separable, so they are tested separately.

1. **The score no longer rejects.** `target_score_threshold` defaults to 0.0, so a frame with a
   cluster has a target. The measurement behind it (T5d): over all 51 chunks of ep1807, reading only
   the ranking gives zero empty chunks, while the old 0.25 gate emptied 35 of them — and because the
   fine ESDF window is centred on the grounded target, those 35 also had no fine layer at all (T5c).

2. **A different object has to earn the target.** Frame-independent ranking flickered on ep1807: pear
   led for 2 chunks and apple for 1 inside an otherwise stable crate run. Three consecutive frames of
   leading flattens both; two does not flatten the pear. `GraspLatch` picked 3 for the same reason.

The identity test is the one that guards the downstream contract: `trajopt/safe_policy.py` names the
target by feeding `target.centroid` to `CentroidIdentity`, so a target that hops between objects
renames it, and `GraspLatch` sees a new object. Hysteresis has to make that name *more* stable, never
less — and the name must still be `obj0`-shaped, not something new.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.config import ClusteringConfig
from benchmark.ag3s.stages.target_grounding import (
    ClusterInfo,
    TargetConfirm,
    ground_target,
)
from benchmark.ag3s.types import AttentionPointCloud, GroundingStatus, PointCloud
from benchmark.trajopt.grasp_latch import CentroidIdentity

CFG = ClusteringConfig(min_points=10)

#: Two objects half a metre apart — far beyond `eps`, so connectivity can never merge them, and far
#: beyond `target_identity_tolerance`, so they can never be read as the same object.
A = np.array([0.50, 0.00, 0.80])
B = np.array([0.50, 0.50, 0.80])


def _lattice(centre: np.ndarray, side: int = 3, spacing: float = 0.008) -> np.ndarray:
    """A `side`-cubed block of points around `centre`, every point within `eps` of a neighbour."""
    g = (np.arange(side) - (side - 1) / 2.0) * spacing
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return np.asarray(centre, np.float64) + grid


def _frame(*attention_per_object: float, centres=(A, B), side: int = 3,
           spacing: float = 0.008) -> AttentionPointCloud:
    """One frame holding `len(centres)` blobs, each at its own attention level.

    `seed_threshold` below the lowest level makes every point a seed, so every blob clusters and the
    ranking — not the seeding — is what decides. The level is therefore the whole story: the highest
    one carries the raw peak, so it wins both terms of the score.

    The 1e-3 ramp inside each blob only keeps the map from being *exactly* flat, which grounding
    rightly refuses to read at all (`NO_ATTENTION`); it is three orders below the gap between levels,
    so it cannot reorder anything.
    """
    blobs = [_lattice(c, side, spacing) for c in centres]
    attention = np.concatenate(
        [np.float32(a) + np.linspace(0.0, 1e-3, len(b), dtype=np.float32)
         for a, b in zip(attention_per_object, blobs)]
    )
    return AttentionPointCloud(PointCloud(np.concatenate(blobs)), attention, attention)


def _spread_frame() -> AttentionPointCloud:
    """A frame whose best cluster scores **under the retired 0.25 gate**.

    Nothing exotic makes that happen: the score is 0.7 x mean attention + 0.3 x spatial consistency,
    and spatial consistency falls off with the cluster's RMS radius. A 6 cm blob — an object seen
    edge-on, a half-occluded one, any cluster loose enough to have a 3 cm RMS radius — halves the
    geometry term, and weak attention cannot make up the difference. This is the shape of the 35
    ep1807 chunks the gate was emptying.
    """
    return _frame(0.02, 0.01, side=7, spacing=0.01)


def _ground(apc: AttentionPointCloud, confirm=None, config: ClusteringConfig | None = None):
    return ground_target(apc, config or CFG, seed_threshold=0.005, confirm=confirm)


def _cluster(centroid: np.ndarray, score: float, index: int) -> ClusterInfo:
    """A candidate reduced to what `TargetConfirm` actually reads: a centroid and a score."""
    return ClusterInfo(
        id=index,
        point_indices=np.array([index], np.int64),
        centroid=np.asarray(centroid, np.float64),
        mean_attention=score,
        max_attention=score,
        attention_mass=score,
        point_count=27,
        spatial_compactness=1.0,
        distance_from_attention_peak=0.0,
        target_score=score,
    )


def _ranked(*pairs) -> list[ClusterInfo]:
    """Clusters sorted by descending score, the order `select` is specified against."""
    ordered = sorted(enumerate(pairs), key=lambda p: -p[1][1])
    return [_cluster(c, s, i) for i, (c, s) in ordered]


# ------------------------------------------------------------------ 1. the score cannot reject


def test_rank_one_is_the_target_even_far_below_the_old_threshold():
    """The whole point of T5e: a low score is a low score, not an absent object."""
    result = _ground(_spread_frame())
    assert result.status is GroundingStatus.OK
    assert result.target is not None
    assert result.best_score < 0.25, "fixture must sit below the retired 0.25 gate to mean anything"
    assert result.target.confidence == pytest.approx(result.best_score)
    assert result.target.centroid == pytest.approx(A, abs=1e-9)


def test_the_threshold_still_works_when_a_caller_sets_it():
    """Not deleted, just off. The judgement it encodes stays reversible."""
    apc = _spread_frame()
    assert _ground(apc).best_score < 0.25
    gated = _ground(apc, config=dataclasses.replace(CFG, target_score_threshold=0.25))
    assert gated.target is None and gated.status is GroundingStatus.LOW_SCORE
    assert gated.best_score > 0.0, "the clusters are still reported, so tuning is still diagnosable"


def test_no_cluster_is_still_no_target():
    """The one honest way to have no target. `min_points` nothing can reach reaches nothing."""
    result = _ground(_frame(0.9, 0.3), config=dataclasses.replace(CFG, min_points=10_000))
    assert result.target is None and result.status is GroundingStatus.NO_CLUSTER


def test_an_empty_cloud_is_still_no_geometry():
    result = ground_target(
        AttentionPointCloud(PointCloud.empty(), np.zeros(0, np.float32)),
        CFG, confirm=TargetConfirm(CFG),
    )
    assert result.target is None and result.status is GroundingStatus.NO_GEOMETRY


def test_select_refuses_to_invent_a_target_from_nothing():
    """`TargetConfirm` never returns "no target" — it is only ever asked when a cluster exists."""
    with pytest.raises(ValueError):
        TargetConfirm(CFG).select([])


# ------------------------------------------------------------ 2. three consecutive frames to switch


def test_two_frames_of_a_new_leader_do_not_move_the_target():
    """The ep1807 pear: it led 2 chunks. Two is not three, so the target stays put."""
    confirm = TargetConfirm(CFG)
    assert _ground(_frame(0.9, 0.3), confirm).target.centroid == pytest.approx(A, abs=1e-9)
    for frame in (1, 2):
        held = _ground(_frame(0.3, 0.9), confirm).target
        assert held.centroid == pytest.approx(A, abs=1e-9), f"switched after {frame} frame(s)"
        assert held.metrics["target_rank"] == 1.0, "the held object was not the leader"
        assert held.metrics["target_confirm_streak"] == float(frame)


def test_three_frames_of_a_new_leader_move_it():
    confirm = TargetConfirm(CFG)
    _ground(_frame(0.9, 0.3), confirm)
    centroids = [_ground(_frame(0.3, 0.9), confirm).target.centroid for _ in range(3)]
    assert centroids[0] == pytest.approx(A, abs=1e-9)
    assert centroids[1] == pytest.approx(A, abs=1e-9)
    assert centroids[2] == pytest.approx(B, abs=1e-9), "3 consecutive frames must take the target"
    assert confirm.last.mode == "switch" and confirm.last.rank == 0


def test_a_broken_run_starts_the_count_over():
    """Why counting consecutively matters: 2 + 2 is not 3.

    A challenger that leads twice, loses the lead once, then leads twice again has led four frames
    and still must not own the target. Without the reset, alternating attention would switch the
    target on the frame the *old* object was leading.
    """
    confirm = TargetConfirm(CFG)
    _ground(_frame(0.9, 0.3), confirm)
    for pattern in (0.3, 0.3, 0.9, 0.3, 0.3):
        target = _ground(_frame(pattern, 1.2 - pattern), confirm).target
    assert target.centroid == pytest.approx(A, abs=1e-9)
    assert confirm.streak == 2


def test_the_first_frame_of_an_episode_takes_the_leader_with_no_argument():
    """Nothing is held yet, so there is nothing to defend the target against.

    Requiring 3 frames here would mean no target at all for the first two frames of every episode —
    and on `run_0002` the apple appears *only* in the leading frames.
    """
    confirm = TargetConfirm(CFG)
    first = _ground(_frame(0.3, 0.9), confirm).target
    assert first.centroid == pytest.approx(B, abs=1e-9)
    assert confirm.last.mode == "first" and confirm.last.streak == 0


def test_reset_is_the_episode_boundary():
    confirm = TargetConfirm(CFG)
    _ground(_frame(0.9, 0.3), confirm)
    confirm.reset()
    assert confirm.held_centroid is None and confirm.last is None
    # A new episode: whatever leads its first frame is the target immediately.
    assert _ground(_frame(0.3, 0.9), confirm).target.centroid == pytest.approx(B, abs=1e-9)


def test_confirm_frames_one_is_the_old_frame_independent_behaviour():
    """The escape hatch, and the proof that the hysteresis is the only thing holding the target."""
    confirm = TargetConfirm(dataclasses.replace(CFG, target_confirm_frames=1))
    _ground(_frame(0.9, 0.3), confirm)
    assert _ground(_frame(0.3, 0.9), confirm).target.centroid == pytest.approx(B, abs=1e-9)


def test_a_held_object_that_is_not_in_the_frame_is_occluded_not_replaced():
    """T20 turned this test around. It was `..._cannot_be_held`: the leader was used at once.

    That was the T14 seq 12 failure — the apple went behind the fingers, the crate led at score
    0.015, and the crate became the target for fine window, exclusion and contact permission. Now
    the held object is returned as `occluded` with its **last observed** geometry. The switch count
    still runs, so an object that is simply gone hands over on schedule (third frame) rather than
    at once — the part of the old test that still holds.
    """
    confirm = TargetConfirm(CFG)
    _ground(_frame(0.9, 0.3), confirm)
    result = _ground(_frame(0.9, centres=(B,)), confirm)  # only B survives reconstruction
    assert result.target.centroid == pytest.approx(A, abs=1e-9)
    assert confirm.last.mode == "occluded" and confirm.last.streak == 1
    assert confirm.last.cluster is None
    _ground(_frame(0.9, centres=(B,)), confirm)
    assert confirm.last.mode == "occluded" and confirm.last.streak == 2
    switched = _ground(_frame(0.9, centres=(B,)), confirm)
    assert confirm.last.mode == "switch"
    assert switched.target.centroid == pytest.approx(B, abs=1e-9)


def test_the_held_centroid_follows_its_object_as_it_moves():
    """A grasped object travels, and identity has to travel with it.

    A leads while moving 20 mm per frame — well inside the 60 mm tolerance per step, but 120 mm in
    total, twice the tolerance. Then B takes the lead for one frame. The hold can only happen if the
    held centroid tracked A the whole way; pinned to where A started, it would match nothing within
    tolerance and the frame would read `unobserved` and hand the target to B.
    """
    confirm = TargetConfirm(CFG)
    for step in range(7):
        moved = A + np.array([0.0, 0.0, 0.02 * step])
        target = _ground(_frame(0.9, 0.3, centres=(moved, B)), confirm).target
        assert target.centroid == pytest.approx(moved, abs=1e-9)
    assert float(np.linalg.norm(target.centroid - A)) > 2 * confirm.tolerance

    challenged = _ground(_frame(0.3, 0.9, centres=(moved, B)), confirm).target
    assert confirm.last.mode == "hold"
    assert challenged.centroid == pytest.approx(moved, abs=1e-9)


# ------------------------------------------------------------------- 3. what the latch downstream reads


def test_the_label_identity_the_latch_reads_stays_obj0_through_a_flicker():
    """`safe_policy._run_latch` names the target with `CentroidIdentity`. That name must not flicker.

    Without hysteresis the two-frame challenge renames the target `obj1` and back, and `GraspLatch`
    counts it as a different object — the streak it needs for a lock resets. With hysteresis the name
    is `obj0` on every frame of the flicker, which is the whole point.
    """
    identity, confirm = CentroidIdentity(tolerance=0.06), TargetConfirm(CFG)
    levels = [(0.9, 0.3), (0.3, 0.9), (0.3, 0.9), (0.9, 0.3), (0.9, 0.3)]
    labels = [identity.label(_ground(_frame(*lv), confirm).target.centroid) for lv in levels]
    assert labels == ["obj0"] * 5

    bare = CentroidIdentity(tolerance=0.06)
    flickered = [bare.label(_ground(_frame(*lv)).target.centroid) for lv in levels]
    assert flickered == ["obj0", "obj1", "obj1", "obj0", "obj0"], "control: this is what we removed"


def test_runner_up_score_is_the_strongest_other_candidate():
    """The key `GraspLatch` divides by. Unchanged when rank 1 wins; the leader when it is overruled.

    Reporting the *selected* cluster's own score as its runner-up would make the ratio exactly 1 and
    the ambiguity test meaningless, and reporting `clusters[1]` blindly would do exactly that on a
    held frame — `clusters[1]` *is* the target there.
    """
    confirm = TargetConfirm(CFG)
    lead = _ground(_frame(0.9, 0.3), confirm).target
    assert lead.metrics["runner_up_score"] < lead.confidence

    held = _ground(_frame(0.3, 0.9), confirm).target
    assert held.metrics["target_rank"] == 1.0
    assert held.metrics["runner_up_score"] > held.confidence, "an overruled leader is the competitor"

    alone = _ground(_frame(0.9, centres=(A,)), TargetConfirm(CFG)).target
    assert alone.metrics["runner_up_score"] == 0.0, "one candidate has no runner-up"


def test_without_a_confirm_object_grounding_is_still_frame_independent():
    """Every direct caller — reports, diagrams, ablations — must keep getting a pure function."""
    frames = [_frame(0.9, 0.3), _frame(0.3, 0.9), _frame(0.3, 0.9)]
    centroids = [_ground(f).target.centroid for f in frames]
    assert centroids[0] == pytest.approx(A, abs=1e-9)
    assert centroids[1] == pytest.approx(B, abs=1e-9)
    for f in frames:
        target = _ground(f).target
        assert target.metrics["target_rank"] == 0.0
        assert target.metrics["target_confirm_streak"] == 0.0


# ----------------------------------------------------------------------------- selection mechanics


def test_select_reads_only_the_ranking_and_the_centroids():
    """A unit test on the rule itself, with no point cloud in the way.

    The scores here would have failed the retired gate on every frame; the decision must not depend
    on their level at all.
    """
    confirm = TargetConfirm(CFG)
    assert confirm.select(_ranked((A, 0.02), (B, 0.01))).mode == "first"
    first = confirm.select(_ranked((A, 0.01), (B, 0.02)))
    assert first.mode == "hold" and first.cluster.centroid == pytest.approx(A)
    assert confirm.select(_ranked((A, 0.01), (B, 0.02))).mode == "hold"
    third = confirm.select(_ranked((A, 0.01), (B, 0.02)))
    assert third.mode == "switch" and third.cluster.centroid == pytest.approx(B)


def test_the_nearer_cluster_is_the_same_object_when_two_are_within_tolerance():
    """Identity is a distance question, and `clusters` is ordered by score, which is unrelated."""
    confirm = TargetConfirm(CFG)
    confirm.select(_ranked((A, 0.5),))
    near = A + np.array([0.01, 0.0, 0.0])
    far = A + np.array([0.05, 0.0, 0.0])
    # The far one leads, and both are inside the 60 mm tolerance. The near one is the held object.
    held = confirm.select(_ranked((near, 0.1), (far, 0.9)))
    assert held.mode == "hold" and held.cluster.centroid == pytest.approx(near)
