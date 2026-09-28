"""T20 — the manipulated object has an identity, and being hidden is not being replaced.

T14 seq 12: the apple went behind the fingers, the crate led grounding at score 0.015, and
`TargetConfirm.select()` handed the frame to the crate (`unobserved`). Fine window, target-free
exclusion, `manipulated_link_margin`, `target_field_exclude` and the T19 guard all followed it.

What is pinned here:

1. `select()` — an unobserved held object is `occluded` (same id, last geometry), never the leader.
   The switch count keeps running; a challenger switches only after `target_confirm_frames` frames
   running at `target_switch_min_score` or above. After `target_lost_frames` it is `lost` — no
   target, `GroundingStatus.LOST`, and never the leader either.
2. Defaults: `target_switch_min_score = 0.0`, `target_lost_frames = None` (never lost). The switch
   schedule is then the T5e one exactly; only the unobserved frames' *answer* changed.
3. Consumers read the manipulated object: `build_constraint_set(manipulated_geometry=...)` for
   contact permission and target exclusion; `AG3S` passes it to `_build_esdf` and the guard.
4. Record: `metrics["manipulated"]` and the chunk summary's `manipulated`.
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from benchmark.ag3s.runtime.pipeline import AG3S
from benchmark.ag3s.stages.target_grounding import (
    LOST,
    OCCLUDED,
    VISIBLE,
    ClusterInfo,
    TargetConfirm,
    ground_target,
)
from benchmark.ag3s.types import AttentionPointCloud, GroundingStatus, PointCloud
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    DEFAULT_OBJECTS,
    LABEL_BOX,
    LABEL_SPHERE,
    PlanarTwoLinkArm,
    make_scene,
)

CFG = ClusteringConfig(min_points=10)
APPLE = np.array([0.50, 0.00, 0.80])
CRATE = np.array([0.50, 0.50, 0.80])
BANANA = np.array([0.50, -0.50, 0.80])


# ----------------------------------------------------------------------------- helpers


def _cluster(centroid, score: float, index: int) -> ClusterInfo:
    return ClusterInfo(
        id=index, point_indices=np.array([index], np.int64),
        centroid=np.asarray(centroid, np.float64), mean_attention=score, max_attention=score,
        attention_mass=score, point_count=27, spatial_compactness=1.0,
        distance_from_attention_peak=0.0, target_score=score)


def _ranked(*pairs) -> list[ClusterInfo]:
    ordered = sorted(enumerate(pairs), key=lambda p: -p[1][1])
    return [_cluster(c, s, i) for i, (c, s) in ordered]


def _lattice(centre, side: int = 3, spacing: float = 0.008) -> np.ndarray:
    g = (np.arange(side) - (side - 1) / 2.0) * spacing
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return np.asarray(centre, np.float64) + grid


def _frame(*pairs) -> AttentionPointCloud:
    """Blobs `(centre, attention_level)`; every point is a seed, so the level alone ranks them."""
    blobs = [_lattice(c) for c, _ in pairs]
    att = np.concatenate([np.float32(a) + np.linspace(0.0, 1e-3, len(b), dtype=np.float32)
                          for (_, a), b in zip(pairs, blobs)])
    return AttentionPointCloud(PointCloud(np.concatenate(blobs)), att, att)


def _ground(apc, confirm, cfg=CFG, ts=0.0):
    return ground_target(apc, cfg, seed_threshold=0.005, confirm=confirm, timestamp=ts)


# ------------------------------------------------------------------ 1. select(): occluded, not replaced


def test_unobserved_returns_no_cluster_and_the_same_id_as_occluded():
    """The T20 contract at the rule level: `cluster is None`, never `clusters[0]`."""
    confirm = TargetConfirm(CFG)
    first = confirm.select(_ranked((APPLE, 0.46), (CRATE, 0.01)))
    assert first.mode == "first" and first.manipulated_id == 0 and first.state == VISIBLE

    crate_only = _ranked((CRATE, 0.015))
    d1 = confirm.select(crate_only)
    assert d1.cluster is None and d1.rank == -1
    assert (d1.mode, d1.state, d1.manipulated_id, d1.age_frames) == ("occluded", OCCLUDED, 0, 1)
    d2 = confirm.select(crate_only)
    assert (d2.mode, d2.age_frames, d2.streak) == ("occluded", 2, 2)
    np.testing.assert_array_equal(confirm.held_centroid, APPLE)


def test_the_default_switch_schedule_is_the_t5e_one():
    """min_score 0.0 filters nothing: an object that stays gone still hands over on frame 3."""
    confirm = TargetConfirm(CFG)
    assert confirm.switch_min_score == 0.0 and confirm.lost_frames is None
    confirm.select(_ranked((APPLE, 0.46),))
    modes = [confirm.select(_ranked((CRATE, 0.015),)).mode for _ in range(3)]
    assert modes == ["occluded", "occluded", "switch"]
    m = confirm.manipulated
    assert (m.id, m.state, m.switched_from, m.age_frames) == (1, VISIBLE, 0, 0)
    np.testing.assert_array_equal(m.centroid, CRATE)


def test_a_challenger_under_the_minimum_never_switches_and_the_apple_stays_occluded():
    """T14 seq 12-21 in miniature: the crate leads at 0.015-0.05 for ten frames."""
    confirm = TargetConfirm(CFG, switch_min_score=0.1)
    confirm.select(_ranked((APPLE, 0.46), (CRATE, 0.01)))
    for k, score in enumerate(np.linspace(0.015, 0.05, 10), start=1):
        d = confirm.select(_ranked((CRATE, float(score)),))
        assert (d.mode, d.cluster, d.manipulated_id, d.age_frames, d.streak) == (
            "occluded", None, 0, k, 0)
    np.testing.assert_array_equal(confirm.held_centroid, APPLE)


def test_a_challenger_at_the_minimum_for_n_frames_does_switch():
    """The identity is not a permanent lock: a real re-selection still goes through."""
    confirm = TargetConfirm(CFG, switch_min_score=0.1)
    confirm.select(_ranked((APPLE, 0.46),))
    modes = [confirm.select(_ranked((CRATE, 0.3),)).mode for _ in range(3)]
    assert modes == ["occluded", "occluded", "switch"]
    assert confirm.manipulated.id == 1 and confirm.manipulated.switched_from == 0


def test_a_sub_minimum_frame_breaks_the_run():
    """"N frames running at the minimum": 2 + gap + 2 is not 3."""
    confirm = TargetConfirm(CFG, switch_min_score=0.1)
    confirm.select(_ranked((APPLE, 0.46),))
    for score in (0.3, 0.3, 0.05, 0.3, 0.3):
        d = confirm.select(_ranked((CRATE, score),))
    assert d.mode == "occluded" and d.streak == 2
    assert confirm.select(_ranked((CRATE, 0.3),)).mode == "switch"


def test_the_minimum_also_guards_a_visible_held_object():
    """While the apple is visible and outranked, a sub-minimum leader never counts either."""
    confirm = TargetConfirm(CFG, switch_min_score=0.5)
    confirm.select(_ranked((APPLE, 0.46),))
    for _ in range(5):
        d = confirm.select(_ranked((APPLE, 0.2), (CRATE, 0.4)))
        assert d.mode == "hold" and d.streak == 0 and d.manipulated_id == 0


def test_the_nearest_rule_re_finds_the_object_when_it_reappears_same_id():
    confirm = TargetConfirm(CFG, switch_min_score=0.1)
    confirm.select(_ranked((APPLE, 0.46),))
    confirm.select(_ranked((CRATE, 0.02),))
    back = confirm.select(_ranked((APPLE + [0.0, 0.0, 0.01], 0.4), (CRATE, 0.02)))
    assert back.mode == "keep" and back.manipulated_id == 0 and back.state == VISIBLE
    assert confirm.manipulated.age_frames == 0


# ------------------------------------------------------------------------------ 2. lost


def test_lost_after_lost_frames_and_then_visible_again_under_the_same_id():
    confirm = TargetConfirm(CFG, switch_min_score=0.1, lost_frames=2)
    confirm.select(_ranked((APPLE, 0.46),))
    states = [confirm.select(_ranked((CRATE, 0.02),)).state for _ in range(4)]
    assert states == [OCCLUDED, OCCLUDED, LOST, LOST]
    assert confirm.last.mode == "lost" and confirm.last.cluster is None
    back = confirm.select(_ranked((APPLE, 0.4), (CRATE, 0.02)))
    assert back.state == VISIBLE and back.manipulated_id == 0


def test_lost_frames_zero_means_lost_on_the_first_unobserved_frame():
    confirm = TargetConfirm(CFG, switch_min_score=0.1, lost_frames=0)
    confirm.select(_ranked((APPLE, 0.46),))
    assert confirm.select(_ranked((CRATE, 0.02),)).state == LOST


def test_lost_grounding_has_no_target_and_says_lost_not_the_leader():
    cfg = dataclasses.replace(CFG, target_switch_min_score=0.99, target_lost_frames=1)
    confirm = TargetConfirm(cfg)
    _ground(_frame((APPLE, 0.9), (CRATE, 0.3)), confirm)
    occluded = _ground(_frame((CRATE, 0.9)), confirm, cfg)
    assert occluded.status is GroundingStatus.OK and occluded.target is not None
    lost = _ground(_frame((CRATE, 0.9)), confirm, cfg)
    assert lost.target is None and lost.status is GroundingStatus.LOST
    assert lost.status is not GroundingStatus.NO_SEED
    assert len(lost.clusters) == 1, "the clusters are still reported"


# -------------------------------------------------------- 3. ground_target: the last geometry stands in


def test_the_occluded_target_is_the_last_observation_not_this_frames_leader():
    confirm = TargetConfirm(CFG)
    seen = _ground(_frame((APPLE, 0.9), (CRATE, 0.3)), confirm, ts=1.0).target
    hidden = _ground(_frame((CRATE, 0.9)), confirm, ts=2.0)
    t = hidden.target
    assert hidden.status is GroundingStatus.OK
    np.testing.assert_array_equal(t.points, seen.points)
    np.testing.assert_array_equal(t.centroid, seen.centroid)
    assert t.point_indices.size == 0, "no point of this frame's cloud is the apple"
    assert t.id == -1 and t.timestamp == 1.0, "when the geometry was true, not now"
    assert t.metrics["manipulated_occluded"] == 1.0
    assert t.metrics["manipulated_age_frames"] == 1.0
    assert t.metrics["target_rank"] == -1.0
    assert t.metrics["runner_up_score"] == pytest.approx(hidden.best_score), \
        "every cluster of this frame competes with a stale object"
    assert t.confidence == seen.confidence
    m = confirm.manipulated
    assert m.state == OCCLUDED and m.last_seen_frame == 0 and m.frame == 1
    assert m.geometry is seen


def test_a_frame_with_no_cluster_ages_the_object_and_keeps_its_status():
    """NO_SEED/NO_CLUSTER stay what they were (certification is T23's); the identity ages."""
    confirm = TargetConfirm(CFG)
    _ground(_frame((APPLE, 0.9), (CRATE, 0.3)), confirm)
    empty = _ground(_frame((CRATE, 0.9)), confirm,
                    dataclasses.replace(CFG, min_points=10_000))
    assert empty.target is None and empty.status is GroundingStatus.NO_CLUSTER
    m = confirm.manipulated
    assert (m.state, m.age_frames, m.id) == (OCCLUDED, 1, 0)
    assert confirm.streak == 0, "a frame with no leader does not touch the switch count"


def test_attention_on_the_basket_with_the_apple_still_clustered_holds_the_apple():
    """Guide §4.1 'transport: attention = basket, manipulated = apple' — two frames of it."""
    confirm = TargetConfirm(CFG)
    _ground(_frame((APPLE, 0.9), (CRATE, 0.3)), confirm)
    for _ in range(2):
        t = _ground(_frame((APPLE, 0.3), (CRATE, 0.9)), confirm).target
        np.testing.assert_allclose(t.centroid, APPLE, atol=1e-9)
        assert confirm.last.mode == "hold" and confirm.manipulated.id == 0


def test_reset_restarts_ids():
    confirm = TargetConfirm(CFG)
    confirm.select(_ranked((APPLE, 0.4),))
    for _ in range(3):
        confirm.select(_ranked((CRATE, 0.4),))
    assert confirm.manipulated.id == 1
    confirm.reset()
    assert confirm.manipulated is None and confirm.state is None
    confirm.select(_ranked((BANANA, 0.4),))
    assert confirm.manipulated.id == 0 and confirm.manipulated.switched_from is None


def test_record_is_json_ready():
    confirm = TargetConfirm(CFG)
    _ground(_frame((APPLE, 0.9)), confirm)
    rec = confirm.manipulated.record()
    json.dumps(rec)
    assert set(rec) == {"id", "state", "age_frames", "centroid", "radius", "n_points",
                        "last_seen_frame", "switched_from", "frame"}
    assert rec["n_points"] == 27 and rec["state"] == "visible" and rec["radius"] > 0.0


# ------------------------------------------------------------------------------- config


def test_config_defaults_and_round_trip():
    cfg = AG3SConfig()
    assert cfg.clustering.target_switch_min_score == 0.0
    assert cfg.clustering.target_lost_frames is None
    over = AG3SConfig.from_dict({"clustering": {"target_switch_min_score": 0.1,
                                                "target_lost_frames": 8}})
    assert AG3SConfig.from_dict(over.to_dict()) == over
    t = TargetConfirm(over.clustering)
    assert (t.switch_min_score, t.lost_frames) == (0.1, 8)


@pytest.mark.parametrize("bad", [{"target_switch_min_score": -0.1},
                                 {"target_switch_min_score": float("nan")},
                                 {"target_lost_frames": -1},
                                 {"target_lost_frames": 1.5},
                                 {"target_lost_frames": True},
                                 {"target_lost_frames": "3"}])
def test_config_rejects_malformed_values(bad):
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": bad})


# --------------------------------------------------------- 4. to_adapter reads the manipulated object


def _adapter(policy="exclude_authorized"):
    from benchmark.ag3s.constraints.clearance import ClearancePolicy
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import TargetGeometry
    from tests.ag3s.fixtures import FINGERTIP_LINK

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig.from_dict({
        "contact": {"contact_links": {"right": [FINGERTIP_LINK]}},
        "constraint": {"horizon": 3, "target_field_policy": policy},
    })
    builder = ConstraintBuilder(
        arm, cfg.constraint, cfg.geometry,
        clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry))
    q = np.array([-0.60, -0.60])
    pts = arm.end_effector_numeric(q) + np.random.default_rng(0).normal(0.0, 0.01, (60, 3))
    apple = TargetGeometry(id=0, points=pts, point_indices=np.arange(60),
                           centroid=pts.mean(axis=0), bounding_geometry=fit_sphere(pts),
                           attention_score=0.9, confidence=0.9)
    crate_pts = pts + np.array([0.0, 0.4, 0.0])
    crate = dataclasses.replace(apple, points=crate_pts, centroid=crate_pts.mean(axis=0))
    field = types.SimpleNamespace(stats={"n_occupied": 10}, has_target_free=True)
    return builder, q, apple, crate, field


def _build(builder, q, target, field, **kw):
    from benchmark.ag3s.constraints.to_adapter import build_constraint_set
    from benchmark.ag3s.types import ContactPolicyContext, Manipulator, Phase

    return build_constraint_set(
        builder, candidates=[], support_surfaces=[], target=target, robot_state=q,
        phase=Phase.GRASP, esdf=field, build_spec=False,
        contact_context=ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT]), **kw)


def test_not_given_the_manipulated_object_is_the_target_as_before():
    builder, q, apple, _, field = _adapter()
    old = _build(builder, q, apple, field)
    assert old.manipulated is not None and old.target_field_exclude is not None
    np.testing.assert_array_equal(old.manipulated.points, apple.points)


def test_permission_and_exclusion_follow_the_manipulated_object_not_the_target():
    builder, q, apple, crate, field = _adapter()
    cs = _build(builder, q, crate, field, manipulated_geometry=apple)
    np.testing.assert_array_equal(cs.manipulated.points, apple.points)
    assert cs.manipulated_link_margin is not None and cs.target_field_exclude is not None
    assert cs.target is crate, "target itself is untouched"


def test_no_target_but_an_occluded_object_keeps_permission_and_exclusion():
    """A NO_SEED frame while the apple is behind the fingers."""
    builder, q, apple, _, field = _adapter()
    cs = _build(builder, q, None, field, manipulated_geometry=apple)
    assert cs.manipulated is not None
    assert cs.manipulated_link_margin is not None and cs.target_field_exclude is not None
    assert cs.target is None


def test_a_lost_object_grants_nothing_even_with_a_target():
    builder, q, _, crate, field = _adapter()
    cs = _build(builder, q, crate, field, manipulated_geometry=None)
    assert cs.manipulated is None
    assert cs.manipulated_link_margin is None and cs.target_field_exclude is None


# ------------------------------------------------------------------------------ 5. the pipeline


SCENE_KW = dict(phase="approach", robot_state=DEFAULT_ARM_Q)
NO_APPLE = tuple(o for o in DEFAULT_OBJECTS if o.label != LABEL_SPHERE)


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scenes(arm):
    """Frame A: the apple is there and attended. Frame B: the apple is gone, attention on the box."""
    with_apple = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    hidden = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=NO_APPLE,
                        target_label=LABEL_BOX)
    return with_apple, hidden


def _inputs(scene):
    return dict(attention_map=scene.attention_grid, depth=scene.depth,
                camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                image_hw=scene.hw, **SCENE_KW)


def _ag3s(arm, *, esdf_spy=None, **clustering):
    data = {"clustering": clustering} if clustering else {}
    if esdf_spy is not None:
        data["collision_backend"] = "esdf"
    ag3s = AG3S(AG3SConfig.from_dict(data), robot_model=arm)
    if esdf_spy is not None:
        def spy(depth_cameras, target, **kw):
            esdf_spy.append(target)
            return None, []
        ag3s._build_esdf = spy
    return ag3s


def _apple_first(ag3s, scenes):
    cs, debug = ag3s.process_debug(**_inputs(scenes[0]))
    if cs.target is None:
        pytest.skip("grounding declined the apple scene")
    apple_centre = next(o.center for o in DEFAULT_OBJECTS if o.label == LABEL_SPHERE)
    assert np.linalg.norm(cs.target.centroid - apple_centre) < 0.03
    return cs


def test_pipeline_keeps_the_apple_through_an_occlusion_and_records_it(arm, scenes):
    seen = []
    ag3s = _ag3s(arm, esdf_spy=seen)
    cs0 = _apple_first(ag3s, scenes)
    cs1 = ag3s.process(**_inputs(scenes[1]))
    rec = cs1.metrics["manipulated"]
    assert rec["state"] == "occluded" and rec["id"] == 0 and rec["age_frames"] == 1
    assert rec["last_seen_frame"] == 0 and rec["frame"] == 1
    np.testing.assert_allclose(rec["centroid"], cs0.target.centroid)
    np.testing.assert_allclose(cs1.target.centroid, cs0.target.centroid)
    assert ag3s.manipulated.state == OCCLUDED
    # the ESDF consumer (fine window, target-free layer, ball, TARGET_LABEL) got the apple's points
    np.testing.assert_array_equal(seen[1].points, cs0.target.points)
    # contact permission follows it too
    np.testing.assert_array_equal(cs1.manipulated.points, cs0.target.points)
    assert any("target occluded" in n for n in cs1.notes)


def test_pipeline_lost_says_lost_with_id_and_age_and_names_no_one(arm, scenes):
    seen = []
    ag3s = _ag3s(arm, esdf_spy=seen, target_switch_min_score=0.999, target_lost_frames=1)
    _apple_first(ag3s, scenes)
    ag3s.process(**_inputs(scenes[1]))
    cs = ag3s.process(**_inputs(scenes[1]))
    assert cs.target is None and cs.grounding_status is GroundingStatus.LOST
    assert cs.metrics["manipulated"]["state"] == "lost"
    assert any(n.startswith("lost(id=0, age=2)") for n in cs.notes)
    assert cs.manipulated is None and cs.manipulated_link_margin is None
    assert seen[-1] is None, "no geometry reaches the ESDF consumers once lost"


def test_pipeline_switch_under_the_default_is_on_schedule_with_a_new_id(arm, scenes):
    ag3s = _ag3s(arm)
    _apple_first(ag3s, scenes)
    states = [ag3s.process(**_inputs(scenes[1])).metrics["manipulated"] for _ in range(3)]
    assert [s["state"] for s in states] == ["occluded", "occluded", "visible"]
    assert states[-1]["id"] == 1 and states[-1]["switched_from"] == 0


def test_the_guard_centre_is_the_manipulated_centroid_while_occluded(arm, scenes):
    ag3s = AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_target_guard_radius": 0.08}}),
                robot_model=arm)
    cs0 = _apple_first(ag3s, scenes)
    ag3s.process(**_inputs(scenes[1]))  # occluded
    _, debug = ag3s.process_debug(**_inputs(scenes[1]))  # guard fixed from the occluded identity
    assert debug["self_filter_stats"]["guard_centroid"] == pytest.approx(
        cs0.target.centroid.tolist())


def test_reset_clears_the_manipulated_object(arm, scenes):
    ag3s = _ag3s(arm)
    _apple_first(ag3s, scenes)
    assert ag3s.manipulated is not None
    ag3s.reset()
    assert ag3s.manipulated is None


def test_the_chunk_summary_carries_the_manipulated_record(arm, scenes, tmp_path):
    from benchmark.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    ag3s = _ag3s(arm)
    _apple_first(ag3s, scenes)
    cs, debug = ag3s.process_debug(**_inputs(scenes[1]))
    writer = ConstraintRecordWriter(tmp_path, esdf_mode="none")
    path = writer.record(t_step=0, chunk_index=0, constraint_set=cs, debug=debug)
    summary = json.loads(str(np.load(path)["summary_json"]))
    assert summary["manipulated"]["state"] == "occluded" and summary["manipulated"]["id"] == 0
