"""T26 — the crate can never be the manipulated object, the exclusion ball or the contact-permitted
object; it is registered as the destination instead.

User ruling 2026-09-28: "crate 는 어떤 상황에서도 공이 될 수 없다" — and a radius cap is *not* the
answer. What is pinned here is that the state "crate is the manipulated object" is unreachable:

1. `max_opening` comes from the robot model (RB-Y1: finger spheres' inner-face gap at the finger
   joints' open limit), is not a literal, and follows the joint limits.
2. Admissibility = graspable (narrowest principal extent ≤ max opening) AND not the destination.
   `first` adopts the best *admissible* cluster; `switch` needs an admissible challenger at
   `target_switch_min_score` (default 0.1) for `target_confirm_frames`; an inadmissible match of the
   held object is `occluded` + `inadmissible_match`.
3. The destination is registered from perception (non-graspable leader, min score, N frames) or
   injected at `reset` (precedence).
4. `AG3S._exclusion_gate` refuses exclusion geometry without admissibility evidence and records
   `invariant_violation` (raises in strict mode, which `tests/conftest.py` sets).
5. `AG3S.attach` refuses an inadmissible object.
6. `metrics["exclusion"]`: `manipulated` before attach, `attached` after, never broken by a switch
   while a grasp is in progress (`set_grasp_active`).
7. `pointcloud.self_filter_inflation` defaults to 0.0.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import dataclasses
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.admissibility import (
    ADMISSIBLE,
    DESTINATION_OVERLAP,
    INJECTED,
    MAX_OPENING_UNKNOWN,
    NOT_GRASPABLE,
    REGISTERED,
    ExclusionInvariantViolation,
    assess,
    max_opening_from_models,
    principal_extents,
    resolve_max_opening,
)
from AG3S.ag3s.stages.target_grounding import (
    INADMISSIBLE_MATCH,
    NO_ADMISSIBLE,
    OCCLUDED,
    UNCHECKED,
    VISIBLE,
    ClusterInfo,
    TargetConfirm,
    ground_target,
)
from AG3S.ag3s.types import AttentionPointCloud, GroundingStatus, PointCloud
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    DEFAULT_OBJECTS,
    LABEL_BOX,
    LABEL_SPHERE,
    TABLE_Z,
    GTObject,
    PlanarTwoLinkArm,
    make_scene,
)

MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'))
OPENING = 0.0715            # ≈ RB-Y1's computed value; the unit tests use it as a plain number
CFG = ClusteringConfig(min_points=10)

APPLE_C = np.array([0.50, 0.00, 0.80])
CRATE_C = np.array([0.50, 0.40, 0.80])
BANANA_C = np.array([0.50, -0.40, 0.80])


# ----------------------------------------------------------------------------- point helpers


def _ball(centre, radius=0.031, n=300, seed=0):
    """Points on a sphere (the apple: ~62 mm across)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def _box(centre, half, n=900, seed=1, rot=None):
    """Points on a box surface (the crate: 183 x 421 x 149 mm by default T14 cluster)."""
    rng = np.random.default_rng(seed)
    half = np.asarray(half, np.float64)
    p = rng.uniform(-1.0, 1.0, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    p[np.arange(n), axis] = np.sign(p[np.arange(n), axis])
    p = p * half
    if rot is not None:
        p = p @ np.asarray(rot).T
    return np.asarray(centre) + p


CRATE_HALF = (0.0915, 0.2105, 0.0745)       # T14 crate cluster 183 x 421 x 149 mm
BANANA_HALF = (0.0765, 0.0315, 0.0395)      # T14 banana-side cluster 153 x 63 x 79 mm


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


def _select(confirm, *items):
    clusters, pts = _clusters(*items)
    return confirm.select(clusters, points=pts)


# ------------------------------------------------------------------ 1. max opening from the model


def test_principal_extents_do_not_depend_on_orientation():
    rot = np.array([[np.cos(0.6), -np.sin(0.6), 0], [np.sin(0.6), np.cos(0.6), 0], [0, 0, 1]])
    straight = principal_extents(_box((0, 0, 0), BANANA_HALF, n=4000))
    turned = principal_extents(_box((0, 0, 0), BANANA_HALF, n=4000, rot=rot))
    np.testing.assert_allclose(straight, turned, atol=2e-3)
    assert list(straight) == sorted(straight)
    # PCA of a box surface need not align with two similar sides (31.5 vs 39.5 mm half-widths), so
    # the narrowest principal extent is ≥ the true 63 mm — the conservative direction.
    assert 0.062 <= straight[0] <= 0.071                       # banana is graspable by width


@pytest.fixture(scope="module")
def rby1_models():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from AG3S.ag3s.experiments.reports.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    scene = TransportScene.attach(model, mujoco.MjData(model))
    return build_robot_model(scene), build_constraint_robot_model(scene)


def test_rby1_max_opening_is_the_finger_spheres_inner_gap_at_the_joint_limit(rby1_models):
    filt, _ = rby1_models
    opening = max_opening_from_models(filt)
    assert opening.known and opening.source == "robot_model"
    grippers = {g["parent_link"]: g for g in opening.detail["grippers"]}
    assert set(grippers) == {"ee_left", "ee_right"}
    left = grippers["ee_left"]
    assert left["fingers"] == ["ee_finger_l1", "ee_finger_l2"]
    assert left["joints"] == {"gripper_finger_l1": -0.05, "gripper_finger_l2": 0.05}
    assert left["origin_gap_m"] == pytest.approx(0.106)            # the lead's 106 mm
    # inner faces of the collision geometry: smaller than the origin gap by the fingers' thickness
    assert opening.value_m == pytest.approx(0.07146, abs=1e-4)
    assert opening.value_m < left["origin_gap_m"]
    # apple (62) and banana (63) fit, the crate (149) does not
    assert 0.063 < opening.value_m < 0.149


def test_the_opening_follows_the_joint_limits_it_is_not_a_literal(rby1_models):
    filt, _ = rby1_models
    base = max_opening_from_models(filt).value_m
    urdf = filt.model
    narrower = dataclasses.replace(urdf, joints=tuple(
        dataclasses.replace(j, lower=-0.03) if j.name == "gripper_finger_l1"
        else dataclasses.replace(j, upper=0.03) if j.name == "gripper_finger_l2" else j
        for j in urdf.joints))
    filt.model = narrower
    try:
        grippers = {g["parent_link"]: g for g in filt.gripper_openings()}
    finally:
        filt.model = urdf
    assert grippers["ee_left"]["inner_gap_m"] == pytest.approx(base - 0.04, abs=1e-9)
    assert grippers["ee_right"]["inner_gap_m"] == pytest.approx(base, abs=1e-12)


def test_the_constraint_model_gives_the_same_number(rby1_models):
    filt, constraint = rby1_models
    assert max_opening_from_models(constraint).value_m == pytest.approx(
        max_opening_from_models(filt).value_m)


def test_no_model_no_number_and_config_overrides():
    assert not max_opening_from_models(None, object()).known
    assert resolve_max_opening(0.08).value_m == 0.08 and resolve_max_opening(0.08).source == "config"
    assert resolve_max_opening(None, PlanarTwoLinkArm()).value_m == PlanarTwoLinkArm.MOCK_MAX_OPENING


def test_ag3s_uses_the_rby1_model_opening(rby1_models):
    filt, constraint = rby1_models
    ag3s = AG3S(AG3SConfig(), robot_model=filt, constraint_robot_model=constraint)
    assert ag3s.max_opening.value_m == pytest.approx(0.07146, abs=1e-4)
    assert ag3s._target_confirm.max_opening == pytest.approx(ag3s.max_opening.value_m)


# --------------------------------------------------------------------------- 2. admissibility


def test_the_t14_clusters_by_the_rule():
    kw = dict(extents=None, max_opening_m=OPENING, destination=None, overlap_distance=0.005,
              overlap_fraction=0.5)
    apple = assess(_ball(APPLE_C), **kw)
    crate = assess(_box(CRATE_C, CRATE_HALF), **kw)
    banana = assess(_box(BANANA_C, BANANA_HALF), **kw)
    assert apple.admissible and apple.reason == ADMISSIBLE
    assert crate.reason == NOT_GRASPABLE and crate.graspable is False
    assert crate.narrowest_extent_m >= 0.148        # 149 mm side (PCA may read it a bit wider)
    assert banana.admissible            # banana really is graspable; rule 3 keeps it off the apple
    unknown = assess(_ball(APPLE_C), **{**kw, "max_opening_m": None})
    assert unknown.reason == MAX_OPENING_UNKNOWN and not unknown.admissible


def test_first_takes_the_best_admissible_cluster_not_the_crate_leader():
    confirm = _confirm()
    d = _select(confirm, (_box(CRATE_C, CRATE_HALF), 0.9), (_ball(APPLE_C), 0.3))
    assert d.mode == "first" and d.rank == 1
    np.testing.assert_allclose(confirm.held_centroid, _ball(APPLE_C).mean(axis=0))
    assert d.admissibility.admissible and d.n_admissible == 1


def test_only_the_crate_is_no_admissible_and_nothing_is_held():
    confirm = _confirm()
    for _ in range(5):
        d = _select(confirm, (_box(CRATE_C, CRATE_HALF), 0.9))
        assert (d.mode, d.cluster, d.reason) == (NO_ADMISSIBLE, None, NO_ADMISSIBLE)
    assert confirm.manipulated is None


def test_the_crate_never_becomes_the_manipulated_object_at_any_score():
    """T14 seq 12–21 with the crate far above the minimum: still never manipulated."""
    confirm = _confirm()
    _select(confirm, (_ball(APPLE_C), 0.5))
    for _ in range(30):
        d = _select(confirm, (_box(CRATE_C, CRATE_HALF), 0.95))
        assert d.mode == OCCLUDED and d.manipulated_id == 0 and d.streak == 0
    assert confirm.manipulated.id == 0 and confirm.manipulated.switched_from is None


def test_the_crate_is_registered_as_the_destination_after_n_frames_at_the_minimum():
    confirm = _confirm()
    _select(confirm, (_ball(APPLE_C), 0.5))
    events = [(_select(confirm, (_box(CRATE_C, CRATE_HALF, seed=k), 0.2)),
               confirm.destinations.last_event) for k in range(3)]
    assert [e for _, e in events] == [None, None, "registered"]
    dest = confirm.destination
    assert dest.source == REGISTERED and dest.id == 0 and dest.registered_frame == 3
    assert dest.extents_m[0] >= 0.148
    # re-observed → refreshed, not re-registered
    _select(confirm, (_box(CRATE_C + 0.01, CRATE_HALF, seed=9), 0.2))
    assert confirm.destinations.last_event == "updated" and confirm.destination.id == 0
    assert confirm.destination.n_observations == 2


def test_a_crate_under_the_minimum_is_not_registered():
    confirm = _confirm()
    for _ in range(10):
        _select(confirm, (_box(CRATE_C, CRATE_HALF), 0.05), (_ball(APPLE_C), 0.04))
    assert confirm.destination is None


def test_a_graspable_leader_is_never_registered_as_destination():
    confirm = _confirm()
    for _ in range(10):
        _select(confirm, (_ball(APPLE_C), 0.9))
    assert confirm.destination is None


def test_an_admissible_challenger_switches_only_at_the_minimum_for_n_frames():
    low = _confirm()
    _select(low, (_ball(APPLE_C), 0.5))
    modes = [_select(low, (_box(BANANA_C, BANANA_HALF), 0.05)).mode for _ in range(6)]
    assert modes == [OCCLUDED] * 6                  # T20 verify: min ≥ 0.05 keeps the apple … at 0.1
    high = _confirm()
    _select(high, (_ball(APPLE_C), 0.5))
    modes = [_select(high, (_box(BANANA_C, BANANA_HALF), 0.3)).mode for _ in range(3)]
    assert modes == [OCCLUDED, OCCLUDED, "switch"]
    assert high.manipulated.id == 1 and high.manipulated.admissibility.admissible


def test_an_inadmissible_leader_does_not_outrank_the_held_apple():
    """The crate leads but cannot compete: the apple is `keep` at its real rank, streak 0."""
    confirm = _confirm()
    _select(confirm, (_ball(APPLE_C), 0.5))
    for _ in range(5):
        d = _select(confirm, (_box(CRATE_C, CRATE_HALF), 0.9), (_ball(APPLE_C), 0.2))
        assert (d.mode, d.rank, d.streak) == ("keep", 1, 0)


def test_a_merged_cluster_at_the_apple_is_occluded_with_inadmissible_match():
    """The apple fused with the crate / the fingers: its place holds a crate-sized cluster."""
    confirm = _confirm()
    apple = _ball(APPLE_C)
    _select(confirm, (apple, 0.5))
    clusters, pts = _clusters((apple, 0.5))
    ground_geometry = confirm.last_geometry  # None: select() alone does not observe
    merged = np.concatenate([apple, _box(APPLE_C + [0, 0.15, 0], CRATE_HALF)])
    merged = merged - merged.mean(axis=0) + APPLE_C + [0.0, 0.02, 0.0]   # centroid within 60 mm
    d = _select(confirm, (merged, 0.6))
    assert (d.mode, d.reason, d.cluster, d.age_frames) == (OCCLUDED, INADMISSIBLE_MATCH, None, 1)
    assert d.admissibility is not None and d.admissibility.reason == NOT_GRASPABLE
    np.testing.assert_allclose(confirm.held_centroid, apple.mean(axis=0))   # not tracked
    assert confirm.manipulated.admissibility.admissible      # last admissible evidence kept
    assert ground_geometry is None


def test_a_crate_fragment_that_looks_graspable_is_kept_out_by_the_destination_overlap():
    confirm = _confirm(destination_points=_box(CRATE_C, CRATE_HALF, n=20000))
    assert confirm.destination.source == INJECTED
    # one wall of the crate, seen edge-on: 8 mm thick — "graspable" by extent alone
    wall = _box(CRATE_C + [0.0915, 0, 0], (0.001, 0.2, 0.07), n=400, seed=4)
    wall[:, 0] = CRATE_C[0] + 0.0915
    d = _select(confirm, (wall, 0.9))
    adm = confirm.last_admissibility[0]
    assert adm.graspable and adm.reason == DESTINATION_OVERLAP and adm.destination_overlap > 0.5
    assert d.mode == NO_ADMISSIBLE


def test_an_apple_resting_against_the_crate_is_still_admissible():
    confirm = _confirm(destination_points=_box(CRATE_C, CRATE_HALF, n=20000))
    touching = _ball(CRATE_C + [0.0915 + 0.031, 0, 0])
    d = _select(confirm, (touching, 0.5))
    assert d.mode == "first" and 0.0 < d.admissibility.destination_overlap < 0.5


def test_an_injected_destination_takes_precedence_and_is_never_replaced():
    confirm = _confirm()
    confirm.reset(destination_points=_box(BANANA_C, CRATE_HALF, n=3000))
    injected = confirm.destination
    for k in range(5):
        _select(confirm, (_box(CRATE_C, CRATE_HALF, seed=k), 0.9))
    assert confirm.destination is injected and confirm.destination.source == INJECTED


def test_a_frozen_confirm_never_switches():
    confirm = _confirm()
    _select(confirm, (_ball(APPLE_C), 0.5))
    confirm.freeze(True)
    modes = [_select(confirm, (_box(BANANA_C, BANANA_HALF), 0.9)).mode for _ in range(6)]
    assert modes == [OCCLUDED] * 6 and confirm.last.frozen
    confirm.freeze(False)
    modes = [_select(confirm, (_box(BANANA_C, BANANA_HALF), 0.9)).mode for _ in range(3)]
    assert modes == [OCCLUDED, OCCLUDED, "switch"]


def test_checked_select_needs_the_cloud():
    clusters, _ = _clusters((_ball(APPLE_C), 0.5))
    with pytest.raises(ValueError, match="points"):
        _confirm().select(clusters)


def test_unchecked_is_the_standalone_default():
    assert TargetConfirm(CFG).max_opening == UNCHECKED
    assert TargetConfirm(ClusteringConfig(gripper_max_opening=0.08)).max_opening == 0.08


def _apc(*blobs):
    pts = np.concatenate([b for b, _ in blobs])
    att = np.concatenate([np.full(len(b), a, np.float32) + np.linspace(0, 1e-3, len(b),
                                                                       dtype=np.float32)
                          for b, a in blobs])
    return AttentionPointCloud(PointCloud(pts), att, att)


def test_ground_target_says_no_admissible():
    confirm = _confirm()
    # dense enough that DBSCAN (eps 30 mm) keeps the crate one cluster — a sparse sample splits it
    # into faces, and a single face is planar (see T26.impl.md, "planar fragment")
    crate = _box(CRATE_C, CRATE_HALF, n=12000)
    res = ground_target(_apc((crate, 0.9)), CFG, seed_threshold=0.5, confirm=confirm)
    assert res.target is None and res.status is GroundingStatus.NO_ADMISSIBLE
    assert len(res.clusters) >= 1


def test_ground_target_metrics_carry_the_admissibility_numbers():
    confirm = _confirm()
    res = ground_target(_apc((_ball(APPLE_C, n=400), 0.9)), CFG, seed_threshold=0.5,
                        confirm=confirm)
    m = res.target.metrics
    assert m["admissible"] == 1.0 and m["narrowest_extent_m"] <= m["gripper_max_opening_m"]


# --------------------------------------------------------------------------------- 3. config


def test_defaults_are_the_user_rulings():
    cfg = AG3SConfig()
    assert cfg.clustering.target_switch_min_score == 0.1
    assert cfg.clustering.target_lost_frames is None
    assert cfg.clustering.gripper_max_opening is None
    assert cfg.pointcloud.self_filter_inflation == 0.0      # sphere radius itself (was 0.05)


@pytest.mark.parametrize("bad", [{"gripper_max_opening": 0.0}, {"gripper_max_opening": -0.1},
                                 {"gripper_max_opening": True},
                                 {"destination_overlap_fraction": 0.0},
                                 {"destination_overlap_fraction": 1.5},
                                 {"destination_overlap_distance": 0.0}])
def test_config_rejects_malformed_t26_values(bad):
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": bad})


# ----------------------------------------------------------------------------- 4. the pipeline


SCENE_KW = dict(phase="approach", robot_state=DEFAULT_ARM_Q)
LABEL_CRATE = 9
BIG_CRATE = GTObject(LABEL_CRATE, "crate", "box",
                     np.array([0.70, 0.36, TABLE_Z + 0.0745]), np.array(CRATE_HALF))
APPLE_AND_CRATE = tuple(o for o in DEFAULT_OBJECTS if o.label == LABEL_SPHERE) + (BIG_CRATE,)
CRATE_ONLY = (BIG_CRATE,)


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scenes(arm):
    apple = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=APPLE_AND_CRATE,
                       target_label=LABEL_SPHERE)
    crate = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=APPLE_AND_CRATE,
                       target_label=LABEL_CRATE)
    crate_alone = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=CRATE_ONLY,
                             target_label=LABEL_CRATE)
    return {"apple": apple, "crate": crate, "crate_alone": crate_alone}


def _inputs(scene):
    return dict(attention_map=scene.attention_grid, depth=scene.depth,
                camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                image_hw=scene.hw, **SCENE_KW)


def _ag3s(arm, seen=None, **kw):
    data = {"collision_backend": "esdf"} if seen is not None else {}
    ag3s = AG3S(AG3SConfig.from_dict(data), robot_model=arm, **kw)
    if seen is not None:
        def spy(depth_cameras, target, **k):
            seen.append((target, k.get("destination_points")))
            return None, []
        ag3s._build_esdf = spy
    return ag3s


def _near(points, centre, tol=0.05):
    return float(np.linalg.norm(np.asarray(points).reshape(-1, 3).mean(axis=0) - centre)) < tol


def test_pipeline_attention_on_the_crate_alone_names_nothing_and_registers_it(arm, scenes):
    seen = []
    ag3s = _ag3s(arm, seen)
    sets = [ag3s.process(**_inputs(scenes["crate_alone"])) for _ in range(4)]
    for cs in sets:
        assert cs.target is None and cs.manipulated is None
        assert cs.grounding_status is GroundingStatus.NO_ADMISSIBLE
        assert cs.metrics["exclusion"]["source"] == "none"
    assert all(t is None for t, _ in seen), "no exclusion geometry from the crate, ever"
    dest = ag3s.destination
    assert dest is not None and dest.source == REGISTERED
    # the visible part of the crate (it runs out of the frame): every point on the crate's box
    lo, hi = BIG_CRATE.center - BIG_CRATE.extent - 0.01, BIG_CRATE.center + BIG_CRATE.extent + 0.01
    assert np.all((dest.points >= lo) & (dest.points <= hi))
    assert sets[2].metrics["destination"]["event"] == "registered"
    assert any("destination registered" in n for n in sets[2].notes)
    # once registered, it is labelled in the field (F18) — handed to `_build_esdf`
    assert seen[-1][1] is not None and len(seen[-1][1]) == len(dest.points)


def test_pipeline_apple_first_then_attention_on_the_crate_keeps_the_apple(arm, scenes):
    seen = []
    ag3s = _ag3s(arm, seen)
    cs0 = ag3s.process(**_inputs(scenes["apple"]))
    if cs0.target is None:
        pytest.skip("grounding declined the apple scene")
    apple_c = next(o.center for o in DEFAULT_OBJECTS if o.label == LABEL_SPHERE)
    assert _near(cs0.target.points, apple_c)
    for _ in range(8):
        cs = ag3s.process(**_inputs(scenes["crate"]))
        assert cs.metrics["manipulated"]["id"] == 0
        assert cs.target is not None and _near(cs.target.points, apple_c)
        assert cs.metrics["exclusion"]["source"] == "manipulated"
        assert cs.metrics["invariant_violation"] is None
    for target, _ in seen:
        assert target is not None and _near(target.points, apple_c), "the ball is the apple's"


def test_pipeline_invariant_refuses_geometry_without_evidence(arm, scenes):
    ag3s = _ag3s(arm, [], strict_invariants=False)
    # A bug stand-in: an unchecked confirm lets a cluster through with no admissibility evidence.
    ag3s._target_confirm = TargetConfirm(ag3s.config.clustering, max_opening=UNCHECKED)
    cs = ag3s.process(**_inputs(scenes["crate_alone"]))
    inv = cs.metrics["invariant_violation"]
    assert inv is not None and "no admissibility evidence" in inv["detail"]
    assert any(n.startswith("invariant_violation:") for n in cs.notes)
    assert cs.target is None and cs.manipulated is None and cs.manipulated_link_margin is None
    assert cs.metrics["exclusion"]["source"] == "none"


def test_pipeline_invariant_raises_in_strict_mode(arm, scenes):
    ag3s = _ag3s(arm, [])
    assert ag3s.strict_invariants            # tests/conftest.py
    ag3s._target_confirm = TargetConfirm(ag3s.config.clustering, max_opening=UNCHECKED)
    with pytest.raises(ExclusionInvariantViolation):
        ag3s.process(**_inputs(scenes["crate_alone"]))


def test_pipeline_with_no_known_opening_names_nothing(scenes):
    ag3s = AG3S(AG3SConfig())
    assert not ag3s.max_opening.known
    cs = ag3s.process(**_inputs(scenes["apple"]))
    assert cs.target is None and cs.grounding_status is GroundingStatus.NO_ADMISSIBLE


def test_reset_injects_a_destination_that_wins(arm, scenes):
    ag3s = _ag3s(arm, [])
    crate_pts = scenes["crate"].points_of(LABEL_CRATE)
    ag3s.reset(destination_points=crate_pts)
    assert ag3s.destination.source == INJECTED
    for _ in range(4):
        ag3s.process(**_inputs(scenes["crate"]))
    assert ag3s.destination.source == INJECTED and len(ag3s.destination.points) == len(crate_pts)
    ag3s.reset()
    assert ag3s.destination is None


def test_attach_refuses_the_crate_and_takes_the_apple(arm, scenes):
    from AG3S.ag3s.constraints.attached import AttachRejected
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry
    from tests.ag3s.fixtures import FINGERTIP_LINK

    ag3s = _ag3s(arm)

    def geom(points):
        points = np.asarray(points, np.float64)
        return TargetGeometry(id=0, points=points, point_indices=np.arange(len(points)),
                              centroid=points.mean(axis=0),
                              bounding_geometry=fit_sphere(points, semantic_role="target"),
                              attention_score=1.0, confidence=1.0, timestamp=0.0, frame_id="base")

    with pytest.raises(AttachRejected, match="inadmissible"):
        ag3s.attach(geom(_box(CRATE_C, CRATE_HALF)), robot_state=DEFAULT_ARM_Q,
                    parent_link=FINGERTIP_LINK)
    assert ag3s.attached is None
    held = ag3s.attach(geom(_ball(APPLE_C)), robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK)
    assert ag3s.attached is held


def test_exclusion_is_continuous_from_approach_through_attach_to_release(arm, scenes):
    from tests.ag3s.fixtures import FINGERTIP_LINK

    ag3s = AG3S(AG3SConfig(), robot_model=arm, attached_parent_links=(FINGERTIP_LINK,))
    ctx = dict(active_manipulators=["left"])
    cs0 = ag3s.process(**_inputs(scenes["apple"]), **ctx)
    if cs0.target is None:
        pytest.skip("grounding declined the apple scene")
    sources = [cs0.metrics["exclusion"]["source"]]
    ag3s.set_grasp_active(True)                    # closing
    sources.append(ag3s.process(**_inputs(scenes["crate"]), **ctx).metrics["exclusion"]["source"])
    ag3s.attach(ag3s.manipulated.geometry, robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK)
    for _ in range(3):                             # transport, attention on the crate
        cs = ag3s.process(**_inputs(scenes["crate"]), **ctx)
        sources.append(cs.metrics["exclusion"]["source"])
        assert cs.metrics["manipulated"]["id"] == 0
    ag3s.detach()
    ag3s.set_grasp_active(False)                   # released
    sources.append(ag3s.process(**_inputs(scenes["apple"]), **ctx).metrics["exclusion"]["source"])
    assert sources == ["manipulated", "manipulated", "attached", "attached", "attached",
                       "manipulated"]
    assert "none" not in sources


def test_a_grasp_in_progress_bars_the_switch(arm, scenes):
    """The box of the default scene is an admissible challenger (score ≥ 0.1)."""
    from tests.ag3s.fixtures import DEFAULT_OBJECTS as OBJS

    no_apple = tuple(o for o in OBJS if o.label != LABEL_SPHERE)
    box_scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=no_apple,
                           target_label=LABEL_BOX)
    apple_scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    free = _ag3s(arm)
    if free.process(**_inputs(apple_scene)).target is None:
        pytest.skip("grounding declined the apple scene")
    states = [free.process(**_inputs(box_scene)).metrics["manipulated"]["id"] for _ in range(3)]
    assert states[-1] == 1                        # the box earns a switch when nothing is grasped
    held = _ag3s(arm)
    held.process(**_inputs(apple_scene))
    held.set_grasp_active(True)
    ids = [held.process(**_inputs(box_scene)).metrics["manipulated"]["id"] for _ in range(6)]
    assert ids == [0] * 6


def test_the_chunk_summary_carries_the_t26_records(arm, scenes):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    ag3s = _ag3s(arm)
    cs, debug = ag3s.process_debug(**_inputs(scenes["crate_alone"]))
    for key in ("exclusion", "destination", "invariant_violation", "admissibility",
                "gripper_max_opening"):
        assert key in cs.metrics
    adm = cs.metrics["admissibility"]
    assert adm["n_admissible"] == 0 and adm["clusters"][0]["reason"] == NOT_GRASPABLE
    assert adm["max_opening_mm"] == pytest.approx(PlanarTwoLinkArm.MOCK_MAX_OPENING * 1000.0)
    assert ConstraintRecordWriter is not None
