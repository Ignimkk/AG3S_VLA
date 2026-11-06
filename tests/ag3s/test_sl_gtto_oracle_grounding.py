"""SL-GTTO §2-1 / §3-4 — the GT grounding input (`clustering.grounding_source = "oracle"`, default `"attention"`), CPU.

| # | what is pinned |
|---|---|
| 1 | config default / validation; serve_safe `--grounding-source` default adds no key; refused without `--target-extent oracle-visible` |
| 2 | `AG3S._oracle_grounding_input` on a fused, ray-cast scene: the code-1 points are exactly the GT target's (a voxel that also holds a neighbour pixel is out), attention 1 / exclude mask = not code 1, same indices as the cloud; no segmentation / no code-1 point / no fusion -> an empty cloud |
| 3 | `ground_target` on it: the target cluster is the target only, although a neighbour box touches it 2 mm away (the attention cloud with the same seeds grows into the box); empty -> `NO_GEOMETRY` and `TargetConfirm` keeps / ages the last geometry |
| 4 | the pipeline (`process_multi`, 3 cameras + arm): attention on the **neighbour**, GT on the sphere -> default grounds the box, oracle grounds the sphere; `metrics["grounding_source"]` only when on |
| 5 | off identity: default config with or without a segmentation handed in -> `ground_target` gets the very same cloud / mask objects, metrics and target are identical, no new key |
"""
from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.oracle_segmentation import CODE_NONE, CODE_OBJECT_BASE, CODE_STATIC, CODE_TARGET
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    LABEL_BOX,
    LABEL_CYLINDER,
    LABEL_NONE,
    LABEL_ROBOT,
    LABEL_SPHERE,
    LABEL_TABLE,
    TABLE_Z,
    GTObject,
    PlanarTwoLinkArm,
    make_three_camera_scene,
)

SPHERE_C = np.array([0.62, 0.00, TABLE_Z + 0.040])
SPHERE_R = 0.040
#: a box touching the target sphere: its face is 2 mm from the sphere surface (DBSCAN eps 3 cm joins them)
BOX_C = np.array([0.62, SPHERE_R + 0.045 + 0.002, TABLE_Z + 0.050])
BOX_H = np.array([0.045, 0.045, 0.050])
TOUCHING = (
    GTObject(LABEL_SPHERE, "target_sphere", "sphere", SPHERE_C, np.array([SPHERE_R] * 3)),
    GTObject(LABEL_BOX, "neighbour_box", "box", BOX_C, BOX_H),
    GTObject(LABEL_CYLINDER, "far_cylinder", "cylinder", np.array([0.86, -0.190, TABLE_Z + 0.060]),
             np.array([0.035, 0.035, 0.060])),
)
#: the simulator's code table on this scene (oracle_segmentation / oracle_codes): target 1, table 2, others 3+
LABEL_TO_CODE = {LABEL_NONE: CODE_NONE, LABEL_TABLE: CODE_STATIC, LABEL_SPHERE: CODE_TARGET,
                 LABEL_BOX: CODE_OBJECT_BASE, LABEL_CYLINDER: CODE_OBJECT_BASE + 1, LABEL_ROBOT: CODE_NONE}
CFG = AG3SConfig().with_overrides({"timing": {"expected_cameras": ["head", "left_wrist", "right_wrist"]}})


def _codes(labels):
    out = np.zeros(labels.shape, np.uint8)
    for lab, code in LABEL_TO_CODE.items():
        out[labels == lab] = code
    return out


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scene(arm):
    # attention on the **neighbour box** (the identity attention would get wrong); GT target = the sphere
    return make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, objects=TOUCHING,
                                   target_label=LABEL_BOX)


def _seg(scene):
    return {"labels": {cam.value: _codes(view.labels) for cam, view in scene.views.items()},
            "meta": {"version": 1, "target": "target_sphere", "codes": {"target_sphere": 1, "neighbour_box": 3}}}


def _fused(scene, arm):
    from AG3S.ag3s.runtime.multiview import fuse_observations
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import AttentionPointCloud

    fusion = fuse_observations(list(scene.observations), CFG, robot_model=arm)
    cloud = fusion.pointcloud
    _, support_mask = fit_support_surfaces(cloud, CFG.support_surface)
    return fusion, AttentionPointCloud(cloud, fusion.attention, fusion.raw_attention), support_mask


def _input(seg, attention_cloud, support_mask, fusion):
    from AG3S.ag3s.runtime.pipeline import AG3S

    fake = types.SimpleNamespace(_oracle_pending=seg, config=CFG)
    return AG3S._oracle_grounding_input(fake, attention_cloud, support_mask, fusion)


def _near_sphere(points, tol=0.006):
    return np.abs(np.linalg.norm(np.asarray(points) - SPHERE_C, axis=1) - SPHERE_R) < tol


def _in_box(points, pad=0.001):   # box surface points (the sphere is 2 mm off the face)
    return np.all(np.abs(np.asarray(points) - BOX_C) <= BOX_H + pad, axis=1)


# ============================================================================ 1 config · CLI
def test_config_default_round_trip_and_validation():
    assert ClusteringConfig().grounding_source == "attention"
    assert ClusteringConfig.GROUNDING_SOURCES == ("attention", "oracle")
    cfg = AG3SConfig.from_dict({"clustering": {"grounding_source": "oracle"}})
    assert cfg.clustering.grounding_source == "oracle"
    assert AG3SConfig.from_dict(cfg.to_dict()).clustering.grounding_source == "oracle"
    for bad in ("gt", "", None, 1):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"clustering": {"grounding_source": bad}})


def _parse(*extra):
    from AG3S.adapters.panda import profile as pf

    return pf.parse_v11ab_args(argv=pf.v11ab_server_argv(pre_merge=False) + list(extra))


def test_cli_default_adds_no_identity_key_and_oracle_maps_to_the_clustering_key():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    args = _parse()
    assert args.grounding_source == "attention" and ss.build_parser().get_default("grounding_source") == "attention"
    assert "grounding_source" not in ss.t43t_identity_section(args)
    assert "grounding_source" not in (pf.v11ab_build_kwargs(args)["identity"] or {})
    on = _parse("--target-extent", "oracle-visible", "--grounding-source", "oracle")
    assert ss.t43t_identity_section(on)["grounding_source"] == "oracle"
    kw = pf.v11ab_build_kwargs(on)
    cfg = pf.v11ab_ag3s_config(kw)                     # build_ag3s's assembly + its survival check
    assert cfg.clustering.grounding_source == "oracle" and cfg.esdf.target_extent == "oracle_visible"


@pytest.mark.parametrize("extra", [("--grounding-source", "oracle"),
                                   ("--grounding-source", "oracle", "--target-extent", "attention-surface")])
def test_cli_refuses_the_gt_grounding_without_the_oracle_extent(extra):
    with pytest.raises(SystemExit):
        _parse(*extra)


# ============================================================================ 2 the GT grounding input
def test_code1_points_are_the_target_only_on_the_same_indices(scene, arm):
    fusion, att, support = _fused(scene, arm)
    cloud, exclude, rec = _input(_seg(scene), att, support, fusion)
    code1 = ~exclude
    pts = att.points
    assert cloud.cloud is att.cloud, "same cloud, same indices (every point_indices consumer is unchanged)"
    assert np.array_equal(cloud.attention, code1.astype(np.float32))
    assert np.array_equal(cloud.raw_attention, code1.astype(np.float32))
    assert rec["source"] == "oracle" and rec["why"] is None and rec["n_code1"] == int(code1.sum()) > 50
    assert sorted(rec["cameras"]) == ["head", "left_wrist", "right_wrist"] and rec["target"] == "target_sphere"
    assert _near_sphere(pts[code1]).all(), "every code-1 point lies on the GT sphere"
    assert not _in_box(pts[code1]).any(), "no code-1 point on the touching box"
    # what is not code 1 on the sphere: voxels that also hold a non-target pixel (the rim) or no label
    rim = _near_sphere(pts, tol=0.002) & ~code1
    assert rec["n_mixed"] >= 0 and rim.sum() < 0.5 * code1.sum()
    # the GT mask, per observation: a code-1 point has no observation off code 1
    fused = fusion.cloud
    labels = _seg(scene)["labels"]
    for i in np.flatnonzero(code1)[:200]:
        blk = fused.observation_slice(int(i))
        for c, (u, v) in zip(fused.obs_camera[blk], fused.obs_uv[blk]):
            assert labels[fused.cameras[int(c)].value][v, u] == CODE_TARGET


def test_a_voxel_that_also_holds_a_neighbour_pixel_is_not_code1(scene, arm):
    fusion, att, support = _fused(scene, arm)
    seg = _seg(scene)
    _, exclude, rec = _input(seg, att, support, fusion)
    code1 = ~exclude
    # relabel one camera's pixel of one code-1 point as the neighbour: that point leaves the code-1 set
    fused = fusion.cloud
    n_obs = np.diff(fused.obs_offset)
    i = int(np.flatnonzero(code1 & (n_obs >= 2))[0])          # a code-1 voxel seen by two observations
    blk = fused.observation_slice(i)
    c, (u, v) = int(fused.obs_camera[blk][0]), fused.obs_uv[blk][0]
    labels = {k: np.array(x, copy=True) for k, x in seg["labels"].items()}
    labels[fused.cameras[c].value][v, u] = CODE_OBJECT_BASE
    _, exclude2, rec2 = _input({**seg, "labels": labels}, att, support, fusion)
    assert not (~exclude2)[i] and rec2["n_code1"] == rec["n_code1"] - 1 and rec2["n_mixed"] == rec["n_mixed"] + 1


@pytest.mark.parametrize("case", ["no_segmentation", "no_code1_point", "no_fusion"])
def test_no_gt_target_this_request_gives_grounding_no_geometry(scene, arm, case):
    from AG3S.ag3s.stages.target_grounding import ground_target
    from AG3S.ag3s.types import GroundingStatus

    fusion, att, support = _fused(scene, arm)
    seg = _seg(scene)
    if case == "no_segmentation":
        seg = None
    elif case == "no_code1_point":
        seg = {**seg, "labels": {k: np.where(v == CODE_TARGET, CODE_STATIC, v).astype(np.uint8)
                                 for k, v in seg["labels"].items()}}
    cloud, exclude, rec = _input(seg, att, support, None if case == "no_fusion" else fusion)
    assert len(cloud) == 0 and exclude is None and rec["why"] == case and rec["n_code1"] == 0
    assert ground_target(cloud, CFG.clustering).status is GroundingStatus.NO_GEOMETRY


# ============================================================================ 3 ground_target on it
def test_ground_target_on_the_gt_input_does_not_grow_into_the_touching_box(scene, arm):
    from AG3S.ag3s.stages.target_grounding import ground_target
    from AG3S.ag3s.types import AttentionPointCloud

    fusion, att, support = _fused(scene, arm)
    cloud, exclude, _ = _input(_seg(scene), att, support, fusion)
    kw = dict(seed_percentile=CFG.attention.seed_percentile, seed_threshold=CFG.attention.seed_threshold,
              min_radius=CFG.geometry.min_radius)
    gt = ground_target(cloud, CFG.clustering, exclude_mask=exclude, **kw)
    tp = gt.target.points
    assert len(tp) > 50 and _near_sphere(tp).all() and not _in_box(tp).any()
    assert np.linalg.norm(gt.target.centroid - SPHERE_C) < SPHERE_R
    # contrast: the same seeds (the code-1 points) in the attention path's connectivity (support mask only) reach
    # the box 2 mm away -- the cluster AG3S's own grounding would grow
    seeds_only = AttentionPointCloud(att.cloud, cloud.attention, cloud.raw_attention)
    grown = ground_target(seeds_only, dataclasses.replace(CFG.clustering, attention_split_radius=None),
                          exclude_mask=support, **kw)                  # (T31b split off: pure connectivity)
    assert grown.target is not None and _in_box(grown.target.points).sum() > 20


def test_target_confirm_keeps_then_ages_the_geometry_when_the_target_is_not_seen(scene, arm):
    from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target

    fusion, att, support = _fused(scene, arm)
    confirm = TargetConfirm(CFG.clustering)
    kw = dict(seed_percentile=CFG.attention.seed_percentile, seed_threshold=CFG.attention.seed_threshold,
              min_radius=CFG.geometry.min_radius, confirm=confirm)
    cloud, exclude, _ = _input(_seg(scene), att, support, fusion)
    for t in range(CFG.clustering.target_confirm_frames + 1):
        ground_target(cloud, CFG.clustering, exclude_mask=exclude, timestamp=float(t), **kw)
    held = confirm.manipulated
    assert held is not None and _near_sphere(held.points).all()
    empty, ex2, _ = _input(None, att, support, fusion)
    ground_target(empty, CFG.clustering, exclude_mask=ex2, timestamp=10.0, **kw)
    assert confirm.manipulated is not None and confirm.manipulated.age_frames == 1
    assert np.array_equal(confirm.manipulated.points, held.points), "the last observed geometry stands in"


# ============================================================================ 4 / 5 the pipeline
def _ag3s(arm, **clustering):
    from AG3S.ag3s.runtime.pipeline import AG3S

    over = {"clustering": clustering} if clustering else {}
    if clustering.get("grounding_source") == "oracle":
        over["esdf"] = {"target_extent": "oracle_visible"}
    return AG3S(CFG.with_overrides(over) if over else CFG, robot_model=arm)


def _spy(monkeypatch):
    from AG3S.ag3s.runtime import pipeline as pl

    calls = []
    real = pl.ground_target

    def spy(cloud, *a, **kw):
        calls.append((cloud, kw.get("exclude_mask")))
        return real(cloud, *a, **kw)
    monkeypatch.setattr(pl, "ground_target", spy)
    return calls


def _run(ag3s, scene, seg=None, frames=3):
    out = None
    for _ in range(frames):
        if seg is not None:
            ag3s.set_oracle_segmentation(seg)
        out = ag3s.process_multi_debug(scene.observations, phase="approach")
    return out


def test_pipeline_oracle_grounds_the_gt_target_where_attention_grounds_the_neighbour(arm, scene):
    cs_att, dbg_att = _run(_ag3s(arm), scene)
    assert dbg_att["grounding"].target is not None and _in_box(dbg_att["grounding"].target.points).mean() > 0.5
    assert "grounding_source" not in cs_att.metrics
    ag = _ag3s(arm, grounding_source="oracle")
    cs, dbg = _run(ag, scene, _seg(scene))
    tgt = dbg["grounding"].target
    assert tgt is not None and _near_sphere(tgt.points).all() and not _in_box(tgt.points).any()
    rec = cs.metrics["grounding_source"]
    assert rec["source"] == "oracle" and rec["n_code1"] > 50 and rec["why"] is None
    assert rec["status"] == dbg["grounding"].status.value and rec["n_target_points"] == len(tgt.point_indices)
    json.dumps(rec)                                                         # recordable as is
    assert dbg["attention_cloud"].attention.max() > 0 and len(dbg["attention_cloud"]) == len(dbg["filtered_cloud"])
    ag.reset()
    assert ag._grounding_frame is None and ag._oracle_pending is None


def test_pipeline_oracle_without_a_segmentation_keeps_then_ages_the_target(arm, scene):
    ag = _ag3s(arm, grounding_source="oracle")
    _run(ag, scene, _seg(scene))
    held = ag._target_confirm.manipulated
    assert held is not None and _near_sphere(held.points).all()
    ag.set_oracle_segmentation(None)              # this request carries no code image (as SafePolicy hands it on)
    cs, dbg = _run(ag, scene, None, frames=1)
    rec = cs.metrics["grounding_source"]
    assert rec["why"] == "no_segmentation" and rec["n_code1"] == 0 and rec["status"] == "no_geometry"
    m = ag._target_confirm.manipulated
    assert m is not None and m.age_frames == 1 and np.array_equal(m.points, held.points)


def test_off_is_identical_with_or_without_a_segmentation(arm, scene, monkeypatch):
    calls = _spy(monkeypatch)
    a, b = _ag3s(arm), _ag3s(arm)
    out_a = _run(a, scene)
    n = len(calls)
    out_b = _run(b, scene, _seg(scene))                                   # ignored: default config stores nothing
    assert b._oracle_pending is None and b._grounding_frame is None
    for (cs1, d1), (cs2, d2) in zip([out_a], [out_b]):
        assert json.dumps(cs1.metrics, default=str, sort_keys=True) == json.dumps(cs2.metrics, default=str,
                                                                                   sort_keys=True)
        assert np.array_equal(d1["grounding"].target.point_indices, d2["grounding"].target.point_indices)
        assert "grounding_source" not in cs1.metrics
    # the default path hands ground_target the attention cloud and the support mask themselves
    for (cloud, mask), dbg in ((calls[n - 1], out_a[1]), (calls[-1], out_b[1])):
        assert cloud is dbg["attention_cloud"] and mask is dbg["support_mask"]
