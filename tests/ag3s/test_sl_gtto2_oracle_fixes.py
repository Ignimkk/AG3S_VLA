"""SL-GTTO2 §1 — the GT identity / extent reaches the TO (`clustering.oracle_admissible` F1, `clustering.oracle_held_target`
F2; both default False), CPU.

| # | what is pinned |
|---|---|
| 1 | config defaults / validation (bool, needs `grounding_source == "oracle"`); CLI: default adds no identity key, the flags map to the clustering keys, refused without `--grounding-source oracle` |
| 2 | F1 `TargetConfirm`: a GT cluster refused by `not_graspable` (max opening below it) or `not_supported` (support plane far below) is adopted when on and recorded (`last_would_refuse`); off = refused as before; on, a non-graspable GT cluster is never registered as the destination |
| 3 | F1 pipeline: `no_admissible` off -> grounding `ok` on, `metrics["grounding_source"]` carries `admissibility_would_refuse` and the exclusion gate's would-be refusal; no `invariant_violation` note; off: no new key |
| 4 | F2 `AG3S._oracle_held_target_mask`: inactive (no grasp, nothing attached) -> the same camera objects; closing / held -> every code-1 pixel is in the TSDF mask, `device_depth` dropped, counts recorded |
| 5 | F2 pipeline (`collision_backend: esdf`, the ESDF build replaced on the instance to read the cameras it gets): grasp active -> the cameras' masks = the self-filter masks OR code 1; no grasp -> identical masks; the fused cloud / grounding are unchanged; off -> no code-1 pixel added, no new key |
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.oracle_segmentation import CODE_TARGET
from tests.ag3s.test_sl_gtto_oracle_grounding import (  # noqa: F401 (fixtures)
    CFG, TABLE_Z, _fused, _input, _near_sphere, _seg, arm, scene,
)

ON = {"grounding_source": "oracle"}


# ============================================================================ 1 config · CLI
def test_config_defaults_and_validation():
    c = ClusteringConfig()
    assert c.oracle_admissible is False and c.oracle_held_target is False
    for name in ("oracle_admissible", "oracle_held_target"):
        cfg = AG3SConfig.from_dict({"clustering": {**ON, name: True}})
        assert getattr(cfg.clustering, name) is True
        assert getattr(AG3SConfig.from_dict(cfg.to_dict()).clustering, name) is True
        with pytest.raises(AG3SConfigError, match="grounding_source"):
            AG3SConfig.from_dict({"clustering": {name: True}})                 # attention grounding
        for bad in (1, "yes", None):
            with pytest.raises(AG3SConfigError):
                AG3SConfig.from_dict({"clustering": {**ON, name: bad}})


def _parse(*extra):
    from AG3S.adapters.panda import profile as pf

    return pf.parse_v11ab_args(argv=pf.v11ab_server_argv(pre_merge=False) + list(extra))


GT_ARGS = ("--target-extent", "oracle-visible", "--grounding-source", "oracle")


def test_cli_default_adds_no_key_and_the_flags_map_to_the_clustering_keys():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    for args in (_parse(), _parse(*GT_ARGS)):
        sec = ss.t43t_identity_section(args)
        assert "oracle_admissible" not in sec and "oracle_held_target" not in sec
    on = _parse(*GT_ARGS, "--oracle-admissible", "--oracle-held-target")
    sec = ss.t43t_identity_section(on)
    assert sec["oracle_admissible"] is True and sec["oracle_held_target"] is True
    cfg = pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(on))
    assert cfg.clustering.oracle_admissible and cfg.clustering.oracle_held_target


@pytest.mark.parametrize("flag", ["--oracle-admissible", "--oracle-held-target"])
def test_cli_refuses_the_flags_without_the_gt_grounding(flag):
    with pytest.raises(SystemExit):
        _parse(flag)
    with pytest.raises(SystemExit):
        _parse("--target-extent", "oracle-visible", flag)


# ============================================================================ 2 F1 TargetConfirm
def _ground(scene, arm, clustering, frames=4, support=None):
    from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target

    cfg = CFG.with_overrides({"clustering": clustering})
    fusion, att, sup = _fused(scene, arm)
    confirm = TargetConfirm(cfg.clustering)
    cloud, exclude, _ = _input(_seg(scene), att, sup, fusion)
    out = None
    for t in range(frames):
        if support is not None:
            confirm.note_support(support)
        out = ground_target(cloud, cfg.clustering, exclude_mask=exclude, timestamp=float(t),
                            seed_percentile=cfg.attention.seed_percentile, seed_threshold=cfg.attention.seed_threshold,
                            min_radius=cfg.geometry.min_radius, confirm=confirm)
    return out, confirm


@pytest.mark.parametrize("rule", ["not_graspable", "not_supported"])
def test_f1_the_gt_cluster_is_adopted_and_the_refusal_recorded(scene, arm, rule):
    if rule == "not_graspable":
        kw, support = {"gripper_max_opening": 0.03}, None           # the 80 mm target sphere does not fit
    else:   # the bowl in the drawer: the only support plane 200 mm below the target (T43 T 3(c))
        kw, support = {"supported_max_bottom": 0.06, "gripper_max_opening": 0.09}, np.array([[0.0, 0.0, 1.0, TABLE_Z - 0.2]])
    off, c_off = _ground(scene, arm, {**ON, **kw}, support=support)
    assert off.status.value == "no_admissible" and off.target is None and c_off.last_would_refuse is None
    on, c_on = _ground(scene, arm, {**ON, **kw, "oracle_admissible": True}, support=support)
    assert on.status.value == "ok" and _near_sphere(on.target.points).all()
    assert [r["reason"] for r in c_on.last_would_refuse] == [rule]
    assert all(a.admissible for a in c_on.last_admissibility)
    assert c_on.manipulated is not None and c_on.manipulated.admissibility.admissible
    if rule == "not_graspable":   # off: the non-graspable GT cluster leading the ranking becomes the destination
        assert c_off.destination is not None and c_on.destination is None


# ============================================================================ 3 F1 pipeline
def _ag3s(arm, esdf=None, **clustering):
    from AG3S.ag3s.runtime.pipeline import AG3S

    over = {"clustering": {**ON, **clustering}, "esdf": {"target_extent": "oracle_visible", **(esdf or {})}}
    return AG3S(CFG.with_overrides(over), robot_model=arm)


def _run(ag, scene, frames=3, grasp=None):
    out = None
    for _ in range(frames):
        if grasp is not None:
            ag.set_grasp_active(grasp)
        ag.set_oracle_segmentation(_seg(scene))
        out = ag.process_multi_debug(scene.observations, phase="approach")
    return out


def test_f1_pipeline_grounds_the_gt_target_the_admissibility_refused(arm, scene):
    cs_off, dbg_off = _run(_ag3s(arm, gripper_max_opening=0.03), scene)
    assert dbg_off["grounding"].status.value == "no_admissible"
    rec_off = cs_off.metrics["grounding_source"]
    assert not {"oracle_admissible", "admissibility_would_refuse", "exclusion_gate_would_refuse",
                "held_target_mask"} & set(rec_off)
    ag = _ag3s(arm, gripper_max_opening=0.03, oracle_admissible=True)
    cs, dbg = _run(ag, scene)
    assert dbg["grounding"].status.value == "ok" and _near_sphere(dbg["grounding"].target.points).all()
    rec = cs.metrics["grounding_source"]
    assert rec["oracle_admissible"] is True and [r["reason"] for r in rec["admissibility_would_refuse"]] == ["not_graspable"]
    assert "exceeds the max opening" in rec["exclusion_gate_would_refuse"]["detail"]
    assert ag._gated_manipulated is not None                     # the exclusion geometry was built from it
    assert not any("invariant_violation" in n for n in cs.notes)
    json.dumps(rec)


# ============================================================================ 4 F2 mask
def _cams(scene, arm):
    from AG3S.ag3s.fields.esdf import CameraDepth

    out = []
    for cam in scene.views:
        obs = next(o for o in scene.observations if o.camera_id == cam)
        d = np.asarray(obs.depth, np.float64)
        out.append(CameraDepth(name=cam.value, depth=d, camera_intrinsics=np.asarray(obs.camera_intrinsics),
                               T_base_cam=np.eye(4), robot_mask=np.zeros(d.shape, bool), device_depth=object()))
    return out


def _mask(seg, cams, *, attached=None, grasp=False, grounding=None):
    from AG3S.ag3s.runtime.pipeline import AG3S

    fake = types.SimpleNamespace(_oracle_pending=seg, _attached=attached, _grasp_active=grasp,
                                 _grounding_frame=grounding)
    return AG3S._oracle_held_target_mask(fake, cams)


def test_f2_mask_inactive_closing_and_held(scene, arm):
    seg, cams = _seg(scene), _cams(scene, arm)
    g = {}
    assert _mask(seg, cams, grounding=g) is cams and g["held_target_mask"] == {
        "active": False, "phase": None, "n_px": {}, "n_px_new": {}}
    for kw, phase in (({"grasp": True}, "closing"), ({"attached": object()}, "held")):
        g = {}
        out = _mask(seg, cams, grounding=g, **kw)
        rec = g["held_target_mask"]
        assert rec["active"] and rec["phase"] == phase and rec["why"] is None
        for c0, c1 in zip(cams, out):
            target = seg["labels"][c0.name] == CODE_TARGET
            assert np.array_equal(np.asarray(c1.robot_mask), target)
            assert rec["n_px"][c0.name] == rec["n_px_new"][c0.name] == int(target.sum())
            assert (c1.device_depth is None) == bool(target.any())   # the host depth carries the new mask
            assert np.array_equal(np.asarray(c1.depth), np.asarray(c0.depth))
        assert sum(rec["n_px"].values()) > 0
    g = {}
    assert _mask(None, cams, grasp=True, grounding=g) is cams and g["held_target_mask"]["why"] == "no_segmentation"


# ============================================================================ 5 F2 pipeline
def _capture(ag):
    seen = []

    def _no_esdf(depth_cameras, *_a, **_k):
        seen.append(list(depth_cameras))
        return None, ["test: no ESDF"]
    ag._build_esdf = _no_esdf
    return seen


@pytest.mark.parametrize("grasp", [False, True])
def test_f2_pipeline_masks_the_code1_pixels_only_while_grasping(arm, scene, grasp):
    from AG3S.ag3s.runtime.pipeline import AG3S

    def make(**kw):
        cfg = CFG.with_overrides({"collision_backend": "esdf",
                                  "clustering": {**ON, **kw}, "esdf": {"target_extent": "oracle_visible"}})
        return AG3S(cfg, robot_model=arm)
    off, on = make(), make(oracle_held_target=True)
    seen_off, seen_on = _capture(off), _capture(on)
    cs0, d0 = _run(off, scene, frames=2, grasp=grasp)
    cs1, d1 = _run(on, scene, frames=2, grasp=grasp)
    seg = _seg(scene)
    for c0, c1 in zip(seen_off[-1], seen_on[-1]):
        m0 = np.asarray(c0.robot_mask, bool)
        m1 = np.asarray(c1.robot_mask, bool)
        target = seg["labels"][c0.name] == CODE_TARGET
        if grasp:
            assert np.array_equal(m1, m0 | target) and target.any()
        else:
            assert np.array_equal(m1, m0)
    # the fused cloud and grounding are untouched
    assert np.array_equal(d0["filtered_cloud"].points, d1["filtered_cloud"].points)
    assert d0["grounding"].status == d1["grounding"].status
    r0, r1 = cs0.metrics["grounding_source"], cs1.metrics["grounding_source"]
    assert "held_target_mask" not in r0 and r1["held_target_mask"]["active"] is grasp
    assert {k: v for k, v in r1.items() if k != "held_target_mask"} == r0
