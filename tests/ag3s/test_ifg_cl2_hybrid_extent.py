"""IFG-CL2 — hybrid target extent (`esdf.target_extent = "attention_surface_hybrid"`): the `ball_owned` range, protected
by the attention surface. Default off; off must be bit-identical.

| # | property |
|---|---|
| 1 | config: default unchanged · round trip · `ball_owned` required · identity rule validated; CLI → esdf keys only when given; start-up refusals; record meta and server metadata only when on |
| 2 | `surface_protect_split`: removed ⊆ `ball_owned` removed · sign ⊆ · a neighbour the cloud missed is protected · the target's unseen side (far from every node) stays removed · far-away other nodes change nothing |
| 3 | `_ball_exclusion(protect=)` (no torch): counts · trace (`protected_ijk`, `nontarget_kept_ijk` ⊇ protected); without `protect` the call is as before; `update()` refuses `surface_protect` without `ball_owned` |
| 4 | identity guard: instance (majority · no selected node · no node near) · centre (outside the ball) · rule selection; `hybrid_evidence` reasons (abstain · no ball · identity off) |
| 5 | `hybrid_samples`: target = selected nodes, other = not selected ∧ not support |
| 6 | pipeline (fake builder, ray-cast rig): the LOCK selection protects the `ball_owned` split (`surface_protect` kwarg, no oracle memory, records); abstain / identity off → no kwarg + reason; default never calls it |
| 7 | constraint record: `protected_ijk` · `attention_surface_hybrid` · `surface_protect` only when present |
| 8 | (storage) offline replay: the recorded C1aN `ball_owned` trace is reproduced from the record; a SURF chunk's code images are reproduced pixel-exactly and the hybrid removes a subset of `ball_owned` |
| 9 | IFG-CL2b: other nodes within V × fusion voxel of the manipulated geometry leave the other evidence (≤, the target's own unselected surface stops protecting target seeds, a farther neighbour stays protected); `0` = IFG-CL2 bit-identical (evidence · split · record); no geometry → no exclusion; config default 1.0 · validation; CLI / record meta / server metadata; pipeline passes the config value |
"""

from __future__ import annotations

import os
import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import (
    CuroboFieldBuilder, OwnedEvidence, SurfaceProtectEvidence, TargetBall, _ball_owned_split, surface_protect_split)
from AG3S.ag3s.runtime.attention_surface_hybrid import (
    HybridSamples, hybrid_evidence, hybrid_samples, identity_guard)

VS = 0.005
N = 40
ORIGIN = np.array([0.40, 0.20, 0.78])          # first voxel centre
A = np.array([0.50, 0.30, 0.875])              # target centre
B = np.array([0.50, 0.345, 0.875])             # neighbour centre (45 mm: surfaces 5 mm apart)
R = 0.020
TABLE_Z = 0.85


def _centres_of(ijk):
    return ORIGIN + np.asarray(ijk, np.float64) * VS


def _is_seed(c):
    """seed = |sdf| < 0.75 vs of either sphere, above the table."""
    da = np.abs(np.linalg.norm(c - A, axis=1) - R)
    db = np.abs(np.linalg.norm(c - B, axis=1) - R)
    return (np.minimum(da, db) < 0.75 * VS) & (c[:, 2] > TABLE_Z + VS)


def _owner(c):
    return np.where(np.linalg.norm(c - A, axis=1) <= np.linalg.norm(c - B, axis=1), 0, 1)


def _sphere(c, n, seed, *, z_min=-1.0, z_max=1.0, r=R):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(20 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[(v[:, 2] >= z_min) & (v[:, 2] <= z_max)][:n]
    return c + r * v


def _nodes(c, *, z_min=-1.0, step=VS):
    """5 mm 'graph nodes' on a sphere surface (voxel centroids of a dense sampling)."""
    p = _sphere(c, 20000, 7, z_min=z_min)
    k = np.floor(p / step).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    out = np.zeros((inv.max() + 1, 3))
    np.add.at(out, inv, p)
    return out / np.bincount(inv)[:, None]


def _ball(radius=0.035):
    return TargetBall(centre=A, radius=radius, z_min=TABLE_Z + VS)


def _scene():
    b = CuroboFieldBuilder(AG3SConfig.from_dict({"esdf": {"backend": "curobo", "fine_voxel_size": VS,
                                                            "target_exclusion": "ball_owned"}}).esdf)
    cand = b._ball_candidates(_ball(), ORIGIN, VS, (N, N, N))
    c = _centres_of(cand)
    return b, cand, c, _is_seed(c), _owner(c)


def _owned(target=None, nontarget=None, **kw):
    # the cloud saw the target's top and **missed** the neighbour (no non-target evidence) — the case to protect
    return OwnedEvidence(target_points=_sphere(A, 60, 0, z_min=0.2) if target is None else target,
                         nontarget_points=np.zeros((0, 3)) if nontarget is None else nontarget,
                         max_dist_voxels=3.0, **kw)


def _protect(other=None):
    return SurfaceProtectEvidence(target_points=_nodes(A, z_min=0.0),                       # visible upper half
                                  other_points=_nodes(B, z_min=-0.3) if other is None else other)


# ============================================================================ 1 config · CLI · metadata


def test_config_default_round_trip_and_validation():
    d = AG3SConfig().esdf
    assert d.target_extent == "estimated" and d.surface_hybrid_identity == "both"
    cfg = AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_extent": "attention_surface_hybrid",
                                         "surface_hybrid_identity": "centre"}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    with pytest.raises(AG3SConfigError, match="ball_owned"):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface_hybrid"}})
    with pytest.raises(AG3SConfigError, match="surface_hybrid_identity"):
        AG3SConfig.from_dict({"esdf": {"surface_hybrid_identity": "gt"}})
    # F1 · F2 compose with the hybrid (they change the `ball_owned` split it protects) — accepted
    AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_extent": "attention_surface_hybrid",
                                   "nontarget_evidence": "surface", "owned_ambiguous": "remove"}})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def _reject(*argv):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])
    reject_bad_flag_combinations(ap, args)
    return args


ON = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface-hybrid",
      "--target-field-policy", "exclude-authorized", "--attention-source", "noun_target:L10H7")


def test_cli_maps_to_esdf_keys_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    assert target_exclusion_section(_parse()) == {} and ifg_cl1_record_meta(_parse()) == {}
    args = _parse(*ON)
    sec = target_exclusion_section(args)
    assert sec == {"target_exclusion": "ball_owned", "target_extent": "attention_surface_hybrid"}
    assert ifg_cl1_record_meta(args) == {"target_extent": "attention_surface_hybrid", "surface_hybrid_identity": "both",
                                         "surface_hybrid_other_exclude_voxels": 1.0}   # IFG-CL2b: value in effect
    args = _parse(*ON, "--hybrid-identity", "instance", "--robot-mask", "render")
    sec = target_exclusion_section(args)
    assert sec["surface_hybrid_identity"] == "instance" and sec["robot_mask"] == "render"
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert {k: getattr(cfg, k) for k in sec} == sec


def test_cli_refusals(capsys):
    args = _reject(*ON)
    assert args.target_extent == "attention-surface-hybrid"
    _reject(*ON, "--robot-mask", "render", "--hybrid-identity", "centre")
    cases = (
        (("--esdf-backend", "curobo", "--target-extent", "attention-surface-hybrid", "--target-field-policy",
          "exclude-authorized", "--attention-source", "noun_target:L10H7"), "ball-owned"),
        (ON[:-2], "noun_target:L10H7"),
        (ON[:-2] + ("--attention-source", "noun_target:L10H6"), "noun_target:L10H7"),
        (ON[:6] + ON[8:], "--target-field-policy"),
        (("--hybrid-identity", "both"), "--hybrid-identity"),
        (ON + ("--no-attention",), "attention"),
        # the oracle-path flags are not read by the hybrid
        (ON + ("--oracle-max-dist-voxels", "5"), "oracle"),
    )
    for argv, needle in cases:
        with pytest.raises(SystemExit) as e:
            _reject(*argv)
        assert e.value.code == 2
        assert needle in capsys.readouterr().err, argv


def test_server_metadata_only_when_on():
    from AG3S.trajopt import wire

    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    base = t46e._safe_policy_with().metadata
    on = t46e._safe_policy_with(target_exclusion="ball_owned", target_extent="attention_surface_hybrid").metadata
    assert wire.ORACLE_SEG_REQUIRED not in on
    assert on.pop("target_extent") == "attention_surface_hybrid" and on.pop("surface_hybrid_identity") == "both"
    assert on.pop("surface_hybrid_other_exclude_voxels") == 1.0   # IFG-CL2b
    assert on == base


# ============================================================================ 2 the split


def test_protection_keeps_the_missed_neighbour_and_never_widens_the_range():
    _, cand, c, seed, owner = _scene()
    owned = _owned()
    est = _ball_owned_split(c, seed, _ball(), owned, VS)
    hyb = surface_protect_split(c, seed, _ball(), owned, _protect(), VS)
    nb = seed & (owner == 1)
    assert int((est["removed"] & nb).sum()) > 10, "the cloud missed the neighbour: ball_owned removes its seeds"
    assert not np.any(hyb["removed"] & ~est["removed"]), "removed ⊆ ball_owned removed"
    assert not np.any(hyb["sign"] & ~est["sign"]), "sign ⊆ ball_owned sign"
    assert np.array_equal(hyb["protected"], est["removed"] & ~hyb["removed"])
    assert int((hyb["removed"] & nb).sum()) <= 0.1 * int((est["removed"] & nb).sum()), "neighbour protected"
    # the target's unseen lower side, farther than max_dist from every node: still removed
    d_t = np.min(np.linalg.norm(c[:, None] - _protect().target_points[None], axis=2), axis=1)
    d_o = np.min(np.linalg.norm(c[:, None] - _protect().other_points[None], axis=2), axis=1)
    hidden = seed & (owner == 0) & (np.minimum(d_t, d_o) > 3 * VS)
    assert int(hidden.sum()) > 5
    assert np.all(hyb["removed"][hidden] == est["removed"][hidden]) and np.all(est["removed"][hidden])
    # classes of `est` are kept (the record's ball_owned counts keep their meaning)
    for k in ("target", "nontarget", "ambiguous", "unowned", "d_t", "d_o"):
        assert np.array_equal(hyb[k], est[k]), k
    # no target seed nearer the selected nodes (by more than the tie) is protected
    assert not np.any(hyb["protected"] & (d_t < d_o - VS))


def test_far_other_nodes_change_nothing():
    _, cand, c, seed, owner = _scene()
    owned = _owned()
    est = _ball_owned_split(c, seed, _ball(), owned, VS)
    hyb = surface_protect_split(c, seed, _ball(), owned, _protect(other=_nodes(A + [0.5, 0.5, 0.0])), VS)
    assert np.array_equal(hyb["removed"], est["removed"]) and not hyb["protected"].any()


# ============================================================================ 3 builder (no torch)


def _fake_builder(seed_of):
    b, *_ = _scene()
    cleared = []
    b._seed_mask = lambda itg, cand: seed_of(cand)
    b._clear_seeds = lambda itg, idx, shape: (cleared.append(np.asarray(idx)) or (len(idx), None))
    return b, cleared


def test_ball_exclusion_with_and_without_protection():
    seed_of = lambda cand: _is_seed(_centres_of(cand))
    b, cleared = _fake_builder(seed_of)
    n0, sign0, ex0 = b._ball_exclusion(None, _ball(), ORIGIN, VS, (N, N, N), owned=_owned())
    n1, sign1, ex1 = b._ball_exclusion(None, _ball(), ORIGIN, VS, (N, N, N), owned=_owned(), protect=_protect())
    assert "surface_protect" not in ex0["counts"] and "protected_ijk" not in ex0["trace"], "default call as before"
    c0, c1 = ex0["counts"], ex1["counts"]
    sp = c1["surface_protect"]
    assert n1 == c1["n_removed"] == n0 - sp["n_protected"] and sp["n_protected"] > 10
    assert sp["n_removed_estimated"] == n0
    assert c1["n_kept_nontarget"] == c0["n_kept_nontarget"] + sp["n_protected"]
    for k in ("n_seeds_in_ball", "n_target_owned", "n_ambiguous"):
        assert c1[k] == c0[k], k
    key = lambda g: set(map(tuple, np.asarray(g)))
    t0, t1 = ex0["trace"], ex1["trace"]
    assert key(t1["removed_ijk"]) <= key(t0["removed_ijk"])
    assert key(t1["protected_ijk"]) == key(t0["removed_ijk"]) - key(t1["removed_ijk"])
    assert key(t1["protected_ijk"]) <= key(t1["nontarget_kept_ijk"])
    assert key(sign1) <= key(sign0)
    assert np.array_equal(cleared[-1], t1["removed_ijk"])
    assert {"split_ms", "n_target_points", "n_other_points", "srf"} <= set(sp)


def test_update_refuses_protection_without_ball_owned():
    b = CuroboFieldBuilder(AG3SConfig.from_dict({"esdf": {"backend": "curobo", "fine_voxel_size": VS}}).esdf)
    with pytest.raises(ValueError, match="ball_owned"):
        b.update([object()], target_points=A[None], target_free_points=A[None], target_free_ball=_ball(),
                 surface_protect=_protect())
    b2, *_ = _scene()
    with pytest.raises(ValueError, match="oracle_evidence"):
        b2.update([object()], target_points=A[None], target_free_points=A[None], target_free_ball=_ball(),
                  surface_protect=_protect(), oracle_evidence=object())
    with pytest.raises(ValueError, match="선택 영역"):
        SurfaceProtectEvidence(target_points=np.zeros((0, 3)))


# ============================================================================ 4 identity guard · evidence


def _samples(target_c=A, other_c=B):
    t = _nodes(target_c, z_min=0.0)
    o = _nodes(other_c, z_min=-0.3)
    table = np.stack(np.meshgrid(np.arange(0.4, 0.6, VS), np.arange(0.2, 0.4, VS), [TABLE_Z]), -1).reshape(-1, 3)
    pos = np.vstack([t, o, table])
    sel = np.r_[np.ones(len(t), bool), np.zeros(len(o) + len(table), bool)]
    return HybridSamples(target=t, other=o, node_pos=pos, node_selected=sel, n_support_other=len(table))


def test_identity_guard_rules():
    geo = _sphere(A, 50, 3, z_min=0.2)                       # the manipulated geometry: the target's top
    ok, rec = identity_guard(geo, _samples(), _ball(), radius=0.01)
    assert ok and rec["instance"]["ok"] and rec["centre"]["ok"] and rec["instance"]["fraction"] > 0.9
    # the region is on the neighbour: the manipulated points land on unselected nodes
    wrong = _samples(target_c=B, other_c=A)
    big = _ball(radius=0.055)                                # a pear-sized ball (CL1 records: 54-59 mm)
    ok, rec = identity_guard(geo, wrong, big, radius=0.01)
    assert not ok and rec["why"].startswith("instance:no_selected_node")
    assert rec["centre"]["ok"] is True, "~46 mm < ball radius: the centre rule alone would have let it through"
    assert identity_guard(geo, wrong, big, radius=0.01, rule="centre")[0] is True
    far = _samples(target_c=A + [0.3, 0, 0], other_c=A)
    ok, rec = identity_guard(geo, far, _ball(), radius=0.01, rule="centre")
    assert not ok and rec["why"] == "centre:outside_ball" and rec["instance"]["ok"] is False
    # half on the region, half off: no majority
    s = _samples()
    half = np.vstack([_sphere(A, 20, 4, z_min=0.5), _sphere(B, 40, 5, z_min=0.5)])
    ok, rec = identity_guard(half, s, _ball(), radius=0.01, rule="instance")
    assert not ok and rec["why"] == "instance:no_majority"
    ok, rec = identity_guard(A[None] + [2.0, 0, 0], s, _ball(), radius=0.01, rule="instance")
    assert not ok and rec["why"] == "instance:no_node_near"
    with pytest.raises(ValueError):
        identity_guard(geo, s, _ball(), radius=0.01, rule="gt")


def test_hybrid_evidence_reasons():
    geo = _sphere(A, 50, 3, z_min=0.2)
    ev, rec = hybrid_evidence(None, {"why": "no_region"}, geo, _ball(), radius=0.01)
    assert ev is None and rec["why"] == "attention_surface_abstained (no_region)" and "ms" in rec
    ev, rec = hybrid_evidence(_samples(), {}, geo, None, radius=0.01)
    assert ev is None and rec["why"] == "no_ball"
    ev, rec = hybrid_evidence(_samples(target_c=B, other_c=A), {}, geo, _ball(), radius=0.01)
    assert ev is None and rec["why"].startswith("identity_off (instance:") and not rec["applied"]
    ev, rec = hybrid_evidence(_samples(), {}, geo, _ball(), radius=0.01)
    assert rec["applied"] and isinstance(ev, SurfaceProtectEvidence)
    assert len(ev.target_points) == rec["n_target_points"] and len(ev.other_points) == rec["n_other_points"]
    assert rec["n_support_points_excluded"] > 0


def test_hybrid_samples_excludes_support_nodes():
    pos = np.arange(30, dtype=float).reshape(10, 3)
    geo = types.SimpleNamespace(pos=pos, support=np.r_[np.zeros(6, bool), np.ones(4, bool)])
    res = types.SimpleNamespace(codes={"head": np.zeros((2, 2), np.uint8)}, geometry=geo,
                                selected_nodes=np.array([0, 1, 8]))
    s = hybrid_samples(res)
    assert np.array_equal(s.target, pos[[0, 1, 8]]) and np.array_equal(s.other, pos[2:6])
    assert s.n_support_other == 3 and s.node_selected.sum() == 3
    assert hybrid_samples(types.SimpleNamespace(codes=None, geometry=geo, selected_nodes=[])) is None
    assert hybrid_samples(types.SimpleNamespace(codes={}, geometry=None, selected_nodes=[])) is None


# ============================================================================ 6 pipeline wiring (fake builder)


def _cl1():
    return pytest.importorskip("tests.ag3s.test_ifg_cl1_attention_surface")


def test_pipeline_hybrid_protects_the_ball_owned_split(monkeypatch):
    cl1 = _cl1()
    fw, ag3s = cl1._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface_hybrid")
    obs, gt, nb = cl1._raycast_rig(fw)
    cams = cl1._cameras(obs)
    ag3s._attention_surface_hybrid_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    surf, s = ag3s._surface_frame, ag3s._surface_hybrid
    assert surf["applied"] and surf["arm"] == "top1-only-ns:N" and "samples" in surf["ms"]
    assert ag3s._oracle_memory is None, "the hybrid does not use the oracle memory"
    assert ag3s._surface_codes is not None and set(ag3s._surface_codes) == {"head", "left_wrist", "right_wrist"}
    # target samples on the apple; the neighbour among the other samples; the table is support (not other)
    d_app = np.abs(np.linalg.norm(s.target - fw.APPLE, axis=1) - fw.APPLE_R)
    assert np.median(d_app) < 0.004
    assert np.any(np.abs(np.linalg.norm(s.other - nb, axis=1) - fw.APPLE_R) < 0.004)
    assert np.mean(np.abs(s.other[:, 2] - fw.TABLE_Z) < 0.003) < 0.05 and s.n_support_other > 100
    field, _ = fw._build(ag3s)
    kw = ag3s._esdf_builder.calls[-1]
    assert "oracle_evidence" not in kw and isinstance(kw["surface_protect"], SurfaceProtectEvidence)
    assert np.array_equal(kw["surface_protect"].target_points, s.target)
    tf = field.stats["target_free"]
    h = tf["attention_surface_hybrid"]
    assert h["applied"] and h["identity"]["instance"]["ok"] and h["identity"]["centre"]["ok"]
    assert tf["attention_surface"]["applied"] and "oracle" not in tf
    ag3s.reset()
    assert ag3s._surface_hybrid is None and ag3s._surface_frame is None


def test_pipeline_hybrid_abstain_and_identity_off(monkeypatch):
    cl1 = _cl1()
    fw, ag3s = cl1._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface_hybrid")
    obs, _, _ = cl1._raycast_rig(fw)
    for o in obs:
        o.attention_map = None
    cams = cl1._cameras(obs)
    ag3s._attention_surface_hybrid_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    assert ag3s._surface_hybrid is None and ag3s._surface_codes is None
    field, _ = fw._build(ag3s)
    assert "surface_protect" not in ag3s._esdf_builder.calls[-1]
    h = field.stats["target_free"]["attention_surface_hybrid"]
    assert not h["applied"] and h["why"] == "attention_surface_abstained (no_attention:head)"
    # identity off: the region is somewhere else
    s = _samples(target_c=fw.APPLE + [0.3, 0.0, 0.0], other_c=fw.APPLE)
    ag3s._surface_hybrid, ag3s._surface_frame = s, {"applied": True}
    field, _ = fw._build(ag3s)
    assert "surface_protect" not in ag3s._esdf_builder.calls[-1]
    h = field.stats["target_free"]["attention_surface_hybrid"]
    assert h["why"].startswith("identity_off (") and "centre:outside_ball" in h["why"]
    # centre-only rule: the same frame is still refused (the region's centre is 300 mm away)
    ag3s.config = ag3s.config.with_overrides({"esdf": {"surface_hybrid_identity": "centre"}})
    fw._build(ag3s)
    assert "surface_protect" not in ag3s._esdf_builder.calls[-1]


def test_pipeline_default_never_calls_the_hybrid(monkeypatch):
    from AG3S.ag3s.runtime import attention_surface_hybrid as H

    def boom(*a, **k):
        raise AssertionError("called with the default config")

    monkeypatch.setattr(H, "hybrid_evidence", boom)
    monkeypatch.setattr(H, "hybrid_samples", boom)
    for esdf in ({"target_exclusion": "ball_owned"}, {"target_exclusion": "ball_owned", "target_extent": "estimated"},
                 {"target_exclusion": "ball_owned", "target_extent": "attention_surface"}):
        fw, ag3s = _cl1()._wired(monkeypatch, **esdf)
        field, _ = fw._build(ag3s)
        assert "surface_protect" not in ag3s._esdf_builder.calls[-1]
        assert "attention_surface_hybrid" not in field.stats["target_free"] and ag3s._surface_hybrid is None


# ============================================================================ 7 record


def test_record_keys_only_when_present():
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    trace = {"tier": "fine_no_target", "method": "ball_owned", "origin": ORIGIN, "voxel_size": VS,
             **{k: np.zeros((2, 3), np.int64) for k in ("removed_ijk", "nontarget_kept_ijk", "ambiguous_ijk",
                                                        "unowned_kept_ijk")}}
    tf = {"mode": "ball", "exclusion_method": "ball_owned", "n_removed": 2}
    payload: dict = {}
    info = ConstraintRecordWriter._write_exclusion_trace(
        types.SimpleNamespace(stats={"target_free": dict(tf)}, exclusion_trace=dict(trace)), payload)
    assert "target_free_protected_ijk" not in payload
    assert not {"attention_surface_hybrid", "surface_protect"} & set(info["target_exclusion"])
    payload = {}
    tf2 = {**tf, "attention_surface_hybrid": {"applied": True, "ms": 1.0}, "surface_protect": {"n_protected": 3},
           "attention_surface": {"applied": True}}
    info = ConstraintRecordWriter._write_exclusion_trace(
        types.SimpleNamespace(stats={"target_free": tf2},
                              exclusion_trace={**trace, "protected_ijk": np.ones((3, 3), np.int64)}), payload)
    assert payload["target_free_protected_ijk"].shape == (3, 3) and payload["target_free_protected_ijk"].dtype == np.int16
    te = info["target_exclusion"]
    assert te["attention_surface_hybrid"]["applied"] and te["surface_protect"]["n_protected"] == 3
    assert "oracle" not in te


# ============================================================================ 8 offline replay (storage)

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
CL1 = STORE / "results/development/20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1"
needs_cl1 = pytest.mark.skipif(not (CL1 / "analysis/SURF_ep1769_s17691.json").exists(),
                               reason="IFG-CL1 closed-loop record (shared storage) not mounted")


@needs_cl1
def test_replay_reproduces_the_record():
    from AG3S.ag3s.experiments.studies.cl2_hybrid_replay import run_one

    c = run_one(CL1, "C1aN_ep1769_s17691", only_k=(0, 9, 12))
    assert [r["k"] for r in c["chunks"]] == [0, 9, 12]
    assert all(r["replay_ok"] for r in c["chunks"]), "the recorded ball_owned trace is reproduced"
    assert all(r["estimated"] == {k: v for k, v in r["recorded"].items() if k != "keptseeds_kept_target"}
               for r in c["chunks"])
    s = run_one(CL1, "SURF_ep1769_s17691", only_k=(0, 18))
    for r in s["chunks"]:
        assert r["codes_parity_px"] == 0, "the recorded code images are reproduced from the client depth"
        for rule, h in r["hybrid"].items():
            assert h["n_removed"] <= r["estimated"]["n_removed"], rule
            assert h["removed_neighbour"] <= r["estimated"]["removed_neighbour"], rule
            assert h["kept_target"] >= r["estimated"]["kept_target"], rule


# ============================================================================ 9 IFG-CL2b — other-evidence exclusion near the manipulated geometry

from AG3S.ag3s.runtime.attention_surface_hybrid import other_near_geometry  # noqa: E402

FUSION = 0.010


def test_cl2b_other_near_geometry_rule():
    geo = np.array([[0.0, 0.0, 0.0]])
    other = np.array([[0.0, 0.0, 0.0099], [0.0, 0.0, 0.010], [0.0, 0.0, 0.0101], [0.5, 0.0, 0.0]])
    assert other_near_geometry(other, geo, FUSION).tolist() == [True, True, False, False], "within = ≤ radius"
    assert not other_near_geometry(other, geo, 0.0).any(), "0 → nothing excluded"
    assert not other_near_geometry(other, np.zeros((0, 3)), FUSION).any(), "no geometry → nothing excluded"
    assert other_near_geometry(np.zeros((0, 3)), geo, FUSION).shape == (0,)


def _partial_region_case():
    """The IFG-CL2 failure: the region covers only the target's top cap; the target's unselected side is "other"."""
    nodes_a = _nodes(A, z_min=-0.3)
    cap = (nodes_a[:, 2] - A[2]) / R >= 0.6
    nodes_b = _nodes(B + [0.0, 0.015, 0.0], z_min=-0.3)          # neighbour 20 mm from the target surface
    table = np.stack(np.meshgrid(np.arange(0.4, 0.6, VS), np.arange(0.2, 0.4, VS), [TABLE_Z]), -1).reshape(-1, 3)
    pos = np.vstack([nodes_a, nodes_b, table])
    sel = np.r_[cap, np.zeros(len(nodes_b) + len(table), bool)]
    s = HybridSamples(target=nodes_a[cap], other=np.vstack([nodes_a[~cap], nodes_b]), node_pos=pos,
                      node_selected=sel, n_support_other=len(table))
    geo = _sphere(A, 400, 11, z_min=-0.3)                         # the manipulated geometry: the target's visible part
    return s, geo, int((~cap).sum())


def test_cl2b_target_own_surface_stops_protecting():
    s, geo, n_own = _partial_region_case()
    ev0, rec0 = hybrid_evidence(s, {}, geo, _ball(), radius=FUSION, rule="centre")
    ev1, rec1 = hybrid_evidence(s, {}, geo, _ball(), radius=FUSION, rule="centre", other_exclude_voxels=1.0)
    assert rec0["applied"] and rec1["applied"] and "other_exclude" not in rec0
    ox = rec1["other_exclude"]
    assert ox == {"voxels": 1.0, "radius_mm": 10.0, "n_before": len(s.other), "n_excluded": ox["n_excluded"]}
    assert ox["n_excluded"] == n_own, "exactly the target's own unselected nodes leave the other evidence"
    assert rec1["n_other_points"] == len(ev1.other_points) == len(s.other) - n_own
    assert np.array_equal(ev1.target_points, ev0.target_points), "target evidence unchanged"
    # the split: target seeds lose their (wrong) protection; the neighbour's seeds stay protected
    nb_c = B + [0.0, 0.015, 0.0]
    b = CuroboFieldBuilder(AG3SConfig.from_dict({"esdf": {"backend": "curobo", "fine_voxel_size": VS,
                                                            "target_exclusion": "ball_owned"}}).esdf)
    ball = _ball(radius=0.05)
    cand = b._ball_candidates(ball, ORIGIN, VS, (N, N, N))
    c = _centres_of(cand)
    da = np.abs(np.linalg.norm(c - A, axis=1) - R)
    db = np.abs(np.linalg.norm(c - nb_c, axis=1) - R)
    seed = (np.minimum(da, db) < 0.75 * VS) & (c[:, 2] > TABLE_Z + VS)
    tgt = seed & (da <= db)
    owned = OwnedEvidence(target_points=_sphere(A, 60, 0, z_min=0.2), nontarget_points=np.zeros((0, 3)),
                          max_dist_voxels=6.0)
    h0 = surface_protect_split(c, seed, ball, owned, ev0, VS)
    h1 = surface_protect_split(c, seed, ball, owned, ev1, VS)
    assert int((h0["protected"] & tgt).sum()) > 10, "IFG-CL2: the target's own nodes protect target seeds"
    assert int((h1["protected"] & tgt).sum()) <= 0.1 * int((h0["protected"] & tgt).sum())
    assert int((h1["protected"] & ~tgt).sum()) == int((h0["protected"] & ~tgt).sum()) > 10, "neighbour still protected"
    assert not np.any(h0["removed"] & ~h1["removed"]), "less other evidence never protects more"
    assert not np.any(h1["removed"] & ~_ball_owned_split(c, seed, ball, owned, VS)["removed"]), "still ⊆ ball_owned"


def test_cl2b_zero_is_cl2_bit_identical():
    s, geo, _ = _partial_region_case()
    ev, rec = hybrid_evidence(s, {}, geo, _ball(), radius=FUSION, rule="centre")
    ev0, rec0 = hybrid_evidence(s, {}, geo, _ball(), radius=FUSION, rule="centre", other_exclude_voxels=0.0)
    rec.pop("ms"), rec0.pop("ms")
    assert rec == rec0 and ev.info == ev0.info
    assert np.array_equal(ev.target_points, ev0.target_points) and np.array_equal(ev.other_points, ev0.other_points)
    assert np.array_equal(ev0.other_points, s.other)
    _, cand, c, seed, _ = _scene()
    a, b2 = surface_protect_split(c, seed, _ball(), _owned(), ev, VS), surface_protect_split(c, seed, _ball(), _owned(),
                                                                                             ev0, VS)
    for k in ("removed", "sign", "protected"):
        assert np.array_equal(a[k], b2[k]), k


def test_cl2b_no_geometry_no_exclusion():
    s, _, _ = _partial_region_case()
    ev, rec = hybrid_evidence(s, {}, np.zeros((0, 3)), _ball(), radius=FUSION, rule="centre", other_exclude_voxels=1.0)
    assert rec["applied"] and rec["other_exclude"]["why"] == "no_geometry" and rec["other_exclude"]["n_excluded"] == 0
    assert np.array_equal(ev.other_points, s.other)
    # with the instance rule, no geometry already turns the protection off (IFG-CL2 behaviour, unchanged)
    ev, rec = hybrid_evidence(s, {}, np.zeros((0, 3)), _ball(), radius=FUSION, other_exclude_voxels=1.0)
    assert ev is None and rec["why"] == "identity_off (instance:no_points)" and "other_exclude" not in rec


def test_cl2b_config_and_cli():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    assert AG3SConfig().esdf.surface_hybrid_other_exclude_voxels == 1.0
    cfg = AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_extent": "attention_surface_hybrid",
                                         "surface_hybrid_other_exclude_voxels": 0.0}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in (-1.0, float("nan"), float("inf"), True, "1"):
        with pytest.raises(AG3SConfigError, match="surface_hybrid_other_exclude_voxels"):
            AG3SConfig.from_dict({"esdf": {"surface_hybrid_other_exclude_voxels": bad}})
    assert "surface_hybrid_other_exclude_voxels" not in target_exclusion_section(_parse(*ON)), "absent = config 1.0"
    args = _parse(*ON, "--hybrid-other-exclude-voxels", "0")
    assert target_exclusion_section(args)["surface_hybrid_other_exclude_voxels"] == 0.0
    assert ifg_cl1_record_meta(args)["surface_hybrid_other_exclude_voxels"] == 0.0
    _reject(*ON, "--hybrid-other-exclude-voxels", "0")


def test_cl2b_cli_refusals(capsys):
    for argv in (("--hybrid-other-exclude-voxels", "1.0"),
                 ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface",
                  "--target-field-policy", "exclude-authorized", "--attention-source", "noun_target:L10H7",
                  "--hybrid-other-exclude-voxels", "1.0"),
                 ON + ("--hybrid-other-exclude-voxels", "-1"),
                 ON + ("--hybrid-other-exclude-voxels", "nan")):
        with pytest.raises(SystemExit) as e:
            _reject(*argv)
        assert e.value.code == 2
        assert "--hybrid-other-exclude-voxels" in capsys.readouterr().err, argv


def test_cl2b_server_metadata_value():
    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    on = t46e._safe_policy_with(target_exclusion="ball_owned", target_extent="attention_surface_hybrid",
                                surface_hybrid_other_exclude_voxels=0.0).metadata
    assert on["surface_hybrid_other_exclude_voxels"] == 0.0
    assert "surface_hybrid_other_exclude_voxels" not in t46e._safe_policy_with().metadata


def test_cl2b_pipeline_passes_the_config_value(monkeypatch):
    cl1 = _cl1()
    for v, expect_key in ((None, True), (0.0, False)):
        kw = {} if v is None else {"surface_hybrid_other_exclude_voxels": v}
        fw, ag3s = cl1._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface_hybrid", **kw)
        obs, _, _ = cl1._raycast_rig(fw)
        cams = cl1._cameras(obs)
        ag3s._attention_surface_hybrid_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
        s = ag3s._surface_hybrid
        field, _ = fw._build(ag3s)
        h = field.stats["target_free"]["attention_surface_hybrid"]
        ev = ag3s._esdf_builder.calls[-1]["surface_protect"]
        assert h["applied"] and ("other_exclude" in h) is expect_key
        if expect_key:
            assert h["other_exclude"]["voxels"] == 1.0 and h["other_exclude"]["radius_mm"] == 10.0
            assert len(ev.other_points) == len(s.other) - h["other_exclude"]["n_excluded"]
            assert h["other_exclude"]["n_excluded"] > 0, "the apple's unselected nodes near its geometry"
        else:
            assert np.array_equal(ev.other_points, s.other), "0 = IFG-CL2"
