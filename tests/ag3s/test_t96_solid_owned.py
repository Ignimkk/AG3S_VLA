"""T96 — target solid ownership of the hand layer (`esdf.surface_solid_owned`, `--surface-solid-owned`), default off.
Off must be the oracle split (T46e / T78 / T78f) call for call.

| # | property |
|---|---|
| 1 | config defaults (off · τ 5 · δ 0) · round trip · refusals (needs attention_surface + A3 + A5; refuses hull fill · contact allowance · sticky memory); CLI flags → esdf keys / record meta only when given; start-up refusals |
| 2 | off: `oracle_visible_split` = the pre-T96 rule re-derived, no `solid` key; the pipeline never calls the T96 steps, stores no region, records no `solid` key, builds the same call as an explicit-off config; the tier is value-for-value the oracle tier |
| 3 | pure, non-ball shapes (cylinder · box · non-convex L) on a table with a neighbour ≥ 5 mm away: the solid removes the hidden back and the seeds between the visible surface and the table (above the guard), keeps the neighbour, the τ band and the table; invariants of every path (seed only, `d_N ≤ τ` never removed, below `z_floor` never removed, solid path inside `S ⊕ δ`) |
| 4 | tier (fake cuRobo, torch): solid counts by path, trace, `oracle_split.solid`, candidates beyond `max_dist` |
| 5 | pipeline: the region is stored at an accepted request, kept through A4 hold, cleared on reset; R⊥ on the support plane; `target_exclusion.attention_surface.solid` (pipeline record + builder voxel counts) in the constraint record JSON; no region → τ-only evidence |
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import (OracleVisibleEvidence, SolidOwned, oracle_candidates,
                                             oracle_visible_split, solid_candidates, solid_facet_value)
from AG3S.ag3s.runtime import attention_surface_post as P

ON_BASE = {"target_extent": "attention_surface", "surface_keep_last": True, "surface_no_ball": True}
P0M_SURF = {"surface_target_check": "overlap", "surface_keep_last": True, "surface_grasp_hold": True,
            "surface_no_ball": True}


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


# ============================================================================ 1 config · CLI


def test_config_defaults_round_trip_and_refusals():
    e = AG3SConfig().esdf
    assert (e.surface_solid_owned, e.surface_solid_protect_mm, e.surface_solid_dilate_mm) == (False, 5.0, 0.0)
    cfg = AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_owned": True, "surface_solid_dilate_mm": 5.0}})
    assert cfg.esdf.surface_solid_owned and cfg.esdf.surface_solid_dilate_mm == 5.0
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in ({}, {"target_extent": "attention_surface"},
                {"target_extent": "attention_surface", "surface_keep_last": True},
                {"target_extent": "oracle_visible"}):
        with pytest.raises(AG3SConfigError, match="surface_solid_owned"):
            AG3SConfig.from_dict({"esdf": {**bad, "surface_solid_owned": True}})
    for clash in ({"surface_hull_fill": True}, {"surface_contact_allowance_mm": 10.0},
                  {"surface_memory_target_sticky": True}):
        with pytest.raises(AG3SConfigError, match="대체"):
            AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, "surface_solid_owned": True, **clash}})
        AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, **clash}})   # alone: still fine
    for name in ("surface_solid_protect_mm", "surface_solid_dilate_mm"):
        for v in (-1.0, float("nan"), float("inf"), True, "5"):
            with pytest.raises(AG3SConfigError, match=name):
                AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_owned": True, name: v}})
    with pytest.raises(AG3SConfigError, match="surface_solid_owned "):
        AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_owned": 1}})


def test_cli_flags_map_to_esdf_keys_and_meta_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    base = A._parse(*A.SURF, *A.ALL)
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert not {k for k in set(sec0) | set(meta0) if k.startswith("surface_solid")}
    on = A._parse(*A.SURF, *A.ALL, "--surface-solid-owned", "--surface-solid-protect-mm", "5",
                  "--surface-solid-dilate-mm", "5")
    want = {"surface_solid_owned": True, "surface_solid_protect_mm": 5.0, "surface_solid_dilate_mm": 5.0}
    sec = target_exclusion_section(on)
    assert {k: v for k, v in sec.items() if k not in sec0} == want
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == want
    only = A._parse(*A.SURF, *A.ALL, "--surface-solid-owned")
    assert {k: v for k, v in target_exclusion_section(only).items() if k not in sec0} == {"surface_solid_owned": True}
    c = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert c.surface_solid_owned and c.surface_solid_protect_mm == 5.0 and c.surface_solid_dilate_mm == 5.0


def test_cli_refusals(capsys):
    A = _t78a()
    P0M = ("--surface-target-check", "overlap", "--surface-keep-last", "--surface-grasp-hold", "--surface-no-ball")
    args = A._reject(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-dilate-mm", "5")
    assert args.surface_solid_owned and args.surface_solid_dilate_mm == 5.0
    cases = (((), "attention-surface"),
             (A.SURF, "--surface-keep-last"),
             ((*A.SURF, "--surface-keep-last"), "--surface-no-ball"),
             ((*A.SURF, *P0M, "--surface-hull-fill"), "--surface-hull-fill"),
             ((*A.SURF, *P0M, "--surface-contact-allowance-mm", "10"), "--surface-contact-allowance-mm"),
             ((*A.SURF, *P0M, "--surface-memory-target-sticky"), "--surface-memory-target-sticky"))
    for extra, msg in cases:
        with pytest.raises(SystemExit):
            A._reject(*extra, "--surface-solid-owned")
        assert msg in capsys.readouterr().err, (extra, msg)
    for flag in ("--surface-solid-protect-mm", "--surface-solid-dilate-mm"):
        with pytest.raises(SystemExit):
            A._reject(*A.SURF, *P0M, flag, "5")                          # without --surface-solid-owned
        assert "--surface-solid-owned 와 함께만" in capsys.readouterr().err
        for bad in ("-1", "nan", "inf"):
            with pytest.raises(SystemExit):
                A._reject(*A.SURF, *P0M, "--surface-solid-owned", flag, bad)
            assert flag in capsys.readouterr().err


# ============================================================================ 2 off = the oracle split


def _old_rule(c, seed, ev, vs):
    """The pre-T96 split (T46e surface + band), re-derived independently."""
    from scipy.spatial import cKDTree

    from AG3S.ag3s.fields.curobo_builder import oracle_band_masks

    d_t = cKDTree(ev.target_points).query(c)[0]
    max_d = float(ev.max_dist_voxels) * vs
    op = ev.other_points
    lo, hi = ev.target_points.min(0) - (2 * max_d + vs), ev.target_points.max(0) + (2 * max_d + vs)
    op = op[np.all((op >= lo) & (op <= hi), axis=1)]
    d_o = cKDTree(op).query(c)[0] if len(op) else np.full(len(c), np.inf)
    above = np.ones(len(c), bool) if ev.z_floor is None else c[:, 2] >= ev.z_floor
    surface = (d_t <= max_d) & (d_t < d_o) & above
    band = np.zeros(len(c), bool)
    if len(ev.band_keys):
        m = oracle_band_masks(c, ev.band_keys, ev.band_masks, ev.band_cell)
        band = ((m == (np.uint32(1) << np.uint32(ev.code))) & seed & (d_t <= max_d) & above
                & (d_o > ev.surface_protect_voxels * vs) & ~surface)
    removed = seed & (surface | band)
    return removed, surface | removed, band


def test_off_split_is_the_pre_t96_rule_and_adds_no_key():
    T, tp, op, c, seed, owner = pytest.importorskip("tests.ag3s.test_t78f_contact_allowance")._scene()
    from AG3S.ag3s.fields.curobo_builder import oracle_cell_keys

    # a band too: every candidate's cell carries the target's bit only
    keys = np.unique(oracle_cell_keys(c, 0.0025))
    ev = OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=6.0, z_floor=T.TABLE_Z + T.VS,
                               band_cell=0.0025, band_keys=keys, band_masks=np.full(len(keys), 2, np.uint32))
    assert ev.solid is None
    sp = oracle_visible_split(c, seed, ev, T.VS)
    assert "solid" not in sp
    rm, sign, band = _old_rule(c, seed, ev, T.VS)
    assert band.any() and rm.any()
    np.testing.assert_array_equal(sp["removed"], rm)
    np.testing.assert_array_equal(sp["sign"], sign)
    np.testing.assert_array_equal(sp["band"], band)
    # the evidence refuses the combinations the config refuses
    sol = SolidOwned(protect_m=0.005)
    with pytest.raises(ValueError, match="solid"):
        OracleVisibleEvidence(target_points=tp, contact_allowance_m=0.01, solid=sol)
    h, _ = P.convex_hull(tp)
    from AG3S.ag3s.fields.curobo_builder import HullFill

    with pytest.raises(ValueError, match="solid"):
        OracleVisibleEvidence(target_points=tp, solid=sol,
                              hull_fill=HullFill(equations=h["equations"], lo=h["lo"], hi=h["hi"], pad_m=0.0))
    for bad in (-0.001, float("nan"), True):
        with pytest.raises(ValueError, match="SolidOwned"):
            SolidOwned(protect_m=bad)
        with pytest.raises(ValueError, match="SolidOwned"):
            SolidOwned(protect_m=0.005, dilate_m=bad)


def test_pipeline_off_never_calls_the_t96_steps_and_builds_the_same_call(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    from AG3S.ag3s.fields import curobo_builder as cb
    from AG3S.ag3s.runtime.pipeline import AG3S

    def boom(*a, **k):
        raise AssertionError("T96 step called with its flag off")

    for target, name in ((AG3S, "_surface_solid"), (AG3S, "_surface_solid_store"), (cb, "solid_candidates"),
                         (cb, "solid_facet_value"), (cb.CuroboFieldBuilder, "_solid_record_base")):
        monkeypatch.setattr(target, name, boom)
    A = _t78a()
    out = []
    for extra in ({}, {"surface_solid_owned": False, "surface_solid_protect_mm": 5.0,
                       "surface_solid_dilate_mm": 0.0}):
        fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", **P0M_SURF, target_extent="attention_surface",
                            **extra)
        obs, _, _ = A._rig(fw)
        rec = A._update(ag3s, obs, ats_points=A._apple_points(fw))
        ag3s.set_grasp_latched(True)
        field, _ = fw._build(ag3s)
        kw = ag3s._esdf_builder.calls[-1]
        t78 = ag3s._surface_frame["t78"]
        assert rec["segmentation"], "the off run must exercise an accepted region"
        assert kw["oracle_evidence"].solid is None and len(kw["oracle_evidence"].target_points)
        assert "solid_region" not in t78 and getattr(ag3s, "_surface_solid_region", None) is None
        assert "solid" not in field.stats["target_free"]["attention_surface"]
        info = ConstraintRecordWriter._write_exclusion_trace(field, {})
        assert "solid" not in info["target_exclusion"]["attention_surface"]
        out.append((rec, kw, t78, field.stats["target_free"]["oracle"]))
    (r0, k0, t0, o0), (r1, k1, t1, o1) = out
    assert k0.keys() == k1.keys()
    np.testing.assert_array_equal(k0["oracle_evidence"].target_points, k1["oracle_evidence"].target_points)
    np.testing.assert_array_equal(k0["oracle_evidence"].other_points, k1["oracle_evidence"].other_points)
    assert k0["oracle_evidence"].z_floor == k1["oracle_evidence"].z_floor
    strip = lambda d: {k: v for k, v in d.items() if k not in ("memory", "ms")}   # noqa: E731 — timings differ
    assert strip(r0) == strip(r1) and strip(t0) == strip(t1) and strip(o0) == strip(o1)


def test_tier_off_is_value_for_value_the_oracle_tier(monkeypatch):
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields import curobo_builder as cb

    ev, _, _ = _t44_evidence(T44)
    b = T44._builder(target_exclusion="ball_owned")

    def tier(evidence):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=evidence)
        return t

    base = tier(ev)
    monkeypatch.setattr(cb, "solid_candidates", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    again = tier(ev)
    assert torch.equal(again.values, base.values)
    assert "solid" not in again.extra["exclusion"]["oracle"]
    assert not {"solid_removed_ijk", "solid_protected_ijk"} & set(again.extra["exclusion_trace"])


# ============================================================================ 3 pure: non-ball shapes

VS = 0.005
TABLE = 0.80
EYE_DIR = np.array([0.0, -1.0, 1.2]) / np.linalg.norm([0.0, -1.0, 1.2])   # the camera looks from front-above


def _box_sdf(p, lo, hi):
    c, h = 0.5 * (np.asarray(lo) + hi), 0.5 * (np.asarray(hi) - lo)
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def _cyl_sdf(p, c, r, z0, z1):
    dr = np.linalg.norm(p[..., :2] - np.asarray(c)[:2], axis=-1) - r
    dz = np.maximum(z0 - p[..., 2], p[..., 2] - z1)
    q = np.stack([dr, dz], -1)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def _shape(kind):
    """`(target sdf, neighbour sdf)`: the target stands on the table, the neighbour's nearest face is 8 mm away
    (≥ 5 mm, the τ assumption). L: the neighbour sits in the L's notch (inside the target's convex hull)."""
    if kind == "cylinder":
        tgt = lambda p: _cyl_sdf(p, (0.0, 0.0), 0.030, TABLE, TABLE + 0.060)   # noqa: E731
        nb = lambda p: _box_sdf(p, (-0.03, 0.038, TABLE), (0.03, 0.078, TABLE + 0.05))   # noqa: E731
    elif kind == "box":
        tgt = lambda p: _box_sdf(p, (-0.03, -0.02, TABLE), (0.03, 0.02, TABLE + 0.05))   # noqa: E731
        nb = lambda p: _cyl_sdf(p, (0.0, 0.048), 0.020, TABLE, TABLE + 0.05)   # noqa: E731
    elif kind == "L":
        # an L of two boxes: the long arm along x (y ∈ [-0.04, -0.01]), the short arm up y at x ∈ [-0.04, -0.01]
        tgt = lambda p: np.minimum(_box_sdf(p, (-0.04, -0.04, TABLE), (0.04, -0.01, TABLE + 0.05)),   # noqa: E731
                                   _box_sdf(p, (-0.04, -0.04, TABLE), (-0.01, 0.04, TABLE + 0.05)))
        # the neighbour in the notch, 8 mm off both inner faces
        nb = lambda p: _box_sdf(p, (-0.002, -0.002, TABLE), (0.03, 0.03, TABLE + 0.03))   # noqa: E731
    else:
        raise ValueError(kind)
    return tgt, nb


def _normals(f, p, h=1e-4):
    g = np.stack([(f(p + h * e) - f(p - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)
    return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)


def _surface_samples(f, lo, hi, step=0.0015, visible_only=True, z_min=None):
    ax = [np.arange(lo[i], hi[i], step) for i in range(3)]
    p = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    p = p[np.abs(f(p)) < 0.5 * step]
    p = p[p[:, 2] > TABLE + 0.5 * step]                                 # the bottom face rests on the table
    if visible_only:
        p = p[_normals(f, p) @ EYE_DIR > 0.15]
    if z_min is not None:
        p = p[p[:, 2] >= z_min]
    return p


def _rig(kind):
    """The grid (5 mm), seeds (|sdf| < 0.75 vs of target ∪ neighbour ∪ table, all faces — the episode's memory of
    other views and the truncation band put seeds on faces this camera does not see), the evidence of one camera:
    target samples = its visible faces above TABLE + 15 mm (the lower front is occluded), region R = those samples at
    5 mm node size, other samples = the neighbour's visible faces + the visible table around."""
    tgt, nb = _shape(kind)
    lo, hi = np.array([-0.08, -0.08, TABLE - 0.02]), np.array([0.08, 0.10, TABLE + 0.08])
    ax = [np.arange(lo[i], hi[i], VS) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    c = g.reshape(-1, 3)
    d = np.stack([tgt(c), nb(c), c[:, 2] - TABLE], -1)
    owner = d.argmin(-1)
    seed = np.abs(d.min(-1)) < 0.75 * VS
    blo, bhi = lo + 0.01, hi - 0.01
    tp = _surface_samples(tgt, blo, bhi, z_min=TABLE + 0.015)
    nbp = _surface_samples(nb, blo, bhi)
    tx, ty = np.meshgrid(np.arange(blo[0], bhi[0], 0.0025), np.arange(blo[1], bhi[1], 0.0025), indexing="ij")
    table = np.column_stack([tx.ravel(), ty.ravel(), np.full(tx.size, TABLE)])
    table = table[(tgt(table + [0, 0, 0.001]) > 0.002) & (nb(table + [0, 0, 0.001]) > 0.002)]
    op = np.vstack([nbp, table])
    nodes = np.unique(np.floor(tp / VS).astype(np.int64), axis=0)
    R = np.array([tp[(np.floor(tp / VS).astype(np.int64) == k).all(1)].mean(0) for k in nodes])
    R_perp = R.copy()
    R_perp[:, 2] = TABLE
    zf = TABLE + VS                                                     # the A5 guard: support + 1 fine voxel
    return dict(c=c, seed=seed, owner=owner, tp=tp, op=op, R=R, pts=np.vstack([R, R_perp]), zf=zf, tgt=tgt)


def _solid(r, protect=0.005, dilate=0.0):
    h, why = P.convex_hull(r["pts"])
    assert h is not None, why
    return SolidOwned(protect_m=protect, dilate_m=dilate, equations=h["equations"], lo=h["lo"], hi=h["hi"],
                      target_points=r["pts"])


@pytest.mark.parametrize("kind", ["cylinder", "box", "L"])
@pytest.mark.parametrize("dilate", [0.0, 0.005])
def test_solid_removes_hidden_back_and_bottom_keeps_neighbour_tau_band_and_table(kind, dilate):
    from scipy.spatial import cKDTree

    r = _rig(kind)
    c, seed, owner = r["c"], r["seed"], r["owner"]
    ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
    sol = _solid(r, dilate=dilate)
    off = oracle_visible_split(c, seed, ev, VS)
    on = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=sol), VS)
    d_n = cKDTree(r["op"]).query(c)[0]
    above = c[:, 2] >= r["zf"]
    fv = solid_facet_value(sol, c)
    tau = d_n <= 0.005
    # invariants (every path)
    assert not np.any(on["removed"] & ~seed), "only seeds"
    assert not np.any(on["removed"] & tau), "the τ band is removed by no path"
    assert not np.any(on["removed"] & ~above), "nothing below the support guard"
    solid_only = on["removed"] & ~on["surface"] & ~on["band"]
    assert np.all(fv[solid_only] <= dilate + 1e-12), "the solid path stays inside S ⊕ δ"
    np.testing.assert_array_equal(on["solid"]["solid"] | on["solid"]["dilate"], solid_only)
    np.testing.assert_array_equal(on["removed"] | (off["removed"] & tau), on["removed"] | off["removed"])
    assert not np.any(off["removed"] & ~on["removed"] & ~tau), "on removes everything off removed outside τ"
    # the neighbour and the table stay
    assert not np.any(on["removed"] & (owner == 1)), "the neighbour keeps every seed"
    assert not np.any(on["removed"] & (owner == 2)), "the table keeps every seed"
    # the solid path is exactly its rule (recomputed independently): x ∈ S ⊕ δ ∧ z ≥ z_floor ∧ d_N > τ ∧ d_T ≤ d_N
    d_T = np.minimum(cKDTree(r["tp"]).query(c)[0], cKDTree(r["pts"]).query(c)[0])
    rule = seed & (fv <= dilate) & above & ~tau & (d_T <= d_n)
    np.testing.assert_array_equal(on["removed"], (off["removed"] & ~tau) | rule)
    # what it is for: target seeds above the guard, outside τ, that the surface · band paths leave
    tgt_seed = seed & (owner == 0) & above & ~tau
    low = tgt_seed & (c[:, 2] < TABLE + 0.015)                           # between the visible surface and the table
    new = on["removed"] & ~off["removed"]
    assert (low & new).sum() >= 5, (kind, dilate, (low & new).sum())
    hidden = tgt_seed & (_normals(r["tgt"], c) @ EYE_DIR <= 0.15)        # faces this camera does not see
    if kind != "box":                                                    # box: every hidden face is S's boundary
        assert (hidden & new).sum() >= 5, (kind, dilate, (hidden & new).sum())
    left_off = (tgt_seed & ~off["removed"]).sum() / tgt_seed.sum()
    left_on = (tgt_seed & ~on["removed"]).sum() / tgt_seed.sum()
    assert left_on < left_off, (kind, dilate, left_on, left_off)
    if dilate:
        assert on["solid"]["dilate"].any()
    else:
        assert not on["solid"]["dilate"].any()
    # the rule's documented limit (impl.md): a hidden face inside S ⊕ δ whose nearest observed surface is a
    # neighbour / the table (d_N < d_T — d_T is to the points R ∪ R⊥ ∪ samples, not to S) stays
    kept_by_dt = tgt_seed & (fv <= dilate) & (d_T > d_n) & ~on["removed"]
    assert kept_by_dt.any()
    # τ protects only where something would have removed a seed; the count matches
    assert on["solid"]["protected"].sum() == (seed & tau & (off["surface"] | off["band"]
                                                             | on["solid"]["in_solid"] & above
                                                             & (on["solid"]["d_T"] <= on["d_o"]))).sum()
    # the sign set gains only voxels outside τ, inside S ⊕ δ, above the guard
    extra = on["sign"] & ~off["sign"]
    assert not np.any(extra & tau) and not np.any(extra & ~above) and np.all(fv[extra] <= dilate + 1e-12)


def test_l_shape_notch_neighbour_is_protected_and_the_notch_risk_is_visible():
    """The documented risk: the convex hull of an L fills its notch. A neighbour standing there keeps its seeds (τ ·
    `d_T ≤ d_N`); free space in the notch inside S counts as target (`fill` → the sign set), which is the limitation."""
    r = _rig("L")
    ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
    sol = _solid(r)
    on = oracle_visible_split(r["c"], r["seed"], dataclasses.replace(ev, solid=sol), VS)
    nb_seed = r["seed"] & (r["owner"] == 1)
    assert nb_seed.sum() > 20 and not on["removed"][nb_seed].any()
    in_s = solid_facet_value(sol, r["c"]) <= 0.0
    assert (nb_seed & in_s).sum() > 10, "the rig really puts the neighbour inside the hull"


def test_solid_candidates_reach_beyond_max_dist_and_match_the_facet_test():
    big = np.array([[x, y, z] for x in (0.0, 0.2) for y in (0.0, 0.2) for z in (0.0, 0.2)], float)
    h, _ = P.convex_hull(big)
    sol = SolidOwned(protect_m=0.005, equations=h["equations"], lo=h["lo"], hi=h["hi"], target_points=big)
    org, shape = np.array([-0.05, -0.05, -0.05]), (64, 64, 64)
    cand = solid_candidates(sol, org, VS, shape)
    centres = org + cand * VS
    assert np.all(solid_facet_value(sol, centres) <= 0.0)
    assert len(cand) == 41 ** 3                                          # 0 .. 0.2 m inclusive at 5 mm
    near = oracle_candidates(big, org, VS, shape, 6 * VS)
    far = np.setdiff1d(cand[:, 0] * 10**6 + cand[:, 1] * 10**3 + cand[:, 2],
                       near[:, 0] * 10**6 + near[:, 1] * 10**3 + near[:, 2])
    assert len(far) > 0, "the centre of a 20 cm box is beyond max_dist of every corner sample"
    sol5 = dataclasses.replace(sol, dilate_m=0.0075)                     # 7.5 mm: off the 5 mm grid (no float tie)
    assert len(solid_candidates(sol5, org, VS, shape)) > len(cand)
    assert solid_candidates(SolidOwned(protect_m=0.005), org, VS, shape) is None
    assert np.all(np.isinf(solid_facet_value(SolidOwned(protect_m=0.005), centres[:3])))


# ============================================================================ 4 tier (fake cuRobo)


def _t44_evidence(T44):
    VS_ = T44.VS
    g = T44._centres()
    sdf, owner = T44._sdf(g)
    surf = np.abs(sdf) < 0.4 * VS_
    pts, own = g[surf], owner[surf]
    a_pts = pts[own == 0]
    upper = a_pts[:, 2] >= T44.A[2]
    ev = OracleVisibleEvidence(target_points=a_pts[upper], other_points=pts[own != 0], max_dist_voxels=6.0,
                               z_floor=T44.TABLE_Z + VS_)
    return ev, a_pts[upper], (g, sdf, owner)


def test_tier_solid_clears_the_lower_half_keeps_neighbour_and_table_and_records_by_path():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from scipy.spatial import cKDTree

    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    VS_ = T44.VS
    ev, R, (g, sdf, owner) = _t44_evidence(T44)
    R_perp = R.copy()
    R_perp[:, 2] = T44.TABLE_Z
    pts = np.vstack([R, R_perp])
    h, _ = P.convex_hull(pts)
    sol = SolidOwned(protect_m=0.005, equations=h["equations"], lo=h["lo"], hi=h["hi"], target_points=pts,
                     info={"region_source": "this_request"})
    b = T44._builder(target_exclusion="ball_owned")

    def tier(evidence):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=evidence)
        return t

    base, on = tier(ev), tier(dataclasses.replace(ev, solid=sol))
    v0, v1 = base.values.numpy(), on.values.numpy()
    above = g[..., 2] >= T44.TABLE_Z + VS_
    d_n = cKDTree(ev.other_points).query(g.reshape(-1, 3))[0].reshape(g.shape[:3])
    d_T = cKDTree(np.vstack([ev.target_points, pts])).query(g.reshape(-1, 3))[0].reshape(g.shape[:3])
    a_low = ((owner == 0) & (np.abs(sdf) < 0.75 * VS_) & (g[..., 2] < T44.A[2] - VS_) & above & (d_n > 0.005)
             & (d_T <= d_n)
             & (solid_facet_value(sol, g.reshape(-1, 3)).reshape(g.shape[:3]) <= 0.0))
    assert a_low.sum() > 5
    assert np.min(np.abs(v1[a_low])) > 0.0, "the lower half inside S, outside τ, is gone"
    b_surf = (owner == 1) & (np.abs(sdf) < 0.4 * VS_)
    assert np.abs(v1[b_surf]).max() < 1e-6, "the neighbour 5 mm away keeps its surface"
    table = (owner == 2) & (np.abs(sdf) < 0.5 * VS_) & (np.linalg.norm(g[..., :2] - T44.A[:2], axis=-1) < 0.03)
    assert np.abs(v1[table]).max() < 1e-6, "the table under the target stays"
    ex, ex0 = on.extra["exclusion"], base.extra["exclusion"]
    s = ex["oracle"]["solid"]
    assert s["has_solid"] and s["protect_mm"] == 5.0 and s["dilate_mm"] == 0.0 and s["region_source"] == "this_request"
    assert s["n_removed_solid"] > 0 and s["n_removed_dilate"] == 0 and s["n_solid_voxels"] > 0
    assert s["n_removed_total"] == s["n_removed_surface"] + s["n_removed_band"] + s["n_removed_solid"]
    assert ex["n_removed"] == s["n_removed_total"] == ex["n_target_owned"]
    assert s["n_protected_tau"] == (s["n_protected_tau_surface"] + s["n_protected_tau_band"]
                                    + s["n_protected_tau_solid"]) or s["n_protected_tau_band"] > 0
    for k in ("candidates_ms", "split_ms"):
        assert s[k] >= 0.0
    tr = on.extra["exclusion_trace"]
    assert len(tr["solid_removed_ijk"]) == s["n_removed_solid"] + s["n_removed_dilate"]
    assert len(tr["solid_protected_ijk"]) == s["n_protected_tau"]
    assert len(tr["removed_ijk"]) == ex["n_removed"]
    assert CuroboFieldBuilder._exclusion_counts([on])["oracle_split"]["solid"] == s
    assert ex["n_removed"] > ex0["n_removed"]
    # τ-only evidence (no solid this request): the surface path with the τ band, no solid path
    tau_only = tier(dataclasses.replace(ev, solid=SolidOwned(protect_m=0.005)))
    st = tau_only.extra["exclusion"]["oracle"]["solid"]
    assert not st["has_solid"] and st["n_removed_solid"] == 0 and st["n_solid_voxels"] == 0
    assert tau_only.extra["exclusion"]["n_removed"] <= ex0["n_removed"]
    # an empty target-sample evidence with a solid still removes (the solid's own candidates)
    empty = tier(OracleVisibleEvidence(target_points=np.zeros((0, 3)), other_points=ev.other_points,
                                       z_floor=ev.z_floor, solid=sol))
    assert empty.extra["exclusion"]["oracle"]["solid"]["n_removed_solid"] > 0


# ============================================================================ 5 pipeline · record


def _wired_on(monkeypatch, **extra):
    A = _t78a()
    return A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **P0M_SURF,
                    surface_solid_owned=True, **extra)


def test_pipeline_stores_the_region_builds_r_perp_and_records(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch, surface_solid_dilate_mm=5.0)
    obs, _, _ = A._rig(fw)
    rec = A._update(ag3s, obs, ats_points=A._apple_points(fw))
    assert rec["segmentation"]
    sr = ag3s._surface_frame["t78"]["solid_region"]
    assert sr["stored"] and sr["n_nodes"] == len(ag3s._surface_solid_region["pos"]) > 10
    field, _ = fw._build(ag3s)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    sol = ev.solid
    assert isinstance(sol, SolidOwned) and sol.has_solid
    assert sol.protect_m == pytest.approx(0.005) and sol.dilate_m == pytest.approx(0.005)
    R = ag3s._surface_solid_region["pos"]
    n = len(R)
    np.testing.assert_allclose(sol.target_points[:n], R)
    np.testing.assert_allclose(sol.target_points[n:, 2], fw.TABLE_Z, atol=1e-9)    # R⊥ on the (horizontal) table
    np.testing.assert_allclose(sol.target_points[n:, :2], R[:, :2], atol=1e-12)
    assert ev.hull_fill is None and ev.contact_allowance_m is None
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert s["applied"] and s["region_source"] == "this_request" and s["n_projected"] == n
    assert s["volume_cm3"] > 0 and s["z_floor_m"] == pytest.approx(fw.TABLE_Z + fw.FINE)
    assert s["plane_z_at_centroid_m"] == pytest.approx(fw.TABLE_Z) and s["voxels"] is None   # fake builder
    assert s["ms"] >= 0.0
    # a builder that splits: its counts reach `target_exclusion.attention_surface.solid.voxels`
    rb = ag3s._esdf_builder
    orig = type(rb).update

    def update(self, cameras, **kw):
        f = orig(self, cameras, **kw)
        f.stats["target_free"]["oracle_split"] = {"solid": {"n_removed_solid": 7, "n_protected_tau": 3}}
        return f

    monkeypatch.setattr(type(rb), "update", update)
    field, _ = fw._build(ag3s)
    te = json.loads(json.dumps(ConstraintRecordWriter._write_exclusion_trace(field, {})["target_exclusion"]))
    assert te["attention_surface"]["solid"]["voxels"] == {"n_removed_solid": 7, "n_protected_tau": 3}
    assert te["attention_surface"]["solid"]["applied"]


def test_pipeline_hold_keeps_the_last_region_and_no_region_is_tau_only(monkeypatch):
    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch)
    obs, _, _ = A._rig(fw)
    for o in obs:
        o.attention_map = None
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    sol = ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid
    assert not s["applied"] and s["why"] == "no_accepted_region"
    assert sol is not None and not sol.has_solid and sol.protect_m == pytest.approx(0.005)   # τ still protects
    obs, _, _ = A._rig(fw)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    R0 = ag3s._surface_solid_region["pos"].copy()
    ag3s.frame_index += 1
    ag3s.set_grasp_active(True)                                          # A4 hold: no selection, the region stands
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    assert ag3s._surface_frame["why"] == "grasp_hold"
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert s["applied"] and s["region_source"] == "last_accepted" and s["region_age_frames"] == 1
    np.testing.assert_array_equal(ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid.target_points[:len(R0)], R0)
    ag3s.set_grasp_active(False)
    A._update(ag3s, obs, ats_points=A._apple_points(fw) + [0.0, 0.3, 0.0])   # A2 rejects: the region stands
    assert ag3s._surface_frame["why"] == "not_ats_target"
    np.testing.assert_array_equal(ag3s._surface_solid_region["pos"], R0)
    # no support plane under the region → no solid (never a solid without its floor)
    field, _ = ag3s._build_esdf([object()], fw._apple_geometry(), support_surfaces=[], robot_state=np.zeros(2),
                                contact_context=fw.LEFT)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert not s["applied"] and s["why"] in ("no_support_plane", "no_support_guard")
    ag3s.reset()
    assert ag3s._surface_solid_region is None


# ============================================================================ 6 the §1.2 replay variant (palm cover on a P0m record)


def test_ta_offline_drop_server_args_and_p0m_pc_variant(tmp_path):
    from AG3S.trajopt.experiments import ta_offline as TA

    rec = tmp_path / "server" / "server_constraints" / "run_0000"
    rec.mkdir(parents=True)
    (tmp_path / "server" / "server_proc.txt").write_text(
        "python -m AG3S.trajopt.serve_safe --config c --checkpoint k --model-xml x.xml --obstacle-margin 0.010 "
        "--palm-shell on --palm-shell-margin 5 --links gripper\n")
    a0, line0 = TA.server_args(rec)
    assert a0.palm_shell == "on" and a0.palm_shell_margin == 5.0 and a0.palm_cover == "off"
    assert TA.server_args(rec, ())[1] == line0                           # no drop: the recorded line as before
    extra = TA.variant_server_args("P0M_PC")
    a1, line1 = TA.server_args(rec, extra)
    assert a1.palm_shell == "off" and a1.palm_shell_margin is None and a1.palm_cover == "on"
    assert "--drop:" not in line1 and a1.links == "gripper" and a1.obstacle_margin == 0.010
    assert TA.VARIANTS["P0M_PC"]["collision"] == {"palm_shell_margin": None}


# ============================================================================ 7 T96b (D28) — d_N without the support-plane samples


def test_t96b_dn_support_config_cli_and_refusals(capsys):
    """D28 config default `exclude` · round trip · bad value; CLI key only when given; refused without the solid."""
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    assert AG3SConfig().esdf.surface_solid_dn_support == "exclude"
    for mode in ("exclude", "include"):
        cfg = AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_owned": True, "surface_solid_dn_support": mode}})
        assert cfg.esdf.surface_solid_dn_support == mode
        assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in ("", "off", None, True):
        with pytest.raises(AG3SConfigError, match="surface_solid_dn_support"):
            AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_owned": True, "surface_solid_dn_support": bad}})
    A = _t78a()
    P0M = ("--surface-target-check", "overlap", "--surface-keep-last", "--surface-grasp-hold", "--surface-no-ball")
    only = A._parse(*A.SURF, *A.ALL, "--surface-solid-owned")
    assert "surface_solid_dn_support" not in target_exclusion_section(only)
    assert "surface_solid_dn_support" not in ifg_cl1_record_meta(only)
    inc = A._parse(*A.SURF, *A.ALL, "--surface-solid-owned", "--surface-solid-dn-support", "include")
    assert target_exclusion_section(inc)["surface_solid_dn_support"] == "include"
    assert ifg_cl1_record_meta(inc)["surface_solid_dn_support"] == "include"
    assert A._reject(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-dn-support",
                     "exclude").surface_solid_dn_support == "exclude"
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, *P0M, "--surface-solid-dn-support", "include")   # without --surface-solid-owned
    assert "--surface-solid-owned 와 함께만" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        A._parse(*A.SURF, *A.ALL, "--surface-solid-owned", "--surface-solid-dn-support", "both")
    capsys.readouterr()


def _dn(plane_z, thr=0.008):
    return {"dn_plane": np.array([0.0, 0.0, 2.0, 2.0 * plane_z]), "dn_threshold_m": thr}   # normalised by SolidOwned


def test_t96b_solid_owned_dn_plane_validation():
    s = SolidOwned(protect_m=0.005, **_dn(TABLE))
    np.testing.assert_allclose(s.dn_plane, [0.0, 0.0, 1.0, TABLE])
    assert s.dn_threshold_m == 0.008
    with pytest.raises(ValueError, match="함께"):
        SolidOwned(protect_m=0.005, dn_plane=np.array([0.0, 0.0, 1.0, TABLE]))
    with pytest.raises(ValueError, match="함께"):
        SolidOwned(protect_m=0.005, dn_threshold_m=0.008)
    for bad in (0.0, -0.001, float("nan"), True):
        with pytest.raises(ValueError, match="dn_threshold_m"):
            SolidOwned(protect_m=0.005, dn_plane=np.array([0.0, 0.0, 1.0, TABLE]), dn_threshold_m=bad)
    with pytest.raises(ValueError, match="dn_plane"):
        SolidOwned(protect_m=0.005, dn_plane=np.zeros(4), dn_threshold_m=0.008)


@pytest.mark.parametrize("kind", ["cylinder", "box", "L"])
@pytest.mark.parametrize("dilate", [0.0, 0.005])
def test_t96b_exclude_drops_the_table_samples_from_d_n_only(kind, dilate):
    """`exclude`: d_N (solid path + τ of every path) is the nearest non-target sample off the support plane; the surface ·
    band paths' own `d_o` is untouched. The rule recomputed independently; neighbour and table keep every seed; target
    residual ≤ `include`; a plane no sample is near is value-for-value `include` (the no-drop branch)."""
    from scipy.spatial import cKDTree

    r = _rig(kind)
    c, seed, owner = r["c"], r["seed"], r["owner"]
    ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
    inc_sol = _solid(r, dilate=dilate)
    exc_sol = dataclasses.replace(inc_sol, **_dn(TABLE))
    inc = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=inc_sol), VS)
    exc = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=exc_sol), VS)
    # the dropped samples: within the RANSAC inlier threshold of the table (of the samples near the solid)
    on_plane = np.abs(r["op"][:, 2] - TABLE) <= 0.008
    assert exc["solid"]["n_dn_support_dropped"] > 0 and inc["solid"]["n_dn_support_dropped"] == 0
    assert exc["solid"]["n_dn_support_dropped"] <= int(on_plane.sum())
    np.testing.assert_array_equal(exc["d_o"], inc["d_o"])                 # the surface · band paths' d_o as is
    np.testing.assert_array_equal(inc["solid"]["d_N"], inc["d_o"])        # include: d_N = d_o (T96)
    d_n = cKDTree(r["op"][~on_plane]).query(c)[0]
    np.testing.assert_allclose(exc["solid"]["d_N"], d_n, rtol=0, atol=1e-12)
    above = c[:, 2] >= r["zf"]
    fv = solid_facet_value(exc_sol, c)
    tau = d_n <= 0.005
    d_T = np.minimum(cKDTree(r["tp"]).query(c)[0], cKDTree(r["pts"]).query(c)[0])
    d_o = exc["d_o"]
    max_d = 6.0 * VS
    surface = seed & (cKDTree(r["tp"]).query(c)[0] <= max_d) & (cKDTree(r["tp"]).query(c)[0] < d_o) & above
    rule = seed & (fv <= dilate) & above & ~tau & (d_T <= d_n)
    np.testing.assert_array_equal(exc["removed"], (surface & ~tau) | exc["band"] | rule)
    assert not np.any(exc["removed"] & tau) and not np.any(exc["removed"] & ~above)
    nb_lost = int((exc["removed"] & (owner == 1)).sum())
    if kind == "L":
        # the documented cost (T96 impl.md note 2): without the table samples in d_N, the hidden lower faces of the
        # neighbour standing in the L's notch (inside S) are nearer R⊥ than any kept neighbour sample → d_T ≤ d_N
        assert 0 < nb_lost <= int((seed & (owner == 1)).sum()), nb_lost
        assert not np.any(inc["removed"] & (owner == 1)), "include keeps the notch neighbour (T96)"
        assert np.all(fv[exc["removed"] & (owner == 1)] <= dilate + 1e-12), "only inside S ⊕ δ"
        print(f"T96b L notch neighbour seeds removed by exclude (dilate {dilate}): {nb_lost}")
    else:
        assert nb_lost == 0, "the neighbour keeps every seed"
    assert not np.any(exc["removed"] & (owner == 2)), "the table keeps every seed (z_floor)"
    tgt_seed = seed & (owner == 0) & above
    left_inc = (tgt_seed & ~inc["removed"]).sum()
    left_exc = (tgt_seed & ~exc["removed"]).sum()
    assert left_exc <= left_inc, (kind, dilate, left_exc, left_inc)
    if kind == "box":                                                    # impl.md T96 note 2: the box's hidden lower sides
        assert left_exc < left_inc, (kind, dilate, left_exc, left_inc)
    # no sample near the plane (a plane 1 m below) → the no-drop branch is `include` key for key, value for value
    far = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=dataclasses.replace(inc_sol, **_dn(TABLE - 1.0))),
                               VS)
    assert far["solid"]["n_dn_support_dropped"] == 0
    assert set(far) == set(inc) and set(far["solid"]) == set(inc["solid"])
    for k in inc:
        if k == "solid":
            for kk in inc["solid"]:
                np.testing.assert_array_equal(far["solid"][kk], inc["solid"][kk])
        else:
            np.testing.assert_array_equal(far[k], inc[k])


def test_t96b_tier_records_the_dropped_samples():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")

    ev, R, _ = _t44_evidence(T44)
    R_perp = R.copy()
    R_perp[:, 2] = T44.TABLE_Z
    pts = np.vstack([R, R_perp])
    h, _ = P.convex_hull(pts)
    sol = SolidOwned(protect_m=0.005, equations=h["equations"], lo=h["lo"], hi=h["hi"], target_points=pts,
                     info={"dn_support": "include"})
    b = T44._builder(target_exclusion="ball_owned")

    def tier(s):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=dataclasses.replace(ev, solid=s))
        return t

    t_inc = tier(sol)
    t_exc = tier(dataclasses.replace(sol, info={"dn_support": "exclude"}, **_dn(T44.TABLE_Z)))
    si, se = (t.extra["exclusion"]["oracle"]["solid"] for t in (t_inc, t_exc))
    assert si["dn_support"] == "include" and not si["dn_support_plane"] and si["dn_threshold_mm"] is None
    assert si["n_dn_support_dropped"] == 0
    assert se["dn_support"] == "exclude" and se["dn_support_plane"] and se["dn_threshold_mm"] == 8.0
    assert se["n_dn_support_dropped"] > 0
    assert se["n_removed_total"] >= si["n_removed_total"]


@pytest.mark.parametrize("mode", ["exclude", "include", None])
def test_t96b_pipeline_passes_the_plane_of_r_perp_and_records(monkeypatch, mode):
    A = _t78a()
    extra = {} if mode is None else {"surface_solid_dn_support": mode}
    fw, ag3s = _wired_on(monkeypatch, **extra)
    obs, _, _ = A._rig(fw)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    sol = ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid
    s = field.stats["target_free"]["attention_surface"]["solid"]
    want = mode or "exclude"
    assert s["applied"] and s["dn_support"] == want and sol.info["dn_support"] == want
    if want == "exclude":
        thr = ag3s.config.support_surface.distance_threshold
        assert s["dn_support_applied"] and s["dn_threshold_mm"] == pytest.approx(thr * 1e3)
        np.testing.assert_allclose(sol.dn_plane, [0.0, 0.0, 1.0, fw.TABLE_Z], atol=1e-9)   # the plane of R⊥
        assert sol.dn_threshold_m == pytest.approx(thr)
        n = len(ag3s._surface_solid_region["pos"])
        np.testing.assert_allclose(sol.target_points[n:] @ sol.dn_plane[:3], sol.dn_plane[3], atol=1e-9)
    else:
        assert not s["dn_support_applied"] and "dn_threshold_mm" not in s
        assert sol.dn_plane is None and sol.dn_threshold_m is None
    # no accepted region → τ only, no plane (the plane is the one under the region)
    fw, ag3s = _wired_on(monkeypatch, **extra)
    obs, _, _ = A._rig(fw)
    for o in obs:
        o.attention_map = None
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    sol = ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid
    assert s["why"] == "no_accepted_region" and not s["dn_support_applied"] and sol.dn_plane is None


def test_t96b_off_is_untouched(monkeypatch):
    """Solid off: neither the config value nor the flag reaches anything (the off tests above hold; here the D28 field)."""
    A = _t78a()
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **P0M_SURF)
    obs, _, _ = A._rig(fw)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    assert ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid is None
    assert "solid" not in (field.stats["target_free"].get("attention_surface") or {})
