"""T78f — target contact allowance of the hand layer (`esdf.surface_contact_allowance_mm`, `--surface-contact-allowance-mm`),
default off. Off must be the oracle split (T46e / T78) call for call.

| # | property |
|---|---|
| 1 | config default None · round trip · refusals (needs `attention_surface` + `surface_keep_last`; 0 < M ≤ `oracle_max_dist_voxels` × fine voxel); CLI flag → esdf key / record meta only when given; start-up refusals |
| 2 | off: `oracle_visible_split` = the pre-T78f rule re-derived (surface `d_t ≤ max_d ∧ d_t < d_o ∧ above`, band), no `allowance_extra` key; `OracleVisibleEvidence` default None; the pipeline never builds an allowance (a `dataclasses.replace(contact_allowance_m=…)` explodes) and records no new key; the builder records no new key |
| 3 | on (pure): removed = off removed ∪ {seed, d_t ≤ M, above z_floor}; `allowance_extra` = the difference; the seeds the `d_t < d_o` rule kept near the target (stale "other" samples) go; seeds beyond M, below the floor, and non-seeds are not added; `sign` gains only the removed seeds; `surface` · `band` unchanged |
| 4 | tier (fake cuRobo, torch): the allowance clears the target seeds the oracle split kept, keeps the table under the guard and the neighbour beyond M; `oracle_split.allowance_mm` · `n_allowance_extra`; trace / counts add up |
| 5 | pipeline: on → the evidence carries `contact_allowance_m`, `target_exclusion.oracle.allowance_mm` · `n_allowance_extra` (the builder's count) in the constraint record JSON |
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence, oracle_candidates, oracle_visible_split
from AG3S.ag3s.runtime.oracle_segmentation import CODE_TARGET, OracleMemory

ON_BASE = {"target_extent": "attention_surface", "surface_keep_last": True}


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


def _t46e():
    return pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")


# ============================================================================ 1 config · CLI


def test_config_default_off_round_trip_and_refusals():
    assert AG3SConfig().esdf.surface_contact_allowance_mm is None
    cfg = AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, "surface_contact_allowance_mm": 10.0}})
    assert cfg.esdf.surface_contact_allowance_mm == 10.0
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    # the range end: oracle_max_dist_voxels (6) × fine voxel (5 mm) = 30 mm is allowed, beyond is refused
    AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, "surface_contact_allowance_mm": 30.0}})
    with pytest.raises(AG3SConfigError, match="후보 거리"):
        AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, "surface_contact_allowance_mm": 30.5}})
    with pytest.raises(AG3SConfigError, match="후보 거리"):
        AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, "oracle_max_dist_voxels": 2.0,
                                       "surface_contact_allowance_mm": 15.0}})
    for v in (0.0, -1.0, float("nan"), float("inf"), True, "10"):
        with pytest.raises(AG3SConfigError, match="surface_contact_allowance_mm"):
            AG3SConfig.from_dict({"esdf": {**ON_BASE, "fine_voxel_size": 0.005, "surface_contact_allowance_mm": v}})
    for bad in ({}, {"target_extent": "oracle_visible"}, {"target_extent": "attention_surface"}):
        with pytest.raises(AG3SConfigError, match="surface_contact_allowance_mm"):
            AG3SConfig.from_dict({"esdf": {**bad, "surface_contact_allowance_mm": 10.0}})


def test_cli_flag_maps_to_esdf_key_and_meta_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    base = A._parse(*A.SURF, *A.ALL)
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert base.surface_contact_allowance_mm is None
    assert "surface_contact_allowance_mm" not in set(sec0) | set(meta0)
    on = A._parse(*A.SURF, *A.ALL, "--surface-contact-allowance-mm", "10")
    sec = target_exclusion_section(on)
    assert {k: v for k, v in sec.items() if k not in sec0} == {"surface_contact_allowance_mm": 10.0}
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == {
        "surface_contact_allowance_mm": 10.0}
    assert AG3SConfig.from_dict({"esdf": {**sec, "fine_voxel_size": 0.005}}).esdf.surface_contact_allowance_mm == 10.0


def test_cli_refusals(capsys):
    A = _t78a()
    for m in ("10", "15", "30"):
        assert A._reject(*A.SURF, *A.ALL, "--surface-contact-allowance-mm", m).surface_contact_allowance_mm == float(m)
    for extra, msg in (((), "attention-surface"),                                     # no attention-surface
                       (A.SURF, "--surface-keep-last")):                              # no keep-last
        with pytest.raises(SystemExit):
            A._reject(*extra, "--surface-contact-allowance-mm", "10")
        assert msg in capsys.readouterr().err
    for bad in ("0", "-5", "nan", "inf", "30.5"):
        with pytest.raises(SystemExit):
            A._reject(*A.SURF, *A.ALL, "--surface-contact-allowance-mm", bad)
        assert "후보 거리" in capsys.readouterr().err
    # the reach follows --oracle-max-dist-voxels and --fine-voxel
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, *A.ALL, "--oracle-max-dist-voxels", "2", "--surface-contact-allowance-mm", "15")
    assert "10 mm" in capsys.readouterr().err
    assert A._reject(*A.SURF, *A.ALL, "--fine-voxel", "0.01", "--surface-contact-allowance-mm", "45")


# ============================================================================ toy scene (T46e: apple + neighbour 1 mm away)


def _scene():
    """The T46e rig: memory of two views, the target's lower half remembered as 'other' (stale samples — the T78c
    cause), the candidate seeds of the fine grid."""
    T = _t46e()
    mem = OracleMemory()
    mem.update(T._views())
    tp, op = mem.evidence(CODE_TARGET)
    low = tp[:, 2] < T.A[2] + 0.005
    tp, op = tp[~low], np.vstack([op, tp[low]])
    g, lo, shape, seed_grid, owner_grid = T._seed_grid()
    cand = oracle_candidates(tp, lo, T.VS, shape, 6.0 * T.VS)
    c = lo + cand * T.VS
    seed = seed_grid[cand[:, 0], cand[:, 1], cand[:, 2]]
    owner = owner_grid[cand[:, 0], cand[:, 1], cand[:, 2]]
    return T, tp, op, c, seed, owner


def _old_rule(c, seed, ev, vs):
    """The pre-T78f split, re-derived independently (no band here)."""
    from scipy.spatial import cKDTree

    d_t = cKDTree(ev.target_points).query(c)[0]
    max_d = float(ev.max_dist_voxels) * vs
    op = ev.other_points
    lo, hi = ev.target_points.min(0) - (2 * max_d + vs), ev.target_points.max(0) + (2 * max_d + vs)
    op = op[np.all((op >= lo) & (op <= hi), axis=1)]
    d_o = cKDTree(op).query(c)[0] if len(op) else np.full(len(c), np.inf)
    above = np.ones(len(c), bool) if ev.z_floor is None else c[:, 2] >= ev.z_floor
    surface = (d_t <= max_d) & (d_t < d_o) & above
    return seed & surface, surface | (seed & surface)


# ============================================================================ 2 off = the oracle split


def test_off_split_is_the_pre_t78f_rule_and_adds_no_key():
    T, tp, op, c, seed, owner = _scene()
    ev = OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=6.0, z_floor=T.TABLE_Z + T.VS)
    assert ev.contact_allowance_m is None
    sp = oracle_visible_split(c, seed, ev, T.VS)
    assert "allowance_extra" not in sp
    rm, sign = _old_rule(c, seed, ev, T.VS)
    np.testing.assert_array_equal(sp["removed"], rm)
    np.testing.assert_array_equal(sp["sign"], sign)
    sp2 = oracle_visible_split(c, seed, dataclasses.replace(ev, contact_allowance_m=None), T.VS)
    assert sp2.keys() == sp.keys()
    for k in sp:
        np.testing.assert_array_equal(np.asarray(sp[k]), np.asarray(sp2[k]))
    for bad in (0.0, -0.01, float("nan"), True):
        with pytest.raises(ValueError, match="contact_allowance_m"):
            OracleVisibleEvidence(target_points=tp, contact_allowance_m=bad)


def test_pipeline_off_never_builds_an_allowance_and_records_no_new_key(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    from AG3S.ag3s.runtime import pipeline as pl

    real = dataclasses.replace

    def boom(obj, **kw):
        if "contact_allowance_m" in kw:
            raise AssertionError("T78f allowance built with the flag off")
        return real(obj, **kw)

    monkeypatch.setattr(pl.dataclasses, "replace", boom)
    A = _t78a()
    out = []
    for extra in ({}, {"surface_contact_allowance_mm": None}):
        fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", **ON_BASE, **extra)
        obs, _, _ = A._rig(fw)
        A._update(ag3s, obs)
        field, _ = fw._build(ag3s)
        ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
        orc = field.stats["target_free"]["oracle"]
        info = ConstraintRecordWriter._write_exclusion_trace(field, {})
        assert ev.contact_allowance_m is None and len(ev.target_points), "the off run must carry target samples"
        assert not {"allowance_mm", "n_allowance_extra"} & set(orc)
        assert not {"allowance_mm", "n_allowance_extra"} & set(info["target_exclusion"]["oracle"])
        out.append((ag3s._esdf_builder.calls[-1], orc))
    (k0, o0), (k1, o1) = out
    assert k0.keys() == k1.keys()
    np.testing.assert_array_equal(k0["oracle_evidence"].target_points, k1["oracle_evidence"].target_points)
    assert k0["oracle_evidence"].z_floor == k1["oracle_evidence"].z_floor
    strip = lambda d: {k: v for k, v in d.items() if k not in ("memory", "ms")}   # noqa: E731 — timings differ
    assert strip(o0) == strip(o1)


# ============================================================================ 3 on (pure)


@pytest.mark.parametrize("m_mm", [10.0, 15.0])
def test_on_removes_the_seeds_within_m_whatever_the_other_samples(m_mm):
    T, tp, op, c, seed, owner = _scene()
    zf = T.TABLE_Z + T.VS
    ev = OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=6.0, z_floor=zf)
    off = oracle_visible_split(c, seed, ev, T.VS)
    on = oracle_visible_split(c, seed, dataclasses.replace(ev, contact_allowance_m=m_mm * 1e-3), T.VS)
    m = m_mm * 1e-3
    allow = seed & (on["d_t"] <= m) & (c[:, 2] >= zf)
    np.testing.assert_array_equal(on["removed"], off["removed"] | allow)            # union
    np.testing.assert_array_equal(on["allowance_extra"], allow & ~off["removed"])
    assert not np.any(on["removed"] & ~seed), "only seeds"
    assert not np.any(on["allowance_extra"] & (on["d_t"] > m)), "nothing beyond M"
    assert not np.any(on["allowance_extra"] & (c[:, 2] < zf)), "nothing below the support guard"
    np.testing.assert_array_equal(on["surface"], off["surface"])
    np.testing.assert_array_equal(on["band"], off["band"])
    np.testing.assert_array_equal(on["sign"] & ~seed, off["sign"] & ~seed)            # no non-seed sign flip added
    np.testing.assert_array_equal(on["sign"], off["sign"] | on["removed"])
    # what it is for: target seeds the d_t < d_o rule kept (stale 'other' samples on the lower half) within M go
    stale = seed & (owner == 0) & ~off["removed"] & (on["d_t"] <= m) & (c[:, 2] >= zf)
    assert stale.sum() > 10 and on["removed"][stale].all()
    assert int(on["allowance_extra"].sum()) >= int(stale.sum())
    # the documented limit: the neighbour 1 mm away loses its seeds within M of a target sample (scenario assumption)
    nb = seed & (owner == 1) & (on["d_t"] <= m) & (c[:, 2] >= zf)
    assert nb.any() and on["removed"][nb].all() and not off["removed"][nb].any()
    nb_far = seed & (owner == 1) & (on["d_t"] > m)
    assert nb_far.any() and not on["removed"][nb_far].any()


# ============================================================================ 4 tier (fake cuRobo)


def test_tier_allowance_clears_the_kept_target_seeds_and_keeps_table_and_far_neighbour():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    VS = T44.VS
    b = T44._builder(target_exclusion="ball_owned")
    g = T44._centres()
    sdf, owner = T44._sdf(g)
    surf = np.abs(sdf) < 0.4 * VS
    pts, own = g[surf], owner[surf]
    a_pts = pts[own == 0]
    upper = a_pts[:, 2] >= T44.A[2]
    zf = T44.TABLE_Z + VS
    ev = OracleVisibleEvidence(target_points=a_pts[upper], other_points=np.vstack([a_pts[~upper], pts[own != 0]]),
                               max_dist_voxels=6.0, z_floor=zf)

    def tier(evidence):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=evidence)
        return t

    base = tier(ev)
    on = tier(dataclasses.replace(ev, contact_allowance_m=0.010))
    v0, v1 = base.values.numpy(), on.values.numpy()
    above = g[..., 2] >= zf
    from scipy.spatial import cKDTree

    d_t = cKDTree(a_pts[upper]).query(g.reshape(-1, 3))[0].reshape(g.shape[:3])
    a_kept = (owner == 0) & (np.abs(sdf) < 0.75 * VS) & above & (d_t <= 0.010)
    assert np.min(np.abs(v0[a_kept])) == 0.0, "the oracle split keeps target seeds near stale 'other' samples"
    assert np.min(np.abs(v1[a_kept])) > 0.0, "every target seed within M above the guard is gone"
    table = (owner == 2) & (np.abs(sdf) < 0.5 * VS) & (np.linalg.norm(g[..., :2] - T44.A[:2], axis=-1) < 0.03)
    assert np.abs(v1[table]).max() < 1e-6, "the table under the target stays"
    b_far = (owner == 1) & (np.abs(sdf) < 0.4 * VS) & (d_t > 0.012)
    assert b_far.any() and np.abs(v1[b_far]).max() < 1e-6, "the neighbour beyond M stays"
    ex, ex0 = on.extra["exclusion"], base.extra["exclusion"]
    orc, orc0 = ex["oracle"], ex0["oracle"]
    assert not {"allowance_mm", "n_allowance_extra"} & set(orc0)
    assert orc["allowance_mm"] == 10.0 and orc["n_allowance_extra"] > 0
    assert ex["n_removed"] == ex0["n_removed"] + orc["n_allowance_extra"]
    assert len(on.extra["exclusion_trace"]["removed_ijk"]) == ex["n_removed"]
    assert CuroboFieldBuilder._exclusion_counts([on])["oracle_split"]["n_allowance_extra"] == orc["n_allowance_extra"]
    # off again = the base tier, value for value
    assert torch.equal(tier(ev).values, base.values)
    # no candidates (empty evidence): the keys are there, the count is 0
    empty = tier(OracleVisibleEvidence(target_points=np.zeros((0, 3)), contact_allowance_m=0.010))
    assert empty.extra["exclusion"]["oracle"]["n_allowance_extra"] == 0


# ============================================================================ 5 pipeline · record


def test_pipeline_on_passes_the_allowance_and_records_it(monkeypatch):
    import json

    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    A = _t78a()
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", **ON_BASE, surface_contact_allowance_mm=15.0)
    obs, _, _ = A._rig(fw)
    A._update(ag3s, obs)
    field, _ = fw._build(ag3s)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    assert ev.contact_allowance_m == pytest.approx(0.015)
    orc = field.stats["target_free"]["oracle"]
    assert orc["allowance_mm"] == 15.0 and orc["n_allowance_extra"] is None   # the fake builder makes no split
    # a builder that splits: its count reaches `target_exclusion.oracle`
    rec_builder = ag3s._esdf_builder
    orig = type(rec_builder).update

    def update(self, cameras, **kw):
        field = orig(self, cameras, **kw)
        field.stats["target_free"]["oracle_split"] = {"allowance_mm": 15.0, "n_allowance_extra": 7}
        return field

    monkeypatch.setattr(type(rec_builder), "update", update)
    field, _ = fw._build(ag3s)
    info = ConstraintRecordWriter._write_exclusion_trace(field, {})
    te = json.loads(json.dumps(info["target_exclusion"]))
    assert te["oracle"]["allowance_mm"] == 15.0 and te["oracle"]["n_allowance_extra"] == 7
    ag3s.reset()
    ag3s._esdf_builder = rec_builder          # reset makes a new (real) builder; the flag must survive it
    assert ag3s.config.esdf.surface_contact_allowance_mm == 15.0
    A._update(ag3s, obs)
    fw._build(ag3s)
    assert ag3s._esdf_builder.calls[-1]["oracle_evidence"].contact_allowance_m == pytest.approx(0.015)
