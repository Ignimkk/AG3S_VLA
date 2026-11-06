"""T98 — the support plane's samples leave `d_o` of the surface · band paths (`esdf.surface_do_support`,
`--surface-do-support`), default `include` (off). Off must be the oracle split (T46e / T78 / T96) call for call.

| # | property |
|---|---|
| 1 | config default `include` · round trip · refusals (bad value; `exclude` needs attention_surface + keep_last + no_ball); CLI flag → esdf key / record meta only for `exclude`; start-up refusals |
| 2 | off: `OracleVisibleEvidence` default None; `oracle_visible_split` = the pre-T98 rule re-derived (T46e scene), no `n_do_support_dropped` key; a plane that holds no sample changes nothing; the pipeline never builds a plane (`dataclasses.replace(do_plane=…)` explodes) and records no `do_support` key |
| 3 | evidence validation (same normalisation / checks as `SolidOwned.dn_plane`) |
| 4 | synthetic: a sphere sitting on a table with its bottom unseen. `include`: the seed just above the guard stays; `exclude`: it goes. The table seeds below `z_floor` stay in both; the neighbour sphere's samples (≥ 10 mm above the plane) keep protecting its seeds; `exclude` removes a superset of `include`; the T96 `d_N` is untouched |
| 5 | pipeline: `exclude` → the evidence carries the plane of `z_floor` + the RANSAC threshold; `target_exclusion.oracle.do_support` {applied, why, n_excluded, plane, threshold_mm}; no plane / no z_floor / no region → nothing applied and the record says why |
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence, SolidOwned, oracle_visible_split

ON_BASE = {"target_extent": "attention_surface", "surface_keep_last": True, "surface_no_ball": True}
P0M_SURF = {"surface_target_check": "overlap", "surface_keep_last": True, "surface_grasp_hold": True,
            "surface_no_ball": True}
VS = 0.005
TABLE = 0.80
THR = 0.008
NEW_KEYS = {"n_do_support_dropped"}


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


# ============================================================================ 1 config · CLI


def test_config_default_include_round_trip_and_refusals():
    assert AG3SConfig().esdf.surface_do_support == "include"
    AG3SConfig.from_dict({"esdf": {"surface_do_support": "include"}})            # include needs nothing
    cfg = AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_do_support": "exclude"}})
    assert cfg.esdf.surface_do_support == "exclude"
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in ("", "off", None, True, "both"):
        with pytest.raises(AG3SConfigError, match="surface_do_support"):
            AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_do_support": bad}})
    for missing in ({"target_extent": "attention_surface", "surface_keep_last": True},      # no surface_no_ball
                    {"target_extent": "attention_surface"},                                 # no keep_last
                    {"target_extent": "oracle_visible"},
                    {}):
        with pytest.raises(AG3SConfigError, match="surface_do_support"):
            AG3SConfig.from_dict({"esdf": {**missing, "surface_do_support": "exclude"}})


def test_cli_flag_maps_to_esdf_key_and_meta_only_for_exclude(capsys):
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    base = A._parse(*A.SURF, *A.ALL)
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert base.surface_do_support is None and "surface_do_support" not in set(sec0) | set(meta0)
    inc = A._parse(*A.SURF, *A.ALL, "--surface-do-support", "include")                      # explicit include = absent
    assert target_exclusion_section(inc) == sec0 and ifg_cl1_record_meta(inc) == meta0
    exc = A._parse(*A.SURF, *A.ALL, "--surface-do-support", "exclude")
    assert {k: v for k, v in target_exclusion_section(exc).items() if k not in sec0} == {"surface_do_support": "exclude"}
    assert {k: v for k, v in ifg_cl1_record_meta(exc).items() if k not in meta0} == {"surface_do_support": "exclude"}
    assert AG3SConfig.from_dict({"esdf": target_exclusion_section(exc)}).esdf.surface_do_support == "exclude"
    # start-up refusals: the effect would otherwise be believed
    assert A._reject(*A.SURF, *A.ALL, "--surface-do-support", "exclude").surface_do_support == "exclude"
    assert A._reject(*A.SURF, *A.ALL, "--surface-do-support", "include").surface_do_support == "include"
    for drop, msg in (("--surface-keep-last", "--surface-keep-last"), ("--surface-no-ball", "--surface-no-ball")):
        argv = [a for a in A.ALL if a != drop]
        with pytest.raises(SystemExit):
            A._reject(*A.SURF, *argv, "--surface-do-support", "exclude")
        assert msg in capsys.readouterr().err
    with pytest.raises(SystemExit):
        A._parse(*A.SURF, *A.ALL, "--surface-do-support", "both")
    capsys.readouterr()


# ============================================================================ 3 evidence validation


def test_evidence_do_plane_normalised_and_validated():
    kw = dict(target_points=np.zeros((1, 3)))
    assert OracleVisibleEvidence(**kw).do_plane is None and OracleVisibleEvidence(**kw).do_threshold_m is None
    ev = OracleVisibleEvidence(**kw, do_plane=np.array([0.0, 0.0, 2.0, 2.0 * TABLE]), do_threshold_m=THR)
    np.testing.assert_allclose(ev.do_plane, [0.0, 0.0, 1.0, TABLE])
    assert ev.do_threshold_m == THR
    with pytest.raises(ValueError, match="함께"):
        OracleVisibleEvidence(**kw, do_plane=np.array([0.0, 0.0, 1.0, TABLE]))
    with pytest.raises(ValueError, match="함께"):
        OracleVisibleEvidence(**kw, do_threshold_m=THR)
    for bad in (0.0, -0.001, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="do_threshold_m"):
            OracleVisibleEvidence(**kw, do_plane=np.array([0.0, 0.0, 1.0, TABLE]), do_threshold_m=bad)
    for bad in (np.zeros(4), np.array([0.0, 0.0, 1.0, np.nan])):
        with pytest.raises(ValueError, match="do_plane"):
            OracleVisibleEvidence(**kw, do_plane=bad, do_threshold_m=THR)


# ============================================================================ 4 synthetic scene (table + sphere + neighbour)


def _fib_sphere(centre, r, n=3000):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.asarray(centre) + r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _scene():
    """A target sphere (r 30 mm) sitting on the table; its bottom (z < table + 12 mm) was never seen. A neighbour sphere
    15 mm beside it (also bottom unseen). Table samples every 2 mm at the plane. Explicit seeds, see `SEEDS`."""
    tgt = _fib_sphere([0.0, 0.0, TABLE + 0.03], 0.03)
    tgt = tgt[tgt[:, 2] >= TABLE + 0.012]                                     # bottom unseen
    nb = _fib_sphere([0.075, 0.0, TABLE + 0.03], 0.03)
    nb = nb[nb[:, 2] >= TABLE + 0.012]                                        # >= 12 mm above the plane
    gx, gy = np.meshgrid(np.arange(-0.10, 0.18, 0.002), np.arange(-0.08, 0.08, 0.002))
    table = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, TABLE)]
    return tgt, np.vstack([nb, table]), nb, table


Z_FLOOR = TABLE + VS
# name -> centre. A: the target's bottom seed just above the guard. B/C: table seeds below the guard.
# D: a neighbour surface seed beside the target. E: a seed between target and neighbour, above the guard, near the table.
SEEDS = {"A": (0.0, 0.0, TABLE + 0.0075), "B": (0.0, 0.0, TABLE - 0.0025), "C": (0.05, 0.0, TABLE - 0.0025),
         "D": (0.045, 0.0, TABLE + 0.03), "E": (0.0375, 0.0, TABLE + 0.0075)}


def _split(ev):
    c = np.array(list(SEEDS.values()))
    out = oracle_visible_split(c, np.ones(len(c), bool), ev, VS)
    return {k: out for k in SEEDS}, out, dict(zip(SEEDS, range(len(SEEDS))))


def _ev(**kw):
    tgt, other, _, _ = _scene()
    return OracleVisibleEvidence(target_points=tgt, other_points=other, max_dist_voxels=6.0, z_floor=Z_FLOOR, **kw)


EXCL = dict(do_plane=np.array([0.0, 0.0, 1.0, TABLE]), do_threshold_m=THR)


def test_the_bottom_seed_above_the_guard_is_kept_by_include_and_removed_by_exclude():
    _, inc, ix = _split(_ev())
    _, exc, _ = _split(_ev(**EXCL))
    assert not inc["removed"][ix["A"]], "the table sample 7.5 mm below is nearer than the target's visible ring"
    assert exc["removed"][ix["A"]], "without the plane's samples the target is the nearest sample"
    assert exc["surface"][ix["A"]] and exc["sign"][ix["A"]]
    # the table seeds below z_floor stay in both (the guard, not the samples, protects them)
    for k in ("B", "C"):
        assert not inc["removed"][ix[k]] and not exc["removed"][ix[k]]
    # the neighbour stays protected by its samples above the plane
    assert not inc["removed"][ix["D"]] and not exc["removed"][ix["D"]]
    # exclude removes a superset of include, only seeds above z_floor
    assert not np.any(inc["removed"] & ~exc["removed"])
    c = np.array(list(SEEDS.values()))
    assert not np.any(exc["removed"] & (c[:, 2] < Z_FLOOR))
    assert "n_do_support_dropped" not in inc and exc["n_do_support_dropped"] > 1000     # the 2 mm table grid inliers


def test_neighbour_samples_off_the_plane_keep_competing_and_only_inliers_leave():
    tgt, other, nb, table = _scene()
    ev = _ev(**EXCL)
    out = oracle_visible_split(np.array([SEEDS["D"]]), np.ones(1, bool), ev, VS)
    assert out["d_o"][0] < 0.003, "the neighbour's own sample is still the nearest 'other'"
    # a sample 9 mm above the plane is an inlier (threshold 8 mm: no), 7 mm is (yes): the boundary is the RANSAC threshold
    probe = np.array([[0.2, 0.2, TABLE + 0.007], [0.3, 0.3, TABLE + 0.009]])
    ev2 = OracleVisibleEvidence(target_points=np.array([[0.2, 0.2, TABLE + 0.02], [0.3, 0.3, TABLE + 0.02]]),
                                other_points=probe, max_dist_voxels=6.0, **EXCL)
    out2 = oracle_visible_split(probe, np.ones(2, bool), ev2, VS)
    assert out2["n_do_support_dropped"] == 1
    assert out2["d_o"][1] == pytest.approx(0.0), "the 9 mm sample is not an inlier: it is still the nearest 'other'"
    assert out2["d_o"][0] > 0.1, "the 7 mm sample is an inlier: the seed on it no longer sees it"


def test_include_split_is_the_pre_t98_rule_and_a_plane_holding_no_sample_changes_nothing():
    from tests.ag3s.test_t78f_contact_allowance import _old_rule, _scene as scene78

    T, tp, op, c, seed, owner = scene78()
    zf = T.TABLE_Z + T.VS
    ev = OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=6.0, z_floor=zf)
    assert ev.do_plane is None and ev.do_threshold_m is None
    out = oracle_visible_split(c, seed, ev, T.VS)
    rem, sign = _old_rule(c, seed, ev, T.VS)
    np.testing.assert_array_equal(out["removed"], rem)
    np.testing.assert_array_equal(out["sign"], sign)
    assert not NEW_KEYS & set(out)
    # a plane 1 m below every sample: nothing is dropped, every array equals the off result
    far = oracle_visible_split(c, seed, dataclasses.replace(ev, do_plane=[0, 0, 1, -1.0], do_threshold_m=THR), T.VS)
    assert far["n_do_support_dropped"] == 0
    for k in ("removed", "kept_other", "sign", "surface", "band", "d_t", "d_o"):
        np.testing.assert_array_equal(far[k], out[k])


def test_t96_tau_protection_d_n_is_not_changed_by_the_plane():
    tgt, other, _, _ = _scene()
    c = np.array(list(SEEDS.values()))
    sol = SolidOwned(protect_m=0.005)
    base = oracle_visible_split(c, np.ones(len(c), bool), _ev(solid=sol), VS)
    on = oracle_visible_split(c, np.ones(len(c), bool), _ev(solid=sol, **EXCL), VS)
    np.testing.assert_array_equal(on["solid"]["d_N"], base["solid"]["d_N"])        # still every sample (T96 as is)
    np.testing.assert_array_equal(on["solid"]["protected"], base["solid"]["protected"])
    assert on["solid"]["n_dn_support_dropped"] == 0
    assert np.array_equal(on["d_o"][[0]] > base["d_o"][[0]], [True])                # d_o itself did change at A


# ============================================================================ 2 pipeline off / 5 pipeline on


def test_pipeline_off_never_builds_a_plane_and_records_no_new_key(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    from AG3S.ag3s.runtime import pipeline as pl

    real = dataclasses.replace

    def boom(obj, **kw):
        if "do_plane" in kw:
            raise AssertionError("T98 plane built with the flag off")
        return real(obj, **kw)

    monkeypatch.setattr(pl.dataclasses, "replace", boom)
    A = _t78a()
    out = []
    for extra in ({}, {"surface_do_support": "include"}):
        fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **P0M_SURF,
                            **extra)
        obs, _, _ = A._rig(fw)
        A._update(ag3s, obs, ats_points=A._apple_points(fw))
        field, _ = fw._build(ag3s)
        ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
        orc = field.stats["target_free"]["oracle"]
        info = ConstraintRecordWriter._write_exclusion_trace(field, {})
        assert ev.do_plane is None and ev.do_threshold_m is None and len(ev.target_points)
        assert "do_support" not in orc and "do_support" not in info["target_exclusion"]["oracle"]
        out.append((ag3s._esdf_builder.calls[-1], orc))
    (k0, o0), (k1, o1) = out
    assert k0.keys() == k1.keys()
    np.testing.assert_array_equal(k0["oracle_evidence"].target_points, k1["oracle_evidence"].target_points)
    assert k0["oracle_evidence"].z_floor == k1["oracle_evidence"].z_floor
    strip = lambda d: {k: v for k, v in d.items() if k not in ("memory", "ms")}   # noqa: E731 — timings differ
    assert strip(o0) == strip(o1)


def test_pipeline_exclude_passes_the_z_floor_plane_and_records_it(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    A = _t78a()
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **P0M_SURF,
                        surface_do_support="exclude")
    obs, _, _ = A._rig(fw)
    rec = A._update(ag3s, obs, ats_points=A._apple_points(fw))
    assert rec["segmentation"]
    field, _ = fw._build(ag3s)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    thr = float(ag3s.config.support_surface.distance_threshold)
    assert ev.do_threshold_m == pytest.approx(thr)
    assert np.linalg.norm(ev.do_plane[:3]) == pytest.approx(1.0)
    # the plane is the one z_floor was made from: its height at the region's centre + one fine voxel = z_floor
    c = ag3s._oracle_memory.evidence(1)[0].mean(axis=0)
    z_plane = (ev.do_plane[3] - ev.do_plane[0] * c[0] - ev.do_plane[1] * c[1]) / ev.do_plane[2]
    assert z_plane == pytest.approx(fw.TABLE_Z) and ev.z_floor == pytest.approx(fw.TABLE_Z + fw.FINE)
    d = field.stats["target_free"]["oracle"]["do_support"]
    assert d["applied"] and d["why"] == "support_plane_under_region" and d["mode"] == "exclude"
    assert d["threshold_mm"] == pytest.approx(thr * 1e3) and d["n_excluded"] is None            # the fake builder: no split
    np.testing.assert_allclose(d["plane"], ev.do_plane, atol=1e-8)
    # a builder that splits: its count reaches `target_exclusion.oracle.do_support`
    rb = ag3s._esdf_builder
    orig = type(rb).update

    def update(self, cameras, **kw):
        f = orig(self, cameras, **kw)
        f.stats["target_free"]["oracle_split"] = {"n_do_support_dropped": 1234}
        return f

    monkeypatch.setattr(type(rb), "update", update)
    field, _ = fw._build(ag3s)
    te = json.loads(json.dumps(ConstraintRecordWriter._write_exclusion_trace(field, {})["target_exclusion"]))
    assert te["oracle"]["do_support"]["n_excluded"] == 1234 and te["oracle"]["do_support"]["applied"]


def test_pipeline_exclude_without_plane_or_region_changes_nothing_and_says_why(monkeypatch):
    A = _t78a()
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **P0M_SURF,
                        surface_do_support="exclude")
    obs, _, _ = A._rig(fw)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    # no support plane under the region → no z_floor → nothing applied
    field, _ = ag3s._build_esdf([object()], fw._apple_geometry(), support_surfaces=[], robot_state=np.zeros(2),
                                contact_context=fw.LEFT)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    d = field.stats["target_free"]["oracle"]["do_support"]
    assert ev.do_plane is None and ev.z_floor is None
    assert not d["applied"] and d["why"] == "no_z_floor" and d["n_excluded"] is None
    # no accepted region yet → an empty evidence that removes nothing; the record does not claim an exclusion
    fw2, ag3s2 = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **P0M_SURF,
                          surface_do_support="exclude")
    for o in A._rig(fw2)[0]:
        o.attention_map = None
    obs2, _, _ = A._rig(fw2)
    for o in obs2:
        o.attention_map = None
    A._update(ag3s2, obs2, ats_points=A._apple_points(fw2))
    field2, _ = fw2._build(ag3s2)
    ev2 = ag3s2._esdf_builder.calls[-1]["oracle_evidence"]
    d2 = field2.stats["target_free"]["oracle"].get("do_support")
    assert ev2.do_plane is None and len(ev2.target_points) == 0
    assert d2 is None or not d2["applied"]
