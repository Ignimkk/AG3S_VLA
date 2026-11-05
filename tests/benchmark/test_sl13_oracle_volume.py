"""SL13 -- ``--oracle-target-volume``: the GT target volume removed from the fine target-free layer, whoever owns the seed. CPU only.

What this file pins:

(a) Off is unchanged. The server argv (``version_server_argv`` + the version's flags) of every one of the 61 versions that
    existed before SL13 is bit-identical (sha256 measured on the registry of 8a7ba963); the flag's dest joins a record only
    when set; ``EsdfConfig.oracle_target_volume`` defaults off and ``serve_safe`` adds nothing to the ``esdf`` section.
(b) The flag is valid only with ``--target-extent oracle-visible`` (parser and config refuse it otherwise; the RB-Y1 server too).
(c) Versions ``-pc-oraclevol`` / ``-pc-mi10-oraclevol`` = the ``-oraclegt`` flags + the flag; they ask for the code image.
(d) The removal on a synthetic fine layer (a bowl on a table with a neighbour object): without the flag the seeds touching the table
    stay (owned by the table samples), with it the whole hull x [z_floor, top] column goes, the neighbour and the table below the
    floor stay; no code-1 sample adds nothing.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl12_oracle_panda import MI, NEW as GT, PC, _args

VOL = (f"{PC}-oraclevol", f"{MI}-oraclevol")
#: sha256 of ``json.dumps({name: version_server_argv(name) + list(flags)}, sort_keys=True)`` over the 61 AG3S_VERSIONS entries
#: of 8a7ba963 (the registry without the SL13 pair)
OLD_ARGV_SHA256 = "42e6c7f5165d1fc1de6dd614a09a16914eac3725a1a653129d8aa6bb1855af8c"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical():
    from AG3S.adapters.panda import profile as pf

    old = {v: pf.version_server_argv(v) + list(pf.AG3S_VERSIONS[v]) for v in pf.AG3S_VERSIONS if v not in pf.SL13_VERSIONS}
    assert len(old) == 61
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == OLD_ARGV_SHA256
    assert pf.SL13_VERSIONS == VOL
    assert not any("--oracle-target-volume" in f for f in old.values())
    assert not any(v.endswith("-lsfork") for v in VOL) and not set(VOL) & set(pf.PC_LATCH_VERSIONS)


def test_off_by_default_in_config_parser_and_section():
    from AG3S.ag3s.config import EsdfConfig
    from AG3S.trajopt import serve_safe as ss

    assert EsdfConfig().oracle_target_volume is False
    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    off = ap.parse_args(B + ["--target-extent", "oracle-visible", "--oracle-extent-identity", "gt"])
    on = ap.parse_args(B + ["--target-extent", "oracle-visible", "--oracle-extent-identity", "gt", "--oracle-target-volume"])
    assert off.oracle_target_volume is False and on.oracle_target_volume is True
    assert "oracle_target_volume" not in ss.target_exclusion_section(off)
    assert ss.target_exclusion_section(on)["oracle_target_volume"] is True
    assert {k: v for k, v in ss.target_exclusion_section(on).items() if k != "oracle_target_volume"} == ss.target_exclusion_section(off)


# ---------------------------------------------------------------------------------------------------- (b) refused without oracle-visible
def test_flag_refused_without_oracle_visible():
    from AG3S.ag3s.config import AG3SConfigError, EsdfConfig
    from AG3S.trajopt import serve_safe as ss

    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml", "--robot", "panda"]
    for extra in ([], ["--target-extent", "estimated"], ["--target-extent", "attention-surface"]):
        ap = ss.build_parser()
        args = ap.parse_args(B + extra + ["--oracle-target-volume"])
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, args)
    with pytest.raises(AG3SConfigError, match="oracle_target_volume"):
        EsdfConfig(oracle_target_volume=True).validate()
    with pytest.raises(AG3SConfigError, match="oracle_target_volume"):
        EsdfConfig(oracle_target_volume=True, target_extent="attention_surface").validate()
    EsdfConfig(oracle_target_volume=True, target_extent="oracle_visible").validate()
    # the RB-Y1 server refuses it too (Panda server only)
    ap = ss.build_parser()
    args = ap.parse_args(["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml",
                          "--target-extent", "oracle-visible", "--oracle-target-volume"])
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, args)


# ---------------------------------------------------------------------------------------------------- (c) the versions
def test_versions_are_the_oraclegt_flags_plus_the_flag(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    for v, gt, base in zip(VOL, GT, (PC, MI)):
        assert list(reg[v]) == list(reg[gt]) + ["--oracle-target-volume"]
        assert pf.is_final_version(v) and pf.needs_oracle_segmentation(v)
        assert pf.version_server_argv(v) == pf.version_server_argv(base)
        a = _args(v, monkeypatch)
        ps.check_unported_flags(a, None)
        assert a.oracle_target_volume is True and a.target_extent == "oracle-visible" and a.oracle_extent_identity == "gt"
        assert pf.match_ag3s_version(a) == v
        g = _args(gt, monkeypatch)
        assert pf.match_ag3s_version(g) == gt and a.oracle_target_volume != g.oracle_target_volume
        assert "oracle_target_volume" not in pf.version_flag_values(g) and pf.version_flag_values(a)["oracle_target_volume"] is True
        assert pf.version_flag_values(a)["oracle_extent_identity"] == "gt"
        assert "oracle_target_volume" not in pf.version_flag_values(_args(base, monkeypatch))


def test_launchers_pass_the_client_flag_for_the_volume_versions(gpu_ok):
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    for v in VOL:
        got = lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")
        assert got[got.index("--target-extent") + 1] == "oracle-visible" and "--oracle-target-volume" in got
        a = lp.build_parser().parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                          "--ag3s-version", v])
        assert lp.oracle_client_argv("vla_v11ab", a) == ["--oracle-segmentation"]
        assert lp.oracle_client_argv("vla_only", a) == []
        assert lp.oracle_settings("vla_v11ab", a)["oracle_segmentation"]["target_rule"]
        wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, v)
        assert "--oracle-target-volume" in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
    for v in GT:
        assert "--oracle-target-volume" not in lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")


@pytest.mark.parametrize("version", VOL)
def test_volume_versions_build_on_cpu(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import wire
    from tests.benchmark.test_sl6a_panda_final import _Stub

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args(version, monkeypatch), gpu_warm_up=False)
    e = served._policy.ag3s.config.esdf
    assert e.target_extent == "oracle_visible" and e.oracle_extent_identity == "gt" and e.oracle_target_volume is True
    assert served.metadata[wire.ORACLE_SEG_REQUIRED] is True
    assert info["final_config"]["oracle_visible"]["target_volume"] is True
    json.dumps(info, default=str)
    gt = GT[0] if version == VOL[0] else GT[1]
    served0, info0 = ps.build_safe_server(_Stub(), _args(gt, monkeypatch), gpu_warm_up=False)
    assert served0._policy.ag3s.config.esdf.oracle_target_volume is False
    assert "target_volume" not in info0["final_config"]["oracle_visible"]


def test_call_record_carries_the_volume_numbers_only_when_present():
    from AG3S.adapters.panda import server as ps

    mk = lambda orc: type("S", (), {"ag3s": type("A", (), {   # noqa: E731
        "config": type("C", (), {"esdf": type("E", (), {"target_extent": "oracle_visible"})()})(),
        "_oracle_frame": {"segmentation": True},
        "_last_esdf_field": type("F", (), {"stats": {"target_free": {"oracle": orc}}})()})()})()
    off = ps.oracle_call_record(mk({"applied": True}))
    assert not any(k.startswith("volume_") for k in off["field"])
    on = ps.oracle_call_record(mk({"applied": True, "volume_applied": True, "volume_n_removed": 12,
                                   "volume_bbox_lo": [0.1, 0.1, 0.05], "volume_bbox_hi": [0.2, 0.2, 0.1]}))
    assert on["field"]["volume_applied"] is True and on["field"]["volume_n_removed"] == 12
    assert on["field"]["volume_bbox_hi"] == [0.2, 0.2, 0.1]
    json.dumps(on)


# ---------------------------------------------------------------------------------------------------- (d) the removal itself
VS = 0.005
SHAPE = (60, 60, 40)
ORG = np.zeros(3)
CODE_TARGET_XY = np.array([0.15, 0.15])
NEIGH_XY = np.array([0.24, 0.15])
Z_TABLE, Z_FLOOR = 0.05, 0.052


def _ring(c, r, z0, z1, step=0.0025, code_n=72):
    out = []
    for z in np.arange(z0, z1 + 1e-9, step):
        for t in np.linspace(0, 2 * np.pi, code_n, endpoint=False):
            out.append([c[0] + r * np.cos(t), c[1] + r * np.sin(t), z])
    return np.asarray(out)


def _scene():
    """Bowl (code 1) on a table, the neighbour (code 3) 90 mm away; the table samples run under the bowl too."""
    bowl = _ring(CODE_TARGET_XY, 0.04, 0.055, 0.09)
    neigh = _ring(NEIGH_XY, 0.025, 0.056, 0.08)
    g = np.arange(0.02, 0.28, 0.0025)
    table = np.array([[x, y, Z_TABLE] for x in g for y in g])
    return bowl, np.vstack([neigh, table]), neigh


def _vox(p):
    return np.floor(np.asarray(p, np.float64) / VS + 0.5).astype(np.int64)


def _layer(bowl, neigh, table_z_row=True):
    """The seeds a fine TSDF would hold: the surface voxels of every sample + the bowl's base (interior voxels just above the table)."""
    sites = np.full(SHAPE, -1, np.int32)
    pts = [bowl, neigh]
    g = np.arange(0.02, 0.28, VS)
    pts.append(np.array([[x, y, Z_TABLE] for x in g for y in g]))               # the table surface
    base = [[CODE_TARGET_XY[0] + dx, CODE_TARGET_XY[1] + dy, 0.055] for dx in np.arange(-0.04, 0.0401, VS)
            for dy in np.arange(-0.04, 0.0401, VS) if np.hypot(dx, dy) < 0.04]
    pts.append(np.asarray(base))                                                # the base voxels touching the table
    idx = _vox(np.vstack(pts))
    sites[idx[:, 0], idx[:, 1], idx[:, 2]] = 0
    return sites, _vox(np.asarray(base)), _vox(neigh)


class _Itg:
    def __init__(self, sites):
        self._site_index = sites


def _patch_numpy_layer(monkeypatch):
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder as B

    def seed_mask(itg, cand):
        return itg._site_index[cand[:, 0], cand[:, 1], cand[:, 2]] >= 0

    def clear_seeds(self, itg, cand, shape):
        nx, ny, nz = (int(v) for v in shape)
        cand = cand[np.all((cand >= 0) & (cand < np.array([nx, ny, nz])), axis=1)]
        if not len(cand):
            return 0, None
        before = int((itg._site_index[cand[:, 0], cand[:, 1], cand[:, 2]] >= 0).sum())
        itg._site_index[cand[:, 0], cand[:, 1], cand[:, 2]] = -1
        return before, cand

    monkeypatch.setattr(B, "_seed_mask", staticmethod(seed_mask))
    monkeypatch.setattr(B, "_clear_seeds", clear_seeds)
    return B


def _run(B, sites, ev, volume):
    itg = _Itg(sites.copy())
    b = object.__new__(B)
    n, sign, excl = b._oracle_exclusion(itg, ev, ORG, VS, SHAPE)
    if volume:
        n, sign, excl = b._volume_exclusion(itg, ev, ORG, VS, SHAPE, n, sign, excl)
    return itg._site_index >= 0, n, sign, excl


def _evidence(bowl, others, **kw):
    from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence

    return OracleVisibleEvidence(target_points=bowl, other_points=others, max_dist_voxels=6.0, z_floor=Z_FLOOR, **kw)


def test_volume_removes_the_bowl_column_and_keeps_the_neighbour(monkeypatch):
    B = _patch_numpy_layer(monkeypatch)
    bowl, others, neigh = _scene()
    sites, base_ijk, neigh_ijk = _layer(bowl, neigh)
    ev = _evidence(bowl, others)
    left0, n0, _, ex0 = _run(B, sites, ev, volume=False)
    # SL12 as it was: the seeds touching the table are owned by the table samples and stay
    inner = base_ijk[np.hypot(*(base_ijk[:, :2] * VS - CODE_TARGET_XY).T) <= 0.025]
    assert len(inner) > 20 and left0[tuple(inner.T)].all() and "target_volume" not in ex0["counts"]
    left1, n1, sign1, ex1 = _run(B, sites, dataclasses.replace(ev, target_volume=True), volume=True)
    assert not left1[tuple(base_ijk.T)].any()                                   # every base seed is gone
    assert left1[tuple(neigh_ijk.T)].all()                                      # the neighbour is untouched
    tv = ex1["counts"]["target_volume"]
    assert tv["applied"] is True and tv["n_extra_removed"] == n1 - n0 > 0
    assert tv["n_extra_removed"] == int((left0 & ~left1).sum()) and int((left1 & ~left0).sum()) == 0
    assert ex1["counts"]["n_removed"] == ex0["counts"]["n_removed"] + tv["n_extra_removed"]
    # the table below the floor and every voxel outside hull x [z_floor, top] is as without the flag
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    c = np.stack([ii, jj, kk], -1) * VS
    from scipy.spatial import ConvexHull
    eq = ConvexHull(bowl[:, :2]).equations
    in_hull = (c[..., :2] @ eq[:, :2].T + eq[:, 2]).max(-1) <= VS
    in_vol = in_hull & (c[..., 2] >= Z_FLOOR) & (c[..., 2] <= bowl[:, 2].max() + VS)
    assert (left0 & ~left1 & ~in_vol).sum() == 0
    assert not (left1 & in_vol).any()                                           # nothing of any owner left inside the volume
    assert left1[~in_vol].sum() == left0[~in_vol].sum()
    table_under_bowl = _vox([CODE_TARGET_XY[0], CODE_TARGET_XY[1], Z_TABLE])
    assert left1[tuple(table_under_bowl)]                                       # below z_floor -> the table stays
    # the removed voxels are in the trace and the sign set; the record has the hull's bbox
    removed = {tuple(r) for r in ex1["trace"]["removed_ijk"]}
    assert {tuple(r) for r in base_ijk} <= removed and sign1 is not None
    assert {tuple(r) for r in base_ijk} <= {tuple(r) for r in sign1}
    lo, hi = np.array(tv["bbox_lo"]), np.array(tv["bbox_hi"])
    assert np.allclose(lo[:2], bowl[:, :2].min(0) - VS, atol=1e-5) and np.allclose(hi[:2], bowl[:, :2].max(0) + VS, atol=1e-5)
    assert abs(lo[2] - Z_FLOOR) < 1e-9 and abs(hi[2] - (bowl[:, 2].max() + VS)) < 1e-5
    json.dumps(tv)


def test_volume_flag_off_is_the_same_call(monkeypatch):
    """``target_volume`` False (the default): the tier never calls ``_volume_exclusion`` -- the evidence field defaults off."""
    from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence

    bowl, others, _ = _scene()
    assert OracleVisibleEvidence(target_points=bowl).target_volume is False
    assert _evidence(bowl, others).target_volume is False


def test_no_code1_sample_adds_nothing(monkeypatch):
    B = _patch_numpy_layer(monkeypatch)
    bowl, others, neigh = _scene()
    sites, _, _ = _layer(bowl, neigh)
    ev = dataclasses.replace(_evidence(np.zeros((0, 3)), others), target_volume=True)
    left0, _, _, _ = _run(B, sites, ev, volume=False)
    left1, n1, sign1, ex1 = _run(B, sites, ev, volume=True)
    assert np.array_equal(left0, left1) and n1 == 0
    tv = ex1["counts"]["target_volume"]
    assert tv["applied"] is False and tv["n_extra_removed"] == 0 and tv["bbox_lo"] is None


def test_split_hull_dilation_degenerate_and_no_floor():
    from AG3S.ag3s.fields.curobo_builder import target_volume_split

    # a square of samples: the volume is the square dilated by one voxel (centre test), z within [floor, top + vs]
    sq = np.array([[x, y, z] for x in (0.1, 0.2) for y in (0.1, 0.2) for z in (0.06, 0.09)])
    out = target_volume_split(sq, ORG, VS, SHAPE, z_floor=0.07)
    assert out["applied"] and not out["degenerate"] and out["n_points"] == 8
    c = ORG + out["cand"] * VS
    assert c[:, 0].min() == pytest.approx(0.095) and c[:, 0].max() == pytest.approx(0.205)
    assert c[:, 2].min() >= 0.07 - 1e-9 and c[:, 2].max() <= 0.09 + VS + 1e-9
    assert len(out["cand"]) == len({tuple(r) for r in out["cand"]}) == 23 * 23 * 6       # x, y: 95..205 mm; z: 70..95 mm
    # collinear samples: no hull -> the xy AABB +- one voxel
    line = np.array([[0.1, 0.1, 0.06], [0.2, 0.1, 0.06], [0.15, 0.1, 0.07]])
    out = target_volume_split(line, ORG, VS, SHAPE, z_floor=0.055)
    assert out["applied"] and out["degenerate"] and len(out["cand"]) > 0
    c = ORG + out["cand"] * VS
    assert c[:, 1].min() == pytest.approx(0.095) and c[:, 1].max() == pytest.approx(0.105)
    # no z_floor: from the lowest sample - one voxel
    out = target_volume_split(sq, ORG, VS, SHAPE, z_floor=None)
    assert (ORG + out["cand"] * VS)[:, 2].min() == pytest.approx(0.055)
    assert not target_volume_split(np.zeros((0, 3)), ORG, VS, SHAPE, z_floor=0.05)["applied"]


def test_volume_on_a_torch_layer():
    """The same removal through the real ``_seed_mask`` / ``_clear_seeds`` on a torch ``site_index`` (skipped without torch)."""
    torch = pytest.importorskip("torch")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder as B

    bowl, others, neigh = _scene()
    sites, base_ijk, neigh_ijk = _layer(bowl, neigh)
    itg = _Itg(torch.as_tensor(sites))
    ev = dataclasses.replace(_evidence(bowl, others), target_volume=True)
    b = object.__new__(B)
    n, sign, excl = b._oracle_exclusion(itg, ev, ORG, VS, SHAPE)
    left_mid = (itg._site_index >= 0).numpy().copy()
    inner = base_ijk[np.hypot(*(base_ijk[:, :2] * VS - CODE_TARGET_XY).T) <= 0.025]
    assert left_mid[tuple(inner.T)].all()
    n, sign, excl = b._volume_exclusion(itg, ev, ORG, VS, SHAPE, n, sign, excl)
    left = (itg._site_index >= 0).numpy()
    assert not left[tuple(base_ijk.T)].any() and left[tuple(neigh_ijk.T)].all()
    assert excl["counts"]["target_volume"]["n_extra_removed"] == int((left_mid & ~left).sum())
