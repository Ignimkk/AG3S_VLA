"""T96e — the hand layer's removed seeds are free space (`esdf.surface_removed_sign_free`, `--surface-removed-sign-free`),
default off.

Diagnosis (T96.impl.md "T96e"): every hand-layer voxel of the replay's "listed as removed but the hand layer still ≤ 0"
class (B0p 1,447 · P0f 1,163) has `|d|` = 1 or √2 voxels — a remaining seed within the sign correction's purity radius
(`_force_sign_on_pure_voxels`, `|d| > 1.5 vs`), so the TSDF-interior sign stayed. The flag gives every removed-list voxel
`|d|` after the distance transform; nothing else changes.

| # | property |
|---|---|
| 1 | config default off · round trip · refusals; CLI key / record meta only when given; start-up refusal |
| 2 | off: `_removed_sign_free` never runs, values and record keys = the explicit-off builder = the default builder |
| 3 | synthetic (fake cuRobo), every removal path (surface · band, contact allowance, hull fill, solid, solid + shadow): removed seeds next to kept seeds were ≤ 0 off and are `|d|` > 0 on; every other voxel (kept seeds, the neighbour, the table, the non-seed sign candidates) is bit-identical; the record counts |
| 4 | the numpy and the torch path of `_removed_sign_free` agree; a 0 (still a site) is counted and left |
| 5 | the server-arg lines (P0m + flag, Q4 + flag) parse, pass the start-up refusals and reach the configs |
"""

from __future__ import annotations

import dataclasses
import pathlib
import shlex

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, OracleVisibleEvidence

RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261104T050351+0900_054748e6a73c_34e19a21953d/"
                   "impl/T96")
P0M = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261104T045052+0900_054748e6a73c_a02802211ca9/"
                   "verify/T96/kit/server_flags_p0m.txt")


# ============================================================================ 1 config · CLI


def test_config_default_round_trip_and_refusals():
    e = AG3SConfig().esdf
    assert e.surface_removed_sign_free is False
    on = AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_removed_sign_free": True}})
    assert on.esdf.surface_removed_sign_free is True
    assert AG3SConfig.from_dict(on.to_dict()).esdf.surface_removed_sign_free is True
    with pytest.raises(AG3SConfigError, match="attention_surface"):
        AG3SConfig.from_dict({"esdf": {"surface_removed_sign_free": True}})
    with pytest.raises(AG3SConfigError, match="attention_surface"):
        AG3SConfig.from_dict({"esdf": {"target_extent": "oracle_visible", "surface_removed_sign_free": True}})
    for bad in (1, "yes", None):
        with pytest.raises(AG3SConfigError, match="true 또는 false"):
            AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_removed_sign_free": bad}})


def test_cli_key_and_meta_only_when_given_and_refusal(capsys):
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml", "--target-extent", "attention-surface"]
    off = p.parse_args(base)
    assert off.surface_removed_sign_free is False
    assert "surface_removed_sign_free" not in serve_safe.target_exclusion_section(off)
    assert "surface_removed_sign_free" not in serve_safe.ifg_cl1_record_meta(off)
    serve_safe._t96e_reject(p, off)
    on = p.parse_args(base + ["--surface-removed-sign-free"])
    assert serve_safe.target_exclusion_section(on)["surface_removed_sign_free"] is True
    assert serve_safe.ifg_cl1_record_meta(on)["surface_removed_sign_free"] is True
    serve_safe._t96e_reject(p, on)
    for ext in ([], ["--target-extent", "oracle-visible"]):
        a = p.parse_args(["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml", "--surface-removed-sign-free"] + ext)
        with pytest.raises(SystemExit):
            serve_safe._t96e_reject(p, a)
        assert "--surface-removed-sign-free" in capsys.readouterr().err


# ============================================================================ 2 · 3 tier (fake cuRobo)


def _t44():
    return pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")


def _builder(T44, flag):
    extra = {} if flag is None else {"target_extent": "attention_surface", "surface_removed_sign_free": flag}
    return T44._builder(target_exclusion="ball_owned", **extra)


def _tier(b, T44, torch, ev):
    (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                          ball=None, oracle=ev)
    return t


def _stale_evidence(T44):
    """The replay's mechanism: the target's upper half labelled target, its lower half remembered as "other" (a later
    frame relabelled the target's own surface code 2), the neighbour B (5 mm away) and the table. The split removes
    the target seeds nearer the target samples and keeps the ones nearer the stale samples → removed seeds touch kept
    target seeds."""
    VS = T44.VS
    g = T44._centres()
    sdf, owner = T44._sdf(g)
    surf = np.abs(sdf) < 0.4 * VS
    pts, own = g[surf], owner[surf]
    a_pts = pts[own == 0]
    upper = a_pts[:, 2] >= T44.A[2]
    ev = OracleVisibleEvidence(target_points=a_pts[upper], other_points=np.vstack([a_pts[~upper], pts[own != 0]]),
                               max_dist_voxels=6.0, z_floor=T44.TABLE_Z + VS)
    return ev, a_pts, pts[own == 1]


def _path_evidence(T44, path):
    from AG3S.ag3s.fields.curobo_builder import HullFill
    from AG3S.ag3s.runtime import attention_surface_post as P

    VS = T44.VS
    ev, a_pts, b_pts = _stale_evidence(T44)
    if path == "surface_band":
        return ev
    if path == "allowance":
        return dataclasses.replace(ev, contact_allowance_m=0.010)
    if path == "hull_fill":
        h, _ = P.convex_hull(a_pts)
        return dataclasses.replace(ev, hull_fill=HullFill(equations=h["equations"], lo=h["lo"], hi=h["hi"], pad_m=VS,
                                                          protect_points=b_pts, z_floor=T44.TABLE_Z + VS))
    T96c = pytest.importorskip("tests.ag3s.test_t96c_solid_shadow")
    ev2, sol, sol_shadow = T96c._t44_solid(T44)
    return dataclasses.replace(ev2, solid=sol if path == "solid" else sol_shadow)


def test_off_never_runs_t96e_and_is_the_default_builder(monkeypatch):
    torch = pytest.importorskip("torch")
    T44 = _t44()
    ev, _, _ = _stale_evidence(T44)
    monkeypatch.setattr(CuroboFieldBuilder, "_removed_sign_free",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("T96e ran while off"))))
    default = _tier(_builder(T44, None), T44, torch, ev)
    off = _tier(_builder(T44, False), T44, torch, ev)
    assert torch.equal(default.values, off.values)
    for t in (default, off):
        assert "removed_sign_free" not in t.extra["exclusion"]["oracle"]
        assert "removed_sign_free" not in CuroboFieldBuilder._exclusion_counts([t])["oracle_split"]
    for k in ("removed_ijk", "nontarget_kept_ijk"):
        np.testing.assert_array_equal(default.extra["exclusion_trace"][k], off.extra["exclusion_trace"][k])
    strip = lambda ex: {k: v for k, v in ex.items() if k != "classify_ms"}   # noqa: E731
    o0, o1 = dict(default.extra["exclusion"]["oracle"]), dict(off.extra["exclusion"]["oracle"])
    for o in (o0, o1):
        o.get("solid", {}).pop("split_ms", None)
    assert strip({**default.extra["exclusion"], "oracle": o0}) == strip({**off.extra["exclusion"], "oracle": o1})
    # the ball (non-oracle) path is never touched, even on
    on_b = _builder(T44, True)
    (t_ball,) = on_b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                                  ball=T44._ball(), owned=T44._owned())
    (t_ball0,) = _builder(T44, None)._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine",
                                                   tier_name="fine_no_target", ball=T44._ball(), owned=T44._owned())
    assert torch.equal(t_ball.values, t_ball0.values)


@pytest.mark.parametrize("path", ["surface_band", "allowance", "hull_fill", "solid", "solid_shadow"])
def test_removed_seeds_become_free_kept_seeds_and_every_other_voxel_unchanged(path):
    torch = pytest.importorskip("torch")
    T44 = _t44()
    VS = T44.VS
    ev = _path_evidence(T44, path)
    off = _tier(_builder(T44, False), T44, torch, ev)
    on = _tier(_builder(T44, True), T44, torch, ev)
    # the split is before the distance transform: same removed list, same kept list
    for k in off.extra["exclusion_trace"]:
        np.testing.assert_array_equal(off.extra["exclusion_trace"][k], on.extra["exclusion_trace"][k])
    rem = np.asarray(on.extra["exclusion_trace"]["removed_ijk"], np.int64).reshape(-1, 3)
    assert len(rem) > 20
    v0, v1 = off.values.numpy(), on.values.numpy()
    mask = np.zeros(v0.shape, bool)
    mask[rem[:, 0], rem[:, 1], rem[:, 2]] = True
    # (a) nothing but the removed list changes — bit for bit
    np.testing.assert_array_equal(v1[~mask], v0[~mask])
    # (b) every removed voxel is free: |d| of the off value, > 0
    r0, r1 = v0[mask], v1[mask]
    assert np.all(r1 > 0.0), "a removed seed is still <= 0"
    np.testing.assert_array_equal(r1, np.abs(r0))
    # (c) the mechanism: removed seeds next to a kept seed (|d| <= 1.5 vs) were <= 0 off and are now small and positive
    near = (r0 < 0) & (np.abs(r0) <= np.float32(1.5 * VS))
    assert near.sum() >= 5, f"{path}: no removed seed next to a kept seed (the test scene lost its point)"
    assert np.all(r1[near] <= np.float32(1.5 * VS) + 1e-7) and np.all(r1[near] >= VS - 1e-7)
    # (d) kept seeds stay sites (value 0, i.e. <= 0) — the neighbour B, the table and the kept target seeds
    g = T44._centres()
    sdf, owner = T44._sdf(g)
    sites0 = (v0 == 0.0)
    assert sites0.sum() > 0 and not (sites0 & mask).any()
    assert np.all(v1[sites0] == 0.0)
    b_surf = (owner == 1) & (np.abs(sdf) < 0.4 * VS)
    # the contact allowance (10 mm, "no obstacle at the grasp" assumption) removes part of B 5 mm away by design (T78f);
    # T96e only reads the removed list, so B's voxels off the list keep their surface in every path
    if path != "allowance":
        assert not (b_surf & mask).any()
    assert np.abs(v1[b_surf & ~mask]).max() < 1e-6, "the neighbour 5 mm away keeps its surface"
    table = (owner == 2) & (np.abs(sdf) < 0.5 * VS) & (np.linalg.norm(g[..., :2] - T44.A[:2], axis=-1) < 0.03)
    assert not (table & mask).any()
    assert np.abs(v1[table]).max() < 1e-6, "the table under the target stays"
    kept = np.asarray(on.extra["exclusion_trace"]["nontarget_kept_ijk"], np.int64).reshape(-1, 3)
    if len(kept):   # kept by the split and not removed later (hull fill · allowance remove some kept seeds afterwards)
        still = ~mask[kept[:, 0], kept[:, 1], kept[:, 2]]
        assert still.any() and np.all(v1[kept[still, 0], kept[still, 1], kept[still, 2]] <= 0.0)
    # (e) the record
    rec = on.extra["exclusion"]["oracle"]["removed_sign_free"]
    assert rec["applied"] is True and rec["n_removed_listed"] == len(rem)
    assert rec["n_flipped"] == int((r0 < 0).sum()) and rec["n_flipped_within_purity"] == int(near.sum())
    assert rec["n_zero_left"] == 0 and rec["purity_radius_m"] == pytest.approx(1.5 * VS)
    assert CuroboFieldBuilder._exclusion_counts([on])["oracle_split"]["removed_sign_free"] == rec
    # the negative values of the removed list off were all within the purity radius (the replay finding)
    assert rec["n_flipped"] == rec["n_flipped_within_purity"]


# ============================================================================ 4 numpy · torch


def test_numpy_and_torch_paths_agree_and_a_site_is_left():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    vs = 0.005
    vals = rng.choice(np.float32([-0.005, -0.00707, -0.02, 0.0, 0.005, 0.03]), size=(8, 8, 8)).astype(np.float32)
    idx = np.argwhere(rng.random((8, 8, 8)) < 0.4).astype(np.int64)
    a, t = vals.copy(), torch.as_tensor(vals.copy())
    ra = CuroboFieldBuilder._removed_sign_free(a, idx, vs)
    rt = CuroboFieldBuilder._removed_sign_free(t, idx, vs)
    np.testing.assert_array_equal(a, t.numpy())
    assert ra == rt
    m = np.zeros(vals.shape, bool)
    m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    np.testing.assert_array_equal(a[~m], vals[~m])
    np.testing.assert_array_equal(a[m], np.abs(vals[m]))
    assert ra["n_zero_left"] == int((vals[m] == 0).sum()) > 0
    assert ra["n_flipped"] == int((vals[m] < 0).sum())
    assert ra["n_flipped_within_purity"] == int(((vals[m] < 0) & (np.abs(vals[m]) <= np.float32(1.5 * vs))).sum())
    empty = CuroboFieldBuilder._removed_sign_free(vals.copy(), np.zeros((0, 3), np.int64), vs)
    assert empty["n_removed_listed"] == 0 and empty["n_flipped"] == 0


# ============================================================================ 5 server-arg lines


@pytest.mark.parametrize("name, base", [("server_flags_p0m_sign.txt", P0M),
                                        ("server_flags_q5_delta0.txt", RUN / "server_flags_q4_delta0.txt")])
def test_server_args_parse_refuse_check_and_config(monkeypatch, name, base):
    if not (RUN / name).exists() or not base.exists():
        pytest.skip("server-arg 파일이 없다")
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))
    line = (RUN / name).read_text().strip()
    assert shlex.split(line) == shlex.split(base.read_text().strip()) + ["--surface-removed-sign-free"]
    p = serve_safe.build_parser()
    a = p.parse_args(["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"] + shlex.split(line))
    serve_safe.reject_bad_flag_combinations(p, a)
    serve_safe.trajopt_config_from_args(a)
    sec = serve_safe.target_exclusion_section(a)
    assert sec["surface_removed_sign_free"] is True
    e = AG3SConfig.from_dict({"esdf": {**sec, "fine_voxel_size": 0.005}}).esdf
    assert e.surface_removed_sign_free and e.target_extent == "attention_surface"
    if "p0m" in name:
        assert e.surface_hull_fill and e.surface_contact_allowance_mm == 10.0 and not e.surface_solid_owned
    else:
        assert e.surface_solid_owned and e.surface_solid_dn_ever_target and e.surface_solid_shadow_mm == 80.0
    assert serve_safe.ifg_cl1_record_meta(a)["surface_removed_sign_free"] is True
