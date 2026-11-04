"""T96c — the camera-ray shadow of the target on the T96 hand layer (`esdf.surface_solid_shadow_mm`,
`--surface-solid-shadow-mm L`), default off. Off must be the T96 split call for call.

| # | property |
|---|---|
| 1 | config default None · round trip · refusals (needs `surface_solid_owned`; L finite > 0); CLI key / record meta only when given; start-up refusals |
| 2 | off (solid on, shadow off): no T96c step runs (pipeline · split · tier), no `shadow` key anywhere, same values |
| 3 | pure, non-ball shapes (cylinder · box · L) on a table, one camera seeing the front only, a neighbour 8 mm behind: the shadow removes the hidden back; the neighbour (observed), the τ band and the table keep every seed; the path is exactly its rule recomputed independently |
| 4 | the documented risk: a neighbour face no camera ever observed, inside the shadow and farther than τ from every sample, is removed (and only such seeds) |
| 5 | tier (fake cuRobo, torch): shadow counts · per camera · trace; candidates = the box's seeds beyond S and max_dist |
| 6 | pipeline: the cameras are stored with the accepted region (code-1 · depth · K · T), kept through A4 hold, frame mismatch → no shadow, reset clears; record `attention_surface.solid.shadow` |
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import (OracleVisibleEvidence, ShadowCamera, SolidOwned, SolidShadow,
                                             oracle_visible_split, shadow_box_voxels, shadow_hits)

ON_BASE = {"target_extent": "attention_surface", "surface_keep_last": True, "surface_no_ball": True,
           "surface_solid_owned": True}
L80 = 0.080


def _t96():
    return pytest.importorskip("tests.ag3s.test_t96_solid_owned")


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


# ============================================================================ 1 config · CLI


def test_config_default_round_trip_and_refusals():
    assert AG3SConfig().esdf.surface_solid_shadow_mm is None
    cfg = AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_shadow_mm": 80.0}})
    assert cfg.esdf.surface_solid_shadow_mm == 80.0
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    with pytest.raises(AG3SConfigError, match="surface_solid_shadow_mm"):
        AG3SConfig.from_dict({"esdf": {**{k: v for k, v in ON_BASE.items() if k != "surface_solid_owned"},
                                       "surface_solid_shadow_mm": 80.0}})
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, "80"):
        with pytest.raises(AG3SConfigError, match="surface_solid_shadow_mm"):
            AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_shadow_mm": bad}})


def test_cli_key_and_meta_only_when_given_and_refusals(capsys):
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    P0M = ("--surface-target-check", "overlap", "--surface-keep-last", "--surface-grasp-hold", "--surface-no-ball")
    base = A._parse(*A.SURF, *P0M, "--surface-solid-owned")
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert "surface_solid_shadow_mm" not in sec0 and "surface_solid_shadow_mm" not in meta0
    on = A._parse(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-shadow-mm", "80")
    assert {k: v for k, v in target_exclusion_section(on).items() if k not in sec0} == {"surface_solid_shadow_mm": 80.0}
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == {"surface_solid_shadow_mm": 80.0}
    assert AG3SConfig.from_dict({"esdf": target_exclusion_section(on)}).esdf.surface_solid_shadow_mm == 80.0
    assert A._reject(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-shadow-mm", "80").surface_solid_shadow_mm == 80.0
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, *P0M, "--surface-solid-shadow-mm", "80")
    assert "--surface-solid-owned 와 함께만" in capsys.readouterr().err
    for bad in ("0", "-5", "nan", "inf"):
        with pytest.raises(SystemExit):
            A._reject(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-shadow-mm", bad)
        assert "--surface-solid-shadow-mm" in capsys.readouterr().err


# ============================================================================ rigs: one camera, ray-cast


def _raycast(T, K, hw, parts, t0=0.05, t1=1.0, iters=400):
    """Sphere-traced pinhole render of `parts` (list of sdf callables): `(label (H, W) int, -1 = miss; depth (H, W)
    z-depth m, 0 = miss)`. Pixel (v, u) centre = integer (v, u) (the `region_from_codes` pinhole)."""
    H, W = hw
    vv, uu = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
    dirs = dc @ T[:3, :3].T
    norm = np.linalg.norm(dirs, axis=1)
    dirs /= norm[:, None]
    eye = T[:3, 3]
    t = np.full(len(dirs), t0)
    done = np.zeros(len(dirs), bool)
    f = lambda p: np.min(np.stack([g(p) for g in parts], -1), -1)   # noqa: E731
    for _ in range(iters):
        live = ~done & (t < t1)
        if not live.any():
            break
        d = f(eye + t[live, None] * dirs[live])
        hit = d < 2e-5
        idx = np.nonzero(live)[0]
        done[idx[hit]] = True
        t[idx[~hit]] += np.maximum(d[~hit], 1e-4)
    p = eye + t[:, None] * dirs
    lab = np.where(done, np.stack([g(p) for g in parts], -1).argmin(-1), -1)
    depth = np.where(done, t / norm, 0.0)                               # ray length → z-depth (dc has z = 1)
    return lab.reshape(H, W), depth.reshape(H, W)


def _camera(kind, hw=(120, 160), f=150.0, dist=0.40):
    """The front-above camera of the T96 rig (`EYE_DIR`), aimed at the target, and its target shadow camera."""
    T96 = _t96()
    tgt, nb = T96._shape(kind)
    aim = np.array([0.0, 0.0, T96.TABLE + 0.025])
    T = _t78a()._look_at(aim + dist * T96.EYE_DIR, aim)
    K = np.array([[f, 0.0, hw[1] / 2], [0.0, f, hw[0] / 2], [0.0, 0.0, 1.0]])
    table = lambda p: p[..., 2] - T96.TABLE   # noqa: E731
    lab, depth = _raycast(T, K, hw, [tgt, nb, table])
    return ShadowCamera(name="head", target=lab == 0, depth_m=depth, K=K, T_base_cam=T), lab


def _rig(kind, observed_neighbour="all"):
    """The T96 rig + one shadow camera. `observed_neighbour`: "all" = the episode memory holds every neighbour face
    (other views / frames), "visible" = only the faces this camera's direction sees (the T96 rig's `op`)."""
    T96 = _t96()
    r = T96._rig(kind)
    if observed_neighbour == "all":
        _, nb = T96._shape(kind)
        blo, bhi = np.array([-0.07, -0.07, T96.TABLE - 0.01]), np.array([0.07, 0.09, T96.TABLE + 0.07])
        nbp = T96._surface_samples(nb, blo, bhi, visible_only=False)
        table = r["op"][np.abs(r["op"][:, 2] - T96.TABLE) < 1e-9]
        r["op"] = np.vstack([nbp, table])
    cam, lab = _camera(kind)
    assert cam.target.sum() > 200 and (lab == 1).sum() > 0, "the camera sees the target and part of the neighbour"
    return r, cam


def _solid_shadow(r, cam, L=L80, protect=0.005):
    sol = _t96()._solid(r, protect=protect)
    sh = SolidShadow(length_m=L, cameras=(cam,), lo=sol.lo - L, hi=sol.hi + L)
    return sol, dataclasses.replace(sol, shadow=sh)


def _shadow_rule(r, cam, sol_sh, taken):
    """The shadow path re-derived independently (brute force per voxel, no shared helper)."""
    from scipy.spatial import cKDTree

    c, seed = r["c"], r["seed"]
    sh = sol_sh.shadow
    T, K = cam.T_base_cam, cam.K
    pc = (c - T[:3, 3]) @ T[:3, :3]
    hit = np.zeros(len(c), bool)
    H, W = cam.target.shape
    for i in np.nonzero(seed & (c[:, 2] >= r["zf"]) & np.all((c >= sh.lo) & (c <= sh.hi), axis=1) & ~taken)[0]:
        x, y, z = pc[i]
        if z <= 1e-6:
            continue
        u, v = int(np.floor(K[0, 0] * x / z + K[0, 2] + 0.5)), int(np.floor(K[1, 1] * y / z + K[1, 2] + 0.5))
        if 0 <= u < W and 0 <= v < H and cam.target[v, u]:
            dz = z - cam.depth_m[v, u]
            hit[i] = cam.depth_m[v, u] > 0 and 0 < dz <= sh.length_m
    d_n = cKDTree(r["op"]).query(c)[0]
    return hit & (d_n > sol_sh.protect_m), hit


# ============================================================================ 2 off = T96


def test_off_no_shadow_step_runs_and_the_split_is_t96(monkeypatch):
    from AG3S.ag3s.fields import curobo_builder as cb

    def boom(*a, **k):
        raise AssertionError("T96c step called with its flag off")

    r, cam = _rig("cylinder")
    sol, sol_sh = _solid_shadow(r, cam)
    ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
    ref = oracle_visible_split(r["c"], r["seed"], dataclasses.replace(ev, solid=sol), _t96().VS)
    for name in ("_shadow_split", "shadow_hits", "shadow_box_voxels"):
        monkeypatch.setattr(cb, name, boom)
    for e in (ev, dataclasses.replace(ev, solid=sol)):
        out = cb.oracle_visible_split(r["c"], r["seed"], e, _t96().VS)
        if e.solid is not None:
            assert not {k for k in out["solid"] if "shadow" in k}
            for k in ("removed", "sign", "surface", "band", "kept_other"):
                np.testing.assert_array_equal(out[k], ref[k])
    with pytest.raises(ValueError, match="T96c"):
        SolidOwned(protect_m=0.005, shadow=sol_sh.shadow)               # a shadow needs a solid
    for bad in (0.0, -0.01, float("nan"), True):
        with pytest.raises(ValueError, match="SolidShadow"):
            SolidShadow(length_m=bad, cameras=(cam,), lo=sol.lo, hi=sol.hi)


def test_pipeline_off_never_calls_the_t96c_steps(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    from AG3S.ag3s.runtime.pipeline import AG3S

    def boom(*a, **k):
        raise AssertionError("T96c step called with its flag off")

    for name in ("_surface_solid_shadow_store", "_surface_solid_shadow"):
        monkeypatch.setattr(AG3S, name, boom)
    A = _t78a()
    fw, ag3s = _t96()._wired_on(monkeypatch)
    obs, _, _ = A._rig(fw)
    assert A._update(ag3s, obs, ats_points=A._apple_points(fw))["segmentation"]
    field, _ = fw._build(ag3s)
    assert getattr(ag3s, "_surface_solid_cams", None) is None
    assert "solid_shadow_views" not in ag3s._surface_frame["t78"]
    sol = ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid
    assert sol.has_solid and sol.shadow is None
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert "shadow" not in s
    te = ConstraintRecordWriter._write_exclusion_trace(field, {})["target_exclusion"]
    assert "shadow" not in te["attention_surface"]["solid"]


def test_tier_shadow_off_is_value_for_value_t96(monkeypatch):
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields import curobo_builder as cb

    ev, sol, _ = _t44_solid(T44)
    b = T44._builder(target_exclusion="ball_owned")

    def tier(e):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=e)
        return t

    base = tier(dataclasses.replace(ev, solid=sol))
    monkeypatch.setattr(cb, "shadow_box_voxels", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    again = tier(dataclasses.replace(ev, solid=sol))
    assert torch.equal(again.values, base.values)
    s = again.extra["exclusion"]["oracle"]["solid"]
    assert not {k for k in s if "shadow" in k}
    assert not {"solid_shadow_removed_ijk", "solid_shadow_protected_ijk"} & set(again.extra["exclusion_trace"])


# ============================================================================ 3 pure: non-ball shapes, one camera


@pytest.mark.parametrize("kind", ["cylinder", "box", "L"])
def test_shadow_removes_the_hidden_back_keeps_neighbour_tau_band_and_table(kind):
    from scipy.spatial import cKDTree

    T96 = _t96()
    r, cam = _rig(kind)
    c, seed, owner = r["c"], r["seed"], r["owner"]
    ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
    sol, sol_sh = _solid_shadow(r, cam)
    t96 = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=sol), T96.VS)
    on = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=sol_sh), T96.VS)
    sp = on["solid"]
    d_n = cKDTree(r["op"]).query(c)[0]
    above = c[:, 2] >= r["zf"]
    tau = d_n <= 0.005
    # the T96 paths are untouched; the shadow only adds seeds
    for k in ("surface", "band"):
        np.testing.assert_array_equal(on[k], t96[k])
    np.testing.assert_array_equal(sp["fill"], t96["solid"]["fill"])
    np.testing.assert_array_equal(on["removed"], t96["removed"] | sp["shadow"])
    np.testing.assert_array_equal(on["sign"], t96["sign"] | sp["shadow"])
    assert not np.any(sp["shadow"] & t96["removed"]), "the shadow counts only seeds no other path removed"
    # invariants
    assert not np.any(sp["shadow"] & ~seed) and not np.any(sp["shadow"] & ~above) and not np.any(sp["shadow"] & tau)
    # exactly its rule
    rule, hit = _shadow_rule(r, cam, sol_sh, t96["removed"])
    np.testing.assert_array_equal(sp["shadow"], rule)
    np.testing.assert_array_equal(sp["protected_shadow"], hit & ~rule)
    assert sp["shadow_by_camera"] == {"head": int(rule.sum())} and sp["n_shadow_hits"] == int(hit.sum())
    # the neighbour (observed) and the table keep every seed
    assert not np.any(on["removed"] & (owner == 1)), "the neighbour 8 mm behind keeps every seed"
    assert not np.any(on["removed"] & (owner == 2)), "the table keeps every seed"
    # what it is for: the hidden back of the target (faces this camera does not see)
    tgt_seed = seed & (owner == 0) & above & ~tau
    hidden = tgt_seed & (T96._normals(r["tgt"], c) @ T96.EYE_DIR <= 0.15)
    new = sp["shadow"]
    assert (hidden & new).sum() >= 5, (kind, (hidden & new).sum())
    left96 = (tgt_seed & ~t96["removed"]).sum() / tgt_seed.sum()
    left_on = (tgt_seed & ~on["removed"]).sum() / tgt_seed.sum()
    assert left_on < left96, (kind, left_on, left96)
    # the shadow is bounded by L: every removed seed lies within L behind a target pixel
    sh_short = dataclasses.replace(sol_sh, shadow=dataclasses.replace(sol_sh.shadow, length_m=0.010))
    short = oracle_visible_split(c, seed, dataclasses.replace(ev, solid=sh_short), T96.VS)["solid"]["shadow"]
    assert short.sum() < new.sum() and not np.any(short & ~new)


def test_shadow_d_n_drops_the_support_plane_samples_like_d28():
    """D28 applies to the shadow's τ too: with `dn_plane`, table samples do not protect (z_floor guards the table)."""
    T96 = _t96()
    r, cam = _rig("cylinder")
    ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
    _, sol_sh = _solid_shadow(r, cam)
    plane = dataclasses.replace(sol_sh, dn_plane=np.array([0.0, 0.0, 1.0, T96.TABLE]), dn_threshold_m=0.008)
    a = oracle_visible_split(r["c"], r["seed"], dataclasses.replace(ev, solid=sol_sh), T96.VS)["solid"]
    b = oracle_visible_split(r["c"], r["seed"], dataclasses.replace(ev, solid=plane), T96.VS)["solid"]
    assert b["n_shadow_dn_support_dropped"] > 0 and a["n_shadow_dn_support_dropped"] == 0
    assert not np.any(a["shadow"] & ~b["shadow"] & ~b["fill"]) and b["shadow"].sum() + b["fill"].sum() >= a["shadow"].sum()
    assert not np.any(a["shadow"] & (r["owner"] != 0)), "without D28: no neighbour · table seed"
    assert not np.any(b["shadow"] & (r["owner"] == 2)), "the table: z_floor"
    # the D28 cost the shadow inherits (no `d_T ≤ d_N` here): a neighbour seed at the guard height whose samples
    # below plane + 8 mm left d_N — only in the band [z_floor, plane + threshold], only with d_N > τ (impl.md T96c)
    from scipy.spatial import cKDTree

    lost = b["shadow"] & (r["owner"] == 1)
    z = r["c"][lost, 2]
    assert np.all((z >= r["zf"]) & (z <= T96.TABLE + 0.008))
    op = r["op"][np.abs(r["op"][:, 2] - T96.TABLE) > 0.008]
    assert np.all(cKDTree(op).query(r["c"][lost])[0] > 0.005)


# ============================================================================ 4 the documented risk


def test_a_never_observed_neighbour_face_in_the_shadow_is_the_risk():
    """With only the camera-facing neighbour faces observed, any neighbour seed the shadow removes lies in the
    shadow and farther than τ from every observed sample — exactly the config's documented risk, nothing else."""
    from scipy.spatial import cKDTree

    T96 = _t96()
    for kind in ("cylinder", "box", "L"):
        r, cam = _rig(kind, observed_neighbour="visible")
        ev = OracleVisibleEvidence(target_points=r["tp"], other_points=r["op"], max_dist_voxels=6.0, z_floor=r["zf"])
        _, sol_sh = _solid_shadow(r, cam)
        sp = oracle_visible_split(r["c"], r["seed"], dataclasses.replace(ev, solid=sol_sh), T96.VS)["solid"]
        lost = sp["shadow"] & (r["owner"] == 1)
        d_n = cKDTree(r["op"]).query(r["c"])[0]
        assert np.all(d_n[lost] > 0.005)
        assert not np.any(sp["shadow"] & (r["owner"] == 2))


# ============================================================================ 5 tier (fake cuRobo)


def _t44_solid(T44, L=L80):
    T96 = _t96()
    ev, R, (g, sdf, owner) = T96._t44_evidence(T44)
    R_perp = R.copy()
    R_perp[:, 2] = T44.TABLE_Z
    pts = np.vstack([R, R_perp])
    from AG3S.ag3s.runtime import attention_surface_post as P

    h, _ = P.convex_hull(pts)
    sol = SolidOwned(protect_m=0.005, equations=h["equations"], lo=h["lo"], hi=h["hi"], target_points=pts,
                     info={"region_source": "this_request"})
    # a camera straight above the target (looking down): the visible cap shadows the lower half
    T = _t78a()._look_at(T44.A + [0.0, -0.01, 0.35], T44.A)
    K = np.array([[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
    parts = [lambda p: np.linalg.norm(p - T44.A, axis=-1) - T44.R, lambda p: np.linalg.norm(p - T44.B, axis=-1) - T44.R,
             lambda p: p[..., 2] - T44.TABLE_Z]
    lab, depth = _raycast(T, K, (120, 160), parts)
    cam = ShadowCamera(name="head", target=lab == 0, depth_m=depth, K=K, T_base_cam=T)
    return ev, sol, dataclasses.replace(sol, shadow=SolidShadow(length_m=L, cameras=(cam,), lo=sol.lo - L,
                                                                  hi=sol.hi + L))


def test_tier_records_the_shadow_by_path_and_camera_and_the_trace():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    ev, sol, sol_sh = _t44_solid(T44)
    b = T44._builder(target_exclusion="ball_owned")

    def tier(e):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=e)
        return t

    base, on = tier(dataclasses.replace(ev, solid=sol)), tier(dataclasses.replace(ev, solid=sol_sh))
    s0, s = base.extra["exclusion"]["oracle"]["solid"], on.extra["exclusion"]["oracle"]["solid"]
    assert s["shadow_mm"] == 80.0 and s["n_shadow_box_seeds"] > 0
    assert s["n_removed_shadow"] > 0
    assert s["n_removed_shadow_by_camera"] == {"head": s["n_removed_shadow"]}
    assert s["n_removed_total"] == s0["n_removed_total"] + s["n_removed_shadow"]
    for k in ("n_removed_surface", "n_removed_band", "n_removed_solid", "n_protected_tau"):
        assert s[k] == s0[k], k
    tr = on.extra["exclusion_trace"]
    assert len(tr["solid_shadow_removed_ijk"]) == s["n_removed_shadow"]
    assert len(tr["solid_shadow_protected_ijk"]) == s["n_protected_tau_shadow"]
    assert on.extra["exclusion"]["n_removed"] == base.extra["exclusion"]["n_removed"] + s["n_removed_shadow"]
    assert CuroboFieldBuilder._exclusion_counts([on])["oracle_split"]["solid"] == s
    for k in ("shadow_ms", "shadow_candidates_ms"):
        assert s[k] >= 0.0
    # the neighbour B (5 mm from A) keeps its surface, the table under A stays
    v1 = on.values.numpy()
    g = T44._centres()
    sdf, owner = T44._sdf(g)
    assert np.abs(v1[(owner == 1) & (np.abs(sdf) < 0.4 * T44.VS)]).max() < 1e-6
    table = (owner == 2) & (np.abs(sdf) < 0.5 * T44.VS) & (np.linalg.norm(g[..., :2] - T44.A[:2], axis=-1) < 0.03)
    assert np.abs(v1[table]).max() < 1e-6
    # every removed shadow seed is inside the box; the box reaches beyond S
    org = np.asarray(T44.ORIGIN, float)                                  # first voxel centre
    P_ = org + tr["solid_shadow_removed_ijk"] * T44.VS
    assert np.all((P_ >= sol_sh.shadow.lo - 1e-9) & (P_ <= sol_sh.shadow.hi + 1e-9))
    box = shadow_box_voxels(sol_sh.shadow, org, T44.VS, (T44.N,) * 3)
    assert box is not None and len(box) > s["n_solid_voxels"]


def test_shadow_hits_geometry():
    """A camera at the origin looking along +z at a target pixel patch observed at 0.5 m: points behind it within L
    hit, in front / beyond L / off-target do not."""
    K = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    tgt = np.zeros((21, 21), bool)
    tgt[8:13, 8:13] = True
    cam = ShadowCamera(name="c", target=tgt, depth_m=np.full((21, 21), 0.5), K=K, T_base_cam=np.eye(4))
    sh = SolidShadow(length_m=0.08, cameras=(cam,), lo=-np.ones(3), hi=np.ones(3))
    p = np.array([[0, 0, 0.49], [0, 0, 0.5], [0, 0, 0.501], [0, 0, 0.58], [0, 0, 0.581], [0.05, 0, 0.55],
                  [0, 0, -0.55]])
    np.testing.assert_array_equal(shadow_hits(sh, p)[0], [False, False, True, True, False, False, False])


# ============================================================================ 6 pipeline · record


def test_pipeline_stores_the_cameras_with_the_region_and_records(monkeypatch):
    import json

    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    from AG3S.ag3s.runtime.oracle_segmentation import CODE_TARGET

    A = _t78a()
    fw, ag3s = _t96()._wired_on(monkeypatch, surface_solid_shadow_mm=80.0)
    obs, _, _ = A._rig(fw)
    assert A._update(ag3s, obs, ats_points=A._apple_points(fw))["segmentation"]
    st = ag3s._surface_solid_cams
    assert st["frame_index"] == ag3s._surface_solid_region["frame_index"]
    codes = ag3s._surface_codes
    names = [c["name"] for c in st["cameras"]]
    assert names and all((codes[n] == CODE_TARGET).sum() > 0 for n in names)
    by = {o.camera_id.value: o for o in obs}
    for c in st["cameras"]:
        np.testing.assert_array_equal(c["target"], codes[c["name"]] == CODE_TARGET)
        np.testing.assert_array_equal(c["depth_m"], np.asarray(by[c["name"]].depth, np.float64))
        np.testing.assert_array_equal(c["T_base_cam"], by[c["name"]].T_base_cam)
    assert ag3s._surface_frame["t78"]["solid_shadow_views"]["stored"]
    field, _ = fw._build(ag3s)
    sol = ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid
    assert sol.has_solid and sol.shadow is not None and sol.shadow.length_m == pytest.approx(0.08)
    np.testing.assert_allclose(sol.shadow.lo, sol.lo - 0.08)
    np.testing.assert_allclose(sol.shadow.hi, sol.hi + 0.08)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert s["shadow"]["applied"] and s["shadow"]["shadow_mm"] == 80.0
    assert set(s["shadow"]["cameras"]) == set(names)
    te = json.loads(json.dumps(ConstraintRecordWriter._write_exclusion_trace(field, {})["target_exclusion"]))
    assert te["attention_surface"]["solid"]["shadow"]["applied"]
    # A4 hold: no selection, the region and its cameras stand (same frame index → the shadow still applies)
    ag3s.frame_index += 1
    ag3s.set_grasp_active(True)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    assert ag3s._surface_frame["why"] == "grasp_hold"
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert s["region_source"] == "last_accepted" and s["shadow"]["applied"]
    # a mismatched frame (cameras of another request) → no shadow, recorded
    ag3s._surface_solid_cams = {**ag3s._surface_solid_cams, "frame_index": -5}
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert not s["shadow"]["applied"] and s["shadow"]["why"] == "camera_frame_mismatch"
    assert ag3s._esdf_builder.calls[-1]["oracle_evidence"].solid.shadow is None
    ag3s.reset()
    assert ag3s._surface_solid_cams is None


def test_pipeline_no_region_records_no_solid_shadow(monkeypatch):
    A = _t78a()
    fw, ag3s = _t96()._wired_on(monkeypatch, surface_solid_shadow_mm=80.0)
    obs, _, _ = A._rig(fw)
    for o in obs:
        o.attention_map = None
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert not s["applied"] and s["shadow"] == {"applied": False, "why": "no_solid", "shadow_mm": 80.0}


# ============================================================================ 7 the Q3 server-arg line


def test_q3_server_args_parse_refuse_check_and_config(monkeypatch):
    import pathlib
    import shlex

    run = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261104T050351+0900_054748e6a73c_34e19a21953d/"
                       "impl/T96")
    if not (run / "server_flags_q3_delta0.txt").exists():
        pytest.skip("Q3 server-arg 파일이 없다")
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))
    line = (run / "server_flags_q3_delta0.txt").read_text().strip()
    q2 = (run / "server_flags_q2_delta0.txt").read_text().strip()
    assert shlex.split(line) == shlex.split(q2) + ["--surface-solid-shadow-mm", "80"]
    p = serve_safe.build_parser()
    a = p.parse_args(["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"] + shlex.split(line))
    serve_safe.reject_bad_flag_combinations(p, a)
    c = serve_safe.trajopt_config_from_args(a)
    assert c.collision.palm_cover_margin == 0.005 and c.collision.palm_shell_margin is None
    sec = serve_safe.target_exclusion_section(a)
    assert sec["surface_solid_shadow_mm"] == 80.0 and sec["surface_solid_owned"] and sec["surface_solid_dilate_mm"] == 0.0
    e = AG3SConfig.from_dict({"esdf": {**sec, "fine_voxel_size": 0.005}}).esdf
    assert e.surface_solid_shadow_mm == 80.0 and e.surface_solid_dn_support == "exclude"
    assert serve_safe.ifg_cl1_record_meta(a)["surface_solid_shadow_mm"] == 80.0
    for absent in ("--surface-hull-fill", "--surface-contact-allowance-mm", "--palm-shell", "--palm-shell-margin"):
        assert absent not in shlex.split(line)
