"""T78b — split-fragment re-attach (`esdf.surface_reattach_hull_mm`) and the hull fill just before the grasp
(`esdf.surface_hull_fill`), both default off. Off must be the T78 (A1–A5) call for call.

| # | property |
|---|---|
| 1 | config defaults off · round trip · refusals; CLI flags → esdf keys / record meta only when given; start-up refusals |
| 2 | off: `split_region` without / with `reattach_hull=None` is the same; the pipeline never calls the T78b steps, no new record key, the builder gets the same kwargs as an explicit-off config; the tier never calls `_hull_fill_exclusion` without a `HullFill` |
| 3 | `convex_hull` · `hull_offset_distance` (inside ≤ 0, offset polytope) · degenerate sets |
| 4 | `reattach_components`: a fragment of the target's surface just outside its hull is re-attached, a neighbour beside it is not; record per component (attached, hull distance) |
| 5 | pipeline: re-attach runs inside A1; the accepted region's hull and the dropped components' samples are kept per episode |
| 6 | hull-fill window: outside → evidence unchanged (`outside_window`); latched / grasp active / palm within 80 mm → `HullFill` on the evidence; A4 hold keeps the last accepted hull; reset clears it |
| 7 | `hull_fill_split` (pure): voxel centres inside hull + pad, support guard, voxels holding dropped samples protected |
| 8 | tier (fake cuRobo, torch): the hull fill clears the target's seeds the oracle split kept (stale "other" samples), keeps protected voxels, the table and the guard; counts and trace |
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import HullFill, OracleVisibleEvidence, hull_fill_split
from AG3S.ag3s.runtime import attention_surface_post as P

T78_ON = {"surface_split": True, "surface_target_check": "overlap", "surface_keep_last": True,
          "surface_grasp_hold": True, "surface_no_ball": True}
T78B_ON = {"surface_reattach_hull_mm": 10.0, "surface_hull_fill": True}


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


def _split_ready(ag3s):
    """The T76 server's split values (12 mm, partition) and a known max opening (the toy robot has no fingers)."""
    import dataclasses

    from AG3S.ag3s.stages.admissibility import MaxOpening

    c = dataclasses.replace(ag3s.config.clustering, attention_split_radius=0.012, attention_split_partition=True,
                            attention_split_ratio_source="raw")
    ag3s.config = dataclasses.replace(ag3s.config, clustering=c)
    ag3s._max_opening = MaxOpening(0.0715, "test")
    return ag3s


# ============================================================================ 1 config · CLI


def test_config_defaults_off_round_trip_and_refusals():
    e = AG3SConfig().esdf
    assert (e.surface_reattach_hull_mm, e.surface_hull_fill, e.surface_hull_fill_palm_mm) == (None, False, 80.0)
    cfg = AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", **T78_ON, **T78B_ON}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    assert cfg.esdf.surface_reattach_hull_mm == 10.0 and cfg.esdf.surface_hull_fill
    with pytest.raises(AG3SConfigError, match="attention_surface"):
        AG3SConfig.from_dict({"esdf": {"surface_hull_fill": True}})
    with pytest.raises(AG3SConfigError, match="surface_split"):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_reattach_hull_mm": 10.0}})
    for bad in ({}, {"surface_keep_last": True}, {"surface_keep_last": True, "surface_split": True}):
        with pytest.raises(AG3SConfigError, match="surface_no_ball"):
            AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_hull_fill": True, **bad}})
    for v in (-1.0, float("nan"), True, "10"):
        with pytest.raises(AG3SConfigError, match="surface_reattach_hull_mm"):
            AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", **T78_ON,
                                           "surface_reattach_hull_mm": v}})
    with pytest.raises(AG3SConfigError, match="surface_hull_fill "):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", **T78_ON, "surface_hull_fill": 1}})


def test_cli_flags_map_to_esdf_keys_and_meta_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    base = A._parse(*A.SURF, *A.ALL)
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert not {"surface_reattach_hull_mm", "surface_hull_fill"} & (set(sec0) | set(meta0))
    on = A._parse(*A.SURF, *A.ALL, "--surface-reattach-hull", "10", "--surface-hull-fill")
    sec = target_exclusion_section(on)
    assert {k: v for k, v in sec.items() if k not in sec0} == T78B_ON
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == T78B_ON
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert cfg.surface_reattach_hull_mm == 10.0 and cfg.surface_hull_fill


def test_cli_refusals(capsys):
    A = _t78a()
    args = A._reject(*A.SURF, *A.ALL, "--surface-reattach-hull", "10", "--surface-hull-fill")
    assert args.surface_hull_fill and args.surface_reattach_hull == 10.0
    for extra, msg in ((("--surface-reattach-hull", "10"), "attention-surface"),
                       (("--surface-hull-fill",), "attention-surface")):
        with pytest.raises(SystemExit):
            A._reject(*extra)
        assert msg in capsys.readouterr().err
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, "--surface-reattach-hull", "10")
    assert "--surface-split" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, *A.ALL, "--surface-reattach-hull", "-1")
    assert "--surface-reattach-hull" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, "--surface-split", "--surface-keep-last", "--surface-hull-fill")
    assert "--surface-no-ball" in capsys.readouterr().err


# ============================================================================ 2 off = T78


def test_split_region_without_reattach_is_the_t78_call():
    A = _t78a()
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    obs, _, _ = A._rig(fw, gap=0.001)
    views = A._views(obs)
    from AG3S.ag3s.runtime.attention_surface import select_target_surface

    res = select_target_surface(views)
    reg = P.region_from_codes(views, res.codes, voxel=0.005)
    kw = dict(max_opening=0.0715, radius=0.012, min_points=20, min_ratio=2.0, partition=True)
    k0, r0 = P.split_region(reg, **kw)
    k1, r1 = P.split_region(reg, **kw, reattach_hull=None)
    np.testing.assert_array_equal(k0, k1)
    assert r0 == r1 and "reattach" not in r0 and r0["applied"]


def test_off_never_calls_the_t78b_steps_and_builds_the_same_call(monkeypatch):
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder
    from AG3S.ag3s.runtime.pipeline import AG3S

    A = _t78a()

    def boom(*a, **k):
        raise AssertionError("T78b step called with its flags off")

    for target, name in ((AG3S, "_surface_hull_fill"), (AG3S, "_surface_hull_store"), (AG3S, "_grasp_palm_points"),
                         (P, "reattach_components"), (P, "pixel_points"),
                         (CuroboFieldBuilder, "_hull_fill_exclusion")):
        monkeypatch.setattr(target, name, boom)
    out = []
    for extra in ({}, {"surface_reattach_hull_mm": None, "surface_hull_fill": False}):
        fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                            **T78_ON, **extra)
        _split_ready(ag3s)
        obs, _, _ = A._rig(fw, gap=0.001)
        rec = A._update(ag3s, obs, ats_points=A._apple_points(fw))
        ag3s.set_grasp_latched(True)                  # inside the window: still nothing with the flag off
        field, _ = fw._build(ag3s)
        kw = ag3s._esdf_builder.calls[-1]
        t78 = ag3s._surface_frame["t78"]
        assert "hull" not in t78 and "reattach" not in t78["split"]
        assert kw["oracle_evidence"].hull_fill is None and "hull_fill" not in field.stats["target_free"]["oracle"]
        assert getattr(ag3s, "_surface_hull", None) is None
        assert rec["segmentation"] and t78["split"]["applied"], "the off run must exercise an accepted split"
        out.append((rec, kw, t78))
    (r0, k0, t0), (r1, k1, t1) = out
    assert k0.keys() == k1.keys()
    np.testing.assert_array_equal(k0["oracle_evidence"].target_points, k1["oracle_evidence"].target_points)
    assert k0["oracle_evidence"].z_floor == k1["oracle_evidence"].z_floor
    strip = lambda d: {k: v for k, v in d.items() if k not in ("memory", "ms")}   # noqa: E731 — timings differ
    assert strip(r0) == strip(r1) and strip(t0) == strip(t1)


# ============================================================================ 3 hull


def _cap(n=500, r=0.035, c=(0.5, 0.3, 0.9), zmin=-0.4, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(8 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] >= zmin][:n]
    return np.asarray(c) + r * v


def test_convex_hull_and_offset_distance():
    cube = np.array([[x, y, z] for x in (0, 0.1) for y in (0, 0.1) for z in (0, 0.1)], float)
    h, why = P.convex_hull(cube)
    assert why == "ok" and h["volume"] == pytest.approx(1e-3) and h["n_vertices"] == 8
    np.testing.assert_allclose(np.linalg.norm(h["equations"][:, :3], axis=1), 1.0)
    q = np.array([[0.05, 0.05, 0.05], [0.1, 0.05, 0.05], [0.12, 0.05, 0.05], [0.13, 0.13, 0.05]])
    d = P.hull_offset_distance(h["equations"], q)
    np.testing.assert_allclose(d, [-0.05, 0.0, 0.02, 0.03], atol=1e-12)   # corner: max facet offset (offset polytope)
    assert P.convex_hull(cube[:3]) == (None, "too_few_points")
    flat = np.c_[np.random.default_rng(0).random((50, 2)), np.zeros(50)]
    assert P.convex_hull(flat)[1] == "hull_degenerate"


# ============================================================================ 4 re-attach


def test_reattach_takes_back_a_fragment_of_the_target_and_not_the_neighbour():
    c = np.array([0.5, 0.3, 0.9])
    target = _cap(400, c=c, zmin=-0.2)                                   # the apple's upper surface (A1 target)
    frag = _cap(600, c=c, zmin=-0.8, seed=1)
    dz = frag[:, 2] - c[2]
    frag = frag[(dz < -0.009) & (dz > -0.014) & (frag[:, 0] < c[0])][:40]  # a low-attention strip of the same apple
    nb = c + np.array([0.0, 0.075, -0.01]) + np.random.default_rng(2).normal(scale=[0.012, 0.03, 0.01], size=(200, 3))
    pos = np.vstack([target, frag, nb])
    reg = P.Region(pos, np.ones(len(pos)), np.ones(len(pos), np.int64), [])
    comps = [types.SimpleNamespace(indices=np.arange(len(target))),
             types.SimpleNamespace(indices=len(target) + len(frag) + np.arange(len(nb))),
             types.SimpleNamespace(indices=len(target) + np.arange(len(frag)))]
    keep, rec = P.reattach_components(reg, comps, 0.010)
    a, f, n = (np.zeros(len(pos), bool) for _ in range(3))
    a[:len(target)] = True
    f[len(target):len(target) + len(frag)] = True
    n[len(target) + len(frag):] = True
    assert keep[a].all() and keep[f].all() and not keep[n].any()
    nb_rec, fr_rec = rec["components"]
    assert fr_rec["attached"] and 0.0 < fr_rec["hull_dist_mm"] <= 10.0, fr_rec   # just outside the hull
    assert not nb_rec["attached"] and nb_rec["hull_dist_mm"] > 20.0, nb_rec
    assert rec["n_attached"] == 1 and rec["n_dropped"] == 1 and rec["n_nodes_attached"] == len(frag)
    assert rec["applied"] and rec["target_hull"]["volume_cm3"] > 0
    keep0, rec0 = P.reattach_components(reg, comps, 0.0)                 # MM = 0: only centroids inside the hull
    assert not keep0[f].any() and rec0["n_attached"] == 0
    flat = P.Region(np.c_[pos[:, :2], np.zeros(len(pos))], np.ones(len(pos)), np.ones(len(pos), np.int64), [])
    keep2, rec2 = P.reattach_components(flat, comps, 0.010)
    assert keep2.sum() == len(target) and rec2["why"] == "hull_degenerate" and rec2["n_attached"] == 0


# ============================================================================ 5 · 6 pipeline


def _wired_on(monkeypatch, **extra):
    A = _t78a()
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **T78_ON,
                        **T78B_ON, **extra)
    return fw, _split_ready(ag3s)


def test_reattach_runs_in_a1_and_the_hull_is_kept_with_the_dropped_samples(monkeypatch):
    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch)
    obs, _, nb = A._rig(fw, gap=0.001)
    rec = A._update(ag3s, obs, ats_points=A._apple_points(fw))
    t78 = ag3s._surface_frame["t78"]
    assert rec["segmentation"] and t78["split"]["applied"]
    ra = t78["split"]["reattach"]
    assert ra["margin_mm"] == 10.0 and all(not c["attached"] for c in ra["components"]), ra   # the neighbour stays out
    assert t78["hull"]["stored"] and t78["hull"]["n_dropped_points"] > 100
    hull = ag3s._surface_hull
    # the dropped samples lie on the neighbour, the hull on the apple
    assert np.linalg.norm(hull["dropped_points"].mean(0) - nb) < np.linalg.norm(hull["dropped_points"].mean(0) - fw.APPLE)
    assert np.linalg.norm(hull["centroid"][:2] - fw.APPLE[:2]) < 0.02
    ag3s.reset()
    assert ag3s._surface_hull is None


def test_window_outside_keeps_the_evidence_inside_attaches_the_hull(monkeypatch):
    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch)
    obs, _, _ = A._rig(fw, gap=0.001)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    hf = field.stats["target_free"]["oracle"]["hull_fill"]
    assert ev.hull_fill is None and not hf["applied"] and hf["why"] == "outside_window", hf
    assert hf["palm_source"] == "no_authorized_gripper" and hf["palm_mm"] is None   # the toy robot has no gripper pairs
    for setter in ("set_grasp_latched", "set_grasp_active"):
        getattr(ag3s, setter)(True)
        field, _ = fw._build(ag3s)
        ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
        hf = field.stats["target_free"]["oracle"]["hull_fill"]
        assert hf["applied"] and hf["window"] == ("latched" if setter == "set_grasp_latched" else "grasp_active")
        h = ev.hull_fill
        assert isinstance(h, HullFill) and h.pad_m == pytest.approx(fw.FINE)
        assert h.z_floor == pytest.approx(ev.z_floor) == pytest.approx(fw.TABLE_Z + fw.FINE)   # the A5 guard
        np.testing.assert_array_equal(h.protect_points, ag3s._surface_hull["dropped_points"])
        np.testing.assert_array_equal(h.equations, ag3s._surface_hull["equations"])
        assert hf["hull_source"] == "this_request" and hf["volume_cm3"] > 0
        getattr(ag3s, setter)(False)


def test_palm_near_opens_the_window(monkeypatch):
    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch)

    class _Gripper(type(ag3s.robot_model)):
        def gripper_joint_pairs(self):
            return [{"parent_link": "ee_left", "children": ("ee_finger_l1", "ee_finger_l2")},
                    {"parent_link": "ee_right", "children": ("ee_finger_r1", "ee_finger_r2")}]

    ag3s.constraint_robot_model = _Gripper()
    obs, _, _ = A._rig(fw, gap=0.001)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    cen = ag3s._surface_hull["centroid"]
    for dx in (-0.10, 0.06):
        q = np.array([dx, 0.0])
        field, _ = fw._build(ag3s, robot_state=q)
        hf = field.stats["target_free"]["oracle"]["hull_fill"]
        palm = _Gripper().sphere_centers_numeric(q)[0][2:4].mean(0)        # the authorized (left) fingers
        d = float(np.linalg.norm(palm - cen)) * 1e3
        assert hf["palm_source"] == "finger_spheres" and hf["palm_mm"] == pytest.approx(d, abs=0.06)
        assert hf["applied"] == (d <= 80.0) and hf["window"] == ("palm_near" if d <= 80.0 else None), (d, hf)
    assert not fw._build(ag3s, robot_state=np.array([-0.10, 0.0]))[0].stats["target_free"]["oracle"]["hull_fill"]["applied"]
    assert fw._build(ag3s, robot_state=np.array([0.06, 0.0]))[0].stats["target_free"]["oracle"]["hull_fill"]["applied"]


def test_grasp_hold_and_rejected_requests_use_the_last_accepted_hull(monkeypatch):
    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch)
    obs, _, _ = A._rig(fw, gap=0.001)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    eq0 = ag3s._surface_hull["equations"].copy()
    ag3s.frame_index += 1
    ag3s.set_grasp_active(True)                                          # A4 hold: no selection, the hull stands
    obs2, _, _ = A._rig(fw, gap=0.001)
    A._update(ag3s, obs2, ats_points=A._apple_points(fw))
    assert ag3s._surface_frame["why"] == "grasp_hold" and "hull" not in ag3s._surface_frame["t78"]
    field, _ = fw._build(ag3s)
    hf = field.stats["target_free"]["oracle"]["hull_fill"]
    assert hf["applied"] and hf["hull_source"] == "last_accepted" and hf["hull_age_frames"] == 1
    np.testing.assert_array_equal(ag3s._esdf_builder.calls[-1]["oracle_evidence"].hull_fill.equations, eq0)
    ag3s.set_grasp_active(False)
    ag3s.set_grasp_latched(True)
    A._update(ag3s, obs2, ats_points=A._apple_points(fw) + [0.0, 0.3, 0.0])   # A2 rejects: the hull stands
    assert ag3s._surface_frame["why"] == "not_ats_target"
    field, _ = fw._build(ag3s)
    np.testing.assert_array_equal(ag3s._esdf_builder.calls[-1]["oracle_evidence"].hull_fill.equations, eq0)


def test_no_accepted_region_no_hull(monkeypatch):
    A = _t78a()
    fw, ag3s = _wired_on(monkeypatch)
    obs, _, _ = A._rig(fw)
    for o in obs:
        o.attention_map = None
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    ag3s.set_grasp_latched(True)
    field, _ = fw._build(ag3s)
    hf = field.stats["target_free"]["oracle"]["hull_fill"]
    assert not hf["applied"] and hf["why"] == "no_accepted_region"
    assert ag3s._esdf_builder.calls[-1]["oracle_evidence"].hull_fill is None


# ============================================================================ 7 hull_fill_split


def _cube_fill(**kw):
    cube = np.array([[x, y, z] for x in (0.02, 0.06) for y in (0.02, 0.06) for z in (0.02, 0.06)], float)
    h, _ = P.convex_hull(cube)
    return HullFill(equations=h["equations"], lo=h["lo"], hi=h["hi"], pad_m=kw.pop("pad", 0.0), **kw)


def test_hull_fill_split_inside_guard_and_protection():
    vs, shape, org = 0.01, (10, 10, 10), np.zeros(3)
    s = hull_fill_split(_cube_fill(), org, vs, shape)
    assert len(s["cand"]) == 5 ** 3 and s["fill"].all() and not s["protected"].any()   # centres 0.02 … 0.06
    s = hull_fill_split(_cube_fill(pad=0.0105), org, vs, shape)
    assert len(s["cand"]) == 7 ** 3                                     # + one voxel each side (facet offset)
    s = hull_fill_split(_cube_fill(z_floor=0.035), org, vs, shape)
    assert int(s["fill"].sum()) == 5 * 5 * 3 and int((~s["above"]).sum()) == 5 * 5 * 2
    s2 = hull_fill_split(_cube_fill(), org, vs, shape, z_floor=0.035)      # the oracle's guard when the hull has none
    np.testing.assert_array_equal(s2["fill"], s["fill"])
    prot = np.array([[0.041, 0.039, 0.044], [0.5, 0.5, 0.5]])           # one sample in voxel (4, 4, 4), one off-grid
    s = hull_fill_split(_cube_fill(protect_points=prot), org, vs, shape)
    assert int(s["protected"].sum()) == 1 and tuple(s["cand"][s["protected"]][0]) == (4, 4, 4)
    assert int(s["fill"].sum()) == 5 ** 3 - 1
    assert len(hull_fill_split(_cube_fill(), np.full(3, 1.0), vs, shape)["cand"]) == 0   # hull off the grid
    with pytest.raises(ValueError):
        _cube_fill(pad=-0.001)
    ev = OracleVisibleEvidence(target_points=np.zeros((1, 3)))
    assert ev.hull_fill is None


# ============================================================================ 8 tier (fake cuRobo)


def test_tier_hull_fill_clears_the_kept_target_seeds_and_keeps_protected_voxels(monkeypatch):
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
    # the oracle memory: the target's upper half as target, its lower half remembered as "other" (cause 2), B, table
    ev = OracleVisibleEvidence(target_points=a_pts[upper], other_points=np.vstack([a_pts[~upper], pts[own != 0]]),
                               max_dist_voxels=6.0, z_floor=T44.TABLE_Z + VS)
    h, _ = P.convex_hull(a_pts)                                            # the accepted region = the whole target
    b_pts = pts[own == 1]

    def tier(evidence):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=evidence)
        return t

    base = tier(ev)
    above = g[..., 2] >= T44.TABLE_Z + VS
    a_low = (owner == 0) & (np.abs(sdf) < 0.75 * VS) & (g[..., 2] < T44.A[2] - VS) & above
    assert np.min(np.abs(base.values.numpy()[a_low])) == 0.0, "the oracle split keeps the stale lower half"
    full = tier(dataclasses_replace(ev, HullFill(equations=h["equations"], lo=h["lo"], hi=h["hi"], pad_m=VS,
                                                 protect_points=b_pts, z_floor=T44.TABLE_Z + VS)))
    v = full.values.numpy()
    assert np.min(np.abs(v[a_low])) > 0.0, "every target seed above the guard is gone"
    b_surf = (owner == 1) & (np.abs(sdf) < 0.4 * VS)
    assert np.abs(v[b_surf]).max() < 1e-6, "voxels holding the dropped component's samples stay"
    table = (owner == 2) & (np.abs(sdf) < 0.5 * VS) & (np.linalg.norm(g[..., :2] - T44.A[:2], axis=-1) < 0.03)
    assert np.abs(v[table]).max() < 1e-6, "the table under the target stays"
    ex, ex0 = full.extra["exclusion"], base.extra["exclusion"]
    hf = ex["hull_fill"]
    assert hf["n_extra_removed"] > 0 and ex["n_removed"] == ex0["n_removed"] + hf["n_extra_removed"]
    assert ex["n_target_owned"] == ex0["n_target_owned"]
    assert len(full.extra["exclusion_trace"]["removed_ijk"]) == ex["n_removed"]
    assert hf["n_hull_voxels"] > 0 and hf["n_protect_points"] == len(b_pts) and hf["pad_mm"] == 5.0
    assert CuroboFieldBuilder._exclusion_counts([full])["hull_fill"] == hf
    assert "hull_fill" not in CuroboFieldBuilder._exclusion_counts([base])
    # without the protection, the neighbour's surface inside the hull pad is cleared too (the protection matters)
    bare = tier(dataclasses_replace(ev, HullFill(equations=h["equations"], lo=h["lo"], hi=h["hi"], pad_m=VS,
                                                 z_floor=T44.TABLE_Z + VS)))
    assert bare.extra["exclusion"]["hull_fill"]["n_extra_removed"] > hf["n_extra_removed"]
    # no hull → `_hull_fill_exclusion` is not called and the tier is the oracle one
    monkeypatch.setattr(CuroboFieldBuilder, "_hull_fill_exclusion",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called without a hull")))
    again = tier(ev)
    assert torch.equal(again.values, base.values)


def dataclasses_replace(ev, hull):
    import dataclasses

    return dataclasses.replace(ev, hull_fill=hull)
