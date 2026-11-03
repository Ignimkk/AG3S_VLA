"""T78 갈래 A — attention-surface region post-processing (`esdf.surface_*`, `runtime.attention_surface_post`), all default
off. Off must be IFG-CL1 call for call.

| # | property |
|---|---|
| 1 | config defaults off · round trip · refusals (flag without `attention_surface`, A4 / A5 without A3, A5 with the coarse tier, bad A2 value); CLI flags → esdf keys / record meta only when given; start-up refusals |
| 2 | off: `_attention_surface_update_t78` is never called, the `t78` record key never appears, the builder gets the same kwargs as an explicit-off config (attention_surface rig) |
| 3 | `region_from_codes` = the graph's selected nodes (same voxels, same centroids, node attention = `node_patch_mean` of the seed map) |
| 4 | A1 `split_region`: touching apple + neighbour region with attention on the apple → the target component is the apple; flat attention / not triggered → whole; `restrict_codes` turns the dropped nodes' pixels into code 2 |
| 5 | A2 `target_check` (rule and its edge cases) |
| 6 | A3: rejected / abstained request → the memory is carved, not folded, and the hand layer removes the last accepted samples; none yet → an empty evidence (nothing removed, no ball / point fallback) |
| 7 | A4: grasp active / attached → no selection, no fold (carve only) |
| 8 | A5: no ball reaches the builder; the support guard = support plane under the region + pad |
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.runtime import attention_surface_post as P
from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER, SurfaceView, select_target_surface
from AG3S.ag3s.runtime.oracle_segmentation import CODE_STATIC, CODE_TARGET, OracleMemory, OracleView

T78_ON = {"surface_split": True, "surface_target_check": "overlap", "surface_keep_last": True,
          "surface_grasp_hold": True, "surface_no_ball": True}


# ============================================================================ 1 config · CLI


def test_config_defaults_off_round_trip_and_refusals():
    e = AG3SConfig().esdf
    assert (e.surface_split, e.surface_target_check, e.surface_keep_last, e.surface_grasp_hold,
            e.surface_no_ball) == (False, "off", False, False, False)
    cfg = AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", **T78_ON}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in ({"surface_split": True}, {"surface_target_check": "overlap"}, {"surface_keep_last": True}):
        with pytest.raises(AG3SConfigError, match="attention_surface"):
            AG3SConfig.from_dict({"esdf": {"target_extent": "oracle_visible", **bad}})
        with pytest.raises(AG3SConfigError, match="attention_surface"):
            AG3SConfig.from_dict({"esdf": bad})
    for bad in ({"surface_grasp_hold": True}, {"surface_no_ball": True}):
        with pytest.raises(AG3SConfigError, match="surface_keep_last"):
            AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", **bad}})
    with pytest.raises(AG3SConfigError, match="target_free_coarse"):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_keep_last": True,
                                       "surface_no_ball": True, "target_free_coarse": True}})
    with pytest.raises(AG3SConfigError, match="surface_target_check"):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_target_check": "iou"}})
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"target_extent": "attention_surface", "surface_split": 1}})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def _reject(*argv):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])
    reject_bad_flag_combinations(ap, args)
    return args


SURF = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface",
        "--target-field-policy", "exclude-authorized", "--attention-source", "noun_target:L10H7",
        "--attention-split-radius-mm", "12")
ALL = ("--surface-split", "--surface-target-check", "overlap", "--surface-keep-last", "--surface-grasp-hold",
       "--surface-no-ball")


def test_cli_flags_map_to_esdf_keys_and_meta_only_when_given():
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    base = _parse(*SURF)
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert not any(k.startswith("surface_") for k in sec0) and not any(k.startswith("surface_") for k in meta0)
    off = _parse(*SURF, "--surface-target-check", "off")
    assert target_exclusion_section(off) == sec0 and ifg_cl1_record_meta(off) == meta0
    on = _parse(*SURF, *ALL)
    sec = target_exclusion_section(on)
    assert {k: v for k, v in sec.items() if k not in sec0} == T78_ON
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert {k: getattr(cfg, k) for k in T78_ON} == T78_ON
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == T78_ON


def test_cli_refusals(capsys):
    args = _reject(*SURF, *ALL)
    assert args.surface_no_ball
    for extra, msg in ((("--surface-split",), "attention-surface"),
                       (("--surface-keep-last",), "attention-surface")):
        with pytest.raises(SystemExit):
            _reject(*extra)
        assert msg in capsys.readouterr().err
    for extra in (("--surface-grasp-hold",), ("--surface-no-ball",)):
        with pytest.raises(SystemExit):
            _reject(*SURF, *extra)
        assert "--surface-keep-last" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _reject(*SURF, "--surface-keep-last", "--surface-no-ball", "--target-free-coarse")
    assert "--target-free-coarse" in capsys.readouterr().err


# ============================================================================ rigs


def _look_at(eye, target):
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _rig(fw, gap=0.083, att_on="apple", hw=(240, 320), f=260.0, apple=None):
    """Three cameras over an apple and a neighbour sphere `gap` (centre offset in y beyond 2R) away. Attention = the
    fraction of each patch covered by the apple (`att_on="apple"`), by both (`"both"`) or by the neighbour."""
    from AG3S.ag3s.types import CameraID

    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    A = fw.APPLE if apple is None else np.asarray(apple, float)
    nb = A + np.array([0.0, 2 * fw.APPLE_R + gap, 0.0])
    sph = ((A, fw.APPLE_R, CODE_TARGET), (nb, fw.APPLE_R, 3))
    eyes = (A + [0.0, -0.05, 0.50], A + [-0.40, -0.15, 0.30], A + [-0.35, 0.20, 0.35])
    ids = (CameraID.HEAD, CameraID.LEFT_WRIST, CameraID.RIGHT_WRIST)
    obs, gt = [], {}
    for cid, eye in zip(ids, eyes):
        v = t46e._raycast(_look_at(eye, A), hw=hw, f=f, spheres=sph, table_z=fw.TABLE_Z)
        h, w = hw
        lab = v.labels
        on = {"apple": lab == CODE_TARGET, "both": (lab == CODE_TARGET) | (lab == 3), "neighbour": lab == 3}[att_on]
        cov = on.reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))
        obs.append(types.SimpleNamespace(camera_id=cid, depth=v.depth_m, camera_intrinsics=v.K,
                                         T_base_cam=v.T_base_cam, attention_map=cov.astype(np.float32),
                                         robot_state=np.zeros(20)))
        gt[cid.value] = lab
    return obs, gt, nb


def _cameras(obs):
    from AG3S.ag3s.fields.esdf import CameraDepth

    return [CameraDepth(name=o.camera_id.value, depth=o.depth, camera_intrinsics=o.camera_intrinsics,
                        T_base_cam=o.T_base_cam, robot_mask=np.zeros(o.depth.shape, bool)) for o in obs]


def _views(obs):
    return [SurfaceView(name=o.camera_id.value, depth_m=np.asarray(o.depth, np.float64), K=o.camera_intrinsics,
                        T_base_cam=o.T_base_cam, robot_mask=np.zeros(o.depth.shape, bool),
                        attention=o.attention_map) for o in obs]


def _wired(monkeypatch, **esdf):
    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    return t46e._wired(monkeypatch, **esdf)


def _update(ag3s, obs, **kw):
    cams = _cameras(obs)
    ag3s._oracle_frame = ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {}, **kw)
    return ag3s._oracle_frame


# ============================================================================ 2 off = IFG-CL1


def test_off_never_calls_the_t78_path_and_builds_the_same_call(monkeypatch):
    from AG3S.ag3s.runtime.pipeline import AG3S

    def boom(*a, **k):
        raise AssertionError("T78 path called with the flags off")

    monkeypatch.setattr(AG3S, "_attention_surface_update_t78", boom)
    monkeypatch.setattr(AG3S, "_surface_memory_evidence", boom)
    out = []
    for extra in ({}, {"surface_split": False, "surface_target_check": "off", "surface_keep_last": False,
                       "surface_grasp_hold": False, "surface_no_ball": False}):
        fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **extra)
        assert not ag3s._t78_on()
        obs, _, _ = _rig(fw)
        rec = _update(ag3s, obs)
        field, _ = fw._build(ag3s)
        kw = ag3s._esdf_builder.calls[-1]
        assert "t78" not in ag3s._surface_frame and "t78" not in field.stats["target_free"]["attention_surface"]
        out.append((rec, kw, field.stats["target_free"]))
    (r0, k0, s0), (r1, k1, s1) = out
    assert k0.keys() == k1.keys()
    np.testing.assert_array_equal(k0["oracle_evidence"].target_points, k1["oracle_evidence"].target_points)
    assert k0["target_free_ball"].summary() == k1["target_free_ball"].summary()
    strip = lambda r: {k: v for k, v in r.items() if k != "memory"}   # noqa: E731 — update_ms differs
    assert strip(r0) == strip(r1)


# ============================================================================ 3 region nodes = graph nodes


def test_region_from_codes_reproduces_the_selected_graph_nodes(monkeypatch):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    obs, _, _ = _rig(fw)
    views = _views(obs)
    res = select_target_surface(views, keep_debug=True)
    assert res.codes is not None
    reg = P.region_from_codes(views, res.codes, voxel=0.005)
    geo = res.geometry
    sel = res.selected_nodes
    assert len(reg) == len(sel)
    ref = geo.pos[sel]
    d = np.linalg.norm(reg.pos[:, None, :] - ref[None, :, :], axis=2).min(axis=1)
    assert d.max() < 1e-9, d.max()
    # node attention = the graph's pixel-weighted patch mean of the seed map
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.runtime.attention_surface import attention_vector

    smap, _ = attention_vector(views)
    a_ref = G.node_patch_mean(geo, smap)[sel]
    j = np.linalg.norm(reg.pos[:, None, :] - ref[None, :, :], axis=2).argmin(axis=1)
    np.testing.assert_allclose(reg.att, a_ref[j], rtol=1e-9, atol=1e-12)
    assert int(reg.n_px.sum()) == sum(int((c == CODE_TARGET).sum()) for c in res.codes.values())


# ============================================================================ 4 A1 split


def test_split_cuts_a_touching_neighbour_by_attention(monkeypatch):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    obs, gt, _ = _rig(fw, gap=0.001)
    views = _views(obs)
    res = select_target_surface(views)
    assert res.codes is not None
    lab_t = {k: (c == CODE_TARGET) for k, c in res.codes.items()}
    merged = sum(int((gt[k][m] == 3).sum()) for k, m in lab_t.items())
    assert merged > 200, "the rig must merge the touching neighbour into the region"
    reg = P.region_from_codes(views, res.codes, voxel=0.005)
    keep, rec = P.split_region(reg, max_opening=0.0715, radius=0.012, min_points=20, min_ratio=2.0, partition=True)
    assert rec["applied"] and rec["reason"] == "split", rec
    codes = P.restrict_codes(res.codes, reg, keep)
    for k, c in codes.items():
        # dropped pixels became static, nothing else changed
        changed = c != res.codes[k]
        assert np.all(res.codes[k][changed] == CODE_TARGET) and np.all(c[changed] == CODE_STATIC)
    n = lambda cs, g: sum(int((gt[k][cs[k] == CODE_TARGET] == g).sum()) for k in cs)   # noqa: E731
    # the apple is kept whole; the neighbour loses most of its pixels (attention is per 16x16 patch, so the patches
    # straddling the contact keep a rim of the neighbour — measured 1 mm gap: 3,824 → 620 px, apple 3,799 → 3,799)
    assert n(codes, CODE_TARGET) == n(res.codes, CODE_TARGET)
    assert n(codes, 3) <= 0.25 * n(res.codes, 3), (n(codes, 3), n(res.codes, 3))
    sub = P.subset(reg, keep)
    assert len(sub) == int(keep.sum()) and sum(len(p[1]) for p in sub.pixels) == int(sub.n_px.sum())


def test_split_leaves_the_region_whole_when_it_does_not_hold():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(600, 3))
    pts = 0.035 * v / np.linalg.norm(v, axis=1, keepdims=True)
    one = P.Region(pts, np.ones(len(pts)), np.ones(len(pts), np.int64), [])
    keep, rec = P.split_region(one, max_opening=0.0715, radius=0.012, min_points=20, min_ratio=2.0)
    assert keep.all() and not rec["applied"] and rec["reason"] == "not_triggered"   # 70 mm ball: largest fits
    two = P.Region(np.vstack([pts, pts + [0.0, 0.071, 0.0]]), np.ones(2 * len(pts)), np.ones(2 * len(pts), np.int64),
                   [])
    two.pos[:, 2] *= 0.3   # a flat pair: narrowest fits, largest does not → triggered
    keep, rec = P.split_region(two, max_opening=0.0715, radius=0.012, min_points=20, min_ratio=2.0, partition=True)
    assert keep.all() and not rec["applied"] and rec["reason"] in ("attention_not_discriminative", "attention_flat"), rec
    keep, rec = P.split_region(two, max_opening=None, radius=0.012, min_points=20, min_ratio=2.0)
    assert keep.all() and rec["reason"] == "max_opening_unknown"
    keep, rec = P.split_region(P.Region(np.zeros((1, 3)), np.ones(1), np.ones(1, np.int64), []), max_opening=0.07,
                               radius=0.012, min_points=20, min_ratio=2.0)
    assert keep.all() and rec["reason"] == "too_few_nodes"


# ============================================================================ 5 A2 target check


def _region(pts):
    pts = np.asarray(pts, float)
    return P.Region(pts, np.ones(len(pts)), np.ones(len(pts), np.int64), [])


def test_target_check_rule():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(400, 3))
    apple = 0.035 * v / np.linalg.norm(v, axis=1, keepdims=True) + [0.5, 0.3, 0.9]
    ats = apple + rng.normal(scale=0.002, size=apple.shape)
    ok, rec = P.target_check(_region(apple), ats)
    assert ok and rec["passed"], rec
    ok, rec = P.target_check(_region(apple + [0.0, 0.25, 0.0]), ats)   # another object 25 cm away
    assert not ok and rec["why"] == "no_overlap", rec
    # the region side is measured: a partial region on the ATS object passes (few ATS points in it), a region
    # half on the object and mostly on a neighbour does not (pixel-weighted)
    side = apple[apple[:, 0] > 0.52]
    ok, rec = P.target_check(_region(side), apple + rng.normal(scale=0.002, size=apple.shape))
    assert ok and rec["fraction"] == 1.0
    nb = apple + [0.0, 0.071, 0.0]
    reg = P.Region(np.vstack([apple[:100], nb]), np.ones(100 + len(nb)), np.ones(100 + len(nb), np.int64), [])
    ok, rec = P.target_check(reg, ats)
    assert not ok and rec["fraction"] < 0.5
    reg.n_px[:100] = 50                                                  # the same nodes, the object's pixels dominate
    ok, rec = P.target_check(reg, ats)
    assert ok, rec
    ok, rec = P.target_check(_region(apple), None)
    assert not ok and rec["why"] == "no_ats_target"
    ok, rec = P.target_check(_region(np.zeros((0, 3))), ats)
    assert not ok and rec["why"] == "empty_region"


# ============================================================================ 6 A2 · A3 in the pipeline


def _apple_points(fw):
    return fw._apple_geometry().points


def test_keep_last_uses_the_last_accepted_region_and_never_the_ball(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                      surface_keep_last=True)
    assert ag3s._t78_on()
    # no accepted region yet (attention missing) → an empty evidence: nothing removed, no ball, no points
    obs, _, _ = _rig(fw)
    for o in obs:
        o.attention_map = None
    rec = _update(ag3s, obs)
    assert rec["segmentation"] is False and rec["memory"]["carved"] and rec["memory"]["updated"] is False
    field, _ = fw._build(ag3s)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    assert len(ev.target_points) == 0 and ev.info.get("empty")
    tf = field.stats["target_free"]
    assert tf["oracle"]["removes_nothing"] and tf["oracle"]["region_source"] == "none_yet"
    assert tf["attention_surface"]["why"] == "no_attention:head" and "t78" in tf["attention_surface"]
    # an accepted region → folded and used
    obs, _, _ = _rig(fw)
    rec = _update(ag3s, obs)
    assert rec["segmentation"] and rec["memory"]["updated"]
    n_mem = len(ag3s._oracle_memory)
    fw._build(ag3s)
    ev1 = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    assert len(ev1.target_points) > 100
    assert ag3s._esdf_builder.calls[-1]["target_free_ball"] is not None    # A5 off: the ball is still made (record)
    # the next request abstains → carved (nothing seen through: same scene), not folded; the last region is removed
    obs2, _, _ = _rig(fw)
    for o in obs2:
        o.attention_map = None
    n_up = ag3s._oracle_memory.n_updates
    rec = _update(ag3s, obs2)
    assert ag3s._oracle_memory.n_updates == n_up and len(ag3s._oracle_memory) == n_mem
    field, _ = fw._build(ag3s)
    ev2 = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    np.testing.assert_array_equal(ev2.target_points, ev1.target_points)
    assert field.stats["target_free"]["oracle"]["region_source"] == "last_accepted"


def test_carve_only_drops_samples_seen_through_and_adds_none():
    t46e = pytest.importorskip("tests.ag3s.test_t46e_oracle_extent")
    views = t46e._views()
    mem = OracleMemory(cell=0.0025, roi_margin=0.065, carve_tol=0.01)
    mem.update(views)
    n0, k0 = len(mem), mem.n_updates
    out = mem.carve_only(views)                        # same depth: nothing seen through
    assert out["n_carved"] == 0 and len(mem) == n0 and mem.n_updates == k0
    far = [OracleView(depth_m=np.where(v.depth_m > 0, v.depth_m + 0.2, 0.0), K=v.K, T_base_cam=v.T_base_cam,
                      labels=np.zeros_like(v.labels)) for v in views]
    out = mem.carve_only(far)                          # everything moved away: the samples are seen through
    assert out["n_carved"] > 0.9 * n0 and len(mem) == n0 - out["n_carved"] and mem.n_updates == k0


def test_target_check_rejects_a_region_off_the_ats_target(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                      surface_keep_last=True, surface_target_check="overlap")
    obs, _, _ = _rig(fw)
    rec = _update(ag3s, obs, ats_points=_apple_points(fw) + [0.0, 0.3, 0.0])   # the ATS target is elsewhere
    surf = ag3s._surface_frame
    assert rec["segmentation"] is False and surf["why"] == "not_ats_target" and surf["selection_applied"]
    assert surf["t78"]["target_check"]["why"] == "no_overlap" and ag3s._surface_codes is None
    assert len(ag3s._oracle_memory) == 0
    rec = _update(ag3s, obs, ats_points=_apple_points(fw))
    assert rec["segmentation"] and ag3s._surface_frame["t78"]["target_check"]["passed"]
    assert len(ag3s._oracle_memory) > 0


def test_run_wires_the_manipulated_points_only_when_on():
    """Wiring (source): `_run` hands `_attention_surface_update` the manipulated geometry's points as `ats_points` only
    when a T78 flag is on (the off call keeps its four positional arguments — no new keyword)."""
    import inspect

    from AG3S.ag3s.runtime.pipeline import AG3S

    src = inspect.getsource(AG3S._run)
    i = src.index("self._attention_surface_update(")
    call = src[i:src.index("))", i) + 2]
    assert '{"ats_points": None if manipulated_geometry is None else manipulated_geometry.points}' in call
    assert "if self._t78_on() else {}" in call


# ============================================================================ 7 A4 · 8 A5


def test_grasp_hold_makes_no_selection_and_keeps_the_memory(monkeypatch):
    from AG3S.ag3s.runtime import attention_surface as AS

    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                      surface_keep_last=True, surface_grasp_hold=True)
    obs, _, _ = _rig(fw)
    _update(ag3s, obs)
    n0, k0 = len(ag3s._oracle_memory), ag3s._oracle_memory.n_updates
    fw._build(ag3s)
    ev0 = ag3s._esdf_builder.calls[-1]["oracle_evidence"]

    def boom(*a, **k):
        raise AssertionError("selected during a grasp")

    monkeypatch.setattr(AS, "select_target_surface", boom)
    ag3s.set_grasp_active(True)
    obs2, _, _ = _rig(fw)
    rec = _update(ag3s, obs2)
    assert rec["memory"]["frozen"] and rec["memory"]["carved"] and ag3s._surface_frame["why"] == "grasp_hold"
    assert len(ag3s._oracle_memory) == n0 and ag3s._oracle_memory.n_updates == k0   # same scene: nothing carved
    fw._build(ag3s)
    np.testing.assert_array_equal(ag3s._esdf_builder.calls[-1]["oracle_evidence"].target_points, ev0.target_points)
    ag3s.set_grasp_active(False)
    monkeypatch.undo()


def test_no_ball_derives_the_support_guard_from_the_region(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                      surface_keep_last=True, surface_no_ball=True)
    obs, _, _ = _rig(fw)
    _update(ag3s, obs)
    field, _ = fw._build(ag3s)
    kw = ag3s._esdf_builder.calls[-1]
    assert kw["target_free_ball"] is None and "nontarget_points" not in kw
    ev = kw["oracle_evidence"]
    assert len(ev.target_points) > 100
    assert ev.z_floor == pytest.approx(fw.TABLE_Z + 1.0 * fw.FINE)
    rec = field.stats["target_free"]["oracle"]
    assert rec["z_floor_source"] == "support_plane_under_region" and rec["applied"]
    # with the ball (A5 off) the guard is the ball's z_min — the same plane + pad here
    fw2, b = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                    surface_keep_last=True)
    obs, _, _ = _rig(fw2)
    _update(b, obs)
    fw2._build(b)
    kb = b._esdf_builder.calls[-1]
    assert kb["oracle_evidence"].z_floor == pytest.approx(kb["target_free_ball"].z_min)
