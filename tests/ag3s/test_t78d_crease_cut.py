"""T78d — concave-crease edge cut on a 2.5 mm attention-surface graph + small-component re-attach (default off).

| # | property |
|---|---|
| 1 | config: defaults off (False · 2.5 · 20 · None), round trip, refusals (not attention_surface · ratio without cut · ≤ 0 / NaN / inf / bool · angle ≥ 180) |
| 2 | serve_safe: absent flags → no esdf key, no record-meta key, `_t78d_crease` None; given → keys; refusals |
| 3 | off = unchanged: with every crease function rigged to raise, the default call and the pipeline (IFG-CL1 and T78 A paths) run and give the same record (no `crease` key) and the T78 B call (no `crease` kwarg) |
| 4 | the CPU crease path = the study `t78c_crease.run_variant(0.0025, cut=True)` (copied inline) on a touching-spheres scene: same selected nodes, the touching neighbour is cut off, the record block |
| 5 | re-attach on a hand-made graph: a small component behind cut edges is added whole, a large one / a support one / one that touches only through uncut edges is not; one pass |
| 6 | device solve (PCG, torch CPU) = the locked `ifg_graph.solve`; GPU (skipped without CUDA): radius normals = `normals_pca`; GPU crease path = CPU crease path (IoU ≥ 0.99, codes ≥ 0.99, re-attach); warm-up with crease |
| 7 | storage (skipped otherwise): T78c requests — CPU (and GPU) crease = the study script |
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib

import numpy as np
import pytest

from AG3S.ag3s import config as ag3s_config
from AG3S.ag3s.config import AG3SConfigError, EsdfConfig
from AG3S.ag3s.runtime import attention_surface as AS
from AG3S.ag3s.runtime import attention_surface_crease as CR
from AG3S.ag3s.runtime.attention_surface import (
    ARM_SPHERES, CAMERA_ORDER, SurfaceView, _pixel_nodes, attention_vector, graph_config, select_target_surface)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:   # noqa: BLE001 — torch missing in the CPU venv
        return False


needs_cuda = pytest.mark.skipif(not _cuda(), reason="no CUDA (torch) in this interpreter")


# ------------------------------------------------------------------------------------------- touching scene

TARGET = np.array([0.55, 0.05, 0.035])
R = 0.035
NEIGHBOUR = TARGET + np.array([0.0, 2 * R + 0.001, 0.0])   # 1 mm gap: one continuous depth surface at 5 mm


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


def _raycast(T, hw=(240, 320), f=260.0):
    h, w = hw
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    v, u = np.mgrid[0:h, 0:w].astype(float)
    dc = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1)
    db = dc @ T[:3, :3].T
    o = T[:3, 3]
    tbest = np.full((h, w), np.inf)
    lab = np.zeros((h, w), np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = -o[2] / db[..., 2]
    ok = np.isfinite(tp) & (tp > 0)
    tbest[ok] = tp[ok]
    lab[ok] = 2
    for c, code in ((TARGET, 1), (NEIGHBOUR, 3)):
        oc = o - c
        a = np.sum(db * db, -1)
        b = 2 * np.sum(db * oc, -1)
        cc = np.sum(oc * oc) - R * R
        disc = b * b - 4 * a * cc
        hit = disc >= 0
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        better = hit & (t > 0) & (t < tbest)
        tbest[better] = t[better]
        lab[better] = code
    return np.where(np.isfinite(tbest), tbest, 0.0), K, lab


def _scene(hw=(240, 320)):
    mid = (TARGET + NEIGHBOUR) / 2
    eyes = (mid + [0.0, -0.10, 0.45], mid + [-0.40, -0.15, 0.30], mid + [-0.35, 0.20, 0.35])
    views, labels = [], {}
    h, w = hw
    for n, eye in zip(CAMERA_ORDER, eyes):
        T = _look_at(eye, mid)
        d, K, lab = _raycast(T, hw)
        cov = (lab == 1).reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))
        rm = np.zeros(hw, bool)
        rm[:12, :] = True
        views.append(SurfaceView(n, d, K, T, rm, cov))
        labels[n] = lab
    return views, labels


@pytest.fixture(scope="module")
def scene():
    return _scene()


def _iou(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return 1.0 if not (a or b) else len(a & b) / len(a | b)


def _code_px(res, labels, lab):
    return sum(int(((res.codes[n] == 1) & (labels[n] == lab)).sum()) for n in CAMERA_ORDER)


# ------------------------------------------------------------------------------------------- the study (inline copy)


def _study_run_variant(views, smap, voxel, cut):
    """`verify/T78c/t78c_crease.py` `run_variant` (lead 2026-11-01), copied line for line (minus timing)."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_round_ns as RNS
    from AG3S.ag3s.experiments.studies import ifg_s1 as S1

    def normals_pca(pos, r):
        pairs = cKDTree(pos).query_pairs(r, output_type="ndarray")
        n = len(pos)
        A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        A = (A + A.T + sp.identity(n)).tocsr(); A.data[:] = 1.0
        deg = np.asarray(A.sum(1)).ravel()
        mean = (A @ pos) / deg[:, None]
        cov = (A @ (pos[:, :, None] * pos[:, None, :]).reshape(n, 9)).reshape(n, 3, 3) / deg[:, None, None] \
            - mean[:, :, None] * mean[:, None, :]
        w, vec = np.linalg.eigh(cov)
        nrm = vec[:, :, 0].copy(); nrm[(deg < 5) | (w[:, 1] < 1e-10)] = np.nan
        return nrm

    cfg = dataclasses.replace(graph_config(), voxel=voxel)
    depth = [np.asarray(v.depth_m, np.float64) for v in views]
    geo = G.build_geometry(depth, [np.asarray(v.K, float) for v in views], [np.asarray(v.T_base_cam, float) for v in views],
                           [np.asarray(v.robot_mask, bool) for v in views], [np.zeros(d.shape, np.int64) for d in depth],
                           ["target"], cfg)
    vp = RNS.eligible(geo, cfg, True)
    if cut:
        six = S1.top_k(smap, S1.SEED_K, vp)
        if six.size:
            sw = smap[six] / smap[six].sum()
            roi = G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius)
            idx = np.flatnonzero(roi)
            nrm = np.full((len(geo.pos), 3), np.nan)
            nrm[idx] = normals_pca(geo.pos[idx], max(2 * voxel, 0.0075))
            pix, _ = _pixel_nodes(views, cfg, geo.pos)
            view = np.zeros((len(geo.pos), 3))
            for v, (u, vv, node) in zip(views, pix):
                ok = node >= 0
                c = np.asarray(v.T_base_cam, float)[:3, 3]
                d = c[None] - geo.pos[node[ok]]; d /= np.linalg.norm(d, axis=1, keepdims=True)
                np.add.at(view, node[ok], d)
            s = np.sign(np.einsum("ij,ij->i", nrm, view)); s[s == 0] = 1; nrm *= s[:, None]
            E = geo.edges; a, b = E[:, 0], E[:, 1]
            inroi = roi[a] & roi[b] & np.isfinite(nrm[a, 0]) & np.isfinite(nrm[b, 0])
            conv = np.einsum("ij,ij->i", nrm[a] - nrm[b], geo.pos[a] - geo.pos[b])
            ang = np.arccos(np.clip(np.einsum("ij,ij->i", nrm[a], nrm[b]), -1, 1))
            cutm = inroi & (conv < 0) & (ang >= np.deg2rad(20.0))
            geo = dataclasses.replace(geo, edges=E[~cutm])
    res = RNS.g_and_top1(geo, cfg, smap, vp) if len(geo.pos) else None
    if res is None:
        return geo, np.zeros(0, np.int64)
    sub, six, m, mt, info = res
    return geo, np.searchsorted(geo.node_eval, sub.node_eval[np.isfinite(mt) & (mt >= 0.5)])


# ============================================================================ 1 config


def test_config_defaults_round_trip_and_refusals():
    e = EsdfConfig()
    assert (e.surface_crease_cut, e.surface_crease_voxel_mm, e.surface_crease_angle_deg,
            e.surface_crease_reattach_ratio) == (False, 2.5, 20.0, None)
    on = dataclasses.replace(e, target_extent="attention_surface", surface_crease_cut=True,
                             surface_crease_reattach_ratio=0.25)
    on.validate() if hasattr(on, "validate") else None
    p = CR.from_esdf(on)
    assert p == CR.CreaseParams(2.5, 20.0, 0.25) and math.isclose(p.voxel_m, 0.0025)
    assert math.isclose(p.normal_radius_m, 0.0075) and CR.from_esdf(e) is None
    bad = [dict(surface_crease_cut=True),                                                   # not attention_surface
           dict(target_extent="attention_surface", surface_crease_reattach_ratio=0.25),     # ratio without cut
           dict(target_extent="attention_surface", surface_crease_cut=1),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_voxel_mm=0.0),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_voxel_mm=-2.5),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_voxel_mm=float("nan")),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_angle_deg=0),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_angle_deg=180.0),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_angle_deg=float("inf")),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_reattach_ratio=0.0),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_reattach_ratio=-0.25),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_reattach_ratio=float("nan")),
           dict(target_extent="attention_surface", surface_crease_cut=True, surface_crease_reattach_ratio=True)]
    for kw in bad:
        with pytest.raises(AG3SConfigError):
            cfg = dataclasses.replace(EsdfConfig(), **kw)
            cfg.validate() if hasattr(cfg, "validate") else None
    for kw in (dict(voxel_mm=0), dict(angle_deg=float("nan")), dict(angle_deg=200), dict(reattach_ratio=0)):
        with pytest.raises(ValueError):
            CR.CreaseParams(**kw)


# ============================================================================ 2 serve_safe flags


def _args(*argv):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])
    reject_bad_flag_combinations(ap, args)
    return args


SURF_ON = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface",
           "--target-field-policy", "exclude-authorized", "--attention-source", "noun_target:L10H7")


def test_cli_absent_changes_nothing_and_given_maps_to_keys():
    from AG3S.trajopt.serve_safe import _t78d_crease, ifg_cl1_record_meta, target_exclusion_section

    for argv in ((), SURF_ON):
        args = _args(*argv)
        assert not any(k.startswith("surface_crease") for k in ifg_cl1_record_meta(args))
        assert not any(k.startswith("surface_crease") for k in target_exclusion_section(args))
        assert _t78d_crease(args) is None
    args = _args(*SURF_ON, "--surface-crease-cut", "--surface-crease-reattach-ratio", "0.25")
    keys = {"surface_crease_cut": True, "surface_crease_reattach_ratio": 0.25}
    assert {k: v for k, v in ifg_cl1_record_meta(args).items() if k.startswith("surface_crease")} == keys
    assert {k: v for k, v in target_exclusion_section(args).items() if k.startswith("surface_crease")} == keys
    assert _t78d_crease(args) == CR.CreaseParams(2.5, 20.0, 0.25)
    args = _args(*SURF_ON, "--surface-crease-cut", "--surface-crease-voxel-mm", "2", "--surface-crease-angle-deg", "30")
    assert _t78d_crease(args) == CR.CreaseParams(2.0, 30.0, None)


def test_cli_refusals(capsys):
    for argv in (("--surface-crease-cut",),
                 (*SURF_ON, "--surface-crease-reattach-ratio", "0.25"),
                 (*SURF_ON, "--surface-crease-voxel-mm", "2.5"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-reattach-ratio", "0"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-reattach-ratio", "nan"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-voxel-mm", "-1"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-angle-deg", "180")):
        with pytest.raises(SystemExit) as e:
            _args(*argv)
        assert e.value.code == 2 and "--surface-crease" in capsys.readouterr().err, argv


# ============================================================================ 3 off = unchanged


def _rig_crease_to_raise(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("T78d crease step called with the flag off")

    for name in ("cut_cpu", "reattach", "normals_pca", "from_esdf", "cut_edges", "orient"):
        monkeypatch.setattr(CR, name, boom)
    monkeypatch.setattr(AS, "_crease_reattach_cpu", boom)
    try:
        from AG3S.ag3s.runtime import attention_surface_gpu as GPU

        monkeypatch.setattr(GPU, "_crease_cut", boom)
        monkeypatch.setattr(GPU, "_radius_normals", boom)
    except Exception:   # noqa: BLE001 — torch missing: the GPU module is never reached on the CPU path
        pass


def test_off_select_never_calls_the_crease_steps(monkeypatch, scene):
    views, _ = scene
    ref = select_target_surface(views, arm=ARM_SPHERES)
    _rig_crease_to_raise(monkeypatch)
    a = select_target_surface(views, arm=ARM_SPHERES)
    b = select_target_surface(views, arm=ARM_SPHERES, crease=None)
    for r in (a, b):
        assert "crease" not in r.record
        assert {k: v for k, v in r.record.items() if k != "ms"} == {k: v for k, v in ref.record.items() if k != "ms"}
        assert np.array_equal(r.selected_nodes, ref.selected_nodes)
        assert all(np.array_equal(r.codes[n], ref.codes[n]) for n in CAMERA_ORDER)


@pytest.mark.parametrize("t78", [False, True])
def test_off_pipeline_makes_the_t78b_call(monkeypatch, t78):
    """Flag off: the pipeline's `select_target_surface` call has exactly the T78 B kwargs (no `crease`), on the IFG-CL1
    path and on the T78 A path, and no crease function runs."""
    cl1 = pytest.importorskip("tests.ag3s.test_ifg_cl1_attention_surface")
    extra = dict(surface_target_check="overlap") if t78 else {}
    fw, ag3s = cl1._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface", **extra)
    obs, _, _ = cl1._raycast_rig(fw)
    cams = cl1._cameras(obs)
    seen, orig = [], AS.select_target_surface

    def spy(views, **kw):
        seen.append(sorted(kw))
        return orig(views, **kw)

    _rig_crease_to_raise(monkeypatch)
    monkeypatch.setattr(AS, "select_target_surface", spy)
    assert ag3s._surface_crease_kwargs() == {}
    if t78:
        ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {}, ats_points=None)
    else:
        ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    assert seen == [["arm", "device"]]
    assert "crease" not in ag3s._surface_frame


def test_on_pipeline_passes_the_params(monkeypatch):
    cl1 = pytest.importorskip("tests.ag3s.test_ifg_cl1_attention_surface")
    fw, ag3s = cl1._wired(monkeypatch, target_exclusion="ball_owned", target_extent="attention_surface",
                          surface_crease_cut=True, surface_crease_reattach_ratio=0.25)
    obs, _, _ = cl1._raycast_rig(fw)
    cams = cl1._cameras(obs)
    rec = ag3s._attention_surface_update(obs, cams, {c.name: c.robot_mask for c in cams}, {})
    assert ag3s._surface_crease_kwargs() == {"crease": CR.CreaseParams(2.5, 20.0, 0.25)}
    cr = ag3s._surface_frame["crease"]
    assert cr["voxel_mm"] == 2.5 and cr["reattach_ratio"] == 0.25 and "n_cut" in cr
    assert isinstance(rec, dict)
    json.dumps(ag3s._surface_frame, default=float)


# ============================================================================ 4 CPU crease = the study


def test_cpu_crease_matches_the_study_and_cuts_the_touching_neighbour(scene):
    views, labels = scene
    smap, _ = attention_vector(views)
    off = select_target_surface(views, arm=ARM_SPHERES)
    on = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams())
    geo, ssel = _study_run_variant(views, smap, 0.0025, True)
    assert on.record["applied"] and on.record["n_nodes"] == len(geo.pos)
    assert np.array_equal(np.sort(on.selected_nodes), np.sort(ssel))
    cr = on.record["crease"]
    assert cr["voxel_mm"] == 2.5 and cr["angle_deg"] == 20.0 and cr["n_cut"] > 0 and cr["n_roi_nodes"] > 0
    assert cr["n_cut"] <= cr["n_edges_roi"] and "reattach" not in cr
    # the 5 mm LOCK takes the touching neighbour, the crease graph does not
    assert _code_px(off, labels, 3) > 0.2 * _code_px(off, labels, 1), "scene: the LOCK should merge the pair"
    assert _code_px(on, labels, 3) <= 0.02 * _code_px(on, labels, 1)
    assert _code_px(on, labels, 1) >= 0.8 * _code_px(off, labels, 1)
    # the pixel → node map follows the node size: every code-1 pixel lifts into a selected 2.5 mm voxel
    pix, _ = _pixel_nodes(views, dataclasses.replace(graph_config(), voxel=0.0025), geo.pos)
    sel = np.zeros(len(geo.pos), bool); sel[ssel] = True
    for v, (u, vv, node) in zip(views, pix):
        ok = node >= 0
        assert np.array_equal(on.codes[v.name][vv[ok], u[ok]] == 1, sel[node[ok]])


def test_cpu_crease_with_keep_debug_and_codes_false(scene):
    views, _ = scene
    a = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams())
    b = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams(), keep_debug=True, codes=False)
    assert np.array_equal(a.selected_nodes, b.selected_nodes) and b.codes is None
    assert abs(float(b.geometry.pos[:, 0].max() - b.geometry.pos[:, 0].min())) > 0   # the 2.5 mm geometry
    assert b.record["n_nodes"] == a.record["n_nodes"]


# ============================================================================ 5 re-attach


def _grid_graph():
    """Node 0..9 = region (selected), 10..12 = small part (cut from the region), 13..42 = big part (cut), 43..44 =
    support (cut), 45..46 = a part touching the region through an uncut and a cut edge (same component as the
    region → not a candidate), 47 = an isolated node. Chains inside each part."""
    n = 48
    pos = np.c_[np.arange(n) * 0.0025, np.zeros(n), np.zeros(n)]
    E = []
    for lo, hi in ((0, 10), (10, 13), (13, 43), (43, 45), (45, 47)):
        E += [(i, i + 1) for i in range(lo, hi - 1)]
    E += [(9, 45)]                                   # uncut: 45–46 belong to the region's component
    cut = np.array([(0, 10), (1, 13), (2, 43), (3, 46)])
    support = np.zeros(n, bool); support[43:45] = True
    sel = np.zeros(n, bool); sel[:10] = True
    return n, np.array(E), cut, support, sel, pos


def test_reattach_adds_small_cut_components_whole_once():
    n, E, cut, support, sel, pos = _grid_graph()
    add, rec = CR.reattach(n, E, cut, support, sel, pos, 0.5)
    assert np.flatnonzero(add).tolist() == [10, 11, 12]
    assert rec["n_selected_before"] == 10 and rec["n_candidates"] == 2 and rec["n_attached_components"] == 1
    comp = {c["n_nodes"]: c for c in rec["components"]}
    assert comp[3]["attached"] and not comp[30]["attached"] and comp[3]["ratio"] == 0.3 and comp[30]["n_cut_edges"] == 1
    add, rec = CR.reattach(n, E, cut, support, sel, pos, 3.0)       # the big part fits under R = 3
    assert add.sum() == 33 and rec["n_attached_components"] == 2
    add, rec = CR.reattach(n, E, cut, support, sel, pos, 0.25)      # 3 > 0.25 × 10
    assert not add.any() and rec["n_attached_components"] == 0
    add, rec = CR.reattach(n, E, np.zeros((0, 2), np.int64), support, sel, pos, 0.5)
    assert not add.any() and rec["n_candidates"] == 0


def test_reattach_in_select_adds_nodes_on_both_paths(scene):
    views, _ = scene
    a = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams())
    b = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams(reattach_ratio=0.25))
    rr = b.record["crease"]["reattach"]
    assert rr["n_selected_before"] == a.record["n_selected_nodes"]
    assert b.record["n_selected_nodes"] == a.record["n_selected_nodes"] + rr["n_attached_nodes"]
    assert set(map(int, a.selected_nodes)) <= set(map(int, b.selected_nodes))
    for c in rr["components"]:
        assert c["attached"] == (c["n_nodes"] <= 0.25 * rr["n_selected_before"])


# ============================================================================ 6 GPU


def test_device_solve_matches_the_locked_solve(scene):
    """`attention_surface_gpu._solve_device` (PCG, crease path only) = `ifg_graph.solve` (spsolve) on the crease
    ROI sub-graph of the touching scene — torch on the CPU is enough (skipped without torch)."""
    torch = pytest.importorskip("torch")
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_round_ns as RNS
    from AG3S.ag3s.experiments.studies import ifg_run as RN
    from AG3S.ag3s.experiments.studies import ifg_s1 as S1
    from AG3S.ag3s.runtime import attention_surface_gpu as GPU

    views, _ = scene
    smap, _ = attention_vector(views)
    p = CR.CreaseParams()
    cfg = p.graph_config(graph_config())
    depth = [np.asarray(v.depth_m, np.float64) for v in views]
    geo = G.build_geometry(depth, [v.K for v in views], [v.T_base_cam for v in views],
                           [np.asarray(v.robot_mask, bool) for v in views], [np.zeros(d.shape, np.int64) for d in depth],
                           ["target"], cfg)
    vp = RNS.eligible(geo, cfg, True)
    geo, _ = CR.cut_cpu(views, geo, cfg, smap, vp, p)
    six = S1.top_k(smap, S1.SEED_K, vp)
    sw = smap[six] / smap[six].sum()
    sub = G.subset(geo, G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius))
    n = len(sub.pos)
    sl = G.seed_unary(sub, six, sw)
    lam, s = RN._combine([(sl, np.ones(n)), (cfg.kappa_support * sub.support.astype(float), np.zeros(n))])
    w = G.edge_weights(sub, cfg, None)
    m0, i0 = G.solve(sub, np.ones(n, bool), w, lam, s, cfg.lam)
    m1, i1 = GPU._solve_device(sub, w, lam, s, cfg.lam, torch.device("cpu"))
    assert np.array_equal(np.isfinite(m0), np.isfinite(m1)) and i1["cg_iter"] > 0 and "cg_fallback" not in i1
    assert np.nanmax(np.abs(m0 - m1)) < 1e-8 and np.array_equal(m0 >= 0.5, m1 >= 0.5)
    for k in ("n_roi", "n_components", "n_unknown_components", "n_unknown_nodes"):
        assert i0[k] == i1[k], k
    # no unary anywhere → the locked "-1" (outside) answer
    m2, _ = GPU._solve_device(sub, w, np.zeros(n), s, cfg.lam, torch.device("cpu"))
    m3, _ = G.solve(sub, np.ones(n, bool), w, np.zeros(n), s, cfg.lam)
    assert np.array_equal(m2, m3)


@needs_cuda
def test_gpu_radius_normals_match_normals_pca():
    import types

    import torch

    from AG3S.ag3s.runtime import attention_surface_gpu as GPU

    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 6000); ph = np.arccos(rng.uniform(-1, 1, 6000))
    pts = 0.035 * np.c_[np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)] + 0.3
    v = 0.0025
    keys = np.floor(pts / v).astype(np.int64); keys -= keys.min(0); ext = keys.max(0) + 1
    packed = (keys[:, 0] * ext[1] + keys[:, 1]) * ext[2] + keys[:, 2]
    nd, inv = np.unique(packed, return_inverse=True)
    pos = np.stack([np.bincount(inv, weights=pts[:, k]) for k in range(3)], 1) / np.bincount(inv)[:, None]
    dev = GPU.cuda_device()
    g = types.SimpleNamespace(pos=torch.as_tensor(pos, device=dev), nd_keys=torch.as_tensor(nd, device=dev),
                              ext=ext.tolist())
    idx = torch.arange(len(pos), device=dev)
    got = GPU._radius_normals(g, idx, 0.0075, v, dev).cpu().numpy()
    ref = CR.normals_pca(pos, 0.0075)
    fin = np.isfinite(ref[:, 0])
    assert np.array_equal(fin, np.isfinite(got[:, 0]))
    assert np.abs(np.abs(np.einsum("ij,ij->i", got[fin], ref[fin])) - 1).max() < 1e-9


@needs_cuda
@pytest.mark.parametrize("ratio", [None, 0.25])
def test_gpu_crease_matches_cpu(scene, ratio):
    views, _ = scene
    p = CR.CreaseParams(reattach_ratio=ratio)
    c = select_target_surface(views, arm=ARM_SPHERES, device="cpu", crease=p)
    g = select_target_surface(views, arm=ARM_SPHERES, device="cuda", crease=p)
    assert g.record["device"] == "cuda" and c.record["applied"] and g.record["applied"]
    assert _iou(c.selected_nodes, g.selected_nodes) >= 0.99
    assert np.mean([(c.codes[n] == g.codes[n]).mean() for n in CAMERA_ORDER]) >= 0.99
    for k in ("n_roi_nodes", "n_edges_roi", "n_cut"):
        assert abs(c.record["crease"][k] - g.record["crease"][k]) <= max(2, 0.01 * c.record["crease"][k]), k
    assert ("reattach" in g.record["crease"]) == (ratio is not None)
    assert "crease" in g.record["ms_detail"]


@needs_cuda
def test_gpu_warmup_with_crease():
    from AG3S.ag3s.runtime import attention_surface_gpu as GPU

    assert GPU.warmup(repeat=1, crease=CR.CreaseParams(reattach_ratio=0.25)) > 0


# ============================================================================ 7 storage (T78c requests)


STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")) / "results/development"
S2 = STORE / "20261030T170909+0900_8ab3c7ef7757_9e05cdff58b9/verify/T78/S2"
needs_s2 = pytest.mark.skipif(not (S2 / "REG_k0/replay.json").is_file(), reason="T78 S2 records not mounted")
CAMS = (("head", "zed_left"), ("left_wrist", "wrist_cam_l"), ("right_wrist", "wrist_cam_r"))


def _s2_views(group: str, client_suffix: str, k: int):
    rep = [r for r in json.loads((S2 / group / "replay.json").read_text())]
    chunks = sorted((S2 / group / "server_constraints/run_0000").glob("chunk_*.npz"))
    i = 0
    for r in rep:
        steps = sorted((pathlib.Path(r["client_dir"]) / "ag3s/run_0000").glob("step_*.npz"),
                       key=lambda p: int(np.load(p)["t_step"]))
        if pathlib.Path(r["client_dir"]).name.endswith(client_suffix):
            z = np.load(chunks[i + k], allow_pickle=True)
            s = np.load(steps[k], allow_pickle=True)
            break
        i += len(steps)
    else:
        pytest.skip(f"{client_suffix} not in {group}")
    views = []
    for name, mj in CAMS:
        codes = z[f"attention_surface_codes_{name}"]
        views.append(SurfaceView(name, np.asarray(s[f"depth_{mj}"], np.float64) / 1000.0, s[f"K_{mj}"],
                                 s[f"T_base_cam_{mj}"], np.asarray(codes) == 0,
                                 np.asarray(z[f"attention_{name}"], np.float64)))
    return views


@needs_s2
@pytest.mark.parametrize("group,client,k", [("REG_k0", "ep211_s2113", 15), ("HURDLE_A", "ep119_s1193", 9)])
def test_crease_matches_the_study_on_t78c_requests(group, client, k):
    views = _s2_views(group, client, k)
    smap, _ = attention_vector(views)
    geo, ssel = _study_run_variant(views, smap, 0.0025, True)
    c = select_target_surface(views, arm=ARM_SPHERES, device="cpu", crease=CR.CreaseParams())
    assert _iou(c.selected_nodes, ssel) >= 0.99
    if _cuda():
        g = select_target_surface(views, arm=ARM_SPHERES, device="cuda", crease=CR.CreaseParams())
        assert _iou(g.selected_nodes, ssel) >= 0.99
