"""T97 — protrusion re-attach (`esdf.surface_crease_protrusion`) and hand freeze (`esdf.surface_crease_hand_freeze_mm`)
on top of the T78d crease cut, both default off.

| # | property |
|---|---|
| 1 | config: defaults off (False · 0.15 · None), round trip, refusals (no crease cut · with the ratio rule · freeze without keep_last · ≤ 0 / NaN / bool); `CreaseParams` refusals; `record()` unchanged when off |
| 2 | serve_safe: absent flags → no esdf / meta key, `_t97_crease` = `_t78d_crease`; given → keys · params; refusals |
| 3 | off = unchanged: with `protrude` rigged to raise, the crease selection (CPU) runs, its record has no `protrusion`; the pipeline with freeze off never reads the palm, stores no region, records no `freeze` |
| 4 | rendered scene (cylinder body + stem on top + a touching same-height cylinder beside it; cut = the GT label boundaries): the stem component is attached, the neighbour is not |
| 5 | `protrude` on hand-made point sets: each of the three conditions alone rejects; no support plane / degenerate hull → nothing added; one pass, whole components |
| 6 | `select_target_surface(crease=CreaseParams(protrusion=True))`: record, node count = before + attached, decisions = rule; GPU (CUDA only) = CPU |
| 7 | pipeline hand freeze: palm within D of the last accepted centroid while searching / latched → no selection (carve only, `why = crease_hand_freeze`); far / held / attached / placed → selection runs; reset clears |
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, EsdfConfig
from AG3S.ag3s.runtime import attention_surface as AS
from AG3S.ag3s.runtime import attention_surface_crease as CR
from AG3S.ag3s.runtime.attention_surface import ARM_SPHERES, CAMERA_ORDER, SurfaceView, graph_config, select_target_surface


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:   # noqa: BLE001 — torch missing in the CPU venv
        return False


needs_cuda = pytest.mark.skipif(not _cuda(), reason="no CUDA (torch) in this interpreter")

ON = dict(target_extent="attention_surface", surface_crease_cut=True)


# ============================================================================ the rendered scene

BODY = np.array([0.55, 0.05])
RB, HB = 0.035, 0.070                     # body cylinder: radius, height (flat top)
STEM_R, STEM_H = 0.008, 0.020             # stem on the body's top, centred
GAP = -0.002                              # neighbour: same cylinder, pressed 2 mm into the body (touching)
NB = BODY + np.array([0.0, 2 * RB + GAP])
LAB_BODY, LAB_TABLE, LAB_NB, LAB_STEM = 1, 2, 3, 4


def _look_at(eye, target):
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, eye
    return T


def _cylinder(o, d, c, r, z0, z1):
    """Ray parameter of the first hit on a vertical cylinder (side z ∈ [z0, z1] + top cap), inf if none."""
    ox, oy = o[0] - c[0], o[1] - c[1]
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (ox * d[..., 0] + oy * d[..., 1])
    disc = b * b - 4 * a * (ox * ox + oy * oy - r * r)
    t = np.full(a.shape, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        zs = o[2] + ts * d[..., 2]
        side = (disc >= 0) & (ts > 0) & (zs >= z0) & (zs <= z1)
        t[side] = ts[side]
        tc = (z1 - o[2]) / d[..., 2]
        px, py = o[0] + tc * d[..., 0] - c[0], o[1] + tc * d[..., 1] - c[1]
        cap = (tc > 0) & (px * px + py * py <= r * r)
    return np.where(cap & (tc < t), tc, t)


def _render(T, hw=(240, 320), f=260.0):
    h, w = hw
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    v, u = np.mgrid[0:h, 0:w].astype(float)
    db = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1) @ T[:3, :3].T
    o = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = -o[2] / db[..., 2]
    t = np.where(np.isfinite(tp) & (tp > 0), tp, np.inf)
    lab = np.where(np.isfinite(t), LAB_TABLE, 0).astype(np.uint8)
    for c, r, z0, z1, code in ((BODY, RB, 0.0, HB, LAB_BODY), (BODY, STEM_R, HB - 0.005, HB + STEM_H, LAB_STEM),
                               (NB, RB, 0.0, HB, LAB_NB)):
        tc = _cylinder(o, db, c, r, z0, z1)
        better = tc < t
        t[better] = tc[better]
        lab[better] = code
    return np.where(np.isfinite(t), t, 0.0), K, lab   # the camera z of a ray (dc.z = 1) is its parameter


def _scene(hw=(240, 320)):
    mid = np.r_[(BODY + NB) / 2, 0.04]
    views, labels = [], {}
    h, w = hw
    for n, off in zip(CAMERA_ORDER, ([0.0, -0.10, 0.45], [-0.40, -0.15, 0.30], [-0.35, 0.20, 0.35])):
        T = _look_at(mid + off, mid)
        d, K, lab = _render(T, hw)
        att = ((lab == LAB_BODY) | (lab == LAB_STEM)).reshape(16, h // 16, 16, w // 16).mean(axis=(1, 3))
        rm = np.zeros(hw, bool)
        rm[:12, :] = True
        views.append(SurfaceView(n, d, K, T, rm, att))
        labels[n] = lab
    return views, labels


@pytest.fixture(scope="module")
def scene():
    return _scene()


# ============================================================================ 1 config


def test_config_defaults_round_trip_and_refusals():
    e = EsdfConfig()
    assert (e.surface_crease_protrusion, e.surface_crease_protrusion_area, e.surface_crease_hand_freeze_mm) == (
        False, 0.15, None)
    cfg = AG3SConfig.from_dict({"esdf": {**ON, "surface_keep_last": True, "surface_crease_protrusion": True,
                                         "surface_crease_protrusion_area": 0.2,
                                         "surface_crease_hand_freeze_mm": 80.0}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    p = CR.from_esdf(cfg.esdf)
    assert p == CR.CreaseParams(2.5, 20.0, None, True, 0.2)
    assert CR.from_esdf(dataclasses.replace(e, **ON)) == CR.CreaseParams()   # off: the T78d params
    bad = [dict(surface_crease_protrusion=True, target_extent="attention_surface"),          # no crease cut
           dict(**ON, surface_crease_protrusion=True, surface_crease_reattach_ratio=0.25),   # with the ratio rule
           dict(**ON, surface_crease_protrusion=1),
           dict(**ON, surface_crease_protrusion_area=0.0),
           dict(**ON, surface_crease_protrusion_area=float("nan")),
           dict(**ON, surface_crease_protrusion_area=True),
           dict(**ON, surface_crease_hand_freeze_mm=80.0),                                  # no keep_last
           dict(target_extent="attention_surface", surface_keep_last=True, surface_crease_hand_freeze_mm=80.0),
           dict(**ON, surface_keep_last=True, surface_crease_hand_freeze_mm=0.0),
           dict(**ON, surface_keep_last=True, surface_crease_hand_freeze_mm=float("inf")),
           dict(**ON, surface_keep_last=True, surface_crease_hand_freeze_mm=True)]
    for kw in bad:
        with pytest.raises(AG3SConfigError):
            dataclasses.replace(EsdfConfig(), **kw).validate()
    for kw in (dict(protrusion=True, reattach_ratio=0.25), dict(protrusion=1), dict(protrusion_area=0.0),
               dict(protrusion_area=float("nan"))):
        with pytest.raises(ValueError):
            CR.CreaseParams(**kw)
    # off: the T78d record exactly (no new key)
    assert CR.CreaseParams().record() == {"voxel_mm": 2.5, "angle_deg": 20.0, "normal_radius_mm": 7.5,
                                          "reattach_ratio": None}
    assert CR.CreaseParams(protrusion=True).record()["protrusion_area"] == 0.15


# ============================================================================ 2 serve_safe


def _args(*argv):
    from AG3S.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])
    reject_bad_flag_combinations(ap, args)
    return args


SURF_ON = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "attention-surface",
           "--target-field-policy", "exclude-authorized", "--attention-source", "noun_target:L10H7")
T97_KEYS = ("surface_crease_protrusion", "surface_crease_protrusion_area", "surface_crease_hand_freeze_mm")


def test_cli_absent_changes_nothing_and_given_maps_to_keys():
    from AG3S.trajopt.serve_safe import _t78d_crease, _t97_crease, ifg_cl1_record_meta, target_exclusion_section

    for argv in ((), SURF_ON, (*SURF_ON, "--surface-crease-cut"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-reattach-ratio", "0.25")):
        args = _args(*argv)
        assert not any(k in T97_KEYS for k in ifg_cl1_record_meta(args))
        assert not any(k in T97_KEYS for k in target_exclusion_section(args))
        assert _t97_crease(args) == _t78d_crease(args)
    args = _args(*SURF_ON, "--surface-keep-last", "--surface-crease-cut", "--surface-crease-protrusion",
                 "--surface-crease-hand-freeze-mm", "80")
    keys = {"surface_crease_protrusion": True, "surface_crease_hand_freeze_mm": 80.0}
    assert {k: v for k, v in ifg_cl1_record_meta(args).items() if k in T97_KEYS} == keys
    assert {k: v for k, v in target_exclusion_section(args).items() if k in T97_KEYS} == keys
    assert _t97_crease(args) == CR.CreaseParams(protrusion=True, protrusion_area=0.15)
    args = _args(*SURF_ON, "--surface-crease-cut", "--surface-crease-protrusion", "--surface-crease-protrusion-area",
                 "0.1")
    assert _t97_crease(args) == CR.CreaseParams(protrusion=True, protrusion_area=0.1)


def test_cli_refusals(capsys):
    for argv in ((*SURF_ON, "--surface-crease-protrusion"),
                 (*SURF_ON, "--surface-keep-last", "--surface-crease-hand-freeze-mm", "80"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-protrusion-area", "0.15"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-protrusion", "--surface-crease-reattach-ratio",
                  "0.25"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-protrusion", "--surface-crease-protrusion-area",
                  "0"),
                 (*SURF_ON, "--surface-crease-cut", "--surface-crease-hand-freeze-mm", "80"),   # no keep_last
                 (*SURF_ON, "--surface-keep-last", "--surface-crease-cut", "--surface-crease-hand-freeze-mm", "nan")):
        with pytest.raises(SystemExit) as e:
            _args(*argv)
        assert e.value.code == 2 and "--surface-crease" in capsys.readouterr().err, argv


# ============================================================================ 3 off = unchanged


def _rig_t97_to_raise(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("T97 step called with its flag off")

    for name in ("protrude", "support_plane", "_cut_candidates"):
        monkeypatch.setattr(CR, name, boom)
    monkeypatch.setattr(AS, "_crease_protrude_cpu", boom)


def test_off_crease_selection_never_calls_protrude(monkeypatch, scene):
    views, _ = scene
    ref = {r: select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams(reattach_ratio=r))
           for r in (None, 0.25)}
    _rig_t97_to_raise(monkeypatch)
    for r, a in ref.items():
        b = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams(reattach_ratio=r))
        assert "protrusion" not in b.record["crease"] and "protrusion_area" not in b.record["crease"]
        strip = lambda d: {k: v for k, v in d.items() if k != "ms"}   # noqa: E731 — timings differ
        sa, sb = strip(a.record), strip(b.record)
        sa["crease"], sb["crease"] = strip(sa["crease"]), strip(sb["crease"])
        for s_ in (sa, sb):
            if "reattach" in s_["crease"]:
                s_["crease"]["reattach"] = strip(s_["crease"]["reattach"])
        assert sa == sb
        assert np.array_equal(a.selected_nodes, b.selected_nodes)
        assert all(np.array_equal(a.codes[n], b.codes[n]) for n in CAMERA_ORDER)


def test_off_pipeline_has_no_freeze(monkeypatch):
    from AG3S.ag3s.runtime.pipeline import AG3S

    A = pytest.importorskip("tests.ag3s.test_t78a_surface_post")

    def boom(*a, **k):
        raise AssertionError("palm read with the freeze off")

    monkeypatch.setattr(AG3S, "_grasp_palm_points", boom)
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", **ON, surface_target_check="overlap",
                        surface_keep_last=True)
    obs, _, _ = A._rig(fw, gap=0.001)
    ag3s.set_grasp_latched(True)
    rec = A._update(ag3s, obs, ats_points=A._apple_points(fw))
    assert ag3s._t97_freeze_check() is None
    assert rec["segmentation"] and "freeze" not in ag3s._surface_frame["t78"] and "freeze" not in ag3s._surface_frame
    assert getattr(ag3s, "_t97_last_region", None) is None and not hasattr(ag3s, "_t97_freeze_inputs")


# ============================================================================ 4 rendered scene, GT cut


def _gt_graph(views, labels):
    from AG3S.ag3s.experiments.studies import ifg_graph as G

    cfg = dataclasses.replace(graph_config(), voxel=0.0025)
    geo = G.build_geometry([v.depth_m for v in views], [v.K for v in views], [v.T_base_cam for v in views],
                           [v.robot_mask for v in views], [labels[n].astype(np.int64) for n in CAMERA_ORDER],
                           ["none", "target", "table", "nb", "stem"], cfg)
    return geo, np.argmax(geo.eval_class_frac[geo.node_eval], 1)


def test_scene_stem_attaches_and_the_touching_neighbour_does_not(scene):
    views, labels = scene
    geo, lab = _gt_graph(views, labels)
    E = geo.edges
    cut = lab[E[:, 0]] != lab[E[:, 1]]      # the crease cut, idealised: every object boundary edge is cut
    assert ((lab[E[cut, 0]] == LAB_BODY) & (lab[E[cut, 1]] == LAB_NB) | (lab[E[cut, 0]] == LAB_NB)
            & (lab[E[cut, 1]] == LAB_BODY)).sum() > 10, "scene: the neighbour must touch the body"
    sel = (lab == LAB_BODY) & ~geo.support
    add, rec = CR.protrude(len(geo.pos), E[~cut], E[cut], geo.support, sel, geo.pos, 0.15, 0.0025)
    json.dumps(rec)
    stem, nb = lab == LAB_STEM, lab == LAB_NB
    assert add[stem].mean() > 0.9 and not add[nb].any() and (lab[add] == LAB_STEM).all()
    assert rec["plane"]["normal"][2] > 0.999 and abs(rec["plane"]["offset_m"]) < 1e-3   # the table z = 0
    def nearest(mask):   # the recorded component whose centroid is nearest the GT part's centroid
        c = geo.pos[mask].mean(0)
        return min(rec["components"], key=lambda k: float(np.linalg.norm(np.asarray(k["centroid"]) - c)))

    comps = {"stem": nearest(stem), "nb": nearest(nb)}
    s, n = comps["stem"], comps["nb"]
    assert s["attached"] and s["rule"] == {"above": True, "small": True, "on_body": True} and s["dh_mm"] > 5
    assert s["area_ratio"] < 0.15 and s["overlap"] > 0.9
    # pressed 2 mm into the body, its projection touches the body's hull — the height (and size) condition rejects it
    assert not n["attached"] and not n["rule"]["above"] and not n["rule"]["small"] and n["dh_mm"] < 0
    # ρ small enough → the stem is rejected by the area condition alone
    add2, rec2 = CR.protrude(len(geo.pos), E[~cut], E[cut], geo.support, sel, geo.pos, 0.5 * s["area_ratio"], 0.0025)
    assert add2[stem].mean() < 0.1   # (single-node stem fragments stay small enough)


# ============================================================================ 5 hand-made point sets


def _disk(c, r, z, step=0.0025):
    g = np.arange(-r, r + 1e-9, step)
    xx, yy = np.meshgrid(g, g)
    m = xx ** 2 + yy ** 2 <= r * r
    return np.c_[c[0] + xx[m], c[1] + yy[m], np.full(m.sum(), z)]


def _chain(n, offset=0):
    return np.c_[offset + np.arange(n - 1), offset + np.arange(1, n)]


def _parts(*parts):
    """Point sets → (pos, kept edges = chains inside each part, part id per node)."""
    pos = np.vstack(parts)
    pid = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    off = np.cumsum([0] + [len(p) for p in parts])
    E = np.vstack([_chain(len(p), off[i]) for i, p in enumerate(parts)])
    return pos, E, pid, off


def _table(n_side=20):
    g = np.linspace(0.40, 0.70, n_side)
    xx, yy = np.meshgrid(g, g)
    return np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]


def test_protrude_each_condition_alone_rejects():
    body = _disk((0.55, 0.55), 0.035, 0.070)                 # the region: a flat top at 70 mm
    stem = _disk((0.55, 0.55), 0.008, 0.085)                 # above, small, on the body
    tall_big = _disk((0.555, 0.55), 0.030, 0.090)            # above, on the body, too large
    beside_high = _disk((0.63, 0.55), 0.008, 0.085)          # above, small, beside the body (off the hull)
    low_on = _disk((0.55, 0.56), 0.008, 0.060)               # small, on the body's footprint, below the top
    table = _table()
    pos, E, pid, off = _parts(body, stem, tall_big, beside_high, low_on, table)
    support = pid == 5
    sel = pid == 0
    cut = np.array([[off[0], off[k]] for k in (1, 2, 3, 4)])   # each part touches the region through one cut edge
    add, rec = CR.protrude(len(pos), E, cut, support, sel, pos, 0.15, 0.0025)
    assert set(np.unique(pid[add])) == {1} and add[pid == 1].all()
    by = {int(np.argmin([np.linalg.norm(np.asarray(c["centroid"]) - pos[pid == k].mean(0)) for k in range(1, 5)])) + 1: c
          for c in rec["components"]}
    assert by[1]["rule"] == {"above": True, "small": True, "on_body": True}
    assert by[2]["rule"] == {"above": True, "small": False, "on_body": True}
    assert by[3]["rule"] == {"above": True, "small": True, "on_body": False} and by[3]["overlap"] == 0.0
    assert by[4]["rule"] == {"above": False, "small": True, "on_body": True} and by[4]["dh_mm"] == pytest.approx(-10, abs=0.1)
    assert rec["n_candidates"] == 4 and rec["n_attached_components"] == 1 and rec["n_attached_nodes"] == len(stem)
    assert rec["h_region_m"] == pytest.approx(0.070, abs=1e-6)
    # ρ large enough → the big one too (the area condition was the only one it failed)
    add, _ = CR.protrude(len(pos), E, cut, support, sel, pos, 1.0, 0.0025)
    assert set(np.unique(pid[add])) == {1, 2}
    # no support plane → nothing added
    add, rec = CR.protrude(len(pos), E, cut, np.zeros(len(pos), bool), sel, pos, 0.15, 0.0025)
    assert not add.any() and rec["why"] == "no_support_plane" and rec["plane"] is None
    # no cut edge → no candidate
    add, rec = CR.protrude(len(pos), E, np.zeros((0, 2), np.int64), support, sel, pos, 0.15, 0.0025)
    assert not add.any() and rec["why"] == "no_cut_edge"
    # a degenerate region hull (a line) overlaps nothing
    line = np.c_[np.linspace(0.5, 0.6, 30), np.full(30, 0.55), np.full(30, 0.07)]
    pos2, E2, pid2, off2 = _parts(line, stem, table)
    add, rec = CR.protrude(len(pos2), E2, np.array([[0, off2[1]]]), pid2 == 2, pid2 == 0, pos2, 0.15, 0.0025)
    assert not add.any() and rec["hull_why"] and rec["components"][0]["overlap"] == 0.0


def test_protrude_same_candidates_as_reattach():
    import tests.ag3s.test_t78d_crease_cut as T78D

    n, E, cut, support, sel, pos = T78D._grid_graph()
    _, rr = CR.reattach(n, E, cut, support, sel, pos, 0.5)
    comp, cand, touch = CR._cut_candidates(n, E, cut, support, sel)
    assert rr["n_candidates"] == cand.size == 2 and sorted(touch.tolist()) == [1, 1]


# ============================================================================ 6 select end-to-end


def _check_select_record(r0, r1):
    pr = r1.record["crease"]["protrusion"]
    assert pr["n_selected_before"] == r0.record["n_selected_nodes"]
    assert r1.record["n_selected_nodes"] == r0.record["n_selected_nodes"] + pr["n_attached_nodes"]
    assert set(map(int, r0.selected_nodes)) <= set(map(int, r1.selected_nodes))
    for c in pr["components"]:
        assert c["attached"] == all(c["rule"].values())
    json.dumps(r1.record)
    return pr


def test_select_with_protrusion_cpu(scene):
    views, _ = scene
    r0 = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams())
    r1 = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams(protrusion=True))
    pr = _check_select_record(r0, r1)
    assert r1.record["crease"]["protrusion_area"] == 0.15 and "ms" in pr
    assert r1.record["crease"]["reattach_ratio"] is None and "reattach" not in r1.record["crease"]


@needs_cuda
def test_select_with_protrusion_gpu_equals_cpu(scene):
    views, _ = scene
    p = CR.CreaseParams(protrusion=True)
    c = select_target_surface(views, arm=ARM_SPHERES, crease=p, device="cpu")
    g = select_target_surface(views, arm=ARM_SPHERES, crease=p, device="cuda")
    g0 = select_target_surface(views, arm=ARM_SPHERES, crease=CR.CreaseParams(), device="cuda")
    assert g.record["device"] == "cuda" and "protrusion" in g.record["ms_detail"]
    _check_select_record(g0, g)
    a, b = set(map(int, c.selected_nodes)), set(map(int, g.selected_nodes))
    assert len(a & b) / len(a | b) >= 0.99
    pc, pg = c.record["crease"]["protrusion"], g.record["crease"]["protrusion"]
    assert [x["attached"] for x in pc["components"]] == [x["attached"] for x in pg["components"]]
    assert pc["n_candidates"] == pg["n_candidates"]


# ============================================================================ 7 pipeline hand freeze


def _freeze_wired(monkeypatch, palm_offset, d_mm=80.0, **extra):
    A = pytest.importorskip("tests.ag3s.test_t78a_surface_post")
    esdf = {"surface_target_check": "overlap", "surface_keep_last": True, "surface_grasp_hold": True,
            "surface_crease_hand_freeze_mm": d_mm, **extra}
    fw, ag3s = A._wired(monkeypatch, target_exclusion="ball_owned", **ON, **esdf)
    state = {"palm": None}

    def palms(self, robot_state, context):
        return ([] if state["palm"] is None else [state["palm"]]), "test"

    monkeypatch.setattr(type(ag3s), "_grasp_palm_points", palms)
    obs, _, _ = A._rig(fw, gap=0.001)
    return A, fw, ag3s, obs, state


def test_freeze_reuses_the_last_accepted_region_when_the_palm_is_near(monkeypatch):
    A, fw, ag3s, obs, state = _freeze_wired(monkeypatch, None)
    ats = A._apple_points(fw)
    rec = A._update(ag3s, obs, ats_points=ats)                       # first request: no accepted region yet
    fz = ag3s._surface_frame["t78"]["freeze"]
    assert rec["segmentation"] and not fz["applied"] and fz["why"] == "no_accepted_region" and fz["latch"] == "searching"
    last = ag3s._t97_last_region
    assert last is not None and np.linalg.norm(last["centroid"][:2] - fw.APPLE[:2]) < 0.02
    n_up = ag3s._oracle_memory.n_updates
    # palm far → selects again
    state["palm"] = last["centroid"] + np.array([0.0, 0.0, 0.20])
    ag3s.frame_index += 1
    rec = A._update(ag3s, obs, ats_points=ats)
    fz = ag3s._surface_frame["t78"]["freeze"]
    assert rec["segmentation"] and not fz["applied"] and fz["why"] == "palm_far" and fz["palm_mm"] == pytest.approx(200)
    assert ag3s._oracle_memory.n_updates == n_up + 1
    # palm near (searching, then latched) → no selection, no fold; the reference region is not replaced
    for latched in (False, True):
        ag3s.set_grasp_latched(latched)
        state["palm"] = ag3s._t97_last_region["centroid"] + np.array([0.0, 0.0, 0.05])
        ag3s.frame_index += 1
        ref = dict(ag3s._t97_last_region)
        n_up = ag3s._oracle_memory.n_updates
        rec = A._update(ag3s, obs, ats_points=ats)
        surf = ag3s._surface_frame
        assert not rec["segmentation"] and rec["memory"]["frozen"] and rec["memory"]["carved"]
        assert surf["why"] == "crease_hand_freeze" and not surf["applied"] and surf["freeze"]["applied"]
        assert surf["freeze"]["palm_mm"] == pytest.approx(50) and surf["freeze"]["latch"] == (
            "latched" if latched else "searching")
        assert surf["freeze"]["region_age_frames"] >= 1 and ag3s._oracle_memory.n_updates == n_up
        assert ag3s._t97_last_region["frame_index"] == ref["frame_index"]
        json.dumps(surf, default=float)
    # the hand layer then uses the memory of the last accepted region (keep_last)
    field, _ = fw._build(ag3s)
    orc = field.stats["target_free"]["oracle"]
    assert orc["region_source"] == "last_accepted" and orc["n_target_samples"] > 0
    ag3s.reset()
    assert ag3s._t97_last_region is None


def test_freeze_window_is_searching_latched_closing_only(monkeypatch):
    A, fw, ag3s, obs, state = _freeze_wired(monkeypatch, None, surface_grasp_hold=False)
    ats = A._apple_points(fw)
    A._update(ag3s, obs, ats_points=ats)
    state["palm"] = ag3s._t97_last_region["centroid"].copy()
    # held (grasp active, not latched): outside the window → selection runs
    ag3s.set_grasp_active(True)
    rec = A._update(ag3s, obs, ats_points=ats)
    fz = ag3s._surface_frame["t78"]["freeze"]
    assert rec["segmentation"] and fz["latch"] == "held" and fz["why"] == "outside_latch_window"
    # closing (active + latched) → frozen
    ag3s.set_grasp_latched(True)
    rec = A._update(ag3s, obs, ats_points=ats)
    assert not rec["segmentation"] and ag3s._surface_frame["freeze"]["latch"] == "closing"
    ag3s.set_grasp_active(False)
    ag3s.set_grasp_latched(False)
    ag3s.set_placed(True)                                              # placed: sticky → outside
    rec = A._update(ag3s, obs, ats_points=ats)
    assert rec["segmentation"] and ag3s._surface_frame["t78"]["freeze"]["latch"] == "placed"
    # no palm → selection runs
    ag3s.reset()
    A._update(ag3s, obs, ats_points=ats)
    state["palm"] = None
    rec = A._update(ag3s, obs, ats_points=ats)
    assert rec["segmentation"] and ag3s._surface_frame["t78"]["freeze"]["why"] == "no_palm"


def test_freeze_check_reads_the_request_inputs(monkeypatch):
    """The check reads this request's `(robot_state, contact context)` that `process` stores (`_t97_freeze_inputs`)."""
    A, fw, ag3s, obs, state = _freeze_wired(monkeypatch, None)
    seen = []

    def palms(self, robot_state, context):
        seen.append((None if robot_state is None else np.asarray(robot_state).tolist(), context is not None))
        return [], "test"

    monkeypatch.setattr(type(ag3s), "_grasp_palm_points", palms)
    A._update(ag3s, obs, ats_points=A._apple_points(fw))   # accepted → reference stored
    ag3s._t97_freeze_inputs = (np.array([0.1, 0.2]), object())
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    assert seen[-1] == ([0.1, 0.2], True)
    assert math.isfinite(ag3s._surface_frame["t78"]["freeze"]["region_age_frames"])
