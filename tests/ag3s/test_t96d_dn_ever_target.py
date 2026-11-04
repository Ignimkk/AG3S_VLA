"""T96d (D34) — the solid's `d_N` without the memory cells ever labelled target (`esdf.surface_solid_dn_ever_target`,
`--surface-solid-dn-ever-target`), default off. Off must be the T96 · T96b · T96c split call for call.

| # | property |
|---|---|
| 1 | config default off · round trip · refusals (needs `surface_solid_owned`, bool); CLI key / record meta only when given; start-up refusal |
| 2 | memory off = before (state · `last`, no new key, `evidence_dn` refuses); on = the same state + the union of the code-1 cells after every update (never carved, cleared by reset) |
| 3 | off: no T96d step runs (split · pipeline · tier), no T96d key; on with nothing dropped = off value for value |
| 4 | pure: an apple-like target whose earlier-frame samples a later frame relabels code 2 — off, τ protects the seeds next to them; on, τ no longer does (the solid path takes them); a real neighbour 8 mm away keeps its τ band and every seed; `d_o` (surface · band paths) unchanged |
| 5 | shadow (T96c) on too: the shadow path reads the same reduced `d_N`; its dropped count |
| 6 | tier (fake cuRobo): `n_dn_ever_target_dropped` in `oracle.solid` |
| 7 | pipeline: the memory tracks the cells only when on; the evidence carries `dn_other_points`; record `attention_surface.solid.dn_ever_target` |
| 8 | the Q4 server-arg line |
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence, SolidOwned, SolidShadow, oracle_visible_split
from AG3S.ag3s.runtime.oracle_segmentation import (CODE_NONE, CODE_OBJECT_BASE, CODE_STATIC, CODE_TARGET,
                                                   OracleMemory, OracleView)

ON_BASE = {"target_extent": "attention_surface", "surface_keep_last": True, "surface_no_ball": True,
           "surface_solid_owned": True}
TAU = 0.005


def _t96():
    return pytest.importorskip("tests.ag3s.test_t96_solid_owned")


def _t96c():
    return pytest.importorskip("tests.ag3s.test_t96c_solid_shadow")


def _t78a():
    return pytest.importorskip("tests.ag3s.test_t78a_surface_post")


def _t78e():
    return pytest.importorskip("tests.ag3s.test_t78e_memory_target_sticky")


# ============================================================================ 1 config · CLI


def test_config_default_round_trip_and_refusals():
    assert AG3SConfig().esdf.surface_solid_dn_ever_target is False
    cfg = AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_dn_ever_target": True}})
    assert cfg.esdf.surface_solid_dn_ever_target is True
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    with pytest.raises(AG3SConfigError, match="surface_solid_dn_ever_target"):
        AG3SConfig.from_dict({"esdf": {**{k: v for k, v in ON_BASE.items() if k != "surface_solid_owned"},
                                       "surface_solid_dn_ever_target": True}})
    for bad in (1, "true"):
        with pytest.raises(AG3SConfigError, match="surface_solid_dn_ever_target"):
            AG3SConfig.from_dict({"esdf": {**ON_BASE, "surface_solid_dn_ever_target": bad}})


def test_cli_key_and_meta_only_when_given_and_refusal(capsys):
    from AG3S.trajopt.serve_safe import ifg_cl1_record_meta, target_exclusion_section

    A = _t78a()
    P0M = ("--surface-target-check", "overlap", "--surface-keep-last", "--surface-grasp-hold", "--surface-no-ball")
    base = A._parse(*A.SURF, *P0M, "--surface-solid-owned")
    assert base.surface_solid_dn_ever_target is False
    sec0, meta0 = target_exclusion_section(base), ifg_cl1_record_meta(base)
    assert "surface_solid_dn_ever_target" not in set(sec0) | set(meta0)
    on = A._parse(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-dn-ever-target")
    want = {"surface_solid_dn_ever_target": True}
    assert {k: v for k, v in target_exclusion_section(on).items() if k not in sec0} == want
    assert {k: v for k, v in ifg_cl1_record_meta(on).items() if k not in meta0} == want
    assert AG3SConfig.from_dict({"esdf": target_exclusion_section(on)}).esdf.surface_solid_dn_ever_target is True
    assert A._reject(*A.SURF, *P0M, "--surface-solid-owned", "--surface-solid-dn-ever-target").surface_solid_dn_ever_target
    with pytest.raises(SystemExit):
        A._reject(*A.SURF, *P0M, "--surface-solid-dn-ever-target")
    assert "--surface-solid-dn-ever-target 는 --surface-solid-owned 와 함께만" in capsys.readouterr().err


# ============================================================================ 2 memory


def _state(mem):
    return mem.keys.copy(), mem.points.copy(), mem.codes.copy(), mem.frame.copy()


def test_memory_off_is_before_and_on_adds_only_the_ever_target_cells():
    E = _t78e()
    for seq in E._sequence():
        a, b, on = OracleMemory(), OracleMemory(track_ever_target=False), OracleMemory(track_ever_target=True)
        ever: set = set()
        for views in seq:
            sa, sb, so = a.update(views), b.update(views), on.update(views)
            assert sa == sb and "n_ever_target_cells" not in sa
            assert {k: v for k, v in so.items() if k != "n_ever_target_cells"} == sa
            for x, y, z in zip(_state(a), _state(b), _state(on)):
                np.testing.assert_array_equal(x, y)
                np.testing.assert_array_equal(x, z)
            ever |= set(int(k) for k in on.keys[on.codes == CODE_TARGET])
            assert set(on.ever_target_keys.tolist()) == ever and so["n_ever_target_cells"] == len(ever)
            assert np.all(np.diff(on.ever_target_keys) > 0)
            assert len(a.ever_target_keys) == 0
        with pytest.raises(RuntimeError, match="track_ever_target"):
            a.evidence_dn(CODE_TARGET)
    with pytest.raises(ValueError, match="track_ever_target"):
        OracleMemory(track_ever_target=1)


def test_memory_relabel_carve_and_reset():
    E = _t78e()
    mem = OracleMemory(track_ever_target=True)
    mem.update([E._view(E._halves(CODE_TARGET, CODE_OBJECT_BASE))])         # left target, right neighbour
    tgt1 = set(int(k) for k in mem.keys[mem.codes == CODE_TARGET])
    tp, op = mem.evidence(CODE_TARGET)
    dn, n = mem.evidence_dn(CODE_TARGET)
    assert n == 0 and len(tgt1) == 32
    np.testing.assert_array_equal(dn, op)
    # frame 2: the target's half is relabelled static (newest wins) → those samples leave d_N, labels as is
    mem.update([E._view(E._halves(CODE_STATIC, CODE_OBJECT_BASE), depth=E.Z - 0.0004)])
    assert not (mem.codes == CODE_TARGET).any()
    tp, op = mem.evidence(CODE_TARGET)
    dn, n = mem.evidence_dn(CODE_TARGET)
    assert len(tp) == 0 and len(op) == 64 and n == 32 and len(dn) == 32
    rel = np.isin(mem.keys, list(tgt1))
    np.testing.assert_array_equal(dn, mem.points[~rel])                      # the evidence order, minus those cells
    assert (mem.codes[rel] == CODE_STATIC).all() and (mem.codes[~rel] == CODE_OBJECT_BASE).all()
    # carving drops samples but not the cells' history; a sample seen there again later is still left out
    st = mem.update([E._view(np.zeros((8, 8), np.uint8), depth=E.Z + 0.1)])
    assert st["n_carved"] == 64 and len(mem) == 0 and st["n_ever_target_cells"] == 32
    mem.update([E._view(np.full((8, 8), CODE_OBJECT_BASE))])
    dn, n = mem.evidence_dn(CODE_TARGET)
    assert n == 32 and len(dn) == 32
    mem.reset()
    assert len(mem.ever_target_keys) == 0 and mem.track_ever_target


# ============================================================================ rig: one camera, a relabelling frame


Z_REL = 0.030   # the target's lower front (above the table by < 30 mm) is relabelled static in frame 2


def _scene(kind="cylinder"):
    """The T96 rig (5 mm grid, seeds on every face) + an episode memory built by `OracleMemory.update` from the T96c
    front-above camera: frame 1 labels the target code 1, the neighbour (8 mm behind) ≥ 3, the table 2; frame 2 (same
    depth) relabels the target's lower front code 2 (newest wins). R = the frame-2 region's nodes (the last accepted
    region = code 1 after frame 2), R⊥ on the table."""
    T96, C = _t96(), _t96c()
    r = T96._rig(kind)
    cam, lab = C._camera(kind)
    codes = np.full(lab.shape, CODE_NONE, np.uint8)
    codes[lab == 0], codes[lab == 1], codes[lab == 2] = CODE_TARGET, CODE_OBJECT_BASE, CODE_STATIC
    v, u = np.nonzero(lab == 0)
    z = cam.depth_m[v, u]
    K, T = cam.K, cam.T_base_cam
    p = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]) @ T[:3, :3].T + T[:3, 3]
    low = p[:, 2] < T96.TABLE + Z_REL
    codes2 = codes.copy()
    codes2[v[low], u[low]] = CODE_STATIC
    mem = OracleMemory(cell=0.0025, track_ever_target=True)
    view = lambda c: OracleView(depth_m=cam.depth_m, K=K, T_base_cam=T, labels=c, name="head")   # noqa: E731
    mem.update([view(codes)])
    mem.update([view(codes2)])
    tp, op = mem.evidence(CODE_TARGET)
    dn, n_drop = mem.evidence_dn(CODE_TARGET)
    assert n_drop > 50 and len(op) - len(dn) == n_drop
    nodes = np.unique(np.floor(tp / T96.VS).astype(np.int64), axis=0)
    key = np.floor(tp / T96.VS).astype(np.int64)
    R = np.array([tp[(key == k).all(1)].mean(0) for k in nodes])
    R_perp = R.copy()
    R_perp[:, 2] = T96.TABLE
    pts = np.vstack([R, R_perp])
    from AG3S.ag3s.runtime import attention_surface_post as P

    h, why = P.convex_hull(pts)
    assert h is not None, why
    sol = SolidOwned(protect_m=TAU, equations=h["equations"], lo=h["lo"], hi=h["hi"], target_points=pts)
    keep = ~np.isin(mem.keys, mem.ever_target_keys) & (mem.codes != CODE_TARGET)
    rel = mem.points[(mem.codes != CODE_TARGET) & ~keep]                    # the relabelled target samples
    ev = OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=6.0, z_floor=r["zf"], solid=sol)
    return dict(r=r, ev=ev, dn=dn, rel=rel, sol=sol, cam=cam, mem=mem)


def _near(pts, c):
    from scipy.spatial import cKDTree

    return cKDTree(pts).query(c)[0] if len(pts) else np.full(len(c), np.inf)


# ============================================================================ 3 off


def test_off_no_t96d_step_runs_and_on_with_nothing_dropped_is_off(monkeypatch):
    from AG3S.ag3s.fields import curobo_builder as cb

    s = _scene()
    r, ev = s["r"], s["ev"]
    ref = oracle_visible_split(r["c"], r["seed"], ev, _t96().VS)

    def boom(*a, **k):
        raise AssertionError("T96d step called with its flag off")

    for name in ("_t96d_crop", "_t96d_in_box"):
        monkeypatch.setattr(cb, name, boom)
    out = cb.oracle_visible_split(r["c"], r["seed"], ev, _t96().VS)
    assert "n_dn_ever_target_dropped" not in out["solid"]
    for k in ("removed", "sign", "surface", "band", "kept_other", "d_t", "d_o"):
        np.testing.assert_array_equal(out[k], ref[k])
    monkeypatch.undo()
    # on, but d_N's samples are every other sample → the same values, the count 0
    same = cb.oracle_visible_split(r["c"], r["seed"], dataclasses.replace(ev, dn_other_points=ev.other_points),
                                   _t96().VS)
    assert same["solid"]["n_dn_ever_target_dropped"] == 0
    for k in ("removed", "sign", "surface", "band", "kept_other", "d_t", "d_o"):
        np.testing.assert_array_equal(same[k], ref[k])
    for k in ("fill", "protected", "d_N", "d_T"):
        np.testing.assert_array_equal(same["solid"][k], ref["solid"][k])


def test_pipeline_off_builds_no_tracking_memory_and_no_field(monkeypatch):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    from AG3S.ag3s.runtime.oracle_segmentation import OracleMemory as OM

    monkeypatch.setattr(OM, "evidence_dn", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    A = _t78a()
    fw, ag3s = _t96()._wired_on(monkeypatch)
    obs, _, _ = A._rig(fw)
    assert A._update(ag3s, obs, ats_points=A._apple_points(fw))["segmentation"]
    field, _ = fw._build(ag3s)
    assert ag3s._oracle_memory.track_ever_target is False and len(ag3s._oracle_memory.ever_target_keys) == 0
    assert "n_ever_target_cells" not in ag3s._oracle_memory.last
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    assert ev.solid is not None and ev.dn_other_points is None
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert "dn_ever_target" not in s
    te = ConstraintRecordWriter._write_exclusion_trace(field, {})["target_exclusion"]
    assert "dn_ever_target" not in te["attention_surface"]["solid"]


# ============================================================================ 4 pure: the relabelled apple


@pytest.mark.parametrize("dn_plane", [False, True])
def test_on_tau_no_longer_protects_the_relabelled_target_but_keeps_the_neighbour(dn_plane):
    T96 = _t96()
    s = _scene()
    r, ev, dn, rel = s["r"], s["ev"], s["dn"], s["rel"]
    if dn_plane:   # with D28 too (the support samples leave d_N as well)
        ev = dataclasses.replace(ev, solid=dataclasses.replace(ev.solid, dn_plane=[0, 0, 1, T96.TABLE],
                                                               dn_threshold_m=0.008))
    c, seed, owner = r["c"], r["seed"], r["owner"]
    off = oracle_visible_split(c, seed, ev, T96.VS)
    on = oracle_visible_split(c, seed, dataclasses.replace(ev, dn_other_points=dn), T96.VS)
    so, sn = off["solid"], on["solid"]
    # labels and d_o are untouched: the surface · band paths compare the same samples
    np.testing.assert_array_equal(on["d_o"], off["d_o"])
    np.testing.assert_array_equal(on["d_t"], off["d_t"])
    assert not (off["band"] & ~on["band"]).any()                                  # fewer samples → d_N ≥, τ ⊆
    assert on["n_other_near"] == off["n_other_near"]
    # the rule re-derived: d_N ≤ τ ⇔ a kept (never-target, off-plane) sample within τ — near the solid
    kept = dn if not dn_plane else dn[np.abs(dn[:, 2] - T96.TABLE) > 0.008]
    near_s = np.all((c >= ev.solid.lo - 0.03) & (c <= ev.solid.hi + 0.03), axis=1)
    np.testing.assert_array_equal((sn["d_N"] <= TAU)[near_s], (_near(kept, c) <= TAU)[near_s])
    # the count: the relabelled samples within the T96 cut (here: all of them)
    assert sn["n_dn_ever_target_dropped"] == len(rel) > 0
    assert "n_dn_ever_target_dropped" not in so
    # A = target seeds above the guard protected only by the relabelled samples
    above = c[:, 2] >= r["zf"]
    rel_d = rel if not dn_plane else rel[np.abs(rel[:, 2] - T96.TABLE) > 0.008]   # D28 drops the low ones anyway
    A = seed & (owner == 0) & above & (_near(rel_d, c) <= TAU) & (_near(kept, c) > TAU)
    assert A.sum() >= 5
    assert (so["d_N"][A] <= TAU).all() and not off["removed"][A].any()             # off: τ protects them
    assert not (sn["d_N"][A] <= TAU).any() and not sn["protected"][A].any()      # on: τ no longer does
    assert on["removed"][A].sum() >= 3                                            # the solid path takes them
    # more removed overall, every newly removed one is a target seed (never the neighbour, never the table)
    new = on["removed"] & ~off["removed"]
    assert new.sum() >= 3 and (owner[new] == 0).all()
    assert not (off["removed"] & ~on["removed"]).any()
    # the neighbour (8 mm behind) keeps every seed and its τ band
    nb = seed & (owner == 1)
    assert not on["removed"][nb].any()
    nb_obs = s["mem"].points[s["mem"].codes == CODE_OBJECT_BASE]
    band = near_s & (_near(nb_obs, c) <= TAU)
    assert band.sum() > 0 and (sn["d_N"][band] <= TAU).all()
    np.testing.assert_array_equal(sn["protected"][nb], so["protected"][nb])
    # the table never goes
    assert not on["removed"][seed & (owner == 2)].any() and not on["removed"][~above].any()


# ============================================================================ 5 shadow too


def test_shadow_reads_the_same_d_n_and_counts_its_drops():
    from AG3S.ag3s.fields.curobo_builder import _t96d_in_box

    T96 = _t96()
    s = _scene()
    r, ev, dn, cam = s["r"], s["ev"], s["dn"], s["cam"]
    L = 0.08
    sol = dataclasses.replace(ev.solid, shadow=SolidShadow(length_m=L, cameras=(cam,), lo=ev.solid.lo - L,
                                                           hi=ev.solid.hi + L))
    evs = dataclasses.replace(ev, solid=sol)
    off = oracle_visible_split(r["c"], r["seed"], evs, T96.VS)
    on = oracle_visible_split(r["c"], r["seed"], dataclasses.replace(evs, dn_other_points=dn), T96.VS)
    sn = on["solid"]
    assert sn["n_shadow_dn_ever_target_dropped"] == (_t96d_in_box(ev.other_points, sol.shadow, TAU)
                                                     - _t96d_in_box(dn, sol.shadow, TAU)) > 0
    assert "n_shadow_dn_ever_target_dropped" not in off["solid"]
    # every shadow seed on is beyond τ of the kept samples; the protected ones are within τ of them
    c = r["c"]
    np.testing.assert_array_equal(_near(dn, c[sn["shadow"]]) > TAU, np.ones(sn["shadow"].sum(), bool))
    assert (_near(dn, c[sn["protected_shadow"]]) <= TAU).all()
    assert on["removed"].sum() >= off["removed"].sum()
    # T96d adds no neighbour seed (the never-observed neighbour face in the shadow is T96c's documented risk, off too)
    nb = r["seed"] & (r["owner"] == 1)
    assert not (on["removed"] & ~off["removed"])[nb].any()


# ============================================================================ 6 tier (fake cuRobo)


def test_tier_records_the_dropped_count():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    ev, sol, _ = _t96c()._t44_solid(T44)
    ev = dataclasses.replace(ev, solid=sol)
    op = ev.other_points
    drop = np.linalg.norm(op - T44.A, axis=1) < T44.R + 0.004                # pretend: samples on A's own surface
    if not drop.any():
        drop[: max(1, len(op) // 10)] = True
    b = T44._builder(target_exclusion="ball_owned")

    def tier(e):
        (t,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=None, oracle=e)
        return t

    base = tier(ev)
    assert not {k for k in base.extra["exclusion"]["oracle"]["solid"] if "ever_target" in k}
    on = tier(dataclasses.replace(ev, dn_other_points=op[~drop]))
    s = on.extra["exclusion"]["oracle"]["solid"]
    assert s["dn_ever_target"] is True and 0 < s["n_dn_ever_target_dropped"] <= int(drop.sum())
    assert CuroboFieldBuilder._exclusion_counts([on])["oracle_split"]["solid"] == s
    assert s["n_removed_total"] >= base.extra["exclusion"]["oracle"]["solid"]["n_removed_total"]


# ============================================================================ 7 pipeline · record


def test_pipeline_on_tracks_the_cells_and_carries_the_samples(monkeypatch):
    import json

    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    A = _t78a()
    fw, ag3s = _t96()._wired_on(monkeypatch, surface_solid_dn_ever_target=True)
    obs, _, _ = A._rig(fw)
    assert A._update(ag3s, obs, ats_points=A._apple_points(fw))["segmentation"]
    mem = ag3s._oracle_memory
    assert mem.track_ever_target and mem.last["n_ever_target_cells"] == len(mem.ever_target_keys) > 0
    field, _ = fw._build(ag3s)
    ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
    dn, n = mem.evidence_dn(CODE_TARGET)
    np.testing.assert_array_equal(ev.dn_other_points, dn)
    assert len(ev.other_points) - len(ev.dn_other_points) == n
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert s["dn_ever_target"] == {"applied": True, "n_ever_target_cells": len(mem.ever_target_keys),
                                   "n_other_samples": len(ev.other_points), "n_dn_ever_target_dropped_memory": n}
    te = json.loads(json.dumps(ConstraintRecordWriter._write_exclusion_trace(field, {})["target_exclusion"]))
    assert te["attention_surface"]["solid"]["dn_ever_target"]["applied"]
    # no accepted region: still carried (the τ protection reads d_N too)
    fw, ag3s = _t96()._wired_on(monkeypatch, surface_solid_dn_ever_target=True)
    obs, _, _ = A._rig(fw)
    for o in obs:
        o.attention_map = None
    A._update(ag3s, obs, ats_points=A._apple_points(fw))
    field, _ = fw._build(ag3s)
    s = field.stats["target_free"]["attention_surface"]["solid"]
    assert s["why"] == "no_accepted_region" and s["dn_ever_target"]["applied"]
    assert ag3s._esdf_builder.calls[-1]["oracle_evidence"].dn_other_points is not None


# ============================================================================ 8 the Q4 server-arg line


def test_q4_server_args_parse_refuse_check_and_config(monkeypatch):
    import pathlib
    import shlex

    run = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261104T050351+0900_054748e6a73c_34e19a21953d/"
                       "impl/T96")
    if not (run / "server_flags_q4_delta0.txt").exists():
        pytest.skip("Q4 server-arg 파일이 없다")
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))
    line = (run / "server_flags_q4_delta0.txt").read_text().strip()
    q3 = (run / "server_flags_q3_delta0.txt").read_text().strip()
    assert shlex.split(line) == shlex.split(q3) + ["--surface-solid-dn-ever-target"]
    p = serve_safe.build_parser()
    a = p.parse_args(["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"] + shlex.split(line))
    serve_safe.reject_bad_flag_combinations(p, a)
    c = serve_safe.trajopt_config_from_args(a)
    assert c.collision.palm_cover_margin == 0.005 and c.collision.palm_shell_margin is None
    sec = serve_safe.target_exclusion_section(a)
    assert sec["surface_solid_dn_ever_target"] is True and sec["surface_solid_shadow_mm"] == 80.0
    e = AG3SConfig.from_dict({"esdf": {**sec, "fine_voxel_size": 0.005}}).esdf
    assert e.surface_solid_dn_ever_target and e.surface_solid_shadow_mm == 80.0 and e.surface_solid_dn_support == "exclude"
    assert serve_safe.ifg_cl1_record_meta(a)["surface_solid_dn_ever_target"] is True
    for absent in ("--surface-hull-fill", "--surface-contact-allowance-mm", "--palm-shell", "--palm-shell-margin"):
        assert absent not in shlex.split(line)
