"""IFG-SPHERE stage 1 — why does each GT target voxel stay in the target-free layer? CPU only, read only.

  cd /mnt/dev/AG3S_VLA-development; OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src \
      .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib collect --run <RUN> --kit IFG-APPLE-SPLIT --arm SPL \
      --eps 211,320 --precheck <PRECHECK-A raw_v2> --out $AG3S_RUN_DIR/verify/IFG-SPHERE-ATTRIB/raw [--variants F1,CORE5]
  ... report --raw <dir> --out <dir>/summary.json

The categories are fixed in `AG3S/ag3s/docs/handoff/IFG-SPHERE.impl.md` section 0 (written before this script measured anything).

Per request (summary grasp.state latched / closing, ball_owned fine target-free layer with a removal trace):
  replay      : `target_free_replay.rebuild_target_free(z, s)` with the record's settings -> trace equality with the record and
                the value difference (fidelity). The split (`d_t`, `d_o`, `sign`) is `_ball_owned_split` on the same inputs.
  residual    : as IFG-AUDIT-LTE `free_residual_target_voxels`: fine voxels within +-6 cm of the GT target centre with target-free
                value F <= vs/2 and the GT target rule (d_target <= 5 mm and <= every other geom). GT = ifg_audit_lte.Scene.
  category    : request 'stale' (ball fit centre > 30 mm from the GT centre), else per voxel the first match of VCATS (impl.md).
  sub labels  : kept_nontarget / kept_ambiguous / interior_not_sign_set -> GT class of the nearest non-target evidence point (10 mm);
                removed_sign_left / interior_impure -> category and GT-target flag of the nearest remaining seed.
  rows        : IFG-PRECHECK-A raw rows with src == target on the target-free layer; the voxel a row touched = the most negative
                weighted trilinear corner when a corner is negative, else the nearest remaining seed (F == 0).
  variants    : the same residual / removal / row metrics on `rebuild_target_free` variants (offline counterfactuals, see VARIANTS).
"""
from __future__ import annotations

import argparse, ast, contextlib, datetime, glob, io, json, pathlib, sys, time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

SCRIPT = "AG3S/ag3s/experiments/studies/ifg_sphere_attrib.py"
VCATS = ("outside_ball_centre", "outside_ball_radius", "below_zmin", "kept_nontarget", "kept_ambiguous", "kept_unowned",
         "surface_protect", "removed_sign_left", "interior_not_sign_set", "interior_impure", "other")
CATS = ("stale",) + VCATS
EVCLS = ("target", "neighbour", "fruit_other", "table", "crate", "obstacle", "other", "none")
GTCLS = ("target", "neighbour", "fruit_other", "table", "crate", "obstacle", "other")
VOX = 0.005
STALE_MM = 30.0
PHASES = ("latched", "closing")


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


# ------------------------------------------------------------------------------------------------ variants
def _core_split(core_mm):
    """CORE<mm> — the fitted-sphere core: every voxel of the ball candidates within (fit radius - core_mm) of the fit centre is
    target (seeds removed, non-seeds in the sign set) whatever the evidence says. Off when the fit was not accepted."""
    def patch(split, centres, seed, ball, vs):
        prov = dict(ball.provenance or {})
        fit = prov.get("fit") or {}
        if prov.get("centre_method") != "surface_fit" or fit.get("radius_mm") is None:
            return split, {"applied": False}
        c = np.asarray(fit.get("centre_m", ball.centre), np.float64)
        lim = float(fit["radius_mm"]) / 1000.0 - core_mm / 1000.0
        core = np.linalg.norm(centres - c[None], axis=1) <= lim
        out = dict(split)
        out["removed"] = split["removed"] | (seed & core)
        out["sign"] = split["sign"] | core
        return out, {"applied": True, "limit_mm": round(lim * 1000, 2), "n_core": int(core.sum()),
                     "n_core_seed_newly_removed": int((seed & core & ~split["removed"]).sum())}
    return patch


#: name -> (rebuild kwargs, split patch or None)
VARIANTS = {
    "OFF": ({}, None),
    "F1": ({"nontarget_evidence": "surface"}, None),
    "F2": ({"owned_ambiguous": "remove"}, None),
    "F1F2": ({"nontarget_evidence": "surface", "owned_ambiguous": "remove"}, None),
}
for _mm in (0, 3, 5, 8):
    VARIANTS[f"CORE{_mm}"] = ({}, _core_split(float(_mm)))
    VARIANTS[f"F1CORE{_mm}"] = ({"nontarget_evidence": "surface"}, _core_split(float(_mm)))


def rebuild_variant(z, s, name):
    """`target_free_replay.rebuild_target_free` with an optional split patch (offline only; the serving code is unchanged).
    Returns (values grid, removed ijk, info)."""
    import AG3S.ag3s.fields.curobo_builder as CB
    from AG3S.trajopt.experiments import target_free_replay as TFR
    kw, patch = VARIANTS[name]
    if patch is None:
        rep = TFR.rebuild_target_free(z, s, **kw)
        return rep["values"], rep["removed_ijk"], {"counts": rep["counts"], "evidence": rep["evidence"].get("evidence")}
    orig = CB._ball_owned_split
    info = {}

    def patched(centres, is_seed, ball, owned, voxel_size):
        sp = orig(centres, is_seed, ball, owned, voxel_size)
        out, inf = patch(sp, np.asarray(centres, np.float64), np.asarray(is_seed, bool), ball, float(voxel_size))
        info.update(inf)
        return out
    CB._ball_owned_split = patched
    try:
        rep = TFR.rebuild_target_free(z, s, **kw)
    finally:
        CB._ball_owned_split = orig
    return rep["values"], rep["removed_ijk"], {"counts": rep["counts"], "core": info, "evidence": rep["evidence"].get("evidence")}


# ------------------------------------------------------------------------------------------------ geometry helpers
def lin(ijk, shape):
    ijk = np.asarray(ijk, np.int64).reshape(-1, 3)
    return np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), tuple(shape))


def member(ijk, ref_lin_sorted, shape):
    """bool: ijk in the sorted linear index set."""
    if not len(ref_lin_sorted) or not len(ijk):
        return np.zeros(len(ijk), bool)
    q = lin(ijk, shape)
    pos = np.clip(np.searchsorted(ref_lin_sorted, q), 0, len(ref_lin_sorted) - 1)
    return ref_lin_sorted[pos] == q


def trilinear_corners(p, origin, vs, shape):
    f = (np.asarray(p, np.float64) - origin) / vs
    n = np.asarray(shape) - 1
    fc = np.clip(f, 0, n); i0 = np.minimum(np.floor(fc).astype(np.int64), np.maximum(n - 1, 0)); t = fc - i0
    cs, ws = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cs.append(np.minimum(i0 + np.array([dx, dy, dz]), n))
                ws.append((t[0] if dx else 1 - t[0]) * (t[1] if dy else 1 - t[1]) * (t[2] if dz else 1 - t[2]))
    return np.array(cs), np.array(ws)


class GTLabeler:
    """GT for world points with ifg_audit_lte.Scene (collision geoms; boxes / planes exact, meshes via a 0.1 mm probe)."""

    def __init__(self, scene, cls):
        self.S, self.cls = scene, cls
        self.is_t = cls == 0

    def target_rule(self, W):
        """(dt, dnt, nearest non-target class, target_rule bool)."""
        W = np.asarray(W, np.float64).reshape(-1, 3)
        if not len(W):
            z = np.zeros(0)
            return z, z, np.zeros(0, int), np.zeros(0, bool)
        tg = self.S.cand[self.is_t]
        D, _ = self.S.query(W, force=tg)
        dt = D[:, self.is_t].min(1)
        Dnt = np.where(self.is_t[None], np.inf, D)
        a = np.argmin(Dnt, 1); dnt = Dnt[np.arange(len(W)), a]
        return dt, dnt, self.cls[a], (dt <= VOX) & (dt <= dnt)

    def point_class(self, W, tol=0.010):
        """nearest GT class within tol (index into EVCLS; 'none' = len-1)."""
        W = np.asarray(W, np.float64).reshape(-1, 3)
        if not len(W):
            return np.zeros(0, int)
        D, _ = self.S.query(W)
        a = np.argmin(D, 1); d = D[np.arange(len(W)), a]
        return np.where(d <= tol, self.cls[a], len(EVCLS) - 1)


# ------------------------------------------------------------------------------------------------ per request
def classify(ijk, ctx):
    """ijk (M,3) -> (cat index into VCATS, evidence-class sub label (-1 n/a), nearest-remaining-seed category (-1 n/a),
    nearest-remaining-seed ijk (M,3) (-1 n/a))."""
    ijk = np.asarray(ijk, np.int64).reshape(-1, 3)
    M = len(ijk)
    cat = np.full(M, VCATS.index("other")); ev = np.full(M, -1); sc = np.full(M, -1); sijk = np.full((M, 3), -1)
    if not M:
        return cat, ev, sc, sijk
    o, vs, shape = ctx["origin"], ctx["vs"], ctx["shape"]
    c = o[None] + ijk * vs
    ball = ctx["ball"]
    r_b = np.linalg.norm(c - ball.centre[None], axis=1); r_g = np.linalg.norm(c - ctx["gt_centre"][None], axis=1)
    out = r_b > ball.radius
    cat[out & (r_g <= ball.radius)] = VCATS.index("outside_ball_centre")
    cat[out & (r_g > ball.radius)] = VCATS.index("outside_ball_radius")
    low = ~out & (c[:, 2] < ball.z_min)
    cat[low] = VCATS.index("below_zmin")
    rest = np.flatnonzero(~out & ~low)
    if len(rest):
        q = lin(ijk[rest], shape)
        pos = np.clip(np.searchsorted(ctx["cand_lin_sorted"], q), 0, len(ctx["cand_lin_sorted"]) - 1)
        found = ctx["cand_lin_sorted"][pos] == q
        ci = ctx["cand_order"][pos]          # index into cand / split arrays
        main = ctx["main"][ijk[rest, 0], ijk[rest, 1], ijk[rest, 2]]
        F = ctx["F"][ijk[rest, 0], ijk[rest, 1], ijk[rest, 2]]
        seed = main == 0
        rec = ctx["rec"]
        for j, (ii, fnd, cidx, sd, fv) in enumerate(zip(rest, found, ci, seed, F)):
            if not fnd:
                continue
            v = ijk[ii][None]
            if sd:
                if member(v, rec["removed"], shape)[0]:
                    cat[ii] = VCATS.index("removed_sign_left")
                elif member(v, rec["protected"], shape)[0]:
                    cat[ii] = VCATS.index("surface_protect")
                elif member(v, rec["nontarget"], shape)[0]:
                    cat[ii] = VCATS.index("kept_nontarget")
                elif member(v, rec["ambiguous"], shape)[0]:
                    cat[ii] = VCATS.index("kept_ambiguous")
                elif member(v, rec["unowned"], shape)[0]:
                    cat[ii] = VCATS.index("kept_unowned")
            else:
                if fv < 0:
                    if not ctx["split"]["sign"][cidx]:
                        cat[ii] = VCATS.index("interior_not_sign_set")
                    elif abs(fv) <= 1.5 * vs:
                        cat[ii] = VCATS.index("interior_impure")
            if cat[ii] in (VCATS.index("kept_nontarget"), VCATS.index("kept_ambiguous"), VCATS.index("interior_not_sign_set")):
                ev[ii] = ctx["ev_label"](int(ctx["ev_idx"][cidx]))
    # nearest remaining seed for the sign categories
    need = np.flatnonzero(np.isin(cat, (VCATS.index("removed_sign_left"), VCATS.index("interior_impure"))))
    if len(need) and ctx["seed_tree"] is not None:
        _, j = ctx["seed_tree"].query(c[need], k=1)
        s_ijk = ctx["seed_ijk"][j]
        sc[need], _, _, _ = classify_seed_only(s_ijk, ctx)
        sijk[need] = s_ijk
    return cat, ev, sc, sijk


def classify_seed_only(ijk, ctx):
    """remaining seeds (F == 0) only: categories 1-7 (no recursion into sign categories)."""
    ctx2 = dict(ctx); ctx2["seed_tree"] = None
    return classify(ijk, ctx2)


def request_context(z, s, gt_centre, labeler, variant=None):
    """`variant` = None (the record) or {"F": replayed grid, "trace": {key: ijk}, "nontarget_evidence": "surface" | ...,
    "owned_ambiguous": ...} — the same classification on an offline variant (`attrib`)."""
    import AG3S.ag3s.fields.curobo_builder as CB
    from AG3S.trajopt.experiments import target_free_replay as TFR
    from scipy.spatial import cKDTree
    tf = TFR.target_free_stats(s)
    ball = TFR.recorded_ball(tf)
    main = np.asarray(z["esdf_distance"], np.float32)
    F = np.asarray(z["esdf_free_distance"], np.float32) if variant is None else variant["F"]
    o = np.asarray(z["esdf_origin"], np.float64); vs = float(z["esdf_voxel_size"]); shape = main.shape
    rec_owned = dict(tf.get("owned") or {})
    nt = TFR.nontarget_from_record(z, fine_voxel=vs, nontarget_evidence="proximity")
    ntp = np.asarray(nt["points"], np.float64).reshape(-1, 3)
    tp = np.asarray(z["target_points"], np.float64).reshape(-1, 3)
    if variant is not None and variant.get("nontarget_evidence") == "surface":
        drop, _ = CB.nontarget_surface_evidence(ntp, tp, ball, shell_m=2.0 * vs, tie_m=float(rec_owned.get("tie_voxels", 1.0)) * vs)
        ntp = ntp[~drop]
    cand = CB.CuroboFieldBuilder._ball_candidates(ball, o, vs, shape)
    centres = o[None] + cand * vs
    seed = (main == 0)[cand[:, 0], cand[:, 1], cand[:, 2]]
    ow = CB.OwnedEvidence(target_points=tp, nontarget_points=ntp, tie_voxels=float(rec_owned.get("tie_voxels", 1.0)),
                          max_dist_voxels=float(rec_owned.get("max_dist_voxels", 2.0)), unowned=str(rec_owned.get("unowned", "remove")),
                          ambiguous="keep" if variant is None else str(variant.get("owned_ambiguous", "keep")))
    split = CB._ball_owned_split(centres, seed, ball, ow, vs)
    # nearest evidence point per candidate (the point that made d_o) -- same filtering as _ball_owned_split
    r = float(ball.radius)
    t_reach = float(np.max(np.linalg.norm(tp - ball.centre[None], axis=1))) if len(tp) else 0.0
    reach = r + (r + t_reach + ow.tie_voxels * vs) + vs
    near = np.flatnonzero(np.linalg.norm(ntp - ball.centre[None], axis=1) <= reach) if len(ntp) else np.zeros(0, int)
    if len(near):
        dd, jj = cKDTree(ntp[near]).query(centres, k=1)
        ev_idx = near[jj]
        assert np.allclose(dd, split["d_o"], atol=1e-9)
    else:
        ev_idx = np.full(len(cand), -1)
    ev_cache = {}

    def ev_label(j):
        if j < 0:
            return EVCLS.index("none")
        if j not in ev_cache:
            ev_cache[j] = int(labeler.point_class(ntp[j][None])[0])
        return ev_cache[j]
    cl = lin(cand, shape); order = np.argsort(cl)
    recset = {}
    for key, f in (("removed", "target_free_removed_ijk"), ("nontarget", "target_free_nontarget_kept_ijk"),
                   ("ambiguous", "target_free_ambiguous_ijk"), ("unowned", "target_free_unowned_kept_ijk"),
                   ("protected", "target_free_protected_ijk")):
        if variant is None:
            a = np.asarray(z[f], np.int64).reshape(-1, 3) if f in z.files else np.zeros((0, 3), np.int64)
        else:
            a = np.asarray(variant["trace"].get(key, np.zeros((0, 3))), np.int64).reshape(-1, 3)
        recset[key] = np.sort(lin(a, shape)) if len(a) else np.zeros(0, np.int64)
    # remaining seeds near the ball (F == 0) for the nearest-seed sub label
    lo = np.clip(np.floor((ball.centre - r - 0.03 - o) / vs).astype(int), 0, np.array(shape) - 1)
    hi = np.clip(np.ceil((ball.centre + r + 0.03 - o) / vs).astype(int), 0, np.array(shape) - 1)
    sub = F[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    seed_ijk = np.argwhere(sub == 0) + lo
    seed_tree = cKDTree(o[None] + seed_ijk * vs) if len(seed_ijk) else None
    return {"ball": ball, "gt_centre": np.asarray(gt_centre, np.float64), "main": main, "F": F, "origin": o, "vs": vs, "shape": shape,
            "cand": cand, "cand_lin_sorted": cl[order], "cand_order": order, "split": split, "seed_cand": seed,
            "ev_idx": ev_idx, "ev_label": ev_label, "rec": recset, "seed_ijk": seed_ijk, "seed_tree": seed_tree,
            "n_nontarget": int(len(ntp)), "nontarget_source": nt.get("source"), "tf": tf}


def residual_box(z, ct, labeler, F=None):
    """LTE residual: candidates in the +-6 cm box -> (ijk, dt, dnt, nnt class, tgt rule), plus the box (lo, hi)."""
    Dg = np.asarray(z["esdf_distance"], np.float32); og = np.asarray(z["esdf_origin"], np.float64); vs = float(z["esdf_voxel_size"])
    F = np.asarray(z["esdf_free_distance"], np.float32) if F is None else F
    shp = np.array(Dg.shape)
    lo = np.clip(np.floor((ct - 0.06 - og) / vs).astype(int), 0, shp - 1); hi = np.clip(np.ceil((ct + 0.06 - og) / vs).astype(int), 0, shp - 1)
    if not np.all(hi > lo):
        return None
    sub = Dg[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    Fs = F[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    cand = np.argwhere((sub <= vs / 2) | (Fs <= vs / 2)) + lo
    dt, dnt, ncl, tgt = labeler.target_rule(og + cand * vs)
    return {"ijk": cand, "dt": dt, "dnt": dnt, "ncl": ncl, "tgt": tgt, "lo": lo, "hi": hi}


def removal_metrics(removed_ijk, ctx_o, vs, labeler, cache):
    """LTE label rule on removed seeds: counts by label (target<=5mm / target5-10mm / neighbour / table / ...)."""
    removed_ijk = np.asarray(removed_ijk, np.int64).reshape(-1, 3)
    if not len(removed_ijk):
        return {"n": 0}
    keys = [tuple(v) for v in removed_ijk.tolist()]
    miss = [i for i, k_ in enumerate(keys) if k_ not in cache]
    if miss:
        dt, dnt, ncl, _ = labeler.target_rule(ctx_o + removed_ijk[miss] * vs)
        for i, a, b, c_ in zip(miss, dt, dnt, ncl):
            cache[keys[i]] = (float(a), float(b), int(c_))
    arr = np.array([cache[k_] for k_ in keys])
    dt, dnt, ncl = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)
    ok5 = dt <= VOX
    rest = (~ok5) & (dt > 0.010)
    out = {"n": int(len(dt)), "target<=5mm": int(ok5.sum()), "target5-10mm": int(((~ok5) & (dt <= 0.010)).sum())}
    for ci, cn in enumerate(GTCLS):
        if ci:
            out[cn] = int((rest & (dnt <= 0.010) & (ncl == ci)).sum())
    out["no_surface<=10mm"] = int((rest & (dnt > 0.010)).sum())
    # direct: removed seeds that are a neighbour-fruit / table surface voxel (that geom within 5 mm and nearer than the target)
    out["neighbour_surface_seed"] = int(((dnt <= VOX) & (dnt < dt) & (ncl == 1)).sum())
    out["table_surface_seed"] = int(((dnt <= VOX) & (dnt < dt) & (ncl == 3)).sum())
    return out


def target_surface_window(z, scene, target_g, ct, dirs, is_t):
    """GT target surface points (48 dirs, > 15 mm from other geoms): outside the fine free grid, composed < 5 mm while F >= 5 mm."""
    from AG3S.ag3s.experiments.studies.ifg_audit_lte import surface_points
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    St, _, ok = surface_points(scene, target_g, ct, dirs)
    St = St[ok]
    D, _ = scene.query(St)
    far = np.where(is_t[None], np.inf, D).min(1) > 0.015
    St = St[far]
    if not len(St):
        return {"n": 0}
    field = build_field(z, outside_distance=1.0)
    comp = field.target_free_distance(St)
    tiers = np.array(field.answer_tier(St, target_free=True))
    F = np.asarray(z["esdf_free_distance"], np.float32); o = np.asarray(z["esdf_free_origin"], np.float64); vs = float(z["esdf_free_voxel_size"])
    f = (St - o) / vs; inside = np.all((f >= 0) & (f <= np.array(F.shape) - 1), 1)
    fv = np.empty(len(St))   # fine target-free trilinear value
    for i, p in enumerate(St):
        cs, ws = trilinear_corners(p, o, vs, F.shape)
        fv[i] = float(np.dot(ws, F[cs[:, 0], cs[:, 1], cs[:, 2]]))
    return {"n": int(len(St)), "outside_free_grid": int((~inside).sum()), "composed_lt5": int((comp < 0.005).sum()),
            "fine_lt5": int((fv < 0.005).sum()), "composed_lt5_fine_ge5": int(((comp < 0.005) & (fv >= 0.005)).sum()),
            "tiers_composed_lt5": dict(Counter(tiers[comp < 0.005].tolist()))}


def process_request(z, s, row, js, scene, dirs, variants, rows_here, rm_cache):
    from AG3S.ag3s.experiments.studies.ifg_audit_lte import classes_for
    from AG3S.trajopt.experiments import target_free_replay as TFR
    mj, m = scene.mj, scene.m
    gs = js["geom_sets"]
    gid = lambda n: (int(n[5:]) if n.startswith("geom#") else mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n))  # noqa: E731
    target_g = [gid(n) for n in gs["target"]]; fruit_g = [gid(n) for n in gs["fruit_other"]]; obst_g = [gid(n) for n in gs.get("obstacle", [])]
    scene.set_q(row["qpos"])
    fb = {}
    for g in fruit_g:
        fb.setdefault(scene.bname[g], []).append(g)
    gap = {b: min(mj.mj_geomDistance(m, scene.d, t, g, 1.0, scene.ft) for t in target_g for g in gg) for b, gg in fb.items()}
    nb = min(gap, key=gap.get); ngap = float(gap[nb])
    cls = classes_for(scene, set(target_g), set(fb[nb]), set(fruit_g), set(obst_g))
    lab = GTLabeler(scene, cls)
    ct, _ = scene.body_centre(target_g)
    tf = TFR.target_free_stats(s)
    ball = tf.get("ball") or {}
    fit = ball.get("fit") or {}
    fit_c = fit.get("centre_m")
    fit_err = None if fit_c is None else float(np.linalg.norm(np.asarray(fit_c) - ct)) * 1000
    stale = fit_err is not None and fit_err > STALE_MM
    man = lit(s.get("manipulated")) or {}
    req = {"phase": str((s.get("grasp") or {}).get("state")), "scene_type": "neighbour" if ngap <= 0.10 else "clear", "neighbour": nb,
           "neighbour_gap_mm": round(ngap * 1000, 1), "stale": bool(stale), "fit_centre_err_mm": None if fit_err is None else round(fit_err, 2),
           "ball_centre_err_mm": round(float(np.linalg.norm(np.asarray(ball["centre_m"]) - ct)) * 1000, 2),
           "ball_radius_mm": ball.get("radius_mm"), "fit_radius_mm": fit.get("radius_mm"), "centre_method": ball.get("centre_method"),
           "z_min_m": ball.get("z_min_m"), "gt_centre": [round(float(v), 5) for v in ct],
           "gt_centre_z_minus_zmin_mm": round((ct[2] - float(ball["z_min_m"])) * 1000, 1),
           "manip_state": man.get("state") if isinstance(man, dict) else None,
           "manip_centroid_err_mm": (round(float(np.linalg.norm(np.asarray(man["centroid"]) - ct)) * 1000, 2)
                                     if isinstance(man, dict) and man.get("centroid") is not None else None),
           "nontarget_source": (tf.get("nontarget") or {}).get("source"),
           "counts_record": {k: tf.get(k) for k in ("n_seeds_in_ball", "n_removed", "n_kept_nontarget", "n_ambiguous", "n_unowned_removed",
                                                      "n_target_owned", "n_sign_candidates", "n_sign_candidates_nonseed")}}
    # replay fidelity
    rep = TFR.rebuild_target_free(z, s)
    F = np.asarray(z["esdf_free_distance"], np.float32)
    dv = np.abs(rep["values"] - F)
    req["fidelity"] = {"trace_equal": TFR.recorded_split_equal(z, rep), "n_vox_gt_0p6mm": int((dv > 6e-4).sum()), "max_abs_mm": round(float(dv.max()) * 1000, 3)}
    ctx = request_context(z, s, ct, lab)
    req["n_nontarget_points"] = ctx["n_nontarget"]
    # residual
    box = residual_box(z, ct, lab)
    out_vox = None
    if box is not None:
        ijk = box["ijk"]; Fv = F[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        res = box["tgt"] & (Fv <= ctx["vs"] / 2)
        Dv = ctx["main"][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        ts = (np.abs(Dv) <= ctx["vs"] / 2) & box["tgt"]
        req["recall"] = None if ts.sum() == 0 else round(float(np.mean(Fv[ts] > ctx["vs"] / 2)), 4)
        req["n_residual"] = int(res.sum()); req["n_residual_negative"] = int((res & (Fv < -ctx["vs"] / 2)).sum())
        cat, ev, sc, sijk = classify(ijk[res], ctx)
        # GT-target flag of the nearest remaining seed (sign categories)
        sg = np.full(len(cat), -1)
        okk = sc >= 0
        if okk.any():
            u, inv = np.unique(sijk[okk], axis=0, return_inverse=True)
            _, _, _, tg = lab.target_rule(ctx["origin"] + u * ctx["vs"])
            sg[okk] = tg[inv.ravel()].astype(int)
        req["cat_counts"] = {VCATS[i]: int((cat == i).sum()) for i in range(len(VCATS))}
        req["cat_ev"] = {f"{VCATS[i]}|{EVCLS[e]}": int(((cat == i) & (ev == e)).sum()) for i in (3, 4, 8) for e in range(len(EVCLS))
                         if ((cat == i) & (ev == e)).any()}
        req["cat_seed"] = {f"{VCATS[i]}|{VCATS[j]}|{'tgt' if g else 'nontgt'}": int(((cat == i) & (sc == j) & (sg == g)).sum())
                           for i in (7, 9) for j in range(len(VCATS)) for g in (0, 1) if ((cat == i) & (sc == j) & (sg == g)).any()}
        # all kept seeds in the ball (not only residual): how many are GT target, by category and evidence class
        out_vox = {"ijk": ijk[res].astype(np.int16), "cat": cat.astype(np.int8), "ev": ev.astype(np.int8), "sc": sc.astype(np.int8),
                   "sg": sg.astype(np.int8), "dt": box["dt"][res].astype(np.float32)}
    else:
        req["n_residual"] = None
    # kept seeds overall
    for key, nm in (("target_free_nontarget_kept_ijk", "kept_nontarget"), ("target_free_ambiguous_ijk", "kept_ambiguous")):
        a = np.asarray(z[key], np.int64).reshape(-1, 3)
        if len(a):
            _, _, _, tg = lab.target_rule(ctx["origin"] + a * ctx["vs"])
            req[f"{nm}_all"] = {"n": int(len(a)), "gt_target": int(tg.sum())}
    req["removal_record"] = removal_metrics(z["target_free_removed_ijk"], ctx["origin"], ctx["vs"], lab, rm_cache)
    req["target_surface_window"] = target_surface_window(z, scene, target_g, ct, dirs, lab.is_t)
    # precheck rows
    rows_out = []
    for r in rows_here:
        p = np.asarray(r["p"], np.float64)
        cs, ws = trilinear_corners(p, ctx["origin"], ctx["vs"], ctx["shape"])
        Fc = F[cs[:, 0], cs[:, 1], cs[:, 2]]
        f_rec = float(np.dot(ws, Fc))
        if (Fc < 0).any():
            v = cs[int(np.argmin(ws * Fc))][None]; how = "negative_corner"
        elif ctx["seed_tree"] is not None:
            _, j = ctx["seed_tree"].query(p, k=1); v = ctx["seed_ijk"][j][None]; how = "nearest_seed"
        else:
            v = None; how = "none"
        rr = {k: r.get(k) for k in ("ep", "kit", "k", "state", "traj", "step", "q", "link", "h_mm", "d_mm", "r_mm", "pen", "tier", "binding")}
        rr.update(f_fine_mm=round(f_rec * 1000, 3), how=how)
        if v is not None:
            c_, e_, s_, sij = classify(v, ctx)
            _, _, ncl_, tg_ = lab.target_rule(ctx["origin"] + v * ctx["vs"])
            rr.update(voxel=v[0].tolist(), cat=VCATS[int(c_[0])], ev=None if e_[0] < 0 else EVCLS[int(e_[0])],
                      seed_cat=None if s_[0] < 0 else VCATS[int(s_[0])], gt_target=bool(tg_[0]),
                      voxel_F_mm=round(float(F[tuple(v[0])]) * 1000, 2), seed_dist_mm=round(float(np.linalg.norm(ctx["origin"] + v[0] * ctx["vs"] - p)) * 1000, 2))
            if s_[0] >= 0:
                _, _, _, tg2 = lab.target_rule(ctx["origin"] + sij * ctx["vs"]); rr["seed_gt_target"] = bool(tg2[0])
        rows_out.append(rr)
    # variants
    var_out = {}
    for vn in variants:
        vals, rem, info = rebuild_variant(z, s, vn)
        vr = {"counts": {k: info["counts"].get(k) for k in ("n_removed", "n_kept_nontarget", "n_ambiguous", "n_sign_candidates")},
              "core": info.get("core"), "f1": (info.get("evidence") or {}).get("n_dropped") if isinstance(info.get("evidence"), dict) else None}
        if box is not None:
            ijk = box["ijk"]; Fv = vals[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            res = box["tgt"] & (Fv <= ctx["vs"] / 2)
            Dv = ctx["main"][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            ts = (np.abs(Dv) <= ctx["vs"] / 2) & box["tgt"]
            vr["n_residual"] = int(res.sum()); vr["recall"] = None if ts.sum() == 0 else round(float(np.mean(Fv[ts] > ctx["vs"] / 2)), 4)
            # non-target surface voxels lost (box): GT non-target surface (dnt <= 2.5 mm, nearer than the target), main seed, free seed in
            # the record but no longer within vs/2 in the variant
            nts = (box["dnt"] <= ctx["vs"] / 2) & (box["dnt"] < box["dt"]) & (Dv == 0)
            Frec = F[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            for ci, cn in ((1, "neighbour"), (3, "table")):
                sel = nts & (box["ncl"] == ci)
                vr[f"{cn}_surface_vox"] = int(sel.sum())
                vr[f"{cn}_surface_kept_record"] = int((sel & (Frec <= ctx["vs"] / 2)).sum())
                vr[f"{cn}_surface_kept_variant"] = int((sel & (Fv <= ctx["vs"] / 2)).sum())
        vr["removal"] = removal_metrics(rem, ctx["origin"], ctx["vs"], lab, rm_cache)
        if rows_here:
            hs = []
            for r in rows_here:
                p = np.asarray(r["p"], np.float64)
                cs, ws = trilinear_corners(p, ctx["origin"], ctx["vs"], ctx["shape"])
                df = float(np.dot(ws, vals[cs[:, 0], cs[:, 1], cs[:, 2]]) - np.dot(ws, F[cs[:, 0], cs[:, 1], cs[:, 2]]))
                hs.append(round(float(r["h_mm"]) + df * 1000, 3))
            vr["row_h_mm"] = hs
        var_out[vn] = vr
    req["variants"] = var_out
    return req, out_vox, rows_out


# ------------------------------------------------------------------------------------------------ collect
def collect(a):
    from AG3S.ag3s.experiments.studies.ifg_audit_lte import Scene, fib_dirs
    K = a.run / "verify" / a.kit
    xml = a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
    scene = Scene(xml); dirs = fib_dirs(48)
    a.out.mkdir(parents=True, exist_ok=True)
    jfs = sorted(f for f in glob.glob(str(K / "replay" / f"runs_{a.arm}_{a.arm}_ep*.json")))
    eps = None if a.eps in ("", "all") else {int(e) for e in a.eps.split(",")}
    variants = [v for v in a.variants.split(",") if v]
    for jf in jfs:
        js = json.load(open(jf)); ep = int(js["episode"])
        if eps is not None and ep not in eps:
            continue
        outp = a.out / f"{a.kit}_ep{ep}.json"
        if outp.exists() and not a.force:
            print("skip", outp); continue
        b = js["server_chunk_alignment"]["block"]; rd = pathlib.Path(js["run_dir"])
        pl = [json.loads(x) for x in open(rd / "frames.jsonl")]
        pl = [r for r in pl if r.get("kind") == "planning"]
        rows_all = []
        if a.precheck is not None:
            pf = a.precheck / f"{a.kit}_ep{ep}.json"
            if pf.exists():
                rows_all = [r for r in json.load(open(pf))["rows"] if r.get("src") == "target" and r.get("layer") == "target_free"]
        rows_by_k = defaultdict(list)
        for r in rows_all:
            rows_by_k[int(r["k"])].append(r)
        reqs, vox, rows_out = [], {}, []
        t0 = time.time(); rm_cache = {}
        for k in range(b["block_len"]):
            p = pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz"
            with np.load(p, allow_pickle=True) as z:
                s = json.loads(str(z["summary_json"]))
                st = str((s.get("grasp") or {}).get("state"))
                if st not in PHASES:
                    continue
                if "target_free_removed_ijk" not in z.files or "esdf_free_distance" not in z.files:
                    reqs.append({"k": k, "phase": st, "skipped": "no target-free trace"}); continue
                from AG3S.trajopt.experiments import target_free_replay as TFR
                tf = TFR.target_free_stats(s)
                if "target_points" not in z.files:
                    # the server built the ball from its kept geometry but the record has no target points (has_target False):
                    # the split cannot be replayed (8 requests: ep1313 k18, ep320 k22/53-55/63-65)
                    reqs.append({"k": k, "phase": st, "skipped": f"no target_points recorded (has_target {s.get('has_target')})"}); continue
                if tf.get("exclusion_method") != "ball_owned" or not len(z["target_free_removed_ijk"]):
                    reqs.append({"k": k, "phase": st, "skipped": f"method {tf.get('exclusion_method')} n_removed {len(z['target_free_removed_ijk'])}"}); continue
                with contextlib.redirect_stdout(io.StringIO()):
                    req, ov, ro = process_request(z, s, pl[k], js, scene, dirs, variants, rows_by_k.get(k, []), rm_cache)
                req.update(ep=ep, k=k, kit=a.kit)
                reqs.append(req); rows_out += ro
                if ov is not None:
                    for kk, vv in ov.items():
                        vox[f"k{k}_{kk}"] = vv
        meta = {"ep": ep, "kit": a.kit, "arm": a.arm, "run_dir": str(rd), "chunk_dir": b["chunk_dir"], "offset": b["offset"],
                "target_fruit": js.get("target_fruit"), "seconds": round(time.time() - t0, 1), "variants": variants,
                "n_rows_precheck_target": len(rows_all)}
        outp.write_text(json.dumps({"meta": meta, "requests": reqs, "rows": rows_out}, default=float))
        np.savez_compressed(a.out / f"{a.kit}_ep{ep}_voxels.npz", **vox)
        print(f"ep{ep}: {sum(1 for r in reqs if 'skipped' not in r)} requests, {len(rows_out)} rows, {time.time() - t0:.0f}s", flush=True)


def support_sign(vals, ctx):
    """P2 (offline): `(values, flipped ijk)`. Sign candidates (`split["sign"]`) with value < 0 and |value| <= 1.5 vs (left alone by
    `_force_sign_on_pure_voxels`) whose nearest remaining seed (F == 0) is below the ball's `z_min` get |value|."""
    vals2 = vals.copy()
    cand, sign, vs = ctx["cand"], ctx["split"]["sign"], ctx["vs"]
    v = vals[cand[:, 0], cand[:, 1], cand[:, 2]]
    sel = sign & (v < 0) & (np.abs(v) <= np.float32(1.5 * vs))
    if not sel.any() or ctx["seed_tree"] is None:
        return vals2, np.zeros((0, 3), np.int64)
    c = ctx["origin"][None] + cand[sel] * vs
    _, j = ctx["seed_tree"].query(c, k=1)
    sz = ctx["origin"][2] + ctx["seed_ijk"][j][:, 2] * vs
    f = cand[sel][sz < float(ctx["ball"].z_min)]
    vals2[f[:, 0], f[:, 1], f[:, 2]] = np.abs(vals2[f[:, 0], f[:, 1], f[:, 2]])
    return vals2, f


def attrib_variant(a):
    """The same residual categories on an offline variant's layer (what the variant leaves behind)."""
    from AG3S.ag3s.experiments.studies.ifg_audit_lte import Scene, classes_for
    from AG3S.trajopt.experiments import target_free_replay as TFR
    K = a.run / "verify" / a.kit
    scene = Scene(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    mj, m = scene.mj, scene.m
    eps = None if a.eps in ("", "all") else {int(e) for e in a.eps.split(",")}
    kw, patch = VARIANTS[a.variant]
    assert patch is None, "attrib supports the rebuild-only variants (F1, F2, F1F2)"
    for jf in sorted(glob.glob(str(K / "replay" / f"runs_{a.arm}_{a.arm}_ep*.json"))):
        js = json.load(open(jf)); ep = int(js["episode"])
        if eps is not None and ep not in eps:
            continue
        outp = a.out / f"{a.kit}_ep{ep}_attrib_{a.variant}.json"
        if outp.exists() and not a.force:
            print("skip", outp); continue
        base = json.load(open(a.out / f"{a.kit}_ep{ep}.json"))
        breq = {r["k"]: r for r in base["requests"] if "skipped" not in r}
        pre_by_k = defaultdict(list)   # the same rows (same order) as the base pass
        pf = None if a.precheck is None else a.precheck / f"{a.kit}_ep{ep}.json"
        if pf is not None and pf.exists():
            for r in json.load(open(pf))["rows"]:
                if r.get("src") == "target" and r.get("layer") == "target_free":
                    pre_by_k[int(r["k"])].append(r)
        b = js["server_chunk_alignment"]["block"]; rd = pathlib.Path(js["run_dir"])
        pl = [r for r in (json.loads(x) for x in open(rd / "frames.jsonl")) if r.get("kind") == "planning"]
        gs = js["geom_sets"]
        gid = lambda n: (int(n[5:]) if n.startswith("geom#") else mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n))  # noqa: E731
        target_g = [gid(n) for n in gs["target"]]; fruit_g = [gid(n) for n in gs["fruit_other"]]; obst_g = [gid(n) for n in gs.get("obstacle", [])]
        out, t0 = [], time.time()
        for k, br in sorted(breq.items()):
            with np.load(pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz", allow_pickle=True) as z:
                s = json.loads(str(z["summary_json"]))
                scene.set_q(pl[k]["qpos"])
                fb = {}
                for g in fruit_g:
                    fb.setdefault(scene.bname[g], []).append(g)
                cls = classes_for(scene, set(target_g), set(fb[br["neighbour"]]), set(fruit_g), set(obst_g))
                lab = GTLabeler(scene, cls)
                ct = np.asarray(br["gt_centre"], np.float64)
                rep = TFR.rebuild_target_free(z, s, **kw)
                vals = rep["values"]
                var = {"F": vals, "nontarget_evidence": kw.get("nontarget_evidence", "proximity"),
                       "owned_ambiguous": kw.get("owned_ambiguous", "keep"),
                       "trace": {"removed": rep["removed_ijk"], "nontarget": rep["nontarget_kept_ijk"], "ambiguous": rep["ambiguous_ijk"]}}
                with contextlib.redirect_stdout(io.StringIO()):
                    ctx = request_context(z, s, ct, lab, variant=var)
                box = residual_box(z, ct, lab, F=vals)
                rq = {"ep": ep, "k": k, "kit": a.kit, "phase": br["phase"], "scene_type": br["scene_type"], "stale": br["stale"]}
                if box is not None:
                    ijk = box["ijk"]; Fv = vals[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    res = box["tgt"] & (Fv <= ctx["vs"] / 2)
                    cat, ev, sc, sijk = classify(ijk[res], ctx)
                    rq["n_residual"] = int(res.sum())
                    rq["cat_counts"] = {VCATS[i]: int((cat == i).sum()) for i in range(len(VCATS))}
                    rq["cat_ev"] = {f"{VCATS[i]}|{EVCLS[e]}": int(((cat == i) & (ev == e)).sum()) for i in (3, 4, 8) for e in range(len(EVCLS))
                                    if ((cat == i) & (ev == e)).any()}
                    rq["cat_seed"] = {f"{VCATS[i]}|{VCATS[j]}": int(((cat == i) & (sc == j)).sum()) for i in (7, 9) for j in range(len(VCATS))
                                      if ((cat == i) & (sc == j)).any()}
                else:
                    rq["n_residual"] = None
                rq["rows"] = []
                prow = pre_by_k.get(k, [])
                F0 = np.asarray(z["esdf_free_distance"], np.float32)
                # P2 counterfactual "support-aware sign": sign-candidate voxels left negative by the purity rule only because the
                # nearest remaining seed lies below the ball's z_min (support plane + pad) get the positive sign. No seed is removed.
                vals2, flip = support_sign(vals, ctx)
                ss = {"n_flipped": int(len(flip))}
                if len(flip):
                    _, dnt_f, ncl_f, _ = lab.target_rule(ctx["origin"] + flip * ctx["vs"])
                    ss["n_flipped_inside_nontarget"] = int((dnt_f < 0).sum())
                    ss["n_flipped_inside_nontarget_by_class"] = dict(Counter(GTCLS[int(c_)] for c_ in ncl_f[dnt_f < 0]))
                if box is not None:
                    ijk = box["ijk"]; Fv2 = vals2[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    ss["n_residual"] = int((box["tgt"] & (Fv2 <= ctx["vs"] / 2)).sum())
                ss["rows_h_mm"] = []
                for r in prow:
                    cs, ws = trilinear_corners(np.asarray(r["p"], np.float64), ctx["origin"], ctx["vs"], ctx["shape"])
                    ss["rows_h_mm"].append(round(float(r["h_mm"]) + float(np.dot(ws, vals2[cs[:, 0], cs[:, 1], cs[:, 2]])
                                                                            - np.dot(ws, F0[cs[:, 0], cs[:, 1], cs[:, 2]])) * 1000, 3))
                rq["support_sign"] = ss
                for r in prow:
                    p = np.asarray(r["p"], np.float64)
                    cs, ws = trilinear_corners(p, ctx["origin"], ctx["vs"], ctx["shape"])
                    Fc = vals[cs[:, 0], cs[:, 1], cs[:, 2]]
                    h = float(r["h_mm"]) + float(np.dot(ws, Fc) - np.dot(ws, F0[cs[:, 0], cs[:, 1], cs[:, 2]])) * 1000
                    o_ = {"traj": r["traj"], "h_mm": r["h_mm"], "h_variant_mm": round(h, 3)}
                    if h < 0 or (r["traj"] == "out" and h <= 1.0):
                        if (Fc < 0).any():
                            v = cs[int(np.argmin(ws * Fc))][None]
                        else:
                            _, j = ctx["seed_tree"].query(p, k=1); v = ctx["seed_ijk"][j][None]
                        c_, e_, s_, _ = classify(v, ctx)
                        o_.update(cat=VCATS[int(c_[0])], ev=None if e_[0] < 0 else EVCLS[int(e_[0])], seed_cat=None if s_[0] < 0 else VCATS[int(s_[0])])
                    rq["rows"].append(o_)
                out.append(rq)
        outp.write_text(json.dumps({"variant": a.variant, "requests": out}, default=float))
        print(f"ep{ep}: {len(out)} requests ({a.variant}), {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ report
def _med(v):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.median(v)), 2)


def report(a):
    files = sorted(f for f in glob.glob(str(a.raw / "*_ep*.json")) if "_attrib_" not in f)
    reqs, rows = [], []
    for f in files:
        d = json.load(open(f))
        reqs += [r for r in d["requests"] if "skipped" not in r]
        rows += d["rows"]
    out = {"n_requests": len(reqs), "n_episodes": len({(r["kit"], r["ep"]) for r in reqs})}
    split = [r for r in reqs if r["kit"] == "IFG-APPLE-SPLIT"]
    # fidelity
    out["fidelity"] = {"trace_equal_all": sum(1 for r in split if all(v for v in r["fidelity"]["trace_equal"].values() if v is not None)),
                       "n": len(split), "n_vox_gt_0p6mm_total": sum(r["fidelity"]["n_vox_gt_0p6mm"] for r in split),
                       "max_abs_mm": max(r["fidelity"]["max_abs_mm"] for r in split)}

    def table(rr):
        tot = Counter(); stale_inner = Counter(); n_stale = 0
        for r in rr:
            if r.get("n_residual") is None:
                continue
            if r["stale"]:
                tot["stale"] += r["n_residual"]; n_stale += 1
                for k_, v in r["cat_counts"].items():
                    stale_inner[k_] += v
            else:
                for k_, v in r["cat_counts"].items():
                    tot[k_] += v
        n = sum(tot.values())
        res = [r["n_residual"] for r in rr if r.get("n_residual") is not None]
        res_ns = [r["n_residual"] for r in rr if r.get("n_residual") is not None and not r["stale"]]
        return {"n_requests": len(rr), "n_stale": n_stale, "residual_total": n, "residual_median": _med(res), "residual_median_nonstale": _med(res_ns),
                "recall_median_nonstale": _med([r.get("recall") for r in rr if not r["stale"]]),
                "by_category": {c: tot.get(c, 0) for c in CATS}, "frac": {c: (round(tot.get(c, 0) / n, 4) if n else None) for c in CATS},
                "stale_inner": dict(stale_inner),
                "per_request_median_nonstale": {c: _med([r["cat_counts"].get(c, 0) for r in rr if not r["stale"] and r.get("n_residual") is not None]) for c in VCATS}}
    out["by_phase"] = {p: table([r for r in split if r["phase"] == p]) for p in PHASES}
    out["by_phase_scene"] = {f"{p}|{st}": table([r for r in split if r["phase"] == p and r["scene_type"] == st]) for p in PHASES for st in ("neighbour", "clear")}
    out["all_pre_grasp"] = table(split)
    ev = Counter(); seedc = Counter()
    for r in split:
        if r["stale"] or r.get("n_residual") is None:
            continue
        for k_, v in r["cat_ev"].items():
            ev[k_] += v
        for k_, v in r["cat_seed"].items():
            seedc[k_] += v
    out["evidence_sub"] = dict(sorted(ev.items(), key=lambda x: -x[1]))
    out["nearest_seed_sub"] = dict(sorted(seedc.items(), key=lambda x: -x[1]))
    kept = {nm: {"n": sum(r.get(f"{nm}_all", {}).get("n", 0) for r in split if not r["stale"]),
                 "gt_target": sum(r.get(f"{nm}_all", {}).get("gt_target", 0) for r in split if not r["stale"])} for nm in ("kept_nontarget", "kept_ambiguous")}
    out["kept_seeds_all_nonstale"] = kept
    rm = Counter()
    for r in split:
        if r["stale"]:
            continue
        for k_, v in r["removal_record"].items():
            rm[k_] += v
    out["removal_record_nonstale"] = dict(rm)
    tsw = Counter(); tiers = Counter()
    for r in split:
        if r["stale"]:
            continue
        w = r["target_surface_window"]
        for k_ in ("n", "outside_free_grid", "composed_lt5", "fine_lt5", "composed_lt5_fine_ge5"):
            tsw[k_] += w.get(k_, 0)
        for k_, v in (w.get("tiers_composed_lt5") or {}).items():
            tiers[k_] += v
    out["target_surface_window_nonstale"] = dict(tsw) | {"tiers_composed_lt5": dict(tiers)}
    # geometry
    out["geometry_nonstale"] = {"ball_radius_mm": _med([r["ball_radius_mm"] for r in split if not r["stale"]]),
                                "fit_radius_mm": _med([r["fit_radius_mm"] for r in split if not r["stale"]]),
                                "fit_centre_err_mm": _med([r["fit_centre_err_mm"] for r in split if not r["stale"]]),
                                "manip_centroid_err_mm": _med([r["manip_centroid_err_mm"] for r in split if not r["stale"]]),
                                "gt_centre_z_minus_zmin_mm": _med([r["gt_centre_z_minus_zmin_mm"] for r in split if not r["stale"]]),
                                "centre_method": dict(Counter(r["centre_method"] for r in split)),
                                "nontarget_source": dict(Counter(r["nontarget_source"] for r in split))}
    out["stale_requests"] = sorted({(r["ep"], r["phase"]) for r in split if r["stale"]})
    out["stale_by_ep"] = dict(Counter(r["ep"] for r in split if r["stale"]))
    out["stale_manip_state"] = dict(Counter(str(r.get("manip_state")) for r in split if r["stale"]))
    # rows
    def row_table(rs):
        c = Counter(r.get("cat") for r in rs); g = Counter(bool(r.get("gt_target")) for r in rs)
        sub = Counter(f"{r.get('cat')}|{r.get('ev') or r.get('seed_cat')}" for r in rs)
        return {"n": len(rs), "by_category": dict(c), "gt_target": g.get(True, 0), "sub": dict(sub.most_common(12)),
                "how": dict(Counter(r.get("how") for r in rs)),
                "fidelity_abs_f_minus_d_mm_median": _med([abs(r["f_fine_mm"] - r["d_mm"]) for r in rs if r.get("tier") == "target_free"])}
    out["rows"] = {}
    for kit in sorted({r["kit"] for r in rows}):
        rk = [r for r in rows if r["kit"] == kit]
        for traj in ("ref", "out"):
            out["rows"][f"{kit}|{traj}|all"] = row_table([r for r in rk if r["traj"] == traj])
            for ep in sorted({r["ep"] for r in rk}):
                re_ = [r for r in rk if r["traj"] == traj and r["ep"] == ep]
                if re_:
                    out["rows"][f"{kit}|{traj}|ep{ep}"] = row_table(re_)
        for ep, ks in ((211, range(11, 15)),):
            for traj in ("ref", "out"):
                for k in ks:
                    re_ = [r for r in rk if r["traj"] == traj and r["ep"] == ep and r["k"] == k]
                    if re_:
                        out["rows"][f"{kit}|{traj}|ep{ep}|k{k}"] = row_table(re_)
    # variants
    vn = sorted({v for r in split for v in r.get("variants", {})})
    vo = {}
    for v in vn:
        d = {}
        for p in PHASES:
            rr = [r for r in split if r["phase"] == p and not r["stale"] and v in r.get("variants", {})]
            d[p] = {"n": len(rr), "residual_median": _med([r["variants"][v].get("n_residual") for r in rr]),
                    "recall_median": _med([r["variants"][v].get("recall") for r in rr]),
                    "residual_total": sum(r["variants"][v].get("n_residual") or 0 for r in rr)}
            for cn in ("neighbour", "table"):
                d[p][f"{cn}_surface_vox"] = sum(r["variants"][v].get(f"{cn}_surface_vox", 0) for r in rr)
                d[p][f"{cn}_surface_kept_record"] = sum(r["variants"][v].get(f"{cn}_surface_kept_record", 0) for r in rr)
                d[p][f"{cn}_surface_kept_variant"] = sum(r["variants"][v].get(f"{cn}_surface_kept_variant", 0) for r in rr)
            rem = Counter()
            for r in rr:
                for k_, x in r["variants"][v]["removal"].items():
                    rem[k_] += x
            d[p]["removal"] = dict(rem)
            d[p]["removal_frac_neighbour"] = round(rem["neighbour"] / rem["n"], 4) if rem["n"] else None
            d[p]["removal_frac_table"] = round(rem["table"] / rem["n"], 4) if rem["n"] else None
        vo[v] = d
    out["variants"] = vo
    # variant effect on precheck target rows (active_ref h < 0); rows align with variants[...]["row_h_mm"] per request
    out["variant_rows"] = variant_rows(a.raw, vn)
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


def report_attrib(a):
    """Aggregate `attrib` outputs: what a variant leaves (categories) and the P2 support-aware sign on top of it."""
    out = {}
    for v in a.variants.split(","):
        reqs = []
        for f in sorted(glob.glob(str(a.raw / f"IFG-APPLE-SPLIT_ep*_attrib_{v}.json"))):
            reqs += json.load(open(f))["requests"]
        d = {"n_requests": len(reqs)}
        for p in PHASES:
            rr = [r for r in reqs if r["phase"] == p and not r["stale"] and r.get("n_residual") is not None]
            tot, ev, sd = Counter(), Counter(), Counter()
            for r in rr:
                tot.update(r["cat_counts"]); ev.update(r["cat_ev"]); sd.update(r["cat_seed"])
            n = sum(tot.values())
            d[p] = {"n_requests": len(rr), "residual_total": n, "residual_median": _med([r["n_residual"] for r in rr]),
                    "frac": {c: round(tot[c] / n, 4) for c in VCATS if tot[c]} if n else {},
                    "per_request_median": {c: _med([r["cat_counts"].get(c, 0) for r in rr]) for c in VCATS},
                    "evidence_sub": dict(ev.most_common()), "nearest_seed_sub": dict(sd.most_common()),
                    "support_sign": {"residual_median": _med([r["support_sign"].get("n_residual") for r in rr]),
                                     "residual_total": sum(r["support_sign"].get("n_residual") or 0 for r in rr),
                                     "n_flipped_total": sum(r["support_sign"]["n_flipped"] for r in rr),
                                     "n_flipped_inside_nontarget": sum(r["support_sign"].get("n_flipped_inside_nontarget", 0) for r in rr),
                                     "flipped_inside_by_class": dict(sum((Counter(r["support_sign"].get("n_flipped_inside_nontarget_by_class", {})) for r in rr), Counter()))}}
        rows = {}
        for r in reqs:
            for i, x in enumerate(r["rows"]):
                key = f"ep{r['ep']}|{x['traj']}"
                o = rows.setdefault(key, {"n": 0, "before": 0, "after": 0, "after_plus_P2": 0, "remaining_by_cat": Counter(), "by_k_after": Counter()})
                hv, h2 = x["h_variant_mm"], r["support_sign"]["rows_h_mm"][i]
                act = (lambda h: h < 0) if x["traj"] == "ref" else (lambda h: h <= 1.0)
                o["n"] += 1; o["before"] += int(act(x["h_mm"])); o["after"] += int(act(hv)); o["after_plus_P2"] += int(act(h2))
                if act(hv):
                    o["remaining_by_cat"]["stale" if r["stale"] else f"{x.get('cat')}|{x.get('ev') or x.get('seed_cat')}"] += 1
                    o["by_k_after"][f"k{r['k']}"] += 1
        d["rows"] = {k_: {**v_, "remaining_by_cat": dict(v_["remaining_by_cat"]), "by_k_after": dict(v_["by_k_after"])} for k_, v_ in rows.items()}
        out[v] = d
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


def variant_rows(raw, vn):
    out = {}
    for f in sorted(f for f in glob.glob(str(raw / "*_ep*.json")) if "_attrib_" not in f):
        d = json.load(open(f))
        rows = d["rows"]; reqs = [r for r in d["requests"] if "skipped" not in r]
        byk = defaultdict(list)
        for r in rows:
            byk[int(r["k"])].append(r)
        for req in reqs:
            rs = byk.get(int(req["k"]), [])
            if not rs:
                continue
            for v in vn:
                hs = req.get("variants", {}).get(v, {}).get("row_h_mm")
                if hs is None:
                    continue
                for r, h in zip(rs, hs):
                    key = f"{req['kit']}|ep{req['ep']}|{r['traj']}"
                    o = out.setdefault(key, {}).setdefault(v, {"n": 0, "active_before": 0, "active_after": 0, "binding_before": 0, "binding_after": 0})
                    o["n"] += 1
                    if r["traj"] == "ref":
                        o["active_before"] += int(r["h_mm"] < 0); o["active_after"] += int(h < 0)
                    else:
                        o["binding_before"] += int(r["h_mm"] <= 1.0); o["binding_after"] += int(h <= 1.0)
    return out


# ------------------------------------------------------------------------------------------------ figures
#: display groups of the categories (8 = the CATEGORICAL slots, in slot order)
GROUPS = (("stale", "stale ball (fit centre > 30 mm off)"),
          ("outside", "outside the ball (centre / radius)"),
          ("zmin", "inside the ball, below z_min"),
          ("kept_tgt_ev", "kept seed: nearest evidence point is the target itself"),
          ("kept_other_ev", "kept seed: other evidence"),
          ("interior_nosign", "interior voxel not in the sign set"),
          ("sign_left", "sign left negative (impure: a kept seed within 7.5 mm)"),
          ("other", "other"))


def group_of(cat, ev):
    c = CATS[cat] if isinstance(cat, int) else cat
    if c == "stale":
        return "stale"
    if c in ("outside_ball_centre", "outside_ball_radius"):
        return "outside"
    if c == "below_zmin":
        return "zmin"
    if c in ("kept_nontarget", "kept_ambiguous", "kept_unowned", "surface_protect"):
        return "kept_tgt_ev" if ev == "target" else "kept_other_ev"
    if c == "interior_not_sign_set":
        return "interior_nosign"
    if c in ("removed_sign_left", "interior_impure"):
        return "sign_left"
    return "other"


def figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from AG3S.ag3s.experiments.common import figstyle as FS
    a.fig.mkdir(parents=True, exist_ok=True)
    S = json.load(open(a.summary))
    files = sorted(f for f in glob.glob(str(a.raw / "IFG-APPLE-SPLIT_ep*.json")) if "_attrib_" not in f)
    reqs = []
    for f in files:
        reqs += [r for r in json.load(open(f))["requests"] if "skipped" not in r and r.get("n_residual") is not None]

    def save(fig, name, data):
        p = a.fig / name
        if p.exists() and not a.force:
            raise SystemExit(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=160, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
        p.with_suffix(".json").write_text(json.dumps(data | {"script": SCRIPT}, indent=1, default=float))
        print("wrote", p)

    # (1) stacked shares by phase x scene type
    def group_counts(rr):
        g = Counter()
        for r in rr:
            if r["stale"]:
                g["stale"] += r["n_residual"]; continue
            evc = Counter()
            for k_, v in r["cat_ev"].items():
                c, e = k_.split("|"); evc[(c, e)] += v
            for c, v in r["cat_counts"].items():
                if c in ("kept_nontarget", "kept_ambiguous"):
                    t = evc.get((c, "target"), 0); g["kept_tgt_ev"] += t; g["kept_other_ev"] += v - t
                else:
                    g[group_of(c, None)] += v
        return g
    # every 'clear' request is ep320 after the apple fell off the table (stale ball, no target voxel in the fine window): bars by phase,
    # the record (OFF) next to the F1 replay (`attrib --variant F1`) when it exists
    f1 = []
    for f in sorted(glob.glob(str(a.raw / "IFG-APPLE-SPLIT_ep*_attrib_F1.json"))):
        f1 += [r for r in json.load(open(f))["requests"] if r.get("n_residual") is not None]
    bars = []
    for p in PHASES:
        bars.append((f"{p}\nrecord (F1 off)", [r for r in reqs if r["phase"] == p]))
        if f1:
            bars.append((f"{p}\nF1 replay", [r for r in f1 if r["phase"] == p]))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]}); FS.style_axes(fig, axs)
    data = {"bars": {}}
    x = np.arange(len(bars))
    bottom = np.zeros(len(bars))
    gcs = [group_counts(rr) for _, rr in bars]
    tot = np.array([sum(g.values()) for g in gcs], float)
    for gi, (gk, gl) in enumerate(GROUPS):
        v = np.array([g.get(gk, 0) for g in gcs], float)
        frac = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
        axs[0].bar(x, frac, bottom=bottom, color=FS.CATEGORICAL[gi], label=gl, width=0.7)
        bottom += frac
    for i, (lab, rr) in enumerate(bars):
        axs[0].text(i, 1.01, f"n req {len(rr)}\nvox {int(tot[i])}\nmedian/req {_med([r['n_residual'] for r in rr if not r['stale']])}", ha="center", va="bottom", fontsize=7)
        data["bars"][lab.replace("\n", "|")] = {"n_requests": len(rr), "n_voxels": int(tot[i]), **{gk: int(gcs[i].get(gk, 0)) for gk, _ in GROUPS}}
    axs[0].set_xticks(x); axs[0].set_xticklabels([b[0] for b in bars], fontsize=8); axs[0].set_ylim(0, 1.2)
    axs[0].set_ylabel("share of remaining GT target voxels", fontsize=8)
    axs[0].set_title("why GT target voxels remain in the target-free layer (IFG-APPLE-SPLIT, pooled voxels)", fontsize=9, loc="left")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    # (2) variants: residual median per phase vs neighbour / table surface kept
    V = S.get("variants", {})
    names = [n for n in ("OFF", "F1", "F2", "F1F2", "CORE0", "CORE5", "F1CORE5") if n in V or n == "OFF"]
    base = S["by_phase"]
    for pi, p in enumerate(PHASES):
        ys = [base[p]["residual_median_nonstale"] if n == "OFF" else V[n][p]["residual_median"] for n in names]
        axs[1].plot(range(len(names)), ys, "o-", color=FS.CATEGORICAL[pi], label=f"{p}: residual median (voxels / request)")
    axs[1].set_xticks(range(len(names))); axs[1].set_xticklabels(names, fontsize=8)
    axs[1].set_ylabel("remaining GT target voxels per request (median, non-stale)", fontsize=8)
    ax2 = axs[1].twinx()
    for pi, p in enumerate(PHASES):
        lost = []
        for n in names:
            if n == "OFF":
                lost.append(0.0); continue
            d = V[n][p]
            k0 = d["neighbour_surface_kept_record"]; k1 = d["neighbour_surface_kept_variant"]
            lost.append(100.0 * (k0 - k1) / k0 if k0 else 0.0)
        ax2.plot(range(len(names)), lost, "s--", color=FS.CATEGORICAL[pi + 2], label=f"{p}: neighbour surface voxels lost vs record (%)")
    ax2.set_ylabel("neighbour-fruit surface voxels lost (% of kept in record)", fontsize=8)
    h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    axs[1].set_title("offline counterfactuals (replay of the server's split; not served)", fontsize=9, loc="left")
    data["variants"] = {n: V.get(n) for n in names if n != "OFF"}
    save(fig, "ifg-sphere-attrib-shares.png", data)

    # (3) scene: one request, residual voxels by group, projected
    ep, k = a.scene_ep, a.scene_k
    d = json.load(open(a.raw / f"IFG-APPLE-SPLIT_ep{ep}.json"))
    req = next(r for r in d["requests"] if r.get("k") == k)
    vz = np.load(a.raw / f"IFG-APPLE-SPLIT_ep{ep}_voxels.npz")
    rows = [r for r in d["rows"] if r["k"] == k]
    with np.load(pathlib.Path(d["meta"]["chunk_dir"]) / f"chunk_{k + d['meta']['offset']:05d}.npz", allow_pickle=True) as z:
        o = np.asarray(z["esdf_origin"], np.float64); vs = float(z["esdf_voxel_size"])
        s = json.loads(str(z["summary_json"]))
        tp = np.asarray(z["target_points"], np.float64)
    from AG3S.trajopt.experiments import target_free_replay as TFR
    ball = TFR.recorded_ball(TFR.target_free_stats(s))
    ijk = vz[f"k{k}_ijk"].astype(int); cat = vz[f"k{k}_cat"].astype(int); ev = vz[f"k{k}_ev"].astype(int)
    W = o + ijk * vs
    grp = [group_of(VCATS[c], EVCLS[e] if e >= 0 else None) for c, e in zip(cat, ev)]
    if req["stale"]:
        grp = ["stale"] * len(grp)
    gt = np.asarray(req["gt_centre"]); fitc = np.asarray((ball.provenance.get("fit") or {}).get("centre_m", ball.centre))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6)); FS.style_axes(fig, axs)
    for ax, (i, j, ln) in zip(axs, ((0, 2, ("x (m, base)", "z (m)")), (0, 1, ("x (m, base)", "y (m)")))):
        for gi, (gk, gl) in enumerate(GROUPS):
            m_ = np.array([g == gk for g in grp], bool)
            if m_.any():
                ax.scatter(W[m_, i], W[m_, j], s=6, color=FS.CATEGORICAL[gi], label=f"{gl} ({int(m_.sum())})", alpha=0.75, lw=0)
        ax.scatter(tp[:, i], tp[:, j], s=5, marker="x", color=FS.INK_2 if hasattr(FS, "INK_2") else "#555", lw=0.6, label="target points (manipulated geometry)")
        ax.add_patch(plt.Circle((ball.centre[i], ball.centre[j]), ball.radius, fill=False, lw=1.2, color=FS.INK, label="exclusion ball (projection of the sphere)"))
        ax.plot(*gt[[i, j]], "*", ms=12, color=FS.CATEGORICAL[2], mec=FS.INK, label="GT target centre")
        ax.plot(*fitc[[i, j]], "P", ms=9, color=FS.CATEGORICAL[0], mec=FS.INK, label="ball fit centre")
        if j == 2:
            ax.axhline(ball.z_min, color=FS.CATEGORICAL[7], lw=1, ls="--", label="z_min (support + 5 mm)")
        ra =np.array([r_["voxel"] for r_ in rows if r_["traj"] == "ref" and r_["h_mm"] < 0 and r_.get("voxel") is not None])
        if len(ra):
            Wr = o + ra * vs
            ax.scatter(Wr[:, i], Wr[:, j], s=40, facecolor="none", edgecolor=FS.INK, lw=1.0, label=f"voxels touched by target-sourced active rows ({len(ra)} rows)")
        ax.set_aspect("equal"); ax.set_xlabel(ln[0], fontsize=8); ax.set_ylabel(ln[1], fontsize=8)
        ax.set_xlim(ball.centre[i] - 0.07, ball.centre[i] + 0.07); ax.set_ylim(ball.centre[j] - 0.07, ball.centre[j] + 0.07)
    axs[0].set_title(f"side view (x-z projection, all y) — ep{ep} request {k} ({req['phase']}, {req['scene_type']})", fontsize=9, loc="left")
    axs[1].set_title("top view (x-y projection, all z)", fontsize=9, loc="left")
    axs[1].legend(fontsize=6.8, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("IFG-SPHERE: remaining GT target voxels of the target-free layer by cause (projection, not a section)", fontsize=10, x=0.01, ha="left")
    save(fig, f"ifg-sphere-attrib-scene-ep{ep}-k{k}.png", {"ep": ep, "k": k, "request": req, "groups": dict(Counter(grp))})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--run", type=pathlib.Path, required=True); c.add_argument("--kit", default="IFG-APPLE-SPLIT"); c.add_argument("--arm", default="SPL")
    c.add_argument("--eps", default="all"); c.add_argument("--precheck", type=pathlib.Path, default=None)
    c.add_argument("--out", type=pathlib.Path, required=True); c.add_argument("--variants", default="")
    c.add_argument("--force", action="store_true")
    r = sp.add_parser("report")
    r.add_argument("--raw", type=pathlib.Path, required=True); r.add_argument("--out", type=pathlib.Path, required=True)
    f = sp.add_parser("figure")
    f.add_argument("--raw", type=pathlib.Path, required=True); f.add_argument("--summary", type=pathlib.Path, required=True)
    f.add_argument("--fig", type=pathlib.Path, required=True); f.add_argument("--scene-ep", type=int, default=211)
    f.add_argument("--scene-k", type=int, default=12); f.add_argument("--force", action="store_true")
    v = sp.add_parser("attrib")
    v.add_argument("--run", type=pathlib.Path, required=True); v.add_argument("--kit", default="IFG-APPLE-SPLIT"); v.add_argument("--arm", default="SPL")
    v.add_argument("--eps", default="all"); v.add_argument("--precheck", type=pathlib.Path, default=None)
    v.add_argument("--out", type=pathlib.Path, required=True, help="the collect output dir (reads <kit>_ep<E>.json)")
    v.add_argument("--variant", default="F1"); v.add_argument("--force", action="store_true")
    ra = sp.add_parser("report-attrib")
    ra.add_argument("--raw", type=pathlib.Path, required=True); ra.add_argument("--out", type=pathlib.Path, required=True)
    ra.add_argument("--variants", default="OFF,F1")
    a = ap.parse_args()
    {"collect": collect, "report": report, "figure": figures, "attrib": attrib_variant, "report-attrib": report_attrib}[a.cmd](a)


if __name__ == "__main__":
    main()
