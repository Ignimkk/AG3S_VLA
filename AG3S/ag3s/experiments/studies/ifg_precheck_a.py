"""IFG-PRECHECK-A — which GT object the field pointed at in every TrajOpt collision row that was active before the grasp.
CPU only, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
      PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_precheck_a \
      collect --run <RUN> --kit IFG-APPLE-SPLIT --arm SPL --eps 211,320 --out $AG3S_RUN_DIR/verify/IFG-PRECHECK-A/raw
  ... report --raw <dir> --fig AG3S/ag3s/docs/figures/ifg-precheck --verify AG3S/ag3s/docs/handoff/IFG-PRECHECK-A.verify.json

The definitions (active row, GT classes, attribution rules R1 / R2, correction, HOLD, palm-target distance) were fixed in
`AG3S/ag3s/docs/handoff/IFG-PRECHECK-A.verify.json` `definitions` before this script measured anything; this script follows them.

Reconstruction = `AG3S.trajopt.experiments.ta_offline.Run` (server flags, robot model, field, q_now, finger path), plus what that
harness leaves out for this server: the recorded `target_permission.target_field_exclude` mask, the strict target permission
(`target_contact_rows_of`, `target_membership = layer_diff` for `ball_owned`) and, for IFG-NBC, `neighbour_contact`.
GT = `ifg_audit_ats.Scene` (class distances) and `ifg_audit_ats.obstacle_of` (post cylinder).
"""
from __future__ import annotations

import argparse, ast, contextlib, dataclasses, datetime, glob, io, json, pathlib, sys, time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

SCRIPT = "AG3S/ag3s/experiments/studies/ifg_precheck_a.py"
CLASSES = ("target", "banana", "orange", "pear", "crate", "post", "table", "room", "robot_other")
GROUP = {"target": "target", "banana": "neighbour", "orange": "neighbour", "pear": "neighbour", "crate": "crate_post",
         "post": "crate_post", "table": "table", "room": "other", "robot_other": "other", "unknown": "unknown"}
GROUPS = ("target", "neighbour", "crate_post", "table", "other", "unknown")
FINE_TIERS = ("fine", "target_free")
PRE = ("latched", "closing")
K_EXEC = 8
PREFILTER = 0.12


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


# ------------------------------------------------------------------------------------------------ GT
class GT:
    """ifg_audit_ats.Scene with the target named 'target' and a robot class without the two grippers."""

    def __init__(self, xml, geom_sets):
        from AG3S.ag3s.experiments.studies.ifg_audit_ats import Scene
        self.S = S = Scene(str(xml))
        m = S.m; mj = S.mj
        gid = lambda n: int(n[5:]) if n.startswith("geom#") else mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        hands = {gid(n) for k in ("hand", "inactive_hand") for n in geom_sets.get(k, [])}
        self.robot_other = [g for g in S.cls["robot"] if g not in hands]
        self.n_hand_excluded = len([g for g in S.cls["robot"] if g in hands])

    def class_dist(self, c, X):
        S = self.S
        if c == "target":
            return S.class_dist("apple", X, prefilter=PREFILTER)
        if c == "post":
            return S.class_dist("obstacle", X, prefilter=PREFILTER)
        if c == "robot_other":
            X = np.asarray(X, float).reshape(-1, 3)
            out = np.full(len(X), np.inf)
            for g in self.robot_other:
                c0 = S.d.geom_xpos[g]
                if np.min(np.linalg.norm(X - c0, axis=1)) > S.m.geom_rbound[g] + PREFILTER:
                    continue
                out = np.minimum(out, S._geom_dist(g, X, PREFILTER))
            return out
        return S.class_dist(c, X, prefilter=PREFILTER)

    def all_dist(self, X):
        X = np.asarray(X, float).reshape(-1, 3)
        if not len(X):
            return np.zeros((0, len(CLASSES)))
        return np.stack([self.class_dist(c, X) for c in CLASSES], 1)


def attribute(D, d, tiers):
    """R1: (class index or -1, error, ambiguous)."""
    err = np.abs(D - d[:, None])
    order = np.argsort(err, 1)
    best = order[:, 0]; second = order[:, 1]
    r = np.arange(len(d))
    e1 = err[r, best]; e2 = err[r, second]
    tol = np.where(np.isin(tiers, FINE_TIERS), 0.010, 0.020)
    ok = np.isfinite(e1) & (e1 <= tol)
    amb = ok & np.isfinite(e2) & (e2 <= tol) & (e2 - e1 <= 0.003)
    return np.where(ok, best, -1), e1, amb, np.where(np.isfinite(e2), second, -1)


def attribute_r2(Ds):
    a = np.argmin(np.abs(Ds), 1); v = np.abs(Ds)[np.arange(len(Ds)), a]
    return np.where(v <= 0.010, a, -1)


def cname(i):
    return "unknown" if i < 0 else CLASSES[i]


# ------------------------------------------------------------------------------------------------ reconstruction
class Cli:
    hm = False
    no_plane_rows = False


def make_scene(run, inp):
    """The server's SceneSnapshot for this chunk (ta_offline.Run.inputs + recorded mask + strict permission + NBC)."""
    from AG3S.trajopt.linearize import target_contact_rows_of
    s, scene = inp["summary"], inp["scene"]
    info = {}
    tp = s.get("target_permission") or {}
    rec_mask = tp.get("target_field_exclude")
    field = scene.esdf
    has_free = bool(getattr(field, "target_free_layers", ()))
    rep = {}
    if rec_mask is not None and has_free and any(rec_mask) and scene.manipulated_points is not None:
        rec_mask = np.asarray(rec_mask, bool)
        old = scene.target_free_mask
        info["mask_diff_rows"] = int(np.sum(rec_mask != (np.zeros_like(rec_mask) if old is None else np.asarray(old, bool))))
        rep["target_free_mask"] = rec_mask
    else:
        info["mask_diff_rows"] = 0 if scene.target_free_mask is None else int(np.asarray(scene.target_free_mask).sum())
        rep["target_free_mask"] = None
    cfg = run.base_cfg
    # support plane: ta_offline falls back to a plane fitted to filtered_cloud[support_mask] when the run never held the
    # object; that fit is tilted (normal z 0.95-0.98 on IFG no-grasp runs). Use the table height AG3S recorded on this chunk
    # (esdf_stats target_free ball.support_plane_z_m), else the run's first recorded one; horizontal normal.
    if run.table_plane is None and inp["held"] is None:
        zt = support_z(s) or getattr(run, "_ifg_support_z", None)
        if zt is not None:
            run._ifg_support_z = run.__dict__.get("_ifg_support_z", zt)
            plane = np.array([0.0, 0.0, 1.0, float(zt)])
            info["support_plane"] = "ball.support_plane_z_m"
            rep["support_planes"] = plane[None]
            if scene.plane_offset is not None and len(scene.plane_offset):
                rep["plane_normal"] = plane[None, :3]
                rep["plane_offset"] = np.asarray([plane[3] + run.support_margin])
                rep["plane_active"] = np.ones(1, bool)
        else:
            info["support_plane"] = "fitted (ta_offline)"
    else:
        info["support_plane"] = "held.support.plane (ta_offline)"
    if bool(getattr(cfg.collision, "target_permission_strict", False)):
        rep["target_permission_strict"] = True
        if scene.manipulated_points is not None:
            rows = target_contact_rows_of(rep["target_free_mask"], None, scene.manipulated_link_margin,
                                          str(tp.get("policy") or "exclude_authorized"), manipulated=True)
            if rows is not None:
                rep["target_contact_rows"] = rows
            if ((s.get("target_exclusion") or {}).get("exclusion_method")) == "ball_owned":
                rep["target_membership"] = "layer_diff"
    depth = float(getattr(cfg.collision, "neighbour_contact_depth", 0.0) or 0.0)
    nbr = ((s.get("grasp") or {}).get("neighbour_contact")) or {}
    info["nbc_active"] = bool(nbr.get("active"))
    info["nbc_palm_to_target_mm"] = nbr.get("palm_to_target_mm")
    if depth > 0 and nbr.get("active"):
        man = lit(s.get("manipulated")) or {}
        names = [str(n) for n in run.robot.sphere_link_names]
        links = set(nbr.get("links") or [])
        rep["neighbour_contact"] = {"links_mask": np.array([n in links for n in names], bool), "depth": depth,
                                    "centre": np.asarray(man["centroid"], np.float64),
                                    "radius": float(cfg.collision.neighbour_contact_radius)}
    return dataclasses.replace(scene, **rep), info


def support_z(s):
    import re
    m = re.search(r"support_plane_z_m'?\"?: ([-0-9.e]+)", json.dumps(s.get("esdf_stats") or {}, default=str))
    return None if m is None else float(m.group(1))


def clearance_rows(run, lin, traj, q_now, scene, fingers):
    """(H, Q) TO row value h, d (layer value), r, centres (H, Q, 3), TO obstacle class (H, Q) or None."""
    lin.set_joint_parameter_path(fingers)
    lin.set_attached(None, None)
    centres = lin.sphere_states(traj, q_now)[0]
    centres = np.array(centres, copy=True)
    h = np.array(lin._esdf_clearance(centres, scene)[..., 0], copy=True)
    cls = getattr(lin, "_last_obstacle_class", None)
    cls = None if cls is None else np.array(cls, copy=True)
    return h, centres, cls


def layer_values(field, pts, mask_rows):
    """value, tier and surface point of the layer each row reads."""
    d = np.asarray(field.distance(pts), np.float64)
    tier = np.asarray(field.answer_tier(pts), dtype=object)
    g = np.asarray(field.gradient(pts), np.float64).reshape(-1, 3)
    if mask_rows.any():
        pf = pts[mask_rows]
        d[mask_rows] = np.asarray(field.target_free_distance(pf), np.float64)
        gf, tf = field.target_free_gradient(pf, return_tier=True)
        g[mask_rows] = np.asarray(gf, np.float64).reshape(-1, 3)
        tier[mask_rows] = np.asarray(tf, dtype=object)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    s = pts - d[:, None] * np.where(n > 1e-12, g / np.maximum(n, 1e-12), 0.0)
    return d, tier.astype(str), s


def collect_scene(run_root, kit, arm, ep, ks_extra, out_dir, log):
    from AG3S.ag3s.experiments.studies.ifg_audit_ats import obstacle_of
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    K = run_root / "verify" / kit
    jf = next(f for f in glob.glob(str(K / "replay" / f"runs_{arm}_{arm}_ep{ep}_s*.json")))
    js = json.load(open(jf)); an = json.load(open(next(iter(glob.glob(str(K / "analysis" / f"{arm}_ep{ep}_s*.json"))))))
    rd = pathlib.Path(js["run_dir"])
    entry = {"run_dir": str(rd), "server_segment": an["server_segment"]}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        run = TA.Run(f"{kit}/{ep}", entry, Cli())
    snap = pathlib.Path(js["snapshot"]).parent.parent  # .../snapshot_<x>
    xml = snap / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
    gt = GT(xml, js["geom_sets"])
    obst, ocfg = obstacle_of(run_root, rd)
    gt.S.obstacle = obst
    cfg = run.base_cfg
    H = int(cfg.horizon.planned); lf = cfg.horizon.lookahead_from
    lf = lf() if callable(lf) else lf
    holder = {}
    refiner = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["v"], cfg)
    lin = refiner.optimizer.linearizer
    assert lin.lookahead_from == lf and lin.horizon == H, (lin.lookahead_from, lf)
    lin8 = run.meter(K_EXEC)
    names = [str(n) for n in run.robot.sphere_link_names]
    hold_seqs = {int(h["seq"]) for h in an.get("hold") or []}
    pl = [json.loads(x) for x in open(rd / "frames.jsonl") if x.strip()]
    pl = [r for r in pl if r.get("kind") == "planning"]
    reqs, rows_out = [], []
    hand = run.hand
    fl = "l" if hand == "left" else "r"
    for k in range(len(pl)):
        seq = k + 1
        with np.load(run.chunks[seq], allow_pickle=True) as z0:
            s0 = json.loads(str(z0["summary_json"]))
        state = str((s0.get("grasp") or {}).get("state"))
        if state not in PRE and k not in ks_extra:
            continue
        inp, why = run.inputs(seq)
        req = {"ep": ep, "kit": kit, "k": k, "seq": seq, "state": state, "pre_grasp": state in PRE}
        if inp is None:
            req["skipped"] = why; reqs.append(req); continue
        s, q_now = inp["summary"], inp["q_now"]
        scene, info = make_scene(run, inp)
        req.update(info)
        gt.S.set_qpos(np.asarray(inp["row"]["qpos"], float))
        # fidelity
        met = (s.get("to") or {}).get("metrics") or {}
        rec_traj = run.layout.chunk_to_trajectory(inp["rec_chunk"])
        ref_traj = run.layout.chunk_to_trajectory(inp["ref_chunk"])
        if met.get("clearance_m") is not None and scene.esdf is not None:
            lin8.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], K_EXEC))
            lin8.set_attached(None, None)
            mine, pair = lin8.worst_row(rec_traj[:, :K_EXEC], q_now, scene)
            recp = lit(met.get("max_violation_pair")) or {}
            req["recorded_clearance_mm"] = float(met["clearance_m"]) * 1e3
            req["recon_clearance_mm"] = float(mine) * 1e3
            req["fidelity_abs_diff_mm"] = abs(float(mine) - float(met["clearance_m"])) * 1e3
            req["fidelity_ok"] = req["fidelity_abs_diff_mm"] <= 1.0
            req["fidelity_same_row"] = bool(pair and recp and int(recp.get("step", -1)) == int(pair["step"])
                                            and int(recp.get("query", -1)) == int(pair["query"]))
        req["to_status"] = (s.get("to") or {}).get("status")
        req["to_iterations"] = (s.get("to") or {}).get("sqp_iterations")
        vd = s.get("verdict") or {}
        reasons = vd.get("reasons") or []
        req["hold"] = any(r.get("action") == "hold" for r in reasons)
        req["hold_kinds"] = sorted({str(r.get("kind")) for r in reasons if r.get("action") == "hold"})
        req["hold_detail"] = [str(r.get("detail"))[:160] for r in reasons if r.get("action") == "hold"]
        req["hold_in_analysis"] = seq in hold_seqs
        req["penetration_to_mm"] = max(0.0, -float(met["clearance_m"]) * 1e3) if met.get("clearance_m") is not None else None
        # correction: fingertip midpoint over the executed window
        tips_ref = np.array([TA.finger_mid(run, ref_traj, q_now, i) for i in range(K_EXEC)])
        tips_out = np.array([TA.finger_mid(run, rec_traj, q_now, i) for i in range(K_EXEC)])
        req["correction_mm"] = float(np.linalg.norm(tips_out - tips_ref, axis=1).max() * 1e3)
        req["correction_by_step_mm"] = [round(float(v) * 1e3, 2) for v in np.linalg.norm(tips_out - tips_ref, axis=1)]
        # palm - target
        palm = np.asarray(run.robot.link_pose(q_now, f"ee_{hand}"))[:3, 3]
        gtc = gt.S.d.geom_xpos[gt.S.cls["apple"][0]].copy()
        req["palm_target_gt_mm"] = float(np.linalg.norm(palm - gtc) * 1e3)
        req["reach_mm_recorded"] = ((s.get("grasp") or {}).get("evidence") or {}).get("reach_mm")
        req["gt_target_centre"] = gtc.tolist()
        req["has_target"] = bool(s.get("has_target"))
        req["manip_state"] = (lit(s.get("manipulated")) or {}).get("state") if s.get("manipulated") else None
        if scene.esdf is None:
            req["skipped"] = "no field"; reqs.append(req); continue
        fingers = run.finger_path(s, inp["ref_chunk"], H)
        mask = np.zeros(len(names), bool) if scene.target_free_mask is None else np.asarray(scene.target_free_mask, bool)
        for tname, traj in (("ref", ref_traj[:, :H]), ("out", rec_traj[:, :H])):
            h, C, tcls = clearance_rows(run, lin, traj, q_now, scene, fingers)
            thr = 0.0 if tname == "ref" else 0.001
            act = np.argwhere(h < thr) if tname == "ref" else np.argwhere(h <= thr)
            req[f"n_rows_{tname}"] = int(len(act))
            if tname == "out":
                win = h[:K_EXEC]
                kq = np.unravel_index(np.argmin(win), win.shape)
                req["out_worst_window"] = {"step": int(kq[0]), "query": int(kq[1]), "link": names[kq[1]], "h_mm": float(win[kq] * 1e3)}
            if tname == "ref":
                kq = np.unravel_index(np.argmin(h), h.shape)
                req["ref_worst"] = {"step": int(kq[0]), "query": int(kq[1]), "link": names[kq[1]], "h_mm": float(h[kq] * 1e3)}
            sel = act
            if tname == "out":
                # make sure the window worst row is attributed even if it is not binding
                w = (int(kq[0]), int(kq[1]))
                if not any((a[0] == w[0] and a[1] == w[1]) for a in act):
                    sel = np.vstack([act, np.array([w])]) if len(act) else np.array([w])
            if not len(sel):
                continue
            P = C[sel[:, 0], sel[:, 1]]
            mrows = mask[sel[:, 1]]
            d, tier, surf = layer_values(scene.esdf, P, mrows)
            r = np.asarray(lin.robot_radii, np.float64)[sel[:, 1]]
            D = gt.all_dist(P)
            c1, e1, amb, c2 = attribute(D, d, tier)
            Ds = gt.all_dist(surf)
            cr2 = attribute_r2(Ds)
            for i in range(len(sel)):
                kk, qq = int(sel[i, 0]), int(sel[i, 1])
                rows_out.append({"ep": ep, "kit": kit, "k": k, "state": state, "traj": tname, "step": kk, "q": qq,
                                 "link": names[qq], "layer": "target_free" if mrows[i] else "main", "tier": str(tier[i]),
                                 "h_mm": round(float(h[kk, qq]) * 1e3, 3), "d_mm": round(float(d[i]) * 1e3, 3),
                                 "r_mm": round(float(r[i]) * 1e3, 3), "pen": bool(d[i] - r[i] < 0),
                                 "to_class": None if tcls is None else int(tcls[kk, qq]),
                                 "src": cname(int(c1[i])), "err_mm": round(float(e1[i]) * 1e3, 3) if np.isfinite(e1[i]) else None,
                                 "second": cname(int(c2[i])), "amb": bool(amb[i]), "best_any": CLASSES[int(np.argmin(np.abs(D[i] - d[i])))] if np.isfinite(D[i]).any() else "none", "src_r2": cname(int(cr2[i])),
                                 "binding": bool(h[kk, qq] < thr) if tname == "ref" else bool(h[kk, qq] <= thr),
                                 "D_mm": {CLASSES[j]: (round(float(D[i, j]) * 1e3, 2) if np.isfinite(D[i, j]) else None) for j in range(len(CLASSES))},
                                 "p": [round(float(v), 5) for v in P[i]]})
        reqs.append(req)
        log(f"  {kit} ep{ep} k{k} {state} ref {req.get('n_rows_ref')} out {req.get('n_rows_out')} fid {req.get('fidelity_abs_diff_mm', float('nan')):.2f} hold {req['hold']}")
    meta = {"ep": ep, "kit": kit, "arm": arm, "run_dir": str(rd), "xml": str(xml), "obstacle": ocfg, "horizon_planned": H,
            "lookahead_from": lf, "server_cmdline": run.cmdline, "success": an.get("success"), "failure_mode": an.get("failure_mode"),
            "n_requests": len(pl), "hand": hand, "n_robot_geoms_hand_excluded": gt.n_hand_excluded, "seconds": round(time.time() - t0, 1)}
    out = out_dir / f"{kit}_ep{ep}.json"
    out.write_text(json.dumps({"meta": meta, "requests": reqs, "rows": rows_out}, default=float))
    log(f"{kit} ep{ep}: {len(reqs)} requests, {len(rows_out)} rows, {time.time() - t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--run", type=pathlib.Path, required=True); c.add_argument("--kit", required=True); c.add_argument("--arm", required=True)
    c.add_argument("--eps", required=True); c.add_argument("--extra-k", default=""); c.add_argument("--out", type=pathlib.Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--raw", type=pathlib.Path, required=True); r.add_argument("--fig", type=pathlib.Path, required=True)
    r.add_argument("--verify", type=pathlib.Path, required=True); r.add_argument("--run", type=pathlib.Path, required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        a.out.mkdir(parents=True, exist_ok=True)
        extra = {int(x) for x in a.extra_k.split(",") if x}
        logf = open(a.out / f"collect_{a.kit}_{a.eps.replace(',', '_')}.log", "a")

        def log(msg):
            print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
        for ep in [int(x) for x in a.eps.split(",")]:
            collect_scene(a.run, a.kit, a.arm, ep, extra, a.out, log)
    else:
        report(a)


# ------------------------------------------------------------------------------------------------ report
FAIL = (211, 320, 526, 954, 979, 1313)
CTRL = (250, 126, 572)
G_LABEL = {"target": ("target 사과", "target apple"), "neighbour": ("이웃 과일", "neighbour fruit"),
           "crate_post": ("바구니 · 기둥", "crate / post"), "table": ("탁자", "table"), "other": ("기타 (방 · 로봇)", "other (room, robot)"),
           "unknown": ("미상 (GT 표면 없음)", "unknown (no GT surface)"), "none": ("활성 행 없음", "no active row")}


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.median(v)), 2)


def pct(v, q):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.percentile(v, q)), 2)


def load_raw(raw):
    data = {}
    for f in sorted(raw.glob("*_ep*.json")):
        j = json.load(open(f)); data[(j["meta"]["kit"], j["meta"]["ep"])] = j
    return data


def group_counts(rows, key="src"):
    c = Counter(GROUP[r[key]] for r in rows)
    n = sum(c.values())
    return {"n_rows": n, "counts": {g: c.get(g, 0) for g in GROUPS},
            "fractions": {g: (round(c.get(g, 0) / n, 4) if n else None) for g in GROUPS}}


def per_request(j):
    """request dicts with the attribution of their worst rows."""
    rows = j["rows"]
    by = defaultdict(list)
    for r in rows:
        by[(r["k"], r["traj"])].append(r)
    out = []
    for q in j["requests"]:
        if "skipped" in q:
            out.append(q); continue
        ref = [r for r in by[(q["k"], "ref")] if r["binding"]]
        outr = by[(q["k"], "out")]
        q = dict(q)
        if ref:
            w = min(ref, key=lambda r: r["h_mm"])
            q["dominant_src"] = w["src"]; q["dominant_group"] = GROUP[w["src"]]; q["dominant_layer"] = w["layer"]
            q["ref_groups"] = group_counts(ref)["counts"]
        else:
            q["dominant_src"] = "none"; q["dominant_group"] = "none"; q["ref_groups"] = {g: 0 for g in GROUPS}
        ow = q.get("out_worst_window")
        wr = [r for r in outr if ow and r["step"] == ow["step"] and r["q"] == ow["query"]]
        q["out_worst_src"] = wr[0]["src"] if wr else None
        q["out_worst_group"] = GROUP[wr[0]["src"]] if wr else None
        q["out_worst_pen_geom_mm"] = (round(max(0.0, -(wr[0]["d_mm"] - wr[0]["r_mm"])), 3) if wr else None)
        q["n_binding_out"] = sum(1 for r in outr if r["binding"])
        q["out_groups"] = group_counts([r for r in outr if r["binding"]])["counts"]
        out.append(q)
    return out


def a3_table(reqs, key):
    res = {}
    for g in GROUPS + ("none",):
        rr = [q for q in reqs if q.get(key) == g]
        if not rr:
            continue
        hold = [q for q in rr if q["hold"]]
        res[g] = {"n_requests": len(rr), "correction_mm_median": med([q["correction_mm"] for q in rr]),
                  "correction_mm_p75": pct([q["correction_mm"] for q in rr], 75), "correction_mm_max": round(max(q["correction_mm"] for q in rr), 2),
                  "n_hold": len(hold), "hold_requests": [f"{q['kit'].split('-')[-1]}:{q['ep']}:k{q['k']}" for q in hold],
                  "penetration_to_mm_median_hold": med([q["penetration_to_mm"] for q in hold]),
                  "penetration_to_mm_median_all": med([q["penetration_to_mm"] for q in rr])}
    return res


def report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from AG3S.ag3s.experiments.common import figstyle as FS
    KO = FS.use_korean()
    T = (lambda ko, en: ko if KO else en)  # noqa: E731
    data = load_raw(a.raw)
    a.fig.mkdir(parents=True, exist_ok=True)
    reqs_all = {key: per_request(j) for key, j in data.items()}
    split = [k for k in data if k[0] == "IFG-APPLE-SPLIT"]
    N = {}
    per_scene = {}
    # ---------------- A1 / A2 per scene and phase
    for key in sorted(data, key=lambda k: (k[0], k[1] not in FAIL, k[1])):
        j = data[key]; kit, ep = key
        rq = reqs_all[key]
        rows = j["rows"]
        sc = {"kit": kit, "episode": ep, "role": "fail" if (kit == "IFG-APPLE-SPLIT" and ep in FAIL) else ("control" if ep in CTRL and kit == "IFG-APPLE-SPLIT" else "nbc"),
              "success": j["meta"]["success"], "failure_mode": j["meta"]["failure_mode"], "obstacle": bool(j["meta"]["obstacle"]),
              "n_requests_measured": len(rq), "fidelity": {"n_checked": sum(1 for q in rq if q.get("fidelity_abs_diff_mm") is not None),
                                                          "n_over_1mm": sum(1 for q in rq if q.get("fidelity_ok") is False),
                                                          "over_1mm_k": [q["k"] for q in rq if q.get("fidelity_ok") is False],
                                                          "max_abs_diff_mm": round(max(q.get("fidelity_abs_diff_mm") or 0 for q in rq), 4)},
              "by_phase": {}}
        for ph in PRE + ("pre_grasp",):
            qq = [q for q in rq if q["pre_grasp"] and (ph == "pre_grasp" or q["state"] == ph)]
            ks = {q["k"] for q in qq}
            ref = [r for r in rows if r["traj"] == "ref" and r["binding"] and r["k"] in ks]
            outb = [r for r in rows if r["traj"] == "out" and r["binding"] and r["k"] in ks]
            gc = group_counts(ref)
            sc["by_phase"][ph] = {
                "n_requests": len(qq), "n_requests_with_active_ref": sum(1 for q in qq if q.get("n_rows_ref", 0) > 0),
                "active_ref_rows": gc["n_rows"], "active_ref_penetrating_rows": sum(r["pen"] for r in ref),
                "active_ref_by_layer": dict(Counter(r["layer"] for r in ref)),
                "active_ref_source_counts": gc["counts"], "active_ref_source_fractions": gc["fractions"],
                "active_ref_source_counts_fine": dict(Counter(r["src"] for r in ref)),
                "n_requests_with_source": {g: sum(1 for q in qq if q["ref_groups"].get(g, 0) > 0) for g in GROUPS},
                "binding_out_rows": len(outb), "binding_out_source_counts": group_counts(outb)["counts"],
                "n_hold": sum(1 for q in qq if q["hold"]),
                "r1_r2_agreement": round(float(np.mean([r["src"] == r["src_r2"] for r in ref])), 4) if ref else None,
                "n_ambiguous": sum(r["amb"] for r in ref),
                "unknown_best_any": dict(Counter(r["best_any"] for r in ref if r["src"] == "unknown")),
                "unknown_r2": dict(Counter(r["src_r2"] for r in ref if r["src"] == "unknown"))}
        per_scene[f"{kit}:{ep}"] = sc
    # pooled
    def pool(keys, ph="pre_grasp", excl_fid=False):
        rows, qq = [], []
        for key in keys:
            rq = [q for q in reqs_all[key] if q["pre_grasp"] and (ph == "pre_grasp" or q["state"] == ph)]
            if excl_fid:
                rq = [q for q in rq if q.get("fidelity_ok") is not False]
            ks = {q["k"] for q in rq}
            rows += [r for r in data[key]["rows"] if r["traj"] == "ref" and r["binding"] and r["k"] in ks]
            qq += rq
        gc = group_counts(rows)
        return {"n_scenes": len(keys), "n_requests": len(qq), "n_requests_with_active_ref": sum(1 for q in qq if q.get("n_rows_ref", 0) > 0),
                "active_ref_rows": gc["n_rows"], "source_counts": gc["counts"], "source_fractions": gc["fractions"],
                "n_requests_with_source": {g: sum(1 for q in qq if q["ref_groups"].get(g, 0) > 0) for g in GROUPS},
                "by_layer": {L: group_counts([r for r in rows if r["layer"] == L])["counts"] for L in ("target_free", "main")}}
    fails = [("IFG-APPLE-SPLIT", e) for e in FAIL]; ctrls = [("IFG-APPLE-SPLIT", e) for e in CTRL]
    N["A1_A2_pooled"] = {grp: {ph: pool(keys, ph) for ph in PRE + ("pre_grasp",)} for grp, keys in (("split_fail_6", fails), ("split_control_3", ctrls))}
    N["A1_A2_pooled_fidelity_ok_only"] = {grp: pool(keys, "pre_grasp", True) for grp, keys in (("split_fail_6", fails), ("split_control_3", ctrls))}
    allref = [r for key in split for r in data[key]["rows"] if r["traj"] == "ref" and r["binding"]
              and any(q["k"] == r["k"] and q["pre_grasp"] for q in reqs_all[key])]
    N["attribution_checks_split_pre_grasp"] = {"n_rows": len(allref), "r1_r2_agreement": round(float(np.mean([r["src"] == r["src_r2"] for r in allref])), 4),
                                               "n_ambiguous": int(sum(r["amb"] for r in allref)),
                                               "tier_counts": dict(Counter(r["tier"] for r in allref)),
                                               "unknown_best_any": dict(Counter(r["best_any"] for r in allref if r["src"] == "unknown")),
                                               "unknown_r2": dict(Counter(r["src_r2"] for r in allref if r["src"] == "unknown")),
                                               "unknown_err_mm_median": med([r["err_mm"] for r in allref if r["src"] == "unknown"])}
    # ---------------- A3
    for grp, keys in (("split_fail_6", fails), ("split_control_3", ctrls)):
        rq = [q for key in keys for q in reqs_all[key] if q["pre_grasp"] and "skipped" not in q]
        N.setdefault("A3", {})[grp] = {
            "n_requests": len(rq), "n_hold": sum(q["hold"] for q in rq),
            "by_dominant_source_ref": a3_table(rq, "dominant_group"),
            "by_out_window_worst_source": a3_table(rq, "out_worst_group"),
            "hold_requests_detail": [{"scene": f"{q['ep']}", "k": q["k"], "state": q["state"], "dominant_ref": q["dominant_src"],
                                      "out_worst": q["out_worst_src"], "out_worst_link": (q.get("out_worst_window") or {}).get("link"),
                                      "penetration_to_mm": round(q["penetration_to_mm"] or 0, 3), "hold_detail": q["hold_detail"],
                                      "correction_mm": round(q["correction_mm"], 2)} for q in rq if q["hold"]],
            "requests_with_any_target_row": {"n": sum(1 for q in rq if q["ref_groups"]["target"] > 0),
                                             "correction_mm_median": med([q["correction_mm"] for q in rq if q["ref_groups"]["target"] > 0]),
                                             "n_hold": sum(1 for q in rq if q["ref_groups"]["target"] > 0 and q["hold"])},
            "requests_with_any_neighbour_row": {"n": sum(1 for q in rq if q["ref_groups"]["neighbour"] > 0),
                                                "correction_mm_median": med([q["correction_mm"] for q in rq if q["ref_groups"]["neighbour"] > 0]),
                                                "n_hold": sum(1 for q in rq if q["ref_groups"]["neighbour"] > 0 and q["hold"])},
            "requests_without_active_ref": {"n": sum(1 for q in rq if q["dominant_group"] == "none"),
                                            "correction_mm_median": med([q["correction_mm"] for q in rq if q["dominant_group"] == "none"])}}
    # ---------------- A4
    a4 = {}
    for kit in ("IFG-APPLE-SPLIT", "IFG-APPLE-NBC"):
        key = (kit, 211)
        if key not in data:
            continue
        rows = []
        for q in reqs_all[key]:
            if not 10 <= q["k"] <= 16:
                continue
            rows.append({"k": q["k"], "seq": q["seq"], "state": q["state"], "n_active_ref": q.get("n_rows_ref"),
                         "ref_source_counts": q["ref_groups"], "dominant_ref": q["dominant_src"], "n_binding_out": q.get("n_binding_out"),
                         "out_worst_src": q["out_worst_src"], "out_worst_link": (q.get("out_worst_window") or {}).get("link"),
                         "out_worst_h_mm": round((q.get("out_worst_window") or {}).get("h_mm", float("nan")), 3),
                         "hold": q["hold"], "hold_detail": q["hold_detail"], "penetration_to_mm": round(q["penetration_to_mm"] or 0.0, 3),
                         "palm_target_gt_mm": round(q["palm_target_gt_mm"], 1), "reach_mm_recorded": q["reach_mm_recorded"],
                         "nbc_active": q.get("nbc_active"), "nbc_palm_to_target_mm": q.get("nbc_palm_to_target_mm"),
                         "correction_mm": round(q["correction_mm"], 2), "fidelity_abs_diff_mm": round(q.get("fidelity_abs_diff_mm") or 0, 4)})
        a4[kit] = rows
    N["A4_ep211_k10_16"] = a4
    # ---------------- figures
    figs = []
    figs.append(fig_scene(a, data, reqs_all, FS, T, plt))
    figs.append(fig_graph(a, per_scene, N, reqs_all, FS, T, plt))
    figs.append(fig_table(a, a4, FS, T, plt))
    # ---------------- verify.json
    v = json.load(open(a.verify))
    kst = datetime.timezone(datetime.timedelta(hours=9))
    v["written"] = v["written"].split(" (")[0] + " (definitions) / " + datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST") + " (numbers)"
    v["status"] = "measured"
    v["runs"] = [{"id": f"{k[0]}:{k[1]}", "raw_output": str(a.raw / f"{k[0]}_ep{k[1]}.json"), "seconds": data[k]["meta"]["seconds"]} for k in sorted(data)]
    v["numbers"] = N
    v["per_scene"] = per_scene
    v["figures"] = figs
    v["scripts"] = [SCRIPT]
    a.verify.write_text(json.dumps(v, indent=1, ensure_ascii=False, default=float))
    print("wrote", a.verify)


def save(fig, path, data, FS, plt):
    path = pathlib.Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({**data, "script": SCRIPT}, indent=1, ensure_ascii=False, default=float))
    print("wrote", path)
    return {"path": str(path), "sidecar": str(path.with_suffix(".json"))}


def gcol(FS):
    return {"target": FS.CATEGORICAL[1], "neighbour": FS.CATEGORICAL[3], "crate_post": FS.CATEGORICAL[6], "table": FS.CATEGORICAL[2],
            "other": FS.CATEGORICAL[4], "unknown": FS.INK, "none": FS.GRID_INK}


def fig_scene(a, data, reqs_all, FS, T, plt):
    """Real scene: ep211 k12 (SPLIT) and the same request in NBC; free camera near the target, active_ref rows coloured by source."""
    import mujoco
    from AG3S.ag3s.experiments.studies.ifg_audit_ats import Scene
    col = gcol(FS)
    kits = [k for k in ("IFG-APPLE-SPLIT", "IFG-APPLE-NBC") if (k, 211) in data]
    K_SHOW = 12
    fig, axs = plt.subplots(len(kits), 2, figsize=(13.5, 5.6 * len(kits)), gridspec_kw={"width_ratios": [1.35, 1]}, squeeze=False)
    FS.style_axes(fig, axs)
    side = {}
    W, H = 720, 540
    for ii, kit in enumerate(kits):
        j = data[(kit, 211)]
        S = Scene(j["meta"]["xml"])
        pl = [json.loads(x) for x in open(pathlib.Path(j["meta"]["run_dir"]) / "frames.jsonl") if x.strip()]
        pl = [r for r in pl if r.get("kind") == "planning"]
        q = next(q for q in reqs_all[(kit, 211)] if q["k"] == K_SHOW)
        S.set_qpos(np.asarray(pl[K_SHOW]["qpos"], float))
        rnd = mujoco.Renderer(S.m, H, W)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        tc = np.asarray(q["gt_target_centre"]); cam.lookat[:] = tc + np.array([0.0, 0.0, 0.02]); cam.distance = 0.55
        cam.azimuth = 180.0; cam.elevation = -30.0
        rnd.update_scene(S.d, camera=cam); img = rnd.render()
        gc = rnd.scene.camera[0]
        # mono render = midpoint of the two stereo eyes (scene.camera[0] is the left eye, offset by ipd / 2)
        pos = 0.5 * (np.array(rnd.scene.camera[0].pos) + np.array(rnd.scene.camera[1].pos)); fwd = np.array(gc.forward); up = np.array(gc.up); right = np.cross(fwd, up)
        near = gc.frustum_near; ht = gc.frustum_top; hw = ht * W / H
        rnd.close()

        def proj(p):
            v = np.atleast_2d(p) - pos; zc = v @ fwd
            return ((v @ right) * near / zc + hw) / (2 * hw) * W, (ht - (v @ up) * near / zc) / (2 * ht) * H
        rows = [r for r in j["rows"] if r["k"] == K_SHOW and r["traj"] == "ref" and r["binding"]]
        ax = axs[ii, 0]; ax.imshow(img)
        cnt = Counter(GROUP[r["src"]] for r in rows)
        for g in GROUPS:
            rr = [r for r in rows if GROUP[r["src"]] == g]
            if not rr:
                continue
            u, vv = proj(np.array([r["p"] for r in rr]))
            ax.scatter(u, vv, s=10, color=col[g], edgecolor="white", linewidth=0.3, label=f"{T(*G_LABEL[g])} ({cnt[g]})", zorder=3)
        u, vv = proj(tc); ax.plot(u, vv, "+", color="white", ms=12, mew=2)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, frameon=True, loc="lower left", title=T("활성 제약 행 출처 (R1)", "active-row source (R1)"), title_fontsize=7.5)
        ax.set_title(T(f"{kit.split('-')[-1]} ep211 요청 {K_SHOW} ({q['state']}) — 실제 씬, 정책 청크 위의 활성 행 {len(rows)}개 (16 step), HOLD={q['hold']}",
                       f"{kit.split('-')[-1]} ep211 request {K_SHOW} ({q['state']}): real scene, {len(rows)} active rows on the policy chunk, HOLD={q['hold']}"),
                     fontsize=8.5, loc="left")
        # top view: GT outlines (mesh vertices) + rows
        bx = axs[ii, 1]
        for f, c_ in (("apple", FS.GRID_INK), ("banana", FS.INK_2)):
            for g in S.cls[f]:
                if S.m.geom_type[g] == 7:
                    vtx = S.mesh_world_vertices(g)
                    bx.scatter(vtx[:, 0] * 1000, vtx[:, 1] * 1000, s=1, color=c_, alpha=0.35, zorder=1)
        for g in S.cls["crate"]:
            if S.m.geom_type[g] == 6:
                c0 = S.d.geom_xpos[g]; R = S.d.geom_xmat[g].reshape(3, 3); sz = S.m.geom_size[g]
                cs = np.array([[sx, sy, 0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]) * sz
                w = cs @ R.T + c0
                bx.plot(w[:, 0] * 1000, w[:, 1] * 1000, color=col["crate_post"], lw=0.8, zorder=1)
        ob = j["meta"]["obstacle"]
        if ob:
            th = np.linspace(0, 2 * np.pi, 60)
            bx.plot((ob["position"][0] + ob["radius"] * np.cos(th)) * 1000, (ob["position"][1] + ob["radius"] * np.sin(th)) * 1000, color=col["crate_post"], lw=1.2)
        for g in GROUPS:
            rr = [r for r in rows if GROUP[r["src"]] == g]
            if rr:
                P = np.array([r["p"] for r in rr]) * 1000
                bx.scatter(P[:, 0], P[:, 1], s=12, color=col[g], edgecolor="white", linewidth=0.3, zorder=3)
        bx.plot(tc[0] * 1000, tc[1] * 1000, "+", color=FS.INK, ms=10, mew=1.5)
        bx.set_xlim(tc[0] * 1000 - 140, tc[0] * 1000 + 140); bx.set_ylim(tc[1] * 1000 - 140, tc[1] * 1000 + 140); bx.set_aspect("equal")
        bx.set_xlabel("x (mm, base)", fontsize=8); bx.set_ylabel("y (mm, base)", fontsize=8)
        bx.set_title(T("위에서 본 같은 순간: 회색 점 = GT mesh 꼭짓점 (연한 = 사과, 진한 = 바나나), 보라 선 = 바구니 벽, + = 사과 중심",
                       "top view: dots = GT mesh vertices (apple, banana), lines = crate walls / post"), fontsize=8, loc="left")
        side[kit] = {"k": K_SHOW, "state": q["state"], "hold": q["hold"], "n_active_ref": len(rows), "source_counts": dict(cnt),
                     "camera": {"lookat": (tc + [0, 0, 0.02]).round(4).tolist(), "distance": 0.55, "azimuth": 180, "elevation": -30},
                     "gt_target_centre": tc.tolist(), "palm_target_gt_mm": round(q["palm_target_gt_mm"], 1),
                     "rows": [{"p": r["p"], "src": r["src"], "link": r["link"], "step": r["step"], "h_mm": r["h_mm"], "layer": r["layer"]} for r in rows]}
    fig.suptitle(T("IFG-PRECHECK-A 실제 씬 — TrajOpt 활성 충돌 행이 가리킨 GT 물체 (ep211, 쥐기 전)",
                   "IFG-PRECHECK-A real scene: GT object behind each active TrajOpt collision row (ep211, pre-grasp)"), fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, a.fig / "ifg-precheck-a-scene-ep211.png", {"panels": side}, FS, plt)


def fig_graph(a, per_scene, N, reqs_all, FS, T, plt):
    col = gcol(FS)
    order = [f"IFG-APPLE-SPLIT:{e}" for e in FAIL] + [f"IFG-APPLE-SPLIT:{e}" for e in CTRL]
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"width_ratios": [1.15, 1.15, 1.2], "wspace": 0.35})
    FS.style_axes(fig, axs)
    data = {}
    for ax, ph in zip(axs[:2], PRE):
        labels, left = [], None
        M = np.array([[per_scene[s]["by_phase"][ph]["active_ref_source_counts"][g] for g in GROUPS] for s in order], float)
        tot = M.sum(1, keepdims=True)
        F = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
        y = np.arange(len(order))[::-1]
        left = np.zeros(len(order))
        for gi, g in enumerate(GROUPS):
            ax.barh(y, F[:, gi], left=left, color=col[g], height=0.7, label=T(*G_LABEL[g]))
            left += F[:, gi]
        for yi, s in zip(y, order):
            ps = per_scene[s]["by_phase"][ph]
            ax.text(1.02, yi, f"{int(ps['active_ref_rows'])} / {ps['n_requests_with_active_ref']}/{ps['n_requests']}", va="center", fontsize=7, color=FS.INK_2)
        ax.set_yticks(y); ax.set_yticklabels([f"ep{s.split(':')[1]} ({T('실패', 'fail') if int(s.split(':')[1]) in FAIL else T('대조', 'ctrl')})" for s in order], fontsize=8)
        ax.set_xlim(0, 1.32); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axhline(len(CTRL) - 0.5, color=FS.GRID_INK, lw=0.8)
        ax.set_xlabel(T("활성 행 비율 (R1 출처)", "fraction of active rows (R1 source)"), fontsize=8)
        ax.set_title(T(f"{ph}: 장면별 활성 행 출처 (오른쪽 숫자 = 행 수 / 활성 요청 / 요청)", f"{ph}: source of active rows (rows / active req / req)"), fontsize=8.5, loc="left")
        data[ph] = {s: per_scene[s]["by_phase"][ph]["active_ref_source_counts"] for s in order}
    axs[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.1), ncol=6)
    # A3: correction by dominant source, HOLD marked
    ax = axs[2]
    rq = [q for s in order for q in reqs_all[("IFG-APPLE-SPLIT", int(s.split(":")[1]))] if q["pre_grasp"] and "skipped" not in q]
    cats = [g for g in GROUPS + ("none",) if any(q["dominant_group"] == g for q in rq)]
    rng = np.random.default_rng(0)
    pts = {}
    for ci, g in enumerate(cats):
        rr = [q for q in rq if q["dominant_group"] == g]
        x = ci + rng.uniform(-0.25, 0.25, len(rr))
        yv = np.array([q["correction_mm"] for q in rr])
        hold = np.array([q["hold"] for q in rr]); fail = np.array([q["ep"] in FAIL for q in rr])
        ax.scatter(x[~hold & fail], yv[~hold & fail], s=12, color=col[g], alpha=0.8, label=None)
        ax.scatter(x[~hold & ~fail], yv[~hold & ~fail], s=12, facecolor="none", edgecolor=col[g], alpha=0.9)
        ax.scatter(x[hold], yv[hold], s=55, marker="x", color=FS.INK, zorder=4)
        ax.hlines(np.median(yv), ci - 0.35, ci + 0.35, color=FS.INK, lw=1.2)
        ax.text(ci, ax.get_ylim()[1] if False else max(yv.max(), 1) * 1.05, f"n={len(rr)}\nHOLD {int(hold.sum())}", ha="center", fontsize=7, color=FS.INK_2)
        pts[g] = {"correction_mm": yv.round(2).tolist(), "hold": hold.tolist(), "fail_scene": fail.tolist(), "median_mm": round(float(np.median(yv)), 2)}
    ax.set_yscale("symlog", linthresh=5); ax.set_xticks(range(len(cats))); ax.set_xticklabels([T(*G_LABEL[g]) for g in cats], fontsize=7.5, rotation=20)
    ax.set_ylabel(T("손끝 수정량 (mm, 실행 창 최대)", "fingertip correction (mm, max over executed window)"), fontsize=8)
    ax.scatter([], [], s=12, color=FS.INK_2, label=T("실패 장면", "fail scene")); ax.scatter([], [], s=12, facecolor="none", edgecolor=FS.INK_2, label=T("대조 장면", "control"))
    ax.scatter([], [], s=55, marker="x", color=FS.INK, label="HOLD"); ax.plot([], [], color=FS.INK, lw=1.2, label=T("중앙값", "median"))
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title(T("요청별 수정량 — 가장 많이 침범한 활성 행의 출처로 묶음 (쥐기 전, SPLIT 9 장면)", "correction per request by the source of its worst active row"), fontsize=8.5, loc="left")
    fig.suptitle(T("IFG-PRECHECK-A: 쥐기 전 TrajOpt 활성 충돌 행의 출처 (정책 청크 기준, h < 0)", "IFG-PRECHECK-A: source of active pre-grasp collision rows"), fontsize=10.5, x=0.01, ha="left")
    return save(fig, a.fig / "ifg-precheck-a-sources.png", {"source_counts_by_phase": data, "correction_by_dominant_source": pts,
                                                            "groups": list(GROUPS)}, FS, plt)


def fig_table(a, a4, FS, T, plt):
    cols = ["kit", T("요청 k", "k"), T("단계", "state"), T("활성 행", "active"), "target", T("이웃", "neigh"), T("바구니·기둥", "crate/post"), T("탁자", "table"),
            T("미상", "unknown"), T("최다 침범\n(정책)", "worst (ref)"), T("출력 창\n최악 행", "worst (out)"), T("NBC 발동", "NBC on"), "HOLD", T("침범 mm", "pen mm"),
            T("손바닥–target\nmm", "palm-target mm"), T("수정 mm", "corr mm")]
    body, kinds = [], []
    for kit, rows in a4.items():
        for r in rows:
            g = r["ref_source_counts"]
            body.append([kit.split("-")[-1], str(r["k"]), r["state"], str(r["n_active_ref"]), str(g["target"]), str(g["neighbour"]), str(g["crate_post"]), str(g["table"]),
                         str(g["unknown"]), r["dominant_ref"], str(r["out_worst_src"]), "-" if kit.endswith("SPLIT") else ("on" if r["nbc_active"] else "off"), "HOLD" if r["hold"] else "-", f"{r['penetration_to_mm']:.1f}",
                         f"{r['palm_target_gt_mm']:.0f}", f"{r['correction_mm']:.1f}"])
            kinds.append(kit)
    fig, ax = plt.subplots(figsize=(17.5, 0.3 * (len(body) + 2)))
    FS.style_axes(fig, ax); ax.axis("off")
    tb = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.25)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor(FS.GRID_INK)
        if ri == 0:
            cell.set_facecolor(FS.SEQ_BLUE[1]); cell.set_text_props(fontweight="bold"); cell.set_height(cell.get_height() * 1.9)
        elif kinds[ri - 1] == "IFG-APPLE-NBC":
            cell.set_facecolor("#f3f2ee")
        if ri > 0 and ci == 13 and body[ri - 1][13] == "HOLD":
            cell.set_text_props(color=FS.CATEGORICAL[7], fontweight="bold")
    n_split = sum(1 for k in kinds if k == "IFG-APPLE-SPLIT")
    ax.set_title(T(f"ep211 요청 10–16: 위 {n_split}줄 = SPLIT, 회색 {len(kinds) - n_split}줄 = NBC (이웃 접촉 20 mm). 행 수 = 정책 청크 16 step 의 활성 행 (h < 0), 출처 = R1",
                   f"ep211 requests 10-16: top {n_split} rows SPLIT, grey rows NBC; counts = active rows on the policy chunk, source = R1"),
                 fontsize=8.5, loc="left")
    return save(fig, a.fig / "ifg-precheck-a-ep211-table.png", {"columns": cols, "rows": body, "kit_per_row": kinds, "A4": a4}, FS, plt)


if __name__ == "__main__":
    main()
