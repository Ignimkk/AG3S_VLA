"""T44c D-sign2 aggregation (verifier). Reads raw_demo/ raw_lb/ (e2c_measure.py) [+ timing/agg_timing.json], writes agg_e2c.json.

Definitions (fixed before reading results):
 safety (T44E2b): D = fine-lattice centres inside GT distractor hulls within ball centre +- (r+60 mm);
   positive = value > 0; clearance >= margin = value - r_finger(frame median authorized finger radius) >= 10 mm.
 over-blocking: authorized spheres (q) [and grid = q + 125 offsets, ROI subset]; delta = d_off - d_on (m);
   decreased = delta > 1e-6 m; cross = (d_off - r >= margin) & (d_on - r < margin).
   shared-caused = decreased & (d_on < d_on_ns - 1e-6) where on_ns = same coarse tier with shared preservation disabled.
   near-shared = decreased & (dist(sphere centre, nearest shared-kept coarse voxel centre) - r <= 20 mm).
   on_eq_main = decreased & |d_on - d_main| <= 1 mm (target-free answer equals the target-including composite).
   gtNT = centre distance to nearest GT non-target, non-robot point (depth pixels + fruit/crate mesh samples, unsigned);
   underestimate = d < gtNT - 10 mm; with gtT < gtNT = the GT target is the nearer surface.
   grasp-adjacent subsets: demo last1 = frame == grasp_close_start_frame (4th of 4), last2 = 3rd+4th.
 target interior: T = fine-lattice centres inside GT target hull (same box); neg = value < 0.
 gradient: points where on-composition answered target_free_coarse (D, T, grid; <=4000/frame/identity/method sampled);
   cos(analytic target_free_gradient, central FD h=1 mm); same_branch = all 6 probes answered by the same tier.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, collections, numpy as np

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2c'))
IDS = ("gt", "gt_label")
MS = ("ball", "bo_d3", "oracle")
MARGIN = 0.010
EPS = 1e-6
TC = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")


def pct(a, p, s=1.0, nd=3):
    a = np.asarray(a, np.float64)
    return round(float(np.percentile(a, p)) * s, nd) if len(a) else None


def dist(a, s=1e3):
    a = np.asarray(a, np.float64)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "median": pct(a, 50, s), "p95": pct(a, 95, s), "max": round(float(a.max()) * s, 3),
            "min": round(float(a.min()) * s, 3)}


def load():
    recs = []
    for d in ("raw_demo", "raw_lb"):
        for f in sorted(glob.glob(str(D / d / "*.json"))):
            r = json.load(open(f)); r["_npz"] = f[:-5] + ".npz"; recs.append(r)
    # grasp-adjacent rank within demo episode
    by_ep = collections.defaultdict(list)
    for r in recs:
        if r["set"] == "demo":
            by_ep[r["source"]["episode_index"]].append(r["seq"])
    for r in recs:
        if r["set"] == "demo":
            fr = sorted(by_ep[r["source"]["episode_index"]])
            r["_rank_from_end"] = len(fr) - 1 - fr.index(r["seq"])
        else:
            r["_rank_from_end"] = None
    return recs


def safety(rows, ident):
    out = {"n_frames": 0, "n_points": 0}
    acc = collections.defaultdict(lambda: collections.Counter())
    vmax = collections.defaultdict(float)
    tiers = collections.defaultdict(collections.Counter)
    for r in rows:
        z = np.load(r["_npz"])
        k = f"{ident}__D_pts"
        if k not in z.files:
            continue
        out["n_frames"] += 1
        rf = float(z["r_finger"])
        dep = z[f"{ident}__D_depth"].astype(np.float64)
        out["n_points"] += len(dep)
        vals = {"main_composite": z[f"{ident}__D_dmain"]}
        for m in MS:
            for c in ("off", "on"):
                vals[f"{m}|{c}"] = z[f"{ident}__D_d_{m}|{c}"]
            if m != "oracle":
                vals[f"{m}|on_ns"] = z[f"{ident}__D_d_{m}|on_ns"]
            t = z[f"{ident}__D_tier_{m}|on"]
            for i, n in enumerate(TC):
                tiers[m][n] += int((t == i).sum())
        for kk, v in vals.items():
            v = v.astype(np.float64)
            pos = v > 0
            clr = v - rf >= MARGIN
            a = acc[kk]
            a["n_positive"] += int(pos.sum()); a["n_frames_with_positive"] += int(pos.any())
            a["n_clearance_ge_margin"] += int(clr.sum()); a["n_frames_clearance_ge_margin"] += int(clr.any())
            a["n_positive_depth_ge_10mm"] += int((pos & (dep >= 0.010)).sum())
            a["n_frames_positive_depth_ge_10mm"] += int((pos & (dep >= 0.010)).any())
            a["n_negative"] += int((v < 0).sum())
            vmax[kk] = max(vmax[kk], float(v.max()))
    for kk, a in acc.items():
        out[kk] = {**dict(a), "frac_positive": round(a["n_positive"] / max(1, out["n_points"]), 4),
                   "max_value_mm": round(vmax[kk] * 1e3, 2)}
    out["answer_tier_on"] = {m: {k: v for k, v in tiers[m].items() if v} for m in MS}
    return out


def target_interior(rows, ident):
    out = {}
    for m in MS:
        a = collections.Counter(); dec = []
        for r in rows:
            z = np.load(r["_npz"])
            if f"{ident}__T_pts" not in z.files:
                continue
            off = z[f"{ident}__T_d_{m}|off"].astype(np.float64); on = z[f"{ident}__T_d_{m}|on"].astype(np.float64)
            main = z[f"{ident}__T_dmain"].astype(np.float64)
            a["n_frames"] += 1; a["n_points"] += len(on)
            a["n_on_negative"] += int((on < 0).sum()); a["n_off_negative"] += int((off < 0).sum())
            a["n_main_negative"] += int((main < 0).sum())
            a["n_frames_on_negative"] += int((on < 0).any())
            a["n_on_negative_off_nonneg"] += int(((on < 0) & (off >= 0)).sum())
            d = off - on
            a["n_decreased"] += int((d > EPS).sum())
            a["n_on_gt_off"] += int((on > off + EPS).sum())
            dec.append(d[d > EPS])
            t = z[f"{ident}__T_tier_{m}|on"]
            a["n_answer_target_free_coarse"] += int((t == TC.index("target_free_coarse")).sum())
            neg = on < 0
            P = z[f"{ident}__T_pts"].astype(np.float64)
            sz = r["identities"][ident]["ball"].get("support_plane_z_m")
            h = P[:, 2] - (sz if sz is not None else -np.inf)
            dep = z[f"{ident}__T_depth"].astype(np.float64)
            inb = z[f"{ident}__T_inball"]
            cf = z[f"{ident}__T_cfree_{m}"].astype(np.float64)
            a["n_on_negative_height_lt_20mm"] += int((neg & (h < 0.020)).sum())
            a["n_on_negative_height_ge_20mm"] += int((neg & (h >= 0.020)).sum())
            a["n_on_negative_in_ball"] += int((neg & inb).sum())
            a["n_on_negative_depth_ge_10mm"] += int((neg & (dep >= 0.010)).sum())
            a["n_on_negative_coarse_free_negative"] += int((neg & (cf < 0)).sum())
            a["n_points_height_lt_20mm"] += int((h < 0.020).sum())
            for i, n in enumerate(TC):
                a[f"on_negative_answer_{n}"] += int((neg & (t == i)).sum())
            a["n_points_coarse_free_negative"] += int((cf < 0).sum())
        out[m] = {**dict(a), "decrease_mm": dist(np.concatenate(dec) if dec else [])}
    return out


def blocking(rows, ident, qn="q", roi_only=False):
    out = {}
    for m in MS:
        a = collections.Counter(); dec_all = []; per_frame = []
        for r in rows:
            z = np.load(r["_npz"])
            k = f"{ident}__{qn}_d_{m}|off"
            if k not in z.files:
                continue
            rad = z["auth_radii"].astype(np.float64)
            if qn == "grid":
                rad = np.repeat(rad, 125)
            sel = z[f"{ident}__{qn}_roi"] if roi_only else np.ones(len(rad), bool)
            off = z[k].astype(np.float64)[sel]; on = z[f"{ident}__{qn}_d_{m}|on"].astype(np.float64)[sel]
            main = z[f"{ident}__{qn}_dmain"].astype(np.float64)[sel]; rr = rad[sel]
            d = off - on
            decd = d > EPS
            a["n_frames"] += 1; a["n_queries"] += int(len(d))
            a["n_decreased"] += int(decd.sum()); a["n_decreased_gt_1mm"] += int((d > 1e-3).sum())
            a["n_frames_decreased"] += int(decd.any())
            cross = (off - rr >= MARGIN) & (on - rr < MARGIN)
            a["n_cross_margin"] += int(cross.sum()); a["n_frames_cross_margin"] += int(cross.any())
            a["n_off_below_margin"] += int((off - rr < MARGIN).sum()); a["n_on_below_margin"] += int((on - rr < MARGIN).sum())
            a["n_on_negative"] += int((on < 0).sum()); a["n_off_negative"] += int((off < 0).sum())
            a["n_on_gt_off"] += int((on > off + EPS).sum())
            a["n_on_lt_min_off_main"] += int((on < np.minimum(off, main) - EPS).sum())
            t = z[f"{ident}__{qn}_tier_{m}|on"][sel]
            a["n_answer_target_free_coarse"] += int((t == TC.index("target_free_coarse")).sum())
            a["n_decreased_on_eq_main_1mm"] += int((decd & (np.abs(on - main) <= 1e-3)).sum())
            gnt = z[f"{qn}_gt_nontarget_dist"].astype(np.float64)[sel]
            gtt = z[f"{qn}_gt_target_dist"].astype(np.float64)[sel]
            a["n_on_lt_gtNT_minus_10mm"] += int((on < gnt - 0.010).sum())
            a["n_off_lt_gtNT_minus_10mm"] += int((off < gnt - 0.010).sum())
            a["n_on_lt_gtNT_minus_10mm_and_gtT_lt_gtNT"] += int(((on < gnt - 0.010) & (gtt < gnt)).sum())
            a["n_off_lt_gtNT_minus_10mm_and_gtT_lt_gtNT"] += int(((off < gnt - 0.010) & (gtt < gnt)).sum())
            if m != "oracle":
                ns = z[f"{ident}__{qn}_d_{m}|on_ns"].astype(np.float64)[sel]
                sc = decd & (on < ns - EPS)
                a["n_decreased_shared_caused"] += int(sc.sum())
                a["n_cross_margin_shared_caused"] += int((cross & (on < ns - EPS)).sum())
                ds = z[f"{ident}__{qn}_dist_sharedkept_{m}"].astype(np.float64)[sel]
                a["n_decreased_near_shared"] += int((decd & (ds - rr <= 0.020)).sum())
            dec_all.append(d[decd])
            per_frame.append(float(d.max()) if len(d) else 0.0)
        dd = np.concatenate(dec_all) if dec_all else np.zeros(0)
        o = {**dict(a), "decrease_mm_among_decreased": dist(dd), "per_frame_max_decrease_mm": dist(per_frame)}
        if m != "oracle" and a["n_decreased"]:
            o["frac_decreased_shared_caused"] = round(a["n_decreased_shared_caused"] / a["n_decreased"], 4)
            o["frac_decreased_near_shared"] = round(a["n_decreased_near_shared"] / a["n_decreased"], 4)
        out[m] = o
    return out


def vs_oracle(rows, ident):
    """(3) falsely-feasible vs oracle, q + grid ROI subset (T44E2b criterion (3) definition), off and on."""
    out = {}
    for comp in ("off", "on"):
        for m in ("ball", "bo_d3"):
            c = collections.Counter(); over = []
            for qn in ("q", "grid"):
                for r in rows:
                    z = np.load(r["_npz"])
                    if f"{ident}__{qn}_d_{m}|off" not in z.files:
                        continue
                    roi = z[f"{ident}__{qn}_roi"]
                    rad = np.repeat(z["auth_radii"].astype(np.float64), 1 if qn == "q" else 125)[roi]
                    d = z[f"{ident}__{qn}_d_{m}|{comp}"].astype(np.float64)[roi]
                    o = z[f"{ident}__{qn}_d_oracle|{comp}"].astype(np.float64)[roi]
                    c[f"{qn}_n"] += int(len(d))
                    c[f"{qn}_ff"] += int(((d - rad >= MARGIN) & (o - rad < MARGIN)).sum())
                    c[f"{qn}_fi"] += int(((d - rad < MARGIN) & (o - rad >= MARGIN)).sum())
                    if qn == "q":
                        over.append(d - o)
            ov = np.concatenate(over) if over else np.zeros(0)
            out[f"{m}|{comp}"] = {**dict(c), "q_over_p95_mm": pct(ov, 95, 1e3), "q_over_max_mm": round(float(ov.max()) * 1e3, 3) if len(ov) else None}
    return out


def coarse_seeds(rows, ident):
    out = {}
    hit_ball = None
    for m in ("ball", "bo_d3", "ball_ns", "bo_d3_ns", "oracle"):
        L = []
        for r in rows:
            c = r["identities"][ident].get("coarse", {}).get(m)
            if c is not None:
                L.append(c)
        cnt = lambda k: [int(x["counts"].get(k, 0)) for x in L]  # noqa: E731
        rec = [x["target_recall"] for x in L if x["target_recall"] is not None]
        lost = np.array([x["distractor_seeds_lost"] for x in L])
        lab = collections.Counter()
        for x in L:
            lab.update(x.get("shared_kept_label_counts") or {})
        o = {"n_frames": len(L),
             "n_shared_kept": {"sum": int(sum(cnt("n_shared_kept"))), "median": pct(cnt("n_shared_kept"), 50),
                               "p90": pct(cnt("n_shared_kept"), 90), "max": int(max(cnt("n_shared_kept"), default=0)),
                               "n_frames_gt0": int(sum(v > 0 for v in cnt("n_shared_kept")))},
             "n_seeds_shared_sum": int(sum(cnt("n_seeds_shared"))), "n_shared_cell_sum": int(sum(cnt("n_shared_cell"))),
             "n_seeds_in_ball_sum": int(sum(cnt("n_seeds_in_ball"))), "n_removed_sum": int(sum(cnt("n_removed"))),
             "n_sign_candidates_sum": int(sum(cnt("n_sign_candidates"))),
             "n_kept_nontarget_sum": int(sum(cnt("n_kept_nontarget"))), "n_ambiguous_sum": int(sum(cnt("n_ambiguous"))),
             "n_unowned_removed_sum": int(sum(cnt("n_unowned_removed"))),
             "shared_kept_gt_label_counts": dict(lab),
             "target_recall_median": pct(rec, 50, 1, 4), "target_recall_p10": pct(rec, 10, 1, 4),
             "distractor_seeds_lost_sum": int(lost.sum()),
             "distractor_seeds_in_ball_sum": int(sum(x["distractor_seeds_in_ball"] for x in L)),
             "support_seeds_lost_sum": int(sum(x["support_seeds_lost"] for x in L)),
             "build_ms_once_median": pct([x["build_ms_once"] for x in L if x["build_ms_once"] == x["build_ms_once"]], 50)}
        if m == "ball":
            hit_ball = lost
        elif m == "bo_d3" and hit_ball is not None:
            hit = hit_ball >= 1
            o["n_frames_ball_lost_distractor"] = int(hit.sum())
            if hit.any():
                o["loss_reduction_pooled_on_hit_frames"] = round(1 - lost[hit].sum() / hit_ball[hit].sum(), 4)
        out[m] = o
    return out


def gradient(rows, ident):
    out = {}
    for m in MS:
        P = collections.defaultdict(list)
        for r in rows:
            z = np.load(r["_npz"])
            if f"{ident}__gc_{m}_cos" not in z.files:
                continue
            for k in ("cos", "norm_an", "norm_fd", "same_branch", "value", "value_answer_diff", "src",
                      "grad_vs_layer_absdiff", "layer_cos", "maincoarse_cos", "maincoarse_norm_fd"):
                P[k].append(z[f"{ident}__gc_{m}_{k}"])
        if not P:
            out[m] = {"n": 0}; continue
        A = {k: np.concatenate(v) for k, v in P.items()}
        same = A["same_branch"].astype(bool)
        smooth = same & (np.abs(A["value"]) >= 0.020)
        o = {"n": int(len(A["cos"])), "n_by_set": {n: int((A["src"] == i).sum()) for i, n in enumerate(("D", "T", "grid"))},
             "grad_equals_coarse_layer_max_absdiff": round(float(A["grad_vs_layer_absdiff"].max()), 9),
             "value_answer_vs_target_free_distance_max_absdiff_m": float(A["value_answer_diff"].max()),
             "frac_same_branch": round(float(same.mean()), 4)}
        for name, s in (("all", np.ones(len(same), bool)), ("same_branch", same), ("smooth_same_branch_absv_ge_20mm", smooth)):
            o[name] = {"n": int(s.sum()), "cos_median": pct(A["cos"][s], 50, 1, 4), "cos_p5": pct(A["cos"][s], 5, 1, 4),
                       "frac_cos_ge_0.9": round(float((A["cos"][s] >= 0.9).mean()), 4) if s.any() else None,
                       "eikonal_fd_median": pct(A["norm_fd"][s], 50, 1, 4), "eikonal_fd_p5": pct(A["norm_fd"][s], 5, 1, 4),
                       "eikonal_analytic_median": pct(A["norm_an"][s], 50, 1, 4), "eikonal_analytic_p5": pct(A["norm_an"][s], 5, 1, 4),
                       "ref_layer_cos_median": pct(A["layer_cos"][s], 50, 1, 4), "ref_layer_cos_p5": pct(A["layer_cos"][s], 5, 1, 4),
                       "ref_main_coarse_cos_median": pct(A["maincoarse_cos"][s], 50, 1, 4),
                       "ref_main_coarse_cos_p5": pct(A["maincoarse_cos"][s], 5, 1, 4),
                       "ref_main_coarse_eikonal_fd_median": pct(A["maincoarse_norm_fd"][s], 50, 1, 4),
                       "ref_main_coarse_eikonal_fd_p5": pct(A["maincoarse_norm_fd"][s], 5, 1, 4)}
        for i, n in enumerate(("D", "T", "grid")):
            s = A["src"] == i
            o[f"by_set_{n}"] = {"n": int(s.sum()), "cos_median": pct(A["cos"][s], 50, 1, 4), "cos_p5": pct(A["cos"][s], 5, 1, 4),
                                "eikonal_fd_median": pct(A["norm_fd"][s], 50, 1, 4), "eikonal_fd_p5": pct(A["norm_fd"][s], 5, 1, 4)}
        out[m] = o
    return out


def main():
    recs = load()
    sets = {"demo": [r for r in recs if r["set"] == "demo"], "lb": [r for r in recs if r["set"] == "lb"]}
    sets["demo_last1"] = [r for r in sets["demo"] if r["_rank_from_end"] == 0]
    sets["demo_last2"] = [r for r in sets["demo"] if r["_rank_from_end"] in (0, 1)]
    out = {"definitions": __doc__, "n_frames": {k: len(v) for k, v in sets.items()}, "sets": {}}
    for s, rows in sets.items():
        so = {}
        for ident in IDS:
            o = {"identity_resolved": dict(collections.Counter(r["identities"][ident].get("identity_mode_resolved") for r in rows)),
                 "skipped": int(sum("skipped" in r["identities"][ident] for r in rows))}
            o["blocking_q"] = blocking(rows, ident, "q")
            o["blocking_grid_roi"] = blocking(rows, ident, "grid", roi_only=True)
            o["target_interior"] = target_interior(rows, ident)
            o["coarse_seeds"] = coarse_seeds(rows, ident)
            if s in ("demo", "lb"):
                o["safety"] = safety(rows, ident)
                o["vs_oracle"] = vs_oracle(rows, ident)
                o["gradient"] = gradient(rows, ident)
            so[ident] = o
        out["sets"][s] = so
    tp = D / "timing" / "agg_timing.json"
    if tp.exists():
        out["timing"] = json.load(open(tp))
    json.dump(out, open(D / "agg_e2c.json", "w"), indent=1, ensure_ascii=False)
    for s in ("demo", "lb"):
        g = out["sets"][s]["gt_label"]
        if "ball|off" not in g["safety"]:
            continue
        print(s, "safety ball off/on pos frames", g["safety"]["ball|off"]["n_frames_with_positive"], g["safety"]["ball|on"]["n_frames_with_positive"],
              "clr>=m", g["safety"]["ball|off"]["n_clearance_ge_margin"], g["safety"]["ball|on"]["n_clearance_ge_margin"])


if __name__ == "__main__":
    main()
