"""T44e V-E2e aggregation (verifier). Reads raw_demo/ raw_lb/ (e2e_measure.py), E2d raw (reproduction), timing/agg_timing.json.
Definitions identical to E2d/agg_e2d.py (T44c-A6): margin 10 mm; clearance = value - r_finger (frame median authorized finger radius);
 cross = (off - r >= margin) & (v - r < margin) over authorized-sphere centres q (demo_last2 = 3rd+4th demo frame of each episode).
 T neg = value < 0 on T (inside GT target hull). D clr = value - r_finger >= margin on D (inside GT distractor hull), frames with any;
 split by height h = z - support plane: h < 0 | 0 <= h <= 15 mm (support band) | h > 15 mm.
 TB = z in [plane-30mm, plane) table band; TA = (plane, plane+10mm] outside every hull.
 table optimistic: TB flips (off < 0 & v >= 0) ; TA (v > off + 1e-6).
 new>old: v > off + 1e-6 ;  new<min(old,d): v < min(off, d_main) - 1e-6   (d_main = target-including composite), over D,T,TB,TA,q,grid.
 zero level (T44dA): lowest z >= support plane with target-free distance > 0 minus support z (mm), 1 mm steps; ball centre column /
 ring at fit radius (8 pts, nanmedian) / ring at r+10 mm; median over frames.
 variants v: guard (A5), keep (A7 plane), exclude (A8), exclude_target (A8b); reference off.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, pathlib, collections, numpy as np

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2e'))
E2D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2d'))
IDS = ("gt", "gt_label"); MS = ("ball", "bo_d3")
VS = ("off", "guard", "keep", "exclude", "exclude_target"); CV = VS[1:]
MARGIN, EPS = 0.010, 1e-6
TC = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")


def load(base=D):
    recs = []
    for d in ("raw_demo", "raw_lb"):
        for f in sorted(glob.glob(str(base / d / "*.json"))):
            r = json.load(open(f)); r["_npz"] = f[:-5] + ".npz"; r["_dir"] = d; recs.append(r)
    by = collections.defaultdict(list)
    for r in recs:
        if r["set"] == "demo": by[r["source"]["episode_index"]].append(r["seq"])
    for r in recs:
        if r["set"] == "demo":
            fr = sorted(by[r["source"]["episode_index"]]); r["_rank"] = len(fr) - 1 - fr.index(r["seq"])
        else: r["_rank"] = None
    return recs


def pct(a, p, s=1.0, nd=3):
    a = np.asarray(a, np.float64)
    return round(float(np.percentile(a, p)) * s, nd) if len(a) else None


def agg(rows, ident, m, v):
    a = collections.Counter()
    sumTA = 0.0
    for r in rows:
        z = np.load(r["_npz"])
        if f"{ident}__q_d_{m}|off" not in z.files:
            continue
        rf = float(z["r_finger"]); ra = z["auth_radii"].astype(np.float64)
        a["n_frames"] += 1
        for s in ("D", "T", "TB", "TA", "q", "grid"):
            k = f"{ident}__{s}_d_{m}|{v}"
            if k not in z.files: continue
            x = z[k].astype(np.float64); xo = z[f"{ident}__{s}_d_{m}|off"].astype(np.float64)
            dm = z[f"{ident}__{s}_dmain"].astype(np.float64)
            a[f"{s}_n"] += len(x)
            a[f"{s}_neg"] += int((x < 0).sum()); a[f"{s}_neg_frames"] += int((x < 0).any())
            a[f"{s}_new_gt_old"] += int((x > xo + EPS).sum())
            a[f"{s}_new_lt_min_old_d"] += int((x < np.minimum(xo, dm) - EPS).sum())
            if v != "off":
                t = z[f"{ident}__{s}_tier_{m}|{v}"]
                a[f"{s}_ans_target_free_coarse"] += int((t == TC.index("target_free_coarse")).sum())
            if s == "T":
                h = z[f"{ident}__T_height"].astype(np.float64) if f"{ident}__T_height" in z.files else None
                if h is not None:
                    a["T_neg_h_lt_20mm"] += int(((x < 0) & (h < 0.020)).sum())
                    a["T_neg_h_ge_20mm"] += int(((x < 0) & (h >= 0.020)).sum())
            if s == "D":
                c = x - rf >= MARGIN
                a["D_clr_ge_margin_pts"] += int(c.sum()); a["D_clr_ge_margin_frames"] += int(c.any())
                a["D_pos"] += int((x > 0).sum())
                h = z[f"{ident}__D_height"].astype(np.float64) if f"{ident}__D_height" in z.files else None
                if h is not None:
                    for nm, sel in (("below", h < 0), ("band0_15", (h >= 0) & (h <= 0.015)), ("above15", h > 0.015)):
                        cc = c & sel
                        a[f"D_clr_{nm}_pts"] += int(cc.sum()); a[f"D_clr_{nm}_frames"] += int(cc.any())
                        a[f"D_{nm}_n"] += int(sel.sum())
            if s == "TB":
                a["TB_optimistic_flip"] += int(((xo < 0) & (x >= 0)).sum())
            if s == "TA":
                sumTA += float((x - xo).sum() * 1e3)
            if s == "q":
                rr = ra
                cr = (xo - rr >= MARGIN) & (x - rr < MARGIN)
                a["q_cross"] += int(cr.sum()); a["q_cross_frames"] += int(cr.any())
                a["q_below_margin"] += int((x - rr < MARGIN).sum())
                if r["_rank"] in (0, 1):
                    a["last2_q_cross"] += int(cr.sum()); a["last2_q_cross_frames"] += int(cr.any()); a["last2_frames"] += 1
                if r["_rank"] == 0:
                    a["last1_q_cross"] += int(cr.sum()); a["last1_q_cross_frames"] += int(cr.any()); a["last1_frames"] += 1
            if s == "grid":
                rr = np.repeat(ra, 125)
                cr = (xo - rr >= MARGIN) & (x - rr < MARGIN)
                a["grid_cross"] += int(cr.sum())
    a["TA_mean_delta_vs_off_sum_mm"] = round(sumTA, 4)
    return dict(a)


def zero_level(rows, ident, m, v):
    out = {}
    for key in ("centre", "ringfit", "ringout"):
        vals = [r["identities"][ident]["zero_level_mm"][f"{m}|{v}"][key] for r in rows
                if "zero_level_mm" in r["identities"][ident]]
        vals = np.asarray(vals, np.float64)
        out[key] = {"n": int(len(vals)), "nan": int(np.isnan(vals).sum()), "median": pct(vals[~np.isnan(vals)], 50, 1, 2),
                    "p90": pct(vals[~np.isnan(vals)], 90, 1, 2), "mean": round(float(np.nanmean(vals)), 3) if len(vals) else None}
    return out


def gradient(rows, ident, m, v):
    P = collections.defaultdict(list)
    for r in rows:
        z = np.load(r["_npz"])
        if f"{ident}__gc_{m}_{v}_cos" not in z.files: continue
        for k in ("cos", "norm_an", "norm_fd", "same_branch", "value_answer_diff", "grad_vs_layer_absdiff", "layer_cos", "src"):
            P[k].append(z[f"{ident}__gc_{m}_{v}_{k}"])
    if not P: return {"n": 0}
    A = {k: np.concatenate(x) for k, x in P.items()}
    same = A["same_branch"].astype(bool)
    return {"n": int(len(A["cos"])), "frac_same_branch": round(float(same.mean()), 4),
            "grad_equals_coarse_layer_max_absdiff": float(A["grad_vs_layer_absdiff"].max()),
            "value_answer_vs_target_free_distance_max_absdiff_m": float(A["value_answer_diff"].max()),
            "cos_median_all": pct(A["cos"], 50, 1, 4), "cos_p5_all": pct(A["cos"], 5, 1, 4),
            "cos_median_same_branch": pct(A["cos"][same], 50, 1, 4), "cos_p5_same_branch": pct(A["cos"][same], 5, 1, 4),
            "ref_layer_cos_median_same_branch": pct(A["layer_cos"][same], 50, 1, 4),
            "ref_layer_cos_p5_same_branch": pct(A["layer_cos"][same], 5, 1, 4),
            "eikonal_fd_median_same_branch": pct(A["norm_fd"][same], 50, 1, 4)}


def repro(rows):
    """E2d reproduction: this run off/guard/keep vs E2d off/guard/plane (same frame key), max abs diff over D,T,q,grid."""
    out = {}
    pairs = (("off", "off"), ("guard", "guard"), ("keep", "plane"))
    for ident in IDS:
        for m in MS:
            mx = {f"{a}_vs_E2d_{b}": 0.0 for a, b in pairs}; n = 0; nf = 0
            for r in rows:
                f = E2D / r["_dir"] / (r["key"] + ".npz")
                if not f.exists(): continue
                a_ = np.load(r["_npz"]); b_ = np.load(f); nf += 1
                for s in ("D", "T", "q", "grid"):
                    for va, vb in pairs:
                        ka, kb = f"{ident}__{s}_d_{m}|{va}", f"{ident}__{s}_d_{m}|{vb}"
                        if ka in a_.files and kb in b_.files:
                            n += len(a_[ka])
                            mx[f"{va}_vs_E2d_{vb}"] = max(mx[f"{va}_vs_E2d_{vb}"], float(np.abs(a_[ka].astype(np.float64) - b_[kb]).max())) if a_[ka].shape == b_[kb].shape else float("inf")
            out[f"{ident}|{m}"] = {**mx, "n_frames_compared": nf, "n_values_compared": n}
    return out


def coarse_counts(rows, ident):
    out = {}
    for m in MS:
        for v in CV:
            L = [r["identities"][ident]["coarse"][f"{m}|{v}"] for r in rows if "coarse" in r["identities"][ident]]
            keys = ("n_removed", "n_sign_candidates", "n_sign_forced", "n_shared_kept", "n_seeds_in_ball")
            o = {k: int(sum(int(x.get(k, 0) or 0) for x in L)) for k in keys}
            sup = [x.get("support") for x in L if isinstance(x.get("support"), dict)]
            if sup:
                o["support_applied_frames"] = int(sum(bool(s.get("applied")) for s in sup))
                o["support_n_removed"] = int(sum(int(s.get("n_removed", 0) or 0) for s in sup))
                o["support_n_seeds"] = int(sum(int(s.get("n_seeds", 0) or 0) for s in sup))
                fps = [s["footprint"] for s in sup if isinstance(s.get("footprint"), dict)]
                if fps:
                    o["footprint"] = {k: int(sum(int(f.get(k, 0) or 0) for f in fps)) for k in
                                      ("n_band_seeds", "n_in_target", "n_overlap_kept", "n_outside_target_kept")}
            out[f"{m}|{v}"] = o
    return out


def main():
    recs = load()
    sets = {"demo": [r for r in recs if r["set"] == "demo"], "lb": [r for r in recs if r["set"] == "lb"]}
    sets["demo_last2"] = [r for r in sets["demo"] if r["_rank"] in (0, 1)]
    out = {"definitions": __doc__, "n_frames": {k: len(v) for k, v in sets.items()}, "sets": {}}
    for s, rows in sets.items():
        so = {}
        for ident in IDS:
            o = {"skipped": int(sum("skipped" in r["identities"][ident] for r in rows)), "methods": {}}
            for m in MS:
                o["methods"][m] = {v: agg(rows, ident, m, v) for v in VS}
                if s in ("demo", "lb"):
                    o["methods"][m]["gradient"] = {v: gradient(rows, ident, m, v) for v in CV}
                    o["methods"][m]["zero_level_mm"] = {v: zero_level(rows, ident, m, v) for v in VS}
            if s in ("demo", "lb"):
                o["coarse_counts"] = coarse_counts(rows, ident)
            so[ident] = o
        out["sets"][s] = so
    out["repro_vs_E2d"] = {s: repro(sets[s]) for s in ("demo", "lb")}
    tp = D / "timing" / "agg_timing.json"
    if tp.exists(): out["timing"] = json.load(open(tp))
    json.dump(out, open(D / "agg_e2e.json", "w"), indent=1, ensure_ascii=False)
    for s in ("demo", "lb"):
        for ident in IDS:
            for m in MS:
                r = out["sets"][s][ident]["methods"][m]
                print(s, ident, m, {v: (r[v].get("T_neg"), r[v].get("D_clr_ge_margin_frames"), r[v].get("TB_neg"), r[v].get("q_cross"), r[v].get("last2_q_cross")) for v in VS})


if __name__ == "__main__":
    main()
