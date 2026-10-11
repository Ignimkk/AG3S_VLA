"""T44c-A6 aggregation (verifier). Reads raw_demo/ raw_lb/ (e2d_measure.py), E2c raw (guard reproduction), timing/agg_timing.json.
Definitions identical to E2c/agg_e2c.py (D-sign2): margin 10 mm; clearance = value - r_finger(frame median authorized finger radius);
 cross = (off - r >= margin) & (v - r < margin) over authorized-sphere centres q (demo_last2 = 3rd+4th demo frame of each episode).
 T neg = value < 0 on T (inside GT target hull). D clr = value - r_finger >= margin on D (inside GT distractor hull), frames with any.
 TB = z in [plane-30mm, plane) table band; TA = (plane, plane+10mm] outside every hull.
 table optimistic: TB flips (off < 0 & v >= 0) ; TA (v > off + 1e-6) ; also TB/TA v > off + 1e-6.
 new>old: v > off + 1e-6 ;  new<min(old,d): v < min(off, d_main) - 1e-6   (d_main = target-including composite), over D,T,TB,TA,q,grid.
 variants v: guard (A5), plane (A6); reference off.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, pathlib, collections, numpy as np

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2d'))
E2C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2c'))
IDS = ("gt", "gt_label"); MS = ("ball", "bo_d3"); VS = ("off", "guard", "plane")
MARGIN, EPS = 0.010, 1e-6
TC = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")


def load():
    recs = []
    for d in ("raw_demo", "raw_lb"):
        for f in sorted(glob.glob(str(D / d / "*.json"))):
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
        if f"{ident}__D_d_{m}|off" not in z.files and f"{ident}__q_d_{m}|off" not in z.files:
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
    """guard of this run vs E2c '|on' arrays, off vs E2c '|off' (same frame key), max abs diff over all sets."""
    out = {}
    for ident in IDS:
        for m in MS:
            mx = {"off": 0.0, "guard_vs_E2c_on": 0.0}; n = 0; nf = 0
            for r in rows:
                if r["set"] == "lb" and False: pass
                f = E2C / r["_dir"] / (r["key"] + ".npz")
                if not f.exists(): continue
                a = np.load(r["_npz"]); b = np.load(f)
                nf += 1
                for s in ("D", "T", "q", "grid"):
                    ka, kb = f"{ident}__{s}_d_{m}|off", f"{ident}__{s}_d_{m}|off"
                    if ka in a.files and kb in b.files:
                        mx["off"] = max(mx["off"], float(np.abs(a[ka].astype(np.float64) - b[kb]).max())) if a[ka].shape == b[kb].shape else float("inf")
                    ka, kb = f"{ident}__{s}_d_{m}|guard", f"{ident}__{s}_d_{m}|on"
                    if ka in a.files and kb in b.files:
                        n += len(a[ka])
                        mx["guard_vs_E2c_on"] = max(mx["guard_vs_E2c_on"], float(np.abs(a[ka].astype(np.float64) - b[kb]).max())) if a[ka].shape == b[kb].shape else float("inf")
            out[f"{ident}|{m}"] = {**mx, "n_frames_compared": nf, "n_values_compared": n}
    return out


def coarse_counts(rows, ident):
    out = {}
    for m in MS:
        for fl in ("guard", "plane"):
            L = [r["identities"][ident]["coarse"][f"{m}|{fl}"] for r in rows if "coarse" in r["identities"][ident]]
            keys = ("n_removed", "n_sign_candidates", "n_sign_forced", "n_shared_kept", "n_seeds_in_ball")
            out[f"{m}|{fl}"] = {k: int(sum(int(x.get(k, 0) or 0) for x in L)) for k in keys}
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
                    o["methods"][m]["gradient"] = {v: gradient(rows, ident, m, v) for v in ("guard", "plane")}
            if s in ("demo", "lb"):
                o["coarse_counts"] = coarse_counts(rows, ident)
            so[ident] = o
        out["sets"][s] = so
    out["repro_vs_E2c"] = {s: repro(sets[s]) for s in ("demo", "lb")}
    tp = D / "timing" / "agg_timing.json"
    if tp.exists(): out["timing"] = json.load(open(tp))
    json.dump(out, open(D / "agg_e2d.json", "w"), indent=1, ensure_ascii=False)
    for s in ("demo", "lb"):
        for ident in IDS:
            for m in MS:
                r = out["sets"][s][ident]["methods"][m]
                print(s, ident, m, {v: (r[v].get("T_neg"), r[v].get("D_clr_ge_margin_frames"), r[v].get("TB_neg"), r[v].get("q_cross")) for v in VS})


if __name__ == "__main__":
    main()
