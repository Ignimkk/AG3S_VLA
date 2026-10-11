"""T44e A8b smoke aggregation (copied from A8/a8_agg.py). Definitions = E2d/agg_e2d.py (D-sign2), identity gt_label, variants off | keep | exclude | exclude_target.
margin 10 mm; clearance = value - r_finger; cross = (off - r >= margin) & (v - r < margin) over authorized-sphere centres q
(demo_last2 = 3rd + 4th demo frame of each episode = grasp 직전). T neg = value < 0 inside GT target hull. D clr = value - r_finger
>= margin inside GT distractor hull. TB = z in [plane-30 mm, plane); TA = (plane, plane+10 mm] outside every hull.
property: new > off + 1e-6 ; new < min(off, d_main) - 1e-6 over D T TB TA q grid.
vs keep: points where exclude > keep + 1e-6 (less conservative than A7) — counted per set with height / in-ball split.
zero level: T44dA table_lift definition (centre · ring at fit radius · ring at r + 10 mm), median over frames.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, pathlib, collections, numpy as np

D = pathlib.Path(_ag3s_workspace_path('outputs/impl/T44/A8b'))
D8 = pathlib.Path(_ag3s_workspace_path('outputs/impl/T44/A8'))
IDENT = "gt_label"; MS = ("ball", "bo_d3"); VS = ("off", "keep", "exclude", "exclude_target")
MARGIN, EPS = 0.010, 1e-6
TC = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")


def load():
    recs = []
    for f in sorted(glob.glob(str(D / "raw_demo" / "*.json"))):
        r = json.load(open(f)); r["_npz"] = f[:-5] + ".npz"; recs.append(r)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["source"]["episode_index"]].append(r["seq"])
    for r in recs:
        fr = sorted(by[r["source"]["episode_index"]]); r["_rank"] = len(fr) - 1 - fr.index(r["seq"])
    return recs


def agg(rows, m, v):
    a = collections.Counter(); sumTA = 0.0; nTA = 0
    for r in rows:
        if "skipped" in r: continue
        z = np.load(r["_npz"])
        rf = float(z["r_finger"]); ra = z["auth_radii"].astype(np.float64)
        a["n_frames"] += 1
        for s in ("D", "T", "TB", "TA", "q", "grid"):
            k = f"{IDENT}__{s}_d_{m}|{v}"
            if k not in z.files: continue
            x = z[k].astype(np.float64); xo = z[f"{IDENT}__{s}_d_{m}|off"].astype(np.float64)
            xk = z[f"{IDENT}__{s}_d_{m}|keep"].astype(np.float64)
            dm = z[f"{IDENT}__{s}_dmain"].astype(np.float64)
            a[f"{s}_n"] += len(x)
            a[f"{s}_neg"] += int((x < 0).sum()); a[f"{s}_neg_frames"] += int((x < 0).any())
            a[f"{s}_new_gt_off"] += int((x > xo + EPS).sum())
            a[f"{s}_new_lt_min_off_d"] += int((x < np.minimum(xo, dm) - EPS).sum())
            a[f"{s}_gt_keep"] += int((x > xk + EPS).sum())
            if v != "off":
                t = z[f"{IDENT}__{s}_tier_{m}|{v}"]
                a[f"{s}_ans_target_free_coarse"] += int((t == TC.index("target_free_coarse")).sum())
            if s == "D":
                c = x - rf >= MARGIN
                a["D_clr_ge_margin_pts"] += int(c.sum()); a["D_clr_ge_margin_frames"] += int(c.any())
                h = z[f"{IDENT}__D_height"].astype(np.float64) * 1e3
                lo = c & (h < 15.0); hi = c & (h >= 15.0)
                a["D_clr_ge_margin_h0_15_frames"] += int(lo.any()); a["D_clr_ge_margin_h0_15_pts"] += int(lo.sum())
                a["D_clr_ge_margin_h15up_frames"] += int(hi.any()); a["D_clr_ge_margin_h15up_pts"] += int(hi.sum())
            if s == "TB":
                a["TB_optimistic_flip_vs_off"] += int(((xo < 0) & (x >= 0)).sum())
                a["TB_optimistic_flip_vs_keep"] += int(((xk < 0) & (x >= 0)).sum())
            if s == "TA":
                sumTA += float(x.sum() * 1e3); nTA += len(x)
            if s == "q":
                cr = (xo - ra >= MARGIN) & (x - ra < MARGIN)
                a["q_cross"] += int(cr.sum()); a["q_cross_frames"] += int(cr.any())
                if r["_rank"] in (0, 1):
                    a["last2_q_cross"] += int(cr.sum()); a["last2_q_cross_frames"] += int(cr.any()); a["last2_frames"] += 1
            if s == "grid":
                rr = np.repeat(ra, 125)
                cr = (xo - rr >= MARGIN) & (x - rr < MARGIN)
                a["grid_cross"] += int(cr.sum())
    a["TA_mean_value_mm"] = round(sumTA / max(nTA, 1), 3)
    return dict(a)


def less_conservative_than_keep(rows, m, v="exclude"):
    """Where exclude > keep: per set, count · height above support (p10/50/90 mm) · in-ball fraction · max increase mm."""
    out = {}
    for s in ("D", "T", "TB", "TA"):
        H, IB, DLT = [], [], []
        for r in rows:
            if "skipped" in r: continue
            z = np.load(r["_npz"])
            k = f"{IDENT}__{s}_d_{m}|{v}"
            if k not in z.files: continue
            x = z[k].astype(np.float64); xk = z[f"{IDENT}__{s}_d_{m}|keep"].astype(np.float64)
            sel = x > xk + EPS
            if sel.any():
                H.append(z[f"{IDENT}__{s}_height"][sel]); IB.append(z[f"{IDENT}__{s}_inball"][sel])
                DLT.append((x - xk)[sel])
        if H:
            h = np.concatenate(H) * 1e3; ib = np.concatenate(IB); dl = np.concatenate(DLT) * 1e3
            out[s] = {"n": int(len(h)), "height_mm_p10_50_90": [round(float(np.percentile(h, p)), 1) for p in (10, 50, 90)],
                      "inball_frac": round(float(ib.mean()), 3),
                      "increase_mm_p50_90_max": [round(float(np.percentile(dl, 50)), 2), round(float(np.percentile(dl, 90)), 2),
                                                 round(float(dl.max()), 2)]}
        else:
            out[s] = {"n": 0}
    return out


def zero(rows, m):
    out = {}
    for v in VS:
        L = [r["zero_level_mm"][f"{m}|{v}"] for r in rows if "zero_level_mm" in r]
        out[v] = {k: round(float(np.nanmedian([x[k] for x in L])), 2) for k in ("centre", "ringfit", "ringout")}
        out[v]["centre_p90"] = round(float(np.nanpercentile([x["centre"] for x in L], 90)), 2)
        out[v]["ringout_p90"] = round(float(np.nanpercentile([x["ringout"] for x in L], 90)), 2)
        out[v]["n"] = len(L)
    return out


def counts(rows, m):
    out = {}
    for v in ("keep", "exclude", "exclude_target"):
        L = [r["coarse"][f"{m}|{v}"] for r in rows if "coarse" in r]
        sup = [x.get("support") or {} for x in L]
        out[v] = {"n_removed_ball": int(sum(x.get("n_removed", 0) for x in L)),
                  "n_shared_kept_ball": int(sum(x.get("n_shared_kept", 0) for x in L)),
                  "n_sign_forced": int(sum(x.get("n_sign_forced", 0) for x in L)),
                  "support_n_removed": int(sum(x.get("n_removed", 0) for x in sup)),
                  "support_n_shared_kept": int(sum(x.get("n_shared_kept", 0) for x in sup)),
                  "support_classify_ms_median": (round(float(np.median([x["classify_ms"] for x in sup if "classify_ms" in x])), 3)
                                                 if any("classify_ms" in x for x in sup) else None),
                  "n_support_protected_in_ball": int(sum(x.get("n_support_protected_in_ball", 0) for x in L)),
                  "footprint": {k: int(sum((x.get("footprint") or {}).get(k, 0) for x in sup))
                                for k in ("n_band_seeds", "n_in_target", "n_overlap_kept", "n_outside_target_kept",
                                          "n_nontarget_points")},
                  "frames_with_overlap": int(sum(((x.get("footprint") or {}).get("n_overlap_kept", 0) > 0) for x in sup)),
                  "build_ms_median": round(float(np.median([x["build_ms_once"] for x in L])), 3),
                  "build_ms_p90": round(float(np.percentile([x["build_ms_once"] for x in L], 90)), 3)}
    return out


def bit_identity():
    """A8b smoke 의 off · keep · exclude 배열이 A8 smoke (A8b 코드 이전) 와 같은가 — exclude 경로 불변의 실측."""
    res = collections.Counter()
    for f in sorted(glob.glob(str(D / "raw_demo" / "*.npz"))):
        g = D8 / "raw_demo" / pathlib.Path(f).name
        if not g.exists():
            res["missing"] += 1; continue
        a, b = np.load(f), np.load(g)
        for k in a.files:
            if not any(f"|{v}" in k for v in ("off", "keep", "exclude")) or "exclude_target" in k:
                continue
            if k not in b.files:
                res["key_missing_in_A8"] += 1; continue
            res["equal" if np.array_equal(a[k], b[k]) else "different"] += 1
    return dict(res)


def main():
    recs = load()
    sets = {"demo": recs, "demo_last2": [r for r in recs if r["_rank"] in (0, 1)]}
    out = {"definitions": __doc__, "n_frames": {k: len(v) for k, v in sets.items()},
           "n_skipped": int(sum("skipped" in r for r in recs)), "sets": {}}
    for s, rows in sets.items():
        so = {}
        for m in MS:
            so[m] = {"vars": {v: agg(rows, m, v) for v in VS}, "zero_level_mm": zero(rows, m),
                     "exclude_gt_keep": less_conservative_than_keep(rows, m),
                     "exclude_target_gt_keep": less_conservative_than_keep(rows, m, "exclude_target"),
                     "counts": counts(rows, m)}
        out["sets"][s] = so
    out["bit_identity_vs_A8"] = bit_identity()
    json.dump(out, open(D / "a8b_agg.json", "w"), indent=1, ensure_ascii=False)
    print("bit identity vs A8 raw (keep / exclude / off arrays):", out["bit_identity_vs_A8"])
    for m in MS:
        r = out["sets"]["demo"][m]
        print(m, "zero", {v: (r["zero_level_mm"][v]["centre"], r["zero_level_mm"][v]["ringfit"], r["zero_level_mm"][v]["ringout"]) for v in VS})
        for v in VS:
            x = r["vars"][v]
            print(" ", v, "T_neg", x.get("T_neg"), "D>=m frames", x.get("D_clr_ge_margin_frames"), "pts", x.get("D_clr_ge_margin_pts"),
                  "q_cross", x.get("q_cross"), "/", x.get("q_cross_frames"), "TA mean", x.get("TA_mean_value_mm"),
                  "viol", sum(x.get(f"{s}_new_gt_off", 0) for s in ("D", "T", "TB", "TA", "q", "grid")),
                  sum(x.get(f"{s}_new_lt_min_off_d", 0) for s in ("D", "T", "TB", "TA", "q", "grid")))
        l2 = out["sets"]["demo_last2"][m]["vars"]
        print("  last2 q_cross", {v: (l2[v].get("q_cross"), l2[v].get("q_cross_frames")) for v in VS})


if __name__ == "__main__":
    main()
