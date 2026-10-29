"""T44b D-merge: aggregate merge_analyze.py output -> stage decomposition + candidate table.
usage: agg_merge.py merge_gt.json [merge_vla.json] --out agg.json"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np


def med(x):
    x = [v for v in x if v is not None]
    return None if not x else round(float(np.median(x)), 4)


def decompose(R):
    M = [r for r in R if r.get("e2_merged")]
    out = {"n_frames": len(R), "n_e2_merged": len(M), "n_mine_merged": sum(bool(r.get("merged_mine")) for r in R),
           "n_agree_e2_vs_mine": sum(bool(r.get("merged_mine")) == r["e2_merged"] for r in R if r.get("parent"))}
    # stage table on e2 merged frames
    reason = collections.Counter(r["split_recorded"]["reason"] for r in M)
    out["merged_split_reason"] = dict(reason)
    out["merged_with"] = dict(collections.Counter(r.get("merge_with") for r in M))
    gaps = [r.get("mesh_gap_merged_mm") for r in M]
    cg = [r.get("cloud_gap_mm") for r in M]
    out["mesh_gap_mm"] = {"median": med(gaps), "min": min(g for g in gaps if g is not None), "max": max(g for g in gaps if g is not None),
                          "n_lt_5": sum(g < 5 for g in gaps if g is not None), "n_lt_10": sum(g < 10 for g in gaps if g is not None),
                          "n_lt_15": sum(g < 15 for g in gaps if g is not None), "n_lt_30": sum(g < 30 for g in gaps if g is not None)}
    out["cloud_gap_mm"] = {"median": med(cg), "min": min(cg), "max": max(cg), "n_lt_10": sum(g < 10 for g in cg),
                           "n_lt_15": sum(g < 15 for g in cg), "n_lt_30": sum(g < 30 for g in cg)}
    ntm = [r.get("nearest_mesh_gap_mm") for r in R if not r.get("e2_merged")]
    out["non_merged_nearest_mesh_gap_mm"] = {"median": med(ntm), "min": min(ntm), "max": max(ntm), "n_lt_30": sum(g < 30 for g in ntm)}
    out["trigger"] = {"n_triggered_recorded": sum(r["split_recorded"]["triggered"] for r in M),
                      "n_triggered_now": sum(r["trigger_check"]["triggered_now"] for r in M)}
    c15 = [r["candidates"]["current: recluster15 + ratio2 (norm att)"] for r in M]
    out["recluster15"] = {"n_ge2_components": sum(c["n_components"] >= 2 for c in c15),
                          "n_oracle_pure": sum(c["oracle"]["precision"] >= .95 for c in c15),
                          "n_oracle_pure_complete": sum(c["oracle"]["precision"] >= .95 and c["oracle"]["recall"] >= .9 for c in c15)}
    rat = [r["split_recorded"]["ratio"] for r in M if r["split_recorded"]["ratio"] is not None]
    rat = [float(x) if x != "inf" else np.inf for x in rat]
    out["recorded_ratio_norm"] = {"n": len(rat), "median": med(rat), "max": None if not rat else round(float(max(rat)), 3),
                                  "n_ge_2": sum(x >= 2 for x in rat)}
    at = [r["attention_target_vs_other"] for r in M if r.get("attention_target_vs_other")]
    out["attention_target_vs_merged_other"] = {
        "norm_ratio_median": med([a["norm_ratio"] for a in at]), "raw_ratio_median": med([a["raw_ratio"] for a in at]),
        "n_raw_ratio_ge_2": sum(a["raw_ratio"] >= 2 for a in at), "n_norm_ratio_ge_2": sum(a["norm_ratio"] >= 2 for a in at),
        "n_raw_ratio_gt_1": sum(a["raw_ratio"] > 1 for a in at),
        "frac_saturated_target_median": med([a["frac_norm_saturated_t"] for a in at]),
        "frac_saturated_other_median": med([a["frac_norm_saturated_o"] for a in at]),
        "mass_share_target_median": med([a["mass_share_t"] for a in at])}
    sd = [r["seeds"] for r in M]
    out["seeds_merged"] = {"median_target": med([s["target"] for s in sd]), "median_other_fruit": med([s["other_fruit"] for s in sd]),
                           "n_frames_seed_on_other_fruit": sum(s["other_fruit"] > 0 for s in sd)}
    # per-frame cause class
    cls = collections.Counter()
    per = []
    for r in M:
        c = r["candidates"]["current: recluster15 + ratio2 (norm att)"]
        geo_ok = c["oracle"]["precision"] >= .95 and c["oracle"]["recall"] >= .9
        if not r["trigger_check"]["triggered_now"]:
            k = "not_triggered"
        elif not geo_ok:
            k = "geometry: no clean 15 mm component (gap < split radius)"
        elif r["split_recorded"]["reason"] == "split" and r["pipeline_cluster"]["target_fraction"] >= .95:
            k = "resolved by the existing split"
        elif r["split_recorded"]["reason"] == "split":
            k = "split taken, wrong component"
        else:
            k = "attention: components separate, ratio < 2"
        cls[k] += 1
        per.append({"key": r["key"], "target": r["target"], "merge_with": r.get("merge_with"),
                    "mesh_gap_mm": r.get("mesh_gap_merged_mm"), "cloud_gap_mm": r.get("cloud_gap_mm"),
                    "parent_target_fraction": r["parent"]["target_fraction"], "extents_mm": r["parent"]["extents_mm"],
                    "triggered": r["trigger_check"]["triggered_now"], "reason": r["split_recorded"]["reason"],
                    "ratio_norm": r["split_recorded"]["ratio"],
                    "raw_ratio_t_vs_o": (r.get("attention_target_vs_other") or {}).get("raw_ratio"),
                    "recluster15_components": c["n_components"], "recluster15_oracle_precision": c["oracle"]["precision"],
                    "cause": k})
    out["cause_classes"] = dict(cls)
    ch = [r.get("chosen") for r in R]
    out["chosen_all"] = {"n": len(R), "n_none": sum(c is None for c in ch),
                         "n_is_target_cluster": sum(bool(c and c["is_target_cluster"]) for c in ch),
                         "n_top_label": dict(collections.Counter(c["top_label"] if c else None for c in ch)),
                         "n_pure_target": sum(bool(c and c["target_fraction"] >= .95 and c["recall"] >= .9) for c in ch)}
    out["pipeline_cluster_merged"] = {"n_pure_complete": sum(r["pipeline_cluster"]["target_fraction"] >= .95 and r["pipeline_cluster"]["recall"] >= .9 for r in M),
                                      "target_fraction_median": med([r["pipeline_cluster"]["target_fraction"] for r in M])}
    out["pipeline_cluster_nonmerged"] = {"n_pure_complete": sum(r["pipeline_cluster"]["target_fraction"] >= .95 and r["pipeline_cluster"]["recall"] >= .9 for r in R if not r.get("e2_merged") and r.get("pipeline_cluster")),
                                         "n": sum(1 for r in R if not r.get("e2_merged"))}
    out["per_frame"] = per
    return out


def candidates(R):
    M = [r for r in R if r.get("e2_merged") and r.get("candidates")]
    N = [r for r in R if not r.get("e2_merged") and r.get("candidates")]
    names = list(M[0]["candidates"]) if M else list(N[0]["candidates"])
    out = {}
    for nme in names:
        o = {}
        for sel in ("att", "mean_raw", "oracle"):
            pm = [r["candidates"][nme][sel]["precision"] for r in M]
            rm = [r["candidates"][nme][sel]["recall"] for r in M]
            pn = [r["candidates"][nme][sel]["precision"] for r in N]
            rn = [r["candidates"][nme][sel]["recall"] for r in N]
            o[sel] = {"merged_n": len(M), "merged_pure_and_complete": sum(p >= .95 and q >= .9 for p, q in zip(pm, rm)),
                      "merged_pure": sum(p >= .95 for p in pm), "merged_precision_median": med(pm),
                      "merged_recall_median": med(rm), "merged_precision_mean": round(float(np.mean(pm)), 4),
                      "nonmerged_n": len(N), "nonmerged_complete": sum(q >= .9 for q in rn),
                      "nonmerged_recall_median": med(rn), "nonmerged_precision_median": med(pn),
                      "all_pure_and_complete": sum(p >= .95 and q >= .9 for p, q in zip(pm + pn, rm + rn))}
        out[nme] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for f in a.inputs:
        R = json.loads(pathlib.Path(f).read_text())
        src = R[0]["attention_source"]
        res[src] = {"decomposition": decompose(R), "candidates": candidates(R)}
    pathlib.Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for src, v in res.items():
        d = v["decomposition"]
        print("==", src, {k: d[k] for k in d if k != "per_frame"})
        for n, c in v["candidates"].items():
            print(f"  {n:42s}", " | ".join(f"{s}: M {c[s]['merged_pure_and_complete']}/{c[s]['merged_n']} "
                                           f"p{c[s]['merged_precision_median']} r{c[s]['merged_recall_median']} "
                                           f"N {c[s]['nonmerged_complete']}/{c[s]['nonmerged_n']}" for s in ("att", "mean_raw", "oracle")))


if __name__ == "__main__":
    main()
