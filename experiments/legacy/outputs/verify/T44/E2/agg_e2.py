"""T44 V-E2 aggregation (verifier). Reads raw_demo/ raw_lb/ (e2_measure.py), writes agg_e2.json.
Criteria fixed in T44.task.md (before measurement):
 (1) on frames where ball lost >= 1 distractor seed (ROI), method loss reduced >= 50 % vs ball  [primary: pooled sum over
     those frames; also: per-frame median reduction and #frames with >= 50 %]
 (2) median target recall >= ball median - 5 pp
 (3) falsely-feasible queries (vs oracle, margin 10 mm) <= ball   [q and grid, pooled]
 (4) build ms p90 (pooled runs) increase vs ball <= 10 ms
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, numpy as np, collections

E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
METHODS = ("ball", "ball_owned_remove", "ball_owned_keep", "oracle", "oracle_prime")
CMP = ("ball_owned_remove", "ball_owned_keep")
IDS = ("gt", "gt_label")


def load():
    recs = []
    for d in ("raw_demo", "raw_lb"):
        for f in sorted(glob.glob(str(E2 / d / "*.json"))):
            r = json.load(open(f)); r["_npz"] = f[:-5] + ".npz"; recs.append(r)
    return recs


def q(a, p):
    return round(float(np.percentile(a, p)), 3) if len(a) else None


def agg(recs, ident):
    out = {"n_frames": 0, "frames": [], "methods": {}}
    rows = [r for r in recs if "metrics" in r["identities"].get(ident, {})]
    out["n_frames"] = len(rows)
    out["n_skipped"] = sum(1 for r in recs if "metrics" not in r["identities"].get(ident, {}))
    if not rows:
        return out
    res = collections.Counter(r["identities"][ident]["identity_mode_resolved"] for r in rows)
    out["identity_resolved_counts"] = dict(res)
    M = {m: [r["identities"][ident]["metrics"][m] for r in rows] for m in METHODS}
    ball_loss = np.array([x["distractor_seeds_lost"] for x in M["ball"]])
    hit = ball_loss >= 1
    out["n_frames_ball_lost_distractor"] = int(hit.sum())
    # query raw pooling
    Q = {}
    dl_unowned = []
    for r in rows:
        z = np.load(r["_npz"], allow_pickle=True)
        lab = z[f"{ident}__seed_label"]; dis = lab == 1
        pr, pk = z[f"{ident}__processed_ball_owned_remove"] == 1, z[f"{ident}__processed_ball_owned_keep"] == 1
        dl_unowned.append(int((dis & pr & ~pk).sum()))
        for qn in ("q", "grid"):
            roi = z[f"{ident}__{qn}_roi"]
            rad = np.repeat(z["auth_radii"], 1 if qn == "q" else 125).astype(np.float64)
            for m in METHODS:
                Q.setdefault((qn, m), []).append(z[f"{ident}__{qn}_d_{m}"][roi].astype(np.float64))
            Q.setdefault((qn, "_rad"), []).append(rad[roi])
    Q = {k: np.concatenate(v) for k, v in Q.items()}
    for m in METHODS:
        x = M[m]
        rec = np.array([v["target_recall"] for v in x if v["target_recall"] is not None])
        loss = np.array([v["distractor_seeds_lost"] for v in x])
        mm = {
            "recall_median": q(rec, 50), "recall_p10": q(rec, 10), "recall_min": round(float(rec.min()), 4) if len(rec) else None,
            "distractor_lost_sum": int(loss.sum()), "distractor_lost_sum_on_ball_hit_frames": int(loss[hit].sum()),
            "distractor_in_ball_sum": int(sum(v["distractor_seeds_in_ball"] for v in x)),
            "obstacle_retention_roi_median": q([v["obstacle_retention_roi"] for v in x], 50),
            "nontarget_labelled_retention_roi_median": q([v["nontarget_labelled_retention_roi"] for v in x], 50),
            "unlabelled_seeds_roi_sum": int(sum(v["unlabelled_seeds_roi"] for v in x)),
            "unlabelled_removed_sum": int(sum(v["unlabelled_removed"] for v in x)),
            "shared_removed_sum": int(sum(v["shared_removed"] for v in x)),
            "support_lost_sum": int(sum(v["support_seeds_lost"] for v in x)),
            "destination_lost_sum": int(sum(v["destination_seeds_lost"] for v in x)),
            "ambiguous_sum": int(sum(v["n_ambiguous"] for v in x)), "ambiguous_median": q([v["n_ambiguous"] for v in x], 50),
            "unowned_removed_sum": int(sum(v["n_unowned_removed"] for v in x)),
            "unowned_kept_sum": int(sum(v["n_unowned_kept"] for v in x)),
            "n_seeds_in_ball_sum": int(sum(v["n_seeds_in_ball"] for v in x)),
        }
        if m in ("ball",) + CMP:
            runs = np.concatenate([v["build_ms_runs"] for v in x])
            mm["build_ms_median"], mm["build_ms_p90"], mm["build_ms_n"] = q(runs, 50), q(runs, 90), int(len(runs))
            fm = np.array([np.median(v["build_ms_runs"]) for v in x])
            mm["build_ms_frame_median_median"], mm["build_ms_frame_median_p90"] = q(fm, 50), q(fm, 90)
            cm = [v["classify_ms"] for v in x if v["classify_ms"] is not None]
            mm["classify_ms_median"], mm["classify_ms_p90"] = q(cm, 50), q(cm, 90)
        if m in CMP:
            if hit.any():
                red = 1 - loss[hit] / ball_loss[hit]
                mm["loss_reduction_pooled_on_hit_frames"] = round(1 - loss[hit].sum() / ball_loss[hit].sum(), 4)
                mm["loss_reduction_frame_median"] = q(red, 50)
                mm["n_hit_frames_reduction_ge_50pct"] = int((red >= 0.5).sum())
            paired = np.concatenate([np.asarray(v["build_ms_runs"]) - np.asarray(b["build_ms_runs"]) for v, b in zip(x, M["ball"])])
            mm["build_ms_paired_diff_median"], mm["build_ms_paired_diff_p90"] = q(paired, 50), q(paired, 90)
        # queries
        for ref in ("oracle", "oracle_prime"):
            if m == ref:
                continue
            for qn in ("q", "grid"):
                d, dr, rad = Q[(qn, m)], Q[(qn, ref)], Q[(qn, "_rad")]
                over = (d - dr) * 1000
                cl, clr = d - rad, dr - rad
                mm[f"{qn}_vs_{ref}"] = {"n": int(len(d)), "over_median_mm": q(over, 50), "over_p95_mm": q(over, 95),
                                         "over_max_mm": round(float(over.max()), 3) if len(over) else None,
                                         "n_over_gt_1mm": int((over > 1).sum()),
                                         "n_falsely_feasible": int(((cl >= .010) & (clr < .010)).sum()),
                                         "n_falsely_infeasible": int(((cl < .010) & (clr >= .010)).sum()),
                                         "n_ref_infeasible": int((clr < .010).sum())}
        out["methods"][m] = mm
    b = out["methods"]["ball"]
    crit = {}
    for m in CMP:
        c = out["methods"][m]
        crit[m] = {
            "(1)_loss_reduction_pooled": c.get("loss_reduction_pooled_on_hit_frames"),
            "(1)_pass": (None if not hit.any() else bool(c["loss_reduction_pooled_on_hit_frames"] >= 0.5)),
            "(1)_n_hit_frames": int(hit.sum()),
            "(2)_recall_median": c["recall_median"], "(2)_ball_recall_median": b["recall_median"],
            "(2)_pass": bool(c["recall_median"] >= b["recall_median"] - 0.05),
            "(3)_falsely_feasible_q": c["q_vs_oracle"]["n_falsely_feasible"], "(3)_ball_q": b["q_vs_oracle"]["n_falsely_feasible"],
            "(3)_falsely_feasible_grid": c["grid_vs_oracle"]["n_falsely_feasible"], "(3)_ball_grid": b["grid_vs_oracle"]["n_falsely_feasible"],
            "(3)_pass": bool(c["q_vs_oracle"]["n_falsely_feasible"] <= b["q_vs_oracle"]["n_falsely_feasible"]
                             and c["grid_vs_oracle"]["n_falsely_feasible"] <= b["grid_vs_oracle"]["n_falsely_feasible"]),
            "(4)_p90_increase_ms": round(c["build_ms_p90"] - b["build_ms_p90"], 3),
            "(4)_pass": bool(c["build_ms_p90"] - b["build_ms_p90"] <= 10.0),
        }
    out["criteria"] = crit
    out["ball_owned_remove_distractor_lost_unowned_only_sum"] = int(sum(dl_unowned))
    out["ball_owned_remove_distractor_lost_sum"] = out["methods"]["ball_owned_remove"]["distractor_lost_sum"]
    # merge / identity
    mg = [r["identities"][ident]["merge"]["target_cluster"] for r in rows if r["identities"][ident]["merge"].get("target_cluster")]
    out["merge"] = {"n": len(mg), "target_cluster_target_fraction_median": q([x["target_fraction"] for x in mg], 50),
                    "n_target_cluster_target_fraction_lt_0.8": sum(x["target_fraction"] < 0.8 for x in mg),
                    "n_target_cluster_with_distractor_points": sum(x["distractor_fraction"] > 0 for x in mg),
                    "target_cluster_share_of_gt_target_median": q([x["share_of_gt_target"] for x in mg if x["share_of_gt_target"] is not None], 50)}
    comp = [r["identities"][ident]["identity"].get("identity_completeness") for r in rows]
    comp = [c for c in comp if c is not None]
    out["nontarget_source_counts"] = dict(collections.Counter(r["identities"][ident]["nontarget"].get("source") for r in rows))
    px = [r for r in rows if r["identities"][ident]["nontarget"].get("source") == "proximity"]
    if px:
        out["proximity_frames"] = {"n": len(px), "ball_recall_median": q([r["identities"][ident]["metrics"]["ball"]["target_recall"] for r in px], 50),
                                   "ball_owned_remove_recall_median": q([r["identities"][ident]["metrics"]["ball_owned_remove"]["target_recall"] for r in px], 50),
                                   "note": "point_indices did not index the current cloud (cloud[idx] != points); completeness via indices reads 0 there"}
    out["identity_completeness_median"] = q(comp, 50); out["identity_completeness_min"] = round(min(comp), 4) if comp else None
    # sign check
    sc = {}
    for m in METHODS:
        fn, eik, eikm, fnm = [], [], [], []
        for r in rows:
            s = r["identities"][ident].get("sign_check", {}).get(m)
            if not s:
                continue
            for di in s["distractor_interior"]:
                fn.append(di["frac_negative"])
                if "main_layer" in di:
                    fnm.append(di["main_layer"]["frac_negative"])
            if s.get("eikonal"):
                eik.append(s["eikonal"]["median"])
            if s.get("eikonal_main_layer"):
                eikm.append(s["eikonal_main_layer"]["median"])
        sc[m] = {"distractor_interior_frac_negative_median": q(fn, 50), "n": len(fn),
                 "eikonal_median_of_frame_medians": q(eik, 50), "eikonal_min_frame_median": round(min(eik), 4) if eik else None}
        if fnm:
            sc["main_layer"] = {"distractor_interior_frac_negative_median": q(fnm, 50), "eikonal_median_of_frame_medians": q(eikm, 50)}
    out["sign_check"] = sc
    # per frame compact
    for r in rows:
        ir = r["identities"][ident]
        out["frames"].append({"key": r["key"], "bin": r["bin"], "identity_resolved": ir["identity_mode_resolved"],
                              "completeness": ir["identity"].get("identity_completeness"),
                              "target_cluster_target_fraction": (ir["merge"].get("target_cluster") or {}).get("target_fraction"),
                              **{f"{m}__recall": ir["metrics"][m]["target_recall"] for m in METHODS},
                              **{f"{m}__dlost": ir["metrics"][m]["distractor_seeds_lost"] for m in METHODS},
                              "distractor_in_ball": ir["metrics"]["ball"]["distractor_seeds_in_ball"],
                              "ambiguous": ir["metrics"]["ball_owned_remove"]["n_ambiguous"],
                              "unowned": ir["metrics"]["ball_owned_remove"]["n_unowned_removed"],
                              "unlabelled_roi": ir["metrics"]["ball"]["unlabelled_seeds_roi"],
                              **{f"{m}__ff_q": ir["queries"]["q"][f"{m}|vs_oracle|all"]["n_falsely_feasible"] for m in ("ball",) + CMP},
                              **{f"{m}__ff_grid": ir["queries"]["grid"][f"{m}|vs_oracle|all"]["n_falsely_feasible"] for m in ("ball",) + CMP}})
    return out


def load_stats(recs):
    la = [float(r["load_before"]["loadavg"][0]) for r in recs] + [float(r["load_after"]["loadavg"][0]) for r in recs]
    thr = [(r["load_after"]["throttled_usec"] - r["load_before"]["throttled_usec"]) / 1e6 for r in recs]
    per = [(r["load_after"]["nr_throttled"] - r["load_before"]["nr_throttled"]) / max(1, r["load_after"]["nr_periods"] - r["load_before"]["nr_periods"]) for r in recs]
    gu = [r[k].get("gpu_util_pct") for r in recs for k in ("load_before", "load_after") if r[k].get("gpu_util_pct") is not None]
    gm = [r[k].get("gpu_mem_used_mib") for r in recs for k in ("load_before", "load_after") if r[k].get("gpu_mem_used_mib") is not None]
    ws = [r[k]["wset_gib"] for r in recs for k in ("load_before", "load_after")]
    return {"loadavg1_median": q(la, 50), "loadavg1_max": max(la), "cgroup_throttled_s_per_frame_median": q(thr, 50),
            "cgroup_throttled_period_fraction_median": q(per, 50), "gpu_util_pct_median": q(gu, 50), "gpu_util_pct_max": max(gu),
            "gpu_mem_used_mib_max": max(gm), "wset_gib_max": max(ws), "wall_s_median": q([r["wall_s"] for r in recs], 50),
            "concurrency_note": "2 e2_measure.py shards of this verifier ran concurrently on the same GPU; another verifier's CPU replays and two policy servers/clients ran on the pod"}


def main():
    recs = load()
    out = {"definitions": __doc__, "n_records": len(recs), "sets": {}}
    sets = {"demo": [r for r in recs if r["set"] == "demo"], "lb": [r for r in recs if r["set"] == "lb"],
            "demo_lt_0.08": [r for r in recs if r["bin"] == "lt_0.08"],
            "demo_0.08_0.10": [r for r in recs if r["bin"] == "0.08_0.10"]}
    for s, rr in sets.items():
        out["sets"][s] = {i: agg(rr, i) for i in IDS}
        rc = [r for r in rr if r["identities"]["gt"].get("identity_mode_resolved") in ("pipeline", "nearest_cluster")]
        out["sets"][s]["gt_cluster_only"] = agg(rc, "gt")
        out["sets"][s]["n_records"] = len(rr)
    out["load"] = load_stats(recs) if recs else None
    nt = [r["identities"][i]["nontarget_points_ms"] for r in recs for i in IDS if "nontarget_points_ms" in r["identities"][i]]
    out["nontarget_points_ms"] = {"median": q(nt, 50), "p90": q(nt, 90), "n": len(nt)}
    json.dump(out, open(E2 / "agg_e2.json", "w"), indent=1)
    for s in sets:
        for i in IDS + ("gt_cluster_only",):
            a = out["sets"][s][i]
            if not a.get("n_frames"):
                continue
            print(s, i, a["n_frames"], a.get("identity_resolved_counts"), "hit", a["n_frames_ball_lost_distractor"])
            for m, c in a["criteria"].items():
                print("   ", m, {k: v for k, v in c.items()})


if __name__ == "__main__":
    main()
