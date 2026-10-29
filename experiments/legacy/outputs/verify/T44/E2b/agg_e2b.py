"""T44b D-sign aggregation (verifier). Reads raw_demo/ raw_lb/ (e2b_measure.py) [+ timing/agg_timing.json], writes agg_e2b.json.
Criteria fixed in T44.task.md before V-E2 (unchanged here):
 (1) on frames where ball lost >= 1 distractor seed (ROI), loss reduced >= 50 % vs ball (pooled over those frames)
 (2) median target recall >= ball median - 5 pp
 (3) falsely-feasible authorized-sphere queries (vs oracle, margin 10 mm, ROI subset, q and grid pooled) <= ball
 (4) build ms p90 increase vs ball <= 10 ms  -> from the dedicated timing pass (timing/agg_timing.json)
Sign diagnosis: per distractor hull fraction of interior voxels with value < 0 (E2 definition); zero = |v| < 1e-6 m.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, collections, numpy as np

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2b'))
E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
IDS = ("gt", "gt_label")
BO = ("bo_d2", "bo_d3", "bo_d4")
METHODS = ("ball",) + BO + ("oracle", "oracle_prime")
VALS = ("none", "ball", "bo_d2", "bo_d3", "bo_d4", "oracle", "oracle_prime", "ball_noforce", "bo_d2_noforce",
        "main_fine_layer", "coarse_layer", "main_composite")
ZERO = 1e-6
MARGIN = 0.010


def q(a, p):
    a = np.asarray(a, np.float64)
    return round(float(np.percentile(a, p)), 4) if len(a) else None


def load():
    recs = []
    for d in ("raw_demo", "raw_lb"):
        for f in sorted(glob.glob(str(D / d / "*.json"))):
            r = json.load(open(f)); r["_npz"] = f[:-5] + ".npz"; recs.append(r)
    return recs


def crit_block(rows, ident):
    M = {m: [r["identities"][ident]["metrics"][m] for r in rows] for m in METHODS}
    ball_loss = np.array([x["distractor_seeds_lost"] for x in M["ball"]])
    hit = ball_loss >= 1
    Q = collections.defaultdict(list)
    for r in rows:
        z = np.load(r["_npz"], allow_pickle=True)
        for qn in ("q", "grid"):
            roi = z[f"{ident}__{qn}_roi"]
            rad = np.repeat(z["auth_radii"], 1 if qn == "q" else 125).astype(np.float64)
            for m in METHODS:
                Q[(qn, m)].append(z[f"{ident}__{qn}_d_{m}"][roi].astype(np.float64))
            Q[(qn, "_rad")].append(rad[roi])
    Q = {k: np.concatenate(v) for k, v in Q.items()}
    out = {"n_frames": len(rows), "n_frames_ball_lost_distractor": int(hit.sum()),
           "identity_resolved_counts": dict(collections.Counter(r["identities"][ident]["identity_mode_resolved"] for r in rows)),
           "methods": {}, "criteria": {}}
    for m in METHODS:
        x = M[m]
        rec = np.array([v["target_recall"] for v in x if v["target_recall"] is not None])
        loss = np.array([v["distractor_seeds_lost"] for v in x])
        mm = {"recall_median": q(rec, 50), "recall_p10": q(rec, 10), "recall_min": round(float(rec.min()), 4),
              "distractor_lost_sum": int(loss.sum()), "distractor_in_ball_sum": int(sum(v["distractor_seeds_in_ball"] for v in x)),
              "support_lost_sum": int(sum(v["support_seeds_lost"] for v in x)),
              "unlabelled_removed_sum": int(sum(v["unlabelled_removed"] for v in x)),
              "ambiguous_sum": int(sum(v["n_ambiguous"] for v in x)), "kept_nontarget_sum": int(sum(v["n_kept_nontarget"] for v in x)),
              "unowned_removed_sum": int(sum(v["n_unowned_removed"] for v in x)),
              "target_owned_sum": int(sum(v["n_target_owned"] for v in x)),
              "n_removed_sum": int(sum(v["n_removed_total"] for v in x))}
        if m != "ball" and hit.any():
            mm["loss_reduction_pooled_on_hit_frames"] = round(1 - loss[hit].sum() / ball_loss[hit].sum(), 4)
            red = 1 - loss[hit] / ball_loss[hit]
            mm["loss_reduction_frame_median"] = q(red, 50)
            mm["n_hit_frames_reduction_ge_50pct"] = int((red >= 0.5).sum())
        if m != "oracle":
            for qn in ("q", "grid"):
                d, dr, rad = Q[(qn, m)], Q[(qn, "oracle")], Q[(qn, "_rad")]
                over = (d - dr) * 1000
                mm[f"{qn}_vs_oracle_roi"] = {"n": int(len(d)), "over_p95_mm": q(over, 95), "over_max_mm": round(float(over.max()), 3) if len(over) else None,
                                             "n_falsely_feasible": int(((d - rad >= MARGIN) & (dr - rad < MARGIN)).sum()),
                                             "n_falsely_infeasible": int(((d - rad < MARGIN) & (dr - rad >= MARGIN)).sum())}
        out["methods"][m] = mm
    b = out["methods"]["ball"]
    for m in BO:
        c = out["methods"][m]
        out["criteria"][m] = {
            "(1)_loss_reduction_pooled": c.get("loss_reduction_pooled_on_hit_frames"),
            "(1)_pass": None if not hit.any() else bool(c["loss_reduction_pooled_on_hit_frames"] >= 0.5),
            "(2)_recall_median": c["recall_median"], "(2)_ball_recall_median": b["recall_median"],
            "(2)_pass": bool(c["recall_median"] >= b["recall_median"] - 0.05),
            "(3)_ff_q": c["q_vs_oracle_roi"]["n_falsely_feasible"], "(3)_ball_ff_q": b["q_vs_oracle_roi"]["n_falsely_feasible"],
            "(3)_ff_grid": c["grid_vs_oracle_roi"]["n_falsely_feasible"], "(3)_ball_ff_grid": b["grid_vs_oracle_roi"]["n_falsely_feasible"],
            "(3)_pass": bool(c["q_vs_oracle_roi"]["n_falsely_feasible"] <= b["q_vs_oracle_roi"]["n_falsely_feasible"]
                             and c["grid_vs_oracle_roi"]["n_falsely_feasible"] <= b["grid_vs_oracle_roi"]["n_falsely_feasible"])}
    return out


def reproduction(rows, ident):
    """bo_d2 / ball / oracle vs V-E2 raw per frame (recall, distractor lost, unowned removed)."""
    n, mism = 0, []
    pairs = (("ball", "ball"), ("bo_d2", "ball_owned_remove"), ("oracle", "oracle"))
    for r in rows:
        sub = "raw_demo" if r["set"] == "demo" else "raw_lb"
        f = E2 / sub / f"{r['key']}.json"
        if not f.exists():
            continue
        o = json.load(open(f))["identities"][ident]
        if "metrics" not in o:
            continue
        n += 1
        for a, bname in pairs:
            x, y = r["identities"][ident]["metrics"][a], o["metrics"][bname]
            for k in ("target_recall", "distractor_seeds_lost", "n_unowned_removed", "n_removed_total"):
                if x.get(k) != y.get(k):
                    mism.append({"key": r["key"], "method": a, "field": k, "e2b": x.get(k), "e2": y.get(k)})
    return {"n_frames_compared": n, "n_mismatch": len(mism), "mismatches": mism[:20]}


def unowned_block(rows, ident):
    out = {}
    for m in BO:
        cc = collections.Counter(); lab = collections.Counter(); towned = collections.Counter()
        arr = collections.defaultdict(list); n_in_hull = 0; frames_with = 0; pc = []
        for r in rows:
            u = r["identities"][ident].get("unowned", {}).get(m)
            if not u:
                continue
            lab.update(u.get("unowned_by_label", {})); towned.update(u.get("target_owned_by_label", {}))
            if u.get("n_unowned_distractor"):
                frames_with += 1
                cc.update(u.get("cause_counts", {})); n_in_hull += u.get("inside_distractor_hull", 0)
                z = np.load(r["_npz"], allow_pickle=True)
                for k in ("raw_depth", "per_camera_min", "fused", "nontarget_evidence", "d_t", "distractor_mesh"):
                    arr[k].append(z[f"{ident}__un{m}__{k}"].astype(np.float64))
                pc.append((list(z[f"{ident}__per_camera_names"]), z[f"{ident}__un{m}__per_camera"].astype(np.float64)))
        A = {k: np.concatenate(v) * 1000 for k, v in arr.items()}
        rec = {"unowned_by_label": dict(lab), "target_owned_by_label": dict(towned),
               "n_unowned_distractor": int(sum(len(v) for v in arr.get("fused", []))) if arr else 0,
               "n_frames_with_unowned_distractor": frames_with, "inside_distractor_hull": n_in_hull,
               "cause_counts": dict(cc),
               "dist_mm": {k: {"median": q(v, 50), "p10": q(v, 10), "p90": q(v, 90), "max": round(float(v.max()), 2),
                               "frac_le_10mm": round(float((v <= 10).mean()), 4), "frac_le_15mm": round(float((v <= 15).mean()), 4),
                               "frac_le_20mm": round(float((v <= 20).mean()), 4)} for k, v in A.items()}}
        cams = collections.defaultdict(list)
        for names, mat in pc:
            for i, nme in enumerate(names):
                cams[str(nme)].append(mat[i])
        rec["per_camera_dist_mm_median"] = {k: q(np.concatenate(v) * 1000, 50) for k, v in cams.items()}
        out[m] = rec
    return out


def sign_block(rows, ident):
    hull_frac = collections.defaultdict(list)
    P = collections.defaultdict(list)
    per_frame = []
    for r in rows:
        sg = r["identities"][ident].get("sign", {})
        if not sg.get("n_points"):
            continue
        z = np.load(r["_npz"], allow_pickle=True)
        hid = z[f"{ident}__sg_hull"]
        v = {k: z[f"{ident}__sg_val_{k}"].astype(np.float64) for k in VALS}
        comp = {k: z[f"{ident}__sg_comp_{k}"].astype(np.float64) for k in ("none", "ball", "bo_d2", "bo_d3", "bo_d4", "oracle")}
        for h in np.unique(hid):
            s = hid == h
            for k, x in v.items():
                hull_frac[k].append(float((x[s] < 0).mean()))
            for k, x in comp.items():
                hull_frac["composed_" + k].append(float((x[s] < 0).mean()))
            hull_frac["none_le0"].append(float((v["none"][s] <= ZERO).mean()))
            hull_frac["none_zero"].append(float((np.abs(v["none"][s]) <= ZERO).mean()))
        for k, x in v.items():
            P["v_" + k].append(x)
        for k, x in comp.items():
            P["c_" + k].append(x)
        for k in ("fine_state", "fine_sdf", "coarse_state", "coarse_sdf", "depth", "inball", "bound"):
            P[k].append(z[f"{ident}__sg_{k}"])
        P["ans_ball"].append(z[f"{ident}__sg_ans_ball"])
        P["frame"].append(np.full(len(hid), len(per_frame)))
        per_frame.append({"key": r["key"], "n": int(len(hid)), "e": sg.get("e_none_vs_main_fine_box"),
                          "frac_neg": sg["frac_negative"], "safety": sg.get("safety"),
                          "r_finger": r["finger_sphere_radius_m"]["median"]})
    P = {k: np.concatenate(v) for k, v in P.items()}
    out = {"n_frames": len(per_frame), "n_hulls": len(hull_frac.get("none", [])), "n_points": int(len(P.get("v_none", []))),
           "hull_frac_negative_median": {k: q(x, 50) for k, x in hull_frac.items()},
           "hull_frac_negative_mean": {k: round(float(np.mean(x)), 4) for k, x in hull_frac.items()}}
    if not per_frame:
        return out
    vn = P["v_none"]
    cls_none = np.where(np.abs(vn) <= ZERO, "zero", np.where(vn < 0, "neg", "pos"))
    fs, fsdf = P["fine_state"], P["fine_sdf"]
    tsdf_cls = np.where(fs == 0, "unallocated", np.where(fs == 1, "unobserved(w<=min)", np.where(fsdf < 0, "observed_sdf<0", "observed_sdf>=0")))
    cs, csdf = P["coarse_state"], P["coarse_sdf"]
    ctsdf = np.where(cs == 0, "unallocated", np.where(cs == 1, "unobserved(w<=min)", np.where(csdf < 0, "observed_sdf<0", "observed_sdf>=0")))
    out["pooled_frac_negative"] = {k[2:]: round(float((x < 0).mean()), 4) for k, x in P.items() if k.startswith("v_")}
    out["pooled_frac_negative_composed"] = {k[2:]: round(float((x < 0).mean()), 4) for k, x in P.items() if k.startswith("c_")}
    out["a_fine_tsdf_x_none_value"] = {t: {c: int(((tsdf_cls == t) & (cls_none == c)).sum()) for c in ("neg", "zero", "pos")}
                                       for t in np.unique(tsdf_cls)}
    out["a_coarse_tsdf_x_coarse_sign"] = {t: {"neg": int(((ctsdf == t) & (P["v_coarse_layer"] < -ZERO)).sum()),
                                              "zero": int(((ctsdf == t) & (np.abs(P["v_coarse_layer"]) <= ZERO)).sum()),
                                              "pos": int(((ctsdf == t) & (P["v_coarse_layer"] > ZERO)).sum())} for t in np.unique(ctsdf)}
    out["a_fine_tsdf_state_counts"] = dict(collections.Counter(tsdf_cls.tolist()))
    out["a_coarse_tsdf_state_counts"] = dict(collections.Counter(ctsdf.tolist()))
    # (b) sign correction & removal flips
    fl = {}
    for m in ("ball", "bo_d2"):
        nf, v = P[f"v_{m}_noforce"], P[f"v_{m}"]
        fl[m] = {"flipped_by_removal(neg->nonneg)": int(((vn < -ZERO) & (nf >= -ZERO)).sum()),
                 "flipped_by_force(neg->pos)": int(((nf < -ZERO) & (v > ZERO)).sum()),
                 "value_changed_gt_1mm": int((np.abs(v - vn) > 0.001).sum())}
    out["b_flips"] = fl
    out["b_e_none_equals_methods"] = {m: int((np.abs(P[f"v_{m}"] - vn) > ZERO).sum()) for m in ("ball", "bo_d2", "bo_d3", "bo_d4", "oracle", "oracle_prime")}
    # (e) only="fine" rebuild vs main fine layer
    ee = [f["e"] for f in per_frame if f["e"]]
    out["e_none_vs_main_fine_layer_box"] = {"n_frames": len(ee), "max_abs_diff_mm_max": max(x["max_abs_diff_mm"] for x in ee),
                                            "n_sign_diff_sum": int(sum(x["n_sign_diff"] for x in ee)),
                                            "all_same_origin": all(x["same_origin"] for x in ee), "all_same_shape": all(x["same_shape"] for x in ee)}
    mf = P["v_main_fine_layer"]
    out["e_main_fine_layer_interior"] = {"n_neg_strict(<-1e-6)": int((mf < -ZERO).sum()), "n_neg_tiny(-1e-6..0)": int(((mf < 0) & (mf >= -ZERO)).sum()),
                                         "n_none_zero": int((np.abs(vn) <= ZERO).sum())}
    # (c) composition
    cb = P["c_ball"]; mc = P["v_main_composite"]
    out["c_composition"] = {"n_main_composite_neg": int((mc < 0).sum()), "n_coarse_neg": int((P["v_coarse_layer"] < 0).sum()),
                            "n_composed_ball_neg": int((cb < 0).sum()),
                            "n_main_neg_but_composed_ball_nonneg": int(((mc < 0) & (cb >= 0)).sum()),
                            "n_main_neg_composed_nonneg_and_coarse_neg": int(((mc < 0) & (cb >= 0) & (P["v_coarse_layer"] < 0)).sum()),
                            "answer_tier_ball": dict(collections.Counter(P["ans_ball"].tolist())),
                            "window_bound_mm_median": q(P["bound"] * 1000, 50), "window_bound_mm_min": round(float(P["bound"].min()) * 1000, 2)}
    # (d) in ball / out
    ib = P["inball"].astype(bool)
    out["d_inball"] = {"n_in": int(ib.sum()), "n_out": int((~ib).sum()),
                       "frac_neg_none_in": round(float((vn[ib] < 0).mean()), 4) if ib.any() else None,
                       "frac_neg_none_out": round(float((vn[~ib] < 0).mean()), 4) if (~ib).any() else None,
                       "frac_neg_ball_in": round(float((P["v_ball"][ib] < 0).mean()), 4) if ib.any() else None,
                       "frac_neg_main_composite_in": round(float((mc[ib] < 0).mean()), 4) if ib.any() else None,
                       "frac_neg_main_composite_out": round(float((mc[~ib] < 0).mean()), 4) if (~ib).any() else None}
    # depth bins
    dep = P["depth"] * 1000
    bins = ((0, 5), (5, 10), (10, 20), (20, 100))
    out["by_depth_mm"] = {f"{a}-{b}": {"n": int(((dep >= a) & (dep < b)).sum()),
                                       "frac_neg_none": q([], 50) if not ((dep >= a) & (dep < b)).any() else round(float((vn[(dep >= a) & (dep < b)] < 0).mean()), 4),
                                       "frac_neg_coarse": None if not ((dep >= a) & (dep < b)).any() else round(float((P["v_coarse_layer"][(dep >= a) & (dep < b)] < 0).mean()), 4),
                                       "frac_neg_main_composite": None if not ((dep >= a) & (dep < b)).any() else round(float((mc[(dep >= a) & (dep < b)] < 0).mean()), 4),
                                       "frac_unobserved_fine_tsdf": None if not ((dep >= a) & (dep < b)).any() else round(float((fs[(dep >= a) & (dep < b)] <= 1).mean()), 4)}
                         for a, b in bins}
    # safety: authorized sphere centre at interior points
    rf = np.concatenate([np.full(f["n"], f["r_finger"]) for f in per_frame])
    fr = P["frame"]
    saf = {"r_finger_median_m": q(rf, 50), "n_points": int(len(vn)), "n_frames": len(per_frame)}
    for name, x in (("main_composite", mc), ("target_free_ball", P["c_ball"]), ("target_free_bo_d2", P["c_bo_d2"]),
                    ("target_free_bo_d3", P["c_bo_d3"]), ("target_free_bo_d4", P["c_bo_d4"]), ("target_free_oracle", P["c_oracle"]),
                    ("target_free_none", P["c_none"])):
        pos = x > 0
        ok = x - rf >= MARGIN
        saf[name] = {"n_positive": int(pos.sum()), "frac_positive": round(float(pos.mean()), 4),
                     "n_frames_with_positive": int(len(np.unique(fr[pos]))),
                     "n_positive_depth_ge_10mm": int((pos & (dep >= 10)).sum()),
                     "n_frames_positive_depth_ge_10mm": int(len(np.unique(fr[pos & (dep >= 10)]))),
                     "n_clearance_ge_margin": int(ok.sum()), "n_frames_clearance_ge_margin": int(len(np.unique(fr[ok]))),
                     "max_value_mm": round(float(x.max()) * 1000, 2)}
    saf["n_depth_ge_10mm"] = int((dep >= 10).sum())
    out["safety"] = saf
    out["per_frame"] = [{"key": f["key"], "n": f["n"], **{k: f["frac_neg"].get(k) for k in ("none", "ball", "main_fine_layer", "coarse_layer", "main_composite")},
                         "tf_ball_pos": (f["safety"] or {}).get("ball", {}).get("n_positive"),
                         "main_pos": (f["safety"] or {}).get("main_composite", {}).get("n_positive")} for f in per_frame]
    return out


def main():
    recs = load()
    out = {"definitions": __doc__, "n_records": len(recs), "sets": {}}
    sets = {"demo": [r for r in recs if r["set"] == "demo"], "lb": [r for r in recs if r["set"] == "lb"],
            "demo_lt_0.08": [r for r in recs if r["bin"] == "lt_0.08"], "demo_0.08_0.10": [r for r in recs if r["bin"] == "0.08_0.10"]}
    for s, rr in sets.items():
        out["sets"][s] = {}
        for i in IDS:
            rows = [r for r in rr if "metrics" in r["identities"].get(i, {})]
            if not rows:
                continue
            blk = crit_block(rows, i)
            blk["reproduction_vs_E2"] = reproduction(rows, i)
            blk["unowned"] = unowned_block(rows, i)
            if s in ("demo", "lb"):
                blk["sign"] = sign_block(rows, i)
            out["sets"][s][i] = blk
        rc = [r for r in rr if r["identities"]["gt"].get("identity_mode_resolved") in ("pipeline", "nearest_cluster") and "metrics" in r["identities"]["gt"]]
        if rc:
            out["sets"][s]["gt_cluster_only"] = crit_block(rc, "gt")
    out["settings_seen"] = {k: sorted({json.dumps(r["settings"].get(k), sort_keys=True) for r in recs})
                            for k in ("fusion_voxel_size_m", "pointcloud_voxel_size_m", "per_camera_final_voxel_size_m", "fine_truncation_m",
                                      "coarse_truncation_m", "minimum_tsdf_weight", "adjacent_skip_steps", "seeding_method", "tsdf_voxel_m")}
    T = D / "timing" / "agg_timing.json"
    if T.exists():
        t = json.load(open(T))
        out["timing"] = t
        for s in out["sets"]:
            for i in out["sets"][s]:
                for m, c in out["sets"][s][i].get("criteria", {}).items():
                    tm = t["methods"].get(m, {})
                    c["(4)_p90_increase_ms_dedicated"] = tm.get("p90_increase_vs_ball_ms")
                    c["(4)_pass"] = None if tm.get("p90_increase_vs_ball_ms") is None else bool(tm["p90_increase_vs_ball_ms"] <= 10.0)
    json.dump(out, open(D / "agg_e2b.json", "w"), indent=1, default=str)
    for s in out["sets"]:
        for i in IDS:
            b = out["sets"][s].get(i)
            if not b:
                continue
            print(s, i, b["n_frames"], b["identity_resolved_counts"], "hit", b["n_frames_ball_lost_distractor"], "repro", b["reproduction_vs_E2"]["n_mismatch"])
            for m, c in b["criteria"].items():
                print("   ", m, c)


if __name__ == "__main__":
    main()
