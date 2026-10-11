"""Aggregate raw/cmp/*.cmp.json -> agg_cmp.json (numbers only). Pre-grasp window = the 8 chunks before the first `closing` chunk."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json
import numpy as np

D = _ag3s_workspace_path('outputs/verify/T45/V')
RANK = {"optimal": 0, "feasible": 1, "violated": 2}
files = sorted(glob.glob(f"{D}/raw/cmp/*.cmp.json"))

def q(a, p):
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else None

def summ(a):
    a = np.asarray([x for x in a if x is not None], float)
    if not len(a):
        return None
    return {"n": int(len(a)), "min": float(a.min()), "median": float(np.median(a)), "p90": q(a, 90), "max": float(a.max())}

runs = {}
allc = []
for f in files:
    d = json.load(open(f))
    cl = d["close"]
    close = cl["close_seq"] if cl["close_seq"] is not None else cl["attach_seq"]
    runs[d["name"]] = {"close_seq": cl["close_seq"], "attach_seq": cl["attach_seq"], "hand": d["hand"], "n_spheres": d["n_spheres"]}
    for c in d["chunks"]:
        if "skipped" in c:
            continue
        c = dict(c)
        c["run"] = d["name"]
        c["hand"] = d["hand"]
        c["rel"] = None if close is None else c["seq"] - close      # -8..-1 = pre8, 0 = closing chunk
        allc.append(c)

def sel(lo, hi):
    return [c for c in allc if c["rel"] is not None and lo <= c["rel"] <= hi]

def own(c, v):
    return c["variants"][v]["eval_under"][v]

def window_stats(chs, label):
    out = {"label": label, "n_chunks": len(chs), "runs": sorted({c["run"] for c in chs})}
    if not chs:
        return out
    out["n_runs"] = len(out["runs"])
    d_rad = [c["diff_window_max_abs_rad"] for c in chs]
    d_own = [c["diff_own_max_abs_rad"] for c in chs]
    out["refined_diff_joint_window_deg"] = summ(np.degrees(d_rad))
    out["refined_diff_chunk_all_cols_deg"] = summ(np.degrees(d_own))
    out["n_chunks_identical_sha1"] = int(sum(c["same_sha1"] for c in chs))
    out["n_chunks_joint_diff_gt_1deg"] = int(sum(np.degrees(x) > 1.0 for x in d_rad))
    out["n_chunks_joint_diff_gt_3deg"] = int(sum(np.degrees(x) > 3.0 for x in d_rad))
    out["base_vs_recorded_deg"] = summ(np.degrees([c["variants"]["base"]["own_vs_recorded_max_abs_rad"] for c in chs]))
    out["PC_vs_recorded_deg"] = summ(np.degrees([c["variants"]["PC"]["own_vs_recorded_max_abs_rad"] for c in chs]))
    # status / verdict
    sb = [c["variants"]["base"]["status"] for c in chs]
    sp = [c["variants"]["PC"]["status"] for c in chs]
    out["status_counts"] = {"base": {s: sb.count(s) for s in sorted(set(sb))}, "PC": {s: sp.count(s) for s in sorted(set(sp))}}
    hb = [own(c, "base")["n_collision"] > 0 for c in chs]
    hp = [own(c, "PC")["n_collision"] > 0 for c in chs]
    out["hold_candidates_base_own_model"] = int(sum(hb))
    out["hold_candidates_PC_own_model"] = int(sum(hp))
    out["new_hold_PC_vs_base"] = int(sum((not a) and b for a, b in zip(hb, hp)))
    out["resolved_hold_PC_vs_base"] = int(sum(a and (not b) for a, b in zip(hb, hp)))
    out["new_hold_chunks"] = [f'{c["run"]}:{c["seq"]}' for c, a, b in zip(chs, hb, hp) if (not a) and b]
    out["status_worse_PC_vs_base"] = int(sum(RANK[b] > RANK[a] for a, b in zip(sb, sp)))
    out["status_better_PC_vs_base"] = int(sum(RANK[b] < RANK[a] for a, b in zip(sb, sp)))
    out["n_collision_rows_increase_chunks"] = int(sum(own(c, "PC")["n_collision"] > own(c, "base")["n_collision"] for c in chs))
    out["n_collision_rows_sum"] = {"base": int(sum(own(c, "base")["n_collision"] for c in chs)),
                                   "PC": int(sum(own(c, "PC")["n_collision"] for c in chs))}
    out["verdict_worse_total"] = int(sum(((not a) and b) or (RANK[sp_] > RANK[sb_]) for a, b, sb_, sp_ in zip(hb, hp, sb, sp)))
    # cross: base trajectory under PC model, PC trajectory under base model
    out["cross_hold_base_traj_under_PC_model"] = int(sum(c["variants"]["base"]["eval_under"]["PC"]["n_collision"] > 0 for c in chs))
    out["cross_hold_PC_traj_under_base_model"] = int(sum(c["variants"]["PC"]["eval_under"]["base"]["n_collision"] > 0 for c in chs))
    # recorded (server) HOLD in these chunks
    out["recorded_hold_chunks"] = int(sum("collision:hold" in (c["recorded"]["reasons"] or []) for c in chs))
    out["recorded_collision_rows_sum"] = int(sum((c["recorded"]["n_collision"] or 0) for c in chs))
    # worst row
    out["worst_row_is_palm_own"] = {"base": int(sum(own(c, "base")["worst_is_palm"] for c in chs)),
                                    "PC": int(sum(own(c, "PC")["worst_is_palm"] for c in chs))}
    out["worst_link_counts_PC"] = {}
    out["worst_link_counts_base"] = {}
    for v, key in (("PC", "worst_link_counts_PC"), ("base", "worst_link_counts_base")):
        for c in chs:
            l = own(c, v)["worst_link_to_view"]
            out[key][l] = out[key].get(l, 0) + 1
    out["worst_row_is_palm_solver_metric"] = {v: int(sum(str(c["variants"][v]["worst_link_solver"]) == f'ee_{c["hand"]}' for c in chs)) for v in ("base", "PC")}
    out["to_min_clearance_mm"] = {v: summ([own(c, v)["to_min_mm"] for c in chs]) for v in ("base", "PC")}
    out["palm_group_min_to_mm"] = {v: summ([own(c, v)["group_min_to_mm"]["palm"] for c in chs]) for v in ("base", "PC")}
    out["fingers_group_min_to_mm"] = {v: summ([own(c, v)["group_min_to_mm"]["fingers"] for c in chs]) for v in ("base", "PC")}
    # palm geometry (own trajectory, own model)
    geo = {}
    for v in ("base", "PC"):
        P = [own(c, v)["palm"] for c in chs if "palm" in own(c, v)]
        g = {"n_chunks": len(P), "n_spheres": summ([p["n_spheres"] for p in P])["median"] if P else None,
             "n_queries_total": int(sum(p["n_queries"] for p in P))}
        for key in ("gap_target", "gap_table", "gap_other_obstacle", "gap_field_main", "gap_field_target_free"):
            vals = [p[key] for p in P if p.get(key)]
            if not vals:
                g[key] = None
                continue
            ent = {"min_mm_over_chunks": float(min(x["min_mm"] for x in vals)),
                   "median_of_chunk_min_mm": float(np.median([x["min_mm"] for x in vals]))}
            for t in ("0", "5", "10", "20"):
                if f"n_lt_{t}mm" in vals[0]:
                    ent[f"n_queries_lt_{t}mm"] = int(sum(x[f"n_lt_{t}mm"] for x in vals))
            g[key] = ent
        g["target_free_mask_palm_all_true_chunks"] = int(sum(1 for p in P if p["target_free_mask_palm"] == p["n_spheres"]))
        g["target_free_mask_palm_present_chunks"] = int(sum(1 for p in P if p["target_free_mask_palm"] is not None))
        g["target_contact_rows_palm_all_true_chunks"] = int(sum(1 for p in P if p["target_contact_rows_palm"] == p["n_spheres"]))
        g["manipulated_link_margin_palm_mm"] = summ([p["manipulated_link_margin_palm_mm"] for p in P])
        g["n_queries_free_minus_main_gt_2mm"] = int(sum(p["n_queries_free_minus_main_gt_2mm"] or 0 for p in P))
        g["n_queries_target_gap_lt_20mm_and_free_minus_main_gt_2mm"] = int(sum(p["n_queries_target_gap_lt_20mm_and_free_minus_main_gt_2mm"] or 0 for p in P))
        geo[v] = g
    out["palm_geometry"] = geo
    # per-chunk PC - base palm gap changes
    dg = []
    for c in chs:
        a, b = own(c, "base").get("palm"), own(c, "PC").get("palm")
        if a and b and a["gap_target"] and b["gap_target"]:
            dg.append(b["gap_target"]["min_mm"] - a["gap_target"]["min_mm"])
    out["palm_min_target_gap_PC_minus_base_mm"] = summ(dg)
    return out

res = {"definitions": "cmp pass: open loop (ref.reset before each solve), sqp.time_budget_ms=1e6, ta_offline reconstruction + strict target permission rows + layer_diff membership; "
       "window = executed 8 steps; pre8 = the 8 chunks before the first `closing` chunk of the run (rel -8..-1); close = rel 0; hold candidate = classify_violations n_collision>0 under the variant's own model; "
       "palm gaps = geometric d - r (no margin) of the variant's own palm spheres on its own refined window; gap_target = nearest recorded target point; gap_table = support plane; gap_other_obstacle = nearest recorded non-target non-support cloud point; "
       "gap_field_main = main ESDF; gap_field_target_free = target-free ESDF (the layer a permitted palm row reads).",
       "runs": runs, "n_runs": len(files)}
res["pre8"] = window_stats(sel(-8, -1), "pre8")
res["close"] = window_stats(sel(0, 0), "close chunk (rel 0)")
res["pre8_plus_close"] = window_stats(sel(-8, 0), "pre8 + close")
res["post1"] = window_stats(sel(1, 1), "attach chunk (rel +1)")
res["all_chunks"] = window_stats(allc, "all chunks (cmp)")
res["no_close_runs"] = [n for n, r in runs.items() if r["close_seq"] is None]
json.dump(res, open(f"{D}/agg_cmp.json", "w"), indent=1)
json.dump(allc, open(f"{D}/agg_cmp_chunks.json", "w"))
print(json.dumps({k: res[k] for k in ("pre8",)}, indent=1)[:9000])
print(res["no_close_runs"], {k: (v["close_seq"], v["attach_seq"]) for k, v in runs.items()})
