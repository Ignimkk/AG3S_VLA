"""T35-diag F — per-run fact table for candidates 1-5 (the 4 failed T34 runs), assembled from a/ b/ c/ d/ e/ outputs.
usage: python3 f_table.py -> f/f_table.json"""
import json, pathlib
from collections import defaultdict
A = json.load(open("a/a_runs.json"))["runs"]; ST = json.load(open("a/settle.json"))["runs"]
B = json.load(open("b/b_summary.json"))["runs"]; Cp = json.load(open("c/c_perception.json"))["runs"]
E = json.load(open("e/revoke_gt.json"))
pathlib.Path("f").mkdir(exist_ok=True)
same_ep = defaultdict(lambda: defaultdict(list))
for k, v in A.items():
    grp, run = k.split("/"); ep = run.split("_")[1]
    same_ep[ep][grp].append(bool(v["success"]))
RUNS = ["E3b_ep1800_r1", "E3b_ep1808_r1", "E3b_ep1808_r2", "E3a_ep1800_r1"]
out = {"definitions": __doc__, "runs": {}, "same_episode_success": {ep: {g: f"{sum(x)}/{len(x)}" for g, x in d.items()} for ep, d in same_ep.items()}}
for run in RUNS:
    cond = run.split("_")[0]; key = f"{cond}_rec/{run}"; ep = run.split("_")[1]
    a = A[key]; st = ST[key]; b = B[run]; cp = Cp[run]["window"]
    bc = {x["t"]: x for x in b["chunks"]}
    post = b["post11"]; pre = b["pre10"]
    mx = bc[post["t_max_dev"]]
    s_chunk = max(t for t in bc if t <= st["t_settle"])
    all_rows = b["chunks"]
    out["runs"][run] = {
        "C1_TO": {"pre10_max_dev_deg": pre["max_dev_deg"], "post11_max_dev_deg": post["max_dev_deg"], "t_max_dev": post["t_max_dev"],
                  "joint_max_dev": post["joint_max_dev"], "driver_at_max_dev": post["driver_at_max_dev"],
                  "variants_dev_at_max_deg": mx["variants_dev_deg"], "ref_worst_at_max": mx["ref_worst"], "returned_at_max": mx["returned"],
                  "returned_counts_window": {k: pre["returned_counts"].get(k, 0) + post["returned_counts"].get(k, 0) for k in set(pre["returned_counts"]) | set(post["returned_counts"])},
                  "driver_counts_window": {k: pre["driver_counts"].get(k, 0) + post["driver_counts"].get(k, 0) for k in set(pre["driver_counts"]) | set(post["driver_counts"])},
                  "n_chunks_ref_neg_rows": pre["n_ref_neg_rows_chunks"] + post["n_ref_neg_rows_chunks"],
                  "n_projection": pre["n_projection"] + post["n_projection"],
                  "n_ref_beyond_TO_position": pre["n_ref_beyond_TO_position"] + post["n_ref_beyond_TO_position"],
                  "ref_beyond_TO_position_deg": {x["t"]: x["ref_beyond_TO_position"] for x in all_rows if x["ref_beyond_TO_position"]},
                  "n_ref_beyond_MJCF": pre["n_ref_beyond_MJCF"] + post["n_ref_beyond_MJCF"], "n_refined_beyond_MJCF": pre["n_refined_beyond_MJCF"] + post["n_refined_beyond_MJCF"],
                  "settle_chunk_t": s_chunk, "settle_chunk_dev_deg": bc[s_chunk]["dev_deg"], "settle_chunk_tip_disp_mm": bc[s_chunk]["tip_disp_mm"],
                  "settle_chunk_returned": bc[s_chunk]["returned"],
                  "refined_minus_ref_body_mid_at_settle_mm": st["refined_minus_ref_body_mid_norm_mm"],
                  "fidelity_max_abs_diff_mm": max(pre["fidelity_max_abs_diff_mm"], post["fidelity_max_abs_diff_mm"]),
                  "fidelity_all_same_row": pre["fidelity_all_same_row"] and post["fidelity_all_same_row"]},
        "C2_policy": {"t_close": a["t_close"], "t_close_ref": a["t_close_ref"], "d_exec_at_close_mm": a["at_close"]["d_exec_mm"],
                      "d_ref_at_close_mm": a["at_close"]["d_ref_mm"], "d_refined_at_close_mm": a["at_close"]["d_refined_mm"],
                      "window16_min_d_ref_mm": a["window16"]["min_d_ref_mm"], "window16_min_d_exec_mm": a["window16"]["min_d_exec_mm"],
                      "body_mid_rel_apple_ref_at_settle_mm": st["body_mid_rel_apple_ref_mm"], "body_mid_rel_apple_meas_at_settle_mm": st["body_mid_rel_apple_mm"],
                      "ref_minus_measured_body_mid_at_settle_mm": st["ref_minus_measured_body_mid_norm_mm"],
                      "n_close_cycles": a["n_close_cycles"], "same_episode_success": out["same_episode_success"][ep]},
        "C3_perception_window": {k: cp[k] for k in ("n", "n_manip_not_apple", "manip_state_counts", "n_occluded_by_hand", "n_split_triggered", "n_split_accepted",
                                                     "n_ball", "n_ball_not_whole_inside", "association_counts", "exclusion_source_counts", "n_switched", "max_to_apple_mm")}
                                | {"held_cover": cp["held_cover"]},
        "C4_grasp": {k: st[k] for k in ("t_settle", "open_settle", "d_l1_settle_mm", "d_l2_settle_mm", "contact_mid_rel_apple_mm", "hold_cmd_median", "hold_min_open",
                                        "hold_open_drop", "contact_lost_t", "dz_at_contact_lost_mm", "hold_dz_max_mm", "hold_mid_z_rise_mm", "hold_xy_max_mm")}
                     | {"pre_close_max_apple_xy_mm": a["pre_close"]["max_disp_xy_mm"], "apple_dz_max_run_mm": a["apple_dz_max_mm"],
                        "window16_min_l1_mm": a["window16"]["min_l1_mm"], "window16_min_l2_mm": a["window16"]["min_l2_mm"]},
    }
    lost = st["contact_lost_t"]
    D = json.load(open(f"b/b_{run}.json"))
    out["runs"][run]["C1_TO"]["settle_to_contact_loss_chunks"] = [
        {"t": e["t"], "held": e["held"], "returned": e["recorded"]["returned"], "dev_deg": e["recorded_dev_deg"]["max"],
         "tip_disp_mm_per_step": e["recorded_refined_vs_reference"]["fingertip_disp_mm_per_step"],
         "variants_tip_mm": {k: v.get("fingertip_disp_mm_max") for k, v in e["ablation"].items()},
         "variants_dev_deg": {k: v.get("dev_max_deg") for k, v in e["ablation"].items()},
         "driver": bc[e["t"]]["driver"], "ref_worst": bc[e["t"]]["ref_worst"], "ref_n_neg_rows": e["reference"]["n_neg_rows"],
         "join_deg": (e.get("join_ref0_vs_prev_refined8_deg") or {}).get("max_arm")}
        for e in D["chunks"] if s_chunk <= e["t"] <= (lost if lost else s_chunk)]
    if run == "E3b_ep1800_r1":
        out["runs"][run]["C5_revoke"] = {k: E[k] for k in ("feedback_of_attach_chunk_t144", "attach_chunk_exec_t144_151", "feedback_of_revoke_chunk_t160",
                                                            "after_revoke_t160_176", "last_in_hand_t", "apple_dz_max_mm_136_176", "t_apple_dz_max")}
json.dump(out, open("f/f_table.json", "w"), indent=1, default=float)
print(json.dumps(out["same_episode_success"]))
for r, v in out["runs"].items():
    print("==", r)
    for k, x in v.items():
        print("  ", k, json.dumps(x, default=float)[:900])
