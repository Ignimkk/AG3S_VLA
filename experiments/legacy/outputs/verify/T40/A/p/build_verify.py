"""T40 A: assemble AG3S/ag3s/docs/handoff/T40A.verify.json from the raw files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, datetime
import numpy as np

A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A'))
RAW = A / "raw"
TAGS = ["1807_18071", "1968_19681", "1982_19822", "1828_18281", "1995_19952", "1967_19672"]
mm = lambda v: None if v is None else round(1e3 * v, 2)

allp = json.load(open(RAW / "carry_paths_22pairs.json"))["pairs"]
cs = json.load(open(RAW / "carry_stats.json"))

# ---- 22-pair overview (stage 1, v1 listing) + binding sensitivity
pairs22 = []
for p in allp:
    tag = f"{p['episode']}_{p['seed']}"
    s = json.load(open(RAW / f"search_{tag}.json")); bd = json.load(open(RAW / f"binding_{tag}.json"))
    row = {"tag": tag, "crate_xy_m": s["crate_xy"], "apple_t0_xy_m": s["apple_t0_xy"], "windows": s["windows"]}
    for slot, k in s["kinds"].items():
        b = k["best"]
        row[slot] = {"n_proxy_blocking_both": k["n_proxy_blocking_both"],
                     "n_proxy_blocking_and_t0_feasible": k.get("n_kept_after_filters"),
                     "best_t0_feasible_xy_m": b and b["xy"], "best_t0_feasible_proxy_apple_mm": b and mm(b["proxy_score_m"])}
    row["bollard_binding_5mm_grid"] = {m: {"n_blocking_not_in_crate": r["n_blocking_not_in_crate"], "n_t0_min_gt_mm": r["n_t0_min_gt_mm"],
                                          "smallest_item_counts": r["smallest_item_counts"]} for m, r in bd.items()}
    pairs22.append(row)


def series_counts(e, w):
    out = {}
    for c in ("E0", "E3b"):
        s = e["replay"][c]["_series"]; t = np.array(s["t"]); A_ = np.array(s["A"]); C_ = np.array(s["C"]); B_ = np.array(s["B"])
        ww = w[c]; ap = t < ww["lift_t"]; win = (t >= ww["lift_t"]) & (t <= ww["place_t"])
        rob = np.minimum(A_, C_)
        out[c] = {"n_rows_robot_A_or_C_le_20mm_before_lift": int(((rob <= 0.02) & ap).sum()),
                  "first_t_robot_A_or_C_le_20mm": int(t[rob <= 0.02][0]) if (rob <= 0.02).any() else None,
                  "n_rows_A_le_0_window": int(((A_ <= 0) & win).sum()), "n_rows_B_le_0_window": int(((B_ <= 0) & win).sum()),
                  }
    return out


episodes = {}
for tag in TAGS:
    rp = json.load(open(RAW / f"replay_{tag}.json")); dt = json.load(open(RAW / f"detour_{tag}.json"))
    ent = {"crate_xy_m": rp["crate_xy"], "apple_t0_xy_m": rp["apple_t0_xy"], "windows": rp["windows"],
           "carry_stats": {c: cs["runs"][f"{c}_{tag}"] for c in ("E0", "E3b")}, "kinds": {}}
    for slot, k in rp["kinds"].items():
        cand = []
        for e in k["evaluated"]:
            r = e["replay"]
            cand.append({"xy_m": e["xy"], "yaw_rad": e["yaw_rad"], "proxy_apple_mm": mm(e["proxy_score_m"]),
                         "exact_score_mm": mm(e["exact_score_m"]), "blocks_E0": e["blocks_E0"], "blocks_E3b": e["blocks_E3b"],
                         "t0_clearance_min_by_item_mm": {it: mm(min(e["t0_clearance_m"][c][it] for c in e["t0_clearance_m"])) for it in e["t0_clearance_m"]["E0"]},
                         "phases": {c: {ph: (None if r[c][ph] is None else {g: {"min_mm": mm(v["min_m"]), "t": v["t"], "n_rows_le_0": v["n_rows_le_0"],
                                                                               **({"pair": v["pair"]} if "pair" in v else {})} for g, v in r[c][ph].items()})
                                        for ph in ("approach", "lift", "carry_grasp_to_place", "window_lift_to_place", "after_place", "all")} for c in r},
                         "counts": series_counts(e, rp["windows"])})
        d = dt["kinds"].get(slot)
        ent["kinds"][slot] = {"search": {"n_candidates": k["n_candidates"], "n_proxy_blocking_both": k["n_proxy_blocking_both"],
                                         "n_checked": k["n_checked"], "rejected": k["rejected"],
                                         "n_t0_feasible_listed": k["n_t0_feasible_listed"],
                                         "n_proxy_blocking_among_listed": k["n_proxy_blocking_among_listed"]},
                              "listed_candidates_exact": cand, "best_index": 0,
                              "best_blocks_E0_and_E3b": bool(cand and cand[0]["blocks_E0"] and cand[0]["blocks_E3b"]),
                              "detour": d and {"obstacle_top_above_table_mm": mm(d["obstacle_top_above_table_m"]),
                                               "tcp_z_max_above_table_mm": mm(d["tcp_z_max_above_table_m"]),
                                               "tcp_minus_apple_bottom_median_mm": mm(d["tcp_minus_apple_bottom_median_m"]),
                                               "apple_bottom_max_est_mm": mm(d["apple_bottom_max_est_m"]),
                                               "free_height_over_top_mm": mm(d["free_height_over_top_m"]),
                                               "recorded_apple_bottom_max_mm": {c: mm(v) for c, v in d["recorded_apple_bottom_max_m"].items()},
                                               "lateral_t0_tall_part_gap_mm": {n: mm(v) for n, v in d["lateral_t0_tall_part_gap_m"].items()},
                                               "t0_top_height_above_table_mm": {n: mm(v) for n, v in d["t0_top_height_above_table_m"].items()}}}
    episodes[tag] = ent

cw = json.load(open(RAW / "configs_written.json"))
dry = {}
for f in [x for x in sorted(glob.glob(str(A / "dry" / "*_*_*_stock.json"))) + sorted(glob.glob(str(A / "dry" / "*_*_*_cf.json"))) if not pathlib.Path(x).name.startswith("stock_")]:
    r = json.load(open(f)); nm = pathlib.Path(f).stem
    clr = [l for l in r["stdout"].splitlines() if "initial robot/object clearance" in l]
    err = [l for l in r["stdout"].splitlines() if "error:" in l]
    dry[nm] = {"outcome": r["outcome"], "obstacles_line": clr[0].strip() if clr else None, "error_line": err[0].strip() if err else None,
               "probe_object_clearance_m": (r.get("probe") or {}).get("object_clearance_m"),
               "probe_pairs_mj_ne_gjk": [x for x in (r.get("probe") or {}).get("object_pairs_mj_vs_gjk", []) if abs(x[2] - min(x[3], 0.25)) > 1e-4]}

out = {
    "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
    "step": "T40A", "date": "2026-10-03",
    "runs": [
        {"id": "A-1 carry paths", "command": "outputs/verify/T40/A/p/extract_paths.py ; carry_stats.py", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "outputs/verify/T40/A/raw/carry_paths_22pairs.json, raw/carry_stats.json", "stdout_log": "raw/extract_paths.log, raw/carry_stats.log"},
        {"id": "A-2 distance check", "command": "p/survey_dist.py ; p/test_gjk.py", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "raw/gjk_validation.log", "stdout_log": "raw/gjk_validation.log"},
        {"id": "A-3 search (22 pairs, stage 1 v1 listing)", "command": "p/run_stage1.sh", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "raw/search_<ep>_<seed>.json", "stdout_log": "raw/search_stage1.log"},
        {"id": "A-4 bollard binding sensitivity (22 pairs)", "command": "p/run_binding.sh", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "raw/binding_<ep>_<seed>.json", "stdout_log": "raw/binding_all.log"},
        {"id": "A-5 exact replay (6 pairs, v2 listing)", "command": "p/search.py --replay <tag>", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "raw/replay_<tag>.json (superseded v1 files in raw/v1/)", "stdout_log": "raw/replay_stage2_<tag>.log"},
        {"id": "A-6 detour", "command": "p/detour.py <tag>", "venv": ".venv-openpi-live (CPU)", "raw_output": "raw/detour_<tag>.json", "stdout_log": "raw/detour_<tag>.log"},
        {"id": "A-7 configs + dry check", "command": "p/make_configs.py ; p/run_dry.sh ; p/check_manager_dist.py", "venv": ".venv-openpi-live (CPU, JAX_PLATFORMS=cpu)",
         "raw_output": "configs/*.json, raw/configs_written.json, dry/*.json", "stdout_log": "dry/run_dry.log"},
        {"id": "A-8 figures", "command": "p/fig_t40a.py <6 tags>", "venv": ".venv-openpi-live (CPU, osmesa)", "raw_output": "AG3S/ag3s/docs/figures/t40/", "stdout_log": None}],
    "code_state": {"repo_main_head": "ff8de30", "benchmark_head": "db9dc2f", "pi05_TO_hybrid_head": "22d8dae",
                   "dirty_files": [], "scene_xml_md5_equal_to_T39_snapshot": True,
                   "model_xml": "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml",
                   "mujoco": "3.11.0"},
    "numbers": {
        "_comment": "distances in mm, signed (negative = penetration depth), GJK (outputs/verify/T40/A/p/gjk.py); A = EE_BODY_L + ee_finger_l1 + ee_finger_l2 geoms, B = apple_geom, C = all other robot geoms; window = [lift_t, place_t]; t = control step (15 Hz)",
        "definitions": {
            "lift_t": "first control row after the last row <= grasp_t whose apple centre z < z0 + 2 mm",
            "grasp_t_place_t": "T39 s23/analysis.json (planning t_step)",
            "t0_scene": "recorded planning row 0 qpos (E0 == E3b, max diff 0.0)",
            "hard_filters": "t0 GJK separation obstacle <-> apple, banana, orange, pear, crate, all robot geoms > 20 mm in both runs' t0 scene; obstacle origin not inside crate wall footprint",
            "blocks": "window min of min(A, B) <= 0 mm",
            "kept": "blocks in both E0 and E3b windows and passes the hard filters",
            "selection_rule": "episodes ranked by stage-1 best t0-feasible bollard proxy (apple-only) score, one pair per episode (lowest score, ties -> lower seed), first 6 with distinct crate xy",
            "detour_free_height": "kinematic max TCP z at obstacle xy (SLSQP over left arm 7 joints, xy tol 5 mm, approach axis within 15 deg of recorded carry, no collision check) - median(TCP z - apple bottom z) - obstacle top",
            "lateral_gap": "t0 GJK separation from the tall part only (bollard post / divider panel) to crate geoms and each fruit"},
        "n_pairs_both_success_T39": 22,
        "selected": [{"tag": t, "crate_xy_m": episodes[t]["crate_xy_m"]} for t in TAGS],
        "episodes": episodes,
        "pairs22_stage1": pairs22,
        "divider_22pairs": {"n_pairs_with_proxy_blocking_and_t0_feasible_divider": sum(1 for r in pairs22 if r["divider_0"]["n_proxy_blocking_and_t0_feasible"]),
                            "n_pairs": 22},
        "bollard_22pairs_stage1": {"n_pairs_with_proxy_blocking_and_t0_feasible_bollard": sum(1 for r in pairs22 if r["bollard_0"]["n_proxy_blocking_and_t0_feasible"]),
                                   "tags": [r["tag"] for r in pairs22 if r["bollard_0"]["n_proxy_blocking_and_t0_feasible"]]},
        "carry_height_44runs": cs["summary"],
        "divider_top_above_table_mm": 80.0, "bollard_top_above_table_mm": 287.0,
        "crate_top_above_table_mm": round(1e3 * cs["runs"]["E0_1807_18071"]["crate_top_above_table_m"], 2),
        "configs": cw,
        "dry_check": dry,
        "dry_check_stock_path": {"pi05_infer_model_xml_branch": "pi05_infer.py:1024-1029 (obstacle profile != clear and model != rby1_transport_14d -> MODEL_XML_PICK_PLACE_OBSTACLES = model_pick_place_obstacles.xml, block scene)",
                                 "error": "ValueError: joint 'apple_free' missing from the loaded MuJoCo model (pi05_infer.py:327 via reset_randomized_scene :363, called at :1101)",
                                 "n_configs": len([k for k in dry if k.endswith("_stock")]),
                                 "n_stock_reached_policy": sum(1 for k, v in dry.items() if k.endswith("_stock") and v["outcome"].startswith("reached")),
                                 "n_counterfactual_reached_policy": sum(1 for k, v in dry.items() if k.endswith("_cf") and v["outcome"].startswith("reached"))},
        "mj_geomDistance_vs_gjk": {"survey_box_box_n": 5850, "survey_box_box_disagree_gt_0p1mm": 82, "survey_box_mesh_n": 10590, "survey_box_mesh_disagree_gt_0p1mm": 38,
                                   "survey_cyl_any_disagree": 0, "worst_abs_diff_mm": 284.18, "gjk_vs_exact_box_box_max_err_mm": 0.0015},
    },
    "figures": [{"path": f"AG3S/ag3s/docs/figures/t40/t40a-scene-{t}.png", "kind": "scene"} for t in TAGS] + [
        {"path": "AG3S/ag3s/docs/figures/t40/t40a-distance.png", "kind": "graph"},
        {"path": "AG3S/ag3s/docs/figures/t40/t40a-table.png", "kind": "table"},
        {"path": "AG3S/ag3s/docs/figures/t40/t40a-layout.png", "kind": "table"}],
    "baseline_check": {"matched": None, "note": "regression baseline not run: lead constraint for stage A (no .venv-ag3s, no GPU); stage A runs no AG3S/trajopt code, only MuJoCo replay of recorded qpos"},
    "verdict": "facts only. 6 episodes selected; bollard best placement blocks both recorded paths in 6/6; divider best placement blocks both in 1/6 (1967_19672). stock pi05_infer rejects every config (ValueError apple_free); counterfactual transport xml reaches the policy call for 6/6 bollard configs and 0/1 divider config (reset object clearance 0.0 from mj_geomDistance, GJK 117 mm).",
    "not_measured": ["closed-loop behaviour with the obstacle (stages B/C)",
                     "self-collision / scene collision of the detour (workspace limit is kinematic only)",
                     "E3b --links gripper constraint view of the obstacle (server side)"],
}
pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40A.verify.json')).write_text(json.dumps(out, indent=1))
print("written")
