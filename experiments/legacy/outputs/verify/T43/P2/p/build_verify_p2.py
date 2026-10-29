"""T43 P2: write AG3S/ag3s/docs/handoff/T43P2.verify.json from the raw files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
import p2lib as F
H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43P2.verify.json'))
J = lambda p: json.loads(pathlib.Path(p).read_text())
judge = J(F.ROOT / "baseline/judge.json")["legacy"]
probe = J(F.RAW / "probe.json")
md5s = (F.RAW / "code_md5_start.txt").read_text().split("\n")
md5e = (F.RAW / "code_md5_end.txt").read_text().split("\n")
eps = {}
for tag in F.TAGS:
    s = J(F.RAW / f"select_p2_{tag}.json"); k = J(F.RAW / f"check_seeds_{tag}.json")
    srch = json.loads((F.RAW / f"p2search_{tag}.json").read_text())
    e = s["chosen"]; cc = s["crosscheck"]; pr = probe[tag]
    eps[tag] = {
        "arm": pr["arm"], "apple_side": "+y (left handle)" if s["side"] > 0 else "-y (right handle)",
        "crate_xy_m": s["crate_xy"], "crate_yaw_rad": s["crate_yaw_rad"],
        "reset_qpos_maxabs_vs_T43P_capture": pr.get("reset_qpos_maxabs_vs_T43P"),
        "windows": s["windows"], "warmup_replay_final_vs_planning_row0_maxabs_qpos": srch["warmup_final_vs_planning_row0_maxabs_qpos"],
        "search_counts": srch["counts"], "n_candidates": srch["n_candidates"], "n_pass_main": s["n_pass_main"],
        "config": s["config"], "profile": "t43p2_overbar",
        "chosen": {kk: e[kk] for kk in ("u", "v", "dyaw_deg", "x", "y", "yaw", "h", "L", "r", "legs_crate_frame", "legs_outside_crate",
                                       "bar_over_interior", "inside_handles", "place_spot_clear", "pass_c_apple")},
        "a_prime_min_m_uncapped": k["uncapped_search_seed"]["a_min"], "a_prime_argmin": k["uncapped_search_seed"]["args"]["warmup"][1],
        "d_min_m_uncapped": k["uncapped_search_seed"]["d_min"],
        "b_crate_min_m": e["b_crate_min"], "b_crate_capped_at_0.06": e["b_crate_min"] >= 0.06, "b_obj_min_m_capped": e["b_obj_min"],
        "client_reset_dryrun": {kk: s["dryrun"].get(kk) for kk in ("outcome", "active_slots", "robot_clearance_m", "object_clearance_m", "obstacles_line")},
        "c_min_per_run_m": e["c_min_per_run"], "c_apple_per_run_m": e["c_apple_per_run"],
        "c_argmin_pairs": {kk: cc["bruteforce"]["args"].get(f"{kk}_car") for kk in ("E0", "E3b")},
        "e_free_height_over_bar_m": e["free_height_over_bar_m"], "e_tcp_z_max_above_table_m": e["tcp_z_max_above_table_m"],
        "e_tcp_minus_apple_bottom_median_m": e["tcp_minus_apple_bottom_median_m"], "e_crossing_crate_frame": e.get("crossing_crate_frame"),
        "e_drop_gap_m": e["drop_gap_m"], "apple_extent_xyz_m": pr["apple_extent_xyz"],
        "apple_final_y_rel_m": s["apple_final_y_rel"],
        "other_seeds": {n: {kk: r.get(kk) for kk in ("success", "lift_t", "car_window", "car_min", "car_apple", "pre_min", "pre_min_all_rows", "t0_robot")}
                        for n, r in k["runs"].items()},
        "crosscheck": {"activate_vs_place_geom_xpos_maxabs": cc["activate_vs_place"]["geom_xpos_maxabs"],
                       "activate_vs_place_geom_xmat_maxabs": cc["activate_vs_place"]["geom_xmat_maxabs"],
                       "bruteforce_vs_search": {"bruteforce": {kk: cc["bruteforce"][kk] for kk in ("a_min", "d_min", "b_crate_min", "c_min_per_run")},
                                                "search": cc["search"]},
                       "gjk_vs_manager_at_argmin_m": {kk: [v["manager"], v["gjk_signed"]] for kk, v in cc["gjk_at_argmin"].items()}},
        "tradeoff": s["tradeoff"] | {"maximin": {kk: v for kk, v in (s["tradeoff"].get("maximin") or {}).items() if kk in ("u", "v", "dyaw_deg", "h", "L", "slack")}},
    }
out = {
    "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
    "step": "T43P2", "date": "2026-10-05",
    "runs": [
        {"id": "baseline legacy (CPU)", "command": "outputs/verify/T43/P2/baseline/run_baseline.sh; judge.py", "venv": ".venv-openpi-live (JAX_PLATFORMS=cpu, nice 19)",
         "raw_output": "outputs/verify/T43/P2/baseline/{legacy.json,judge.json}", "stdout_log": "outputs/verify/T43/P2/baseline/legacy.log"},
        {"id": "capture", "command": "outputs/verify/T43/P2/p/capture.py (p2lib.save_capture, live pi05_TO_hybrid fce4af8 client main() to the reset check)",
         "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T43/P2/raw/capture/", "stdout_log": "outputs/verify/T43/P2/raw/capture.log"},
        {"id": "probe", "command": "p/probe.py", "venv": ".venv-openpi-live (CPU)", "raw_output": "raw/probe.json", "stdout_log": "raw/probe.log"},
        {"id": "search", "command": "p/run_p2search.sh (p2search.py, 4 workers, nice 19, 1 thread)", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "raw/p2search_<TAG>.json", "stdout_log": "raw/p2search_w{1..4}.log"},
        {"id": "select + cross-check", "command": "p/select_p2.py (dryrun_p2.py = client reset check)", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "raw/select_p2_<TAG>.json, raw/dryrun_p2_<TAG>.json, configs/t43p2_ep<EP>_s<SEED>.json (v1 ranking: raw/select_v1/)", "stdout_log": "raw/select_p2.log"},
        {"id": "other seeds + uncapped a', d", "command": "p/check_seeds.py", "venv": ".venv-openpi-live (CPU)", "raw_output": "raw/check_seeds_<TAG>.json", "stdout_log": "raw/check_seeds.log"},
        {"id": "figures", "command": "p/render_p2.py; p/fig_scene_p2.py; p/fig_layout_p2.py; p/fig_table_p2.py", "venv": ".venv-openpi-live (CPU, osmesa)",
         "raw_output": "outputs/verify/T43/P2/render/", "stdout_log": "raw/render_p2.log"},
    ],
    "code_state": {
        "benchmark_head": subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "pi05_TO_hybrid": "live tree fce4af8, clean (git status empty at start)",
        "md5_start": [l for l in md5s if l], "md5_end": [l for l in md5e if l], "md5_unchanged": md5s == md5e,
        "dirty_files": [], "scripts": "outputs/verify/T43/P2/p/ (p2search.bb_multi = T43 P psearch.bb_multi with the overbar geoms)",
    },
    "numbers": {
        "_comment": "distances in m; pair_distance = fce4af8 PickPlaceObstacleManager.pair_distance (certified mj_geomDistance / exact GJK); search values capped at 0.06 m (a', d uncapped values from check_seeds with cap 0.30)",
        "baseline_legacy_cpu": {k: judge[k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")},
        "definitions": {"search": J(F.RAW / f"p2search_{F.TAGS[0]}.json")["definitions"], "selection": J(F.RAW / f"select_p2_{F.TAGS[0]}.json")["selection"],
                        "other_seeds": J(F.RAW / f"check_seeds_{F.TAGS[0]}.json")["definitions"]},
        "geometry": {"crate_frame_from_probe": {"outer": "|x_rel| <= 0.090, |y_rel| <= 0.150", "interior": "|x_rel| < 0.082, |y_rel| < 0.142",
                                                "wall_top_above_table_m": 0.12, "handle_bar_y_rel": "0.192..0.208", "handle_top_above_table_m": 0.157},
                     "overbar": "capsule bar along slot x at bar_height h (bar centre), radius r; 2 cylinder legs r 0.012 at position +- R(yaw)[L/2,0,0] from the table to the bar axis",
                     "colour": {"rgba": F.RED, "note": "per-slot rgba in every config (visual geoms only)"}},
        "episodes": eps,
        "resources": {"processes": "<= 4 concurrent (search 4; select/check/render 1 + S2 replays), nice 19", "worker_rss_gb_observed": "0.56-0.57 (search)",
                      "working_set_before_start_gib": [19, 40], "servers": 0, "gpu": "not used"},
    },
    "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43p2-layout.png", "kind": "graph (top-down layout)", "sidecar": "AG3S/ag3s/docs/figures/t43/t43p2-layout.json"},
                {"path": "AG3S/ag3s/docs/figures/t43/t43p2-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t43/t43p2-scene.json"},
                {"path": "AG3S/ag3s/docs/figures/t43/t43p2-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t43/t43p2-table.json"}],
    "baseline_check": {"matched": judge["matched"], "note": f"legacy CPU: violated_start {judge['violated_start']}, has_target {judge['has_target']}, frame0 {judge['frame0_clearance_before_mm']} mm, sha1 {judge['sha1']['repr_json_list'][:7]}; cuRobo not run (CPU only)"},
    "verdict": "기준선 일치 (legacy, CPU). cuRobo 기준선은 돌리지 않음.",
    "not_measured": ["closed-loop runs with the P2 configs (E0 / E3b / V1 / V3 / V4)", "cuRobo regression baseline (GPU)",
                     "criterion e is a kinematic upper bound (no collision check), as in T43 P", "crate motion during the place (crate clearance is a reset/t0 value)",
                     "whether the arm (not the apple) clears the bar while descending into the crate after passing over it"],
}
H.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(H)
