"""T43 P: assemble AG3S/ag3s/docs/handoff/T43P.verify.json from the raw files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw')); TAGS = ["1807_18071", "1968_19681", "1982_19822", "1995_19952", "1967_19672"]
KEYS = ("u", "v", "dyaw_deg", "x", "y", "yaw", "h", "a_min", "d_min", "b_crate_min", "b_obj_min", "client_reset_arms", "client_reset_object",
        "c_min_per_run", "c_apple_per_run", "free_height_over_bar_m", "post_in_crate", "inside_handle")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
base = json.load(open(_ag3s_workspace_path('outputs/verify/T43/P/baseline/judge.json')))["legacy"]
eps = {}
for t in TAGS:
    s = json.load(open(RAW / f"select_p_{t}.json")); old = json.load(open(RAW / f"old_f_{t}.json"))
    ps = {suf or "coarse": json.load(open(RAW / f"psearch_{t}{suf}.json")) for suf in ("", "_refine", "_ext") if (RAW / f"psearch_{t}{suf}.json").exists()}
    rec = {"n_candidates": {k: v["n_candidates"] for k, v in ps.items()}, "counts": {k: v["counts"] for k, v in ps.items()},
           "crate_xy_m": ps["coarse"]["crate_xy"], "crate_yaw_rad": ps["coarse"]["crate_yaw_rad"],
           "warmup_replay_final_vs_planning_row0_maxabs_qpos": ps["coarse"]["warmup_final_vs_planning_row0_maxabs_qpos"],
           "n_pass_main": s["n_main"], "n_pass_relaxed": s["n_relaxed"],
           "old_F_r1": {k: old.get(k) for k in KEYS if k in old} | {"config": old["config"], "pass": {k: old[k] for k in ("pass_a", "pass_b", "pass_c", "pass_c_relaxed", "pass_d")}},
           "chosen_main": s["chosen"], "chosen_relaxed": s["chosen_relaxed"], "tradeoff": s["tradeoff"]}
    for k in ("main", "relaxed"):
        if k in s:
            g = s[k]["gjk_check"]
            rec[f"{k}_config"] = s[k]["config"]; rec[f"{k}_profile"] = s[k]["profile"]
            rec[f"{k}_gjk_max_abs_diff_m"] = max((v.get("abs_diff_m") or 0) for v in g.values() if isinstance(v, dict))
            rec[f"{k}_gjk_check"] = g
            dr = s[k]["dryrun"]; rec[f"{k}_client_dryrun"] = {q: dr.get(q) for q in ("outcome", "obstacles_line", "robot_clearance_m", "object_clearance_m", "model_xml")}
    if "relaxed" not in s and s.get("chosen_relaxed"):
        rec["relaxed_config"] = "same placement as main"
    eps[t] = rec
st = json.load(open(RAW / "selftest_bound.json"))
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "T43P", "date": "2026-10-05",
 "runs": [
  {"id": "baseline legacy (CPU)", "command": "outputs/verify/T43/P/baseline/run_baseline.sh; judge.py", "venv": ".venv-openpi-live (JAX_PLATFORMS=cpu, nice 19)",
   "raw_output": "outputs/verify/T43/P/baseline/{legacy.json,judge.json}", "stdout_log": "outputs/verify/T43/P/baseline/legacy.log"},
  {"id": "capture", "command": "outputs/verify/T43/P/p/capture.py (flib.capture_reset on snapshot ${AG3S_PROJECT_ROOT}-t40h)", "venv": ".venv-openpi-live (CPU)",
   "raw_output": "outputs/verify/T43/P/raw/capture/", "stdout_log": "outputs/verify/T43/P/raw/capture.log"},
  {"id": "search", "command": "p/run_psearch.sh (psearch.py coarse), p/run_refine.sh (refine.py), p/run_ext.sh (ep1995 extended), p/fix_e.py", "venv": ".venv-openpi-live (CPU, <= 4 processes, nice 19)",
   "raw_output": "outputs/verify/T43/P/raw/psearch_<TAG>{,_refine,_ext}.json", "stdout_log": "raw/psearch_w*.log, refine_r*.log, psearch_ext.log, fix_e.log"},
  {"id": "select + cross-check", "command": "p/select_p.py (gjk.py at argmin pairs; dryrun.py = client reset check)", "venv": ".venv-openpi-live (CPU)",
   "raw_output": "raw/select_p_<TAG>.json, raw/dryrun_t43p_*.json, configs/t43p_*.json", "stdout_log": "raw/select_p.log"},
  {"id": "old F under P metrics", "command": "p/old_f.py", "venv": ".venv-openpi-live (CPU)", "raw_output": "raw/old_f_<TAG>.json", "stdout_log": "raw/old_f.log"},
  {"id": "self-test", "command": "p/selftest_bound.py", "venv": ".venv-openpi-live (CPU)", "raw_output": "raw/selftest_bound.json", "stdout_log": "raw/selftest_bound.log"},
  {"id": "figures", "command": "p/render_p.py (RP_KIND main/relaxed/tradeoff), p/fig_scene.py, p/fig_layout.py, p/fig_table.py", "venv": ".venv-openpi-live (CPU, osmesa)",
   "raw_output": "outputs/verify/T43/P/render/", "stdout_log": "-"}],
 "code_state": {"benchmark_head_at_baseline": "d309550 (baseline/pre.txt)", "benchmark_head_at_write": git("-C", _ag3s_workspace_path('legacy/benchmark'), "rev-parse", "--short", "HEAD"), "benchmark_used_by_search": "figstyle only (search/distance code = pi05_TO_hybrid bd6a002 snapshot)",
                "pi05_TO_hybrid": "bd6a002 via snapshot ${AG3S_PROJECT_ROOT}-t40h (pick_place_obstacles.py, convex_distance.py, pi05_infer.py md5 = git show bd6a002)",
                "dirty_files": [], "scripts": "outputs/verify/T43/P/p/ (copied from outputs/verify/T40/F/p/, SNAP -> ${AG3S_PROJECT_ROOT}-t40h)"},
 "numbers": {
  "_comment": "distances in m unless _mm; distance = bd6a002 PickPlaceObstacleManager.pair_distance (certified mj_geomDistance / GJK); values are capped at 0.06 m (0.06 = '>= 60 mm')",
  "baseline_legacy_cpu": {k: base[k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")} | {"ref": "14/15, 15/15, -29.031048280806342 (T36/T40 legacy sha1 40798fb...)", "curobo": "not run (lane P is CPU only)"},
  "definitions": {"search": ps_doc if (ps_doc := json.load(open(RAW / "psearch_1807_18071.json"))["definitions"]) else None,
                  "selection": json.load(open(RAW / "select_p_1807_18071.json"))["definitions"]},
  "geometry": {"hurdle": "post centres +-0.120 m from bar centre; foot cylinder r 0.035 (z 0.820-0.832); sleeve r 0.014 to 0.18 m above table; bar capsule r 0.012 half-length 0.12",
               "crate_frame": "walls |x_rel| <= 0.094, |y_rel| <= 0.150 (outer); left handle bar y_rel 0.192-0.208, x_rel +-0.050, top 157 mm above table; handle posts x_rel +-0.058; handle arms y_rel 0.150-0.200, 73-87 mm above table",
               "straddle_check_m": {"crate_outer_width_x": 0.188, "post_spacing": 0.24, "needed_for_both_feet_clear_of_crate_x_walls": 0.258}},
  "self_test": {"segment_bound_vs_sphere_bound_on_old_F_max_abs_m": max(v["max_abs_diff"] for v in st["vs_old_f"].values()),
                "bb_vs_bruteforce_6_random_max_abs_m": max(r["max_abs_diff"] for r in st["brute"]),
                "eval_s_per_candidate_single": [round(v["eval_s"], 3) for v in st["vs_old_f"].values()]},
  "episodes": eps,
  "resources": {"processes": "<= 4 concurrent, nice 19, CPU only, 1 thread", "worker_rss_gb_observed": "0.49-0.58",
                "working_set_before_start_gib": [45, 41, 32], "servers": 0, "gpu": "not used"}},
 "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43p-scene.png", "kind": "scene"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43p-layout.png", "kind": "graph (top-down layout)"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43p-table.png", "kind": "table"}],
 "baseline_check": {"matched": base["matched"], "note": "legacy CPU: violated_start 14/15, has_target 15/15, frame0 -29.031048280806342 mm, sha1 40798fb (= T40 legacy); cuRobo not run"},
 "verdict": "기준선 일치 (legacy, CPU). cuRobo 기준선은 돌리지 않음.",
 "not_measured": ["closed-loop runs with the new configs (E0/E3b)", "cuRobo regression baseline (GPU)",
                  "criterion e is a kinematic upper bound (no collision check), as in T40 F",
                  "crate motion during place (crate clearance of the chosen posts is a t0/reset value)"]}
pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43P.verify.json')).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("ok")
