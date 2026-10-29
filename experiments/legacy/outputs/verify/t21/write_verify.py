from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np
from pathlib import Path
O = Path(_ag3s_workspace_path('outputs/verify/t21'))
H = Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
S = {r: json.loads((O / f"summary_{r}.json").read_text()) for r in ("t14", "t17")}
R = {r: json.loads((O / f"replay_{r}.json").read_text()) for r in ("t14", "t17")}
base = json.loads((O / "baseline.json").read_text())
KEYS = ["live_exact", "live_proxy", "a_rec", "b1_rec", "b2_rec", "a_t20", "b1_t20", "b2_t20", "a_default", "b1_default", "b2_default"]

def m1(run):
    out = {"chunks": S[run]["chunks"], "t_steps": S[run]["t_steps"], "rows_per_chunk": S[run]["per_rule"]["live_exact"]["per_chunk"][0]["n_all"],
           "rows_definition": "authorized-link spheres (ee_left, ee_finger_l1, ee_finger_l2; 61) x (q_now + reference steps 1..8) = 549",
           "rules": {}}
    for k in KEYS:
        v = S[run]["per_rule"][k]
        out["rules"][k] = {
            "field": "exact" if k == "live_exact" else "proxy",
            "fraction_in_window_per_chunk": [r["frac_all"] for r in v["per_chunk"]],
            "fraction_in_window_now_per_chunk": [r["frac_now"] for r in v["per_chunk"]],
            "by_tier_total": v["by_tier_all_total"], "finger_by_tier_total": v["finger_by_tier_all_total"],
            "coarse_or_outside_rows_per_chunk": [r["by_tier_all"]["coarse"] + r["by_tier_all"]["outside"] for r in v["per_chunk"]],
            "n_unresolved_total": v["n_unresolved_total"], "n_rows_total": v["n_rows_total"],
            "chunks_finger_rows_all_fine_or_target_free": f"{v['chunks_finger_all_resolved']}/{v['n_chunks']}",
            "chunks_finger_q_now_answered_by_fine_or_target_free": f"{v['chunks_finger_now_in_fine_or_tf']}/{v['n_chunks']}",
        }
    col = S[run]["collision"]["live_exact"]
    out["TO_side_authorized_rows_exact_field_returned_trajectory"] = {
        "source": "linearize worst_row.authorized_rows.by_tier (T21 implementation) of variant 'collision' on the recorded field",
        "by_tier_total": col["authorized_rows_by_tier_total_returned"], "verifiable_per_chunk": col["authorized_rows_verifiable"]}
    return out

def m4(run):
    return {k: {"basis": S[run]["per_rule"][k]["basis_counts"], "overflow": S[run]["per_rule"][k]["overflow_counts"]}
            for k in KEYS if k not in ("live_exact", "live_proxy")}

def m2(run):
    return {k: {kk: v[kk] for kk in ("per_chunk_median_mm", "per_chunk_max_mm", "pooled_median_mm", "pooled_max_mm", "returned", "sqp_iterations", "worst_row_tier")}
            for k, v in S[run]["collision"].items()}

def m3(run):
    out = {"ball_removed_cloud_points_by_label": {}, "finger_rules": {}}
    for b in S[run]["ball_removed_by_label"]:
        out["ball_removed_cloud_points_by_label"][b["chunk"]] = {m: (None if v is None else {l: x for l, x in v.items() if l in ("table", "crate", "apple", "banana", "pear", "orange")}) for m, v in b.items() if m != "chunk"}
    for k in ("live_exact", "b1_rec", "b2_rec", "a_t20", "b1_t20", "b2_t20"):
        rows = S[run]["per_rule"][k]["per_chunk"]
        out["finger_rules"][k] = {
            "target_free_distance_min_mm": [r["m3"]["target_free_distance_min_mm"] for r in rows],
            "truth_table_min_mm": [r["m3"]["truth_table_min_mm"] for r in rows],
            "truth_crate_min_mm": [r["m3"]["truth_crate_min_mm"] for r in rows],
            "truth_apple_min_mm": [r["m3"]["truth_apple_min_mm"] for r in rows],
            "nearest_kept_cloud_point_label_counts": [r["m3_nearest"] for r in rows] if rows[0]["m3_nearest"] else None,
            "cloud_in_window_kept_in_target_free": ([{l: [x["in_window"], x["kept_in_target_free"]] for l, x in r["m3_cloud"].items() if l in ("table", "crate", "apple")} for r in rows] if rows[0]["m3_cloud"] else None),
        }
    return out

fid = {}
for run in ("t14", "t17"):
    F = S[run]["fidelity"]
    fid[run] = {"tier_agreement_live_window": [f["tier_agreement_all"] for f in F],
                "fine_query_diff_mm_median": [(f.get("fine_query_diff_mm") or {}).get("median") for f in F],
                "fine_query_diff_mm_max_abs": [(f.get("fine_query_diff_mm") or {}).get("max_abs") for f in F],
                "free_query_diff_mm_median": [(f.get("free_query_diff_mm") or {}).get("median") for f in F],
                "fine_voxel_diff_above_support_mm_median": [f["fine_voxel_diff_mm_exact_0_100mm_above_support"]["median"] for f in F],
                "fine_voxel_diff_above_support_mm_p90": [f["fine_voxel_diff_mm_exact_0_100mm_above_support"]["p90"] for f in F],
                "fine_voxel_diff_all_mm_p90": [f["fine_voxel_diff_mm_exact_0_100mm"]["p90"] for f in F],
                "collision_control_pooled_mm": {"exact": [S[run]["collision"]["live_exact"]["pooled_median_mm"], S[run]["collision"]["live_exact"]["pooled_max_mm"]],
                                                "proxy_same_window": [S[run]["collision"]["live_proxy"]["pooled_median_mm"], S[run]["collision"]["live_proxy"]["pooled_max_mm"]]}}

manip = {run: {c["chunk"]: {m: [v["n"], None if v["to_apple_mm"] is None else round(v["to_apple_mm"], 1)] for m, v in c.items() if m != "chunk"} for c in S[run]["manipulated"]} for run in ("t14", "t17")}

V = {
 "_comment": "ag3s-verifier. 숫자와 경로만. 해석·판정 없음.",
 "step": "T21", "date": "2026-09-28",
 "runs": [
  {"id": "baseline", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms",
   "venv": ".venv-ag3s", "raw_output": "outputs/verify/t21/baseline.json", "stdout_log": "outputs/verify/t21/baseline.log", "concurrent_load": "none of mine (run first, alone)"},
  {"id": "pytest_all", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s", "venv": "third_party/openpi/.venv",
   "stdout_log": "outputs/verify/t21/pytest_openpi.log", "git_status": "outputs/verify/t21/git_status_at_pytest.txt"},
  {"id": "pytest_t21_files", "command": ".venv-ag3s/bin/python -m pytest -q tests/ag3s/test_fine_window.py tests/trajopt/test_worst_row_tier.py tests/trajopt/test_violation_pair.py", "venv": ".venv-ag3s", "stdout_log": "outputs/verify/t21/pytest_ag3s_t21files.log"},
  {"id": "manipulated_replay", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python outputs/verify/t21/replay_manip.py", "venv": ".venv-ag3s",
   "raw_output": ["outputs/verify/t21/manip_t14.npz", "outputs/verify/t21/manip_t14.json", "outputs/verify/t21/manip_t17.npz", "outputs/verify/t21/manip_t17.json"], "stdout_log": "outputs/verify/t21/replay_manip.log"},
  {"id": "window_replay", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python outputs/verify/t21/t21_replay.py <t14|t17>", "venv": ".venv-ag3s",
   "raw_output": ["outputs/verify/t21/replay_t14.json", "outputs/verify/t21/replay_t17.json"], "stdout_log": ["outputs/verify/t21/replay_t14.log", "outputs/verify/t21/replay_t17.log"]},
  {"id": "extraction", "scripts": ["outputs/verify/t21/extract.py", "outputs/verify/t21/write_verify.py", "outputs/verify/t21/make_figures.py"],
   "raw_output": ["outputs/verify/t21/summary_t14.json", "outputs/verify/t21/summary_t17.json", "outputs/verify/t21/extract_t14.log", "outputs/verify/t21/extract_t17.log", "outputs/verify/t21/m3_extract.log"]}],
 "code_state": {"repo": "benchmark", "head": "368d723", "dirty_files_snapshot": "outputs/verify/t21/git_status_at_verify.txt", "md5": "outputs/verify/t21/code_md5.txt",
                "note": "T23 implementer edits in parallel (safe_policy/wire/client/pi05_infer); pytest snapshot in git_status_at_pytest.txt"},
 "reproduction_scope": {
   "records": {"T14": "outputs/live_test/20260928_t14 chunks 11-20 (seq 12-21, t 88-160)", "T17": "outputs/live_test/20260928_t17 chunks 9-15 (t 72-120)"},
   "window_placement": "exact rule replay: curobo_builder.place_fine_window (working tree), shape 128^3, voxel 5 mm, hand_pad = geometry.safety_margin 0.05 + 2 x 5 mm, manipulated_pad 10 mm; hand = contact-authorized link spheres from robot.sphere_centers_numeric; b1 = q_now only; b2 = q_now + reference_chunk steps 0..7 via layout.full_q (8 rows)",
   "rules": {"live": "recorded fine window of the server record (T21 before)", "a": "placement=target_centroid", "b1": "hand_swept, execution_path none (live wiring today)", "b2": "hand_swept, execution_path = reference 8 steps"},
   "manipulated_sources": {"rec": "recorded target_points of the chunk (live grounding)", "t20": "working-tree TargetConfirm replay, target_switch_min_score 0.1, target_lost_frames 20", "default": "working-tree TargetConfirm replay, AG3SConfig defaults (min_score 0.0, lost None)"},
   "manipulated_per_chunk_n_points_and_centroid_to_apple_mm": manip,
   "tiers_exact": "only for the recorded live window: fixed_input_ablation.build_field (recorded coarse/fine/free grids) -> CuroboEsdfField.answer_tier(target_free=True)",
   "tiers_proxy": "field NOT rebuilt by cuRobo (no depth in records). proxy = recorded coarse grid (exact) + fine and target-free layers at the replayed window from the union of recorded filtered_cloud of chunks 0..k (2.5 mm dedup), unsigned nearest-point distance on 128^3 x 5 mm, clipped 0.4 m; target-free = same minus cloud points inside the TargetBall (centre=mean, r = max spread + 5 mm, z >= support_plane_z + 5 mm; recorded ball for 'rec'). No TSDF carving, no signed interior.",
   "collision_variant": "fixed_input_ablation 'collision' (w_smooth 0, w_continuity 0, clock off, max_iterations default), fresh solver per chunk; field+scene swapped; m3 definition of T24 (left fingertip = mean of ee_finger_l1/l2 sphere centres, displacement vs reference, 8 exec steps)",
   "m3_truth": "MuJoCo model_transport.xml at the planning row qpos (base = world, qpos[0:3] = 0); table/crate collision boxes exact, apple = ellipsoid on geom_size; cloud points labelled by nearest body within 15 mm, else 'other'"},
 "numbers": {
  "baseline": {"violated_start": 13, "frames": 15, "has_target": 15, "frame0_clearance_before_mm": base["frames"][0]["clearance_before_mm"]},
  "m5_regression": {"pytest_tests_trajopt_ag3s_openpi_venv": {"passed": 1292, "failed": 0}, "pytest_t21_files_venv_ag3s": {"passed": 58, "failed": 0}},
  "replay_checks": {"fk_sphere_centers_numeric_vs_linearizer_max_mm": {r: S[r]["fk_vs_linearizer_max_mm"] for r in S},
                    "live_window_vs_place_fine_window_target_centroid_recorded_points_max_mm": {r: S[r]["live_vs_a_rec_lower_mm_max"] for r in S},
                    "collision_exact_field_T17_reproduces_T24_m3": {"pooled_median_mm": S["t17"]["collision"]["live_exact"]["pooled_median_mm"], "pooled_max_mm": S["t17"]["collision"]["live_exact"]["pooled_max_mm"], "T24": [51.033594499171954, 195.47857252345915]}},
  "proxy_fidelity": fid,
  "m1_window_and_tiers": {"T14": m1("t14"), "T17": m1("t17")},
  "m2_collision_fingertip": {"T14": m2("t14"), "T17": m2("t17"),
     "recorded_grid_window_move": "not possible: recorded fine/free grids exist only at the live window; values at moved windows are proxy (see reproduction_scope)"},
  "m3_table_crate_in_target_free": {"T14": m3("t14"), "T17": m3("t17")},
  "m4_overflow": {"T14": m4("t14"), "T17": m4("t17")},
 },
 "figures": [{"path": "AG3S/ag3s/docs/figures/t21/t21-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t21/t21-scene.json"},
             {"path": "AG3S/ag3s/docs/figures/t21/t21-graph.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t21/t21-graph.json"},
             {"path": "AG3S/ag3s/docs/figures/t21/t21-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t21/t21-table.json"}],
 "baseline_check": {"matched": True, "note": "위반으로 시작 13/15 · has_target 15/15 · frame0 clearance_before +0.18597810857728936 mm — skill 기준값과 같다"},
 "verdict": "기준선 일치. tests 1292 passed / 0 failed. 창 배치·계층 수치는 위 numbers 에만 있다.",
 "not_measured": [
  "cuRobo field rebuild at the new windows (no depth in T14/T17 records) — fine/target-free values at b1/b2/a_t20 windows are proxy",
  "TSDF free-space carving and signed interior in the proxy",
  "live AG3S stats['window'] / notes from a real run (no new closed-loop run; 8123 untouched)",
  "T19 guard rule effect on the cloud (records are pre-T19 self-filter output)",
  "latency of the new window path",
  "T14 chunks outside 11-20, T17 chunks outside 9-15"],
 "process_notes": [
  "impl.md 는 1-190 행(요약·바뀐 파일·규칙·config·기록 형식·단위 검증·verifier 가 알아야 할 것·재지 않은 것)만 측정 전에 읽었다. '내가 기대하는 결과' 절(191-)은 읽지 않았다.",
  "proxy 첫 시도(단일 chunk cloud)는 T17 chunk 12 에서 fine voxel diff median 7.5 mm · p90 65 mm 였고 누적 cloud 로 바꿨다 (위 fidelity 는 누적 기준).",
  "nearest kept point label 'other' 가 T14 finger 에서 342/342: T14 chunk 12 · T17 chunk 12 확인 — 그 점들은 q_now 손 구 표면에서 median 14.0 / 18.0 mm, chunk 0-6 에 처음 나온 점 (누적 proxy 에 carving 없음)."],
}
(H / "T21.verify.json").write_text(json.dumps(V, ensure_ascii=False, indent=1))
p = json.loads((H / "T21.verify.partial.json").read_text())
p["entries"].append({"kind": "final", "verify_json": "AG3S/ag3s/docs/handoff/T21.verify.json"})
(H / "T21.verify.partial.json").write_text(json.dumps(p, ensure_ascii=False))
print("ok")
