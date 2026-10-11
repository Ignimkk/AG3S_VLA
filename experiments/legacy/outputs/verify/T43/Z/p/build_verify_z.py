"""T43 Z — write/merge numbers.Z1 and numbers.Z4 into AG3S/ag3s/docs/handoff/T43Z.verify.json (numbers and paths only).
Other sections (e.g. numbers.Z5 written by another verifier) are kept as they are."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
Z = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Z')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43Z.verify.json'))
N = json.load(open(Z / "raw/numbers_z.json")); RL = json.load(open(Z / "raw/z4_release.json"))
d = json.load(open(OUT)) if OUT.exists() else {"_comment": "ag3s-verifier (T43 Z, 2026-10-06). 숫자와 경로만 — 해석 없음.", "step": "T43Z", "date": "2026-10-06",
                                               "runs": [], "numbers": {}, "figures": [], "not_measured": []}
d.setdefault("numbers", {}); d.setdefault("runs", []); d.setdefault("figures", []); d.setdefault("not_measured", [])
d["runs"] = [r for r in d["runs"] if not str(r.get("id", "")).startswith("Z1_") and not str(r.get("id", "")).startswith("Z4_")] + [
    {"id": "Z4_physics_replay", "command": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa outputs/verify/T43/Z/p/run_replays.sh "
     "(T43/Y/p/replay_y.py unchanged; E0 V1 V3 with ${AG3S_PROJECT_ROOT}-t43w, V5S V5B with T43W_SNAP=${AG3S_PROJECT_ROOT}-t43b; 3 workers)",
     "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/Z/replay/*.npz|*.json", "stdout_log": "outputs/verify/T43/Z/raw/replay_*.log"},
    {"id": "Z1_held", "command": "outputs/verify/T43/Z/p/z1_held.py outputs/verify/T43/Z/raw/z1_held.json", "raw_output": "outputs/verify/T43/Z/raw/z1_held.json",
     "stdout_log": "outputs/verify/T43/Z/raw/z1_held.log"},
    {"id": "Z4_release", "command": "outputs/verify/T43/Z/p/z4_release.py outputs/verify/T43/Z/raw/z4_release.json", "raw_output": "outputs/verify/T43/Z/raw/z4_release.json",
     "stdout_log": "outputs/verify/T43/Z/raw/z4_release.log"},
    {"id": "Z1_Z4_aggregate", "command": "outputs/verify/T43/Z/p/agg_z.py; p/fig_z1_closeup.py; p/fig_z1_scene.py; p/build_verify_z.py", "raw_output": "outputs/verify/T43/Z/raw/numbers_z.json"}]
d["code_state_Z1_Z4"] = {"root_head": "9514c22", "inputs": "outputs/verify/T43/W/{runs,runs_blind,analysis,analysis_blind} + outputs/verify/T43/Y/{replay,raw}",
                         "no_new_closed_loop_runs": True}
z1 = N["Z1"]
d["numbers"]["Z1"] = {"definitions": "outputs/verify/T43/Z/p/z1_held.py (docstring)", "margin_mm": z1["margin_mm"], "crate_inner_mm": z1["crate_inner_mm"],
                      "apple_half_extent_mm": z1["apple_half_extent_mm"], "arms": z1["arms"], "per_run": z1["per_run"]}
reps = sorted(Z.glob("replay/*.json"))
mism = sum(json.load(open(p))["qpos_mismatch_rows"] for p in reps)
z4 = N["Z4"]
d["numbers"]["Z4"] = {"definitions": "outputs/verify/T43/Z/p/z4_release.py (docstring)",
                      "replay": {"n_runs_new": len(reps), "qpos_mismatch_rows_total_new": mism, "n_runs_reused_from_Y": 38},
                      "arms": z4["arms"], "pairs": z4["pairs"], "n_flips": len(z4["flips"]), "flips": z4["flips"], "crosscheck_vs_y_metrics": z4["crosscheck_vs_y_metrics"],
                      "per_run": z4["release_only_in_runs"]}
figs = [("AG3S/ag3s/docs/figures/t43/t43z-held-sphere-closeup.png", "table_diagram", "Z1"), ("AG3S/ag3s/docs/figures/t43/t43z-held-sphere-scene.png", "scene", "Z1"),
        ("AG3S/ag3s/docs/figures/t43/t43z-held-sphere-radii.png", "graph", "Z1"), ("AG3S/ag3s/docs/figures/t43/t43z-release-rescore.png", "graph_table", "Z4")]
paths = {f["path"] for f in d["figures"]}
d["figures"] += [{"path": p, "kind": k, "section": s, "sidecar": p.replace(".png", ".json")} for p, k, s in figs if p not in paths]
d["baseline_check_Z1_Z4"] = {"matched": None, "note": "regression-baseline 미실행 (작업 지시: CPU 재생 · 집계만, .venv-ag3s 금지). 재생 충실도: 새 재생 75 run qpos_mismatch_rows 합 %d; V4/V5 38 run release_t · release_mode 가 Y y_metrics 와 불일치 %d / %d." % (
    mism, len(z4["crosscheck_vs_y_metrics"]["release_t_mismatch"]), len(z4["crosscheck_vs_y_metrics"]["release_mode_mismatch"]))}
d["verdict_Z1_Z4"] = "사실만: release 규칙으로 성공이 바뀐 run 수 %d. 판정은 lead 와 사용자." % len(z4["flips"])
nm = ["Z1: crate 안 hold 시점 손 자세로 회전한 구 배치의 crate 벽 맞춤 (attach 시 손 자세 · world 축 기준으로만 계산)",
      "Z1: V1 / V3 의 held 구 (Z1 범위 V4 · V5, 참고로 V5S · V5B 포함)"]
d["not_measured"] = [x for x in d["not_measured"] if x not in nm] + nm
OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=float))
print(OUT, len(reps), mism)
