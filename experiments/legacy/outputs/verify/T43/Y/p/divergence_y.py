"""T43 Y — V4 vs V5 first divergence per pair, and the server state of both arms at that chunk (verifier, CPU only).

usage: divergence_y.py OUT_JSON
first divergence = first control step with max_j |executed_actions V4 - V5| > 1e-3 rad (analyze_t39 DIV_TOL); chunk = planning row with that t.
At that chunk, both arms: max |actions_reference V4 - V5| (policy chunk, rows 0-7) and max |actions V4 - V5| (refined), grasp_state, attach,
decision_reason, latched_hold, n_not_supported clusters, closing carve ball radius, worst TO row (binding category / link / clearance),
held query radii (raw/server_chunks.json), tcp_deflection_mm."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]
S = json.load(open(_ag3s_workspace_path('outputs/verify/T43/Y/raw/server_chunks.json')))["runs"]
out = {"definitions": __doc__, "pairs": {}}
for k4 in sorted(A):
    if not k4.startswith("V4/"):
        continue
    k5 = k4.replace("V4", "V5")
    if k5 not in A:
        continue
    e4 = np.load(pathlib.Path(A[k4]["run_dir"]) / "traj.npz")["executed_actions"]; e5 = np.load(pathlib.Path(A[k5]["run_dir"]) / "traj.npz")["executed_actions"]
    n = min(len(e4), len(e5)); dif = np.abs(e4[:n] - e5[:n]).max(axis=1); w = np.where(dif > 1e-3)[0]
    rec = {"first_step": None if not len(w) else int(w[0])}
    if len(w):
        t = int(8 * (w[0] // 8)); rec["chunk_t"] = t
        P = {}
        for k in (k4, k5):
            plan = {p["t_step"]: p for p in (json.loads(l) for l in open(pathlib.Path(A[k]["run_dir"]) / "frames.jsonl")) if p["kind"] == "planning"}
            P[k] = plan[t]
        r4, r5 = (np.asarray(P[k].get("actions_reference") or P[k]["actions"], float)[:8] for k in (k4, k5))
        a4, a5 = (np.asarray(P[k]["actions"], float)[:8] for k in (k4, k5))
        rec["reference_rows0_7_max_abs_diff_rad"] = float(np.abs(r4 - r5).max()); rec["refined_rows0_7_max_abs_diff_rad"] = float(np.abs(a4 - a5).max())
        for lab, k in (("V4", k4), ("V5", k5)):
            sc = next((c for c in A[k].get("server_chunks", []) if c["t"] == t), {}); sy = next((c for c in S[k]["chunks"] if c["t"] == t), {})
            rec[lab] = {"grasp_state": sc.get("grasp_state"), "attach": sc.get("attach_now"), "decision_reason": sc.get("decision_reason"), "latched_hold": sc.get("latched_hold"),
                        "n_not_supported": sc.get("n_not_supported"), "carve_ball_radius_mm": sc.get("carve_ball_radius_mm"), "binding": sc.get("binding"),
                        "to_status": sc.get("to_status"), "tcp_deflection_mm": sc.get("tcp_deflection_mm"), "held_query_radii_mm": (sy.get("held_query") or {}).get("radii_mm"),
                        "closing_tip": sc.get("closing_tip")}
    out["pairs"][k4.split("/", 1)[1].replace("V4_", "")] = rec
    print(k4, json.dumps(rec, default=float)[:600], flush=True)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
