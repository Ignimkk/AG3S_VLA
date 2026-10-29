"""T43GD per-pair comparison V7 (gate OFF) vs V7g (gate ON). usage: gd_pairs.py AGG_JSON OUT_JSON    (CPU, numbers only)
Per pair (stage, episode, seed):
  first_exec_diff_step   : first control step where traj.npz executed_actions (n, 16) differ (> 0) between the arms (None = identical over the common length); chunk = step // 8 (server t_step = 8 * chunk).
  first_chunk_diff_t     : first chunk t_step where any of (HOLD flag, confirmed label, latch, grounding_status, admissibility mode, exclusion.active) differs between the arms.
  max_arm_dJ_between_rad : max over the common control steps of || q_V7[s, arm] - q_V7g[s, arm] ||_2 (traj.npz measured_qpos, active arm columns); mean_ likewise.
  n_steps_exec_differ    : control steps (common length) with max |executed_actions V7 - V7g| > 1e-3.
  d_*                    : V7g minus V7 of the per-run numbers in agg.json.
rank = max_arm_dJ_between_rad (descending); the three largest are the video pairs."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); ROOTS = {"V7": T / "GD/runs", "V7g": T / "V7g/runs"}
agg = json.load(open(sys.argv[1])); R = agg["runs"]
out = {"definitions": __doc__, "pairs": {}}
keys = sorted({(o["stage"], o["episode"], o["seed"]) for o in R.values() if o["arm"] == "V7"} & {(o["stage"], o["episode"], o["seed"]) for o in R.values() if o["arm"] == "V7g"})
for st, ep, sd in keys:
    a = R[f"V7/{st}/V7_ep{ep}_s{sd}"]; b = R[f"V7g/{st}/V7g_ep{ep}_s{sd}"]
    za = np.load(ROOTS["V7"] / f"V7/{st}/V7_ep{ep}_s{sd}/traj.npz"); zb = np.load(ROOTS["V7g"] / f"V7g/{st}/V7g_ep{ep}_s{sd}/traj.npz")
    n = min(len(za["executed_actions"]), len(zb["executed_actions"]))
    ea, eb = za["executed_actions"][:n].astype(float), zb["executed_actions"][:n].astype(float)
    dif = np.abs(ea - eb).max(axis=1)
    cols = list(range(0, 7)) if a["pose"]["side"] == "left" else list(range(8, 15))
    qa, qb = za["measured_qpos"][:n].astype(float), zb["measured_qpos"][:n].astype(float)
    dq = np.linalg.norm(qa[:, cols] - qb[:, cols], axis=1)
    ta = dict(zip(a["t_seq"], zip(a["verdict_seq"], a["label_seq"], a["latch_seq"], a["grounding_seq"], a["mode_seq"], a["excl_seq"])))
    tb = dict(zip(b["t_seq"], zip(b["verdict_seq"], b["label_seq"], b["latch_seq"], b["grounding_seq"], b["mode_seq"], b["excl_seq"])))
    cd = next((t for t in sorted(set(ta) & set(tb)) if ta[t] != tb[t]), None)
    first_div = int(np.argmax(dif > 0)) if (dif > 0).any() else None
    o = {"stage": st, "episode": ep, "seed": sd, "n_steps_common": int(n), "n_steps_V7": len(za["executed_actions"]), "n_steps_V7g": len(zb["executed_actions"]),
         "first_exec_diff_step": first_div, "first_chunk_diff_t": cd, "n_steps_exec_differ": int((dif > 1e-3).sum()),
         "max_arm_dJ_between_rad": float(dq.max()), "mean_arm_dJ_between_rad": float(dq.mean()), "argmax_step": int(dq.argmax()),
         "mode": [a["mode"], b["mode"]], "success": [a["success"], b["success"]], "success_rel": [a["success_rel"], b["success_rel"]],
         "t_placed": [a["t_placed"], b["t_placed"]],
         "d_n_hold": b["n_hold"] - a["n_hold"], "d_n_hold_after_placed": b["after_placed"]["n_hold"] - a["after_placed"]["n_hold"],
         "d_n_switch_after_placed": b["after_placed"]["n_switch"] - a["after_placed"]["n_switch"], "d_n_carve_after_placed": b["after_placed"]["n_carve"] - a["after_placed"]["n_carve"],
         "n_subtask_no_target": [a["n_subtask_no_target"], b["n_subtask_no_target"]], "n_hold": [a["n_hold"], b["n_hold"]],
         "n_switch_after_placed": [a["after_placed"]["n_switch"], b["after_placed"]["n_switch"]], "n_carve_after_placed": [a["after_placed"]["n_carve"], b["after_placed"]["n_carve"]],
         "grounding_status_after_placed": [a["grounding_status_after_placed"], b["grounding_status_after_placed"]],
         "d_dJ_rad_last": b["pose"]["dJ_rad_last"] - a["pose"]["dJ_rad_last"], "d_dEE_mm_last": b["pose"]["dEE_mm_last"] - a["pose"]["dEE_mm_last"],
         "dJ_rad_last": [a["pose"]["dJ_rad_last"], b["pose"]["dJ_rad_last"]], "dEE_mm_last": [a["pose"]["dEE_mm_last"], b["pose"]["dEE_mm_last"]],
         "hurdle_contact_steps": [a["hurdle_contact_steps"], b["hurdle_contact_steps"]]}
    out["pairs"][f"{st}:{ep}_s{sd}"] = o
rank = sorted(out["pairs"], key=lambda k: -out["pairs"][k]["max_arm_dJ_between_rad"])
out["rank_by_max_arm_dJ_between_rad"] = rank
out["n_pairs_identical_exec"] = sum(1 for o in out["pairs"].values() if o["first_exec_diff_step"] is None and o["n_steps_V7"] == o["n_steps_V7g"])
out["n_pairs_exec_differ"] = sum(1 for o in out["pairs"].values() if o["first_exec_diff_step"] is not None)
out["n_pairs_chunk_state_differ"] = sum(1 for o in out["pairs"].values() if o["first_chunk_diff_t"] is not None)
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))
for k in rank:
    o = out["pairs"][k]
    print(k, "maxdJ", round(o["max_arm_dJ_between_rad"], 3), "firstdiff", o["first_exec_diff_step"], "chunkdiff", o["first_chunk_diff_t"], "mode", o["mode"], "S", o["success"], "hold", o["n_hold"], "sw", o["n_switch_after_placed"], "carve", o["n_carve_after_placed"], "snt", o["n_subtask_no_target"], "dEE", [round(x) for x in o["dEE_mm_last"]])
