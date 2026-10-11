"""T41c — write the current per-run comparison into AG3S/ag3s/docs/handoff/T41c.verify.partial.json (key numbers_partial; numbers only,
same flock as update_partial.py). Source: T41/c/analysis/compare_c.json (compare_c.py)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import fcntl, json, pathlib, datetime
PARTIAL = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T41c.verify.partial.json'))
LOCK = pathlib.Path(_ag3s_workspace_path('outputs/verify/T41/c/p/.partial.lock'))
CMP = json.load(open(_ag3s_workspace_path('outputs/verify/T41/c/analysis/compare_c.json')))
LOCK.touch(exist_ok=True)
with open(LOCK, "a") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    cur = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {"_comment": "ag3s-verifier partial record for T41c (numbers and paths only).", "step": "T41c"}
    per = {}
    for k, r in CMP["runs"].items():
        f = r.get("T41c")
        if not f:
            continue
        per[k] = {"success": f["success"], "grasp": f["grasp"], "place": f["place"], "obstacle_stop_t": f["obstacle_stop_t"],
                  "client_robot_collision": f["client_robot_collision"], "client_payload_collision": f["client_payload_collision"],
                  "min_mm_all": {g: (v["all"] or {}).get("min_mm") for g, v in f["min_mm"].items()},
                  "verdict_reason_kind_counts": f["verdict_reason_kind_counts"], "replay": f.get("replay"),
                  "holds": [{x: h[x] for x in ("t_first", "duration_steps", "kinds", "mode", "phase", "joint_max_deg", "tcp_max_mm", "dev_from_q_hold_max_deg")}
                            | {"slip_max_mm": h["fruit"]["slip_max_mm"], "in_hand_at_entry": h["fruit"]["in_hand_at_entry"],
                               "robot_contact_substeps": h["contacts_in_hold"]["robot_substeps"], "payload_contact_substeps": h["contacts_in_hold"]["payload_substeps"]}
                            for h in f.get("holds") or []],
                  "contact_events": f.get("contact_events")}
    cur["numbers_partial"] = {"source": "outputs/verify/T41/c/analysis/compare_c.json", "T41c_runs": per, "summary_counts": CMP["summary_counts"]}
    cur["numbers_partial_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    tmp = PARTIAL.with_suffix(".json.tmp"); tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False, default=float)); tmp.replace(PARTIAL)
    fcntl.flock(lk, fcntl.LOCK_UN)
print(cur["numbers_partial_updated"], len(per))
