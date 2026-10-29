"""T40 G/D: compact per-run numbers from analysis JSON(s) -> partial/final verify JSON section 'G'. usage: summarize.py ANALYSIS_JSON..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, datetime
import numpy as np

PARTIAL = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40FG.verify.partial.json'))


def mm(x):
    return None if x is None else round(1e3 * x, 2)


def compact(k, r):
    h = r.get("hurdle") or {}
    o = {"stage": r["stage"], "cond": r["cond"], "episode": r["episode"], "seed": r["key"], "run_dir": r["run_dir"],
         "success": r["success"], "grasp": r["grasp"], "place": r["place"], "grasp_t": r["grasp_t"], "place_t": r["place_t"],
         "apple_dz_max_mm": r["apple_dz_max_mm"], "n_control": r["n_control"], "n_planning": r["n_planning"], "wall_s": r["wall_s"],
         "n_hold_chunks": r["n_hold_chunks"], "hold_t": r["hold_t"], "hold_kind_counts": r["hold_kind_counts"],
         "gate_action_counts": r["gate_action_counts"], "verdict_reason_kind_counts": r["verdict_reason_kind_counts"], "ipc_counts": r["ipc_counts"],
         "hold_chunks": [{kk: c.get(kk) for kk in ("t", "hold_kinds", "worst", "verdict_reason_kinds", "max_violation_m")} for c in r["chunks"] if c["hold"]],
         "time_ms_excl_first": r["time_ms_excl_first"], "first_chunk_round_trip_ms": r["first_chunk_round_trip_ms"],
         "n_round_trip_gt_chunk_period": r["n_round_trip_gt_chunk_period"],
         "server_segment": r.get("server_segment"), "contact_settle": r.get("contact")}
    if r.get("bit_identity_vs_T39") is not None:
        b = r["bit_identity_vs_T39"]
        o["bit_identical_to_T39"] = b["all_bit_identical"]
        o["bit_identity_detail"] = {"traj_npz": {kk: v["equal"] for kk, v in b["traj_npz"].items()},
                                    "frames": {kk: v["equal"] for kk, v in b["frames"].items()},
                                    "server_record": {kk: v for kk, v in (b.get("server_record") or {}).items()}}
    if h:
        o["hurdle"] = {"physical": h["physical_hurdle"], "config": h["config"], "profile": h["profile"], "placement": h["hurdle"],
                       "lift_t": h["lift_t"], "apple_left_hand_t": h["apple_left_hand_t"],
                       "min_mm_by_phase": {ph: {g: (None if v is None else {"min_mm": mm(v["min_m"]), "t": v["t"], "n_rows_le_0": v["n_rows_le_0"], "pair": v.get("pair")})
                                               for g, v in d.items()} for ph, d in h["phases"].items()},
                       "client_obstacle_safety": h.get("client_obstacle_safety"), "client_obstacle_stop": h.get("client_obstacle_stop_lines"),
                       "apple_hurdle_min_grasp_to_left_hand": h.get("apple_hurdle_min_grasp_to_left_hand"),
                       "ag3s": {kk: h.get(kk) for kk in ("ag3s_n_chunks", "ag3s_n_target_nearest_hurdle", "ag3s_n_manipulated_nearest_hurdle",
                                                          "ag3s_n_chunks_with_target_points_on_hurdle", "ag3s_min_target_to_hurdle_m", "ag3s_min_manipulated_to_hurdle_m")} if r["cond"] == "E3b" else None}
    return o


def main():
    runs = {}
    for f in sys.argv[1:]:
        a = json.loads(pathlib.Path(f).read_text())
        for k, r in a["runs"].items():
            runs[k] = compact(k, r)
    cur = json.loads(PARTIAL.read_text())
    cur.setdefault("G", {})["per_run"] = runs
    cur["G"]["analysis_files"] = sys.argv[1:]
    cur["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    PARTIAL.write_text(json.dumps(cur, indent=1, ensure_ascii=False, default=float))
    print(len(runs), "runs summarized")


if __name__ == "__main__":
    main()
