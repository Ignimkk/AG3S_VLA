"""T41c — per-run comparison T41c vs T41a vs T40 (G for B/C, D-chain for D) vs T40 J (verifier). Numbers only.
usage: compare_c.py OUT_JSON
Run facts (all four sets through the SAME scripts): analyze_t41.py (T39 rules: success / grasp / place; HOLD chunks + kinds; gate /
verdict reason counts) + hurdle_metrics (t41a client distance code = T40 H fix; per body group min distance to the hurdle, mm, true
MuJoCo collision geoms on recorded control-row qpos; B runs: hurdle placed virtually = the episode's r1 config, no physics).
  T41c: T41/c/analysis/analysis_c.json · T41a: T41/a/analysis/analysis_a.json · T40: T41/a/analysis/t40g_reanalysis.json
  (T40 G records) · T40J: T41/a/analysis/t40j_reanalysis.json (T40 J reruns of the false-zero stops).
HOLD / contact facts: hold_c.py physics replay (T41/c/analysis/hold_c.json for T41c, hold_ref.json for the others).
  contact phase: before_lift if lift_t is None or t < lift_t; carry if lift_t <= t <= end_carry (place_t, else apple_left_hand_t,
  else last row); after otherwise (lift_t / place_t / apple_left_hand_t from hurdle_metrics)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))
OUT = pathlib.Path(sys.argv[1])
SRC = {"T41c": V / "T41/c/analysis/analysis_c.json", "T41a": V / "T41/a/analysis/analysis_a.json",
       "T40": V / "T41/a/analysis/t40g_reanalysis.json", "T40J": V / "T41/a/analysis/t40j_reanalysis.json"}
A = {k: json.load(open(p))["runs"] for k, p in SRC.items() if p.exists()}
H = {}
for f in (V / "T41/c/analysis/hold_c.json", V / "T41/c/analysis/hold_ref.json"):
    if f.exists():
        H.update(json.load(open(f))["runs"])
HROOT = {"T41c": "T41/c/runs", "T41a": "T41/a/runs", "T40": "T40/G/runs", "T40J": "T40/J/runs"}
GRP = {"fingers": "Af", "EE_BODY_L": "Ae", "apple": "B", "other_robot": "C"}


def mm(x):
    return None if x is None else round(1e3 * float(x), 2)


def phase(t, h):
    lift, place, left, last = h.get("lift_t"), h.get("place_t"), h.get("apple_left_hand_t"), (h.get("series") or {}).get("t", [None])[-1]
    if lift is None or t < lift:
        return "before_lift"
    end = place if place is not None else (left if left is not None else last)
    return "carry" if t <= end else "after"


def facts(r, hk):
    h = r.get("hurdle") or {}; ph = h.get("phases") or {}; cs = h.get("client_obstacle_safety") or {}
    stop = bool(h.get("client_obstacle_stop_lines")); last_t = int(h["series"]["t"][-1]) if h.get("series") else None
    o = {"success": r["success"], "grasp": r["grasp"], "place": r["place"], "grasp_t": r["grasp_t"], "place_t": r["place_t"],
         "lift_t": h.get("lift_t"), "n_control": r["n_control"], "n_hold_chunks": r["n_hold_chunks"], "hold_t": r["hold_t"],
         "hold_kind_counts": r["hold_kind_counts"], "gate_action_counts": r["gate_action_counts"],
         "verdict_reason_kind_counts": r["verdict_reason_kind_counts"], "physical_hurdle": h.get("physical_hurdle"),
         "obstacle_stop": stop, "obstacle_stop_t": last_t if stop else None,
         "client_robot_collision": cs.get("robot_collision"), "client_payload_collision": cs.get("payload_collision"),
         "client_contact_steps": cs.get("contact_steps"), "client_min_robot_clearance_mm": mm(cs.get("min_robot_clearance_m")),
         "client_contact_pairs": cs.get("contact_pairs"), "min_mm": {}}
    for g, k in GRP.items():
        o["min_mm"][g] = {p: (None if (ph.get(p) or {}).get(k) is None else {"min_mm": mm(ph[p][k]["min_m"]), "t": ph[p][k]["t"],
                                                                               "n_rows_le_0": ph[p][k]["n_rows_le_0"], "pair": ph[p][k].get("pair")})
                          for p in ("all", "before_lift", "carry_lift_to_place_or_end")}
    x = H.get(hk)
    if x is None:
        o["replay"] = None
        return o
    o["replay"] = {"qpos_mismatch_rows": x["qpos_mismatch_rows"], "n_rows": x["n_rows"], "hold_mode_manifest": x["hold_mode_manifest"],
                   "n_hold_rows": x["n_hold_rows"], "n_hold_episodes": len(x["hold_episodes"])}
    o["holds"] = [{**{k: e[k] for k in ("t_first", "t_last", "duration_steps", "n_chunks", "chunk_seqs", "kinds", "mode", "source", "gripper_source",
                                         "entry_command_dev_rad_recorded", "q_hold_vs_entry_measured_deg", "joint_max_deg", "tcp_max_mm",
                                         "dev_from_q_hold_max_deg", "fruit", "contacts_in_hold")},
                   "phase": phase(e["t_first"], h)} for e in x["hold_episodes"]]
    o["contact_events"] = [{**e, "phase": phase(e["t_first"], h)} for e in x["contact_events"]]
    return o


def ident(ka, kc):
    """T41c vs T41a record identity: traj.npz arrays equal? first control step whose executed action differs; T41a HOLD rows before it."""
    ra, rc = A["T41a"].get(ka), A["T41c"].get(kc)
    if not ra or not rc:
        return None
    za, zc = np.load(pathlib.Path(ra["run_dir"]) / "traj.npz"), np.load(pathlib.Path(rc["run_dir"]) / "traj.npz")
    o = {k: bool(za[k].shape == zc[k].shape and np.array_equal(za[k], zc[k])) for k in ("executed_actions", "measured_qpos", "predicted_chunks")}
    ea, ec = za["executed_actions"], zc["executed_actions"]; n = min(len(ea), len(ec))
    dif = np.where(np.abs(ea[:n] - ec[:n]).max(axis=1) > 0)[0]
    o["n_steps"] = [int(len(ea)), int(len(ec))]
    o["first_executed_step_diff"] = int(dif[0]) if dif.size else (None if len(ea) == len(ec) else n)
    fa = [h["t_first"] for h in (H.get(f"T41/a/runs/{ka}") or {}).get("hold_episodes", [])]
    fc = [h["t_first"] for h in (H.get(f"T41/c/runs/{kc}") or {}).get("hold_episodes", [])]
    o["first_hold_t_T41a"] = min(fa) if fa else None; o["first_hold_t_T41c"] = min(fc) if fc else None
    return o


res = {"definitions": __doc__, "runs": {}}
keys = sorted(set().union(*[set(v) for v in A.values()]), key=lambda k: ("BCD".index(k.split("/")[1]), k))
for key in keys:
    st = key.split("/")[1]; ep = int(key.split("_ep")[1].split("_")[0])
    row = {"stage": st, "episode": ep}
    for s, runs in A.items():
        if key in runs:
            row[s] = facts(runs[key], f"{HROOT[s]}/{key}")
    row["T41c_vs_T41a_identity"] = ident(key, key)
    res["runs"][key] = row
    def brief(f):
        if not f:
            return "  n/a"
        hs = ";".join(f"t{e['t_first']}x{e['duration_steps']}{'/'.join(e['kinds'])} j{e['joint_max_deg']:.1f} tcp{e['tcp_max_mm']:.0f} slip{e['fruit']['slip_max_mm']:.0f}{'H' if e['fruit']['in_hand_at_entry'] else ''} c{e['contacts_in_hold']['robot_substeps']}"
                      for e in f.get("holds") or [])
        ce = ";".join(f"{e['t_first']}{e['phase'][:2]}{'R' if e['robot_substeps'] else ''}{'P' if e['payload_substeps'] else ''}{'h' if e['in_hold_first_row'] else ''}" for e in f.get("contact_events") or [])
        mn = f["min_mm"]
        return (f"{'S' if f['success'] else '-'}{'g' if f['grasp'] else '-'}{'p' if f['place'] else '-'} stop={f['obstacle_stop_t']} fing={(mn['fingers']['all'] or {}).get('min_mm')} "
                f"ee={(mn['EE_BODY_L']['all'] or {}).get('min_mm')} app={(mn['apple']['all'] or {}).get('min_mm')} oth={(mn['other_robot']['all'] or {}).get('min_mm')} holds[{hs}] contacts[{ce}]")
    print(key)
    for s in ("T40", "T40J", "T41a", "T41c"):
        if s in row:
            print(f"  {s:5s}", brief(row[s]))
    if row.get("T41c_vs_T41a_identity"):
        print("  ident", row["T41c_vs_T41a_identity"])
summ = {}
for st in "BCD":
    summ[st] = {}
    for s in ("T40", "T40J", "T41a", "T41c"):
        rr = [r[s] for k, r in res["runs"].items() if r["stage"] == st and s in r]
        summ[st][s] = {"n": len(rr), "success": sum(r["success"] for r in rr), "grasp": sum(r["grasp"] for r in rr), "place": sum(r["place"] for r in rr),
                       "robot_contact": sum(bool(r["client_robot_collision"]) for r in rr), "payload_contact": sum(bool(r["client_payload_collision"]) for r in rr),
                       "obstacle_stop": sum(r["obstacle_stop"] for r in rr), "hold_chunks": sum(r["n_hold_chunks"] for r in rr),
                       "hold_episodes": sum(len(r.get("holds") or []) for r in rr),
                       "contact_events_in_hold": sum(1 for r in rr for e in (r.get("contact_events") or []) if e["in_hold_first_row"]),
                       "holds_with_robot_contact": sum(1 for r in rr for e in (r.get("holds") or []) if e["contacts_in_hold"]["robot_substeps"] > 0)}
res["summary_counts"] = summ
print(json.dumps(summ, indent=1))
OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
