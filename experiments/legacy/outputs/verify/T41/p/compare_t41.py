"""T41a — per-run comparison T41a vs T40 G/D (and T40 J when T40FG.verify.json has a 'J' key). Numbers only.
usage: compare_t41.py T41A_ANALYSIS T40_REANALYSIS T41A_REPLAY OUT_JSON [T40_REPLAY [T40J_REANALYSIS]]
T40J = T40 J reruns (snapshot work-t40h: T40 code + client false-zero fix, margin off) of the T40 runs that ended in a false-zero stop.

T41a / T40 run facts: analyze_t41.py (T39 rules: success / grasp / place, HOLD chunks + kinds, gate / verdict counts) and
hurdle_metrics (snapshot t41a client distance code = T40 H fix) — for T40 G/D the SAME scripts re-run on the T40 records
(T40 originally measured with the pre-fix distance code; its original outcome_table is copied alongside as 'T40_original').
min distance to the hurdle per body group, mm (true MuJoCo collision geoms, recorded control-row qpos):
  fingers = ee_finger_l1/l2 (Af), EE_BODY_L (Ae), apple (B), other robot = every other collision geom under 'base' (C).
  B runs: hurdle placed virtually (the episode's r1 config), no physics.
contact: client traj.npz obstacle_safety (MuJoCo contacts every sim step; robot_collision / payload_collision / contact_pairs /
  min_robot_clearance_m) + '[SAFETY] obstacle stop' (with --obstacle-stop-distance 0 the client stops at the first contact).
  contact_t = last control row of an obstacle-stopped run; relative to lift-off: before_lift if lift_t is None or contact_t < lift_t.
  first_t_dist_le_0: first control row whose true geom distance <= 0 per group (recorded qpos, no contact physics) and its side of
  lift-off; this is the only time stamp for a payload (apple) contact that did not stop the client.
TO rows: replay_t41.py (classes per phase; closest-approach TO view vs true mesh). T40_TO_rows_classified_by_T41a_rule = the same replay
  on the T40 records (TO ran with margin off there; classes and clear_off counts are what the T41a rule would have seen on those plans)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

A1 = json.load(open(sys.argv[1]))["runs"]; A0 = json.load(open(sys.argv[2]))["runs"]
RP = json.load(open(sys.argv[3]))["runs"] if pathlib.Path(sys.argv[3]).exists() else {}
OUT = pathlib.Path(sys.argv[4])
RP0 = json.load(open(sys.argv[5]))["runs"] if len(sys.argv) > 5 and pathlib.Path(sys.argv[5]).exists() else {}
AJ = json.load(open(sys.argv[6]))["runs"] if len(sys.argv) > 6 and pathlib.Path(sys.argv[6]).exists() else {}
FG = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40FG.verify.json')))["numbers"]
J = FG.get("J")
GRP = {"fingers": "Af", "EE_BODY_L": "Ae", "apple": "B", "other_robot": "C"}


def mm(x):
    return None if x is None else round(1e3 * float(x), 2)


def facts(r):
    h = r.get("hurdle") or {}
    ph = h.get("phases") or {}
    cs = h.get("client_obstacle_safety") or {}
    stop = bool(h.get("client_obstacle_stop_lines"))
    last_t = int(h["series"]["t"][-1]) if h.get("series") else None
    lift = h.get("lift_t")
    o = {"success": r["success"], "grasp": r["grasp"], "place": r["place"], "grasp_t": r["grasp_t"], "place_t": r["place_t"],
         "lift_t": lift, "n_control": r["n_control"], "n_hold_chunks": r["n_hold_chunks"], "hold_t": r["hold_t"],
         "hold_kind_counts": r["hold_kind_counts"], "gate_action_counts": r["gate_action_counts"],
         "verdict_reason_kind_counts": r["verdict_reason_kind_counts"], "physical_hurdle": h.get("physical_hurdle"),
         "obstacle_stop": stop, "obstacle_stop_t": last_t if stop else None,
         "client_robot_collision": cs.get("robot_collision"), "client_payload_collision": cs.get("payload_collision"),
         "client_contact_steps": cs.get("contact_steps"), "client_min_robot_clearance_mm": mm(cs.get("min_robot_clearance_m")),
         "client_contact_pairs": cs.get("contact_pairs")}
    if stop and (cs.get("robot_collision") or cs.get("payload_collision")):
        o["contact_t"] = last_t; o["contact_vs_lift"] = "before_lift" if (lift is None or last_t < lift) else "after_lift"
    else:
        o["contact_t"] = None; o["contact_vs_lift"] = None
    # kinematic contact (true geom distance <= 0 on recorded control-row qpos), first row per group, vs lift-off
    o["first_t_dist_le_0"] = {}
    ser = h.get("series") or {}
    for g, k in GRP.items():
        if ser.get(k):
            tt = np.asarray(ser["t"]); v = np.asarray(ser[k]); idx = np.where(v <= 0)[0]
            ft = int(tt[idx[0]]) if idx.size else None
            o["first_t_dist_le_0"][g] = None if ft is None else {"t": ft, "vs_lift": "before_lift" if (lift is None or ft < lift) else "after_lift",
                                                                 "n_rows": int(idx.size)}
    o["min_mm"] = {}
    for g, k in GRP.items():
        o["min_mm"][g] = {p: (None if (ph.get(p) or {}).get(k) is None else {"min_mm": mm(ph[p][k]["min_m"]), "t": ph[p][k]["t"],
                                                                               "n_rows_le_0": ph[p][k]["n_rows_le_0"], "pair": ph[p][k].get("pair")})
                          for p in ("all", "before_lift", "carry_lift_to_place_or_end")}
    return o


def rp_facts(key, src=None):
    x = (RP if src is None else src).get(key)
    if not x:
        return None
    ca = next((c for c in x["chunks"] if "closest_approach" in c), None)
    o = {"by_phase": x["by_phase"], "fidelity_summary": x["fidelity_summary"], "t_star_min_gripper_hurdle": x["t_star_min_gripper_hurdle"]}
    if ca:
        o["closest_approach_chunk"] = {"seq": ca["seq"], "t": ca["t"], "phase": ca["phase"], "gate": ca["gate"], "hold_kinds": ca["hold_kinds"],
                                       "fidelity": ca["fidelity"],
                                       **{ln: {k: v for k, v in ca["closest_approach"][ln].items() if "per_step" not in k}
                                          for ln in ("ee_left", "ee_finger_l1", "ee_finger_l2")}}
    # chunks with obstacle rows < 10 mm (margin binding) and the hand among them
    o["chunks_with_obstacle_lt10mm"] = [{"seq": c["seq"], "t": c["t"], "phase": c["phase"], "n": c["counts"]["obstacle_lt10mm"],
                                         "n_hand": c["counts"]["obstacle_lt10mm_hand"], "gate": c["gate"], "hold_kinds": c["hold_kinds"]}
                                        for c in x["chunks"] if c["counts"]["obstacle_lt10mm"] > 0]
    return o


res = {"definitions": __doc__, "runs": {}}
for key in sorted(A1, key=lambda k: ("BCD".index(A1[k]["stage"]), A1[k]["episode"])):
    r1 = A1[key]
    st, ep, sd = r1["stage"], r1["episode"], r1["key"]
    k0 = f"E3b/{st}/E3b_ep{ep}_s{sd}"
    row = {"stage": st, "episode": ep, "seed": sd, "T41a": facts(r1), "T41a_TO_rows": rp_facts(key)}
    if k0 in A0:
        row["T40"] = facts(A0[k0]); row["T40"]["source"] = f"T40 G record re-analysed: {A0[k0]['run_dir']}"
    row["T40_original"] = (FG.get("outcome_table", {}).get(st) or {}).get(f"E3b_ep{ep}")
    if RP0.get(k0):
        row["T40_TO_rows_classified_by_T41a_rule"] = rp_facts(k0, RP0)  # T40 records (margin off) replayed with the T41a classification
    if r1.get("bit_identity_vs_T40G") is not None:
        b = r1["bit_identity_vs_T40G"]
        row["B_vs_T40G_bit_identity"] = {"all": b["all_bit_identical"], "first_executed_step_diff": b.get("first_executed_step_diff"),
                                         "traj_npz": {k: v["equal"] for k, v in b["traj_npz"].items()},
                                         "server_record": {k: v for k, v in (b.get("server_record") or {}).items()}}
    if k0 in AJ:
        row["T40J"] = facts(AJ[k0]); row["T40J"]["source"] = f"T40 J record re-analysed: {AJ[k0]['run_dir']}"
        row["T40J_summary_as_recorded"] = ((J or {}).get("outcome_summary") or {}).get(f"E3b {st}{ep}")
    res["runs"][key] = row
    t1, t0 = row["T41a"], row.get("T40") or {}
    print(st, ep, "T41a", "S" if t1["success"] else "-", "hold", t1["n_hold_chunks"], "stop", t1["obstacle_stop_t"], "coll", t1["client_robot_collision"],
          "fing/ee", (t1["min_mm"]["fingers"]["all"] or {}).get("min_mm"), (t1["min_mm"]["EE_BODY_L"]["all"] or {}).get("min_mm"),
          "| T40", "S" if t0.get("success") else "-", "hold", t0.get("n_hold_chunks"), "stop", t0.get("obstacle_stop_t"), "coll", t0.get("client_robot_collision"),
          "fing/ee", ((t0.get("min_mm") or {}).get("fingers", {}).get("all") or {}).get("min_mm"), ((t0.get("min_mm") or {}).get("EE_BODY_L", {}).get("all") or {}).get("min_mm"))
OUT.write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
