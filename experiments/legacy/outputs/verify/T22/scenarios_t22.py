"""T22 verifier — #4 empty-hand close, #5 HOLD/release, last-gate on legacy attaches, reach sweep, per-step reach."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np
import mujoco
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T22'))
import replay_t22 as R
from AG3S.trajopt import wire
from AG3S.trajopt.grasp_latch import GraspLatch, LatchConfig, grasp_signal_from_feedback
from AG3S.ag3s.constraints.attached import parent_reach, AttachRejected

OUT = _ag3s_workspace_path('outputs/verify/T22/scenarios_t22.json')
res = {}


def fb(seq, applied, measured, executed=None):
    applied = np.asarray(applied, np.float64).reshape(-1); n = len(applied)
    flags = [True] * n if executed is None else list(executed)
    held = not all(flags)
    blob = wire.make_exec_feedback(seq=seq, t_step_start=(seq - 1) * 8, executed=flags,
        executed_chunk="none" if not any(flags) else "reference", ipc="unsafe" if held else "ok",
        hold_kind="unsafe" if held else None, hold_reason="synthetic HOLD" if held else None,
        planned_gripper=np.column_stack([applied, np.ones(n)]), applied_gripper=np.column_stack([applied, np.ones(n)]),
        applied_arm=np.zeros((n, 14)), measured_gripper=(float(measured), 1.0))
    return wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: blob})


def run_seq(name, chunks, reach=0.042, placed=None):
    """chunks: list of (applied[8], measured_at_next_request, executed or None)."""
    latch = GraspLatch(LatchConfig(), evidence=True)
    for _ in range(3):
        latch.update(label="apple", score=1.0, runner_up=0.0)
    rows = []
    # first request: measured open, no closing (establishes opening_previous)
    seq = 1
    for i, (app, meas, exe) in enumerate(chunks):
        seq += 1
        pl = bool(placed[i]) if placed is not None else False
        sig = grasp_signal_from_feedback(fb(seq - 1, app, meas, exe), 0, 0.85)
        before = latch.phase.value
        ev = latch.observe_grasp(sig, reach_m=reach, manipulated_id=0, manipulated_state="visible", placed=pl)
        rows.append({"req": i, "applied": [round(float(a), 4) for a in app], "executed": exe, "measured": round(float(meas), 4),
                     "placed_in": pl, "state_before": before, "state": latch.phase.value, "attach": ev.attach, "detach": ev.detach,
                     "checks": (ev.evidence or {}).get("checks"), "note": ev.note})
    return {"name": name, "reach_m_given": reach, "n_attach": sum(r["attach"] for r in rows),
            "attach_req": [r["req"] for r in rows if r["attach"]], "detach_req": [r["req"] for r in rows if r["detach"]], "rows": rows}


# ---------- #4a pure synthetic empty-hand close (to full closure) ----------
OPEN8 = [1.0] * 8
s4 = []
s4.append(run_seq("4a synthetic: cmd 0.0 x8, opening -> 0.004 (full close), stays 0.0",
                  [(OPEN8, 1.0, None), ([1, 1, 1, 0, 0, 0, 0, 0], 0.004, None), ([0.0] * 8, 0.003, None), ([0.0] * 8, 0.003, None)]))
s4.append(run_seq("4b synthetic: T17 command sequence, empty hand (opening follows command)",
                  [(OPEN8, 1.0, None), ([1.0, 0.99, 0.99, 0.0, 0.0, 0.0, 0.01, 0.01], 0.01, None),
                   ([0.0, 0.0, 0.0, 0.0, 0.02, 0.56, 0.58, 0.58], 0.575, None), ([0.57] * 8, 0.571, None), ([0.58] * 8, 0.58, None)]))
s4.append(run_seq("4c synthetic: cmd 0.0, opening stops at 0.03 (below empty_below)",
                  [(OPEN8, 1.0, None), ([0.0] * 8, 0.03, None), ([0.0] * 8, 0.03, None)]))

# ---------- #4d MuJoCo-simulated empty-hand close (hand far from any object) ----------
m, d = R._m, mujoco.MjData(R._m)
P, C, S = R.load_run("t17")
q0 = np.asarray(P[2]["qpos"], np.float64)
d.qpos[:] = q0; mujoco.mj_forward(m, d)
act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
for i in range(m.nu):            # hold every actuator at its joint's current position
    j = m.actuator_trnid[i][0]; d.ctrl[i] = d.qpos[m.jnt_qposadr[j]]
d.ctrl[act["gripper_l_act"]] = -0.045
apple_b = np.asarray(P[2]["object_poses"]["apple"]["pos"])
reach_far = parent_reach(R.RM, q0[R.Q20], "ee_finger_l1", apple_b)
cmds = [[1.0] * 8, [1, 1, 1, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.02, 0.56, 0.58, 0.58], [0.57] * 8, [0.57] * 8]
trace, chunks = [], []
for c in cmds:
    for u in c:
        d.ctrl[act["gripper_l_act"]] = u * -0.045
        for _ in range(33):
            mujoco.mj_step(m, d)
        trace.append({"cmd": u, "opening": abs(d.qpos[R.GL]) / 0.045})
    chunks.append((c, abs(d.qpos[R.GL]) / 0.045, None))
sim = run_seq("4d MuJoCo-sim empty close (T17 seq2 pose, arm held, T17 command pattern); reach forced 0.042 m", chunks)
sim["sim"] = {"start_pose": "T17 planning seq 2 qpos", "sim_steps_per_action": 33, "timestep": float(m.opt.timestep),
              "urdf_l1_to_apple_mm_at_pose": round(reach_far * 1000, 2), "opening_trace": [round(t["opening"], 4) for t in trace],
              "cmd_trace": [t["cmd"] for t in trace]}
s4.append(sim)
res["4_empty_hand"] = s4

# ---------- #5 HOLD after grasp + release ----------
T17 = [(OPEN8, 1.009, None), ([1.0, 0.99, 0.99, 0.0, -0.003, 0.005, 0.006, 0.005], 0.715, None),
       ([0.003, -0.005, -0.005, 0.001, 0.023, 0.564, 0.575, 0.576], 0.720, None)]
HOLD = ([0.72] * 8, 0.72, [False] * 8)
OPENC = ([1.0] * 8, 1.0, None)
s5 = []
s5.append(run_seq("5a grasp then 4 HOLD chunks (opening kept 0.72) then 2 open chunks", T17 + [HOLD] * 4 + [OPENC, OPENC]))
s5.append(run_seq("5b grasp, HOLD, one open glitch then closed again, then open x2", T17 + [HOLD, OPENC, ([0.57] * 8, 0.72, None), OPENC, OPENC]))
s5.append(run_seq("5c placed_fn=True while closed (HOLD), then placed & open", T17 + [HOLD, HOLD, OPENC],
                  placed=[False, False, False, True, True, True]))
s5.append(run_seq("5d HOLD chunk whose re-command (0.72) reads below 0.85 — partial HOLD 3/8 exec", T17 + [([0.57, 0.57, 0.57, 0.72, 0.72, 0.72, 0.72, 0.72], 0.72, [True] * 3 + [False] * 5)] + [OPENC, OPENC]))
res["5_hold_release"] = s5

# ---------- last gate on the legacy attaches (real server geometry) ----------
gate = []
for run, seq_attach in (("t17", 16), ("t16", 23), ("t14", 25)):
    P, C, S = R.load_run(run)
    for k in (seq_attach, seq_attach + 1):
        srv = S.get(k)
        if not srv or srv["centroid"] is None:
            continue
        q20 = np.asarray(P[k]["qpos"], np.float64)[R.Q20]
        stub, fn = R.make_ag3s()
        g = R.target_geom(srv["centroid"], srv["points"])
        rec = {"run": run, "seq": k, "t_step": int(P[k]["t_step"]), "n_target_points": 0 if srv["points"] is None else int(len(srv["points"])),
               "reach_mm": round(parent_reach(R.RM, q20, "ee_finger_l1", srv["centroid"]) * 1000, 2)}
        try:
            fn(stub, g, robot_state=q20, parent_link="ee_finger_l1", label="manipulated", reach=0.12)
            rec["gate"] = "accepted"
        except AttachRejected as e:
            rec["gate"] = "rejected"; rec["rejection"] = e.record()
        # also: attach with reach=None (old default) passes
        try:
            fn(stub, g, robot_state=q20, parent_link="ee_finger_l1", label="manipulated", reach=None); rec["gate_reach_None"] = "accepted"
        except Exception as e:
            rec["gate_reach_None"] = f"{type(e).__name__}: {e}"
        gate.append(rec)
res["last_gate_on_server_targets"] = gate

# ---------- reach sweep ----------
sweep = {}
limits = [0.03, 0.04, 0.042, 0.045, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.60, 0.62, 0.65, 0.70, 1.0]
for run in ("t17", "t16", "t14"):
    for src in ("apple", "server_prev"):
        sweep[f"{run}/{src}"] = {str(L): (lambda rows: {"n_attach": sum(r["attach"] for r in rows),
                                                       "attach_t": [r["t_step"] for r in rows if r["attach"]]})(R.replay(run, src, L)) for L in limits}
res["reach_sweep"] = sweep

# ---------- per control step: ee_finger_l1 (URDF origin) <-> apple ----------
per = {}
for run in ("t17", "t16", "t14"):
    P, C, S = R.load_run(run)
    out = []
    for cs in sorted(C):
        for r in C[cs]:
            q = np.asarray(r["qpos"], np.float64)
            mj, Tb = R.mj_state(q)
            ap = mj["apple"]
            out.append({"t": int(r["t_step"]), "executed": bool(r["executed"]),
                        "cmd": float(np.asarray(P[cs]["actions_reference"])[r["step_in_chunk"], R.COL_L]),
                        "opening": abs(q[R.GL]) / 0.045,
                        "urdf_l1_apple_mm": parent_reach(R.RM, q[R.Q20], "ee_finger_l1", ap) * 1000,
                        "mj_l1_body_apple_mm": float(np.linalg.norm(mj["ee_finger_l1"] - ap)) * 1000,
                        "apple_z_mm": float(ap[2]) * 1000})
    per[run] = out
res["per_step"] = per
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False, default=float)

for s in s4 + s5:
    print(s["name"], "| attach", s["attach_req"], "detach", s["detach_req"])
    for r in s["rows"]:
        print("    ", r["req"], r["state_before"], "->", r["state"], r["measured"], {k: v for k, v in (r["checks"] or {}).items() if not v}, r["note"][:90])
print("sim trace", sim["sim"]["opening_trace"][:24], "far reach", sim["sim"]["urdf_l1_to_apple_mm_at_pose"])
for g in gate: print(g)
for k, v in sweep.items(): print(k, {L: x["n_attach"] for L, x in v.items()})
