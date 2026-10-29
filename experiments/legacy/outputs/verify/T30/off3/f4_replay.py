"""T30 #3 — F4 (settled = measured opening standing still, settle_epsilon) replayed with the working-tree GraspLatch.

(a) T14 / T16 / T17 (ep1807, live_test 20260928): T22 verifier replay reused as is — outputs/verify/T22/replay_t22.py
    `replay(run, source)` (feedback reconstructed from local control rows; GraspLatch(LatchConfig(), evidence=True);
    last gate = real AG3S.attach on a stub). source 'apple' (GT apple centroid) and 'server_prev' (previous chunk's server
    target centroid), as T22.
(b) Empty-hand MuJoCo sweep: T22 #4e (outputs/verify/T22/empty_sweep_t22.py) logic copied here (that script imports
    scenarios_t22, which rewrites a T22 file on import) — same q0 (T17 planning seq 2), 33 mj_step per control step,
    close target {0.0, 0.3, 0.57} x start step 0..7, reach forced 0.042 m, run_seq identical to scenarios_t22.run_seq.
(c) T28 E3a/E3b (12 runs): per run, server chunks in order (T28 p3/<COND>_analysis.json pairing). Signal =
    grasp_signal_from_feedback(recorded summary.exec_feedback, hand 0, cfg.gripper_closed_below) — the feedback the server
    actually received. reach = parent_reach(load_rby1(), q20 of the paired planning row, 'ee_finger_l1', centroid) with
    centroid = previous chunk's recorded manipulated centroid (base) ['server_prev'] or GT apple (base) ['apple'];
    manipulated_id/state = previous chunk's recorded manipulated. Last gate = AG3S.attach stub (as T22). After each request
    latch.update(label='manipulated', score=1, runner_up=0) (as T22).
The T22 AG3S.attach stub gains `admissibility_of` (AG3S.attach calls it since T26; see _make_ag3s).
Reported: attach / detach request t_step, with the old rule (recorded summary.grasp.attach) alongside for (c).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T22'))
import replay_t22 as R  # noqa: E402
from AG3S.trajopt import wire  # noqa: E402
from AG3S.trajopt.grasp_latch import GraspLatch, LatchConfig, grasp_signal_from_feedback  # noqa: E402
from AG3S.ag3s.constraints.attached import parent_reach, AttachRejected  # noqa: E402

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30/off3'))


def _make_ag3s():
    """T22 stub + `admissibility_of` (AG3S.attach calls it since T26): same `assess` call as AG3S.admissibility_of with
    max opening 71.46 mm, no destination, overlap distance 5 mm, default overlap fraction."""
    import types
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.stages.admissibility import assess
    from AG3S.ag3s.config import AG3SConfig
    frac = float(AG3SConfig().clustering.destination_overlap_fraction)
    stub = types.SimpleNamespace(constraint_robot_model=R.RM, _attached=None)
    stub.admissibility_of = lambda pts: assess(np.asarray(pts, np.float64).reshape(-1, 3), extents=None, max_opening_m=0.07146,
                                               destination=None, overlap_distance=0.005, overlap_fraction=frac)
    return stub, AG3S.attach


R.make_ag3s = _make_ag3s
T28 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))


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
    latch = GraspLatch(LatchConfig(), evidence=True)
    for _ in range(3):
        latch.update(label="apple", score=1.0, runner_up=0.0)
    rows = []
    seq = 1
    for i, (app, meas, exe) in enumerate(chunks):
        seq += 1
        pl = bool(placed[i]) if placed is not None else False
        sig = grasp_signal_from_feedback(fb(seq - 1, app, meas, exe), 0, 0.85)
        before = latch.phase.value
        ev = latch.observe_grasp(sig, reach_m=reach, manipulated_id=0, manipulated_state="visible", placed=pl)
        rows.append({"req": i, "applied": [round(float(a), 4) for a in app], "executed": exe, "measured": round(float(meas), 4),
                     "state_before": before, "state": latch.phase.value, "attach": ev.attach, "detach": ev.detach,
                     "checks": (ev.evidence or {}).get("checks"), "opening_delta": (ev.evidence or {}).get("opening_delta"), "note": ev.note})
    return {"name": name, "reach_m_given": reach, "n_attach": sum(r["attach"] for r in rows),
            "attach_req": [r["req"] for r in rows if r["attach"]], "detach_req": [r["req"] for r in rows if r["detach"]], "rows": rows}


def empty_sweep():
    m = R._m
    P, C, S = R.load_run("t17")
    q0 = np.asarray(P[2]["qpos"], np.float64)
    act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}

    def simulate(cmd_chunks):
        d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d)
        for i in range(m.nu):
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i][0]]]
        chunks, trace = [], []
        for c in cmd_chunks:
            for u in c:
                d.ctrl[act["gripper_l_act"]] = u * -0.045
                for _ in range(33):
                    mujoco.mj_step(m, d)
                trace.append(abs(d.qpos[R.GL]) / 0.045)
            chunks.append((c, abs(d.qpos[R.GL]) / 0.045, None))
        return chunks, trace

    out = []
    for target in (0.0, 0.3, 0.57):
        for s in range(8):
            c2 = [1.0] * s + [target] * (8 - s)
            chunks, trace = simulate([[1.0] * 8, c2, [target] * 8, [target] * 8])
            r = run_seq(f"empty close to {target} starting at step {s}", chunks)
            att = r["attach_req"]
            out.append({"close_target": target, "start_step_in_chunk": s,
                        "opening_at_boundaries": [round(ch[1], 4) for ch in chunks], "attach_req": att,
                        "attach_opening": None if not att else r["rows"][att[0]]["measured"],
                        "states": [x["state"] for x in r["rows"]],
                        "failed_checks": [[k for k, v in (x["checks"] or {}).items() if not v] for x in r["rows"]],
                        "opening_delta": [x["opening_delta"] for x in r["rows"]]})
    return out


def e3_replay(cond, run, source):
    an = json.loads((T28 / f"p3/{cond}_analysis.json").read_text())
    chunks = [c for c in an["runs"][run]["chunks"] if c.get("server_chunk")]
    fr = [json.loads(l) for l in (T28 / f"{cond}_rec/{run}/frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {x["seq"]: x for x in fr if x["kind"] == "planning"}
    cfg = LatchConfig()
    latch = GraspLatch(cfg, evidence=True)
    stub, gate = R.make_ag3s()
    rows = []
    prev_man = None
    for c in chunks:
        z = np.load(T28 / f"{cond}_rec" / c["server_chunk"], allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        pr = plan[c["seq"]]
        qpos = np.asarray(pr["qpos"], np.float64)
        q20 = qpos[R.Q20]
        mj, Tb = R.mj_state(qpos)
        apple_base = (Tb @ np.r_[np.asarray(pr["object_poses"]["apple"]["pos"], np.float64), 1.0])[:3]
        if source == "apple":
            centroid = apple_base
        else:
            centroid = None if not prev_man else np.asarray(prev_man["centroid"], np.float64)
        reach = None if centroid is None else parent_reach(R.RM, q20, "ee_finger_l1", centroid)
        fbk = s.get("exec_feedback") or {"available": False}
        sig = grasp_signal_from_feedback(fbk, 0, cfg.gripper_closed_below)
        before = latch.phase.value
        mid = None if not prev_man else prev_man["id"]
        mst = None if not prev_man else prev_man["state"]
        if source == "apple":
            mid, mst = 0, "visible"
        ev = latch.observe_grasp(sig, reach_m=reach, manipulated_id=mid, manipulated_state=mst, placed=False)
        rejected = None
        if ev.attach:
            try:
                gate(stub, R.target_geom(centroid, None), robot_state=q20, parent_link="ee_finger_l1", label="manipulated", reach=cfg.reach)
            except AttachRejected as exc:
                rejected = exc.record(); ev = latch.revert_attach(exc.reason)
        latch.update(label="manipulated", score=1.0, runner_up=0.0)
        g = s.get("grasp") or {}
        rows.append({"t": c["t"], "seq": c["seq"], "state_before": before, "state": latch.phase.value, "attach": bool(ev.attach),
                     "detach": bool(ev.detach), "gate_rejected": rejected, "opening": sig.opening,
                     "opening_delta": (ev.evidence or {}).get("opening_delta"), "checks": (ev.evidence or {}).get("checks"),
                     "reach_mm": None if reach is None else 1000 * reach, "note": ev.note,
                     "recorded_state": g.get("state"), "recorded_attach": bool(g.get("attach")), "recorded_detach": bool(g.get("detach"))})
        prev_man = s.get("manipulated")
    return {"cond": cond, "run": run, "source": source,
            "attach_t": [r["t"] for r in rows if r["attach"]], "detach_t": [r["t"] for r in rows if r["detach"]],
            "recorded_attach_t": [r["t"] for r in rows if r["recorded_attach"]], "recorded_detach_t": [r["t"] for r in rows if r["recorded_detach"]],
            "n_gate_rejected": sum(1 for r in rows if r["gate_rejected"]),
            "transitions": [{"t": r["t"], "from": r["state_before"], "to": r["state"], "note": r["note"]} for r in rows if r["state"] != r["state_before"] or r["attach"] or r["detach"]],
            "rows": rows}


def main():
    res = {"definitions": __doc__, "latch_config": {"settle_epsilon": LatchConfig().settle_epsilon, "reach": LatchConfig().reach,
                                                    "gripper_closed_below": LatchConfig().gripper_closed_below}}
    live = {}
    for run in ("t14", "t16", "t17"):
        for src in ("apple", "server_prev"):
            rows = R.replay(run, src)
            s = R.summarize(rows)
            live[f"{run}_{src}"] = {"summary": s, "rows": rows}
            print(run, src, "attach", s["attach_t"], "detach", s["detach_t"], "legacy", s["legacy_attach_t"], flush=True)
    res["live"] = live
    sw = empty_sweep()
    res["empty_sweep"] = sw
    print("empty sweep attaches", sum(1 for r in sw if r["attach_req"]), "/", len(sw), flush=True)
    e3 = {}
    for cond in ("E3a", "E3b"):
        an = json.loads((T28 / f"p3/{cond}_analysis.json").read_text())
        for run in an["runs"]:
            for src in ("server_prev", "apple"):
                r = e3_replay(cond, run, src)
                e3[f"{run}_{src}"] = r
                print(run, src, "attach", r["attach_t"], "detach", r["detach_t"], "recorded", r["recorded_attach_t"], r["recorded_detach_t"], flush=True)
    res["e3"] = e3
    (OUT / "f4_replay.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
