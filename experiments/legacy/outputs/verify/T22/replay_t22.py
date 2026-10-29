"""T22 verifier replay — GraspLatch(evidence=True) on old records (#1-#3) and synthetic sequences (#4-#5).

Feedback reconstruction (old records carry no `exec_feedback`):
  request seq k (k>=2) gets the facts of chunk seq k-1:
    executed[i]        = control frame (chunk_seq=k-1, step_in_chunk=i).executed
    applied_gripper[i] = planning(seq k-1).actions_reference[i][col] if executed[i]
                         else |control.qpos[adr]| / 0.045   (HOLD re-commands current state)
    planned_gripper[i] = planning(seq k-1).actions_reference[i][col]
    measured_gripper   = |planning(seq k).qpos[adr]| / 0.045   (capture-time state of request k)
  seq 1: available=False ("first chunk").
  col = 7 (left) / 15 (right) = wire.gripper_columns(); adr = MuJoCo jnt_qposadr (l1=27, r1=18).
  `actions` and `actions_reference` gripper columns are identical in all three records (max diff 0).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import argparse, glob, json, pathlib, sys, types
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.trajopt import wire
from AG3S.trajopt.grasp_latch import GraspLatch, LatchConfig, GraspPhase, grasp_signal_from_feedback
from AG3S.ag3s.robot_models import load_rby1, DEFAULT_RBY1_JOINTS
from AG3S.ag3s.constraints.attached import parent_reach, AttachRejected, DEFAULT_ATTACH_REACH_M

ROOT = pathlib.Path(_ag3s_workspace_path(''))
MODEL = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {
    "t17": "outputs/live_test/20260928_t17/shadow_ep1807",
    "t16": "outputs/live_test/20260928_t16/execute_ep1807",
    "t14": "outputs/live_test/20260928_t14/execute_ep1807",
}
OPEN = 0.045
N_ARM = 7

_m = mujoco.MjModel.from_xml_path(MODEL)
_d = mujoco.MjData(_m)
JADR = {mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_JOINT, j): int(_m.jnt_qposadr[j]) for j in range(_m.njnt)}
Q20 = [JADR[j] for j in DEFAULT_RBY1_JOINTS]
GL, GR = JADR["gripper_finger_l1"], JADR["gripper_finger_r1"]
COL_L, COL_R = wire.gripper_columns()
RM = load_rby1()


def body_T(name):
    b = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, name)
    T = np.eye(4); T[:3, :3] = _d.xmat[b].reshape(3, 3); T[:3, 3] = _d.xpos[b]
    return T


def mj_state(qpos):
    _d.qpos[:] = np.asarray(qpos, np.float64); mujoco.mj_forward(_m, _d)
    Tb = np.linalg.inv(body_T("base"))
    return {n: (Tb @ body_T(n))[:3, 3].copy() for n in ("ee_finger_l1", "ee_finger_l2", "apple")}, Tb


def load_run(run):
    L = [json.loads(l) for l in open(ROOT / RUNS[run] / "frames.jsonl")]
    P = {r["seq"]: r for r in L if r["kind"] == "planning"}
    C = {}
    for r in L:
        if r["kind"] == "control":
            C.setdefault(r["chunk_seq"], []).append(r)
    for v in C.values():
        v.sort(key=lambda r: r["step_in_chunk"])
    S = {}
    for f in sorted(glob.glob(str(ROOT / pathlib.Path(RUNS[run]).parent / "server_constraints/run_0000/chunk_*.npz"))):
        z = np.load(f, allow_pickle=True)
        k = int(z["chunk_index"])
        c = np.asarray(z["target_centroid"], np.float64) if "target_centroid" in z.files else np.zeros(0)
        s = json.loads(str(z["summary_json"]))
        es = s.get("esdf_stats") if isinstance(s.get("esdf_stats"), dict) else {}
        S[k + 1] = {"centroid": c if c.size == 3 else None,
                    "points": np.asarray(z["target_points"], np.float64) if "target_points" in z.files else None,
                    "n_attached": (es or {}).get("n_attached_points")}
    return P, C, S


def feedback_for(P, C, k, hand=0):
    if k - 1 not in C:
        return wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: wire.no_exec_feedback("first chunk")})
    ctl = C[k - 1]; prev = P[k - 1]
    ref = np.asarray(prev["actions_reference"], np.float64)
    flags = [bool(r["executed"]) for r in ctl]
    applied, planned = [], []
    for i, r in enumerate(ctl):
        pl = ref[r["step_in_chunk"], [COL_L, COL_R]]
        planned.append(pl)
        if r["executed"]:
            applied.append(pl)
        else:
            q = np.asarray(r["qpos"])
            applied.append([abs(q[GL]) / OPEN, abs(q[GR]) / OPEN])
    q = np.asarray(P[k]["qpos"])
    blob = wire.make_exec_feedback(
        seq=k - 1, t_step_start=int(ctl[0]["t_step"]), executed=flags,
        executed_chunk="none" if not any(flags) else "reference",
        ipc="ok" if all(flags) else "unsafe", hold_kind=None if all(flags) else "unsafe",
        hold_reason=None if all(flags) else "record: executed=False",
        planned_gripper=np.asarray(planned), applied_gripper=np.asarray(applied),
        applied_arm=np.zeros((len(flags), 2 * N_ARM)),
        measured_gripper=(abs(q[GL]) / OPEN, abs(q[GR]) / OPEN))
    return wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: blob})


def make_ag3s():
    """Last gate: the real `AG3S.attach` bound to a stub carrying only the robot model."""
    from AG3S.ag3s.runtime.pipeline import AG3S
    stub = types.SimpleNamespace(constraint_robot_model=RM, _attached=None)
    return stub, AG3S.attach


def target_geom(centroid, points=None):
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry
    if points is None or len(points) < 4:
        points = np.asarray(centroid) + np.random.default_rng(0).normal(0, 0.01, (60, 3))
    points = np.asarray(points, np.float64)
    return TargetGeometry(id=0, points=points, point_indices=np.arange(len(points)),
                          centroid=np.asarray(centroid, np.float64), bounding_geometry=fit_sphere(points),
                          attention_score=1.0, confidence=1.0)


def replay(run, source, reach_limit=None):
    P, C, S = load_run(run)
    cfg = LatchConfig() if reach_limit is None else LatchConfig(reach=reach_limit)
    latch = GraspLatch(cfg, evidence=True)
    legacy = GraspLatch(LatchConfig(), evidence=False)
    stub, gate = make_ag3s()
    rows = []
    for k in sorted(P):
        pr = P[k]
        qpos = np.asarray(pr["qpos"], np.float64)
        q20 = qpos[Q20]
        mj, Tb = mj_state(qpos)
        apple_world = np.asarray(pr["object_poses"]["apple"]["pos"], np.float64)
        apple_base = (Tb @ np.r_[apple_world, 1.0])[:3]
        srv_prev = S.get(k - 1, {}).get("centroid")      # latest grounding before this request's
        srv_same = S.get(k, {}).get("centroid")
        if source == "apple":
            centroid, points = apple_base, None
        else:
            centroid, points = srv_prev, S.get(k - 1, {}).get("points")
        reach = None if centroid is None else parent_reach(RM, q20, "ee_finger_l1", centroid)
        fb = feedback_for(P, C, k)
        sig = grasp_signal_from_feedback(fb, 0, cfg.gripper_closed_below)
        before = latch.phase.value
        ev = latch.observe_grasp(sig, reach_m=reach, manipulated_id=0 if centroid is not None else None,
                                 manipulated_state="visible" if centroid is not None else None, placed=False)
        rejected = None
        if ev.attach:
            try:
                gate(stub, target_geom(centroid, points), robot_state=q20, parent_link="ee_finger_l1",
                     label="manipulated", reach=cfg.reach)
            except AttachRejected as exc:
                rejected = exc.record()
                ev = latch.revert_attach(exc.reason)
        # lock after (as `_run_latch` does after grounding) — constant label (T20 holds one object)
        latch.update(label="manipulated", score=1.0, runner_up=0.0)
        planned_first = float(np.asarray(pr["actions_reference"])[0, COL_L])
        lev = legacy.update(label="manipulated", score=1.0, runner_up=0.0, gripper=planned_first)
        rows.append({
            "seq": k, "t_step": int(pr["t_step"]),
            "n_exec": sig.n_exec, "n_steps": sig.n_steps, "closing_steps": sig.closing_steps,
            "last_command": sig.last_command, "measured": sig.opening,
            "feedback_available": sig.available,
            "reach_mm": None if reach is None else round(reach * 1000, 2),
            "reach_apple_urdf_l1_mm": round(parent_reach(RM, q20, "ee_finger_l1", apple_base) * 1000, 2),
            "reach_server_prev_mm": None if srv_prev is None else round(parent_reach(RM, q20, "ee_finger_l1", srv_prev) * 1000, 2),
            "reach_server_same_mm": None if srv_same is None else round(parent_reach(RM, q20, "ee_finger_l1", srv_same) * 1000, 2),
            "mj_l1_body_apple_mm": round(float(np.linalg.norm(mj["ee_finger_l1"] - apple_base)) * 1000, 2),
            "mj_finger_mid_apple_mm": round(float(np.linalg.norm(0.5 * (mj["ee_finger_l1"] + mj["ee_finger_l2"]) - apple_base)) * 1000, 2),
            "server_prev_to_apple_mm": None if srv_prev is None else round(float(np.linalg.norm(srv_prev - apple_base)) * 1000, 2),
            "state_before": before, "state": latch.phase.value,
            "attach": bool(ev.attach), "detach": bool(ev.detach), "gate_rejected": rejected,
            "checks": (ev.evidence or {}).get("checks"), "opening_before": (ev.evidence or {}).get("opening_before"),
            "settled_steps": (ev.evidence or {}).get("settled_steps"), "note": ev.note,
            "planned_first_row": planned_first, "legacy_attach": bool(lev.attach), "legacy_detach": bool(lev.detach),
            "legacy_phase": lev.phase.value,
            "server_n_attached_points": S.get(k, {}).get("n_attached"),
            "urdf_l1_origin": np.round(RM.link_pose(q20, "ee_finger_l1")[:3, 3], 5).tolist(),
            "apple_base": np.round(apple_base, 5).tolist(),
            "server_prev_centroid": None if srv_prev is None else np.round(srv_prev, 5).tolist(),
        })
    return rows


def summarize(rows):
    att = [r for r in rows if r["attach"]]; det = [r for r in rows if r["detach"]]
    rej = [r for r in rows if r["gate_rejected"]]
    failed = {}
    for r in rows:
        if r["checks"]:
            for kk, v in r["checks"].items():
                if not v:
                    failed[kk] = failed.get(kk, 0) + 1
    return {"n_requests": len(rows), "n_attach": len(att), "attach_t": [r["t_step"] for r in att],
            "attach_rows": [{k: r[k] for k in ("seq", "t_step", "reach_mm", "measured", "last_command", "opening_before", "settled_steps", "checks", "note")} for r in att],
            "n_detach": len(det), "detach_t": [r["t_step"] for r in det],
            "n_gate_rejected": len(rej), "gate_rejections": [{"t_step": r["t_step"], **r["gate_rejected"]} for r in rej],
            "phases_visited": sorted({r["state"] for r in rows}),
            "transitions": [{"t_step": r["t_step"], "from": r["state_before"], "to": r["state"], "note": r["note"]} for r in rows if r["state"] != r["state_before"] or r["attach"] or r["detach"]],
            "failed_check_counts_in_closing": failed,
            "legacy_attach_t": [r["t_step"] for r in rows if r["legacy_attach"]],
            "legacy_detach_t": [r["t_step"] for r in rows if r["legacy_detach"]],
            "n_hold_chunks": sum(1 for r in rows if r["feedback_available"] and r["n_exec"] < r["n_steps"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", default="apple", choices=["apple", "server_prev"])
    ap.add_argument("--reach", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = replay(a.run, a.source, a.reach)
    out = {"run": a.run, "record": RUNS[a.run], "centroid_source": a.source,
           "reach_limit_m": a.reach if a.reach is not None else LatchConfig().reach,
           "default_attach_reach_m": DEFAULT_ATTACH_REACH_M, "summary": summarize(rows), "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
    s = out["summary"]
    print(a.run, a.source, "attach", s["n_attach"], s["attach_t"], "detach", s["detach_t"], "gate_rej", s["n_gate_rejected"],
          "legacy_attach_t", s["legacy_attach_t"], "holds", s["n_hold_chunks"], "failed", s["failed_check_counts_in_closing"])
    for t in s["transitions"]:
        print("  ", t)
