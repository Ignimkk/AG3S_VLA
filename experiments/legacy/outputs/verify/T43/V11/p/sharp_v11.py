"""T43 V11 path sharpness of the EXECUTED COMMAND (verifier, CPU only, numbers only). usage: V11_ARM=V11A T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 sharp_v11.py OUT_JSON
Arms: V8 (V8/analysis/analysis.json) and ARM (V11/analysis/analysis_<ARM>.json); runs H and B.
Method = T43TA boundary_metrics (AG3S/trajopt/experiments/ta_offline.py:526) applied to the commands that were actually executed:
  u(t) = traj.npz executed_actions[t] (control row t, arm columns 0-6 / 8-14; 14 joints), chunk boundary at control row t0 = 8 n (n >= 1, first row of a chunk).
  clean boundary = control rows t0-2 .. t0+1 all exist and are executed (frames.jsonl control rows 'executed' true) -- boundaries next to a HOLD chunk are excluded.
  acc_boundary = max over the 14 joints and both second differences ([u(t0-2), u(t0-1), u(t0)] and [u(t0-1), u(t0), u(t0+1)]) of |second difference| (rad / row^2)
                 (= TA acc_boundary_max_rad: ext = [u-2, u-1, Q0, Q1]).
  acc_inner = same inside a chunk: triples that lie within rows t0 .. t0+7 of one chunk (t0 > 0), all executed.
  vel_boundary_ratio = max_j |u(t0) - u(t0-1)| / max_step_j  (max_step = v_urdf * 0.9 / 15, 14 values: 0.1885 x4, 0.377 x2, 0.126 per arm = pi,pi,pi,pi,2pi,2pi,2.0944 rad/s)
  palm path (hurdle runs H only, and also B): palm = MuJoCo body EE_BODY_<L|R> origin (model_transport_pick_place_obstacles.xml) with the 7 active-arm joints set to u(t), every other joint = the run's control row 0 qpos
    (so the palm path depends only on the commanded joints); displacements v0 = P(t0-1)-P(t0-2), v1 = P(t0)-P(t0-1), v2 = P(t0+1)-P(t0); turns (v0,v1), (v1,v2) in degrees where both norms > 2 mm;
    palm_reversal = any turn > 90 deg (TA rule). palm_acc_boundary_mm = max over the two second differences of |P second difference| (mm / row^2).
Also the policy reference: the same acc_boundary / reversal on actions_reference rows is NOT computed (the reference chunks of different chunks are not a continuous command)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402
T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); ARM = os.environ.get("V11_ARM", "V11A")
AN = {"V8": T43 / "V8/analysis/analysis.json", ARM: T43 / f"V11/analysis/analysis_{ARM}.json"}
J14 = list(range(0, 7)) + list(range(8, 15))
MAX_STEP = np.array([np.pi, np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2.094395102] * 2) * 0.9 / 15.0
LEFT_Q = list(range(20, 27)); RIGHT_Q = list(range(11, 18))


def q(x, ps=(50, 90)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max()), "mean": float(x.mean())}


def one(key, a):
    rd = pathlib.Path(a["run_dir"]); ep = int(a["episode"]); side = "right" if ep == 1834 else "left"
    HM.set_arm(side); M = HM.model(); m, d = M["m"], M["d"]
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    z = np.load(rd / "traj.npz", allow_pickle=True); EA = np.asarray(z["executed_actions"], float); T = len(EA)
    ex = np.array([bool(ctrl[t].get("executed", True)) if t in ctrl else False for t in range(T)])
    qi = LEFT_Q if side == "left" else RIGHT_Q; c0 = 0 if side == "left" else 8
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L" if side == "left" else "EE_BODY_R")
    q0 = np.asarray(ctrl[0]["qpos"], float)
    P = np.zeros((T, 3))
    for t in range(T):
        d.qpos[:] = q0; d.qpos[qi] = EA[t, c0:c0 + 7]; mujoco.mj_kinematics(m, d); P[t] = d.xpos[body]
    U = EA[:, J14]
    acc_b, acc_i, velr, turns_all, rev, pacc = [], [], [], [], [], []
    n_clean = n_excl = 0
    for t0 in range(8, T - 1, 8):
        if not ex[t0 - 2:t0 + 2].all():
            n_excl += 1; continue
        n_clean += 1
        ext = U[t0 - 2:t0 + 2]
        acc_b.append(float(np.abs(np.diff(ext, 2, axis=0)).max()))
        velr.append(float((np.abs(U[t0] - U[t0 - 1]) / MAX_STEP).max()))
        Pe = P[t0 - 2:t0 + 2]; v = np.diff(Pe, axis=0); tn = []
        for x, y in ((v[0], v[1]), (v[1], v[2])):
            nx, ny = np.linalg.norm(x), np.linalg.norm(y)
            tn.append(float(np.degrees(np.arccos(np.clip(x @ y / (nx * ny), -1, 1)))) if nx > 2e-3 and ny > 2e-3 else None)
        turns_all += [x for x in tn if x is not None]
        rev.append(float(any(x is not None and x > 90.0 for x in tn)))
        pacc.append(float(np.linalg.norm(np.diff(Pe, 2, axis=0), axis=1).max() * 1e3))
    for t0 in range(0, T - 7, 8):
        if t0 == 0 or not ex[t0:t0 + 8].all():
            continue
        acc_i.append(float(np.abs(np.diff(U[t0:t0 + 8], 2, axis=0)).max()))
    return {"key": key, "n_boundaries_clean": n_clean, "n_boundaries_excluded_hold": n_excl, "acc_boundary_rad": acc_b, "acc_inner_chunk_max_rad": acc_i,
            "vel_boundary_ratio": velr, "palm_turn_deg": turns_all, "palm_reversal": rev, "palm_acc_boundary_mm": pacc}


def main():
    out = {"definitions": __doc__, "arm_new": ARM, "arms": {}}
    for arm, f in AN.items():
        R = json.load(open(f))["runs"]; per = {}
        for key in sorted(R):
            if not key.startswith(arm + "/"):
                continue
            per[key] = one(key, R[key]); print(key, per[key]["n_boundaries_clean"], flush=True)
        out["arms"][arm] = {"per_run": per}
        for stage in ("H", "B", "ALL"):
            sel = [v for k, v in per.items() if stage == "ALL" or k.split("/")[1] == stage]
            cat = lambda f_: [x for v in sel for x in v[f_]]
            rv = cat("palm_reversal")
            out["arms"][arm][stage] = {"n_runs": len(sel), "n_boundaries_clean": sum(v["n_boundaries_clean"] for v in sel), "n_boundaries_excluded_hold": sum(v["n_boundaries_excluded_hold"] for v in sel),
                                       "acc_boundary_rad": q(cat("acc_boundary_rad")), "acc_inner_chunk_max_rad": q(cat("acc_inner_chunk_max_rad")), "vel_boundary_ratio": q(cat("vel_boundary_ratio")),
                                       "n_vel_boundary_gt1": int(sum(x > 1.0 + 1e-9 for x in cat("vel_boundary_ratio"))), "palm_turn_deg": q(cat("palm_turn_deg")),
                                       "palm_reversal_share": (float(np.mean(rv)) if rv else None), "palm_reversal_n": int(sum(rv)), "palm_acc_boundary_mm": q(cat("palm_acc_boundary_mm"))}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    for arm in out["arms"]:
        for st in ("H", "B", "ALL"):
            v = out["arms"][arm][st]; print(arm, st, v["n_boundaries_clean"], "acc_b", {k: round(v["acc_boundary_rad"].get(k, 0), 4) for k in ("median", "p90", "max")}, "rev", v["palm_reversal_share"], "vel>1", v["n_vel_boundary_gt1"])


if __name__ == "__main__":
    main()
