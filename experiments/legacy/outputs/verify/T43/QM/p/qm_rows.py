"""T43 QM — per-row kinematic series of the V8 and V11AB runs (verifier, CPU only, recorded data only, no new robot runs).
usage: qm_rows.py OUT_DIR KEY_FILTER...      (KEY_FILTER: substrings of 'ARM/STAGE/NAME'; empty = all 38 runs)
Inputs: analysis.json of V8 (V8/analysis/analysis.json) and V11AB (V11/analysis/analysis_V11AB.json): run_dir, episode, side.
        run_dir/frames.jsonl (control rows: t_step, chunk_seq, step_in_chunk, executed, qpos = state when the row's action was commanded;
        planning rows: seq, t_step, qpos, actions = refined chunk, actions_reference = policy chunk), run_dir/traj.npz (executed_actions, measured_qpos).
Model: model_transport_pick_place_obstacles.xml of the client snapshot ${AG3S_PROJECT_ROOT}-t43v11 (qpos layout nq 66 as recorded); FK only (mj_kinematics).
Per control row t (T rows):
  M state  = control row t+1 qpos (post state of row t); last row: row t qpos with both arms <- measured_qpos[t]  (XM rule, xm_cross.py).
  C state  = M state with both arms <- executed_actions[t] (cols 0-6 left, 8-14 right); the apple is NOT moved (only TCP / palm are used from C).
  tcp      = site '<side>_ee' position; palm = body EE_BODY_<S> origin; apple = body 'apple' centre (M only).
Per planning chunk seq, step k = 0..7, at the planning-row qpos q_p (chunk deflection, XM / analyze_t43w rule):
  refined = actions[k], reference = actions_reference[k] (both arms replaced); apple moved rigidly with EE_BODY_<S> when in hand at q_p
  (in hand: |apple - finger midpoint| <= 60 mm and apple z - apple z(row 0) >= 20 mm, XM rule; fingers = MJCF ee_finger_<s>1/2 midpoint —
  both fingers, so the MJCF/URDF l1/l2 name swap does not enter).
  dpalm_mag = |palm(refined) - palm(reference)| (mm); dpalm_dz, dtcp_mag;
  apple_bottom(q) = apple geom centre z - sqrt(sum((R_geom[2,:] * geom_size)^2)) (XM apple_bottom rule);
  dapple_bottom_dz = apple_bottom(refined) - apple_bottom(reference) (mm; only when in hand at q_p, else NaN).
  Row t (seq, k) carries the values of its chunk step.
Output OUT_DIR/<ARM>_<STAGE>_<NAME>.npz with arrays t, seq, k, executed, tcp_C, tcp_M, palm_M, apple_M, dpalm_mag, dpalm_dz, dtcp_mag, dapple_bottom_dz, inhand_plan."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, time
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
AN = {"V8": T43 / "V8/analysis/analysis.json", "V11AB": T43 / "V11/analysis/analysis_V11AB.json"}
XML = "${AG3S_PROJECT_ROOT}-t43v11/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
QL = np.arange(20, 27); QR = np.arange(11, 18)

m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
APPLE_B = bid("apple"); APPLE_Q = int(m.jnt_qposadr[m.body_jntadr[APPLE_B]])
APPLE_G = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"); APPLE_SIZE = m.geom_size[APPLE_G][:3].copy()


def setq(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)


def body_T(b):
    T = np.eye(4); T[:3, :3] = d.xmat[b].reshape(3, 3); T[:3, 3] = d.xpos[b]; return T


def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def one(key, a):
    rd = pathlib.Path(a["run_dir"]); side = a["side"]; S = side[0].upper(); fl = side[0]
    ee = bid(f"EE_BODY_{S}"); f1, f2 = bid(f"ee_finger_{fl}1"), bid(f"ee_finger_{fl}2")
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    z = np.load(rd / "traj.npz", allow_pickle=True); EA = np.asarray(z["executed_actions"], float); MQ = np.asarray(z["measured_qpos"], float)
    T = len(EA); assert T == len(ctrl), (T, len(ctrl))
    z0 = float(np.asarray(ctrl[0]["qpos"], float)[APPLE_Q + 2])
    o = {k: [] for k in ("t", "seq", "k", "executed", "tcp_C", "tcp_M", "palm_M", "apple_M")}
    for t in range(T):
        r = ctrl[t]
        if t + 1 in ctrl:
            qM = np.asarray(ctrl[t + 1]["qpos"], float)
        else:
            qM = np.asarray(r["qpos"], float).copy(); qM[QL] = MQ[t, 0:7]; qM[QR] = MQ[t, 8:15]
        setq(qM); o["tcp_M"].append(d.site_xpos[site].copy()); o["palm_M"].append(d.xpos[ee].copy()); o["apple_M"].append(d.xpos[APPLE_B].copy())
        qC = qM.copy(); qC[QL] = EA[t, 0:7]; qC[QR] = EA[t, 8:15]
        setq(qC); o["tcp_C"].append(d.site_xpos[site].copy())
        o["t"].append(t); o["seq"].append(int(r["chunk_seq"])); o["k"].append(int(r["step_in_chunk"])); o["executed"].append(bool(r.get("executed", True)))
    # chunk deflection at the planning-row state
    defl = {}
    for seq, p in plan.items():
        qp = np.asarray(p["qpos"], float); A = np.asarray(p["actions"], float); R_ = np.asarray(p["actions_reference"], float)
        setq(qp); T0 = body_T(ee); tip = 0.5 * (d.xpos[f1] + d.xpos[f2]); ap = d.xpos[APPLE_B].copy()
        inhand = bool(np.linalg.norm(ap - tip) <= 0.060 and ap[2] - z0 >= 0.020)
        Ta = np.eye(4); Ta[:3, :3] = qmat(qp[APPLE_Q + 3:APPLE_Q + 7]); Ta[:3, 3] = qp[APPLE_Q:APPLE_Q + 3]
        for k in range(min(8, len(A))):
            res = {}
            for nm, act in (("rf", A[k]), ("rc", R_[k])):
                q = qp.copy(); q[QL] = act[0:7]; q[QR] = act[8:15]
                setq(q)
                if inhand:
                    Tn = body_T(ee) @ np.linalg.inv(T0) @ Ta; qa = np.zeros(4); mujoco.mju_mat2Quat(qa, Tn[:3, :3].reshape(-1))
                    q[APPLE_Q:APPLE_Q + 3] = Tn[:3, 3]; q[APPLE_Q + 3:APPLE_Q + 7] = qa; setq(q)
                gR = d.geom_xmat[APPLE_G].reshape(3, 3)
                res[nm] = (d.xpos[ee].copy(), d.site_xpos[site].copy(), float(d.geom_xpos[APPLE_G][2] - np.sqrt(np.sum((gR[2, :] * APPLE_SIZE) ** 2))))
            dp = res["rf"][0] - res["rc"][0]
            defl[(seq, k)] = (1e3 * float(np.linalg.norm(dp)), 1e3 * float(dp[2]), 1e3 * float(np.linalg.norm(res["rf"][1] - res["rc"][1])),
                              1e3 * (res["rf"][2] - res["rc"][2]) if inhand else np.nan, inhand)
    nanrow = (np.nan, np.nan, np.nan, np.nan, False)
    D = [defl.get((s, k), nanrow) for s, k in zip(o["seq"], o["k"])]
    out = {k: np.asarray(v) for k, v in o.items()}
    out["dpalm_mag"] = np.array([x[0] for x in D]); out["dpalm_dz"] = np.array([x[1] for x in D]); out["dtcp_mag"] = np.array([x[2] for x in D])
    out["dapple_bottom_dz"] = np.array([x[3] for x in D]); out["inhand_plan"] = np.array([x[4] for x in D])
    return out


if __name__ == "__main__":
    od = pathlib.Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True); flt = sys.argv[2:]
    for arm, f in AN.items():
        runs = json.load(open(f))["runs"]
        for key in sorted(k for k in runs if k.startswith(arm + "/")):
            if flt and not any(x in key for x in flt):
                continue
            t0 = time.time(); o = one(key, runs[key])
            np.savez_compressed(od / (key.replace("/", "_") + ".npz"), **o)
            print(key, len(o["t"]), f"{time.time() - t0:.1f}s", flush=True)
