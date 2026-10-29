"""T43 DX-1 tracking lag at the 25 contact rows (t*): which planned (refined) row the actual hand pose is closest to.
usage: dx1_lag.py OUT_JSON      (reads raw/dx1_summary.json; MuJoCo FK only, model = work-t43w model_transport_pick_place_obstacles.xml,
                                 same qpos layout nq 66 in every snapshot)
actual  = EE body (EE_BODY_L|R) of the post state of row t* (= control row t*+1 qpos; t*+1 exists for every carry row here).
planned = EE body of the planning-row qpos of a chunk with the active-arm joints = refined_chunk row j (what the TO checked),
          for the previous chunk (rows 0..7) and the executing chunk (rows 0..7) -> sequence index i = 8*(chunk offset) + j.
lag_steps = (index of row k of the executing chunk) - argmin_i |EE_actual - EE_planned(i)|;  d_at_k_mm, d_best_mm.
hand_speed_mm_per_step = |EE planned(row k) - EE planned(row k-1)| (k-1 from the previous chunk when k = 0)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco
import dx_common as C

XML = "${AG3S_PROJECT_ROOT}-t43w/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
S = json.load(open(_ag3s_workspace_path('outputs/verify/T43/DX/raw/dx1_summary.json')))["rows"]
out = []
for r in S:
    a = C.run_entry(r["key"]); side = a["side"]; ctrl, plan = C.frames(a["run_dir"]); ch = C.segment_chunks(a["server_segment"])
    qi, c0 = (C.LEFT_Q, 0) if side == "left" else (C.RIGHT_Q, 8); ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L" if side == "left" else "EE_BODY_R")

    def ee_of(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); return d.xpos[ee].copy()
    t = r["t_star"]; seq = r["chunk_seq"]; k = r["k"]
    act = ee_of(np.asarray(ctrl[t + 1]["qpos"], float))
    P = []
    for sq in (seq - 1, seq):
        if sq not in ch:
            P += [None] * 8; continue
        Rf = np.asarray(np.load(ch[sq], allow_pickle=True)["refined_chunk"], float)
        for j in range(8):
            q = np.asarray(plan[sq]["qpos"], float).copy(); q[qi] = Rf[j, c0:c0 + 7]; P.append(ee_of(q))
    dist = np.array([np.inf if p is None else 1e3 * np.linalg.norm(act - p) for p in P])
    ik = 8 + k; ib = int(np.argmin(dist))
    sp = None if P[ik - 1] is None else 1e3 * float(np.linalg.norm(P[ik] - P[ik - 1]))
    out.append({"key": r["key"], "t_star": t, "chunk_seq": seq, "k": k, "d_at_k_mm": float(dist[ik]), "d_best_mm": float(dist[ib]), "best_index": ib,
                "lag_steps": ik - ib, "hand_speed_mm_per_step": sp, "dist_by_index_mm": dist.tolist()})
    print(r["key"], t, "k", k, "d@k %.1f best %.1f lag %d speed %s" % (dist[ik], dist[ib], ik - ib, None if sp is None else round(sp, 1)))
lags = [o["lag_steps"] for o in out]
res = {"definitions": __doc__, "rows": out, "lag_steps_values": sorted(lags), "median_lag_steps": float(np.median(lags)),
       "median_speed_mm_per_step": float(np.median([o["hand_speed_mm_per_step"] for o in out if o["hand_speed_mm_per_step"] is not None]))}
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("lags", sorted(lags), "median speed", res["median_speed_mm_per_step"])
