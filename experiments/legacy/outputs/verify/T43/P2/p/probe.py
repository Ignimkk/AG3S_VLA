"""T43 P2 probe: reset state vs T43 P capture (bd6a002), crate geometry in the crate frame, windows, apple path."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, math
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
import numpy as np, mujoco
import p2lib as F
out = {}
for tag in F.TAGS:
    c = F.Ctx(tag); m, d = c.m, c.d; c.restore()
    r = {"arm": c.arm, "target": c.episode.get("target_fruit")}
    old = F.pathlib.Path(f"{_ag3s_workspace_path()}/outputs/verify/T43/P/raw/capture/{tag}_reset.npz")
    if old.exists():
        z = np.load(old); r["reset_qpos_maxabs_vs_T43P"] = float(np.abs(z["qpos"] - c.q_reset).max())
    bc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
    cxy = d.xpos[bc].copy(); Rc = d.xmat[bc].reshape(3, 3); yaw = math.atan2(Rc[1, 0], Rc[0, 0])
    r["crate_pos"] = cxy.tolist(); r["crate_yaw_deg"] = math.degrees(yaw)
    geoms = []
    for g in c.CRATE:
        # world AABB corners via geom aabb in geom frame -> crate frame
        ctr = m.geom_aabb[g, :3]; ext = m.geom_aabb[g, 3:]
        pts = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pg = ctr + ext * np.array([sx, sy, sz])
                    pw = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ pg
                    pts.append(Rc.T @ (pw - cxy))
        pts = np.array(pts)
        geoms.append({"name": c.gname(g), "type": int(m.geom_type[g]), "lo": pts.min(0).round(4).tolist(), "hi": pts.max(0).round(4).tolist(),
                      "top_above_table": round(float((cxy[2] + pts[:, 2].max()) - F.TABLE_Z), 4)})
    r["crate_geoms_crate_frame"] = geoms
    ap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    r["apple_crate_frame"] = (Rc.T @ (d.xpos[ap] - cxy)).round(4).tolist()
    r["apple_rbound"] = float(m.geom_rbound[c.B_APPLE].max())
    sup = F.gjk.make_support(m, d, int(c.B_APPLE[0]))
    r["apple_extent_xyz"] = [float(sup(e)[k] - sup(-e)[k]) for k, e in enumerate(np.eye(3))]
    # reset gripper (both) z range above table, crate-frame xy box
    for S in ("L", "R"):
        gg = [g for g in c.ROBOT_ALL if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) in (f"EE_BODY_{S}", f"ee_finger_{S.lower()}1", f"ee_finger_{S.lower()}2")]
        P = []
        for g in gg:
            s = F.gjk.make_support(m, d, int(g))
            for v in np.vstack([np.eye(3), -np.eye(3)]):
                P.append(s(v))
        P = np.array(P); Pc = (Rc.T @ (P - cxy).T).T
        r[f"reset_grip_{S}_crate_frame_lo_hi"] = [Pc.min(0).round(4).tolist(), Pc.max(0).round(4).tolist()]
        r[f"reset_grip_{S}_bottom_above_table"] = float(P[:, 2].min() - F.TABLE_Z)
    for cond, R in c.runs.items():
        Q = R["Q"]; t = R["t"]
        w = {k: R[k] for k in ("lift_t", "grasp_t", "place_t", "source")}
        # T40A-rule lift recomputed (check against T40A for the 5 T40 tags)
        z = Q[:, 40]; ok = np.where((t >= 0) & (t <= R["grasp_t"]) & (z < z[0] + 0.002))[0]; w["lift_t_recomputed"] = int(t[ok[-1] + 1])
        car = (t >= R["lift_t"]) & (t < R["place_t"])
        A = Q[car][:, 38:41]; Ac = (Rc.T @ (A - cxy).T).T
        w["apple_car_crate_frame_first_last"] = [Ac[0].round(4).tolist(), Ac[-1].round(4).tolist()]
        w["apple_car_z_above_table_max"] = float(A[:, 2].max() - F.TABLE_Z)
        Af = Q[-1, 38:41]; w["apple_final_crate_frame"] = (Rc.T @ (Af - cxy)).round(4).tolist()
        w["run_dir"] = R["run_dir"]; w["t0_vs_reset_maxabs_robot_qpos"] = float(np.abs(Q[0][:31] - c.q_reset[:31]).max())
        r[cond] = w
    out[tag] = r; print(tag, json.dumps({k: v for k, v in r.items() if k != "crate_geoms_crate_frame"}), flush=True)
json.dump(out, open(F.RAW / "probe.json", "w"), indent=1)
