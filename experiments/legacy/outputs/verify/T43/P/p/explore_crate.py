"""T43 P: crate geometry (collision geoms of body 'crate' and its children) at the captured reset scene + warmup replay check
(1000 mj_step from the captured reset state, hurdle parked) vs recorded planning row 0 qpos (E0, E3b). usage: explore_crate.py TAG..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F
out = {}
for tag in sys.argv[1:]:
    c = F.Ctx(tag); m, d = c.m, c.d
    bc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
    geoms = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g]); k = b
        while k > 0 and k != bc: k = int(m.body_parentid[k])
        if k != bc: continue
        R = d.geom_xmat[g].reshape(3, 3)
        geoms.append({"id": g, "name": c.gname(g), "type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
                      "xpos": d.geom_xpos[g].tolist(), "xmat": R.tolist(), "coll": bool(m.geom_contype[g] or m.geom_conaffinity[g]),
                      "group": int(m.geom_group[g]), "rbound": float(m.geom_rbound[g])})
    Rb = d.xmat[bc].reshape(3, 3)
    q0 = c.q_reset.copy()
    # warmup replay
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
    z = np.load(F.CAPDIR / f"{tag}_reset.npz"); d.qvel[:] = z["qvel"]; d.ctrl[:] = z["ctrl"]
    for _ in range(1000): mujoco.mj_step(m, d)
    rec = {cond: np.abs(d.qpos - R["Q"][0]).max() for cond, R in c.runs.items()}
    rec_e = {cond: float(np.abs(R["Q"][0] - c.runs["E0"]["Q"][0]).max()) for cond, R in c.runs.items()}
    out[tag] = {"crate_body_xpos": d.xpos[bc].tolist() if False else None, "crate_geoms": geoms,
                "warmup_replay_maxabs_vs_planning_row0": {k: float(v) for k, v in rec.items()}, "E3b_vs_E0_row0": rec_e,
                "robot_reset_vs_row0_maxabs": float(np.abs(q0 - c.runs["E0"]["Q"][0]).max())}
    c.restore(); out[tag]["crate_body_xpos"] = c.d.xpos[bc].tolist(); out[tag]["crate_body_xmat"] = c.d.xmat[bc].reshape(3,3).tolist()
    print(tag, out[tag]["warmup_replay_maxabs_vs_planning_row0"], out[tag]["E3b_vs_E0_row0"], out[tag]["robot_reset_vs_row0_maxabs"], flush=True)
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T43/P/raw/explore_crate.json'), "w"), indent=1)
