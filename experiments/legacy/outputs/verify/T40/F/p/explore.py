"""T40 F step 0: scene geometry facts per episode (reset gripper box, crate box, fruits, carry path) -> raw/explore_<tag>.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F, gjk
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/raw'))


def aabb(ctx, geoms):
    lo, hi = np.full(3, 9.0), np.full(3, -9.0)
    for g in geoms:
        s = gjk.make_support(ctx.m, ctx.d, int(g))
        for k in range(3):
            e = np.zeros(3); e[k] = 1
            hi[k] = max(hi[k], s(e)[k]); lo[k] = min(lo[k], s(-e)[k])
    return lo, hi


for tag in sys.argv[1:]:
    c = F.Ctx(tag); m, d = c.m, c.d
    res = {"tag": tag, "model_xml": c.model_xml}
    c.restore()
    lo, hi = aabb(c, c.A_GRIP); res["reset_gripper_aabb"] = [lo.tolist(), hi.tolist()]
    lo, hi = aabb(c, c.CRATE); res["crate_aabb"] = [lo.tolist(), hi.tolist()]
    for n in ("apple", "banana", "orange", "pear"):
        gs = [g for g in np.r_[c.B_APPLE, c.FRUITS] if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) == n]
        lo, hi = aabb(c, gs); res[f"{n}_aabb"] = [lo.tolist(), hi.tolist()]
    # left arm geoms below table+0.45 at reset
    for cond, R in c.runs.items():
        e = {}
        for nm, rows in (("t0", [0]),):
            d.qpos[:] = R["Q"][0]; mujoco.mj_kinematics(m, d)
            lo, hi = aabb(c, c.A_GRIP); e["t0_gripper_aabb"] = [lo.tolist(), hi.tolist()]
        mk = (R["t"] >= R["lift_t"]) & (R["t"] <= R["place_t"])
        bapple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        path = []
        for i in np.where(mk)[0]:
            d.qpos[:] = R["Q"][i]; mujoco.mj_kinematics(m, d)
            s = gjk.make_support(m, d, int(c.B_APPLE[0]))
            path.append([int(R["t"][i])] + d.xpos[bapple].tolist() + [s(np.array([0, 0, -1.0]))[2] - F.TABLE_Z])
        e["carry_path_t_x_y_z_bottom"] = path
        e["windows"] = {k: R[k] for k in ("lift_t", "grasp_t", "place_t")}
        res[cond] = e
    c.restore()
    (OUT / f"explore_{tag}.json").write_text(json.dumps(res))
    P0 = np.array(res["E0"]["carry_path_t_x_y_z_bottom"]); P3 = np.array(res["E3b"]["carry_path_t_x_y_z_bottom"])
    print(tag, "reset grip", np.round(res["reset_gripper_aabb"], 3).tolist(), "crate", np.round(res["crate_aabb"], 3).tolist())
    print("   apple", np.round(res["apple_aabb"], 3).tolist(), "banana", np.round(res["banana_aabb"], 3).tolist(),
          "orange", np.round(res["orange_aabb"], 3).tolist(), "pear", np.round(res["pear_aabb"], 3).tolist())
    for nm, Pp in (("E0", P0), ("E3b", P3)):
        sel = Pp[::6]
        print("   ", nm, [(int(r[0]), round(r[1], 3), round(r[2], 3), round(r[4], 3)) for r in sel])
