"""T40 A step 1-2: carry path extraction for the 22 both-success T39 pairs (verifier).
grasp_t / place_t from T39 s23/analysis.json (analyze_e0 rule, planning t_step).
Per control row t in [grasp_t, place_t]: recorded qpos (66) -> mj_forward on model_transport_pick_place_obstacles.xml
(obstacles parked) -> apple centre (qpos 38:41 == xpos apple), left TCP = site 'left_ee', finger midpoint
(ee_finger_l1/l2 body xpos), apple height above table = apple centre z - table_surface_z; apple bottom = min z of apple_geom
AABB (ellipsoid, world)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np, mujoco
X = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
T39 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/raw'))
TABLE_Z = json.load(open(_ag3s_workspace_path('integration/rby1/manipulation/integration/rby1/manipulation/config/pick_place_obstacles.json')))["table_surface_z"]
m = mujoco.MjModel.from_xml_path(X); d = mujoco.MjData(m)
nm = lambda t, n: mujoco.mj_name2id(m, t, n)
B = mujoco.mjtObj.mjOBJ_BODY
site_lee = nm(mujoco.mjtObj.mjOBJ_SITE, "left_ee")
b_f1, b_f2, b_ee = nm(B, "ee_finger_l1"), nm(B, "ee_finger_l2"), nm(B, "EE_BODY_L")
b_apple, b_crate = nm(B, "apple"), nm(B, "crate")
g_apple = nm(mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")


def ellipsoid_zmin(d, g):
    R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]
    return float(d.geom_xpos[g][2] - np.sqrt(np.sum((R[2, :] * s) ** 2)))


def load(rd):
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]
    plan = [r for r in rows if r["kind"] == "planning"]
    return ctrl, plan


def path_of(rd, gt, pt):
    ctrl, plan = load(rd)
    out = []
    for c in ctrl:
        d.qpos[:] = c["qpos"]; mujoco.mj_forward(m, d)
        out.append({"t": c["t_step"], "apple": d.xpos[b_apple].tolist(), "tcp": d.site_xpos[site_lee].tolist(),
                    "finger_mid": (0.5 * (d.xpos[b_f1] + d.xpos[b_f2])).tolist(), "ee_body": d.xpos[b_ee].tolist(),
                    "apple_bottom_z": ellipsoid_zmin(d, g_apple), "carry": gt <= c["t_step"] <= pt})
    return out, plan[0]["object_poses"]


if __name__ == "__main__":
    a = json.load(open(T39 / "s23/analysis.json"))
    res = {"table_surface_z": TABLE_Z, "model_xml": X, "definitions": __doc__, "pairs": []}
    for p in a["pairs"]:
        e0 = a["runs"][p["E0_run"]]; e3 = a["runs"][p["E3b_run"]]
        if not (e0["success"] and e3["success"]):
            continue
        ent = {"episode": p["episode"], "seed": p["key"], "runs": {}}
        for cond, r in (("E0", e0), ("E3b", e3)):
            rd = pathlib.Path(r["run_dir"])
            man = json.loads((rd / "manifest.json").read_text())
            path, op0 = path_of(rd, r["grasp_t"], r["place_t"])
            car = [e for e in path if e["carry"]]
            az = np.array([e["apple"][2] for e in car]) - TABLE_Z
            bz = np.array([e["apple_bottom_z"] for e in car]) - TABLE_Z
            ent["runs"][cond] = {"run_dir": str(rd), "grasp_t": r["grasp_t"], "place_t": r["place_t"],
                                 "used_arm": man["scene"]["episode_used_arm"], "target": man["scene"]["episode_target_fruit"],
                                 "object_poses_t0": op0, "n_carry_rows": len(car),
                                 "apple_centre_above_table_mm": {"min": 1e3 * az.min(), "max": 1e3 * az.max(), "median": 1e3 * float(np.median(az))},
                                 "apple_bottom_above_table_mm": {"min": 1e3 * bz.min(), "max": 1e3 * bz.max()},
                                 "path": path}
        res["pairs"].append(ent)
        r0 = ent["runs"]["E0"]; r3 = ent["runs"]["E3b"]
        print(ent["episode"], ent["seed"], "crate", np.round(r0["object_poses_t0"]["crate"]["pos"], 3), "apple0", np.round(r0["object_poses_t0"]["apple"]["pos"], 3),
              r0["used_arm"], "E0 h", {k: round(v) for k, v in r0["apple_bottom_above_table_mm"].items()},
              "E3b h", {k: round(v) for k, v in r3["apple_bottom_above_table_mm"].items()})
    (OUT / "carry_paths_22pairs.json").write_text(json.dumps(res))
