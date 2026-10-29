"""T40: obstacle stops with clearance exactly 0.0 and no contact. For each such run: the client's robot_clearance (snapshot manager,
cap 0.25) on the last recorded control row qpos and on 400 small perturbations of it (arm + gripper joints +- 2e-3 rad / m,
uniform); every pair returning exactly 0.0 is listed with its mj_geomDistance witness gap |from - to| and an independent GJK
separation (T40 A gjk.py). usage: zero_probe.py OUT_JSON RUN_DIR CONFIG PROFILE [RUN_DIR CONFIG PROFILE ...]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import hurdle_metrics as HM, gjk
out = {"definitions": __doc__, "runs": {}}
args = sys.argv[2:]
for i in range(0, len(args), 3):
    rd, cfg, prof = pathlib.Path(args[i]), args[i + 1], args[i + 2]
    mg = HM.manager(cfg, prof); M = HM.model(); m, d = M["m"], M["d"]
    ctrl = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in ctrl if r["kind"] == "control"]
    q0 = np.array(ctrl[-1]["qpos"]); rng = np.random.default_rng(0)
    H = sorted(mg.collision_geom_ids); ft = np.zeros(6)
    zero = {}; mins = []
    for k in range(401):
        q = q0.copy()
        if k:
            q[7:29] += rng.uniform(-2e-3, 2e-3, 22)
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        rc = mg.robot_clearance(0.25); mins.append(rc)
        if rc <= 0:
            for o in H:
                for g in mg.robot_geom_ids:
                    v = mg.pair_distance(o, g, 0.25, ft)
                    if v <= 0:
                        key = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])), int(g))
                        e = zero.setdefault(str(key), {"n": 0, "values": [], "witness_gap_m": [], "gjk_sep_m": []})
                        e["n"] += 1
                        if len(e["values"]) < 5:
                            e["values"].append(float(v)); e["witness_gap_m"].append(float(np.linalg.norm(ft[:3] - ft[3:])))
                            e["gjk_sep_m"].append(float(gjk.gjk_dist(gjk.make_support(m, d, o), gjk.make_support(m, d, int(g)), d.geom_xpos[g] - d.geom_xpos[o] + 1e-9)))
    out["runs"][str(rd)] = {"t_last": ctrl[-1]["t_step"], "clearance_last_row": mins[0], "n_perturbed_le0": int(sum(x <= 0 for x in mins[1:])),
                            "min_perturbed": float(min(mins)), "median_perturbed": float(np.median(mins)), "pairs_le0": zero}
    print(rd.name, rd.parent.name, ctrl[-1]["t_step"], round(mins[0], 5), int(sum(x <= 0 for x in mins[1:])), list(zero.items())[:2], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
