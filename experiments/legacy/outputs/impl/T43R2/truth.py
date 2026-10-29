"""Truth for flagged held chunks: MuJoCo held fruit (moved rigidly with the EE body) vs scene geoms (crate, table,
other fruits, hurdle) along the recorded refined plan."""
import sys, json, glob, pathlib, numpy as np, mujoco
from multiprocessing import Pool
OUT = pathlib.Path("/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/r2")
_M = {}

def model(xml):
    if xml not in _M:
        m = mujoco.MjModel.from_xml_path(xml); _M[xml] = (m, mujoco.MjData(m))
    return _M[xml]

def bid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

def one(path):
    d = json.load(open(path)); run = pathlib.Path(d["run"])
    man = json.loads((run / "manifest.json").read_text())
    xml = man["scene"]["model_xml"]
    argv = (man.get("extra") or {}).get("argv") or []
    obstacle = None
    if "--obstacle-config" in argv:
        xml = xml.replace("model_transport.xml", "model_transport_pick_place_obstacles.xml")
        obstacle = (argv[argv.index("--obstacle-config") + 1], argv[argv.index("--obstacle-profile") + 1])
    m, dd = model(xml)
    mgr = None
    if obstacle:
        sys.path.insert(0, str(pathlib.Path(xml).parents[4] / "rby1_manipulation/src"))
        from rby1_manipulation.simulation import pick_place_obstacles as PPO
        mgr = PPO.PickPlaceObstacleManager(m, dd, PPO.load_pick_place_obstacle_config(obstacle[0])); mgr.activate(obstacle[1])
    fruit = man["scene"].get("episode_target_fruit") or "apple"
    fb = bid(m, fruit)
    fgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and (m.geom_contype[g] or m.geom_conaffinity[g])]
    jnt = [j for j in range(m.njnt) if m.jnt_bodyid[j] == fb][0]; adr = int(m.jnt_qposadr[jnt])
    others = []
    names = ["crate", "table"] + [f for f in ("apple", "orange", "pear", "banana") if f != fruit]
    for n in names:
        b = bid(m, n)
        if b < 0: continue
        others += [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if mgr is not None:
        others += [int(g) for g in mgr.collision_geom_ids]
    J = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
    rows = [json.loads(l) for l in (run / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    idx = json.load(open(sys.argv[1]))[str(run)]
    out = []
    for A in d["attaches"]:
        if "error" in A: continue
        ee = bid(m, "EE_BODY_L" if A["hand"] == "left" else "EE_BODY_R")
        for c in A["chunks"]:
            if not (min(c["new_mm"]) < 0 <= min(c["old_mm"])) and not c["last_held"] and not (min(c["new_mm"]) < min(c["old_mm"]) - 0.5 and min(c["new_mm"]) < 0): continue
            q0 = np.asarray(plan[c["seq"]]["qpos"], float)
            qi = np.asarray(plan[min(plan)]["qpos"], float); dd.qpos[:] = qi; mujoco.mj_kinematics(m, dd); z_init = float(dd.xpos[fb][2])
            dd.qpos[:] = q0; mujoco.mj_kinematics(m, dd)
            in_hand = bool(dd.xpos[fb][2] - z_init > 0.02 and np.linalg.norm(dd.xpos[fb] - dd.xpos[ee]) < 0.15)
            dd.qpos[:] = q0; mujoco.mj_kinematics(m, dd)
            Re, pe = dd.xmat[ee].reshape(3, 3).copy(), dd.xpos[ee].copy()
            Ra, pa = dd.xmat[fb].reshape(3, 3).copy(), dd.xpos[fb].copy()
            Rrel, prel = Re.T @ Ra, Re.T @ (pa - pe)
            with np.load(idx["chunks"][str(c["seq"])]["npz"], allow_pickle=True) as z:
                fin = np.asarray(z["refined_chunk"], float)
            true = []; who = []
            for k in range(8):
                q = q0.copy()
                for i in range(7):
                    q[J[f"left_arm_{i}"]] = fin[k, i]; q[J[f"right_arm_{i}"]] = fin[k, 8 + i]
                dd.qpos[:] = q; mujoco.mj_kinematics(m, dd)
                Re2, pe2 = dd.xmat[ee].reshape(3, 3).copy(), dd.xpos[ee].copy()
                q[adr:adr + 3] = pe2 + Re2 @ prel; qq = np.zeros(4); mujoco.mju_mat2Quat(qq, (Re2 @ Rrel).reshape(-1)); q[adr + 3:adr + 7] = qq
                dd.qpos[:] = q; mujoco.mj_kinematics(m, dd)
                best, bn = 1.0, None; ft = np.zeros(6)
                for g1 in fgeoms:
                    for g2 in others:
                        dist = mujoco.mj_geomDistance(m, dd, g1, g2, 0.2, ft)
                        if dist < best: best, bn = dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2)
                true.append(1e3 * best); who.append(bn)
            kmin = int(np.argmin(c["new_mm"]))
            out.append(dict(run=str(run).split("verify/")[1], fruit=fruit, hand=A["hand"], seq=c["seq"], last=c["last_held"], gap=c["gap_mm"],
                            radius=A["body"]["body_radius_mm"], merged=A["body"].get("mode")=="segments", n=len(A["body"]["spheres"]),
                            old=min(c["old_mm"]), new=min(c["new_mm"]), k=kmin, true_at_k=true[kmin], who_at_k=who[kmin], true_min=min(true),
                            new_k=c["new_mm"][kmin], in_hand=in_hand, near=c["near"], true_all=true, new_all=c["new_mm"], old_all=c["old_mm"], mode=A["body"].get("mode")))
    return out

if __name__ == "__main__":
    files = sorted(glob.glob(str(OUT / "scan4" / (sys.argv[2] + "*.json"))))
    res = []
    with Pool(int(sys.argv[3])) as p:
        for r in p.imap_unordered(one, files): res.extend(r)
    (OUT / f"truth_{sys.argv[2].replace('/','_')}.json").write_text(json.dumps(res))
    for r in sorted(res, key=lambda r: (r["run"], r["seq"])):
        print(f"{'H' if r['in_hand'] else '-'} {r['mode'][:3]} {r['run'][-24:]:24s} {r['fruit']:6s} {r['hand'][0]} seq {r['seq']:3d} {'L' if r['last'] else ' '} R {r['radius']:5.1f} n{r['n']} {'S' if r['merged'] else ' '} gap {r['gap'] if r['gap'] is not None else float('nan'):6.1f} old {r['old']:6.1f} new {r['new']:6.1f} | true@k {r['true_at_k']:6.1f} ({r['who_at_k']}) true_min {r['true_min']:6.1f}")
