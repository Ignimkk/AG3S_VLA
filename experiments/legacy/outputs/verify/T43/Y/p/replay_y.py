"""T43 Y — physics replay of recorded runs: crate contacts, apple-hand contacts, apple / crate / TCP state per control row.
(verifier, CPU only: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa; .venv-openpi-live python)

usage: replay_y.py OUT_DIR RUN_DIR [RUN_DIR ...]     -> OUT_DIR/<ARM>_<STAGE>_<name>.npz + .json

Replay = T43/W/p/hold_w.py replay() method unchanged (pi05_infer.main() reset with the run's episode / seed / obstacle config,
1000 warm-up mj_steps, then each recorded control row's applied_ctrl for spa = 1 / (CTRL_HZ * timestep) = 33 sub-steps).
fidelity: qpos_mismatch_rows = control rows whose recorded qpos differs from the replay pre-step qpos (any element != 0.0).

Geom sets (collision geoms only, contype|conaffinity != 0):
  apple  = geoms of body 'apple';  crate = geoms of body 'crate' (floor, 4 walls, handles);
  hand   = geoms of the active arm's EE_BODY_{L,R} subtree (palm) + ee_finger_{l,r}1/2 subtrees (the fingers hang off link_*_arm_6);  robot_other = every other geom under body 'base';
  obstacle = obstacle manager collision geoms; fruit_other = banana / orange / pear; table = every remaining geom (world body).
Per control row (state = after the row's sub-steps, 'post'; index 0 also has the pre-state of row 0):
  apple_pos, apple_quat, apple_linvel (free-joint qvel[0:3], world), crate_pos, crate_quat, tcp (site <side>_ee), tcp_R,
  finger_mid (bodies ee_finger_{l,r}1/2), finger_open_norm = gripper_finger_{l,r}1 qpos / RBY1_GRIPPER_OPEN (1 = open),
  grip_cmd = applied_ctrl gripper of the active arm (1 = open command, 0 = close), executed / hold flags of the record,
  apple_floor_gap = mj_geomDistance(apple_geom, crate_floor) (m, < 0 penetration), apple_crate_gap = min over crate geoms,
  hand_crate_gap = min over (hand geoms x crate geoms) (distmax 0.10 m).
  contacts per row: sub-steps with >= 1 MuJoCo contact for each pair class
     apple-crate, hand-crate, robot_other-crate, apple-hand, apple-table, apple-fruit_other, apple-obstacle, crate-obstacle
  and per class the min contact dist (m) and max normal force (N, mj_contactForce) over the row's sub-steps; pair names."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, os, pathlib, time
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hold_w as HW  # noqa: E402  (load_client, reset: snapshot T43W_SNAP, default ${AG3S_PROJECT_ROOT}-t43w)

CLASSES = ("apple-crate", "hand-crate", "robot_other-crate", "apple-hand", "apple-table", "apple-fruit_other", "apple-obstacle", "crate-obstacle")


def subtree(m, root):
    out = set()
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x == root:
                out.add(b); break
            x = int(m.body_parentid[x])
    return out


def replay(cl, rd):
    man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"])
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; rows = [r for r in rows if r["kind"] == "control"]
    st = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in st.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d, mg, act, idx = L["m"], L["d"], L.get("obstacle_manager"), L["act"], L["idx"]
    spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
    for _ in range(1000):
        mujoco.mj_step(m, d)
    side = "right" if int(ep) == 1834 else "left"
    fl = "l" if side == "left" else "r"; SD = "L" if side == "left" else "R"
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
    gb = lambda bodies: {g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in bodies}
    apple = gb({B("apple")}); crate = gb({B("crate")}); hand = gb(subtree(m, B(f"EE_BODY_{SD}")) | subtree(m, B(f"ee_finger_{fl}1")) | subtree(m, B(f"ee_finger_{fl}2")))   # fingers are children of link_*_arm_6, not of EE_BODY
    robot = gb(subtree(m, B("base"))) - hand
    obst = set(mg.collision_geom_ids) if mg is not None else set()
    fruit = gb({B("banana"), B("orange"), B("pear")})
    other = {g for g in range(m.ngeom) if coll[g]} - apple - crate - hand - robot - obst - fruit
    sets = {"apple": apple, "crate": crate, "hand": hand, "robot_other": robot, "obstacle": obst, "fruit_other": fruit, "table": other}
    cls_of = {}
    for c in CLASSES:
        a, b = c.split("-")
        cls_of[c] = (sets[a], sets[b])
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    b_apple, b_crate = B("apple"), B("crate"); f1, f2 = B(f"ee_finger_{fl}1"), B(f"ee_finger_{fl}2")
    ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "apple_free"); dofa = int(m.jnt_dofadr[ja])
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_finger_{fl}1"); qf = int(m.jnt_qposadr[jf])
    g_floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "crate_floor"); g_apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
    crate_l = sorted(crate); hand_l = sorted(hand)
    ft = np.zeros(6); f6 = np.zeros(6)
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"

    def gd(g1, g2, dmax=0.10):
        return float(mujoco.mj_geomDistance(m, d, int(g1), int(g2), dmax, ft))

    def state():
        return {"apple_pos": d.xpos[b_apple].copy(), "apple_quat": d.xquat[b_apple].copy(), "apple_linvel": d.qvel[dofa:dofa + 3].copy(),
                "crate_pos": d.xpos[b_crate].copy(), "crate_quat": d.xquat[b_crate].copy(), "tcp": d.site_xpos[site].copy(),
                "tcp_R": d.site_xmat[site].reshape(3, 3).copy(), "finger_mid": 0.5 * (d.xpos[f1] + d.xpos[f2]),
                "finger_open_norm": float(d.qpos[qf] / cl.RBY1_GRIPPER_OPEN),
                "apple_floor_gap": gd(g_apple, g_floor, 0.5), "apple_crate_gap": min(gd(g_apple, g, 0.5) for g in crate_l),
                "hand_crate_gap": min(gd(h, g) for h in hand_l for g in crate_l)}
    n = 7; mism = 0; S = [state()]; C = []
    for r in rows:
        if np.max(np.abs(np.asarray(r["qpos"]) - d.qpos)) != 0.0:
            mism += 1
        a = r["applied_ctrl"]
        for i in range(n):
            d.ctrl[act["left_a"][i]] = a["arm"][i]; d.ctrl[act["right_a"][i]] = a["arm"][n + i]
        d.ctrl[act["left_grip_a"]] = a["gripper"][0] * cl.RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = a["gripper"][1] * cl.RBY1_GRIPPER_OPEN
        cnt = {c: 0 for c in CLASSES}; mind = {c: None for c in CLASSES}; maxf = {c: 0.0 for c in CLASSES}; pairs = {c: set() for c in CLASSES}
        for s in range(spa):
            mujoco.mj_step(m, d)
            hit = set()
            for k in range(d.ncon):
                con = d.contact[k]; g1, g2 = int(con.geom1), int(con.geom2)
                for c, (A, Bs) in cls_of.items():
                    if (g1 in A and g2 in Bs) or (g2 in A and g1 in Bs):
                        hit.add(c)
                        mind[c] = float(con.dist) if mind[c] is None else min(mind[c], float(con.dist))
                        mujoco.mj_contactForce(m, d, k, f6); maxf[c] = max(maxf[c], float(abs(f6[0])))
                        pairs[c].add(tuple(sorted((gname(g1), gname(g2)))))
            for c in hit:
                cnt[c] += 1
        C.append({"t": int(r["t_step"]), "seq": int(r["chunk_seq"]), "executed": bool(r.get("executed", True)), "hold": bool(r.get("hold")),
                  "grip_cmd": float(a["gripper"][0 if side == "left" else 1]), "cnt": cnt, "mind": mind, "maxf": maxf,
                  "pairs": {c: sorted(v) for c, v in pairs.items() if v}})
        S.append(state())
    return dict(prof=prof, ep=ep, seed=seed, side=side, spa=spa, n_rows=len(rows), qpos_mismatch_rows=mism, S=S, C=C,
                sets={k: sorted(gname(g) for g in v) for k, v in sets.items() if k != "table"}, n_table_geoms=len(other))


def save(R, out_dir, key):
    S, C = R["S"], R["C"]
    arr = {k: np.array([s[k] for s in S]) for k in S[0]}
    arr["t"] = np.array([c["t"] for c in C]); arr["seq"] = np.array([c["seq"] for c in C]); arr["executed"] = np.array([c["executed"] for c in C])
    arr["hold"] = np.array([c["hold"] for c in C]); arr["grip_cmd"] = np.array([c["grip_cmd"] for c in C])
    for c in CLASSES:
        arr["cnt_" + c] = np.array([x["cnt"][c] for x in C]); arr["maxf_" + c] = np.array([x["maxf"][c] for x in C])
        arr["mind_" + c] = np.array([np.nan if x["mind"][c] is None else x["mind"][c] for x in C])
    np.savez_compressed(out_dir / f"{key}.npz", **arr)
    ev = {}
    for c in CLASSES:
        e, cur = [], None
        for x in C:
            if x["cnt"][c]:
                if cur and cur["t_last"] + 1 == x["t"]:
                    cur["t_last"] = x["t"]; cur["substeps"] += x["cnt"][c]; cur["n_hold_rows"] += x["hold"]
                    cur["min_dist_m"] = min(cur["min_dist_m"], x["mind"][c]); cur["max_force_N"] = max(cur["max_force_N"], x["maxf"][c]); cur["pairs"] |= {tuple(p) for p in x["pairs"].get(c, [])}
                else:
                    cur = {"t_first": x["t"], "t_last": x["t"], "substeps": x["cnt"][c], "n_hold_rows": int(x["hold"]), "min_dist_m": x["mind"][c],
                           "max_force_N": x["maxf"][c], "pairs": {tuple(p) for p in x["pairs"].get(c, [])}}
                    e.append(cur)
            else:
                cur = None
        for z in e:
            z["pairs"] = sorted(z["pairs"])
        ev[c] = e
    js = {"definitions": __doc__, "run_dir": R["run_dir"], "profile": R["prof"], "episode": R["ep"], "seed": R["seed"], "side": R["side"], "spa": R["spa"],
          "n_rows": R["n_rows"], "qpos_mismatch_rows": R["qpos_mismatch_rows"], "geom_sets": R["sets"], "n_table_geoms": R["n_table_geoms"],
          "contact_events": ev, "contact_rows": {c: int(sum(1 for x in C if x["cnt"][c])) for c in CLASSES}}
    (out_dir / f"{key}.json").write_text(json.dumps(js, indent=1, default=float))
    return js


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    cl = HW.load_client()
    for rd in map(pathlib.Path, sys.argv[2:]):
        key = f"{rd.parent.parent.name}_{rd.parent.name}_{rd.name}"
        if (out / f"{key}.json").exists():
            print("skip", key, flush=True); continue
        t0 = time.time(); R = replay(cl, rd); R["run_dir"] = str(rd)
        js = save(R, out, key)
        print(key, "rows", js["n_rows"], "mism", js["qpos_mismatch_rows"], "contact_rows", js["contact_rows"], f"{time.time() - t0:.0f}s", flush=True)
