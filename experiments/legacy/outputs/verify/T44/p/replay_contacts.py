"""T44 contact replay (verifier). COPY of T43/CT/p/replay_ct.py (original untouched), extended for the target-exclusion question:
who touches which fruit, and in which planning seq / server chunk.
(CPU only: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa; .venv-openpi-live python; run with nice -n 19, <= 4 workers)

usage: replay_contacts.py OUT_DIR RUN_DIR [RUN_DIR ...]      -> OUT_DIR/<ARM>_<STAGE>_<name>.npz + .json
       env T43W_SNAP = client snapshot (default: parsed from the run manifest scene.model_xml, e.g. ${AG3S_PROJECT_ROOT}-t43v11); all runs of one process must share it.

Replay = T43/W/p/hold_w.py replay() method unchanged (pi05_infer.main() reset with the run's episode / seed / obstacle config, 1000 warm-up mj_steps,
then each recorded control row's applied_ctrl for spa sub-steps). qpos_mismatch_rows = control rows whose recorded qpos differs from the replay pre-step qpos.

Geom sets (collision geoms only, contype|conaffinity != 0). All hand / finger sets are SUBTREE based, so the MJCF / URDF swap of the left finger names
(MJCF ee_finger_l2 = URDF ee_finger_l1, MJCF ee_finger_l1 = URDF ee_finger_l2) cannot misclassify:
  hand        = subtree(EE_BODY_<side>) + subtree(ee_finger_<s>1) + subtree(ee_finger_<s>2) of the ACTIVE arm (manifest episode_used_arm)
  robot_other = every other geom under body 'base' (includes the INACTIVE hand; flagged 'inactive_hand' in the geom detail)
  target      = geoms of the fruit body named in the manifest (episode_target_fruit)
  fruit_other = geoms of the other three fruit bodies (apple / banana / orange / pear minus target)
  crate, obstacle, table as in replay_ct.py.
LEGACY classes keep the replay_ct.py definitions LITERALLY (body 'apple' is the 'apple' set, banana / orange / pear is the legacy 'fruit_other'), so for an apple-target
episode every legacy number is identical to replay_ct.py; legacy_fruit_other_literal is stored as sets['fruit_other_legacy'].
NEW classes (target = manifest target fruit): hand-fruit_other, robot_other-target, robot_other-fruit_other (+ hand-target for completeness).
Per class and run: contact_chunks[class] = list over local planning seq with contact: {seq_local, server_chunk_k, plan_index, t_first, t_last, rows, substeps, min_dist_m, max_force_N}.
Seq mapping rule (T44): a control row with t_step t belongs to planning row plan_index = (#planning rows with t_step <= t) - 1; local seq = plan_index + 1 (server
numbering, 1-based) and server chunk_k = plan_index = seq - 1, i.e. chunk_k <-> local seq k+1. (In a policy-only E0 record the frames.jsonl 'seq' / 'chunk_seq' fields are t_step, not
chunk numbers; plan_index is computed from the planning rows' t_step in both cases.) A contact seen in rows of plan_index p is caused by executing the chunk planned at planning row p.
geom_detail[class] = list of {A_geom, A_body, A_flags, B_geom, B_body, B_flags, substeps, substeps_dist_le_0, min_dist_m, max_force_N, t_first, t_last, rows, chunks}.
server_chunk_alignment (best effort): the server_constraints block of this run, offset, and a target-centroid check against the recorded object pose."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, os, pathlib, time, re, glob
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco


def _snap_from_manifest(rd):
    try:
        mx = json.load(open(pathlib.Path(rd) / "manifest.json"))["scene"]["model_xml"]
        return mx.split("/pi05_TO_hybrid")[0]
    except Exception:
        return "${AG3S_PROJECT_ROOT}-t43w"


if "T43W_SNAP" not in os.environ and len(sys.argv) > 2:
    os.environ["T43W_SNAP"] = _snap_from_manifest(sys.argv[2])
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hold_w as HW  # noqa: E402  (load_client, reset)

LEGACY = ("apple-crate", "hand-crate", "robot_other-crate", "apple-hand", "apple-table", "apple-fruit_other", "apple-obstacle", "crate-obstacle", "hand-obstacle", "robot_other-obstacle")
NEW = ("hand-fruit_other", "robot_other-target", "robot_other-fruit_other", "hand-target")
CLASSES = LEGACY + NEW
FRUIT_BODIES = ("apple", "banana", "orange", "pear")
URDF_ALIAS = {"ee_finger_l1": "ee_finger_l2", "ee_finger_l2": "ee_finger_l1"}   # left hand only: MJCF name -> URDF link of the same physical finger


def subtree(m, root):
    out = set()
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x == root:
                out.add(b); break
            x = int(m.body_parentid[x])
    return out


def find_server_block(rd, n_planning):
    """best effort: locate this client run's chunk block in the server_constraints record. returns dict (never raises)."""
    rd = pathlib.Path(rd)
    try:
        st = (rd.parent / f"{rd.name}.start").read_text().splitlines()
        t_start = st[0].strip(); port = next(l.split("=", 1)[1] for l in st if l.startswith("port="))
        servers = sorted(glob.glob(str(rd.parent.parent / f"server_{port}_*")))
        from datetime import datetime, timezone
        ts = lambda s: datetime.fromisoformat(s).astimezone(timezone.utc)
        t0 = ts(t_start)
        best = None
        for sdir in servers:
            m = re.search(r"_(\d{8}T\d{6})$", sdir)
            if m and datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) <= t0:
                best = sdir
        if best is None:
            return {"found": False, "reason": "no server dir before the client start"}
        peers = []
        for p in glob.glob(str(rd.parent.parent / "*" / "*.start")) + glob.glob(str(rd.parent / "*.start")):
            lines = pathlib.Path(p).read_text().splitlines()
            if any(l.strip() == f"port={port}" for l in lines):
                peers.append((ts(lines[0].strip()), pathlib.Path(p).name[:-6]))
        m = re.search(r"_(\d{8}T\d{6})$", best); s0 = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        later = [d for d in glob.glob(str(rd.parent.parent / f"server_{port}_*")) if d > best]
        s1 = None
        if later:
            m2 = re.search(r"_(\d{8}T\d{6})$", min(later)); s1 = datetime.strptime(m2.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        peers = sorted(x for x in set(peers) if x[0] >= s0 and (s1 is None or x[0] < s1))
        order = [n for _, n in peers]
        idx = order.index(rd.name)
        runs = sorted(glob.glob(best + "/server_constraints/run_*"))
        chunks = sorted(glob.glob(runs[-1] + "/chunk_*.npz"))
        ts_ = [int(np.load(c, allow_pickle=True)["t_step"]) for c in chunks]
        starts = [i for i, t in enumerate(ts_) if t == 1]
        blocks = [(a, (starts[i + 1] if i + 1 < len(starts) else len(chunks))) for i, a in enumerate(starts)]
        return {"found": True, "server_dir": best, "chunk_dir": runs[-1], "peer_runs_in_order": order, "run_rank": idx, "n_blocks": len(blocks),
                "block": blocks[idx] if idx < len(blocks) else None, "offset": blocks[idx][0] if idx < len(blocks) else None,
                "block_len": (blocks[idx][1] - blocks[idx][0]) if idx < len(blocks) else None, "n_planning_rows": n_planning}
    except Exception as e:   # noqa: BLE001
        return {"found": False, "reason": f"{type(e).__name__}: {e}"}


def replay(cl, rd):
    man = json.load(open(rd / "manifest.json")); sc = man["scene"]; prof = sc.get("obstacle_profile"); ep = sc["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"])
    allrows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    rows = [r for r in allrows if r["kind"] == "control"]
    pl = [r for r in allrows if r["kind"] == "planning"]; pl_t = np.array([r["t_step"] for r in pl])
    st = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in st.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None) if st.exists() else None
    L = HW.reset(cl, ep, seed, prof, cfg); m, d, mg, act, idx = L["m"], L["d"], L.get("obstacle_manager"), L["act"], L["idx"]
    spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
    for _ in range(1000):
        mujoco.mj_step(m, d)
    side = sc.get("episode_used_arm") or ("right" if int(ep) == 1834 else "left")
    tfruit = sc.get("episode_target_fruit") or "apple"
    fl = "l" if side == "left" else "r"; SD = "L" if side == "left" else "R"
    ofl = "r" if side == "left" else "l"; OSD = "R" if side == "left" else "L"
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
    gb = lambda bodies: {g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in bodies}
    apple = gb({B("apple")}); crate = gb({B("crate")})
    hand = gb(subtree(m, B(f"EE_BODY_{SD}")) | subtree(m, B(f"ee_finger_{fl}1")) | subtree(m, B(f"ee_finger_{fl}2")))   # subtree based (finger bodies hang off link_*_arm_6)
    ihand = gb(subtree(m, B(f"EE_BODY_{OSD}")) | subtree(m, B(f"ee_finger_{ofl}1")) | subtree(m, B(f"ee_finger_{ofl}2")))
    robot = gb(subtree(m, B("base"))) - hand
    obst = set(mg.collision_geom_ids) if mg is not None else set()
    fruit_legacy = gb({B("banana"), B("orange"), B("pear")})
    target = gb({B(tfruit)}); fruit_other = gb({B(f) for f in FRUIT_BODIES if f != tfruit})
    other = {g for g in range(m.ngeom) if coll[g]} - apple - crate - hand - robot - obst - fruit_legacy
    sets = {"apple": apple, "crate": crate, "hand": hand, "robot_other": robot, "obstacle": obst, "fruit_other": fruit_legacy, "table": other,
            "target": target, "fruit_other_t": fruit_other}
    assert hand and target and fruit_other and robot, {k: len(v) for k, v in sets.items()}
    assert not (hand & robot) and not (target & fruit_other)
    cls_of = {}
    for c in CLASSES:
        a, b = c.split("-")
        if c in NEW:
            a = {"fruit_other": "fruit_other_t"}.get(a, a); b = {"fruit_other": "fruit_other_t"}.get(b, b)
        cls_of[c] = (sets[a], sets[b])
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    b_apple, b_crate, b_tgt = B("apple"), B("crate"), B(tfruit); f1, f2 = B(f"ee_finger_{fl}1"), B(f"ee_finger_{fl}2")
    ja = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "apple_free"); dofa = int(m.jnt_dofadr[ja])
    jf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_finger_{fl}1"); qf = int(m.jnt_qposadr[jf])
    g_floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "crate_floor"); g_apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
    crate_l = sorted(crate); hand_l = sorted(hand)
    ft = np.zeros(6); f6 = np.zeros(6)
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
    bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(g)])) or f"body#{int(m.geom_bodyid[int(g)])}"

    def flags(g):
        f = [k for k, v in (("hand", hand), ("inactive_hand", ihand), ("robot_other", robot), ("target", target), ("fruit_other", fruit_other), ("crate", crate), ("obstacle", obst)) if g in v]
        b = bname(g)
        if side == "left" and g in hand and b in URDF_ALIAS:
            f.append(f"urdf_link={URDF_ALIAS[b]}")
        return f

    def gd(g1, g2, dmax=0.10):
        return float(mujoco.mj_geomDistance(m, d, int(g1), int(g2), dmax, ft))

    def state():
        return {"apple_pos": d.xpos[b_apple].copy(), "apple_quat": d.xquat[b_apple].copy(), "apple_linvel": d.qvel[dofa:dofa + 3].copy(),
                "crate_pos": d.xpos[b_crate].copy(), "crate_quat": d.xquat[b_crate].copy(), "tcp": d.site_xpos[site].copy(),
                "tcp_R": d.site_xmat[site].reshape(3, 3).copy(), "finger_mid": 0.5 * (d.xpos[f1] + d.xpos[f2]),
                "finger_open_norm": float(d.qpos[qf] / cl.RBY1_GRIPPER_OPEN),
                "apple_floor_gap": gd(g_apple, g_floor, 0.5), "apple_crate_gap": min(gd(g_apple, g, 0.5) for g in crate_l),
                "hand_crate_gap": min(gd(h, g) for h in hand_l for g in crate_l), "target_pos": d.xpos[b_tgt].copy()}
    n = 7; mism = 0; S = [state()]; C = []; OB = []
    gbody = bname
    ob_kind = {}
    for g in apple: ob_kind[g] = "apple"
    for g in hand: ob_kind[g] = "hand"
    for g in robot: ob_kind[g] = "robot_other"
    detail = {c: {} for c in CLASSES}
    for r in rows:
        if np.max(np.abs(np.asarray(r["qpos"]) - d.qpos)) != 0.0:
            mism += 1
        a = r["applied_ctrl"]
        for i in range(n):
            d.ctrl[act["left_a"][i]] = a["arm"][i]; d.ctrl[act["right_a"][i]] = a["arm"][n + i]
        d.ctrl[act["left_grip_a"]] = a["gripper"][0] * cl.RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = a["gripper"][1] * cl.RBY1_GRIPPER_OPEN
        pidx = int(np.searchsorted(pl_t, int(r["t_step"]), side="right") - 1)
        obrow = {}
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
                        mujoco.mj_contactForce(m, d, k, f6); fN = float(abs(f6[0])); maxf[c] = max(maxf[c], fN)
                        pairs[c].add(tuple(sorted((gname(g1), gname(g2)))))
                        ga, gb_ = (g1, g2) if g1 in A else (g2, g1)
                        e = detail[c].setdefault((ga, gb_), dict(A_geom=gname(ga), A_body=bname(ga), A_flags=flags(ga), B_geom=gname(gb_), B_body=bname(gb_), B_flags=flags(gb_),
                                                               substeps=0, substeps_dist_le_0=0, min_dist_m=float(con.dist), max_force_N=0.0, t_first=int(r["t_step"]), t_last=int(r["t_step"]),
                                                               _rows=set(), _chunks=set()))
                        e["substeps"] += 1; e["substeps_dist_le_0"] += int(con.dist <= 0); e["min_dist_m"] = min(e["min_dist_m"], float(con.dist)); e["max_force_N"] = max(e["max_force_N"], fN)
                        e["t_last"] = int(r["t_step"]); e["_rows"].add(int(r["t_step"])); e["_chunks"].add(pidx)
            for k in range(d.ncon):
                con = d.contact[k]; g1, g2 = int(con.geom1), int(con.geom2)
                if g1 in obst and g2 in ob_kind or g2 in obst and g1 in ob_kind:
                    gr = g2 if g1 in obst else g1
                    mujoco.mj_contactForce(m, d, k, f6)
                    e = obrow.setdefault(gr, [gname(gr), gbody(gr), ob_kind[gr], 0, float(con.dist), 0.0, 0])
                    e[3] += 1; e[4] = min(e[4], float(con.dist)); e[5] = max(e[5], float(abs(f6[0]))); e[6] += int(con.dist > 0)
            for c in hit:
                cnt[c] += 1
        C.append({"t": int(r["t_step"]), "seq": int(r["chunk_seq"]), "plan_index": pidx, "executed": bool(r.get("executed", True)), "hold": bool(r.get("hold")),
                  "grip_cmd": float(a["gripper"][0 if side == "left" else 1]), "cnt": cnt, "mind": mind, "maxf": maxf,
                  "pairs": {c: sorted(v) for c, v in pairs.items() if v}})
        S.append(state())
        if obrow:
            OB.append({"t": int(r["t_step"]), "geoms": {str(g): v for g, v in obrow.items()}})
    gdet = {}
    for c, dd in detail.items():
        out = []
        for e in dd.values():
            e = dict(e); e["rows"] = len(e.pop("_rows")); e["chunks"] = sorted(e.pop("_chunks")); out.append(e)
        gdet[c] = sorted(out, key=lambda x: (-x["substeps"], x["A_geom"], x["B_geom"]))
    return dict(OB=OB, obst_geoms={str(g): gname(g) for g in obst}, prof=prof, ep=ep, seed=seed, side=side, target_fruit=tfruit, spa=spa, n_rows=len(rows), n_planning=len(pl),
                qpos_mismatch_rows=mism, S=S, C=C, geom_detail=gdet,
                sets={k: sorted(gname(g) for g in v) for k, v in sets.items() if k != "table"} | {"fruit_other_legacy": sorted(gname(g) for g in fruit_legacy), "inactive_hand": sorted(gname(g) for g in ihand)},
                n_table_geoms=len(other), planning_t=[int(x) for x in pl_t], planning_target_pos=[pl_.get("object_poses", {}).get(tfruit, {}).get("pos") for pl_ in pl])


def chunk_rows(C, CL, offset):
    """per class: list over plan_index with contact -> {seq_local, server_chunk_k, ...}"""
    out = {}
    for c in CL:
        per = {}
        for x in C:
            if not x["cnt"][c]:
                continue
            e = per.setdefault(x["plan_index"], dict(plan_index=x["plan_index"], seq_local=x["plan_index"] + 1, server_chunk_k=x["plan_index"], server_chunk_file_index=(None if offset is None else x["plan_index"] + offset),
                                                     t_first=x["t"], t_last=x["t"], rows=0, substeps=0, min_dist_m=x["mind"][c], max_force_N=0.0))
            e["t_last"] = x["t"]; e["rows"] += 1; e["substeps"] += x["cnt"][c]; e["min_dist_m"] = min(e["min_dist_m"], x["mind"][c]); e["max_force_N"] = max(e["max_force_N"], x["maxf"][c])
        out[c] = [per[k] for k in sorted(per)]
    return out


def align_check(R, blk):
    """target centroid of the server chunk (npz) vs recorded object pose at the planning row (best effort)."""
    if not blk.get("found") or blk.get("offset") is None:
        return None
    res = []
    for p in (0, min(10, R["n_planning"] - 1), R["n_planning"] - 1):
        f = pathlib.Path(blk["chunk_dir"]) / f"chunk_{p + blk['offset']:05d}.npz"
        if not f.exists():
            res.append(dict(plan_index=p, file=str(f), exists=False)); continue
        z = np.load(f, allow_pickle=True)
        tp = R["planning_target_pos"][p]
        if "target_centroid" not in z.files:
            res.append(dict(plan_index=p, file=f.name, t_step_in_npz=int(z["t_step"]), chunk_index_in_npz=int(z["chunk_index"]), seq_matches=bool(int(z["t_step"]) == p + 1), target_centroid=None, recorded_target_pos=tp, dist_m=None)); continue
        res.append(dict(plan_index=p, file=f.name, t_step_in_npz=int(z["t_step"]), chunk_index_in_npz=int(z["chunk_index"]), seq_matches=bool(int(z["t_step"]) == p + 1),
                        target_centroid=[float(x) for x in z["target_centroid"]], recorded_target_pos=tp,
                        dist_m=(None if tp is None else float(np.linalg.norm(np.array(z["target_centroid"]) - np.array(tp))))))
    return res


def save(R, out_dir, key):
    S, C = R["S"], R["C"]
    arr = {k: np.array([s[k] for s in S]) for k in S[0]}
    arr["t"] = np.array([c["t"] for c in C]); arr["seq"] = np.array([c["seq"] for c in C]); arr["plan_index"] = np.array([c["plan_index"] for c in C]); arr["executed"] = np.array([c["executed"] for c in C])
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
    blk = find_server_block(R["run_dir"], R["n_planning"]) if os.environ.get("T44_SERVER_ALIGN", "1") == "1" else {"found": False, "reason": "disabled"}
    offset = blk.get("offset") if blk.get("found") else None
    js = {"definitions": __doc__, "run_dir": R["run_dir"], "profile": R["prof"], "episode": R["ep"], "seed": R["seed"], "side": R["side"], "target_fruit": R["target_fruit"], "spa": R["spa"],
          "n_rows": R["n_rows"], "n_planning_rows": R["n_planning"], "qpos_mismatch_rows": R["qpos_mismatch_rows"], "geom_sets": R["sets"], "n_table_geoms": R["n_table_geoms"],
          "obstacle_contact_rows": R["OB"], "obstacle_geoms": R["obst_geoms"], "snapshot": str(HW.HY), "contact_events": ev,
          "contact_rows": {c: int(sum(1 for x in C if x["cnt"][c])) for c in CLASSES},
          "contact_substeps": {c: int(sum(x["cnt"][c] for x in C)) for c in CLASSES},
          "contact_chunks": chunk_rows(C, CLASSES, offset), "geom_detail": R["geom_detail"],
          "seq_rule": "plan_index = (#planning rows with t_step <= t) - 1 ; seq_local = plan_index + 1 ; server chunk_k = plan_index (chunk_k <-> seq k+1)",
          "server_chunk_alignment": {"block": blk, "checks": align_check(R, blk)}}
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
