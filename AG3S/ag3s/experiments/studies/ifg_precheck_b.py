"""IFG-PRECHECK-B — actual robot contacts in the IFG-APPLE-SPLIT rollouts, the field value at those contacts, and checked vs unchecked links.
CPU only, read only. Two stages, two interpreters:

  # stage 1: physics replay of every client rollout (client = run snapshot), all robot-vs-world contacts at every MuJoCo sub-step
  source <RUN>/verify/IFG-APPLE-SPLIT/p/env.sh   (or set the same variables)
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=1 CL_SNAP=<RUN>/snapshot_split PYTHONPATH=$SNAP_PYTHONPATH \
      .venv-openpi-live/bin/python -u AG3S/ag3s/experiments/studies/ifg_precheck_b.py replay --kit <RUN>/verify/IFG-APPLE-SPLIT --out <OUT>/replay RUN_DIR...
  # stage 2: phases, field vs GT at the contacts, link classes, figures, verify.json
  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_precheck_b analyze \
      --run <RUN> --out <OUT> --fig AG3S/ag3s/docs/figures/ifg-precheck --verify AG3S/ag3s/docs/handoff/IFG-PRECHECK-B.verify.json

The definitions below were written before any contact was counted.
"""
import argparse, glob, json, os, pathlib, sys, time

import numpy as np

SCRIPT = "AG3S/ag3s/experiments/studies/ifg_precheck_b.py"

DEFINITIONS = {
    "data": "IFG-APPLE-SPLIT arm SPL, 24 apple-target scenes (16 clear, 8 with the basket-side pillar 'pick_obstacle_crate_post_0_collision'), 600 control rows "
            "(15 Hz) x 33 MuJoCo sub-steps (2 ms) per scene. Inputs read only.",
    "actual_trajectory": "physics replay of the recorded client rollout: the run snapshot client (p/hold_w.py reset: pi05_infer.main() scene with the run's "
                         "episode / seed / obstacle profile, 1000 warm-up mj_steps) then each recorded control row's applied_ctrl for 33 sub-steps "
                         "(the method of p/replay_contacts.py). fidelity check: qpos_mismatch_rows = control rows whose recorded (measured) qpos differs "
                         "from the replay pre-step qpos in any element; must be 0 for the scene to be used. Contacts are MuJoCo d.contact of the "
                         "executed physics, not of the commanded chunk.",
    "contact": "a MuJoCo contact (d.contact entry, exclude == 0) between a robot collision geom (subtree of body 'base') and a non-robot collision geom "
               "(contype|conaffinity != 0) with dist <= 0 (penetration >= 0 mm). Secondary threshold: penetration >= 1 mm (dist <= -0.001 m). "
               "Entries with dist > 0 (inside the geom margin, not touching) are not contacts; counted separately as 'margin_only'.",
    "contact_step": "a control row (one 15 Hz step, 33 sub-steps) with at least one contact of that (link, partner) class in any of its sub-steps; "
                    "substeps = number of sub-steps with such a contact.",
    "partner_classes": {"target": "geoms of the manifest target fruit body (apple)",
                        "neighbour_fruit": "geoms of the other fruit bodies (banana / orange / pear); counted separately from the target",
                        "crate": "geoms of body 'crate' (basket floor, walls, handles)",
                        "pillar": "obstacle-manager collision geoms (pick_obstacle_crate_post_0_collision, cylinder r 15 mm, h 180 mm beside the basket)",
                        "table": "geoms of body 'table'",
                        "other": "every other non-robot collision geom (floor plane, office walls, shelf)"},
    "excluded_by_rule": {
        "intended_target_contact": "contact between the ACTIVE hand (MuJoCo bodies EE_BODY_<side>, ee_finger_<s>1, ee_finger_<s>2 of manifest "
                                   "episode_used_arm) and the target fruit. Not counted as a collision; its count is reported as 'intended'. "
                                   "Target contacts of any other robot link (inactive hand, wrist, arm) ARE counted (partner 'target').",
        "support_contact": "robot bodies base / wheel_l / wheel_r against the floor (partner 'other', body 'world'): wheel support, not counted.",
    },
    "link_classes": {"checked": "the 6 URDF links that carry TrajOpt collision spheres (summary sphere_link_names: ee_left, ee_right, ee_finger_l1, "
                                "ee_finger_l2, ee_finger_r1, ee_finger_r2; 270 spheres) = MuJoCo bodies EE_BODY_L, EE_BODY_R, ee_finger_l1, ee_finger_l2, "
                                "ee_finger_r1, ee_finger_r2 (left finger names swapped between MJCF and URDF; the class is body-set based so the swap "
                                "cannot misclassify)",
                     "unchecked_wrist": "link_<side>_arm_6 and FT_SENSOR_<S> (wrist / flange, no TrajOpt sphere)",
                     "unchecked_arm": "link_<side>_arm_0 .. link_<side>_arm_5 (shoulder, upper arm, elbow, forearm)",
                     "unchecked_body": "torso, head, base, wheels"},
    "phase": "server request k = plan_index of the control row (plan_index = #planning rows with t_step <= t - 1; the chunk executed in that row "
             "was planned at request k); phase = summary grasp.state of server chunk k (searching / latched / closing / held / placed).",
    "field_vs_gt": {
        "which_contacts": "every counted contact row (link, partner geom) pair, at its deepest sub-step in that row (min dist).",
        "query_points": "p_s = c + s * n_out for s in {0, 5, 10, 20} mm; c = MuJoCo contact position (midway between the two surfaces), n_out = unit "
                        "contact normal oriented from the partner geom toward the robot geom.",
        "field": "server constraint record chunk k (the request whose plan was executing), offline `fixed_input_ablation.build_field(z, outside_distance=1.0)`; "
                 "layers evaluated separately: coarse = layers[0] (20 mm voxel, whole workspace), fine = layers[1] (5 mm voxel window; NaN where the point "
                 "is outside the window's interpolation box), main = composed min(coarse, fine) = the value unauthorised TrajOpt rows read, "
                 "free = target-free fine layer (where present). answer_tier = which layer answered main.",
        "gt": "GT_main(p) = min signed distance over all non-robot collision geoms at the deepest sub-step's MuJoCo state (qpos snapshot), the target left "
              "out in 'held' rows where the active hand touches the target in that row (the server frees the held object on purpose); GT_free = same "
              "without the target; GT_partner = to the contacted partner body's geoms. Boxes / planes / cylinders analytic, fruit meshes by mj_geomDistance "
              "to a 0.1 mm probe sphere (`ifg_audit_lte.Scene`); the runtime pillar (absent from model_transport.xml) analytic from its replay pose.",
        "error": "field - GT (mm); > 0 = field says farther than reality.",
        "layer_gt_pairing": "coarse, fine, main vs GT_main; free vs GT_free.",
    },
}


# =============================================================================================== stage 1: replay

def _subtree(m, root):
    out = set()
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x == root:
                out.add(b); break
            x = int(m.body_parentid[x])
    return out


CHECKED_BODIES = ("EE_BODY_L", "EE_BODY_R", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
FRUITS = ("apple", "banana", "orange", "pear")


def replay_one(cl, HW, rd):
    import mujoco
    man = json.load(open(rd / "manifest.json")); sc = man["scene"]; prof = sc.get("obstacle_profile"); ep = sc["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"])
    allrows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    rows = [r for r in allrows if r["kind"] == "control"]
    pl = [r for r in allrows if r["kind"] == "planning"]; pl_t = np.array([r["t_step"] for r in pl])
    st = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in st.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None) if st.exists() else None
    L = HW.reset(cl, ep, seed, prof, cfg); m, d, mg, act = L["m"], L["d"], L.get("obstacle_manager"), L["act"]
    spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
    for _ in range(1000):
        mujoco.mj_step(m, d)
    side = sc.get("episode_used_arm"); tfruit = sc.get("episode_target_fruit") or "apple"
    s = "l" if side == "left" else "r"; S = "L" if side == "left" else "R"
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"  # noqa: E731
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or f"body#{int(b)}"  # noqa: E731
    coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    robot_b = _subtree(m, B("base"))
    is_robot = np.array([int(m.geom_bodyid[g]) in robot_b for g in range(m.ngeom)])
    obst = set(int(g) for g in mg.collision_geom_ids) if mg is not None else set()
    active_hand_b = {B(f"EE_BODY_{S}"), B(f"ee_finger_{s}1"), B(f"ee_finger_{s}2")}
    gcls = {}
    for g in range(m.ngeom):
        if not coll[g] or is_robot[g]:
            continue
        bn = bname(m.geom_bodyid[g])
        gcls[g] = ("pillar" if g in obst else "target" if bn == tfruit else "neighbour_fruit" if bn in FRUITS else "crate" if bn == "crate"
                   else "table" if bn == "table" else "other")
    support_b = {B("base"), B("wheel_l"), B("wheel_r")}
    f6 = np.zeros(6)
    # records
    REC = {k: [] for k in ("row", "t", "sub", "rg", "pg", "dist", "pos", "nout", "force", "exclude")}
    best = {}      # (row, rg_body, pg) -> deepest (dist, qpos, pos, nout)
    mism = 0; intended_rows = set()
    for ri, r in enumerate(rows):
        if np.max(np.abs(np.asarray(r["qpos"]) - d.qpos[:len(r["qpos"])])) != 0.0 or len(r["qpos"]) != m.nq:
            mism += 1
        a = r["applied_ctrl"]
        for i in range(7):
            d.ctrl[act["left_a"][i]] = a["arm"][i]; d.ctrl[act["right_a"][i]] = a["arm"][7 + i]
        d.ctrl[act["left_grip_a"]] = a["gripper"][0] * cl.RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = a["gripper"][1] * cl.RBY1_GRIPPER_OPEN
        for sub in range(spa):
            mujoco.mj_step(m, d)
            for k in range(d.ncon):
                con = d.contact[k]; g1, g2 = int(con.geom1), int(con.geom2)
                r1, r2 = is_robot[g1], is_robot[g2]
                if r1 == r2:
                    continue
                rg, pg = (g1, g2) if r1 else (g2, g1)
                if pg not in gcls:
                    continue
                rb = int(m.geom_bodyid[rg])
                if rb in support_b and gcls[pg] == "other" and bname(m.geom_bodyid[pg]) == "world":
                    continue
                n = np.array(con.frame[:3], np.float64)   # normal points geom1 -> geom2
                nout = n if g1 == pg else -n               # partner -> robot
                mujoco.mj_contactForce(m, d, k, f6)
                REC["row"].append(ri); REC["t"].append(int(r["t_step"])); REC["sub"].append(sub); REC["rg"].append(rg); REC["pg"].append(pg)
                REC["dist"].append(float(con.dist)); REC["pos"].append(np.array(con.pos, np.float64)); REC["nout"].append(nout)
                REC["force"].append(float(abs(f6[0]))); REC["exclude"].append(int(con.exclude))
                if gcls[pg] == "target" and rb in active_hand_b and con.dist <= 0:
                    intended_rows.add(ri)
                key = (ri, rb, pg)
                if con.dist <= 0 and (key not in best or con.dist < best[key][0]):
                    best[key] = (float(con.dist), d.qpos.copy(), np.array(con.pos, np.float64), nout.copy(), rg, sub)
    # obstacle poses (static) for analytic GT in stage 2
    obst_pose = {str(g): {"name": gname(g), "type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(), "xpos": d.geom_xpos[g].tolist(),
                          "xmat": d.geom_xmat[g].tolist()} for g in sorted(obst)}
    keys = sorted(best)
    arr = {k: np.array(v) for k, v in REC.items()}
    arr["best_row"] = np.array([k[0] for k in keys], np.int64); arr["best_rbody"] = np.array([k[1] for k in keys], np.int64)
    arr["best_pg"] = np.array([k[2] for k in keys], np.int64)
    arr["best_dist"] = np.array([best[k][0] for k in keys]); arr["best_qpos"] = np.array([best[k][1] for k in keys]).reshape(len(keys), m.nq)
    arr["best_pos"] = np.array([best[k][2] for k in keys]).reshape(-1, 3); arr["best_nout"] = np.array([best[k][3] for k in keys]).reshape(-1, 3)
    arr["best_rg"] = np.array([best[k][4] for k in keys], np.int64); arr["best_sub"] = np.array([best[k][5] for k in keys], np.int64)
    arr["row_t"] = np.array([int(r["t_step"]) for r in rows]); arr["row_plan_index"] = np.searchsorted(pl_t, arr["row_t"], side="right") - 1
    arr["row_hold"] = np.array([bool(r.get("hold")) for r in rows]); arr["intended_rows"] = np.array(sorted(intended_rows), np.int64)
    meta = {"episode": int(ep), "seed": seed, "profile": prof, "side": side, "target_fruit": tfruit, "spa": spa, "n_rows": len(rows), "n_planning": len(pl),
            "nq": int(m.nq), "qpos_mismatch_rows": mism, "obstacle_geoms": obst_pose,
            "geom_name": {str(g): gname(g) for g in set(arr["rg"].tolist()) | set(arr["pg"].tolist()) | set(arr["best_rg"].tolist())},
            "geom_body": {str(g): bname(m.geom_bodyid[g]) for g in set(arr["rg"].tolist()) | set(arr["pg"].tolist()) | set(arr["best_rg"].tolist())},
            "body_name": {str(b): bname(b) for b in set(arr["best_rbody"].tolist()) | {int(m.geom_bodyid[g]) for g in set(arr["rg"].tolist())}},
            "partner_class": {str(g): gcls[g] for g in set(arr["pg"].tolist())}, "run_dir": str(rd)}
    return arr, meta


def cmd_replay(a):
    sys.path.insert(0, str(a.kit / "p"))
    import hold_w as HW   # noqa: E402  (kit copy; CL_SNAP selects the run snapshot client)
    cl = HW.load_client()
    a.out.mkdir(parents=True, exist_ok=True)
    for rd in a.runs:
        rd = pathlib.Path(rd); key = rd.name
        if (a.out / f"{key}.json").exists():
            print("skip", key, flush=True); continue
        mx = json.load(open(rd / "manifest.json"))["scene"]["model_xml"]
        assert mx.startswith(str(HW.SNAP)), (mx, str(HW.SNAP))
        t0 = time.time()
        arr, meta = replay_one(cl, HW, rd)
        np.savez_compressed(a.out / f"{key}.npz", **arr)
        meta["seconds"] = round(time.time() - t0, 1)
        (a.out / f"{key}.json").write_text(json.dumps(meta, indent=1))
        print(key, "mism", meta["qpos_mismatch_rows"], "records", len(arr["row"]), "best", len(arr["best_row"]), f"{meta['seconds']}s", flush=True)


# =============================================================================================== stage 2: analysis

def prim_sdf(P, typ, size, xpos, xmat):
    """signed distance to a MuJoCo primitive (box 6, cylinder 5, capsule 3, sphere 2, plane 0)."""
    R = np.asarray(xmat, np.float64).reshape(3, 3); loc = (np.atleast_2d(P) - np.asarray(xpos)) @ R; s = np.asarray(size)
    if typ == 6:
        q = np.abs(loc) - s[:3]
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    if typ == 5:
        dr = np.linalg.norm(loc[:, :2], axis=1) - s[0]; dz = np.abs(loc[:, 2]) - s[1]
        q = np.stack([dr, dz], 1)
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    if typ == 3:
        z = np.clip(loc[:, 2], -s[1], s[1]); return np.linalg.norm(loc - np.stack([0 * z, 0 * z, z], 1), axis=1) - s[0]
    if typ == 2:
        return np.linalg.norm(loc, axis=1) - s[0]
    if typ == 0:
        return loc[:, 2]
    raise ValueError(f"geom type {typ}")


LINK_CLASSES = ("checked", "unchecked_wrist", "unchecked_arm", "unchecked_body")
PARTNERS = ("target", "neighbour_fruit", "crate", "pillar", "table", "other")
PHASES = ("searching", "latched", "closing", "held", "placed")
OFFS = (0.0, 0.005, 0.010, 0.020)
LAYERS = ("coarse", "fine", "main", "free")


def link_class(body):
    if body in CHECKED_BODIES:
        return "checked"
    if body.startswith("FT_SENSOR") or body.endswith("_arm_6"):
        return "unchecked_wrist"
    if "_arm_" in body:
        return "unchecked_arm"
    return "unchecked_body"


def hand_of(body, side):
    """'active' / 'inactive' for the 6 hand bodies (by the manifest used arm), None otherwise."""
    if body not in CHECKED_BODIES:
        return None
    own = body.endswith("_L") or body.startswith("ee_finger_l")
    return "active" if own == (side == "left") else "inactive"


URDF_OF = {"EE_BODY_L": "ee_left", "EE_BODY_R": "ee_right", "ee_finger_l1": "ee_finger_l2", "ee_finger_l2": "ee_finger_l1",
           "ee_finger_r1": "ee_finger_r1", "ee_finger_r2": "ee_finger_r2"}   # MJCF body -> URDF link (left finger names swapped)


def stats_mm(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    mm = v * 1000
    return {"n": int(len(v)), "median_mm": round(float(np.median(mm)), 2), "min_mm": round(float(mm.min()), 2), "max_mm": round(float(mm.max()), 2),
            "p10_mm": round(float(np.percentile(mm, 10)), 2), "p90_mm": round(float(np.percentile(mm, 90)), 2),
            "n_over_5mm": int((mm > 5).sum()), "n_over_10mm": int((mm > 10).sum())}


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
    from AG3S.ag3s.experiments.studies.ifg_audit_lte import Scene
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField
    K = a.run / "verify" / a.kit
    if a.from_cache:
        summarize(a, *(json.load(open(a.out / f"{n}.json")) for n in ("scenes", "contacts", "rowcounts")))
        return
    xml = a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
    scene = Scene(xml); mj = scene.mj
    cand_body = np.array([scene.bname[g] for g in scene.cand])
    rep = a.out / "replay"
    contacts = []           # one per (row, link body, partner geom) deepest sub-step
    rowcounts = []          # per (scene, link_class, link body, partner class, phase): rows / substeps
    scenes = []
    for jf in sorted(glob.glob(str(rep / "*.json"))):
        meta = json.load(open(jf)); z = np.load(jf[:-5] + ".npz", allow_pickle=True)
        rd = pathlib.Path(meta["run_dir"]); ep = meta["episode"]
        rj = json.load(open(K / "replay" / f"runs_SPL_{rd.name}.json")); blk = rj["server_chunk_alignment"]["block"]
        an = json.load(open(K / "analysis" / f"{rd.name}.json"))
        # phases per request
        ph = []; summ = {}
        for k in range(blk["block_len"]):
            zz = np.load(pathlib.Path(blk["chunk_dir"]) / f"chunk_{k + blk['offset']:05d}.npz", allow_pickle=True)
            s_ = json.loads(str(zz["summary_json"])); summ[k] = s_
            ph.append(str((s_.get("grasp") or {}).get("state")))
            del zz
        rows_pi = z["row_plan_index"]
        gb = meta["geom_body"]; pc = meta["partner_class"]
        R_ = {k: z[k] for k in ("row", "sub", "rg", "pg", "dist", "force", "exclude")}
        rbody = np.array([gb[str(g)] for g in R_["rg"]]); pcls = np.array([pc[str(g)] for g in R_["pg"]])
        lcls = np.array([link_class(b) for b in rbody])
        side = meta["side"]; S = "L" if side == "left" else "R"; s_ = "l" if side == "left" else "r"
        active = {f"EE_BODY_{S}", f"ee_finger_{s_}1", f"ee_finger_{s_}2"}
        intended = (pcls == "target") & np.isin(rbody, list(active))
        touch = (R_["dist"] <= 0) & (R_["exclude"] == 0)
        deep = R_["dist"] <= -0.001
        sc = {"episode": ep, "side": side, "profile": meta["profile"], "success": an.get("success"), "failure_mode": an.get("failure_mode"),
              "qpos_mismatch_rows": meta["qpos_mismatch_rows"], "n_rows": meta["n_rows"], "replay_seconds": meta.get("seconds"),
              "intended_target_rows": int(len(set(R_["row"][touch & intended].tolist()))),
              "margin_only_records": int(((R_["dist"] > 0) & ~intended).sum()),
              "by_partner": {}, "by_partner_link": {}, "by_partner_phase": {}, "events": []}
        cnt = touch & ~intended
        for p in PARTNERS:
            mk = cnt & (pcls == p)
            sc["by_partner"][p] = {"rows": int(len(set(R_["row"][mk].tolist()))), "substeps": int(len(set(zip(R_["row"][mk].tolist(), R_["sub"][mk].tolist())))),
                                   "rows_pen_ge_1mm": int(len(set(R_["row"][mk & deep].tolist()))),
                                   "max_pen_mm": None if not mk.any() else round(float(-R_["dist"][mk].min()) * 1000, 2),
                                   "max_force_N": None if not mk.any() else round(float(R_["force"][mk].max()), 2)}
            for lc in LINK_CLASSES:
                m2 = mk & (lcls == lc)
                if m2.any():
                    sc["by_partner_link"].setdefault(p, {})[lc] = {"rows": int(len(set(R_["row"][m2].tolist()))),
                                                                  "bodies": sorted(set(rbody[m2].tolist())),
                                                                  "rows_pen_ge_1mm": int(len(set(R_["row"][m2 & deep].tolist())))}
            for phn in PHASES:
                m3 = mk & np.isin(R_["row"], np.flatnonzero(np.array([ph[i] for i in rows_pi]) == phn))
                if m3.any():
                    sc["by_partner_phase"].setdefault(p, {})[phn] = int(len(set(R_["row"][m3].tolist())))
        # events: maximal runs of consecutive rows per (link body, partner class)
        for (b, p) in sorted(set(zip(rbody[cnt].tolist(), pcls[cnt].tolist()))):
            rr = sorted(set(R_["row"][cnt & (rbody == b) & (pcls == p)].tolist()))
            run = [rr[0]]
            for x in rr[1:] + [None]:
                if x is not None and x == run[-1] + 1:
                    run.append(x); continue
                mm = cnt & (rbody == b) & (pcls == p) & np.isin(R_["row"], run)
                sc["events"].append({"link_body": b, "link_class": link_class(b), "hand": hand_of(b, side), "partner": p, "partner_geoms": sorted({meta["geom_name"][str(g)] for g in R_["pg"][mm]}),
                                     "t_first": int(z["row_t"][run[0]]), "t_last": int(z["row_t"][run[-1]]), "rows": len(run),
                                     "requests": sorted({int(rows_pi[i]) for i in run}), "phases": sorted({ph[int(rows_pi[i])] for i in run}),
                                     "max_pen_mm": round(float(-R_["dist"][mm].min()) * 1000, 2), "max_force_N": round(float(R_["force"][mm].max()), 2),
                                     "hold_rows": int(z["row_hold"][run].sum())})
                if x is not None:
                    run = [x]
        for lc in LINK_CLASSES:
            for p in PARTNERS:
                for phn in PHASES:
                    m4 = cnt & (lcls == lc) & (pcls == p) & np.isin(R_["row"], np.flatnonzero(np.array([ph[i] for i in rows_pi]) == phn))
                    if m4.any():
                        rowcounts.append({"episode": ep, "link_class": lc, "partner": p, "phase": phn, "rows": int(len(set(R_["row"][m4].tolist())))})
        # field vs GT at counted contacts (deepest sub-step per row, link body, partner geom)
        fields = {}
        bb = z["best_rbody"]; bpg = z["best_pg"]
        bbody = [meta["body_name"][str(x)] for x in bb]
        for i in range(len(z["best_row"])):
            p = pc[str(int(bpg[i]))]; b = bbody[i]
            if p == "target" and b in active:
                continue
            ri = int(z["best_row"][i]); k = int(rows_pi[ri])
            if k not in fields:
                zz = np.load(pathlib.Path(blk["chunk_dir"]) / f"chunk_{k + blk['offset']:05d}.npz", allow_pickle=True)
                fields = {k: (build_field(zz, outside_distance=1.0), summ[k])}
                del zz
            field, s_k = fields[k]
            qp = z["best_qpos"][i]
            scene.set_q(qp[:scene.m.nq])
            c0 = z["best_pos"][i]; n0 = z["best_nout"][i]
            Pq = np.array([c0 + o * n0 for o in OFFS])
            D, _ = scene.query(Pq)
            is_t = cand_body == meta["target_fruit"]
            hand_t_row = ri in set(z["intended_rows"].tolist())
            held = ph[k] == "held" and hand_t_row
            obd = np.full(len(Pq), np.inf)
            for og in meta["obstacle_geoms"].values():
                obd = np.minimum(obd, prim_sdf(Pq, og["type"], og["size"], og["xpos"], og["xmat"]))
            gt_all = np.minimum(np.where(is_t[None] & held, np.inf, D).min(1), obd)
            gt_free = np.minimum(np.where(is_t[None], np.inf, D).min(1), obd)
            pgname = meta["geom_name"][str(int(bpg[i]))]; pbody = meta["geom_body"][str(int(bpg[i]))]
            if p == "pillar":
                og = meta["obstacle_geoms"][str(int(bpg[i]))]
                gt_part = prim_sdf(Pq, og["type"], og["size"], og["xpos"], og["xmat"])
            else:
                gt_part = np.where((cand_body == pbody)[None], D, np.inf).min(1)
            coarse = np.asarray(field.layers[0].distance(Pq), float)
            fine = np.full(len(Pq), np.nan)
            if len(field.layers) > 1:
                cov = CuroboEsdfField._covers(field.layers[1].grid, Pq)
                if cov.any():
                    fine[cov] = np.asarray(field.layers[1].distance(Pq[cov]), float)
            main = np.asarray(field.distance(Pq), float); tier = list(field.answer_tier(Pq))
            free = np.asarray(field.target_free_distance(Pq), float) if field.has_target_free else np.full(len(Pq), np.nan)
            ftier = list(field.answer_tier(Pq, target_free=True)) if field.has_target_free else [None] * len(Pq)
            links = s_k.get("sphere_link_names") or []
            contacts.append({"episode": ep, "side": side, "hand": hand_of(b, side), "row": ri, "t": int(z["row_t"][ri]), "sub": int(z["best_sub"][i]), "request": k, "phase": ph[k],
                             "link_body": b, "link_class": link_class(b), "robot_geom": meta["geom_name"][str(int(z["best_rg"][i]))],
                             "partner": p, "partner_geom": pgname, "partner_body": pbody, "pen_mm": round(-float(z["best_dist"][i]) * 1000, 3),
                             "pos": [round(float(v), 5) for v in c0], "nout": [round(float(v), 4) for v in n0], "held_target_left_out": bool(held),
                             "offsets_mm": [o * 1000 for o in OFFS],
                             "gt_main_mm": (gt_all * 1000).round(2).tolist(), "gt_free_mm": (gt_free * 1000).round(2).tolist(),
                             "gt_partner_mm": (gt_part * 1000).round(2).tolist(),
                             "coarse_mm": (coarse * 1000).round(2).tolist(), "fine_mm": [None if not np.isfinite(v) else round(v * 1000, 2) for v in fine],
                             "main_mm": (main * 1000).round(2).tolist(), "free_mm": [None if not np.isfinite(v) else round(v * 1000, 2) for v in free],
                             "main_tier": tier, "free_tier": ftier, "fine_window_covers": [bool(np.isfinite(v)) for v in fine],
                             "has_target_free": bool(field.has_target_free), "n_sphere_links": len(set(links))})
        scenes.append(sc)
        print(f"ep{ep}: counted contact pairs {sum(1 for c in contacts if c['episode'] == ep)}", flush=True)
    out = a.out
    (out / "cover_check.json").write_text(json.dumps(sphere_cover_check(scene, K), indent=1))
    (out / "contacts.json").write_text(json.dumps(contacts, indent=1))
    (out / "scenes.json").write_text(json.dumps(scenes, indent=1))
    (out / "rowcounts.json").write_text(json.dumps(rowcounts, indent=1))
    summarize(a, scenes, contacts, rowcounts)


def sphere_cover_check(scene, K, ep=119, k=0):
    """position check of the checked / unchecked body mapping: at request k of ep (robot at rest), the fraction of each robot body's
    collision-geom points (mesh vertices; primitive centres) within 5 mm of the recorded TrajOpt spheres (sphere_centres, sphere_radii)."""
    mj = scene.mj; m = scene.m
    rj = json.load(open(next(iter(glob.glob(str(K / "replay" / f"runs_SPL_SPL_ep{ep}_s*.json"))))))
    blk = rj["server_chunk_alignment"]["block"]; rd = pathlib.Path(rj["run_dir"])
    pl = [r for r in (json.loads(x) for x in open(rd / "frames.jsonl")) if r.get("kind") == "planning"]
    zz = np.load(pathlib.Path(blk["chunk_dir"]) / f"chunk_{k + blk['offset']:05d}.npz", allow_pickle=True)
    C = zz["sphere_centres"].astype(float); Rr = zz["sphere_radii"].astype(float)
    sm = json.loads(str(zz["summary_json"])); names = sm.get("sphere_link_names") or []
    scene.set_q(pl[k]["qpos"])
    coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    out = {"episode": ep, "request": k, "n_spheres": int(len(C)), "sphere_links": sorted(set(names)) if isinstance(names, list) else names,
           "planning_vs_sphere_note": "sphere_centres = server FK at refined_chunk[0] (commanded first step); request 0 of ep119 with the robot at rest",
           "bodies": {}}
    for g in range(m.ngeom):
        if not coll[g] or not scene.robot[g]:
            continue
        b = scene.bname[g]
        if not (b in CHECKED_BODIES or "_arm_" in b or b.startswith("FT_SENSOR")):
            continue
        pts = scene.mesh_world(g) if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH else scene.d.geom_xpos[g][None]
        gap = (np.linalg.norm(pts[:, None] - C[None], axis=2) - Rr[None]).min(1)
        e = out["bodies"].setdefault(b, {"n_points": 0, "n_within_5mm": 0, "link_class": link_class(b), "urdf_link": URDF_OF.get(b)})
        e["n_points"] += int(len(pts)); e["n_within_5mm"] += int((gap <= 0.005).sum())
    for b, e in out["bodies"].items():
        e["frac_within_5mm"] = round(e["n_within_5mm"] / max(e["n_points"], 1), 4)
    return out


NOTES = [
    "definitions were written (script + <out>/definitions_prereg.json) before the 24-scene replay; a one-scene smoke test (ep901) was run first.",
    "replay fidelity: qpos_mismatch_rows = 0 in all 24 scenes (recorded measured qpos == replay pre-step qpos, every element, every control row).",
    "MuJoCo contacts with dist > 0 (margin) were recorded too: margin_only_records per scene in per_scene (all 0 in this data).",
    "B2 field: the request whose chunk was executing (plan_index of the row). The field is built from observations up to that request; the contact happens "
    "0-7 control rows (0-0.47 s) after the request's capture.",
    "B2 GT uses model_transport.xml collision geoms with the deepest sub-step qpos (66) of the replay; the pillar is analytic from its replay pose; "
    "fruit meshes are MuJoCo convex hulls (visual-vs-collision geometry differs by up to ~2 mm on fruit surfaces, see IFG-AUDIT-LTE notes).",
    "checked / unchecked body mapping position check: B3_sphere_cover_check (ep119 request 0, fraction of each body's collision points within 5 mm of the "
    "recorded TrajOpt spheres).",
    "the scene figure renders model_transport.xml + a visual-only pillar cylinder at the replay pose; the contact marker is drawn in 2D over the render "
    "(visible even when occluded).",
]


def crosscheck(a, scenes):
    """row counts of the kit's replay_contacts.py classes vs this replay (union of event rows)."""
    K = a.run / "verify" / a.kit
    res = {"n_compared": 0, "n_equal": 0, "diff": []}
    for s_ in scenes:
        rj = json.load(open(next(iter(glob.glob(str(K / "replay" / f"runs_SPL_SPL_ep{s_['episode']}_s*.json"))))))
        mine = {}
        for e in s_["events"]:
            who = "hand" if e.get("hand") == "active" else "robot_other"
            cls = {"target": "target", "neighbour_fruit": "fruit_other", "crate": "crate", "pillar": "obstacle"}.get(e["partner"])
            if cls is None:
                continue
            mine.setdefault(f"{who}-{cls}", set()).update(range(e["t_first"], e["t_last"] + 1))
        for c in ("hand-fruit_other", "robot_other-fruit_other", "robot_other-crate", "hand-crate", "robot_other-target", "hand-obstacle", "robot_other-obstacle"):
            a_ = int(rj["contact_rows"].get(c, 0)); b_ = len(mine.get(c, ()))
            res["n_compared"] += 1; res["n_equal"] += int(a_ == b_)
            if a_ != b_:
                res["diff"].append({"episode": s_["episode"], "class": c, "kit": a_, "this": b_})
        a_ = int(rj["contact_rows"].get("hand-target", 0)); b_ = s_["intended_target_rows"]
        res["n_compared"] += 1; res["n_equal"] += int(a_ == b_)
        if a_ != b_:
            res["diff"].append({"episode": s_["episode"], "class": "hand-target (intended)", "kit": a_, "this": b_})
    return res


def err_lists(contacts, sel, oi):
    e = {}
    for L in LAYERS:
        gk = "gt_free_mm" if L == "free" else "gt_main_mm"
        e[L] = [None if c[f"{L}_mm"][oi] is None else (c[f"{L}_mm"][oi] - c[gk][oi]) / 1000 for c in contacts if sel(c)]
    return e


def summarize(a, scenes, contacts, rowcounts):
    import datetime
    numbers = {"n_scenes": len(scenes), "qpos_mismatch_rows_total": int(sum(s["qpos_mismatch_rows"] for s in scenes))}
    # B1
    tot = {p: {"scenes_with_contact": sum(1 for s in scenes if s["by_partner"][p]["rows"] > 0), "rows": sum(s["by_partner"][p]["rows"] for s in scenes),
               "rows_pen_ge_1mm": sum(s["by_partner"][p]["rows_pen_ge_1mm"] for s in scenes),
               "max_pen_mm": max([s["by_partner"][p]["max_pen_mm"] for s in scenes if s["by_partner"][p]["max_pen_mm"] is not None], default=None)} for p in PARTNERS}
    numbers["B1_contact_rows_by_partner_all_scenes"] = tot
    numbers["B1_intended_target_rows_total"] = int(sum(s["intended_target_rows"] for s in scenes))
    numbers["B1_by_partner_phase_all_scenes"] = {p: {phn: sum(s["by_partner_phase"].get(p, {}).get(phn, 0) for s in scenes) for phn in PHASES} for p in PARTNERS}
    numbers["B1_per_scene_rows"] = {str(s["episode"]): {p: s["by_partner"][p]["rows"] for p in PARTNERS} for s in scenes}
    # B3
    b3 = {}
    for lc in LINK_CLASSES:
        b3[lc] = {p: sum(r["rows"] for r in rowcounts if r["link_class"] == lc and r["partner"] == p) for p in PARTNERS}
    numbers["B3_rows_by_link_class_partner"] = b3
    numbers["B3_events_by_link_class_partner"] = {lc: {p: sum(1 for s in scenes for e in s["events"] if e["link_class"] == lc and e["partner"] == p) for p in PARTNERS} for lc in LINK_CLASSES}
    numbers["B3_bodies_in_contact"] = sorted({(e["link_body"], e["link_class"], e.get("hand") or "-") for s in scenes for e in s["events"]})
    numbers["B3_body_rows_by_link_class_hand"] = {}
    for s in scenes:
        for e in s["events"]:
            kk = f"{e['link_class']}|{e.get('hand') or '-'}|{e['partner']}"
            numbers["B3_body_rows_by_link_class_hand"][kk] = numbers["B3_body_rows_by_link_class_hand"].get(kk, 0) + e["rows"]
    cc = a.out / "cover_check.json"
    if cc.exists():
        numbers["B3_sphere_cover_check"] = json.load(open(cc))
    # B2
    b2 = {"n_contact_pairs": len(contacts)}
    for p in PARTNERS + ("all",):
        for lc in LINK_CLASSES + ("all",):
            sel = lambda c, p=p, lc=lc: (p == "all" or c["partner"] == p) and (lc == "all" or c["link_class"] == lc)  # noqa: E731
            n = sum(1 for c in contacts if sel(c))
            if not n:
                continue
            ent = {"n": n}
            for oi, o in enumerate(OFFS):
                el = err_lists(contacts, sel, oi)
                ent[f"offset_{int(o * 1000)}mm"] = {L: stats_mm(el[L]) for L in LAYERS}
                ent[f"offset_{int(o * 1000)}mm"]["main_tier_counts"] = {t: sum(1 for c in contacts if sel(c) and c["main_tier"][oi] == t)
                                                                        for t in sorted({c["main_tier"][oi] for c in contacts if sel(c)})}
            b2[f"{p}|{lc}"] = ent
    numbers["B2_field_minus_gt"] = b2
    numbers["crosscheck_vs_kit_replay"] = crosscheck(a, scenes)
    a.notes = NOTES + [f"cross-check vs the kit's replay_contacts.py output (same replay method, classes hand / robot_other): "
                       f"{numbers['crosscheck_vs_kit_replay']['n_equal']} of {numbers['crosscheck_vs_kit_replay']['n_compared']} (scene, class) row counts equal"]
    a.numbers = numbers
    figures = make_figures(a, scenes, contacts, rowcounts, numbers)
    kst = datetime.timezone(datetime.timedelta(hours=9))
    v = {"step": "IFG-PRECHECK-B", "written": datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST"),
         "runs": [{"id": "replay", "venv": ".venv-openpi-live", "raw_output": str(a.out / "replay")},
                  {"id": "analyze", "venv": ".venv-ag3s", "raw_output": [str(a.out / n) for n in ("contacts.json", "scenes.json", "rowcounts.json")]}],
         "data": {"run": str(a.run), "kit": a.kit, "arm": "SPL"},
         "definitions": DEFINITIONS, "numbers": numbers,
         "per_scene": scenes, "scripts": [SCRIPT, str(a.out / "run_replay.sh")], "figures": figures, "notes": getattr(a, "notes", []),
         "baseline_check": json.load(open(a.out / "baseline_check.json")) if (a.out / "baseline_check.json").exists() else None,
         "verdict": "replay fidelity: qpos_mismatch_rows 0 / 24 scenes; regression baseline (legacy) matched (see baseline_check)",
         "not_measured": ["contacts between robot links (self-collision)", "contacts of the held apple with the world (not robot links)",
                          "IFG-APPLE-NBC (task: B covers SPLIT only)"]}
    if a.verify:
        if a.verify.exists():
            raise SystemExit(f"refusing to overwrite {a.verify}")
        a.verify.write_text(json.dumps(v, indent=1, default=float, ensure_ascii=False))
        print("wrote", a.verify)


def make_figures(a, scenes, contacts, rowcounts, numbers):
    if a.skip_fig:
        return []
    return figures_all(a, scenes, contacts, rowcounts, numbers)


PCOL = {"target": 0, "neighbour_fruit": 1, "crate": 2, "pillar": 3, "table": 4, "other": 5}   # figstyle.CATEGORICAL slots, fixed order


def _save(fig, path, data, plt, FS):
    path = pathlib.Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(data | {"script": SCRIPT}, indent=1, default=float, ensure_ascii=False))
    print("wrote", path)
    return str(path)


def figures_all(a, scenes, contacts, rowcounts, numbers):
    import matplotlib.pyplot as plt
    from AG3S.ag3s.experiments.common import figstyle as FS
    KO = FS.use_korean(); T = (lambda ko, en: ko if KO else en)  # noqa: E731
    a.fig.mkdir(parents=True, exist_ok=True)
    PN = {"target": T("target 사과 (손 외 링크)", "target apple (non-hand link)"), "neighbour_fruit": T("이웃 과일", "neighbour fruit"),
          "crate": T("바구니", "basket"), "pillar": T("기둥", "pillar"), "table": T("탁자", "table"), "other": T("기타 (바닥·벽·선반)", "other")}
    out = []
    # ---------------------------------------------------------------- (1) real scene
    import mujoco
    xml = a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
    rep = a.out / "replay"
    groups = {}
    for c in contacts:
        kk = (c["episode"], c["partner"])
        if kk not in groups or c["pen_mm"] > groups[kk]["pen_mm"]:
            groups[kk] = c
    sel = [groups[k] for k in sorted(groups)]
    n = len(sel); ncol = min(4, max(n, 1)); nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 5.0 * nrow)); FS.style_axes(fig, axs); axs = np.atleast_1d(axs).ravel()
    panels = []
    for ax, c in zip(axs, sel):
        meta = json.load(open(rep / f"{pathlib.Path(next(s_['run_dir'] for s_ in [json.load(open(j)) for j in glob.glob(str(rep / '*.json'))] if s_['episode'] == c['episode'])).name}.json"))
        z = np.load(rep / f"{pathlib.Path(meta['run_dir']).name}.npz")
        spec = mujoco.MjSpec.from_file(str(xml))
        for og in meta["obstacle_geoms"].values():
            q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(og["xmat"], float))
            spec.worldbody.add_geom(name="precheck_pillar_vis", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=og["size"], pos=og["xpos"], quat=q,
                                    contype=0, conaffinity=0, rgba=[0.55, 0.55, 0.6, 1])
        m = spec.compile(); d = mujoco.MjData(m)
        idx = np.flatnonzero((z["best_row"] == c["row"]) & (np.array([meta["body_name"][str(b)] for b in z["best_rbody"]]) == c["link_body"])
                             & (np.array([meta["geom_name"][str(g)] for g in z["best_pg"]]) == c["partner_geom"]))[0]
        d.qpos[:] = z["best_qpos"][idx][:m.nq]; mujoco.mj_forward(m, d)
        cpos = np.asarray(c["pos"]); nout = np.asarray(c["nout"])
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = cpos; cam.distance = 0.42
        cam.azimuth = float(np.degrees(np.arctan2(cpos[1], cpos[0]))) + 180 + 55; cam.elevation = -28
        rnd = mujoco.Renderer(m, 520, 520); rnd.update_scene(d, camera=cam)
        scn = rnd.scene
        g = scn.geoms[scn.ngeom]; mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]), cpos, np.eye(3).ravel(), np.array([0.89, 0.29, 0.28, 1], np.float32)); scn.ngeom += 1
        g = scn.geoms[scn.ngeom]; mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([0.89, 0.29, 0.28, 1], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.003, cpos, cpos + 0.06 * nout); scn.ngeom += 1
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)   # MuJoCo free camera (mono, no stereo eye offset)
        fw = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]); up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
        rt = np.cross(fw, up); cpos_cam = cpos - cam.distance * fw
        fpx = 0.5 * 520 / np.tan(np.radians(m.vis.global_.fovy) / 2)

        def proj(pw):
            v = np.asarray(pw) - cpos_cam
            return np.array([260 + fpx * (v @ rt) / (v @ fw), 260 - fpx * (v @ up) / (v @ fw)])
        img = rnd.render(); rnd.close()
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        u0 = proj(cpos); u1 = proj(cpos + 0.06 * nout)
        ax.annotate("", xy=u1, xytext=u0, arrowprops=dict(arrowstyle="->", color=FS.CATEGORICAL[7], lw=1.6))
        ax.plot(*u0, "o", ms=13, mfc="none", mec=FS.CATEGORICAL[7], mew=1.8)
        ax.set_xlim(0, 520); ax.set_ylim(520, 0)
        urdf = URDF_OF.get(c["link_body"])
        lab = c["link_body"] + (f" (URDF {urdf})" if urdf and urdf != c["link_body"] else "")
        ok = next(s_ for s_ in scenes if s_["episode"] == c["episode"])
        ax.set_title(f"ep{c['episode']} t={c['t']} req {c['request']} ({c['phase']})\n{lab} ↔ {c['partner_geom']}\n"
                     + T(f"침투 {c['pen_mm']:.2f} mm · {PN[c['partner']]} · ", f"pen {c['pen_mm']:.2f} mm · {PN[c['partner']]} · ")
                     + ("success" if ok["success"] else f"fail ({ok['failure_mode']})"), fontsize=7.5, loc="left")
        panels.append({k: c[k] for k in ("episode", "t", "sub", "request", "phase", "link_body", "link_class", "hand", "partner", "partner_geom", "pen_mm", "pos", "nout",
                                          "coarse_mm", "fine_mm", "main_mm", "free_mm", "gt_main_mm", "offsets_mm")} | {"camera": {"lookat": cpos.tolist(), "distance": 0.42,
                                          "azimuth": cam.azimuth, "elevation": -28}, "pillar_rendered": bool(meta["obstacle_geoms"])})
    for ax in axs[len(sel):]:
        ax.axis("off")
    fig.suptitle(T("IFG-PRECHECK-B 실제 장면: 집계된 접촉마다 (장면, 상대) 묶음의 가장 깊은 순간. 빨간 원 = MuJoCo 접촉점 (가려져도 표시), 화살표 = 상대→로봇 법선 (60 mm)",
                   "IFG-PRECHECK-B scenes: deepest counted contact per (scene, partner); red dot = MuJoCo contact point, arrow = partner→robot normal"),
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    out.append(_save(fig, a.fig / "b-scene-contacts.png", {"rule": "per (episode, partner) the counted contact pair with the largest penetration; "
                                                                  "render = model_transport.xml + pillar visual from the replay pose, qpos = deepest sub-step", "panels": panels}, plt, FS))
    # ---------------------------------------------------------------- (2a) timeline graph
    order = sorted(scenes, key=lambda s_: (bool(s_["success"]), s_["episode"]))
    fig, ax = plt.subplots(figsize=(13, 7.6)); FS.style_axes(fig, ax)
    tl = {}
    for yi, s_ in enumerate(order):
        meta = next(json.load(open(j)) for j in glob.glob(str(rep / "*.json")) if json.load(open(j))["episode"] == s_["episode"])
        z = np.load(rep / f"{pathlib.Path(meta['run_dir']).name}.npz")
        it = z["row_t"][z["intended_rows"]] if len(z["intended_rows"]) else np.array([])
        ax.scatter(it, np.full(len(it), yi), s=9, marker="|", color=FS.GRID_INK, lw=1.0)
        tl[str(s_["episode"])] = {"intended_target_t": it.astype(int).tolist(), "counted": {}}
        for e in s_["events"]:
            tt = np.arange(e["t_first"], e["t_last"] + 1)
            off = {"checked": 0.0, "unchecked_wrist": 0.18, "unchecked_arm": 0.18, "unchecked_body": 0.18}[e["link_class"]]
            ax.scatter(tt, np.full(len(tt), yi + off - 0.18 * (e.get("hand") == "inactive")), s=11, marker="s" if e["link_class"] == "checked" else "^",
                       color=FS.CATEGORICAL[PCOL[e["partner"]]], lw=0)
            tl[str(s_["episode"])]["counted"].setdefault(f"{e['link_body']}|{e['partner']}", []).append([e["t_first"], e["t_last"]])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"ep{s_['episode']} {'S' if s_['success'] else 'F:' + str(s_['failure_mode'])}{' +pillar' if s_['profile'] != 'clear' else ''}" for s_ in order], fontsize=7)
    ax.set_xlim(0, 600); ax.set_xlabel(T("control step t (15 Hz)", "control step t (15 Hz)"), fontsize=8.5)
    for p, ci in PCOL.items():
        ax.scatter([], [], s=14, marker="s", color=FS.CATEGORICAL[ci], label=PN[p])
    ax.scatter([], [], s=14, marker="|", color=FS.GRID_INK, label=T("의도된 손–target 접촉 (집계 제외)", "intended hand–target (not counted)"))
    ax.scatter([], [], s=14, marker="s", color=FS.INK_2, label=T("■ 검사 링크 (손 6) · ▲ 미검사 링크", "■ checked hand link · ▲ unchecked link"))
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
    ax.set_title(T("IFG-PRECHECK-B: 실제 실행(physics replay)의 로봇–환경 접촉 step, 장면별 (위 = 실패 6). 비활성 손은 0.18 아래로",
                   "IFG-PRECHECK-B: robot–world contact steps in the executed rollouts per scene (failures first); inactive hand drawn 0.18 below"), fontsize=9.5, loc="left")
    fig.tight_layout()
    out.append(_save(fig, a.fig / "b-contact-timeline.png", {"scene_order": [s_["episode"] for s_ in order], "timeline": tl,
                                                             "rows_by_partner": numbers["B1_contact_rows_by_partner_all_scenes"]}, plt, FS))
    # ---------------------------------------------------------------- (2b) field - GT graph
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True); FS.style_axes(fig, axs)
    rng = np.random.default_rng(0); fe = {}
    for li, L in enumerate(LAYERS):
        ax = axs[li]; fe[L] = {}
        gk = "gt_free_mm" if L == "free" else "gt_main_mm"
        for oi, o in enumerate(OFFS):
            for p in PARTNERS:
                v = [c[f"{L}_mm"][oi] - c[gk][oi] for c in contacts if c["partner"] == p and c[f"{L}_mm"][oi] is not None]
                if not v:
                    continue
                pj = list(PCOL).index(p)
                x = oi + (pj - 2.5) * 0.11 + rng.uniform(-0.03, 0.03, len(v))
                ax.scatter(x, v, s=7, color=FS.CATEGORICAL[PCOL[p]], alpha=0.7, lw=0)
                fe[L].setdefault(f"{int(o * 1000)}mm", {})[p] = {"n": len(v), "median": round(float(np.median(v)), 2), "min": round(float(min(v)), 2), "max": round(float(max(v)), 2)}
        ax.axhline(0, color=FS.INK_2, lw=0.7)
        for yv in (5, 10):
            ax.axhline(yv, color=FS.GRID_INK, lw=0.6, ls="--")
        ax.set_xticks(range(len(OFFS))); ax.set_xticklabels([f"{int(o * 1000)}" for o in OFFS])
        ax.set_xlabel(T("접촉점에서 법선 방향 거리 s (mm)", "offset s along the normal (mm)"), fontsize=8)
        nm = {"coarse": T("거친 층 (20 mm)", "coarse (20 mm)"), "fine": T("고운 층 (5 mm 창)", "fine (5 mm window)"), "main": T("합성 main = min(거친, 고운)", "main = min(coarse, fine)"),
              "free": T("target-free 층 (vs GT target 제외)", "target-free (vs GT w/o target)")}[L]
        ax.set_title(nm, fontsize=9, loc="left")
    axs[0].set_ylabel(T("field − GT (mm); 양수 = 실제보다 멀다고 봄", "field − GT (mm); > 0 = says farther"), fontsize=8.5)
    for p in PARTNERS:
        if any(c["partner"] == p for c in contacts):
            axs[3].scatter([], [], s=12, color=FS.CATEGORICAL[PCOL[p]], label=PN[p])
    axs[3].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(T(f"IFG-PRECHECK-B: 집계된 접촉 {len(contacts)}개(행·링크·상대 geom별 가장 깊은 sub-step)에서 그 요청의 거리장 값 − GT", 
                   f"IFG-PRECHECK-B: field − GT at {len(contacts)} counted contacts (request that planned the executing chunk)"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    out.append(_save(fig, a.fig / "b-field-minus-gt.png", {"definition": DEFINITIONS["field_vs_gt"], "by_layer_offset_partner": fe}, plt, FS))
    # ---------------------------------------------------------------- (3) tables
    cols = ["ep", T("팔", "arm"), T("장면", "scene"), T("결과", "outcome"), T("의도 손–target", "intended"), "target*", T("이웃 과일", "neighbour"),
            T("바구니", "basket"), T("기둥", "pillar"), T("탁자", "table"), T("기타", "other"), T("접촉 링크 (분류)", "links (class)"), T("최대 침투 mm", "max pen mm")]
    rows = []
    for s_ in sorted(scenes, key=lambda s_: (bool(s_["success"]), s_["episode"])):
        lk = sorted({f"{e['link_body']} ({'C' if e['link_class'] == 'checked' else 'U'}{'/inact' if e.get('hand') == 'inactive' else ''})" for e in s_["events"]})
        mp = max([e["max_pen_mm"] for e in s_["events"]], default=None)
        rows.append([str(s_["episode"]), s_["side"][0].upper(), "pillar" if s_["profile"] != "clear" else "clear",
                     "S" if s_["success"] else f"F {s_['failure_mode']}", str(s_["intended_target_rows"])]
                    + [str(s_["by_partner"][p]["rows"]) if s_["by_partner"][p]["rows"] else "·" for p in PARTNERS] + [", ".join(lk) if lk else "·", "·" if mp is None else f"{mp:.2f}"])
    totals = numbers["B1_contact_rows_by_partner_all_scenes"]
    rows.append([T("합계", "total"), "", "", "", str(numbers["B1_intended_target_rows_total"])] + [str(totals[p]["rows"]) for p in PARTNERS] + ["", ""])
    fig, ax = plt.subplots(figsize=(17, 0.36 * len(rows) + 1.6)); FS.style_axes(fig, ax); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=cols, loc="upper left", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.2); tb.scale(1, 1.25)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor(FS.GRID_INK)
        if r_ == 0:
            cell.set_facecolor("#eeeeea")
    w = [0.04, 0.03, 0.045, 0.085, 0.06, 0.05, 0.055, 0.05, 0.045, 0.045, 0.045, 0.30, 0.06]
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_width(w[c_])
    ax.set_title(T("IFG-PRECHECK-B 표 1: 장면별 접촉 step 수 (control row, 15 Hz; 침투 ≥ 0 mm). target* = 활성 손 외 링크의 target 접촉. C = 검사 링크(손 6), U = 미검사, inact = 비활성 손",
                   "IFG-PRECHECK-B table 1: contact steps per scene (penetration >= 0 mm). target* = target touched by a non-active-hand link. C checked, U unchecked"),
                 fontsize=9, loc="left")
    out.append(_save(fig, a.fig / "b-table-scenes.png", {"columns": cols, "rows": rows, "partner_order": list(PARTNERS)}, plt, FS))
    # summary table: B3 link class x partner, B2 medians
    b3 = numbers["B3_rows_by_link_class_partner"]
    r1 = [[lc] + [str(b3[lc][p]) for p in PARTNERS] for lc in LINK_CLASSES]
    b2 = numbers["B2_field_minus_gt"]
    r2 = []
    for key in sorted(k for k in b2 if k != "n_contact_pairs" and k.endswith("|all")):
        e = b2[key]
        for o in ("offset_0mm", "offset_5mm", "offset_10mm"):
            r2.append([key.split("|")[0], str(e["n"]), o.split("_")[1]] + [("·" if e[o][L]["n"] == 0 else f"{e[o][L]['median_mm']:+.1f} [{e[o][L]['min_mm']:+.1f}, {e[o][L]['max_mm']:+.1f}] n={e[o][L]['n']}")
                                                                   for L in LAYERS])
    fig, axs = plt.subplots(2, 1, figsize=(15, 0.33 * (len(r1) + len(r2)) + 3.2), gridspec_kw={"height_ratios": [len(r1) + 2, len(r2) + 2]})
    FS.style_axes(fig, axs)
    for ax_, cl_, rr_, ttl in ((axs[0], [T("링크 분류", "link class")] + [PN[p] for p in PARTNERS], r1,
                                T("표 2 (B3): 링크 분류 × 상대, 접촉 step 수 (장면별로 그 분류 안 중복 없는 step, 24 장면 합)", "table 2 (B3): contact steps by link class x partner (unique steps per scene within the class, summed over scenes)")),
                               (axs[1], [T("상대", "partner"), "n", "s mm"] + [f"{L}: median [min, max]" for L in LAYERS], r2,
                                T("표 3 (B2): field − GT (mm) 중앙값 [최소, 최대], 집계 접촉 전부 (링크 분류 합침)", "table 3 (B2): field − GT (mm) median [min, max], all link classes"))):
        ax_.axis("off")
        tb = ax_.table(cellText=rr_, colLabels=cl_, loc="upper left", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(7.4); tb.scale(1, 1.25)
        for (r_, c_), cell in tb.get_celld().items():
            cell.set_edgecolor(FS.GRID_INK)
            if r_ == 0:
                cell.set_facecolor("#eeeeea")
        ax_.set_title(ttl, fontsize=9, loc="left")
    fig.tight_layout()
    out.append(_save(fig, a.fig / "b-table-summary.png", {"B3_rows_by_link_class_partner": b3, "B2_rows": r2, "B2_columns": [f"{L}" for L in LAYERS]}, plt, FS))
    return out


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("replay"); r.add_argument("--kit", type=pathlib.Path, required=True); r.add_argument("--out", type=pathlib.Path, required=True)
    r.add_argument("runs", nargs="+")
    z = sp.add_parser("analyze"); z.add_argument("--run", type=pathlib.Path, required=True); z.add_argument("--out", type=pathlib.Path, required=True)
    z.add_argument("--fig", type=pathlib.Path, required=True); z.add_argument("--verify", type=pathlib.Path, default=None)
    z.add_argument("--kit", default="IFG-APPLE-SPLIT"); z.add_argument("--skip-fig", action="store_true")
    z.add_argument("--from-cache", action="store_true", help="reuse <out>/contacts.json, scenes.json, rowcounts.json")
    a = ap.parse_args()
    (cmd_replay if a.cmd == "replay" else cmd_analyze)(a)


if __name__ == "__main__":
    main()
