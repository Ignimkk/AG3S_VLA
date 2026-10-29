"""T41c — physics replay of recorded runs: per-HOLD motion, contacts and fruit slip (verifier, CPU only, MUJOCO_GL=osmesa).

usage: hold_c.py OUT_JSON RUN_DIR [RUN_DIR ...]      (RUN_DIR = .../E3b/<STAGE>/E3b_ep<EP>_s<SEED>)

Replay (T40/H/replay.py method, client code of the snapshot ${AG3S_PROJECT_ROOT}-t41c): pi05_infer.main() reset with the run's episode,
seed and obstacle profile (manifest scene.obstacle_profile; 'clear' / none = no obstacle args), 1000 warm-up mj_steps, then every
recorded control row's applied_ctrl (arm 7+7, gripper 2 x RBY1_GRIPPER_OPEN) for spa sub-steps. fidelity: qpos_mismatch_rows =
control rows whose recorded qpos differs from the replay's pre-step qpos (any element != 0.0).
Per control row (pre = recorded pose before the row's physics; post = after its spa sub-steps):
  left-arm joints (LEFT_ARM_JOINTS qpos), TCP = site left_ee world position, apple centre (body 'apple'), apple in the TCP frame
  (R_site^T (p_apple - p_tcp)), apple to finger-midpoint distance (bodies ee_finger_l1/l2).
  contacts: every sub-step, MuJoCo contacts (d.contact) between active obstacle collision geoms and robot geoms (manager.robot_geom_ids
  = arm subtrees) / payload geoms (manager.payload_geom_ids); counted per row in sub-steps, pairs by geom name.
  robot clearance: manager.robot_clearance(0.25) at sub-steps 0, 5, 10, ... (client distance code, H fix), min per row.
HOLD = control rows with a 'hold' record (executed false). A HOLD episode = maximal run of consecutive HOLD rows (consecutive t_step).
Window of a HOLD episode = pre-state of its first row + post-states of all its rows (so the physics of the last HOLD step counts).
  duration_steps = rows; n_chunks = distinct chunk_seq; kinds = union of hold.kinds / entry_kinds; mode / source / gripper_source.
  joint_max_deg = max over window and over the 7 left-arm joints of |q - q(entry pre)|; tcp_max_mm = max |p_tcp - p_tcp(entry pre)|.
  dev_from_q_hold_max_deg = max over window of max |q_left_measured - q_hold(left 7)|; q_hold_vs_entry_measured_deg = at entry.
  fruit: in_hand_at_entry = apple-finger-mid <= 60 mm AND apple z - its z at t=0 >= 10 mm;
         slip_max_mm = max |apple_in_tcp - apple_in_tcp(entry)|; slip_end_mm = value at the last post-state; apple_dz_mm = world z change.
  contacts_in_hold: robot / payload contact sub-steps inside the window rows, pairs; min robot clearance over the window rows.
Contact events (whole run): maximal runs of rows with robot or payload contact; first t, last t, n_substeps, kind, pairs, in_hold (first row)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, os, io, inspect, contextlib, importlib.util, pathlib, time
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco

HY = pathlib.Path("${AG3S_PROJECT_ROOT}-t41c/pi05_TO_hybrid")
sys.path.insert(0, str(HY / "rby1_manipulation" / "src"))
F = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/configs'))


class _R(Exception):
    pass


def load_client():
    spec = importlib.util.spec_from_file_location("_cl", HY / "rby1_bringup/pi05_infer.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.__file__.startswith(str(HY)), mod.__file__
    return mod


def reset(cl, ep, seed, prof):
    cap = {}

    def stub(remote):
        cap.update(inspect.currentframe().f_back.f_locals); raise _R
    orig = cl.load_remote_policy; cl.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index", str(ep),
                "--headless", "--warmup-steps", "1000", "--policy-seed", str(seed), "--obstacle-stop-distance", "0"]
    if prof and prof != "clear":
        sys.argv += ["--obstacle-config", str(F / f"{prof}.json"), "--obstacle-profile", prof]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            cl.main()
    except _R:
        pass
    finally:
        cl.load_remote_policy = orig
    return cap


def gname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"


def replay(cl, rd):
    man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"])
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; rows = [r for r in rows if r["kind"] == "control"]
    L = reset(cl, ep, seed, prof); m, d, mg, act, idx = L["m"], L["d"], L.get("obstacle_manager"), L["act"], L["idx"]
    spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
    for _ in range(1000):
        mujoco.mj_step(m, d)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("apple", "ee_finger_l1", "ee_finger_l2")}
    lq = np.asarray(idx["left_q"], int)
    obst = set(mg.collision_geom_ids) if mg is not None else set()
    robot = set(mg.robot_geom_ids) if mg is not None else set()
    payload = set(mg.payload_geom_ids) if mg is not None else set()

    def state():
        R = d.site_xmat[site].reshape(3, 3); p = d.site_xpos[site].copy(); a = d.xpos[bid["apple"]].copy()
        fm = 0.5 * (d.xpos[bid["ee_finger_l1"]] + d.xpos[bid["ee_finger_l2"]])
        return {"q": d.qpos[lq].copy(), "tcp": p, "apple": a, "apple_tcp": R.T @ (a - p), "apple_fm": float(np.linalg.norm(a - fm))}
    n = 7; mism = 0; per = []
    for r in rows:
        if np.max(np.abs(np.asarray(r["qpos"]) - d.qpos)) != 0.0:
            mism += 1
        pre = state()
        a = r["applied_ctrl"]
        for i in range(n):
            d.ctrl[act["left_a"][i]] = a["arm"][i]; d.ctrl[act["right_a"][i]] = a["arm"][n + i]
        d.ctrl[act["left_grip_a"]] = a["gripper"][0] * cl.RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = a["gripper"][1] * cl.RBY1_GRIPPER_OPEN
        rc, pc, pairs, clr = 0, 0, set(), np.inf
        for s in range(spa):
            mujoco.mj_step(m, d)
            if obst:
                hit_r = hit_p = False
                for k in range(d.ncon):
                    c = d.contact[k]; pr = {int(c.geom1), int(c.geom2)}
                    if not pr & obst:
                        continue
                    if pr & robot:
                        hit_r = True
                    elif pr & payload:
                        hit_p = True
                    else:
                        continue
                    pairs.add(tuple(sorted(gname(m, g) for g in pr)))
                rc += hit_r; pc += hit_p
                if s % 5 == 0:
                    clr = min(clr, mg.robot_clearance(0.25))
        per.append({"t": r["t_step"], "seq": r.get("chunk_seq"), "hold": r.get("hold"), "executed": r.get("executed"), "pre": pre, "post": state(),
                    "robot_contact_substeps": rc, "payload_contact_substeps": pc, "pairs": sorted(pairs),
                    "robot_clearance_min_m": None if not np.isfinite(clr) else float(clr)})
    return dict(prof=prof, ep=ep, seed=seed, spa=spa, n_rows=len(rows), qpos_mismatch_rows=mism, per=per,
                hold_mode=(man.get("policy") or {}).get("safe_hold_mode"))


def analyse(R):
    per = R["per"]; z0 = per[0]["pre"]["apple"][2]
    # HOLD episodes
    eps, cur = [], []
    for i, p in enumerate(per):
        if p["hold"]:
            if cur and per[cur[-1]]["t"] + 1 != p["t"]:
                eps.append(cur); cur = []
            cur.append(i)
        elif cur:
            eps.append(cur); cur = []
    if cur:
        eps.append(cur)
    holds = []
    for e in eps:
        P0 = per[e[0]]["pre"]; W = [P0] + [per[i]["post"] for i in e]
        hr = [per[i]["hold"] for i in e]
        kinds = sorted({k for h in hr for k in (h.get("kinds") or []) + (h.get("entry_kinds") or [])})
        qh = np.asarray(hr[0]["q_hold"], float)[:7]
        jm = max(float(np.max(np.abs(w["q"] - P0["q"]))) for w in W)
        tm = max(float(np.linalg.norm(w["tcp"] - P0["tcp"])) for w in W)
        dev = max(float(np.max(np.abs(w["q"] - np.asarray(per[i]["hold"]["q_hold"], float)[:7])))
                  for w, i in zip(W, [e[0]] + e))
        sl = [float(np.linalg.norm(w["apple_tcp"] - P0["apple_tcp"])) for w in W]
        rc = sum(per[i]["robot_contact_substeps"] for i in e); pc = sum(per[i]["payload_contact_substeps"] for i in e)
        cl = [per[i]["robot_clearance_min_m"] for i in e if per[i]["robot_clearance_min_m"] is not None]
        holds.append({"t_first": per[e[0]]["t"], "t_last": per[e[-1]]["t"], "duration_steps": len(e),
                      "chunk_seqs": sorted({per[i]["seq"] for i in e}), "n_chunks": len({per[i]["seq"] for i in e}),
                      "kinds": kinds, "mode": hr[0].get("mode"), "source": hr[0].get("source"), "gripper_source": hr[0].get("gripper_source"),
                      "entry_command_dev_rad_recorded": hr[0].get("entry_command_dev_rad"),
                      "q_hold_vs_entry_measured_deg": float(np.degrees(np.max(np.abs(qh - P0["q"])))),
                      "joint_max_deg": float(np.degrees(jm)), "tcp_max_mm": 1e3 * tm,
                      "dev_from_q_hold_max_deg": float(np.degrees(dev)),
                      "fruit": {"apple_finger_mid_entry_mm": 1e3 * P0["apple_fm"], "apple_lift_entry_mm": 1e3 * (P0["apple"][2] - z0),
                                "in_hand_at_entry": bool(P0["apple_fm"] <= 0.060 and P0["apple"][2] - z0 >= 0.010),
                                "slip_max_mm": 1e3 * max(sl), "slip_end_mm": 1e3 * sl[-1],
                                "apple_dz_end_mm": 1e3 * (W[-1]["apple"][2] - P0["apple"][2]),
                                "apple_finger_mid_end_mm": 1e3 * W[-1]["apple_fm"]},
                      "contacts_in_hold": {"robot_substeps": rc, "payload_substeps": pc,
                                           "pairs": sorted({tuple(x) for i in e for x in per[i]["pairs"]}),
                                           "robot_clearance_min_mm": (1e3 * min(cl)) if cl else None}})
    # contact events
    ev, cur = [], None
    for p in per:
        k = ("robot" if p["robot_contact_substeps"] else "") + ("+payload" if p["payload_contact_substeps"] else "")
        if k:
            if cur and cur["t_last"] + 1 == p["t"]:
                cur["t_last"] = p["t"]; cur["robot_substeps"] += p["robot_contact_substeps"]; cur["payload_substeps"] += p["payload_contact_substeps"]
                cur["pairs"] |= {tuple(x) for x in p["pairs"]}; cur["n_hold_rows"] += bool(p["hold"])
            else:
                cur = {"t_first": p["t"], "t_last": p["t"], "robot_substeps": p["robot_contact_substeps"], "payload_substeps": p["payload_contact_substeps"],
                       "pairs": {tuple(x) for x in p["pairs"]}, "in_hold_first_row": bool(p["hold"]), "n_hold_rows": int(bool(p["hold"]))}
                ev.append(cur)
        else:
            cur = None
    for e in ev:
        e["pairs"] = sorted(e["pairs"])
    return {"profile": R["prof"], "episode": R["ep"], "seed": R["seed"], "hold_mode_manifest": R["hold_mode"], "spa": R["spa"],
            "n_rows": R["n_rows"], "qpos_mismatch_rows": R["qpos_mismatch_rows"], "n_hold_rows": sum(bool(p["hold"]) for p in per),
            "hold_episodes": holds, "contact_events": ev,
            "series": {"t": [p["t"] for p in per], "hold": [bool(p["hold"]) for p in per],
                       "robot_clearance_min_mm": [None if p["robot_clearance_min_m"] is None else 1e3 * p["robot_clearance_min_m"] for p in per],
                       "tcp": [p["pre"]["tcp"].tolist() for p in per], "apple_tcp_mm": [(1e3 * p["pre"]["apple_tcp"]).tolist() for p in per],
                       "apple_finger_mid_mm": [1e3 * p["pre"]["apple_fm"] for p in per]}}


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]); res = json.loads(out.read_text()) if out.exists() else {"definitions": __doc__, "runs": {}}
    cl = load_client()
    for rd in map(pathlib.Path, sys.argv[2:]):
        key = str(rd.relative_to(_ag3s_workspace_path('outputs/verify')))
        if key in res["runs"]:
            print("skip", key); continue
        t0 = time.time(); R = replay(cl, rd); A = analyse(R); A["run_dir"] = str(rd)
        res["runs"][key] = A
        out.write_text(json.dumps(res, indent=1, default=float))
        print(key, "rows", A["n_rows"], "mism", A["qpos_mismatch_rows"], "holds", [(h["t_first"], h["duration_steps"], h["kinds"], round(h["joint_max_deg"], 2),
              round(h["tcp_max_mm"], 1), round(h["fruit"]["slip_max_mm"], 1), h["contacts_in_hold"]["robot_substeps"]) for h in A["hold_episodes"]],
              "contacts", [(e["t_first"], e["t_last"], e["robot_substeps"], e["payload_substeps"], e["in_hold_first_row"]) for e in A["contact_events"]],
              f"{time.time() - t0:.0f}s", flush=True)
