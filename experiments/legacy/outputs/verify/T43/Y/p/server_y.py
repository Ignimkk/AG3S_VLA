"""T43 Y — per server chunk record of one run, with the held query spheres placed against the TRUE crate geometry.
(verifier, CPU only; MUJOCO_GL=osmesa)

usage: server_y.py OUT_JSON KEY [KEY ...]       KEY = <ARM>/<STAGE>/<name> as in T43/W/analysis/analysis.json

Server segment = analysis.json server_segment (analyze_t41 pairing rule: every refined_chunk == the run's planning actions).
Per chunk (t = planning-row t_step):
  grasp: state, state_before, attach, detach, transition, rejected, attach_revoked, evidence.{placed, open_streak, opening_after,
         last_command}; subtask: label, released, placed_seen; exec_feedback: n_exec, hold_reason; verdict reasons (kind, action, detail);
  to: status, clearance_mm, worst row {query, link, obstacle, step, tier, clearance_mm}; verdict.classification collision_by_why;
  binding / category = analysis.json server_chunks (analyze_t43w rule: base q < 122, gripper_cover 122 <= q < n_robot, held q >= n_robot).
  held query spheres = grasp.attached.held.body_cover.{lifted_parent_frame if frame.near_support else body_parent_frame}
         (checked: the worst row's point_m equals that sphere's centre at refined row `step`, |diff| = 0.0 mm on 4 chunks),
         frame link ee_left == MJCF body EE_BODY_L (mujoco_source alias). Placed with MuJoCo FK on
         (a) the planning-row qpos (measured pose at the chunk start) and (b) the planning-row qpos with the active-arm joints of
         refined_chunk row `step` (the worst row's pose).
  true crate signed distance of a sphere = min over crate collision boxes (floor, 4 walls, handles; crate pose from the
         planning-row qpos) of (point-to-OBB signed distance of the centre) - radius; < 0 = the sphere overlaps the crate solid.
  apple (true): mj_geomDistance(apple_geom, crate geoms) min at the planning-row qpos; apple centre in the crate frame.
  gripper: refined (= executed if not HOLD) and reference (policy) gripper column of the active arm, rows 0-7 (min / max);
         applied gripper (control rows of the chunk, mean)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

AN = _ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')
LEFT_Q = list(range(20, 27)); RIGHT_Q = list(range(11, 18))


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def obb_sd(p, c, R, h):
    q = R.T @ (np.asarray(p) - c); dv = np.abs(q) - h
    return float(np.linalg.norm(np.maximum(dv, 0.0)) + min(float(dv.max()), 0.0))


def one(key, a):
    r = a["runs"][key]; side = r["side"]; HM.set_arm(side); HM._M.clear()
    M = HM.model(); m, d = M["m"], M["d"]
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    SD = "L" if side == "left" else "R"; ee = B(f"EE_BODY_{SD}"); cb = B("crate"); ab = B("apple")
    crate_g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == cb and (m.geom_contype[g] or m.geom_conaffinity[g])]
    g_apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
    qi, c0 = (LEFT_Q, 0) if side == "left" else (RIGHT_Q, 8); gc = 7 if side == "left" else 15
    rows = [json.loads(l) for l in open(pathlib.Path(r["run_dir"]) / "frames.jsonl") if l.strip()]
    plan = {p["seq"]: p for p in rows if p["kind"] == "planning"}
    ctrl = [p for p in rows if p["kind"] == "control"]
    seg = r.get("server_segment")
    if not seg:
        return {"key": key, "error": "no server segment"}
    first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1]); dd = first.parent
    sc_an = {c["seq"]: c for c in r.get("server_chunks", [])}
    ft = np.zeros(6); out = []

    def crate_sd(pts_r):
        res = []
        for c, rad in pts_r:
            res.append(min(obb_sd(c, d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]) for g in crate_g) - rad)
        return res
    for n in range(seg["n"]):
        f = dd / f"chunk_{i0 + n:05d}.npz"; z = np.load(f, allow_pickle=True); s = json.loads(str(z["summary_json"])); seq = int(z["t_step"])
        p = plan[seq]; g = s.get("grasp") or {}; ev = lit(g.get("evidence")) or {}; sub = s.get("subtask") or {}; ef = s.get("exec_feedback") or {}
        to = s.get("to") or {}; met = to.get("metrics") or {}; mv = lit(met.get("max_violation_pair")) or {}
        cls = lit((s.get("verdict") or {}).get("classification")) or {}
        e = {"seq": seq, "t": p["t_step"], "grasp_state": g.get("state"), "state_before": g.get("state_before"), "attach": g.get("attach"),
             "detach": g.get("detach"), "transition": g.get("transition"), "rejected": g.get("rejected"), "attach_revoked": g.get("attach_revoked"),
             "placed": ev.get("placed"), "open_streak": ev.get("open_streak"), "opening_after": ev.get("opening_after"), "last_command": ev.get("last_command"),
             "subtask": sub.get("label"), "subtask_released": sub.get("released"), "placed_seen": sub.get("placed_seen"),
             "fb_n_exec": ef.get("n_exec"), "fb_hold_reason": ef.get("hold_reason"),
             "verdict": [{"kind": x.get("kind"), "action": x.get("action"), "detail": x.get("detail")} for x in ((s.get("verdict") or {}).get("reasons") or [])],
             "to_status": to.get("status"), "to_clearance_mm": None if met.get("clearance_m") is None else 1e3 * met["clearance_m"],
             "worst": {"query": mv.get("query"), "link": mv.get("link"), "obstacle": mv.get("obstacle"), "step": mv.get("step"), "tier": mv.get("tier"),
                       "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * mv["clearance_m"]},
             "collision_by_why": cls.get("collision_by_why"), "binding": (sc_an.get(seq) or {}).get("binding"),
             "attached": bool(lit(g.get("attached")))}
        A = np.asarray(p["actions"], float); Ar = np.asarray(p.get("actions_reference") or p["actions"], float)
        cs = [c for c in ctrl if c["chunk_seq"] == seq]
        e["grip_refined_rows0_7"] = [float(A[:8, gc].min()), float(A[:8, gc].max())]
        e["grip_reference_rows0_7"] = [float(Ar[:8, gc].min()), float(Ar[:8, gc].max())]
        e["grip_applied_mean"] = float(np.mean([c["applied_ctrl"]["gripper"][0 if side == "left" else 1] for c in cs])) if cs else None
        e["n_executed"] = sum(1 for c in cs if c.get("executed", True)); e["hold"] = e["n_executed"] < len(cs)
        d.qpos[:] = p["qpos"]; mujoco.mj_kinematics(m, d)
        e["apple_crate_gap_mm"] = 1e3 * min(float(mujoco.mj_geomDistance(m, d, g_apple, gg, 0.5, ft)) for gg in crate_g)
        Rc = d.xmat[cb].reshape(3, 3); e["apple_in_crate_mm"] = (1e3 * (Rc.T @ (d.xpos[ab] - d.xpos[cb]))).tolist()
        att = lit(g.get("attached"))
        if isinstance(att, dict):
            held = lit(att.get("held")) or {}; bc = lit(held.get("body_cover")) or {}
            fr = bc.get("frame") or {}; near = bool(fr.get("near_support"))
            sph = bc.get("lifted_parent_frame") if near else bc.get("body_parent_frame")
            if sph:
                loc = [(np.asarray(x["centre_m"], float), 1e-3 * float(x["radius_mm"])) for x in sph]
                e["held_query"] = {"set": "lifted" if near else "body", "radii_mm": [x["radius_mm"] for x in sph]}
                Re, pe = d.xmat[ee].reshape(3, 3).copy(), d.xpos[ee].copy()
                w_now = [(pe + Re @ c, rad) for c, rad in loc]
                e["held_query"]["crate_sd_mm_now"] = [1e3 * v for v in crate_sd(w_now)]
                e["held_query"]["centre_world_now"] = [w.tolist() for w, _ in w_now]
                e["held_query"]["bottom_z_now"] = [float(w[2] - rad) for w, rad in w_now]
                k = mv.get("step")
                if k is not None and 0 <= int(k) < len(A):
                    d.qpos[qi] = A[int(k), c0:c0 + 7]; mujoco.mj_kinematics(m, d)
                    Re, pe = d.xmat[ee].reshape(3, 3).copy(), d.xpos[ee].copy()
                    w_k = [(pe + Re @ c, rad) for c, rad in loc]
                    e["held_query"]["crate_sd_mm_worst_step"] = [1e3 * v for v in crate_sd(w_k)]
                    e["held_query"]["centre_world_worst_step"] = [w.tolist() for w, _ in w_k]
        out.append(e)
    return {"key": key, "side": side, "server_segment": seg, "chunks": out}


if __name__ == "__main__":
    a = json.load(open(AN)); outp = pathlib.Path(sys.argv[1])
    res = json.loads(outp.read_text()) if outp.exists() else {"definitions": __doc__, "runs": {}}
    for key in sys.argv[2:]:
        res["runs"][key] = one(key, a); print(key, len(res["runs"][key].get("chunks", [])), flush=True)
        outp.write_text(json.dumps(res, indent=1, default=float))
