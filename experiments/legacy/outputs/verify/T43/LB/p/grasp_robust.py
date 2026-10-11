"""T42-1 grasp robustness per run (verifier, 2026-10-05; user instruction 2026-10-05: 'pick must succeed cleanly, no slip').
CPU only. Model = ${AG3S_PROJECT_ROOT}-t42 model_transport.xml (the T42-1 client model); distances = mj_geomDistance kept only if the T40 H
certificate passes (snapshot work-t42c rby1_manipulation, convex_distance.py md5-identical to work-t42b.simulation.convex_distance.certify), else the exact gjk_distance
(the manager._checked_distance rule).

Per run (recorded control rows; target / arm from the manifest; arm gripper = applied_ctrl gripper[0 left | 1 right], 1 = open):
  close cycles     : number of transitions of the applied arm gripper from >= 0.9 (open) to < 0.9 (a close command; the holding
                     command while a fruit is held is ~0.57) over the whole run;
                     n_close_before_lift = those at t <= lift_t; regrasp = n_close_before_lift >= 2
  lift_t           : first control row after the last row t <= grasp_t whose target centre z < z0 + 2 mm (T40 A rule; grasp_t =
                     analyze_t42 planning rule)
  release_t        : first control row t > lift_t whose applied arm gripper >= 0.9 (an open command)
  hold window      : lift_t <= t < min(release_t, place_t) (place_t = analyze_t42 planning t of the final in-crate streak; when the
                     run has no place, up to release_t or the end)
  drift_mm         : max over the window of |p(t) - p(lift_t)|, p = target centre in the frame of body EE_BODY_<arm> (slip of the
                     fruit relative to the hand)
  finger_gap_mm    : per row min over the two finger bodies (ee_finger_<arm>1/2, collision geoms) of the target-to-finger distance;
                     reported: max over the window; n_rows_no_finger_within_3mm = rows where both fingers are > 3 mm from the target
  dz_drop_mm       : max over the window of (running max of target z) - target z (a fall while carried)
  slip             : drift_mm > 15 or n_rows_no_finger_within_3mm > 0. (First version also had dz_drop_mm > 20; removed after the
                     first evaluation of the p1/p2a runs: the hand lowers the held fruit into the crate inside the window, 129-153 mm
                     in runs whose fingers never left the fruit, so dz_drop is reported only.)
  robust_grasp_strict (A) : grasp and not slip and not regrasp
  (B, primary; lead decision 2026-10-05): the first slip event (a finger loss, or drift > 15 mm -- the fruit starts sliding a few
                     rows before the fingers are 3 mm away) counts as place, not slip, when at that row the target centre is
                     inside the crate interior footprint (crate-local |x| < 0.077, |y| < 0.137, crate pose = that control row's qpos)
                     AND above the crate rim or already inside the crate (crate-local z > -0.052, the success rule's lower bound; the
                     lead's wording was 'above the rim' -- a loss with the fruit already inside the crate volume, e.g. 1977 s3 at
                     local z 0.023, is counted as place too) AND the run ends with the target in the crate (analyze_t42 place). Rows from that first exempt loss on are not checked. slip_B = drift (rows before the
                     exempt loss) > 15 mm or a finger loss that is not exempt.
  robust_grasp     : (B) grasp and not slip_B and not regrasp
  first_slip_t / first_slip_xy_from_reset_m / first_slip_z_above_table_m : the first slip event's row, the target's xy distance
                     from its control-row-0 position and its height above the table (used by failure_mode)
failure_mode(r)    : no_grasp (grasp false) | regrasp (n_close_before_lift >= 2) | slip_at_lift (first non-exempt slip event with
                     the target <= 0.05 m (xy) from its start) | slip_in_transport (other non-exempt slip event) | not_placed
                     (grasp, no slip, place false) | ok
usage: grasp_robust.py OUT_JSON RUN_DIR [RUN_DIR ...]   (or import: robust(run_dir))"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t42c/integration/rby1/manipulation/src")
from rby1_manipulation.simulation.convex_distance import GeomSupport, certify, gjk_distance  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402

OPEN = 0.9   # applied gripper >= 0.9 = open command (holding command while grasping is ~0.57, closing 0.0)
XML = "${AG3S_PROJECT_ROOT}-t42/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
_M = {}


def model():
    if not _M:
        m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        coll = lambda b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == B(b) and (m.geom_contype[g] or m.geom_conaffinity[g])]
        _M.update(m=m, d=d, B=B, coll=coll, sup=GeomSupport(m, d))
    return _M


def dist(g1, g2, cap=0.05):
    M = model(); m, d = M["m"], M["d"]; ft = np.zeros(6)
    raw = float(mujoco.mj_geomDistance(m, d, g1, g2, cap, ft))
    if raw >= cap:
        return cap
    if certify(M["sup"], g1, g2, raw, ft):
        return raw
    ex = gjk_distance(M["sup"], g1, g2)
    return min(raw, 0.0) if ex <= 0 else min(cap, ex)


def robust(rd):
    rd = pathlib.Path(rd); a = AN.analyze(rd)
    M = model(); m, d = M["m"], M["d"]
    tf, arm = a["target_fruit"], a["used_arm"]; gi = 0 if arm == "left" else 1; s = "L" if arm == "left" else "R"
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
    adr = AN.QADR[tf]; z0 = plan[0]["object_poses"][tf]["pos"][2]
    t = np.array([c["t_step"] for c in ctrl]); Q = np.array([c["qpos"] for c in ctrl]); z = Q[:, adr + 2]
    g = np.array([((c.get("applied_ctrl") or {}).get("gripper") or [1, 1])[gi] for c in ctrl], float)
    closes = [int(t[i]) for i in range(1, len(g)) if g[i - 1] >= OPEN and g[i] < OPEN] + ([int(t[0])] if g[0] < OPEN else [])
    out = {"run_dir": str(rd), "target_fruit": tf, "used_arm": arm, "success": a["success"], "grasp": a["grasp"], "place": a["place"],
           "grasp_t": a["grasp_t"], "place_t": a["place_t"], "close_cmd_t": sorted(closes), "n_close_cycles": len(closes)}
    lift_t = None
    if a["grasp_t"] is not None:
        k = np.where((t <= a["grasp_t"]) & (z < z0 + 0.002))[0]
        lift_t = int(t[k[-1] + 1]) if len(k) and k[-1] + 1 < len(t) else None
    out["lift_t"] = lift_t
    out["n_close_before_lift"] = None if lift_t is None else sum(1 for c in closes if c <= lift_t)
    out["regrasp"] = None if lift_t is None else out["n_close_before_lift"] >= 2
    if lift_t is None:
        out.update(release_t=None, slip=None, slip_B=None, robust_grasp=False, robust_grasp_strict=False); return out
    rel = np.where((t > lift_t) & (g >= OPEN))[0]
    release_t = int(t[rel[0]]) if len(rel) else None
    end = min(x for x in (release_t, a["place_t"], int(t[-1]) + 1) if x is not None)
    w = np.where((t >= lift_t) & (t < end))[0]
    out["release_t"] = release_t; out["hold_window"] = [lift_t, end]; out["n_hold_rows"] = int(len(w))
    ee = M["B"](f"EE_BODY_{s}"); tb = M["B"](tf)
    fing = [M["coll"](f"ee_finger_{s.lower()}1"), M["coll"](f"ee_finger_{s.lower()}2")]; tg = M["coll"](tf)
    loc, gap, zz = [], [], []
    for i in w:
        d.qpos[:] = Q[i]; mujoco.mj_kinematics(m, d)
        R = d.xmat[ee].reshape(3, 3); loc.append(R.T @ (d.xpos[tb] - d.xpos[ee]))
        gap.append(min(min(dist(x, y) for x in tg for y in f) for f in fing)); zz.append(Q[i, adr + 2])
    loc = np.array(loc); gap = np.array(gap); zz = np.array(zz)
    drift = np.linalg.norm(loc - loc[0], axis=1) if len(loc) else np.zeros(0)
    drop = (np.maximum.accumulate(zz) - zz) if len(zz) else np.zeros(0)
    out["drift_mm_max"] = float(1e3 * drift.max()) if len(drift) else None
    out["drift_mm_final"] = float(1e3 * drift[-1]) if len(drift) else None
    out["finger_gap_mm_max"] = float(1e3 * gap.max()) if len(gap) else None
    out["n_rows_no_finger_within_3mm"] = int((gap > 0.003).sum())
    out["first_row_no_finger_within_3mm_t"] = int(t[w[np.argmax(gap > 0.003)]]) if (gap > 0.003).any() else None
    out["dz_drop_mm_max"] = float(1e3 * drop.max()) if len(drop) else None
    out["slip"] = bool((out["drift_mm_max"] or 0) > 15 or out["n_rows_no_finger_within_3mm"] > 0)
    out["robust_grasp_strict"] = bool(a["grasp"] and not out["slip"] and not out["regrasp"])
    # (B) exemption: first finger loss over the crate interior, above the rim, and the run ends placed
    lost = np.flatnonzero((gap > 0.003) | (drift > 0.015)); cadr = AN.QADR["crate"]   # first slip event: finger loss or drift > 15 mm
    exempt_i = None
    if len(lost):
        i0 = int(lost[0]); q = Q[w[i0]]
        ic, loc = AN.in_crate(q[adr:adr + 3], q[cadr:cadr + 3], q[cadr + 3:cadr + 7])
        over = abs(loc[0]) < 0.077 and abs(loc[1]) < 0.137 and loc[2] > -0.052   # over the interior: above the rim or inside the crate
        out["first_loss_crate_local"] = [float(v) for v in loc]; out["first_loss_over_crate_above_rim"] = bool(over)
        out["first_slip_t"] = int(t[w[i0]]); out["first_slip_z_above_table_m"] = float(q[adr + 2] - 0.82)
        out["first_slip_xy_from_reset_m"] = float(np.linalg.norm(q[adr:adr + 2] - Q[0, adr:adr + 2]))
        if over and a["place"]:
            exempt_i = i0
    upto = exempt_i if exempt_i is not None else len(w)
    out["loss_exempt_as_place"] = exempt_i is not None
    out["drift_mm_max_B"] = float(1e3 * drift[:upto].max()) if upto > 0 else 0.0
    out["slip_B"] = bool(out["drift_mm_max_B"] > 15 or (len(lost) > 0 and exempt_i is None))
    out["robust_grasp"] = bool(a["grasp"] and not out["slip_B"] and not out["regrasp"])
    out["series"] = {"t": t[w].tolist(), "drift_mm": (1e3 * drift).tolist(), "finger_gap_mm": (1e3 * gap).tolist()}
    return out


def failure_mode(r):
    if not r.get("grasp"):
        return "no_grasp"
    if r.get("regrasp"):
        return "regrasp"
    if r.get("slip_B"):
        x = r.get("first_slip_xy_from_reset_m")
        return "slip_at_lift" if (x is not None and x <= 0.05) else "slip_in_transport"
    if not r.get("place"):
        return "not_placed"
    return "ok"


if __name__ == "__main__":
    res = {"definitions": __doc__, "runs": {}}
    for x in sys.argv[2:]:
        r = robust(x); res["runs"][pathlib.Path(x).name] = r
        print(pathlib.Path(x).name, r["target_fruit"], r["used_arm"], "S" if r["success"] else "-", "closes", r["close_cmd_t"], "lift", r["lift_t"],
              "rel", r.get("release_t"), "drift", None if r.get("drift_mm_max") is None else round(r["drift_mm_max"], 1),
              "gapmax", None if r.get("finger_gap_mm_max") is None else round(r["finger_gap_mm_max"], 1), "noF", r.get("n_rows_no_finger_within_3mm"),
              "drop", None if r.get("dz_drop_mm_max") is None else round(r["dz_drop_mm_max"], 1), "slip", r.get("slip"), "slipB", r.get("slip_B"), "strict", r.get("robust_grasp_strict"), "robust(B)", r["robust_grasp"], flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
