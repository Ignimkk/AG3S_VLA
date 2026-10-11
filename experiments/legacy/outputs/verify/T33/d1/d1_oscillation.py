"""T33 copy of T31-diag d1_oscillation.py (same metric code). RUNS: E2 (T28, 6), T30 E3a/E3b (all runs with grasp and place),
T33 E3a/E3b (all runs with grasp and place). Output outputs/verify/T33/d1/d1_oscillation.json.
--- original ---
T31-diag D1 — oscillation metrics on the EXECUTED joint trajectory, grasp -> place window.

usage: PYTHONPATH=outputs/verify/T31diag/snap .venv-ag3s/bin/python d1_oscillation.py

Runs: E2 (T28/E2_rec, 6 runs), E3a ep1807 r1 r2 r3, E3b ep1807 r1 r2 (T30).  E3a r2 is extra (task names r1, r3).
grasp_t / place_t : analyze_e0 rules (T28/E0/analyze_e0.py) — grasp = first planning row apple dz >= 50 mm;
                    place = first planning t of the final uninterrupted apple-in-crate streak.
release_t         : first control row t >= place_t with applied_ctrl.gripper[0] >= 0.5 (left gripper opening cmd).
Windows           : W_transport = control rows with grasp_t <= t < place_t ; W_place = place_t <= t <= release_t ;
                    W_all = grasp_t <= t <= release_t.
Signal            : control rows 'qpos' of the left arm (MuJoCo qpos adr 20..26 = left_arm_0..6), dt = 1/15 s (manifest ctrl_hz 15),
                    and separately the commanded 'applied_ctrl.arm' left 7 (same metrics, 'cmd_*').
Metrics (per window):
  sign_reversals      : per joint, velocity v_k = (q_{k+1}-q_k)/dt; count of k where sign(v_k) differs from the sign of the last
                        v with |v| > deadband (deadband 0.02 rad/s); summed over the 7 left-arm joints. Also per second of window.
  jerk_rms            : sqrt(mean over joints x steps of ((q_{k+3}-3q_{k+2}+3q_{k+1}-q_k)/dt^3)^2)  rad/s^3
  fingertip path      : midpoint of MuJoCo bodies ee_finger_l1 / ee_finger_l2 (FK of control-row qpos), world frame.
                        path_length_m, chord_m, path/chord ratio; turning: displacement vectors merged until >= 2 mm,
                        total turning angle (deg) between successive merged segments; mean curvature = total turning (rad) / path length (1/m);
                        n_turns_gt_90deg = merged-segment turns > 90 deg.
  hold                : control rows executed == False — intervals [t_first, t_last], resume t = t_last + 1.
  refined-reference   : planning rows in the window: max over left arm (16D cols 0-6) and executed rows 0-7 of |actions - actions_reference| deg,
                        and the same over both arms (cols 0-6, 8-14).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
import analyze_e0 as A

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/d1'))
DT = 1.0 / 15.0
LADR = list(range(20, 27))
DEADBAND = 0.02
RUNS = {}
for p in sorted(pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/E2_rec')).glob("E2_ep*")):
    if p.is_dir():
        RUNS[p.name] = ("E2", p)
for tag in ("T30", "T33"):
    for cond in ("E3a", "E3b"):
        for p in sorted(pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/{tag}/{cond}_rec").glob(f"{cond}_ep*")):
            if p.is_dir() and (p / "frames.jsonl").exists():
                RUNS[f"{tag}_{p.name}"] = (f"{tag}_{cond}", p)


def reversals(q, dt=DT, db=DEADBAND):
    v = np.diff(q, axis=0) / dt
    tot = 0; per = []
    for j in range(q.shape[1]):
        last = 0; n = 0
        for x in v[:, j]:
            if abs(x) <= db:
                continue
            s = 1 if x > 0 else -1
            if last and s != last:
                n += 1
            last = s
        per.append(n); tot += n
    return tot, per


def jerk_rms(q, dt=DT):
    if q.shape[0] < 4:
        return None, None
    j = (q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]) / dt ** 3
    return float(np.sqrt(np.mean(j ** 2))), np.sqrt(np.mean(j ** 2, axis=0)).tolist()


def path_metrics(P, seg_min=0.002):
    if len(P) < 3:
        return None
    d = np.diff(P, axis=0)
    L = float(np.linalg.norm(d, axis=1).sum())
    chord = float(np.linalg.norm(P[-1] - P[0]))
    segs = []; acc = np.zeros(3)
    for x in d:
        acc = acc + x
        if np.linalg.norm(acc) >= seg_min:
            segs.append(acc.copy()); acc = np.zeros(3)
    ang = []
    for a, b in zip(segs[:-1], segs[1:]):
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        ang.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    ang = np.array(ang)
    return {"path_length_m": L, "chord_m": chord, "path_over_chord": (L / chord) if chord > 1e-6 else None,
            "n_segments_2mm": len(segs), "total_turning_deg": float(ang.sum()) if ang.size else 0.0,
            "mean_curvature_per_m": float(np.radians(ang.sum()) / L) if L > 0 else None,
            "n_turns_gt_90deg": int((ang > 90).sum()), "n_turns_gt_45deg": int((ang > 45).sum()),
            "median_turn_deg": float(np.median(ang)) if ang.size else None}


def window_metrics(rows, m, d, fid):
    q = np.array([[r["qpos"][a] for a in LADR] for r in rows])
    cmd = np.array([r["applied_ctrl"]["arm"][:7] for r in rows])
    out = {"n_rows": len(rows), "t_first": rows[0]["t_step"] if rows else None, "t_last": rows[-1]["t_step"] if rows else None,
           "duration_s": len(rows) * DT}
    if len(rows) < 4:
        return out
    tot, per = reversals(q); ct, cper = reversals(cmd)
    jr, jper = jerk_rms(q); cjr, cjper = jerk_rms(cmd)
    P = []
    for r in rows:
        d.qpos[:len(r["qpos"])] = r["qpos"]; mujoco.mj_kinematics(m, d)
        P.append(0.5 * (d.xpos[fid[0]] + d.xpos[fid[1]]))
    P = np.array(P)
    out.update({"sign_reversals": tot, "sign_reversals_per_joint": per, "sign_reversals_per_s": tot / out["duration_s"],
                "cmd_sign_reversals": ct, "cmd_sign_reversals_per_joint": cper, "cmd_sign_reversals_per_s": ct / out["duration_s"],
                "jerk_rms": jr, "jerk_rms_per_joint": jper, "cmd_jerk_rms": cjr, "cmd_jerk_rms_per_joint": cjper,
                "fingertip": path_metrics(P), "fingertip_path_world": P.tolist()})
    return out


def main():
    m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
    fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("ee_finger_l1", "ee_finger_l2")]
    # check applied_ctrl.arm ordering against qpos (left first?)
    res = {"definitions": __doc__, "deadband_rad_s": DEADBAND, "dt_s": DT, "runs": {}}
    for name, (cond, rd) in RUNS.items():
        a = A.analyze(rd)
        rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
        ctrl = [r for r in rows if r["kind"] == "control"]
        plan = [r for r in rows if r["kind"] == "planning"]
        g, p = a["grasp_t"], a["place_t"]
        if g is None or p is None:
            res["runs"][name] = {"condition": cond, "run_dir": str(rd), "success": a["success"], "grasp_t": g, "place_t": p, "skipped": "no grasp or no place"}
            print(name, "skipped", g, p, flush=True)
            continue
        # ordering check: correlation of applied arm[:7] with qpos 20..26 over the run
        qa = np.array([[r["qpos"][x] for x in LADR] for r in ctrl]); ca = np.array([r["applied_ctrl"]["arm"] for r in ctrl])
        err_left_first = float(np.abs(ca[:, :7] - qa).mean()); err_right_first = float(np.abs(ca[:, 7:] - qa).mean())
        rel = next((r["t_step"] for r in ctrl if r["t_step"] >= p and r["applied_ctrl"]["gripper"][0] >= 0.5), None)
        W = {"transport": [r for r in ctrl if g <= r["t_step"] < p],
             "place": [r for r in ctrl if p <= r["t_step"] <= (rel if rel is not None else 10 ** 9)],
             "all": [r for r in ctrl if g <= r["t_step"] <= (rel if rel is not None else 10 ** 9)]}
        e = {"condition": cond, "run_dir": str(rd), "success": a["success"], "grasp_t": g, "place_t": p, "release_t": rel,
             "applied_arm_order_check_mean_abs_rad": {"left_first": err_left_first, "right_first": err_right_first},
             "windows": {k: window_metrics(v, m, d, fid) for k, v in W.items()}}
        # hold intervals (whole run and within window)
        holds = []; cur = None
        for r in ctrl:
            if r.get("executed") is False:
                if cur is None:
                    cur = [r["t_step"], r["t_step"], r["chunk_seq"]]
                else:
                    cur[1] = r["t_step"]
            elif cur is not None:
                holds.append({"t_first": cur[0], "t_last": cur[1], "resume_t": cur[1] + 1, "n_steps": cur[1] - cur[0] + 1,
                              "first_chunk_seq": cur[2]}); cur = None
        if cur is not None:
            holds.append({"t_first": cur[0], "t_last": cur[1], "resume_t": None, "n_steps": cur[1] - cur[0] + 1, "first_chunk_seq": cur[2]})
        e["hold_intervals"] = holds
        e["hold_steps_in_all_window"] = sum(1 for r in W["all"] if r.get("executed") is False)
        rr = []
        for pl in plan:
            if pl.get("actions") is None or pl.get("actions_reference") is None:
                continue
            ac = np.asarray(pl["actions"], float)[:8]; ar = np.asarray(pl["actions_reference"], float)[:8]
            dl = np.degrees(np.abs(ac[:, :7] - ar[:, :7])); db = np.degrees(np.abs(ac[:, list(range(7)) + list(range(8, 15))] - ar[:, list(range(7)) + list(range(8, 15))]))
            rr.append({"seq": pl["seq"], "t": pl["t_step"], "left_arm_max_deg": float(dl.max()), "both_arms_max_deg": float(db.max()),
                       "argmax_left_joint": int(dl.max(axis=0).argmax()), "per_left_joint_max_deg": dl.max(axis=0).tolist(),
                       "executed": bool(pl.get("safe")) if cond.endswith("E3b") else True,
                       "in_window": bool(g <= pl["t_step"] <= (rel if rel is not None else 10 ** 9))})
        e["refined_minus_reference"] = rr
        res["runs"][name] = e
        w = e["windows"]["all"]
        print(name, g, p, rel, "rev", w.get("sign_reversals"), round(w.get("sign_reversals_per_s", 0), 2), "jerk", round(w.get("jerk_rms") or 0, 2),
              "path/chord", w.get("fingertip", {}) and round(w["fingertip"]["path_over_chord"], 2), "curv", w.get("fingertip") and round(w["fingertip"]["mean_curvature_per_m"], 1),
              "holds", len(holds), "order", round(err_left_first, 4), round(err_right_first, 4), flush=True)
    (OUT / "d1_oscillation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
