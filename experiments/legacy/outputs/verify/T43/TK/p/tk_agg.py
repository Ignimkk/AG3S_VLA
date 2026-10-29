"""T43 TK aggregate of raw/tk_{V7g,V8,E0}.json -> raw/tk_summary.json (numbers only).

usage: tk_agg.py
Row sets: exec rows (cls exec) unless stated; 'other' = before + after phases.  |.| per joint = absolute value (rad);
EE values in mm.  Moving rows for the time offset: speed_cmd_mm >= 3 mm/row and t >= 7.
cross_track_mm = d_at_tau_ee_mm = min over tau in [0, 6] rows of |EE(M(t)) - EE_C(t - tau)| (distance of M from the interpolated command path).
clear_row_k_link_mm = record 'clearance' (safe_policy.py:1266-1275: cs.esdf.distance at the sphere centres of the TO result trajectory
  = refined, minus radius, minus esdf_margin) at row k, min over the contacting link's spheres; esdf_link_minus_geom_P_mm = that - bar_P_mm.
by_k_exec: rows t with t, t-1, t-2 all exec; speed = |EE_C(t) - EE_C(t-1)|; cos = cosine between EE_C(t)-EE_C(t-1) and EE_C(t-1)-EE_C(t-2)
  (computed when both > 0.1 mm); cmd_accel = |(EE_C(t)-EE_C(t-1)) - (EE_C(t-1)-EE_C(t-2))|.  k = 0 is the first row of a new chunk.
Speed bins on speed_cmd_mm (EE command speed, mm per control row of 0.066 s).
contact_runs = runs with >= 1 CT contact (any phase); clean_runs = H-stage runs with none (+ B-stage runs, no hurdle).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np

R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/TK/raw'))
BINS = [0, 1, 3, 10, 20, 30, 45, 1e9]


def st(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 4), "p90": round(float(np.percentile(x, 90)), 4),
            "max": round(float(x.max()), 4)}


def joint_st(rows, key):
    A = np.abs(np.asarray([r[key] for r in rows if r.get(key) is not None], float))
    if len(A) == 0:
        return None
    return {"median": np.round(np.median(A, 0), 4).tolist(), "p90": np.round(np.percentile(A, 90, 0), 4).tolist()}


def block(rows):
    mv = [r for r in rows if (r["speed_cmd_mm"] or 0) >= 3 and r["t"] >= 7]
    return {"n_rows": len(rows),
            "PC_mm": st([r["PC_mm"] for r in rows]), "CM_mm": st([r["CM_mm"] for r in rows]), "PM_mm": st([r["PM_mm"] for r in rows]),
            "PC_rad_joint": joint_st(rows, "PC_rad"), "CM_rad_joint": joint_st(rows, "CM_rad"),
            "CM_rad_maxjoint": st([float(np.max(np.abs(r["CM_rad"]))) for r in rows]),
            "speed_cmd_mm": st([r["speed_cmd_mm"] for r in rows]),
            "tau_ee_moving_steps": st([r["tau_ee"] for r in mv]), "tau_j_moving_steps": st([r["tau_j"] for r in mv]),
            "d_at_tau_ee_moving_mm": st([r["d_at_tau_ee_mm"] for r in mv]),
            "RP_mm_refined_vs_reference": st([r["RP_mm"] for r in rows])}


def main():
    out = {"definitions": __doc__, "arms": {}}
    allc = []
    for arm in ("V7g", "V8", "E0"):
        J = json.load(open(R / f"tk_{arm}.json")); runs = J["runs"]
        A = {"n_runs": len(runs), "model": J.get("model")}
        rows_all = [r for k, v in runs.items() for r in v["rows"]]
        ex = [r for r in rows_all if r["cls"] == "exec"]
        A["exec"] = {"all": block(ex), "carry": block([r for r in ex if r["phase"] == "carry"]),
                     "other": block([r for r in ex if r["phase"] != "carry"])}
        hold = [r for r in rows_all if r["cls"] == "hold"]
        A["hold"] = {"all": block(hold), "carry": block([r for r in hold if r["phase"] == "carry"])}
        A["n_rows_by_cls"] = {c: sum(r["cls"] == c for r in rows_all) for c in ("exec", "hold", "stop")}
        # speed bins (exec rows)
        sb = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            rr = [r for r in ex if r["speed_cmd_mm"] is not None and lo <= r["speed_cmd_mm"] < hi]
            sb.append({"bin_mm_per_row": [lo, hi if hi < 1e8 else None], "n": len(rr), "CM_mm": st([r["CM_mm"] for r in rr]),
                       "CM_over_speed": st([r["CM_mm"] / r["speed_cmd_mm"] for r in rr if r["speed_cmd_mm"] > 0.5]),
                       "tau_ee": st([r["tau_ee"] for r in rr if r["t"] >= 7]),
                       "CM_rad_joint": joint_st(rr, "CM_rad")})
        A["speed_bins_exec"] = sb
        # by step-in-chunk k (exec rows whose previous two rows are exec): command step size, C-M, reversal of the command direction
        byk = []
        for run in runs.values():
            rw = run["rows"]
            for t in range(2, len(rw)):
                if all(rw[x]["cls"] == "exec" for x in (t, t - 1, t - 2)):
                    a1 = np.subtract(rw[t]["EE_C"], rw[t - 1]["EE_C"]); a0 = np.subtract(rw[t - 1]["EE_C"], rw[t - 2]["EE_C"])
                    n1, n0 = np.linalg.norm(a1), np.linalg.norm(a0)
                    cosv = float(a1 @ a0 / (n1 * n0)) if n1 > 1e-4 and n0 > 1e-4 else None
                    byk.append((rw[t]["k"], rw[t]["phase"], 1e3 * n1, rw[t]["CM_mm"], cosv, 1e3 * float(np.linalg.norm(a1 - a0))))
        A["by_k_exec"] = {}
        for ph in ("carry", "all"):
            A["by_k_exec"][ph] = [{"k": k, "n": len(v), "speed_cmd_mm": st([x[2] for x in v]), "CM_mm": st([x[3] for x in v]),
                                   "cos_cmd_dir_vs_prev": st([x[4] for x in v]),
                                   "frac_cos_lt_0": (None if not [x for x in v if x[4] is not None] else
                                                     round(float(np.mean([x[4] < 0 for x in v if x[4] is not None])), 4)),
                                   "cmd_accel_mm_per_row2": st([x[5] for x in v])}
                                  for k in range(8) for v in [[x for x in byk if x[0] == k and (ph == "all" or x[1] == ph)]]]
        # run lags
        A["run_lag_tau_steps"] = st([v["run_lag"]["tau_steps"] for v in runs.values() if v["run_lag"]])
        A["run_lag_median_mm_at_tau"] = st([v["run_lag"]["median_mm_at_tau"] for v in runs.values() if v["run_lag"]])
        A["run_lag_median_mm_at_0"] = st([v["run_lag"]["median_mm_at_0"] for v in runs.values() if v["run_lag"]])
        # contact vs clean runs (carry exec)
        crun = {k for k, v in runs.items() if v["contacts"]}
        A["contact_runs"] = sorted(crun)
        A["carry_exec_contact_runs"] = block([r for k, v in runs.items() if k in crun for r in v["rows"] if r["cls"] == "exec" and r["phase"] == "carry"])
        A["carry_exec_clean_runs"] = block([r for k, v in runs.items() if k not in crun for r in v["rows"] if r["cls"] == "exec" and r["phase"] == "carry"])
        # q_now
        qn = [q for v in runs.values() for q in v["qnow"]]
        A["qnow"] = {
            "n_chunks": len(qn),
            "cap_minus_pre0_rad_max": max([q["cap_minus_pre0_rad_max"] for q in qn if q.get("cap_minus_pre0_rad_max") is not None], default=None),
            "cap_minus_Mprev_rad_max": max([q.get("cap_minus_Mprev_rad_max", 0) for q in qn], default=None),
            "cap_minus_infstate_rad_max": max([q["cap_minus_infstate_rad_max"] for q in qn if q.get("cap_minus_infstate_rad_max") is not None], default=None),
            "qnow_minus_Cprev_mm_prev_exec": {ph: st([q["qnow_minus_Cprev_mm"] for q in qn if "qnow_minus_Cprev_mm" in q and q["prev_row_cls"] == "exec"
                                                       and (q["phase"] == "carry") == (ph == "carry")]) for ph in ("carry", "other")},
            "qnow_minus_Cprev_mm_prev_hold": st([q["qnow_minus_Cprev_mm"] for q in qn if "qnow_minus_Cprev_mm" in q and q["prev_row_cls"] == "hold"]),
            "qnow_minus_Cprev_rad_joint_prev_exec": joint_st([q for q in qn if "qnow_minus_Cprev_rad" in q and q["prev_row_cls"] == "exec"], "qnow_minus_Cprev_rad"),
            "qnow_minus_Pprev_mm_prev_exec": st([q["qnow_minus_Pprev_mm"] for q in qn if "qnow_minus_Pprev_mm" in q and q["prev_row_cls"] == "exec"]),
            "P0_minus_qnow_rad_max": st([q.get("P0_minus_qnow_rad_max") for q in qn]),
            "R0_minus_qnow_rad_max": st([q.get("R0_minus_qnow_rad_max") for q in qn]),
            "refined_vs_EA_max": max([q.get("refined_vs_EA_max", 0) for q in qn], default=None),
            "refined_vs_traj_predicted_max": max([q.get("refined_vs_traj_predicted_max", 0) for q in qn], default=None),
            "feedback_applied_vs_EA_max": max([q.get("feedback_applied_vs_EA_max", 0) for q in qn], default=None),
            "n_feedback_checked": sum("feedback_applied_vs_EA_max" in q for q in qn),
            "n_refined_checked": sum("refined_vs_EA_max" in q for q in qn)}
        cons = [c for v in runs.values() for c in v["contacts"]]
        for c in cons:
            c["arm"] = arm
            rw = runs[c["key"]]["rows"][c["t"]]
            c["cross_track_mm"] = rw["d_at_tau_ee_mm"]
            c["esdf_link_minus_geom_P_mm"] = (None if c.get("clear_row_k_link_mm") is None or c.get("bar_P_mm") is None
                                              else c["clear_row_k_link_mm"] - c["bar_P_mm"])
        allc += cons
        out["arms"][arm] = A
    keep = ["arm", "key", "kind", "body", "phase_ct", "t", "k", "seq", "row_cls", "ct_max_pen_mm", "bar_P_mm", "bar_C_mm", "bar_M_mm", "bar_M_pre_mm",
            "bar_Cm1_mm", "bar_Cm2_mm", "bar_Cm3_mm", "bar_Prows_min_mm", "bar_Prows_argmin_t", "PC_mm", "CM_mm", "PM_mm", "tau_ee", "cross_track_mm", "esdf_link_minus_geom_P_mm", "speed_cmd_mm",
            "to_status", "to_clearance_mm", "clear_row_k_mm", "clear_row_k_link_mm", "clear_chunk_link_mm", "max_violation_pair", "post_src"]
    out["contacts"] = [{k: c.get(k) for k in keep} for c in allc]
    grp = {}
    for c in allc:
        g = f"{c['arm']}|{c['kind']}|{c['phase_ct']}"
        grp.setdefault(g, []).append(c)
    out["contact_groups"] = {g: {"n": len(v), **{f: st([c.get(f) for c in v]) for f in
                                 ("bar_P_mm", "bar_C_mm", "bar_M_mm", "bar_Cm2_mm", "bar_Prows_min_mm", "CM_mm", "PC_mm", "tau_ee", "speed_cmd_mm",
                                  "to_clearance_mm", "clear_row_k_mm", "clear_row_k_link_mm", "esdf_link_minus_geom_P_mm", "cross_track_mm", "bar_M_pre_mm")}} for g, v in grp.items()}
    json.dump(out, open(R / "tk_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
