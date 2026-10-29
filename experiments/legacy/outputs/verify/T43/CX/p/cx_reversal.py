"""T43 CX — where the palm reversals at chunk boundaries happen, V11AB vs V11ABC (verifier, CPU only, numbers only).
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 cx_reversal.py OUT_JSON

Same rule as V11/p/sharp_v11.py (unchanged): executed command u(t) = traj.npz executed_actions; boundary t0 = 8n (n >= 1); clean = rows t0-2 .. t0+1
executed; palm = EE_BODY_<side> origin with the 7 active-arm joints = u(t), others = control row 0 qpos; v0, v1, v2 = palm displacements
(t0-2 -> t0-1, t0-1 -> t0, t0 -> t0+1); turns (v0, v1), (v1, v2) where both |v| > 2 mm; reversal = any turn > 90 deg.
Added: per boundary t0, chunk index n = t0 / 8, |v0| |v1| |v2| (mm/row), turn angles, which pair reversed, phase, and the C row values of the command
across the boundary (max_j |u(t0) - 2u(t0-1) + u(t0-2)| and |u(t0+1) - 2u(t0) + u(t0-1)|, rad/row^2).
phase of t0 (server grasp.state of the chunk that starts at t0, analysis server_chunks; grasp_t / place_t of analysis):
  approach   t0 < first closing chunk t;  grasp  closing chunks and t0 < grasp_t + 8;  carry  grasp_t + 8 <= t0 and state held;
  after      everything later (placed / latched again / released)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

V11 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11'))
J14 = list(range(0, 7)) + list(range(8, 15))
LEFT_Q = list(range(20, 27)); RIGHT_Q = list(range(11, 18))


def one(a):
    rd = pathlib.Path(a["run_dir"]); ep = int(a["episode"]); side = "right" if ep == 1834 else "left"
    HM.set_arm(side); M = HM.model(); m, d = M["m"], M["d"]
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    z = np.load(rd / "traj.npz", allow_pickle=True); EA = np.asarray(z["executed_actions"], float); T = len(EA)
    ex = np.array([bool(ctrl[t].get("executed", True)) if t in ctrl else False for t in range(T)])
    qi = LEFT_Q if side == "left" else RIGHT_Q; c0 = 0 if side == "left" else 8
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_BODY_L" if side == "left" else "EE_BODY_R")
    q0 = np.asarray(ctrl[0]["qpos"], float); P = np.zeros((T, 3))
    for t in range(T):
        d.qpos[:] = q0; d.qpos[qi] = EA[t, c0:c0 + 7]; mujoco.mj_kinematics(m, d); P[t] = d.xpos[body]
    U = EA[:, J14]
    st = {c["t"]: c.get("grasp_state") for c in a.get("server_chunks") or [] if c.get("t") is not None}
    first_close = min((t for t, s in st.items() if s == "closing"), default=None)
    gt = a.get("grasp_t")
    out = []
    for t0 in range(8, T - 1, 8):
        if not ex[t0 - 2:t0 + 2].all():
            continue
        Pe = P[t0 - 2:t0 + 2]; v = np.diff(Pe, axis=0); nv = np.linalg.norm(v, axis=1)
        tn = []
        for i, (x, y) in enumerate(((v[0], v[1]), (v[1], v[2]))):
            tn.append(float(np.degrees(np.arccos(np.clip(x @ y / (nv[i] * nv[i + 1]), -1, 1)))) if nv[i] > 2e-3 and nv[i + 1] > 2e-3 else None)
        s = st.get(t0)
        if first_close is None or t0 < first_close:
            ph = "approach" if (gt is None or t0 < gt) else "after"
        elif s == "closing" or (gt is not None and t0 < gt + 8):
            ph = "grasp"
        elif s == "held":
            ph = "carry"
        else:
            ph = "after"
        e2 = U[t0 - 2:t0 + 2]
        out.append({"t0": t0, "n": t0 // 8, "phase": ph, "state": s, "v_mm": (1e3 * nv).tolist(), "turn_deg": tn,
                    "reversal": bool(any(x is not None and x > 90 for x in tn)), "rev_pair": [bool(x is not None and x > 90) for x in tn],
                    "c_row_rad": [float(np.abs(e2[2] - 2 * e2[1] + e2[0]).max()), float(np.abs(e2[3] - 2 * e2[2] + e2[1]).max())]})
    return out


def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"n": int(len(x))} if not len(x) else {"n": int(len(x)), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def main():
    res = {"definitions": __doc__, "arms": {}}
    for arm in ("V11AB", "V11ABC"):
        R = json.load(open(V11 / f"analysis/analysis_{arm}.json"))["runs"]; per = {}
        for key in sorted(R):
            per[key] = one(R[key]); print(key, len(per[key]), sum(b["reversal"] for b in per[key]), flush=True)
        allb = [(k, b) for k, L in per.items() for b in L]
        summ = {"n_boundaries": len(allb), "n_reversal": sum(b["reversal"] for _, b in allb)}
        for ph in ("approach", "grasp", "carry", "after"):
            sel = [b for _, b in allb if b["phase"] == ph]
            rv = [b for b in sel if b["reversal"]]
            summ[ph] = {"n": len(sel), "n_reversal": len(rv), "share": len(rv) / len(sel) if sel else None,
                        "rev_vmin_mm": q([min(b["v_mm"]) for b in rv]), "rev_vmax_mm": q([max(b["v_mm"]) for b in rv]),
                        "rev_turn_deg": q([max(x for x in b["turn_deg"] if x is not None) for b in rv]),
                        "c_row_rad_rev": q([max(b["c_row_rad"]) for b in rv]), "c_row_rad_all": q([max(b["c_row_rad"]) for b in sel])}
        for stg in ("H", "B"):
            sel = [b for k, b in allb if k.split("/")[1] == stg]
            summ[stg] = {"n": len(sel), "n_reversal": sum(b["reversal"] for b in sel)}
        rv = [b for _, b in allb if b["reversal"]]
        summ["rev_by_chunk_n"] = {str(n): sum(1 for b in rv if b["n"] == n) for n in sorted({b["n"] for b in rv})}
        summ["bnd_by_chunk_n"] = {str(n): sum(1 for _, b in allb if b["n"] == n) for n in sorted({b["n"] for _, b in allb})}
        summ["rev_vmin_lt5mm"] = sum(1 for b in rv if min(b["v_mm"]) < 5.0)
        summ["rev_vmin_mm"] = q([min(b["v_mm"]) for b in rv])
        summ["per_run"] = {k: {"n": len(L), "n_reversal": sum(b["reversal"] for b in L),
                               "by_phase": {ph: [sum(1 for b in L if b["phase"] == ph), sum(1 for b in L if b["phase"] == ph and b["reversal"])] for ph in ("approach", "grasp", "carry", "after")}}
                           for k, L in per.items()}
        res["arms"][arm] = {"summary": summ, "per_run": per}
        print(arm, {k: v for k, v in summ.items() if k not in ("per_run", "rev_by_chunk_n", "bnd_by_chunk_n")}, flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
