"""T28 P1-1 detail: per control step left-finger->apple surface distance (MuJoCo geoms, witness-checked
mj_geomDistance, same helper as P1-2), applied left gripper cmd, measured opening. Adds per-run summary
fields to e0_summary.json (key `detail`)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/E0'))
m, d, probe, obj, fing = P.build()
S = json.loads((OUT / "e0_summary.json").read_text())
for name, r in S["runs"].items():
    rows = [json.loads(l) for l in (OUT / name / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [x for x in rows if x["kind"] == "control"]
    fa, gl_cmd, gl_meas, ts = [], [], [], []
    for c in ctrl:
        P.set_q(m, d, c["qpos"])
        fa.append(1000 * min(P.gdist(m, d, gf, ga, 1.0) for gf in fing for ga in obj["apple"]))
        gl_cmd.append(c["applied_ctrl"]["gripper"][0]); gl_meas.append(abs(c["qpos"][27]) / 0.045); ts.append(c["t_step"])
    fa = np.array(fa); gl_cmd = np.array(gl_cmd); gl_meas = np.array(gl_meas)
    close_t = next((t for t, g in zip(ts, gl_cmd) if g < 0.5), None)
    r["detail"] = {
        "t": ts, "finger_to_apple_surface_mm": fa.tolist(), "grip_l_applied": gl_cmd.tolist(), "grip_l_measured": gl_meas.tolist(),
        "min_finger_to_apple_surface_mm": float(fa.min()), "t_min_finger_to_apple": int(ts[int(fa.argmin())]),
        "t_first_finger_apple_contact_le_0mm": next((t for t, x in zip(ts, fa) if x <= 0), None),
        "t_first_applied_grip_l_lt_0p5": close_t,
        "finger_to_apple_mm_at_first_close": float(fa[ts.index(close_t)]) if close_t is not None else None,
        "min_grip_l_applied": float(gl_cmd.min()), "min_grip_l_measured": float(gl_meas.min()),
        "n_close_open_cycles_applied_lt_0p5": int(np.sum((gl_cmd[1:] < 0.5) & (gl_cmd[:-1] >= 0.5))),
    }
    print(name, {k: v for k, v in r["detail"].items() if not isinstance(v, list)})
S["n_invalid_geomdistance_pairs_skipped_detail"] = P.N_INVALID[0]
(OUT / "e0_summary.json").write_text(json.dumps(S, indent=1))
