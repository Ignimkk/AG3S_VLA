"""#4e MuJoCo-sim empty-hand close: sweep the step (within the chunk) at which cmd 1.0 -> 0.0 starts.
Also sweep close target (0.0 / 0.3 / 0.57). reach is forced to 0.042 m (isolates the opening evidence)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T22'))
import replay_t22 as R
import scenarios_t22  # noqa  (reuses fb/run_seq; running it again rewrites scenarios_t22.json identically)
from scenarios_t22 import run_seq

m = R._m
P, C, S = R.load_run("t17")
q0 = np.asarray(P[2]["qpos"], np.float64)
act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}


def simulate(cmd_chunks):
    d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_forward(m, d)
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i][0]]]
    chunks, trace = [], []
    for c in cmd_chunks:
        for u in c:
            d.ctrl[act["gripper_l_act"]] = u * -0.045
            for _ in range(33):
                mujoco.mj_step(m, d)
            trace.append(abs(d.qpos[R.GL]) / 0.045)
        chunks.append((c, abs(d.qpos[R.GL]) / 0.045, None))
    return chunks, trace


out = []
for target in (0.0, 0.3, 0.57):
    for s in range(8):
        c2 = [1.0] * s + [target] * (8 - s)
        chunks, trace = simulate([[1.0] * 8, c2, [target] * 8, [target] * 8])
        r = run_seq(f"empty close to {target} starting at step {s}", chunks)
        att = r["attach_req"]
        row = {"close_target": target, "start_step_in_chunk": s, "closing_steps_in_first_chunk": 8 - s,
               "opening_at_boundaries": [round(ch[1], 4) for ch in chunks], "attach_req": att,
               "attach_opening": None if not att else r["rows"][att[0]]["measured"],
               "states": [x["state"] for x in r["rows"]],
               "failed_checks": [[k for k, v in (x["checks"] or {}).items() if not v] for x in r["rows"]],
               "trace": [round(t, 4) for t in trace]}
        out.append(row)
        print(target, s, row["opening_at_boundaries"], "attach", att, row["attach_opening"], row["failed_checks"])
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T22/empty_sweep_t22.json'), "w"), indent=1)
