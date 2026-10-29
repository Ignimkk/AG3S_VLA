"""Base re-solve bytes: the same reconstructed inputs through whichever `benchmark` is on PYTHONPATH."""
import json, sys, types, hashlib, numpy as np
from AG3S.trajopt.experiments.ta_offline import Run, K_EXEC
from AG3S.trajopt.refiner import TrajOptChunkRefiner
import AG3S.trajopt.sqp as S
print("sqp from", S.__file__, flush=True)
idx, key, out, with_hist = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
entry = json.load(open(idx))["runs"][key]
run = Run(key, entry, types.SimpleNamespace(hm=False, no_plane_rows=False))
holder = {}
ref = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["v"], run.base_cfg)
res = {}
prev_seq = None
for seq in sorted(run.chunks):
    inp, why = run.inputs(seq)
    if inp is None:
        prev_seq = None; continue
    s, fb = inp["summary"], inp["feedback"]
    ctx = {"t_step": seq}
    if prev_seq == seq - 1:
        with np.load(run.chunks[prev_seq], allow_pickle=True) as pz:
            ctx["previous_physical_chunk"] = np.asarray(pz["refined_chunk"], np.float64)
    if fb.get("available") and fb.get("n_exec") is not None:
        ctx["previous_executed_steps"] = int(fb["n_exec"])
    if with_hist and fb.get("applied_arm") is not None:
        arm = np.asarray(fb["applied_arm"], np.float64)
        if arm.ndim == 2 and arm.shape[0] >= 2:
            ctx["command_history"] = np.ascontiguousarray(arm[-2:].T)
            ctx["deflection_history"] = arm[-1] * 0.0 + 0.01
    ref.reset()
    ref.optimizer.linearizer.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], ref.optimizer.horizon))
    holder["v"] = (inp["scene"], inp["q_now"], True)
    chunk = ref.refine(inp["ref_chunk"], ctx)
    r = ref.last_result
    res[str(seq)] = {"chunk": hashlib.sha1(np.ascontiguousarray(chunk).tobytes()).hexdigest(),
                     "traj": hashlib.sha1(np.ascontiguousarray(r.trajectory).tobytes()).hexdigest(),
                     "status": r.status.value, "viol": float(r.max_violation), "cost": float(r.cost),
                     "keys": sorted(k for k in r.metrics if k in ("lookahead", "servo_accel", "deflection_rate"))}
    inp["z"].close(); prev_seq = seq
json.dump(res, open(out, "w"))
print(key, len(res), "chunks", flush=True)
