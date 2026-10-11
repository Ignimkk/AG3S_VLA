from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, types, numpy as np
from AG3S.trajopt.experiments.ta_offline import Run, variant_config
from AG3S.trajopt.refiner import TrajOptChunkRefiner
from AG3S.trajopt.problem import objective
idx = json.load(open(_ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json')))["runs"]
key, seq = sys.argv[1], int(sys.argv[2]); vs = sys.argv[3:]
run = Run(key, idx[key], types.SimpleNamespace(hm=False, no_plane_rows=False))
inp, _ = run.inputs(seq); s = inp["summary"]; fb = inp["feedback"]
with np.load(run.chunks[seq - 1], allow_pickle=True) as pz:
    prev = np.asarray(pz["refined_chunk"], np.float64)
ctx0 = {"t_step": seq, "previous_physical_chunk": prev, "previous_executed_steps": int(fb["n_exec"])}
for v in vs:
    name, _, extra = v.partition("@")
    cfg = variant_config(run.base_cfg, name, False); holder = {}
    if extra == "eps5": cfg = cfg.with_overrides({"qp": {"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 40000}})
    if extra == "eps4": cfg = cfg.with_overrides({"qp": {"eps_abs": 1e-4, "eps_rel": 1e-4, "max_iter": 20000}})
    if extra == "polish": cfg = cfg.with_overrides({"qp": {"polish": True}})
    ref = TrajOptChunkRefiner(run.robot, run.layout, lambda c: holder["v"], cfg)
    H = cfg.horizon.planned
    ref.optimizer.linearizer.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], H))
    holder["v"] = (inp["scene"], inp["q_now"], True)
    import time; t0=time.process_time(); ref.refine(inp["ref_chunk"], dict(ctx0)); print("cpu_ms", round((time.process_time()-t0)*1e3,1), "qp_ms", round(ref.last_result.metrics.get("timing_ms",{}).get("qp",0),1) if isinstance(ref.last_result.metrics.get("timing_ms"),dict) else None)
    r = ref.last_result; m = r.metrics
    R = run.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :H]
    D = r.trajectory - R
    print(f"== {v}: init merit {m['initial_merit']:.5f} viol {m['initial_violation_m']} returned {m['returned']}")
    for c in m["candidates"]: print("   cand", {k: (round(x, 5) if isinstance(x, float) else x) for k, x in c.items()})
    print("   |D| per step (deg max):", np.degrees(np.abs(D).max(axis=0)).round(2).tolist())
    print("   lookahead", {k: x for k, x in (m.get("lookahead") or {}).items()})
    print("   cost", round(r.cost, 5), "track", round(float(np.sum(D**2)), 5), "continuity_active", m.get("continuity_active"))
