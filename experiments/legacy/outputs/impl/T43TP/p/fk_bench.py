"""T43 TP — micro-benchmark of the FK pieces at H = 8 / 16 (the V8 server robot model)."""
import json, time, sys
import numpy as np
from AG3S.trajopt.experiments import ta_offline as TA
import argparse
key = "V8/H/V8_ep1807_s18071"
entry = json.load(open("outputs/verify/T43/V8/analysis/analysis.json"))["runs"][key]
run = TA.Run(key, entry, argparse.Namespace(hm=False, no_plane_rows=False))
from AG3S.trajopt.linearize import CollisionLinearizer
inp, _ = run.inputs(25)
q_now = inp["q_now"]
ref = run.layout.chunk_to_trajectory(inp["ref_chunk"])
for H in (8, 16):
    lin = CollisionLinearizer(run.robot, run.layout, H)
    traj = ref[:, :H]
    full = run.layout.full_q(traj, q_now)
    args = lin._fk_args(full)
    def t(fn, n=30):
        fn(); ts=[]
        for _ in range(n):
            a=time.perf_counter(); fn(); ts.append((time.perf_counter()-a)*1e3)
        return np.median(ts)
    t_all = t(lambda: lin.sphere_states(traj, q_now))
    t_cas = t(lambda: lin._fk_map(*args))
    pos_fn, cd, _, _ = lin._config_maps(H, False)
    t_pos = t(lambda: pos_fn(*lin._args_for(full, lin.param_path())))
    c1 = lin.sphere_states(traj, q_now)[0]
    c2, _ = lin.config_states(full, lin.param_path(), with_jac=False)
    print(f"H={H} spheres={lin.n_spheres} sphere_states {t_all:.2f} ms | casadi fk+jac {t_cas:.2f} | casadi pos only {t_pos:.2f} | pos bit-identical {np.array_equal(c1, c2)} maxdiff {np.abs(c1-c2).max():.2e}")
