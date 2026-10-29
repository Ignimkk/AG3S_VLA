"""T43 TP — capture the OSQP subproblems V11AB solves on one recorded run (server mode), with the
warm start each one was given, for an offline settings study (qp_study.py)."""
import argparse, json, pickle, sys, time
import numpy as np
from AG3S.trajopt.experiments import tp_timing as TP
from AG3S.trajopt import qp as QP

key, out, setting = sys.argv[1], sys.argv[2], sys.argv[3]
cap = []
orig = QP.QpSolver._solve_osqp
def wrapped(self, problem, warm_start):
    ws = None
    if (self._backend is not None and self._signature == QP._signature(problem) and warm_start
            and self._last_x is not None and self._last_x.shape[0] == problem.n_var):
        ws = (self._last_x.copy(), self._last_y.copy())
    t0 = time.perf_counter()
    sol = orig(self, problem, warm_start)
    dt = (time.perf_counter() - t0) * 1e3
    cap.append({"P": problem.P, "q": problem.q, "A": problem.A, "l": problem.l, "u": problem.u,
                "n_q": problem.n_q, "nq": problem.nq, "horizon": problem.horizon, "ws": ws,
                "eps": self.config.eps_abs, "iters": sol.iterations, "ms": dt, "status": sol.status,
                "x": sol.x, "seq": CUR["seq"]})
    return sol
QP.QpSolver._solve_osqp = wrapped
CUR = {"seq": None}
idx = "outputs/verify/T43/V8/analysis/analysis.json"
entry = json.load(open(idx))["runs"][key]
import AG3S.trajopt.experiments.ta_offline as TA
orig_inputs = TA.Run.inputs
def inputs(self, seq):
    CUR["seq"] = seq
    return orig_inputs(self, seq)
TA.Run.inputs = inputs
cli = argparse.Namespace(variants=[setting], mode="server", repeat=1, keep_traj=False, progress=False,
                         max_chunks=0, first_chunk=0)
TP.run_timing(key, entry, cli)
pickle.dump(cap, open(out, "wb"))
it = np.array([c["iters"] for c in cap]); ms = np.array([c["ms"] for c in cap])
print(len(cap), "solves; iters median", np.median(it), "p90", np.quantile(it, .9), "max", it.max(), "ms sum", ms.sum())
