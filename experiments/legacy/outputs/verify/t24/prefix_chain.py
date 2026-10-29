"""T24 verifier — independent check of measurement #4 with the PRE-FIX code.

Loads `sqp_HEAD_368d723.py` (= `git show HEAD:trajopt/sqp.py`, before the T24 change) as
`AG3S.trajopt.sqp`, then drives the tool's chained replay with it. What the pre-fix code
returned is labelled by matching `result` against every trajectory `_merit` was evaluated on:
call 0 = initial iterate, later calls = QP candidates; acceptance = `merit <= iterate_merit`
(same test as the HEAD loop). Fresh-solver variants are not run (`run_variants` = no-op).
usage: prefix_chain.py <tool args ...>   (pass --variants replay)
"""
import importlib.util, sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("AG3S.trajopt.sqp", HERE / "sqp_HEAD_368d723.py")
old = importlib.util.module_from_spec(spec)
sys.modules["AG3S.trajopt.sqp"] = old
spec.loader.exec_module(old)
import AG3S.trajopt as _pkg
_pkg.sqp = old

LOG = []
_orig_merit = old.TrajectoryOptimizer._merit


def _merit(self, trajectory, *a, **k):
    m = _orig_merit(self, trajectory, *a, **k)
    LOG.append((np.array(trajectory, np.float64, copy=True), float(m)))
    return m


old.TrajectoryOptimizer._merit = _merit

from AG3S.trajopt.experiments import fixed_input_ablation as tool  # noqa: E402
from AG3S.trajopt import refiner as _ref  # noqa: E402
assert _ref.TrajectoryOptimizer is old.TrajectoryOptimizer, "refiner did not pick up HEAD sqp"

tool.run_variants = lambda *a, **k: None
_orig_refine = None


def label(result, captured):
    log = list(LOG)
    LOG.clear()
    if not log:
        return {"returned": "no_merit_log", "returned_index": None, "trajectory": None, "capture_ok": False}
    it_m = log[0][1]
    flags = []
    for traj, m in log[1:]:
        acc = m <= it_m
        if acc:
            it_m = m
        flags.append(acc)
    r = np.asarray(result.trajectory, np.float64)
    hits = [i for i, (t, _) in enumerate(log) if t.shape == r.shape and np.array_equal(t, r)]
    if not hits:
        lab, idx = "unmatched", None
    elif hits[-1] == 0:
        lab, idx = "initial", -1
    else:
        idx = hits[-1] - 1
        lab = "accepted" if flags[idx] else "best_unaccepted"
    return {"returned": lab, "returned_index": idx, "trajectory": None, "capture_ok": bool(hits),
            "n_candidates": len(log) - 1, "candidate_accepted": flags}


tool.legacy_selection = label
# LOG must start empty for each refine: clear at every chunk entry of the chain
_orig_prepare = tool.prepare_chunk


def prepare(sh, path):
    LOG.clear()
    return _orig_prepare(sh, path)


tool.prepare_chunk = prepare

if __name__ == "__main__":
    sys.exit(tool.main(sys.argv[1:]))
