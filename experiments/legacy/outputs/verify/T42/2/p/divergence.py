"""T42-2: first divergence of each obstacle run from the no-obstacle run of the same episode, seed and condition (T42-1; T39 for
apple + left rep 1-2). traj.npz executed_actions (n x 16): first control step with max |diff| > 0 and > 1e-3; the reference run's
lift_t (grasp_robust) and whether the divergence step is before it. usage: divergence.py ANALYSIS_JSON (adds 'divergence' in place)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p'))
import plib as L  # noqa: E402
A = json.loads(pathlib.Path(sys.argv[1]).read_text())
for k, r in A["runs"].items():
    ref = L.run_dir(r["cond"], r["episode"], r["seed"])
    za = np.load(pathlib.Path(r["obst"]["run_dir"]) / "traj.npz")["executed_actions"]; zb = np.load(ref / "traj.npz")["executed_actions"]
    n = min(len(za), len(zb)); dd = np.abs(za[:n] - zb[:n]).max(axis=1)
    sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p')); import plan as PL
    rb = PL.robust_cached(ref)
    f0 = int(np.argmax(dd > 0)) if (dd > 0).any() else None; f3 = int(np.argmax(dd > 1e-3)) if (dd > 1e-3).any() else None
    r["divergence"] = {"reference_run": str(ref), "n_compared": n, "first_step_diff_gt0": f0, "first_step_diff_gt_1e-3": f3,
                       "reference_lift_t": rb.get("lift_t"), "reference_grasp_t": rb.get("grasp_t"),
                       "diverges_before_reference_lift": None if (f3 is None or rb.get("lift_t") is None) else bool(f3 < rb["lift_t"])}
    print(k, r["divergence"]["first_step_diff_gt0"], r["divergence"]["first_step_diff_gt_1e-3"], "ref lift", rb.get("lift_t"))
pathlib.Path(sys.argv[1]).write_text(json.dumps(A, indent=1, default=float))
