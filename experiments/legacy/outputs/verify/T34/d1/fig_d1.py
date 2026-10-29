"""T34 D1 before/after figure: oscillation metrics in the grasp->release window (W_all) per run, groups E2 (T28), T30 E3a, T30 E3b,
T33 E3a, T33 E3b, T34 E3a, T34 E3b (only runs with grasp and place). Source outputs/verify/T34/d1/d1_oscillation.json (T31diag d1 metric code)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean(); C = fs.CATEGORICAL
D = json.load(open(_ag3s_workspace_path('outputs/verify/T34/d1/d1_oscillation.json')))["runs"]
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t34'))
GROUPS = ["E2", "T30_E3a", "T30_E3b", "T33_E3a", "T33_E3b", "T34_E3a", "T34_E3b"]
MET = [("sign_reversals_per_s", "joint velocity sign reversals / s (qpos, 7 left joints, deadband 0.02 rad/s)"),
       ("cmd_sign_reversals_per_s", "commanded sign reversals / s"), ("jerk_rms", "jerk RMS (rad/s^3, qpos)"),
       ("fingertip.mean_curvature_per_m", "fingertip mean curvature (rad/m)"), ("fingertip.path_over_chord", "fingertip path / chord"),
       ("rr_max", "refined - reference max in window (deg)")]
def get(w, k, run):
    if k == "rr_max":
        v = [x["both_arms_max_deg"] for x in run["refined_minus_reference"] if x["in_window"]]
        return max(v) if v else None
    if "." in k:
        a, b = k.split("."); return (w.get(a) or {}).get(b)
    return w.get(k)
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5)); side = {"source": "outputs/verify/T34/d1/d1_oscillation.json", "runs": {}, "group_median": {}}
for ax, (k, lab) in zip(axes.ravel(), MET):
    for gi, gname in enumerate(GROUPS):
        pts = []
        for n, r in D.items():
            if r.get("condition") != gname or "windows" not in r:
                continue
            v = get(r["windows"]["all"], k, r)
            if v is None:
                continue
            pts.append(v); side["runs"].setdefault(n, {"condition": gname, "grasp_t": r["grasp_t"], "place_t": r["place_t"], "release_t": r["release_t"]})[k] = v
            ax.plot(gi + np.random.default_rng(hash(n) % 2**32).uniform(-0.12, 0.12), v, "o", color=C[gi], ms=6, alpha=0.85)
            ax.annotate(n.replace(gname + "_", "").replace("E2_", "").replace("E3a_", "").replace("E3b_", ""), (gi + 0.14, v), fontsize=5.5, color=fs.INK_2)
        if pts:
            ax.hlines(np.median(pts), gi - 0.25, gi + 0.25, color=fs.INK, lw=1.5)
            side["group_median"].setdefault(gname, {})[k] = {"n": len(pts), "median": float(np.median(pts))}
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS, fontsize=8); ax.set_title(lab, fontsize=8.5)
fig.suptitle("T34 D1 before/after: oscillation in the grasp -> release window, executed left-arm joint trajectory (black bar = group median)", fontsize=10.5)
fs.style_axes(fig, axes.ravel()); fig.tight_layout()
p = F / "t34-d1-oscillation-before-after.png"; fig.savefig(p, dpi=130); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)
