"""T28 Phase 2R keyframe contact sheet per condition, extracted from the recorded third_person_front.mp4.

usage: fig_keyframes.py COND   (reads p2r/<COND>_rec_summary.json)

Frame mapping: pi05_infer.py appends one rendered frame per control step AFTER that step's mj_step loop
(pi05_infer.py:1840-1842), so mp4 frame k = state after control step k = state logged by control row t=k+1.
A keyframe for control-row time t uses mp4 frame max(t-1, 0). t=0 uses frame 0 (after the first step).
Last column: frame n-1, numbers from control row n-1. finger->apple distance is capped at 1000 mm (mj_geomDistance distmax 1.0 m).
Columns: t=0 | first close command (applied_ctrl.gripper[0] < 0.5) | grasp t (control-row apple dz >= 50 mm) or,
if no grasp, t of minimum finger->apple surface distance | t of max apple dz | last frame.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import imageio.v2 as iio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean()

cond = sys.argv[1]
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28'))
S = json.loads((V / "p2r" / f"{cond}_rec_summary.json").read_text())
out_png = F / f"t28-p2-keyframes-{cond}.png"
assert not out_png.exists(), f"{out_png} exists — pick a new name"

order = [f"{cond}_ep{ep}_r{k}" for ep in (1807, 1800, 1808) for k in (1, 2)]
order = [k for k in order if k in S["runs"]]
cols = ["t=0", "first close cmd", "grasp t / closest", "max lift", "last"]
fig, axes = plt.subplots(len(order), 5, figsize=(16, 2.55 * len(order)))
axes = np.atleast_2d(axes)
side = {"condition": cond, "frame_mapping": __doc__, "rows": {}}
for i, name in enumerate(order):
    r = S["runs"][name]
    rd = iio.get_reader(r["mp4"])
    n = rd.count_frames()
    last_t = n  # control row t=n would be the state after frame n-1
    gt = r.get("grasp_t_control_rows")
    ts = [0, r["t_first_close_cmd"], gt if gt is not None else r["t_min_finger_to_apple"], r["t_apple_dz_max"], last_t]
    labs = ["start", "close cmd", "grasp" if gt is not None else "closest (no grasp)", "max lift", f"last frame {n-1} (row {n-1})"]
    idx = [None if t is None else min(max(t - 1, 0), n - 1) for t in ts]
    tl = r["timeline_control"]
    row = []
    for j, (t, k, lab) in enumerate(zip(ts, idx, labs)):
        ax = axes[i, j]
        ax.set_xticks([]); ax.set_yticks([])
        if k is None:
            ax.text(0.5, 0.5, "none", ha="center", va="center", transform=ax.transAxes, color=fs.INK_2)
            row.append({"col": lab, "t": None, "frame": None}); continue
        img = rd.get_data(k)
        ax.imshow(img[40:420, 60:600])
        tt = min(t, len(tl["t"]) - 1)
        dz = tl["apple_dz_mm"][tt]; gm = tl["grip_l_measured"][tt]; fa = tl["finger_to_apple_surface_mm"][tt]
        ax.set_title(f"{lab}  " + (f"t={t}  " if j < 4 else "") + f"dz={dz:.0f}mm  grip={gm:.2f}  f-a={'>=1000' if fa >= 999.9 else format(fa, '.0f')}mm", fontsize=7.5, color=fs.INK)
        row.append({"col": lab, "t": t, "frame": k, "apple_dz_mm": dz, "grip_l_measured": gm, "finger_to_apple_mm": fa})
        for s in ax.spines.values():
            s.set_color(fs.CATEGORICAL[2] if r["success"] else (fs.CATEGORICAL[3] if r["grasp"] else fs.CATEGORICAL[7])); s.set_linewidth(2)
    res = "success" if r["success"] else ("grasp, no place" if r["grasp"] else "no grasp")
    axes[i, 0].set_ylabel(f"{name.replace(cond + '_', '')}\n{res}", fontsize=8.5, color=fs.INK)
    side["rows"][name] = {"success": r["success"], "grasp": r["grasp"], "place": r["place"], "n_frames": n, "mp4": r["mp4"], "keyframes": row}
    rd.close()
fig.patch.set_facecolor(fs.SURFACE)
fig.suptitle(f"T28 P2R {cond} — third-person front view keyframes (border: green success / amber grasp only / red no grasp)", fontsize=11, color=fs.INK)
fig.tight_layout()
fig.savefig(out_png, dpi=110); plt.close(fig)
out_png.with_suffix(".json").write_text(json.dumps(side, indent=1))
print(out_png)
