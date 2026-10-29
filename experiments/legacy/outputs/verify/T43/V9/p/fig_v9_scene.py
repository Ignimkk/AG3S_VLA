"""T43 V9 scene figure: real MuJoCo scene at the hurdle crossing, V8 vs V9, same ep:seed (1807 s18071, both arms succeed under the old rule).
Each arm is drawn at its own closest apple-to-bar row (min of hurdle series B over the run); qpos = the recorded control row qpos.
usage: fig_v9_scene.py OUT_PNG OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ["MUJOCO_GL"] = "osmesa"; os.environ["T43W_SNAP"] = "${AG3S_PROJECT_ROOT}-t43v8"
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import render_y as RY  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V9": json.load(open(T / "V9/analysis/analysis.json"))["runs"]}
KEY = {"V8": "V8/H/V8_ep1807_s18071", "V9": "V9/H/V9_ep1807_s18071"}
figstyle.use_korean(); C = figstyle.CATEGORICAL
fig, axes = plt.subplots(2, 2, figsize=(13, 10.4)); figstyle.style_axes(fig, axes); meta = {}
for r_, arm in enumerate(("V8", "V9")):
    r = AN[arm][KEY[arm]]; s = r["hurdle"]["series"]; t_arr = np.array(s["t"]); B = np.array(s["B"]); i = int(B.argmin()); t = int(t_arr[i])
    mins = {k: 1e3 * float(np.array(s[k])[i]) for k in ("Af", "Ae", "B", "C")}
    M, ctrl = RY.scene_for(r["run_dir"], "left"); q = np.asarray(ctrl[t]["qpos"], float)
    m, d = M["m"], M["d"]; d.qpos[:] = q; mujoco.mj_forward(m, d)
    ap = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")].copy()
    cams = [dict(lookat=tuple(ap), distance=0.65, azimuth=150.0, elevation=-18.0), RY.REC]
    for c_, cam in enumerate(cams):
        axes[r_, c_].imshow(RY.render(M, q, cam)); axes[r_, c_].set_xticks([]); axes[r_, c_].set_yticks([])
        axes[r_, c_].set_title(("close-up on the apple (az 150, el -18)" if c_ == 0 else "recorded video view (front)"), fontsize=9, color=figstyle.INK)
    axes[r_, 0].set_ylabel(f"{arm}  {KEY[arm].split('/')[-1]}  row t {t}\nmin over groups at this row: fingers {mins['Af']:.1f} palm {mins['Ae']:.1f} apple {mins['B']:.1f} other {mins['C']:.1f} mm",
                           fontsize=8.5, color=figstyle.INK)
    meta[arm] = {"run": KEY[arm], "run_dir": r["run_dir"], "t": t, "rule": "row of min apple-to-bar distance (series B)", "apple_world": [float(x) for x in ap], "dist_mm_at_row": mins,
                 "run_min_mm": {k: 1e3 * float(np.array(s[k]).min()) for k in ("Af", "Ae", "B", "C")}, "success_old": bool(r["success"]), "grasp_t": r["grasp_t"], "place_t": r["place_t"]}
fig.suptitle("T43 V9 (--servo-model) vs V8 at the hurdle crossing, same ep1807 s18071, each at its closest apple-bar row (recorded qpos, MuJoCo render)", fontsize=10.5, color=figstyle.INK)
fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.02, hspace=0.08, wspace=0.03)
out = pathlib.Path(sys.argv[1]); assert not out.exists() and not pathlib.Path(sys.argv[2]).exists(); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=105, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "panels": meta, "cams": [str(c) for c in cams]}, open(sys.argv[2], "w"), indent=1, default=float)
print("ok", out)
