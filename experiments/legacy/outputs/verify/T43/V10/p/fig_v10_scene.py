"""T43 V10 scene figure: the hurdle crossing, V8 vs V10 (same ep:seed), real MuJoCo scene rendered from the recorded control row qpos at the carry row with the smallest
apple-bar distance (hurdle_metrics series B, rows between grasp_t and release_t). Apple centre trail (replay apple_pos, grasp_t..release_t) drawn as small spheres.
usage: fig_v10_scene.py OUT_PNG OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ["T43W_SNAP"] = "${AG3S_PROJECT_ROOT}-t43v10"
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import render_y as RY  # noqa: E402
import mujoco  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V10": json.load(open(T / "V10/analysis/analysis.json"))["runs"]}
CT = json.load(open(T / "V10/ct.json"))["runs"]
REP = {"V8": T / "V8/replay", "V10": T / "V10/replay"}
C = figstyle.CATEGORICAL; COL = {"V8": C[0], "V10": C[1]}
hexrgb = lambda h, a: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (a,)
RUNS = [("1807", "18073"), ("1968", "19683")]
CAM = dict(distance=0.75, azimuth=165.0, elevation=-12.0)
kor = figstyle.use_korean()
fig, axes = plt.subplots(len(RUNS), 2, figsize=(13.5, 5.4 * len(RUNS))); figstyle.style_axes(fig, axes); meta = []
for r_, (ep, sd) in enumerate(RUNS):
    for c_, arm in enumerate(("V8", "V10")):
        key = f"{arm}/H/{arm}_ep{ep}_s{sd}"; r = AN[arm][key]; o = CT[key]; h = r["hurdle"]
        gt, rt = o["grasp_t"], o["release_t"]
        ser = h["series"]; ts = np.asarray(ser["t"]); Bm = np.asarray(ser["B"], float)
        sel = (ts >= gt) & ((ts <= rt) if rt is not None else True)
        i = int(np.argmin(np.where(sel, Bm, np.inf))); t = int(ts[i]); bmm = 1e3 * float(Bm[i])
        M, ctrl = RY.scene_for(r["run_dir"], "right" if int(ep) == 1834 else "left")
        z = np.load(REP[arm] / f"{arm}_H_{arm}_ep{ep}_s{sd}.npz"); tt = z["t"]; ap = z["apple_pos"][1:]
        trail = [(ap[k], 0.004, hexrgb(COL[arm], 0.9)) for k in range(len(tt)) if gt <= tt[k] <= (rt if rt is not None else 10 ** 9)]
        k0 = int(np.where(tt == t)[0][0]); ctr = ap[k0]
        q = np.array(ctrl[t]["qpos"], float)
        cm = dict(lookat=tuple(ctr), **CAM)
        img = RY.render(M, q, cm, trail, size=(540, 720))
        ax = axes[r_, c_]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        cont = [(x["phase"], x["body"], x["first_t"], round(x["max_pen_mm"], 1)) for x in o["contacts"]]
        ax.set_title(f"{arm}  ep{ep} s{sd}  row t {t}: apple-bar min {bmm:.1f} mm  ({'S' if o['success'] else 'F'})\ncontacts (MJCF names): {cont if cont else 'none'}", fontsize=8.5, color=figstyle.INK, loc="left")
        meta.append({"arm": arm, "run": key, "row_t": t, "apple_bar_min_mm_at_row": bmm, "grasp_t": gt, "release_t": rt, "contacts": cont, "camera": {"lookat_world": [float(x) for x in ctr], **CAM}, "n_trail_points": len(trail)})
fig.suptitle("T43 V10 vs V8 hurdle crossing (real MuJoCo scene at the carry row of smallest apple-bar distance; dots = apple centre path, grasp to release; V8 blue / V10 orange)", fontsize=10.5, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0)
out = pathlib.Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True); assert not out.exists(); fig.savefig(out, dpi=105, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "panels": meta, "sources": [str(T / "V8/analysis/analysis.json"), str(T / "V10/analysis/analysis.json"), str(T / "V10/ct.json"), str(T / "V8/replay"), str(T / "V10/replay")],
           "pose": "recorded control-row qpos (pre-state of that row)"}, open(sys.argv[2], "w"), indent=1, default=float)
print("ok", out)
