"""T35-diag figure 2 — keyframe sheet (real scene): failed T34 run vs successful E2 (VLA + TO without perception, T28) run on
the same episode. MuJoCo offscreen render of the recorded control-row qpos (model_transport.xml, MUJOCO_GL=osmesa).
Columns per run (relative to that run's own events, a/settle.json): first close command, settle (last row of the first close cycle),
settle+6, settle+12, settle+20 control rows. Camera: free camera looking at the run's initial GT apple centre (+15 mm z),
distance 0.30 m, elevation -8 deg, azimuth = horizontal direction perpendicular to the ee_finger_l1 -> ee_finger_l2 axis at settle
(side view of the finger pair). Caption per frame: t, measured opening, GT apple dz, contact midpoint z - apple centre z (mm,
witness points; '-' when a finger is > 1 mm from the apple).
Writes figures/t35/t35-keyframes-failed-vs-E2.png + .json.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
from AG3S.ag3s.experiments.common import figstyle as F
import p12_records as P

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t35'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))
A = json.load(open(V / "T35diag/a/a_runs.json"))["runs"]
ST = json.load(open(V / "T35diag/a/settle.json"))["runs"]
PAIRS = [("T34/E3b_rec/E3b_ep1800_r1", "T28/E2_rec/E2_ep1800_r1"), ("T34/E3a_rec/E3a_ep1800_r1", "T28/E2_rec/E2_ep1800_r2"),
         ("T34/E3b_rec/E3b_ep1808_r1", "T28/E2_rec/E2_ep1808_r1"), ("T34/E3b_rec/E3b_ep1808_r2", "T28/E2_rec/E2_ep1808_r2")]
OFFS = [("close", None), ("settle", 0), ("settle+6", 6), ("settle+12", 12), ("settle+20", 20)]

m, d, probe, obj, fing = P.build()
B1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1"); B2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l2")
ren = mujoco.Renderer(m, 300, 400)
side = {"figure": None, "script": "outputs/verify/T35diag/figs/fig_keyframes.py", "doc": __doc__, "rows": []}
nrow = 2 * len(PAIRS)
fig, axes = plt.subplots(nrow, len(OFFS), figsize=(2.9 * len(OFFS), 2.35 * nrow))
F.style_axes(fig, axes)
for pi, pair in enumerate(PAIRS):
    for ri, rp in enumerate(pair):
        key = rp.split("/", 1)[1]
        r = A[key]; st = ST[key]
        rows = [json.loads(l) for l in open(V / rp / "frames.jsonl") if l.strip()]
        ctrl = {c["t_step"]: c for c in rows if c["kind"] == "control"}
        tl = {e["t"]: e for e in r["timeline"]}
        q0 = np.asarray(ctrl[min(ctrl)]["qpos"], float)
        P.set_q(m, d, ctrl[st["t_settle"]]["qpos"]); mujoco.mj_forward(m, d)
        ax_ = d.xpos[B2] - d.xpos[B1]
        v = np.array([-ax_[1], ax_[0]]); v /= np.linalg.norm(v) + 1e-12
        az = float(np.degrees(np.arctan2(v[1], v[0])))
        row_i = 2 * pi + ri
        rec = {"run": key, "success": r["success"], "azimuth_deg": az, "frames": []}
        for ci, (name, off) in enumerate(OFFS):
            t = r["t_close"] if off is None else st["t_settle"] + off
            t = min(t, max(ctrl))
            c = ctrl[t]
            P.set_q(m, d, c["qpos"]); mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera(); cam.lookat[:] = q0[38:41] + np.array([0, 0, 0.015]); cam.distance = 0.30
            cam.azimuth = az; cam.elevation = -8
            ren.update_scene(d, camera=cam)
            img = ren.render()
            ax = axes[row_i, ci]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            e = tl[t]
            inh = e["d_l1_mm"] <= 1.0 and e["d_l2_mm"] <= 1.0
            mz = e["mid_rel_apple_mm"][2]
            cap = f"t {t}  open {e['open_meas']:.2f}\ndz {e['dz_mm']:.1f} mm  mid z {mz:+.1f}" if inh else f"t {t}  open {e['open_meas']:.2f}\ndz {e['dz_mm']:.1f} mm  mid z -"
            ax.text(4, 4, cap, fontsize=7, va="top", color=F.INK, bbox=dict(facecolor=F.SURFACE, alpha=0.8, lw=0, pad=1.5))
            if row_i == 0:
                ax.set_title(name, fontsize=9, color=F.INK)
            if ci == 0:
                ax.set_ylabel(("FAIL " if not r["success"] else "OK ") + key.split("/")[1], fontsize=8, color=F.INK)
            rec["frames"].append({"col": name, "t": t, "open_meas": e["open_meas"], "dz_mm": e["dz_mm"], "d_l1_mm": e["d_l1_mm"],
                                  "d_l2_mm": e["d_l2_mm"], "mid_rel_apple_mm": e["mid_rel_apple_mm"], "in_hand": inh})
        side["rows"].append(rec)
fig.suptitle("T35-diag keyframes: failed T34 runs (FAIL) vs successful E2 runs, same episode (side view of the finger pair)", fontsize=11, color=F.INK)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = FIG / "t35-keyframes-failed-vs-E2.png"
fig.savefig(out, dpi=110); plt.close(fig)
side["figure"] = str(out)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print("wrote", out)
