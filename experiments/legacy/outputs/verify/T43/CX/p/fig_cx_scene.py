"""T43 CX figure 1 (real scene): drop / grasp-loss moments, V11AB vs V11ABC at the same ep:seed (verifier).
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 PYTHONPATH=${AG3S_PROJECT_ROOT} fig_cx_scene.py OUT_PNG
Rows: H 1807 s18071 (t = 222, 4 rows before the V11ABC separation, inside its HOLD stretch), H 1995 s19953 (t = 168, V11ABC separation row - 1),
H 1982 s19822 (t = 132, V11ABC separation row - 2). Columns 1-2: third_person_front.mp4 frame n = control row t of each run.
Column 3, row 1: side view across the bar (y-z plane, world, mm): hurdle collision geoms (hurdle_metrics shapes, projected), V11ABC worst held query sphere
of each HOLD chunk (cx_hold point_world, radius 38.5 mm, at refined row step 6-7) and the measured apple centre of both runs at that chunk's t (control-row qpos).
Column 3, rows 2-3: measured apple centre z and fingertip-body midpoint z vs control row (control-row qpos), both runs; vertical line = V11ABC separation row."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, subprocess, sys, tempfile
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
import hurdle_metrics as HM  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists() and not out.with_suffix(".json").exists(), f"{out} exists"
X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX')); V11 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11'))
figstyle.use_korean(); C = figstyle.CATEGORICAL
CS = json.load(open(X / "raw/cx_cases.json"))["cases"]; HD = json.load(open(X / "raw/cx_hold.json"))["arms"]
ROWS = [("H:1807_s18071", 222), ("H:1995_s19953", 168), ("H:1982_s19822", 132)]
ARMC = {"V11AB": C[0], "V11ABC": C[1]}
HM.set_arm("left"); M = HM.model(); m, d = M["m"], M["d"]
B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def run_dir(arm, ck):
    return pathlib.Path(json.load(open(V11 / f"analysis/analysis_{arm}.json"))["runs"][CS[ck]["arms"][arm]["key"]]["run_dir"])


def ctrl_rows(rd):
    return {r["t_step"]: r for r in (json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()) if r["kind"] == "control"}


def state(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    return d.xpos[B("apple")].copy(), 0.5 * (d.xpos[B("ee_finger_l1")] + d.xpos[B("ee_finger_l2")])


side = {"script": __file__, "rows": {}}
fig = plt.figure(figsize=(20, 15)); gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.9])
for i, (ck, tf) in enumerate(ROWS):
    st, nm = ck.split(":"); side["rows"][ck] = {"t_frame": tf, "panels": {}}
    for j, arm in enumerate(("V11AB", "V11ABC")):
        rd = run_dir(arm, ck)
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(rd / "third_person_front.mp4"), "-vf", f"select='eq(n\\,{tf})'", "-vsync", "0", f"{td}/f.png"], check=True)
            img = mpimg.imread(f"{td}/f.png")
        ax = fig.add_subplot(gs[i, j]); ax.imshow(img); ax.axis("off")
        tl = CS[ck]["arms"][arm]["timeline"]; stg = CS[ck]["fail_stage"][arm]["stage"]
        ax.set_title(f"{arm} {st} ep{nm}  t={tf}  ({stg}; separation row {tl.get('t_sep')}, grasp row {tl.get('grasp_t_used')})", fontsize=10, color=ARMC[arm])
        side["rows"][ck]["panels"][arm] = {"run_dir": str(rd), "stage": stg, "t_sep": tl.get("t_sep"), "grasp_t_used": tl.get("grasp_t_used")}
    ax = fig.add_subplot(gs[i, 2])
    if i == 0:
        rd = run_dir("V11ABC", ck); stf = rd.parent / f"{rd.name}.start"
        cfg, prof = next(l.split("=", 1)[1].split(":", 1) for l in stf.read_text().splitlines() if l.startswith("obstacle="))
        mg = HM.manager(cfg, prof); shapes = HM.hurdle_shapes(mg); sh = []
        for t_, c_, ax_, hl, r_ in shapes:
            if abs(ax_[0]) > 0.9:   # bar along x -> circle in y-z
                ax.add_patch(plt.Circle((1e3 * c_[1], 1e3 * c_[2]), 1e3 * r_, color=C[7], zorder=5)); sh.append({"kind": "bar_x", "y_mm": 1e3 * c_[1], "z_mm": 1e3 * c_[2], "r_mm": 1e3 * r_})
            elif abs(ax_[2]) > 0.9:   # post
                ax.add_patch(plt.Rectangle((1e3 * (c_[1] - r_), 1e3 * (c_[2] - hl)), 2e3 * r_, 2e3 * hl, color=C[7], alpha=0.35, zorder=4)); sh.append({"kind": "post_z", "y_mm": 1e3 * c_[1], "z_lo_mm": 1e3 * (c_[2] - hl), "z_hi_mm": 1e3 * (c_[2] + hl)})
        held = [x for x in HD["V11ABC"] if x["key"].endswith("ep1807_s18071") and x["kind"] == "held"]
        pts = []
        for k, x in enumerate(held):
            w = np.asarray(x["point_world_m"]); ax.add_patch(plt.Circle((1e3 * w[1], 1e3 * w[2]), x["radius_mm"], fill=False, ec=C[1], lw=0.9, alpha=0.7, zorder=6))
            pts.append({"t": x["t"], "y_mm": 1e3 * w[1], "z_mm": 1e3 * w[2], "r_mm": x["radius_mm"], "to_clearance_mm": x["to_clearance_mm"], "bar_gap_mm": x["bar_gap_mm"]})
        ax.scatter([p["y_mm"] for p in pts], [p["z_mm"] for p in pts], s=12, color=C[1], zorder=7, label="V11ABC HOLD: worst held sphere centre (refined row 6-7), r 38.5 mm")
        app = {}
        for arm in ("V11AB", "V11ABC"):
            cr = ctrl_rows(run_dir(arm, ck)); ts_ = [x["t"] for x in held]
            P = np.array([state(np.asarray(cr[t]["qpos"], float))[0] for t in ts_])
            ax.plot(1e3 * P[:, 1], 1e3 * P[:, 2], "o-", ms=4, color=ARMC[arm], lw=1, zorder=8, label=f"{arm}: measured apple centre at t = {ts_[0]}..{ts_[-1]}")
            app[arm] = {"t": ts_, "y_mm": (1e3 * P[:, 1]).tolist(), "z_mm": (1e3 * P[:, 2]).tolist()}
        ax.set_aspect("equal"); ax.set_xlabel("world y (mm)"); ax.set_ylabel("world z (mm)")
        ax.set_xlim(0, 450); ax.set_ylim(880, 1200); ax.legend(fontsize=7, loc="lower left")
        ax.set_title("H 1807 s18071 side view (y-z): bar (red), V11ABC planned held sphere vs measured apple", fontsize=10)
        side["rows"][ck]["side_view"] = {"hurdle": sh, "held_sphere_points": pts, "apple": app}
    else:
        ts_abc = CS[ck]["arms"]["V11ABC"]["timeline"]["t_sep"]; t0, t1 = tf - 40, tf + 30; tr = {}
        for arm in ("V11AB", "V11ABC"):
            cr = ctrl_rows(run_dir(arm, ck)); T = list(range(t0, t1))
            S = [state(np.asarray(cr[t]["qpos"], float)) for t in T]
            az = [1e3 * s[0][2] for s in S]; fz = [1e3 * s[1][2] for s in S]
            ax.plot(T, az, color=ARMC[arm], lw=2, label=f"{arm} apple centre z"); ax.plot(T, fz, color=ARMC[arm], lw=1, ls="--", label=f"{arm} fingertip-body midpoint z")
            tr[arm] = {"t": T, "apple_z_mm": az, "finger_mid_z_mm": fz}
        ax.axvline(ts_abc, color=figstyle.INK_2, lw=0.8); ax.text(ts_abc + 0.5, ax.get_ylim()[1] - 5, f"V11ABC separation t={ts_abc}", fontsize=8, va="top")
        ax.set_xlabel("control row t (1/15 s)"); ax.set_ylabel("z (mm)"); ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"{ck}: apple vs fingertip midpoint height", fontsize=10); side["rows"][ck]["heights"] = tr
figstyle.style_axes(fig, [a for a in fig.axes if a.get_xlabel()])
fig.suptitle("T43 CX: V11AB (blue) vs V11ABC (orange), same ep:seed - drop / grasp-loss moment", fontsize=13)
fig.tight_layout(); fig.savefig(out, dpi=90, facecolor=figstyle.SURFACE)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(out)
