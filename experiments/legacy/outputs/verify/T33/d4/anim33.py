"""T33 copy of T31-diag d4_anim.py: T33 records (common33.Run), held object = recorded T32 held spheres on ee_left (circles, reference amber / refined blue),
window = grasp..release from T33/d1/d1_oscillation.json (run key T33_<run>). Output figures/t33/t33-place-esdf-anim-<run>.mp4.
--- original ---
T31-diag D4 animation — composite ESDF (the value the TO reads: CuroboEsdfField.distance = min over covering tiers) on two
planes that follow the hand, with reference vs refined 8-step finger trajectories, over grasp..release of one run.

usage: ... d4_anim.py RUN
Per chunk: plane xz at y = mean left-finger-sphere y of the reference step 0; plane yz at x = the same point's x. Field sampled every
4 mm. Overlays: reference (amber, dashed) and refined (blue) mean-finger path over 8 steps with the current step's finger spheres
(circles) and held points; executed fingertip trail (ink; D1 path, ee_finger_l1/l2 body midpoint) up to the current control row;
GT crate box sections; GT apple centre (green). 8 video frames per chunk (one per control row t..t+7), right panel = the recorded
third_person_front.mp4 frame of that control row. Chunk HOLD (E3b gate) is written in the title.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import imageio.v2 as iio
import subprocess, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T33/d4'))
import common33 as FC
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean(); C = fs.CATEGORICAL
run = sys.argv[1]
R = FC.Run(run)
D1 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/d1/d1_oscillation.json')).read_text())["runs"]["T33_" + run]
g, rel = D1["grasp_t"], D1["release_t"]
trail = np.array(D1["windows"]["all"]["fingertip_path_world"]); t0 = D1["windows"]["all"]["t_first"]
VID = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T33/d4/video_{run}"); VID.mkdir(exist_ok=True)
if not any(VID.iterdir()):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(R.run_dir / "third_person_front.mp4"), "-start_number", "0", str(VID / "f%04d.png")], check=True)
FR = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T33/d4/frames_{run}"); shutil.rmtree(FR, ignore_errors=True); FR.mkdir()
nframe = [0]
rows_ctrl = {}
for l in (R.run_dir / "frames.jsonl").read_text().splitlines():
    x = json.loads(l)
    if x.get("kind") == "control":
        rows_ctrl[x["t_step"]] = x
cm = fs.sequential_cmap()
out_mp4 = FC.FIG / f"t33-place-esdf-anim-{run}.mp4"
side = {"run": run, "definitions": __doc__, "window": [g, rel], "fps": 8, "frames": []}
for c in R.chunks:
    if not (g <= c["t"] <= rel):
        continue
    ch = R.load(c["t"])
    ref, fin = ch.traj["reference"]["centres"], ch.traj["refined"]["centres"]
    rad = ch.traj["reference"]["radii"]
    p0 = ref[0][R.finger_idx].mean(0)
    planes = []
    for axis, val in ((1, p0[1]), (0, p0[0])):
        other = [k for k in range(3) if k != axis]
        la = (p0[other[0]] - 0.3, p0[other[0]] + 0.3); lb = (ch.table_top_z - 0.05, ch.table_top_z + 0.6)
        ua = np.arange(*la, 0.004); ub = np.arange(*lb, 0.004)
        A, B = np.meshgrid(ua, ub)
        pts = np.zeros((A.size, 3)); pts[:, other[0]] = A.ravel(); pts[:, other[1]] = B.ravel(); pts[:, axis] = val
        V = np.asarray(ch.field.distance(pts)).reshape(A.shape)
        planes.append((axis, val, other, ua, ub, A, B, V))
    gate = c.get("gate") or {}
    n_exec, n_ctl = c.get("n_executed"), c.get("n_control")
    side["frames"].append({"t": c["t"], "plane_y": float(p0[1]), "plane_x": float(p0[0]), "refined_minus_reference_deg": c.get("refined_minus_reference_deg"),
                           "n_executed": n_exec, "n_control": n_ctl, "gate": gate.get("action") or gate.get("would_hold")})
    for k in range(8):
        tt = c["t"] + k
        fig = plt.figure(figsize=(18, 6.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
        for j, (axis, val, other, ua, ub, A, B, V) in enumerate(planes):
            ax = fig.add_subplot(gs[0, j])
            ext = [ua[0], ua[-1], ub[0], ub[-1]]
            ax.imshow(np.clip(V, 0, 0.1), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=0.1, interpolation="nearest")
            ax.imshow(np.where(V < 0, 1.0, np.nan), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([C[7]]), alpha=0.55)
            for lev, ls in ((0.0, "-"), (0.005, "--"), (0.010, "-."), (0.020, ":")):
                if np.nanmin(V) < lev < np.nanmax(V):
                    ax.contour(A, B, V, levels=[lev], colors=fs.INK, linestyles=ls, linewidths=0.9)
            for bx in ch.boxes:
                poly = FC.plane_box_polygon(bx, axis, val)
                if poly is not None:
                    ax.add_patch(Polygon(poly, closed=True, fill=False, ec=C[1], lw=1.2))
            ax.plot(ch.apple_centre[other[0]], ch.apple_centre[other[1]], "o", color=C[2], ms=6, mfc="none", mew=1.5)
            for tr, col, ls in ((ref, C[3], "--"), (fin, C[0], "-")):
                pm = tr[:, R.finger_idx].mean(1)
                ax.plot(pm[:, other[0]], pm[:, other[1]], ls, color=col, lw=1.6)
                for q in R.finger_idx:
                    ax.add_patch(Circle(tr[k, q, other], rad[q], fill=False, ec=col, lw=0.7))
            for tr, col in ((ch.traj["reference"]["held"], C[3]), (ch.traj["refined"]["held"], C[0])):
                for q_, r_ in zip(tr[k], ch.held_radii):
                    ax.add_patch(Circle(q_[other], r_, fill=False, ec=col, lw=1.0, ls=":"))
            it = min(max(tt - t0, 0), len(trail) - 1)
            ax.plot(trail[:it + 1, other[0]], trail[:it + 1, other[1]], color=fs.INK, lw=0.9, alpha=0.8)
            ax.set_xlim(ua[0], ua[-1]); ax.set_ylim(ub[0], ub[-1]); ax.set_aspect("equal")
            lab = "xyz"
            ax.set_xlabel(f"{lab[other[0]]} (m)"); ax.set_ylabel(f"{lab[other[1]]} (m)")
            ax.set_title((f"x–z @ y={val:.3f} (follows hand)" if axis == 1 else f"y–z @ x={val:.3f} (follows hand)") if not KO else
                         (f"x–z 단면 @ y={val:.3f} (손을 따라감)" if axis == 1 else f"y–z 단면 @ x={val:.3f} (손을 따라감)"), fontsize=10)
        ax = fig.add_subplot(gs[0, 2])
        try:
            ax.imshow(iio.imread(VID / f"f{tt:04d}.png"))
        except Exception:
            pass
        ax.axis("off")
        exe = rows_ctrl.get(tt, {}).get("executed")
        ax.set_title(f"third_person_front.mp4  t={tt}  executed={exe}", fontsize=10)
        fs.style_axes(fig, fig.axes[:2])
        dev = c.get("refined_minus_reference_deg")
        fig.suptitle(f"{run}  chunk t={c['t']} step {k}  refined−reference {dev:.2f}°  executed {n_exec}/{n_ctl}   "
                     + ("amber dashed = reference, blue = refined, dotted circles = held spheres, ink = executed fingertip trail, red = d<0 (composite field)" if not KO else
                        "주황 점선 = reference, 파랑 = refined, 점선 원 = 쥔 물체 구, 검정 = 실행된 손끝 궤적, 빨강 = d<0 (합성 필드)"), fontsize=11)
        fig.patch.set_facecolor(fs.SURFACE)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(FR / f"g{nframe[0]:05d}.png", dpi=80); nframe[0] += 1; plt.close(fig)
    print(run, c["t"], flush=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "8", "-i", str(FR / "g%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)], check=True)
side["n_frames"] = nframe[0]
out_mp4.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(out_mp4)
