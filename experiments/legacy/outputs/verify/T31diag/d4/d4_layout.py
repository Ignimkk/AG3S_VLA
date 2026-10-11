"""T31-diag D4 layout — where the slices cut (base frame; base == MuJoCo world for these runs, checked: base body pose identity).
usage: ... d4_layout.py RUN T1 T2 T3
Top view (x-y) and side view (x-z): GT crate collision boxes (projection), table top edge, fingertip path over grasp..release
(D1, midpoint of ee_finger_l1/l2 bodies, control rows), GT apple centre at the planning rows, the fine window box of each selected
chunk, and the slice planes of d4_slices.py (xy @ crate rim z, xy @ held-apple z per chunk, xz @ crate centre y)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d4'))
import fig_common as FC
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean(); C = fs.CATEGORICAL
run = sys.argv[1]; ts = [int(x) for x in sys.argv[2:]]
R = FC.Run(run)
D1 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')).read_text())["runs"][run]
path = np.array(D1["windows"]["all"]["fingertip_path_world"])
chs = [R.load(t) for t in ts]
ch0 = chs[0]
assert np.allclose(ch0.Rb, np.eye(3), atol=1e-3) and np.allclose(ch0.tb, 0, atol=1e-3), (ch0.Rb, ch0.tb)
apple = np.array([R.rows[c["seq"]]["object_poses"]["apple"]["pos"] for c in R.chunks if D1["grasp_t"] <= c["t"] <= D1["release_t"]])
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
side = {"run": run, "chunks": ts, "base_equals_world": "within 1e-3 (rot) and 1e-3 m (pos)", "planes": {}, "fine_windows": {}}
for ax, (a, b) in zip(axes, ((0, 1), (0, 2))):
    for bx in ch0.boxes:
        from scipy.spatial import ConvexHull
        q = bx[:, [a, b]]; h = ConvexHull(q)
        ax.add_patch(Polygon(q[h.vertices], closed=True, fill=True, fc=C[1], alpha=0.25, ec=C[1], lw=0.8))
    ax.plot(path[:, a], path[:, b], color=C[0], lw=1.2, label="fingertip path grasp→release (D1)" if not KO else "손끝 경로 grasp→release (D1)")
    ax.plot(apple[:, a], apple[:, b], ".", color=C[2], ms=5, label="GT apple centre (planning rows)" if not KO else "GT 사과 중심 (planning 행)")
    for i, ch in enumerate(chs):
        g = ch.field.layers[1].grid; lo = np.asarray(g.origin); hi = lo + (np.asarray(g.shape) - 1) * g.voxel_size
        ax.add_patch(Rectangle((lo[a], lo[b]), hi[a] - lo[a], hi[b] - lo[b], fill=False, ec=C[3 + i], ls="--", lw=1.1,
                               label=f"fine window t={ch.t}" if not KO else f"fine 창 t={ch.t}"))
        side["fine_windows"][ch.t] = {"lo": lo.tolist(), "hi": hi.tolist()}
        tip = ch.traj["refined"]["centres"][0][R.finger_idx].mean(0)
        ax.plot(tip[a], tip[b], "o", color=C[3 + i], ms=7)
        ax.annotate(f"t={ch.t}", (tip[a], tip[b]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=fs.INK)
    cc = ch0.crate_centre
    if b == 1:
        ax.axhline(cc[1], color=fs.INK, lw=1.4, ls="-.", label="slice xz @ crate centre y" if not KO else "단면 xz @ crate 중심 y")
        ax.set_ylabel("y (m)")
    else:
        ax.axhline(ch0.crate_rim_z - 0.005, color=fs.INK, lw=1.4, ls="-", label="slice xy @ crate rim z − 5 mm" if not KO else "단면 xy @ crate 테두리 z − 5 mm")
        for i, ch in enumerate(chs):
            ax.axhline(ch.apple_centre[2], color=C[3 + i], lw=1.0, ls=":", label=(f"slice xy @ held-apple z (t={ch.t})" if not KO else f"단면 xy @ 쥔 사과 z (t={ch.t})"))
            side["planes"][f"xy_apple_t{ch.t}"] = float(ch.apple_centre[2])
        ax.axhline(ch0.table_top_z, color=fs.INK_2, lw=0.8, ls="--", label="table top z" if not KO else "테이블 상판 z")
        ax.set_ylabel("z (m)")
    ax.set_xlabel("x (m)"); ax.set_aspect("equal")
    ax.set_title(("top view x–y (base = world)" if b == 1 else "side view x–z (base = world)") if not KO else
                 ("위에서 본 x–y (base = world)" if b == 1 else "옆에서 본 x–z (base = world)"), fontsize=11)
    ax.legend(fontsize=8, loc="best")
side["planes"].update({"xy_rim_minus_5mm_z": float(ch0.crate_rim_z - 0.005), "xz_y": float(ch0.crate_centre[1])})
side["crate_centre"] = ch0.crate_centre.tolist(); side["table_top_z"] = float(ch0.table_top_z)
side["fingertip_z_range_m"] = [float(path[:, 2].min()), float(path[:, 2].max())]
side["fingertip_min_height_above_rim_m"] = float(path[:, 2].min() - ch0.crate_rim_z)
fs.style_axes(fig, list(axes))
fig.suptitle(f"{run} — slice layout (D4)" if not KO else f"{run} — 단면 배치도 (D4)", fontsize=13)
fig.tight_layout()
out = FC.FIG / f"t31-d4-layout-{run}.png"
fig.savefig(out, dpi=120); out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(out, side["fingertip_min_height_above_rim_m"])
