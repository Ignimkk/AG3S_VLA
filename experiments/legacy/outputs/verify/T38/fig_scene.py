"""T38 figure (1) real scene: the replayed request of one chunk — the 3 re-rendered depth images and the fused cloud
the server recorded for the same chunk (base frame, top view, coloured by attention), with the target points.
writes AG3S/ag3s/docs/figures/t38/t38-scene.png + .json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-o4")
from AG3S.ag3s.experiments.common import figstyle

run = "E3b_ep1807_r2"; seq = int(sys.argv[1]) if len(sys.argv) > 1 else 22
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t38')); FIG.mkdir(parents=True, exist_ok=True)
idx = json.loads((OUT / "frames" / f"{run}.json").read_text())
s = [x for x in idx["seqs"] if x["seq"] == seq][0]
z = np.load(OUT / "frames" / run / f"seq{seq:03d}.npz")
rec = np.load(pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/E3b_rec')) / s["server_chunk"], allow_pickle=False)
summ = json.loads(str(rec["summary_json"]))
figstyle.use_korean()
fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.3)
cams = ["zed_left", "wrist_cam_l", "wrist_cam_r"]
side = {}
for i, c in enumerate(cams):
    ax = fig.add_subplot(gs[0, i])
    d = z[f"depth_mm/{c}"].astype(float) / 1000.0
    im = ax.imshow(np.where(d > 0, d, np.nan), cmap=figstyle.sequential_cmap(), vmin=0.2, vmax=2.0)
    att = z[f"attention/{c}"]
    ax.contour(np.kron(att, np.ones((d.shape[0] // 16, d.shape[1] // 16))), levels=[np.percentile(att, 95)],
               colors=[figstyle.CATEGORICAL[1]], linewidths=1.0)
    ax.set_title(f"{c} depth {d.shape[1]}x{d.shape[0]}\n{int((d > 0).sum())} valid px; orange = attention top 5 % cells", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    side[c] = {"valid_px": int((d > 0).sum()), "shape": list(d.shape)}
fig.colorbar(im, cax=fig.add_axes([0.12, 0.2, 0.4, 0.025]), orientation="horizontal", label="depth (m)")
ax = fig.add_subplot(gs[0, 3])
pts = rec["raw_cloud"]; att = rec["attention_values"]; tgt = rec["target_points"]; sup = rec["support_mask"]
ax.scatter(pts[sup, 0], pts[sup, 1], s=0.3, c=figstyle.GRID_INK, label=f"support-surface points ({int(sup.sum())})")
o = ~sup
sc = ax.scatter(pts[o, 0], pts[o, 1], s=0.5, c=att[o], cmap=figstyle.sequential_cmap(), vmin=0, vmax=1)
ax.scatter(tgt[:, 0], tgt[:, 1], s=4, c=figstyle.CATEGORICAL[1], label=f"target points ({len(tgt)})")
ax.set_aspect("equal"); ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)")
ax.set_xlim(-0.1, 1.2); ax.set_ylim(-0.7, 0.7)
ax.legend(fontsize=7, loc="lower left", markerscale=4)
fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, label="fused attention (normalised)")
ax.set_title(f"fused cloud recorded by the server for this chunk: {len(pts)} points\n"
             f"(top view; scene_reconstruction output -> target_grounding input)", fontsize=8)
figstyle.style_axes(fig, fig.axes)
fig.suptitle(f"T38 replay input: {run} seq {seq} (t_step {s['t_step']}, phase {s['phase_label']}, grasp {s['recorded_grasp_state']}) "
             f"- 3 cameras re-rendered from frames.jsonl qpos, attention 16x16 from the server record", fontsize=10)
fig.savefig(FIG / "t38-scene.png", dpi=130, bbox_inches="tight", facecolor=figstyle.SURFACE)
(FIG / "t38-scene.json").write_text(json.dumps({
    "run": run, "seq": seq, "t_step": s["t_step"], "server_chunk": s["server_chunk"], "cameras": side,
    "fused_points_recorded": int(len(pts)), "support_points": int(sup.sum()), "target_points": int(len(tgt)),
    "n_clusters_recorded": summ.get("n_clusters"), "source_frames": f"outputs/verify/T38/frames/{run}/seq{seq:03d}.npz",
    "script": "outputs/verify/T38/fig_scene.py"}, indent=1))
print("ok")
