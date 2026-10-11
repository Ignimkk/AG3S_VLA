from common.workspace import workspace_path as _ag3s_workspace_path
import sys, os, json
sys.path.insert(0, _ag3s_workspace_path('')); os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.runtime.asset_path import resolve_asset
ko = figstyle.use_korean()
OUT = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t19'); RAW = _ag3s_workspace_path('outputs/verify/T19')
RUN = _ag3s_workspace_path('outputs/live_test/20260928_t14')
rows = [json.loads(l) for l in open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T19.verify.partial.json'))]
R = {(r["setting"], r["seq"], r["camera"]): r for r in rows}
SETS = ["A", "B", "C", "C2", "D"]
LABEL = {"A": "A  default (0.05 m, guard off)", "B": "B  gripper 0.010 m", "C": "C  B + guard 0.06 m @ chunk_{k-1} centroid",
         "C2": "C2 B + guard 0.06 m @ chunk_{k-2} centroid (suppl.)", "D": "D  B + guard 0.06 m @ apple ground truth"}
COL = {s: figstyle.CATEGORICAL[i] for i, s in enumerate(SETS)}
SEQS = [10, 11, 12, 13, 14]; TS = [72, 80, 88, 96, 104]

# ---------- Fig 1: scene overlay wrist_cam_l t=96 ----------
p = resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
m = mujoco.MjModel.from_xml_path(str(p)); d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
fr = [json.loads(l) for l in open(f"{RUN}/execute_ep1807/frames.jsonl") if json.loads(l).get("kind") == "planning" and json.loads(l)["seq"] == 13][0]
d.qpos[:] = np.asarray(fr["qpos"], float); mujoco.mj_forward(m, d)
rend = mujoco.Renderer(m, scene.height, scene.width); rend.update_scene(d, camera="wrist_cam_l"); img = np.asarray(rend.render())
f = scene.capture("wrist_cam_l"); K = f.camera_intrinsics; Tinv = np.linalg.inv(f.T_base_cam)
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
robot_root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
panels = ["A", "C", "D"]; side1 = {}
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9)); figstyle.style_axes(fig, axes)
for ax, s in zip(axes, panels):
    z = np.load(f"{RAW}/pix_{s}_seq13_wrist_cam_l.npz")
    us, vs, bids, passed = z["us"], z["vs"], z["bids"], z["passed"]
    ap = bids == apple_bid; rb = m.body_rootid[bids] == robot_root
    ax.imshow(img)
    ax.scatter(us[ap & passed], vs[ap & passed], s=1.0, c=figstyle.CATEGORICAL[2], lw=0, label="apple px passed")
    ax.scatter(us[ap & ~passed], vs[ap & ~passed], s=1.0, c=figstyle.CATEGORICAL[7], lw=0, label="apple px removed")
    ax.scatter(us[rb & passed], vs[rb & passed], s=1.0, c=figstyle.CATEGORICAL[3], lw=0, label="robot px passed (leak)")
    gtxt = "guard off"
    if bool(z["has_guard"]):
        c = Tinv[:3, :3] @ z["guard_centre"] + Tinv[:3, 3]
        if c[2] > 0.01:
            u, v = K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]
            ax.add_patch(Circle((u, v), 0.06 * K[0, 0] / c[2], fill=False, ec=figstyle.CATEGORICAL[0], lw=1.0, ls="--"))
            ax.plot([u], [v], "+", color=figstyle.CATEGORICAL[0], ms=8)
            gtxt = f"guard centre in view (u={u:.0f}, v={v:.0f})"
        else:
            gtxt = "guard centre behind camera"
    r = R[(s, 13, "wrist_cam_l")]
    ax.text(8, 22, gtxt, fontsize=7.5, color=figstyle.INK, va="top", bbox=dict(fc=figstyle.SURFACE, ec="none", alpha=0.8))
    ax.set_xlim(0, scene.width); ax.set_ylim(scene.height, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{LABEL[s]}\napple {r['apple_px_passed']}/{r['apple_px_visible']} px passed · robot {r['robot_px_passed']}/{r['robot_px_visible']} px passed\n"
                 f"guard centre {r['guard_source'] or '-'}", fontsize=8.2, color=figstyle.INK)
    side1[s] = {k: r[k] for k in ("apple_px_visible", "apple_px_passed", "robot_px_visible", "robot_px_passed",
                                    "guard_centre", "guard_source", "guard_centre_to_apple_mm")}
    side1[s]["guard_projection"] = gtxt
axes[0].legend(fontsize=7, frameon=False, loc="lower left", markerscale=7)
fig.suptitle("Fig T19-1 — T14 t=96 (seq 13) wrist_cam_l, MuJoCo re-posed from recorded qpos; decision by robot_filter.filter_robot_points "
             "(218 spheres). Dashed blue: guard sphere R=0.06 m projected", fontsize=9.6, color=figstyle.INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/t19-selffilter-scene-t96.png", dpi=170, facecolor=figstyle.SURFACE)
json.dump({"seq": 13, "t": 96, "camera": "wrist_cam_l", "panels": side1}, open(f"{OUT}/t19-selffilter-scene-t96.json", "w"), indent=1)

# ---------- Fig 2: graph ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4)); figstyle.style_axes(fig, axes)
side2 = {}
for ax, cam in zip(axes[:2], ("zed_left", "wrist_cam_l")):
    vis = [R[("A", q, cam)]["apple_px_visible"] for q in SEQS]
    ax.plot(TS, vis, color=figstyle.INK_2 if hasattr(figstyle, "INK_2") else "#555", lw=1.0, ls=":", marker="o", ms=3, label="apple px visible")
    for i, s in enumerate(SETS):
        y = [R[(s, q, cam)]["apple_px_passed"] for q in SEQS]
        ax.plot(TS, np.asarray(y) + (i - 2) * 0.0, color=COL[s], lw=1.6, marker="os^vD"[i], ms=4.5, alpha=0.9, label=LABEL[s])
        side2[f"{cam}_{s}_apple_passed"] = y
    side2[f"{cam}_apple_visible"] = vis
    ax.set_title(f"apple px passing the self-filter — {cam}", fontsize=9.5, color=figstyle.INK)
    ax.set_xlabel("t_step (T14 planning seq 10–14)"); ax.set_ylabel("px"); ax.set_xticks(TS)
axes[0].legend(fontsize=6.6, frameon=False, loc="upper right")
ax = axes[2]
for i, s in enumerate(SETS):
    y = [sum(R[(s, q, c)]["robot_px_passed"] for c in ("zed_left", "wrist_cam_l", "wrist_cam_r")) for q in SEQS]
    ax.plot(TS, y, color=COL[s], lw=1.6, marker="os^vD"[i], ms=4.5, label=s)
    side2[f"robot_passed_3cams_{s}"] = y
vis = [sum(R[("A", q, c)]["robot_px_visible"] for c in ("zed_left", "wrist_cam_l", "wrist_cam_r")) for q in SEQS]
side2["robot_visible_3cams"] = vis
ax.set_ylim(-1, 10); ax.set_xticks(TS)
ax.set_title(f"robot px passing (leak), sum of 3 cameras\n(robot px visible per pose: {min(vis)}–{max(vis)})", fontsize=9.5, color=figstyle.INK)
ax.set_xlabel("t_step"); ax.set_ylabel("px"); ax.legend(fontsize=7, frameon=False)
fig.suptitle("Fig T19-2 — per-pose self-filter outcome, settings A/B/C/C2/D (wrist_cam_r sees 0 apple px at every pose)", fontsize=10, color=figstyle.INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/t19-selffilter-per-pose.png", dpi=170, facecolor=figstyle.SURFACE)
json.dump(side2, open(f"{OUT}/t19-selffilter-per-pose.json", "w"), indent=1)

# ---------- Fig 3: table ----------
tabs = json.load(open(f"{RAW}/inflation_tables.json")); att = json.load(open(f"{RAW}/attribution.json"))
links = list(tabs["tables"]["A"].keys())
fig = plt.figure(figsize=(15, 9.2)); fig.patch.set_facecolor(figstyle.SURFACE)
ax1 = fig.add_axes([0.02, 0.33, 0.40, 0.60]); ax1.axis("off")
cell = [[ln] + [f"{tabs['tables'][s][ln]:.3f}" for s in ("A", "B", "C", "D")] for ln in links]
cell.append(["guard R (m)", "0", "0", "0.06", "0.06"])
cell.append(["guard centre", "-", "-", "chunk_{k-1}", "apple GT"])
t = ax1.table(cellText=cell, colLabels=["link (self-filter model)", "A", "B", "C", "D"], loc="upper center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(7.0); t.scale(1, 1.02)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0: c.set_facecolor("#eeede8")
    elif i <= len(links) and j > 1 and cell[i - 1][j] != "0.050": c.set_facecolor(figstyle.SEQ_BLUE[1])
ax1.set_title(f"effective self_filter inflation per link (m) — {tabs['n_spheres']} spheres, from resolve_self_filter_inflation / inflation_table", fontsize=8.8, color=figstyle.INK, loc="left")
ax2 = fig.add_axes([0.46, 0.55, 0.53, 0.38]); ax2.axis("off")
cell2 = []
for key, v in att.items():
    ln = "  ".join(f"{k}:{x['erased_px_B']}" for k, x in v["erased_B_by_link"].items())
    bare = "  ".join(f"{k}:{x}" for k, x in v["inside_bare_sphere_by_link"].items()) or "-"
    cell2.append([key.replace("_", " ", 2), v["apple_px"], v["erased_px_B"], ln, v["inside_any_bare_sphere"], bare])
t2 = ax2.table(cellText=cell2, colLabels=["pose / camera", "apple px", "removed (B)", "removed by link (B, px; overlaps)", "in bare\nsphere (any)", "in bare sphere by link\n(inflation 0)"],
               loc="upper center", cellLoc="center", colWidths=[0.16, 0.06, 0.07, 0.33, 0.08, 0.34])
t2.auto_set_font_size(False); t2.set_fontsize(6.8); t2.scale(1, 1.5)
for (i, j), c in t2.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0: c.set_facecolor("#eeede8")
ax2.set_title("which link's sphere removes the apple px under B (robot_sphere_mask per link subset)", fontsize=8.8, color=figstyle.INK, loc="left")
ax3 = fig.add_axes([0.46, 0.05, 0.53, 0.42]); ax3.axis("off")
cell3 = []
for q, tt in zip(SEQS, TS):
    for s in SETS:
        rz, rl = R[(s, q, "zed_left")], R[(s, q, "wrist_cam_l")]
        near = sum(R[(s, q, c)]["cloud_passed_within_50mm_of_apple"] for c in ("zed_left", "wrist_cam_l", "wrist_cam_r"))
        napple = sum(R[(s, q, c)]["cloud_passed_within_50mm_by_body"].get("apple", 0) for c in ("zed_left", "wrist_cam_l", "wrist_cam_r"))
        gd = "-" if rz["guard_centre_to_apple_mm"] is None else f"{rz['guard_centre_to_apple_mm']:.0f}"
        cell3.append([f"t={tt}", s, f"{rz['apple_px_passed']}/{rz['apple_px_visible']}", f"{rl['apple_px_passed']}/{rl['apple_px_visible']}",
                      sum(R[(s, q, c)]["robot_px_passed"] for c in ("zed_left", "wrist_cam_l", "wrist_cam_r")), f"{near} ({napple})", gd])
t3 = ax3.table(cellText=cell3, colLabels=["pose", "set", "apple zed_left\npassed/vis", "apple wrist_l\npassed/vis", "robot px\npassed (3 cam)",
                                          "pts <50 mm of apple\n(of which apple body)", "guard centre\nto apple (mm)"], loc="upper center", cellLoc="center")
t3.auto_set_font_size(False); t3.set_fontsize(6.3); t3.scale(1, 0.62)
for j in range(7): t3[0, j].set_height(t3[0, j].get_height() * 2.2)
for (i, j), c in t3.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0: c.set_facecolor("#eeede8")
    elif (i - 1) // 5 % 2 == 1: c.set_facecolor("#f4f3ef")
fig.suptitle("Fig T19-3 — settings, per-link inflation, removal attribution and per-pose counts (T14 seq 10–14)", fontsize=10.5, color=figstyle.INK, x=0.02, ha="left")
fig.savefig(f"{OUT}/t19-selffilter-settings-table.png", dpi=170, facecolor=figstyle.SURFACE)
json.dump({"inflation_tables": tabs, "attribution_B": att, "rows": cell3}, open(f"{OUT}/t19-selffilter-settings-table.json", "w"), indent=1)
print("ok", ko)
