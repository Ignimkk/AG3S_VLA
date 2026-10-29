"""T31-diag D3 figures: (1) representation table, (2) held apple / crate distance per chunk."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean(); C = fs.CATEGORICAL
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t31'))
V = json.loads(pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T31.diag.verify.json')).read_text())["D3_partial"]
R3 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d3/d3_records.json')).read_text())
OFF = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d3/d3_attached_offset.json')).read_text())
VOX = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d4/voxels_E3b_ep1807_r1_all.json')).read_text())
A = V["records_all_chunks"]
nch = sum(a["n_chunks"] for a in A.values())
ncand0 = sum(a["n_candidates_counts"].get("0", 0) for a in A.values())
nstat0 = sum(a["n_static_shapes_counts"].get("0", 0) for a in A.values())
rows = [
    ["crate (destination)", "TSDF → ESDF voxels (coarse 20 mm + fine 5 mm)", "label 'destination' on its points (margin 20 mm for held points whose nearest label is destination; placed_fn AABB)",
     f"none — n_candidates 0 in {ncand0}/{nch} chunks; n_static_shapes 0 in {nstat0}/{nch}", "pipeline.py:878, 1627-1628; linearize.py:697-703; placed.py:51"],
    ["table", "voxels (exclude_support_surfaces False)", "support planes built but TO use_support_planes False", "none", "serve_safe.py:713, 1187"],
    ["held apple (after attach)", "attached points = query points, radius 0 (25 / 33 pts); seeds removed only at those voxels", "1 bounding sphere (fit_sphere, r 30.6–32.3 mm) used only for the is-object margin test",
     "1 sphere (held apple, not crate)", "attached.py:177-187; clearance.py:115-155; linearize.py:704-722"],
    ["target apple (before grasp)", "voxels + target-free fine layer", "target-free exclusion ball (r 35.6–48.0 mm), nearest GT centre apple (orange after E3b r2 detach)", "ball (target, not crate)", "curobo_builder.py:1002"],
    ["robot", "self-filter 587 spheres; constraint 122 spheres (links gripper)", "", "robot spheres", "serve_safe log"],
]
cols = ["object", "distance field (what the TO reads)", "other geometry", "sphere / primitive?", "code"]
fig, ax = plt.subplots(figsize=(24, 4.2)); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colWidths=[0.1, 0.24, 0.33, 0.17, 0.16])
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 2.6)
for (i, j), cc in tb.get_celld().items():
    cc.set_edgecolor(fs.GRID_INK)
    if i == 0: cc.set_facecolor(fs.SEQ_BLUE[0])
    if i == 1: cc.set_facecolor("#fdf0ea")
fig.patch.set_facecolor(fs.SURFACE)
ax.set_title("D3 — how each object reaches the TO (E3a/E3b ep1807 records, 5 runs × 75 chunks; code = T30 working tree snapshot)" if not KO else
             "D3 — 각 물체가 TO 에 어떤 형태로 들어가나 (E3a/E3b ep1807 기록 5 run × 75 청크; 코드 = T30 working tree 사본)", fontsize=12)
fig.savefig(F / "t31-d3-representation-table.png", dpi=110, bbox_inches="tight")
(F / "t31-d3-representation-table.json").write_text(json.dumps({"cols": cols, "rows": rows, "n_chunks": nch, "n_candidates_0": ncand0, "n_static_shapes_0": nstat0}, indent=1, ensure_ascii=False))
# graph
runs = ["E3a_ep1807_r1", "E3a_ep1807_r3", "E3b_ep1807_r1", "E3b_ep1807_r2"]
fig, axes = plt.subplots(2, 2, figsize=(17, 9.5)); axes = axes.ravel()
side = {"sources": ["outputs/verify/T31diag/d3/d3_records.json", "outputs/verify/T31diag/d3/d3_attached_offset.json", "outputs/verify/T31diag/d4/voxels_E3b_ep1807_r1_all.json"]}
for i, r in enumerate(runs):
    ch = R3["B"][r]["chunks"]
    t = [c["t"] for c in ch]
    axes[0].plot(t, [c["held_apple_in_field"]["fine"]["value_at_gt_apple_centre_mm"] for c in ch], color=C[i], lw=1.4, label=f"{r} fine")
    axes[0].plot(t, [c["held_apple_in_field"]["coarse"]["value_at_gt_apple_centre_mm"] for c in ch], color=C[i], lw=1.0, ls="--", label=f"{r} coarse")
    oc = [c for c in OFF["runs"][r]["chunks"] if c["attached_centroid_to_gt_apple_mm"] < 100]
    axes[1].plot([c["t"] for c in oc], [c["attached_centroid_to_gt_apple_mm"] for c in oc], color=C[i], marker=".", label=r)
    axes[3].plot(t, [c["query_tiers_reference"]["min_crate_dist_mm"] for c in ch], color=C[i], marker=".", label=r)
axes[0].axhline(0, color=fs.INK, lw=0.8)
axes[0].set_ylabel("field value at GT apple centre (mm)"); axes[0].set_title("held apple in the field: d at GT apple centre (<0 = inside a surface)" if not KO else "쥔 사과가 필드에 있는가: GT 사과 중심의 d (<0 = 표면 안)", fontsize=10)
axes[0].legend(fontsize=7, ncol=2)
axes[1].set_ylabel("mm"); axes[1].set_title("|centroid(attached points) − GT apple centre| (before the drop)" if not KO else "|attached 점 중심 − GT 사과 중심| (떨어뜨리기 전)", fontsize=10); axes[1].legend(fontsize=8)
vc = VOX["chunks"]
for j, k in enumerate(["crate", "no_gt_surface_30mm", "apple", "other_fruit", "table", "robot"]):
    axes[2].plot([c["t"] for c in vc], [c["fine_class_counts"].get(k, 0) for c in vc], color=C[["crate", "table", "apple", "other_fruit", "robot", "other", "no_gt_surface_30mm"].index(k)], marker=".", label=k)
axes[2].set_yscale("log"); axes[2].set_ylabel("fine voxels with d ≤ 2.5 mm")
axes[2].set_title("E3b_ep1807_r1 fine window: occupied voxels by GT nearest body" if not KO else "E3b_ep1807_r1 fine 창: GT 최근접 물체별 점유 복셀", fontsize=10); axes[2].legend(fontsize=8)
axes[3].axhline(50, color=fs.INK_2, ls="--", lw=0.8)
axes[3].set_ylabel("mm"); axes[3].set_title("min GT distance, left-hand spheres + held points (reference, 8 steps) → crate" if not KO else "왼손 구 + 쥔 점 (reference 8 스텝) → crate 최소 GT 거리", fontsize=10); axes[3].legend(fontsize=8)
for a in axes: a.set_xlabel("chunk t")
fs.style_axes(fig, list(axes))
fig.suptitle("D3 — held apple and crate in the recorded field, grasp → release" if not KO else "D3 — 기록된 필드 속 쥔 사과와 crate, grasp → release", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(F / "t31-d3-held-apple-crate.png", dpi=110)
(F / "t31-d3-held-apple-crate.json").write_text(json.dumps(side, indent=1))
print("ok")
