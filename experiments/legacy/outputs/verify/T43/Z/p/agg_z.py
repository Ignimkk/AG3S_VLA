"""T43 Z — aggregation of raw/z1_held.json and raw/z4_release.json into raw/numbers_z.json (numbers only), plus the two figures
t43z-held-sphere-radii (graph) and t43z-release-rescore (bars + table). usage: agg_z.py [--overwrite]
Z1 per arm (V4 / V5 = stages H + B; V5S, V5B = H): over runs with an attach chunk, the query set at attach:
  largest sphere radius (median / min / max), all radii, body_cover_mode counts, need_x (x half-span + 20 mm) vs inner 82 mm,
  need_y vs 142 mm, apple_bottom_above_floor_needed_mm, carve-out ball radius; apple half-extents (mesh, all runs)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

Z = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Z')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
H = json.load(open(Z / "raw/z1_held.json")); RL = json.load(open(Z / "raw/z4_release.json"))
C = figstyle.CATEGORICAL; OW = "--overwrite" in sys.argv


def st(v):
    v = [x for x in v if x is not None]
    return {"n": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))} if v else {"n": 0}


Z1 = {"margin_mm": H["margin_mm"], "crate_inner_mm": H["crate_inner_mm"], "arms": {}, "per_run": {}}
ARMS1 = ["V4", "V5", "V5S", "V5B"]
for arm in ARMS1:
    rs = [o for k, o in H["runs"].items() if k.split("/")[0] == arm]
    att = [o for o in rs if o.get("attach")]
    q = [o["fit"][o["query_set"]] for o in att]
    Z1["arms"][arm] = {"n_runs": len(rs), "n_attach": len(att), "n_no_attach": len(rs) - len(att),
                       "largest_radius_mm": st([f["max_radius_mm"] for f in q]),
                       "body_cover_mode_counts": {m: sum(o["body_cover_mode"] == m for o in att) for m in sorted({o["body_cover_mode"] for o in att})},
                       "largest_radius_mm_by_mode": {m: st([o["fit"][o["query_set"]]["max_radius_mm"] for o in att if o["body_cover_mode"] == m]) for m in sorted({o["body_cover_mode"] for o in att})},
                       "n_spheres": st([len(o["sets"][o["query_set"]]) for o in att]),
                       "need_x_mm": st([f["need_x_mm"] for f in q]), "need_y_mm": st([f["need_y_mm"] for f in q]),
                       "n_fits_x": sum(f["fits_x"] for f in q), "n_fits_y": sum(f["fits_y"] for f in q),
                       "largest_r_plus_margin_mm": st([f["max_radius_mm"] + H["margin_mm"] for f in q]),
                       "apple_bottom_above_floor_needed_mm": st([f["apple_bottom_above_floor_needed_mm"] for f in q]),
                       "cover_radius_about_apple_mm": st([f["cover_radius_about_apple_mm"] for f in q]),
                       "carve_ball_radius_mm": st([(o.get("carve_ball") or {}).get("radius_mm") for o in att]),
                       "query_set_counts": {s: sum(o["query_set"] == s for o in att) for s in ("body_parent_frame", "lifted_parent_frame")}}
    for o in att:
        f = o["fit"][o["query_set"]]
        Z1["per_run"][o["key"]] = {"attach_t": o["attach_t"], "mode": o["body_cover_mode"], "query_set": o["query_set"],
                                   "radii_mm": [s["radius_mm"] for s in o["sets"][o["query_set"]]], "need_x_mm": f["need_x_mm"], "need_y_mm": f["need_y_mm"],
                                   "fits_x": f["fits_x"], "fits_y": f["fits_y"], "apple_bottom_above_floor_needed_mm": f["apple_bottom_above_floor_needed_mm"],
                                   "carve_ball_radius_mm": (o.get("carve_ball") or {}).get("radius_mm"), "success_old": o["success"]}
allatt = [o for o in H["runs"].values() if o.get("attach")]
Z1["apple_half_extent_mm"] = {ax: st([o["apple_half_extent_mm"][i] for o in allatt]) for i, ax in enumerate("xyz")}
Z1["apple_mesh_local_extent_mm"] = "x -29.9..31.1, y/z -33.3..33.3 (mesh vertices, before geom quat)"

Z4 = {"arms": RL["arms"], "pairs": RL["pairs"], "flips": RL["flips"], "crosscheck_vs_y_metrics": RL["crosscheck_vs_y_metrics"],
      "release_only_in_runs": {k: {f: o[f] for f in ("success", "success_rel", "release_t", "release_mode", "release_with_cmd", "in_crate_end", "sep_kept")}
                               for k, o in RL["runs"].items()}}
json.dump({"Z1": Z1, "Z4": Z4}, open(Z / "raw/numbers_z.json", "w"), indent=1, default=float)

# ---- figure: radii graph
figstyle.use_korean()
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6)); figstyle.style_axes(fig, axes)
MK = {"sphere": "o", "segments": "D"}
side = {"definitions": __doc__, "panels": {}}
for ax, (fld, title, ref) in zip(axes, (("max_radius_mm", "largest held query sphere radius at attach (mm)", None),
                                        ("need_x_mm", "x half-span of the sphere set + 20 mm margin (mm)", 82.0),
                                        ("apple_bottom_above_floor_needed_mm", "apple bottom height above crate floor needed\nfor lowest sphere clearance = 20 mm (mm)", None))):
    side["panels"][fld] = {}
    for i, arm in enumerate(ARMS1):
        pts = [(o, o["fit"][o["query_set"]]) for k, o in H["runs"].items() if k.split("/")[0] == arm and o.get("attach")]
        rng = np.random.default_rng(i)
        for o, f in pts:
            v = f[fld]; xj = i + rng.uniform(-0.18, 0.18)
            ax.scatter(xj, v, marker=MK[o["body_cover_mode"]], s=36, color=C[i], edgecolor=figstyle.INK if o["key"].endswith("1925_s19253") else "none", lw=1.2, zorder=3)
        side["panels"][fld][arm] = {o["key"]: f[fld] for o, f in pts}
    if ref:
        ax.axhline(ref, color=figstyle.INK, lw=1, ls="--"); ax.text(3.45, ref + 0.8, "crate inner half-width x = 82", fontsize=8, ha="right", color=figstyle.INK)
    ax.set_xticks(range(len(ARMS1))); ax.set_xticklabels([f"{a}\n(n {Z1['arms'][a]['n_attach']})" for a in ARMS1])
    ax.set_title(title, fontsize=10, color=figstyle.INK); ax.grid(True, axis="y", color=figstyle.GRID_INK, lw=0.5)
from matplotlib.lines import Line2D  # noqa: E402
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=figstyle.INK_2, label="body cover mode 'sphere'"),
                        Line2D([], [], marker="D", ls="", color=figstyle.INK_2, label="body cover mode 'segments'"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=figstyle.INK, label="ep1925 s19253 (black edge)")], fontsize=8, frameon=False, loc="center left")
fig.suptitle("T43 Z1 — held query spheres at attach, all runs with an attach chunk (V4, V5 = H + B; V5S, V5B = H)", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = FIG / "t43z-held-sphere-radii.png"; assert OW or not out.exists(), out
fig.savefig(out, dpi=110, facecolor=figstyle.SURFACE)
side.update(figure=str(out), arms=Z1["arms"], sources=["outputs/verify/T43/Z/raw/z1_held.json"]); json.dump(side, open(FIG / "t43z-held-sphere-radii.json", "w"), indent=1, default=float)
plt.close(fig)

# ---- figure: release re-score
ORDER = [("E0", "H"), ("V1", "H"), ("V3", "H"), ("V3", "B"), ("V4", "H"), ("V4", "B"), ("V5", "H"), ("V5", "B"), ("V5S", "H"), ("V5B", "H")]
A = RL["arms"]
fig = plt.figure(figsize=(16, 9.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], width_ratios=[1.25, 1], hspace=0.35, wspace=0.12)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, :]); figstyle.style_axes(fig, [ax1, ax2, ax3])
x = np.arange(len(ORDER)); lab = [f"{a} {s}\n(n {A[a][s]['n']})" for a, s in ORDER]
for k, (fld, nm) in enumerate((("success", "old success (grasp + place at last planning row)"), ("success_rel", "new success (grasp + release + separated + in crate at end)"))):
    v = [A[a][s][fld] for a, s in ORDER]
    ax1.bar(x + (k - 0.5) * 0.38, v, 0.38, color=C[k], label=nm)
    for xi, vi in zip(x, v):
        ax1.text(xi + (k - 0.5) * 0.38, vi + 0.15, str(vi), ha="center", fontsize=8, color=figstyle.INK)
ax1.set_xticks(x); ax1.set_xticklabels(lab, fontsize=8); ax1.set_ylabel("runs"); ax1.legend(fontsize=8, frameon=False, loc="upper left")
ax1.set_title("success per arm: old rule vs release rule", fontsize=10, color=figstyle.INK)
b0 = np.zeros(len(ORDER))
for k, (fld, nm) in enumerate((("release_open_cmd", "released, applied gripper command open"), ("release_closed_cmd", "released while command closed"), ("no_release", "no release (hand contact to the end / no grasp)"))):
    v = np.array([A[a][s][fld] for a, s in ORDER]); ax2.bar(x, v, 0.6, bottom=b0, color=C[k + 2], label=nm)
    for xi, vi, bi in zip(x, v, b0):
        if vi:
            ax2.text(xi, bi + vi / 2, str(vi), ha="center", va="center", fontsize=8, color=figstyle.SURFACE)
    b0 += v
ax2.set_xticks(x); ax2.set_xticklabels(lab, fontsize=7); ax2.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0)); ax2.set_ylim(0, 19)
ax2.set_title("release mode (Y definition)", fontsize=10, color=figstyle.INK)
ax3.axis("off")
hdr = ["pair (stage H, by ep:seed)", "rule", "both", "A only", "B only", "neither", "McNemar exact p"]
rows = []
for p, v in RL["pairs"].items():
    a_, b_ = p.split("_vs_")
    for fld, nm in (("success", "old success"), ("success_rel", "new success (release)")):
        q = v[fld]; rows.append([f"{a_} (A) vs {b_} (B)", nm, q["both"], q[f"{a_}_only"], q[f"{b_}_only"], q["neither"], f"{q['mcnemar_exact_p']:.4g}"])
tb = ax3.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center", colWidths=[0.2, 0.2, 0.08, 0.08, 0.08, 0.08, 0.12])
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.45)
for (r_, c_), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r_ == 0:
        cell.set_facecolor("#ecebe6")
nf = len(RL["flips"])
ax3.text(0.5, 0.02, f"runs whose success changes under the release rule: {nf}" + ("" if nf else " (none)") +
         f" · release_t / release_mode identical to Y y_metrics on {RL['crosscheck_vs_y_metrics']['n']} V4/V5 runs", transform=ax3.transAxes, ha="center", fontsize=9.5, color=figstyle.INK)
fig.suptitle("T43 Z4 — W runs re-scored with release (physics replay, qpos mismatch 0 on all 113 runs)", fontsize=11.5, color=figstyle.INK)
out = FIG / "t43z-release-rescore.png"; assert OW or not out.exists(), out
fig.savefig(out, dpi=110, facecolor=figstyle.SURFACE, bbox_inches="tight")
json.dump({"definitions": RL["definitions"], "figure": str(out), "arms": A, "pairs": RL["pairs"], "flips": RL["flips"], "table_rows": rows,
           "sources": ["outputs/verify/T43/Z/raw/z4_release.json"]}, open(FIG / "t43z-release-rescore.json", "w"), indent=1, default=float)
print("ok")
print(json.dumps(Z1["arms"], indent=0, default=float)[:4000]); print(Z1["apple_half_extent_mm"])
