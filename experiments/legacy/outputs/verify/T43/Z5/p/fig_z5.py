"""T43 Z5 figures: t43z5-release-graph (graph), t43z5-release-table (table). Numbers: ../aggregate.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
Z = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Z5')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
A = json.load(open(Z / "aggregate.json")); G = A["groups"]
C = figstyle.CATEGORICAL; figstyle.use_korean()
ORDER = [("E0_Z5_14", "E0 Z5\nn=14"), ("E0_T39_48", "E0 T39\nn=48"), ("V4_H_14", "V4 H\nhurdle\nn=14"), ("V5_H_14", "V5 H\nhurdle\nn=14"),
         ("V4_B_5", "V4 B\nn=5"), ("V5_B_5", "V5 B\nn=5")]
MODE_COL = {"closed_cmd": C[1], "open_cmd": C[0]}
for name in ("t43z5-release-graph", "t43z5-release-table"):
    assert not (FIG / f"{name}.png").exists(), name

# ---- graph
fig, ax = plt.subplots(1, 4, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
figstyle.style_axes(fig, ax)
x = np.arange(len(ORDER)); lab = [l for _, l in ORDER]
cl = np.array([G[k]["summary"]["release_mode"]["closed_cmd"] for k, _ in ORDER]); op = np.array([G[k]["summary"]["release_mode"]["open_cmd"] for k, _ in ORDER])
nr = np.array([G[k]["summary"]["n_runs"] for k, _ in ORDER]); nn = nr - cl - op
ax[0].bar(x, cl / nr, color=MODE_COL["closed_cmd"], label="closed command at release"); ax[0].bar(x, op / nr, bottom=cl / nr, color=MODE_COL["open_cmd"], label="open command at release")
ax[0].bar(x, nn / nr, bottom=(cl + op) / nr, color=figstyle.GRID_INK, label="no release (no grasp)")
for i in range(len(ORDER)):
    ax[0].text(i, cl[i] / nr[i] / 2, f"{cl[i]}", ha="center", va="center", color="white", fontsize=10)
    ax[0].text(i, (cl[i] + op[i] / 2) / nr[i], f"{op[i]}", ha="center", va="center", color="white", fontsize=10)
    if nn[i]: ax[0].text(i, (cl[i] + op[i] + nn[i] / 2) / nr[i], f"{nn[i]}", ha="center", va="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=8); ax[0].set_ylabel("share of runs"); ax[0].set_title("release mode (applied gripper command at release)")
ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
rng = np.random.default_rng(0)
panels = [("release_apple_floor_gap_mm", "apple-floor gap at release (mm)"), ("release_v_horiz_mm_s", "|v_xy| at release (mm/s)"), ("post_release_horiz_travel_mm", "xy travel 2 s after release (mm)")]
side = {}
for a, (m, ttl) in zip(ax[1:], panels):
    side[m] = {}
    for i, (k, _) in enumerate(ORDER):
        runs = [r for r in G[k]["runs"] if r["release_t"] is not None]; side[m][k] = []
        for r in runs:
            v = r[m]
            if v is None: continue
            a.scatter(i + rng.uniform(-0.18, 0.18), v, color=MODE_COL[r["release_mode"]], s=26, alpha=0.85, edgecolor="none")
            side[m][k].append([r["pair"], r["release_mode"], v])
        vv = [r[m] for r in runs if r[m] is not None]
        if vv: a.hlines(np.median(vv), i - 0.3, i + 0.3, color=figstyle.INK, lw=2)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=8); a.set_title(ttl)
ax[1].set_ylabel("mm"); ax[1].text(0.02, 0.97, "dot = run (orange closed, blue open); bar = median", transform=ax[1].transAxes, fontsize=8, va="top")
fig.suptitle("T43 Z5: E0 (policy only) release metrics, Y2 definitions unchanged; V4 / V5 from Y (H runs carry the hurdle)")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = FIG / "t43z5-release-graph.png"; fig.savefig(out, dpi=120, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "source": str(Z / "aggregate.json"),
           "release_mode_counts": {k: G[k]["summary"]["release_mode"] for k, _ in ORDER}, "n_runs": {k: G[k]["summary"]["n_runs"] for k, _ in ORDER},
           "points_pair_mode_value": side, "median_all_release": {k: G[k]["summary"]["median_all_release"] for k, _ in ORDER}}, open(FIG / "t43z5-release-graph.json", "w"), indent=1, default=float)
print(out)

# ---- table
cols = ["group", "S/G/P", "release\nn", "closed /\nopen", "gap mm\n(med)", "v_z mm/s\n(med)", "|v_xy| mm/s\n(med)", "xy 2 s mm\n(med)", "slip@rel mm\n(med)", "TCP mm/s\nat open (med)", "closed-cmd only:\ngap mm / xy 2 s mm (med, n)"]
rows = []
f1 = lambda v: "-" if v is None else f"{v:.1f}"
for k, l in ORDER + [("E0_T39_same_pairs", "E0 T39, same ep:seed\n(no obst., n=6 of 14)")]:
    s = G[k]["summary"]; m = s["median_all_release"]; c = s["by_mode"]["closed_cmd"]
    rows.append([l.replace("\n", " "), f'{s["success"]}/{s["grasp"]}/{s["place"]} of {s["n_runs"]}', str(s["n_release"]), f'{s["release_mode"]["closed_cmd"]} / {s["release_mode"]["open_cmd"]}',
                 f1(m["release_apple_floor_gap_mm"]["median"]), f1(m["release_v_vert_mm_s"]["median"]), f1(m["release_v_horiz_mm_s"]["median"]), f1(m["post_release_horiz_travel_mm"]["median"]),
                 f1(m["in_hand_slip_at_release_mm"]["median"]), f1(m["tcp_speed_at_open_mm_s"]["median"]),
                 f'{f1(c["release_apple_floor_gap_mm"]["median"])} / {f1(c["post_release_horiz_travel_mm"]["median"])}  (n={c["n"]})'])
fig, ax = plt.subplots(figsize=(19, 3.9)); figstyle.style_axes(fig, [ax]); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 2.0)
tb.auto_set_column_width(list(range(len(cols))))
for (r, c_), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK); cell.set_facecolor(figstyle.SURFACE); cell.get_text().set_color(figstyle.INK)
    if r == 0: cell.set_height(cell.get_height() * 1.5); cell.get_text().set_fontweight("bold")
ax.set_title("T43 Z5: release metrics (Y2 definitions), median over runs with a release; V4 / V5 H runs carry the hurdle, B / E0 do not", fontsize=10)
out = FIG / "t43z5-release-table.png"; fig.savefig(out, dpi=120, facecolor=figstyle.SURFACE, bbox_inches="tight"); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "source": str(Z / "aggregate.json"), "columns": cols, "rows": rows}, open(FIG / "t43z5-release-table.json", "w"), indent=1, default=float)
print(out)
