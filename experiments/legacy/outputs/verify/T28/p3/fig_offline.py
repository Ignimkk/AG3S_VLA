"""T28 Phase 3 offline 0a-0e figures (numbers from p3/off_0ab.json, off_0ab_summary.json, off_0c.json, off_0d.json, off_0e_summary.json).
t28-p3-offline-grounding.png : 0a/0b graph — per chunk manipulated centroid -> GT apple (mm), apple-cluster narrowest extent vs 71.46 mm,
                               ball radius, T14 / T17, variants A (never frozen) / B (frozen from grasp attempt).
t28-p3-offline-table.png     : 0a-0e table (old -> new numbers)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean()
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p3'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28'))
C = fs.CATEGORICAL


def new(name):
    p = F / name
    assert not p.exists(), f"{p} exists"
    return p


AB = json.loads((V / "off_0ab.json").read_text())
ABS = json.loads((V / "off_0ab_summary.json").read_text())
C0 = json.loads((V / "off_0c.json").read_text())
D0 = json.loads((V / "off_0d.json").read_text())
E0 = json.loads((V / "off_0e_summary.json").read_text())
FR = {"t14": _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl'),
      "t17": _ag3s_workspace_path('outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl')}

# ---------------- graph
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
side = {"source": ["outputs/verify/T28/p3/off_0ab.json", "outputs/verify/T28/p3/off_0ab_summary.json"], "panels": {}}
for j, run in enumerate(("t14", "t17")):
    fr = {}
    for l in open(FR[run]):
        r = json.loads(l)
        if r["kind"] == "planning":
            fr[r["seq"]] = r
    z0 = fr[1]["object_poses"]["apple"]["pos"][2]
    for vi, (var, col, mk) in enumerate((("A_never_frozen", C[0], "o"), ("B_freeze_from_grasp_attempt", C[1], "x"))):
        ch = [e for e in AB["runs"][f"{run}_{var}"]["chunks"] if e["has_local_row"]]
        t = [8 * e["chunk"] for e in ch]
        d = [e["manipulated"]["to_apple_mm"] if e["manipulated"] else np.nan for e in ch]
        vis = [e["manipulated"]["state"] == "visible" if e["manipulated"] else False for e in ch]
        ax = axes[0, j]
        ax.plot(t, d, color=col, lw=1.2, label=f"{var}")
        ax.scatter([a for a, v in zip(t, vis) if v], [b for b, v in zip(d, vis) if v], color=col, marker=mk, s=16, zorder=3)
        side["panels"][f"{run}_{var}_manipulated_to_apple_mm"] = {"t": t, "mm": d, "visible": vis,
                                                                   "id": [e["manipulated"]["id"] if e["manipulated"] else None for e in ch]}
        if vi == 0:
            dz = [1000 * (fr[e["chunk"] + 1]["object_poses"]["apple"]["pos"][2] - z0) for e in ch]
            ax2 = ax.twinx(); ax2.plot(t, dz, color=C[2], lw=1, ls="--"); ax2.set_ylabel("GT apple dz (mm, dashed)", fontsize=8, color=C[2])
            side["panels"][f"{run}_apple_dz_mm"] = dz
            # narrowest extents of apple clusters
            ax = axes[1, j]
            ts, ne = [], []
            for e in ch:
                for c in e["clusters"]:
                    if c.get("nearest_centre") == "apple" and c["admissibility"]:
                        ts.append(8 * e["chunk"]); ne.append(c["admissibility"]["narrowest_extent_mm"])
            ax.scatter(ts, ne, color=C[0], s=18, label="apple-nearest cluster narrowest PCA extent")
            ax.axhline(71.46, color=C[7], lw=1.2, label="max opening 71.46 mm")
            ax.axhline(61.04, color=fs.GRID_INK, lw=1, ls=":", label="true apple narrowest 61.04 mm")
            side["panels"][f"{run}_apple_cluster_narrowest_mm"] = {"t": ts, "mm": ne}
            ax.set_ylabel("mm"); ax.set_ylim(0, 90)
            ax.legend(fontsize=7, loc="lower left")
            ax.set_title(f"{run.upper()} 0a apple-cluster narrowest extent (n={len(ne)}, max {max(ne):.1f}, margin {71.46 - max(ne):.1f} mm)", fontsize=9)
        ax = axes[2, j]
        br = [e["gate"].get("ball_radius_mm", np.nan) if e.get("gate") else np.nan for e in ch]
        near = [e["gate"].get("ball_centre_nearest") if e.get("gate") else None for e in ch]
        ax.plot(t, br, color=col, lw=1.1, label=f"{var}")
        for a, b, n in zip(t, br, near):
            if n and n != "apple":
                ax.scatter([a], [b], color=C[7], s=10, zorder=3)
        side["panels"][f"{run}_{var}_ball"] = {"t": t, "radius_mm": br, "centre_nearest": near}
    axes[0, j].set_title(f"{run.upper()} 0a manipulated centroid -> GT apple centre (markers = visible)", fontsize=9)
    axes[0, j].set_ylabel("mm"); axes[0, j].set_yscale("symlog", linthresh=50); axes[0, j].legend(fontsize=7, loc="upper left")
    axes[2, j].set_title(f"{run.upper()} 0b replayed ball radius (red dots: ball centre nearest a non-apple object)", fontsize=9)
    axes[2, j].set_ylabel("ball radius mm"); axes[2, j].set_xlabel("t (control steps, chunk k -> t = 8k)")
    axes[2, j].legend(fontsize=7)
for ax in axes.ravel():
    ax.grid(color=fs.GRID_INK, lw=0.5)
fig.patch.set_facecolor(fs.SURFACE)
fig.suptitle("T28 P3 offline 0a/0b — T26 identity replay on T14 / T17 recorded attention clouds (TargetConfirm max_opening 71.46 mm)", fontsize=11)
fig.tight_layout()
p = new("t28-p3-offline-grounding.png"); fig.savefig(p, dpi=120); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(p)

# ---------------- table
rows, side = [], {"source": ["off_0ab_summary.json", "off_0c.json", "off_0d.json", "off_0e_summary.json", "P1-3/p13b_selffilter_summary.json", "P1-2/p12_records.json"]}
for run in ("t14", "t17"):
    for var in ("A_never_frozen", "B_freeze_from_grasp_attempt"):
        s = ABS[f"{run}_{var}"]
        rows.append([f"0a {run} {var[:1]}", "manipulated != apple, pre-grasp / post-grasp",
                     "-", f"{s['pre_grasp']['manipulated_nearest_not_apple']}/{s['pre_grasp']['n']} | {s['post_grasp']['manipulated_nearest_not_apple']}/{s['post_grasp']['n']} (switched id {s['post_grasp']['manipulated_id_not_first(switched)']})"])
    a = ABS[f"{run}_A_never_frozen"]["apple_cluster_narrowest_extent_mm"]
    rows.append([f"0a {run}", "apple cluster narrowest extent (mm)", "-", f"n {a['n']} min {a['min']:.1f} p50 {a['p50']:.1f} max {a['max']:.1f}; margin {a['min_margin_to_max_opening_mm']:.1f}"])
    s = ABS[f"{run}_A_never_frozen"]["summary_from_off_0ab"]
    rows.append([f"0b {run} A", "gate refused (invariant_violation) / ball radius mm", "-",
                 f"{s['0b_gate_refused(invariant_violation)']} / {s['0b_ball_radius_mm']['min']:.1f}-{s['0b_ball_radius_mm']['max']:.1f} (centre nearest {s['0b_ball_centre_nearest_counts']})"])
for run in ("t14", "t16", "t17"):
    o = C0["summary"][f"{run}:constraint_e3:old:left"]; n = C0["summary"][f"{run}:constraint_e3:new:left"]
    rows.append([f"0c {run}", "E3 finger spheres: centre offset / pair gap model-real (mm, median)",
                 f"{o['centre_offset_mm']['median']:.1f} / {o['pair_gap_model_minus_real_mm']['median']:.1f}",
                 f"{n['centre_offset_mm']['median']:.1f} / {n['pair_gap_model_minus_real_mm']['median']:.1f}"])
h = D0["summary"]["hold_chunks"]
rows.append(["0d t16", f"HOLD chunks (n={h['n']}): worst point inside apple",
             f"T16 model, closed fingers: {h['T16model_oldfinger']['n_worst_point_inside_apple']} (P1-2 recorded apple-nearest {h['p12_recorded_worst_nearest_category_counts'].get('apple')})",
             f"T16 model: {h['T16model_T29finger']['n_worst_point_inside_apple']} | E3 model: {h['E3model_T29finger']['n_worst_point_inside_apple']}"])
rows.append(["0d t16", "HOLD worst nearest body", str(h["T16model_oldfinger"]["nearest_body_counts"]), f"{h['T16model_T29finger']['nearest_body_counts']} | {h['E3model_T29finger']['nearest_body_counts']}"])
for run in ("t14", "t17"):
    e = E0[run]; o = e["P1-3b_old_model_inflation0_for_reference"]
    rows.append([f"0e {run}", "robot leak px / apple retention / apple whole-deleted frames",
                 f"{o['robot_px_passed_total']} / {o['apple_retention']:.3f} / {o['n_frames_apple_visible_but_all_removed']}",
                 f"{e['robot_px_passed_total(leak)']} / {e['apple_retention']:.3f} / {len(e['frames_apple_whole_deleted_all_cameras'])}"])
side["rows"] = rows
fig, ax = plt.subplots(figsize=(17, 0.42 * len(rows) + 1.2)); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["item", "quantity", "old (pre-T26/T29 model or record)", "new (working tree T26+T29)"], loc="upper center", cellLoc="left",
              colWidths=[0.1, 0.3, 0.27, 0.33])
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
ax.set_title("T28 P3 offline 0a-0e (A = never frozen, B = frozen from grasp attempt; pre-grasp = before GT apple dz >= 50 mm)", fontsize=10)
fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
p = new("t28-p3-offline-table.png"); fig.savefig(p, dpi=125, bbox_inches="tight"); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(p)
