"""T43 DX graphs + table (verifier). usage: fig_graphs_dx.py   (reads raw/dx1_summary.json, dx1_lag.json, dx2_summary.json, dx2_t43*.json, dx1_t43*.json)
Writes AG3S/ag3s/docs/figures/t43/t43dx-2-apple-graph, t43dx-5-finger-graph, t43dx-3-table (.png + .json); refuses to overwrite."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle

R = _ag3s_workspace_path('outputs/verify/T43/DX/raw/'); F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
figstyle.use_korean(); CAT = figstyle.CATEGORICAL
S1 = json.load(open(R + "dx1_summary.json")); LAG = json.load(open(R + "dx1_lag.json")); S2 = json.load(open(R + "dx2_summary.json"))
ev2 = [e for f in sorted(glob.glob(R + "dx2_t43*.json")) for e in json.load(open(f))["events"]]


def save(fig, name, data):
    p = F / f"{name}.png"; assert not p.exists(), p
    fig.savefig(p, dpi=130, facecolor=figstyle.SURFACE); json.dump(data, open(p.with_suffix(".json"), "w"), indent=1, default=float); print("wrote", p)


# ---------------- problem 1 graph
rows = S1["rows"]; lab = [f"{r['key'].split('/')[0]} {r['key'].split('_ep')[1].replace('_s', ' s')} t{r['t_star']}" for r in rows]
x = np.arange(len(rows))
fig, ax = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw={"width_ratios": [2.2, 1, 1]})
series = [("to_held_row_k_mm", "TO held rows at row k (recorded field, margin 0)", CAT[0], "o"),
          ("held_planned_true_bar_mm", "held spheres at TO-planned pose vs true bar", CAT[2], "s"),
          ("held_true_bar_mm", "held spheres at actual pose vs true bar", CAT[1], "D"),
          ("contact_pt_sd_mm", "apple contact point to held-sphere union (>0 = outside)", CAT[7], "^")]
for k, l, c, mk in series:
    ax[0].plot(x, [r[k] for r in rows], mk, color=c, label=l, ms=6)
ax[0].axhline(0, color=figstyle.INK, lw=0.8); ax[0].axhline(10, color=CAT[3], lw=1, ls="--", label="obstacle_margin 10 mm (robot rows only)")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=75, fontsize=7); ax[0].set_ylabel("mm"); ax[0].set_title("25 apple-hurdle carry contacts: clearance seen by TO vs true geometry", fontsize=10)
ax[0].legend(fontsize=7.5, loc="upper right")
base = [v for k_, r_ in json.load(open(glob.glob(R + "dx1_t43w3.json")[0]))["runs"].items() for c in r_["carry_chunks"] for _, _, v in c["ee_plan_vs_actual_post_mm"]]
allb = []
for f in glob.glob(R + "dx1_t43*.json"):
    for k_, r_ in json.load(open(f))["runs"].items():
        for c in r_["carry_chunks"]:
            allb += [v for _, _, v in c["ee_plan_vs_actual_post_mm"]]
ax[1].hist(allb, bins=np.arange(0, 150, 5), color=CAT[0], alpha=0.6, density=True, label=f"all carry rows of these runs (n={len(allb)})")
ax[1].hist([r["ee_plan_vs_actual_mm"] for r in rows], bins=np.arange(0, 150, 5), color=CAT[1], alpha=0.8, density=True, label="contact rows t* (n=25)")
ax[1].set_xlabel("|EE actual - EE planned (refined row k)| (mm)"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=7.5); ax[1].set_title("hand tracking offset", fontsize=10)
lg = [o["lag_steps"] for o in LAG["rows"]]
ax[2].scatter([r["slip_mm"] for r in rows], [r["slip_deg"] for r in rows], c=[CAT[1] if (r["contact_pt_sd_mm"] or 0) > 0 else CAT[0] for r in rows], s=30)
ax[2].axvline(5.0, color=CAT[3], ls="--", lw=1, label="body_cover slip pad 5 mm")
ax[2].set_xlabel("apple slip in hand since attach (mm)"); ax[2].set_ylabel("rotation since attach (deg)")
ax[2].set_title(f"slip (orange = contact point outside spheres); lag median {np.median(lg):.0f} steps", fontsize=9); ax[2].legend(fontsize=7.5)
figstyle.style_axes(fig, list(ax)); fig.tight_layout()
save(fig, "t43dx-2-apple-graph", {"rows": [{k: r[k] for k in ("key", "t_star", "to_held_row_k_mm", "held_planned_true_bar_mm", "held_true_bar_mm", "contact_pt_sd_mm", "ee_plan_vs_actual_mm", "slip_mm", "slip_deg")} for r in rows],
                                  "carry_rows_ee_offset_mm": allb, "lag_steps": lg, "source": [R + "dx1_summary.json", R + "dx1_lag.json"]})

# ---------------- problem 2 graph
r2 = S2["rows"]; x2 = np.arange(len(r2)); lab2 = [f"{r['key'].split('/')[0]} {r['key'].split('_ep')[1].replace('_s', ' s')} t{r['t']}" for r in r2]
fig, ax = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw={"width_ratios": [1, 1.4]})
for k, l, c, mk in (("composed_full_pb_mm", "full layer d at bar contact point", CAT[0], "o"), ("composed_free_pb_mm", "target-free layer d at bar contact point", CAT[1], "s"),
                    ("fc_esdf_new_mm", "FC: finger d - r (TO composition)", CAT[2], "^")):
    ax[0].plot(x2, [r[k] for r in r2], mk, color=c, ms=7, label=l)
ax[0].plot(x2, [-r["pb_dist_minus_ball_radius_mm"] for r in r2], "D", color=CAT[6], ms=6, label="depth of contact point inside carve-out ball")
ax[0].axhline(0, color=figstyle.INK, lw=0.8); ax[0].axhline(5, color=CAT[3], lw=1, ls="--", label="fine voxel 5 mm")
ax[0].set_xticks(x2); ax[0].set_xticklabels(lab2, rotation=60, fontsize=7.5); ax[0].set_ylabel("mm"); ax[0].legend(fontsize=7.5); ax[0].set_title("8 finger-bar contacts without bar in the TO field", fontsize=10)
for i, e in enumerate(ev2):
    h = [x_ for x_ in e["history"] if x_["t"] is not None]
    tt = [x_["t"] - e["t"] for x_ in h]
    c = CAT[i % len(CAT)]
    ax[1].plot(tt, [x_["fine_mm"] for x_ in h], "-", color=c, lw=1.2, label=f"{lab2[i]} full")
    ax[1].plot(tt, [x_["free_mm"] for x_ in h], ":", color=c, lw=1.6)
    inb = [x_ for x_ in h if x_.get("pb_in_ball")]
    ax[1].plot([x_["t"] - e["t"] for x_ in inb], [x_["free_mm"] for x_ in inb], "o", color=c, ms=5)
ax[1].set_xlabel("chunk planning t - contact t (control steps)"); ax[1].set_ylabel("recorded fine ESDF at bar contact point (mm)"); ax[1].set_ylim(-20, 120)
ax[1].set_title("history at the contact point: solid = full layer, dotted = target-free layer, dot = point inside carve-out ball", fontsize=9)
ax[1].legend(fontsize=6.5, ncol=2, loc="upper left")
figstyle.style_axes(fig, list(ax)); fig.tight_layout()
save(fig, "t43dx-5-finger-graph", {"rows": [{k: r[k] for k in ("key", "t", "composed_full_pb_mm", "composed_free_pb_mm", "fc_esdf_new_mm", "pb_dist_minus_ball_radius_mm")} for r in r2],
                                   "history": {f"{e['key']}:{e['t']}": e["history"] for e in ev2}, "source": [R + "dx2_summary.json"]})

# ---------------- table
c1, c2 = S1["counts"], S2["counts"]
t1 = [["apple contact point outside held-sphere union (actual pose)", f"{c1['not_covering']}/25"],
      ["apple slip in hand since attach > 5 mm pad", f"{c1['slip_gt_pad_5mm']}/25"],
      ["held rows get obstacle_margin 10 mm (rule: class 'held')", f"{c1['held_obstacle_margin_rule']}/25"],
      ["  recorded worst row held -> class 'held'", f"{c1['worst_row_held_class_held']}/{c1['worst_row_is_held']}"],
      ["bar absent in recorded ESDF at contact (d > voxel)", f"{c1['bar_absent_esdf']}/25"],
      ["field sees held-sphere overlap at actual pose (d - r < 0)", f"{c1['field_sees_overlap_actual']}/25"],
      ["TO held-row clearance at row k >= 0", f"{c1['to_held_row_k_ge0']}/25"],
      ["held spheres at TO-planned pose clear true bar", f"{c1['held_planned_true_gt0']}/25"],
      ["apple at TO-planned pose clear true bar", f"{c1['apple_planned_true_gt0']}/25"],
      ["chunk feasible / certified / executed", f"{c1['to_feasible']} / {c1['certified']} / {c1['chunk_all_rows_executed']}"],
      ["EE actual vs planned at t* (min / median / max mm)", " / ".join(f"{v:.1f}" for v in S1["ranges"]["ee_plan_vs_actual_mm"])],
      ["  all carry rows (median / p90 mm)", f"{S1['tracking_baseline_carry_rows_mm']['median']:.1f} / {S1['tracking_baseline_carry_rows_mm']['p90']:.1f}"]]
t2 = [["bar points absent in raw cloud near contact", f"{c2['points_absent_raw']}/8"], ["removed by self-filter (raw vs filtered cloud)", f"{c2['removed_by_selffilter_cloud']}/8"],
      ["bar pixels near contact self-filtered (any camera, partial)", f"{c2['bar_px_selffiltered']}/8"], ["bar not visible in any camera", f"{c2['not_visible_any_cam']}/8"],
      ["head pixel of contact point occluded (hand / arm)", f"{c2['head_pb_occluded']}/8"], ["left wrist camera sees bar near contact", f"{c2['wrist_l_bar_px_near_gt0']}/8"],
      ["contact point outside fine / coarse grid", f"{c2['outside_fine_grid']} / {c2['outside_coarse_grid']}"],
      ["bar present in full layer (d <= 5 mm)", f"{c2['bar_in_full_layer']}/8"], ["finger rows ask target-free layer", f"{c2['query_target_free_layer_fc']}/8"],
      ["bar absent in target-free layer (d > 5 mm)", f"{c2['bar_absent_target_free']}/8"], ["contact point inside target-free carve-out ball", f"{c2['pb_inside_ball']}/8"],
      ["AG3S target = bar (>= 90 % target points within 20 mm)", f"{c2['target_is_bar']}/8"]]
fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
for a_, tb, ttl in ((ax[0], t1, "Problem 1: 25 apple-hurdle carry contacts"), (ax[1], t2, "Problem 2: 8 finger-bar contacts without bar in TO field")):
    a_.axis("off"); tab = a_.table(cellText=tb, colLabels=["condition", "count"], loc="center", cellLoc="left", colWidths=[0.8, 0.2])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.45); a_.set_title(ttl, fontsize=10)
fig.tight_layout()
save(fig, "t43dx-3-table", {"problem1": t1, "problem2": t2, "counts1": c1, "counts2": c2})
