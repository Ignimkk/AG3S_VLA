"""T44dA figure (c): (top) 7 C1a-ok / C1b-fail pairs, (bottom) question summary. Numbers from CLA/*.json only.
-> AG3S/ag3s/docs/figures/t44/t44da-table.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle
C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44'))
pt = json.load(open(C / "pairs_table.json")); g = json.load(open(FIG / "t44da-graph.json")); ap = json.load(open(C / "approach_stats.json"))["per_arm"]
tl = json.load(open(C / "table_lift.json"))
figstyle.use_korean()
rows = []
for k, o in pt.items():
    pre = o["C1b_before_close"]; w = [s["field_worst_on"] or {} for s in pre]
    tf = [x for x in w if x.get("tier_on") == "target_free_coarse"]
    b = o["C1b_at_first_material"]
    rows.append([k, o["failure_mode"]["C1b"], f'{o["first_fin_diff_seq"]} / {o["first_material_seq"]}',
                 f'{"HOLD" if b["hold"] else "execute"}; {b["worst_row"]["tier"]}',
                 f'{len(tf)}/3', "-" if not tf else f'{min(x["h_bottom_mm"] for x in tf):.1f}-{max(x["h_bottom_mm"] for x in tf):.1f}',
                 "-" if not tf else f'{min(x["gap_target_mm"] for x in tf):.1f}-{max(x["gap_target_mm"] for x in tf):.1f}',
                 f'{max(s["dz7_mm"] for s in pre):.1f}',
                 f'{o["close_geom_c8"]["C1a"]["dz"]:.1f} / {o["close_geom_c8"]["C1b"]["dz"]:.1f}',
                 f'{o["hold_rows_after_close"]["C1a"]} / {o["hold_rows_after_close"]["C1b"]}'])
cols1 = ["pair", "C1b outcome", "first refined diff /\n> 0.01 rad (seq)", "C1b at first > 0.01 rad:\nverdict; worst row tier",
         "pre-close chunks with\nworst field row =\ntarget_free_coarse", "that row: bottom\nabove support (mm)", "that row: GT gap\nto target (mm)",
         "max TO lift dz7\npre-close (mm)", "grasp height dz at\nclose+8, C1a / C1b (mm)", "HOLD rows after\nclose, C1a / C1b"]
h = g["p4_hold_by_cause"]; p5 = g["p5_c2_arm_rows"]; p1 = g["p1_dz7_mm"]
summ = [
 ["Q1 C1b grasp failures", "coarse tier (support lift): confirmed\nHOLD as mechanism: refuted",
  f"approach (n=72): coarse lowers authorized-row clearance by median {ap['C1b']['field_off_minus_on_median']:.1f} mm; worst row tier target_free_coarse in {100*ap['C1b']['field_frac_worst_tfc']:.0f}% of chunks, "
  f"its sphere bottom {ap['C1b']['field_worst_tfc_h_bottom_median']:.1f} mm above support; TO lift median {p1['C1b']['median']:.1f} (C1a {p1['C1a']['median']:.1f}) mm; "
  f"support zero level under ball {tl['C1b']['centre_off_median_mm']:.1f}->{tl['C1b']['centre_on_median_mm']:.1f} mm, ring {tl['C1b']['ringout_off_median_mm']:.1f}->{tl['C1b']['ringout_on_median_mm']:.1f} mm"],
 ["Q2 C2 0/18", "C2-only: unauthorized arm_5 rows\nkeep a 50 mm target margin",
  f"422 spheres, permission mask 135 rows (fingers + palm, not empty); worst TO row = link_*_arm_5 in {p5['n_chunks']} chunks / {p5['n_runs']} runs, implied margin {p5['implied_margin_mm_median']:.0f} mm, "
  f"field d-r {p5['median_d_main_minus_r_mm']:.0f} mm; TO lift median {p1['C2']['median']:.1f} mm (p90 {p1['C2']['p90']:.1f}); HOLD 20 chunks (5 arm-link)"],
 ["Q3 C1a HOLD runs 5->14", "ball_owned ambiguous seeds at close",
  f"close-time HOLD (authorized finger, target-free tier, closing) in {h['C1a']['runs_by_cause'].get('authorized finger, target-free, at close')} runs / "
  f"{h['C1a']['chunks'].get('authorized finger, target-free, at close')} chunks (C0: 1); n_ambiguous 77-233; nearest kept seed = ambiguous in 14/15 (median 11.5 mm); split fired 0; strict 2 chunks"],
 ["Q4 hand-fruit_other C1a", "not an exclusion effect",
  "palm (EE_BODY) events: distractor kept in field (91-93% of vertices < 7.5 mm), TO-predicted -1.1..6.7 mm, sphere-model gap 9-16 mm actual; "
  "banana finger events: only 35-47% of banana vertices in field (main field same), planned GT gap -1..-11 mm"],
]
fig = plt.figure(figsize=(22, 9.5)); fig.patch.set_facecolor(figstyle.SURFACE)
ax1 = fig.add_axes([0.01, 0.47, 0.98, 0.46]); ax2 = fig.add_axes([0.01, 0.02, 0.98, 0.38])
for a_ in (ax1, ax2): a_.axis("off")
t1 = ax1.table(cellText=rows, colLabels=cols1, loc="upper center", cellLoc="center"); t1.auto_set_font_size(False); t1.set_fontsize(8.5); t1.scale(1, 2.0)
for (r_, c_), cell in t1.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r_ == 0: cell.set_facecolor("#eef2f8"); cell.get_text().set_wrap(True); cell.set_height(0.16)
ax1.set_title("C1a ok / C1b fail pairs (7): first divergence and the C1b pre-close chunks (3 chunks before the first close)", fontsize=10, color=figstyle.INK)
t2 = ax2.table(cellText=summ, colLabels=["question", "result", "numbers"], loc="upper center", cellLoc="left", colWidths=[0.11, 0.17, 0.72])
t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 3.2)
for (r_, c_), cell in t2.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK); cell.get_text().set_wrap(True)
    if r_ == 0: cell.set_facecolor("#eef2f8")
ax2.set_title("Question summary (18 pairs per arm; no statistical test beyond the CL-run McNemar values)", fontsize=10, color=figstyle.INK)
fig.savefig(FIG / "t44da-table.png", dpi=130)
json.dump({"pairs_columns": cols1, "pairs_rows": rows, "summary_rows": summ,
           "sources": ["outputs/verify/T44/CLA/pairs_table.json", "AG3S/ag3s/docs/figures/t44/t44da-graph.json", "outputs/verify/T44/CLA/approach_stats.json", "outputs/verify/T44/CLA/table_lift.json"],
           "script": "outputs/verify/T44/CLA/p/fig_table.py"}, open(FIG / "t44da-table.json", "w"), indent=1)
