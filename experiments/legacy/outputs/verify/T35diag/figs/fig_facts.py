"""T35-diag figure 3 — (a) graph over all 42 runs (T28 E0/E0_rec/E1/E1_rec/E2_rec, T34 E3a/E3b; episodes 1800/1807/1808):
contact midpoint z - apple centre z at settle (mm) and minimum measured opening in the 24 control rows after settle, per condition,
success vs failure (a/settle.json). (b) fact table for the 4 failed T34 runs, candidates 1-5 (f/f_table.json).
Writes figures/t35/t35-grasp-geometry-all-runs.png/.json and figures/t35/t35-facts-table.png/.json.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as F

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t35'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag'))
ST = json.load(open(V / "a/settle.json"))["runs"]
FT = json.load(open(V / "f/f_table.json"))
GROUPS = ["E0", "E0_rec", "E1", "E1_rec", "E2_rec", "E3a_rec", "E3b_rec"]
C = F.CATEGORICAL

# ---- (a) graph
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
F.style_axes(fig, ax)
side = {"figure": None, "script": "outputs/verify/T35diag/figs/fig_facts.py", "doc": __doc__, "points": []}
rng = np.random.default_rng(1)
for gi, g in enumerate(GROUPS):
    for k, v in ST.items():
        if k.split("/")[0] != g:
            continue
        x = gi + rng.uniform(-0.18, 0.18)
        ok = bool(v["success"])
        kw = dict(color=C[0] if ok else C[1], marker="o" if ok else "X", s=46, edgecolor=F.SURFACE, linewidth=1, zorder=3)
        ax[0].scatter(x, v["contact_mid_rel_apple_mm"][2], **kw)
        ax[1].scatter(x, v["hold_min_open"], **kw)
        run = k.split("/")[1]
        if g.startswith("E3") and not ok:
            ax[0].annotate(run.split("_", 1)[1], (x, v["contact_mid_rel_apple_mm"][2]), fontsize=6.5, color=F.INK_2, xytext=(4, 2), textcoords="offset points")
        side["points"].append({"run": k, "success": ok, "contact_mid_z_mm": v["contact_mid_rel_apple_mm"][2], "hold_min_open": v["hold_min_open"],
                               "open_settle": v["open_settle"], "contact_lost_t": v["contact_lost_t"]})
for a in ax:
    a.set_xticks(range(len(GROUPS))); a.set_xticklabels(GROUPS, fontsize=8)
    a.grid(axis="y", color=F.GRID_INK, lw=0.6)
ax[0].axhline(0, color=F.INK_2, lw=1)
ax[0].set_ylabel("contact midpoint z - apple centre z\nat settle (mm)")
ax[0].set_title("where the fingers closed on the apple (height)", fontsize=9, color=F.INK)
ax[1].axhline(0.70, color=F.INK_2, lw=1, ls=":"); ax[1].text(-0.4, 0.705, "apple width ~0.70", fontsize=7, color=F.INK_2)
ax[1].set_ylabel("min measured opening,\n24 rows after settle (1 = open)")
ax[1].set_title("did the fingers stay blocked by the apple", fontsize=9, color=F.INK)
ax[0].scatter([], [], color=C[0], marker="o", label="success"); ax[0].scatter([], [], color=C[1], marker="X", label="failure")
ax[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("T35-diag: first-grasp geometry, all 42 runs on episodes 1800 / 1807 / 1808 (first close cycle)", fontsize=10, color=F.INK)
fig.tight_layout()
out = FIG / "t35-grasp-geometry-all-runs.png"; fig.savefig(out, dpi=140); plt.close(fig)
side["figure"] = str(out); out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print("wrote", out)

# ---- (b) table
R = FT["runs"]; runs = list(R)
f1 = lambda x: "-" if x is None else f"{x:.1f}"
f2 = lambda x: "-" if x is None else f"{x:.2f}"
def lost_chunk(v):
    ch = v["C1_TO"]["settle_to_contact_loss_chunks"]
    lost = v["C4_grasp"]["contact_lost_t"]
    return max((c for c in ch if c["t"] < lost), key=lambda c: c["t"])
rows = []
def add(cand, label, fn):
    rows.append([cand, label] + [fn(R[r]) for r in runs])
add("C1 TO", "max refined-ref, 10 before / 11 after close (deg)", lambda v: f"{f2(v['C1_TO']['pre10_max_dev_deg'])} / {f2(v['C1_TO']['post11_max_dev_deg'])} @t{v['C1_TO']['t_max_dev']}")
add("C1 TO", "variant that zeroes the max chunk", lambda v: ",".join(k for k, x in v["C1_TO"]["variants_dev_at_max_deg"].items() if k.startswith("no_") and x is not None and x < 0.05) or "none")
add("C1 TO", "chunk executing when contact was lost: t, refined-ref (deg)", lambda v: f"t{lost_chunk(v)['t']}{' held' if lost_chunk(v)['held'] else ''}, {f2(lost_chunk(v)['dev_deg'])}")
add("C1 TO", "  fingertip shift max (mm): recorded / replay / no_collision / no_continuity", lambda v: f"{f1(max(lost_chunk(v)['tip_disp_mm_per_step']))} / {f1(lost_chunk(v)['variants_tip_mm']['replay'])} / {f1(lost_chunk(v)['variants_tip_mm']['no_collision'])} / {f1(lost_chunk(v)['variants_tip_mm']['no_continuity'])}")
add("C1 TO", "  reference worst row (link, tier, mm, GT nearest)", lambda v: f"{lost_chunk(v)['ref_worst']['link']}, {lost_chunk(v)['ref_worst']['tier']}, {f1(lost_chunk(v)['ref_worst']['clearance_mm'])}, {lost_chunk(v)['ref_worst']['gt_nearest']}")
add("C1 TO", "returned in window (initial/accepted/projection)", lambda v: "/".join(str(v["C1_TO"]["returned_counts_window"].get(k, 0)) for k in ("initial", "accepted", "projection")))
add("C1 TO", "ref beyond TO pos. limit / beyond MJCF (chunks)", lambda v: f"{v['C1_TO']['n_ref_beyond_TO_position']} / {v['C1_TO']['n_ref_beyond_MJCF']}")
add("C2 policy", "first close t; finger-apple at close: meas / ref / refined (mm)", lambda v: f"t{v['C2_policy']['t_close']}; {f1(v['C2_policy']['d_exec_at_close_mm'])} / {f1(v['C2_policy']['d_ref_at_close_mm'])} / {f1(v['C2_policy']['d_refined_at_close_mm'])}")
add("C2 policy", "min finger-apple 16 rows after close: ref / meas (mm)", lambda v: f"{f1(v['C2_policy']['window16_min_d_ref_mm'])} / {f1(v['C2_policy']['window16_min_d_exec_mm'])}")
add("C2 policy", "same episode success E0 / E0_rec / E1 / E1_rec / E2_rec", lambda v: " / ".join(v["C2_policy"]["same_episode_success"][g] for g in ("E0", "E0_rec", "E1", "E1_rec", "E2_rec")))
add("C3 percep.", "manip!=apple / occluded_by_hand (of 21 chunks)", lambda v: f"{v['C3_perception_window']['n_manip_not_apple']} / {v['C3_perception_window']['n_occluded_by_hand']}")
add("C3 percep.", "split triggered / accepted; ball not whole apple", lambda v: f"{v['C3_perception_window']['n_split_triggered']} / {v['C3_perception_window']['n_split_accepted']}; {v['C3_perception_window']['n_ball_not_whole_inside']} of {v['C3_perception_window']['n_ball']}")
add("C4 grasp", "opening at settle; both fingers touching", lambda v: f"{f2(v['C4_grasp']['open_settle'])}; {'yes' if max(v['C4_grasp']['d_l1_settle_mm'], v['C4_grasp']['d_l2_settle_mm']) <= 1 else 'no'}")
add("C4 grasp", "contact midpoint - apple centre at settle (x,y,z mm)", lambda v: ", ".join(f"{x:+.1f}" for x in v["C4_grasp"]["contact_mid_rel_apple_mm"]))
add("C4 grasp", "hold: cmd median / min opening", lambda v: f"{f2(v['C4_grasp']['hold_cmd_median'])} / {f2(v['C4_grasp']['hold_min_open'])}")
add("C4 grasp", "contact lost t; apple dz max (mm); mid z rise (mm)", lambda v: f"t{v['C4_grasp']['contact_lost_t']}; {f1(v['C4_grasp']['hold_dz_max_mm'])}; {f1(v['C4_grasp']['hold_mid_z_rise_mm'])}")
add("C4 grasp", "apple xy pushed before close (mm)", lambda v: f1(v["C4_grasp"]["pre_close_max_apple_xy_mm"]))
add("C5 revoke", "in hand rows: attach exec 144-151 / revoke feedback 152-159", lambda v: (f"{v['C5_revoke']['attach_chunk_exec_t144_151']['n_in_hand']}/8 / {v['C5_revoke']['feedback_of_revoke_chunk_t160']['n_in_hand']}/8, last in hand t{v['C5_revoke']['last_in_hand_t']}") if "C5_revoke" in v else "no attach")
fig, ax = plt.subplots(figsize=(17, 0.30 * len(rows) + 0.6))
F.style_axes(fig, ax); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["cand.", "measured fact"] + runs, loc="center", cellLoc="left", colLoc="left",
              colWidths=[0.06, 0.30] + [0.16] * len(runs))
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.35)
for (i, j), cell in tb.get_celld().items():
    cell.set_edgecolor(F.GRID_INK); cell.set_facecolor(F.SURFACE if i % 2 else "#f2f1ec")
    if i == 0:
        cell.set_text_props(color=F.INK, weight="normal"); cell.set_facecolor("#e6e5df")
ax.set_title("T35-diag: measured facts per failed T34 run (candidates 1-5)", fontsize=11, color=F.INK)
fig.tight_layout()
out = FIG / "t35-facts-table.png"; fig.savefig(out, dpi=140); plt.close(fig)
out.with_suffix(".json").write_text(json.dumps({"figure": str(out), "script": "outputs/verify/T35diag/figs/fig_facts.py", "source": "outputs/verify/T35diag/f/f_table.json",
                                                "columns": ["cand.", "fact"] + runs, "rows": rows}, indent=1))
print("wrote", out)
