"""T43 CX figure 2 (graph): in-hand time lines of the 7 cases, hand acceleration at lift / drop, reversal split (verifier).
usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_cx_graph.py OUT_PNG
Panels 1-7 (one per case): x = control row - grasp row (grasp_t_used, cx_cases), left axis = apple slip in the hand frame (mm, |rel(t) - rel(ref)|,
cx_extract), clipped at 30 mm; right axis = gripper command of the active arm (1 open, 0 closed; thin lines); shaded = V11ABC HOLD rows;
vertical lines = separation rows (3 rows without apple-hand contact) of each run. V11AB blue, V11ABC orange.
Panel 8: measured hand (TCP) acceleration max |a| (m/s^2): lift window grasp row .. +20 (or separation) and drop window separation-24 .. +4 (cx_summary).
Panel 9: palm reversal share at chunk boundaries, split at each run's end-of-task row (cx_reversal_split), n reversals / n boundaries on the bars."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists() and not out.with_suffix(".json").exists(), f"{out} exists"
X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX'))
figstyle.use_korean(); C = figstyle.CATEGORICAL; AC = {"V11AB": C[0], "V11ABC": C[1]}
CS = json.load(open(X / "raw/cx_cases.json"))["cases"]; SM = json.load(open(X / "raw/cx_summary.json"))
side = {"script": __file__, "cases": {}}
fig, axs = plt.subplots(3, 3, figsize=(20, 14)); axs = axs.ravel()
for i, (ck, c) in enumerate(CS.items()):
    ax = axs[i]; ax2 = ax.twinx(); side["cases"][ck] = {}
    span = []
    for arm in ("V11AB", "V11ABC"):
        tl = c["arms"][arm]["timeline"]; gt = tl["grasp_t_used"]; ts = tl.get("t_sep")
        z = np.load(X / f"raw/tl/tl_{arm}_{ck.replace(':', '_')}.npz")
        t1 = min((ts if ts is not None else gt + 80) + 12, 599); T = np.arange(gt - 16, t1)
        sl = np.clip(z["slip_mm"][T + 1], 0, 30); g = z["grip_cmd"][T]
        ax.plot(T - gt, sl, color=AC[arm], lw=2, label=f"{arm} slip ({c['fail_stage'][arm]['stage']})")
        ax2.plot(T - gt, g, color=AC[arm], lw=0.8, alpha=0.8)
        if ts is not None:
            ax.axvline(ts - gt, color=AC[arm], lw=1, ls=":")
        if arm == "V11ABC":
            h = z["hold"][T].astype(bool)
            for k in np.flatnonzero(h):
                ax.axvspan(T[k] - gt - 0.5, T[k] - gt + 0.5, color=AC[arm], alpha=0.10, lw=0)
        span.append(t1 - gt)
        side["cases"][ck][arm] = {"grasp_t_used": gt, "t_sep": ts, "rows_rel": (T - gt).tolist(), "slip_mm_clipped30": sl.tolist(), "grip_cmd": g.tolist(),
                                  "hold_rows_rel": [int(x - gt) for x in T[z["hold"][T].astype(bool)]]}
    ax.set_ylim(0, 31); ax2.set_ylim(0, 1.05); ax.set_xlim(-16, max(span))
    ax.set_xlabel("control row - grasp row"); ax.set_ylabel("apple slip in hand (mm, clip 30)"); ax2.set_ylabel("gripper command", color=figstyle.INK_2, fontsize=8)
    ax2.tick_params(labelsize=7, colors=figstyle.INK_2)
    ax.set_title(f"{ck}  V11AB {c['fail_stage']['V11AB']['stage']} / V11ABC {c['fail_stage']['V11ABC']['stage']}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
# panel 8: accelerations
ax = axs[7]; cks = list(CS); xx = np.arange(len(cks)); w = 0.2; s8 = {}
for k, (arm, win) in enumerate([("V11AB", "lift_window"), ("V11ABC", "lift_window"), ("V11AB", "drop_window"), ("V11ABC", "drop_window")]):
    v = [(SM["cases"][ck][arm].get(win) or {}).get("acc_meas_max") or 0 for ck in cks]
    ax.bar(xx + (k - 1.5) * w, v, w, color=AC[arm], alpha=1.0 if win == "lift_window" else 0.45, label=f"{arm} {'lift (grasp..+20)' if win == 'lift_window' else 'drop (sep-24..+4)'}")
    s8[f"{arm}_{win}"] = dict(zip(cks, v))
ax.set_xticks(xx); ax.set_xticklabels([c.replace("_s", "\ns") for c in cks], fontsize=7); ax.set_ylabel("measured TCP max |a| (m/s^2)")
ax.set_title("measured hand acceleration (V11AB drop window = its release at the crate)", fontsize=10); ax.legend(fontsize=7)
side["accel"] = s8
# panel 9: reversal split
ax = axs[8]; RS = SM["reversal_split"]; groups = ["other_before", "other_after", "cx_case_before", "cx_case_after"]; s9 = {}
for k, arm in enumerate(("V11AB", "V11ABC")):
    a = RS[arm]["agg_n_boundaries_n_reversal"]; v = [a[g][1] / a[g][0] for g in groups]
    bars = ax.bar(np.arange(4) + (k - 0.5) * 0.38, v, 0.38, color=AC[arm], label=arm)
    for b_, g in zip(bars, groups):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.005, f"{a[g][1]}/{a[g][0]}", ha="center", fontsize=7)
    s9[arm] = {g: a[g] for g in groups}
ax.set_xticks(range(4)); ax.set_xticklabels(["other runs\nbefore end", "other runs\nafter end", "7 CX runs\nbefore end", "7 CX runs\nafter end"], fontsize=8)
ax.set_ylabel("palm reversal share (boundaries)"); ax.set_title("reversal share split at end-of-task row (place / drop / first detach)", fontsize=10); ax.legend(fontsize=8)
side["reversal_split"] = s9
figstyle.style_axes(fig, axs)
fig.suptitle("T43 CX: apple slip in the hand, gripper command, HOLD rows (shaded), hand acceleration, reversal share - V11AB vs V11ABC", fontsize=13)
fig.tight_layout(); fig.savefig(out, dpi=90, facecolor=figstyle.SURFACE)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(out)
