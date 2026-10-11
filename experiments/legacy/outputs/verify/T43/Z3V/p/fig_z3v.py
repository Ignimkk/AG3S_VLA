"""T43 Z3V figures: t43z3v-graph (HOLD per run + slip detach t + timeline), t43z3v-table (paired 2x2 + totals), t43z3v-scene (V5 vs V5D, run H 1925 s19253,
third_person_front.mp4 frames at release_t / detach_t / detach_t + 24). Numbers from Z3V/aggregate.json; sidecar .json per figure."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); Z = T / "Z3V"; FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
for n in ("graph", "table", "scene"):
    assert not (FIG / f"t43z3v-{n}.png").exists(), n
agg = json.load(open(Z / "aggregate.json")); P = agg["per_run"]
A5 = json.load(open(T / "W/analysis/analysis.json"))["runs"]; AD = json.load(open(Z / "analysis/analysis.json"))["runs"]
ko = figstyle.use_korean(); C = figstyle.CATEGORICAL; INK, INK2 = figstyle.INK, figstyle.INK_2
names = list(P)
short = lambda n: n.replace(":", " ").replace("_s", " s")

# ---------------- graph
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]}); figstyle.style_axes(fig, ax)
x = np.arange(len(names)); w = 0.4
h5 = [P[n]["V5"]["n_hold"] for n in names]; hd = [P[n]["V5D"]["n_hold"] for n in names]
t5 = [P[n]["V5"]["trailing_hold"] for n in names]; td = [P[n]["V5D"]["trailing_hold"] for n in names]
ax[0].bar(x - w / 2, h5, w, color=C[0], label="V5 HOLD chunks"); ax[0].bar(x + w / 2, hd, w, color=C[1], label="V5D HOLD chunks")
ax[0].bar(x - w / 2, t5, w, color="none", edgecolor=INK, hatch="////", lw=0.6, label="trailing HOLD (V5)")
ax[0].bar(x + w / 2, td, w, color="none", edgecolor=INK, hatch="////", lw=0.6)
ax[0].set_xticks(x); ax[0].set_xticklabels([short(n) for n in names], rotation=70, ha="right", fontsize=7); ax[0].set_ylabel("HOLD chunks (planning rows)", color=INK2)
ax[0].set_title(f"HOLD chunks per run: V5 {sum(h5)} vs V5D {sum(hd)}; trailing V5 {sum(t5)} vs V5D {sum(td)}", fontsize=9.5, color=INK); ax[0].legend(fontsize=8, frameon=False)
sl = [s for s in agg["slip_table"] if s["V5D_slip_flag"]]
y = np.arange(len(sl))
ax[1].scatter([s["V5D_release_t_measured"] for s in sl], y, marker="o", s=42, color=C[0], label="measured release t (MuJoCo replay)", zorder=3)
ax[1].scatter([s["predicted_slip_detach_t"] for s in sl if s["predicted_slip_detach_t"] is not None], [i for i, s in enumerate(sl) if s["predicted_slip_detach_t"] is not None],
              marker="D", s=70, facecolors="none", edgecolors=C[3], lw=1.6, label="predicted detach t (offline replay)", zorder=2)
ax[1].scatter([s["V5D_detach_t"] for s in sl], y, marker="x", s=48, color=C[1], lw=2, label="V5D actual slip-detach t", zorder=4)
ax[1].scatter([s["V5_detach_t"] for s in sl if s["V5_detach_t"] is not None], [i for i, s in enumerate(sl) if s["V5_detach_t"] is not None], marker="|", s=110, color=INK2, lw=1.6, label="V5 detach t (when it happened)", zorder=1)
ax[1].set_yticks(y); ax[1].set_yticklabels([short(s["run"]) for s in sl], fontsize=8); ax[1].invert_yaxis(); ax[1].set_xlabel("t (control step)", color=INK2)
ax[1].set_title(f"slip-detach events ({len(sl)} runs): release vs predicted vs actual", fontsize=9.5, color=INK); ax[1].legend(fontsize=7.5, frameon=False, loc="lower right")
ax[1].grid(axis="x", color=figstyle.GRID_INK, lw=0.6)
k = "H:1925_s19253"; key5 = next(q for q in A5 if q.startswith("V5/H/") and "ep1925_s19253" in q); keyd = next(q for q in AD if "ep1925_s19253" in q and q.startswith("V5D/H"))
for r, (lab, A, kk, col) in enumerate((("V5", A5, key5, C[0]), ("V5D", AD, keyd, C[1]))):
    ch = A[kk]["chunks"]; t = np.array([c["t"] for c in ch]); hold = np.array([c["hold"] for c in ch], float)
    ax[2].bar(t, hold, width=7, bottom=1.2 - r * 1.2, color=col, label=f"{lab} HOLD row ({int(hold.sum())})", align="edge", lw=0)
    ax[2].bar(t, 1 - hold, width=7, bottom=1.2 - r * 1.2, color=figstyle.GRID_INK, align="edge", lw=0)
rel = P[k]["V5D"]["release_t_measured"]; det = P[k]["V5D"]["detach_t"]
ax[2].axvline(rel, color=INK, ls=":", lw=1.2); ax[2].text(rel - 4, 2.78, f"release {rel} ", fontsize=8, color=INK, va="top", ha="right")
ax[2].axvline(det, color=C[1], ls="--", lw=1.4); ax[2].text(det + 4, 2.78, f" V5D slip detach {det}", fontsize=8, color=C[1], va="top")
ax[2].set_yticks([0.6, 1.8]); ax[2].set_yticklabels(["V5D", "V5"]); ax[2].set_xlabel("t (planning row)", color=INK2); ax[2].set_ylim(0, 2.9)
ax[2].set_title("H 1925 s19253: per-chunk HOLD (colored = HOLD, grey = executed)", fontsize=9.5, color=INK); ax[2].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("T43 Z3V — V5D (V5 + --held-slip-detach) vs V5, 19 paired runs (H 14 + B 5)", fontsize=11, color=INK)
fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.2, wspace=0.28)
fig.savefig(FIG / "t43z3v-graph.png", dpi=115, facecolor=figstyle.SURFACE)
json.dump({"figure": str(FIG / "t43z3v-graph.png"), "script": __file__, "source": str(Z / "aggregate.json"),
           "hold_per_run": {n: {"V5": P[n]["V5"]["n_hold"], "V5D": P[n]["V5D"]["n_hold"], "V5_trailing": P[n]["V5"]["trailing_hold"], "V5D_trailing": P[n]["V5D"]["trailing_hold"]} for n in names},
           "slip_rows": sl, "timeline_run": k}, open(FIG / "t43z3v-graph.json", "w"), indent=1, default=float)

# ---------------- table
fig, ax = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1.0, 1.25]}); figstyle.style_axes(fig, ax)
for a in ax:
    a.axis("off")
tab = agg["paired_2x2"]; rows = []
for stg in ("H", "B", "ALL"):
    for f, lab in (("success", "S old"), ("grasp", "G old"), ("place", "P old"), ("success_rel", "S Z4"), ("place_rel", "P Z4"), ("released", "released Z4")):
        x_ = tab[stg][f]
        rows.append([stg, lab, f"{x_['V5_total']}/{x_['n_pairs']}", f"{x_['V5D_total']}/{x_['n_pairs']}", x_["both"], x_["V5_only"], x_["V5D_only"], x_["neither"], f"{x_['mcnemar_exact_p']:.3f}"])
t0 = ax[0].table(cellText=rows, colLabels=["stage", "metric", "V5", "V5D", "both", "V5 only", "V5D only", "neither", "McNemar p"], loc="center", cellLoc="center")
t0.auto_set_font_size(False); t0.set_fontsize(8.5); t0.scale(1, 1.25)
ax[0].set_title("paired 2x2 (V5 vs V5D, same episode and seed), McNemar exact two-sided", fontsize=10, color=INK)
T_ = agg["totals"]; ss = agg["stops_summary"]
rows2 = [["HOLD chunks (19 runs)", T_["V5"]["n_hold"], T_["V5D"]["n_hold"]], ["runs with HOLD", T_["V5"]["runs_with_hold"], T_["V5D"]["runs_with_hold"]],
         ["trailing HOLD chunks (sum)", T_["V5"]["trailing_hold_chunks_sum"], T_["V5D"]["trailing_hold_chunks_sum"]], ["runs with trailing HOLD", T_["V5"]["runs_with_trailing_hold"], T_["V5D"]["runs_with_trailing_hold"]],
         ["held-row violation chunks (all)", T_["V5"]["held_row_viol_all"], T_["V5D"]["held_row_viol_all"]], ["held-row violation chunks, t >= detach_t", T_["V5"]["held_row_viol_ge_detach_t"], T_["V5D"]["held_row_viol_ge_detach_t"]],
         ["runs with detach", T_["V5"]["n_runs_with_detach"], T_["V5D"]["n_runs_with_detach"]], ["runs with slip detach", T_["V5"]["n_runs_slip_detach"], T_["V5D"]["n_runs_slip_detach"]],
         ["chunks after detach", T_["V5"]["finger_apple_after_detach"]["n_chunks_after_detach"], T_["V5D"]["finger_apple_after_detach"]["n_chunks_after_detach"]],
         ["  binding row = finger link", T_["V5"]["finger_apple_after_detach"]["n_binding_finger_link"], T_["V5D"]["finger_apple_after_detach"]["n_binding_finger_link"]],
         ["  finger row <= 20 mm from apple surface", T_["V5"]["finger_apple_after_detach"]["n_finger_apple_le_20mm"], T_["V5D"]["finger_apple_after_detach"]["n_finger_apple_le_20mm"]],
         ["  finger row <= 50 mm from apple surface", T_["V5"]["finger_apple_after_detach"]["n_finger_apple_le_50mm"], T_["V5D"]["finger_apple_after_detach"]["n_finger_apple_le_50mm"]],
         ["runs ended by obstacle stop", ss["V5"]["n_runs_stop"], ss["V5D"]["n_runs_stop"]], ["planning rows (sum)", ss["V5"]["sum_planning_rows"], ss["V5D"]["sum_planning_rows"]],
         ["place_t, 16 runs both placed: mean", f'{agg["place_t_pairs"]["mean_V5"]:.1f}', f'{agg["place_t_pairs"]["mean_V5D"]:.1f}'], ["place_t median diff V5D - V5", "", agg["place_t_pairs"]["median_diff_V5D_minus_V5"]]]
t1 = ax[1].table(cellText=rows2, colLabels=["measure", "V5", "V5D"], loc="center", cellLoc="center", colWidths=[0.6, 0.18, 0.18])
t1.auto_set_font_size(False); t1.set_fontsize(8.5); t1.scale(1, 1.3)
ax[1].set_title("totals over 19 paired runs", fontsize=10, color=INK)
fig.suptitle("T43 Z3V — V5D vs V5 tables", fontsize=11, color=INK); fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
fig.savefig(FIG / "t43z3v-table.png", dpi=115, facecolor=figstyle.SURFACE)
json.dump({"figure": str(FIG / "t43z3v-table.png"), "script": __file__, "source": str(Z / "aggregate.json"), "paired_2x2_rows": rows, "totals_rows": rows2}, open(FIG / "t43z3v-table.json", "w"), indent=1, default=float)

# ---------------- scene
import render_y as RY  # noqa: E402
Rp5 = np.load(T / "Y/replay/V5_H_V5_ep1925_s19253.npz"); RpD = np.load(Z / "replay/V5D_H_V5D_ep1925_s19253.npz")
fig, ax = plt.subplots(2, 3, figsize=(15, 8.2)); figstyle.style_axes(fig, ax)
crop = (slice(80, 330), slice(170, 500)); side = {"run": "H 1925 s19253", "panels": {}, "crop_rows_cols": [[80, 330], [170, 500]]}
for r, (lab, A, kk, rp) in enumerate((("V5", A5, key5, Rp5), ("V5D", AD, keyd, RpD))):
    for c, (tt, tl) in enumerate(((rel, "measured release t"), (det, "V5D slip-detach t"), (det + 24, "slip-detach t + 24"))):
        i = int(np.searchsorted(rp["t"], tt)); dist = float(np.linalg.norm(rp["apple_pos"][i + 1] - rp["finger_mid"][i + 1]) * 1e3)
        ax[r, c].imshow(RY.video_frame(A[kk]["run_dir"], tt)[crop]); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ch = {x["t"]: x for x in A[kk]["chunks"]}; row_t = (tt // 8) * 8
        hold_now = bool(ch[row_t]["hold"]) if row_t in ch else None
        info = {"t": int(tt), "label": tl, "grip_cmd": float(rp["grip_cmd"][i]), "finger_open_norm": float(rp["finger_open_norm"][i]), "apple_finger_mid_mm": dist,
                "apple_hand_contact_substeps": int(rp["cnt_apple-hand"][i]), "planning_row_t": int(row_t), "planning_row_hold": hold_now}
        side["panels"][f"{lab}@{tt}"] = info
        ax[r, c].set_title(f"{lab} t {tt} ({tl})\ncmd {info['grip_cmd']:.2f} · opening {info['finger_open_norm']:.2f} · apple-hand contacts {info['apple_hand_contact_substeps']} · apple-finger {dist:.0f} mm · row HOLD {hold_now}",
                           fontsize=8.2, color=INK)
fig.suptitle(f"T43 Z3V — H 1925 s19253: V5 (top) vs V5D (bottom) at release t {rel}, V5D slip-detach t {det}, +24 (third_person_front.mp4 crops)", fontsize=10.5, color=INK)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04, hspace=0.18)
fig.savefig(FIG / "t43z3v-scene.png", dpi=115, facecolor=figstyle.SURFACE)
json.dump({"figure": str(FIG / "t43z3v-scene.png"), "script": __file__, **side}, open(FIG / "t43z3v-scene.json", "w"), indent=1, default=float)
print("done", ko)
