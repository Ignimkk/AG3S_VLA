"""T43 QM figures (1) paired trajectory, (2) metrics table, (3) paired distributions. Reads QM/qm_metrics.json only. Refuses to overwrite.
usage: fig_qm.py FIG_DIR"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

FD = pathlib.Path(sys.argv[1]); Q = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/QM'))
D = json.load(open(Q / "qm_metrics.json")); P = D["per_run"]; M = D["paired"]; S = D["fig1_series"]
figstyle.use_korean(); C = figstyle.CATEGORICAL; COL = {"V8": C[0], "V11AB": C[1]}
REPS = ["H:1807_s18073", "H:1967_s19672", "H:1968_s19681", "H:1995_s19952"]


def fresh(name):
    p = FD / name
    assert not p.exists() and not p.with_suffix(".json").exists(), f"{p} exists"
    return p


def carry_rows(s, arm_o):
    """in-hand rows up to the Z4 release row (or all in-hand rows)."""
    rel = arm_o["release_t"]
    return [i for i, t in enumerate(s["t"]) if rel is None or t <= rel]


# ------------------------------------------------------------------ (1) trajectory
out1 = fresh("t43qm-trajectory.png")
fig, axs = plt.subplots(2, 3, figsize=(19, 10.5)); side1 = {"script": __file__, "source": str(Q / "qm_metrics.json"), "representative": {}, "pooled": {}}
for ax, key in zip(axs.flat[:4], REPS):
    side1["representative"][key] = {}
    for arm in ("V8", "V11AB"):
        s = S[key][arm]; o = P[key][arm]; ix = carry_rows(s, o)
        sM = [s["s_M"][i] for i in ix]; hM = [s["h_M"][i] for i in ix]; sC = [s["s_C"][i] for i in ix]; hC = [s["h_C"][i] for i in ix]
        ax.plot(sM, hM, color=COL[arm], lw=2.2, label=f"{arm} measured M")
        ax.plot(sC, hC, color=COL[arm], lw=1.2, ls="--", label=f"{arm} commanded C")
        if o.get("crossing_t") is not None and o["crossing_t"] in s["t"]:
            i = s["t"].index(o["crossing_t"]); ax.scatter([s["s_M"][i]], [s["h_M"][i]], color=COL[arm], s=40, zorder=5, edgecolor="black")
        side1["representative"][key][arm] = {"t": [s["t"][i] for i in ix], "s_M": sM, "h_M": hM, "s_C": sC, "h_C": hC, "crossing_t": o.get("crossing_t"),
                                             "contact_apple_carry": o.get("contact_apple_carry"), "success_ct": o["success_ct"],
                                             "h_at_crossing_M": o.get("apple_bottom_minus_bar_top_at_crossing_M_mm"), "h_at_crossing_C": o.get("apple_bottom_minus_bar_top_at_crossing_C_mm")}
    ax.add_patch(plt.Circle((0, -12), 12, color=C[7], zorder=6)); ax.axhline(0, color=figstyle.GRID_INK, lw=0.8); ax.axvline(0, color=figstyle.GRID_INK, lw=0.8)
    ax.set_xlim(-260, 160); ax.set_ylim(-45, 185); ax.set_aspect("equal")
    a8, ab = P[key]["V8"], P[key]["V11AB"]
    ax.set_title(f"{key}  h@cross M: V8 {a8['apple_bottom_minus_bar_top_at_crossing_M_mm']:.1f} / V11AB {ab['apple_bottom_minus_bar_top_at_crossing_M_mm']:.1f} mm\n"
                 f"apple-hurdle contact in carry: V8 {a8['contact_apple_carry']} / V11AB {ab['contact_apple_carry']}", fontsize=9)
    ax.set_xlabel("s = apple centre across the bar plane (mm, + toward crate)"); ax.set_ylabel("apple bottom - bar top (mm)")
axs.flat[0].legend(fontsize=8, loc="upper left")
bins = np.arange(-260, 161, 10); cen = 0.5 * (bins[1:] + bins[:-1])
for ax, st_ in zip(axs.flat[4:], ("M", "C")):
    side1["pooled"][st_] = {"bin_centre_mm": cen.tolist()}
    for arm in ("V8", "V11AB"):
        mat = []
        for key in D["keys"]:
            if not key.startswith("H") or not S[key].get(arm):
                continue
            s = S[key][arm]; o = P[key][arm]
            if o.get("crossing_t") is None:
                continue
            ix = carry_rows(s, o); sv = np.array([s[f"s_{st_}"][i] for i in ix]); hv = np.array([s[f"h_{st_}"][i] for i in ix])
            row = [float(hv[(sv >= lo) & (sv < hi)].mean()) if np.any((sv >= lo) & (sv < hi)) else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
            mat.append(row)
        mat = np.array(mat); n = np.sum(np.isfinite(mat), 0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(mat, 0); q1 = np.nanpercentile(mat, 25, 0); q3 = np.nanpercentile(mat, 75, 0)
        ok = n >= 5
        ax.fill_between(cen[ok], q1[ok], q3[ok], color=COL[arm], alpha=0.22, lw=0)
        ax.plot(cen[ok], med[ok], color=COL[arm], lw=2.2, ls="-" if st_ == "M" else "--", label=f"{arm} median (IQR band), runs {mat.shape[0]}")
        side1["pooled"][st_][arm] = {"n_runs": int(mat.shape[0]), "n_per_bin": n.tolist(), "median": [None if not np.isfinite(x) else float(x) for x in med],
                                     "p25": [None if not np.isfinite(x) else float(x) for x in q1], "p75": [None if not np.isfinite(x) else float(x) for x in q3]}
    ax.add_patch(plt.Circle((0, -12), 12, color=C[7], zorder=6)); ax.axhline(0, color=figstyle.GRID_INK, lw=0.8); ax.axvline(0, color=figstyle.GRID_INK, lw=0.8)
    ax.set_xlim(-260, 160); ax.set_ylim(-45, 185); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"pooled H runs, {'measured M' if st_ == 'M' else 'commanded C'}\nper-run mean per 10 mm s-bin (in hand, up to release); bins with >= 5 runs", fontsize=9)
    ax.set_xlabel("s (mm)"); ax.set_ylabel("apple bottom - bar top (mm)")
figstyle.style_axes(fig, axs.flat)
fig.suptitle("T43 QM: apple height over the bar vs progress, V8 (blue) vs V11AB (orange); solid = measured M, dashed = commanded C; red disc = bar (r 12 mm); dot = measured crossing row", fontsize=11)
fig.tight_layout(); fig.savefig(out1, dpi=100, facecolor=figstyle.SURFACE); out1.with_suffix(".json").write_text(json.dumps(side1, default=float)); print(out1)

# ------------------------------------------------------------------ (2) table
TAB = [  # (label, field, kind, unit, scale)
    ("Success old (19 pairs)", ("success_old", "ALL"), "b"), ("Success Z4 release (19)", ("success_z4", "ALL"), "b"), ("Success CT contact-incl. (19)", ("success_ct", "ALL"), "b"),
    ("Success CT, H only (14)", ("success_ct", "H"), "b"),
    ("Collision run: robot, carry (H)", "contact_robot_carry", "b"), ("Collision run: apple, carry (H)", "contact_apple_carry", "b"),
    ("Collision run: robot, return (H)", "contact_robot_return", "b"), ("Collision run: any phase (H)", "contact_any_run", "b"),
    ("Collision row share any, carry (H) [%]", "contact_any_carry_row_share", "c", 100), ("Collision row share any, return (H) [%]", "contact_any_return_row_share", "c", 100),
    ("Min clearance M fingers (H) [mm]", "minclr_fingers_mm", "c"), ("Min clearance M palm (H) [mm]", "minclr_palm_mm", "c"),
    ("Min clearance M apple, carry (H) [mm]", "minclr_apple_carry_mm", "c"), ("Min clearance M other robot (H) [mm]", "minclr_robot_other_mm", "c"),
    ("Apple bottom - bar top @cross, C (H) [mm]", "apple_bottom_minus_bar_top_at_crossing_C_mm", "c"), ("Apple bottom - bar top @cross, M (H) [mm]", "apple_bottom_minus_bar_top_at_crossing_M_mm", "c"),
    ("Safety viol. plan C rows, window (H) [%]", "viol_C_row_share", "c", 100), ("Safety viol. M rows < 10 mm, all (H) [%]", "viol_M_row_share", "c", 100),
    ("Safety viol. M rows < 10 mm, window (H) [%]", "viol_M_window_row_share", "c", 100), ("Verdict violated chunks [%]", "violated_chunk_share", "c", 100), ("HOLD chunks [%]", "hold_chunk_share", "c", 100),
    ("Grasp retention (19)", ("retained", "ALL"), "b"),
    ("Slip fingermid, pre-open end [mm]", "slip_fm_pre_open_end_mm", "c"), ("Slip fingermid, max to release (Y2 window) [mm]", "slip_fm_max_mm", "c"), ("Slip TCP (Y2) at release [mm]", "slip_tcp_at_release_mm", "c"),
    ("Path apple grasp-release [mm]", "path_apple_mm", "c"), ("Path TCP M grasp-release [mm]", "path_tcp_M_mm", "c"), ("Path apple in window (H) [mm]", "path_apple_window_mm", "c"),
    ("t grasp [s]", "t_grasp_s", "c"), ("t crossing (H) [s]", "t_crossing_s", "c"), ("t release [s]", "t_release_s", "c"), ("t place (old) [s]", "t_place_s", "c"),
    ("grasp -> release [s]", "dur_grasp_to_release_s", "c"),
    ("Deflection onset, rows before crossing (H)", "deflection_onset_rows", "c"), ("  variant contiguous run (H)", "deflection_onset_rows_contiguous", "c"),
    ("Lift hold, rows around crossing (H)", "lift_hold_rows", "c"), ("Lifted rows in window (H)", "lifted_rows_in_window", "c"),
    ("TCP acc M p90, window&carry (H) [m/s2]", "acc_M_window_in_carry_p90", "c"), ("TCP acc C p90, window&carry (H) [m/s2]", "acc_C_window_in_carry_p90", "c"),
    ("TCP jerk M p90, window&carry (H) [m/s3]", "jerk_M_window_in_carry_p90", "c"), ("TCP jerk C p90, window&carry (H) [m/s3]", "jerk_C_window_in_carry_p90", "c"),
    ("TCP acc M p90, chunk boundary [m/s2]", "acc_M_boundary_p90", "c"), ("TCP acc M p90, inner [m/s2]", "acc_M_inner_p90", "c"),
    ("TCP acc C p90, chunk boundary [m/s2]", "acc_C_boundary_p90", "c"), ("TCP acc C p90, inner [m/s2]", "acc_C_inner_p90", "c"),
    ("TCP jerk M p90, boundary [m/s3]", "jerk_M_boundary_p90", "c"), ("TCP jerk C p90, boundary [m/s3]", "jerk_C_boundary_p90", "c"),
    ("TO server-recorded median [ms]", "to_server_median_ms", "c"), ("TO server-recorded p90 [ms]", "to_server_p90_ms", "c"), ("  pod load1 mean during run", "load1_mean_during_run", "c"),
    ("TO controlled TP cuda median [ms]", "to_tp_cuda_median_ms", "c"), ("TO controlled TP cuda p90 [ms]", "to_tp_cuda_p90_ms", "c"),
]
rows = []; side2 = {"script": __file__, "rows": []}
for e in TAB:
    lab, f, kind = e[0], e[1], e[2]; sc = e[3] if len(e) > 3 else 1
    if kind == "b":
        v = M[f[0]][f[1]] if isinstance(f, tuple) else M[f]
        r = [lab, f"{v['V8_true']}/{v['n_pairs']}", f"{v['V11AB_true']}/{v['n_pairs']}", f"McNemar b={v['V8_only']} c={v['V11AB_only']}", f"{v['p']:.3g}"]
    else:
        v = M[f]
        fm = lambda x: "-" if x is None else (f"{x * sc:.3g}")
        r = [lab, fm(v["V8_paired_median"]), fm(v["V11AB_paired_median"]), f"Wilcoxon W={v['W']:.1f} n={v['n_pairs']}" if v.get("W") is not None else f"no nonzero diff, n={v['n_pairs']}",
             "-" if v.get("p") is None else f"{v['p']:.3g}"]
    rows.append(r); side2["rows"].append({"label": lab, "field": f, "cells": r})
out2 = fresh("t43qm-table.png")
fig, ax = plt.subplots(figsize=(15, 0.31 * len(rows) + 1.4)); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["metric (continuous: paired median of per-run values; binary: runs true / pairs)", "V8", "V11AB", "paired statistic", "p"],
              loc="center", cellLoc="left", colLoc="left", colWidths=[0.47, 0.1, 0.1, 0.23, 0.08])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.25)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        c.set_facecolor("#ecebe6")
ax.set_title("T43 QM: V8 vs V11AB, paired by ep:seed (14 H + 5 B). H = hurdle runs only. Wilcoxon signed-rank (two-sided) / exact McNemar. Numbers only.", fontsize=10)
fig.tight_layout(); fig.savefig(out2, dpi=110, facecolor=figstyle.SURFACE); out2.with_suffix(".json").write_text(json.dumps(side2, default=float)); print(out2)

# ------------------------------------------------------------------ (3) distributions
DIST = [("minclr_fingers_mm", "min clearance M fingers (H) mm"), ("minclr_palm_mm", "min clearance M palm (H) mm"), ("minclr_apple_carry_mm", "min clearance M apple carry (H) mm"),
        ("apple_bottom_minus_bar_top_at_crossing_M_mm", "apple bottom - bar top @cross M (H) mm"), ("apple_bottom_minus_bar_top_at_crossing_C_mm", "apple bottom - bar top @cross C (H) mm"),
        ("deflection_onset_rows", "deflection onset, rows before crossing (H)"), ("lift_hold_rows", "lift hold rows around crossing (H)"),
        ("viol_M_window_row_share", "M rows < 10 mm in window (H), share"), ("acc_M_window_in_carry_p90", "TCP acc M p90 window&carry (H) m/s2"),
        ("jerk_M_window_in_carry_p90", "TCP jerk M p90 window&carry (H) m/s3"), ("acc_C_boundary_p90", "TCP acc C p90 chunk boundary m/s2"),
        ("slip_fm_pre_open_end_mm", "slip finger-mid pre-open mm"), ("path_apple_mm", "apple path grasp-release mm"), ("dur_grasp_to_release_s", "grasp -> release s"),
        ("to_server_median_ms", "TO server-recorded median ms (load differs)"), ("to_tp_cuda_median_ms", "TO controlled TP cuda median ms")]
out3 = fresh("t43qm-distributions.png")
fig, axs = plt.subplots(4, 4, figsize=(20, 17)); side3 = {"script": __file__, "panels": {}}
for ax, (f, lab) in zip(axs.flat, DIST):
    ks = [k for k in D["keys"] if P[k]["V8"].get(f) is not None and P[k]["V11AB"].get(f) is not None]
    a = [P[k]["V8"][f] for k in ks]; b = [P[k]["V11AB"][f] for k in ks]
    for x, y, k in zip(a, b, ks):
        ax.plot([0, 1], [x, y], color=figstyle.GRID_INK if k.startswith("H") else "#bdbcb6", lw=1.0, ls="-" if k.startswith("H") else ":", zorder=1)
    rng = np.random.default_rng(0); jit = rng.uniform(-0.05, 0.05, len(ks))
    ax.scatter(0 + jit, a, color=COL["V8"], s=22, zorder=3); ax.scatter(1 + jit, b, color=COL["V11AB"], s=22, zorder=3)
    ax.boxplot([a, b], positions=[-0.25, 1.25], widths=0.18, showfliers=False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["V8", "V11AB"]); ax.set_xlim(-0.5, 1.5)
    v = M[f]; pv = v.get("p")
    ax.set_title(f"{lab}\nn={v['n_pairs']}, median {v['V8_paired_median']:.3g} vs {v['V11AB_paired_median']:.3g}, p={'-' if pv is None else f'{pv:.3g}'}", fontsize=9)
    side3["panels"][f] = {"keys": ks, "V8": a, "V11AB": b, "W": v.get("W"), "p": pv}
figstyle.style_axes(fig, axs.flat)
fig.suptitle("T43 QM: per-run values, lines join the same ep:seed (solid = H, dotted = B); box = IQR; Wilcoxon signed-rank p", fontsize=12)
fig.tight_layout(); fig.savefig(out3, dpi=90, facecolor=figstyle.SURFACE); out3.with_suffix(".json").write_text(json.dumps(side3, default=float)); print(out3)
