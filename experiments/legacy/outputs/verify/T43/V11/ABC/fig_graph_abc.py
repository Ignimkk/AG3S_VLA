"""T43 V11ABC graph figure (verifier; copy of AB/fig_graph_ab.py, four arms, + panels g/h). usage: fig_graph_abc.py OUT_PNG   (reads V11/ABC/compare_ABC.json, V11/sharp_*.json, V11/raw/cross_*.json, XM/raw/cross_V8.json, analysis jsons)
Eight panels, V8 / V11A / V11AB / V11ABC (19 runs paired by ep:seed unless noted).
 (a) hurdle crossing, carry: apple bottom - bar top at the apple's carry crossing row, commanded (hollow) and measured (filled), per H run.
 (b) chunk-boundary command second difference (rad / row^2, executed command, clean boundaries, H + B pooled), cumulative share.
 (c) server TO solve time (to.solve_ms, first chunk of each run excluded), cumulative share.
 (d) TO deflection dominant axis at k* of window chunks with |D palm| > 5 mm (carry with apple, return).
 (e) grasp phase: max closing-chunk outward_mm per run (row-7 fingertip midpoint, refined vs reference; > 0 = away from the apple).
 (f) per-run TO solve median vs mean loadavg(1 min) during the run."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); V = T / "V11"; out = pathlib.Path(sys.argv[1])
assert not out.exists() and not out.with_suffix(".json").exists(), out
ARMS = ("V8", "V11A", "V11AB", "V11ABC"); NA = len(ARMS)
cmp_ = json.load(open(V / "ABC/compare_ABC.json"))
cross = {"V8": json.load(open(T / "XM/raw/cross_V8.json"))["runs"], "V11A": json.load(open(V / "raw/cross_V11A.json"))["runs"], "V11AB": json.load(open(V / "raw/cross_V11AB.json"))["runs"], "V11ABC": json.load(open(V / "raw/cross_V11ABC.json"))["runs"]}
sh = {"V8": json.load(open(V / "sharp_V11AB.json"))["arms"]["V8"], "V11A": json.load(open(V / "sharp_V11A.json"))["arms"]["V11A"], "V11AB": json.load(open(V / "sharp_V11AB.json"))["arms"]["V11AB"], "V11ABC": json.load(open(V / "sharp_V11ABC.json"))["arms"]["V11ABC"]}
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V11A": json.load(open(V / "analysis/analysis_V11A.json"))["runs"], "V11AB": json.load(open(V / "analysis/analysis_V11AB.json"))["runs"], "V11ABC": json.load(open(V / "analysis/analysis_V11ABC.json"))["runs"]}
C = figstyle.CATEGORICAL; COL = {"V8": C[0], "V11A": C[1], "V11AB": C[2], "V11ABC": C[4]}; figstyle.use_korean()
side = {"figure": str(out), "script": __file__}
fig, ax = plt.subplots(2, 4, figsize=(32, 10.5)); figstyle.style_axes(fig, ax.ravel())
# (a)
keys = sorted({k.split("/")[-1].split("_ep")[1] for k in cross["V11ABC"] if "error" not in cross["V11ABC"][k]})
a_rows = {}
for i, kk in enumerate(keys):
    for ai, arm in enumerate(ARMS):
        r = cross[arm].get(f"{arm}/H/{arm}_ep{kk}")
        if not r or "error" in r:
            continue
        cr = r["crossing"]; vals = {}
        for s_ in ("C", "M"):
            c = cr.get(f"carry_apple_{s_}")
            vals[s_] = c["apple_bottom_minus_bar_top_mm"] if c and c["inhand"] else None
        a_rows.setdefault(kk, {})[arm] = vals
        dx = (ai - 1.5) * 0.2
        if vals["M"] is not None:
            ax[0, 0].scatter(i + dx, vals["M"], color=COL[arm], s=34, zorder=3)
        if vals["C"] is not None:
            ax[0, 0].scatter(i + dx, vals["C"], facecolors="none", edgecolors=COL[arm], s=56, linewidths=1.5, zorder=3)
ax[0, 0].axhline(0, color=C[3], lw=1.0); ax[0, 0].set_xticks(range(len(keys))); ax[0, 0].set_xticklabels([k.replace("_s", ":") for k in keys], rotation=60, fontsize=7)
for arm in ARMS:
    ax[0, 0].scatter([], [], color=COL[arm], s=34, label=f"{arm} measured"); ax[0, 0].scatter([], [], facecolors="none", edgecolors=COL[arm], s=56, label=f"{arm} commanded")
ax[0, 0].legend(fontsize=6.5, frameon=False, ncol=2); ax[0, 0].set_ylabel("apple bottom - bar top at carry crossing (mm)", fontsize=8, color=figstyle.INK_2)
ax[0, 0].set_title("(a) hurdle crossing: apple bottom over the bar top (H runs with the apple in hand)", fontsize=9.5, color=figstyle.INK)
side["a_apple_bottom_minus_bar_top_mm"] = a_rows
# (b)
for arm in ARMS:
    v = np.sort([x for k, r in sh[arm]["per_run"].items() for x in r["acc_boundary_rad"]])
    ax[0, 1].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[arm], label=f"{arm} (n {len(v)}), median {np.median(v):.4f} / p90 {np.percentile(v, 90):.4f}")
ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("max |command second difference| across a chunk start (rad / row^2)", fontsize=8, color=figstyle.INK_2); ax[0, 1].set_ylabel("cumulative share", fontsize=8, color=figstyle.INK_2)
ax[0, 1].legend(fontsize=8, frameon=False); ax[0, 1].set_title("(b) chunk-boundary sharpness of the executed command (H + B)", fontsize=9.5, color=figstyle.INK)
side["b_boundary_acc"] = {arm: {s_: sh[arm][s_]["acc_boundary_rad"] for s_ in ("H", "B", "ALL")} for arm in ARMS}
# (c)
solve = {}
for arm in ARMS:
    s = [c["solve_ms"] for r in AN[arm].values() for c in r["server_chunks"][1:] if c["solve_ms"] is not None]; solve[arm] = np.sort(s)
    ld = cmp_["loadavg_per_round"][arm]
    ax[0, 2].plot(solve[arm], np.arange(1, len(s) + 1) / len(s), color=COL[arm], label=f"{arm} (n {len(s)}), median {np.median(s):.0f} / p90 {np.percentile(s, 90):.0f} / max {np.max(s):.0f} ms; load1 median {ld['load1_median']:.0f}")
ax[0, 2].set_xscale("log"); ax[0, 2].set_xlabel("server TO solve (to.solve_ms), ms", fontsize=8, color=figstyle.INK_2); ax[0, 2].set_ylabel("cumulative share", fontsize=8, color=figstyle.INK_2)
ax[0, 2].legend(fontsize=7, frameon=False); ax[0, 2].set_title("(c) TO solve time, 19 runs pooled (first chunk of each run excluded)", fontsize=9.5, color=figstyle.INK)
side["c_to_solve_ms"] = {arm: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max()), "load1_median_during_round": cmp_["loadavg_per_round"][arm]["load1_median"]} for arm, v in solve.items()}
# (d)
cats = ["up", "down", "away", "toward", "along_bar"]; w = 0.2
dd = cmp_["deflection_direction"]
for gi, g in enumerate(("carry_apple", "return")):
    for ai, arm in enumerate(ARMS):
        d = dd[arm][g].get("deflected_dominant_axis", {}) or {}
        x0 = gi * (len(cats) + 1) + np.arange(len(cats)) + (ai - 1.5) * w
        ax[1, 0].bar(x0, [d.get(c_, 0) for c_ in cats], width=w, color=COL[arm], label=arm if gi == 0 else None)
ax[1, 0].set_xticks([gi * (len(cats) + 1) + i for gi in range(2) for i in range(len(cats))]); ax[1, 0].set_xticklabels(cats * 2, fontsize=8, rotation=30)
ax[1, 0].set_ylabel("chunks (|D palm| > 5 mm at k*)", fontsize=8, color=figstyle.INK_2); ax[1, 0].legend(fontsize=8, frameon=False)
ax[1, 0].set_title("(d) deflection dominant axis: left group carry (apple in hand), right group return", fontsize=9.5, color=figstyle.INK)
side["d_dominant_axis"] = {arm: {g: dd[arm][g].get("deflected_dominant_axis") for g in ("carry_apple", "return")} for arm in ARMS}
# (e)
gp = cmp_["grasp_phase"]; rk = sorted(gp["V8"]["per_run"], key=lambda s: (s[0] != "H", s)); e_rows = {}
for i, kk in enumerate(rk):
    for ai, arm in enumerate(ARMS):
        e = gp[arm]["per_run"].get(kk)
        if not e or e["outward_mm_max"] is None:
            continue
        e_rows.setdefault(kk, {})[arm] = {"outward_mm_max": e["outward_mm_max"], "grasp": e["grasp"]}
        ax[1, 1].scatter(i + (ai - 1.5) * 0.2, e["outward_mm_max"], color=COL[arm], s=34, marker="o" if e["grasp"] else "x", zorder=3)
ax[1, 1].axhline(2.0, color=C[3], lw=1.0, ls="--"); ax[1, 1].axhline(0.0, color=figstyle.GRID_INK, lw=0.8)
ax[1, 1].set_xticks(range(len(rk))); ax[1, 1].set_xticklabels([k.replace("_s", ":") for k in rk], rotation=70, fontsize=6.5)
for arm in ARMS:
    ax[1, 1].scatter([], [], color=COL[arm], s=34, label=f"{arm} (G {gp[arm]['G']}/{gp[arm]['n_runs']})")
ax[1, 1].legend(fontsize=8, frameon=False); ax[1, 1].set_ylabel("max closing-chunk outward_mm (dashed = 2 mm)", fontsize=8, color=figstyle.INK_2)
ax[1, 1].set_title("(e) grasp phase: fingertip pushed away from the apple at closing (x = no grasp)", fontsize=9.5, color=figstyle.INK)
side["e_closing_outward_max_by_run"] = e_rows
# (f)
f_rows = {}
for arm in ARMS:
    xs = [v["load1_mean_during_run"] for v in cmp_["per_run_load_vs_to"].values() if v["arm"] == arm and v["load1_mean_during_run"] is not None and v["to_solve_median_ms"]]
    ys = [v["to_solve_median_ms"] for v in cmp_["per_run_load_vs_to"].values() if v["arm"] == arm and v["load1_mean_during_run"] is not None and v["to_solve_median_ms"]]
    ax[1, 2].scatter(xs, ys, color=COL[arm], s=34, label=f"{arm} (n {len(xs)})"); f_rows[arm] = {"load1_mean": xs, "to_solve_median_ms": ys}
ax[1, 2].set_yscale("log"); ax[1, 2].set_xlabel("pod loadavg(1 min), mean over the run", fontsize=8, color=figstyle.INK_2); ax[1, 2].set_ylabel("TO solve median per run (ms)", fontsize=8, color=figstyle.INK_2)
ax[1, 2].legend(fontsize=8, frameon=False); ax[1, 2].set_title("(f) per-run TO solve median vs pod load (pod totals, other sessions included)", fontsize=9.5, color=figstyle.INK)
side["f_load_vs_to"] = f_rows; side["f_spearman"] = cmp_["spearman_load_vs_to_median"]
# (g) boundary velocity ratio
for arm in ARMS:
    v = np.sort([x for k, r in sh[arm]["per_run"].items() for x in r["vel_boundary_ratio"]])
    ax[0, 3].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[arm], label=f"{arm} p90 {np.percentile(v, 90):.3f}, n>1: {int((v > 1).sum())} of {len(v)}, max {v.max():.2f}")
ax[0, 3].axvline(1.0, color=C[3], lw=1.0, ls="--"); ax[0, 3].set_xlabel("|Q0 - u[-1]| / max_step at the chunk boundary (> 1 = over the speed limit)", fontsize=8, color=figstyle.INK_2)
ax[0, 3].set_ylabel("cumulative share", fontsize=8, color=figstyle.INK_2); ax[0, 3].legend(fontsize=7, frameon=False); ax[0, 3].set_title("(g) chunk-boundary velocity ratio (H + B)", fontsize=9.5, color=figstyle.INK)
side["g_boundary_velocity_ratio"] = {arm: {s_: {k: sh[arm][s_][k] for k in ("vel_boundary_ratio", "n_vel_boundary_gt1")} for s_ in ("H", "B", "ALL")} for arm in ARMS}
# (h) palm reversal share + boundary acc p90 per arm / stage
xs_ = np.arange(3); wd = 0.2
for ai, arm in enumerate(ARMS):
    ax[1, 3].bar(xs_ + (ai - 1.5) * wd, [sh[arm][s_]["palm_reversal_share"] for s_ in ("H", "B", "ALL")], width=wd, color=COL[arm], label=arm)
ax[1, 3].set_xticks(xs_); ax[1, 3].set_xticklabels(["H", "B", "H + B"]); ax[1, 3].set_ylabel("share of chunk boundaries where the palm velocity reverses", fontsize=8, color=figstyle.INK_2)
ax[1, 3].legend(fontsize=8, frameon=False); ax[1, 3].set_title("(h) palm reversal share at chunk boundaries", fontsize=9.5, color=figstyle.INK)
side["h_palm_reversal_share"] = {arm: {s_: sh[arm][s_]["palm_reversal_share"] for s_ in ("H", "B", "ALL")} for arm in ARMS}
fig.suptitle("T43 V11ABC vs V11AB vs V11A vs V8 (V8 flags + TA; paired by ep:seed)", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=100, facecolor=figstyle.SURFACE); plt.close(fig)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(out)
