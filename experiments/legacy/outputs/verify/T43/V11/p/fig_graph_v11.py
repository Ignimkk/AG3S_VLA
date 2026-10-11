"""T43 V11A graph figure (verifier). usage: V11_ARM=V11A fig_graph_v11.py OUT_PNG
Four panels, V8 vs ARM (19 runs paired by ep:seed unless noted). Numbers from V11/{xm_summary,sharp,extra,agg}_<ARM>.json, raw/cross_<ARM>.json (+ XM/raw/cross_V8.json), the analysis jsons.
 (a) hurdle crossing, carry: apple bottom - bar top at the apple's carry crossing row, commanded (C, executed_actions) and measured (M), per H run (hollow = commanded, filled = measured).
 (b) chunk-boundary command second difference (rad / row^2, executed command, clean boundaries, H + B pooled), cumulative share.
 (c) server TO solve time (to.solve_ms, first chunk of each run excluded), cumulative share.
 (d) TO deflection direction at k* of window chunks with |D palm| > 5 mm: dominant axis counts, carry (apple in hand) and return groups."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
ARM = os.environ.get("V11_ARM", "V11A"); T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); V = T / "V11"; out = pathlib.Path(sys.argv[1])
assert not out.exists() and not out.with_suffix(".json").exists(), out
cross = {"V8": json.load(open(T / "XM/raw/cross_V8.json"))["runs"], ARM: json.load(open(V / f"raw/cross_{ARM}.json"))["runs"]}
sh = json.load(open(V / f"sharp_{ARM}.json")); xs = json.load(open(V / f"xm_summary_{ARM}.json")); ex = json.load(open(V / f"extra_{ARM}.json"))
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], ARM: json.load(open(V / f"analysis/analysis_{ARM}.json"))["runs"]}
C = figstyle.CATEGORICAL; COL = {"V8": C[0], ARM: C[1]}; figstyle.use_korean()
side = {"figure": str(out), "script": __file__, "arm": ARM}
fig, ax = plt.subplots(2, 2, figsize=(16, 10.5)); figstyle.style_axes(fig, ax.ravel())
# (a)
keys = sorted({k.split("/")[-1].split("_ep")[1] for k in cross[ARM] if "error" not in cross[ARM][k]})
a_rows = {}
for i, kk in enumerate(keys):
    for arm in ("V8", ARM):
        r = cross[arm].get(f"{arm}/H/{arm}_ep{kk}")
        if not r or "error" in r:
            continue
        cr = r["crossing"]; vals = {}
        for st in ("C", "M"):
            c = cr.get(f"carry_apple_{st}")
            vals[st] = c["apple_bottom_minus_bar_top_mm"] if c and c["inhand"] else None
        a_rows.setdefault(kk, {})[arm] = vals
        dx = -0.12 if arm == "V8" else 0.12
        if vals["M"] is not None:
            ax[0, 0].scatter(i + dx, vals["M"], color=COL[arm], s=40, zorder=3)
        if vals["C"] is not None:
            ax[0, 0].scatter(i + dx, vals["C"], facecolors="none", edgecolors=COL[arm], s=60, linewidths=1.6, zorder=3)
ax[0, 0].axhline(0, color=C[3], lw=1.0); ax[0, 0].set_xticks(range(len(keys))); ax[0, 0].set_xticklabels([k.replace("_s", ":") for k in keys], rotation=60, fontsize=7)
ax[0, 0].scatter([], [], color=COL["V8"], s=40, label="V8 measured"); ax[0, 0].scatter([], [], facecolors="none", edgecolors=COL["V8"], s=60, label="V8 commanded")
ax[0, 0].scatter([], [], color=COL[ARM], s=40, label=f"{ARM} measured"); ax[0, 0].scatter([], [], facecolors="none", edgecolors=COL[ARM], s=60, label=f"{ARM} commanded")
ax[0, 0].legend(fontsize=7, frameon=False); ax[0, 0].set_ylabel("apple bottom - bar top at carry crossing (mm)", fontsize=8, color=figstyle.INK_2)
ax[0, 0].set_title("(a) hurdle crossing: apple bottom over the bar top (H runs with the apple in hand)", fontsize=9.5, color=figstyle.INK)
side["a_apple_bottom_minus_bar_top_mm"] = a_rows
# (b)
for arm in ("V8", ARM):
    v = np.sort([x for k, r in sh["arms"][arm]["per_run"].items() for x in r["acc_boundary_rad"]])
    ax[0, 1].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[arm], label=f"{arm} (n {len(v)}), p90 {np.percentile(v, 90):.4f}")
ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("max |command second difference| across a chunk start (rad / row^2)", fontsize=8, color=figstyle.INK_2); ax[0, 1].set_ylabel("cumulative share", fontsize=8, color=figstyle.INK_2)
ax[0, 1].legend(fontsize=8, frameon=False); ax[0, 1].set_title("(b) chunk-boundary sharpness of the executed command (H + B)", fontsize=9.5, color=figstyle.INK)
side["b_boundary_acc"] = {arm: {s: sh["arms"][arm][s]["acc_boundary_rad"] for s in ("H", "B", "ALL")} for arm in ("V8", ARM)}
side["b_palm_reversal"] = {arm: {s: {k: sh["arms"][arm][s][k] for k in ("palm_reversal_share", "palm_reversal_n", "n_boundaries_clean")} for s in ("H", "B", "ALL")} for arm in ("V8", ARM)}
# (c)
solve = {}
for arm in ("V8", ARM):
    s = [c["solve_ms"] for r in AN[arm].values() for c in r["server_chunks"][1:] if c["solve_ms"] is not None]; solve[arm] = np.sort(s)
    ax[1, 0].plot(solve[arm], np.arange(1, len(s) + 1) / len(s), color=COL[arm], label=f"{arm} (n {len(s)}), median {np.median(s):.0f} / p90 {np.percentile(s, 90):.0f} / max {np.max(s):.0f} ms")
ax[1, 0].set_xlabel("server TO solve (to.solve_ms), ms", fontsize=8, color=figstyle.INK_2); ax[1, 0].set_ylabel("cumulative share", fontsize=8, color=figstyle.INK_2)
ax[1, 0].legend(fontsize=8, frameon=False); ax[1, 0].set_title("(c) TO solve time, 19 runs pooled (first chunk of each run excluded)", fontsize=9.5, color=figstyle.INK)
side["c_to_solve_ms"] = {arm: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())} for arm, v in solve.items()}
# (d)
cats = ["up", "down", "away", "toward", "along_bar"]; w = 0.38
for gi, g in enumerate(("carry_apple", "return")):
    for ai, arm in enumerate(("V8", ARM)):
        d = xs["arms"][arm][g].get("deflected_dominant_axis", {}); tot = max(1, sum(d.values()))
        x0 = gi * (len(cats) + 1) + np.arange(len(cats)) + (ai - 0.5) * w
        ax[1, 1].bar(x0, [d.get(c_, 0) for c_ in cats], width=w, color=COL[arm], label=arm if gi == 0 else None)
ax[1, 1].set_xticks([gi * (len(cats) + 1) + i for gi in range(2) for i in range(len(cats))]); ax[1, 1].set_xticklabels(cats * 2, fontsize=8, rotation=30)
ax[1, 1].set_ylabel("chunks (|D palm| > 5 mm at k*)", fontsize=8, color=figstyle.INK_2); ax[1, 1].legend(fontsize=8, frameon=False)
ax[1, 1].set_title("(d) deflection dominant axis: left group carry (apple in hand), right group return", fontsize=9.5, color=figstyle.INK)
side["d_dominant_axis"] = {arm: {g: xs["arms"][arm][g].get("deflected_dominant_axis") for g in ("carry_apple", "carry_empty", "return")} for arm in ("V8", ARM)}
fig.suptitle(f"T43 {ARM} vs V8 (same flags + TA; paired by ep:seed)", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=115, facecolor=figstyle.SURFACE); plt.close(fig)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(out)
