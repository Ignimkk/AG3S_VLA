"""T43 V10 graph + table figures: t43v10-graphs (graph) and t43v10-pairs-table (table). Numbers from V10/agg.json, ct.json, extra.json, hold_join.json and the analysis jsons. usage: fig_v10_graphs.py OUT_DIR"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V10')); T = V.parent; OUT = pathlib.Path(sys.argv[1])
a = json.load(open(V / "agg.json")); ex = json.load(open(V / "extra.json")); ct = json.load(open(V / "ct.json")); hj = json.load(open(V / "hold_join.json"))
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V10": json.load(open(V / "analysis/analysis.json"))["runs"]}
C = figstyle.CATEGORICAL; COL = {"V8": C[0], "V10": C[1]}; ARMS = ("V8", "V10"); figstyle.use_korean()
for n in ("t43v10-graphs", "t43v10-pairs-table"):
    for e in ("png", "json"):
        assert not (OUT / f"{n}.{e}").exists(), n
runs = sorted({(o["episode"], o["seed"]) for o in ct["runs"].values()})
lab = [f"{e}\ns{s}" for e, s in runs]
fig, ax = plt.subplots(2, 3, figsize=(20, 10.5)); figstyle.style_axes(fig, ax); ax = ax.ravel()
# (0) apple-bar min distance per pair
x = np.arange(len(runs)); w = 0.38
for j, arm in enumerate(ARMS):
    v = [1e3 * AN[arm][f"{arm}/H/{arm}_ep{e}_s{s}"]["hurdle"]["phases"]["all"]["B"]["min_m"] for e, s in runs]
    ax[0].bar(x + (j - 0.5) * w, v, w, color=COL[arm], label=arm)
ax[0].axhline(0, color=figstyle.INK, lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7)
ax[0].set_ylabel("min apple-hurdle distance over the run (hurdle_metrics B, mm; < 0 = contact)", fontsize=8, color=figstyle.INK_2); ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title("apple vs hurdle, 14 H pairs (value 164 = apple never lifted to the bar)", fontsize=10, color=figstyle.INK)
# (1) HOLD chunks per run by phase
ph_order = ["carry", "crate_above", "release_after"]; phc = {"carry": C[2], "crate_above": C[3], "release_after": C[4] if len(C) > 4 else C[0]}
for j, arm in enumerate(ARMS):
    per = a["hold"][arm]["per_run"]; bot = np.zeros(len(runs))
    for ph in ph_order:
        v = np.array([sum(1 for c in per.get(f"H:{e}_s{s}", {"chunks": []})["chunks"] if c["phase"] == ph) for e, s in runs], float)
        ax[1].bar(x + (j - 0.5) * w, v, w, bottom=bot, color=phc[ph], alpha=0.95 if arm == "V10" else 0.5, label=f"{arm} {ph}" if True else None); bot += v
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_ylabel("HOLD chunks (H runs)", fontsize=8, color=figstyle.INK_2)
h_, l_ = ax[1].get_legend_handles_labels(); ax[1].legend(h_[:6], l_[:6], fontsize=7, frameon=False, ncol=2)
ax[1].set_title(f"HOLD chunks per pair: V8 {a['hold']['V8']['by_stage_total'].get('H', 0)} H, V10 {a['hold']['V10']['by_stage_total'].get('H', 0)} H (left bar V8, right bar V10; faded = V8)", fontsize=10, color=figstyle.INK)
# (2) new HOLD causes: row type | class, and surface identity
S = hj["summary"]["new_V10_HOLD"]; cats = list(S["by_row_type_class"].items()); ids = list(S["by_surface_identity"].items())
yy = np.arange(len(cats)); ax[2].barh(yy, [v for _, v in cats], color=C[1]); ax[2].set_yticks(yy); ax[2].set_yticklabels([k for k, _ in cats], fontsize=8)
for i, (_, v) in enumerate(cats): ax[2].text(v + 0.3, i, str(v), va="center", fontsize=8, color=figstyle.INK)
ax[2].set_xlabel("new V10 HOLD chunks (51; HOLD at an (ep:seed, t) with no V8 HOLD)", fontsize=8, color=figstyle.INK_2); ax[2].set_title("binding row: row type | recorded margin class", fontsize=10, color=figstyle.INK)
# (3) surface identity
yy = np.arange(len(ids)); ax[3].barh(yy, [v for _, v in ids], color=C[3]); ax[3].set_yticks(yy); ax[3].set_yticklabels([k for k, _ in ids], fontsize=8)
for i, (_, v) in enumerate(ids): ax[3].text(v + 0.3, i, str(v), va="center", fontsize=8, color=figstyle.INK)
ax[3].set_xlabel("new V10 HOLD chunks", fontsize=8, color=figstyle.INK_2)
ax[3].set_title("nearest field surface of the binding row (GT geom group within 20 mm, observation state)", fontsize=10, color=figstyle.INK)
# (4) TO solve histogram
def samples(arm):
    s = []
    for k, r in AN[arm].items():
        s += [c["solve_ms"] for c in r["server_chunks"][1:] if c["solve_ms"] is not None]
    return np.array(s)
bins = np.linspace(0, 300, 61)
for arm in ARMS:
    s = samples(arm); ax[4].hist(np.clip(s, 0, 299), bins=bins, histtype="stepfilled", alpha=0.45, color=COL[arm], label=f"{arm} (n {len(s)})")
    t = a["timing"][arm]["ALL"]["to_solve_ms"]; ax[4].axvline(t["median"], color=COL[arm], lw=1.6); ax[4].axvline(t["p90"], color=COL[arm], lw=1.2, ls="--")
ax[4].set_xlabel("server TO solve (to.solve_ms), ms; clipped at 300; solid = median, dashed = p90", fontsize=8, color=figstyle.INK_2); ax[4].legend(fontsize=8, frameon=False)
ld8 = json.load(open(T / "V8/extra.json"))["loadavg_during_V8"]; ld10 = ex["loadavg_during_V10"]
ax[4].set_title(f"TO solve time (19 runs pooled); pod load1 median V8 {ld8['load1_median']:.1f} / V10 {ld10['load1_median']:.1f}", fontsize=10, color=figstyle.INK)
# (5) timing table
ax[5].axis("off"); rows = []
for lab_, f in (("TO solve median / p90 / max (ms)", lambda t: "%.1f / %.1f / %.1f" % (t["to_solve_ms"]["median"], t["to_solve_ms"]["p90"], t["to_solve_ms"]["max"])),
                ("round trip median / p90 (ms)", lambda t: "%.0f / %.0f" % (t["round_trip_ms"]["median"], t["round_trip_ms"]["p90"])),
                ("time_budget_hit_count", lambda t: str(t["time_budget_hit_count"])), ("chunks (excl. first)", lambda t: str(t["n_chunks_excl_first"]))):
    rows.append([lab_] + [f(a["timing"][arm]["ALL"]) for arm in ARMS])
for st in ("H", "B"):
    rows.append([f"TO solve median / p90 / max, {st}"] + ["%.1f / %.1f / %.1f" % tuple(a["timing"][arm][st]["to_solve_ms"][k] for k in ("median", "p90", "max")) for arm in ARMS])
tb = ax[5].table(cellText=rows, colLabels=["", "V8", "V10"], loc="center", cellLoc="center", colWidths=[0.5, 0.25, 0.25]); tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 2.0)
fig.suptitle("T43 V10 (V8 flags + --held-obstacle-margin) vs V8 — apple-hurdle distance, HOLD, new-HOLD binding rows, TO timing", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); out = OUT / "t43v10-graphs.png"; fig.savefig(out, dpi=105, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "sources": [str(V / x) for x in ("agg.json", "extra.json", "ct.json", "hold_join.json")], "pairs": [f"{e}_s{s}" for e, s in runs],
           "hold_new_summary": S, "timing": {arm: {st: a["timing"][arm][st]["to_solve_ms"] for st in ("ALL", "H", "B")} for arm in ARMS}, "loadavg": {"V8": ld8, "V10": ld10}}, open(OUT / "t43v10-graphs.json", "w"), indent=1, default=float)
# ---- pairs table
P = a["pairs_old_rule"]; Z = ex["H_pairs_z4_rule"]; CTP = ct["pairs"]["V10_vs_V8"]
def cell(x): return f"{x['V10_total']} vs {x['V8_total']}\n+{x['V10_only']} / -{x['V8_only']} / both {x['both']}\np {x['mcnemar_exact_p']:.2f}"
def ctc(x): return cell({"V10_total": x["A_total"], "V8_total": x["B_total"], "V10_only": x["A_only"], "V8_only": x["B_only"], "both": x["both"], "mcnemar_exact_p": x["mcnemar_exact_p"]})
rows = [["S (old rule)", cell(P["H"]["success"]), cell(P["B"]["success"]), cell(P["ALL"]["success"])], ["G (grasp)", cell(P["H"]["grasp"]), cell(P["B"]["grasp"]), cell(P["ALL"]["grasp"])],
        ["P (old rule)", cell(P["H"]["place"]), cell(P["B"]["place"]), cell(P["ALL"]["place"])],
        ["S (Z4 release rule)", cell(Z["success_rel"]), "-", "-"], ["P (Z4: place_rel)", cell(Z["place_rel"]), "-", "-"],
        ["S (CT rule: release + no pre-release contact)", ctc(CTP["success_ct"]), "-", "-"]]
fig, axs = plt.subplots(2, 1, figsize=(14, 13.5), gridspec_kw={"height_ratios": [3.3, 3.4]}); figstyle.style_axes(fig, axs)
for q in axs: q.axis("off")
t1 = axs[0].table(cellText=rows, colLabels=["metric (V10 total vs V8 total; +V10_only / -V8_only / both; McNemar exact p)", "hurdle H (14 pairs)", "no obstacle B (5 pairs)", "H + B (19 pairs)"], loc="center", cellLoc="center", colWidths=[0.34, 0.22, 0.22, 0.22])
t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1, 3.0); axs[0].set_title("T43 V10 vs V8, paired by ep:seed", fontsize=11, color=figstyle.INK)
fc = a["finger_contacts"]; hb = a["hold"]; hp = a["hold_pairs"]; ft = a["follow_to"]; ob = a["obstacle_stop"]; ca = ct["arms"]
def fcc(arm, ph): x = fc[arm]["by_phase"][ph]; return f"{x['n_runs']} runs / {x['n_rows']} rows / pen {x['max_pen_mm']:.1f} mm"
def carry_apple(arm):
    L = [(k, c) for k, o in ct["runs"].items() if o["arm"] == arm for c in o["contacts"] if c["kind"] == "apple" and c["phase"] == "carry"]
    return f"{len({k for k, _ in L})} runs / pen {max([c['max_pen_mm'] for _, c in L] or [0]):.1f} mm"
rows2 = [["apple-hurdle contact, carry", carry_apple("V8"), carry_apple("V10")],
         ["finger-hurdle contact, before grasp | carry (MJCF)", f"{fc['V8']['by_phase']['before_grasp']['n_runs']} runs | {fc['V8']['by_phase']['carry']['n_runs']} runs", f"{fc['V10']['by_phase']['before_grasp']['n_runs']} runs | {fc['V10']['by_phase']['carry']['n_runs']} runs"],
         ["finger-hurdle contact, after release (MJCF)", fcc("V8", "after_release"), fcc("V10", "after_release")],
         ["obstacle stop runs (H)", str(ob["V8"]["runs"]), str(ob["V10"]["runs"])],
         ["HOLD chunks total / runs with HOLD / trailing (19 runs)", f"{hb['V8']['total']} / {hb['V8']['runs_with_hold']} / {hb['V8']['trailing']}", f"{hb['V10']['total']} / {hb['V10']['runs_with_hold']} / {hb['V10']['trailing']}"],
         ["HOLD by binding category", str(hb["V8"]["by_category"]), str(hb["V10"]["by_category"])], ["HOLD by phase", str(hb["V8"]["by_phase"]), str(hb["V10"]["by_phase"])],
         ["HOLD only in this arm (same ep:seed, same t)", str(hp["V8_only_n"]), f"{hp['V10_only_n']}  {hp['V10_only_by_category']}"],
         ["follow_to executed / refused", f"{ft['V8']['n_executed']} / {sum(ft['V8']['refused_counts'].values())}", f"{ft['V10']['n_executed']} / {sum(ft['V10']['refused_counts'].values())}"],
         ["TO solve median / p90 / max (ms)", "%.1f / %.1f / %.1f" % tuple(a["timing"]["V8"]["ALL"]["to_solve_ms"][k] for k in ("median", "p90", "max")), "%.1f / %.1f / %.1f" % tuple(a["timing"]["V10"]["ALL"]["to_solve_ms"][k] for k in ("median", "p90", "max"))],
         ["time_budget_hit_count", str(a["timing"]["V8"]["ALL"]["time_budget_hit_count"]), str(a["timing"]["V10"]["ALL"]["time_budget_hit_count"])]]
t2 = axs[1].table(cellText=rows2, colLabels=["", "V8", "V10"], loc="center", cellLoc="center", colWidths=[0.40, 0.30, 0.30]); t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 2.2)
fig.tight_layout(); out = OUT / "t43v10-pairs-table.png"; fig.savefig(out, dpi=115, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "sources": [str(V / x) for x in ("agg.json", "ct.json", "extra.json")], "rows_pairs": rows, "rows_summary": rows2, "pairs_old_rule": P, "pairs_z4_rule": Z, "pairs_ct_rule": CTP}, open(OUT / "t43v10-pairs-table.json", "w"), indent=1, default=float)
print("ok")
