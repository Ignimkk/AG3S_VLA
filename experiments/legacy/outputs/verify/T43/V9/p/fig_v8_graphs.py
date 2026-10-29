"""T43 V8 graph + table figures: t43v8-timing (graph) and t43v8-pairs-table (table). Numbers from V8/agg.json, ct.json, extra.json and the analysis jsons (raw chunk times). usage: fig_v8_graphs.py OUT_DIR"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V8')); T = V.parent; OUT = pathlib.Path(sys.argv[1])
a = json.load(open(V / "agg.json")); ex = json.load(open(V / "extra.json")); ct = json.load(open(V / "ct.json"))
AN = {"V7g": json.load(open(T / "V7g/analysis/analysis.json"))["runs"], "V8": json.load(open(V / "analysis/analysis.json"))["runs"]}
C = figstyle.CATEGORICAL; COL = {"V7g": C[0], "V8": C[1]}; ARMS = ("V7g", "V8"); figstyle.use_korean()
for n in ("t43v8-timing", "t43v8-pairs-table"):
    for e in ("png", "json"):
        assert not (OUT / f"{n}.{e}").exists(), n
def samples(arm):
    s, rt = [], []
    for k, r in AN[arm].items():
        s += [c["solve_ms"] for c in r["server_chunks"][1:] if c["solve_ms"] is not None]; rt += [c["client_round_trip_ms"] for c in r["chunks"][1:] if c["client_round_trip_ms"] is not None]
    return np.array(s), np.array(rt)
S = {x: samples(x) for x in ARMS}
# ---- timing graph
fig, ax = plt.subplots(1, 4, figsize=(19, 4.8)); figstyle.style_axes(fig, ax)
bins = np.linspace(0, 260, 53)
for x in ARMS:
    ax[0].hist(np.clip(S[x][0], 0, 259), bins=bins, histtype="stepfilled", alpha=0.45, color=COL[x], label=f"{x} (n {len(S[x][0])})")
    t = a["timing"][x]["ALL"]["to_solve_ms"]; ax[0].axvline(t["median"], color=COL[x], lw=1.6); ax[0].axvline(t["p90"], color=COL[x], lw=1.2, ls="--")
ax[0].set_xlabel("server TO solve (to.solve_ms), ms; clipped at 260; solid = median, dashed = p90", fontsize=8, color=figstyle.INK_2); ax[0].set_ylabel("chunks (first chunk of each run excluded)", fontsize=8, color=figstyle.INK_2)
ax[0].legend(fontsize=8, frameon=False); ax[0].set_title("TO solve time, 19 runs pooled", fontsize=10, color=figstyle.INK)
pr = ex["per_run_to_solve_median"]["runs"]; xs = np.array([v["V7g_median_ms"] for v in pr.values()]); ys = np.array([v["V8_median_ms"] for v in pr.values()])
ax[1].scatter(xs, ys, color=C[1], s=28); lim = max(xs.max(), ys.max()) * 1.05; ax[1].plot([0, lim], [0, lim], color=figstyle.GRID_INK); ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
ax[1].set_xlabel("V7g per-run median TO solve, ms", fontsize=8, color=figstyle.INK_2); ax[1].set_ylabel("V8 per-run median TO solve, ms", fontsize=8, color=figstyle.INK_2)
ax[1].set_title(f"paired by ep:seed (19); V8 higher in {ex['per_run_to_solve_median']['n_runs_V8_higher']}; median ratio {ex['per_run_to_solve_median']['ratio_median']:.2f}", fontsize=10, color=figstyle.INK)
for x in ARMS:
    v = np.sort(S[x][1]); ax[2].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[x], label=x)
ax[2].axvline(533.3, color=C[3], ls="--"); ax[2].text(540, 0.05, "533.3 ms (chunk period)", fontsize=8, color=figstyle.INK_2)
ax[2].set_xlabel("client round trip (timing_ms.policy_infer), ms", fontsize=8, color=figstyle.INK_2); ax[2].set_ylabel("cumulative share", fontsize=8, color=figstyle.INK_2); ax[2].legend(fontsize=8, frameon=False)
ax[2].set_title("client round trip", fontsize=10, color=figstyle.INK)
ax[3].axis("off")
rows = []
for lab, f in (("TO solve median / p90 / max (ms)", lambda t: "%.1f / %.1f / %.1f" % (t["to_solve_ms"]["median"], t["to_solve_ms"]["p90"], t["to_solve_ms"]["max"])),
               ("verdict trajopt median / p90 / max", lambda t: "%.1f / %.1f / %.1f" % (t["verdict_trajopt_ms"]["median"], t["verdict_trajopt_ms"]["p90"], t["verdict_trajopt_ms"]["max"])),
               ("round trip median / p90 (ms)", lambda t: "%.0f / %.0f" % (t["round_trip_ms"]["median"], t["round_trip_ms"]["p90"])),
               ("round trip > 533.3 ms", lambda t: "%d / %d (%.0f%%)" % (t["round_trip_gt_533_3_n"], t["round_trip_n"], 100 * t["round_trip_gt_533_3_share"])),
               ("time_budget_hit_count", lambda t: str(t["time_budget_hit_count"])), ("chunks (excl. first)", lambda t: str(t["n_chunks_excl_first"]))):
    rows.append([lab] + [f(a["timing"][x]["ALL"]) for x in ARMS])
for st in ("H", "B"):
    rows.append([f"TO solve median / p90 / max, {st} runs"] + ["%.1f / %.1f / %.1f" % tuple(a["timing"][x][st]["to_solve_ms"][k] for k in ("median", "p90", "max")) for x in ARMS])
tb = ax[3].table(cellText=rows, colLabels=["", "V7g", "V8"], loc="center", cellLoc="center", colWidths=[0.5, 0.25, 0.25]); tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.9)
ld = ex["loadavg_during_V8"]
fig.suptitle(f"T43 V8 vs V7g — GPU server TO timing (V8 = V7g + --finger-cover both, 270 vs 122 spheres). pod loadavg during V8: median {ld['load1_median']:.1f}, max {ld['load1_max']:.1f}", fontsize=10.5, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.93)); out = OUT / "t43v8-timing.png"; fig.savefig(out, dpi=115, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "sources": [str(V / "agg.json"), str(V / "extra.json")], "timing": {x: {s: {k: a["timing"][x][s][k] for k in ("n_chunks_excl_first", "to_solve_ms", "verdict_trajopt_ms", "verdict_total_ms", "round_trip_ms", "round_trip_gt_533_3_n", "round_trip_n", "round_trip_gt_533_3_share", "time_budget_hit_count")} for s in ("ALL", "H", "B")} for x in ARMS},
           "per_run_to_solve_median": ex["per_run_to_solve_median"], "loadavg_during_V8": ex["loadavg_during_V8"]}, open(OUT / "t43v8-timing.json", "w"), indent=1, default=float)
# ---- pairs table
P = a["pairs_old_rule"]; Z = ex["H_pairs_z4_rule"]; CTP = ct["pairs"]["V8_vs_V7g"]
def cell(x): return f"{x['V8_total']} vs {x['V7g_total']}\n+{x['V8_only']} / -{x['V7g_only']} / both {x['both']}\np {x['mcnemar_exact_p']:.2f}"
rows = [["S (old rule)", cell(P["H"]["success"]), cell(P["B"]["success"]), cell(P["ALL"]["success"])],
        ["G (grasp, rule-free)", cell(P["H"]["grasp"]), cell(P["B"]["grasp"]), cell(P["ALL"]["grasp"])],
        ["P (old rule)", cell(P["H"]["place"]), cell(P["B"]["place"]), cell(P["ALL"]["place"])],
        ["S (Z4 release rule)", cell(Z["success_rel"]), "-", "-"], ["P (Z4: place_rel)", cell(Z["place_rel"]), "-", "-"],
        ["S (CT rule: release + no pre-release contact)", cell({"V8_total": CTP["success_ct"]["A_total"], "V7g_total": CTP["success_ct"]["B_total"], "V8_only": CTP["success_ct"]["A_only"], "V7g_only": CTP["success_ct"]["B_only"], "both": CTP["success_ct"]["both"], "mcnemar_exact_p": CTP["success_ct"]["mcnemar_exact_p"]}), "-", "-"]]
fig, axs = plt.subplots(2, 1, figsize=(14, 12.5), gridspec_kw={"height_ratios": [3.3, 3.0]}); figstyle.style_axes(fig, axs)
for x in axs: x.axis("off")
t1 = axs[0].table(cellText=rows, colLabels=["metric (V8 total vs V7g total; +V8_only / -V7g_only / both; McNemar exact p)", "hurdle H (14 pairs)", "no obstacle B (5 pairs)", "H + B (19 pairs)"], loc="center", cellLoc="center", colWidths=[0.34, 0.22, 0.22, 0.22])
t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1, 3.0); axs[0].set_title("T43 V8 vs V7g, paired by ep:seed", fontsize=11, color=figstyle.INK)
fc = a["finger_contacts"]; hb = a["hold"]; hp = a["hold_pairs"]; ft = a["follow_to"]; ob = a["obstacle_stop"]
def fcc(arm, ph): x = fc[arm]["by_phase"][ph]; return f"{x['n_runs']} runs / {x['n_rows']} rows / pen {x['max_pen_mm']:.1f} mm"
rows2 = [["finger-hurdle contact, before grasp", fcc("V7g", "before_grasp"), fcc("V8", "before_grasp")], ["finger-hurdle contact, carry", fcc("V7g", "carry"), fcc("V8", "carry")],
         ["finger-hurdle contact, after release", fcc("V7g", "after_release"), fcc("V8", "after_release")],
         ["obstacle stop runs (H)", str(ob["V7g"]["runs"]), str(ob["V8"]["runs"])],
         ["HOLD chunks total / runs with HOLD (19 runs)", f"{hb['V7g']['total']} / {hb['V7g']['runs_with_hold']}", f"{hb['V8']['total']} / {hb['V8']['runs_with_hold']}"],
         ["HOLD by binding category", str(hb["V7g"]["by_category"]), str(hb["V8"]["by_category"])], ["HOLD by target label", str(hb["V7g"]["by_target"]), str(hb["V8"]["by_target"])],
         ["HOLD by phase", str(hb["V7g"]["by_phase"]), str(hb["V8"]["by_phase"])],
         ["HOLD only in this arm (same ep:seed, same t)", str(hp["V7g_only_n"]), f"{hp['V8_only_n']}  {hp['V8_only_by_category']}"],
         ["follow_to executed / refused", f"{ft['V7g']['n_executed']} / {sum(ft['V7g']['refused_counts'].values())} {ft['V7g']['refused_counts']}", f"{ft['V8']['n_executed']} / {sum(ft['V8']['refused_counts'].values())} {ft['V8']['refused_counts']}"],
         ["TO solve median / p90 / max (ms)", "%.1f / %.1f / %.1f" % tuple(a["timing"]["V7g"]["ALL"]["to_solve_ms"][k] for k in ("median", "p90", "max")), "%.1f / %.1f / %.1f" % tuple(a["timing"]["V8"]["ALL"]["to_solve_ms"][k] for k in ("median", "p90", "max"))],
         ["time_budget_hit_count", str(a["timing"]["V7g"]["ALL"]["time_budget_hit_count"]), str(a["timing"]["V8"]["ALL"]["time_budget_hit_count"])]]
t2 = axs[1].table(cellText=rows2, colLabels=["", "V7g", "V8"], loc="center", cellLoc="center", colWidths=[0.36, 0.32, 0.32]); t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 2.2)
fig.tight_layout(); out = OUT / "t43v8-pairs-table.png"; fig.savefig(out, dpi=115, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "sources": [str(V / "agg.json"), str(V / "ct.json"), str(V / "extra.json")], "rows_pairs": rows, "rows_summary": rows2,
           "pairs_old_rule": P, "pairs_z4_rule": Z, "pairs_ct_rule": CTP}, open(OUT / "t43v8-pairs-table.json", "w"), indent=1, default=float)
print("ok")
