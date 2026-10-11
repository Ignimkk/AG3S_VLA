"""T43 V11A pairs table figure (verifier). usage: V11_ARM=V11A fig_table_v11.py OUT_PNG
Table 1: S / G / P new (ARM) vs base (V8) under the old rule, the Z4 release rule and the CT contact-inclusive rule, paired by ep:seed (new total vs base total; +new_only / -base_only / both; McNemar exact p).
Table 2: contacts (CT replays, MuJoCo names) by phase and body, obstacle stops, HOLD (total, by category / target / phase, only-in-new), follow_to, TO timing.
Numbers from V11/{agg,ct,extra}_<ARM>.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
ARM = os.environ.get("V11_ARM", "V11A"); V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11')); out = pathlib.Path(sys.argv[1])
assert not out.exists() and not out.with_suffix(".json").exists(), out
a = json.load(open(V / f"agg_{ARM}.json")); ct = json.load(open(V / f"ct_{ARM}.json")); ex = json.load(open(V / f"extra_{ARM}.json"))
figstyle.use_korean()
P = a["pairs_old_rule"]; Z = ex["H_pairs_z4_rule"]; CTP = ct["pairs"][f"{ARM}_vs_V8"]


def cell(x, n="new", b="base"):
    return f"{x[n + '_total']} vs {x[b + '_total']}\n+{x[n + '_only']} / -{x[b + '_only']} / both {x['both']}\np {x['mcnemar_exact_p']:.2f}"


def ctcell(x):
    return cell({"new_total": x["A_total"], "base_total": x["B_total"], "new_only": x["A_only"], "base_only": x["B_only"], "both": x["both"], "mcnemar_exact_p": x["mcnemar_exact_p"]})


rows = [["S (old rule)", cell(P["H"]["success"]), cell(P["B"]["success"]), cell(P["ALL"]["success"])],
        ["G (grasp, rule-free)", cell(P["H"]["grasp"]), cell(P["B"]["grasp"]), cell(P["ALL"]["grasp"])],
        ["P (old rule)", cell(P["H"]["place"]), cell(P["B"]["place"]), cell(P["ALL"]["place"])],
        ["S (Z4 release rule)", cell(Z["success_rel"]), "-", "-"], ["P (Z4: place_rel)", cell(Z["place_rel"]), "-", "-"],
        ["S (CT rule: release + no pre-release contact)", ctcell(CTP["success_ct"]), "-", "-"]]
fig, axs = plt.subplots(2, 1, figsize=(14, 13), gridspec_kw={"height_ratios": [3.3, 3.6]}); figstyle.style_axes(fig, axs)
for x in axs:
    x.axis("off")
t1 = axs[0].table(cellText=rows, colLabels=[f"metric ({ARM} total vs V8 total; +{ARM}_only / -V8_only / both; McNemar exact p)", "hurdle H (14 pairs)", "no obstacle B (5 pairs)", "H + B (19 pairs)"], loc="center", cellLoc="center", colWidths=[0.34, 0.22, 0.22, 0.22])
t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1, 3.0); axs[0].set_title(f"T43 {ARM} vs V8, paired by ep:seed", fontsize=11, color=figstyle.INK)
fc = a["finger_contacts"]; hb = a["hold"]; hp = a["hold_pairs"]; ft = a["follow_to"]; ob = a["obstacle_stop"]; tm = a["timing"]


def fcc(arm, ph):
    x = fc[arm]["by_phase"][ph]
    return f"{x['n_runs']} runs / {x['n_rows']} rows / pen {x['max_pen_mm']:.1f} mm"


def apple(arm):
    L = [e for e in ct["contact_table"] if e["kind"] == "apple" and e["runs_by_arm"].get(arm)]
    return "; ".join(f"{e['phase']} {e['runs_by_arm'][arm]} runs" for e in L) or "none"


def to(arm):
    t = tm[arm]["ALL"]["to_solve_ms"]
    return "%.1f / %.1f / %.1f" % (t["median"], t["p90"], t["max"])


rows2 = [["finger-hurdle contact, before grasp", fcc("V8", "before_grasp"), fcc(ARM, "before_grasp")], ["finger-hurdle contact, carry", fcc("V8", "carry"), fcc(ARM, "carry")],
         ["finger-hurdle contact, after release (return)", fcc("V8", "after_release"), fcc(ARM, "after_release")],
         ["apple-hurdle contact (runs by phase)", apple("V8"), apple(ARM)],
         ["obstacle stop runs (H)", str(ob["V8"]["runs"]), str(ob[ARM]["runs"])],
         ["HOLD chunks total / runs with HOLD (19 runs)", f"{hb['V8']['total']} / {hb['V8']['runs_with_hold']}", f"{hb[ARM]['total']} / {hb[ARM]['runs_with_hold']}"],
         ["HOLD by binding category", str(hb["V8"]["by_category"]), str(hb[ARM]["by_category"])], ["HOLD by target label", str(hb["V8"]["by_target"]), str(hb[ARM]["by_target"])],
         ["HOLD by phase", str(hb["V8"]["by_phase"]), str(hb[ARM]["by_phase"])],
         ["HOLD only in this arm (same ep:seed, same t)", str(hp["base_only_n"]), f"{hp['new_only_n']}  {hp['new_only_by_category']}"],
         ["follow_to executed / refused", f"{ft['V8']['n_executed']} / {sum(ft['V8']['refused_counts'].values())} {ft['V8']['refused_counts']}", f"{ft[ARM]['n_executed']} / {sum(ft[ARM]['refused_counts'].values())} {ft[ARM]['refused_counts']}"],
         ["TO solve median / p90 / max (ms)", to("V8"), to(ARM)],
         ["time_budget_hit / max_iterations_hit (all chunks)", f"{ex['raw_to']['V8']['time_budget_hit_count_all']} / {ex['raw_to']['V8']['max_iterations_hit_count_all']}", f"{ex['raw_to'][ARM]['time_budget_hit_count_all']} / {ex['raw_to'][ARM]['max_iterations_hit_count_all']}"]]
t2 = axs[1].table(cellText=rows2, colLabels=["", "V8", ARM], loc="center", cellLoc="center", colWidths=[0.36, 0.32, 0.32]); t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 2.2)
fig.tight_layout(); fig.savefig(out, dpi=115, facecolor=figstyle.SURFACE); plt.close(fig)
out.with_suffix(".json").write_text(json.dumps({"figure": str(out), "script": __file__, "sources": [str(V / f"agg_{ARM}.json"), str(V / f"ct_{ARM}.json"), str(V / f"extra_{ARM}.json")],
                                                "rows_pairs": rows, "rows_summary": rows2, "pairs_old_rule": P, "pairs_z4_rule": Z, "pairs_ct_rule": CTP}, indent=1, default=float))
print(out)
