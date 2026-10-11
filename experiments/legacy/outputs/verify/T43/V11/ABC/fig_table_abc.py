"""T43 V11ABC pairs table figure (verifier; copy of AB/fig_table_ab.py, four arms, six pairs). usage: fig_table_abc.py OUT_PNG   (reads V11/ABC/compare_ABC.json)
Table 1: S / G / P (old rule, H / B / ALL), Z4 rule and CT rule (H) - V11AB vs V8, V11AB vs V11A, V11A vs V8 (A total vs B total; +A_only / -B_only / both; McNemar exact p).
Table 2: per arm V8 / V11A / V11AB - contacts, HOLD, follow_to, grasp phase, TO timing and load."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11')); out = pathlib.Path(sys.argv[1])
assert not out.exists() and not out.with_suffix(".json").exists(), out
c = json.load(open(V / "ABC/compare_ABC.json")); ARMS = ("V8", "V11A", "V11AB", "V11ABC"); PN = ("V11ABC_vs_V8", "V11ABC_vs_V11AB", "V11ABC_vs_V11A", "V11AB_vs_V8", "V11A_vs_V8", "V11AB_vs_V11A")
figstyle.use_korean()


def cell(x):
    return f"{x['A_total']} vs {x['B_total']}\n+{x['A_only']} / -{x['B_only']} / both {x['both']}\np {x['mcnemar_exact_p']:.2f}"


rows = []
for lab, stg, f in (("S old, H (14)", "H", "S"), ("G, H", "H", "G"), ("P old, H", "H", "P"), ("S old, B (5)", "B", "S"), ("G, B", "B", "G"), ("S old, H+B (19)", "ALL", "S"), ("G, H+B", "ALL", "G"), ("P old, H+B", "ALL", "P")):
    rows.append([lab] + [cell(c["pairs_old_rule"][p][stg][f]) for p in PN])
for lab, f in (("S Z4 release rule, H", "success_rel"), ("P Z4 (place_rel), H", "place_rel"), ("S CT rule (release + no pre-release contact), H", "success_ct")):
    rows.append([lab] + [cell(c["H_pairs_z4_ct_rule"][p][f]) for p in PN])
fig, axs = plt.subplots(2, 1, figsize=(24, 17), gridspec_kw={"height_ratios": [4.4, 4.2]}); figstyle.style_axes(fig, axs)
for x in axs:
    x.axis("off")
t1 = axs[0].table(cellText=rows, colLabels=["metric (A total vs B total; +A_only / -B_only / both; McNemar exact p)", "V11ABC (A) vs V8 (B)", "V11ABC (A) vs V11AB (B)", "V11ABC (A) vs V11A (B)", "V11AB (A) vs V8 (B)", "V11A (A) vs V8 (B)", "V11AB (A) vs V11A (B)"], loc="center", cellLoc="center", colWidths=[0.22] + [0.13] * 6)
t1.auto_set_font_size(False); t1.set_fontsize(8.5); t1.scale(1, 2.55); axs[0].set_title("T43 V11ABC, paired by ep:seed", fontsize=11, color=figstyle.INK)
fb = c["finger_hurdle_contacts_by_phase"]; ca = c["contacts_arms"]; h = c["hold3"]; ft = c["follow_to3"]; tm = c["timing3"]; gp = c["grasp_phase"]; ld = c["loadavg_per_round"]


def fcc(a, ph):
    x = fb[a][ph]
    return f"{x['n_runs']} runs / {x['n_rows']} rows / pen {x['max_pen_mm']:.1f} mm"


def to(a):
    t = tm[a]["ALL"]["to_solve_ms"]
    return "%.0f / %.0f / %.0f" % (t["median"], t["p90"], t["max"])


def rt(a):
    t = tm[a]["ALL"]["round_trip_ms"]
    return "%.0f / %.0f" % (t["median"], t["p90"])


def appl(a):
    r = [e for e in c["contact_table_by_body_phase"] if e["kind"] == "apple" and e["n_runs"][a]]
    return "; ".join(f"{e['phase']} {e['n_runs'][a]}" for e in r) or "none"


rows2 = [["finger-hurdle contact, before grasp"] + [fcc(a, "before_grasp") for a in ARMS], ["finger-hurdle contact, carry"] + [fcc(a, "carry") for a in ARMS],
         ["finger-hurdle contact, after release"] + [fcc(a, "after_release") for a in ARMS], ["apple-hurdle contact (runs by phase)"] + [appl(a) for a in ARMS],
         ["pre-release / post-release contact runs (H)"] + [f"{ca[a]['pre_release_contact_runs']} / {ca[a]['post_release_contact_runs']}" for a in ARMS],
         ["obstacle stop runs (H)"] + [str(len(c["obstacle_stop"][a])) for a in ARMS],
         ["HOLD chunks / runs with HOLD"] + [f"{h[a]['total']} / {h[a]['runs_with_hold']}" for a in ARMS], ["HOLD by binding category"] + [str(h[a]["by_category"]) for a in ARMS],
         ["HOLD by target / phase"] + [f"{h[a]['by_target']}\n{h[a]['by_phase']}" for a in ARMS],
         ["follow_to executed / refused"] + [f"{ft[a]['n_executed']} / {sum(ft[a]['refused_counts'].values())}" for a in ARMS],
         ["grasp G / closing chunks > 2 mm outward / pooled max (mm)"] + [f"{gp[a]['G']}/19 / {gp[a]['closing_chunks_pooled']['n_outward_gt_2mm']} of {gp[a]['closing_chunks_pooled']['n']} / {gp[a]['closing_chunks_pooled']['outward_mm']['max']:.1f}" for a in ARMS],
         ["first closing: d_refined median (mm) / outward median (mm)"] + [f"{gp[a]['first_closing_d_refined_mm']['median']:.1f} / {gp[a]['first_closing_outward_mm']['median']:.1f}" for a in ARMS],
         ["boundary 2nd difference median / p90 (rad/row^2), H+B"] + [f"{c['sharpness'][a]['ALL']['acc_boundary_rad']['median']:.4f} / {c['sharpness'][a]['ALL']['acc_boundary_rad']['p90']:.4f}" for a in ARMS],
         ["palm reversal share / boundary velocity ratio p90 / n > 1"] + [f"{c['sharpness'][a]['ALL']['palm_reversal_share']:.3f} / {c['sharpness'][a]['ALL']['vel_boundary_ratio']['p90']:.3f} / {c['sharpness'][a]['ALL']['n_vel_boundary_gt1']}" for a in ARMS],
         ["TO solve median / p90 / max (ms)"] + [to(a) for a in ARMS], ["round trip median / p90 (ms)"] + [rt(a) for a in ARMS],
         ["pod load1 median / p90 / max during round"] + [f"{ld[a]['load1_median']:.0f} / {ld[a]['load1_p90']:.0f} / {ld[a]['load1_max']:.0f}" for a in ARMS]]
t2 = axs[1].table(cellText=rows2, colLabels=["", "V8", "V11A", "V11AB", "V11ABC"], loc="center", cellLoc="center", colWidths=[0.28, 0.18, 0.18, 0.18, 0.18]); t2.auto_set_font_size(False); t2.set_fontsize(8); t2.scale(1, 2.2)
fig.tight_layout(); fig.savefig(out, dpi=100, facecolor=figstyle.SURFACE); plt.close(fig)
out.with_suffix(".json").write_text(json.dumps({"figure": str(out), "script": __file__, "sources": [str(V / "ABC/compare_ABC.json")], "rows_pairs": rows, "rows_summary": rows2}, indent=1, default=float))
print(out)
