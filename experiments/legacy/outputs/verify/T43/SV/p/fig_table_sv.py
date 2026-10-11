"""T43 SV table figure (verifier). usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_table_sv.py OUT_PNG
Top: grasp-collapse table, one line per failure pair (V9 vs V8, then V3 vs V4) from raw/grasp_table.json.
Bottom: alternatives (raw/alt_table.json): contacts seen and new-HOLD upper bound per option.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists(), f"{out} exists"
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))
gt = json.load(open(SV / "grasp_table.json"))["pairs"]; alt = json.load(open(SV / "alt_table.json"))
figstyle.use_korean()
hdr = ["pair", "type", "div10 t (A state)", "1st close A/B", "tip-apple A/B mm", "tip dz A/B mm", "window outward>2 A/B",
       "stay starts A/B", "window binding A (rows:class:near)", "F1-exempt path bind", "revoked"]
rows = []; side = {"script": __file__, "grasp_rows": [], "alt_rows": []}
for pair in ("V9V8", "V3V4"):
    for r in gt[pair]["rows"]:
        if r["kind"] != "failure":
            continue
        A, B = r["A"], r["B"]
        top = sorted(A["window_binding_counts"].items(), key=lambda x: -x[1])[:2]
        ta = (A.get("tip_alignment_first_closing") or {}).get("dz_mm"); tb = (B.get("tip_alignment_first_closing") or {}).get("dz_mm")
        ex = A["window_path_binding_exemptions"]
        row = [f"{pair[:2]}-{pair[2:]} {r['stage']} {r['episode']}_s{r['seed']}", r["failure_type"], f"{r['div10_t']} ({r['div10_state_A']})",
               f"{A['first_closing_t']}/{B['first_closing_t']}", f"{A.get('tip_apple_mm_first_closing', 0):.0f}/{B.get('tip_apple_mm_first_closing', 0):.0f}",
               f"{ta}/{tb}", f"{A['window_n_outward_gt2']}/{B['window_n_outward_gt2']} of 6", f"{A['window_stay_starts']}/{B['window_stay_starts']}",
               ", ".join(f"{k} {v}" for k, v in top), f"{ex['reads_target_free_F1_exempt']}/{ex['n']}",
               ";".join(f"t{t}:{'+'.join(x or [])}" for t, x in A["attach_revoked"][:1])]
        rows.append(row); side["grasp_rows"].append(dict(zip(hdr, row)))
ct = alt["contact_tables"]; ub = alt["hold_upper_bounds"]
ahdr = ["alternative", "return finger-hurdle seen V8 / V7g", "lead rows (median)", "return new-HOLD UB (V8+V7g, of 917)",
        "carry apple-hurdle seen V8 / V7g", "carry / crate-entry new-HOLD UB (of 207 / 224)"]
arows = []
def s(tab, grp, k):
    return f"{tab['V8'][grp][k]['seen']}/{tab['V8'][grp][k]['n']} , {tab['V7g'][grp][k]['seen']}/{tab['V7g'][grp][k]['n']}"
arows.append(["now (obstacle 10, held 0)", s(ct, "return_fingers", "b_margin_10"), "-", f"{ub['V8+V7g']['return']['M10']['new_hold_upper_bound_chunks']}",
              s(ct, "carry_apple", "c_margin_0"), f"{ub['V8+V7g']['carry']['current_M10_held0']['new_hold_upper_bound_chunks']} / {ub['V8+V7g']['crate_entry']['current_M10_held0']['new_hold_upper_bound_chunks']}"])
arows.append(["(a) servo after PLACED (S rows < 10)", s(ct, "return_fingers", "a_servo_10"), f"{ct['V8+V7g']['return_fingers']['a_servo_10']['lead_rows_median']}",
              f"{ub['V8+V7g']['return']['a_servo_S_lt10']['new_hold_upper_bound_chunks']} (S rows < 10)", "carry servo S<0: " + s(ct, "carry_apple", "a_servo_0"), "-"])
for M in (20, 30, 40):
    arows.append([f"(b) return margin {M}", s(ct, "return_fingers", f"b_margin_{M}"), f"{ct['V8+V7g']['return_fingers'][f'b_margin_{M}']['lead_rows_median']}",
                  f"{ub['V8+V7g']['return'][f'M{M}']['new_hold_upper_bound_chunks']} (crate {ub['V8+V7g']['return'][f'M{M}']['by_cause_label'].get('robot:crate', 0)})", "-", "-"])
arows.append(["HM held 10 (carry)", "-", "-", "-", s(ct, "carry_apple", "c_margin_10"),
              f"{ub['V8+V7g']['carry']['HM_M10_held10']['new_hold_upper_bound_chunks']} / {ub['V8+V7g']['crate_entry']['HM_M10_held10']['new_hold_upper_bound_chunks']}"])
for M in (20, 30):
    arows.append([f"carry margin {M} (held M, robot max(10,M))", "-", f"{ct['V8+V7g']['carry_apple'][f'c_margin_{M}']['lead_rows_median']} (carry)", "-",
                  s(ct, "carry_apple", f"c_margin_{M}"),
                  f"{ub['V8+V7g']['carry'][f'M{M}']['new_hold_upper_bound_chunks']} / {ub['V8+V7g']['crate_entry'][f'M{M}']['new_hold_upper_bound_chunks']} (incl. unexplained {ub['V8+V7g']['carry'][f'M{M}_incl_unexplained']['new_hold_upper_bound_chunks']} / {ub['V8+V7g']['crate_entry'][f'M{M}_incl_unexplained']['new_hold_upper_bound_chunks']})"])
side["alt_rows"] = [dict(zip(ahdr, r)) for r in arows]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(24, 13), gridspec_kw={"height_ratios": [2.3, 1.0]})
for ax, h, rr, title in ((a1, hdr, rows, "grasp collapse: servo arm A vs no-servo arm B, same episode:seed (window = 6 chunks before A's first closing)"),
                         (a2, ahdr, arows, "offline alternatives on V8 / V7g records: contacts seen before contact and new-HOLD upper bound")):
    ax.axis("off"); tb = ax.table(cellText=rr, colLabels=h, loc="upper center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7.6); tb.scale(1, 1.32)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor(figstyle.SEQ_BLUE[1] if len(figstyle.SEQ_BLUE) > 1 else "#dde"); c.get_text().set_fontweight("bold")
    tb.auto_set_column_width(list(range(len(h))))
    ax.set_title(title, fontsize=10, loc="left")
fig.suptitle("T43 SV - tables (numbers: outputs/verify/T43/SV/raw/grasp_table.json, alt_table.json)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=115)
json.dump(side, open(out.with_suffix(".json"), "w"), indent=1, default=float)
print("wrote", out)
