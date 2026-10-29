"""T43 XM table figure (verifier). usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_table_xm.py OUT_PNG
Top: per-arm hurdle-crossing intervention (raw/cross_summary.json, group carry_apple + crossing rows).
Middle: V9 per-run class with control-row steps for the third_person_front.mp4 check (frame t = post state of row t, 15 fps).
Bottom: speed-proportional obstacle margin alternatives on V8 / V7g records (raw/speed_table.json), SV alternatives format.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists(), f"{out} exists"
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/XM/raw'))
S = json.load(open(RAW / "cross_summary.json")); SP = json.load(open(RAW / "speed_table.json"))
figstyle.use_korean()
f1 = lambda x: "-" if x is None else f"{x:.1f}"
f0 = lambda x: "-" if x is None else f"{x:.0f}"
side = {"script": __file__, "crossing_rows": [], "v9_rows": [], "alt_rows": []}
# ---- table A
hA = ["arm", "carry+apple window\nchunks (runs)", "palm |D| max\nmed / p90 mm", "executed rows |D|\nmed / p90 mm", "deflected (>5 mm)\ndominant axis",
      "D at k* med mm\ndz / away / du", "stay-start\nchunks", "apple bottom - bar top\nat crossing M med [min,max]", "same, plan C\nmed", "fingers - bar at\ncrossing M med",
      "binding at palm crossing\n(rows:category n) clr med", "ref violates n:\nrefined clr med, [0,2] mm", "|D| vs violation\nslope, r", "accel > a_max\nshare", "no accepted\nSQP candidate"]
rA = []
for arm in ("V5", "V7g", "V8", "V9"):
    A = S["arms"][arm]; G = A["carry_apple"]; cr = A["crossing"]
    dd = G["deflected_dominant_axis"]; comp = G["deflected_components_at_kstar_mm"]
    ab = cr.get("apple_bottom_minus_bar_top_M", {}); abc = cr.get("apple_bottom_minus_bar_top_C", {}); fb = cr.get("fingers_bar_M_apple", {})
    bt = sorted(A["binding_at_palm_crossing"].items(), key=lambda x: -x[1])[:2]
    row = [arm, f"{G['n_chunks']} ({G['n_runs']})", f"{f1(G['palm_D_max_mm']['median'])} / {f1(G['palm_D_max_mm']['p90'])}",
           f"{f1(G['executed_rows']['palm_D_mm']['median'])} / {f1(G['executed_rows']['palm_D_mm']['p90'])}",
           " ".join(f"{k} {v}" for k, v in sorted(dd.items(), key=lambda x: -x[1])),
           f"{f1(comp['dz']['median'])} / {f1(comp['away']['median'])} / {f1(comp['du']['median'])}",
           f"{G['initial_source'].get('stay', 0)} of {G['n_chunks']}",
           f"{f1(ab.get('median'))} [{f1(ab.get('min'))}, {f1(ab.get('max'))}] n{ab.get('n', 0)}", f"{f1(abc.get('median'))}", f"{f1(fb.get('median'))}",
           ", ".join(f"{k.split(':')[0]}:{k.split(':')[1]} {v}" for k, v in bt) + f"; {f1(A['binding_clearance_at_palm_crossing_mm'].get('median'))}",
           f"{G['n_ref_violating']}: {f1(G['refined_binding_clearance_rec_mm_when_ref_violates'].get('median'))}, {G['share_binding_in_0_2mm_when_ref_violates']}",
           f"{G['D_vs_ref_violation']['slope_lsq']:.2f}, {G['D_vs_ref_violation']['pearson_r']:.2f}",
           f"{G['max_step']['accel_gt_1']}", f"{G['sqp']['n_accepted'].get('0', 0)} of {G['n_chunks']}"]
    rA.append(row); side["crossing_rows"].append(dict(zip([h.replace(chr(10), ' ') for h in hA], row)))
# ---- table B
hB = ["V9 run (H)", "class", "closing t", "attach t", "first non-attached chunk t\n(state, attached chunks before)", "apple in hand\nrows", "max lift\nmm @ t",
      "apple leaves hand t: palm s mm,\napple-hurdle mm, chunk start", "palm carry\ncrossing t", "frames to check t (s)"]
rB = []
for key, v in S["v9_classes"].items():
    rv = v["revoke"]; dr = v["drop_row"]; fr = v["video"]["frames"]
    row = [key.split("V9_", 1)[1], v["class"].replace("_", " "), f0(v["first_closing_t"]), f0(v["first_attached_t"]),
           "-" if rv is None else f"{rv['t']} ({rv['state']}, {rv['n_attached_chunks_before']})",
           " ".join(f"{a}-{b}" for a, b in v["inhand_intervals"]) or "-", f"{v['max_apple_lift_mm']:.0f} @ {v['max_apple_lift_t']}",
           "-" if dr is None else f"{dr['t']}: {dr['s_palm_mm']:.0f}, {dr['apple_hurdle_mm_prev_row']:.0f}, {str(dr['chunk_init']).split(' ')[0]}",
           f0(v["palm_carry_crossing_t"]), ", ".join(f"{k} {t} ({t / 15:.1f})" for k, t in fr.items())]
    rB.append(row); side["v9_rows"].append(dict(zip([h.replace(chr(10), ' ') for h in hB], row)))
# ---- table C
ct = SP["contact_tables"]; ub = SP["hold_upper_bounds"]["V8+V7g"]
hC = ["obstacle margin variant (V8 / V7g records, plan rows)", "return finger-hurdle\nseen V8 / V7g", "lead rows\nmedian", "carry apple-hurdle\nseen V8 / V7g", "lead rows\nmedian",
      "new-HOLD UB approach\n(of %d)" % ub["approach"]["cur"]["n_chunks_phase"], "closing\n(of %d)" % ub["closing"]["cur"]["n_chunks_phase"],
      "carry\n(of %d)" % ub["carry"]["cur"]["n_chunks_phase"], "crate entry\n(of %d)" % ub["crate_entry"]["cur"]["n_chunks_phase"], "return\n(of %d)" % ub["return"]["cur"]["n_chunks_phase"]]
lab = {"cur": "now: robot obstacle 10 mm, held 0", "HM": "HM: robot 10, held obstacle 10 (k = 0)", "k0.5": "10 mm + 0.5 x |v_sphere| x 2 rows", "k1.0": "10 mm + 1.0 x |v_sphere| x 2 rows"}
rC = []
for v in ("cur", "HM", "k0.5", "k1.0"):
    s_ = lambda g: f"{ct['V8'][g][v]['seen']}/{ct['V8'][g][v]['n']} , {ct['V7g'][g][v]['seen']}/{ct['V7g'][g][v]['n']}"
    row = [lab[v], s_("return_fingers"), f1(ct["V8+V7g"]["return_fingers"][v]["lead_rows_median"]), s_("carry_apple"), f1(ct["V8+V7g"]["carry_apple"][v]["lead_rows_median"])]
    row += [str(ub[ph][v]["new_hold_upper_bound_chunks"]) for ph in ("approach", "closing", "carry", "crate_entry", "return")]
    rC.append(row); side["alt_rows"].append(dict(zip([h.replace(chr(10), ' ') for h in hC], row)))
fig, axs = plt.subplots(3, 1, figsize=(26, 17), gridspec_kw={"height_ratios": [1.05, 2.0, 0.9]})
for ax, h, rr, title, fs in ((axs[0], hA, rA, "hurdle crossing, carry with apple in hand (window = within 100 mm of the hurdle); D = refined - reference, palm (EE_BODY)", 7.6),
                             (axs[1], hB, rB, "V9 hurdle runs: per-run class and control-row steps (video frame t = post state of row t, 15 fps)", 7.6),
                             (axs[2], hC, rC, "speed-proportional obstacle margin, offline on V8 / V7g plan rows: contacts seen before contact, new-HOLD upper bound per phase (support / target / own-remnant rows excluded)", 8.0)):
    ax.axis("off"); tb = ax.table(cellText=rr, colLabels=h, loc="upper center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.scale(1, 2.0 if ax is not axs[1] else 1.55)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor(figstyle.SEQ_BLUE[1]); c.set_height(c.get_height() * 1.6)
    tb.auto_set_column_width(list(range(len(h))))
    ax.set_title(title, fontsize=10, loc="left")
fig.suptitle("T43 XM: TO intervention at the hurdle crossing (V5 / V7g / V8 / V9) and speed-proportional margin alternatives", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=100)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(out)
