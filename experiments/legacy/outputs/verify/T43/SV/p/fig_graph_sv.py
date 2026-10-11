"""T43 SV graph figure (verifier, CPU only). usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_graph_sv.py OUT_PNG
Inputs: raw/sag_table.json, raw/grasp_table.json, raw/alt_table.json (definitions in sv_sag_agg.py, sv_grasp_agg.py, sv_alt_agg.py).
(a) finger height over the table top in the last 3 chunks up to the first closing chunk, failure keys: median and p10-p90 of P / S / M per arm.
(b) server tip_alignment.now dz (fingertip minus AG3S centroid height, mm) on the first closing chunk, per failure pair (no-servo -> servo).
(c) return phase: finger-hurdle contacts seen (V8 9 + V7g 5) per alternative.
(d) return phase: new-HOLD upper bound (post-PLACED chunks, V8 + V7g, 917) per alternative, stacked by truth label of the binding cell.
(e) carry: apple-hurdle contacts seen (V8 5 + V7g 3) and new-HOLD upper bound chunks in carry (207) and crate entry (224).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists(), f"{out} exists"
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))
sag = json.load(open(SV / "sag_table.json"))["arms"]; gt = json.load(open(SV / "grasp_table.json"))["pairs"]; alt = json.load(open(SV / "alt_table.json"))
figstyle.use_korean(); CAT = figstyle.CATEGORICAL
fig, axes = plt.subplots(1, 5, figsize=(24, 5.4), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0, 1.25, 1.25]})
side = {"script": __file__, "inputs": [str(SV / f) for f in ("sag_table.json", "grasp_table.json", "alt_table.json")]}
# (a)
ax = axes[0]; arms = ["V8", "V9", "V4", "V3"]; xs = np.arange(len(arms)); w = 0.25; sa = {}
for i, st in enumerate(("P", "S", "M")):
    med = [sag[a]["failure"][st]["median"] for a in arms]; lo = [sag[a]["failure"][st]["p10"] for a in arms]; hi = [sag[a]["failure"][st]["p90"] for a in arms]
    ax.bar(xs + (i - 1) * w, med, w, color=CAT[[1, 2, 0][i]], label={"P": "P plan", "S": "S servo prediction", "M": "M measured"}[st])
    ax.errorbar(xs + (i - 1) * w, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor=figstyle.INK_2, lw=0.8, capsize=2)
    sa[st] = {"median": med, "p10": lo, "p90": hi}
ax.set_xticks(xs); ax.set_xticklabels(["V8\nno servo", "V9\nservo", "V4\nno servo", "V3\nservo+sweep"])
ax.set_ylabel("finger sphere height over table top (mm)"); ax.set_ylim(0, 40)
ax.set_title("(a) last 3 chunks before closing, failure keys\nmedian, bars p10-p90 (V9V8 13 pairs, V3V4 10)", fontsize=9); ax.legend(fontsize=7)
side["a_finger_table_mm_failures"] = {"arms": arms, **sa}
# (b)
ax = axes[1]; sb = {}
for j, (pair, A, B, col) in enumerate((("V9V8", "V9", "V8", CAT[0]), ("V3V4", "V3", "V4", CAT[3]))):
    vals = []
    for r in gt[pair]["rows"]:
        if r["kind"] != "failure":
            continue
        zb = (r["B"].get("tip_alignment_first_closing") or {}).get("dz_mm"); za = (r["A"].get("tip_alignment_first_closing") or {}).get("dz_mm")
        if za is None or zb is None:
            continue
        vals.append((zb, za)); ax.plot([2 * j, 2 * j + 1], [zb, za], "-o", color=col, alpha=0.55, ms=3)
    v = np.asarray(vals); ax.plot([2 * j, 2 * j + 1], np.median(v, 0), "-s", color=col, lw=2.5, ms=7, label=f"{B} -> {A} median")
    sb[pair] = {"pairs_dz_B_A": vals, "median_B": float(np.median(v[:, 0])), "median_A": float(np.median(v[:, 1]))}
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["V8", "V9", "V4", "V3"]); ax.set_ylabel("tip dz vs AG3S centroid at first closing (mm)")
ax.set_title("(b) fingertip height at the first closing chunk\nper failure pair (server tip_alignment.now)", fontsize=9); ax.legend(fontsize=7)
side["b_tip_dz_first_closing"] = sb
# (c)
ax = axes[2]; ct = alt["contact_tables"]
opts = [("a_servo_10", "servo\nafter PLACED"), ("b_margin_10", "margin\n10 (now)"), ("b_margin_20", "margin\n20"), ("b_margin_30", "margin\n30"), ("b_margin_40", "margin\n40")]
v8 = [ct["V8"]["return_fingers"][k]["seen"] for k, _ in opts]; v7 = [ct["V7g"]["return_fingers"][k]["seen"] for k, _ in opts]
lead = [ct["V8+V7g"]["return_fingers"][k]["lead_rows_median"] for k, _ in opts]
xs = np.arange(len(opts)); ax.bar(xs, v8, 0.6, color=CAT[0], label="V8 (of 9)"); ax.bar(xs, v7, 0.6, bottom=v8, color=CAT[5], label="V7g (of 5)")
for x, a_, b_, l_ in zip(xs, v8, v7, lead):
    ax.text(x, a_ + b_ + 0.3, f"{a_ + b_}/14" + (f"\nlead {l_:.0f} rows" if l_ is not None else ""), ha="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in opts], fontsize=7.5); ax.set_ylim(0, 18); ax.set_ylabel("contacts seen by the TO before contact")
ax.set_title("(c) return: finger-hurdle contacts seen\n(contact chunk flagged; lead = median rows)", fontsize=9); ax.legend(fontsize=7, loc="upper left")
side["c_return_seen"] = {"options": [k for k, _ in opts], "V8": v8, "V7g": v7, "lead_rows_median": lead}
# (d)
ax = axes[3]; ub = alt["hold_upper_bounds"]["V8+V7g"]["return"]
keys = [("a_servo_S_lt10", "servo S<10\n(path rows)"), ("M20", "margin 20"), ("M30", "margin 30"), ("M40", "margin 40")]
labs = ["crate", "hurdle", "unexplained", "fruit_other"]; bottom = np.zeros(len(keys)); sd = {}
for i, lb in enumerate(labs):
    vals = np.array([ub[k]["by_cause_label"].get(f"robot:{lb}", 0) for k, _ in keys], float)
    ax.bar(np.arange(len(keys)), vals, 0.6, bottom=bottom, color=CAT[i], label=lb); bottom += vals; sd[lb] = vals.tolist()
for x, k in enumerate(keys):
    ax.text(x, bottom[x] + 8, f"{ub[k[0]]['new_hold_upper_bound_chunks']}/{ub[k[0]]['n_chunks_phase']}", ha="center", fontsize=7.5)
ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=7.5); ax.set_ylabel("post-PLACED chunks (V8 + V7g)")
ax.set_title("(d) return: new-HOLD upper bound (plan rows below the new margin)\nservo: chunks with S rows below 10 mm; by truth label", fontsize=9); ax.legend(fontsize=7)
side["d_return_hold_ub"] = {"options": [k for k, _ in keys], "by_label": sd, "totals": [ub[k]["new_hold_upper_bound_chunks"] for k, _ in keys], "n_chunks": ub["M20"]["n_chunks_phase"]}
# (e)
ax = axes[4]; u = alt["hold_upper_bounds"]["V8+V7g"]; ca = ct["V8+V7g"]["carry_apple"]
ck = [("current_M10_held0", "c_margin_0", "now\n(held 0)"), ("HM_M10_held10", "c_margin_10", "HM\nheld 10"), ("M20", "c_margin_20", "carry\n20"), ("M30", "c_margin_30", "carry\n30")]
xs = np.arange(len(ck)); c1 = [u["carry"][k]["new_hold_upper_bound_chunks"] for k, _, _ in ck]; c2 = [u["crate_entry"][k]["new_hold_upper_bound_chunks"] for k, _, _ in ck]
ax.bar(xs - 0.18, c1, 0.36, color=CAT[0], label=f"carry (of {u['carry']['M20']['n_chunks_phase']})")
ax.bar(xs + 0.18, c2, 0.36, color=CAT[1], label=f"crate entry (of {u['crate_entry']['M20']['n_chunks_phase']})")
seen = [ca[c]["seen"] for _, c, _ in ck]
for x, s_, a_, b_ in zip(xs, seen, c1, c2):
    ax.text(x, max(a_, b_) + 5, f"seen {s_}/8", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xticklabels([l for _, _, l in ck], fontsize=7.5); ax.set_ylabel("new-HOLD upper bound chunks (V8 + V7g)")
ax.set_title("(e) carry margin (held rows M, robot rows max(10, M)):\nnew-HOLD upper bound and apple-hurdle contacts seen", fontsize=9); ax.legend(fontsize=7)
side["e_carry"] = {"options": [k for k, _, _ in ck], "carry": c1, "crate_entry": c2, "seen_of_8": seen}
figstyle.style_axes(fig, list(axes))
fig.suptitle("T43 SV - servo model grasp collapse (a, b) and offline alternatives on V8 / V7g records (c-e)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=120)
json.dump(side, open(out.with_suffix(".json"), "w"), indent=1, default=float)
print("wrote", out)
