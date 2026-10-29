"""T43 TK table figures.  usage: fig_table_tk.py OUT_TABLE_PNG OUT_CONTACTS_PNG   (numbers from raw/tk_summary.json)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle

o1, o2 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
assert not o1.exists() and not o2.exists()
S = json.load(open(_ag3s_workspace_path('outputs/verify/T43/TK/raw/tk_summary.json')))
figstyle.use_korean()
A = S["arms"]


def mp(s, f="{:.1f}"):
    if not s or not s.get("n"):
        return "-"
    return (f + " / " + f).format(s["median"], s["p90"])


cols = [(a, ph) for a in ("V7g", "V8", "E0") for ph in ("carry", "other")]
rows = []


def add(label, fn, f="{:.1f}"):
    vals = []
    for a, ph in cols:
        try:
            vals.append(fn(A[a], ph, f))
        except Exception:
            vals.append("-")
    rows.append([label] + vals)


add("P-C EE mm, exec rows", lambda X, ph, f: mp(X["exec"][ph]["PC_mm"], f))
add("P-C EE mm, HOLD rows (all phases)", lambda X, ph, f: mp(X["hold"]["all"]["PC_mm"], f) if ph == "carry" else "")
add("C-M EE mm, exec rows", lambda X, ph, f: mp(X["exec"][ph]["CM_mm"], f))
add("C-M max-joint rad", lambda X, ph, f: mp(X["exec"][ph]["CM_rad_maxjoint"], "{:.3f}"))
add("P-M EE mm (= TO plan vs actual)", lambda X, ph, f: mp(X["exec"][ph]["PM_mm"], f))
add("time offset tau EE (rows), moving", lambda X, ph, f: mp(X["exec"][ph]["tau_ee_moving_steps"], "{:.2f}"))
add("time offset tau joint (rows), moving", lambda X, ph, f: mp(X["exec"][ph]["tau_j_moving_steps"], "{:.2f}"))
add("cross-track |M - C path| mm, moving", lambda X, ph, f: mp(X["exec"][ph]["d_at_tau_ee_moving_mm"], f))
add("commanded EE speed mm/row", lambda X, ph, f: mp(X["exec"][ph]["speed_cmd_mm"], f))
add("C-M EE mm at speed < 1 mm/row (all phases)", lambda X, ph, f: mp(X["speed_bins_exec"][0]["CM_mm"], f) if ph == "carry" else "")
add("C-M / speed, 10-20 mm/row (all phases)", lambda X, ph, f: mp(X["speed_bins_exec"][3]["CM_over_speed"], "{:.2f}") if ph == "carry" else "")
add("q_now - C(t0-1) EE mm (prev row exec)", lambda X, ph, f: mp(X["qnow"]["qnow_minus_Cprev_mm_prev_exec"][ph], f))
add("q_now - M_pre(t0) max rad (all chunks)", lambda X, ph, f: "{:.1g}".format(X["qnow"]["cap_minus_pre0_rad_max"]) if ph == "carry" else "")
add("P(t0) - q_now max-joint rad (all chunks)", lambda X, ph, f: mp(X["qnow"]["P0_minus_qnow_rad_max"], "{:.3f}") if ph == "carry" else "")
add("refined - reference EE mm", lambda X, ph, f: mp(X["exec"][ph]["RP_mm_refined_vs_reference"], f))
add("cmd direction reversal k=0 | k=2..7 (carry)", lambda X, ph, f: ("{:.2f} | {:.2f}".format(X["by_k_exec"]["carry"][0]["frac_cos_lt_0"],
    float(np.mean([r["frac_cos_lt_0"] for r in X["by_k_exec"]["carry"][2:]])))) if ph == "carry" else "")
add("cmd accel mm/row^2 k=0 | k=2..7 med (carry)", lambda X, ph, f: ("{:.1f} | {:.1f}".format(X["by_k_exec"]["carry"][0]["cmd_accel_mm_per_row2"]["median"],
    float(np.median([r["cmd_accel_mm_per_row2"]["median"] for r in X["by_k_exec"]["carry"][2:]])))) if ph == "carry" else "")
add("n exec rows", lambda X, ph, f: str(X["exec"][ph]["n_rows"]))
hdr = ["median / p90"] + [f"{a} {ph}" for a, ph in cols]
J = []
for a in ("V7g", "V8", "E0"):
    for ph in ("carry",):
        X = A[a]["exec"][ph]["CM_rad_joint"]
        J.append([f"{a} {ph} |C-M| rad"] + [f"{m:.3f} / {p:.3f}" for m, p in zip(X["median"], X["p90"])])
mdl = A["V8"]["model"]["left_arm"]
J.append(["model kp | (kv+b)/kp rows"] + [f"{r['kp']:.0f} | {r['ramp_lag_steps']:.2f}" for r in mdl])

fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(2, 1, height_ratios=[len(rows) + 1, len(J) + 1.5])
ax = fig.add_subplot(gs[0]); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center", colLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.45)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[1] if hasattr(figstyle, "SEQ_BLUE") else "#dde"); cell.set_text_props(color=figstyle.INK)
    if c == 0:
        cell.set_width(0.26); cell.set_text_props(ha="left")
    else:
        cell.set_width(0.12)
ax.set_title("T43 TK decomposition: P = TO refined row k, C = applied command, M = measured post-row state (active arm EE_BODY / 7 joints)", fontsize=10)
ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
tb2 = ax2.table(cellText=J, colLabels=["per joint (carry, exec)"] + [f"arm_{j}" for j in range(7)], loc="upper center", cellLoc="center")
tb2.auto_set_font_size(False); tb2.set_fontsize(8.5); tb2.scale(1, 1.45)
for (r, c), cell in tb2.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if c == 0:
        cell.set_width(0.2)
    else:
        cell.set_width(0.11)
m0 = A["V8"]["model"]
ax2.set_title(f"MuJoCo position servo: dt {m0['timestep']} s x {m0['substeps_per_row']} substeps = {m0['row_period_s']:.3f} s per row; "
              f"zeta {min(r['zeta'] for r in mdl):.2f}-{max(r['zeta'] for r in mdl):.2f}; wn {min(r['wn_rad_s'] for r in mdl):.1f}-{max(r['wn_rad_s'] for r in mdl):.1f} rad/s", fontsize=9)
fig.tight_layout(); fig.savefig(o1, dpi=130)
json.dump({"source": "outputs/verify/T43/TK/raw/tk_summary.json", "header": hdr, "rows": rows, "joint_rows": J}, open(o1.with_suffix(".json"), "w"), indent=1)

# contacts
C = S["contacts"]
f = lambda v, n=1: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v))
hdr2 = ["arm", "run", "kind:body", "phase", "t", "k", "row", "pen", "bar P", "bar C", "bar M", "bar M_pre", "bar C(t-2)", "min bar P rows",
        "|C-M|", "tau", "x-track", "speed", "TO clr", "ESDF link@P", "ESDF-geom"]
R2 = [[c["arm"], c["key"].split("_", 1)[1], f"{c['kind']}:{c['body']}", c["phase_ct"].replace("_release", "").replace("_grasp", ""), c["t"], c["k"],
       c["row_cls"], f(c["ct_max_pen_mm"]), f(c["bar_P_mm"]), f(c["bar_C_mm"]), f(c["bar_M_mm"]), f(c["bar_M_pre_mm"]), f(c["bar_Cm2_mm"]),
       f(c["bar_Prows_min_mm"]), f(c["CM_mm"]), f(c["tau_ee"], 2), f(c["cross_track_mm"]), f(c["speed_cmd_mm"]), f(c["to_clearance_mm"]),
       f(c["clear_row_k_link_mm"]), f(c["esdf_link_minus_geom_P_mm"])] for c in C]
fig = plt.figure(figsize=(22, 0.3 * len(R2) + 1.6)); ax = fig.add_subplot(111); ax.axis("off")
tb = ax.table(cellText=R2, colLabels=hdr2, loc="upper center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    cell.set_width({1: 0.075, 2: 0.115}.get(c, 0.04))
    if r > 0 and c in (8, 9, 10) :
        try:
            v = float(cell.get_text().get_text())
            if v <= 0:
                cell.set_facecolor(figstyle.SEQ_BLUE[2] if hasattr(figstyle, "SEQ_BLUE") else "#ccd")
        except ValueError:
            pass
ax.set_title("T43 TK per contact (CT first contact row t): signed distance (mm) of the contacting body to the hurdle for P, C, M; "
             "TO clr = to.metrics.clearance_m of the chunk; ESDF link@P = record clearance of the link spheres at row k (refined); shaded <= 0", fontsize=9)
fig.tight_layout(); fig.savefig(o2, dpi=130)
json.dump({"source": "outputs/verify/T43/TK/raw/tk_summary.json", "header": hdr2, "rows": R2}, open(o2.with_suffix(".json"), "w"), indent=1)
print("wrote", o1, o2)
