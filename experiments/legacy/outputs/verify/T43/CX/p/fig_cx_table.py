"""T43 CX figure 3 (table): per-case cause table + HOLD / offline C-toggle / reversal rows (verifier, numbers from raw/cx_summary.json only).
usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_cx_table.py OUT_PNG"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists() and not out.with_suffix(".json").exists(), f"{out} exists"
X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX')); S = json.load(open(X / "raw/cx_summary.json"))
figstyle.use_korean()
f1 = lambda v, n=1: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
hdr = ["case", "AB / ABC stage", "1st divergent\nchunk (seq,t) state", "ABC sep. row\nrows in hand\nHOLD rows", "ABC C rows on chunks\ngrasp-8..sep:\nn_over>0 / max slack rad\nmax bvel ratio",
       "ABC slip mm\nsep-8 / sep-1", "ABC grip cmd\nsep-8..sep-1", "meas. TCP max|a|\nlift AB / ABC m/s^2", "1st closing outward mm\nAB / ABC", "tip-apple at closing mm\nAB / ABC"]
rows = []
for ck, E in S["cases"].items():
    a, b = E["V11AB"], E["V11ABC"]; d = E["divergence"]
    cr = b.get("c_rows_server") or {}
    rows.append([ck, f"{a['stage']} / {b['stage']}", f"{tuple(d['first_refined_seq_t'])} {d['grasp_state_at_first']['V11ABC']}",
                 f"{b['t_sep']} / {b.get('rows_in_hand')} / {b.get('hold_rows_in_hand')}",
                 f"{cr.get('n_chunks_n_over_gt0')}/{cr.get('n_chunks')} / {f1(cr.get('overshoot_max_rad'), 4)}\n{f1(cr.get('boundary_velocity_ratio_max'), 2)}",
                 f"{f1(b.get('slip_mm_tsep_m8'))} / {f1(b.get('slip_mm_tsep_m1'))}", f1(b.get("grip_cmd_mean_tsep_m8"), 3),
                 f"{f1((a.get('lift_window') or {}).get('acc_meas_max'), 2)} / {f1((b.get('lift_window') or {}).get('acc_meas_max'), 2)}",
                 f"{f1(a['first_closing']['outward_mm'], 2)} / {f1(b['first_closing']['outward_mm'], 2)}",
                 f"{f1(a['first_closing']['measured_tip_apple_mm'])} / {f1(b['first_closing']['measured_tip_apple_mm'])}"])
H = S["hold"]["V11ABC"]; HA = S["hold"]["V11AB"]
off = S["offline_c_toggle"]
def offc(k):
    ch = off[k]["chunks"]; v = off[k]["variant"]
    hb = [c for c in ch if c["base"][2] > 0]; flip = [c["seq"] for c in ch if (c["base"][2] > 0) != (c[v][2] > 0)]
    same = [c["seq"] for c in hb if c[v][3] == c["base"][3] and abs(c[v][1] - c["base"][1]) < 0.05]
    return len(hb), flip, len(same)
RS = S["reversal_split"]
notes = [
    f"V11ABC HOLD: {H['n_hold']} (held {H['n_held']}: {', '.join(k.split('_ep')[1] + ' ' + str(v) for k, v in H['held_by_run'].items())}); held binding row = attached:ee_left[0] {H['held_links'].get('attached:ee_left[0]')} / [1] {H['held_links'].get('attached:ee_left[1]')}, refined step 7 in {H['held_step_counts'].get('7')}/17;"
    f" TO clearance {H['held_clearance_mm_range'][0]:.1f} .. {H['held_clearance_mm_range'][1]:.1f} mm; cloud points inside the sphere {H['held_cloud_inside_total']}, nearest true surface = bar for {H['held_cloud_nearest_bar_total']}; SQP returned initial (reference) {H['held_sqp_returned'].get('initial')}/17.",
    f"V11AB HOLD: {HA['n_hold']} (held {HA['n_held']}, 1807 s18071 t160/168, bar, SQP accepted 2/2).",
    "offline C toggle (ta_offline, recorded inputs, open loop): " + "; ".join(f"{k}: HOLD-candidate chunks base {offc(k)[0]}, collision-row status flips {offc(k)[1]}, identical result {offc(k)[2]}" for k in off),
    "reversal share (n rev / n boundaries): " + "; ".join(f"{arm}: " + ", ".join(f"{g} {RS[arm]['agg_n_boundaries_n_reversal'][g][1]}/{RS[arm]['agg_n_boundaries_n_reversal'][g][0]}" for g in ("other_before", "other_after", "cx_case_before", "cx_case_after")) for arm in RS),
    "C rows (problem.py:135-164, :416-426): |X[j+2]-2X[j+1]+X[j]| <= min(max(a_C, |same on reference|), 0.04) + s, X = [u-2, u-1, Q0..Q15], a_C = eps dt^2 / kappa (servo.py:125) = 0.0076-0.0104 rad/row^2; slack weight w_slack*0.01; |Q0 - u-1| <= max_step (weight 10 w_slack).",
]
fig = plt.figure(figsize=(26, 11)); ax = fig.add_axes([0.01, 0.30, 0.98, 0.64]); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 3.2)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[0]); cell.set_height(cell.get_height() * 1.5)
ax2 = fig.add_axes([0.01, 0.01, 0.98, 0.27]); ax2.axis("off")
import textwrap
ax2.text(0, 1, "\n".join(textwrap.fill(n, 230) for n in notes), va="top", fontsize=9, family="monospace", color=figstyle.INK)
fig.patch.set_facecolor(figstyle.SURFACE)
fig.suptitle("T43 CX: per-case numbers, V11AB vs V11ABC at the same ep:seed (seq 2 = first chunk with exec history; 'sep.' = first of 3 rows without apple-hand contact)", fontsize=12)
fig.savefig(out, dpi=90, facecolor=figstyle.SURFACE)
out.with_suffix(".json").write_text(json.dumps({"script": __file__, "header": hdr, "rows": rows, "notes": notes, "source": str(X / "raw/cx_summary.json")}, indent=1)); print(out)
