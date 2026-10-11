"""T43 Y figures: t43y-y1-timeline (graph), t43y-y1-table (table), t43y-y2-pairs (graph), t43y-y2-table (table).
Numbers: raw/y_metrics.json, raw/server_chunks.json, raw/esdf_held.json, raw/hold_rows.json, raw/divergence.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

Y = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
M = json.load(open(Y / "raw/y_metrics.json")); R = M["runs"]
S = json.load(open(Y / "raw/server_chunks.json"))["runs"]; E = json.load(open(Y / "raw/esdf_held.json"))["runs"]
H = json.load(open(Y / "raw/hold_rows.json"))["runs"]; D = json.load(open(Y / "raw/divergence.json"))["pairs"]
C = figstyle.CATEGORICAL; figstyle.use_korean()
K5, K4 = "V5/H/V5_ep1925_s19253", "V4/H/V4_ep1925_s19253"


def save(fig, name, data):
    out = FIG / f"{name}.png"; fig.savefig(out, dpi=120, facecolor=figstyle.SURFACE); plt.close(fig)
    json.dump({"figure": str(out), "script": __file__, **data}, open(FIG / f"{name}.json", "w"), indent=1, default=float); print(out)


# ---------------- t43y-y1-timeline ----------------
fig, ax = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
figstyle.style_axes(fig, ax)
side = {}
for key, col, lab in ((K4, C[0], "V4"), (K5, C[1], "V5")):
    s = R[key]["series"]; t = np.array(s["t"])
    ax[0].plot(t, s["apple_in_crate_z_mm"], color=col, lw=1.4, label=f"{lab} apple centre z (crate frame)")
    ax[1].plot(t, s["grip_cmd"], color=col, lw=1.4, label=f"{lab} applied gripper command")
    ax[1].plot(t, s["finger_open"], color=col, lw=1.0, ls="--", label=f"{lab} measured finger opening")
    sc = S[key]["chunks"]
    ax[1].scatter([c["t"] for c in sc], [c["grip_reference_rows0_7"][1] for c in sc], color=col, s=10, marker="^", label=f"{lab} policy reference gripper max (chunk)")
    ax[2].step([c["t"] for c in sc], [c["worst"]["clearance_mm"] if c["worst"]["clearance_mm"] is not None else np.nan for c in sc], where="post", color=col, lw=1.3, label=f"{lab} TO worst-row clearance")
    ax[3].plot(t, s["apple_hand_substeps"], color=col, lw=1.2, label=f"{lab} apple-hand contact sub-steps / row")
    ax[4].plot(t, s["apple_crate_substeps"], color=col, lw=1.2, label=f"{lab} apple-crate contact sub-steps / row")
    for a, b in R[key]["hold_intervals"]:
        for k in range(5):
            ax[k].axvspan(a, b + 1, color=col, alpha=0.08, lw=0)
    side[lab] = {k: R[key].get(k) for k in ("grasp_t", "lift_t", "attach_t", "over_crate_t", "in_crate_t", "place_t", "policy_open_t", "applied_open_t", "measured_open_t",
                                              "last_apple_hand_contact_t", "release_t", "release_mode", "first_apple-crate_contact_t", "first_hand-crate_contact_t", "server_detach_t", "hold_intervals")}
e5 = E[K5]
ax[2].scatter([x["t"] for x in e5], [x["esdf_gap_mm"] for x in e5], color=C[2], s=12, label="V5 held row: recorded ESDF - r (no margin)")
ax[2].scatter([x["t"] for x in e5], [x["true_sd_mm"] for x in e5], color=C[3], s=12, marker="s", label="V5 held row: true crate signed distance - r")
ax[2].axhline(0, color=figstyle.INK_2, lw=0.8)
ax[0].axhline(75, color=figstyle.INK_2, lw=0.8, ls=":"); ax[0].axhline(-52, color=figstyle.INK_2, lw=0.8, ls=":")
ax[0].text(102, 79, "place box top (z 75 mm)", fontsize=7, color=figstyle.INK_2); ax[0].text(102, -48, "crate floor top (z -52 mm)", fontsize=7, color=figstyle.INK_2)
ev = [("attach", R[K5]["attach_t"]), ("over crate (both)", R[K5]["over_crate_t"]), ("V4 release", R[K4]["release_t"]), ("V5 first HOLD", R[K5]["hold_intervals"][0][0]), ("V5 release (slip, cmd closed)", R[K5]["release_t"]), ("V5 applied open", R[K5]["applied_open_t"])]
for i, (n, tt) in enumerate(ev):
    for k in range(5):
        ax[k].axvline(tt, color=figstyle.GRID_INK, lw=0.9)
    ax[0].text(tt + 2, 285 - 22 * (i % 4), f"{n} t {tt}", fontsize=7, color=figstyle.INK)
ax[0].set_ylabel("mm"); ax[1].set_ylabel("1 = open"); ax[2].set_ylabel("mm"); ax[3].set_ylabel("sub-steps (of 33)"); ax[4].set_ylabel("sub-steps (of 33)")
ax[2].set_ylim(-40, 140); ax[4].set_xlabel("control row t (15 Hz)"); ax[4].set_xlim(100, 600)
for a in ax:
    a.legend(fontsize=7, loc="upper right", frameon=False, ncol=2)
fig.suptitle("T43 Y1 — ep1925 s19253, V4 vs V5 (physics replay, qpos mismatch 0 rows; shaded = HOLD rows)", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.98))
save(fig, "t43y-y1-timeline", {"events": side, "esdf_held_V5": e5})

# ---------------- t43y-y1-table ----------------
keys = ["V4/H/V4_ep1925_s19253", "V5/H/V5_ep1925_s19253", "V5/H/V5_ep1807_s18071", "V5/H/V5_ep1807_s18073", "V5/H/V5_ep1834_s18343", "V5/B/V5_ep1807_s18071", "V4/H/V4_ep1834_s18341"]
f = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
rows = [("grasp_t", "grasp_t"), ("attach_t (server)", "attach_t"), ("over_crate_t", "over_crate_t"), ("in_crate_t", "in_crate_t"), ("place_t", "place_t"),
        ("HOLD intervals", "hold_intervals"), ("n HOLD / trailing", None), ("policy open (chunk t)", "policy_open_t"), ("applied open", "applied_open_t"),
        ("measured open >= 0.9", "measured_open_t"), ("release_t (last hand contact + 1)", "release_t"), ("release mode", "release_mode"),
        ("applied cmd at release", "release_with_cmd"), ("first apple-crate contact", "first_apple-crate_contact_t"), ("hand-crate contact rows", "n_hand-crate_contact_rows"),
        ("robot_other-crate contact rows", "n_robot_other-crate_contact_rows"), ("server detach", "server_detach_t"), ("server state at end", None),
        ("HOLD worst rows (kind/obstacle: n)", None), ("HOLD TO clearance min / max mm", None), ("held row: ESDF - r min / max mm", None),
        ("held row: true crate sd - r min mm", None), ("held query radii at attach mm", None)]
cells = []
side = {}
for lab, k in rows:
    line = []
    for key in keys:
        o = R[key]; h = H.get(key, []); rel = o["release_t"]
        if k:
            v = o.get(k); line.append(str(v) if k == "hold_intervals" else f(v))
        elif lab.startswith("n HOLD"):
            line.append(f"{o['n_hold_chunks']} / {o['trailing_hold_chunks']}")
        elif lab.startswith("server state"):
            line.append(S[key]["chunks"][-1]["grasp_state"])
        elif lab.startswith("HOLD worst rows"):
            from collections import Counter
            cn = Counter(f"{x['kind']}/{x['obstacle']}" for x in h); line.append("\n".join(f"{a}: {b}" for a, b in cn.items()) or "-")
        elif lab.startswith("HOLD TO"):
            line.append(f"{min(x['to_clearance_mm'] for x in h):.1f} / {max(x['to_clearance_mm'] for x in h):.1f}" if h else "-")
        elif lab.startswith("held row: ESDF"):
            hh = [x for x in h if x["kind"] == "held"]; line.append(f"{min(x['esdf_gap_mm'] for x in hh):.1f} / {max(x['esdf_gap_mm'] for x in hh):.1f}" if hh else "-")
        elif lab.startswith("held row: true"):
            hh = [x for x in h if x["kind"] == "held"]; line.append(f"{min(x['crate_sd_mm'] for x in hh):.1f}" if hh else "-")
        elif lab.startswith("held query radii"):
            a = next((c for c in S[key]["chunks"] if c.get("attach")), None); line.append(", ".join("%.1f" % x for x in ((a or {}).get("held_query") or {}).get("radii_mm") or []) or "-")
    cells.append(line); side[lab] = dict(zip(keys, line))
fig, ax = plt.subplots(figsize=(19, 8.4)); figstyle.style_axes(fig, ax); ax.axis("off")
tb = ax.table(cellText=cells, rowLabels=[r[0] for r in rows], colLabels=[k.replace("/H/", " H ").replace("/B/", " B ").split(" ", 1)[0] + " " + k.split("_", 1)[1] + (" (B)" if "/B/" in k else "") for k in keys],
              loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.5)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0 or j == -1:
        c.set_text_props(color=figstyle.INK, weight="bold")
ax.set_title("T43 Y1 — time line (control row t) and HOLD binding rows: ep1925 s19253 pair + V5 runs with long attached spans (+ V4 ep1834 s18341 reference)", fontsize=10, color=figstyle.INK)
fig.tight_layout()
save(fig, "t43y-y1-table", {"table": side, "definitions": "see outputs/verify/T43/Y/p/y_metrics.py and hold_rows_y.py docstrings"})

# ---------------- t43y-y2-pairs ----------------
pairs = [(k, k.replace("V4", "V5")) for k in sorted(R) if k.startswith("V4/") and k.replace("V4", "V5") in R and R[k]["success"] and R[k.replace("V4", "V5")]["success"]]
names = [k.split("/", 1)[1].replace("V4_", "").replace("/", " ") for k, _ in pairs]
mets = [("release_apple_floor_gap_mm", "apple-floor gap at release (mm)"), ("release_v_vert_mm_s", "apple v_z at release (mm/s)"),
        ("release_v_horiz_mm_s", "apple |v_xy| at release (mm/s)"), ("post_release_horiz_travel_mm", "apple xy travel, 2 s after release (mm)"),
        ("in_hand_slip_max_mm", "in-hand slip attach -> release (mm)"), ("open_minus_release_rows", "release_t - applied_open_t (rows)"),
        ("tcp_speed_at_open_mm_s", "TCP speed at applied open (mm/s)"), ("pdefl", "place-window TO deflection median (mm)")]


def gv(o, m):
    return (o.get("place_window_deflection_mm") or {}).get("median") if m == "pdefl" else o.get(m)


def sign_p(d):
    d = [x for x in d if x != 0]; n = len(d); k = min(sum(x > 0 for x in d), sum(x < 0 for x in d))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else None


fig, ax = plt.subplots(2, 4, figsize=(18, 9.5)); figstyle.style_axes(fig, ax)
side = {"pairs": names, "metrics": {}}
yy = np.arange(len(pairs))
for a, (m, lab) in zip(ax.ravel(), mets):
    v4 = [gv(R[x], m) for x, _ in pairs]; v5 = [gv(R[y], m) for _, y in pairs]
    for i, (p4, p5) in enumerate(zip(v4, v5)):
        if p4 is not None and p5 is not None:
            a.plot([p4, p5], [i, i], color=figstyle.GRID_INK, lw=1.2, zorder=1)
        for p, (x, yk), col, lb in ((p4, pairs[i], C[0], "V4"), (p5, pairs[i], C[1], "V5")):
            if p is None:
                continue
            o = R[x if lb == "V4" else yk]; mk = "o" if o.get("release_mode") == "open_cmd" else "x"
            a.scatter([p], [i], color=col, marker=mk, s=34, zorder=2)
    a.set_yticks(yy); a.set_yticklabels(names if a in ax[:, 0] else [], fontsize=7); a.set_title(lab, fontsize=9, color=figstyle.INK); a.grid(axis="x", color=figstyle.GRID_INK, lw=0.5)
    both = [(p, q) for p, q in zip(v4, v5) if p is not None and q is not None]; dd = [q - p for p, q in both]
    st = {"n": len(both), "V4_median": float(np.median([p for p, _ in both])) if both else None, "V5_median": float(np.median([q for _, q in both])) if both else None,
          "median_paired_diff_V5_minus_V4": float(np.median(dd)) if dd else None, "n_V5_gt_V4": sum(x > 0 for x in dd), "n_V5_lt_V4": sum(x < 0 for x in dd), "sign_test_p": sign_p(dd),
          "V4": v4, "V5": v5}
    side["metrics"][m] = st
    a.text(0.98, 0.02, f"median V4 {st['V4_median']:.1f} / V5 {st['V5_median']:.1f}\nV5>V4 {st['n_V5_gt_V4']}, V5<V4 {st['n_V5_lt_V4']}, sign p {st['sign_test_p']:.2f}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=7, color=figstyle.INK_2)
from collections import Counter
side["release_mode"] = {"V4": dict(Counter(R[x].get("release_mode") for x, _ in pairs)), "V5": dict(Counter(R[y].get("release_mode") for _, y in pairs))}
fig.suptitle(f"T43 Y2 — place metrics, {len(pairs)} pairs with V4 and V5 both successful (blue V4, orange V5; o = released with open command, x = released with closed command)\n"
             f"release mode V4 {side['release_mode']['V4']} · V5 {side['release_mode']['V5']}", fontsize=10, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "t43y-y2-pairs", side)
Y2STAT = side

# ---------------- t43y-y2-table (attribution) ----------------
hdr = ["pair", "S V4/V5", "first div t", "V4 state @div", "V5 decision @div", "ref diff rad", "refined diff rad", "carve ball V4/V5 mm", "attach held radii V4 mm", "attach held radii V5 mm",
       "V5 latched_hold true", "V5 not_supported chunks <= div", "release mode V4/V5"]
A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]
cells = []; side = {}
for p, r in D.items():
    st, nm = p.split("/"); k4, k5 = f"V4/{st}/V4_{nm}", f"V5/{st}/V5_{nm}"
    if r.get("first_step") is None:
        continue
    def rad(k):
        a = next((c for c in S[k]["chunks"] if c.get("attach")), None); return ", ".join("%.1f" % x for x in ((a or {}).get("held_query") or {}).get("radii_mm") or []) or "-"
    ns = sum(1 for c in A[k5].get("server_chunks", []) if c.get("n_not_supported", 0) > 0 and c["t"] <= r["chunk_t"])
    lh = sum(1 for c in A[k5].get("server_chunks", []) if c.get("latched_hold"))
    b4, b5 = (r.get("V4") or {}).get("carve_ball_radius_mm"), (r.get("V5") or {}).get("carve_ball_radius_mm")
    line = [p, f"{'S' if R[k4]['success'] else '-'}/{'S' if R[k5]['success'] else '-'}", str(r["chunk_t"]), str((r.get("V4") or {}).get("grasp_state")),
            str((r.get("V5") or {}).get("decision_reason")), f"{r['reference_rows0_7_max_abs_diff_rad']:.4f}", f"{r['refined_rows0_7_max_abs_diff_rad']:.4f}",
            f"{f(b4)} / {f(b5)}", rad(k4), rad(k5), str(lh), str(ns), f"{R[k4].get('release_mode') or '-'} / {R[k5].get('release_mode') or '-'}"]
    cells.append(line); side[p] = dict(zip(hdr, line))
fig, ax = plt.subplots(figsize=(20, 7.6)); figstyle.style_axes(fig, ax); ax.axis("off")
tb = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(7.2); tb.scale(1, 1.3)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        c.set_text_props(color=figstyle.INK, weight="bold")
ax.set_title("T43 Y2 — V4 vs V5 first divergence (executed actions > 1e-3 rad) and the server state there; V5 = V4 + --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 (+ snapshot 1 -> 3)",
             fontsize=9.5, color=figstyle.INK)
fig.tight_layout()
save(fig, "t43y-y2-table", {"rows": side, "definitions": "outputs/verify/T43/Y/p/divergence_y.py docstring"})
json.dump(Y2STAT, open(Y / "raw/y2_stats.json", "w"), indent=1, default=float)
