"""T43W final figures (verifier): V5 rerun (snapshot 3) + blind round (snapshot ${AG3S_PROJECT_ROOT}-t43b).
inputs : final/numbers_main.json, final/numbers_blind.json, final/agg.json (final_agg.py), analysis/analysis.json, analysis_blind/analysis.json
outputs: AG3S/ag3s/docs/figures/t43/{t43w-v5-arms, t43w-v5b-v5s-pairs, t43w-v5-fail-scene}.{png,json}
arms       : hurdle 14 success / grasp / place per arm E0 V1 V3 V4 V5 V5S V5B (left, bars of count of 14) and the no-obstacle 5 (V3 V4 V5, right).
pairs      : V5B vs V5S, 14 (episode, seed) pairs: outcome S / G / - and failure stage per arm (definitions in final_agg.py), 2x2 + exact McNemar p.
fail-scene : V5 failed hurdle runs (ep1967 s19672, s19673), third-person front frame (recorded third_person_front.mp4, avoidance arrow overlay by the client)
             at the control row of the minimum hurdle distance over fingers, palm, other robot geoms and the apple (client pair_distance, GJK)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
AG = json.loads((V / "final/agg.json").read_text())
NM = json.loads((V / "final/numbers_main.json").read_text()); NB = json.loads((V / "final/numbers_blind.json").read_text())
AM = json.loads((V / "analysis/analysis.json").read_text())["runs"]
KO = figstyle.use_korean()
ARMS = ["E0", "V1", "V3", "V4", "V5", "V5S", "V5B"]
COL = dict(zip(ARMS, [figstyle.CATEGORICAL[i] for i in range(7)]))


def save(fig, name, data):
    p = OUT / f"{name}.png"
    assert not p.exists() or "--overwrite" in sys.argv, f"{p} exists"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    data["definitions"] = __doc__
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


# ---- arms bar graph
fig, (ax, axb) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [7, 3]}); figstyle.style_axes(fig, [ax, axb])
side = {"arms": AG["arms"]}
w = 0.26
for i, a in enumerate(ARMS):
    h = AG["arms"][a]["H"]
    for j, (m, hatch) in enumerate((("success", ""), ("grasp", "//"), ("place", ".."))):
        b = ax.bar(i + (j - 1) * w, h[m], w, color=COL[a], hatch=hatch, edgecolor=figstyle.SURFACE, label=m if i == 0 else None)
        ax.text(i + (j - 1) * w, h[m] + 0.2, str(h[m]), ha="center", fontsize=8, color=figstyle.INK)
ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS); ax.set_ylim(0, 15.5); ax.set_ylabel("runs of 14 (hurdle, P2 overbar)")
ax.set_title("hurdle 14: success (solid) / grasp (//) / place (..) per arm", fontsize=9, color=figstyle.INK)
ax.axhline(14, color=figstyle.INK_2, lw=0.6, ls="--")
for i, a in enumerate(("V3", "V4", "V5")):
    h = AG["arms"][a]["B_no_obstacle"]
    for j, (m, hatch) in enumerate((("success", ""), ("grasp", "//"), ("place", ".."))):
        axb.bar(i + (j - 1) * w, h[m], w, color=COL[a], hatch=hatch, edgecolor=figstyle.SURFACE)
        axb.text(i + (j - 1) * w, h[m] + 0.1, str(h[m]), ha="center", fontsize=8, color=figstyle.INK)
axb.set_xticks(range(3)); axb.set_xticklabels(("V3", "V4", "V5")); axb.set_ylim(0, 5.5); axb.set_ylabel("runs of 5")
axb.set_title("no obstacle 5", fontsize=9, color=figstyle.INK)
fig.suptitle("T43 W: V5 rerun (snapshot 3) and blind round (V5S visible / V5B policy-blind), same 7 episodes x 2 seeds", color=figstyle.INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "t43w-v5-arms", side)

# ---- V5B vs V5S pair table
def key(v): return (v["episode"], v["seed"])
R = {a: {key(v): v for v in (NM if a == "V5" else NB)["per_run"].values() if v["arm"] == a and v["stage"] == "H"} for a in ("V5B", "V5S")}
fs = {a: {k: s for s, ks in AG["fail_stage"][a]["runs"].items() for k in ks} for a in ("V5B", "V5S")}
def cell(a, k):
    v = R[a][k]; o = "S" if v["success"] else ("G" if v["grasp"] else "-")
    st = fs[a].get(f'{k[0]}_s{k[1]}', "")
    return f"{o}" + (f" ({st})" if st else "") + f"  H{v['n_hold_chunks']}" + (f"[{v['trailing_hold_chunks']}]" if v["trailing_hold_chunks"] else "")
keys = sorted(R["V5B"])
rows = [[f"ep{e} s{s}", cell("V5B", (e, s)), cell("V5S", (e, s))] for e, s in keys]
fig = plt.figure(figsize=(12, 9)); fig.patch.set_facecolor(figstyle.SURFACE)
ax = fig.add_axes([0.01, 0.30, 0.98, 0.64]); ax.set_axis_off()
tb = ax.table(cellText=rows, colLabels=["run", "V5B (policy-blind)", "V5S (visible)"], loc="upper center", cellLoc="center", colWidths=[0.2, 0.4, 0.4])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.7)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0 and j > 0: c.set_text_props(color=COL["V5B" if j == 1 else "V5S"])
ax.set_title("S success / G grasp only / - ; (failure stage); H = HOLD chunks [trailing to end]", fontsize=9, color=figstyle.INK)
P = AG["pairs"]["V5B_vs_V5S"]
pc = [[m, str(P[m]["both"]), str(P[m]["V5B_only"]), str(P[m]["V5S_only"]), str(P[m]["neither"]), f"{P[m]['mcnemar_exact_p']:.3f}"] for m in ("success", "grasp", "place")]
ax2 = fig.add_axes([0.01, 0.12, 0.55, 0.14]); ax2.set_axis_off()
t2 = ax2.table(cellText=pc, colLabels=["V5B vs V5S", "both", "V5B only", "V5S only", "neither", "McNemar exact p"], loc="upper center", cellLoc="center")
t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 1.5)
fc = [[a] + [str(AG["fail_stage"][a]["counts"][s]) for s in ("before_pick", "grasp_closing_only", "grasp_attached_not_lifted", "carry", "place")] for a in ("V5B", "V5S")]
ax3 = fig.add_axes([0.01, 0.01, 0.8, 0.10]); ax3.set_axis_off()
t3 = ax3.table(cellText=fc, colLabels=["failure stage (runs)", "before pick", "grasp: closing only", "grasp: attached, not lifted", "carry", "place"], loc="upper center", cellLoc="center")
t3.auto_set_font_size(False); t3.set_fontsize(8.5); t3.scale(1, 1.5)
side = {"pairs": P, "rows": rows, "fail_stage": {a: AG["fail_stage"][a] for a in ("V5B", "V5S")}}
save(fig, "t43w-v5b-v5s-pairs", side)

# ---- failure scene (V5 failed hurdle runs)
FAILS = [("V5", 1967, 19672), ("V5", 1967, 19673)]
fig, axes = plt.subplots(1, len(FAILS), figsize=(6.2 * len(FAILS), 4.8), squeeze=False)
side = {"runs": {}}
for ax, (arm, ep, seed) in zip(axes[0], FAILS):
    r = AM[f"{arm}/H/{arm}_ep{ep}_s{seed}"]; s = r["hurdle"]["series"]
    t = np.asarray(s["t"]); rob = np.minimum.reduce([np.asarray(s[k]) for k in ("Af", "Ae", "C")]); ap = np.asarray(s["B"]); allm = np.minimum(rob, ap); i = int(np.argmin(allm))
    vid = pathlib.Path(r["run_dir"]) / "third_person_front.mp4"
    rd = imageio.get_reader(str(vid)); n = rd.count_frames(); kf = min(int(t[i]), n - 1); img = rd.get_data(kf); rd.close()
    Hh, Ww = img.shape[:2]; crop = (int(0.02 * Hh), int(0.80 * Hh), int(0.30 * Ww), int(0.98 * Ww)); ax.imshow(img[crop[0]:crop[1], crop[2]:crop[3]]); ax.set_axis_off()
    mm = NM["per_run"][f"{arm}/H/{arm}_ep{ep}_s{seed}"]
    ax.set_title(f"{arm} ep{ep} s{seed}  t={int(t[i])} of {int(r['n_control'])}  closest {1e3*allm[i]:.1f} mm ({'apple' if ap[i] < rob[i] else 'robot'})\n"
                 f"fing {mm['hurdle_min_mm']['Af']:.1f} palm {mm['hurdle_min_mm']['Ae']:.1f} apple {mm['hurdle_min_mm']['B']:.1f} arm {mm['hurdle_min_mm']['C']:.1f} mm  "
                 f"closing chunks {mm['closing']['n_chunks']}, attached {mm['attached_chunks']}", color=COL[arm], fontsize=8)
    side["runs"][f"{arm}_ep{ep}_s{seed}"] = {"run_dir": r["run_dir"], "frame_index": kf, "n_frames": n, "t_closest": int(t[i]), "closest_mm": 1e3 * float(allm[i]),
                                              "closest_is_apple": bool(ap[i] < rob[i]), "hurdle_min_mm": mm["hurdle_min_mm"], "crop_rows_cols": crop,
                                              "obstacle_stop_lines": mm["obstacle_stop_lines"], "grasp_state_counts": mm["grasp_state_counts"]}
fig.suptitle("T43 W V5 rerun, failed hurdle runs (overbar P2), frame at closest hurdle approach; arrow = refined - reference TCP x8 (red = HOLD)", color=figstyle.INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "t43w-v5-fail-scene", side)
