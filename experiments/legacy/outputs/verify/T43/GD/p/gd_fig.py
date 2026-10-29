"""T43GD figures (verifier): t43gd-scene (still of a comparison video), t43gd-graph, t43gd-table -> AG3S/ag3s/docs/figures/t43/ (+ .json sidecars; refuses to overwrite).
usage: gd_fig.py AGG_JSON PAIRS_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43')); GDm = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/gate_demo'))
agg = json.load(open(sys.argv[1])); prs = json.load(open(sys.argv[2])); A = agg["arms"]; R = agg["runs"]
KO = figstyle.use_korean(); C = figstyle.CATEGORICAL
for n in ("scene", "graph", "table"):
    for ext in ("png", "json"):
        assert not (FIG / f"t43gd-{n}.{ext}").exists(), f"exists: t43gd-{n}.{ext}"

# ---- scene: still frame of gd_1995_s19953 at control step 300
STEP, PAIR = 300, "H:1995_s19953"; ep, sd = 1995, 19953
subprocess.run(["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-i", str(GDm / f"gd_{ep}_s{sd}.mp4"), "-vf", f"select=eq(n\\,{STEP})", "-frames:v", "1", str(FIG / "t43gd-scene.png")], check=True)
rowat = {}
for arm, key in (("V7", f"V7/H/V7_ep{ep}_s{sd}"), ("V7g", f"V7g/H/V7g_ep{ep}_s{sd}")):
    o = R[key]; i = max(j for j, t in enumerate(o["t_seq"]) if t <= STEP)
    rowat[arm] = {"chunk_t_step": o["t_seq"][i], "label": o["label_seq"][i], "latch": o["latch_seq"][i], "verdict": o["verdict_seq"][i], "follow_to": o["follow_to_seq"][i], "grounding_status": o["grounding_seq"][i],
                  "subtask_no_target": o["snt_seq"][i], "mode": o["mode_seq"][i], "exclusion_active": o["excl_seq"][i], "t_placed": o["t_placed"], "n_hold": o["n_hold"], "hold_t": o["hold_t"]}
(FIG / "t43gd-scene.json").write_text(json.dumps({"figure": "t43gd-scene.png", "source": "outputs/verify/T43/gate_demo/gd_1995_s19953.mp4 (frame index %d = control step %d), left V7 gate OFF, right V7g gate ON; time lines from outputs/verify/T43/GD/agg.json" % (STEP, STEP),
                                                  "pair": PAIR, "step": STEP, "rows_at_step": rowat}, indent=1, default=float))

# ---- graph: (a) ready-posture distance per pair, (b) after-PLACED grounding status chunks, (c) after-PLACED chunk modes + HOLD kinds
keys = sorted(prs["pairs"], key=lambda k: prs["pairs"][k]["dEE_mm_last"][0])
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
figstyle.style_axes(fig, ax)
a0 = ax[0]
for i, k in enumerate(keys):
    o = prs["pairs"][k]; a0.plot([o["dEE_mm_last"][0], o["dEE_mm_last"][1]], [i, i], color="#888888", lw=1.2, zorder=1)
a0.scatter([prs["pairs"][k]["dEE_mm_last"][0] for k in keys], range(len(keys)), color=C[1], s=36, label="V7 gate OFF", zorder=2)
a0.scatter([prs["pairs"][k]["dEE_mm_last"][1] for k in keys], range(len(keys)), color=C[0], s=36, label="V7g gate ON", zorder=2)
a0.set_yticks(range(len(keys))); a0.set_yticklabels([k.replace("_s", " s") + ("*" if "obstacle_stop" in prs["pairs"][k]["mode"] else "") for k in keys], fontsize=7)
a0.set_xlabel("EE distance from the ready pose at the last control step (mm)"); a0.legend(fontsize=8, loc="lower right")
a0.set_title("(a) ready-posture distance per pair (* = a run ended by obstacle stop)", fontsize=9)
b0 = ax[1]; sts = ["ok", "subtask_gated", "no_admissible"]; col = {"ok": "#cccccc", "subtask_gated": C[5], "no_admissible": C[6]}
bot = np.zeros(2)
for s in sts:
    v = np.array([A[a]["ALL"]["grounding_status_after_placed"].get(s, 0) for a in ("V7", "V7g")], float)
    b0.bar(["V7 OFF", "V7g ON"], v, bottom=bot, color=col[s], label=s)
    for i, x in enumerate(v):
        if x:
            b0.text(i, bot[i] + x / 2, int(x), ha="center", va="center", fontsize=8)
    bot += v
b0.set_ylabel("chunks after PLACED (19 pairs)"); b0.legend(fontsize=8); b0.set_title("(b) grounding status after PLACED", fontsize=9)
c0 = ax[2]; labs = ["n_switch", "n_carve", "n_hold"]
vals = {a: [A[a]["ALL"]["after_placed"]["n_switch"], A[a]["ALL"]["after_placed"]["n_carve"], A[a]["ALL"]["n_hold_after_placed"]] for a in ("V7", "V7g")}
x = np.arange(3); c0.bar(x - 0.2, vals["V7"], 0.38, color=C[1], label="V7 gate OFF"); c0.bar(x + 0.2, vals["V7g"], 0.38, color=C[0], label="V7g gate ON")
for i in range(3):
    c0.text(i - 0.2, vals["V7"][i], vals["V7"][i], ha="center", va="bottom", fontsize=8); c0.text(i + 0.2, vals["V7g"][i], vals["V7g"][i], ha="center", va="bottom", fontsize=8)
c0.set_xticks(x); c0.set_xticklabels(["target switch\nchunks", "carve chunks\n(exclusion active)", "HOLD chunks"]); c0.set_ylabel("chunks after PLACED"); c0.legend(fontsize=8)
c0.set_title("(c) after PLACED: switch / carve / HOLD", fontsize=9)
fig.suptitle("T43 GD: subtask gate OFF (V7) vs ON (V7g), same ep:seed pairs (14 hurdle + 5 no obstacle)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "t43gd-graph.png", dpi=130); plt.close(fig)
(FIG / "t43gd-graph.json").write_text(json.dumps({"figure": "t43gd-graph.png", "source": "outputs/verify/T43/GD/agg.json, pairs.json",
    "panel_a_dEE_mm_last_V7_V7g": {k: prs["pairs"][k]["dEE_mm_last"] for k in keys}, "panel_a_modes": {k: prs["pairs"][k]["mode"] for k in keys},
    "panel_b_grounding_after_placed": {a: A[a]["ALL"]["grounding_status_after_placed"] for a in ("V7", "V7g")}, "panel_c": {"labels": labs, **vals}}, indent=1, default=float))

# ---- table
def f1(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"
def st(a, k, f):
    s = A[a]["ALL"][k]; return f"{f1(s[f], 2 if 'rad' in k else 0)}" if s else "-"
rows = []
def add(name, fn):
    rows.append([name, fn("V7"), fn("V7g")])
A_ = lambda a, k: A[a]["ALL"][k]
add("S old / 19", lambda a: A_(a, "success")); add("G old / 19", lambda a: A_(a, "grasp")); add("P old / 19", lambda a: A_(a, "place"))
add("S Z4 release / 19", lambda a: A_(a, "success_rel")); add("S contact-incl. / 19", lambda a: A_(a, "success_ct"))
add("runs ended by obstacle stop", lambda a: A_(a, "mode_counts").get("obstacle_stop", 0))
add("hurdle pre-release contact runs (14 H)", lambda a: A[a]["H"]["n_pre_release_contact"]); add("hurdle post-release contact runs (14 H)", lambda a: A[a]["H"]["n_post_release_contact"])
add("HOLD chunks total (runs)", lambda a: f'{A_(a, "n_hold_total")} ({A_(a, "n_runs_with_hold")})')
add("HOLD kinds", lambda a: ", ".join(f"{k} {v}" for k, v in A_(a, "hold_kinds").items()))
add("HOLD by region (before held / held / after PLACED)", lambda a: " / ".join(str(A_(a, "hold_by_region").get(k, 0)) for k in ("before_first_held", "held_before_placed", "after_placed")))
add("grounding status chunks", lambda a: ", ".join(f"{k} {v}" for k, v in A_(a, "grounding_status_counts").items()))
add("reason subtask_no_target chunks (runs)", lambda a: f'{A_(a, "n_subtask_no_target")} ({A_(a, "n_runs_subtask_no_target")})')
add("subtask_released flag chunks (runs)", lambda a: f'{A_(a, "n_released_flag_chunks")} ({A_(a, "n_runs_released_flag")})')
add("gate blocked first / switch (chunks)", lambda a: f'{A_(a, "n_blocked_first")} / {A_(a, "n_blocked_switch")}')
add("target switch chunks after PLACED", lambda a: A_(a, "after_placed")["n_switch"]); add("carve chunks after PLACED (runs)", lambda a: f'{A_(a, "after_placed")["n_carve"]} ({A_(a, "after_placed")["n_runs_carve"]})')
add("first label home, median step", lambda a: f1(A_(a, "t_first_label_home")["median"], 0)); add("t_placed median step", lambda a: f1(A_(a, "t_placed")["median"], 0))
add("home label after PLACED, median steps", lambda a: f1(A_(a, "home_after_placed_delay")["median"], 0))
add("ready posture dEE last, median mm (19)", lambda a: f1(A_(a, "pose")["dEE_mm_last"]["median"], 1)); add("ready posture dJ last, median rad (19)", lambda a: f1(A_(a, "pose")["dJ_rad_last"]["median"], 3))
add("dEE last, median mm (600-step runs)", lambda a: f"{f1(A_(a, 'pose_full_runs')['dEE_mm_last']['median'], 1)} (n={A_(a, 'pose_full_runs')['dEE_mm_last']['n']})")
add("dEE min from PLACED, median mm", lambda a: f1(A_(a, "pose")["dEE_mm_min_from_placed"]["median"], 1))
fig, ax = plt.subplots(figsize=(11.5, 0.42 * (len(rows) + 2))); ax.axis("off"); figstyle.style_axes(fig, [ax]); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["metric (19 pairs: 14 hurdle H + 5 no obstacle B)", "V7  gate OFF", "V7g  gate ON"], loc="center", cellLoc="left", colWidths=[0.46, 0.27, 0.27])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.35)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#d8d7d2")
    if r == 0:
        cell.set_facecolor("#cde2fb"); cell.set_text_props(fontweight="bold")
ax.set_title("T43 GD: gate OFF (V7) vs gate ON (V7g), paired by ep:seed", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "t43gd-table.png", dpi=130); plt.close(fig)
(FIG / "t43gd-table.json").write_text(json.dumps({"figure": "t43gd-table.png", "source": "outputs/verify/T43/GD/agg.json", "rows": {r[0]: {"V7": r[1], "V7g": r[2]} for r in rows}}, indent=1, default=str))
print("ok", [str(p) for p in sorted(FIG.glob("t43gd-*"))])
