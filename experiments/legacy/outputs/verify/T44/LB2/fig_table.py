"""T44c S1 summary table figure. usage: fig_table.py AGG_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle
A = json.load(open(sys.argv[1]))
ko = figstyle.use_korean()
fig, ax = plt.subplots(figsize=(13, 0.34 * (len(A["episodes"]) + 8) + 1.4)); figstyle.style_axes(fig, [ax]); ax.axis("off")
hdr = ["ep", "fruit", "arm", "bin", "nn xy [mm]", "nn fruit", "E2 demo", "seed1", "seed2", "result", "fail reason"]
def cell(r):
    if r is None: return "-"
    return "ok" if r["ok"] else r["failure_mode"]
rows = []
for e in A["episodes"].values():
    s = e["seeds"]
    rows.append([e["episode"], e["target_fruit"], e["used_arm"], e["bin"].replace("_", " "), f"{1e3*e['nn_xy_m']:.1f}", e["nn_other_fruit"], "yes" if e["in_e2_demo"] else "",
                 cell(s[0]), cell(s[1]), e["status"].upper(), ",".join(e["fail_reasons"])])
t = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK); c.get_text().set_color(figstyle.INK)
    if i == 0: c.set_facecolor(figstyle.SEQ_BLUE[2] if hasattr(figstyle, "SEQ_BLUE") else "#cce"); continue
    st = rows[i - 1][9]
    if j == 9: c.set_facecolor({"PASS": "#cfe8d5", "FAIL": "#f3d0d0", "PENDING": "#eeeeee"}[st])
rf = A["episode_fail_reasons"]; rr = A["run_failure_modes"]
ax.set_title(f"T44c S1 - E0 screening of crowded train episodes: {A['n_pass']} pass / {A['n_fail']} fail / {A['n_pending']} pending of {A['n_candidates']} "
             f"({A['n_runs_done']}/{A['n_runs_planned']} runs)\npass = both seeds (10*ep+1, 10*ep+2) success and robust grasp (LB rule B); "
             f"episode fail reasons: {rf}; run modes: {rr}", fontsize=9, color=figstyle.INK, loc="left")
out = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44/t44c-lb2-table.png'))
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
out.with_suffix(".json").write_text(json.dumps({"source": sys.argv[1], "n_pass": A["n_pass"], "passed": A["passed"], "n_fail": A["n_fail"], "n_pending": A["n_pending"],
    "n_runs_done": A["n_runs_done"], "episode_fail_reasons": rf, "run_failure_modes": rr, "rows": rows, "header": hdr}, indent=1))
print(out)
