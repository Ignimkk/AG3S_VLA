"""T28 P3 0f regression-baseline variants table + frame clearance_before graph (numbers from p3/frag_0f.json)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean(); C = fs.CATEGORICAL
D = json.load(open(_ag3s_workspace_path('outputs/verify/T28/p3/frag_0f.json')))
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28/t28-p3-0f-baseline.png')); assert not p.exists()
V = D["variants"]; order = list(V)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1.1, 1]})
a0.axis("off")
rows = [[v, str(V[v]["n_filter_spheres"]), V[v]["violated_start"], V[v]["has_target"], f"{V[v]['frame0_clearance_before_mm']:+.6f}", V[v]["desc"][:95]] for v in order]
rows.insert(0, ["old reference (skill)", "218", "13/15", "15/15", "+0.185978", "T19/T20/T24 HEAD measurement"])
tb = a0.table(cellText=rows, colLabels=["variant", "self-filter spheres", "violated start", "has_target", "frame0 clearance_before mm", "what"], loc="upper center", cellLoc="left",
              colWidths=[0.17, 0.08, 0.07, 0.07, 0.11, 0.5])
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.4)
a0.set_title("T28 P3 0f regression baseline (esdf_rollout, run alone 03:37-03:43): working tree vs HEAD and single-change copies", fontsize=10)
for i, v in enumerate(order):
    a1.plot(range(15), V[v]["clearance_before_mm_per_frame"], marker="o", ms=3, lw=1.2, color=C[i % 8], label=v)
a1.axhline(0, color=fs.INK, lw=0.7); a1.set_xlabel("frame"); a1.set_ylabel("clearance_before (mm)"); a1.set_yscale("symlog", linthresh=5)
a1.legend(fontsize=7, ncol=4); a1.set_title("clearance_before per frame (pre-SQP)", fontsize=9)
fs.style_axes(fig, [a1]); fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps({"source": "outputs/verify/T28/p3/frag_0f.json", "rows": rows,
    "clearance_before_mm_per_frame": {v: V[v]["clearance_before_mm_per_frame"] for v in order}}, indent=1)); print(p)
