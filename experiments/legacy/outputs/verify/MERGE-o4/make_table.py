from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, shutil
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle
O = _ag3s_workspace_path('outputs/verify/MERGE-o4')
F = _ag3s_workspace_path('AG3S/ag3s/docs/figures/merge-o4/merge-o4-baseline-table')
s = json.load(open(f"{O}/summary.json"))
ko = figstyle.use_korean()
keys = [("violated_start", "위반으로 시작" if ko else "violated at start", lambda v: f"{v}/15"),
        ("has_target", "has_target", lambda v: f"{v}/15"),
        ("frame0_clearance_before_mm", "frame0 clearance_before (mm)", lambda v: repr(v)),
        ("clearance_before_15_sha1_12", "clearance_before x15 sha1[:12]", lambda v: v)]
rows = []
for b in ["legacy", "curobo"]:
    for k, lab, fmt in keys:
        e = s[b]["expected"][k]; g = s[b][k]
        rows.append([b if b == "legacy" else "cuRobo", lab, fmt(e), fmt(g), "같음" if (ko and e == g) else ("equal" if e == g else ("다름" if ko else "differs"))])
fig, ax = plt.subplots(figsize=(11, 3.8))
figstyle.style_axes(fig, [ax]) if True else None
ax.axis("off")
hdr = ["baseline", "항목" if ko else "item", "기대값 (T38 V3, work-ef)" if ko else "expected (T38 V3)",
       "측정값 (merge working tree)" if ko else "measured (merged tree)", "비트 비교" if ko else "bitwise"]
t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="left", colLoc="left",
             colWidths=[0.1, 0.27, 0.25, 0.25, 0.1])
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.45)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK); cell.set_facecolor(figstyle.SURFACE)
    cell.get_text().set_color(figstyle.INK)
    if r == 0: cell.get_text().set_fontweight("bold")
ax.set_title(("MERGE-o4 회귀 기준선 — o4-gpu-parallel 5c72d37 + main 5110dbc (merge, 커밋 전), .venv-openpi-live"
              if ko else "MERGE-o4 regression baselines — merged tree, .venv-openpi-live"), fontsize=10.5, color=figstyle.INK, loc="left")
fig.text(0.01, 0.02, ("import: benchmark.__file__ = ${AG3S_PROJECT_ROOT}/benchmark/__init__.py · 바깥 모듈 0 (legacy 53 / cuRobo 57)  ·  "
          f"wall {56.3} s / {57.0} s  ·  새 프로세스 GPU 최대 {s['legacy']['gpu']['new_process_MiB_max']} / {s['curobo']['gpu']['new_process_MiB_max']} MiB"),
         fontsize=8, color=figstyle.INK_2)
fig.savefig(F + ".png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
json.dump({"source": f"{O}/summary.json", "rows": rows, "header": hdr}, open(F + ".json", "w"), indent=1, ensure_ascii=False)
print(F + ".png", ko)
