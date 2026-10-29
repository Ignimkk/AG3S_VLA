"""T43 TA2 figures 2 (graph) and 3 (table) from agg/agg_V8.json (+ agg_V7g.json for the contacts)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from AG3S.ag3s.experiments.common import figstyle

D = _ag3s_workspace_path('outputs/impl/T43TA/')
a8 = json.load(open(D + "agg/agg_V8.json")); a7 = json.load(open(D + "agg/agg_V7g.json"))
V = list(a8["table"])
T = a8["table"]; T7 = a7["table"]
ko = figstyle.use_korean()
panels = [
    ("hurdle_servo_p10", "servo 예측 허들 clearance p10 (mm, 교차 226 청크)", lambda v: T[v]["hurdle_servo_mm"]["p10"]),
    ("n_servo_lt0", "servo 예측 < 0 청크 수 (교차 226)", lambda v: T[v]["n_cross_servo_lt0"]),
    ("contacts", "접촉 순간 servo 예측 clear (V8 14 + V7g 8 = 22)", lambda v: T[v]["contacts"]["servo_clear"] + T7[v]["contacts"]["servo_clear"]),
    ("acc_bd_p90", "청크 경계 명령 2차 차분 p90 (rad/row²)", lambda v: T[v]["acc_boundary_all_rad"]["p90"]),
    ("reversal", "청크 시작 손바닥 방향 반전 비율", lambda v: T[v]["reversal_share_all"]),
    ("dev_palm_p90", "정책과의 손바닥 차이 p90 (mm, 전 청크)", lambda v: T[v]["dev_palm_all_mm"]["p90"]),
    ("new_viol", "새 충돌 행 청크 (base 대비, 전 단계)", lambda v: T[v]["new_viol_total"]),
    ("cpu", "TO CPU 시간 비 (청크별 중앙, base = 1)", lambda v: T[v]["cpu_ratio"]["median"]),
]
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
figstyle.style_axes(fig, list(axes.ravel()))
data = {}
for ax, (k, title, f) in zip(axes.ravel(), panels):
    vals = [f(v) for v in V]
    data[k] = dict(zip(V, vals))
    cols = [figstyle.CATEGORICAL[0] if v in ("base", "Q4") else figstyle.CATEGORICAL[1] if v.startswith("A") and "B" not in v
            else figstyle.CATEGORICAL[2] if "C" not in v else figstyle.CATEGORICAL[3] for v in V]
    ax.barh(range(len(V)), vals, color=cols)
    ax.set_yticks(range(len(V))); ax.set_yticklabels(V, fontsize=7); ax.invert_yaxis()
    ax.set_title(title if ko else k, fontsize=8.5)
    for i, x in enumerate(vals):
        ax.text(x, i, f" {x:.3g}", va="center", fontsize=6.5)
fig.suptitle("T43 TA2 오프라인 (V8 19 run, 1037 청크 재풀이; 파랑 = 기준 · 주황 = A · 초록 = A+B · 노랑 = A+B+C)" if ko else
             "T43 TA2 offline (V8 19 runs, 1037 chunks re-solved)", fontsize=10)
fig.tight_layout(); fig.savefig(D + "fig/t43ta-2-graph.png", dpi=140)
json.dump({"source": ["agg/agg_V8.json", "agg/agg_V7g.json"], "panels": data}, open(D + "fig/t43ta-2-graph.json", "w"), indent=1)

# table
rows = []
for v in V:
    t, g = T[v], T[v]["grasp"]["at_apple"]
    rows.append([v, t["hurdle_cmd_mm"]["p10"], t["hurdle_servo_mm"]["p10"], t["n_cross_servo_lt0"],
                 f"{t['contacts']['servo_clear'] + T7[v]['contacts']['servo_clear']}/22", t["dev_palm_all_mm"]["median"],
                 t["acc_boundary_all_rad"]["p90"], round(t["reversal_share_all"], 3), t["n_vel_boundary_gt1"],
                 f"{t['new_viol_total']}/{t['fixed_viol_total']}", g["moved_vs_q4"], g["n_outward_gt_2mm_vs_q4"],
                 g["outward_vs_q4_mm"].get("max"), t["cpu_ratio"]["median"]])
hdr = ["variant", "cmd p10", "servo p10", "servo<0", "contact", "palm dev med", "acc bd p90", "reversal",
       "vel bd>1", "new/fixed", "grasp moved", "grasp out>2", "grasp out max", "CPU x"]
fig, ax = plt.subplots(figsize=(17, 0.42 * len(rows) + 1.4))
figstyle.style_axes(fig, [ax]); ax.axis("off")
tb = ax.table(cellText=[[str(c) for c in r] for r in rows], colLabels=hdr, loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
ax.set_title("T43 TA2 — V8 오프라인 표 (grasp = 사과 앞 72 청크, Q4 대비; 교차 = 허들 50 mm 안 226 청크)" if ko else
             "T43 TA2 — V8 offline table", fontsize=10)
fig.tight_layout(); fig.savefig(D + "fig/t43ta-3-table.png", dpi=140)
json.dump({"header": hdr, "rows": rows}, open(D + "fig/t43ta-3-table.json", "w"), indent=1)
print("ok")
