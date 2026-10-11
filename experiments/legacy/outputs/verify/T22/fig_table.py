from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean()
V = _ag3s_workspace_path('outputs/verify/T22/')
OUT = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t22/t22-scenario-table')
sc = json.load(open(V + "scenarios_t22.json")); sw = json.load(open(V + "empty_sweep_t22.json"))
CK = ["settled", "closed", "dropped", "not_empty", "blocked", "manipulated", "reach"]
rows = []


def decisive(rs, key_t, key_meas, key_reach=None):
    """attach frame, else the first frame that has checks (first closing decision)."""
    a = [r for r in rs if r["attach"]]
    if a: return a[0]
    c = [r for r in rs if r.get("checks")]
    return c[0] if c else None


def fmt_checks(r):
    if not r or not r.get("checks"): return ["-"] * 7
    return ["T" if r["checks"][k] else "F" for k in CK]


for run, lab in (("t17", "#1 T17"), ("t16", "#2 T16"), ("t14", "#3 T14")):
    for src in ("apple", "server_prev"):
        d = json.load(open(V + f"replay_{run}_{src}.json")); rs = d["rows"]; s = d["summary"]
        r = decisive(rs, "t_step", "measured")
        rows.append([f"{lab} ({'사과 참값' if src == 'apple' else '서버 centroid'})",
                     f"{s['n_attach']}" + (f" (t={','.join(map(str, s['attach_t']))})" if s["attach_t"] else ""),
                     ",".join(map(str, s["detach_t"])) or "-",
                     "-" if not r else str(r["t_step"]),
                     "-" if not r or r["measured"] is None else f"{r['measured']:.3f}",
                     "-" if not r or r["last_command"] is None else f"{r['last_command']:.3f}",
                     "-" if not r or r["reach_mm"] is None else f"{r['reach_mm']:.1f}"] + fmt_checks(r) +
                    [",".join(map(str, s["legacy_attach_t"])) or "-"])
for key in ("4_empty_hand", "5_hold_release"):
    for s in sc[key]:
        rs = s["rows"]; r = decisive(rs, "req", "measured")
        name = s["name"].split(" ")[0]
        SHORT = {"4a": "빈손 합성: 명령 0, 개도 0.004 까지", "4b": "빈손 합성: T17 명령열, 개도=명령", "4c": "빈손 합성: 명령 0, 개도 0.03 정지",
                 "4d": "빈손 MuJoCo: 닫힘 s=3 시작 (T17 과 같은 위치)", "5a": "파지 후 HOLD x4 → 열림 x2", "5b": "파지 후 HOLD, 열림 1회 튐, 다시 열림 x2",
                 "5c": "파지 후 placed=True (닫힌 채) → placed+열림", "5d": "파지 후 부분 HOLD 3/8 (재명령 0.72) → 열림 x2"}
        lc = None
        if r:
            ex = [a for a, e in zip(r["applied"], r["executed"] or [True] * 8) if e]
            lc = ex[-1] if ex else None
        rows.append([f"#{name} " + SHORT.get(name, s["name"][:34]), f"{s['n_attach']}" + (f" (req {s['attach_req'][0]})" if s["attach_req"] else ""),
                     ",".join(f"req {x}" for x in s["detach_req"]) or "-", "-" if not r else f"req {r['req']}",
                     "-" if not r else f"{r['measured']:.3f}", "-" if lc is None else f"{lc:.3f}", f"{s['reach_m_given'] * 1000:.0f}"] + fmt_checks(r) + ["n/a"])
n_att = sum(1 for x in sw if x["attach_req"])
rows.append([f"#4e 빈손 MuJoCo: 시작 s=0..7 x 목표 0/0.3/0.57", f"{n_att}/24", "-", "-", "0.16-0.76", "0 / 0.3 / 0.57", "42"] + ["(sweep)"] * 7 + ["n/a"])
gate = sc["last_gate_on_server_targets"]
rows.append(["마지막 관문 AG3S.attach(reach=0.12): 옛 attach 시점 서버 target", f"거절 {sum(g['gate']=='rejected' for g in gate)}/{len(gate)}", "-",
             "T17/T16/T14 x2", "-", "-", f"{min(g['reach_mm'] for g in gate):.0f}-{max(g['reach_mm'] for g in gate):.0f}"] + ["-"] * 6 + ["F"] + ["-"])
cols = ["시나리오", "attach", "detach", "판정 프레임", "개도", "마지막 명령", "reach mm"] + CK + ["옛 경로 attach t"]

fig, ax = plt.subplots(figsize=(19, 0.3 * len(rows) + 0.9))
fs.style_axes(fig, ax); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=cols, loc="upper left", cellLoc="center", colLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.45)
widths = [0.25, 0.07, 0.06, 0.06, 0.05, 0.07, 0.05] + [0.045] * 7 + [0.07]
for (i, j), c in tb.get_celld().items():
    c.set_width(widths[j]); c.set_edgecolor(fs.GRID_INK)
    if i == 0:
        c.set_facecolor(fs.SEQ_BLUE[1]); c.set_text_props(color=fs.INK, weight="bold")
    else:
        c.set_facecolor(fs.SURFACE)
        txt = c.get_text().get_text()
        if 7 <= j <= 13 and txt == "F": c.set_facecolor("#fbd9d9")
        if 7 <= j <= 13 and txt == "T": c.set_facecolor("#d9f2e6")
        if j == 1 and txt[:1] not in ("0", "-"): c.set_facecolor("#fde3c6")
    if j == 0: c._loc = "left"; c.set_text_props(ha="left")
ax.set_title("T22 시나리오별 attach/detach 와 7 검사 (판정 프레임 = attach 프레임, 없으면 첫 closing 판정 프레임). T=참 F=거짓. 재생은 기록에서 재구성한 피드백, 폐루프 새 실행 아님",
             fontsize=9, color=fs.INK, loc="left")
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
json.dump({"columns": cols, "rows": rows}, open(OUT + ".json", "w"), indent=1, ensure_ascii=False)
print(OUT)
