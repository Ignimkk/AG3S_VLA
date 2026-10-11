from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs

KO = fs.use_korean()
V = _ag3s_workspace_path('outputs/verify/T22/')
OUT = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t22/t22-grasp-timeline')
sc = json.load(open(V + "scenarios_t22.json"))
sw = json.load(open(V + "empty_sweep_t22.json"))
rp = {r: json.load(open(V + f"replay_{r}_apple.json"))["rows"] for r in ("t17", "t16")}
rps = {r: json.load(open(V + f"replay_{r}_server_prev.json"))["rows"] for r in ("t17", "t16")}
STATE_C = {"searching": "#eeeeec", "latched": fs.SEQ_BLUE[0], "closing": "#fde3c6", "held": "#c9eedd", "placed": "#e6e1f5"}

fig, axes = plt.subplots(3, 1, figsize=(12, 12.5), gridspec_kw={"height_ratios": [1.1, 1.1, 0.9]})
fs.style_axes(fig, axes)
side = {}
for ax, run, title in ((axes[0], "t17", "T17 shadow (성공 파지) — 재구성 피드백 재생" if KO else "T17 shadow (successful grasp) - replay"),
                       (axes[1], "t16", "T16 closed loop --no-collision (닫힘이 HOLD 로 취소)" if KO else "T16 closed loop (close cancelled by HOLD)")):
    ps = sc["per_step"][run]
    t = np.array([p["t"] for p in ps]); op = np.array([p["opening"] for p in ps]); cmd = np.array([p["cmd"] for p in ps])
    ex = np.array([p["executed"] for p in ps]); rch = np.array([p["urdf_l1_apple_mm"] for p in ps])
    rows = rp[run]
    # state bands per request (request at t_step covers [t, t+8))
    for r in rows:
        ax.axvspan(r["t_step"], r["t_step"] + 8, color=STATE_C[r["state"]], lw=0, zorder=0)
    hold = ~ex
    for tt in t[hold]:
        ax.axvspan(tt, tt + 1, ymin=0.0, ymax=0.04, color=fs.CATEGORICAL[7], lw=0)
    ax.step(t, cmd, where="post", color=fs.CATEGORICAL[1], lw=1.4, label="계획 gripper (actions_reference 열 7)" if KO else "planned gripper (col 7)")
    ax.plot(t, op, color=fs.CATEGORICAL[0], lw=1.8, label="측정 개도 |qpos[27]|/0.045" if KO else "measured opening")
    ax.axhline(0.85, color=fs.INK_2, lw=0.8, ls="--"); ax.text(t[-1] + 1, 0.85, "0.85", fontsize=7, color=fs.INK_2, va="center")
    ax.axhline(0.10, color=fs.INK_2, lw=0.8, ls=":"); ax.text(t[-1] + 1, 0.10, "empty_below 0.10", fontsize=7, color=fs.INK_2, va="center")
    ax.set_ylim(-0.05, 1.2); ax.set_ylabel("gripper (1 = 열림)" if KO else "gripper (1 = open)")
    ax2 = ax.twinx(); ax2.set_facecolor("none")
    ax2.plot(t, rch, color=fs.CATEGORICAL[2], lw=1.4, label="ee_finger_l1 원점(URDF FK)↔사과 mm" if KO else "ee_finger_l1 origin - apple mm")
    srv = [(r["t_step"], r["reach_server_prev_mm"]) for r in rps[run] if r["reach_server_prev_mm"] is not None]
    ax2.plot([s[0] for s in srv], [s[1] for s in srv], "o", ms=3, color=fs.CATEGORICAL[6], label="원점↔서버 target_centroid(직전) mm" if KO else "origin - server centroid (prev) mm")
    ax2.axhline(120, color=fs.CATEGORICAL[2], lw=0.9, ls="--"); ax2.text(t[0], 124, "reach 120 mm", fontsize=7, color=fs.CATEGORICAL[2])
    ax2.set_ylim(0, 700); ax2.set_ylabel("mm", color=fs.INK_2); ax2.tick_params(colors=fs.INK_2, labelsize=8)
    for s in ("top",): ax2.spines[s].set_visible(False)
    for r in rows:
        if r["attach"]:
            ax.annotate(f"attach t={r['t_step']}\n개도 {r['measured']:.3f}, reach {r['reach_mm']:.1f} mm" if KO else f"attach t={r['t_step']}",
                        (r["t_step"], r["measured"]), (r["t_step"] + 12, 0.35), fontsize=8, color=fs.INK,
                        arrowprops=dict(arrowstyle="->", color=fs.INK_2))
        if r["detach"]:
            ax.annotate(f"detach t={r['t_step']}", (r["t_step"], r["measured"]), (r["t_step"] + 6, 1.1), fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=fs.INK_2))
        if r["legacy_attach"]:
            ax.axvline(r["t_step"], color=fs.CATEGORICAL[7], lw=1, ls="-.")
            ax.text(r["t_step"] + 1, 1.14, f"옛 경로 attach 판단 t={r['t_step']}" if KO else f"legacy attach t={r['t_step']}", fontsize=7, color=fs.CATEGORICAL[7])
    ax.set_title(title, fontsize=10, color=fs.INK, loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    from matplotlib.patches import Patch
    hs = [Patch(color=c, label=k) for k, c in STATE_C.items() if k != "searching"] + [Patch(color=fs.CATEGORICAL[7], label="HOLD step (executed=False)")]
    ax.legend(h1 + h2 + hs, l1 + l2 + [h.get_label() for h in hs], fontsize=7, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False)
    ax.set_xlim(0, max(t) + 10)
    side[run] = {"t": t.tolist(), "opening": op.round(4).tolist(), "cmd": cmd.round(4).tolist(), "executed": ex.tolist(), "reach_urdf_l1_apple_mm": rch.round(2).tolist(),
                 "server_prev_reach": srv, "states": [(r["t_step"], r["state"]) for r in rows],
                 "attach_t": [r["t_step"] for r in rows if r["attach"]], "detach_t": [r["t_step"] for r in rows if r["detach"]],
                 "legacy_attach_t": [r["t_step"] for r in rows if r["legacy_attach"]]}
axes[1].set_xlabel("control step t", labelpad=40)

# panel 3: opening trace, T17 real (with apple) vs MuJoCo-sim empty close
ax = axes[2]
t17 = [p for p in sc["per_step"]["t17"] if 104 <= p["t"] <= 135]
ax.plot([p["t"] - 115 for p in t17], [p["opening"] for p in t17], color=fs.CATEGORICAL[0], lw=2.2, label="T17 실측 (사과 잡음), 명령 0 시작 = 0" if KO else "T17 measured (apple)")
r0 = next(x for x in sw if x["close_target"] == 0.0 and x["start_step_in_chunk"] == 0)
tr = [1.0] + r0["trace"][8:24]
ax.plot(np.arange(len(tr)), tr, color=fs.SEQ_BLUE[8], lw=1.4, label="MuJoCo 빈손 닫힘 (명령 0.0)" if KO else "sim empty close (cmd 0.0)")
for s in range(8):
    r = next(x for x in sw if x["close_target"] == 0.0 and x["start_step_in_chunk"] == s)
    bt = 8 - s
    col = fs.CATEGORICAL[7] if r["attach_req"] else fs.INK_2
    ax.plot([bt], [r["opening_at_boundaries"][1]], "o", color=col, ms=6, zorder=5)
    ax.annotate(f"s={s}: {'attach' if r['attach_req'] else '거절' if KO else 'no'}", (bt, r["opening_at_boundaries"][1]),
                (bt + 0.3, r["opening_at_boundaries"][1] + 0.06), fontsize=7, color=col)
ax.axhline(0.10, color=fs.INK_2, lw=0.8, ls=":"); ax.axhline(0.85, color=fs.INK_2, lw=0.8, ls="--")
ax.set_xlim(-6, 16); ax.set_ylim(-0.05, 1.1)
ax.set_xlabel("닫힘 명령 첫 적용 뒤 step 수 (점 = 닫힘이 청크의 s 번째 스텝에서 시작할 때 청크 경계=다음 요청에서 읽히는 개도, 빨강 = attach)" if KO else "step since first close command")
ax.set_ylabel("측정 개도" if KO else "opening")
n_att = sum(1 for x in sw if x["attach_req"])
ax.set_title((f"#4e MuJoCo 빈손 닫힘 (T17 seq2 자세, 팔 고정, reach 0.042 m 강제): 24 설정 중 attach {n_att}") if KO else f"#4e sim empty close: attach {n_att}/24", fontsize=10, loc="left", color=fs.INK)
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=150)
side["empty_sweep"] = [{k: x[k] for k in ("close_target", "start_step_in_chunk", "opening_at_boundaries", "attach_req", "attach_opening")} for x in sw]
json.dump(side, open(OUT + ".json", "w"), indent=1, ensure_ascii=False)
print(OUT + ".png", KO)
