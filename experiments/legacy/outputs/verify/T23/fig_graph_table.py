"""T23 figure (2) 그래프 · (3) 표."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from AG3S.ag3s.experiments.common import figstyle
from AG3S.trajopt import wire

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t23'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T23'))
ko = figstyle.use_korean()
s1 = json.loads((V / "m1/t16_reasons_summary.json").read_text())
s2 = json.loads((V / "m2/t14_seq22_verdict.json").read_text())
a3 = json.loads((V / "m3/hold_drift_analysis.json").read_text())
S3 = json.loads((V / "m3/hold_drift_summary.json").read_text())
C = figstyle.CATEGORICAL
KCOL = {"allowed_contact": C[2], "budget_only": C[5], "occluded_target": C[6], "uncertified_waived": C[4],
        "collision": C[7], "unverified": C[3], "uncertified": C[1], "comms": C[4]}

# ---------------------------------------------------------------- graph
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
ax0 = fig.add_subplot(gs[0, :]); ax1 = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[1, 1])
figstyle.style_axes(fig, [ax0, ax1, ax2])
rows = [("old", "T16 기록 (옛 판정)" if ko else "T16 recorded (old)"),
        ("main|allow_unresolved=False", "새 판정 main (기본)" if ko else "new: main (default)"),
        ("main|allow_unresolved=True", "main · allow_unresolved_tier=True"),
        ("distance|allow_unresolved=False", "distance 근거 (사과 공 점)" if ko else "distance evidence"),
        ("nomask|allow_unresolved=False", "target_free_mask 없음" if ko else "no target_free mask"),
        ("nomask|allow_unresolved=True", "no mask · allow_unresolved=True")]
chunks = s1["chunks"]
sidecar = {"graph_a": [], "graph_b": {}, "graph_c": {}}
for yi, (key, label) in enumerate(rows):
    y = len(rows) - 1 - yi
    for xi, c in enumerate(chunks):
        if key == "old":
            kinds, safe = ["violated → HOLD"], False
        else:
            kinds, safe = c["kinds"][key], c["safe"][key]
        n = max(1, len(kinds))
        for j, k in enumerate(kinds):
            ax0.add_patch(Rectangle((xi + j / n, y + 0.08), 1 / n, 0.84, fc=KCOL.get(k, figstyle.INK_2), ec="white", lw=0.5))
        ax0.text(xi + 0.5, y + 0.5, "E" if safe else "H", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        sidecar["graph_a"].append({"row": key, "chunk": c["chunk_index"], "t": c["t_step"], "kinds": kinds, "execute": safe})
ax0.set_xlim(0, len(chunks)); ax0.set_ylim(0, len(rows))
ax0.set_xticks(np.arange(len(chunks)) + 0.5)
ax0.set_xticklabels([f"{c['t_step']}" + ("*" if c["attached"] else "") for c in chunks], fontsize=7)
ax0.set_yticks(np.arange(len(rows)) + 0.5); ax0.set_yticklabels([r[1] for r in rows][::-1], fontsize=8)
ax0.axvspan(0, 14, color=C[2], alpha=0.06, zorder=-1)
ax0.set_xlabel("T16 chunk t_step (* = attached 점 기록, 재구성 불가)" if ko else "T16 chunk t_step (* = attached)")
handles = [Rectangle((0, 0), 1, 1, fc=KCOL[k]) for k in ("allowed_contact", "unverified", "collision")] + [Rectangle((0, 0), 1, 1, fc=figstyle.INK_2)]
ax0.legend(handles, ["allowed_contact", "unverified", "collision", "old: violated"], ncol=4, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.2))
ax0.set_title(("(a) T16 unsafe 25 chunk 재판정 — 색 = kind, E=실행 H=HOLD, 옅은 초록 = t=72–176\n가정: manipulated = 사과 참값, r=45 mm" if ko else
               "(a) T16 25 unsafe chunks re-judged"), fontsize=10, loc="left", pad=22)

t = np.asarray(a3["runs"]["replay|legacy|hold"]["t"])
for key, lab, colr, ls in (("replay|legacy|hold", "legacy (재생)", C[1], "-"), ("replay|fixed|hold", "fixed (재생)", C[0], "-"),
                           ("record|legacy|hold", "legacy (t=72 qpos 에서 시작, qvel=0)", C[1], ":"), ("record|fixed|hold", "fixed (t=72 qpos 에서 시작, qvel=0)", C[0], ":")):
    y = a3["runs"][key]["finger_drift_from_entry_mm"]
    ax1.plot(t, y, ls=ls, color=colr, lw=2 if ls == "-" else 1.4, label=lab if ko else key)
    sidecar["graph_b"][key] = y
rec = a3["runs"]["replay|legacy|hold"]["t16_recorded_finger_drift_from_t72_mm"]
ax1.plot(t[:len(rec)], rec, "--", color=figstyle.INK, lw=1.1, label="T16 기록 qpos" if ko else "T16 recorded")
sidecar["graph_b"]["t16_recorded"] = rec
ax1.set_xlabel("control step t"); ax1.set_ylabel("손가락 구 중심 이동 (mm, t=72 대비)" if ko else "finger centre displacement from t=72 (mm)")
ax1.set_title("(b) HOLD 64 step (8 chunk × 8, 각 33 sim step) 동안 손가락 드리프트" if ko else "(b) finger drift during HOLD", fontsize=10, loc="left")
ax1.legend(fontsize=8); ax1.grid(True, color=figstyle.GRID_INK, lw=0.5)

for key, lab, colr, ls in (("replay|legacy|close_then_hold", "legacy · t=72 닫힘 명령 뒤 HOLD", C[1], "-"),
                           ("replay|fixed|close_then_hold", "fixed · t=72 닫힘 명령 뒤 HOLD", C[0], "-"),
                           ("replay|legacy|hold", "legacy · HOLD (열림)", C[1], ":"), ("replay|fixed|hold", "fixed · HOLD (열림)", C[0], ":")):
    r = a3["runs"][key]
    ax2.plot(t, r["applied_gripper_left"], ls=ls, color=colr, lw=2 if ls == "-" else 1.4, label=lab if ko else key, drawstyle="steps-post")
    sidecar["graph_c"][key] = {"applied_gripper_left": r["applied_gripper_left"], "measured_gripper_left": r["measured_gripper_left"]}
ax2.set_xlabel("control step t"); ax2.set_ylabel("왼 gripper 명령 (d.ctrl / OPEN, 1=열림 0=닫힘)" if ko else "left gripper command (normalized)")
ax2.set_title("(c) HOLD 중 gripper 명령 — t=72 한 step 만 닫힘(0.0) 실행 후 HOLD (합성)" if ko else "(c) gripper command during HOLD", fontsize=10, loc="left")
ax2.legend(fontsize=8, loc="center right"); ax2.grid(True, color=figstyle.GRID_INK, lw=0.5); ax2.set_ylim(-0.05, 1.08)
fig.tight_layout()
fig.savefig(FIG / "t23-reasons-hold-graph.png", dpi=130)
(FIG / "t23-reasons-hold-graph.json").write_text(json.dumps(sidecar))

# ---------------------------------------------------------------- table
pv = s1["per_variant"]
fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), gridspec_kw={"width_ratios": [0.9, 1.35, 1.2]})
figstyle.style_axes(fig, list(axs))
for ax in axs:
    ax.axis("off")
t1 = [[k, wire.GATE_DEFAULT[k]] for k in wire.GATE_DEFAULT]
tb = axs[0].table(cellText=t1, colLabels=["kind", "gate (wire.GATE_DEFAULT)"], loc="upper center", cellLoc="left")
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r > 0 and c == 1:
        cell.set_facecolor("#e3f4ec" if cell.get_text().get_text() == "execute" else "#fbe3e2")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
axs[0].set_title("kind → 처리" if ko else "kind → action", fontsize=10)
t2 = []
for key, lab in rows[1:]:
    p = pv[key]
    t2.append([lab, str(p["n_execute"]), str(p["n_hold"]), str(p["n_collision_chunks"]), str(p["n_unverified_only"]),
               f"{p['grasp_window_t72_176']['n_execute']}/{p['grasp_window_t72_176']['n']}"])
tb = axs[1].table(cellText=t2, colLabels=["T16 재판정 변형" if ko else "variant", "실행", "HOLD", "collision 포함", "unverified 만", "t=72–176 실행"],
                  loc="upper center", cellLoc="center", colWidths=[0.36, 0.1, 0.1, 0.15, 0.15, 0.15])
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.6)
axs[1].set_title("T16 unsafe 25 chunk — 새 kind 집계 (옛: 25/25 HOLD)" if ko else "T16 aggregate", fontsize=10)
axs[1].text(0.0, 0.2, ("가정: manipulated = object_poses.apple.pos, r = 0.045 m, state visible\n"
                       "권한 link = ee_left · ee_finger_l1 · ee_finger_l2 (contact_links['left'])\n"
                       "재구성: T24 fixed_input_ablation (기록 clearance 재현 최대 %.1e mm)\n"
                       "attached 11 chunk(t=184–264): 쥔 물체 점 기록 없음 → 기록 최악 행을 collision 으로" % max(c["fidelity_record_clearance_max_abs_diff_mm"] for c in chunks)),
             fontsize=8.5, transform=axs[1].transAxes, va="top")
t3 = []
for c in s2["cases"]:
    m = c["manipulated"] or {}
    t3.append([f"#2 seq22 {m.get('state') or '없음'} id={m.get('id')}", "budget" if c["time_budget_hit"] else "-", ", ".join(c["kinds"]), "실행" if c["safe"] else "HOLD"])
rl, rf = S3["replay|legacy|hold"], S3["replay|fixed|hold"]
cl, cf = S3["replay|legacy|close_then_hold"], S3["replay|fixed|close_then_hold"]
t3 += [["#3 손가락 이동 t=96 (mm)", "", f"legacy {rl['finger_drift_from_entry_mm@t96']:.1f}", f"fixed {rf['finger_drift_from_entry_mm@t96']:.1f}"],
       ["#3 손가락 이동 t=135 (mm)", "", f"legacy {rl['finger_drift_from_entry_mm@t135']:.1f}", f"fixed {rf['finger_drift_from_entry_mm@t135']:.1f}"],
       ["#3 팔 관절 step 최대 평균 (°/step)", "", f"legacy {rl['arm_step_max_deg_mean_t80_135']:.3f}", f"fixed {rf['arm_step_max_deg_mean_t80_135']:.3f}"],
       ["#3 q_hold 대비 관절 최대 편차 t=135 (°)", "", f"legacy {rl['arm_max_abs_dev_from_q_hold_deg@t135']:.2f}", f"fixed {rf['arm_max_abs_dev_from_q_hold_deg@t135']:.2f}"],
       ["#4 닫힘 명령 뒤 HOLD 끝 gripper 명령", "", f"legacy {cl['applied_gripper_left_first_last'][2]:.3f}", f"fixed {cf['applied_gripper_left_first_last'][2]:.3f}"],
       ["재생 legacy vs T16 기록 손가락 (mm, max)", "", f"{rl['sim_vs_t16_finger_mm_max']:.2f}", ""]]
tb = axs[2].table(cellText=t3, colLabels=["항목", "", "kinds / legacy", "결과 / fixed"], loc="upper center", cellLoc="left", colWidths=[0.38, 0.08, 0.37, 0.17])
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
axs[2].set_title("#2 T14 seq 22 · #3/#4 MuJoCo HOLD" if ko else "#2 / #3 / #4", fontsize=10)
fig.suptitle("T23 검증 — verdict_reasons · HOLD 정책 (숫자만)" if ko else "T23 verification", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "t23-reasons-table.png", dpi=130)
(FIG / "t23-reasons-table.json").write_text(json.dumps({"gate_default": wire.GATE_DEFAULT, "t16_per_variant": pv, "t14_seq22": s2["cases"],
                                                       "hold_summary": {k: S3[k] for k in ("replay|legacy|hold", "replay|fixed|hold", "replay|legacy|close_then_hold", "replay|fixed|close_then_hold")}}, indent=1, default=str))
print("saved")
