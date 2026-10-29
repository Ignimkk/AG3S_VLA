"""PPT figures: TO V8 -> V11AB (why, how, result, TO formulation).

Numbers come only from recorded sources (no new runs):
  * AG3S/ag3s/docs/AG3S_T0T6_LOG.md T43 sections 7-13 (numbers from T43*.verify.json)
  * AG3S/ag3s/docs/handoff/T43XM.verify.json (deflection profile by row k)
  * AG3S/ag3s/docs/figures/t43/t43hx-timeline.json (V8 ep1807 s18073 carry, P vs M distance)
  * T43 TA offline numbers (implementer, open loop) as quoted in the log section 12-1

Run from the repo root:  .venv-ag3s/bin/python doc/archive/presentations/to_v8_v11ab/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "AG3S/ag3s/experiments"))
from common import figstyle as fs  # noqa: E402

OUT = Path(__file__).resolve().parent
KO = fs.use_korean()
C = fs.CATEGORICAL
RED = C[7]
ARM_COL = {"V8": C[0], "V11A": C[1], "V11AB": C[2], "V9": C[6], "V10": C[3], "E0": fs.INK_2}


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=170)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    plt.close(fig)


def label_bars(ax, bars, fmt="{:g}", dy=0.0):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=fs.INK)


# --------------------------------------------------------------------------------------------
# Figure 1 — the plan was clear, the robot was not: plan P = command C, measured M lags ~2 rows
# --------------------------------------------------------------------------------------------
def fig1():
    hx = json.loads((ROOT / "AG3S/ag3s/docs/figures/t43/t43hx-timeline.json").read_text())
    p = next(q for q in hx["panels"] if q["key"] == "V8/H/V8_ep1807_s18073"
             and q["phase"] == "carry" and q["group"] == "apple")
    t, P, M, hold = p["t"], p["P_geo_mm"], p["M_geo_mm"], p["hold"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 8.6))
    # (a) one real crossing
    a = ax[0, 0]
    hold_t = [ti for ti, h in zip(t, hold) if h]
    if hold_t:
        a.axvspan(min(hold_t) - 0.5, max(hold_t) + 0.5, color=fs.GRID_INK, alpha=0.5, lw=0)
        a.text((min(hold_t) + max(hold_t)) / 2, 112, "HOLD", ha="center", fontsize=8, color=fs.INK_2)
    a.plot(t, P, "-o", ms=3.5, color=C[1], label="계획 P = 명령 C (TO 출력)")
    a.plot(t, M, "-o", ms=3.5, color=C[0], lw=2.2, label="실제 M (MuJoCo 측정)")
    a.axhline(0, color=RED, lw=1)
    i0 = t.index(173)
    a.annotate(f"계획 {P[i0]:.1f} mm\n실제 {M[i0]:.1f} mm", xy=(173, M[i0]), xytext=(176.5, 62),
               fontsize=9, arrowprops=dict(arrowstyle="->", color=fs.INK_2))
    a.set_xlabel("control row t (1 row = 0.066 s)")
    a.set_ylabel("사과 – 허들 거리 (mm)")
    a.set_ylim(-8, 120)
    a.set_title("(a) 실제 장면 1개: V8 ep1807 s18073, 사과를 들고 허들 통과")
    a.legend(frameon=False, fontsize=8, loc="upper right")

    # (b) where the gap is: P-C vs C-M, with and without TO
    a = ax[0, 1]
    groups = ["|P − C|\n계획 vs 명령", "|C − M|\n명령 vs 실제"]
    v8 = [(0.0, 0.0), (15.1, 31.0)]
    e0 = [(None, None), (20.0, 31.1)]
    x = [0, 1]
    w = 0.32
    b1 = a.bar([xi - w / 2 for xi in x], [v[0] for v in v8], w, color=ARM_COL["V8"], label="V8 (TO 있음) 중앙값")
    a.scatter([xi - w / 2 for xi in x], [v[1] for v in v8], marker="_", s=300, color=fs.INK, zorder=3,
              label="p90")
    b2 = a.bar([1 + w / 2], [e0[1][0]], w, color=ARM_COL["E0"], label="E0 (TO 없음) 중앙값")
    a.scatter([1 + w / 2], [e0[1][1]], marker="_", s=300, color=fs.INK, zorder=3)
    label_bars(a, b1[1:], "{:.1f}", 0.4)
    label_bars(a, b2, "{:.1f}", 0.4)
    a.text(-w / 2, 0.6, "0.0 (p90 0.0)", ha="center", fontsize=9)
    a.text(w / 2, 0.6, "E0: TO 계획 없음", ha="center", fontsize=8, color=fs.INK_2)
    a.set_xticks(x)
    a.set_xticklabels(groups)
    a.set_ylabel("EE 위치 차이 (mm), 운반 중 실행 행")
    a.set_ylim(0, 36)
    a.set_title("(b) 차이는 전부 명령→실제 구간 · 지연 τ ≈ 1.9 row (TO 없이도 동일)")
    a.legend(frameon=False, fontsize=8, loc="upper left")

    # (c) cause of every below-margin moment
    a = ax[1, 0]
    causes = ["(i) 계획은 clear,\n실제만 진입\n(servo lag)", "(ii) TO가 위반을\n봤지만 못 풂", "(iii) ESDF 오차",
              "(iv) margin 0 행\n(쥔 사과)", "(v) 기타"]
    vals = [232, 0, 0, 2, 0]
    bars = a.barh(range(5)[::-1], vals, color=[C[0], C[1], C[1], C[1], C[1]])
    for b, v in zip(bars, vals):
        a.text(v + 3, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=10)
    a.set_yticks(range(5)[::-1])
    a.set_yticklabels(causes, fontsize=8)
    a.set_xlim(0, 260)
    a.set_xlabel("순간 수 (V7g · V8 · V5 허들 14 run, 합 234)")
    a.set_title("(c) 허들 100 mm 안에서 실제 거리가 margin 10 mm 미만이 된 234 순간의 원인")

    # (d) command vs actual height over the bar
    a = ax[1, 1]
    arms = ["V5", "V7g", "V8"]
    cmd = [21.4, 18.0, 19.6]
    act = [3.1, 2.0, 3.8]
    xs = range(3)
    bb1 = a.bar([i - 0.18 for i in xs], cmd, 0.36, color=C[1], label="명령 C (TO가 보낸 높이)")
    bb2 = a.bar([i + 0.18 for i in xs], act, 0.36, color=C[0], label="실제 M")
    label_bars(a, bb1, "{:.1f}", 0.3)
    label_bars(a, bb2, "{:.1f}", 0.3)
    for i in xs:
        a.annotate("", xy=(i + 0.18, act[i] + 2.5), xytext=(i - 0.18, cmd[i] - 0.5),
                   arrowprops=dict(arrowstyle="->", color=fs.INK_2))
        a.text(i + 0.3, (cmd[i] + act[i]) / 2 + 2, f"−{cmd[i] - act[i]:.1f}", fontsize=9, color=RED)
    a.axhline(10, color=RED, lw=1, ls="--")
    a.text(-0.45, 10.4, "obstacle margin 10 mm", fontsize=8, color=RED, ha="left")
    a.set_xticks(list(xs))
    a.set_xticklabels(arms)
    a.set_ylabel("사과 바닥 – 막대 윗면, 중앙 (mm)")
    a.set_ylim(0, 26)
    a.set_title("(d) 허들 위에서 명령은 18–21 mm, 실제는 2–4 mm로 통과")
    a.legend(frameon=False, fontsize=8, loc="upper left")

    fs.style_axes(fig, ax)
    fig.suptitle("근거: TO 계획(=명령)은 margin을 지켰지만, 실제 로봇이 약 2 row 늦게 따라와 접촉한다", fontsize=13)
    fig.tight_layout()
    save(fig, "fig1_cause_plan_vs_actual", {
        "a_example": {"key": p["key"], "phase": "carry", "group": "apple", "t": t, "P_geo_mm": P,
                      "M_geo_mm": M, "hold": hold, "source": "AG3S/ag3s/docs/figures/t43/t43hx-timeline.json"},
        "b_PC_CM_mm": {"V8_PminusC_median_p90": [0.0, 0.0], "V8_CminusM_median_p90": [15.1, 31.0],
                       "E0_CminusM_median_p90": [20.0, 31.1], "lag_tau_rows_V8_E0": [1.90, 1.95],
                       "source": "T43TK.verify.json via AG3S_T0T6_LOG.md T43 8-1"},
        "c_causes": dict(zip(["servo_lag", "TO_unsolved", "ESDF_error", "margin0_row", "other"], vals))
        | {"source": "T43HX.verify.json via log T43 8-3"},
        "d_height_over_bar_mm": {"arms": arms, "command": cmd, "actual": act,
                                 "source": "T43XM.verify.json via log T43 11"},
    })


# --------------------------------------------------------------------------------------------
# Figure 2 — why lookahead (A) + deflection rate (B), and why not the alternatives
# --------------------------------------------------------------------------------------------
def fig2():
    xm = json.loads((ROOT / "AG3S/ag3s/docs/handoff/T43XM.verify.json").read_text())
    prof = xm["numbers"]["arms"]["V8"]["carry_apple"]["palm_D_profile_by_k_median_mm"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 8.6))
    # (a) V8 deflection restarts at every chunk start
    a = ax[0, 0]
    for c in range(3):
        xs = [c * 8 + k for k in range(8)]
        a.plot(xs, prof, "-o", ms=3.5, color=ARM_COL["V8"], label="V8 TO 편향 |D| (손바닥, 운반 중 중앙값)" if c == 0 else None)
        if c:
            a.plot([c * 8 - 1, c * 8], [prof[-1], prof[0]], ls=":", color=ARM_COL["V8"])
        a.axvline(c * 8 - 0.5, color=fs.GRID_INK, lw=1)
        a.text(c * 8 + 3.5, 17.3, f"chunk {c + 1}", ha="center", fontsize=8, color=fs.INK_2)
    a.annotate("row 0에서 정책 쪽으로\n되돌아감", xy=(8, prof[0]), xytext=(9.5, 3.0), fontsize=9,
               arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    a.set_ylim(0, 19)
    a.set_xlabel("실행 row (chunk 당 8 row; 같은 중앙값 profile을 3 chunk 반복 표시)")
    a.set_ylabel("|D| = |refined − reference| (mm)")
    a.set_title(f"(a) V8: 편향이 chunk 안에서 {prof[0]:.1f} → {prof[-1]:.1f} mm로 늦게 커지고, 매 chunk 다시 시작 → B")
    a.legend(frameon=False, fontsize=8, loc="lower right")

    # (b) what TO leaves vs what the lag takes away
    a = ax[0, 1]
    names = ["TO가 margin 위로\n남긴 여유", "servo lag가\n깎는 높이"]
    vals = [4.6, 15.8]
    bars = a.bar(names, vals, 0.5, color=[C[1], RED])
    label_bars(a, bars, "{:.1f} mm", 0.3)
    a.set_ylabel("mm (V8)")
    a.set_ylim(0, 19)
    a.set_title("(b) V8 TO는 margin을 겨우 넘기는 만큼만 비킨다 → 늦게, 작게 비킴")
    a.text(1, 17.6, "= 명령 19.6 − 실제 3.8 mm", ha="center", fontsize=8, color=fs.INK_2)
    a.text(0, 6.6, "(정책 위반 chunk에서 TO 결과의\nbinding clearance, margin 위 중앙)", ha="center", fontsize=8, color=fs.INK_2)

    # (c) the alternatives that were tried and rejected
    a = ax[1, 0]
    metrics = ["성공 (기존)", "접촉 없는 성공 (CT)", "놓은 뒤 손가락–허들\n접촉 run", "HOLD chunk ÷ 4"]
    data = {"V8": [13, 8, 9, 6 / 4], "V9\nservo model을 TO에": [2, 2, 0, 17 / 4],
            "V10\n쥔 사과 행에 margin": [10, 9, 7, 52 / 4]}
    cols = [ARM_COL["V8"], ARM_COL["V9"], ARM_COL["V10"]]
    w = 0.26
    raw_hold = {"V8": 6, "V9\nservo model을 TO에": 17, "V10\n쥔 사과 행에 margin": 52}
    for i, (k, v) in enumerate(data.items()):
        xs = [j + (i - 1) * w for j in range(len(metrics))]
        b = a.bar(xs, v, w, color=cols[i], label=k.replace("\n", " · "))
        for j, bb in enumerate(b):
            txt = str(raw_hold[k]) if j == 3 else f"{v[j]:g}"
            a.text(bb.get_x() + w / 2, bb.get_height() + 0.2, txt, ha="center", fontsize=8)
    a.set_xticks(range(len(metrics)))
    a.set_xticklabels(metrics, fontsize=8)
    a.set_ylabel("run / 14 (HOLD는 chunk 수 ÷ 4로 축소 표시)")
    a.set_ylim(0, 19)
    a.set_title("(c) 거부된 대안: lag를 TO에 직접 넣거나(V9) margin을 키우면(V10) 파지·HOLD가 무너짐")
    a.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)

    # (d) offline replay of the same V8 chunks with A, A+B, A+B+C
    a = ax[1, 1]
    var = ["base (V8)", "A", "A + B", "A + B + C"]
    p10 = [-0.56, -0.49, 2.94, 2.85]
    clear = [9, 9, 15, 15]
    cols = [ARM_COL["V8"], C[1], C[2], C[3]]
    b = a.bar(range(4), p10, 0.55, color=cols)
    for bb, v in zip(b, p10):
        a.text(bb.get_x() + 0.275, v + (0.12 if v >= 0 else -0.12), f"{v:+.2f}", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=9)
    a.axhline(0, color=RED, lw=1)
    a.set_ylabel("servo 예측 허들 clearance p10 (mm)")
    a.set_ylim(-1.6, 4.4)
    a.set_xticks(range(4))
    a.set_xticklabels([f"{n}\n접촉 22건 중 {c}건 해소" for n, c in zip(var, clear)], fontsize=8)
    a.set_title("(d) 같은 V8 chunk 1037개를 TO만 다시 풂 (offline, open loop): B가 핵심")

    fs.style_axes(fig, ax)
    fig.suptitle("방법 선택 근거: 늦게·작게 비키는 TO → 미리 보고(A), 비킨 상태를 유지(B)", fontsize=13)
    fig.tight_layout()
    save(fig, "fig2_method_rationale", {
        "a_V8_carry_apple_palm_D_profile_by_k_median_mm": prof,
        "a_source": "AG3S/ag3s/docs/handoff/T43XM.verify.json numbers.arms.V8.carry_apple",
        "b_mm": {"TO_binding_clearance_above_margin_V8_median": 4.6,
                 "command_minus_actual_height_V8": round(19.6 - 3.8, 1),
                 "source": "T43XM via log T43 11"},
        "c_alternatives_H14": {k.replace("\n", " "): {"S_old": v[0], "S_CT": v[1],
                                                       "finger_release_contact_runs": v[2],
                                                       "HOLD_chunks": raw_hold[k]}
                               for k, v in data.items()},
        "c_source": "T43V9 / T43V10.verify.json via log T43 9-10",
        "d_offline_TA": {"variants": var, "servo_pred_hurdle_clearance_p10_mm": p10,
                         "contacts_cleared_of_22": clear,
                         "source": "T43 TA implementer offline (open loop), log T43 12-1"},
    })


# --------------------------------------------------------------------------------------------
# Figure 3 — result V8 -> V11A -> V11AB (closed loop, paired, hurdle 14 runs)
# --------------------------------------------------------------------------------------------
def fig3():
    arms = ["V8", "V11A", "V11AB"]
    d = {"success_old": [13, 10, 12], "success_ct": [8, 8, 12],
         "carry_apple_contact_runs": [5, 2, 0], "release_finger_contact_runs": [9, 4, 0],
         "obstacle_stop_runs": [10, 6, 1],
         "apple_bar_median_mm": [3.8, 6.7, 13.6], "apple_bar_min_mm": [-6.4, -0.1, 7.1],
         "release_finger_bar_median_mm": [-1.4, 3.4, 9.6]}
    cols = [ARM_COL[a] for a in arms]
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.6))

    def grouped(a, keys, labels):
        w = 0.26
        for i, arm in enumerate(arms):
            xs = [j + (i - 1) * w for j in range(len(keys))]
            b = a.bar(xs, [d[k][i] for k in keys], w, color=cols[i], label=arm)
            label_bars(a, b, "{:g}", 0.15)
        a.set_xticks(range(len(keys)))
        a.set_xticklabels(labels, fontsize=9)

    a = ax[0, 0]
    grouped(a, ["success_old", "success_ct"], ["성공 (기존 규칙)", "접촉 없는 성공 (CT)"])
    a.set_ylim(0, 15.5)
    a.set_ylabel("run / 14")
    a.set_title("(a) 허들 14 run 성공: 접촉 없는 성공 8 → 12")
    a.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")

    a = ax[0, 1]
    grouped(a, ["carry_apple_contact_runs", "release_finger_contact_runs", "obstacle_stop_runs"],
            ["운반 중 사과–허들 접촉", "놓은 뒤 손가락–허들 접촉", "obstacle stop"])
    a.set_ylim(0, 11.5)
    a.set_ylabel("run / 14 (낮을수록 좋음)")
    a.set_title("(b) 접촉이 생긴 run 수")

    for a, key, kmin, title in [
        (ax[1, 0], "apple_bar_median_mm", "apple_bar_min_mm", "(c) 허들 통과 시 실제 사과 바닥 – 막대 윗면 (중앙, ▼ 최소)"),
        (ax[1, 1], "release_finger_bar_median_mm", None, "(d) 놓은 뒤 ±8 row 실제 손가락–허들 최소 거리 (중앙)")]:
        b = a.bar(range(3), d[key], 0.55, color=cols)
        for bb, v in zip(b, d[key]):
            a.text(bb.get_x() + 0.275, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=9)
        if kmin:
            for i, v in enumerate(d[kmin]):
                a.plot(i + 0.33, v, marker="v", color=fs.INK, ms=7)
                a.text(i + 0.4, v, f"min {v:+.1f}", va="center", fontsize=8)
        a.axhline(0, color=RED, lw=1)
        a.axhline(10, color=RED, lw=1, ls="--")
        a.text(-0.4, 10.4, "obstacle margin 10 mm", fontsize=8, color=RED)
        a.set_xticks(range(3))
        a.set_xticklabels(arms)
        a.set_ylabel("mm")
        a.set_ylim(-8, 16)
        a.set_title(title)

    fs.style_axes(fig, ax)
    fig.suptitle("결과 (closed loop, 같은 episode·seed 짝, 허들 14 run): V8 → V11A → V11AB", fontsize=13)
    fig.tight_layout()
    save(fig, "fig3_result_v8_v11ab", {"arms": arms, "values": d,
                                       "source": "T43V11ABC.verify.json via AG3S_T0T6_LOG.md T43 13",
                                       "note": "n=14; V8 vs V11AB CT paired p=0.219"})


# --------------------------------------------------------------------------------------------
# Figure 4 — TO formulation V8 vs V11AB (horizon schematic + objective + constraints)
# --------------------------------------------------------------------------------------------
def fig4():
    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.1], hspace=0.12)
    a = fig.add_subplot(gs[0])
    a.set_xlim(-3.2, 17)
    a.set_ylim(-0.4, 2.6)
    a.axis("off")

    def box(x, y, w, h, fc, text, ec=None, ls="-"):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc,
                                   ec=ec or fs.INK_2, lw=1, ls=ls))
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    # V8 row
    a.text(-3.1, 1.95, "V8", fontsize=12, va="center", weight="bold", color=ARM_COL["V8"])
    a.text(-3.1, 1.6, "H = 8", fontsize=9, va="center")
    for k in range(8):
        box(k, 1.65, 0.92, 0.6, "#cfe0f6", f"{k}")
    a.text(4, 2.42, "row 0–7: 모든 충돌 행 · 인증 · 실행", ha="center", fontsize=9)
    a.text(12, 1.95, "정책 chunk의 나머지 row는 보지 않음\n→ 허들이 8 row 안에 들어와야 비로소 반응",
           ha="center", va="center", fontsize=9, color=fs.INK_2)
    # V11AB row
    a.text(-3.1, 0.55, "V11AB", fontsize=12, va="center", weight="bold", color=ARM_COL["V11AB"])
    a.text(-3.1, 0.2, "H = 16", fontsize=9, va="center")
    for k in range(8):
        box(k, 0.25, 0.92, 0.6, "#cfe0f6", f"{k}")
    for k in range(8, 16):
        box(k, 0.25, 0.92, 0.6, "#fbe3c8", f"{k}", ls="--")
    a.text(4, -0.2, "row 0–7: 모든 충돌 행 · 인증 · 실행 (V8과 동일)", ha="center", fontsize=9)
    a.text(12, -0.2, "row 8–15 lookahead: 장애물 행만, 같은 slack 가격, 인증·실행 안 함\n"
                     "gate: 정책 꼬리가 장애물 20 mm 안일 때만 사용", ha="center", va="top", fontsize=9)
    a.set_title("실행 창은 같고(8 row), V11AB는 다음 8 row의 장애물을 미리 본다", fontsize=11)

    b = fig.add_subplot(gs[1])
    b.axis("off")
    rows = [
        ("", "V8 (기존 TO)", "V11AB (개선 TO)"),
        ("알고리즘", "SQP (ESDF 거리 선형화 → QP, OSQP)\nmin 3 iter · trust region 0.15 rad · eps 1e-3",
         "같은 SQP\nOSQP eps 1e-4 (16 step QP 정확도)"),
        ("결정 변수", r"$Q=[q_0..q_7]$, slack $s\geq0$", r"$Q=[q_0..q_{15}]$, slack $s\geq0$"),
        ("목적함수",
         r"$w_{tr}\sum_k\|q_k-q^{ref}_k\|^2 + w_{sm}\sum_k\|\Delta^2 q_k\|^2$" "\n"
         r"$+\,w_{co}\sum_k\|q_k-q^{prev}_k\|^2 + w_{sl}\sum s$",
         r"V8 항 그대로 (continuity는 row 0–7만)" "\n"
         r"$+\,w_{dr}\left(\sum_k\|D_{k+1}-D_k\|^2+\|D_0-D_{prev}\|^2\right)$" "\n"
         r"$D=Q-Q^{ref}$ (정책 대비 편향),  $D_{prev}$ = 직전 실행 편향"),
        ("가중치", r"$w_{tr}=1,\ w_{sm}=0.05,\ w_{co}=0.5,\ w_{sl}=10^3$",
         r"동일 $+\ w_{dr}=20$  (편향을 약 $\sqrt{w_{dr}/w_{tr}}\approx4.5$ row에 걸쳐 완만하게)"),
        ("충돌 제약",
         r"$d_{ESDF}(p_i(q_k))-r_i-m_i\geq -s_{ik}$,  k = 0–7" "\n" r"obstacle $m$ = 10 mm (쥔 사과 행 0)",
         r"k = 0–7: V8과 동일" "\n" r"k = 8–15: 장애물 행만 (target·받침면·쥔 사과 잔상 제외)"),
        ("기구 제약", r"joint box, 속도 $|q_{k+1}-q_k|\leq v_{max}\Delta t$ (soft), 가속도 (soft)," "\n"
         r"anchor $|q_0-q_{now}|\leq v_{max}\Delta t$, trust region", "동일"),
        ("인증 · 실행", "row 0–7 인증 → 실행 8 row", "row 0–7만 인증 (꼬리는 기록만) → 실행 8 row"),
    ]
    tbl = b.table(cellText=[list(r) for r in rows[1:]], colLabels=list(rows[0]), loc="center",
                  cellLoc="left", colWidths=[0.1, 0.42, 0.48])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(fs.GRID_INK)
        cell.set_height((0.17 if r == 3 else 0.12) if r else 0.06)
        if r == 0:
            cell.set_facecolor("#e9e8e3")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_facecolor("#f3f2ee")
            cell.set_text_props(weight="bold")
        elif c == 2 and r in (1, 3, 4, 5, 7):
            cell.set_facecolor("#e6f5ee")
    fig.patch.set_facecolor(fs.SURFACE)
    fig.suptitle("TO 비교: V8 vs V11AB (= V8 + A lookahead + B deflection rate)", fontsize=13, y=0.99)
    save(fig, "fig4_to_formulation_compare", {
        "V8_flags": "--obstacle-margin 0.010 --esdf-margin 0 --finger-cover both ... (plan_horizon = execution window 8)",
        "V11AB_added_flags": "--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20",
        "weights": {"w_track": 1.0, "w_smooth": 0.05, "w_continuity": 0.5, "w_slack": 1e3,
                    "w_deflection_rate_V11AB": 20},
        "sqp": {"min_iterations": 3, "trust_radius_rad": 0.15, "osqp_eps_V8": 1e-3, "osqp_eps_V11AB": 1e-4},
        "source": "AG3S/trajopt/problem.py, config.py, sqp.py, linearize.py; T43V8/T43V11ABC.verify.json server cmdline",
    })


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("korean", KO, "->", OUT)
