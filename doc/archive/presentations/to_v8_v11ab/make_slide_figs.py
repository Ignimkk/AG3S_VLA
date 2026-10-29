"""Single-chart, large-font figures for the V8 -> V11AB slide deck.

Same sources as make_figures.py (recorded numbers only); failed variants (V9, V10, C) left out.
Run from the repo root:  .venv-ag3s/bin/python doc/archive/presentations/to_v8_v11ab/make_slide_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "AG3S/ag3s/experiments"))
from common import figstyle as fs  # noqa: E402

OUT = Path(__file__).resolve().parent / "slide_figs"
OUT.mkdir(exist_ok=True)
fs.use_korean()
plt.rcParams.update({"font.size": 17, "axes.labelsize": 17, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "legend.fontsize": 15})
C = fs.CATEGORICAL
RED = C[7]
ARM = {"V8": C[0], "V11A": C[1], "V11AB": C[2]}
BG = "#fbfaf7"


def style(fig, ax):
    fs.style_axes(fig, ax)
    fig.patch.set_facecolor(BG)
    for a in (ax if isinstance(ax, (list, tuple)) else [ax]):
        a.set_facecolor(BG)
        a.tick_params(labelsize=16)


def save(fig, name, data):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150, facecolor=BG)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    plt.close(fig)


def bar_labels(a, bars, fmt, dy, size=16):
    for b in bars:
        v = b.get_height()
        a.text(b.get_x() + b.get_width() / 2, v + (dy if v >= 0 else -dy), fmt.format(v),
               ha="center", va="bottom" if v >= 0 else "top", fontsize=size, color=fs.INK)


def example():
    hx = json.loads((ROOT / "AG3S/ag3s/docs/figures/t43/t43hx-timeline.json").read_text())
    p = next(q for q in hx["panels"] if q["key"] == "V8/H/V8_ep1807_s18073"
             and q["phase"] == "carry" and q["group"] == "apple")
    t, P, M, hold = p["t"], p["P_geo_mm"], p["M_geo_mm"], p["hold"]
    fig, a = plt.subplots(figsize=(11, 6.2))
    ht = [x for x, h in zip(t, hold) if h]
    a.axvspan(min(ht) - 0.5, max(ht) + 0.5, color=fs.GRID_INK, alpha=0.45, lw=0)
    a.text((min(ht) + max(ht)) / 2, 108, "HOLD", ha="center", fontsize=14, color=fs.INK_2)
    a.plot(t, P, "-o", ms=5, lw=2.2, color=C[1], label="계획 = 명령 (TO 출력)")
    a.plot(t, M, "-o", ms=5, lw=3, color=C[0], label="실제 (MuJoCo 측정)")
    a.axhline(0, color=RED, lw=1.2)
    i = t.index(173)
    a.annotate(f"계획 {P[i]:.1f} mm\n실제 {M[i]:.1f} mm", xy=(173, M[i]), xytext=(175.5, 55),
               fontsize=17, arrowprops=dict(arrowstyle="->", color=fs.INK, lw=1.4))
    a.set_xlabel("control row (1 row = 0.066 s)")
    a.set_ylabel("사과 – 허들 거리 (mm)")
    a.set_ylim(-6, 118)
    a.legend(frameon=False, loc="upper right")
    style(fig, a)
    save(fig, "s_example", {"key": p["key"], "t": t, "P_geo_mm": P, "M_geo_mm": M, "hold": hold,
                            "source": "AG3S/ag3s/docs/figures/t43/t43hx-timeline.json"})


def gap():
    fig, a = plt.subplots(figsize=(8, 6.2))
    names = ["계획 vs 명령\n(V8)", "명령 vs 실제\n(V8, TO 있음)", "명령 vs 실제\n(E0, TO 없음)"]
    med = [0.0, 15.1, 20.0]
    p90 = [0.0, 31.0, 31.1]
    b = a.bar(range(3), med, 0.55, color=[C[1], C[0], fs.INK_2])
    a.scatter(range(3), p90, marker="_", s=900, lw=3, color=fs.INK, zorder=3, label="p90")
    for x, m, q in zip(range(3), med, p90):
        a.text(x, m + 0.6, f"{m:.1f}", ha="center", fontsize=18, weight="bold")
        if q:
            a.text(x + 0.33, q, f"p90 {q:.1f}", va="center", fontsize=14, color=fs.INK_2)
    a.set_xticks(range(3))
    a.set_xticklabels(names)
    a.set_ylabel("EE 위치 차이 (mm), 운반 중 중앙값")
    a.set_ylim(0, 36)
    style(fig, a)
    save(fig, "s_gap", {"median_mm": med, "p90_mm": p90, "lag_rows": {"V8": 1.90, "E0": 1.95},
                        "source": "T43TK.verify.json via AG3S_T0T6_LOG.md T43 8-1"})


def causes():
    fig, a = plt.subplots(figsize=(8, 6.2))
    labels = ["계획은 clear,\n실제만 진입 (servo lag)", "TO가 위반을\n풀지 못함", "ESDF 거리 오차",
              "margin 0 행\n(쥔 사과)"]
    vals = [232, 0, 0, 2]
    y = list(range(len(vals)))[::-1]
    bars = a.barh(y, vals, 0.6, color=[C[0], fs.GRID_INK, fs.GRID_INK, C[1]])
    for b, v in zip(bars, vals):
        a.text(v + 4, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=18, weight="bold")
    a.set_yticks(y)
    a.set_yticklabels(labels, fontsize=15)
    a.set_xlim(0, 270)
    a.set_xlabel("순간 수 (허들 14 run × 3 arm, 합 234)")
    style(fig, a)
    save(fig, "s_causes", {"labels": labels, "values": vals,
                           "source": "T43HX.verify.json via AG3S_T0T6_LOG.md T43 8-3"})


def height():
    fig, a = plt.subplots(figsize=(11, 6.2))
    arms = ["V5", "V7g", "V8"]
    cmd, act = [21.4, 18.0, 19.6], [3.1, 2.0, 3.8]
    xs = range(3)
    b1 = a.bar([x - 0.2 for x in xs], cmd, 0.4, color=C[1], label="명령 (TO가 보낸 높이)")
    b2 = a.bar([x + 0.2 for x in xs], act, 0.4, color=C[0], label="실제")
    bar_labels(a, b1, "{:.1f}", 0.4, 17)
    bar_labels(a, b2, "{:.1f}", 0.4, 17)
    a.axhline(10, color=RED, lw=1.4, ls="--")
    a.text(-0.55, 10.5, "obstacle margin 10 mm", fontsize=14, color=RED)
    a.set_xticks(list(xs))
    a.set_xticklabels(arms)
    a.set_ylabel("허들 위 사과 바닥 높이, 중앙 (mm)")
    a.set_ylim(0, 27)
    a.legend(frameon=False, loc="upper right")
    style(fig, a)
    save(fig, "s_height", {"arms": arms, "command_mm": cmd, "actual_mm": act,
                           "source": "T43XM.verify.json via AG3S_T0T6_LOG.md T43 11"})


def deflection():
    xm = json.loads((ROOT / "AG3S/ag3s/docs/handoff/T43XM.verify.json").read_text())
    prof = xm["numbers"]["arms"]["V8"]["carry_apple"]["palm_D_profile_by_k_median_mm"]
    fig, a = plt.subplots(figsize=(11, 6.2))
    for c in range(3):
        xs = [c * 8 + k for k in range(8)]
        a.plot(xs, prof, "-o", ms=6, lw=2.6, color=C[0])
        if c:
            a.plot([c * 8 - 1, c * 8], [prof[-1], prof[0]], ls=":", lw=2.2, color=C[0])
        a.axvline(c * 8 - 0.5, color=fs.GRID_INK, lw=1.2)
        a.text(c * 8 + 3.5, 17.6, f"chunk {c + 1}", ha="center", fontsize=15, color=fs.INK_2)
    a.annotate("다음 chunk row 0에서\n다시 정책 쪽으로", xy=(8, prof[0]), xytext=(9.3, 2.0), fontsize=16,
               color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    a.text(0, prof[0] - 1.6, f"{prof[0]:.1f}", fontsize=15, ha="center")
    a.text(7, prof[-1] + 0.6, f"{prof[-1]:.1f}", fontsize=15, ha="center")
    a.set_ylim(0, 19)
    a.set_xlabel("실행 row (chunk 당 8 row, 같은 중앙값 profile을 반복 표시)")
    a.set_ylabel("TO 편향 |D| (mm)")
    style(fig, a)
    save(fig, "s_deflection", {"V8_carry_apple_palm_D_profile_by_k_median_mm": prof,
                               "note": "one median profile repeated over 3 chunks for illustration",
                               "source": "AG3S/ag3s/docs/handoff/T43XM.verify.json"})


def margin_vs_lag():
    fig, a = plt.subplots(figsize=(8, 6.2))
    b = a.bar(["TO가 margin 위로\n남긴 여유", "servo lag로\n잃는 높이"], [4.6, 15.8], 0.5, color=[C[1], RED])
    bar_labels(a, b, "{:.1f} mm", 0.3, 19)
    a.set_ylabel("mm (V8)")
    a.set_ylim(0, 19)
    style(fig, a)
    save(fig, "s_margin_vs_lag", {"TO_binding_clearance_above_margin_median_mm": 4.6,
                                  "command_minus_actual_height_mm": 15.8,
                                  "source": "T43XM via AG3S_T0T6_LOG.md T43 11"})


def offline():
    fig, a = plt.subplots(figsize=(11, 6.2))
    var = ["V8 TO", "A", "A + B"]
    p10 = [-0.56, -0.49, 2.94]
    clear = [9, 9, 15]
    b = a.bar(range(3), p10, 0.5, color=[C[0], C[1], C[2]])
    bar_labels(a, b, "{:+.2f} mm", 0.12, 18)
    a.axhline(0, color=RED, lw=1.2)
    a.set_xticks(range(3))
    a.set_xticklabels([f"{n}\n접촉 22건 중 {c}건 미리 회피" for n, c in zip(var, clear)])
    a.set_ylabel("예측 허들 clearance p10 (mm)")
    a.set_ylim(-1.5, 3.8)
    style(fig, a)
    save(fig, "s_offline", {"variants": var, "servo_pred_hurdle_clearance_p10_mm": p10,
                            "contacts_cleared_of_22": clear,
                            "source": "T43 TA offline (implementer, open loop), AG3S_T0T6_LOG.md T43 12-1"})


def results():
    arms = ["V8", "V11A", "V11AB"]
    cols = [ARM[x] for x in arms]
    # success
    fig, a = plt.subplots(figsize=(8, 6.2))
    b = a.bar(range(3), [8, 8, 12], 0.55, color=cols)
    bar_labels(a, b, "{:g} / 14", 0.2, 19)
    a.set_xticks(range(3))
    a.set_xticklabels(arms)
    a.set_ylabel("접촉 없는 성공 (run)")
    a.set_ylim(0, 14.5)
    style(fig, a)
    save(fig, "s_res_success", {"arms": arms, "success_ct": [8, 8, 12], "n": 14,
                                "source": "T43V11ABC.verify.json via AG3S_T0T6_LOG.md T43 13"})
    # contacts
    fig, a = plt.subplots(figsize=(11, 6.2))
    keys = ["운반 중\n사과–허들 접촉", "놓은 뒤\n손가락–허들 접촉", "obstacle stop"]
    vals = {"V8": [5, 9, 10], "V11A": [2, 4, 6], "V11AB": [0, 0, 1]}
    w = 0.26
    for i, arm in enumerate(arms):
        bb = a.bar([j + (i - 1) * w for j in range(3)], vals[arm], w, color=ARM[arm], label=arm)
        bar_labels(a, bb, "{:g}", 0.15, 16)
    a.set_xticks(range(3))
    a.set_xticklabels(keys)
    a.set_ylabel("run / 14 (낮을수록 좋음)")
    a.set_ylim(0, 11.5)
    a.legend(frameon=False, ncol=3, loc="upper left")
    style(fig, a)
    save(fig, "s_res_contacts", {"arms": arms, "values": vals,
                                 "source": "T43V11ABC.verify.json via AG3S_T0T6_LOG.md T43 13"})
    # clearance
    fig, a = plt.subplots(figsize=(11, 6.2))
    med, mn = [3.8, 6.7, 13.6], [-6.4, -0.1, 7.1]
    b = a.bar(range(3), med, 0.5, color=cols, label="중앙")
    for i, (m, v) in enumerate(zip(med, mn)):
        a.text(i, m + 0.3, f"{m:+.1f}", ha="center", va="bottom", fontsize=18, weight="bold")
        a.plot(i + 0.32, v, marker="v", ms=11, color=fs.INK)
        a.text(i + 0.4, v, f"최소 {v:+.1f}", va="center", fontsize=15)
    a.axhline(0, color=RED, lw=1.2)
    a.axhline(10, color=RED, lw=1.4, ls="--")
    a.text(-0.4, 10.4, "obstacle margin 10 mm", fontsize=14, color=RED)
    a.set_xticks(range(3))
    a.set_xticklabels(arms)
    a.set_ylabel("허들 위 실제 사과 바닥 높이 (mm)")
    a.set_ylim(-8, 16.5)
    style(fig, a)
    save(fig, "s_res_clearance", {"arms": arms, "median_mm": med, "min_mm": mn,
                                  "source": "T43V11ABC.verify.json via AG3S_T0T6_LOG.md T43 13"})


def scene_crops():
    src = ROOT / "AG3S/ag3s/docs/figures/t43/t43v11ab-crossing-scene.png"
    im = Image.open(src)
    w, h = im.size  # 2400 x 1200 (top row: three video frames)
    sx, sy = w / 2000, h / 1000
    boxes = {"s_scene_v8": (86, 62, 633, 471), "s_scene_v11a": (749, 62, 1296, 471),
             "s_scene_v11ab": (1411, 62, 1958, 471)}
    for name, (l, t, r, b) in boxes.items():
        im.crop((int(l * sx), int(t * sy), int(r * sx), int(b * sy))).save(OUT / f"{name}.png")
    # bottom row: palm trajectories
    traj = {"s_traj_v8": (0, 500, 666, 1000), "s_traj_v11a": (666, 500, 1333, 1000),
            "s_traj_v11ab": (1333, 500, 2000, 1000)}
    for name, (l, t, r, b) in traj.items():
        im.crop((int(l * sx), int(t * sy), int(r * sx), int(b * sy))).save(OUT / f"{name}.png")
    (OUT / "s_scene_source.json").write_text(json.dumps(
        {"source": str(src.relative_to(ROOT)), "pair": "ep1807 s18073", "crops": boxes | traj},
        indent=1))


if __name__ == "__main__":
    example(); gap(); causes(); height(); deflection(); margin_vs_lag(); offline(); results()
    scene_crops()
    print(sorted(p.name for p in OUT.glob("*.png")))
