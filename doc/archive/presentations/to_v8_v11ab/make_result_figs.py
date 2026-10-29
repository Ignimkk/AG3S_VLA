"""Slide-size before/after figures from the PPT_TO_V11AB verifier sidecars (compare_figs/*.json).

No new numbers: everything is read from compare_figs/c_timeline.json and c_deflection_profile.json
(verify: results/development/20261014T071304+0900_cf38c9dbca7b_c2a4f0214bb3/verify/PPT_TO_V11AB.verify.json).
Run from the repo root:  .venv-ag3s/bin/python doc/archive/presentations/to_v8_v11ab/make_result_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "AG3S/ag3s/experiments"))
from common import figstyle as fs  # noqa: E402

fs.use_korean()
plt.rcParams.update({"font.size": 17, "axes.labelsize": 17, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "legend.fontsize": 15})
C = fs.CATEGORICAL
RED = C[7]
BG = "#fbfaf7"
ARM = {"V8": C[0], "V11A": C[1], "V11AB": C[2]}
SRC = HERE / "compare_figs"
OUT = HERE / "slide_figs"


def style(fig, axes):
    fs.style_axes(fig, axes)
    fig.patch.set_facecolor(BG)
    for a in axes:
        a.set_facecolor(BG)
        a.tick_params(labelsize=15)


RAW = Path("/mnt/dev/AG3S_VLA-storage/results/development/20261014T071304+0900_cf38c9dbca7b_c2a4f0214bb3"
           "/verify/PPT_TO_V11AB/raw")
KEY = "1807_s18073"


def cross_rows(arm):
    """Every control row of ep1807 s18073 from the verifier's XM crossing output (same values as the HX timeline,
    without the HX 100 mm window, so no row is dropped)."""
    d = json.loads((RAW / f"cross_{arm}.json").read_text())
    run = next(v for k, v in d["runs"].items() if KEY in k)
    return run["rows"]


def timeline():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    used = {}
    for a, arm in zip(axes, ["V8", "V11AB"]):
        rows = [r for r in cross_rows(arm) if 150 <= r["t"] <= 184]
        t = [r["t"] for r in rows]
        P = [r["geoH_C"]["apple"] for r in rows]
        M = [r["geoH_M"]["apple"] for r in rows]
        hold = [not r["executed"] for r in rows]
        ht = [x for x, h in zip(t, hold) if h]
        if ht:
            a.axvspan(min(ht) - 0.5, max(ht) + 0.5, color=fs.GRID_INK, alpha=0.45, lw=0)
            a.text((min(ht) + max(ht)) / 2, 150, "HOLD", ha="center", fontsize=13, color=fs.INK_2)
        a.plot(t, P, "-o", ms=4, lw=2, color=C[1], label="계획 = 명령")
        a.plot(t, M, "-o", ms=4, lw=2.8, color=C[0], label="실제")
        ex = [i for i, h in enumerate(hold) if not h]
        mp, mm = min(P[i] for i in ex), min(M[i] for i in ex)
        a.axhline(0, color=RED, lw=1.2)
        a.axhline(10, color=RED, lw=1.2, ls="--")
        a.set_title(f"{arm}: 실제 최소 {mm:.1f} mm", fontsize=18, color=ARM[arm] if arm != "V11AB" else "#12805C")
        a.set_xlabel("control row")
        a.set_xlim(149.5, 184.5)
        a.set_ylim(-8, 165)
        used[arm] = {"t": t, "P_geo_mm": P, "M_geo_mm": M, "hold": hold,
                     "min_exec_P_mm": mp, "min_exec_M_mm": mm}
    axes[0].set_ylabel("사과 – 허들 거리 (mm)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].text(184, 12, "margin 10 mm", fontsize=13, color=RED, ha="right")
    style(fig, axes)
    fig.tight_layout()
    fig.savefig(OUT / "r_timeline.png", dpi=150, facecolor=BG)
    (OUT / "r_timeline.json").write_text(json.dumps(
        {"source": str(RAW / "cross_{V8,V11AB}.json"), "field": "geoH_C/geoH_M apple (mj_geomDistance to hurdle)",
         "episode": "ep1807 s18073", "arms": used}, ensure_ascii=False, indent=1))
    plt.close(fig)


def path():
    """Side view of the apple across the bar: s = horizontal distance across the bar, height = apple bottom - bar top."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    used = {}
    spans = {"V8": (160, 180), "V11AB": (152, 176)}
    marks = {"V8": [169, 171, 173, 175], "V11AB": [158, 162, 166, 169, 172]}
    for a, arm in zip(axes, ["V8", "V11AB"]):
        lo, hi = spans[arm]
        rows = [r for r in cross_rows(arm) if lo <= r["t"] <= hi]
        sC = [r["s_C"]["apple"] for r in rows]; sM = [r["s_M"]["apple"] for r in rows]
        zC = [r["apple_bottom_minus_bar_top_C"] for r in rows]; zM = [r["apple_bottom_minus_bar_top_M"] for r in rows]
        a.add_patch(plt.Circle((0, -12), 12, color=RED, alpha=0.85, zorder=1))
        a.text(0, -34, "허들 막대", ha="center", fontsize=14, color=RED)
        a.plot(sC, zC, "-o", ms=4, lw=2, color=C[1], label="명령 (계획)", zorder=3)
        a.plot(sM, zM, "-o", ms=4, lw=2.8, color=C[0], label="실제", zorder=3)
        for r, x1, y1, x2, y2 in zip(rows, sC, zC, sM, zM):
            if r["t"] in marks[arm]:
                a.plot([x1, x2], [y1, y2], ls=":", lw=1.4, color=fs.INK_2, zorder=2)
                a.text(x1, y1 + 5, str(r["t"]), fontsize=13, color=C[1], ha="center")
                a.text(x2, y2 - 11, str(r["t"]), fontsize=13, color=C[0], ha="center")
        a.axhline(0, color=RED, lw=1, ls="--")
        a.set_xlim(-160, 130)
        a.set_ylim(-60, 105)
        a.set_xlabel("허들을 가로지르는 수평 위치 (mm, + = crate 쪽)")
        a.set_title(arm, fontsize=18, color=ARM[arm] if arm != "V11AB" else "#12805C")
        used[arm] = {"t": [r["t"] for r in rows], "s_C": sC, "s_M": sM, "bottom_C": zC, "bottom_M": zM}
    axes[0].set_ylabel("사과 바닥 – 막대 윗면 (mm)")
    axes[0].legend(frameon=False, loc="upper left")
    style(fig, axes)
    fig.tight_layout()
    fig.savefig(OUT / "r_path.png", dpi=150, facecolor=BG)
    (OUT / "r_path.json").write_text(json.dumps(
        {"source": str(RAW / "cross_{V8,V11AB}.json"), "episode": "ep1807 s18073",
         "note": "same row number = same time; dotted line joins command and actual of that row", "arms": used},
        ensure_ascii=False, indent=1))
    plt.close(fig)


def profile(name, arms, size):
    pr = json.loads((SRC / "c_deflection_profile.json").read_text())["arms"]
    fig, a = plt.subplots(figsize=size)
    for arm in arms:
        v = pr[arm]["palm_D_profile_by_k_median_mm"]
        a.plot(range(8), v, "-o", ms=7, lw=3, color=ARM[arm], label=arm)
        a.text(7.15, v[-1], f"{v[-1]:.1f}", va="center", fontsize=15, color=fs.INK)
        a.text(-0.15, v[0], f"{v[0]:.1f}", va="center", ha="right", fontsize=15, color=fs.INK)
    a.set_xlim(-0.9, 7.8)
    a.set_ylim(0, 20)
    a.set_xticks(range(8))
    a.set_xlabel("chunk 안 row k (0 = 새 chunk 첫 row)")
    a.set_ylabel("TO 편향 |D|, 중앙 (mm)")
    a.legend(frameon=False, loc="lower right")
    style(fig, [a])
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150, facecolor=BG)
    (OUT / f"{name}.json").write_text(json.dumps(
        {"source": str((SRC / "c_deflection_profile.json").relative_to(ROOT)),
         "rule": "carry_apple chunks with max_k |D| > 5 mm, hurdle 14 runs",
         "arms": {k: pr[k] for k in arms}}, ensure_ascii=False, indent=1))
    plt.close(fig)


if __name__ == "__main__":
    timeline()
    path()
    profile("r_profile_v8", ["V8"], (7.2, 4.4))
    profile("r_profile_all", ["V8", "V11A", "V11AB"], (8, 5.8))
    print("ok")
