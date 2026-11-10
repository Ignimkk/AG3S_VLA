"""Baseline tranche report: per-block table + figures of every finished tranche run vs the original papers (dev-4a, 2026-11-10).

Collects ``closed_loop_metrics.json`` of ``bl<T>-<arm>-<suite>-L<lvl>[-rN]`` (SafeLIBERO) and ``bl<T>-<arm>-<suite>[-rN]``
(LIBERO) runs. Per block the highest -rN supersedes (same rule as ``bl_tranche_check.py``); superseded runs are listed
as 'replaced' so the before/after of every port fix stays visible. Writes into ``outputs/2026-11-09_baseline_tranche/<T>/``:
  README.md (tables), blocks.csv, fig1_pooled_vs_paper.png, fig2_car_blocks.png
usage: .venv-ag3s/bin/python experiments/segmentation/bl_tranche_report.py [--tag t1]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import re

ROOT = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
OUT_BASE = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "2026-11-09_baseline_tranche"
ARMS = ["none", "aegis", "aegis-gt", "multilink_cbf", "multilink_cbf-gt", "knows", "knows-gt"]
LABEL = {"none": "π0.5", "aegis": "AEGIS", "aegis-gt": "AEGIS-GT", "multilink_cbf": "multi-link CBF",
         "multilink_cbf-gt": "multi-link CBF-GT", "knows": "KNOWS", "knows-gt": "KNOWS-GT"}
SUITES = ["spatial", "object", "goal", "long"]
LIBERO_SUITES = ["spatial", "object", "goal", "10"]
LEVELS = ["I", "II"]
# paper references, levels pooled (TSR, SSR, CAR); KNOWS-paper rows = mean of its LI / LII values (bl_tranche_check.py)
PAPER = {"none": (56.4, 15.75, 17.7), "knows": (69.3, 56.85, 72.6), "knows-gt": (70.4, 59.35, 79.15),
         "aegis": (67.5, None, 68.9), "aegis-gt": (67.5, None, 68.9),
         "multilink_cbf": (60.3, 51.7, 73.0), "multilink_cbf-gt": (60.3, 51.7, 73.0)}
SAFE_RE = re.compile(r"^(\d{8}T\d{6})_([0-9a-f]+)_bl(?P<t>t\d+)-(?P<arm>[a-z_]+(?:-gt)?)-(?P<suite>spatial|object|goal|long)"
                     r"-L(?P<lvl>I{1,2})(?P<r>-r\d)?-")
LIB_RE = re.compile(r"^(\d{8}T\d{6})_([0-9a-f]+)_bl(?P<t>t\d+)-(?P<arm>[a-z_]+)-(?P<suite>spatial|object|goal|10)(?P<r>-r\d)?-"
                    r"(?!LI)")


def rank(r):
    return int(r[2:]) if r else 0


def collect(tag):
    safe, lib = {}, {}
    for d in sorted(ROOT.iterdir()):
        if d.name.startswith("INVALID") or not (d / "closed_loop_metrics.json").exists():
            continue
        m = SAFE_RE.match(d.name)
        if m and m["t"] == tag and m["arm"] in ARMS:
            j = json.loads((d / "closed_loop_metrics.json").read_text())["safelibero_official"]
            rec = {"run": d.name, "commit": d.name.split("_")[1][:8], "rev": m["r"] or "", "n": j["n"],
                   "succ": j["n_success"], "safe": j["n_safe_success"], "coll": j["n_collision"]}
            safe.setdefault((m["arm"], m["suite"], m["lvl"]), []).append(rec)
            continue
        m = LIB_RE.match(d.name)
        if m and m["t"] == tag and m["arm"] in ARMS and "-LI-" not in d.name and "-LII-" not in d.name:
            j = json.loads((d / "closed_loop_metrics.json").read_text())["libero"]
            lib.setdefault((m["arm"], m["suite"]), []).append(
                {"run": d.name, "commit": d.name.split("_")[1][:8], "rev": m["r"] or "", "n": j["n"], "SR": 100 * j["SR"]})
    for dct in (safe, lib):
        for k, v in dct.items():
            v.sort(key=lambda r: (rank(r["rev"]), r["run"]))  # last = current
    return safe, lib


def pct(a, n):
    return 100.0 * a / n if n else float("nan")


def pooled(safe, arm):
    cur = [v[-1] for (a, _, _), v in safe.items() if a == arm]
    n = sum(r["n"] for r in cur)
    if not n:
        return None
    return {"n": n, "blocks": len(cur), "TSR": pct(sum(r["succ"] for r in cur), n),
            "SSR": pct(sum(r["safe"] for r in cur), n), "CAR": pct(n - sum(r["coll"] for r in cur), n)}


def fig_pooled(safe, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ARMS if pooled(safe, a)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (mi, met) in zip(axes, enumerate(["TSR", "SSR", "CAR"])):
        for i, a in enumerate(arms):
            p = pooled(safe, a)
            ax.bar(i, p[met], width=0.62, color="#2a78d6", edgecolor="white", linewidth=2)
            ax.text(i, p[met] + 1.5, f"{p[met]:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
            ref = PAPER[a][mi]
            if ref is not None:
                ax.plot([i - 0.38, i + 0.38], [ref, ref], color="#eb6834", linewidth=2.5,
                        label="paper" if i == 0 or (met == "SSR" and a == "multilink_cbf") else None)
        ax.set_xticks(range(len(arms)), [f"{LABEL[a]}\n({pooled(safe, a)['n']} ep)" for a in arms], fontsize=7.5, rotation=30,
                      ha="right")
        ax.set_title(met, fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#ddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("% (levels + finished suites pooled)")
    h = [plt.Rectangle((0, 0), 1, 1, color="#2a78d6"), plt.Line2D([0], [0], color="#eb6834", linewidth=2.5)]
    fig.legend(h, ["measured (tranche, current port version)", "paper reference"], loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Baseline tranche 1 vs original papers (SafeLIBERO, init 15–18)", fontsize=11, x=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_blocks(safe, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    cols = [(s, l) for s in SUITES for l in LEVELS]
    M = np.full((len(ARMS), len(cols)), np.nan)
    for i, a in enumerate(ARMS):
        for j, (s, l) in enumerate(cols):
            v = safe.get((a, s, l))
            if v:
                M[i, j] = pct(v[-1]["n"] - v[-1]["coll"], v[-1]["n"])
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad("#f2f2f2")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = "–" if np.isnan(M[i, j]) else f"{M[i, j]:.0f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=8.5,
                    color="white" if not np.isnan(M[i, j]) and M[i, j] > 55 else "#222")
    ax.set_xticks(range(len(cols)), [f"{s}\nL{l}" for s, l in cols], fontsize=8)
    ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS], fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ARMS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("CAR % (16 ep per block)")
    ax.set_title("CAR per block, current port version ('–' = not run yet)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t1")
    a = ap.parse_args()
    out = OUT_BASE / a.tag
    out.mkdir(parents=True, exist_ok=True)
    safe, lib = collect(a.tag)
    with open(out / "blocks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "suite", "level", "rev", "commit", "status", "n", "TSR", "SSR", "CAR", "run"])
        for (arm, s, l), v in sorted(safe.items(), key=lambda kv: (ARMS.index(kv[0][0]), SUITES.index(kv[0][1]), kv[0][2])):
            for r in v:
                w.writerow([arm, s, l, r["rev"] or "-", r["commit"], "current" if r is v[-1] else "replaced", r["n"],
                            f"{pct(r['succ'], r['n']):.1f}", f"{pct(r['safe'], r['n']):.1f}",
                            f"{pct(r['n'] - r['coll'], r['n']):.1f}", r["run"]])
        for (arm, s), v in sorted(lib.items()):
            for r in v:
                w.writerow([arm, "libero_" + s, "-", r["rev"] or "-", r["commit"], "current" if r is v[-1] else "replaced",
                            r["n"], f"{r['SR']:.1f}", "", "", r["run"]])
    fig_pooled(safe, out / "fig1_pooled_vs_paper.png")
    fig_blocks(safe, out / "fig2_car_blocks.png")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# Baseline tranche {a.tag} — 결과 수집 (자동 생성 {now} KST, `experiments/segmentation/bl_tranche_report.py`)", "",
         "근거 · 판정: `doc/active/experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md` §4.6–4.14. "
         "SafeLIBERO init 15–18 × task 0–3 = 16 ep / block, seed 7, `main_aegis` step budget, 1 mm 충돌 규칙. "
         "block 마다 가장 높은 `-rN` (현재 port 판) 만 pool 에 들어간다.", "",
         "## 1. arm 별 pooled (현재 판) vs 논문", "",
         "| arm | block | n | TSR | SSR | CAR | 논문 TSR / SSR / CAR | CAR 차이 (pt) |", "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        p = pooled(safe, arm)
        if not p:
            L.append(f"| {LABEL[arm]} | 0 / 8 | 0 | – | – | – | {PAPER[arm]} | – |")
            continue
        ref = PAPER[arm]
        refs = " / ".join("–" if x is None else f"{x:.1f}" for x in ref)
        L.append(f"| {LABEL[arm]} | {p['blocks']} / 8 | {p['n']} | {p['TSR']:.1f} | {p['SSR']:.1f} | {p['CAR']:.1f} | {refs} | "
                 f"{p['CAR'] - ref[2]:+.1f} |")
    L += ["", "논문 값: π0.5 · KNOWS · KNOWS-GT = KNOWS 논문 Tab. 1 의 LI / LII 평균, AEGIS = VLSA Tab. I, multi-link = 그 논문 "
          "Table Ib (static SafeLIBERO). 우리 pool 은 끝난 suite 만 포함하므로 suite 구성이 논문 (4 suite) 과 다를 수 있다.", "",
          "![fig1](fig1_pooled_vs_paper.png)", "", "![fig2](fig2_car_blocks.png)", "",
          "## 2. LIBERO (장애물 없음, 성공률)", "", "| arm | " + " | ".join(LIBERO_SUITES) + " |",
          "|---|" + "---|" * len(LIBERO_SUITES)]
    for arm in ARMS:
        if any((arm, s) in lib for s in LIBERO_SUITES):
            L.append(f"| {LABEL[arm]} | " + " | ".join(f"{lib[(arm, s)][-1]['SR']:.1f} ({lib[(arm, s)][-1]['n']})"
                                                       if (arm, s) in lib else "–" for s in LIBERO_SUITES) + " |")
    L += ["", "## 3. 수정 전 → 수정 후 (같은 block)", "", "| arm | block | 수정 전 (rev, CAR) | 현재 (rev, CAR) |", "|---|---|---|---|"]
    for (arm, s, l), v in sorted(safe.items(), key=lambda kv: (ARMS.index(kv[0][0]), SUITES.index(kv[0][1]), kv[0][2])):
        if len(v) > 1:
            old = ", ".join(f"{r['rev'] or 'r1'} {pct(r['n'] - r['coll'], r['n']):.1f}" for r in v[:-1])
            c = v[-1]
            L.append(f"| {LABEL[arm]} | {s} L{l} | {old} | {c['rev'] or 'r1'} {pct(c['n'] - c['coll'], c['n']):.1f} |")
    L += ["", "전체 run 목록 (현재 / replaced): `blocks.csv`. episode 영상: 각 run 의 `videos/`.", ""]
    (out / "README.md").write_text("\n".join(L))
    print(out / "README.md")


if __name__ == "__main__":
    main()
