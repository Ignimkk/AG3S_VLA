"""T1 항목 1 재측정 figure — grounding 점수 대 target 중심 오차.

`outputs/verify/T1/item1_score_ep*.json` 을 읽는다. 해석하지 않는다 — 산점도 하나만 낸다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/make_item1_score_figure.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AG3S.ag3s.experiments.common import figstyle

EPISODES = [("ep1800", "apple"), ("ep1850", "banana"), ("ep1900", "pear")]
OUT_DIR = pathlib.Path("AG3S/ag3s/docs/figures/t1")
THRESHOLD = 0.25


def main() -> None:
    figstyle.use_korean()
    data = {ep: json.loads(pathlib.Path(f"outputs/verify/T1/item1_score_{ep}.json").read_text())["frames"]
           for ep, _t in EPISODES}

    fig, ax = plt.subplots(figsize=(8, 6))
    for ci, (ep, tgt) in enumerate(EPISODES):
        frames = data[ep]
        scores = np.array([f["target_score"] for f in frames])
        err = np.array([f["would_be_target_center_error_mm_from_best_cluster"] for f in frames])
        passed = np.array([f["actual_target_center_error_mm"] is not None for f in frames])
        c = figstyle.CATEGORICAL[ci]
        ax.scatter(scores[passed], err[passed], c=c, marker="o", s=45,
                  label=f"{ep} ({tgt}) — grounding 통과 (n={passed.sum()})")
        ax.scatter(scores[~passed], err[~passed], c=c, marker="x", s=90, linewidths=2.5,
                  label=f"{ep} ({tgt}) — LOW_SCORE 로 거름, best cluster 기준 (n={(~passed).sum()})")
    ax.axvline(THRESHOLD, color=figstyle.INK_2, ls="--", lw=1,
              label=f"target_score_threshold = {THRESHOLD}")
    ax.set_xlabel("grounding target_score (best_score, LOW_SCORE 프레임은 문턱 미만 최고점)")
    ax.set_ylabel("target 중심 오차 (mm) — 통과 프레임은 실제값, 거른 프레임은 best cluster 기준")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("T1 항목 1 재측정 — grounding 점수 대 target 중심 오차 (세 에피소드)")
    ax.legend(fontsize=8, loc="upper right")
    figstyle.style_axes(fig, [ax])
    fig.tight_layout()

    out_png = OUT_DIR / "t1-item1-score-vs-error.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    numbers = {}
    for ep, tgt in EPISODES:
        frames = data[ep]
        scores = [f["target_score"] for f in frames]
        wouldbe = [f["would_be_target_center_error_mm_from_best_cluster"] for f in frames]
        passed = [f for f in frames if f["actual_target_center_error_mm"] is not None]
        failed = [f for f in frames if f["actual_target_center_error_mm"] is None]
        numbers[ep] = {
            "target": tgt,
            "r_score_vs_wouldbe_error_all15": float(np.corrcoef(scores, wouldbe)[0, 1]),
            "passed": {
                "n": len(passed),
                "score_min": min(f["target_score"] for f in passed) if passed else None,
                "score_max": max(f["target_score"] for f in passed) if passed else None,
                "score_mean": float(np.mean([f["target_score"] for f in passed])) if passed else None,
                "score_median": float(np.median([f["target_score"] for f in passed])) if passed else None,
                "r_score_vs_actual_error": (
                    float(np.corrcoef([f["target_score"] for f in passed],
                                     [f["actual_target_center_error_mm"] for f in passed])[0, 1])
                    if len(passed) >= 2 else None),
            },
            "failed": {
                "n": len(failed),
                "score_min": min(f["target_score"] for f in failed) if failed else None,
                "score_max": max(f["target_score"] for f in failed) if failed else None,
                "would_be_error_mm_min": (min(f["would_be_target_center_error_mm_from_best_cluster"]
                                              for f in failed) if failed else None),
                "would_be_error_mm_max": (max(f["would_be_target_center_error_mm_from_best_cluster"]
                                              for f in failed) if failed else None),
                "would_be_error_mm_mean": (float(np.mean(
                    [f["would_be_target_center_error_mm_from_best_cluster"] for f in failed]))
                    if failed else None),
                "would_be_error_mm_median": (float(np.median(
                    [f["would_be_target_center_error_mm_from_best_cluster"] for f in failed]))
                    if failed else None),
            },
        }

    # 같은 점수대에서 에피소드별 오차 — score 0.1 폭 구간
    bins = np.arange(0.0, 1.01, 0.1)
    cross = {}
    for ep, tgt in EPISODES:
        frames = data[ep]
        scores = np.array([f["target_score"] for f in frames])
        wouldbe = np.array([f["would_be_target_center_error_mm_from_best_cluster"] for f in frames])
        per_bin = {}
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (scores >= b0) & (scores < b1)
            if m.any():
                per_bin[f"{b0:.1f}-{b1:.1f}"] = {
                    "n": int(m.sum()), "mean_wouldbe_error_mm": float(np.mean(wouldbe[m]))}
        cross[ep] = per_bin
    numbers["cross_episode_error_by_score_bin"] = cross

    (OUT_DIR / "t1-item1-score-vs-error.json").write_text(
        json.dumps(numbers, indent=1, ensure_ascii=False))
    print(f"wrote {out_png}")
    print(json.dumps(numbers, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
