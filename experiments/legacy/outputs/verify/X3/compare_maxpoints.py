"""X3 측정 6 — voxel 이 6.3 mm 로 거칠어진 것이 clearance 를 얼마나 바꾸나.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/X3/compare_maxpoints.py

두 실행은 `pointcloud.max_points` 하나만 다르다 (60000 기본 · 200000). 그래서 차이를
voxel 해상도에 돌릴 수 있다. 시간도 같이 본다 — `max_points` 는 성능 설정이 아니라
실시간 성립 조건이라고 `config.py:127` 이 적기 때문이다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import re
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AG3S.ag3s.experiments.common import figstyle

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OK, BAD = C[2], C[7]
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/x3'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/X3'))

BASE = V / "probe_headfix_v2.json"          # max_points = 60000 (기본)
RAISED = V / "probe_headfix_maxpts200k.json"  # max_points = 200000


def T(ko: str, en: str) -> str:
    return ko if KO else en


def voxel_mm(notes: list[str]) -> float:
    for n in notes:
        g = re.search(r"voxel grown to ([\d.]+) mm", n)
        if g:
            return float(g.group(1))
    return 5.0  # 상한에 안 걸렸으면 config 기본값 voxel_size = 0.005


def load(path: pathlib.Path, rollout: pathlib.Path):
    d = json.loads(path.read_text())
    rows = json.loads(rollout.read_text())["frames"]
    for m, r in zip(d["frames"], rows):
        m["ag3s_ms"] = r["ag3s_ms"]
        m["to_ms"] = r["to_ms"]
        m["voxel_mm"] = voxel_mm(m["ag3s_notes"])
    return d


def main() -> None:
    base = load(BASE, V / "probe_headfix_v2_rollout.json")
    rais = load(RAISED, V / "probe_headfix_maxpts200k_rollout.json")

    rows = []
    for b, r in zip(base["frames"], rais["frames"]):
        rows.append({
            "i": b["i"], "phase": b["phase"],
            "voxel_mm": {"cap_60k": b["voxel_mm"], "cap_200k": r["voxel_mm"]},
            "validity": {"cap_60k": b["validity"], "cap_200k": r["validity"]},
            "status": {"cap_60k": b["status"], "cap_200k": r["status"]},
            "clearance_mm": {"cap_60k": b["clearance_after_mm_rollout"],
                             "cap_200k": r["clearance_after_mm_rollout"]},
            "clearance_delta_mm": r["clearance_after_mm_rollout"] - b["clearance_after_mm_rollout"],
            "ag3s_ms": {"cap_60k": b["ag3s_ms"], "cap_200k": r["ag3s_ms"]},
        })

    d_clear = [x["clearance_delta_mm"] for x in rows]
    summary = {
        "n_frames": len(rows),
        "feasible": {"cap_60k": sum(1 for x in rows if x["status"]["cap_60k"] == "feasible"),
                     "cap_200k": sum(1 for x in rows if x["status"]["cap_200k"] == "feasible")},
        "violated": {"cap_60k": sum(1 for x in rows if x["status"]["cap_60k"] == "violated"),
                     "cap_200k": sum(1 for x in rows if x["status"]["cap_200k"] == "violated")},
        "validity_counts": {
            "cap_60k": base["summary"]["validity_counts"],
            "cap_200k": rais["summary"]["validity_counts"]},
        "voxel_mm_used": {
            "cap_60k": sorted({x["voxel_mm"]["cap_60k"] for x in rows}),
            "cap_200k": sorted({x["voxel_mm"]["cap_200k"] for x in rows})},
        "clearance_delta_mm": {
            "median": float(np.median(d_clear)), "min": float(min(d_clear)),
            "max": float(max(d_clear)), "max_abs": float(max(abs(v) for v in d_clear))},
        "clearance_min_mm": {"cap_60k": min(x["clearance_mm"]["cap_60k"] for x in rows),
                             "cap_200k": min(x["clearance_mm"]["cap_200k"] for x in rows)},
        "ag3s_ms_median": {
            "cap_60k": float(np.median([x["ag3s_ms"]["cap_60k"] for x in rows])),
            "cap_200k": float(np.median([x["ag3s_ms"]["cap_200k"] for x in rows]))},
        "n_frames_clearance_got_worse_when_finer": sum(1 for v in d_clear if v < 0),
    }

    (V / "compare_maxpoints.json").write_text(json.dumps(
        {"base": str(BASE), "raised": str(RAISED), "summary": summary, "frames": rows},
        indent=1, ensure_ascii=False))

    # ---- figure (그래프 + 표) -------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    x = [r["i"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    figstyle.style_axes(fig, axes)

    ax = axes[0]
    ax.axhline(0, color=figstyle.INK, lw=1.2)
    ax.plot(x, [r["clearance_mm"]["cap_60k"] for r in rows], "-o", color=C[3], ms=5,
            label=T("max_points 60k (voxel 6.3 mm)", "max_points 60k (voxel 6.3 mm)"))
    ax.plot(x, [r["clearance_mm"]["cap_200k"] for r in rows], "-o", color=C[0], ms=5,
            label=T("max_points 200k (voxel 5.0 mm)", "max_points 200k (voxel 5.0 mm)"))
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel("clearance_after (mm)")
    ax.set_title(T("해상도를 되돌리면 clearance 가 달라지나",
                   "does finer resolution move clearance"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.axhline(0, color=figstyle.INK, lw=1.2)
    ax.bar(x, d_clear, color=[OK if v >= 0 else BAD for v in d_clear], width=0.7)
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("차 (mm)", "delta (mm)"))
    ax.set_title(T("5.0 mm − 6.3 mm", "5.0 mm - 6.3 mm"), fontsize=10)

    ax = axes[2]
    w = 0.38
    ax.bar(np.array(x) - w / 2, [r["ag3s_ms"]["cap_60k"] for r in rows], w, color=C[3],
           label="60k")
    ax.bar(np.array(x) + w / 2, [r["ag3s_ms"]["cap_200k"] for r in rows], w, color=C[0],
           label="200k")
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("AG3S 실행 (ms)", "AG3S runtime (ms)"))
    ax.set_title(T("상한을 올린 값", "the price of the raised cap"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "x3-maxpoints.png", dpi=150, bbox_inches="tight",
                facecolor=figstyle.SURFACE)
    plt.close(fig)

    (OUT / "x3-maxpoints.json").write_text(json.dumps(
        {"figure": "x3-maxpoints.png", "sources": [str(BASE), str(RAISED)],
         "numbers": {"summary": summary, "frames": rows}}, indent=1, ensure_ascii=False))

    print("wrote", OUT / "x3-maxpoints.png")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
