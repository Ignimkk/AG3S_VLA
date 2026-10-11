"""T1-a 2 차 — 재현성. 이미 디스크에 있는 롤아웃 json 을 프레임 단위로 대조한다.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1-a/repro_frames.py --out outputs/verify/T1-a/repro.json

새로 돌리지 않는다. 묻는 것 셋:
  1. 어느 프레임이 뒤집히나 — 매번 같은 프레임인가.
  2. 뒤집힌 프레임의 clearance_after_mm 가 0 에서 얼마나 가까운가
     (feasible/violated 는 0 기준 이진 딱지라 +-0.5 mm 가 판정을 가른다).
  3. 그 프레임들의 iterations / to_ms / ag3s_ms 가 실행마다 어떻게 다른가.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np

VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/R'))

#: 같은 명령(legacy · 실측 attention · --dump-frames 없음)인 것만이 '반복 실행'이다.
IDENTICAL = ["R2_legacy_attn", "R2_repeat_1", "R2_repeat_2", "R2_repeat_3"]
#: attention 만 synthetic 이고 나머지가 같다.
NEAR = ["R2_legacy_control"]
#: --dump-frames 가 붙은 다른 명령. 폭에 넣지 않고 따로 본다.
OTHER = ["R4_dump_base"]


def load(tag):
    return json.loads((VER / f"{tag}.json").read_text())["frames"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups = {"identical": IDENTICAL, "near_synthetic_attention": NEAR, "other_command": OTHER}
    runs = {t: load(t) for g in groups.values() for t in g}
    n = len(next(iter(runs.values())))

    per_run = {t: {"feasible": sum(1 for r in f if r["status"] == "feasible"),
                   "violated": sum(1 for r in f if r["status"] == "violated"),
                   "iter_median": statistics.median([r["iterations"] for r in f]),
                   "iter": [r["iterations"] for r in f],
                   "to_ms_sum": round(sum(r["to_ms"] for r in f), 1),
                   "ag3s_ms_sum": round(sum(r["ag3s_ms"] for r in f), 1)}
               for t, f in runs.items()}

    # --- 프레임별 대조 (같은 명령 4 회) -------------------------------------------------
    frames = []
    for i in range(n):
        st = {t: runs[t][i]["status"] for t in IDENTICAL}
        cl = {t: runs[t][i]["arm_after_mm"] for t in IDENTICAL}
        it = {t: runs[t][i]["iterations"] for t in IDENTICAL}
        vals = list(cl.values())
        flipped = len(set(st.values())) > 1
        frames.append({
            "frame": i, "t_step": runs[IDENTICAL[0]][i]["t_step"],
            "phase": runs[IDENTICAL[0]][i]["phase"],
            "flipped": flipped,
            "statuses": st,
            "arm_after_mm": {t: round(v, 3) for t, v in cl.items()},
            "arm_after_min_mm": round(min(vals), 3), "arm_after_max_mm": round(max(vals), 3),
            "arm_after_spread_mm": round(max(vals) - min(vals), 3),
            "abs_min_distance_to_zero_mm": round(min(abs(v) for v in vals), 3),
            "iterations": it,
            "to_ms": {t: round(runs[t][i]["to_ms"], 1) for t in IDENTICAL},
            "ag3s_ms": {t: round(runs[t][i]["ag3s_ms"], 1) for t in IDENTICAL},
            "arm_before_mm": round(runs[IDENTICAL[0]][i]["arm_before_mm"], 3),
        })

    flipped = [f for f in frames if f["flipped"]]
    stable = [f for f in frames if not f["flipped"]]

    out = {
        "_comment": "이미 있는 json 만 읽었다. 새로 돌린 것 없다.",
        "source_dir": str(VER), "n_frames": n,
        "groups": groups,
        "per_run": per_run,
        "identical_feasible": [per_run[t]["feasible"] for t in IDENTICAL],
        "identical_iter_median": [per_run[t]["iter_median"] for t in IDENTICAL],
        "n_flipped_frames": len(flipped),
        "flipped_frames": [f["frame"] for f in flipped],
        "flipped_detail": flipped,
        "stable_frames_margin_mm": {
            "min_abs_clearance": round(min(f["abs_min_distance_to_zero_mm"] for f in stable), 3),
            "median_abs_clearance": round(
                float(np.median([f["abs_min_distance_to_zero_mm"] for f in stable])), 3)},
        "frames": frames,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"같은 명령 {len(IDENTICAL)} 회 feasible: {out['identical_feasible']}")
    print(f"  반복 중앙값: {out['identical_iter_median']}")
    print(f"뒤집힌 프레임 {len(flipped)}/{n}: {out['flipped_frames']}")
    for f in flipped:
        print(f"  frame {f['frame']:2d} ({f['phase']:9}) before {f['arm_before_mm']:8.2f} mm  "
              f"after {f['arm_after_min_mm']:7.2f}..{f['arm_after_max_mm']:7.2f} "
              f"(폭 {f['arm_after_spread_mm']:.2f} mm, 0 까지 {f['abs_min_distance_to_zero_mm']:.2f} mm)")
        print(f"            iters {list(f['iterations'].values())}  "
              f"to_ms {list(f['to_ms'].values())}")
    print(f"\n뒤집히지 않은 프레임의 |clearance| 최소 "
          f"{out['stable_frames_margin_mm']['min_abs_clearance']} mm, "
          f"중앙 {out['stable_frames_margin_mm']['median_abs_clearance']} mm")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
