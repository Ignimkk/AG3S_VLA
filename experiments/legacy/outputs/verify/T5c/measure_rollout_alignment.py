"""T5c — shadow 실행과 run_0000(T2 기록)이 얼마나 같은 궤적인가, 오른팔로만.

클라이언트 로그는 청크마다 오른팔(비작업 팔) 7 관절만 소수 둘째 자리로 찍는다
(`pi05_infer.py:1593-1595`, obs_format=rby1_16d 는 else 분기). 16D state 의 8..14 가
오른팔이므로(`pi05_infer.py:421-427`) 같은 t_step 끼리 비교한다. 왼팔(작업 팔)은 shadow
기록에 없다.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/measure_rollout_alignment.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import glob
import json
import pathlib
import re

import numpy as np

LOG = pathlib.Path(_ag3s_workspace_path('outputs/live_test/logs/t5shadow_run.log'))
RECORDS = _ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/rollout_alignment_right_arm.json'))


def main() -> None:
    live = {}
    pat = re.compile(r"\[t=\s*(\d+)\] infer=([\d.]+)s chunk=\(50, 16\)\s+q_r=\[([^\]]+)\]")
    for line in LOG.read_text().splitlines():
        m = pat.search(line)
        if m:
            live[int(m.group(1))] = [float(x) for x in m.group(3).split(",")]

    rows = []
    for f in sorted(glob.glob(f"{RECORDS}/step_*.npz")):
        d = np.load(f, allow_pickle=True)
        t = int(d["t_step"])
        if t in live:
            off = np.round(np.asarray(d["state"], float)[8:15], 2)
            rows.append((t, float(np.max(np.abs(off - np.array(live[t]))))))
    rows.sort()
    diffs = [r[1] for r in rows]
    out = {"n_compared": len(rows), "unit": "rad",
           "max_abs_diff_right_arm_per_chunk": rows,
           "min": min(diffs), "p50": float(np.median(diffs)), "max": max(diffs),
           "note": "클라이언트 로그가 0.01 rad 로 반올림해 찍으므로 분해능은 0.01 rad 다"}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, out["n_compared"], out["min"], out["p50"], out["max"])


if __name__ == "__main__":
    main()
