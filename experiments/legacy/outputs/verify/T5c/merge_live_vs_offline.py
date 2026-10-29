"""T5c — live shadow 판정과 같은 t_step 의 참값 기하를 묶는다.

shadow 기록에는 qpos·물체 자세가 없다. 같은 에피소드(ep1807)의 다른 실행인
`outputs/live_test/20260925_ep1807/run_0000` (T2 가 쓴 기록)의 참값 기하를 t_step 으로
맞춰 붙인다 — **두 실행은 다른 rollout 이다.** 오른팔(비작업 팔) 관절로만 정렬을 확인했고
그 값은 `rollout_alignment_right_arm.json` 에 있다.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/merge_live_vs_offline.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

import numpy as np

SH = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/shadow_agg.json'))
OFF = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/offline_grounding_ep1807.json'))
ALIGN = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/rollout_alignment_right_arm.json'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/live_vs_offline_by_chunk.json'))
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def summ(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "p50": xs[len(xs) // 2], "max": xs[-1],
            "mean": float(np.mean(xs))}


def main() -> None:
    sh = json.loads(SH.read_text())
    off = {f["t_step"]: f for f in json.loads(OFF.read_text())["frames"]}
    align = json.loads(ALIGN.read_text())

    rows = []
    for r in sh["per_chunk"]:
        o = off[r["t_step"]]
        rows.append({
            "seq": r["seq"], "t_step": r["t_step"],
            "live_ag3s_status": r["ag3s_status"],
            "live_grounding_status": r["grounding_status"],
            "live_has_target": r["has_target"],
            "live_n_field_tiers": r["n_tiers"],
            "offline_has_target": o["has_target"],
            "offline_target_score": o["target_score"],
            "offline_grounding_status": o["grounding_status"],
            "phase_offline": o["phase"],
            "ee_apple_dist_mm": o["ee_apple_dist_mm"],
            "apple_pixels": o["apple_pixels"],
            "apple_z_m": o["apple_z_m"],
            "apple_crate_xy_dist_mm": o["apple_crate_xy_dist_mm"],
        })

    ht = [r for r in rows if r["live_has_target"]]
    nt = [r for r in rows if not r["live_has_target"]]
    agree = sum(1 for r in rows if r["live_has_target"] == r["offline_has_target"])
    out = {
        "note": "live 판정은 shadow 기록, 기하와 offline 점수는 run_0000(다른 rollout) 재생.",
        "sources": {"live": str(SH), "offline_and_geometry": str(OFF),
                    "alignment": str(ALIGN)},
        "rollout_alignment_right_arm_rad": {k: align[k] for k in ("n_compared", "min", "p50", "max")},
        "n_chunks": len(rows),
        "live_has_target": len(ht), "live_no_target_or_degraded_without_target": len(nt),
        "offline_has_target": sum(1 for r in rows if r["offline_has_target"]),
        "agreement_live_vs_offline_has_target": {
            "same": agree, "different": len(rows) - agree,
            "live_yes_offline_no": [r["seq"] for r in rows
                                    if r["live_has_target"] and not r["offline_has_target"]],
            "live_no_offline_yes": [r["seq"] for r in rows
                                    if not r["live_has_target"] and r["offline_has_target"]],
        },
        "by_live_label": {
            "offline_target_score": {"has_target": summ([r["offline_target_score"] for r in ht]),
                                     "no_target": summ([r["offline_target_score"] for r in nt])},
            "ee_apple_dist_mm": {"has_target": summ([r["ee_apple_dist_mm"] for r in ht]),
                                 "no_target": summ([r["ee_apple_dist_mm"] for r in nt])},
            "apple_z_m": {"has_target": summ([r["apple_z_m"] for r in ht]),
                          "no_target": summ([r["apple_z_m"] for r in nt])},
            "apple_pixels_mean": {c: {"has_target": float(np.mean([r["apple_pixels"][c] for r in ht])),
                                      "no_target": float(np.mean([r["apple_pixels"][c] for r in nt]))}
                                  for c in CAMS},
            "apple_pixels_zero_frames": {c: {"has_target": sum(1 for r in ht if r["apple_pixels"][c] == 0),
                                             "no_target": sum(1 for r in nt if r["apple_pixels"][c] == 0)}
                                         for c in CAMS},
            "phase_counts": {p: {"has_target": sum(1 for r in ht if r["phase_offline"] == p),
                                 "no_target": sum(1 for r in nt if r["phase_offline"] == p)}
                             for p in sorted({r["phase_offline"] for r in rows})},
            "n_field_tiers": {"has_target": sorted({r["live_n_field_tiers"] for r in ht}),
                              "no_target": sorted({r["live_n_field_tiers"] for r in nt})},
        },
        "per_chunk": rows,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "per_chunk"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
