"""T7a s3 — 갈라지는 chunk 를 숫자로 고정한다. 새로 재지 않고 T6d 의 곡선을 다시 쓴다.

`outputs/verify/T6d/m3_curves.npz` — 손끝(ee_finger_l1/l2) 구에서 **사과 표면**까지의 거리
(exec window 8 스텝 최소). shadow 는 `reference`(실행된 것), closed-loop 는 `actions`(실행된 것).
같은 chunk 축에 T7a 의 min margin_slack 을 나란히 놓는다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np

T6 = np.load(_ag3s_workspace_path('outputs/verify/T6d/m3_curves.npz'))
S2 = json.load(open(_ag3s_workspace_path('outputs/verify/T7a/s2_joint_axis.json')))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s3_divergence_context.json'))
EXEC_H = 8

tip_sh = T6["shadow_tip_reference_mm"][:, :EXEC_H].min(axis=1)
tip_shr = T6["shadow_tip_refined_mm"][:, :EXEC_H].min(axis=1)
tip_cl = T6["closed_loop_approach_tip_mm"][:, :EXEC_H].min(axis=1)
sl_sh = np.asarray(S2["runs"]["shadow_approach"]["reference__no_target__left_spheres"]
                   ["chunk_min_slack_mm"], np.float64)
sl_cl = np.asarray(S2["runs"]["closed_loop_approach"]["refined__no_target__left_spheres"]
                   ["chunk_min_slack_mm"], np.float64)

out = {
    "source_curves": _ag3s_workspace_path('outputs/verify/T6d/m3_curves.npz'),
    "exec_window_steps": EXEC_H,
    "metric": "min over exec window of (fingertip sphere surface distance to apple) [mm]",
    "shadow_reference_tip_mm": [round(float(v), 4) for v in tip_sh],
    "shadow_refined_tip_mm": [round(float(v), 4) for v in tip_shr],
    "closed_loop_refined_tip_mm": [round(float(v), 4) for v in tip_cl],
    "first_chunk_shadow_tip_below_0mm": int(np.argmax(tip_sh <= 0.0)) if (tip_sh <= 0).any() else None,
    "n_chunks_shadow_tip_below_0mm": int((tip_sh <= 0).sum()),
    "n_chunks_closed_loop_tip_below_0mm": int((tip_cl <= 0).sum()),
    "closed_loop_tip_min_mm": round(float(tip_cl.min()), 4),
    "closed_loop_tip_min_chunk": int(tip_cl.argmin()),
    "shadow_tip_min_mm": round(float(tip_sh.min()), 4),
    "shadow_refined_tip_min_mm": round(float(tip_shr.min()), 4),
    "chunk_of_max_gap_tip_mm": int(np.argmax(tip_cl - tip_sh)),
    "max_gap_tip_mm": round(float((tip_cl - tip_sh).max()), 4),
    "min_slack_shadow_reference_mm": [round(float(v), 4) for v in sl_sh],
    "min_slack_closed_loop_refined_mm": [round(float(v), 4) for v in sl_cl],
}
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k, v in out.items():
    if not isinstance(v, list):
        print(f"  {k} = {v}")
print("wrote", OUT)
