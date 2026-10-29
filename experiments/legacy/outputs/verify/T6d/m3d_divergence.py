"""T6d 재는 것 3 (d) — 세 궤적이 **언제부터** 갈라지는가, 그리고 파지 전후로 나눈 값.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T6d/m3d_divergence.py
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d'))
z = np.load(V / 'm3_curves.npz'); H = 8
ref = np.nanmin(z['shadow_tip_reference_mm'][:, :H], 1)
ref_f = np.nanmin(z['shadow_tip_refined_mm'][:, :H], 1)
cla = np.nanmin(z['closed_loop_approach_tip_mm'][:, :H], 1)
clg = np.nanmin(z['closed_loop_grasp_tip_mm'][:, :H], 1)
out = {}


def first_exceed(a, b, th):
    d = np.abs(a - b)
    return int(np.argmax(d > th) + 1) if (d > th).any() else None


for name, v in (('closed_loop_approach', cla), ('closed_loop_grasp', clg), ('shadow_refined', ref_f)):
    out[f'{name}_vs_shadow_reference'] = {
        'first_chunk_diff_gt_20mm': first_exceed(v, ref, 20.0),
        'first_chunk_diff_gt_50mm': first_exceed(v, ref, 50.0),
        'first_chunk_diff_gt_100mm': first_exceed(v, ref, 100.0),
        'max_abs_diff_mm': float(np.abs(v - ref).max()), 'min_value_mm': float(v.min())}
touch = int(np.argmax(ref < 0) + 1)
out['reference_first_chunk_tip_below_apple_surface'] = touch
d = ref_f - ref
out['pre_touch'] = {'chunks': f'1..{touch}',
                    'n_refined_further_from_apple': int((d[:touch] > 1e-6).sum()),
                    'n_refined_closer': int((d[:touch] < -1e-6).sum()),
                    'max_push_mm': float(d[:touch].max()), 'max_pull_mm': float(d[:touch].min())}
out['post_touch'] = {'chunks': f'{touch+1}..75',
                     'n_refined_further_from_apple': int((d[touch:] > 1e-6).sum()),
                     'max_push_mm': float(d[touch:].max())}
out['whole_run'] = {'n_refined_further_from_apple': int((d > 1e-6).sum()),
                    'max_push_mm': float(d.max()), 'max_pull_mm': float(d.min())}
out['executed_window_best_tip_mm'] = {'shadow_reference': float(ref.min()), 'shadow_refined': float(ref_f.min()),
                                      'closed_loop_approach': float(cla.min()), 'closed_loop_grasp': float(clg.min())}
print(json.dumps(out, indent=1))
(V / 'm3d_divergence.json').write_text(json.dumps(out, indent=1))
