"""T6d 재는 것 3 (b) — 실행 창(앞 8 스텝)과 그리퍼 이동량.

control 기록의 `step_in_chunk` 는 세 실행 모두 0..7 뿐이다 (600 control step / 75 chunk).
그래서 50 스텝 chunk 중 **앞 8 스텝만** 실행됐다. `m3_curves.npz` 를 그 창으로 다시 요약한다.
그리퍼는 action col 7(왼쪽) 과 qpos[27](gripper_finger_l1) 의 범위를 본다.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T6d/m3b_executed_window.py
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d'))
z = np.load(V / 'm3_curves.npz')
H = 8
out = {}


def summ(a, name):
    full = np.nanmin(a, axis=1); win = np.nanmin(a[:, :H], axis=1)
    print(f"{name:34s} full50 best {np.nanmin(full):8.2f}  exec{H} best {np.nanmin(win):8.2f}  "
          f"exec{H} median {np.nanmedian(win):8.2f}")
    return {'per_chunk_full50_mm': [float(v) for v in full],
            'per_chunk_exec8_mm': [float(v) for v in win],
            'best_full50_mm': float(np.nanmin(full)), 'best_exec8_mm': float(np.nanmin(win))}


out['shadow_tip_reference'] = summ(z['shadow_tip_reference_mm'], 'shadow tip reference')
out['shadow_tip_refined'] = summ(z['shadow_tip_refined_mm'], 'shadow tip refined')
out['closed_loop_approach_tip'] = summ(z['closed_loop_approach_tip_mm'], 'closed_loop approach tip refined')
out['closed_loop_grasp_tip'] = summ(z['closed_loop_grasp_tip_mm'], 'closed_loop grasp tip refined')
out['shadow_clear_reference'] = summ(z['shadow_clear_reference_mm'], 'shadow clearance reference')
out['shadow_clear_refined'] = summ(z['shadow_clear_refined_mm'], 'shadow clearance refined')

a = np.nanmin(z['shadow_tip_refined_mm'][:, :H], 1) - np.nanmin(z['shadow_tip_reference_mm'][:, :H], 1)
c = np.nanmin(z['shadow_clear_refined_mm'][:, :H], 1) - np.nanmin(z['shadow_clear_reference_mm'][:, :H], 1)
print(f"exec{H} tip delta: worse(+) in {int((a>1e-6).sum())} chunks, max +{a.max():.3f} mm, min {a.min():.3f} mm")
print(f"exec{H} clearance delta: worse(-) in {int((c<-1e-6).sum())} chunks, worst {c.min():.3f} mm")
out['exec8_delta'] = {'tip_delta_mm': [float(v) for v in a], 'clear_delta_mm': [float(v) for v in c],
                      'n_tip_worse': int((a > 1e-6).sum()), 'max_tip_push_mm': float(a.max()),
                      'n_clear_worse': int((c < -1e-6).sum()), 'worst_clear_delta_mm': float(c.min())}

for run in ['local_t6_from_pc', 'local_shadow_from_pc', 'local_grasp_from_pc']:
    act, ref, qg = [], [], []
    with open(f'{_ag3s_workspace_path()}/outputs/live_test/{run}/frames.jsonl') as f:
        for l in f:
            d = json.loads(l)
            if d.get('kind') == 'planning':
                act.append(np.asarray(d['actions'])[:, 7])
                if d.get('actions_reference') is not None:
                    ref.append(np.asarray(d['actions_reference'])[:, 7])
            elif d.get('kind') == 'control':
                qg.append(d['qpos'][27])
    A, Q = np.asarray(act), np.asarray(qg)
    r = {'action_col7_min': float(A.min()), 'action_col7_max': float(A.max()),
         'qpos_gripper_l1_min': float(Q.min()), 'qpos_gripper_l1_max': float(Q.max())}
    if ref:
        R = np.asarray(ref); r['reference_col7_min'] = float(R.min()); r['reference_col7_max'] = float(R.max())
    out.setdefault('gripper', {})[run] = r
    print(run, json.dumps(r))
(V / 'm3b_executed_window.json').write_text(json.dumps(out, indent=1))
print('wrote m3b_executed_window.json')
