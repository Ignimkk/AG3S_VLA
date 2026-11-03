#!/usr/bin/env python3
"""Combine real-ATS replay rows with the stored external-baseline rows of the same 552 frames (exact key check)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

METRICS = ('iou', 'recall', 'precision', 'neighbor_fp')
BASE = {'oracle_image': ('sam2_image_instant', 'oracle_image'),
        'grounded_sam2_frame_phrase': ('sam2_image_instant', 'grounded_sam2_frame_phrase'),
        'grounded_sam2_frame_instruction': ('sam2_image_instant', 'grounded_sam2_frame_instruction'),
        'yoloe025': ('yoloe_image_instant', 'yoloe_phrase'),
        'yoloe005_posthoc': ('yoloe_conf005_image_instant', 'yoloe_phrase')}


def key(r):
    return r['episode_id'], int(r['obs_index'])


def collect(root, real):
    real_rows = {key(r): r for r in map(json.loads, (real/'frames.jsonl').open())}
    rows = {'ATS_real_target_cluster': real_rows,
            'ATS_real_erased_extent': {k: {**r, 'metrics': r['extent_metrics']} for k, r in real_rows.items()}}
    for name, (folder, arm) in BASE.items():
        rows[name] = {key(r): r for r in map(json.loads, (root/folder/'frames.jsonl').open()) if r['arm'] == arm}
    adapter = {}
    for shard in sorted(root.glob('factorial_shard[0-9]')):
        for r in map(json.loads, (shard/'frames.jsonl').open()):
            adapter[key(r)] = {**r, 'metrics': r['arms']['noun__fk_spheres']['metrics']}
    rows['adapter_noun_L8H6_fk'] = adapter
    return rows


def aggregate(rows, tasks, keys):
    out = {}
    for m in METRICS + ('identity', 'abstain'):
        per = defaultdict(lambda: defaultdict(list))
        for k in keys:
            r = rows[k]
            x = r['metrics']
            if m == 'identity':
                v = float(x['precision'] is not None and x['precision'] >= 0.5)
            elif m == 'abstain':
                v = float(x['pred_voxels'] == 0)
            else:
                v = x[m]
            if v is not None:
                per[tasks[k[0]]][k[0]].append(float(v))
        task = {t: float(np.mean([np.mean(v) for v in eps.values()])) for t, eps in per.items()}
        out[m] = {'mean': float(np.mean(list(task.values()))), 'task': task}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    expected = {key(r) for r in map(json.loads, (a.root/'all_frames.jsonl').open())}
    tasks = {k[0]: json.loads(next(l for l in (a.root/'all_frames.jsonl').open() if k[0] in l))['task_id'] for k in expected}
    rows = collect(a.root, a.real)
    for name, r in rows.items():
        if set(r) != expected:
            raise ValueError(f'{name}: frame keys differ from the fixed 552 ({len(r)} rows)')
    keys = sorted(expected)
    result = {n: aggregate(r, tasks, keys) for n, r in rows.items()}
    rng = np.random.default_rng(20261102)
    idx = rng.integers(0, 10, (10000, 10))
    contrasts = []
    for mine, other in [(m, o) for m in ('ATS_real_erased_extent', 'ATS_real_target_cluster') for o in result
                        if not o.startswith('ATS_real')]:
        ats = np.array([result[mine]['iou']['task'][t] for t in range(10)])
        o = np.array([result[other]['iou']['task'][t] for t in range(10)])
        d = ats-o
        contrasts.append({'ats': mine, 'baseline': other, 'delta_iou': float(d.mean()),
                          'ci95_exploratory': np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist(),
                          'tasks_ats_lower': int((d < 0).sum()), 'task_delta': d.tolist()})
    ats_rows = rows['ATS_real_target_cluster']
    status = defaultdict(int)
    for r in ats_rows.values():
        status[r['status']] += 1
    summary = {'frames': len(keys), 'tasks': 10, 'aggregation': 'frame -> episode -> task -> equal 10 task mean',
               'identity_rule': 'precision >= 0.5 of the predicted voxel set (fixed before results)',
               'results': result, 'contrasts': contrasts, 'ats_grounding_status_counts': dict(status),
               'ats_mean_wall_s_excluding_first_per_episode': float(np.mean(
                   [r['wall_s'] for k, r in ats_rows.items() if k[1] != min(x[1] for x in keys if x[0] == k[0])])),
               'note': 'exploratory; 10 tasks x 2 inits; wall time measured with a shared GPU, not a latency claim'}
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    lines = ['| 방법 | 3D IoU | recall | precision | 이웃 FP | 대상 선택 | 무출력 |', '|---|---:|---:|---:|---:|---:|---:|']
    for n, r in sorted(result.items(), key=lambda kv: -kv[1]['iou']['mean']):
        lines.append(f"| {n} | {r['iou']['mean']:.3f} | {r['recall']['mean']:.3f} | {r['precision']['mean']:.3f} | "
                     f"{r['neighbor_fp']['mean']:.1f} | {r['identity']['mean']:.3f} | {r['abstain']['mean']:.3f} |")
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
