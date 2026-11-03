#!/usr/bin/env python3
"""Check and summarize the complete paired Object development factorial."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    expected = {(r['episode_id'], int(r['obs_index'])) for r in map(json.loads, a.frames.open())}
    rows = {}
    for shard in a.shards:
        manifest = json.loads((shard/'manifest.json').read_text())
        if manifest['frames_sha256'] != sha(a.frames): raise ValueError('frame list hash mismatch')
        for row in map(json.loads, (shard/'frames.jsonl').open()):
            key = row['episode_id'], row['obs_index']
            if key in rows: raise ValueError(f'duplicate frame {key}')
            rows[key] = row
    if set(rows) != expected or len(rows) != 281:
        raise ValueError(f'frame mismatch: got {len(rows)}, expected {len(expected)}')
    arms = ['noun__none', 'noun__fk_spheres', 'noun__gt_image_privileged',
            'gt_patch_privileged__none', 'gt_patch_privileged__fk_spheres',
            'gt_patch_privileged__gt_image_privileged']
    if any(set(r['arms']) != set(arms) for r in rows.values()): raise ValueError('missing arm')
    previous = {(r['episode_id'], r['obs_index']): r for r in map(json.loads, a.previous.open())
                if r['arm'] == 'surface_no_robot_mask'}
    if set(previous) != expected: raise ValueError('previous frame set differs')
    for key, row in rows.items():
        for m in ('iou', 'recall', 'neighbor_fp'):
            old, new = previous[key]['metrics'][m], row['arms']['noun__none']['metrics'][m]
            if old != new: raise ValueError(f'previous baseline mismatch {key} {m}: {old} != {new}')
    task_ids = sorted({r['task_id'] for r in rows.values()})
    if task_ids != list(range(10)): raise ValueError('expected tasks 0 through 9')
    def task_means(arm, field):
        d = defaultdict(lambda: defaultdict(list))
        for r in rows.values():
            v = r['arms'][arm]['metrics'][field]
            if v is not None: d[r['task_id']][r['episode_id']].append(v)
        return np.array([np.mean([np.mean(vals) for vals in d[t].values()]) for t in task_ids])
    picks = np.random.default_rng(20261101).integers(0, 10, (10000, 10))
    summaries = []
    for arm in arms:
        iou = task_means(arm, 'iou')
        item = {'arm': arm, 'iou': float(iou.mean()), 'recall': float(task_means(arm, 'recall').mean()),
                'neighbor_fp': float(task_means(arm, 'neighbor_fp').mean()), 'task_iou': iou.tolist(),
                'abstain_count': sum(not r['arms'][arm]['applied'] for r in rows.values()),
                'abstain_reasons': dict(Counter(r['arms'][arm]['why'] for r in rows.values()
                                                if not r['arms'][arm]['applied']))}
        summaries.append(item)
    pairs = [('noun__fk_spheres', 'noun__none'), ('noun__gt_image_privileged', 'noun__none'),
             ('gt_patch_privileged__fk_spheres', 'gt_patch_privileged__none'),
             ('gt_patch_privileged__gt_image_privileged', 'gt_patch_privileged__none'),
             ('gt_patch_privileged__none', 'noun__none'),
             ('gt_patch_privileged__fk_spheres', 'noun__fk_spheres'),
             ('gt_patch_privileged__gt_image_privileged', 'noun__gt_image_privileged')]
    contrasts = []
    for left, right in pairs:
        diff = task_means(left, 'iou') - task_means(right, 'iou')
        contrasts.append({'left': left, 'right': right, 'mean': float(diff.mean()),
                          'ci95': np.quantile(diff[picks].mean(axis=1), [.025,.975]).tolist(),
                          'task_differences': diff.tolist()})
    robot = {}
    for c in ('head', 'left_wrist'):
        vals = [r['robot_stats'][c] for r in rows.values()]
        counts = {k: sum(v[k] for v in vals) for k in ('tp','fp','fn','gt_robot_px','target_px','target_deleted_px')}
        by_task = {}
        for t in task_ids:
            rs = [r['robot_stats'][c] for r in rows.values() if r['task_id'] == t]
            tp, fp, fn = (sum(r[k] for r in rs) for k in ('tp','fp','fn'))
            target, deleted = (sum(r[k] for r in rs) for k in ('target_px','target_deleted_px'))
            by_task[str(t)] = {'precision': tp/(tp+fp) if tp+fp else None,
                               'recall': tp/(tp+fn) if tp+fn else None,
                               'target_deleted_fraction': deleted/target if target else None}
        robot[c] = {'pixel_micro_counts': counts,
                    'pixel_micro_precision': counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
                    'pixel_micro_recall': counts['tp']/counts['gt_robot_px'] if counts['gt_robot_px'] else None,
                    'pixel_micro_target_deleted': counts['target_deleted_px']/counts['target_px'] if counts['target_px'] else None,
                    'per_task_pooled_pixels': by_task}
    a.out.mkdir(parents=True, exist_ok=False)
    result = {'frames': len(rows), 'episodes': len({k[0] for k in rows}), 'tasks': task_ids,
              'all_previous_noun_none_match': len(rows), 'summary': summaries, 'contrasts': contrasts,
              'robot_mask': robot, 'bootstrap': 'paired 10 task clusters, seed 20261101, 10000',
              'shard_inputs_sha256': {str(s): sha(s/'frames.jsonl') for s in a.shards},
              'script_sha256': sha(Path(__file__))}
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['| 단서 | robot mask | IoU | recall | 이웃 FP | abstain |', '|---|---|---:|---:|---:|---:|']
    for x in summaries:
        seed, robot_name = x['arm'].split('__')
        lines.append(f"| {seed} | {robot_name} | {x['iou']:.3f} | {x['recall']:.3f} | {x['neighbor_fp']:.1f} | {x['abstain_count']}/281 |")
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('validated and summarized',len(rows),'matched frames')


if __name__ == '__main__': main()
