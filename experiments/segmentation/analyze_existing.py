#!/usr/bin/env python3
"""Reanalyze frozen SL4b dev grounding runs at the task cluster level.

No models are run or tuned. The benchmark is the fixed ten tasks, four inits each;
cluster intervals describe variability over these tasks, not a held-out estimate.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--samples', type=int, default=10000)
    args = p.parse_args()
    src = args.run / 'episodes.jsonl'
    data = defaultdict(dict)
    for line in src.read_text().splitlines():
        row = json.loads(line)
        eid = row['episode_id']
        match = re.fullmatch(r'libero_spatial_t(\d+)_e(\d+)', eid)
        if not match:
            raise ValueError(f'unexpected episode {eid}')
        if eid in data[row['arm']]:
            raise ValueError(f'duplicate {row["arm"]}, {eid}')
        data[row['arm']][eid] = row
    expected = {f'libero_spatial_t{t}_e{e}' for t in range(10) for e in range(1,5)}
    for arm, episodes in data.items():
        if set(episodes) != expected:
            raise ValueError(f'{arm}: episode keys differ from fixed dev population')
        for row in episodes.values():
            if row.get('3d_iou') is None:
                raise ValueError(f'{arm}: missing IoU')
    ref = 'grounded_sam2:per_frame|instruction'
    rng = np.random.default_rng(args.seed)
    picks = rng.integers(0, 10, size=(args.samples, 10))
    task = {arm: np.array([np.mean([eps[f'libero_spatial_t{t}_e{e}']['3d_iou'] for e in range(1,5)])
                           for t in range(10)]) for arm, eps in data.items()}
    rows = []
    for arm in sorted(data):
        a = task[arm]
        delta = a - task[ref]
        ci = np.quantile(a[picks].mean(axis=1), [.025,.975])
        dci = np.quantile(delta[picks].mean(axis=1), [.025,.975])
        rows.append({'arm': arm, 'n_episodes': 40, 'n_tasks': 10, 'task_iou': a.tolist(),
                     'mean_3d_iou': float(a.mean()), 'task_bootstrap_95': ci.tolist(),
                     'delta_vs_reference': float(delta.mean()), 'delta_task_bootstrap_95': dci.tolist(),
                     'tasks_better': int((delta > 0).sum()), 'tasks_worse': int((delta < 0).sum())})
    args.out.mkdir(parents=True, exist_ok=True)
    result = {'input': str(src.resolve()), 'sha256': digest(src), 'reference': ref,
              'split': 'development, spatial tasks 0-9, init 1-4',
              'unit': 'frame mean per episode; episode mean per task; equal task mean',
              'interval': 'paired percentile bootstrap resampling 10 tasks with replacement; exploratory',
              'seed': args.seed, 'resamples': args.samples, 'rows': rows}
    (args.out / 'task_cluster_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    lines = ['| 방법 | 3D soft IoU | task bootstrap 95% CI | Δ vs SAM2 instruction | paired task 95% CI | 우세/열세 task |',
             '|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda x: -x['mean_3d_iou']):
        arm_label = r['arm'].replace('|', r'\|')
        c, d = r['task_bootstrap_95'], r['delta_task_bootstrap_95']
        lines.append(f"| {arm_label} | {r['mean_3d_iou']:.3f} | [{c[0]:.3f}, {c[1]:.3f}] | {r['delta_vs_reference']:+.3f} | [{d[0]:+.3f}, {d[1]:+.3f}] | {r['tasks_better']}/{r['tasks_worse']} |")
    (args.out / 'task_cluster_results.md').write_text('\n'.join(lines)+'\n')
    print(f'wrote {len(rows)} arms; 40 episodes per arm; {args.samples} paired task-cluster resamples')

if __name__ == '__main__':
    main()
