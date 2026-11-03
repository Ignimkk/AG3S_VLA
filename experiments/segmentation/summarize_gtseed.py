#!/usr/bin/env python3
"""Paired task-mean summary of privileged GT-patch graph diagnostics."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include-noun',action='store_true',help='summarize all six cue x robot conditions')
    a=p.parse_args();expected={(r['episode_id'],r['obs_index']) for r in map(json.loads,a.frames.open())};rows={}
    arms=(['noun__none','noun__fk_spheres','noun__gt_image_privileged'] if a.include_noun else [])+[
        'gt_patch_privileged__none','gt_patch_privileged__fk_spheres','gt_patch_privileged__gt_image_privileged']
    for shard in a.shards:
        for r in map(json.loads,(shard/'frames.jsonl').open()):
            k=(r['episode_id'],r['obs_index'])
            if k in rows or set(r['arms'])!=set(arms):raise ValueError(f'duplicate or incorrect arms: {k}')
            rows[k]=r
    if set(rows)!=expected:raise ValueError(f'frame mismatch {len(rows)} vs {len(expected)}')
    tasks=sorted({r['task_id'] for r in rows.values()})
    if tasks!=list(range(10)):raise ValueError('expected ten tasks')
    def metric(arm,name):
        d=defaultdict(lambda:defaultdict(list))
        for r in rows.values():
            value=r['arms'][arm]['metrics'][name]
            if value is not None:d[r['task_id']][r['episode_id']].append(value)
        return np.array([np.mean([np.mean(x) for x in d[t].values()]) for t in tasks])
    result=[]
    for arm in arms:
        result.append({'arm':arm,'iou':float(metric(arm,'iou').mean()),'task_iou':metric(arm,'iou').tolist(),
                       'recall':float(metric(arm,'recall').mean()),'neighbor_fp':float(metric(arm,'neighbor_fp').mean()),
                       'abstain_count':sum(not r['arms'][arm]['applied'] for r in rows.values()),
                       'abstain_reasons':dict(Counter(r['arms'][arm]['why'] for r in rows.values() if not r['arms'][arm]['applied']))})
    rng=np.random.default_rng(20261101);picks=rng.integers(0,10,(10000,10));contrasts=[]
    baseline='gt_patch_privileged__none'
    pairs=[(b,baseline) for b in ('gt_patch_privileged__fk_spheres','gt_patch_privileged__gt_image_privileged')]
    if a.include_noun:
        pairs += [('noun__fk_spheres','noun__none'),('noun__gt_image_privileged','noun__none'),
                  ('gt_patch_privileged__fk_spheres','noun__fk_spheres')]
    for b,baseline in pairs:
        delta=metric(b,'iou')-metric(baseline,'iou')
        contrasts.append({'left':b,'right':baseline,'delta':float(delta.mean()),
                          'ci95':np.quantile(delta[picks].mean(axis=1),[.025,.975]).tolist()})
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps({'frames':len(rows),'episodes':len({k[0] for k in rows}),
        'tasks':tasks,'results':result,'contrasts':contrasts,'bootstrap':'10 tasks, seed 20261101, 10000 samples',
        'privileged':'GT target patch and GT robot mask are privileged in arms that use them'},indent=2)+'\n')
    print('summarized',len(rows),'matched frames',flush=True)


if __name__=='__main__':main()
