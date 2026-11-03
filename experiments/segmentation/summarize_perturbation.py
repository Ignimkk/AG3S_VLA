#!/usr/bin/env python3
"""Summarize completed mask perturbation pilot and check baseline/monotonicity."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--baseline-frames',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20261101)
    ap.add_argument('--resamples',type=int,default=10000)
    args=ap.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    rows=[json.loads(l) for l in (args.run/'frames.jsonl').read_text().splitlines()]
    assert len(rows)==len(manifest['episode_ids'])*manifest['frames_per_episode'], 'incomplete data'
    keys={(r['episode_id'],r['obs_index']) for r in rows}
    assert len(keys)==len(rows), 'duplicate frame'
    found={}
    for line in args.baseline_frames.open():
        b=json.loads(line); k=(b['episode_id'],b['obs_index'])
        if b.get('arm')=='gt_oracle|phrase' and k in keys:
            assert k not in found, 'duplicate oracle'
            found[k]=b
    assert set(found)==keys, 'missing oracle comparisons'
    max_iou_error=max(abs(r['metrics']['oracle']['iou']-found[(r['episode_id'],r['obs_index'])]['3d_iou']) for r in rows)
    assert max_iou_error < 1e-12, f'oracle mismatch {max_iou_error}'
    monotonic=[]
    for r in rows:
        m=r['metrics']
        for key, names, increasing in [('FP',['oracle','dilate_2px','dilate_4px','dilate_8px'],True),
                                      ('FN',['oracle','erode_2px','erode_4px','erode_8px'],True)]:
            v=[m[n][key] for n in names]
            if any(a>b+1e-8 for a,b in zip(v,v[1:])):
                monotonic.append({'episode':r['episode_id'],'frame':r['obs_index'],'metric':key,'values':v})
    assert not monotonic, f'non-monotonic results: {monotonic[:2]}'
    by=defaultdict(list)
    for r in rows:
        for condition, vals in r['metrics'].items():
            by[(condition,r['task_id'],r['episode_id'])].append(vals)
    taskids=sorted({r['task_id'] for r in rows})
    conditions=manifest['mask_errors']
    rng=np.random.default_rng(args.seed)
    picks=rng.integers(0,len(taskids),(args.resamples,len(taskids)))
    summary=[]
    for condition in conditions:
        result={'condition':condition}
        for metric in ('iou','recall','FP','FN','neighbor_fp','fixture_fp','support_fp','robot_fp'):
            vals=[]
            for tid in taskids:
                eps=[]
                for (c,t,e),ms in by.items():
                    if c==condition and t==tid:
                        valid=[m[metric] for m in ms if m[metric] is not None]
                        if valid: eps.append(np.mean(valid))
                if not eps: raise ValueError(f'no valid data {condition} task {tid} metric {metric}')
                vals.append(np.mean(eps))
            arr=np.array(vals)
            result[metric]={'mean':float(arr.mean()),'task_ci95':np.quantile(arr[picks].mean(axis=1),[.025,.975]).tolist(),
                            'task_means':arr.tolist()}
        summary.append(result)
    output={'input_sha256':digest(args.run/'frames.jsonl'), 'baseline_sha256':digest(args.baseline_frames),
            'oracle_max_abs_iou_error':max_iou_error, 'oracle_checked_frames':len(rows),
            'monotonicity_failures':len(monotonic), 'n_episodes':len(manifest['episode_ids']),
            'n_frames':len(rows), 'n_tasks':len(taskids), 'bootstrap':{'unit':'task','seed':args.seed,'resamples':args.resamples},
            'aggregation':'frame -> episode -> task -> mean across tasks', 'results':summary}
    (args.run/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['| 마스크 | 3D IoU | target recall | 비 target FP (voxel mass) | target FN (voxel mass) | 이웃 FP (voxel mass) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| {} | {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:.1f} |'.format(r['condition'],*[r[k]['mean'] for k in ('iou','recall','FP','FN','neighbor_fp')]))
    (args.run/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f'oracle parity max error={max_iou_error:.3g}; {len(rows)} frames; monotonicity verified')

if __name__=='__main__':main()
