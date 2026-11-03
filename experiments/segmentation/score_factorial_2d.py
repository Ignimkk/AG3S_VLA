#!/usr/bin/env python3
"""Score saved 2D factorial predictions against image-time simulator masks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d


def main():
    p=argparse.ArgumentParser()
    for key in ('source','scene','phase','frames','out'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shards',type=Path,nargs='+',required=True)
    a=p.parse_args()
    expected={(r['episode_id'],r['obs_index']) for r in map(json.loads,a.frames.open())}
    metadata={e['stem']:e for e in load_episodes(a.source)}
    rows={}
    for shard in a.shards:
        for r in map(json.loads,(shard/'frames.jsonl').open()):
            k=(r['episode_id'],r['obs_index'])
            if k in rows:raise ValueError(f'duplicate frame {k}')
            rows[k]=(r,shard)
    if set(rows)!=expected:raise ValueError('frame list mismatch')
    values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    counts=defaultdict(lambda:defaultdict(lambda:{'gt_absent':0,'gt_absent_pred_nonempty':0}))
    by_ep=defaultdict(list)
    for stem,t in sorted(rows):by_ep[stem].append(t)
    for stem,times in by_ep.items():
        ep=Episode(metadata[stem],scene_dir=a.scene)
        with np.load(a.phase/f'phase_{stem}.npz',allow_pickle=False) as z:
            phase={k:z[k] for k in ('call_frames','segP/head','segP/left_wrist')}
        for t in times:
            r,shard=rows[(stem,t)]; idx=np.flatnonzero(phase['call_frames']==t)
            if len(idx)!=1:raise ValueError('image-time GT join')
            with np.load(shard/'masks'/f'{stem}_{t:05d}.npz',allow_pickle=False) as z:
                for cam in ('head','left_wrist'):
                    gt=np.isin(phase[f'segP/{cam}'][idx[0]],ep.target_geoms)
                    for arm in r['arms']:
                        pred=np.unpackbits(z[f'{arm}__{cam}'],count=gt.size).reshape(gt.shape).astype(bool)
                        m=mask_metrics_2d(pred,gt)
                        if m['gt_pixels']==0:
                            counts[arm][cam]['gt_absent']+=1
                            counts[arm][cam]['gt_absent_pred_nonempty']+=int(m['pred_pixels']>0)
                        for metric in ('iou','precision','recall'):
                            v=m[metric]
                            if v is not None:values[(arm,cam,metric)][r['task_id']][stem].append(v)
    out={}
    for (arm,cam,metric),by_task in values.items():
        vals=[np.mean([np.mean(v) for v in eps.values()]) for eps in by_task.values() if eps]
        out.setdefault(arm,{}).setdefault(cam,{})[metric]={'mean':float(np.mean(vals)) if vals else None,'n_tasks':len(vals)}
    for arm in out:
        for cam in out[arm]:out[arm][cam]['invisible_target_counts']=counts[arm][cam]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump({'frames':len(rows),'averaging':'frame -> episode -> task; IoU/recall exclude GT-absent frames, precision excludes empty predictions',
                                         'arms':out},f,indent=2);f.write('\n')
    print(f'scored 2D masks on {len(rows)} frames')


if __name__=='__main__':main()
