#!/usr/bin/env python3
"""Task-cluster reanalysis of common-point V11AB and baseline predictions."""
import argparse,hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--seed',type=int,default=20261101);a=ap.parse_args()
    p=a.run/'calls.jsonl'; data=defaultdict(lambda:defaultdict(list)); observed=defaultdict(set)
    with p.open('rb') as f:
        h=hashlib.sha256()
        for b in f:
            h.update(b); r=json.loads(b); arm=r['arm'];eid=r['episode_id']; k=(eid,int(r['obs_index']))
            if k in observed[arm]:raise ValueError(f'duplicate {arm} {k}')
            observed[arm].add(k);data[arm][eid].append(r)
    keys=next(iter(observed.values()))
    if len(keys)!=873 or any(v!=keys for v in observed.values()):raise ValueError('unpaired or incomplete frames')
    ref='grounded_sam2:per_frame|instruction'; taskids=list(range(10))
    def task_values(arm,metric):
        d=defaultdict(list)
        for eid,rs in data[arm].items():
            t=int(re.search(r'_t(\d+)_e',eid).group(1)); v=[r[metric] for r in rs if r.get(metric) is not None]
            if v:d[t].append(np.mean(v))
        return {t:np.mean(v) for t,v in d.items()}
    ref_t=task_values(ref,'iou'); pick=np.random.default_rng(a.seed).integers(0,10,(10000,10));out=[]
    for arm in sorted(data):
        v=task_values(arm,'iou')
        if set(v)!=set(taskids): raise ValueError(f'missing task {arm}')
        vals=np.array([v[t] for t in taskids]); delta=np.array([v[t]-ref_t[t] for t in taskids]);metrics={}
        for metric in ('precision','recall','neighbour_frac'):
            mt=task_values(arm,metric);metrics[metric]=float(np.mean(list(mt.values()))) if mt else None
        out.append({'arm':arm,'iou':float(vals.mean()),'iou_ci95':np.quantile(vals[pick].mean(1),[.025,.975]).tolist(),
                    'delta_vs_sam2':float(delta.mean()),'delta_ci95':np.quantile(delta[pick].mean(1),[.025,.975]).tolist(),
                    'metrics':metrics,'found_n':sum(r['found'] for e in data[arm].values() for r in e)})
    result={'calls_sha256':h.hexdigest(),'n_frames':873,'n_episodes':40,'n_tasks':10,
            'aggregation':'frame -> episode -> task -> equal task mean',
            'ci':'paired percentile bootstrap over 10 task clusters, 10000 resamples; exploratory dev',
            'warning':'ats_v11ab|noun_attention is a historical output label: this specific run uses ACTION L8H2, not noun attention.',
            'results':out}
    (a.run/'task_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| 방법 | point IoU | point recall | 이웃 혼입 비율 | SAM2 대비 Δ IoU [task 95% CI] | 유효 예측 / 873 |','|---|---:|---:|---:|---:|---:|']
    for r in sorted(out,key=lambda x:-x['iou']):
        label=r['arm'].replace('noun_attention','action_L8H2').replace('|',r'\|');c=r['delta_ci95'];m=r['metrics']
        lines.append(f"| {label} | {r['iou']:.3f} | {m['recall']:.3f} | {m['neighbour_frac']:.3f} | {r['delta_vs_sam2']:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] | {r['found_n']} |")
    (a.run/'task_summary.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(out)} arms, 873 matched frames; action L8H2, not noun attention')

if __name__=='__main__':main()
