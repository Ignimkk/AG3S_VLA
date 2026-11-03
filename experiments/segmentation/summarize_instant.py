#!/usr/bin/env python3
"""Summarize matched image-instant segmentation scores at the task level."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()


def mean_task(values):
    return float(np.mean([np.mean(v) for v in values.values()]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--labels',type=Path,help='optional independently generated phase labels')
    ap.add_argument('--seed',type=int,default=20261101)
    args=ap.parse_args()
    rows=[json.loads(l) for l in (args.run/'frames.jsonl').read_text().splitlines()]
    data=defaultdict(dict)
    for r in rows:
        k=(r['episode_id'],r['obs_index']); arm=r['arm']
        if k in data[arm]: raise ValueError(f'duplicate {arm} {k}')
        data[arm][k]=r
    manifest=json.loads((args.run/'manifest.json').read_text())
    expected_arms=4+len(manifest.get('providers',{}))+len(manifest.get('extra_surfaces',[]))
    n_frames=manifest.get('n_frames',200)
    if len(data)!=expected_arms or any(len(v)!=n_frames for v in data.values()):
        raise ValueError(f'incomplete arms: {[(a,len(b)) for a,b in data.items()]}')
    common=set(next(iter(data.values())))
    if any(set(v)!=common for v in data.values()):raise ValueError('frame sets differ')
    labels={}
    if args.labels:
        for line in args.labels.open():
            r=json.loads(line)
            for t,p in zip(r['frames'],r['labels']): labels[(r['stem'],int(t))]=p
    else:
        labels={k:'unclassified' for k in common}
    if not common<=set(labels):raise ValueError('missing phase labels')
    taskids=sorted({r['task_id'] for r in data['oracle_image'].values()})
    assert len(taskids)==10
    pick=np.random.default_rng(args.seed).integers(0,10,(10000,10))
    def by_task(arm,metric,phase=None):
        result=defaultdict(lambda:defaultdict(list))
        for k,r in data[arm].items():
            if phase and labels[k]!=phase:continue
            if metric in ('head_iou','wrist_iou'):
                cam='head' if metric=='head_iou' else 'left_wrist'; val=r['pixel_metrics'][cam]['iou']
            else:val=r['metrics'][metric]
            if val is not None: result[r['task_id']][r['episode_id']].append(float(val))
        return {t:mean_task(episodes) for t,episodes in result.items() if episodes}
    reference=manifest.get('reference','grounded_sam2_frame_instruction')
    if reference not in data: reference=next((x for x in ('yoloe_phrase','grounded_sam_instruction') if x in data),'oracle_image')
    ref=by_task(reference,'iou')
    out=[]
    for arm in sorted(data):
        metrics={}
        for m in ('iou','precision','recall','neighbor_fp','head_iou','wrist_iou'):
            v=by_task(arm,m); metrics[m]={'mean':float(np.mean(list(v.values()))) if v else None,'n_tasks':len(v)}
        tm=by_task(arm,'iou')
        if set(tm)!=set(taskids):raise ValueError(f'missing task IoU for {arm}')
        taskvals=np.array([tm[t] for t in taskids]); delta=np.array([tm[t]-ref[t] for t in taskids])
        phases={}
        for p in sorted(set(labels[k] for k in common)):
            tm_phase=by_task(arm,'iou',p)
            phases[p]={'mean':float(np.mean(list(tm_phase.values()))) if tm_phase else None,
                       'n_tasks':len(tm_phase),'n_frames':sum(labels[k]==p and r['metrics']['iou'] is not None for k,r in data[arm].items())}
        invisible={}
        for c in ('head','left_wrist'):
            camera_rows=[r['pixel_metrics'][c] for r in data[arm].values()]
            absent=[r for r in camera_rows if r['gt_pixels']==0]
            invisible[c]={'gt_absent_frames':len(absent),'predicted_nonempty':sum(r['pred_pixels']>0 for r in absent)}
        out.append({'arm':arm,'metrics':metrics,'invisible_target':invisible,'task_iou':taskvals.tolist(),
                    'iou_ci95':np.quantile(taskvals[pick].mean(axis=1),[.025,.975]).tolist(),
                    'delta_vs_reference':float(delta.mean()),
                    'delta_ci95':np.quantile(delta[pick].mean(axis=1),[.025,.975]).tolist(),'phases':phases})
    changes={c:sum(data['oracle_image'][k]['gt_pixel_change'][c]>0 for k in common) for c in ('head','left_wrist')}
    result={'input_sha256':sha(args.run/'frames.jsonl'),'labels_sha256':sha(args.labels) if args.labels else None,
            'n_frames':n_frames,'n_episodes':len({k[0] for k in common}),'n_tasks':len(taskids),'label_difference_frames':changes,
            'aggregation':'frame -> episode -> task -> equal task mean; precision ignores no-prediction frames',
            'ci':'paired task bootstrap 10000 samples, exploratory dev set, not independent held-out',
            'reference':reference,'results':out}
    (args.run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'| 방법 | 3D IoU | recall | 이웃 FP | {reference} 대비 Δ IoU [task 95% CI] |',
           '|---|---:|---:|---:|---:|']
    for r in sorted(out,key=lambda r:-r['metrics']['iou']['mean']):
        m=r['metrics'];c=r['delta_ci95']
        lines.append(f"| {r['arm']} | {m['iou']['mean']:.3f} | {m['recall']['mean']:.3f} | {m['neighbor_fp']['mean']:.1f} | {r['delta_vs_reference']:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] |")
    (args.run/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f'summarized {len(data)} methods on {n_frames} paired frames; changed GT frames: {changes}')

if __name__=='__main__':main()
