#!/usr/bin/env python3
"""Combine independently scored masks only after exact frame-key checks."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root
    expected={(r['episode_id'],r['obs_index']) for r in map(json.loads,(root/'all_frames.jsonl').open())}
    if len(expected)!=552:raise ValueError('expected 552 fixed frames')
    keys=set()
    for shard in sorted(root.glob('factorial_shard[0-9]')):
        for r in map(json.loads,(shard/'frames.jsonl').open()):
            k=r['episode_id'],r['obs_index']
            if k in keys:raise ValueError('duplicate candidate frame')
            keys.add(k)
    if keys!=expected:raise ValueError('candidate frame mismatch')
    sources={'sam2':root/'sam2_image_instant','yoloe025':root/'yoloe_image_instant',
             'yoloe005_posthoc':root/'yoloe_conf005_image_instant'}
    for name,path in sources.items():
        perarm={}
        for r in map(json.loads,(path/'frames.jsonl').open()):
            k=r['episode_id'],r['obs_index'];perarm.setdefault(r['arm'],[]).append(k)
        if any(len(v)!=len(expected) or set(v)!=expected for v in perarm.values()):
            raise ValueError(f'{name}: arm frame mismatch')
    candidate=json.loads((root/'factorial_summary/summary.json').read_text())
    candidate2=json.loads((root/'factorial_summary/score_2d.json').read_text())
    results=[]
    for x in candidate['results']:
        if x['arm'] not in ('noun__none','noun__fk_spheres','noun__gt_image_privileged'):continue
        c=candidate2['arms'][x['arm']]
        results.append({'arm':x['arm'],'iou':x['iou'],'recall':x['recall'],'neighbor_fp':x['neighbor_fp'],
                        'head_iou':c['head']['iou']['mean'],'wrist_iou':c['left_wrist']['iou']['mean'],
                        'task_iou':x['task_iou'],'abstain':x['abstain_count']})
    for name,path in sources.items():
        summary=json.loads((path/'summary.json').read_text())
        for x in summary['results']:
            if x['arm']=='oracle_image' and name!='sam2':continue
            if x['arm']=='yoloe_phrase':arm=name
            else:arm=x['arm']
            m=x['metrics'];v=m['iou']['task_means']
            if set(v)!=set(map(str,range(10))):raise ValueError(f'{arm} missing tasks')
            results.append({'arm':arm,'iou':m['iou']['mean'],'recall':m['recall']['mean'],
                            'neighbor_fp':m['neighbor_fp']['mean'],'head_iou':m['head_iou']['mean'],
                            'wrist_iou':m['left_wrist_iou']['mean'],'task_iou':[v[str(i)] for i in range(10)]})
    byarm={r['arm']:r for r in results}
    rng=np.random.default_rng(20261101);idx=rng.integers(0,10,(10000,10));contrasts=[]
    for other in ('grounded_sam2_frame_phrase','grounded_sam2_frame_instruction','yoloe025','yoloe005_posthoc'):
        diff=np.array(byarm['noun__fk_spheres']['task_iou'])-np.array(byarm[other]['task_iou'])
        contrasts.append({'candidate':'noun__fk_spheres','baseline':other,'delta':float(diff.mean()),
                          'ci95':np.quantile(diff[idx].mean(axis=1),[.025,.975]).tolist(),'task_differences':diff.tolist()})
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps({'frames':len(expected),'episodes':len({k[0] for k in expected}),
        'aggregation':'frame -> episode -> task -> equal 10 task mean',
        'bootstrap':'paired task bootstrap, 10000 samples, seed 20261101; exploratory',
        'results':results,'contrasts':contrasts,'baseline_paths':{k:str(v) for k,v in sources.items()}},indent=2)+'\n')
    lines=['| 방법 | 3D IoU | recall | 이웃 FP | head / wrist 2D IoU |', '|---|---:|---:|---:|---:|']
    for r in sorted(results,key=lambda r:-r['iou']):
        lines.append(f"| {r['arm']} | {r['iou']:.3f} | {r['recall']:.3f} | {r['neighbor_fp']:.1f} | {r['head_iou']:.3f} / {r['wrist_iou']:.3f} |")
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('combined',len(results),'arms on',len(expected),'frames')


if __name__=='__main__':main()
