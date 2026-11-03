#!/usr/bin/env python3
"""Rescore saved external masks against image-time GT without candidate masks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import Episode,load_episodes,seg_labels
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d
from image_instant_eval import PROVIDERS,saved
from mask_perturbation import scores,CAMERAS


def main():
    p=argparse.ArgumentParser()
    for k in ('source','scene','phase','frames','baseline','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--providers',nargs='+',required=True,choices=tuple(PROVIDERS))
    a=p.parse_args(); meta={e['stem']:e for e in load_episodes(a.source)}; selected=defaultdict(list)
    for r in map(json.loads,a.frames.open()):selected[r['episode_id']].append(r['obs_index'])
    if not set(selected)<=set(meta):raise ValueError('frame episode missing')
    info=json.loads((a.baseline/'summary.json').read_text())
    if info.get('provider_errors'):raise ValueError('provider errors prevent completed comparison')
    prov=json.loads((a.baseline/'provenance.json').read_text())
    if prov.get('provider_build_errors'):raise ValueError('provider build errors')
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for stem,times in selected.items():
        ep=Episode(meta[stem],scene_dir=a.scene)
        with np.load(a.phase/f'phase_{stem}.npz') as z:phase={k:z[k] for k in ('call_frames','segP/head','segP/left_wrist')}
        for t in times:
            i=np.flatnonzero(phase['call_frames']==t)
            if len(i)!=1:raise ValueError('missing frame')
            fr=ep.frame(t);gt={}
            for c in CAMERAS:
                seg=phase[f'segP/{c}'][i[0]];fr['labels'][c]=seg_labels(seg,ep.lut);gt[c]=np.isin(seg,ep.target_geoms)
            masks={'oracle_image':gt}
            for arm in a.providers:masks[arm]=saved(a.baseline/'masks'/PROVIDERS[arm]/f'{stem}_{t:04d}.npz')
            met=scores(fr,masks)
            for arm in masks:
                rows.append({'episode_id':stem,'task_id':meta[stem]['task_id'],'obs_index':t,'arm':arm,
                             'metrics':met[arm], 'pixel_metrics':{c:mask_metrics_2d(masks[arm][c],gt[c]) for c in CAMERAS}})
        print(stem,flush=True)
    with (a.out/'frames.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    out=[]
    for arm in ['oracle_image']+a.providers:
        armrows=[r for r in rows if r['arm']==arm];metrics={}
        for metric in ('iou','recall','precision','neighbor_fp','head_iou','left_wrist_iou'):
            vals=defaultdict(lambda:defaultdict(list))
            for r in armrows:
                if metric in ('head_iou','left_wrist_iou'):
                    cam=metric[:-4];v=r['pixel_metrics'][cam]['iou']
                else:v=r['metrics'][metric]
                if v is not None:vals[r['task_id']][r['episode_id']].append(v)
            task={int(k):float(np.mean([np.mean(v) for v in ep.values()])) for k,ep in vals.items()}
            metrics[metric]={'mean':float(np.mean(list(task.values()))) if task else None,'task_means':task}
        invisible={}
        for c in CAMERAS:
            px=[r['pixel_metrics'][c] for r in armrows];absent=[v for v in px if v['gt_pixels']==0]
            invisible[c]={'gt_absent':len(absent),'pred_nonempty_when_gt_absent':sum(v['pred_pixels']>0 for v in absent),
                          'any_pred_nonempty':sum(v['pred_pixels']>0 for v in px)}
        out.append({'arm':arm,'metrics':metrics,'invisible':invisible})
    (a.out/'summary.json').write_text(json.dumps({'frames':sum(map(len,selected.values())),
        'episodes':len(selected),'aggregation':'frame -> episode -> task; undefined precision excluded',
        'source_baseline':str(a.baseline),'results':out},indent=2)+'\n')
    print('completed',len(rows),'arm-frame records')


if __name__=='__main__':main()
