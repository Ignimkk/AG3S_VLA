#!/usr/bin/env python3
"""Compare surface pilot with saved baselines on exactly the same frames."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def readlines(p):
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frames',type=Path,required=True,help='frame list of perturbation dev run')
    ap.add_argument('--baseline',type=Path,required=True,help='existing provider frames.jsonl')
    ap.add_argument('--surface',type=Path,required=True)
    ap.add_argument('--surface-gtrobot',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20261101)
    a=ap.parse_args()
    frame_rows=readlines(a.frames)
    selection={(r['episode_id'],int(r['obs_index'])):r for r in frame_rows}
    if len(selection)!=len(frame_rows) or len(frame_rows)!=200: raise ValueError('expected 200 unique sample frames')
    phase={}
    for ep in readlines(a.labels):
        for t,label in zip(ep['frames'],ep['labels']): phase[(ep['stem'],int(t))]=label
    if not set(selection)<=set(phase): raise ValueError('missing phase labels')
    data=defaultdict(dict)
    for r in readlines(a.baseline):
        key=(r['episode_id'],int(r['obs_index']))
        if key not in selection: continue
        if r.get('missing_reason') and r['missing_reason']!='target_not_visible':
            raise ValueError(f'baseline error: {r["arm"]} {key} {r["missing_reason"]}')
        if key in data[r['arm']]: raise ValueError(f'duplicate {r["arm"]} {key}')
        data[r['arm']][key]={'iou':r.get('3d_iou'),'precision':r.get('3d_precision'),
                            'recall':r.get('3d_recall'),'neighbor_fp':r.get('3d_neighbor_fp'),
                            'found_any':r.get('found_any'),
                            'head_iou':r.get('cameras',{}).get('head',{}).get('2d_iou'),
                            'wrist_iou':r.get('cameras',{}).get('left_wrist',{}).get('2d_iou')}
    for name,path in [('surface_adapter_no_robot_mask',a.surface),('surface_adapter_gt_robot_privileged',a.surface_gtrobot)]:
        for r in readlines(path):
            key=(r['episode_id'],int(r['obs_index']))
            if key not in selection or key in data[name]: raise ValueError(f'bad candidate frame {key}')
            m=r['metrics']; p=r['pixel_metrics']
            data[name][key]={'iou':m['iou'],'precision':m['precision'],'recall':m['recall'],
                             'neighbor_fp':m['neighbor_fp'],'found_any':r['applied'],
                             'head_iou':p['head']['iou'],'wrist_iou':p['left_wrist']['iou']}
            if r['selection'].get('n_labelled_px',{}).get('right_wrist',0)!=0:
                raise ValueError('dummy camera added algorithm pixels')
    for name in data:
        if set(data[name])!=set(selection): raise ValueError(f'not paired: {name}, n={len(data[name])}')
    ref='grounded_sam2:per_frame|instruction'
    taskids=sorted({int(r['task_id']) for r in selection.values()})
    rng=np.random.default_rng(a.seed); pick=rng.integers(0,len(taskids),(10000,len(taskids)))
    def taskmean(arm,metric,phase_name=None):
        tmp=defaultdict(lambda:defaultdict(list))
        for k,v in data[arm].items():
            if phase_name is not None and phase[k]!=phase_name: continue
            val=v[metric]
            if val is None: continue
            tmp[int(selection[k]['task_id'])][k[0]].append(float(val))
        return {t:float(np.mean([np.mean(v) for v in eps.values()])) for t,eps in tmp.items() if eps}
    ref_iou=taskmean(ref,'iou')
    result=[]
    for arm in sorted(data):
        metrics={}
        for m in ('iou','precision','recall','neighbor_fp','head_iou','wrist_iou'):
            tm=taskmean(arm,m)
            metrics[m]={'mean':float(np.mean(list(tm.values()))) if tm else None,'n_tasks':len(tm)}
        tm=taskmean(arm,'iou')
        if set(tm)!=set(taskids): raise ValueError(f'missing IoU tasks {arm}')
        task_values=np.array([tm[t] for t in taskids]); delta=np.array([tm[t]-ref_iou[t] for t in taskids])
        phases={p:{'mean':float(np.mean(list(tm.values()))) if tm else None,'n_tasks':len(tm),
                   'n_frames':sum(phase[k]==p and data[arm][k]['iou'] is not None for k in selection)}
                for p in sorted(set(phase[k] for k in selection)) for tm in [taskmean(arm,'iou',p)]}
        result.append({'arm':arm,'n_frames':len(data[arm]),'found_n':sum(bool(v['found_any']) for v in data[arm].values()),
                       'metrics':metrics,'iou_task_means':task_values.tolist(),
                       'iou_ci95':np.quantile(task_values[pick].mean(1),[.025,.975]).tolist(),
                       'delta_vs_sam2_instruction':float(delta.mean()),
                       'delta_ci95':np.quantile(delta[pick].mean(1),[.025,.975]).tolist(),
                       'phases':phases})
    a.out.mkdir(parents=True,exist_ok=True)
    output={'n_frames':len(selection),'n_episodes':len({k[0] for k in selection}),'n_tasks':len(taskids),
            'metric':'frame -> episode -> task -> equally weighted task mean; precision ignores undefined empty predictions',
            'ci':'paired percentile bootstrap, 10000 resamples of task clusters, exploratory on 10 dev tasks',
            'reference':ref,'inputs':{str(p):sha(p) for p in (a.frames,a.baseline,a.surface,a.surface_gtrobot,a.labels)},
            'results':result}
    (a.out/'surface_compare.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['| 방법 | 3D IoU | recall | 이웃 FP | 비교기준 대비 Δ IoU [task 95% CI] | 감지/200 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in sorted(result,key=lambda x:-x['metrics']['iou']['mean']):
        label=r['arm'].replace('|',r'\|'); m=r['metrics']; ci=r['delta_ci95']
        lines.append(f"| {label} | {m['iou']['mean']:.3f} | {m['recall']['mean']:.3f} | {m['neighbor_fp']['mean']:.1f} | {r['delta_vs_sam2_instruction']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['found_n']} |")
    (a.out/'surface_compare.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(result)} arms, {len(selection)} paired frames, {len(taskids)} tasks')

if __name__=='__main__':main()
