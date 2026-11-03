#!/usr/bin/env python3
"""Evaluate external segmenters on synchronized static multibench scenes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d
from Baseline.paper_baselines.interface import TargetQuery,validate_results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--captures',nargs='+',type=Path,required=True)
    p.add_argument('--provider',choices=('sam2','yoloe'),required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.provider=='sam2':
        from Baseline.paper_baselines.providers.grounded_sam2 import GroundedSAM2Provider
        model=GroundedSAM2Provider(mode='per_frame',device=a.device)
    else:
        from Baseline.paper_baselines.providers.yoloe import YOLOEProvider
        model=YOLOEProvider(device=a.device)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'masks').mkdir()
    (a.out/'provenance.json').write_text(json.dumps({'provider':model.provenance(),'captures':[str(c.resolve()) for c in a.captures],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input':'synchronized RGB and prompt from instruction; no GT used for prediction',
        'gt':'actual target visible pixels; target-only render for occlusion diagnostics only'},indent=2,default=str)+'\n')
    rows=[];count=0;errors=[]
    for root in a.captures:
        scenes=json.loads((root/'scenes.json').read_text())
        for scene in scenes:
            if scene['status']!='ok':
                errors.append({'scene':scene['id'],'phase':'capture','error':scene['error']});continue
            with np.load(root/(scene['id']+'.npz'),allow_pickle=False) as z:
                rgb={c:np.ascontiguousarray(z[c+'_rgb']) for c in ('head','wrist')}
                gt={c:z[c+'_gt'].astype(bool) for c in rgb}
            for mode in ('instruction','phrase'):
                phrase=scene['phrase_from_instruction']
                if mode=='phrase' and phrase is None:
                    for cam in rgb:rows.append({'scene':scene['id'],'benchmark':scene['benchmark'],'suite':scene['suite'],
                        'level':scene['level'],'task_id':scene['task_id'],'camera':cam,'mode':mode,
                        'error':'instruction_phrase_parse_failed','occlusion':scene['cameras'][cam]['occlusion']})
                    continue
                query=TargetQuery(instruction=scene['language'],target_phrase=phrase or '',prompt_mode=mode,episode_id=scene['id'])
                try:
                    result=model.predict(rgb,query);validate_results(result,rgb)
                    np.savez_compressed(a.out/'masks'/(scene['id']+'__'+mode+'.npz'),**{c:result[c].mask for c in rgb})
                    for cam in rgb:
                        row={'scene':scene['id'],'benchmark':scene['benchmark'],'suite':scene['suite'],
                             'level':scene['level'],'task_id':scene['task_id'],'camera':cam,'mode':mode,
                             'prompt':query.prompt,'occlusion':scene['cameras'][cam]['occlusion'],
                             'visible_px':scene['cameras'][cam]['visible'],'solo_px':scene['cameras'][cam]['solo_in_frame'],
                             'found':bool(result[cam].found),'metrics':mask_metrics_2d(result[cam].mask,gt[cam])}
                        rows.append(row)
                except Exception as exc:
                    errors.append({'scene':scene['id'],'mode':mode,'error':repr(exc)})
                    for cam in rgb:rows.append({'scene':scene['id'],'benchmark':scene['benchmark'],'suite':scene['suite'],
                        'level':scene['level'],'task_id':scene['task_id'],'camera':cam,'mode':mode,'error':repr(exc),
                        'occlusion':scene['cameras'][cam]['occlusion']})
            count+=1;print(a.provider,scene['id'],count,flush=True)
    with (a.out/'frames.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    groups=defaultdict(list)
    for r in rows:
        if 'error' in r:continue
        if r['solo_px']==0:bucket='outside_view'
        elif r['visible_px']==0:bucket='fully_occluded'
        elif r['occlusion']<.1:bucket='0-<10%'
        elif r['occlusion']<.5:bucket='10-<50%'
        else:bucket='50-<100%'
        for key in ((r['benchmark'],r['mode'],r['camera'],'all'),(r['benchmark'],r['mode'],r['camera'],bucket),
                    ('all',r['mode'],r['camera'],bucket)):
            groups[key].append(r)
    result=[]
    for (bench,mode,cam,bucket),rs in sorted(groups.items()):
        vals=[r['metrics']['iou'] for r in rs if r['metrics']['iou'] is not None]
        rec=[r['metrics']['recall'] for r in rs if r['metrics']['recall'] is not None]
        absent=[r for r in rs if r['visible_px']==0]
        result.append({'benchmark':bench,'mode':mode,'camera':cam,'bucket':bucket,'n_images':len(rs),
                       'n_visible_target':len(vals),'iou':float(np.mean(vals)) if vals else None,
                       'recall':float(np.mean(rec)) if rec else None,'found':sum(r['found'] for r in rs),
                       'invisible_target_images':len(absent),'invisible_nonempty':sum(r['found'] for r in absent)})
    (a.out/'summary.json').write_text(json.dumps({'provider':a.provider,'scenes_completed':count,'rows':len(rows),
       'errors':errors,'parser_failure_rows':sum(r.get('error')=='instruction_phrase_parse_failed' for r in rows),
       'aggregation':'image mean within group; camera frames and related tasks are not independent',
       'results':result},indent=2)+'\n')
    print('completed',count,'scenes with',len(errors),'errors',flush=True)


if __name__=='__main__':main()
