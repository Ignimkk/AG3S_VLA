#!/usr/bin/env python3
"""Verify saved oracle-prompt, tracking and Grounded-SAM baseline measurements."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    scenes={}
    for d in ('capture_libero','capture_safelibero','capture_libero_safety','capture_libero_init1',
              'capture_safelibero_init1','capture_libero_safety_init1','capture_controlled'):
        path=a.root/d
        for r in json.loads((path/'scenes.json').read_text()):
            if r['status']!='ok':raise ValueError('capture error')
            scenes[r['id']]=(path,r)
    if len(scenes)!=96:raise ValueError('expected 96 scenes')
    def gt(scene,cam):
        path,_=scenes[scene]
        with np.load(path/(scene+'.npz')) as z:return z[cam+'_gt'].astype(bool)
    oracle=[];none=0
    for r in map(json.loads,(a.root/'sam2_oracle_prompts/frames.jsonl').open()):
        truth=gt(r['scene'],r['camera'])
        if r['status']=='not_applicable':
            if truth.any():raise ValueError('oracle should have prompt')
            none+=1;continue
        if r['status']!='ok':raise ValueError('oracle error')
        y,x=np.nonzero(truth)
        if r['arm']=='oracle_box':expected=[int(x.min()),int(y.min()),int(x.max()),int(y.max())]
        else:
            dt=distance_transform_edt(np.pad(truth,1))[1:-1,1:-1];yy,xx=np.unravel_index(np.argmax(dt),truth.shape);expected=[int(xx),int(yy)]
        if r['prompt']!=expected:raise ValueError('incorrect oracle prompt')
        with np.load(a.root/'sam2_oracle_prompts/masks'/(r['scene']+'.npz')) as z:pred=z[r['camera']+'__'+r['arm']].astype(bool)
        if r['metrics']!=mask_metrics_2d(pred,truth):raise ValueError('oracle metric mismatch')
        oracle.append(r)
    if len(oracle)+none!=384:raise ValueError('oracle row count')
    base=[];parse=0
    folder=a.root/'eval_all_grounded_sam'
    for r in map(json.loads,(folder/'frames.jsonl').open()):
        _,s=scenes[r['scene']]
        if r.get('error')=='instruction_phrase_parse_failed':
            if s['phrase_from_instruction'] is not None:raise ValueError('unexpected parse failure')
            parse+=1;continue
        if 'error' in r:raise ValueError('grounded SAM provider error')
        expected=s['language'] if r['mode']=='instruction' else s['phrase_from_instruction']
        if r['prompt']!=expected:raise ValueError('grounded SAM prompt mismatch')
        with np.load(folder/'masks'/(r['scene']+'__'+r['mode']+'.npz')) as z:pred=z[r['camera']].astype(bool)
        if r['metrics']!=mask_metrics_2d(pred,gt(r['scene'],r['camera'])):raise ValueError('grounded SAM metric mismatch')
        base.append(r)
    if len(base)+parse!=384:raise ValueError('grounded SAM row count')
    track=[];keys=set()
    for r in map(json.loads,(a.root/'baseline_occlusion_tracking/frames.jsonl').open()):
        if 'error' in r:raise ValueError('tracking error')
        k=(r['sequence'],r['arm'],r['index'])
        if k in keys:raise ValueError('duplicate tracking row')
        keys.add(k)
        if r['arm'] in ('sam2_track','sam2_track_redetect_after_empty'):
            with np.load(a.root/'baseline_occlusion_tracking/masks'/f"{r['sequence']}__{r['arm']}__{r['index']}.npz") as z:pred=z['mask'].astype(bool)
        else:
            model='sam2' if r['arm']=='sam2_per_frame' else 'yoloe'
            folder='eval_safelibero_'+model if r['phase'] in ('original','reappeared') else 'eval_controlled_'+model
            with np.load(a.root/folder/'masks'/(r['scene']+'__phrase.npz')) as z:pred=z['head'].astype(bool)
        if r['metrics']!=mask_metrics_2d(pred,gt(r['scene'],'head')):raise ValueError('tracking metric mismatch')
        track.append(r)
    if len(track)!=128 or len(keys)!=128:raise ValueError('tracking row count')
    out={'scenes':len(scenes),'oracle_scored_rows':len(oracle),'oracle_not_applicable_rows':none,
         'grounded_sam_scored_rows':len(base),'grounded_sam_parser_failure_rows':parse,
         'tracking_rows':len(track),'oracle_prompts_and_all_metrics':'verified'}
    with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(out)


if __name__=='__main__':main()
