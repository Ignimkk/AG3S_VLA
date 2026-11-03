#!/usr/bin/env python3
"""Pair natural and deliberately occluded SafeLIBERO rendered scenes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for k in ('reference-capture','stress-capture','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reference-eval',type=Path,nargs='+',required=True)
    p.add_argument('--stress-eval',type=Path,nargs='+',required=True)
    a=p.parse_args();scenes=json.loads((a.stress_capture/'scenes.json').read_text())
    valid=[r for r in scenes if r['status']=='ok'];errors=[r for r in scenes if r['status']!='ok']
    subset=0
    for r in valid:
        with np.load(a.reference_capture/(r['reference_id']+'.npz')) as x,np.load(a.stress_capture/(r['id']+'.npz')) as y:
            for c in ('head','wrist'):
                if not np.array_equal(x[c+'_solo'],y[c+'_solo']):raise ValueError('target-only silhouette differs')
                subset+=1
    if len(a.reference_eval)!=len(a.stress_eval):raise ValueError('provider list lengths differ')
    result=[];pair_rows=[]
    for ref,stress in zip(a.reference_eval,a.stress_eval):
        pr=json.loads((ref/'summary.json').read_text());ps=json.loads((stress/'summary.json').read_text())
        if pr['provider']!=ps['provider'] or pr['errors'] or ps['errors']:raise ValueError('provider mismatch/errors')
        rr={(r['scene'],r['camera'],r['mode']):r for r in map(json.loads,(ref/'frames.jsonl').open())}
        ss={(r['scene'],r['camera'],r['mode']):r for r in map(json.loads,(stress/'frames.jsonl').open())}
        for scene in valid:
            for cam in ('head','wrist'):
                for mode in ('phrase','instruction'):
                    before=rr[(scene['reference_id'],cam,mode)];after=ss[(scene['id'],cam,mode)]
                    if 'error' in before or 'error' in after:
                        pair_rows.append({'provider':ps['provider'],'scene':scene['id'],'camera':cam,'mode':mode,'error':'provider_or_prompt_failure'});continue
                    if before['prompt']!=after['prompt']:raise ValueError('prompt changed across pair')
                    old=before['metrics']['iou'];new=after['metrics']['iou']
                    pair_rows.append({'provider':ps['provider'],'scene':scene['id'],'reference_id':scene['reference_id'],
                        'camera':cam,'mode':mode,'offset_m':scene['offset_m'],'before_occlusion':before['occlusion'],
                        'after_occlusion':after['occlusion'],'before_iou':old,'after_iou':new,
                        'delta_iou':new-old if new is not None and old is not None else None,
                        'before_visible':before['visible_px'],'after_visible':after['visible_px'],
                        'after_solo':after['solo_px'],'before_found':before['found'],'after_found':after['found']})
        for mode in ('phrase','instruction'):
            for cam in ('head','wrist'):
                group=[x for x in pair_rows if x['provider']==ps['provider'] and x['mode']==mode and x['camera']==cam and 'error' not in x]
                paired=[x for x in group if x['delta_iou'] is not None]
                hidden=[x for x in group if x['after_visible']==0 and x['after_solo']>0]
                result.append({'provider':ps['provider'],'mode':mode,'camera':cam,'pairs':len(group),
                    'both_visible_pairs':len(paired),'mean_before_iou':float(np.mean([x['before_iou'] for x in paired])) if paired else None,
                    'mean_after_iou':float(np.mean([x['after_iou'] for x in paired])) if paired else None,
                    'mean_delta_iou':float(np.mean([x['delta_iou'] for x in paired])) if paired else None,
                    'fully_hidden':len(hidden),'fully_hidden_nonempty_prediction':sum(x['after_found'] for x in hidden)})
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps({'controlled_scenes':len(valid),'errors':errors,
        'target_silhouette_checks':subset,'note':'each original scene has two interventions; samples correlated; no causal generalization',
        'results':result,'pairs':pair_rows},indent=2)+'\n')
    print('summarized',len(valid),'controlled scenes with',subset,'silhouette checks')


if __name__=='__main__':main()
