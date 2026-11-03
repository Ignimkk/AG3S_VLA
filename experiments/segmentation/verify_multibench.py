#!/usr/bin/env python3
"""Independently check stored masks, GT joins, prompts and 2D metrics."""
import argparse
from pathlib import Path
import json
import numpy as np
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',nargs='+',type=Path,required=True)
    p.add_argument('--evaluations',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();scene_map={}
    for root in a.captures:
        for s in json.loads((root/'scenes.json').read_text()):
            if s['status']!='ok':continue
            if s['id'] in scene_map:raise ValueError('duplicate scene')
            if s.get('phrase_from_instruction') and s['phrase_from_instruction'].lower() not in s['language'].lower():
                raise ValueError('phrase not present in instruction')
            scene_map[s['id']]=(root,s)
    total=0;parser_fail=0;checked=set()
    for root in a.evaluations:
        provider=json.loads((root/'summary.json').read_text())['provider']
        for r in map(json.loads,(root/'frames.jsonl').open()):
            key=(provider,r['scene'],r['camera'],r['mode'])
            if key in checked:raise ValueError('duplicate provider prediction')
            checked.add(key)
            cap,s=scene_map[r['scene']]
            if r.get('error')=='instruction_phrase_parse_failed':
                if s['phrase_from_instruction'] is not None:raise ValueError('false parse error')
                parser_fail+=1;continue
            if 'error' in r:raise ValueError('provider exception '+str(key))
            prompt=s['language'] if r['mode']=='instruction' else s['phrase_from_instruction']
            if prompt!=r['prompt']:raise ValueError('prompt mismatch')
            with np.load(cap/(s['id']+'.npz'),allow_pickle=False) as z:
                gt=z[r['camera']+'_gt'].astype(bool);solo=z[r['camera']+'_solo'].astype(bool)
            if np.any(gt & ~solo):raise ValueError('visible GT outside target-only')
            with np.load(root/'masks'/(s['id']+'__'+r['mode']+'.npz'),allow_pickle=False) as z:
                pred=z[r['camera']].astype(bool)
            if mask_metrics_2d(pred,gt)!=r['metrics']:raise ValueError('metric mismatch '+str(key))
            occ=float(1-gt.sum()/solo.sum()) if solo.any() else None
            if r['occlusion']!=occ:raise ValueError('occlusion mismatch')
            total+=1
    expected=len(scene_map)*2*2*2
    if len(checked)!=expected:raise ValueError(f'expected {expected} provider-camera-mode rows, got {len(checked)}')
    with a.out.open('x') as f:json.dump({'scenes':len(scene_map),'rows':len(checked),'scored_rows':total,
        'phrase_parser_failure_rows':parser_fail,'metrics_equal':'all','prompt_source':'instruction text only',
        'visible_subset_of_solo':'all'},f,indent=2);f.write('\n')
    print('verified',len(scene_map),'scenes,',total,'metric rows;',parser_fail,'parser failure rows')


if __name__=='__main__':main()
