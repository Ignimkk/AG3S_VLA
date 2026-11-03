#!/usr/bin/env python3
"""Compare automatic Grounded-SAM2 and privileged SAM2 prompts on matched images."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--automatic',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    auto={}
    for path in a.automatic:
        summary=json.loads((path/'summary.json').read_text())
        if summary['provider']!='sam2' or summary['errors']:raise ValueError('automatic run error')
        for row in map(json.loads,(path/'frames.jsonl').open()):
            if row['mode']!='phrase':continue
            key=(row['scene'],row['camera'])
            if key in auto:raise ValueError('duplicate automatic result')
            auto[key]=row
    prompt={};fail=[]
    for r in map(json.loads,(a.oracle/'frames.jsonl').open()):
        k=(r['scene'],r['camera'])
        if (k,r['arm']) in prompt:raise ValueError('duplicate oracle')
        prompt[(k,r['arm'])]=r
        if r['status']=='error':fail.append(r)
    if fail:raise ValueError('oracle inference error')
    if set(k for k,arm in prompt if arm=='oracle_box')!=set(auto):raise ValueError('frame mismatch')
    rows=[];excluded=defaultdict(int)
    for k,x in auto.items():
        box=prompt[(k,'oracle_box')];point=prompt[(k,'oracle_point')]
        if box['status']!='ok' or point['status']!='ok':excluded[box['bucket']]+=1;continue
        if 'error' in x:excluded[x['error']]+=1;continue
        if x['visible_px']!=box['visible_px'] or x['solo_px']!=box['solo_px']:
            raise ValueError('GT mismatch')
        rows.append({'scene':k[0],'camera':k[1],'benchmark':x['benchmark'],'bucket':box['bucket'],
           'auto_iou':x['metrics']['iou'],'auto_found':x['found'],
           'box_iou':box['metrics']['iou'],'point_iou':point['metrics']['iou']})
    groups=defaultdict(list)
    for r in rows:
        domain='controlled' if r['benchmark']=='safelibero_controlled' else 'natural'
        for k in ((domain,'all'),(domain,r['bucket']),(r['benchmark'],'all')):groups[k].append(r)
    out=[]
    for (domain,bucket),v in sorted(groups.items()):
        out.append({'domain':domain,'bucket':bucket,'n_images':len(v),'n_scenes':len({r['scene'] for r in v}),
            **{key:float(np.mean([r[key] for r in v])) for key in ('auto_iou','box_iou','point_iou')},
            'auto_no_detection':sum(not r['auto_found'] for r in v)})
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps({'n_matched':len(rows),'excluded':dict(excluded),'groups':out,
       'warning':'oracle box/point derived from GT; not deployable and not a fair end-to-end comparison',
       'rows':rows},indent=2)+'\n')
    print('matched',len(rows),'images; excluded',dict(excluded))


if __name__=='__main__':main()
