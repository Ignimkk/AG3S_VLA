#!/usr/bin/env python3
"""Aggregate static segmentation by benchmark, camera and physical occlusion."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np


def bucket(r):
    if not r['solo_px']:return 'outside_view'
    if not r['visible_px']:return 'fully_occluded'
    if r['occlusion']<.1:return '0-<10%'
    if r['occlusion']<.5:return '10-<50%'
    return '50-<100%'


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs='+',required=True)
    p.add_argument('--sam2',type=Path,nargs='+',required=True);p.add_argument('--yoloe',type=Path,nargs='+',required=True)
    p.add_argument('--grounded-sam',type=Path,nargs='*',default=[])
    p.add_argument('--ignore-other-scenes',action='store_true',help='explicitly omit evaluation rows outside selected captures')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    scenes={};capture_errors=[]
    for d in a.captures:
        for r in json.loads((d/'scenes.json').read_text()):
            if r['id'] in scenes:raise ValueError('duplicate scene')
            scenes[r['id']]=r
            if r['status']!='ok':capture_errors.append(r)
    expected={(r['id'],cam,mode) for r in scenes.values() if r['status']=='ok'
              for cam in ('head','wrist') for mode in ('phrase','instruction')}
    models={};errors={};ignored={}
    providers=[('sam2',a.sam2),('yoloe',a.yoloe)]
    if a.grounded_sam:providers.append(('grounded_sam',a.grounded_sam))
    for name,dirs in providers:
        rows={};issues=[];ignored_count=0
        for d in dirs:
            summary=json.loads((d/'summary.json').read_text());issues.extend(summary['errors'])
            if summary['provider']!=name:raise ValueError(f'{d}: wrong provider')
            for r in map(json.loads,(d/'frames.jsonl').open()):
                if a.ignore_other_scenes and r['scene'] not in scenes:
                    ignored_count+=1;continue
                k=r['scene'],r['camera'],r['mode']
                if k in rows:raise ValueError('duplicate prediction '+str(k))
                rows[k]=r
        if set(rows)!=expected:raise ValueError(f'{name}: expected {len(expected)} rows, got {len(rows)}')
        models[name]=rows;errors[name]=issues;ignored[name]=ignored_count
    grouped=[]
    for name,rows in models.items():
        groups=defaultdict(list)
        for r in rows.values():
            if 'error' in r:continue
            for key in ((r['benchmark'],r['camera'],r['mode'],'all'),
                        (r['benchmark'],'both',r['mode'],'all'),
                        ('all','both',r['mode'],bucket(r)),
                        (r['benchmark'],'both',r['mode'],bucket(r))):groups[key].append(r)
        for (bench,cam,mode,b),rs in sorted(groups.items()):
            vals=[r['metrics']['iou'] for r in rs if r['metrics']['iou'] is not None]
            recall=[r['metrics']['recall'] for r in rs if r['metrics']['recall'] is not None]
            precision=[r['metrics']['precision'] for r in rs if r['metrics']['precision'] is not None]
            absent=[r for r in rs if r['visible_px']==0]
            grouped.append({'provider':name,'benchmark':bench,'camera':cam,'mode':mode,'occlusion_bucket':b,
                'images':len(rs),'visible_images':len(vals),'iou':float(np.mean(vals)) if vals else None,
                'recall':float(np.mean(recall)) if recall else None,
                'precision_nonempty':float(np.mean(precision)) if precision else None,
                'precision_n':len(precision),'absent_images':len(absent),
                'absent_nonempty':sum(r['found'] for r in absent),'nonempty':sum(r['found'] for r in rs),
                'unique_scenes':len({r['scene'] for r in rs})})
    hist=Counter()
    for r in next(iter(models.values())).values():
        if r.get('mode')=='phrase' and 'error' not in r:hist[(r['benchmark'],bucket(r))]+=1
    out={'scenes_planned':len(scenes),'scenes_ok':sum(r['status']=='ok' for r in scenes.values()),
         'capture_errors':capture_errors,'provider_errors':errors,'explicitly_ignored_other_scene_rows':ignored,
         'parser_failures':{name:sum(r.get('error')=='instruction_phrase_parse_failed' for r in rows.values())//2
                            for name,rows in models.items()},
         'occlusion_counts':{'|'.join(k):v for k,v in sorted(hist.items())},
         'aggregation':'visible target image mean, per benchmark and view; images in a scene are correlated',
         'results':grouped}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print('summarized',out['scenes_ok'],'scenes of',out['scenes_planned'])


if __name__=='__main__':main()
