#!/usr/bin/env python3
"""Summarize paired, consecutive external-baseline video predictions."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np


def mean(xs):
 xs=[float(x) for x in xs if x is not None]
 return float(np.mean(xs)) if xs else None


def main():
 p=argparse.ArgumentParser();p.add_argument('--evals',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();models={};data={}
 for folder in a.evals:
  model=json.loads((folder/'manifest.json').read_text())['provider'];name=folder.name
  summary=json.loads((folder/'summary_input.json').read_text())
  if summary['errors']:raise ValueError('provider has errors '+name)
  rows=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines() if x.strip()]
  if len(rows)!=summary['rows']:raise ValueError('frame count mismatch')
  data[name]=rows;models[name]=model
 keys=lambda rows:{(r['sequence'],r['frame']) for r in rows}
 if any(keys(v)!=keys(next(iter(data.values()))) for v in data.values()):raise ValueError('not identical paired frame sets')
 a.out.mkdir(parents=True,exist_ok=False)
 allresults={};events=[]
 for name,rows in data.items():
  episodes=defaultdict(list)
  for row in rows:episodes[row['sequence']].append(row)
  sums=[]
  def stats(items):
   vis=[r for r in items if r['metrics']['gt_pixels']>0]
   return {'frames':len(items),'visible_frames':len(vis),'mean_iou':mean(r['metrics']['iou'] for r in vis),
    'mean_recall':mean(r['metrics']['recall'] for r in vis),'mean_boundary_f1_2px':mean(r['boundary_f1_2px'] for r in vis),
    'box_iou_given_selected_box':mean(r['bbox_iou'] for r in vis),'selected_box_count':sum(r['bbox_iou'] is not None for r in vis),
    'box_iou_ge_05_count':sum(r['bbox_iou'] is not None and r['bbox_iou']>=.5 for r in vis),
    'mask_iou_given_box_iou_ge_05':mean(r['metrics']['iou'] for r in vis if r['bbox_iou'] is not None and r['bbox_iou']>=.5),
    'boundary_f1_given_box_iou_ge_05':mean(r['boundary_f1_2px'] for r in vis if r['bbox_iou'] is not None and r['bbox_iou']>=.5),
    'visible_empty':sum(not r['found'] for r in vis),'visible_nonempty_no_overlap':sum(r['nonempty_no_target_overlap'] for r in vis),
    'visible_iou_ge_05':sum(r['metrics']['iou']>=.5 for r in vis),'invisible_frames':len(items)-len(vis),
    'invisible_nonempty':sum(r['found'] for r in items if r['metrics']['gt_pixels']==0)}
  for sid,es in sorted(episodes.items()):
   es.sort(key=lambda r:r['frame'])
   if [r['frame'] for r in es]!=list(range(len(es))):raise ValueError('not contiguous '+sid)
   s={'sequence':sid,'suite':es[0]['suite'],'task':es[0]['task'],'initial_iou':es[0]['metrics']['iou'],**stats(es)}
   sums.append(s)
   status=[r['metrics']['gt_pixels']==0 for r in es]
   i=0
   while i<len(es):
    if not status[i]:i+=1;continue
    start=i
    while i<len(es) and status[i]:i+=1
    end=i-1
    if start==0 or i==len(es):continue
    prefix=es[start-1]['metrics']['iou']
    later=es[i:]
    until=next((j for j,r in enumerate(later) if r['metrics']['gt_pixels']==0),len(later))
    later=later[:until]
    first=next((j for j,r in enumerate(later) if r['metrics']['iou']>=.5),None)
    ev={'model':name,'sequence':sid,'kind':'target_not_visible','start':start,'end':end,'duration_frames':end-start+1,
        'iou_before':prefix,'iou_on_reappearance':es[i]['metrics']['iou'],'frames_to_iou_ge_05':first,
        'nonempty_while_invisible':sum(r['found'] for r in es[start:i])}
    if 'solo_px' in es[start]:
     ev['fully_occluded_frames']=sum(r['solo_px']>0 for r in es[start:i]);ev['outside_view_frames']=sum(r['solo_px']==0 for r in es[start:i])
    events.append(ev)
  allresults[name]={'overall':stats(rows),'episode_mean_iou':mean(x['mean_iou'] for x in sums),
      'episode_mean_boundary_f1_2px':mean(x['mean_boundary_f1_2px'] for x in sums),'episodes':sums}
  if 'solo_px' in rows[0]:
   for lo,hi,label in [(0,.1,'0-<10%'),(.1,.5,'10-<50%'),(.5,1,'50-<100%')]:
    es=[r for r in rows if r['metrics']['gt_pixels']>0 and lo<=r['occlusion']<hi]
    allresults[name][label]=stats(es)
   full=[r for r in rows if r['metrics']['gt_pixels']==0 and r['solo_px']>0]
   outside=[r for r in rows if r['solo_px']==0]
   allresults[name]['fully_occluded']={'frames':len(full),'nonempty':sum(r['found'] for r in full)}
   allresults[name]['outside_view']={'frames':len(outside),'nonempty':sum(r['found'] for r in outside)}
 result={'paired_sequences':len(set(r['sequence'] for r in next(iter(data.values())))),
         'paired_frames':len(next(iter(data.values()))),'models':allresults,
         'caveat':'consecutive frames are correlated; descriptive exploration; detection thresholds differ; no ATS',
         'events':events}
 (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'paired_sequences':result['paired_sequences'],'paired_frames':result['paired_frames'],
  'models':{k:{'overall':v['overall'],'episode_mean_iou':v['episode_mean_iou']} for k,v in allresults.items()},'events':len(events)},indent=2))

if __name__=='__main__':main()
