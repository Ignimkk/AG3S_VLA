#!/usr/bin/env python3
"""Independent GT/mask consistency and IoU verification for video evaluations."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--evals',nargs='+',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
checks=[];reference=None
for folder in a.evals:
 by=defaultdict(list)
 for l in (folder/'frames.jsonl').read_text().splitlines():
  r=json.loads(l)
  if 'error' in r:raise ValueError('error row: '+str(r))
  by[r['sequence']].append(r)
 keys={(s,r['frame']) for s,rs in by.items() for r in rs}
 if reference is None:reference=keys
 elif reference!=keys:raise ValueError('different evaluated frame sets')
 count=0
 for sid,rs in sorted(by.items()):
  rs.sort(key=lambda x:x['frame'])
  with np.load(a.source/(sid+'.npz')) as z:
   idx=(np.arange(224)*256/224).astype(int);labs=z['seg_full'][:,idx][:,:,idx]
   name=str(z['manipulated'][0]);nameids={str(x).split(':',1)[1]:int(str(x).split(':',1)[0]) for x in z['id2name']}
   gt=labs==nameids[name];solo=z['solo_full'][:,idx][:,:,idx] if 'solo_full' in z.files else None
  with np.load(folder/'masks'/(sid+'.npz')) as z:pred=z['masks']
  if pred.shape!=gt.shape or len(rs)!=len(gt):raise ValueError('shape/count mismatch')
  for k,r in enumerate(rs):
   if r['frame']!=k:raise ValueError('frame order')
   p,g=pred[k].astype(bool),gt[k];tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum());n=int(g.sum())
   met=r['metrics'];expected_iou=tp/(tp+fp+fn) if n else None
   if (met['tp'],met['fp'],met['fn'],met['gt_pixels'])!=(tp,fp,fn,n):raise ValueError('confusion mismatch')
   if expected_iou is None:
    if met['iou'] is not None:raise ValueError('invisible IoU expected None')
   elif abs(met['iou']-expected_iou)>1e-12:raise ValueError('IoU mismatch')
   if k % 25 == 0 and n:
    if p.any():
     pb=p & ~binary_erosion(p,border_value=0);gb=g & ~binary_erosion(g,border_value=0)
     pp=float((pb & (distance_transform_edt(~gb)<=2)).sum()/pb.sum())
     rr=float((gb & (distance_transform_edt(~pb)<=2)).sum()/gb.sum())
     bf=2*pp*rr/(pp+rr) if pp+rr else 0.
    else:bf=0.
    if abs(r['boundary_f1_2px']-bf)>1e-12:raise ValueError('boundary F1 mismatch')
    if r['bbox'] is not None:
     ys,xs=np.nonzero(g);x1,y1,x2,y2=r['bbox'];X1,Y1,X2,Y2=xs.min(),ys.min(),xs.max()+1,ys.max()+1
     inter=max(0,min(x2,X2)-max(x1,X1))*max(0,min(y2,Y2)-max(y1,Y1))
     union=max(0,x2-x1)*max(0,y2-y1)+(X2-X1)*(Y2-Y1)-inter
     bi=inter/union if union else 0.
     if abs(r['bbox_iou']-bi)>1e-12:raise ValueError('bbox IoU mismatch')
   if solo is not None:
    if (g&~solo[k]).any():raise ValueError('visible outside target-only')
    if r['solo_px']!=int(solo[k].sum()):raise ValueError('solo mismatch')
    expected=1-n/int(solo[k].sum()) if solo[k].any() else None
    if expected is None:
     if r['occlusion'] is not None:raise ValueError('occlusion should be None')
    elif abs(r['occlusion']-expected)>1e-12:raise ValueError('occlusion mismatch')
   count+=1
 checks.append({'model':folder.name,'sequences':len(by),'frames':count,'verified':'saved masks, frame continuity, TP/FP/FN/IoU, target-only occlusion; boundary and box independently sampled every 25th frame'})
a.out.write_text(json.dumps({'checks':checks,'paired_frame_sets_equal':True},indent=2)+'\n');print(json.dumps(checks,indent=2))
