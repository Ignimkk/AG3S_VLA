#!/usr/bin/env python3
"""Diagnose which GT instance occupies most of each predicted mask in recorded LIBERO."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--evals',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
files=[a.source/f'{s}_task{t}_ep0.npz' for s in ('libero_spatial','libero_object') for t in range(3)]
out={};frames=[];idx=(np.arange(224)*256/224).astype(int)
for path in files:
 with np.load(path) as z:
  labels=z['seg_full'][:,idx][:,:,idx];target=str(z['manipulated'][0]);id2name={int(str(x).split(':',1)[0]):str(x).split(':',1)[1] for x in z['id2name']};target_id=next(i for i,n in id2name.items() if n==target);candidates=set(z['candidate_ids'].tolist())
 for folder in a.evals:
  with np.load(folder/'masks'/(path.stem+'.npz')) as z:pred=z['masks']
  if len(pred)!=len(labels):raise ValueError('mask count mismatch')
  for i,(mask,lab) in enumerate(zip(pred,labels)):
   count=int(mask.sum());cls='empty';ident=None;frac=None
   if count:
    vals,counts=np.unique(lab[mask],return_counts=True);k=int(np.argmax(counts));ident=int(vals[k]);frac=float(counts[k]/count)
    if frac<.5:cls='mixed_no_majority'
    elif ident==target_id:cls='target_instance_majority'
    elif ident in candidates:cls='other_scene_instance_majority'
    else:cls='robot_background_or_other_majority'
   frames.append({'model':folder.name,'sequence':path.stem,'frame':i,'class':cls,'dominant_instance':id2name.get(ident) if ident is not None else None,'fraction':frac})
for folder in a.evals:
 rs=[r for r in frames if r['model']==folder.name];counts=Counter(r['class'] for r in rs)
 others=Counter(r['dominant_instance'] for r in rs if r['class']=='other_scene_instance_majority')
 out[folder.name]={'frames':len(rs),'counts':dict(counts),'other_instance_names':dict(others)}
a.out.mkdir(parents=True,exist_ok=False)
(a.out/'summary.json').write_text(json.dumps({'definition':'majority >= 50% of predicted pixels according to co-timed GT instance map; visible-only GT and resize limitations apply','models':out},indent=2)+'\n')
with (a.out/'frames.jsonl').open('w') as f:
 for r in frames:f.write(json.dumps(r)+'\n')
print(json.dumps(out,indent=2))
