#!/usr/bin/env python3
"""Illustrate a fixed consecutive rollout using saved predictions and labels."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--sequence',required=True)
p.add_argument('--frame-eval',type=Path,required=True);p.add_argument('--track-eval',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',nargs=4,type=int)
a=p.parse_args();sid=a.sequence
with np.load(a.source/(sid+'.npz')) as z:
 imgs=z['image'];labels=z['seg_full'];target=str(z['manipulated'][0]);ids={str(v).split(':',1)[1]:int(str(v).split(':',1)[0]) for v in z['id2name']};ix=(np.arange(224)*256/224).astype(int);gt=labels[:,ix][:,:,ix]==ids[target]
frames=a.frames or [0,len(imgs)//3,2*len(imgs)//3,len(imgs)-1]
if any(i<0 or i>=len(imgs) for i in frames):raise ValueError('frame outside sequence')
masks=[];rows=[]
for d in (a.frame_eval,a.track_eval):
 with np.load(d/'masks'/(sid+'.npz')) as z:masks.append(z['masks'])
 rows.append({int(r['frame']):r for r in (json.loads(x) for x in (d/'frames.jsonl').read_text().splitlines()) if r['sequence']==sid})
fig,axes=plt.subplots(3,4,figsize=(12,9))
for c,i in enumerate(frames):
 axes[0,c].imshow(imgs[i]);axes[0,c].set_title('Frame {}'.format(i))
 for k in range(2):
  p=masks[k][i].astype(bool);g=gt[i];img=imgs[i].astype(float)/255*.45
  img[g]+=np.array([0,.55,0]);img[p]+=np.array([.55,0,0]);axes[k+1,c].imshow(np.clip(img,0,1))
  score=rows[k][i]['metrics']['iou'];axes[k+1,c].set_title(('SAM2 per-frame' if k==0 else 'SAM2 track')+' | IoU '+('N/A' if score is None else '{:.2f}'.format(score)))
 for ax in axes[:,c]:ax.set_xticks([]);ax.set_yticks([])
fig.suptitle(sid+' | green: GT only, red: prediction only, yellow: overlap',fontsize=11)
fig.tight_layout();fig.savefig(a.out,dpi=150);print(a.out)
