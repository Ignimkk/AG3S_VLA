#!/usr/bin/env python3
"""Plot target visibility and baseline mask IoUs on the same recorded time axis."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--evals',nargs='+',required=True,type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
records={}
for d in a.evals:
 rows=[json.loads(x) for x in (d/'frames.jsonl').read_text().splitlines() if x.strip()]
 rows=sorted((r for r in rows if r['sequence']==a.sequence),key=lambda r:r['frame'])
 if not rows:raise ValueError('sequence missing '+str(d))
 records[d.name]=rows
first=next(iter(records.values()));n=len(first)
if any(len(v)!=n for v in records.values()):raise ValueError('frame count mismatch')
x=np.arange(n);occ=np.array([r['occlusion'] if r['occlusion'] is not None else np.nan for r in first])
fig,ax=plt.subplots(2,1,figsize=(12,5.8),sharex=True)
ax[0].plot(x,occ,color='black',lw=1.6);ax[0].set_ylabel('Occlusion fraction');ax[0].set_ylim(-.05,1.05);ax[0].grid(alpha=.25)
for name,rows in records.items():
 vals=[r['metrics']['iou'] if r['metrics']['iou'] is not None else np.nan for r in rows]
 ax[1].plot(x,vals,label=name,linewidth=1.3,alpha=.85)
ax[1].set_ylim(-.05,1.05);ax[1].set_ylabel('Visible-target IoU');ax[1].set_xlabel('Consecutive observation index');ax[1].grid(alpha=.25);ax[1].legend(loc='best',ncol=2)
fig.suptitle(a.sequence+' | IoU undefined while target is invisible');fig.tight_layout();fig.savefig(a.out,dpi=150);print(a.out)
