#!/usr/bin/env python3
"""First frame of fixed Object tasks: target GT and comparable 2D predictions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import Episode,load_episodes
from visualize_examples import overlay

BASE=Path('outputs/segmentation_baseline_20261101')
SOURCE=Path('/mnt/dev/AG3S_VLA-storage/results/benchmark/20261101T035128_96fda38abd1b_segmentation-object-dev-10-20261101')
OBJ=BASE/'object_dev'


def main():
    episodes={r['stem']:r for r in load_episodes(SOURCE)}
    frames={}
    for shard in BASE.glob('factorial_full_shard[0-9]'):
        for r in map(json.loads,(shard/'frames.jsonl').open()):frames[(r['episode_id'],r['obs_index'])]=shard
    fig,axs=plt.subplots(6,4,figsize=(13,19),constrained_layout=True)
    for col,title in enumerate(('Image + target GT','SAM2, target phrase','Noun, no robot mask','Noun + FK robot mask')):
        axs[0,col].set_title(title,fontsize=10)
    for i,task in enumerate((0,4,9)):
        stem=f'libero_object_t{task}_e0';ep=Episode(episodes[stem],scene_dir=OBJ/'scene')
        t=ep.eval_frames()[0];fr=ep.frame(t)
        with np.load(OBJ/'phase_v2'/f'phase_{stem}.npz') as z:
            ix=int(np.flatnonzero(z['call_frames']==t)[0]);gt={c:np.isin(z[f'segP/{c}'][ix],ep.target_geoms) for c in ('head','left_wrist')}
        with np.load(OBJ/'external_gpu/masks/grounded_sam2-per_frame__phrase'/f'{stem}_{t:04d}.npz') as z:
            sam={c:np.unpackbits(z[c],count=gt[c].size).reshape(gt[c].shape).astype(bool) for c in gt}
        shard=frames[(stem,t)]
        with np.load(shard/'masks'/f'{stem}_{t:05d}.npz') as z:
            pred={arm:{c:np.unpackbits(z[f'{arm}__{c}'],count=gt[c].size).reshape(gt[c].shape).astype(bool) for c in gt}
                  for arm in ('noun__none','noun__fk_spheres')}
        for j,c in enumerate(('head','left_wrist')):
            row=2*i+j
            for col,mask in enumerate((np.zeros_like(gt[c]),sam[c],pred['noun__none'][c],pred['noun__fk_spheres'][c])):
                ax=axs[row,col];ax.imshow(overlay(fr['rgb'][c],gt[c],mask));ax.axis('off')
                if col==0:label=f'task {task}, init 0, frame {t} / {c}'
                elif gt[c].any():label=f'2D IoU {(mask & gt[c]).sum()/(mask | gt[c]).sum():.3f}'
                else:label='target not visible'
                ax.text(.02,.96,label,transform=ax.transAxes,va='top',color='white',fontsize=8,
                        bbox={'facecolor':'black','alpha':.6,'pad':2})
    fig.suptitle('Green = target GT; red = prediction; yellow = overlap. Fixed first-frame examples.',fontsize=11)
    out=BASE/'factorial_summary/fixed_examples.png';fig.savefig(out,dpi=140);print(out)


if __name__=='__main__':main()
