#!/usr/bin/env python3
"""Fixed example overlays: GT green, prediction red, overlap yellow."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes

BASE=Path('/mnt/dev/AG3S_VLA-storage/results/benchmark')
SOURCE=BASE/'20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4'
BASELINE=BASE/'20261024T040140_5752c64c7006_baseline-eval-v2-sl4bdev'
PHASE=BASE/'20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels/phase'
OUT=Path('outputs/segmentation_baseline_20261101')
SCENE=Path('outputs/verify/SL4b/raw/scene')


def overlay(rgb,gt,pred):
    a=rgb.astype(np.float32)/255
    color=np.zeros_like(a); color[...,0]=pred; color[...,1]=gt
    visible=gt|pred
    a[visible]=.45*a[visible]+.55*color[visible]
    return np.clip(a,0,1)


def main():
    # Fixed task IDs and first init/call chosen before inspecting these examples.
    episodes={r['stem']:r for r in load_episodes(SOURCE)}
    fig,axs=plt.subplots(6,4,figsize=(13,19),constrained_layout=True)
    for col,title in enumerate(('Image + image-time GT','Grounded-SAM-2 instruction','Surface: no robot mask','Surface: GT robot mask*')):
        axs[0,col].set_title(title,fontsize=10)
    for i,task in enumerate((0,4,9)):
        stem=f'libero_spatial_t{task}_e1'; ep=Episode(episodes[stem],scene_dir=SCENE)
        t=ep.eval_frames()[0]; fr=ep.frame(t)
        with np.load(PHASE/f'phase_{stem}.npz') as z:
            ix=int(np.flatnonzero(z['call_frames']==t)[0]); seg={c:z[f'segP/{c}'][ix] for c in ('head','left_wrist')}
        with np.load(BASELINE/'masks/grounded_sam2-per_frame__instruction'/f'{stem}_{t:04d}.npz') as z:
            sam={c:np.unpackbits(z[c],count=int(np.prod(z[c+'__shape']))).reshape(tuple(z[c+'__shape'])).astype(bool) for c in seg}
        predictions=[]
        for name in ('surface_dev40_nomask_v2','surface_dev40_gtrobot_v2'):
            with np.load(OUT/name/'masks'/f'{stem}_{t:05d}.npz') as z:predictions.append({c:z[c].astype(bool) for c in seg})
        for j,c in enumerate(('head','left_wrist')):
            row=2*i+j; gt=np.isin(seg[c],ep.target_geoms); rgb=fr['rgb'][c]
            for col,pred in enumerate((gt,sam[c],predictions[0][c],predictions[1][c])):
                axs[row,col].imshow(overlay(rgb,gt,pred))
                axs[row,col].axis('off')
                if col==0: axs[row,col].text(.02,.96,f'task {task}, init 1, frame {t} / {c}',transform=axs[row,col].transAxes,va='top',color='white',fontsize=8,bbox={'facecolor':'black','alpha':.6,'pad':2})
                if col:
                    union=(gt|pred).sum(); iou=(gt&pred).sum()/union if gt.sum() and union else float('nan')
                    tag = f'2D IoU {iou:.3f}' if np.isfinite(iou) else 'target not visible'
                    axs[row,col].text(.02,.96,tag,transform=axs[row,col].transAxes,va='top',color='white',fontsize=8,bbox={'facecolor':'black','alpha':.6,'pad':2})
    fig.suptitle('Green = target GT; red = prediction; yellow = overlap. *GT robot mask is privileged.',fontsize=11)
    fig.savefig(OUT/'fixed_segmentation_examples.png',dpi=140)
    print(OUT/'fixed_segmentation_examples.png')

if __name__=='__main__':main()
