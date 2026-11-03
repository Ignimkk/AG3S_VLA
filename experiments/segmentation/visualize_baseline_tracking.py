#!/usr/bin/env python3
"""Fixed four-observation tracking illustration; deliberately discontinuous."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from visualize_examples import overlay

BASE=Path('outputs/multibench_segmentation_20261101')
REF='safelibero_spatial_LI_t0'


def main():
    items=[(REF,'capture_safelibero','eval_safelibero_sam2','original'),
           (REF+'_stress_00','capture_controlled','eval_controlled_sam2','partial'),
           (REF+'_stress_06','capture_controlled','eval_controlled_sam2','fully hidden'),
           (REF,'capture_safelibero','eval_safelibero_sam2','reappeared')]
    fig,axs=plt.subplots(4,4,figsize=(14,14),constrained_layout=True)
    for j,title in enumerate(('RGB + visible GT','SAM2 per-frame','SAM2 causal tracking','SAM2 track + redetect')):
        axs[0,j].set_title(title,fontsize=10)
    details=[]
    for i,(name,cap,ev,phase) in enumerate(items):
        with np.load(BASE/cap/(name+'.npz')) as z:
            rgb=z['head_rgb'];gt=z['head_gt'].astype(bool);solo=z['head_solo'].astype(bool)
        with np.load(BASE/ev/'masks'/(name+'__phrase.npz')) as z:per=z['head'].astype(bool)
        pred=[per]
        for arm in ('sam2_track','sam2_track_redetect_after_empty'):
            with np.load(BASE/'baseline_occlusion_tracking/masks'/f'{REF}__{arm}__{i}.npz') as z:pred.append(z['mask'].astype(bool))
        axs[i,0].imshow(overlay(rgb,gt,np.zeros_like(gt)))
        occ=1-gt.sum()/solo.sum();axs[i,0].text(.02,.97,f'{phase}: {occ:.1%} occluded',transform=axs[i,0].transAxes,
            va='top',color='white',fontsize=9,bbox={'facecolor':'black','alpha':.6})
        metrics=[]
        for j,mask in enumerate(pred,1):
            axs[i,j].imshow(overlay(rgb,gt,mask));u=(mask | gt).sum();iou=float((mask & gt).sum()/u) if gt.any() and u else None
            tag=f'IoU {iou:.3f}' if iou is not None else ('GT hidden; mask nonempty' if mask.any() else 'GT hidden; empty mask')
            axs[i,j].text(.02,.97,tag,transform=axs[i,j].transAxes,va='top',color='white',fontsize=8,
                          bbox={'facecolor':'black','alpha':.6});metrics.append(iou)
        for ax in axs[i]:ax.axis('off')
        details.append({'phase':phase,'occlusion':float(occ),'iou_per_frame_track_redetect':metrics})
    fig.suptitle('Illustration: green GT, red prediction, yellow overlap. Four discontinuous simulator renders, not natural video.',fontsize=10)
    out=BASE/'baseline_occlusion_tracking/example.png';fig.savefig(out,dpi=130)
    out.with_suffix('.json').write_text(json.dumps({'fixed_scene':REF,'frames':details},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
