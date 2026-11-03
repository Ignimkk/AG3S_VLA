#!/usr/bin/env python3
"""Fixed SafeLIBERO scene: unchanged, partially and fully occluded render."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from visualize_examples import overlay

BASE=Path('outputs/multibench_segmentation_20261101')


def main():
    ref='safelibero_spatial_LI_t0';conditions=[(ref,'capture_safelibero','eval_safelibero_sam2','eval_safelibero_yoloe'),
        (ref+'_stress_00','capture_controlled','eval_controlled_sam2','eval_controlled_yoloe'),
        (ref+'_stress_06','capture_controlled','eval_controlled_sam2','eval_controlled_yoloe')]
    fig,axs=plt.subplots(3,4,figsize=(14,11),constrained_layout=True)
    for i,t in enumerate(('Simulation RGB','Target GT / solo boundary','Grounded-SAM-2','YOLOE')):axs[0,i].set_title(t,fontsize=11)
    metadata=[]
    for row,(name,cap,sam,yolo) in enumerate(conditions):
        with np.load(BASE/cap/(name+'.npz')) as z:
            rgb=z['head_rgb'];gt=z['head_gt'].astype(bool);solo=z['head_solo'].astype(bool)
        occ=1-gt.sum()/solo.sum() if solo.any() else float('nan')
        axs[row,0].imshow(rgb);axs[row,0].text(.02,.97,'occlusion {:.1%}'.format(occ),transform=axs[row,0].transAxes,
           va='top',color='white',bbox={'facecolor':'black','alpha':.6})
        x=overlay(rgb,gt,np.zeros_like(gt));x[solo&~binary_erosion(solo)]=[.1,.4,1];axs[row,1].imshow(x)
        metrics={}
        for col,d in ((2,sam),(3,yolo)):
            with np.load(BASE/d/'masks'/(name+'__phrase.npz')) as z:pred=z['head'].astype(bool)
            axs[row,col].imshow(overlay(rgb,gt,pred));n=(pred | gt).sum();iou=(pred & gt).sum()/n if gt.any() and n else None
            tag='IoU {:.3f}'.format(iou) if iou is not None else 'target hidden; prediction {}'.format('nonempty' if pred.any() else 'empty')
            axs[row,col].text(.02,.97,tag,transform=axs[row,col].transAxes,va='top',color='white',fontsize=9,
                              bbox={'facecolor':'black','alpha':.6});metrics[d]={'iou':iou,'predicted_pixels':int(pred.sum())}
        for ax in axs[row]:ax.axis('off')
        metadata.append({'scene':name,'camera':'head','occlusion':occ,'visible_target_pixels':int(gt.sum()),'models':metrics})
    fig.suptitle('Fixed controlled example: green GT, red prediction, yellow overlap, blue target-only outline',fontsize=11)
    out=BASE/'summary_controlled/example.png';fig.savefig(out,dpi=140)
    out.with_suffix('.json').write_text(json.dumps({'selection':'fixed first SafeLIBERO scene and its two predeclared occluder offsets; illustrative', 'rows':metadata},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
