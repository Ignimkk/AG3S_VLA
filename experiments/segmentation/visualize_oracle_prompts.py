#!/usr/bin/env python3
"""Fixed example comparing language grounding and privileged SAM2 prompts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from visualize_examples import overlay

BASE=Path('outputs/multibench_segmentation_20261101')


def main():
    ref='safelibero_spatial_LI_t0';cases=[(ref,'capture_safelibero','eval_safelibero_sam2'),
      (ref+'_stress_00','capture_controlled','eval_controlled_sam2')]
    oracle={(r['scene'],r['camera'],r['arm']):r for r in map(json.loads,(BASE/'sam2_oracle_prompts/frames.jsonl').open())}
    fig,axs=plt.subplots(2,4,figsize=(15,8),constrained_layout=True)
    for j,title in enumerate(('RGB + target GT','Grounded-SAM2: phrase','SAM2: GT-derived box','SAM2: GT-derived point')):
        axs[0,j].set_title(title,fontsize=10)
    records=[]
    for i,(scene,cap,auto) in enumerate(cases):
        with np.load(BASE/cap/(scene+'.npz')) as z:
            rgb=z['head_rgb'];gt=z['head_gt'].astype(bool);solo=z['head_solo'].astype(bool)
        axs[i,0].imshow(overlay(rgb,gt,np.zeros_like(gt)))
        occ=1-gt.sum()/solo.sum();axs[i,0].text(.02,.97,f'occlusion {occ:.1%}',transform=axs[i,0].transAxes,
          va='top',color='white',bbox={'facecolor':'black','alpha':.6})
        with np.load(BASE/auto/'masks'/(scene+'__phrase.npz')) as z:auto_pred=z['head'].astype(bool)
        with np.load(BASE/'sam2_oracle_prompts/masks'/(scene+'.npz')) as z:
            box_pred=z['head__oracle_box'].astype(bool);point_pred=z['head__oracle_point'].astype(bool)
        for j,pred in enumerate((auto_pred,box_pred,point_pred),1):
            axs[i,j].imshow(overlay(rgb,gt,pred));iou=(pred & gt).sum()/(pred | gt).sum()
            axs[i,j].text(.02,.97,f'IoU {iou:.3f}',transform=axs[i,j].transAxes,va='top',color='white',
                          bbox={'facecolor':'black','alpha':.6})
        box=oracle[(scene,'head','oracle_box')]['prompt'];point=oracle[(scene,'head','oracle_point')]['prompt']
        axs[i,2].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='cyan',linewidth=2))
        axs[i,3].scatter([point[0]],[point[1]],s=70,c='magenta',marker='+',linewidths=2)
        for ax in axs[i]:ax.axis('off')
        records.append({'scene':scene,'occlusion':occ,'gt_box':box,'gt_point':point})
    fig.suptitle('Green GT, red prediction, yellow overlap. Cyan box / magenta point are privileged GT-derived inputs.',fontsize=11)
    out=BASE/'sam2_prompt_comparison/example.png';fig.savefig(out,dpi=140)
    out.with_suffix('.json').write_text(json.dumps({'selection':'fixed SafeLIBERO first scene and its partial-occlusion condition',
                                                    'privileged':True,'rows':records},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
