#!/usr/bin/env python3
"""Illustrative highest-occlusion examples per benchmark, not a random sample."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
from visualize_examples import overlay


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',nargs='+',type=Path,required=True)
    p.add_argument('--sam2',nargs='+',type=Path,required=True);p.add_argument('--yoloe',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();scenes=[]
    for root in a.captures:
        for r in json.loads((root/'scenes.json').read_text()):
            if r['status']!='ok':continue
            for cam,c in r['cameras'].items():
                has_sam=any((x/'masks'/(r['id']+'__phrase.npz')).exists() for x in a.sam2)
                has_yoloe=any((x/'masks'/(r['id']+'__phrase.npz')).exists() for x in a.yoloe)
                if c['visible']>0 and c['occlusion'] is not None and has_sam and has_yoloe:
                    scenes.append((r,c['occlusion'],cam,root))
    fig,axs=plt.subplots(len(a.captures),4,figsize=(15,4.2*len(a.captures)),squeeze=False,constrained_layout=True)
    for i,(title,roots) in enumerate((('RGB',None),('GT visible + solo boundary',None),('Grounded-SAM-2: phrase',a.sam2),('YOLOE: phrase',a.yoloe))):
        axs[0,i].set_title(title,fontsize=11)
    chosen=[]
    for row,root in enumerate(a.captures):
        possible=[x for x in scenes if x[3]==root]
        if not possible:raise ValueError('no visible scene '+str(root))
        r,occ,cam,_=max(possible,key=lambda x:x[1]);chosen.append({'scene':r['id'],'camera':cam,'occlusion':occ})
        with np.load(root/(r['id']+'.npz')) as z:
            rgb=z[cam+'_rgb'];gt=z[cam+'_gt'].astype(bool);solo=z[cam+'_solo'].astype(bool)
        axs[row,0].imshow(rgb)
        gt_img=overlay(rgb,gt,np.zeros_like(gt));edge=solo&~binary_erosion(solo);gt_img[edge]=[0.15,0.45,1.0]
        axs[row,1].imshow(gt_img)
        for col,roots in ((2,a.sam2),(3,a.yoloe)):
            hits=[x/'masks'/(r['id']+'__phrase.npz') for x in roots if (x/'masks'/(r['id']+'__phrase.npz')).exists()]
            if len(hits)!=1:raise ValueError('missing or duplicate mask '+r['id'])
            with np.load(hits[0]) as z:pred=z[cam].astype(bool)
            axs[row,col].imshow(overlay(rgb,gt,pred));iou=(pred & gt).sum()/(pred | gt).sum()
            axs[row,col].text(.02,.97,'IoU {:.3f}'.format(iou),transform=axs[row,col].transAxes,va='top',color='white',
                fontsize=9,bbox={'facecolor':'black','alpha':.6})
        axs[row,0].text(.02,.97,'{} / {}\nocclusion {:.1%}'.format(r['id'],cam,occ),
            transform=axs[row,0].transAxes,va='top',color='white',fontsize=9,bbox={'facecolor':'black','alpha':.6})
        for ax in axs[row]:ax.axis('off')
    fig.suptitle('Selected highest visible occlusion per benchmark; green GT, red prediction, yellow overlap; blue solo boundary',fontsize=11)
    a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=135)
    a.out.with_suffix('.json').write_text(json.dumps({'selection':'highest occlusion with visible target and both phrase predictions in each capture; illustrative, not representative','rows':chosen},indent=2)+'\n')
    print(a.out)


if __name__=='__main__':main()
