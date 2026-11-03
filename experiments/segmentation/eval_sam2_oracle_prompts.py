#!/usr/bin/env python3
"""SAM2 image segmentation with GT-derived boxes/points, privileged diagnostic."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d
from Baseline.paper_baselines.providers.sam_backend import Sam2ImageBackend


def bucket(solo,visible):
    if solo==0:return 'outside_view'
    if visible==0:return 'fully_occluded'
    v=1-visible/solo
    return '0-<10%' if v<.1 else '10-<50%' if v<.5 else '50-<100%'


def prompts(gt):
    """Visible-GT box and farthest-interior pixel; deterministic row-major tie."""
    y,x=np.nonzero(gt)
    if not len(y):return None,None
    box=[int(x.min()),int(y.min()),int(x.max()),int(y.max())]
    padded=np.pad(gt,1,constant_values=False)
    dist=distance_transform_edt(padded)[1:-1,1:-1]
    iy,ix=np.unravel_index(np.argmax(dist),gt.shape)
    if not gt[iy,ix]:raise ValueError('oracle point outside target')
    return box,[int(ix),int(iy)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,nargs='+',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    model=Sam2ImageBackend(device=a.device);a.out.mkdir(parents=True,exist_ok=False);(a.out/'masks').mkdir()
    (a.out/'provenance.json').write_text(json.dumps({'model':model.provenance(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'captures':[str(x.resolve()) for x in a.captures],
      'privileged':'GT visible target mask used to construct point and box; no end-to-end claim',
      'point':'maximum distance to visible-mask boundary; deterministic row-major tie'},indent=2,default=str)+'\n')
    rows=[];errors=[]
    for root in a.captures:
        for s in json.loads((root/'scenes.json').read_text()):
            if s['status']!='ok':errors.append({'scene':s['id'],'error':'capture failed'});continue
            with np.load(root/(s['id']+'.npz'),allow_pickle=False) as z:
                imgs={c:np.ascontiguousarray(z[c+'_rgb']) for c in ('head','wrist')}
                gts={c:z[c+'_gt'].astype(bool) for c in imgs}
            masks={}
            for cam,im in imgs.items():
                gt=gts[cam];vis=int(gt.sum());solo=int(s['cameras'][cam]['solo_in_frame']);box,point=prompts(gt)
                for arm in ('oracle_box','oracle_point'):
                    base={'scene':s['id'],'benchmark':s['benchmark'],'suite':s['suite'],'level':s['level'],
                          'task_id':s['task_id'],'camera':cam,'arm':arm,'visible_px':vis,'solo_px':solo,
                          'occlusion':s['cameras'][cam]['occlusion'],'bucket':bucket(solo,vis)}
                    if not vis:
                        rows.append({**base,'status':'not_applicable','reason':'no_visible_gt_for_prompt'});continue
                    try:
                        if arm=='oracle_box':pred,score=model.predict_box(im,box)
                        else:pred,score=model.predict_points(im,[point],[1],multimask=True)
                        pred=np.asarray(pred,bool)
                        if pred.shape!=gt.shape:raise ValueError('prediction shape mismatch')
                        masks[cam+'__'+arm]=pred
                        rows.append({**base,'status':'ok','prompt':box if arm=='oracle_box' else point,
                                     'score':float(score),'metrics':mask_metrics_2d(pred,gt)})
                    except Exception as exc:
                        errors.append({'scene':s['id'],'camera':cam,'arm':arm,'error':repr(exc)})
                        rows.append({**base,'status':'error','error':repr(exc)})
            np.savez_compressed(a.out/'masks'/(s['id']+'.npz'),**masks)
            print(s['id'],flush=True)
    with (a.out/'frames.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    groups=defaultdict(list)
    for r in rows:
        for k in ((r['benchmark'],r['arm'],'all'),('natural_all' if r['benchmark']!='safelibero_controlled' else 'controlled',r['arm'],r['bucket'])):
            groups[k].append(r)
    summaries=[]
    for (bench,arm,b),items in sorted(groups.items()):
        ok=[r for r in items if r['status']=='ok'];summaries.append({'group':bench,'arm':arm,'bucket':b,
          'rows':len(items),'n_scored':len(ok),'n_not_applicable':sum(r['status']=='not_applicable' for r in items),
          'n_errors':sum(r['status']=='error' for r in items),
          'iou':float(np.mean([r['metrics']['iou'] for r in ok])) if ok else None,
          'recall':float(np.mean([r['metrics']['recall'] for r in ok])) if ok else None})
    (a.out/'summary.json').write_text(json.dumps({'rows':len(rows),'errors':errors,'results':summaries,
      'note':'GT box and point are privileged; pixel-wise visible target, not full 3D'},indent=2)+'\n')
    print('completed',len(rows),'camera-prompt records;',len(errors),'errors')


if __name__=='__main__':main()
