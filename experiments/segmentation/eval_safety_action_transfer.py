#!/usr/bin/env python3
"""Score instruction-only external segmenters on recorded continuous policy rollouts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d
from Baseline.paper_baselines.interface import TargetQuery, validate_results

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'outputs/continuous_rollout_20261102/libero_safety_capture'
PHRASE = re.compile(r'^\s*(?:pick up|pick|grab|pass me) the (?P<target>.+?)(?: and (?:place|put|bring|cut)\b.*|\s*[.!]?\s*$)', re.I)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def box_iou(box,gt):
    ys,xs=np.nonzero(gt)
    if not len(xs) or box is None:return None
    x1,y1,x2,y2=map(float,box)
    gx1,gy1,gx2,gy2=float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)
    intersection=max(0,min(x2,gx2)-max(x1,gx1))*max(0,min(y2,gy2)-max(y1,gy1))
    area=max(0,x2-x1)*max(0,y2-y1)+(gx2-gx1)*(gy2-gy1)-intersection
    return intersection/area if area else 0.


def boundary_f1(pred,gt):
    if not gt.any():return None
    if not pred.any():return 0.
    a=pred & ~binary_erosion(pred,border_value=0)
    b=gt & ~binary_erosion(gt,border_value=0)
    # Euclidean radius <= 2 in pixels, not an unspecified iteration-shaped tolerance.
    yy,xx=np.ogrid[-2:3,-2:3];foot=(xx*xx+yy*yy)<=4
    prec=float((a & binary_dilation(b,structure=foot)).sum()/a.sum())
    rec=float((b & binary_dilation(a,structure=foot)).sum()/b.sum())
    return 2*prec*rec/(prec+rec) if prec+rec else 0.


def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',choices=['sam2_frame','sam2_track','yoloe'],required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--source',default=SOURCE,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--limit-sequences',type=int,default=0)
    a=p.parse_args()
    if a.provider.startswith('sam2'):
        from Baseline.paper_baselines.providers.grounded_sam2 import GroundedSAM2Provider
        model=GroundedSAM2Provider(mode='track' if a.provider=='sam2_track' else 'per_frame',device=a.device)
    else:
        from Baseline.paper_baselines.providers.yoloe import YOLOEProvider
        model=YOLOEProvider(device=a.device)
    files=[a.source/f'libero_safety_{s}_task0_ep0.npz' for s in ('affordance','human_safety','obstacle_avoidance')]
    if a.limit_sequences:files=files[:a.limit_sequences]
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'masks').mkdir()
    manifest={'provider':model.provenance(),'source_files':{str(x):sha(x) for x in files},
       'evaluation_script_sha256':sha(Path(__file__)),'collection_script_sha256':sha(ROOT/'Baseline/paper_baselines/methods/knows/collect_p0b.py'),
       'protocol':str(ROOT/'doc/active/experiments/BASELINE_LIBERO_SAFETY_ACTION_TRANSFER_PROTOCOL_20261102_KO.md'),
       'target':'recorded simulator instance is GT only, not supplied to models',
       'gt_resize':'nearest floor output_index * 256 / 224', 'camera':'head',
       'track_input':'all stored frames in temporal order','missingness':'target-only rendered at same state; solo zero = outside rendered view'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    allrows=[];sequences=[];errors=[];t0=time.time()
    for path in files:
        model.reset();sid=path.stem
        try:
            with np.load(path,allow_pickle=False) as z:
                prompt=str(z['prompt']);m=PHRASE.fullmatch(prompt)
                if not m:raise ValueError('instruction phrase parser failed: '+prompt)
                manipulated=[str(n) for n in z['manipulated']]
                if len(manipulated)!=1:raise ValueError('expected exactly one target')
                idmap={x.split(':',1)[1]:int(x.split(':',1)[0]) for x in map(str,z['id2name'])}
                target_ids=[i for name,i in idmap.items() if name==manipulated[0]]
                if len(target_ids)!=1:raise ValueError('target instance is not uniquely mapped')
                target_id=target_ids[0];images=z['image'];labels=z['seg_full'];solos=z['solo_full'];ts=z['t']
                if len(ts)!=len(images) or len(ts)!=len(labels) or not np.array_equal(ts,np.arange(len(ts))):raise ValueError('incomplete consecutive observation indices')
                if tuple(images.shape[1:])!=(224,224,3) or tuple(labels.shape[1:])!=(256,256):raise ValueError('unexpected source shape')
                success=bool(z['success']);suite=str(z['suite']);task=int(z['task_id'])
            idx=(np.arange(224)*256/224).astype(np.int64)
            query=TargetQuery(instruction=prompt,target_phrase=m.group('target').strip(),prompt_mode='phrase',episode_id=sid)
            preds=[];rows=[]
            for k,(image,lab) in enumerate(zip(images,labels)):
                rgb=np.ascontiguousarray(image);lab=lab[np.ix_(idx,idx)];gt=lab==target_id
                solo=solos[k][np.ix_(idx,idx)]; solo_px=int(solo.sum())
                if np.any(gt & ~solo):raise ValueError('visible outside solo after resizing')
                try:
                    result=model.predict({'head':rgb},query);validate_results(result,{'head':rgb});out=result['head'];pred=out.mask
                    metric=mask_metrics_2d(pred,gt)
                    selected=out.extras.get('selected');box=None
                    if selected is not None and out.boxes is not None and len(out.boxes)>selected:box=out.boxes[selected].tolist()
                    non_target=pred & (lab!=target_id) & (lab!=0)
                    row={'sequence':sid,'suite':suite,'task':task,'frame':k,'t':int(ts[k]),'found':bool(out.found),
                        'source':out.extras.get('source','detect'),'bbox':box,'bbox_iou':box_iou(box,gt),
                        'solo_px':solo_px,'occlusion':float(1-gt.sum()/solo_px) if solo_px else None,
                        'boundary_f1_2px':boundary_f1(pred,gt),'other_instance_pixels':None,
                        'nonempty_no_target_overlap':bool(pred.any() and gt.any() and not np.any(pred & gt)),
                        'metrics':metric}
                    preds.append(pred)
                except Exception as exc:
                    row={'sequence':sid,'suite':suite,'task':task,'frame':k,'t':int(ts[k]),'error':repr(exc)}
                    errors.append(dict(row,traceback=traceback.format_exc()));preds.append(np.zeros((224,224),bool))
                rows.append(row)
                if (k+1)%50==0:print(a.provider,sid,k+1,'/',len(ts),flush=True)
            np.savez_compressed(a.out/'masks'/f'{sid}.npz',masks=np.stack(preds))
            allrows.extend(rows);sequences.append({'id':sid,'suite':suite,'task':task,'source_success':success,
                 'source_frames':len(ts),'target_gt_instance':manipulated[0],'phrase':query.prompt,'frames':len(rows),
                 'errors':sum('error' in r for r in rows)})
            print(a.provider,sid,'done',len(rows),'elapsed',round(time.time()-t0,1),flush=True)
        except Exception as exc:
            errors.append({'sequence':sid,'error':repr(exc),'traceback':traceback.format_exc()})
            sequences.append({'id':sid,'error':repr(exc)});print(sid,'ERROR',repr(exc),flush=True)
        with (a.out/'frames.jsonl').open('w') as f:
            for r in allrows:f.write(json.dumps(r,allow_nan=False)+'\n')
        (a.out/'progress.json').write_text(json.dumps({'sequences':sequences,'errors':errors},indent=2)+'\n')
    (a.out/'summary_input.json').write_text(json.dumps({'sequences':sequences,'errors':errors,'rows':len(allrows),'elapsed_seconds':time.time()-t0},indent=2)+'\n')
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
