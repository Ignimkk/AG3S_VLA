#!/usr/bin/env python3
"""External baseline tracking on a four-frame discontinuous occlusion stress sequence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from Baseline.paper_baselines.eval.metrics import mask_metrics_2d
from Baseline.paper_baselines.interface import TargetQuery,validate_results
from Baseline.paper_baselines.providers.grounded_sam2 import GroundedSAM2Provider


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    ref=a.root/'capture_safelibero';stress=a.root/'capture_controlled'
    original={r['id']:r for r in json.loads((ref/'scenes.json').read_text()) if r['status']=='ok'}
    variants=defaultdict(dict)
    for r in json.loads((stress/'scenes.json').read_text()):
        if r['status']!='ok':raise ValueError('controlled capture failure')
        variants[r['reference_id']][round(float(r['offset_m']),2)]=r
    if len(variants)!=8 or any(set(x)!={0.,.06} for x in variants.values()):raise ValueError('missing planned sequences')
    model=GroundedSAM2Provider(mode='track',device=a.device)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'masks').mkdir()
    (a.out/'manifest.json').write_text(json.dumps({'sequences':len(variants),'camera':'head',
      'frames':'original, stress_00, stress_06, original; discontinuous synthetic intervention',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'track':model.provenance(),
      'grounded_sam2_per_frame':'saved eval_safelibero_sam2 and eval_controlled_sam2',
      'yoloe_per_frame':'saved eval_safelibero_yoloe and eval_controlled_yoloe',
      'initial_success_threshold_iou':0.5,'control':'no GT mask or target location supplied'},indent=2,default=str)+'\n')
    rows=[];errors=[]
    for ref_id,vr in sorted(variants.items()):
        s=original[ref_id]
        if not s['phrase_from_instruction']:raise ValueError('missing language phrase')
        plan=[(ref,ref_id,'original'),(stress,vr[0.]['id'],'partial'),
              (stress,vr[.06]['id'],'fully_hidden'),(ref,ref_id,'reappeared')]
        frames=[]
        for path,name,phase in plan:
            with np.load(path/(name+'.npz'),allow_pickle=False) as z:
                frames.append((phase,name,np.ascontiguousarray(z['head_rgb']),z['head_gt'].astype(bool)))
        if not np.array_equal(frames[0][2],frames[-1][2]) or not np.array_equal(frames[0][3],frames[-1][3]):
            raise ValueError('reappearance does not equal initial')
        if not (frames[0][3].any() and frames[1][3].any() and not frames[2][3].any()):
            raise ValueError('controlled sequence visibility unexpected')
        query=TargetQuery(instruction=s['language'],target_phrase=s['phrase_from_instruction'],prompt_mode='phrase',episode_id=ref_id)
        for label,redetect in [('sam2_track',False),('sam2_track_redetect_after_empty',True)]:
            model.redetect_on_loss=redetect;model.reset()
            for index,(phase,name,rgb,gt) in enumerate(frames):
                try:
                    outputs=model.predict({'head':rgb},query);validate_results(outputs,{'head':rgb})
                    out=outputs['head'];pred=out.mask
                    np.savez_compressed(a.out/'masks'/f'{ref_id}__{label}__{index}.npz',mask=pred)
                    rows.append({'sequence':ref_id,'arm':label,'index':index,'phase':phase,'scene':name,
                         'found':bool(out.found),'source':out.extras.get('source'),'metrics':mask_metrics_2d(pred,gt)})
                except Exception as exc:
                    errors.append({'sequence':ref_id,'arm':label,'index':index,'error':repr(exc)})
                    rows.append({'sequence':ref_id,'arm':label,'index':index,'phase':phase,'scene':name,'error':repr(exc)})
        for label,refdir,stressdir in [('sam2_per_frame','eval_safelibero_sam2','eval_controlled_sam2'),
                                       ('yoloe_per_frame','eval_safelibero_yoloe','eval_controlled_yoloe')]:
            for index,(phase,name,rgb,gt) in enumerate(frames):
                folder=refdir if phase in ('original','reappeared') else stressdir
                with np.load(a.root/folder/'masks'/(name+'__phrase.npz'),allow_pickle=False) as z:pred=z['head'].astype(bool)
                if pred.shape!=gt.shape:raise ValueError('cached prediction shape mismatch')
                rows.append({'sequence':ref_id,'arm':label,'index':index,'phase':phase,'scene':name,
                             'found':bool(pred.any()),'source':'cached_per_frame','metrics':mask_metrics_2d(pred,gt)})
        print(ref_id,flush=True)
    with (a.out/'frames.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    groups=defaultdict(list)
    for r in rows:groups[(r['arm'],r['phase'])].append(r)
    summaries=[]
    for (arm,phase),group in sorted(groups.items()):
        ok=[r for r in group if 'error' not in r];iou=[r['metrics']['iou'] for r in ok if r['metrics']['iou'] is not None]
        summaries.append({'arm':arm,'phase':phase,'n':len(group),'errors':len(group)-len(ok),
            'visible_iou_mean':float(np.mean(iou)) if iou else None,'visible_iou_n':len(iou),
            'found':sum(r['found'] for r in ok),'iou_at_least_05':sum(v>=.5 for v in iou)})
    splits=[]
    for arm in ('sam2_per_frame','sam2_track','sam2_track_redetect_after_empty','yoloe_per_frame'):
        byseq=defaultdict(dict)
        for r in rows:
            if r['arm']==arm:byseq[r['sequence']][r['phase']]=r
        for label,condition in [('initial_iou_ge_05',True),('initial_iou_lt_05',False)]:
            selected=[x for x in byseq.values() if 'error' not in x['original'] and 'error' not in x['reappeared']
                      and (x['original']['metrics']['iou']>=.5)==condition]
            splits.append({'arm':arm,'group':label,'sequences':len(selected),
                           'reappeared_iou_ge_05':sum(x['reappeared']['metrics']['iou']>=.5 for x in selected),
                           'reappeared_iou_mean':float(np.mean([x['reappeared']['metrics']['iou'] for x in selected])) if selected else None})
    (a.out/'summary.json').write_text(json.dumps({'sequences':len(variants),'errors':errors,'phase_results':summaries,
      'reappearance_by_initial_quality':splits,'caveat':'artificial discontinuous 4-frame sequence; not a natural video'},indent=2)+'\n')
    print('completed',len(variants),'sequences',len(rows),'rows',len(errors),'errors')


if __name__=='__main__':main()
