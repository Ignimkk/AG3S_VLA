#!/usr/bin/env python3
"""Rescore saved mask predictions against segmentation at RGB-D image time.

No provider is rerun or selected. Existing SL4b phase_export segP is OpenCV
orientation, rendered at the recorded image-time state. Scores are not the
same target as the previous return-time segmentation scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes, seg_labels
from Baseline.paper_baselines.eval import metrics as M
from mask_perturbation import scores, CAMERAS

PROVIDERS={
 'grounded_sam_instruction':'grounded_sam__instruction',
 'grounded_sam2_frame_instruction':'grounded_sam2-per_frame__instruction',
 'grounded_sam2_frame_phrase':'grounded_sam2-per_frame__phrase',
 'grounded_sam2_track_instruction':'grounded_sam2-track__instruction',
 'yoloe_phrase':'yoloe__phrase',
 'localization_heads_sam_point':'localization_heads-sam_point__no_text',
 'localization_heads_threshold':'localization_heads-threshold__no_text',
 'tokencut_foreground':'tokencut-foreground__no_text',
 'tokencut_attention_overlap':'tokencut-attention_overlap__no_text',
}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def saved(path):
    with np.load(path,allow_pickle=False) as z:
        masks={}
        for c in CAMERAS:
            h,w=map(int,z[f'{c}__shape'])
            masks[c]=np.unpackbits(z[c],count=h*w).reshape(h,w).astype(bool)
        return masks


def main():
    ap=argparse.ArgumentParser()
    for k in ('source','scene','phase','frame-list','baseline-run','surface','surface-gtrobot','out'):
        ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--extra-surface',nargs=2,action='append',default=[],metavar=('ARM','DIR'))
    ap.add_argument('--expected-episodes',type=int,default=40)
    ap.add_argument('--providers',nargs='*',choices=tuple(PROVIDERS),default=list(PROVIDERS))
    a=ap.parse_args()
    providers={name:PROVIDERS[name] for name in a.providers}
    frame_list={}
    for line in a.frame_list.open():
        r=json.loads(line); frame_list.setdefault(r['episode_id'],[]).append(int(r['obs_index']))
    metadata=sorted((e for e in load_episodes(a.source) if e['stem'] in frame_list),
                    key=lambda r:(r['task_id'],r['episode_idx']))
    if len(metadata)!=a.expected_episodes or len(frame_list)!=a.expected_episodes or any(not x for x in frame_list.values()):
        raise ValueError(f'expected {a.expected_episodes} populated episodes')
    a.out.mkdir(parents=True,exist_ok=False)
    manifest={'kind':'offline re-evaluation of saved predictions against image-instant GT',
              'source':str(a.source.resolve()), 'phase':str(a.phase.resolve()),
              'frame_list_sha256':sha(a.frame_list),'script_sha256':sha(Path(__file__)),
              'baseline_run':str(a.baseline_run.resolve()), 'surface':str(a.surface.resolve()),
              'surface_gtrobot':str(a.surface_gtrobot.resolve()),'providers':providers,
              'n_frames':sum(map(len,frame_list.values())),'n_episodes':len(metadata),
              'gt_note':'segP at image instant, same depth cameras, simulator target and robot labels',
              'warning':'candidate GT-robot run still USED return-time robot masks for its prediction; privileged',
              'extra_surfaces':a.extra_surface,
              'arms':'saved masks with empty masks for selector abstention; no fallback or actions'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    t0=time.monotonic()
    with (a.out/'frames.jsonl').open('x') as out:
        for n,meta in enumerate(metadata,1):
            ep=Episode(meta,cameras=CAMERAS,scene_dir=a.scene)
            with np.load(a.phase/f'phase_{ep.stem}.npz',allow_pickle=False) as z:
                call_frames=z['call_frames']
                seg={c:z[f'segP/{c}'] for c in CAMERAS}
                for t in frame_list[ep.stem]:
                    match=np.flatnonzero(call_frames==t)
                    if len(match)!=1: raise ValueError(f'bad image-instant GT join {ep.stem}/{t}')
                    j=int(match[0]); fr=ep.frame(t)
                    return_gt={c:fr['labels'][c]==M.CLASSES.index('target') for c in CAMERAS}
                    fr['labels']={c:seg_labels(seg[c][j],ep.lut) for c in CAMERAS}
                    gt={c:fr['labels'][c]==M.CLASSES.index('target') for c in CAMERAS}
                    masks={'oracle_image':gt,'oracle_return':return_gt}
                    for name,folder in providers.items():
                        p=a.baseline_run/'masks'/folder/f'{ep.stem}_{t:04d}.npz'
                        masks[name]=saved(p)
                    for name,root in (('surface_no_robot_mask',a.surface),('surface_gtrobot_privileged',a.surface_gtrobot)):
                        with np.load(root/'masks'/f'{ep.stem}_{t:05d}.npz',allow_pickle=False) as z:
                            masks[name]={c:z[c].astype(bool) for c in CAMERAS}
                    for name,root in a.extra_surface:
                        if name in masks: raise ValueError(f'duplicate arm {name}')
                        with np.load(Path(root)/'masks'/f'{ep.stem}_{t:05d}.npz',allow_pickle=False) as z:
                            masks[name]={c:z[c].astype(bool) for c in CAMERAS}
                    metrics=scores(fr,masks)
                    for arm,mask in masks.items():
                        row={'episode_id':ep.stem,'task_id':int(meta['task_id']),'episode_idx':int(meta['episode_idx']),
                             'obs_index':t,'arm':arm,'metrics':metrics[arm],
                             'pixel_metrics':{c:M.mask_metrics_2d(mask[c],gt[c]) for c in CAMERAS},
                             'gt_pixel_change':{c:int(np.count_nonzero(return_gt[c]!=gt[c])) for c in CAMERAS}}
                        out.write(json.dumps(row,allow_nan=False)+'\n')
                    out.flush()
            print(f'{n}/{len(metadata)} {ep.stem} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    print('complete',flush=True)

if __name__=='__main__': main()
