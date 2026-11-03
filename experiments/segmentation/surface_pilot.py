#!/usr/bin/env python3
"""Exploratory Panda port of IFG attention-surface to two recorded cameras.

Calls the existing AG3S runtime selector, but gives its required third camera an
fully robot-masked placeholder. This is a research adapter, not the deployed Panda
configuration. The default removes no robot pixels; optional GT robot mask is a
privileged diagnostic. No policy actions are executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes
from Baseline.paper_baselines.eval import metrics as M
from AG3S.ag3s.runtime.attention_surface import SurfaceView, select_target_surface, CAMERA_ORDER
from mask_perturbation import scores, CAMERAS


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--scene',type=Path,required=True)
    ap.add_argument('--noun-run',type=Path,required=True)
    ap.add_argument('--frame-list',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit-episodes',type=int,default=0)
    ap.add_argument('--robot-mask',choices=('none','gt'),default='none')
    ap.add_argument('--head',default='8,6')
    ap.add_argument('--gt-attention',type=Path,help='privileged diagnostic: phase directory; target-mask patch coverage replaces attention')
    a=ap.parse_args()
    layer,head=map(int,a.head.split(','))
    selected={}
    for line in a.frame_list.open():
        r=json.loads(line); selected.setdefault(r['episode_id'],[]).append(int(r['obs_index']))
    metadata=sorted((m for m in load_episodes(a.source) if m['stem'] in selected),key=lambda m:(m['task_id'],m['episode_idx']))
    if a.limit_episodes: metadata=metadata[:a.limit_episodes]
    if not metadata: raise ValueError('no matching episodes')
    a.out.mkdir(parents=True,exist_ok=False)
    manifest={'description':'offline research adapter; no policy execution; not finalized AG3S',
              'selector':'AG3S.ag3s.runtime.attention_surface.select_target_surface',
              'inputs':'same recorded depth/scene segmentation, pre-extracted VLA noun attention',
              'source':str(a.source.resolve()),'source_episodes_sha256':sha(a.source/'episodes.jsonl'),
              'noun_run':str(a.noun_run.resolve()),'frame_list_sha256':sha(a.frame_list),
              'script_sha256':sha(Path(__file__)),'layer':layer,'head':head,
              'robot_mask':a.robot_mask,'privileged_robot_mask':a.robot_mask=='gt',
              'camera_adapter':'head + left_wrist; mandatory right_wrist is a one-valid-pixel dummy fully robot-masked',
              'attention_source': 'simulator GT target patch coverage (privileged)' if a.gt_attention else 'recorded noun attention',
              'note':'head L8H6 was selected using same dev episodes; surface settings transferred from RB-Y1',
              'episode_ids':[m['stem'] for m in metadata]}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'masks').mkdir()
    start=time.monotonic()
    with (a.out/'frames.jsonl').open('x') as out:
        for idx,meta in enumerate(metadata,1):
            ep=Episode(meta,cameras=CAMERAS,scene_dir=a.scene)
            noun_file=a.noun_run/'noun_rows'/f'{ep.stem}_noun.npz'
            phase_data=None
            if a.gt_attention:
                with np.load(a.gt_attention/f'phase_{ep.stem}.npz',allow_pickle=False) as p:
                    phase_data={key:p[key] for key in ('call_frames','segP/head','segP/left_wrist')}
            with np.load(noun_file,allow_pickle=False) as z:
                attention=z['noun_image']; obs=z['obs_index']; layers=z['layers']; noun_ok=z['noun_ok']
                layer_pos=np.flatnonzero(layers==layer)
                if len(layer_pos)!=1: raise ValueError(f'layer {layer} missing')
                for t in selected[ep.stem]:
                    match=np.flatnonzero(obs==t)
                    if len(match)!=1: raise ValueError(f'attention join {ep.stem}/{t} ambiguous')
                    k=int(match[0])
                    if not noun_ok[k]: raise ValueError(f'noun missing {ep.stem}/{t}')
                    v=attention[k,int(layer_pos[0]),head]
                    fr=ep.frame(t)
                    views=[]
                    for j,c in enumerate(CAMERAS):
                        rob=fr['robot'][c] if a.robot_mask=='gt' else np.zeros_like(fr['robot'][c])
                        # Stored noun maps are policy-oriented. Horizontal unmirror precedes use.
                        if phase_data is None:
                            amap=np.ascontiguousarray(v[j*256:(j+1)*256].reshape(16,16)[:,::-1])
                        else:
                            pidx=np.flatnonzero(phase_data['call_frames']==t)
                            if len(pidx)!=1: raise ValueError(f'GT attention join {ep.stem}/{t}')
                            gt=np.isin(phase_data[f'segP/{c}'][int(pidx[0])],ep.target_geoms)
                            if gt.shape!=(256,256): raise ValueError(f'unexpected GT shape {gt.shape}')
                            amap=gt.reshape(16,16,16,16).mean(axis=(1,3)).astype(np.float32)
                        views.append(SurfaceView(c,fr['depth_m'][c],fr['K'][c],fr['T'][c],rob,amap))
                    dummy_depth=np.zeros_like(fr['depth_m']['head'])
                    dummy_depth[0,0]=1.0  # backproject requires at least one valid pixel; robot mask excludes it
                    views.append(SurfaceView(CAMERA_ORDER[2],dummy_depth,
                                             fr['K']['head'],fr['T']['head'],np.ones_like(fr['robot']['head']),
                                             np.zeros((16,16),np.float32)))
                    result=select_target_surface(views,arm=f'pilot_two_camera_robot_{a.robot_mask}')
                    masks={c:(result.codes[c]==1 if result.codes is not None else np.zeros_like(fr['robot'][c])) for c in CAMERAS}
                    metric=scores(fr,{'surface':masks})['surface']
                    pixel_metrics={c:M.mask_metrics_2d(masks[c],fr['labels'][c]==M.CLASSES.index('target')) for c in CAMERAS}
                    np.savez_compressed(a.out/'masks'/f'{ep.stem}_{t:05d}.npz',**masks)
                    row={'episode_id':ep.stem,'task_id':int(meta['task_id']),'episode_idx':int(meta['episode_idx']),
                         'obs_index':t,'applied':bool(result.record['applied']),'why':result.record.get('why'),
                         'selection':result.record,'metrics':metric,'pixel_metrics':pixel_metrics}
                    out.write(json.dumps(row,allow_nan=False)+'\n');out.flush()
            print(f'{idx}/{len(metadata)} {ep.stem} elapsed={time.monotonic()-start:.1f}s',flush=True)
    print('complete',flush=True)

if __name__=='__main__': main()
