#!/usr/bin/env python3
"""Distinguish full-image robot-mask errors from graph-eligible depth pixels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from AG3S.adapters.panda import model as PM
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.stages.reconstruction import backproject
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes


def main():
    p=argparse.ArgumentParser()
    for k in ('source','scene','phase','frames','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    chosen=defaultdict(list)
    for r in map(json.loads,a.frames.open()):chosen[r['episode_id']].append(r['obs_index'])
    eps={e['stem']:e for e in load_episodes(a.source)}
    if not set(chosen)<=set(eps):raise ValueError('episode mismatch')
    model=PM.build_panda_self_filter_model();cfg=PointCloudConfig();graph_cfg=PointCloudConfig(range_max=2.0)
    total=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    per_task=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(int))));n=0
    for stem,times in chosen.items():
        ep=Episode(eps[stem],scene_dir=a.scene)
        with np.load(eps[stem]['geom_path']) as z:state={c:z[f'robot_state/{c}'] for c in ('head','left_wrist')}
        with np.load(a.phase/f'phase_{stem}.npz') as z:
            phase={k:z[k] for k in ('call_frames','segP/head','segP/left_wrist')}
        for t in times:
            fr=ep.frame(t);pidx=np.flatnonzero(phase['call_frames']==t)
            if len(pidx)!=1:raise ValueError('bad phase join')
            for c in ('head','left_wrist'):
                seg=phase[f'segP/{c}'][pidx[0]];gt_robot=np.isin(seg,ep.robot_geoms);gt_target=np.isin(seg,ep.target_geoms)
                pred=PM.panda_depth_robot_mask(fr['depth_m'][c],fr['K'][c],fr['T'][c],state[c][fr['geom_row']],
                                               model,cfg,device='cpu').mask
                pc=backproject(fr['depth_m'][c],fr['K'][c],fr['T'][c],graph_cfg)
                valid=np.zeros_like(gt_robot);valid[pc.uv[:,1],pc.uv[:,0]]=True
                for scope,eligible in [('full_image',np.ones_like(valid)),('graph_depth_eligible',valid)]:
                    for d in (total[c][scope],per_task[str(eps[stem]['task_id'])][c][scope]):
                        for key,mask in [('tp',pred & gt_robot),('fp',pred & ~gt_robot),('fn',~pred & gt_robot),
                                         ('target',gt_target),('target_deleted',pred & gt_target),('pixels',eligible)]:
                            d[key]+=int((mask & eligible).sum())
            n+=1
        print(stem,n,flush=True)
    def format_scopes(camera_data):
        out={}
        for c,scopes in camera_data.items():
            out[c]={}
            for name,d in scopes.items():
                d=dict(d);d['precision']=d['tp']/(d['tp']+d['fp']) if d['tp']+d['fp'] else None
                d['recall']=d['tp']/(d['tp']+d['fn']) if d['tp']+d['fn'] else None
                d['target_deleted_fraction']=d['target_deleted']/d['target'] if d['target'] else None
                out[c][name]=d
        return out
    out=format_scopes(total)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump({'frames':n,'camera_pixel_micro':out,
        'per_task_pixel_micro':{t:format_scopes(d) for t,d in per_task.items()},
        'note':'graph-eligible = finite z 0.05..3m and radial range <=2m; FK filter uses its default depth gate'},f,indent=2);f.write('\n')


if __name__=='__main__':main()
