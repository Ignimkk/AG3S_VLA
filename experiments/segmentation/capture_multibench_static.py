#!/usr/bin/env python3
"""Capture synchronized static RGB / instance GT / target-only visibility.

Runs in the benchmark's own Python environment. Target-only render hides other
geom groups without moving or stepping the simulator; it is evaluation-only.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
import traceback

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmark/runners'))
import libero_pi05_client as base

PHRASE=re.compile(r'^\s*(?:pick up|pick|grab|pass me) the (?P<target>.+?)(?: and (?:place|put|bring|cut)\b.*|\s*[.!]?\s*$)',re.I)
CAMERAS=(('agentview','head'),('robot0_eye_in_hand','wrist'))


def capture(env,target_ids):
    sim=env.sim;model=sim.model
    sim.forward()
    state_before=np.array(sim.get_state().flatten(),copy=True)
    raw={};visible={};solo={};rgb_after={};seg_after={}
    for cam,name in CAMERAS:
        raw[name]=np.asarray(sim.render(camera_name=cam,height=256,width=256))[::-1].copy()
        z=np.asarray(sim.render(camera_name=cam,height=256,width=256,segmentation=True))[::-1]
        visible[name]=(z[...,0]==5)&np.isin(z[...,1],target_ids)
    groups=np.asarray(model.geom_group).copy()
    try:
        model.geom_group[:]=5
        model.geom_group[target_ids]=groups[target_ids]
        for cam,name in CAMERAS:
            z=np.asarray(sim.render(camera_name=cam,height=256,width=256,segmentation=True))[::-1]
            solo[name]=(z[...,0]==5)&np.isin(z[...,1],target_ids)
    finally:
        model.geom_group[:]=groups
    for cam,name in CAMERAS:
        rgb_after[name]=np.asarray(sim.render(camera_name=cam,height=256,width=256))[::-1].copy()
        z=np.asarray(sim.render(camera_name=cam,height=256,width=256,segmentation=True))[::-1]
        seg_after[name]=(z[...,0]==5)&np.isin(z[...,1],target_ids)
        if not np.array_equal(raw[name],rgb_after[name]) or not np.array_equal(visible[name],seg_after[name]):
            raise ValueError('restored scene render mismatch '+name)
        if np.any(visible[name]&~solo[name]):raise ValueError('visible target not subset of target-only '+name)
    if not np.array_equal(state_before,np.asarray(sim.get_state().flatten())):
        raise ValueError('sim state changed during rendering')
    return raw,visible,solo


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=('libero','safelibero','libero_safety'),required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--limit-scenes',type=int,default=0)
    p.add_argument('--init',type=int,default=0)
    args=p.parse_args()
    if args.benchmark=='safelibero':
        import libero_safe_client as safe
        safe.activate_safelibero()
    elif args.benchmark=='libero_safety':
        import libero_safety_client as lsc
        lsc.activate_libero_safety()
    else:base.require_libero_config()
    from libero.libero import benchmark
    from benchmark.adapters.libero_panda import task_target as tt
    if args.benchmark=='libero':
        specs=[(suite,None,t) for suite in ('libero_spatial','libero_object') for t in (0,3,6,9)]
    elif args.benchmark=='safelibero':
        specs=[(suite,level,t) for suite in ('safelibero_spatial','safelibero_object') for level in ('I','II') for t in (0,2)]
    else:
        specs=[(suite,level,t) for suite in ('affordance','human_safety','obstacle_avoidance','obstacle_avoidance_human')
               for level in (0,1,2) for t in (0,1)]
    if args.limit_scenes:specs=specs[:args.limit_scenes]
    args.out.mkdir(parents=True,exist_ok=False)
    meta={'benchmark':args.benchmark,'planned':len(specs),'seed':7,'init':args.init,
          'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
          'target':'BDDL goal used to generate evaluation GT only','phrase':'regex reads task language only',
          'occlusion':'1-visible / same-state target-only in-frame geom segmentation',
          'cameras':[x[1] for x in CAMERAS]}
    (args.out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    records=[]
    for suite,level,tid in specs:
        ident='{}_{}_t{}'.format(suite,'L'+str(level) if level is not None else 'plain',tid)
        if args.init:ident+='_init{}'.format(args.init)
        env=None;t0=time.time()
        try:
            cls=benchmark.get_benchmark_dict()[suite]
            if args.benchmark=='libero_safety':so=lsc.LevelSuite(cls(),level)
            elif args.benchmark=='safelibero':so=cls(safety_level=level)
            else:so=cls()
            task=so.get_task(tid);init=so.get_task_init_states(tid)
            np.random.seed(7);env,_=base.make_task_env(task,7,False)
            if not 0<=args.init<len(init):raise ValueError('init index outside suite states')
            obs=base.start_episode(env,init,args.init)
            wait=20 if args.benchmark=='safelibero' else 10
            for _ in range(wait):obs,_,_,_=env.step(base.LIBERO_DUMMY_ACTION)
            inner=getattr(env,'env',env)
            if args.benchmark=='libero_safety':
                problem=inner.parsed_problem;objs=set(inner.objects_dict)
                targets=[str(g[1]) for g in problem['goal_state'] if len(g)>1 and str(g[1]) in objs]
                if not targets:raise ValueError('no movable target in first goal')
                target=targets[0]
                if len(set(targets))!=1:raise ValueError('multi-target goal requires union evaluation')
            else:
                bddl=(safe.SAFELIBERO_ROOT/'libero/libero/bddl_files' if args.benchmark=='safelibero'
                      else ROOT/'benchmark/LIBERO/libero/libero/bddl_files')/task.problem_folder/task.bddl_file
                target,_=tt.goal_roles(tt.parse_bddl(bddl))
            model=env.sim.model;names=[model.body_id2name(i) or '' for i in range(model.nbody)]
            target_ids=np.asarray(tt.instance_geom_ids(target,model.geom_bodyid,names),dtype=np.int64)
            language=str(task.language);match=PHRASE.fullmatch(language)
            phrase=match.group('target').strip() if match else None
            rgb,vis,solo=capture(env,target_ids)
            data={}
            for c in ('head','wrist'):
                data[c+'_rgb']=rgb[c];data[c+'_gt']=vis[c];data[c+'_solo']=solo[c]
            np.savez_compressed(args.out/(ident+'.npz'),**data)
            rec={'id':ident,'benchmark':args.benchmark,'suite':suite,'level':level,'task_id':tid,
                 'language':language,'phrase_from_instruction':phrase,'target_gt':target,'init':args.init,
                 'target_geom_count':len(target_ids),'status':'ok','seconds':time.time()-t0,
                 'cameras':{c:{'visible':int(vis[c].sum()),'solo_in_frame':int(solo[c].sum()),
                               'occlusion':float(1-vis[c].sum()/solo[c].sum()) if solo[c].any() else None}
                            for c in ('head','wrist')}}
        except Exception as exc:
            rec={'id':ident,'benchmark':args.benchmark,'suite':suite,'level':level,'task_id':tid,
                 'status':'error','error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.time()-t0}
        finally:
            if env is not None:env.close()
        records.append(rec);print(json.dumps(rec),flush=True)
    (args.out/'scenes.json').write_text(json.dumps(records,indent=2)+'\n')
    print('completed',len(records),'scenes',sum(r['status']=='ok' for r in records),'ok',flush=True)


if __name__=='__main__':main()
