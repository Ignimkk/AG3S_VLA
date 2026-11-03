#!/usr/bin/env python3
"""Move a SafeLIBERO 3D obstacle along camera rays as a controlled stress test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmark/runners'))
import libero_pi05_client as base
import libero_safe_client as safe
from capture_multibench_static import capture,PHRASE


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--limit-scenes',type=int,default=0)
    a=p.parse_args();safe.activate_safelibero()
    from libero.libero import benchmark
    from benchmark.adapters.libero_panda import task_target as tt
    specs=[(s,l,t) for s in ('safelibero_spatial','safelibero_object') for l in ('I','II') for t in (0,2)]
    if a.limit_scenes:specs=specs[:a.limit_scenes]
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'manifest.json').write_text(json.dumps({'planned_original_scenes':len(specs),'offsets_m':[0.,.06],
      'obstacle':'white_storage_box_obstacle_1','method':'place freejoint at 0.85*target + 0.15*head_camera plus horizontal offset',
      'warning':'deliberate test placement may float or intersect objects; not a natural distribution',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    rows=[]
    for suite,level,tid in specs:
        env=None;t0=time.time();name='{}_L{}_t{}'.format(suite,level,tid)
        try:
            so=benchmark.get_benchmark_dict()[suite](safety_level=level);task=so.get_task(tid)
            np.random.seed(7);env,_=base.make_task_env(task,7,False);base.start_episode(env,so.get_task_init_states(tid),0)
            for _ in range(20):env.step(base.LIBERO_DUMMY_ACTION)
            bddl=safe.SAFELIBERO_ROOT/'libero/libero/bddl_files'/task.problem_folder/task.bddl_file
            target,_=tt.goal_roles(tt.parse_bddl(bddl));m=env.sim.model;d=env.sim.data
            names=[m.body_id2name(i) or '' for i in range(m.nbody)]
            ids=np.asarray(tt.instance_geom_ids(target,m.geom_bodyid,names),dtype=np.int64)
            raw,vis,solo=capture(env,ids)
            with np.load(a.reference/(name+'.npz')) as z:
                for cam in ('head','wrist'):
                    if not np.array_equal(raw[cam],z[cam+'_rgb']) or not np.array_equal(vis[cam],z[cam+'_gt']):
                        raise ValueError('reference render differs: '+cam)
            body=tt.instance_body_ids(target,names)[0];cam_id=m.camera_name2id('agentview')
            camera_pos=np.array(d.cam_xpos[cam_id],copy=True);right=np.array(d.cam_xmat[cam_id],copy=True).reshape(3,3)[:,0]
            target_pos=np.array(d.body_xpos[body],copy=True)
            obj=env.env.objects_dict['white_storage_box_obstacle_1'];joint=obj.joints[0]
            initial=np.array(d.get_joint_qpos(joint),copy=True)
            if initial.shape!=(7,):raise ValueError('obstacle freejoint expected')
            phrase=PHRASE.fullmatch(str(task.language));phrase=phrase.group('target').strip() if phrase else None
            for off in (0.,.06):
                q=initial.copy();q[:3]=.85*target_pos+.15*camera_pos+off*right;d.set_joint_qpos(joint,q)
                rgb,visible,solo=capture(env,ids);label='{}_stress_{:02d}'.format(name,int(round(off*100)))
                np.savez_compressed(a.out/(label+'.npz'),**{c+'_'+kind:v[c] for c in ('head','wrist')
                    for kind,v in (('rgb',rgb),('gt',visible),('solo',solo))})
                rows.append({'id':label,'benchmark':'safelibero_controlled','suite':suite,'level':level,'task_id':tid,
                    'status':'ok','language':str(task.language),'phrase_from_instruction':phrase,'target_gt':target,
                    'reference_id':name,'offset_m':off,'obstacle_world_xyz':q[:3].tolist(),
                    'cameras':{c:{'visible':int(visible[c].sum()),'solo_in_frame':int(solo[c].sum()),
                       'occlusion':float(1-visible[c].sum()/solo[c].sum()) if solo[c].any() else None} for c in ('head','wrist')}})
                d.set_joint_qpos(joint,initial);env.sim.forward()
            print(name,'ok',round(time.time()-t0,1),flush=True)
        except Exception as exc:
            rows.append({'id':name,'status':'error','error':repr(exc),'traceback':traceback.format_exc()});print(name,'error',repr(exc),flush=True)
        finally:
            if env is not None:env.close()
    (a.out/'scenes.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('completed',sum(r['status']=='ok' for r in rows),'controlled scenes',flush=True)


if __name__=='__main__':main()
