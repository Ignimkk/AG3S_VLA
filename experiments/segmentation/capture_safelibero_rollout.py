#!/usr/bin/env python3
"""Re-render every recorded policy observation; reject any RGB misalignment."""
import hashlib,json,pathlib,sys,time,traceback
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmark/runners'))
import libero_pi05_client as base
import libero_safe_client as safe
from openpi_client import image_tools

RUN=pathlib.Path('/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only')
OUT=ROOT/'outputs/continuous_rollout_20261102/safelibero_capture'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 safe.activate_safelibero()
 from libero.libero import benchmark
 from benchmark.adapters.libero_panda import task_target as tt
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'manifest.json').write_text(json.dumps({'run':str(RUN),'run_manifest_sha256':sha(RUN/'run_manifest.json'),
   'tasks':[0,1,2],'episode':0,'seed':7,'safety_level':'I','camera':'head','all_recorded_frames':True,
   'source_script_sha256':sha(pathlib.Path(__file__)),'same_state_check':'RGB bit-identical every frame, target only contains visible, state invariant',
   'orientation':'record RGB and segmentation each [::-1,::-1] as source policy input; RGB resized with openpi resize_with_pad'},indent=2)+'\n')
 suite=benchmark.get_benchmark_dict()['safelibero_spatial'](safety_level='I');rows=[]
 for tid in (0,1,2):
  env=None;stem='safelibero_spatial_t{}_e0'.format(tid);t0=time.time()
  try:
   task=suite.get_task(tid);env,_=base.make_task_env(task,7,False);base.start_episode(env,suite.get_task_init_states(tid),0)
   model=env.sim.model;names=[model.body_id2name(i) or '' for i in range(model.nbody)]
   bddl=safe.SAFELIBERO_ROOT/'libero/libero/bddl_files'/task.problem_folder/task.bddl_file
   target,_=tt.goal_roles(tt.parse_bddl(bddl));ids=np.asarray(tt.instance_geom_ids(target,model.geom_bodyid,names),np.int64)
   data=RUN/'episodes'/(stem+'.npz');geom=RUN/'episodes'/(stem+'_geom.npz')
   with np.load(data,allow_pickle=False) as z:
    rgb_saved=z['obs_agentview_image'];actions=z['action'];done=z['done']
   with np.load(geom,allow_pickle=False) as z:
    states=z['sim_state_image'];indices=z['obs_index'];record_names=list(z['model_body_names']);record_bodyids=z['model_geom_bodyid'];stamps=z['stamp/head']
   if names!=record_names or not np.array_equal(model.geom_bodyid,record_bodyids):raise ValueError('body/geom ID table mismatch')
   if not np.array_equal(indices,np.arange(len(states))) or len(states)!=len(rgb_saved):raise ValueError('nonconsecutive source frame indices')
   imgs=[];segs=[];solos=[];mismatches=[];restoration_errors=[]
   for i,state in enumerate(states):
    sim=env.sim;sim.set_state_from_flattened(state.copy());sim.forward();before=np.array(sim.get_state().flatten(),copy=True)
    rgb=sim.render(camera_name='agentview',height=256,width=256)
    if not np.array_equal(rgb,rgb_saved[i]):mismatches.append(i);raise ValueError('RGB mismatch frame {}'.format(i))
    seg=sim.render(camera_name='agentview',height=256,width=256,segmentation=True)
    isgeom=seg[...,0]==5;target_vis=isgeom & np.isin(seg[...,1],ids)
    groups=np.asarray(model.geom_group).copy()
    try:
     model.geom_group[:]=5;model.geom_group[ids]=groups[ids]
     solo=sim.render(camera_name='agentview',height=256,width=256,segmentation=True)
     solo=(solo[...,0]==5)&np.isin(solo[...,1],ids)
    finally:model.geom_group[:]=groups
    if np.any(target_vis&~solo) or not np.array_equal(before,np.asarray(sim.get_state().flatten())):
     restoration_errors.append(i);raise ValueError('target-only or state invariant failed at {}'.format(i))
    restored=sim.render(camera_name='agentview',height=256,width=256,segmentation=True)
    if not np.array_equal(seg,restored):raise ValueError('geom restore mismatch frame {}'.format(i))
    lab=np.where(isgeom,seg[...,1]+2,0).astype(np.int16);lab[target_vis]=1
    imgs.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb[::-1,::-1],224,224)))
    segs.append(lab[::-1,::-1]);solos.append(solo[::-1,::-1])
    if (i+1)%50==0:print(stem,i+1,'/',len(states),flush=True)
   out=OUT/('safelibero_spatial_task{}_ep0.npz'.format(tid))
   np.savez_compressed(out,prompt=str(task.language),suite='safelibero_spatial',task_id=tid,episode=0,
      success=bool(done.any()),id2name=np.array(['0:background','1:'+target]),manipulated=np.array([target]),
      image=np.stack(imgs),seg_full=np.stack(segs),solo_full=np.stack(solos),t=np.arange(len(states)),
      sim_stamp=stamps,action=actions)
   row={'id':stem,'status':'ok','frames':len(states),'source_episode_sha256':sha(data),'source_geom_sha256':sha(geom),
        'saved_sha256':sha(out),'rgb_equal':len(states),'mismatches':mismatches,'restoration_errors':restoration_errors,
        'target_gt':target,'language':str(task.language),'source_success':bool(done.any()),'seconds':time.time()-t0}
  except Exception as exc:
   row={'id':stem,'status':'error','error':repr(exc),'traceback':traceback.format_exc()};print(stem,'ERROR',repr(exc),flush=True)
  finally:
   if env is not None:env.close()
  rows.append(row);(OUT/'sequences.json').write_text(json.dumps(rows,indent=2)+'\n');print(stem,row['status'],flush=True)
 if any(r['status']!='ok' for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
