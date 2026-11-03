#!/usr/bin/env python3
"""Run recorded LIBERO policy actions on Safety scenes as an out-of-domain stress test."""
import hashlib,json,pathlib,sys,time,traceback
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmark/runners'))
sys.path.insert(0,str(ROOT/'experiments/segmentation'))
import libero_pi05_client as base
import libero_safety_client as lsc
from capture_multibench_static import capture
from openpi_client import image_tools

SOURCE=pathlib.Path('/mnt/dev/AG3S_VLA-storage/datasets/workspace/knows_p0b')
OUT=ROOT/'outputs/continuous_rollout_20261102/libero_safety_capture'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 lsc.activate_libero_safety()
 from libero.libero import benchmark
 from benchmark.adapters.libero_panda import task_target as tt
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'manifest.json').write_text(json.dumps({'benchmark':'LIBERO-Safety','source':'off-task recorded LIBERO actions, no safety policy claims',
  'suites':['affordance','human_safety','obstacle_avoidance'],'level':0,'task':0,'init':0,'seed':7,'wait_steps':10,
  'script_sha256':sha(pathlib.Path(__file__)),'camera':'head','no_gt_in_action_or_segmentation_model':True},indent=2)+'\n')
 rows=[]
 for ix,suite in enumerate(('affordance','human_safety','obstacle_avoidance')):
  name='libero_safety_{}_task0_ep0'.format(suite);env=None;t0=time.time()
  try:
   source=SOURCE/'libero_spatial_task{}_ep0.npz'.format(ix)
   with np.load(source,allow_pickle=False) as z:actions=z['action']
   so=lsc.LevelSuite(benchmark.get_benchmark_dict()[suite](),0);task=so.get_task(0)
   np.random.seed(7);env,_=base.make_task_env(task,7,False);base.start_episode(env,so.get_task_init_states(0),0)
   for _ in range(10):env.step(base.LIBERO_DUMMY_ACTION)
   inner=getattr(env,'env',env);problem=inner.parsed_problem;objs=set(inner.objects_dict)
   targets=[str(g[1]) for g in problem['goal_state'] if len(g)>1 and str(g[1]) in objs]
   if not targets or len(set(targets))!=1:raise ValueError('ambiguous or missing target')
   target=targets[0];m=env.sim.model;names=[m.body_id2name(i) or '' for i in range(m.nbody)]
   ids=np.asarray(tt.instance_geom_ids(target,m.geom_bodyid,names),dtype=np.int64)
   imgs=[];segs=[];solos=[];executed=[];done=False
   for i,act in enumerate(actions):
    rgb,vis,solo=capture(env,ids)
    imgs.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb['head'],224,224)))
    # The goal instance is retained; other geom identities are not asserted in this source.
    lab=np.zeros((256,256),dtype=np.int16);lab[vis['head']]=1
    segs.append(lab);solos.append(solo['head']);executed.append(act)
    _,_,done,_=env.step(np.asarray(act,dtype=float).tolist())
    if (i+1)%50==0:print(name,i+1,'/',len(actions),flush=True)
    if done:break
   out=OUT/(name+'.npz')
   np.savez_compressed(out,prompt=str(task.language),suite=suite,task_id=0,episode=0,success=bool(done),
    id2name=np.array(['0:background_or_other','1:'+target]),manipulated=np.array([target]),image=np.stack(imgs),
    seg_full=np.stack(segs),solo_full=np.stack(solos),t=np.arange(len(imgs)),action=np.stack(executed))
   row={'id':name,'status':'ok','frames':len(imgs),'source_actions':str(source),'source_sha256':sha(source),
        'saved_sha256':sha(out),'target_gt':target,'language':str(task.language),'done':bool(done),'seconds':time.time()-t0}
  except Exception as exc:
   row={'id':name,'status':'error','error':repr(exc),'traceback':traceback.format_exc()};print(name,'ERROR',repr(exc),flush=True)
  finally:
   if env is not None:env.close()
  rows.append(row);(OUT/'sequences.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,row['status'],flush=True)
 if any(r['status']!='ok' for r in rows):raise SystemExit(1)

if __name__=='__main__':main()
