"""Re-render recorded episode 1345 request 13; shell geometry from snapshot_sc.
No rollout, optimization, or physics changes. Only rendering overlays/transparency vary.
"""
import os,sys,json,pathlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
P=pathlib.Path('/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581')
sys.path.insert(0,str(P/'verify/AG3S-PRE-CORE-TO/SC/lib'))
from AG3S.trajopt.experiments import ta_offline as TA
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap,openings_from_mujoco
from AG3S.ag3s.experiments.common import figstyle
from occ_common import Scene
out=pathlib.Path(__file__).resolve().parent
idx=json.loads((P/'verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json').read_text())['runs']['E4/B_ROI_FPR_ep1345_s13453']
cd=pathlib.Path(idx['run_dir']); chunk_dir=pathlib.Path(idx['server_segment']['first']).parent
if '--cached' in sys.argv:
 sc=Scene(1345,13453,'clear',W=1000,H=800)
 z=np.load(out/'geometry.npz'); sc.set_q(z['qpos']); m,d=sc.m,sc.d
 cen,rad,old,shell,cp=z['centers'],z['radii'],z['old'].tolist(),z['shell'].tolist(),z['contact']
 bid=m.body('EE_BODY_L').id
else:
 args,_=TA.server_args(chunk_dir,('--palm-shell','left'))
 print('building shell model',flush=True)
 robot,qadr=TA.build_robot(args)
 print('building scene',robot.n_spheres,flush=True)
 sc=Scene(1345,13453,'clear',W=1000,H=800)
 st=np.load(cd/'ag3s/run_0000/step_00013.npz'); sc.set_q(st['qpos'])
 m,d=sc.m,sc.d
 fmap=FingerJointMap(robot)
 robot.set_joint_parameters(fmap.joint_values(openings_from_mujoco(m,d)))
 q=np.array([d.qpos[qadr[n]] for n in robot.joint_names],float)
 cen,rad=robot.sphere_centers_numeric(q)
 links=list(robot.sphere_link_names); roles=list(robot.sphere_roles)
 ids=[i for i,l in enumerate(links) if l=='ee_left']
 old=[i for i in ids if roles[i]!='palm_shell']; shell=[i for i in ids if roles[i]=='palm_shell']
 bid=m.body('EE_BODY_L').id; orange=m.geom('orange_geom').id
 hand=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and (m.geom_contype[g] or m.geom_conaffinity[g])]
 best=None
 for g in hand:
  ft=np.zeros(6); distance=mujoco.mj_geomDistance(m,d,g,orange,0.2,ft)
  if best is None or distance<best[0]: best=(distance,ft.copy(),g)
 cp=best[1][:3]; base=m.body('base').id if 'base' in [m.body(i).name for i in range(m.nbody)] else 1
 print('contact',best[0],cp,'old',len(old),'shell',len(shell),flush=True)
 meta={'episode':1345,'seed':13453,'request':13,'t_step':int(st['t_step']),'qpos_source':str(cd/'ag3s/run_0000/step_00013.npz'),'model_source':str(P/'snapshot_sc'),'old_palm_spheres':len(old),'added_left_shell_spheres':len(shell),'total_robot_spheres':len(rad),'contact_distance_mm':float(best[0]*1000),'contact_point_world':cp.tolist(),'contact_point_to_old_spheres_mm':float(np.min(np.linalg.norm(cen[old]-cp,axis=1)-rad[old])*1000),'contact_point_to_shell_spheres_mm':float(np.min(np.linalg.norm(cen[shell]-cp,axis=1)-rad[shell])*1000),'note':'Same recorded qpos and scene in all panels. Render-only alpha and sphere overlays. No optimization or ESDF recomputation.'}
 (out/'render_metadata.json').write_text(json.dumps(meta,indent=2))
 np.savez(out/'geometry.npz',centers=cen,radii=rad,old=old,shell=shell,contact=cp,qpos=st['qpos'])

def render(az,el,variant='actual',distance=.29):
 cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 cam.lookat[:]=cp+np.array([0,0,.025]); cam.distance=distance; cam.azimuth=az;cam.elevation=el
 sc.r.update_scene(d,camera=cam); scene=sc.r.scene
 if variant!='actual':
  for g in scene.geoms[:scene.ngeom]:
   if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0 and m.geom_bodyid[g.objid]==bid:
    g.rgba[3]=.22
  for i in old+(shell if variant=='shell' else []):
   color=[.1,.4,1.,.95] if i in old else [.53,.18,.88,.60]
   mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([rad[i],0,0]),cen[i],np.eye(3).ravel(),np.array(color,np.float32));scene.ngeom+=1
 for pos in [cp]:
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.0025,0,0]),pos,np.eye(3).ravel(),np.array([1,.08,.07,1],np.float32));scene.ngeom+=1
 cams=scene.camera
 eye=(np.asarray(cams[0].pos)+np.asarray(cams[1].pos))/2
 forward=np.asarray(cams[0].forward);up=np.asarray(cams[0].up);right=np.cross(forward,up)
 delta=cp-eye;dep=float(delta@forward);focal=.5*sc.H/np.tan(np.deg2rad(m.vis.global_.fovy)/2)
 render.contact_pixel=(sc.W/2+focal*float(delta@right)/dep,sc.H/2-focal*float(delta@up)/dep)
 return sc.r.render().copy()

figstyle.use_korean()
if '--preview' in sys.argv:
 fig,axs=plt.subplots(2,4,figsize=(16,7))
 for ax,(az,el) in zip(axs.flat,[(0,-25),(60,-25),(120,-25),(180,-25),(240,-25),(300,-25),(120,-60),(240,-60)]):
  ax.imshow(render(az,el,'old'));ax.set_title(f'{az}, {el}');ax.axis('off')
 fig.tight_layout();fig.savefig(out/'camera_preview.png',dpi=100)
else:
 az=float(os.environ.get('PALM_RENDER_AZ','55'));el=float(os.environ.get('PALM_RENDER_EL','-20'))
 fig,axs=plt.subplots(1,3,figsize=(18,6.9),facecolor='#F7F9FC')
 titles=['실제 씬  |  기록된 손–오렌지 접촉','사용 전  |  기존 손바닥 구 23개','사용 후  |  껍질 구 168개 추가']
 for ax,variant,title in zip(axs,['actual','old','shell'],titles):
  im=render(az,el,variant);plt.imsave(out/f'{variant}.png',im);ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=14,pad=14)
  u,v=render.contact_pixel
  ax.plot(u,v,marker='x',ms=9,mew=2,color='#FF4848',zorder=10)
  ax.annotate('접촉 부위',xy=(u,v),xytext=(.08,.25),textcoords='axes fraction',fontsize=11,color='#A32626',bbox=dict(boxstyle='round,pad=.4',fc='white',ec='none',alpha=.92),arrowprops=dict(arrowstyle='->',color='#FF4848',lw=1.7),zorder=11)
 fig.suptitle('오렌지 1345 · 같은 자세와 카메라에서 본 손바닥 충돌 모델',fontsize=22,x=.5,y=.97)
 fig.text(.5,.91,'episode 1345 / seed 13453 / step 104  ·  기록 qpos로 MuJoCo 재렌더링  ·  회피 실행 전후 비교가 아닌 모델 형상 비교',ha='center',fontsize=11,color='#53657A')
 fig.text(.5,.12,'파랑 = 기존 손바닥 축선 구     보라 = 추가 껍질 구     빨간 × = 손바닥–오렌지 최근접점 (가림 없이 표시)',ha='center',fontsize=13)
 fig.text(.5,.07,'중앙·오른쪽은 손바닥만 반투명 표시. 장애물 거리장·물리 형상·자세는 동일하며, 거리장을 조회하는 구의 범위만 달라집니다.',ha='center',fontsize=11,color='#53657A')
 fig.text(.5,.025,'구는 실제 중심·반지름으로 표시했습니다. 5 mm 요구 여유는 구 자체의 크기와 별도이며, 이 그림에는 여유 띠를 표시하지 않았습니다.',ha='center',fontsize=10,color='#53657A')
 fig.subplots_adjust(left=.015,right=.985,bottom=.18,top=.81,wspace=.025)
 fig.savefig(out/'palm_shell_scene_before_after.png',dpi=160,facecolor=fig.get_facecolor())
print('saved',out,flush=True)
