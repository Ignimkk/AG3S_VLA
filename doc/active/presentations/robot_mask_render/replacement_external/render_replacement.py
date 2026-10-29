"""Render-only replacement (NOT union) at the prior ep1345 external camera.
RGB/depth are freshly rendered from recorded qpos. Sphere baseline uses the
self-filter model, not TrajOpt's thin collision spheres. Replacement is strictly
RobotRenderMask.masks()[0]; no sphere mask is merged into it.
"""
import sys,json,pathlib,dataclasses
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
P=pathlib.Path('/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581')
sys.path.insert(0,str(P/'verify/AG3S-PRE-CORE-TO/SC/lib'))
from occ_common import Scene
from AG3S.trajopt.experiments import ta_offline as TA
from AG3S.trajopt import serve_safe
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap,openings_from_mujoco
from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
from AG3S.ag3s.stages.reconstruction import backproject
from AG3S.ag3s.stages.robot_filter import depth_robot_mask
from AG3S.ag3s.runtime.robot_render_mask import RobotRenderMask
from AG3S.ag3s.experiments.common import figstyle
out=pathlib.Path(__file__).resolve().parent
previous=out.parents[1]/'palm_shell_distance/scene_render'
g=np.load(previous/'geometry.npz');cp=g['contact'];qpos=g['qpos']
idx=json.loads((P/'verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json').read_text())['runs']['E4/B_ROI_FPR_ep1345_s13453']
args,_=TA.server_args(pathlib.Path(idx['server_segment']['first']).parent)
sc=Scene(1345,13453,'clear',W=1000,H=800);sc.set_q(qpos);m,d=sc.m,sc.d
cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:]=cp+[0,0,.025];cam.distance=.29;cam.azimuth=55;cam.elevation=-20
sc.r.update_scene(d,camera=cam)
cams=sc.r.scene.camera
eye=(np.asarray(cams[0].pos)+np.asarray(cams[1].pos))/2
forward=np.asarray(cams[0].forward,dtype=float).copy();up=np.asarray(cams[0].up,dtype=float).copy()
forward/=np.linalg.norm(forward);up/=np.linalg.norm(up);right=np.cross(forward,up);right/=np.linalg.norm(right)
f=.5*sc.H/np.tan(np.deg2rad(m.vis.global_.fovy)/2)
K=np.array([[f,0,sc.W/2],[0,f,sc.H/2],[0,0,1]])
Twc=np.eye(4);Twc[:3,:3]=np.column_stack([right,-up,forward]);Twc[:3,3]=eye
bid=m.body('base').id;Twb=np.eye(4);Twb[:3,:3]=d.xmat[bid].reshape(3,3);Twb[:3,3]=d.xpos[bid]
Tbc=np.linalg.inv(Twb)@Twc
rgb=sc.r.render().copy();sc.r.enable_depth_rendering();depth=np.asarray(sc.r.render(),float).copy();sc.r.disable_depth_rendering()
# Build each method independently. The replacement only reads the mesh renderer.
qadr={m.joint(i).name:int(m.jnt_qposadr[i]) for i in range(m.njnt)}
rs=np.array([d.qpos[qadr[n]] for n in DEFAULT_RBY1_JOINTS]);openings=openings_from_mujoco(m,d)
rr=RobotRenderMask(args.model_xml)
mask_list,record=rr.masks([dict(name='external_1345',depth_m=depth,K=K,T_base_cam=Tbc,robot_state=rs)],openings=openings)
render_only=mask_list[0].copy();assert render_only is not None
fm=mujoco.MjModel.from_xml_path(str(args.model_xml));fs=TransportScene.attach(fm,mujoco.MjData(fm));robot=build_robot_model(fs)
robot.set_joint_parameters(FingerJointMap(robot).joint_values(openings))
q=np.array([d.qpos[qadr[n]] for n in robot.joint_names])
cfg=PointCloudConfig(range_max=args.range_max,**serve_safe.self_filter_options(args));assert cfg.self_filter_target_guard_radius==0
cloud=backproject(depth,K,Tbc,cfg);sphere_only=depth_robot_mask(cloud,depth.shape,robot,q,cfg,device='cpu').mask
assert np.array_equal(render_only,mask_list[0])
np.savez_compressed(out/'masks.npz',sphere_only=sphere_only,render_only=render_only,depth=depth,K=K,T_base_cam=Tbc,qpos=qpos)
# Same RGB; color only the mask of each independent method.
old=rgb.copy();old[sphere_only]=(.43*rgb[sphere_only]+.57*np.array([45,115,245])).astype(np.uint8)
new=rgb.copy();new[render_only]=(.43*rgb[render_only]+.57*np.array([155,75,225])).astype(np.uint8)
for name,im in [('actual',rgb),('sphere_only',old),('render_only',new)]:plt.imsave(out/f'{name}.png',im)
plt.imsave(out/'render_only_binary.png',render_only,cmap='gray',vmin=0,vmax=1)
figstyle.use_korean();fig,axs=plt.subplots(1,3,figsize=(18,6.9),facecolor='#F7F9FC')
uv=K@np.linalg.inv(Twc)[:3]@np.r_[cp,1];uv=uv[:2]/uv[2]
for ax,im,title in zip(axs,[rgb,old,new],['실제 씬  |  기록된 손–오렌지 접촉','기존 방식  |  구 마스크만','대체 방식  |  렌더 마스크만']):
 ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=15,pad=14)
 ax.plot(*uv,marker='x',ms=9,mew=2,color='#FF4848',zorder=10)
 ax.annotate('접촉 부위',xy=uv,xytext=(.08,.25),textcoords='axes fraction',fontsize=11,color='#A32626',bbox=dict(boxstyle='round,pad=.4',fc='white',ec='none',alpha=.92),arrowprops=dict(arrowstyle='->',color='#FF4848',lw=1.7),zorder=11)
fig.suptitle('오렌지 1345 · 기존 구 마스크를 렌더 마스크로 대체',fontsize=22,y=.97)
fig.text(.5,.91,'앞선 손바닥 그림과 동일한 자세·외부 카메라  |  step 104 · 방위각 55° · 고도각 −20° · 거리 0.29 m',ha='center',fontsize=11,color='#53657A')
fig.text(.5,.12,'파랑 = 구 마스크만 사용     보라 = 렌더 마스크만 사용 — 기존 구·껍질 구 마스크를 합치지 않음',ha='center',fontsize=13)
fig.text(.5,.07,'보라색은 로봇 메시를 렌더링해 판정한 제거 영역입니다. 로봇 외곽을 따라가며, 구를 추가하거나 부풀린 모습이 아닙니다.',ha='center',fontsize=11,color='#53657A')
fig.text(.5,.025,'외부 시점에서 마스크 대체를 시각화한 그림입니다. 새 depth를 렌더링해 계산했으며, TrajOpt 충돌 모델이나 실행 설정은 바꾸지 않았습니다.',ha='center',fontsize=10,color='#53657A')
fig.subplots_adjust(left=.015,right=.985,bottom=.18,top=.81,wspace=.025)
fig.savefig(out/'render_only_replacement_ep1345.png',dpi=170,facecolor=fig.get_facecolor())
meta={'qpos_source':str(previous/'geometry.npz'),'camera':{'azimuth':55,'elevation':-20,'distance':.29,'lookat':cam.lookat.tolist(),'width':sc.W,'height':sc.H},'comparison':'sphere-only versus render-only; no union','sphere_model':'self-filter build_robot_model, NOT TrajOpt thin spheres','sphere_count':robot.n_spheres,'sphere_mask_px':int(sphere_only.sum()),'render_only_px':int(render_only.sum()),'sphere_only_exclusive_px':int((sphere_only&~render_only).sum()),'render_only_exclusive_px':int((render_only&~sphere_only).sum()),'renderer':record,'note':'Offline visualization at synthetic external view of recorded pose. Actual robot mask is evaluated in sensor views. No serving configuration changes.'}
(out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False));print(json.dumps(meta),flush=True)
