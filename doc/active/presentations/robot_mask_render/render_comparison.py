"""Recompute pre-render sphere self-filter and added render mask on recorded depth.
Uses episode 1345 step 104, recorded K/T/depth, real serving mask functions.
RGB is re-rendered at the recorded qpos. No TSDF integration or rollout.
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
idx=json.loads((P/'verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json').read_text())['runs']['E4/B_ROI_FPR_ep1345_s13453']
cd=pathlib.Path(idx['run_dir']);src=cd/'ag3s/run_0000/step_00013.npz';st=np.load(src)
args,_=TA.server_args(pathlib.Path(idx['server_segment']['first']).parent)
print('Loading scene and self-filter model',flush=True)
sc=Scene(1345,13453,'clear',W=640,H=480);sc.set_q(st['qpos']);m,d=sc.m,sc.d
fm=mujoco.MjModel.from_xml_path(str(args.model_xml));fs=TransportScene.attach(fm,mujoco.MjData(fm));robot=build_robot_model(fs)
openings=openings_from_mujoco(m,d);fmap=FingerJointMap(robot);robot.set_joint_parameters(fmap.joint_values(openings))
qadr={m.joint(i).name:int(m.jnt_qposadr[i]) for i in range(m.njnt)}
q=np.array([d.qpos[qadr[n]] for n in robot.joint_names]);rs=np.array([d.qpos[qadr[n]] for n in DEFAULT_RBY1_JOINTS])
cfg=PointCloudConfig(range_max=args.range_max,**serve_safe.self_filter_options(args))
assert cfg.self_filter_target_guard_radius==0, 'Need frame-specific target guard before proceeding'
rr=RobotRenderMask(args.model_xml)
views=[]
for cam in ['zed_left','wrist_cam_l']:
 views.append(dict(name=cam,depth_m=st[f'depth_{cam}'].astype(float)/1000,K=st[f'K_{cam}'],T_base_cam=st[f'T_base_cam_{cam}'],robot_state=rs))
rms,rec=rr.masks(views,openings=openings)
figstyle.use_korean();fig,axs=plt.subplots(2,3,figsize=(16,9.6),facecolor='#F7F9FC')
meta={'source':str(src),'code_snapshot':str(P/'snapshot_sc'),'self_filter_spheres':robot.n_spheres,'pointcloud_config':dataclasses.asdict(cfg),'render':rec,'cameras':{},'note':'Same recorded depth/K/T used for both masks; full-resolution RGB re-rendered at recorded qpos. This is an offline mask comparison, not a closed-loop rerun.'}
for row,(view,rm) in enumerate(zip(views,rms)):
 cam=view['name'];dep=view['depth_m'];cloud=backproject(dep,view['K'],view['T_base_cam'],cfg)
 sph=depth_robot_mask(cloud,dep.shape,robot,q,cfg,device='cpu').mask
 add=rm&~sph;union=sph|rm
 rgb=sc.render(cam);before=rgb.copy();after=rgb.copy()
 before[sph]=(.28*rgb[sph]+.72*np.array([50,120,245])).astype(np.uint8)
 after[sph]=before[sph];after[add]=(.20*rgb[add]+.80*np.array([255,150,30])).astype(np.uint8)
 for col,im in enumerate([rgb,before,after]):
  ax=axs[row,col];ax.imshow(im);ax.axis('off')
  if row==0:ax.set_title(['카메라 원본 시야','렌더 사용 전  |  기존 구 마스크','렌더 추가 후  |  구 ∪ 렌더 마스크'][col],fontsize=15,pad=12)
  plt.imsave(out/f'{cam}_{["rgb","before","after"][col]}.png',im)
 label='머리 카메라' if row==0 else '왼손목 카메라'
 axs[row,0].text(.01,.02,label,transform=axs[row,0].transAxes,color='white',fontsize=12,bbox=dict(fc='#172B46',ec='none',alpha=.85,pad=5))
 axs[row,1].text(.01,.02,f'구 마스크: {sph.sum():,} px',transform=axs[row,1].transAxes,color='white',fontsize=11,bbox=dict(fc='#172B46',ec='none',alpha=.85,pad=5))
 axs[row,2].text(.01,.02,f'렌더로 추가: {add.sum():,} px',transform=axs[row,2].transAxes,color='white',fontsize=11,bbox=dict(fc='#172B46',ec='none',alpha=.85,pad=5))
 np.savez_compressed(out/f'{cam}_masks.npz',sphere=sph,render=rm,added=add,union=union,depth=dep,K=view['K'],T_base_cam=view['T_base_cam'])
 meta['cameras'][cam]={'sphere_px':int(sph.sum()),'render_px':int(rm.sum()),'added_px':int(add.sum()),'union_px':int(union.sum())}
fig.suptitle('로봇 렌더 마스크 사용 전: 구 마스크가 제외하던 픽셀',fontsize=23,y=.975)
fig.text(.5,.925,'1345 장면 · step 104 · 같은 관측 depth / 카메라 / 로봇 자세로 마스크를 다시 계산',ha='center',fontsize=12,color='#53657A')
fig.text(.5,.09,'파랑 = 기존 구 마스크가 제외하는 픽셀     주황 = 렌더링 방식이 추가로 제외하는 픽셀',ha='center',fontsize=14)
fig.text(.5,.05,'색은 마스크 위치를 보여주는 표시입니다. 실제 처리에서는 해당 depth 픽셀을 TSDF 적분에서 제외하며, 빈 공간으로 채우지 않습니다.',ha='center',fontsize=11,color='#53657A')
fig.text(.5,.018,'렌더 판정: 관측 depth와 로봇만 렌더링한 depth의 차이 ≤ 10 mm. 이 마스크는 TrajOpt의 충돌 검사 구와 별도입니다.',ha='center',fontsize=11,color='#53657A')
fig.subplots_adjust(left=.02,right=.98,bottom=.14,top=.86,wspace=.025,hspace=.035)
fig.savefig(out/'robot_mask_before_after_ep1345.png',dpi=160,facecolor=fig.get_facecolor())
(out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
print(json.dumps(meta['cameras']),flush=True)
