"""Plot computed mask arrays on re-rendered RGB, with identical crop for each column."""
from pathlib import Path
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
p=Path(__file__).resolve().parent
font=fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name()
rgb=plt.imread(p/'wrist_cam_l_rgb.png')[:,:,:3]
z=np.load(p/'wrist_cam_l_masks.npz');sph,add=z['sphere'],z['added']
pre=rgb.copy();pre[sph]=.28*rgb[sph]+.72*np.array([50,120,245])/255
post=pre.copy();post[add]=.20*rgb[add]+.80*np.array([255,150,30])/255
fig,axs=plt.subplots(2,3,figsize=(15,9),gridspec_kw={'height_ratios':[3,1.15]},facecolor='#F7F9FC')
for i,im in enumerate([rgb,pre,post]):
 axs[0,i].imshow(im);axs[0,i].add_patch(Rectangle((135,345),475,134,fill=False,ec='#19BDB1',lw=1.5));axs[0,i].axis('off')
 axs[0,i].set_title(['마스크 적용 전 RGB 시야','렌더 사용 전: 기존 구 마스크','렌더 추가 후: 기존 구 ∪ 렌더'][i],fontsize=14,pad=13)
 axs[1,i].imshow(im[345:480,135:610]);axs[1,i].axis('off')
 axs[1,i].set_title(['아래쪽 손 주변 확대','파랑: 기존에 제외한 픽셀','주황: 렌더가 추가로 제외한 픽셀'][i],fontsize=12,pad=8)
fig.suptitle('렌더링 방식 사용 전의 로봇 마스크 — 왼손목 카메라',fontsize=23,y=.97)
fig.text(.5,.915,'1345 장면 · step 104 · 같은 기록 depth에서 계산 · RGB는 기록된 자세로 재렌더링',ha='center',fontsize=12,color='#53657A')
fig.text(.5,.105,'사용 전: 구 마스크 11,561 px     →     렌더 추가 후: 17,607 px (추가 6,046 px)',ha='center',fontsize=14)
fig.text(.5,.062,'색칠한 영역은 depth 적분에서 제외됩니다. 원래 RGB 영상의 색이 바뀌는 것은 아닙니다.',ha='center',fontsize=12,color='#53657A')
fig.text(.5,.025,'로봇 마스크는 TrajOpt 충돌 구와 별도입니다. 렌더 depth와 관측 depth가 10 mm 이내로 일치하는 픽셀을 추가합니다.',ha='center',fontsize=11,color='#53657A')
fig.subplots_adjust(left=.025,right=.975,top=.84,bottom=.16,wspace=.035,hspace=.13)
fig.savefig(p/'robot_mask_wrist_before_after_detail.png',dpi=170,facecolor=fig.get_facecolor())
