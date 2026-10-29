"""Explanatory schematic, not an episode reconstruction or measured ESDF.
Sources: AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO-SC.impl.md section 2;
         AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO-SCM.impl.md section 1.
Analytic circular-obstacle distance, identical in both panels. Sphere layout is illustrative.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

font = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'font.size': 12, 'axes.unicode_minus': False, 'svg.fonttype': 'none'})
ink, blue, shell, red = '#172B46', '#2673CE', '#8D4CCE', '#C44440'
fig = plt.figure(figsize=(15, 9), facecolor='#F7F9FC')
fig.text(.055, .94, '손바닥 껍질 구: 같은 거리장, 더 넓어진 충돌 검사 범위', fontsize=24, weight='bold', color=ink)
fig.text(.055, .895, '추가 이유  |  손바닥 겉면이 기존 구 밖에 있으면, 실제 접촉을 거리 검사에서 놓칠 수 있습니다.', fontsize=14, color=ink)
fig.text(.055, .852, '설명용 2D 개념도 · 구의 수와 배치는 단순화 · 1345 장면의 실측 거리장이나 재현 사진이 아닙니다.', fontsize=11, color='#657489')

xx, yy = np.meshgrid(np.linspace(-72, 121, 650), np.linspace(-65, 105, 550))
dist = np.hypot(xx-69, yy-35)-30
axes = [fig.add_axes([.055, .29, .425, .50]), fig.add_axes([.52, .29, .425, .50])]
for k, ax in enumerate(axes):
    ax.set_aspect('equal'); ax.set_xlim(-70, 119); ax.set_ylim(-62, 101); ax.axis('off')
    ax.set_title(['사용 전  |  가운데 축의 기존 구', '사용 후  |  기존 구 + 바깥 껍질 구'][k], loc='left', fontsize=17, color=ink, pad=16, weight='bold')
    # Same analytic obstacle distance field and same robot pose in both panels.
    ax.contourf(xx, yy, dist, levels=[0, 5, 10, 20, 40, 60, 200], colors=['#FCE1BF','#FDEBD4','#FFF1DF','#FFF6E9','#FFF9F1','#FFFCF7'], zorder=0)
    cs = ax.contour(xx, yy, dist, levels=[5, 10, 20, 40, 60], colors='#D5B88B', linewidths=.9, zorder=1)
    ax.clabel(cs, levels=[10, 20, 40, 60], fmt=lambda v: f'{v:.0f}', fontsize=9, inline=True)
    ax.add_patch(Rectangle((-40,10),80,50, facecolor='#CFD7DF', edgecolor='#526175', alpha=.72, linewidth=2, zorder=2))
    ax.text(-38, 65, '실제 손바닥 외곽', fontsize=11, color='#526175')
    ax.add_patch(Circle((69,35),30,facecolor='#EF9B38',edgecolor='#BC6819',lw=1.8,zorder=3))
    ax.text(75,35,'이웃\n오렌지',ha='center',va='center',color='#663A0F',fontsize=13,zorder=6)
    ax.text(67, -2, '장애물 표면 d = 0',ha='center',fontsize=10,color='#966023')
    for x in np.linspace(-22,22,9):
        ax.add_patch(Circle((x,35),3,facecolor='#84B8EE',edgecolor=blue,lw=1.3,zorder=4))
    ax.plot([-22,22],[35,35],'.',color=blue,ms=3,zorder=5)
    if k:
        # Outer surfaces only; no added sphere crosses the illustrative grasp floor y=10.
        centers = [(x,56) for x in np.arange(-32,33,8)] + [(s,y) for s in [-36,36] for y in [17,26,35,44,53]]
        for c in centers:
            ax.add_patch(Circle(c,6,fc='#B995E0',ec=shell,alpha=.62,lw=1.2,zorder=4))
        ax.add_patch(Circle((36,35),11,fill=False,ec=shell,lw=1.8,ls='--',zorder=5))
        ax.annotate('破線: 껍질 구 표면에서\n5 mm 여유'.replace('破線','점선'),xy=(36,46),xytext=(69,85),ha='center',fontsize=10,color=shell,arrowprops=dict(arrowstyle='-',color=shell),zorder=7)
        ax.annotate('껍질 구가 접촉 부위까지 검사',xy=(41,35),xytext=(-62,84),fontsize=11,color=shell,arrowprops=dict(arrowstyle='->',color=shell,connectionstyle='arc3,rad=-.2'),zorder=7)
    else:
        ax.annotate('실제 접촉 부위가 구 밖에 있음',xy=(40,35),xytext=(-62,84),fontsize=11,color=red,arrowprops=dict(arrowstyle='->',color=red,connectionstyle='arc3,rad=-.2'),zorder=7)
    ax.plot(40,35,'x',color=red,ms=9,mew=2,zorder=8)
    ax.plot([-32,32],[10,10],color='#26957F',lw=3,zorder=5)
    ax.add_patch(Circle((0,-26),19,facecolor='#E5F2E7',edgecolor='#579568',ls='--',lw=1.4,zorder=3))
    ax.text(0,-26,'목표 사과',ha='center',va='center',fontsize=11,color='#426E4C')
    ax.text(0,-53,'손가락 사이 파지 공간\n손끝 쪽 바닥면은 껍질로 덮지 않음',ha='center',va='center',fontsize=10,color='#267C69')
    ax.text(48,-54,'배경 등고선 = 장애물까지 거리\n두 패널 동일 · 단위 mm',fontsize=9,color='#856F50')

fig.text(.055,.252,'기존 구와 장애물 사이에는 간격이 남음',fontsize=15,color=blue,weight='bold')
fig.text(.52,.252,'추가 구에서 침범 또는 여유 부족이 드러남',fontsize=15,color=shell,weight='bold')
fig.text(.055,.215,'겉면을 놓친 모델은 실제 손바닥 접촉을 놓칠 수 있음',fontsize=11,color='#53657A')
fig.text(.52,.215,'거리장을 바꾸지 않고, 추가 구 중심에서도 거리를 조회',fontsize=11,color='#53657A')
fig.add_artist(Line2D([.055,.945],[.185,.185],transform=fig.transFigure,color='#D4DDE7',lw=1))
fig.text(.055,.143,'검사식',fontsize=13,color=ink,weight='bold')
fig.text(.14,.143,'여유 값 = d(구 중심) − 구 반지름 − 요구 여유     |     0 미만이면 제약 위반',fontsize=14,color=ink)
fig.text(.055,.10,'껍질 구의 장애물 요구 여유 = 5 mm (해당 설정) · 기존 구의 여유는 유지 · 목표 사과를 제외한 손 전용 층을 예시',fontsize=11,color='#53657A')
fig.text(.055,.058,'MuJoCo 물리 형상과 카메라 관측은 그대로입니다.  구 형상의 바깥 넘침(overflow)과 장애물 여유(margin)는 별도 값입니다.',fontsize=11,color='#53657A')
out = Path(__file__).resolve().parent
fig.savefig(out/'palm_shell_distance_before_after.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(out/'palm_shell_distance_before_after.svg',facecolor=fig.get_facecolor())
plt.close(fig)
print(out/'palm_shell_distance_before_after.png')
