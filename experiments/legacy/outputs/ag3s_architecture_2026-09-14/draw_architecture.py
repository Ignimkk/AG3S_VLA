from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    fm.fontManager.addfont(f)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42})
COL={'blue':('#eaf1f7','#436d91'),'orange':('#fcf1e3','#9e6a2c'),'red':('#f7eaea','#9e5454'),'purple':('#f0edf7','#75628e'),'green':('#fbfdfb','#48745a'),'gray':('#f6f7f8','#545d65')}
TEXT='#17212a'
def canvas(w,h):
    fig,ax=plt.subplots(figsize=(w/10,h/10));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return fig,ax

def txt(ax,x,y,s,size=12,weight='normal',ha='center',va='center',color=TEXT,linespacing=1.5):
    return ax.text(x,y,s,fontsize=size,weight=weight,ha=ha,va=va,color=color,linespacing=linespacing,zorder=5)

def box(ax,x,y,w,h,title,body='',col='gray',fs=13,bs=11,lw=1.3):
    fill,edge=COL[col]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.7',facecolor=fill,edgecolor=edge,linewidth=lw,zorder=2))
    if body:
        txt(ax,x+w/2,y+h*.72,title,fs,'bold')
        txt(ax,x+w/2,y+h*.31,body,bs,linespacing=1.45)
    else: txt(ax,x+w/2,y+h/2,title,fs,'bold')

def arr(ax,points,label=None,lp=None,dash=False,color='#47515a',lw=1.4,size=12):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=size,linewidth=lw,linestyle=(0,(4,3)) if dash else '-',color=color,zorder=3))
    if label: txt(ax,*lp,label,10)

def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

fig,ax=canvas(160,90)
txt(ax,80,85,'VLA + AG3S + trajopt',22,'bold')
txt(ax,80,79.5,'attention 은 무엇이 조작 대상인지만 정한다.  무엇이 충돌할 수 있는지는 기하가 정한다.',12)
box(ax,18,66,124,9,'A. VLA (π0.5)','언어 지시 “put the apple in the basket” + 카메라 영상',fs=15,bs=11)
# AG3S dominates the central area.
box(ax,11,26,138,33,'',col='green',lw=2.8)
txt(ax,18,55.5,'B. AG3S',17,'bold',ha='left',color='#33573f')
box(ax,17,39,37,11,'B1. 무엇이 있나','depth → TSDF → ESDF (거리장)','blue',12,10)
box(ax,61.5,39,37,11,'B2. 어디를 정밀하게','action chunk의 swept volume = ROI','orange',12,9)
box(ax,106,39,37,11,'B3. 무엇이 조작 대상인가','attention → grounding → 잠금(latch)','red',11,9.5)
box(ax,17,28.5,126,7,'B4. 충돌 제약','어느 로봇 구가 · 무엇에 · 얼마나 가까워도 되는가','purple',12,10)
for x,label in [(35.5,'카메라 depth'),(80,'action chunk [50 × 14]'),(124.5,'attention')]:
    arr(ax,[(x,66),(x,50)])
    txt(ax,x,62.5,label,10).set_bbox(dict(facecolor='white',edgecolor='none',pad=2))
arr(ax,[(61.5,44.5),(54,44.5)])
txt(ax,57.8,47.1,'ROI',9)
arr(ax,[(35.5,39),(35.5,35.5)])
arr(ax,[(124.5,39),(124.5,35.5)])
box(ax,40,13,80,8,'C. trajopt','SQP 선형화 → QP → 전 행 재검사',fs=13,bs=10.5)
arr(ax,[(80,28.5),(80,21)])
box(ax,48,2.5,64,6,'D. 안전한 action chunk → 실행',fs=12.5)
arr(ax,[(80,13),(80,8.5)])
arr(ax,[(112,5.5),(155,5.5),(155,44.5),(143,44.5)],dash=True)
txt(ax,133,10,'성공 판정 → 잠금 해제',9)
save(fig,'01_overall')

# A3 portrait proportions. ROI / Geometry / Semantics avoids edge crossings.
fig,ax=canvas(200,282.84)
txt(ax,100,274,'AG3S 내부 아키텍처',25,'bold')
txt(ax,100,265,'depth · 로봇 상태 · action chunk · attention → 궤적 최적화기가 소비하는 충돌 제약',12.5)
principles=[
'01  attention은 조작 대상만 선택한다. 충돌 후보는 기하가 정한다 — 질량이 있으면 후보, attention 문턱 없음.',
'02  접촉 허용은 (로봇 구, 대상) 쌍의 마진이다. 0 mm도 관통 금지 — 익명 거리장에서 대상을 파내지 않는다.\n      거리장은 누가 묻는지 모른다. 대상을 파내면 손끝뿐 아니라 팔꿈치·몸통·반대팔에게도 사라진다.',
'03  미지 링크 · 미지 phase · 미지 source는 모두 full margin으로 처리한다 — fail-closed.',
'04  AG3S는 판정만 한다. phase와 파지 성공은 외부에서 주입받으며, AG3S가 결정하지 않는다.'
]
for y,s in zip([254,247,239,233],principles):txt(ax,7,y,s,11,ha='left')
# lane tints stop above shared policy, with labels inside ample whitespace
for x,w,c,title,sub in [(6,50,'orange','열 B · ROI','어디를 정밀하게 볼 것인가'),(65,52,'blue','열 A · 기하','무엇이 부딪힐 수 있나'),(124,52,'red','열 C · 의미','무엇이 조작 대상인가')]:
    fill,edge=COL[c]
    ax.add_patch(Rectangle((x-1.5,79),w+3,149,facecolor=fill,alpha=.28,edgecolor='none',zorder=0))
    txt(ax,x+w/2,223,title,16,'bold',color=edge)
    txt(ax,x+w/2,217.5,sub,11)
box(ax,6,201,50,12,'입력 · VLA action chunk','50 스텝 × 14 채널','orange',12,11)
box(ax,65,201,52,12,'입력 · depth 카메라 3대 (머리 1 + 손목 2)','내부/외부 파라미터 · 로봇 상태 · phase(주입)','blue',10.4,9.8)
box(ax,124,201,52,12,'입력 · VLA 내부 attention 맵','특정 layer · head, 16×16','red',12,10.5)
A=[
(183,'A1. 로봇 self-filter','로봇 자기 몸의 픽셀·점 제거 → 자기 충돌 오인 방지'),
(163,'A2. 재구성 → point cloud','핀홀 역투영 · uv 보존 · 결정론적 다운샘플'),
(143,'A3. 멀티뷰 융합','카메라별 처리 → base 좌표계 통합 · 같은 칸은 max'),
(123,'A4. 지지면 RANSAC 평면','평면 하나 = 선형 제약 행 하나'),
(103,'A5. Block-Sparse TSDF 5 mm','관측을 매 프레임 적분 · 누적'),
(83,'A6. Local Dense ESDF 10–20 mm','ROI 안에서만 생성 · 거리 d(p)와 기울기 ∇d(p)')]
for y,t,b in A:box(ax,65,y,52,12,t,b,'blue',12,9.6)
for top,bot in [(201,195),(183,175),(163,155),(143,135),(123,115),(103,95)]:arr(ax,[(91,top),(91,bot)])
for y,t,b in [(183,'B1. Forward Kinematics','chunk 매 스텝의 로봇 자세'),(163,'B2. 로봇 충돌 구 궤적','구 61개 × 50 스텝'),(143,'B3. swept volume + padding','이번 chunk가 쓸고 지나갈 부피')]:box(ax,6,y,50,12,t,b,'orange',12,10.5)
for top,bot in [(201,195),(183,175),(163,155)]:arr(ax,[(31,top),(31,bot)])
arr(ax,[(53,143),(53,89),(65,89)],color=COL['orange'][1])
txt(ax,59,92,'ROI',10,'bold')
txt(ax,8,130,'정밀 영역을 attention의 target에 두면\n파지 순간 목적지로 옮겨간다.\nswept volume은 attention이 어디를 보든\n로봇이 실제로 지나가는 곳이다.',10,ha='left',va='top',linespacing=1.7)
box(ax,124,183,52,12,'C1. attention 어댑터 → 픽셀맵 (H, W)','카메라 해상도로 확대 · VLA별 차이는 여기서 격리','red',11.5,9.7)
box(ax,124,163,52,12,'C2. lifting — 점별 attention','uv로 3D 점에 부착 · 어떤 점도 버리지 않음','red',12,10)
box(ax,124,123,52,12,'C3. grounding','씨앗 → 3D 연결성 → 덩어리 → 점수 → 주목 대상','red',12,9.6)
box(ax,124,101,52,16,'C4. 잠금(latch)  ★신규','걸기: 연속 N프레임 같은 물체 + 1·2위 점수 격차\n유지: attention 무시 · 풀기: 성공=detach → 조작 대상','red',13,9.5,lw=2.8)
# distinct highlight border for new latch
ax.add_patch(FancyBboxPatch((124,101),52,16,boxstyle='round,pad=0,rounding_size=0.7',fill=False,edgecolor='#75530b',linewidth=2.4,zorder=4))
box(ax,124,83,52,12,'C5. attached — 쥔 물체를 로봇 쪽 구로','parent_link를 따라 이동 · 해석적 거리 사용','red',11.6,10)
for top,bot in [(201,195),(183,175),(163,135),(123,117),(101,95)]:arr(ax,[(150,top),(150,bot)])
arr(ax,[(117,129),(124,129)],color=COL['blue'][1])
txt(ax,120.5,138,'지지면 마스크',9)
# The only feedback edge originates outside AG3S.
txt(ax,188,122,'과제 소유자\n(AG3S 밖)',9,linespacing=1.4)
arr(ax,[(188,116),(188,109),(176,109)],dash=True)
txt(ax,188,101,'성공 판정 →\n잠금 해제\n(detach)',8.7,linespacing=1.35)
# The geometry and semantic branches feed the pairwise policy; ROI arrives via A6.
arr(ax,[(91,83),(91,68)])
arr(ax,[(150,83),(150,68)])
box(ax,6,42,170,26,'',col='purple',lw=1.8)
txt(ax,91,63,'합류 1 · clearance 정책 — (로봇 구 × 대상 종류) 마진 행렬',14,'bold')
heads=['조작 대상(쥔 것)','목적지(담을 곳) ★신규','장애물','지지면','미지 기하']
vals=['0 mm, 권한 링크만','얇게','full','full','full']
for i,(h,v) in enumerate(zip(heads,vals)):
    x=9+i*32.8
    ax.add_patch(Rectangle((x,49),32.8,10,facecolor='white',edgecolor='#75530b' if i==1 else '#9a90a9',linewidth=2.1 if i==1 else .8,zorder=3))
    txt(ax,x+16.4,56,h,10.2,'bold');txt(ax,x+16.4,51.6,v,10.5)
txt(ax,91,45.2,'알 수 없는 조합은 전부 full margin — fail closed',11,'bold')
box(ax,25,23,132,13,'합류 2 · CollisionConstraintSet','ESDF 핸들 · 조작 대상 · 구별 마진 배열 · 지지면 · validity · phase','purple',14,11)
arr(ax,[(91,42),(91,36)])
arr(ax,[(91,23),(91,17.5)])
txt(ax,91,14.5,'trajopt',13,'bold')
txt(ax,7,7,'주목 대상(attention이 보는 것)과 조작 대상(실제로 쥔 것)은 다르다. 접촉 권한은 조작 대상에 붙는다.',11,ha='left')
save(fig,'02_ag3s_detail')
print('Created both diagrams as PNG, SVG and PDF:',OUT)
