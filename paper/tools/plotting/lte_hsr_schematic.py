"""Conceptual LTE / HSR diagram; no experimental measurements are depicted."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Ellipse

OUT = Path('AG3S/ag3s/docs/figures/core-eval/paper')
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(15.5, 5.7))
fig.patch.set_facecolor('white'); ax.set_xlim(0, 15.5); ax.set_ylim(0, 5.7); ax.axis('off')
navy='#243449'; blue='#2763a5'; red='#ce5147'; gray='#7c8795'; green='#30876c'; light='#f2f5f8'
def text(x,y,s,**kw): ax.text(x,y,s, color=kw.pop('color',navy), **kw)
def box(x,y,w,h,fc='white',ec='#d4dce5',r=.12,lw=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.02,rounding_size={r}',facecolor=fc,edgecolor=ec,linewidth=lw); ax.add_patch(p); return p
def arrow(a,b,color=gray,lw=1.5,style='-|>',**kw): ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=11,color=color,linewidth=lw,**kw))
def apple(x,y,filled=True,alpha=1):
    ax.add_patch(Circle((x,y),.26,facecolor=red if filled else 'none',edgecolor=red,linewidth=1.5,linestyle='-' if filled else '--',alpha=alpha))
    ax.plot([x,x+.04],[y+.25,y+.34],color=green,lw=1.6,alpha=alpha)
    ax.add_patch(Ellipse((x+.10,y+.31),.15,.07,angle=25,facecolor=green,edgecolor='none',alpha=alpha))
def obstacle(x,y):
    ax.add_patch(Rectangle((x-.19,y-.23),.38,.46,facecolor='#a9b4c0',edgecolor='#738294',lw=1))
def gripper(x,y,closed=False):
    ax.plot([x,x],[y+.72,y+.42],color=navy,lw=9,solid_capstyle='round')
    ax.plot([x-.37,x+.37],[y+.42,y+.42],color=blue,lw=7,solid_capstyle='round')
    gap=.30 if closed else .43
    ax.plot([x-gap,x-gap],[y+.40,y-.10],color=blue,lw=6,solid_capstyle='round')
    ax.plot([x+gap,x+gap],[y+.40,y-.10],color=blue,lw=6,solid_capstyle='round')
def header(x,y,n,title,subtitle):
    text(x,y,n,fontsize=10,fontweight='bold',color=blue,va='top')
    text(x+.29,y,title,fontsize=12.5,fontweight='bold',va='top')
    text(x+.29,y-.34,subtitle,fontsize=9.2,color='#526276',va='top')

# Panels
box(.12,.77,3.13,4.55,fc='#f7f9fb'); box(3.52,.77,7.43,4.55); box(11.22,.77,4.16,4.55,fc='#f7f9fb')
header(.35,5.08,'01','HSR: task state','Approach → grasp recognized')
gripper(1.66,3.40,True); apple(1.66,3.31); obstacle(2.55,3.30)
text(.40,2.47,'HSR estimates task phase',fontsize=9.3,fontweight='bold')
text(.40,2.14,'and whether the object is held.',fontsize=9.0)
box(.40,1.15,2.57,.57,fc='#eaf1fa',ec='#c7d9ed'); text(1.685,1.435,'grasp / held signal',ha='center',va='center',fontsize=9.2,fontweight='bold',color=blue)
arrow((3.25,3.14),(3.50,3.14)); arrow((10.96,3.14),(11.21,3.14))
header(3.77,5.08,'02','LTE: link-specific target exemption','Before attachment  •  two views of the same scene')
# field cards
box(3.82,2.03,3.22,2.19,fc='#f5f8fc',ec='#cbd9ea'); box(7.43,2.03,3.22,2.19,fc='#fbf7f6',ec='#ebd5d2')
text(4.02,3.96,'AUTHORIZED LINKS',fontsize=9,fontweight='bold',color=blue)
text(4.02,3.66,'Fingers / palm',fontsize=10,fontweight='bold')
apple(4.52,2.93,False); obstacle(5.45,2.93)
text(4.95,2.48,'target-free field',ha='center',fontsize=9.4,fontweight='bold',color=blue)
text(7.63,3.96,'OTHER LINKS',fontsize=9,fontweight='bold',color=red)
text(7.63,3.66,'Wrist / arm',fontsize=10,fontweight='bold')
apple(8.13,2.93); obstacle(9.06,2.93)
text(8.56,2.48,'full-scene field',ha='center',fontsize=9.4,fontweight='bold',color=red)
text(7.25,1.62,'Target contact allowed for designated links only',ha='center',fontsize=10,fontweight='bold')
text(7.25,1.29,'Other obstacles remain in both fields',ha='center',fontsize=9.4,color='#526276')
header(11.47,5.08,'03','After grasp','Object joins robot collision geometry')
gripper(12.55,3.39,True); apple(12.55,3.30)
for dx,dy in [(-.12,0),(.12,0),(0,.14),(0,-.14),(0,0)]: ax.add_patch(Circle((12.55+dx,3.30+dy),.10,fill=False,edgecolor=blue,lw=1.0))
obstacle(14.27,3.30); arrow((12.96,3.30),(13.95,3.30),color=green,lw=1.5)
text(12.55,2.72,'attached spheres',ha='center',fontsize=8.8,color=blue,fontweight='bold')
text(14.27,2.72,'environment',ha='center',fontsize=8.8,color='#526276')
text(11.48,2.26,'Held volume removed from field;',fontsize=9.2)
text(11.48,1.91,'moves with hand and is checked',fontsize=9.2)
text(11.48,1.56,'against the environment.',fontsize=9.2)
text(.16,.38,'Conceptual schematic  •  Dashed red: target excluded from queried field  •  Gray: other obstacles',fontsize=9,color='#657487')
fig.savefig(OUT/'paper-lte-hsr-schematic.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(OUT/'paper-lte-hsr-schematic.svg',bbox_inches='tight',pad_inches=.04)
fig.savefig(OUT/'paper-lte-hsr-schematic.png',dpi=220,bbox_inches='tight',pad_inches=.04)
