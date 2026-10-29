from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np
sys.argv=[sys.argv[0],'linearize_head.py','/dev/null']
src=open('replay_identity.py').read().split("rep = {")[0]
exec(src)
base=_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/server_constraints/run_0000')
for i in list(range(131,206))+[128,129,130]:
    z=np.load(f'{base}/chunk_{i:05d}.npz',allow_pickle=True)
    summ=json.loads(str(z['summary_json'])); names=np.array(summ['sphere_link_names'])
    radii=z['sphere_radii'].astype(np.float64); c=z['sphere_centres'].astype(np.float64)[None]
    field=build_field(z,outside_distance=1.0); L=lin(NEW,radii)
    off=L._esdf_clearance(c,scene(NEW,z,field,names,radii,dict(support_planes=plane(z))))[...,0]
    on=L._esdf_clearance(c,scene(NEW,z,field,names,radii,dict(support_planes=plane(z),obstacle_margin=0.01)))[...,0]
    cls=L._last_obstacle_class; hand=np.isin(names,AUTH)[None]
    ns=int(((cls==2)&hand&(off<0.01)).sum()); no=int(((cls==0)&hand&(off<0.01)).sum()); nt=int(((cls==1)&hand&(off<0.03)).sum())
    if ns or no or i>=128 and i<=130: print(i,int(z['t_step']),'support<10',ns,'obst<10',no,'target<30',nt, 'min hand off', round(float(off[hand].min())*1e3,2))
