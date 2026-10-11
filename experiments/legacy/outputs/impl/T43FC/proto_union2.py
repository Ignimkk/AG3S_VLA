"""T43 FC prototype 2 — interior-grid candidates, r = per-piece depth + delta (overflow <= delta, exact)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, time, numpy as np, mujoco
from scipy.spatial import cKDTree
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.gripper_cover import link_surface, hull_depth, _gap
MODEL=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(MODEL); scene=TransportScene.attach(m, mujoco.MjData(m))
rob=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05))
link="ee_finger_l1"
fine=link_surface(m, rob, link, spacing=0.0005)
TAU=0.0003
int_face=hull_depth(fine.points+TAU*fine.normals,fine.hulls)>0
print("internal-face samples",int(int_face.sum()),"of",len(int_face))
target=fine.points[~int_face]
tree_t=cKDTree(target); tree_all=cKDTree(fine.points)
def greedy(C,R,target,tol=0.0):
    tt=cKDTree(target)
    covers=[np.asarray(tt.query_ball_point(c,float(r)+tol),np.int64) for c,r in zip(C,R)]
    unc=np.ones(target.shape[0],bool); gain=np.array([len(c) for c in covers]); chosen=[]
    while unc.any():
        order=np.lexsort((R,-gain)); best,bg=-1,0
        for j in order:
            if gain[j]<=bg: break
            g=int(unc[covers[j]].sum()); gain[j]=g
            if g>bg: best,bg=j,g
        if best<0 or bg==0: break
        chosen.append(best); unc[covers[best]]=False
    return np.asarray(chosen), int(unc.sum())
def overflow(C,R,n=600):
    rng=np.random.default_rng(0); out=0.0
    for c,r in zip(C,R):
        v=rng.normal(size=(n,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
        P=c+r*v; d=hull_depth(P,fine.hulls); o=P[d<0]
        if o.size: out=max(out,float(tree_all.query(o)[0].max()))
    return out
EPS=0.0005
lo,hi=fine.vertices.min(0),fine.vertices.max(0)
for h in (0.0005,):
    g=np.stack(np.meshgrid(*[np.arange(l,hh+1e-9,h) for l,hh in zip(lo,hi)],indexing='ij'),-1).reshape(-1,3)
    d=hull_depth(g,fine.hulls); keep=d>0.0002; G=g[keep]; ds=tree_t.query(G)[0]; D=np.maximum(d[keep], ds-EPS)
    print("union>piece depth cands", int((ds-EPS>d[keep]+1e-4).sum()))
    print("grid",h,"cands",G.shape[0],flush=True)
    rows=[]
    for delta in (0.001,0.0015,0.002,0.0025,0.003):
        t=time.time(); R=D+delta
        idx,unc=greedy(G,R,target)
        C,Ri=G[idx],R[idx]
        gg=_gap(target,C,Ri)
        row={"delta_mm":delta*1e3,"n":int(idx.size),"r_mm":[round(float(Ri.min()*1e3),2),round(float(Ri.max()*1e3),2)],"uncovered":unc,"n_targets":int(target.shape[0]),
             "gap_max_mm":round(float(max(gg.max(),0)*1e3),3),"overflow_sampled_mm":round(overflow(C,Ri)*1e3,3),
             "inner_x":round(float((C[:,0]-Ri).min()*1e3),2),"outer_x":round(float((C[:,0]+Ri).max()*1e3),2),"tip_z":round(float((C[:,2]-Ri).min()*1e3),2),"y":round(float((np.abs(C[:,1])+Ri).max()*1e3),2),"sec":round(time.time()-t,1)}
        print(row,flush=True); rows.append(row)
json.dump(rows,open(_ag3s_workspace_path('outputs/impl/T43FC/proto_union2.json'),"w"),indent=1)
