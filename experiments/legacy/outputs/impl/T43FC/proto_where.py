from common.workspace import workspace_path as _ag3s_workspace_path
exec(open(_ag3s_workspace_path('outputs/impl/T43FC/proto_grid.py')).read().split("lo,hi=fine.vertices")[0])
lo,hi=fine.vertices.min(0),fine.vertices.max(0); h=0.0005
g=np.stack(np.meshgrid(*[np.arange(l,hh+1e-9,h) for l,hh in zip(lo,hi)],indexing='ij'),-1).reshape(-1,3)
d=hull_depth(g,fine.hulls); keep=d>0.0002; G,D=g[keep],d[keep]
delta=0.002; R=D+delta; idx,unc=greedy(G,R,target); C,Ri=G[idx],R[idx]
print("n",idx.size, "block(z>-10mm)", int((C[:,2]>-0.010).sum()), "plate", int((C[:,2]<=-0.010).sum()))
print("radii hist", np.histogram(Ri*1e3, bins=[2,3,4,5,6,6.5,7])[0])
# targets by region
print("targets block", int((target[:,2]>-0.010).sum()), "plate", int((target[:,2]<=-0.010).sum()))
# how many covered by first k
tt=cKDTree(target); unc=np.ones(len(target),bool); cum=[]
for c,r in zip(C,Ri):
    unc[tt.query_ball_point(c,r)]=False; cum.append(int((~unc).sum()))
print("cum frac at k=10,20,40,60,80,100:", [round(cum[k-1]/len(target),3) for k in (10,20,40,60,80,100)])
