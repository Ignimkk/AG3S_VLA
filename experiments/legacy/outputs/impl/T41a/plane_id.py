from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, json, glob, sys
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
base=_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/server_constraints/run_0000')
res={'fine':[], 'coarse':[]}
for i in range(0,604,20):
  z=np.load(f'{base}/chunk_{i:05d}.npz',allow_pickle=True)
  cl=z['filtered_cloud'].astype(float); sm=z['support_mask']
  sup=cl[sm]; sup=sup[sup[:,2]>0.4]
  mu=sup.mean(0); _,_,vt=np.linalg.svd(sup-mu,full_matrices=False); n=vt[2]*np.sign(vt[2][2]); off=n@mu
  other=cl[~sm]; other=other[other[:,2]>0.4]
  tree=cKDTree(other)
  o=z['esdf_origin']; vs=float(z['esdf_voxel_size']); sh=z['esdf_shape']
  rng=np.random.default_rng(i); P=o+rng.random((30000,3))*sh*vs
  s=P@n-off; keep=(s>0.002)&(s<0.08)
  P=P[keep]; s=s[keep]
  dn=tree.query(P)[0]
  # table clearly nearest: every non-support point at least 10 mm further than the plane
  st=cKDTree(sup[:,:2]); over=st.query(P[:,:2])[0]<0.010
  tbl=(dn>s+0.010)&over
  # an obstacle clearly nearest: a non-support point at least 10 mm nearer than the plane
  obs=dn<s-0.010
  for tier,(g,oo,vv) in {'fine':(z['esdf_distance'],o,vs),'coarse':(z['esdf_coarse_distance'],z['esdf_coarse_origin'],float(z['esdf_coarse_voxel_size']))}.items():
    d=map_coordinates(g.astype(np.float32), ((P-oo)/vv).T, order=1)
    res[tier].append(dict(i=i, n_tbl=int(tbl.sum()), n_obs=int(obs.sum()),
       tbl_absdiff=np.abs(d-s)[tbl], obs_absdiff=np.abs(d-s)[obs]))
out={}
for tier,L in res.items():
  t=np.concatenate([r['tbl_absdiff'] for r in L]); b=np.concatenate([r['obs_absdiff'] for r in L])
  out[tier]={'n_table_nearest':int(t.size),'n_obstacle_nearest':int(b.size),
    'table_nearest_abs_d_minus_s_mm_q50_q90_q99':[float(np.quantile(t,q)*1e3) for q in (.5,.9,.99)]}
  for tol in (0.005,0.010,0.020):
    out[tier][f'tol_{int(tol*1e3)}mm']={'table_identified_frac':float((t<=tol).mean()),
       'obstacle_misread_as_table_frac':float((b<=tol).mean())}
print(json.dumps(out,indent=1)); json.dump(out,open(sys.argv[1],'w'),indent=1)
