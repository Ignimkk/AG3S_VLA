from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.fields.curobo_field import layer_from_arrays
f = sys.argv[1]
z = np.load(f, allow_pickle=False); s = json.loads(str(z["summary_json"]))
print(z.files)
print({k: (z[k].shape, z[k].dtype) for k in z.files if k.startswith("esdf")})
print("fine origin", z["esdf_origin"], "vox", z["esdf_voxel_size"], "trunc", s["esdf_stats"].get("tsdf_truncation"))
D = np.asarray(z["esdf_distance"], np.float32); o = np.asarray(z["esdf_origin"], float); h = float(z["esdf_voxel_size"])
print("D shape", D.shape, "min/max", D.min(), D.max(), "n==0", int((D==0).sum()), "n<0", int((D<0).sum()))
# a column profile at centre xy
i, j = D.shape[0]//2, D.shape[1]//2
print("col", [round(float(v)*1000,1) for v in D[i, j, :]])
print("z centres", [round(o[2] + (k+0.5)*h, 4) for k in range(D.shape[2])])
