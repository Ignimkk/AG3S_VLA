"""T32 H4 — xz slices (y = GT crate centre y) of rebuilt fields for a few settings, chunk 30 of E3b_ep1807_r1.

usage (cwd = outputs/verify/T32/h4/runroot):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python ../slices.py
Same inputs/build as real_rebuild.py. Slice grid: x in crate centre +-0.2 m, z in [0.80, 1.05] m, 1 mm samples,
EsdfField.distance per tier; TSDF (sdf, weight) sampled at the same points with h4_common.tsdf_at.
Output: real/slices_chunk00030.npz + .json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T32/h4'))
import h4_common as H
import real_rebuild as RR

OUT = RR.OUT
run, k = "E3b_ep1807_r1", 30
c, z, rec_fine, rec_coarse, s = RR.recorded(run, k)
seq = int(c["seq"]); cc, Rc, walls = RR.gt_crate(run, seq)
fc = H.fine_centre_from_origin(z["esdf_origin"], z["esdf_shape"], float(z["esdf_voxel_size"]))
att = np.load(OUT / f"attached_{run}_chunk{k:05d}.npy")
xs = np.arange(cc[0] - 0.2, cc[0] + 0.2 + 1e-9, 0.001); zs = np.arange(0.80, 1.05 + 1e-9, 0.001)
X, Zg = np.meshgrid(xs, zs)
P = np.stack([X.ravel(), np.full(X.size, cc[1]), Zg.ravel()], 1)


def sample(L):
    v = np.full(len(P), np.nan); ins = H.inside(L, P)
    v[ins] = np.asarray(L.distance(P[ins]), np.float64); return v.reshape(X.shape)


out = {"xs": xs, "zs": zs, "recorded_fine": sample(rec_fine), "recorded_coarse": sample(rec_coarse)}
S = dict(H.SERVER)
cfgs = [("server", S, None), ("trunc_20mm", {**S, "trunc_vox": 1.0}, None), ("trunc_30mm", {**S, "trunc_vox": 1.5}, None),
        ("trunc_40mm", {**S, "trunc_vox": 2.0}, None), ("cams_zed_left", S, ("zed_left",))]
meta = {"run": run, "chunk": k, "seq": seq, "y_m": float(cc[1]), "crate_centre_base_m": cc.tolist(), "gt_walls": walls, "configs": {}}
for name, cfg, cams in cfgs:
    frames = RR.load_frames(run, list(range(1, seq + 1)), cams)
    built = H.build(H.make_builder(**cfg), frames, fc, att)
    for tn in ("fine", "coarse"):
        out[f"{name}_{tn}"] = sample(built["tiers"][tn]["layer"])
    sdf, w = H.tsdf_at(built["itg"], P)
    out[f"{name}_tsdf"] = sdf.reshape(X.shape)
    meta["configs"][name] = {"cfg": cfg, "cams": list(cams) if cams else "all", "truncation_m": built["truncation_m"]}
    print(name, flush=True)
    import torch; del built; torch.cuda.empty_cache()
np.savez_compressed(OUT / "slices_chunk00030.npz", **out)
(OUT / "slices_chunk00030.json").write_text(json.dumps(meta, indent=1))
