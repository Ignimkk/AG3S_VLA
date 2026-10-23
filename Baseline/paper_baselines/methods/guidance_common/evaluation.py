"""Helpers shared by the CPU smoke (``scripts/smoke_guidance_methods.py``) and the GPU smoke (``gpu_smoke.py``)."""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.geometry import signed_distance_np
from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel
from Baseline.paper_baselines.methods.interface import PRIVILEGED_PREFIX

#: build_scene kwargs of the smokes: the recorded GT "support" class already removes the table (privileged ignore mask)
SCENE_KW = {"remove_table": False, "voxel": 0.02, "max_points": 1024, "workspace_radius": 0.6}


def path_metrics(raw: np.ndarray, scene, H: int) -> dict:
    pm = EEPathModel(H=H, p0=scene.ee_pos, R_bw=scene.R_bw, y_prev=scene.y_prev)
    p = pm.numpy(np.clip(raw[:, :3], -1, 1))
    o = scene.obstacles
    if o.empty:
        return {"path_base": p.tolist(), "ee_min_sdf_m": None, "hand_min_sdf_m": None}
    sd = signed_distance_np(p, o)
    hs = p[:, None, :] + (scene.hand_c - scene.ee_pos[None])[None]
    hsd = signed_distance_np(hs.reshape(-1, 3), o).reshape(H, -1) - scene.hand_r[None]
    return {"path_base": p.tolist(), "ee_min_sdf_m": float(sd.min()), "hand_min_sdf_m": float(hsd.min()),
            "ee_sdf_per_row_m": sd.tolist()}


def injected_obs(obs, scene, chunk, radius=0.03, row=6, offset=0.01):
    """Copy of ``obs`` with a privileged sphere ``offset`` m beside the recorded EE path at ``row``."""
    import copy

    H = chunk.shape[0]
    pb = EEPathModel(H=H, p0=scene.ee_pos, R_bw=scene.R_bw, y_prev=scene.y_prev).numpy(np.clip(chunk[:, :3], -1, 1))
    k = min(H - 1, row)
    lat = np.cross(pb[k] - pb[max(k - 1, 0)], [0, 0, 1.0])
    c = pb[k] + offset * lat / (np.linalg.norm(lat) + 1e-9)
    o = copy.copy(obs)
    o.extras = {**obs.extras, PRIVILEGED_PREFIX + "obstacle_geoms": [{"type": "sphere", "pos": c, "size": [radius],
                                                                       "name": "injected_sphere"}]}
    return o, c, k



__all__ = ["SCENE_KW", "injected_obs", "path_metrics"]
