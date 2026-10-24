"""Non-privileged scene geometry from the wire depth (server side, numpy; runs in either venv).

``scene_points`` lifts every wire camera (``lift.backproject_pixels``, the V11AB back-projection), then removes

1. the robot: points within its collision spheres + ``robot_margin`` at the request's ``robot_state`` (Panda
   kinematics, ``safety_common.ObstacleCloudSource.remove_robot``; not the simulator segmentation),
2. points farther than ``workspace_radius`` from the EE,
3. the table: points below the densest horizontal plane + ``table_margin`` (``guidance_common.scene.estimate_table_z``),
4. the target: pixels inside the detected target mask (``target_masks[cam]``, e.g. a Grounding DINO box) -- lifted
   separately as ``target_points``.

``clusters`` splits the remaining cloud into connected components on a voxel grid (26-neighbourhood), the
``obstacle_source="given"`` input of the filters (``--obstacles depth``).
"""
from __future__ import annotations

import time

import numpy as np

from Baseline.paper_baselines.methods.interface import SceneObservation

_ROBOT_SOURCE = None


def _robot_remover():
    global _ROBOT_SOURCE
    if _ROBOT_SOURCE is None:
        from Baseline.paper_baselines.methods.safety_common import ObstacleCloudSource

        _ROBOT_SOURCE = ObstacleCloudSource("given")
    return _ROBOT_SOURCE


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    from Baseline.paper_baselines.methods.guidance_common.scene import voxel_downsample

    return voxel_downsample(points, voxel, max_points=10 ** 9)


def scene_points(obs: SceneObservation, *, target_masks: dict | None = None, robot_margin: float = 0.02,
                 workspace_radius: float = 0.6, table_margin: float = 0.015, voxel: float = 0.01,
                 range_max: float = 2.0) -> dict:
    """``{"obstacle_points" (N, 3), "target_points" (M, 3) | None, "table_z", "info"}`` in the base frame."""
    from Baseline.paper_baselines import lift as L
    from Baseline.paper_baselines.methods.guidance_common.scene import estimate_table_z

    t0 = time.perf_counter()
    obst, targ = [], []
    n_raw = 0
    for cam, depth in obs.depth.items():
        if cam not in obs.intrinsics or cam not in obs.T_base_cam:
            continue
        pts, u, v = L.backproject_pixels(depth, obs.intrinsics[cam], obs.T_base_cam[cam], range_max=range_max)
        n_raw += len(pts)
        tm = None if not target_masks or target_masks.get(cam) is None else np.asarray(target_masks[cam], bool)[v, u]
        if tm is not None:
            targ.append(pts[tm])
            pts = pts[~tm]
        obst.append(pts)
    pts = np.concatenate(obst) if obst else np.zeros((0, 3))
    ee = np.asarray(obs.ee_pose, np.float64)[:3, 3]
    pts = pts[np.linalg.norm(pts - ee[None], axis=1) <= workspace_radius]
    pts = voxel_down(pts, voxel)  # before the robot test: fewer KD-tree queries
    n_ws = len(pts)
    remover = _robot_remover()
    remover.robot_margin = float(robot_margin)
    pts = remover.remove_robot(obs, pts)
    n_robot = n_ws - len(pts)
    table_z = estimate_table_z(pts)
    if table_z is not None:
        pts = pts[pts[:, 2] > table_z + table_margin]
    tp = None
    if targ and sum(len(t) for t in targ):
        tp = voxel_down(np.concatenate(targ), voxel)
        tp = remover.remove_robot(obs, tp)
        if table_z is not None:
            tp = tp[tp[:, 2] > table_z + table_margin]
        tp = tp if len(tp) else None
    info = {"n_raw": int(n_raw), "n_workspace_voxels": int(n_ws), "n_robot_removed": int(n_robot),
            "table_z": None if table_z is None else float(table_z), "n_obstacle": int(len(pts)),
            "n_target": 0 if tp is None else int(len(tp)), "voxel_m": voxel, "robot_margin_m": robot_margin,
            "ms": (time.perf_counter() - t0) * 1e3,
            "robot_removal": "Panda collision spheres at robot_state + margin (kinematics, non-privileged)"}
    return {"obstacle_points": pts, "target_points": tp, "table_z": table_z, "info": info}


def clusters(points: np.ndarray, *, voxel: float = 0.02, min_points: int = 30, max_clusters: int = 8) -> dict:
    """Connected components (26-neighbour voxel grid) -> ``{"depth#j": (N_j, 3)}``, largest first."""
    from scipy import ndimage

    p = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(p):
        return {}
    keys = np.floor(p / voxel).astype(np.int64)
    lo = keys.min(0)
    k = keys - lo
    grid = np.zeros(tuple(k.max(0) + 1), bool)
    grid[tuple(k.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    pl = lab[tuple(k.T)]
    sizes = np.bincount(pl, minlength=n + 1)
    order = [i for i in np.argsort(-sizes) if i > 0 and sizes[i] >= min_points][:max_clusters]
    return {f"depth#{j}": p[pl == i] for j, i in enumerate(order)}


def box_masks(boxes: dict, shapes: dict) -> dict:
    """``{cam: (x0, y0, x1, y1) | None}`` -> bool masks of each camera's shape."""
    out = {}
    for cam, hw in shapes.items():
        m = np.zeros(tuple(hw), bool)
        b = boxes.get(cam)
        if b is not None:
            x0, y0, x1, y1 = np.round(np.asarray(b, float)).astype(int)
            m[max(0, y0):min(hw[0], y1), max(0, x0):min(hw[1], x1)] = True
        out[cam] = m
    return out


__all__ = ["box_masks", "clusters", "scene_points", "voxel_down"]
