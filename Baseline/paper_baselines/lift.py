"""Mask + depth + K + T_base_cam -> base-frame target points / 5 mm voxels (the one lift every provider shares).

Pixel frames (the only place a frame conversion happens; fixed by ``tests/test_lift.py``):

* **OpenCV frame** -- the frame of the recorded depth (``depth_mm/<cam>``), segmentation (``segmentation_geom/<cam>``)
  and of ``K`` (``pixel_centre``: pixel centres at integer indices) and ``T_base_cam`` (OpenCV axes). Every
  provider receives RGB in this frame and returns masks in it.
* **robosuite raw** -- the recorded ``obs_*_image`` arrays (OpenGL rows, row 0 = bottom):
  ``opencv = raw[::-1]`` (:func:`opencv_rgb_from_raw`, conventions_json ``image_rows``).
* **policy 224** -- the image pi0.5 sees: ``raw[::-1, ::-1]`` then ``resize_with_pad`` 256->224, so
  ``policy256[i, j] = opencv[i, W-1-j]`` (left-right mirrored, AG3S_BENCHMARK_LOG.md ~562). A mask made in that
  frame (VLA attention, B's providers) goes through :func:`policy_mask_to_opencv` before lifting. The pixel map
  is ``benchmark.adapters.libero_panda.camera.opencv_to_policy_pixel`` (re-used, not re-derived).

Back-projection re-uses V11AB's ``AG3S.ag3s.stages.reconstruction.backproject`` (pinhole, no flip, invalid depth
dropped) with ``PointCloudConfig(range_max=2.0)`` -- the same call the development soft-IoU evaluation makes
(``ifg_graph._cam_pixels``). Depth is the wire depth (uint16 mm / 1000), i.e. what V11AB itself sees.

Robot removal (``ROBOT_REMOVAL``), applied identically to every provider (fairness):

* ``none``          -- the mask is lifted as given;
* ``gt_robot_seg``  -- pixels whose GT segmentation geom belongs to the robot (``robot0_*``, ``gripper0_*``) are
  removed before lifting. Privileged (GT) -- an upper bound on what a robot filter can do.
"""
from __future__ import annotations

import numpy as np

from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.stages.reconstruction import backproject
from benchmark.adapters.libero_panda import camera as libero_camera

VOXEL_M = 0.005
RANGE_MAX_M = 2.0
ROBOT_REMOVAL = ("none", "gt_robot_seg")
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_")
POLICY_HW = (224, 224)


# --- pixel frames ----------------------------------------------------------------------------------------------
def opencv_rgb_from_raw(raw: np.ndarray) -> np.ndarray:
    """Recorded robosuite image (OpenGL rows) -> OpenCV-frame image (same frame as depth / segmentation)."""
    return np.ascontiguousarray(np.asarray(raw)[::-1])


def policy_mask_to_opencv(mask_policy: np.ndarray, src_hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Mask in the policy 224 image -> mask in the OpenCV ``src_hw`` frame (nearest pixel; pad area -> False).

    For each OpenCV pixel centre the policy pixel it maps to is computed with ``opencv_to_policy_pixel`` and read.
    """
    m = np.asarray(mask_policy, bool)
    H, W = int(src_hw[0]), int(src_hw[1])
    v, u = np.mgrid[0:H, 0:W]
    x, y = libero_camera.opencv_to_policy_pixel(u, v, src_hw=(H, W), out_hw=m.shape)
    xi, yi = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
    ok = (xi >= 0) & (xi < m.shape[1]) & (yi >= 0) & (yi < m.shape[0])
    out = np.zeros((H, W), bool)
    out[ok] = m[yi[ok], xi[ok]]
    return out


def opencv_mask_to_policy(mask_opencv: np.ndarray, out_hw: tuple[int, int] = POLICY_HW) -> np.ndarray:
    """Inverse direction (for tests / figures): OpenCV mask -> policy-frame mask (nearest pixel)."""
    m = np.asarray(mask_opencv, bool)
    H, W = m.shape
    y, x = np.mgrid[0:out_hw[0], 0:out_hw[1]]
    u, v = libero_camera.policy_to_opencv_pixel(x, y, src_hw=(H, W), out_hw=out_hw)
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    out = np.zeros(tuple(out_hw), bool)
    out[ok] = m[vi[ok], ui[ok]]
    return out


# --- robot removal --------------------------------------------------------------------------------------------
def robot_geom_ids(model_geom_bodyid: np.ndarray, model_body_names, prefixes=ROBOT_BODY_PREFIXES) -> np.ndarray:
    names = [str(b) for b in model_body_names]
    bodies = {i for i, n in enumerate(names) if n.startswith(tuple(prefixes))}
    return np.array([g for g, b in enumerate(np.asarray(model_geom_bodyid).tolist()) if int(b) in bodies], np.int64)


def apply_robot_removal(mask: np.ndarray, mode: str, robot_pixels: np.ndarray | None = None) -> np.ndarray:
    if mode not in ROBOT_REMOVAL:
        raise ValueError(f"robot removal must be one of {ROBOT_REMOVAL}, got {mode!r}")
    mask = np.asarray(mask, bool)
    if mode == "none":
        return mask
    if robot_pixels is None:
        raise ValueError("gt_robot_seg needs the robot pixel mask")
    return mask & ~np.asarray(robot_pixels, bool)


# --- lifting --------------------------------------------------------------------------------------------------
def backproject_pixels(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, *, range_max: float = RANGE_MAX_M
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All valid pixels of one camera -> ``(points_base (N, 3), u (N,), v (N,))`` via V11AB ``backproject``."""
    pc = backproject(np.asarray(depth_m, np.float64), K, T_base_cam, PointCloudConfig(range_max=range_max))
    return pc.points, pc.uv[:, 0].astype(np.int64), pc.uv[:, 1].astype(np.int64)


def lift_selection(mask: np.ndarray, u: np.ndarray, v: np.ndarray, depth_shape: tuple[int, int], *,
                   robot_removal: str = "none", robot_pixels: np.ndarray | None = None) -> np.ndarray:
    """Boolean selection over the back-projected valid-depth pixels ``(u, v)`` (:func:`backproject_pixels` order):
    ``points[lift_selection(...)]`` are the lifted mask points. Same robot removal and frame check as
    :func:`lift_mask`; lets a caller back-project a frame once and lift many masks of it."""
    mask = apply_robot_removal(mask, robot_removal, robot_pixels)
    if mask.shape != tuple(depth_shape):
        raise ValueError(f"mask {mask.shape} and depth {tuple(depth_shape)} must share the pixel frame")
    return mask[v, u]


def lift_mask(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, *,
              robot_removal: str = "none", robot_pixels: np.ndarray | None = None, range_max: float = RANGE_MAX_M
              ) -> np.ndarray:
    """Base-frame points ``(M, 3)`` of the masked pixels that have valid depth."""
    pts, u, v = backproject_pixels(depth_m, K, T_base_cam, range_max=range_max)
    return pts[lift_selection(mask, u, v, np.asarray(depth_m).shape, robot_removal=robot_removal,
                              robot_pixels=robot_pixels)]


def voxel_keys(points: np.ndarray, voxel: float = VOXEL_M) -> np.ndarray:
    """Unique integer voxel keys ``(K, 3)`` (global grid, ``floor(p / voxel)``; same as ifg_graph)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(p):
        return np.zeros((0, 3), np.int64)
    return np.unique(np.floor(p / voxel).astype(np.int64), axis=0)


def project_points(points_base: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray):
    """Base points -> ``(u, v, z_cam)`` (re-used ``camera.project_base_points``)."""
    return libero_camera.project_base_points(points_base, K, T_base_cam)


__all__ = ["POLICY_HW", "RANGE_MAX_M", "ROBOT_REMOVAL", "VOXEL_M", "apply_robot_removal", "backproject_pixels", "lift_mask",
           "lift_selection", "opencv_mask_to_policy", "opencv_rgb_from_raw", "policy_mask_to_opencv", "project_points", "robot_geom_ids",
           "voxel_keys"]
