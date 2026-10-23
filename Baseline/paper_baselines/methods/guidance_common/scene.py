"""SceneObservation -> what a guidance method needs (obstacles, target, EE state, hand spheres), all in the base frame.

Obstacle sources, in priority order (``obstacle_source="auto"``):

1. ``extras["privileged/obstacle_geoms"]`` -- simulator primitives (SafeLIBERO obstacle geoms; **privileged**);
2. ``extras["obstacle_points"]``           -- an (N, 3) base-frame cloud given by the caller (e.g. a cuRobo ESDF
   surface, a perception provider). Privileged only if the caller says so (``extras["obstacle_points_privileged"]``);
3. depth lift of every camera -- all valid pixels **minus** the target mask, minus the robot mask, minus the table
   plane, inside a workspace ball around the EE, voxel-downsampled. Masks are taken from (first found):
   ``extras["target_mask"][cam]`` (a grounding provider, non-privileged) or ``extras["privileged/target_mask"][cam]``
   (GT segmentation); ``extras["robot_mask"][cam]`` or ``extras["privileged/robot_mask"][cam]``; optional
   ``extras["(privileged/)ignore_mask"][cam]`` for pixels that are not obstacles (table, destination, ...).

Target: ``extras["target_points"]`` (N, 3), else the lifted target mask (provider or privileged GT).
Every privileged key read is listed in ``GuidanceScene.info["privileged_inputs"]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.geometry import Obstacles
from Baseline.paper_baselines.methods.interface import PRIVILEGED_PREFIX, SceneObservation

#: URDF ``panda_grip_site_joint`` (robots/panda/description/panda_libero.urdf:122): hand -> grip site.
T_HAND_GRIP = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.097], [0.0, 0.0, 0.0, 1.0]])
FINGER_Z = 0.0524  # panda_finger_joint{1,2} origin z (URDF:110,117)


def _pick(extras: dict, key: str, cam: str | None = None):
    """Non-privileged ``key`` first, then ``privileged/key``; returns (value, privileged_flag)."""
    for k, priv in ((key, False), (PRIVILEGED_PREFIX + key, True)):
        if k in extras and extras[k] is not None:
            v = extras[k]
            if cam is not None:
                if not isinstance(v, dict) or cam not in v:
                    continue
                v = v[cam]
            return v, priv
    return None, False


def voxel_downsample(points: np.ndarray, voxel: float, max_points: int, seed: int = 0) -> np.ndarray:
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(p):
        return p
    keys = np.floor(p / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1
    cent = np.zeros((n, 3))
    np.add.at(cent, inv, p)
    cent /= np.bincount(inv, minlength=n)[:, None]
    if len(cent) > max_points:  # deterministic subsample: keep the points nearest to ... everything (uniform stride)
        rng = np.random.default_rng(seed)
        cent = cent[np.sort(rng.choice(len(cent), max_points, replace=False))]
    return cent


def hand_spheres(ee_pose_base: np.ndarray, finger_opening_m: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """cuRobo ``franka.yml`` spheres of ``panda_hand`` + fingers at the current EE pose: ``(centres (S,3), radii (S,))``."""
    from AG3S.adapters.panda.model import franka_collision_spheres

    sph = franka_collision_spheres()
    T_hand = np.asarray(ee_pose_base, np.float64) @ np.linalg.inv(T_HAND_GRIP)
    cs, rs = [], []
    w = float(np.clip(finger_opening_m, 0.0, 0.04))
    frames = {"panda_hand": T_hand}
    for link, sgn in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0)):
        T = np.eye(4)
        T[:3, 3] = [0.0, sgn * w, FINGER_Z]
        frames[link] = T_hand @ T
    for link, T in frames.items():
        for c, r in sph.get(link, []):
            cs.append(T[:3, :3] @ c + T[:3, 3])
            rs.append(r)
    return np.asarray(cs), np.asarray(rs)


def estimate_table_z(points: np.ndarray, bin_m: float = 0.005) -> float | None:
    """Height of the dominant horizontal plane (densest z bin) among the lifted points."""
    if len(points) < 50:
        return None
    z = points[:, 2]
    lo, hi = np.percentile(z, 1), np.percentile(z, 60)
    if hi - lo < bin_m:
        return float(np.median(z))
    hist, edges = np.histogram(z[(z >= lo) & (z <= hi)], bins=max(4, int((hi - lo) / bin_m)))
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


@dataclass
class GuidanceScene:
    obstacles: Obstacles
    ee_pose: np.ndarray  # (4, 4) base
    R_bw: np.ndarray  # (3, 3) world -> base
    y_prev: np.ndarray  # (3,) last realised world EE step
    target_points: np.ndarray | None
    hand_c: np.ndarray  # (S, 3) hand + finger sphere centres at the current pose (base)
    hand_r: np.ndarray  # (S,)
    info: dict = field(default_factory=dict)

    @property
    def ee_pos(self) -> np.ndarray:
        return self.ee_pose[:3, 3].copy()

    @property
    def target_centroid(self) -> np.ndarray | None:
        return None if self.target_points is None or not len(self.target_points) else self.target_points.mean(0)


def build_scene(obs: SceneObservation, *, obstacle_source: str = "auto", voxel: float = 0.02, max_points: int = 1024,
                workspace_radius: float = 0.6, table_margin: float = 0.015, target_dilate: float = 0.0,
                remove_table: bool = True) -> GuidanceScene:
    """Collect obstacles / target / EE state from one observation (see module docstring for the priority)."""
    ex = dict(obs.extras or {})
    priv: list[str] = []
    info: dict = {"obstacle_source": None}
    ee_pose = np.asarray(obs.ee_pose, np.float64).reshape(4, 4)
    T_wb = np.asarray(ex.get("T_world_base", np.eye(4)), np.float64)
    R_bw = T_wb[:3, :3].T
    y_prev = np.asarray(ex.get("prev_ee_step_world", np.zeros(3)), np.float64).reshape(3)

    # --- target ---
    target_pts, tsrc = None, None
    if ex.get("target_points") is not None:
        target_pts, tsrc = np.asarray(ex["target_points"], np.float64).reshape(-1, 3), "target_points"
    elif ex.get(PRIVILEGED_PREFIX + "target_points") is not None:
        target_pts, tsrc = np.asarray(ex[PRIVILEGED_PREFIX + "target_points"], np.float64).reshape(-1, 3), "privileged/target_points"
        priv.append(tsrc)

    lifted_all: list[np.ndarray] = []
    lifted_target: list[np.ndarray] = []
    want_lift = obstacle_source in ("auto", "depth") or target_pts is None

    if want_lift and obs.depth:
        from Baseline.paper_baselines import lift as L

        for cam, depth in obs.depth.items():
            if cam not in obs.intrinsics or cam not in obs.T_base_cam:
                continue
            pts, u, v = L.backproject_pixels(depth, obs.intrinsics[cam], obs.T_base_cam[cam])
            keep = np.ones(len(pts), bool)
            tmask, tpriv = _pick(ex, "target_mask", cam)
            if tmask is not None:
                tm = np.asarray(tmask, bool)[v, u]
                lifted_target.append(pts[tm])
                keep &= ~tm
                if tpriv:
                    priv.append(f"{PRIVILEGED_PREFIX}target_mask/{cam}")
            for mkey in ("robot_mask", "ignore_mask"):  # ignore_mask: table / support / destination ... not obstacles
                m, mpriv = _pick(ex, mkey, cam)
                if m is not None:
                    keep &= ~np.asarray(m, bool)[v, u]
                    if mpriv:
                        priv.append(f"{PRIVILEGED_PREFIX}{mkey}/{cam}")
            lifted_all.append(pts[keep])
        if target_pts is None and lifted_target and sum(len(t) for t in lifted_target):
            target_pts = np.concatenate(lifted_target)
            tsrc = "lifted target_mask"
    info["target_source"] = tsrc

    # --- obstacles ---
    obstacles = None
    if obstacle_source in ("auto", "privileged") and ex.get(PRIVILEGED_PREFIX + "obstacle_geoms"):
        prims = ex[PRIVILEGED_PREFIX + "obstacle_geoms"]
        obstacles = Obstacles.from_primitives(list(prims))
        info["obstacle_source"] = "privileged/obstacle_geoms"
        priv.append(PRIVILEGED_PREFIX + "obstacle_geoms")
    elif obstacle_source in ("auto", "points") and ex.get("obstacle_points") is not None:
        pts = voxel_downsample(np.asarray(ex["obstacle_points"]), voxel, max_points)
        obstacles = Obstacles(points=pts, point_radius=0.5 * voxel)
        info["obstacle_source"] = "obstacle_points"
        if ex.get("obstacle_points_privileged"):
            priv.append("obstacle_points (privileged by caller)")
    elif obstacle_source in ("auto", "depth") and lifted_all:
        pts = np.concatenate(lifted_all)
        pts = pts[np.linalg.norm(pts - ee_pose[:3, 3][None], axis=1) <= workspace_radius]
        table_z = ex.get("table_z")
        if table_z is None and remove_table:
            table_z = estimate_table_z(pts)
        if remove_table and table_z is not None:
            pts = pts[pts[:, 2] > float(table_z) + table_margin]
        info["table_z"] = None if table_z is None else float(table_z)
        if target_dilate > 0 and target_pts is not None and len(target_pts) and len(pts):
            from scipy.spatial import cKDTree

            d, _ = cKDTree(target_pts).query(pts, k=1)
            pts = pts[d > target_dilate]
        pts = voxel_downsample(pts, voxel, max_points)
        obstacles = Obstacles(points=pts, point_radius=0.5 * voxel)
        info["obstacle_source"] = "depth_lift"
    if obstacles is None:
        obstacles = Obstacles()
        info["obstacle_source"] = "none"

    hc, hr = hand_spheres(ee_pose, float(ex.get("finger_opening_m", 0.04)))
    info.update(obstacle_counts=obstacles.counts(), n_target_points=0 if target_pts is None else int(len(target_pts)),
                privileged_inputs=sorted(set(priv)), voxel_m=voxel, max_points=max_points)
    return GuidanceScene(obstacles=obstacles, ee_pose=ee_pose, R_bw=R_bw, y_prev=y_prev, target_points=target_pts,
                         hand_c=hc, hand_r=hr, info=info)


__all__ = ["FINGER_Z", "GuidanceScene", "T_HAND_GRIP", "build_scene", "estimate_table_z", "hand_spheres", "voxel_downsample"]
