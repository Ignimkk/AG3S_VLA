"""AEGIS obstacle geometry: point cloud -> workspace / outlier / density filter -> MVEE ellipsoid.

Ported from ``benchmark/SafeLIBERO/main/utils.py`` (THU-RCSCT/vlsa-aegis @ 2457feed; MIT):

* :func:`filtering_points` = ``filtering_points`` (utils.py:257-316). The workspace box is the official one, in the
  **MuJoCo world frame** (our points are in the robot base frame, so they are mapped with ``T_world_base`` first;
  without it the box step is skipped and reported). The DBSCAN step (utils.py:301-312, open3d
  ``cluster_dbscan(eps=0.0001, min_points=50)``) is re-implemented with a scipy KD-tree (open3d is not installed);
  same rule: keep the largest cluster if any cluster exists, else keep everything.
* the ellipsoid fit is the official ``fit_ellipse`` (utils.py:431-445: ConvexHull -> ``mvee_cvxpy`` with SCS ->
  eigen-decomposition, semi-axes sorted descending), imported unchanged through :mod:`.official`.
* point clouds come from :class:`~Baseline.paper_baselines.methods.safety_common.ObstacleCloudSource`; its
  ``detector`` source is the port of ``get_point_cloud`` (utils.py:155-255: Grounding DINO box -> crop of the depth
  image -> back-projection), with A's HF Grounding DINO instead of the original ``groundingdino`` package.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from Baseline.paper_baselines.methods.aegis.official import load_official_utils

#: utils.py:262-282 -- (z, x, y) bounds in the MuJoCo world frame per scene type.
WORKSPACE_WORLD = {
    "table": ((0.92, 1.5), (-0.3, 0.3), (-0.3, 0.3)),  # "spatial" / "goal"
    "floor": ((0.05, 0.5), (-0.3, 0.3), (-0.3, 0.3)),  # "object"
    "living_room": ((0.43, 0.8), (-0.3, 0.3), (-0.3, 0.3)),  # "long"
}


def scene_type(suite: str | None) -> str | None:
    """utils.py:262-282 dispatch on the task-suite name (LIBERO ``libero_10`` = SafeLIBERO ``long``)."""
    s = str(suite or "")
    if "spatial" in s or "goal" in s:
        return "table"
    if "object" in s:
        return "floor"
    if "long" in s or "libero_10" in s:
        return "living_room"
    return None


@dataclasses.dataclass
class ObstacleEllipsoid:
    """AEGIS convention: ``{y : ||diag(semi_axes)^-1 R^T (y - center)|| <= 1}`` (utils.py ``Q_diag`` = semi-axes)."""

    name: str
    center: np.ndarray  # (3,) base frame
    R: np.ndarray  # (3, 3)
    semi_axes: np.ndarray  # (3,)
    n_points: int = 0

    def shape_matrix(self) -> np.ndarray:
        """KNOWS / squared-radius convention ``Q = R diag(a^2) R^T``."""
        return self.R @ np.diag(np.asarray(self.semi_axes) ** 2) @ self.R.T

    def summary(self) -> dict:
        return {"name": self.name, "center": np.round(self.center, 4).tolist(),
                "semi_axes": np.round(self.semi_axes, 4).tolist(), "R": np.round(self.R, 5).tolist(),
                "n_points": int(self.n_points)}

    @staticmethod
    def from_dict(d: dict) -> "ObstacleEllipsoid":
        return ObstacleEllipsoid(str(d.get("name", "obstacle")), np.asarray(d["center"], float).reshape(3),
                                 np.asarray(d.get("R", np.eye(3)), float).reshape(3, 3),
                                 np.asarray(d["semi_axes"], float).reshape(3), int(d.get("n_points", 0)))


def _dbscan_largest(pts: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """open3d ``cluster_dbscan`` + "keep the largest cluster if labels.max() >= 0" (utils.py:302-311)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(pts)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(eps, output_type="ndarray")
    deg = np.bincount(pairs.reshape(-1), minlength=n) + 1  # open3d counts the point itself
    core = deg >= min_points
    if not core.any():
        return pts  # labels all -1 -> official code keeps every point
    keep = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    lab = np.where(core, lab, -1)
    # border points join the cluster of a core neighbour
    for a, b in pairs:
        if lab[a] < 0 and core[b]:
            lab[a] = lab[b]
        elif lab[b] < 0 and core[a]:
            lab[b] = lab[a]
    largest = np.bincount(lab[lab >= 0]).argmax()
    return pts[lab == largest]


def filtering_points(pts_base: np.ndarray, suite: str | None, T_world_base: np.ndarray | None = None,
                     *, keep_frac: float = 0.8, dbscan_eps: float = 0.0001, dbscan_min_points: int = 50
                     ) -> tuple[np.ndarray, dict]:
    """utils.py:257-316 on base-frame points; returns ``(points_base, info)``."""
    pts = np.asarray(pts_base, np.float64).reshape(-1, 3)
    info = {"n_in": int(len(pts))}
    st = scene_type(suite)
    if st is not None and T_world_base is not None:
        T = np.asarray(T_world_base, np.float64)
        w = pts @ T[:3, :3].T + T[:3, 3]
        (z0, z1), (x0, x1), (y0, y1) = WORKSPACE_WORLD[st]
        keep = (w[:, 2] > z0) & (w[:, 2] < z1) & (w[:, 0] > x0) & (w[:, 0] < x1) & (w[:, 1] > y0) & (w[:, 1] < y1)
        pts = pts[keep]
        info["workspace"] = st
    else:
        info["workspace"] = None  # official box needs the world frame / scene type: skipped
    info["n_workspace"] = int(len(pts))
    if len(pts) == 0:
        return pts, info
    d = np.linalg.norm(pts - pts.mean(0), axis=1)  # utils.py:290-296 keep the 80% nearest the centroid
    pts = pts[np.argsort(d)[: int(len(pts) * keep_frac)]]
    info["n_trim"] = int(len(pts))
    if len(pts) == 0:
        return pts, info
    pts = _dbscan_largest(pts, dbscan_eps, dbscan_min_points)
    info["n_out"] = int(len(pts))
    return pts, info


def fit_obstacle(name: str, points: np.ndarray, *, min_points: int = 10) -> ObstacleEllipsoid | None:
    """Official ``fit_ellipse`` (ConvexHull + cvxpy/SCS MVEE). ``None`` when too few / degenerate points."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) < min_points:
        return None
    U = load_official_utils()
    try:
        center, R, S = U.fit_ellipse(pts, plot=False)
    except Exception:  # degenerate hull (planar crop) or SCS failure -> no ellipsoid
        return None
    if not (np.all(np.isfinite(center)) and np.all(np.isfinite(S))):
        return None
    if np.linalg.det(R) < 0:
        R = R.copy()
        R[:, -1] *= -1.0
    return ObstacleEllipsoid(name, np.asarray(center, float), np.asarray(R, float), np.asarray(S, float), len(pts))


def perceive_obstacles(source, obs, *, ref_point: np.ndarray, max_obstacles: int = 8) -> tuple[list[ObstacleEllipsoid], dict]:
    """Per-episode AEGIS obstacle fit (main_aegis.py:184-233) for every cloud of ``source(obs)``.

    ``source`` is an ``ObstacleCloudSource`` or ``None`` (then ``extras["obstacle_ellipsoids"]`` are used as given).
    Obstacles are sorted by distance to ``ref_point`` and capped at ``max_obstacles``.
    """
    import time

    if source is None:
        obstacles = [ObstacleEllipsoid.from_dict(d) for d in obs.extras.get("obstacle_ellipsoids", [])]
        return obstacles[:max_obstacles], {"source": "ellipsoids", "privileged": False, "n": len(obstacles)}
    t0 = time.perf_counter()
    clouds = source(obs)
    suite, Twb = obs.extras.get("suite"), obs.extras.get("T_world_base")
    obstacles, per = [], {}
    for name, pts in clouds.points.items():
        fp, finfo = filtering_points(pts, suite, Twb)
        E = fit_obstacle(name, fp)
        per[name] = {**finfo, "fitted": E is not None}
        if E is not None:
            obstacles.append(E)
    obstacles.sort(key=lambda E: float(np.linalg.norm(E.center - np.asarray(ref_point))))
    info = {"source": clouds.source, "privileged": clouds.privileged, "excluded": clouds.excluded, "clouds": per,
            "detector": clouds.info, "ms": (time.perf_counter() - t0) * 1e3,
            "centroids": {k: v.mean(0).round(4).tolist() for k, v in clouds.points.items()}}
    return obstacles[:max_obstacles], info


__all__ = ["ObstacleEllipsoid", "WORKSPACE_WORLD", "filtering_points", "fit_obstacle", "perceive_obstacles", "scene_type"]
