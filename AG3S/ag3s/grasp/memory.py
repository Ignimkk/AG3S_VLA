"""AG3S-GEO-GRASP — accumulated observation of the target before the grasp.

One frame of the recorded cloud holds ~100 points of the apple (~8 mm apart) — too sparse for pad-sized contact
patches, and one-sided. Before the grasp the object does not move, so the observed points of the same target (ATS
identity) are merged across requests in the base frame (voxel `voxel`, newest point per voxel kept). The memory is
cleared when the target id changes or the observed centroid moves more than `max_jump` from the centroid the memory
was started at (the object was pushed, swapped or re-identified) — stale points are never taken as surface.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class TargetPointMemory:
    def __init__(self, voxel: float = 0.003, max_jump: float = 0.03, max_points: int = 6000,
                 max_surface_gap: float = 0.007, min_points_for_gap: int = 50):
        self.voxel = float(voxel)
        #: the new frame's points lie (median) farther than this from the sphere fitted to the remembered points ->
        #: the object moved. A sphere, not the nearest remembered point: a new viewpoint sees surface not yet in the
        #: memory, which the sphere already accounts for (the apple departs from its best sphere by up to ~5 mm).
        self.max_surface_gap = float(max_surface_gap)
        self.min_points_for_gap = int(min_points_for_gap)
        self.max_jump = float(max_jump)
        self.max_points = int(max_points)
        self.reset()

    def reset(self) -> None:
        self.target_id: Optional[int] = None
        self.ref_centroid: Optional[np.ndarray] = None
        self._vox: dict[tuple[int, int, int], np.ndarray] = {}
        self.n_frames = 0

    @property
    def points(self) -> np.ndarray:
        return np.array(list(self._vox.values())) if self._vox else np.zeros((0, 3))

    def update(self, target_id: Optional[int], points: Optional[np.ndarray], centroid: Optional[np.ndarray],
               observed: bool = True) -> dict[str, Any]:
        """Merge this request's observed target points (when `observed`); returns the record."""
        rec: dict[str, Any] = {"cleared": None}
        if target_id is None:
            if self._vox:
                rec["cleared"] = "no target"
            self.reset()
            return self._rec(rec)
        if self.target_id is not None and int(target_id) != int(self.target_id):
            rec["cleared"] = "target changed"
            self.reset()
        if centroid is not None and self.ref_centroid is not None and observed:
            jump = float(np.linalg.norm(np.asarray(centroid, np.float64) - self.ref_centroid))
            rec["centroid_shift_mm"] = round(jump * 1000, 1)
            if jump > self.max_jump:
                rec["cleared"] = "target moved"
                self.reset()
        self.target_id = int(target_id)
        if observed and points is not None and len(points):
            p = np.asarray(points, np.float64).reshape(-1, 3)
            p = p[np.all(np.isfinite(p), axis=1)]
            if len(self._vox) >= self.min_points_for_gap and len(p) >= 10:
                from AG3S.ag3s.robot_models.held_object import fit_surface_sphere

                fit = fit_surface_sphere(self.points)
                gap = (float(np.median(np.abs(np.linalg.norm(p - fit.centre, axis=1) - fit.radius)))
                       if fit.accepted else 0.0)
                rec["surface_gap_mm"] = round(gap * 1000, 2)
                if fit.accepted and gap > self.max_surface_gap:
                    # seen surface no longer where it was remembered (pushed by the hand, re-identified)
                    rec["cleared"] = "surface moved"
                    tid = self.target_id
                    self.reset()
                    self.target_id = tid
            if self.ref_centroid is None and centroid is not None:
                self.ref_centroid = np.asarray(centroid, np.float64).copy()
            keys = np.floor(p / self.voxel).astype(np.int64)
            for k, q in zip(map(tuple, keys), p):
                self._vox[k] = q
            if len(self._vox) > self.max_points:
                # drop the oldest voxels (dict keeps insertion order; re-inserted keys stay at their first place,
                # so this drops the earliest-seen surface first)
                for k in list(self._vox)[:len(self._vox) - self.max_points]:
                    del self._vox[k]
            self.n_frames += 1
        return self._rec(rec)

    def _rec(self, rec: dict[str, Any]) -> dict[str, Any]:
        rec.update(target_id=self.target_id, n_points=len(self._vox), n_frames=self.n_frames)
        return rec


class ObstacleMemory:
    """Scene points (everything but the target) accumulated over an episode's pre-grasp requests (task §3g-1, G2).

    One request's cloud misses obstacle parts that another view saw (occlusion, the moving wrist camera); the trajectory
    optimizer's TSDF keeps them for the whole episode, so the grasp checks keep them too. Voxel `voxel`, newest point
    per voxel, the oldest voxels dropped beyond `max_points`. Unlike the TSDF nothing is cleared along rays: a point
    stays until `reset` (episode start)."""

    def __init__(self, voxel: float = 0.005, max_points: int = 60000):
        self.voxel = float(voxel)
        self.max_points = int(max_points)
        self.reset()

    def reset(self) -> None:
        self._vox: dict[tuple[int, int, int], np.ndarray] = {}
        self.n_frames = 0

    @property
    def points(self) -> np.ndarray:
        return np.array(list(self._vox.values())) if self._vox else np.zeros((0, 3))

    def update(self, points: Optional[np.ndarray]) -> dict[str, Any]:
        if points is not None and len(points):
            p = np.asarray(points, np.float64).reshape(-1, 3)
            p = p[np.all(np.isfinite(p), axis=1)]
            for k, q in zip(map(tuple, np.floor(p / self.voxel).astype(np.int64)), p):
                self._vox[k] = q
            if len(self._vox) > self.max_points:
                for k in list(self._vox)[:len(self._vox) - self.max_points]:
                    del self._vox[k]
            self.n_frames += 1
        return {"n_points": len(self._vox), "n_frames": self.n_frames}

    def without(self, target_points: Optional[np.ndarray], radius: float = 0.005) -> np.ndarray:
        """The remembered points farther than `radius` from every `target_points` point (the target is not an
        obstacle; its remembered surface grows over the requests)."""
        P = self.points
        if len(P) == 0 or target_points is None or len(target_points) == 0:
            return P
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.asarray(target_points, np.float64).reshape(-1, 3)).query(P, distance_upper_bound=radius)
        return P[~np.isfinite(d)]


__all__ = ["TargetPointMemory", "ObstacleMemory"]
