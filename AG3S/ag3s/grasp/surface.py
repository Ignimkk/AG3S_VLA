"""AG3S-GEO-GRASP — surface normals of the observed target points.

Normals come from a PCA of each point's neighbourhood (smallest-eigenvalue direction). Their sign is set **outward**
from the target's centre — the centre of the surface-sphere fit when it is accepted (`fit_surface_sphere`, convex
objects such as the apple), otherwise the points' centroid. A one-sided view biases the centroid towards the camera;
the sphere fit does not. The planarity ratio ``lambda_min / sum(lambda)`` of each neighbourhood is kept: a large value
means the neighbourhood is not a surface patch (noise, an edge) and the normal is not trusted.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class SurfaceEstimate:
    """`points[N, 3]` (downsampled), unit outward `normals[N, 3]`, `curvature[N]` (planarity ratio), `n_neighbours[N]`,
    the centre used for the sign and how it was found (``"sphere"`` / ``"centroid"``), and the sphere radius (m, or None)."""

    points: np.ndarray
    normals: np.ndarray
    curvature: np.ndarray
    n_neighbours: np.ndarray
    centre: np.ndarray
    centre_method: str
    radius: Optional[float]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One point (the mean) per occupied voxel of size `voxel` (m). Order: sorted voxel keys (deterministic)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) == 0 or voxel <= 0.0:
        return p
    keys = np.floor(p / float(voxel)).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((counts.size, 3))
    np.add.at(out, inv, p)
    return out / counts[:, None]


def estimate_surface(points: np.ndarray, *, radius: float = 0.012, min_neighbours: int = 6,
                     max_object_radius: Optional[float] = None) -> SurfaceEstimate:
    """Outward normals of `points` (already downsampled) from neighbours within `radius` (m).

    A point with fewer than `min_neighbours` neighbours gets a zero normal and curvature 1 (not a usable contact)."""
    from scipy.spatial import cKDTree

    from AG3S.ag3s.robot_models.held_object import fit_surface_sphere

    p = np.asarray(points, np.float64).reshape(-1, 3)
    n = len(p)
    if n == 0:
        z = np.zeros((0, 3))
        return SurfaceEstimate(z, z, np.zeros(0), np.zeros(0, int), np.zeros(3), "none", None)
    try:
        fit = fit_surface_sphere(p, max_radius=max_object_radius)
        if fit.accepted:
            centre, method, rad = np.asarray(fit.centre, np.float64), "sphere", float(fit.radius)
        else:
            centre, method, rad = p.mean(axis=0), "centroid", None
    except ValueError:
        centre, method, rad = p.mean(axis=0), "centroid", None
    # batched PCA over the neighbour pairs (each point is its own neighbour)
    pairs = cKDTree(p).query_pairs(r=float(radius), output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)]) if len(pairs) else np.arange(n)
    j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)]) if len(pairs) else np.arange(n)
    count = np.bincount(i, minlength=n)
    mean = np.zeros((n, 3))
    np.add.at(mean, i, p[j])
    mean /= count[:, None]
    dq = p[j] - mean[i]
    cov = np.zeros((n, 3, 3))
    np.add.at(cov, i, dq[:, :, None] * dq[:, None, :])
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0].copy()
    flip = np.einsum("ij,ij->i", normals, p - centre) < 0.0
    normals[flip] *= -1.0
    curv = w[:, 0] / np.maximum(w.sum(axis=1), 1e-18)
    few = count < int(min_neighbours)
    normals[few] = 0.0
    curv[few] = 1.0
    return SurfaceEstimate(p, normals, curv, count, centre, method, rad)


__all__ = ["SurfaceEstimate", "voxel_downsample", "estimate_surface"]
