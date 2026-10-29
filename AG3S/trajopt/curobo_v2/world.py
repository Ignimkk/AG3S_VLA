"""T63 — AG3S ESDF arrays -> cuRobo v2 voxel world (pure numpy part).

Conventions (checked against the code, not assumed):

| item | AG3S record (`esdf_*` in `chunk_*.npz`) | cuRobo v2 `VoxelGrid` / `VoxelData` |
|---|---|---|
| frame | robot `base` (`frames.jsonl` field `frame_id = base`) | the solver's base link — here `link_torso_5`, so each grid gets the pose `inv(T_base_torso5) · T_base_grid` |
| origin | **centre of voxel (0,0,0)** (`curobo_field.layer_from_arrays`) | `pose` = **centre of the grid box**; voxel `i` centre at `pose + (i - n/2 + 0.5) · voxel` (`VoxelGrid.create_xyzr_tensor`, `sample_voxel_sdf_with_grad`) -> `pose = origin + (n - 1) / 2 · voxel` |
| index order | `(nx, ny, nz)`, z fastest (C order) | `voxel_idx_to_flat = x·ny·nz + y·nz + z` — same, the grid is copied flat without reindexing |
| sign | negative inside, positive outside | `compute_local_sdf`: "negative inside, positive outside" — same, no flip (`get_occupied_voxels` is not used) |
| interpolation | trilinear inside the lattice (first..last voxel centre) | trilinear; at the border only in-bounds corners count; fully outside -> `max_esdf_distance` |
| dtype | float16 | float16 (`VoxelData.features`) — the recorded values are loaded bit-for-bit |

cuRobo's collision kernel does not return a signed distance; it returns
`w · act(r + eta - sdf)` summed over obstacles (`wp_collision_common.apply_collision_activation`).
`sdf_from_activation_cost` inverts that for one obstacle at a time (weight 1, a large `eta`).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

LAYER_PREFIXES = {"main": "esdf", "coarse": "esdf_coarse", "free": "esdf_free"}


def layer_arrays(z: Mapping[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    """`{distance (nx,ny,nz) float16, origin (3,), voxel_size}` of one recorded layer."""
    p = LAYER_PREFIXES[name]
    d = np.asarray(z[f"{p}_distance"])
    origin = np.asarray(z[f"{p}_origin"], np.float64).reshape(3)
    vs = float(np.asarray(z[f"{p}_voxel_size"]))
    if d.ndim != 3:
        raise ValueError(f"{p}_distance must be 3-D, got {d.shape}")
    if f"{p}_shape" in getattr(z, "files", z):
        shape = tuple(int(v) for v in np.asarray(z[f"{p}_shape"]))
        if shape != d.shape:
            raise ValueError(f"{p}: recorded shape {shape} != array shape {d.shape}")
    return {"distance": d, "origin": origin, "voxel_size": vs}


def grid_centre(origin: np.ndarray, shape, voxel_size: float) -> np.ndarray:
    """cuRobo grid `pose` (box centre) from our `origin` (centre of voxel 0)."""
    return (np.asarray(origin, np.float64).reshape(3)
            + 0.5 * (np.asarray(shape, np.float64) - 1.0) * float(voxel_size))


def grid_dims(shape, voxel_size: float) -> np.ndarray:
    """cuRobo `VoxelGrid.dims` (metres). `get_grid_shape` rounds `dims / voxel` back to `shape`."""
    return np.asarray(shape, np.float64) * float(voxel_size)


def voxel_centres(origin: np.ndarray, shape, voxel_size: float) -> np.ndarray:
    """`(nx·ny·nz, 3)` voxel centres in our convention, C order (for tests)."""
    axes = [np.asarray(origin, np.float64)[k] + np.arange(int(shape[k])) * float(voxel_size)
            for k in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return g.reshape(-1, 3)


def rotation_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion `[w, x, y, z]` (cuRobo's pose order), w >= 0."""
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def grid_pose_in_frame(T_base_frame: np.ndarray, centre_base: np.ndarray) -> list[float]:
    """Pose `[x, y, z, qw, qx, qy, qz]` of a base-axis-aligned grid centred at `centre_base`,
    expressed in the solver frame whose base pose is `T_base_frame`."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(centre_base, np.float64).reshape(3)
    T_frame_grid = np.linalg.inv(np.asarray(T_base_frame, np.float64)) @ T
    return [*T_frame_grid[:3, 3].tolist(), *rotation_to_quat_wxyz(T_frame_grid[:3, :3]).tolist()]


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    p = np.asarray(pts, np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


def sdf_from_activation_cost(cost: np.ndarray, radius: np.ndarray, eta: float) -> np.ndarray:
    """Invert `cost = act(r + eta - sdf)` (weight 1, one obstacle) to the signed distance.

    `act(p) = p - eta/2` for `p > eta`, `p² / (2 eta)` for `0 < p <= eta`, `0` for `p <= 0`
    (`apply_collision_activation`). With `eta` larger than the grid's `max_esdf_distance + r`
    every in-grid sphere has `p > 0`. A cost of exactly 0 means `sdf >= r + eta` (reported as
    `+inf`).
    """
    c = np.asarray(cost, np.float64)
    r = np.asarray(radius, np.float64)
    out = np.full(np.broadcast(c, r).shape, np.inf)
    lin = c > 0.5 * eta
    quad = (c > 0.0) & ~lin
    p_lin = c + 0.5 * eta
    p_quad = np.sqrt(np.maximum(2.0 * eta * c, 0.0))
    out = np.where(lin, r + eta - p_lin, out)
    out = np.where(quad, r + eta - p_quad, out)
    return out


def inside_lattice(points: np.ndarray, origin: np.ndarray, shape, voxel_size: float) -> np.ndarray:
    """Same rule as `CuroboEsdfField._covers`: first voxel centre .. last voxel centre."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    o = np.asarray(origin, np.float64).reshape(3)
    last = o + (np.asarray(shape, np.float64) - 1.0) * float(voxel_size)
    return np.all((pts >= o) & (pts <= last), axis=1)


def compose_main(d_coarse: np.ndarray, d_fine: np.ndarray, inside_fine: np.ndarray) -> np.ndarray:
    """Our two-tier main field (`CuroboEsdfField._evaluate`): coarse everywhere, and inside the
    fine lattice the smaller of the two. The composition rule is ours; cuRobo supplies the
    per-layer lookups."""
    d = np.asarray(d_coarse, np.float64).copy()
    f = np.asarray(d_fine, np.float64)
    m = np.asarray(inside_fine, bool)
    d[m] = np.minimum(d[m], f[m])
    return d


def compose_box(d_box: np.ndarray, inside_box: np.ndarray, d_coarse: np.ndarray) -> np.ndarray:
    """The B0 world of user decision 2 (2026-10-21): inside the box lattice the box layer
    (`esdf_free`, or the main fine layer for a chunk without one), outside it the coarse layer.
    Unlike `compose_main` there is **no** `min` with coarse inside the box — the coarse layer still
    holds the target there."""
    return np.where(np.asarray(inside_box, bool), np.asarray(d_box, np.float64),
                    np.asarray(d_coarse, np.float64))


def compose_b0(d_box: np.ndarray, inside_box: np.ndarray, d_coarse: np.ndarray, *,
               has_free: bool) -> np.ndarray:
    """The B0 world as used from T63 step 3 (`composed_world`, mode `b0`). With `esdf_free`:
    `compose_box` (user decision 2). Without it (the box is then the main fine layer): our main
    field `compose_main` — user decision 3, lead default (step 2 used the fine layer alone)."""
    if has_free:
        return compose_box(d_box, inside_box, d_coarse)
    return compose_main(d_coarse, d_box, inside_box)


def compose_lte(d_coarse: np.ndarray, d_fine: np.ndarray, inside_fine: np.ndarray,
                d_free: np.ndarray, inside_free: np.ndarray, excl_rows: np.ndarray) -> np.ndarray:
    """B4's per-link world (`composed_world`, mode `lte`) on `(..., S)` arrays: target-excluded rows
    (`excl_rows (S,)`) get `compose_box(free)`, the others our main field `compose_main`."""
    main = compose_main(d_coarse, d_fine, inside_fine)
    box = compose_box(d_free, inside_free, d_coarse)
    return np.where(np.asarray(excl_rows, bool), box, main)


__all__ = [
    "LAYER_PREFIXES", "compose_b0", "compose_box", "compose_lte", "compose_main", "grid_centre", "grid_dims", "grid_pose_in_frame",
    "inside_lattice", "layer_arrays", "rotation_to_quat_wxyz", "sdf_from_activation_cost",
    "transform_points", "voxel_centres",
]
