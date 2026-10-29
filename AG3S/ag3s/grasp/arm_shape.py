"""AG3S-GEO-GRASP — the arm beyond the hand boxes, as occupancy grids of its real collision meshes (task §3g-1).

The candidate generator's hand is three boxes (palm, two fingers; `gripper.py`). Measured on the six pilot scenes
(task §3f, exact MuJoCo distances), 13 of 188 returned candidates put the wrist force-torque sensor body into an
obstacle and 0 put the arm links 0-6 there — the hand boxes do not cover the sensor. The URDF capsules over-cover the
arm (125 "collisions" for 13 real ones), and one convex hull per link over-covers it by 11-123 %; so every link here is
the **union of its MJCF collision convex meshes**, voxelized once (`res`, a cell is occupied when its centre lies in
one of the hulls) and dilated by the check margin. A point is inside when its cell is occupied; the resolution error
is at most half a cell diagonal.

Two groups:

* **wrist** — `link_<side>_arm_6` and `FT_SENSOR_<S>`, rigid with the hold link: grids in the hold-link frame, checked
  wherever the hand boxes are (candidate poses, approach segment, tracked-grasp re-check, close gate);
* **arm** — `link_<side>_arm_0..5`: grids in their own link frames, placed by forward kinematics of the IK solution
  (the URDF link frames equal the MJCF body frames).

MuJoCo is read only here, at build time; checks are plain numpy.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Callable, Optional, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class OccupancyShape:
    """Occupied cells `occ[nx, ny, nz]` of one body, cell (i, j, k) centred at `origin + (i, j, k) * res`, in `frame`."""

    name: str
    frame: str
    origin: np.ndarray
    res: float
    occ: np.ndarray

    @property
    def n_occupied(self) -> int:
        return int(self.occ.sum())

    def dilated(self, margin: float) -> "OccupancyShape":
        """The shape grown by `margin` (m): every cell within `margin` of an occupied cell (centre to centre)."""
        r = int(np.ceil(max(float(margin), 0.0) / self.res - 1e-9))
        if r == 0:
            return self
        from scipy.ndimage import binary_dilation

        g = np.arange(-r, r + 1)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        ball = (X ** 2 + Y ** 2 + Z ** 2) * self.res ** 2 <= float(margin) ** 2 + 1e-12
        occ = binary_dilation(np.pad(self.occ, r), structure=ball)
        return dataclasses.replace(self, origin=self.origin - r * self.res, occ=occ)

    def surface_points(self, spacing: float = 0.005) -> np.ndarray:
        """Centres (in `frame`) of the occupied cells on the shape's boundary (an unoccupied 6-neighbour), thinned to
        one per `spacing` voxel — where a distance field is queried (task §3j: a surface crossing the shape crosses
        its boundary; a shape is at least a few cells thick)."""
        o = np.pad(self.occ, 1)
        inner = o[1:-1, 1:-1, 1:-1] & o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] & o[1:-1, 2:, 1:-1] \
            & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:]
        P = self.origin + np.argwhere(self.occ & ~inner) * self.res
        if len(P) and spacing > self.res:
            _, keep = np.unique(np.floor(P / spacing).astype(np.int64), axis=0, return_index=True)
            P = P[np.sort(keep)]
        return P

    def contains(self, p: np.ndarray) -> np.ndarray:
        """`(N,)` bool: points `p[N, 3]` (in `frame`) inside an occupied cell."""
        p = np.asarray(p, np.float64).reshape(-1, 3)
        idx = np.floor((p - self.origin) / self.res + 0.5).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.asarray(self.occ.shape)), axis=1)
        out = np.zeros(len(p), bool)
        if ok.any():
            i = idx[ok]
            out[ok] = self.occ[i[:, 0], i[:, 1], i[:, 2]]
        return out

    @functools.cached_property
    def bound(self) -> tuple[np.ndarray, float]:
        """Centre and radius (frame) of a sphere containing every occupied cell — the cheap pre-filter."""
        ii = np.argwhere(self.occ)
        if not len(ii):
            return self.origin.copy(), 0.0
        P = self.origin + ii * self.res
        c = 0.5 * (P.min(0) + P.max(0))
        return c, float(np.linalg.norm(P - c, axis=1).max() + self.res)


def shapes_hits_many(points: np.ndarray, Ts: np.ndarray, shapes: Sequence[OccupancyShape]) -> np.ndarray:
    """`(K,)` number of `points` (base frame) inside any of `shapes` (all in one frame) placed at each pose `Ts[K]`."""
    out = np.zeros(len(Ts), int)
    if len(points) == 0 or len(Ts) == 0 or not shapes:
        return out
    pts = np.asarray(points, np.float64)
    bounds = [s.bound for s in shapes]
    for k, T in enumerate(np.asarray(Ts, np.float64)):
        R, t = T[:3, :3], T[:3, 3]
        inside = np.zeros(len(pts), bool)
        for s, (c, r) in zip(shapes, bounds):
            cw = R @ c + t
            near = np.einsum("ij,ij->i", pts - cw, pts - cw) <= r * r
            if near.any():
                q = (pts[near] - t) @ R
                inside[np.flatnonzero(near)[s.contains(q)]] = True
        out[k] = int(inside.sum())
    return out


def _body_meshes(m, d, body: str, frame_body: str) -> list[np.ndarray]:
    """Vertices (in `frame_body`'s frame) of every collision mesh geom of `body`, one array per geom."""
    import mujoco

    b, f = m.body(body).id, m.body(frame_body).id
    Rf, pf = d.xmat[f].reshape(3, 3), d.xpos[f]
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"{body}: collision geom {g} is not a mesh (type {int(m.geom_type[g])})")
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = (d.geom_xmat[g].reshape(3, 3) @ m.mesh_vert[a:a + n].T).T + d.geom_xpos[g]
        out.append((V - pf) @ Rf)
    return out


def voxelize_hulls(meshes: Sequence[np.ndarray], res: float, name: str, frame: str) -> OccupancyShape:
    """Occupancy grid of the union of the convex hulls of `meshes` (vertex arrays, one frame)."""
    from scipy.spatial import ConvexHull

    allv = np.vstack(meshes)
    lo, hi = allv.min(0) - res, allv.max(0) + res
    n = np.ceil((hi - lo) / res).astype(int) + 1
    ii = np.indices(n).reshape(3, -1).T
    C = lo + ii * res
    occ = np.zeros(len(C), bool)
    grid = occ.reshape(tuple(n))
    for V in meshes:
        h = ConvexHull(V)
        box = np.all((C >= V.min(0) - 1e-9) & (C <= V.max(0) + 1e-9), axis=1)
        if box.any():
            occ[np.flatnonzero(box)[np.all(C[box] @ h.equations[:, :3].T + h.equations[:, 3] <= 1e-9, axis=1)]] = True
        # the hull surface too, sampled finer than a cell: a part thinner than a cell has no cell centre inside it
        S = _hull_surface(V, h, 0.5 * res)
        idx = np.floor((S - lo) / res + 0.5).astype(np.int64)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return OccupancyShape(name=name, frame=frame, origin=lo, res=float(res), occ=grid)


def _hull_surface(V: np.ndarray, hull, step: float) -> np.ndarray:
    """Points on every facet of `hull` (over vertices `V`) no farther than `step` apart along the facet edges."""
    out = [V]
    for tri in hull.simplices:
        a, b, c = V[tri]
        k = max(int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / step)), 1)
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        i, j = i[keep], j[keep]
        out.append(a + np.outer(i / k, b - a) + np.outer(j / k, c - a))
    return np.vstack(out)


def measure_arm_from_mjcf(model_xml: str, hand: str = "left", res: float = 0.003, hold_link: Optional[str] = None
                          ) -> dict[str, tuple[OccupancyShape, ...]]:
    """`{"wrist": (arm_6, FT sensor) in the hold-link frame, "arm": (link 0..5) in their own frames}`."""
    import mujoco

    side, S = ("left", "L") if hand == "left" else ("right", "R")
    hold = hold_link or f"ee_{side}"
    import os

    m = mujoco.MjModel.from_xml_path(os.path.abspath(str(model_xml)))   # relative mesh dirs resolve from the file
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    ee = f"EE_BODY_{S}"
    wrist = tuple(voxelize_hulls(_body_meshes(m, d, b, ee), res, b, hold) for b in (f"link_{side}_arm_6", f"FT_SENSOR_{S}"))
    arm = tuple(voxelize_hulls(_body_meshes(m, d, f"link_{side}_arm_{k}", f"link_{side}_arm_{k}"), res,
                               f"link_{side}_arm_{k}", f"link_{side}_arm_{k}") for k in range(6))
    return {"wrist": wrist, "arm": arm}


class ArmChecker:
    """Obstacles inside the arm links 0..5 at a joint configuration.

    `link_pose(q_full, link) -> 4x4` is the robot's forward kinematics (base frame); `full_q(q_sub) -> q_full` puts the
    hand's arm joints into the current full configuration. The obstacles are either points (counted inside the shapes
    dilated by `margin`) or a distance function `field(points) -> (N,)` (the trajectory optimizer's field, task §3j;
    a boundary point of a link closer than `margin` to a surface is a hit)."""

    def __init__(self, shapes: Sequence[OccupancyShape], link_pose: Callable[[np.ndarray, str], np.ndarray],
                 full_q: Callable[[np.ndarray], np.ndarray], margin: float, surface_spacing: float = 0.005):
        self.raw = tuple(shapes)
        self.shapes = tuple(s.dilated(margin) for s in shapes)
        self.surfaces = tuple(s.surface_points(surface_spacing) for s in shapes)
        self.link_pose = link_pose
        self.full_q = full_q
        self.margin = float(margin)

    def __call__(self, q_sub: np.ndarray, obstacles) -> dict:
        """`{"hits": n, "links": [link with a hit, ...]}` for obstacle points (base frame) or a distance function."""
        if obstacles is None or (not callable(obstacles) and len(obstacles) == 0):
            return {"hits": 0, "links": []}
        q = self.full_q(np.asarray(q_sub, np.float64))
        hits, links = 0, []
        if callable(obstacles):
            Ts = [np.asarray(self.link_pose(q, s.frame), np.float64) for s in self.raw]
            P = np.vstack([S @ T[:3, :3].T + T[:3, 3] for S, T in zip(self.surfaces, Ts)])
            d = np.asarray(obstacles(P), np.float64)
            pos = 0
            for s, S in zip(self.raw, self.surfaces):
                n = int((d[pos:pos + len(S)] < self.margin).sum()); pos += len(S)
                if n:
                    hits += n
                    links.append(s.frame)
            return {"hits": hits, "links": links}
        for s in self.shapes:
            n = int(shapes_hits_many(obstacles, np.asarray(self.link_pose(q, s.frame), np.float64)[None], (s,))[0])
            if n:
                hits += n
                links.append(s.frame)
        return {"hits": hits, "links": links}


def field_hits_many(field: Callable[[np.ndarray], np.ndarray], Ts: np.ndarray, surface: np.ndarray, margin: float
                    ) -> np.ndarray:
    """`(K,)` number of `surface` points (a frame rigid with the poses `Ts[K]`) closer than `margin` to an obstacle
    surface by the distance function `field` — one batched query."""
    Ts = np.asarray(Ts, np.float64)
    if len(Ts) == 0 or len(surface) == 0:
        return np.zeros(len(Ts), int)
    P = np.einsum("kij,nj->kni", Ts[:, :3, :3], surface) + Ts[:, None, :3, 3]
    d = np.asarray(field(P.reshape(-1, 3)), np.float64).reshape(len(Ts), len(surface))
    return (d < margin).sum(axis=1)


__all__ = ["OccupancyShape", "shapes_hits_many", "field_hits_many", "voxelize_hulls", "measure_arm_from_mjcf",
           "ArmChecker"]
