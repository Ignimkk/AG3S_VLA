"""LIBERO scene ground truth in the Panda base frame (SL4), from ``scene_export.py`` files. Python 3.11, numpy.

``benchmark/adapters/libero_panda/scene_export.py`` (``.venv-libero``) restores the recorded image-instant
simulator state of each frame and writes the MuJoCo geom poses, the static geom tables and the meshes of
every non-robot geom. This module turns one such file into:

* **instances** (:meth:`SceneGT.instances`) -- every BDDL object and fixture of the task
  (``task_target.scene_instances``: ``<name>_main`` / ``<name>_<part>`` bodies, ``main_table`` -> ``table``)
  with a role (``target`` / ``destination`` / ``object`` / ``fixture``) plus ``mount`` (``mount0_*``, robot
  pedestal; SL2/SL3 self-filter residue);
* **surface samples** (:meth:`SceneGT.samples`) -- points on each instance's **visual** geoms (group 1:
  what the cameras render; collision geoms are hidden), area-weighted, deterministic (seed), stored in the
  geom frame and placed per frame by :meth:`SceneGT.points_base`;
* **exact distances** (:meth:`SceneGT.exact_distance`) -- Euclidean distance from query points (e.g.
  constraint sphere centres) to each instance's visual surface triangles, base frame. Meshes are used as
  they are (no convex hull); primitives are triangulated (box exact; cylinder / capsule / sphere with
  :data:`SEGMENTS` segments, surface error <= ``r (1 - cos(pi / SEGMENTS))`` = 0.05 % of the radius);
* **containment** (:meth:`SceneGT.inside`) -- whether a point is inside an instance's **collision**
  primitives (box / sphere / cylinder / capsule / ellipsoid analytic; LIBERO object collisions are boxes),
  to give the exact distance a sign;
* :func:`depth_residual` -- recorded depth minus the point's camera z (visibility test for surface samples).

Frames: base = ``robot0_base`` (``T_world_base`` of the export, checked against the record there).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = range(8)
#: Segments of a triangulated cylinder / capsule / sphere ring.
SEGMENTS = 96
#: Rendered (visual) geom group in robosuite / LIBERO; group 0 = collision, not rendered.
VISUAL_GROUP = 1
ROLES = ("target", "destination", "object", "fixture", "mount")
#: Default surface sample spacing (m) and per-instance cap (deterministic subsample beyond it).
SAMPLE_SPACING = 0.004
SAMPLE_CAP = 3000


def _rigid_inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


# --- triangulation ---------------------------------------------------------------------------------------
def _box(size):
    hx, hy, hz = size
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.int64)
    return v, f


def _uv_sphere(r, n=SEGMENTS, z_lo=-np.pi / 2, z_hi=np.pi / 2, centre_z=0.0):
    rings = max(4, n // 2)
    lat = np.linspace(z_lo, z_hi, rings + 1)
    lon = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.array([[r * np.cos(a) * np.cos(b), r * np.cos(a) * np.sin(b), centre_z + r * np.sin(a)] for a in lat for b in lon])
    f = []
    for i in range(rings):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            f += [[a, b, d], [a, d, c]]
    return v, np.array(f, np.int64)


def _cylinder(r, h, n=SEGMENTS, caps=True):
    lon = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(lon), r * np.sin(lon)], 1)
    v = np.concatenate([np.c_[ring, np.full(n, -h)], np.c_[ring, np.full(n, h)], [[0, 0, -h], [0, 0, h]]])
    f = []
    for j in range(n):
        a, b = j, (j + 1) % n
        f += [[a, b, b + n], [a, b + n, a + n]]
        if caps:
            f += [[2 * n, b, a], [2 * n + 1, a + n, b + n]]
    return v, np.array(f, np.int64)


def _merge(parts):
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def primitive_mesh(gtype: int, size: Sequence[float], n: int = SEGMENTS):
    """``(V, F)`` triangle mesh of a MuJoCo primitive in its geom frame."""
    s = np.asarray(size, np.float64)
    if gtype == BOX:
        return _box(s)
    if gtype == SPHERE:
        return _uv_sphere(s[0], n)
    if gtype == ELLIPSOID:
        v, f = _uv_sphere(1.0, n)
        return v * s[:3], f
    if gtype == CYLINDER:
        return _cylinder(s[0], s[1], n)
    if gtype == CAPSULE:
        side = _cylinder(s[0], s[1], n, caps=False)
        top = _uv_sphere(s[0], n, 0.0, np.pi / 2, s[1])
        bottom = _uv_sphere(s[0], n, -np.pi / 2, 0.0, -s[1])
        return _merge([side, top, bottom])
    if gtype == PLANE:
        hx, hy = (s[0] or 3.0), (s[1] or 3.0)
        return np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], np.float64), np.array([[0, 1, 2], [0, 2, 3]])
    raise ValueError(f"unsupported geom type {gtype}")


# --- point / triangle ----------------------------------------------------------------------------------
def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point of every triangle to every point (Ericson, Real-Time Collision Detection 5.1.5).

    ``p`` (N, 3); ``a, b, c`` (T, 3). Returns (N, T, 3).
    """
    p = np.asarray(p, np.float64)[:, None, :]
    a, b, c = (np.asarray(x, np.float64)[None] for x in (a, b, c))
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    dot = lambda x, y: np.einsum("ntk,ntk->nt", np.broadcast_to(x, ap.shape), np.broadcast_to(y, ap.shape))  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        q = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]
        # later assignments win: write in reverse Ericson order (BC, AC, C, AB, B, A)
        den = (d4 - d3) + (d5 - d6)
        w = np.where(np.abs(den) < 1e-300, 0.0, (d4 - d3) / np.where(np.abs(den) < 1e-300, 1.0, den))
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        q = np.where(m[..., None], b + (c - b) * w[..., None], q)
        den = d2 - d6
        w = np.where(np.abs(den) < 1e-300, 0.0, d2 / np.where(np.abs(den) < 1e-300, 1.0, den))
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        q = np.where(m[..., None], a + ac * w[..., None], q)
        m = (d6 >= 0) & (d5 <= d6)
        q = np.where(m[..., None], np.broadcast_to(c, q.shape), q)
        den = d1 - d3
        v = np.where(np.abs(den) < 1e-300, 0.0, d1 / np.where(np.abs(den) < 1e-300, 1.0, den))
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        q = np.where(m[..., None], a + ab * v[..., None], q)
        m = (d3 >= 0) & (d4 <= d3)
        q = np.where(m[..., None], np.broadcast_to(b, q.shape), q)
        m = (d1 <= 0) & (d2 <= 0)
        q = np.where(m[..., None], np.broadcast_to(a, q.shape), q)
    return q


def point_triangles_distance(p: np.ndarray, tri: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """``(N,)`` min distance from each point to a triangle soup ``tri`` (T, 3, 3)."""
    p = np.asarray(p, np.float64).reshape(-1, 3)
    out = np.full(len(p), np.inf)
    if not len(tri) or not len(p):
        return out
    step = max(1, chunk // max(1, len(p)))
    for s in range(0, len(tri), step):
        t = tri[s:s + step]
        q = closest_points_on_triangles(p, t[:, 0], t[:, 1], t[:, 2])
        out = np.minimum(out, np.sqrt(((q - p[:, None, :]) ** 2).sum(-1)).min(1))
    return out


def closest_points_pairwise(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point of triangle ``m`` to point ``m`` (all (M, 3)); the algorithm of
    :func:`closest_points_on_triangles` (Ericson 5.1.5) element-wise instead of all-pairs."""
    p, a, b, c = (np.asarray(x, np.float64).reshape(-1, 3) for x in (p, a, b, c))
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    dot = lambda x, y: np.einsum("mk,mk->m", x, y)  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        q = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
        den = (d4 - d3) + (d5 - d6)
        w = np.where(np.abs(den) < 1e-300, 0.0, (d4 - d3) / np.where(np.abs(den) < 1e-300, 1.0, den))
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        q = np.where(m[:, None], b + (c - b) * w[:, None], q)
        den = d2 - d6
        w = np.where(np.abs(den) < 1e-300, 0.0, d2 / np.where(np.abs(den) < 1e-300, 1.0, den))
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        q = np.where(m[:, None], a + ac * w[:, None], q)
        m = (d6 >= 0) & (d5 <= d6)
        q = np.where(m[:, None], c, q)
        den = d1 - d3
        v = np.where(np.abs(den) < 1e-300, 0.0, d1 / np.where(np.abs(den) < 1e-300, 1.0, den))
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        q = np.where(m[:, None], a + ab * v[:, None], q)
        m = (d3 >= 0) & (d4 <= d3)
        q = np.where(m[:, None], b, q)
        m = (d1 <= 0) & (d2 <= 0)
        q = np.where(m[:, None], a, q)
    return q


def point_triangles_distance_near(p: np.ndarray, tri: np.ndarray, max_distance: float,
                                  pair_budget: int = 4_000_000) -> np.ndarray:
    """``(N,)`` min distance from each point to a triangle soup, **exact where it is <= max_distance** and ``inf``
    otherwise.

    Candidate (point, triangle) pairs come from a KD-tree of the points: triangle ``t`` with centroid ``c_t`` and
    circumradius ``r_t = max |v - c_t|`` can only be within ``max_distance`` of points with
    ``|p - c_t| <= r_t + max_distance`` (triangle inequality through its closest point), so no pair that can be
    within ``max_distance`` is skipped. Each candidate pair is then solved exactly
    (:func:`closest_points_pairwise`, Ericson), ``pair_budget`` pairs at a time."""
    from scipy.spatial import cKDTree

    p = np.asarray(p, np.float64).reshape(-1, 3)
    tri = np.asarray(tri, np.float64).reshape(-1, 3, 3)
    out = np.full(len(p), np.inf)
    if not len(tri) or not len(p):
        return out
    lo, hi = tri.min(1), tri.max(1)
    # a triangle within max_distance of some point has its box inside the points' box grown by max_distance
    keep = np.all((hi >= p.min(0) - max_distance) & (lo <= p.max(0) + max_distance), axis=1)
    if not keep.any():
        return out
    tri = tri[keep]
    centre = tri.mean(1)
    radius = np.sqrt(((tri - centre[:, None]) ** 2).sum(-1)).max(1)
    lists = cKDTree(p).query_ball_point(centre, radius + float(max_distance) + 1e-12)
    counts = np.fromiter((len(x) for x in lists), np.int64, count=len(lists))
    if not counts.sum():
        return out
    ti = np.repeat(np.arange(len(tri)), counts)
    pi = np.concatenate([np.asarray(x, np.int64) for x in lists if len(x)])
    step = max(1, int(pair_budget))
    for s in range(0, len(pi), step):
        a, b = pi[s:s + step], ti[s:s + step]
        cp = closest_points_pairwise(p[a], tri[b, 0], tri[b, 1], tri[b, 2])
        np.minimum.at(out, a, np.sqrt(((cp - p[a]) ** 2).sum(-1)))
    out[out > max_distance] = np.inf
    return out


def sample_triangles(v: np.ndarray, f: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted uniform samples, ``ceil(area / spacing^2)`` points."""
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(area.sum())
    if total <= 0.0:
        return np.zeros((0, 3))
    n = int(np.ceil(total / spacing ** 2))
    idx = rng.choice(len(tri), size=n, p=area / total)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    t = tri[idx]
    return (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]


# --- scene ---------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Instance:
    name: str
    role: str
    visual_geoms: tuple[int, ...]
    collision_geoms: tuple[int, ...]


@dataclasses.dataclass
class Samples:
    """Surface samples in their geom frames (placed per frame by `SceneGT.points_base`)."""

    points_local: np.ndarray  # (P, 3)
    geom: np.ndarray  # (P,) int, geom id
    instance: np.ndarray  # (P,) int, index into SceneGT.instances()
    spacing: float
    cap: int
    seed: int

    def per_instance(self) -> dict[int, int]:
        return {int(k): int(v) for k, v in zip(*np.unique(self.instance, return_counts=True))}


class SceneGT:
    """One ``scene_<stem>.npz`` of ``scene_export.py``."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        with np.load(str(self.path)) as z:
            self.data = {k: z[k] for k in z.files}
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.target_info = json.loads(str(self.data["target_json"]))
        self.provenance = json.loads(str(self.data["provenance_json"]))
        self.frames = np.asarray(self.data["frames"]).astype(np.int64)
        self._mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._instances = self._build_instances()

    # -- tables
    @property
    def n_frames(self) -> int:
        return int(len(self.frames))

    def frame_row(self, obs_index: int) -> int:
        hit = np.flatnonzero(self.frames == int(obs_index))
        if not hit.size:
            raise KeyError(f"{self.path.name}: frame {obs_index} was not exported")
        return int(hit[0])

    def _build_instances(self) -> list[Instance]:
        d = self.data
        group, coll = d["geom_group"], (d["geom_contype"] | d["geom_conaffinity"]) != 0
        out = []
        for name, gids in self.target_info["instances_geom_ids"].items():
            if name == self.target_info["target"]:
                role = "target"
            elif name == self.target_info.get("destination"):
                role = "destination"
            elif name in self.target_info.get("fixtures", ()):
                role = "fixture"
            else:
                role = "object"
            gids = [int(g) for g in gids]
            out.append(Instance(name, role, tuple(g for g in gids if group[g] == VISUAL_GROUP),
                                tuple(g for g in gids if coll[g])))
        mount = [int(g) for g in d["mount_geoms"]]
        out.append(Instance("mount", "mount", tuple(g for g in mount if group[g] == VISUAL_GROUP),
                            tuple(g for g in mount if coll[g])))
        order = {r: i for i, r in enumerate(ROLES)}
        return sorted(out, key=lambda i: (order[i.role], i.name))

    def instances(self) -> list[Instance]:
        return list(self._instances)

    def instance_index(self, name: str) -> int:
        return next(i for i, inst in enumerate(self._instances) if inst.name == name)

    # -- geometry
    def geom_mesh(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(V, F)`` of geom ``g`` in its geom frame."""
        if g not in self._mesh_cache:
            d = self.data
            t = int(d["geom_type"][g])
            if t == MESH:
                mid = int(d["geom_dataid"][g])
                va, vn = int(d["mesh_vertadr"][mid]), int(d["mesh_vertnum"][mid])
                fa, fn = int(d["mesh_faceadr"][mid]), int(d["mesh_facenum"][mid])
                if va < 0:
                    raise ValueError(f"mesh {mid} of geom {g} was not exported")
                self._mesh_cache[g] = (d["mesh_vert"][va:va + vn], d["mesh_face"][fa:fa + fn])
            else:
                self._mesh_cache[g] = primitive_mesh(t, d["geom_size"][g])
        return self._mesh_cache[g]

    def T_base_world(self, row: int) -> np.ndarray:
        return _rigid_inv(self.data["T_world_base"][row])

    def geom_pose_base(self, row: int, g: int) -> tuple[np.ndarray, np.ndarray]:
        Tbw = self.T_base_world(row)
        R = Tbw[:3, :3] @ self.data["geom_xmat"][row, g].reshape(3, 3)
        t = Tbw[:3, :3] @ self.data["geom_xpos"][row, g] + Tbw[:3, 3]
        return R, t

    def triangles_base(self, row: int, g: int) -> np.ndarray:
        v, f = self.geom_mesh(g)
        R, t = self.geom_pose_base(row, g)
        return (v @ R.T + t)[f]

    def samples(self, spacing: float = SAMPLE_SPACING, cap: int = SAMPLE_CAP, seed: int = 0) -> Samples:
        pts, geoms, inst = [], [], []
        for i, instance in enumerate(self._instances):
            rng = np.random.default_rng([seed, i])
            local, gid = [], []
            for g in instance.visual_geoms:
                v, f = self.geom_mesh(g)
                p = sample_triangles(v, f, spacing, rng)
                local.append(p)
                gid.append(np.full(len(p), g, np.int64))
            if not local:
                continue
            local, gid = np.concatenate(local), np.concatenate(gid)
            if cap and len(local) > cap:
                keep = np.sort(rng.choice(len(local), size=cap, replace=False))
                local, gid = local[keep], gid[keep]
            pts.append(local)
            geoms.append(gid)
            inst.append(np.full(len(local), i, np.int64))
        return Samples(np.concatenate(pts), np.concatenate(geoms), np.concatenate(inst), spacing, cap, seed)

    def points_base(self, row: int, samples: Samples) -> np.ndarray:
        Tbw = self.T_base_world(row)
        xpos, xmat = self.data["geom_xpos"][row], self.data["geom_xmat"][row].reshape(-1, 3, 3)
        R = np.einsum("ij,gjk->gik", Tbw[:3, :3], xmat)
        t = xpos @ Tbw[:3, :3].T + Tbw[:3, 3]
        g = samples.geom
        return np.einsum("pij,pj->pi", R[g], samples.points_local) + t[g]

    def exact_distance(self, points: np.ndarray, row: int, *, max_distance: float = 0.5) -> np.ndarray:
        """``(N, I)`` unsigned distance (m) to each instance's visual surface; ``inf`` beyond ``max_distance``."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.full((len(pts), len(self._instances)), np.inf)
        for i, instance in enumerate(self._instances):
            for g in instance.visual_geoms:
                tri = self.triangles_base(row, g)
                lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
                lb = np.linalg.norm(np.maximum(0.0, np.maximum(lo - pts, pts - hi)), axis=1)
                near = lb < max_distance
                if near.any():
                    d = point_triangles_distance(pts[near], tri)
                    out[near, i] = np.minimum(out[near, i], d)
        out[out > max_distance] = np.inf
        return out

    def exact_distance_near(self, points: np.ndarray, row: int, *, max_distance: float) -> np.ndarray:
        """:meth:`exact_distance` for a small ``max_distance`` (e.g. 10 mm labels): the same exact distances where
        they are <= ``max_distance`` (``inf`` beyond), with per-triangle box culling (SL4b evaluator)."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.full((len(pts), len(self._instances)), np.inf)
        for i, instance in enumerate(self._instances):
            for g in instance.visual_geoms:
                tri = self.triangles_base(row, g)
                lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
                lb = np.linalg.norm(np.maximum(0.0, np.maximum(lo - pts, pts - hi)), axis=1)
                near = lb <= max_distance
                if near.any():
                    d = point_triangles_distance_near(pts[near], tri, max_distance)
                    out[near, i] = np.minimum(out[near, i], d)
        return out

    def inside(self, points: np.ndarray, row: int) -> np.ndarray:
        """``(N, I)`` -- point inside one of the instance's collision primitives (analytic)."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.zeros((len(pts), len(self._instances)), bool)
        d = self.data
        for i, instance in enumerate(self._instances):
            for g in instance.collision_geoms:
                R, t = self.geom_pose_base(row, g)
                q = (pts - t) @ R  # geom frame
                s = d["geom_size"][g]
                ty = int(d["geom_type"][g])
                if ty == BOX:
                    hit = np.all(np.abs(q) <= s, axis=1)
                elif ty == SPHERE:
                    hit = np.linalg.norm(q, axis=1) <= s[0]
                elif ty == ELLIPSOID:
                    hit = ((q / s) ** 2).sum(1) <= 1.0
                elif ty == CYLINDER:
                    hit = (np.hypot(q[:, 0], q[:, 1]) <= s[0]) & (np.abs(q[:, 2]) <= s[1])
                elif ty == CAPSULE:
                    z = np.clip(q[:, 2], -s[1], s[1])
                    hit = np.linalg.norm(q - np.c_[np.zeros((len(q), 2)), z], axis=1) <= s[0]
                elif ty == PLANE:
                    hit = q[:, 2] < 0.0
                else:
                    continue  # mesh collision: no analytic containment (none in LIBERO objects)
                out[:, i] |= hit
        return out

    def body_position_base(self, row: int, body: str) -> np.ndarray:
        names = [str(x) for x in self.data["body_names"]]
        Tbw = self.T_base_world(row)
        return Tbw[:3, :3] @ self.data["body_xpos"][row, names.index(body)] + Tbw[:3, 3]

    def target_position_base(self, row: int) -> np.ndarray:
        return self.body_position_base(row, self.target_info["target"] + "_main")

    def instance_table(self) -> list[dict[str, Any]]:
        return [{"name": x.name, "role": x.role, "visual_geoms": list(x.visual_geoms),
                 "collision_geoms": list(x.collision_geoms)} for x in self._instances]


def depth_residual(points_base: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """``(N,)`` recorded depth at each point's pixel minus the point's camera z (m); NaN off-image / no depth.

    Pixel = nearest integer of the pinhole projection (adapter K convention ``pixel_centre``: pixel centres
    at integer indices). ``|residual|`` small (a few mm) = the point is the surface the camera sees there;
    large **negative** = a nearer surface occludes it; large positive = the pixel sees past the point
    (silhouette edge / pixel rounding). The tolerance is the caller's choice.
    """
    pts = np.asarray(points_base, np.float64).reshape(-1, 3)
    Tcb = _rigid_inv(np.asarray(T_base_cam, np.float64))
    pc = pts @ Tcb[:3, :3].T + Tcb[:3, 3]
    out = np.full(len(pts), np.nan)
    z = pc[:, 2]
    ok = z > 1e-6
    u = np.full(len(pts), -1, np.int64)
    v = np.full(len(pts), -1, np.int64)
    u[ok] = np.rint(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(np.int64)
    v[ok] = np.rint(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(np.int64)
    H, W = depth_m.shape
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    dm = np.zeros(len(pts))
    dm[ok] = depth_m[v[ok], u[ok]]
    ok &= dm > 0
    out[ok] = dm[ok] - z[ok]
    return out


__all__ = ["BOX", "CAPSULE", "CYLINDER", "ELLIPSOID", "Instance", "MESH", "PLANE", "ROLES", "SAMPLE_CAP", "SAMPLE_SPACING",
           "SEGMENTS", "SPHERE", "Samples", "SceneGT", "VISUAL_GROUP", "closest_points_on_triangles", "closest_points_pairwise",
           "depth_residual", "point_triangles_distance", "point_triangles_distance_near", "primitive_mesh",
           "sample_triangles"]
