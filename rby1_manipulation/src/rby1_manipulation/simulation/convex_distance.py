"""Convex geom distance that does not trust ``mj_geomDistance`` blindly (T40 H).

Why this exists
---------------
MuJoCo 3.11 ``mj_geomDistance`` (nativeccd GJK, ``engine_support.c:mj_geomDistanceCCD``) can
return **exactly 0.0 for two geoms that are 7-24 cm apart**.  T40 G hit it in 5 closed-loop
runs: hurdle foot (a flat cylinder r 35 mm x 12 mm, axis vertical) vs an arm mesh, GJK
separation 0.070-0.236 m, ``mj_geomDistance`` 0.0 in either argument order whenever
``distmax`` exceeds the true gap (below it, the cut-off exit fires first and the capped value
is right).  The witness points it returns are not 0 apart (|from - to| = 0.07-0.47 m), so the
0.0 is not a contact: GJK concluded "origin enclosed" on a degenerate simplex.  T40 A's
divider (box) "reset object clearance 0.0, GJK 117 mm" is the same failure.  It is rare
(a measure-small set of poses: 4-10 of 400 perturbations of the stop pose) and exact-pose
dependent, which is why a kinematic probe of the recorded control row missed ep1967.

What this module does
---------------------
* :func:`support` -- exact support point of a box / sphere / capsule / cylinder / ellipsoid /
  mesh (convex hull of its vertices) from ``geom_xpos`` / ``geom_xmat`` / ``geom_size``.
* :func:`certify` -- a cheap check of one ``mj_geomDistance`` answer: for a positive value the
  witness points must be ``value`` apart and lie on the two supporting planes of the direction
  ``to - from``.  The plane gap is a lower bound on the true distance, so a value that passes
  cannot be an under-estimate (the failure above), and the witness points pin it from above.
* :func:`gjk_distance` -- an exact GJK (Ericson-style closest point on simplex with explicit
  Voronoi-region tests; a coplanar tetrahedron is never taken as enclosing the origin -- that
  is the case MuJoCo gets wrong) used when :func:`certify` refuses.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

#: witness / supporting-plane tolerance of :func:`certify` (m).  MuJoCo's GJK tolerance is
#: 1e-6; the T40 H requirement is 0.1 mm.
CERTIFY_TOL = 1e-5

_G = mujoco.mjtGeom
SUPPORTED_TYPES = frozenset(int(t) for t in (
    _G.mjGEOM_BOX, _G.mjGEOM_SPHERE, _G.mjGEOM_CAPSULE, _G.mjGEOM_CYLINDER,
    _G.mjGEOM_ELLIPSOID, _G.mjGEOM_MESH,
))


class GeomSupport:
    """Support points of model geoms at the current ``data`` pose (mesh vertices cached)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._mesh_vertices: dict[int, np.ndarray] = {}

    def supported(self, geom: int) -> bool:
        return int(self.model.geom_type[geom]) in SUPPORTED_TYPES

    def _vertices(self, mesh: int) -> np.ndarray:
        verts = self._mesh_vertices.get(mesh)
        if verts is None:
            start = int(self.model.mesh_vertadr[mesh])
            count = int(self.model.mesh_vertnum[mesh])
            verts = np.ascontiguousarray(self.model.mesh_vert[start:start + count], dtype=np.float64)
            self._mesh_vertices[mesh] = verts
        return verts

    def __call__(self, geom: int, direction: np.ndarray) -> np.ndarray:
        """argmax_{p in geom} direction . p (world frame)."""
        m, d = self.model, self.data
        centre = d.geom_xpos[geom]
        rot = d.geom_xmat[geom].reshape(3, 3)
        size = m.geom_size[geom]
        kind = int(m.geom_type[geom])
        local = rot.T @ direction
        if kind == _G.mjGEOM_MESH:
            verts = self._vertices(int(m.geom_dataid[geom]))
            return centre + rot @ verts[int(np.argmax(verts @ local))]
        if kind == _G.mjGEOM_BOX:
            return centre + rot @ np.where(local >= 0.0, size, -size)
        norm = math.sqrt(float(direction @ direction))
        if kind == _G.mjGEOM_SPHERE:
            return centre + (size[0] / norm) * direction if norm > 0.0 else centre.copy()
        if kind == _G.mjGEOM_CAPSULE:
            tip = rot[:, 2] * (size[1] if local[2] >= 0.0 else -size[1])
            return centre + tip + ((size[0] / norm) * direction if norm > 0.0 else 0.0)
        if kind == _G.mjGEOM_CYLINDER:
            radial = math.hypot(float(local[0]), float(local[1]))
            point = np.array([0.0, 0.0, size[1] if local[2] >= 0.0 else -size[1]])
            if radial > 0.0:
                point[0] = size[0] * local[0] / radial
                point[1] = size[0] * local[1] / radial
            return centre + rot @ point
        if kind == _G.mjGEOM_ELLIPSOID:
            scaled = size * size * local
            denom = math.sqrt(float(local @ scaled))
            return centre + (rot @ (scaled / denom) if denom > 0.0 else 0.0)
        raise NotImplementedError(f"no support function for geom type {kind}")


def certify(support: GeomSupport, geom1: int, geom2: int, value: float,
            from_to: np.ndarray, tol: float = CERTIFY_TOL) -> bool:
    """True if a positive ``mj_geomDistance`` answer is consistent with exact supports.

    ``from_to[:3]`` is on geom1, ``from_to[3:]`` on geom2 (mj_geomDistance convention).
    With n = (to - from) / |to - from|: |to - from| == value, n.from == max_{geom1} n.p and
    n.to == min_{geom2} n.p (both within ``tol``).  Then the separating-plane gap
    min_{geom2} n.p - max_{geom1} n.p (a lower bound on the distance) equals ``value``.
    """
    if not value > 0.0:
        return False
    start, end = from_to[:3], from_to[3:]
    witness = end - start
    gap = math.sqrt(float(witness @ witness))
    if gap <= 0.0 or abs(gap - value) > tol:
        return False
    normal = witness / gap
    high1 = float(normal @ support(geom1, normal))
    low2 = float(normal @ support(geom2, -normal))
    return (abs(float(normal @ start) - high1) <= tol
            and abs(float(normal @ end) - low2) <= tol
            and low2 - high1 >= value - tol)


# ----------------------------------------------------------------------------- exact GJK


def _closest_segment(a: np.ndarray, b: np.ndarray):
    """Closest point to the origin on segment ab -> (point, kept vertex indices)."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom <= 0.0 else -float(a @ ab) / denom
    if t <= 0.0:
        return a, (0,)
    if t >= 1.0:
        return b, (1,)
    return a + t * ab, (0, 1)


def _closest_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Ericson, Real-Time Collision Detection 5.1.5, with p = origin."""
    ab, ac = b - a, c - a
    ap = -a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (0,)
    bp = -b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, (1,)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, (0, 1)
    cp = -c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, (2,)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, (0, 2)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), (1, 2)
    denom = va + vb + vc
    if not denom > 0.0:  # degenerate (collinear) triangle: best edge
        options = [(_closest_segment(a, b), (0, 1)), (_closest_segment(a, c), (0, 2)),
                   (_closest_segment(b, c), (1, 2))]
        (point, keep), pair = min(options, key=lambda o: float(o[0][0] @ o[0][0]))
        return point, tuple(pair[i] for i in keep)
    v, w = vb / denom, vc / denom
    return a + ab * v + ac * w, (0, 1, 2)


_FACES = ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0))


def _closest_tetrahedron(pts: list[np.ndarray]):
    """Closest point to the origin in a tetrahedron; ``None`` point if it encloses the origin.

    The origin counts as enclosed only for a tetrahedron with a clear volume, with the origin on
    the inner side of every face; a flat (coplanar) one is treated as its faces.
    """
    scale = max(float(np.max(np.abs(p))) for p in pts) or 1.0
    vol = float(np.dot(pts[1] - pts[0], np.cross(pts[2] - pts[0], pts[3] - pts[0])))
    flat = abs(vol) <= 1e-12 * scale ** 3
    inside = not flat
    best = None
    for i, j, k, opposite in _FACES:
        a, b, c = pts[i], pts[j], pts[k]
        normal = np.cross(b - a, c - a)
        side_origin = float(normal @ (-a))
        side_other = float(normal @ (pts[opposite] - a))
        if not flat and side_origin * side_other > 0.0:
            continue  # origin strictly on the inner side of this face
        inside = False  # outside (or on the plane of) this face: its closest point decides
        point, keep = _closest_triangle(a, b, c)
        norm2 = float(point @ point)
        if best is None or norm2 < best[0]:
            best = (norm2, point, tuple((i, j, k)[n] for n in keep))
    if inside:
        return None, (0, 1, 2, 3)
    return best[1], best[2]


def _closest(simplex: list[np.ndarray]):
    if len(simplex) == 1:
        return simplex[0], (0,)
    if len(simplex) == 2:
        return _closest_segment(*simplex)
    if len(simplex) == 3:
        return _closest_triangle(*simplex)
    return _closest_tetrahedron(simplex)


def gjk_distance(support: GeomSupport, geom1: int, geom2: int,
                 max_iterations: int = 200) -> float:
    """Euclidean distance between two convex geoms; 0.0 when they overlap or touch."""
    centre_gap = support.data.geom_xpos[geom2] - support.data.geom_xpos[geom1]
    direction = centre_gap if float(centre_gap @ centre_gap) > 0.0 else np.array([1.0, 0.0, 0.0])
    point = support(geom1, -direction) - support(geom2, direction)
    simplex: list[np.ndarray] = []
    for _ in range(max_iterations):
        norm2 = float(point @ point)
        if norm2 <= 1e-24:
            return 0.0
        vertex = support(geom1, -point) - support(geom2, point)
        # Frank-Wolfe gap: |x|^2 - x.w >= |x|^2 - |x| * dist; stop when it is negligible
        if norm2 - float(point @ vertex) <= 1e-14 + 1e-12 * norm2:
            return math.sqrt(norm2)
        if any(np.array_equal(vertex, s) for s in simplex):
            return math.sqrt(norm2)  # no progress possible
        simplex.append(vertex)
        closest, keep = _closest(simplex)
        if closest is None:
            return 0.0
        simplex = [simplex[i] for i in keep]
        point = closest
    return math.sqrt(float(point @ point))
