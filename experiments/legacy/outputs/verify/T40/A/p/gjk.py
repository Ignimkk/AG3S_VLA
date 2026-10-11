"""Independent convex distance (GJK, numpy) for T40 A — mj_geomDistance (MuJoCo 3.11.0, native CCD) returned wrong
values for box geoms at distmax 0.3 (survey_dist.py: box-box |mj - exact| up to 211 mm in 87 / 5297 pairs, box-mesh
cap-dependent in 19 / 18900). Shapes by support function in world frame: box, cylinder (local z axis), sphere, capsule,
ellipsoid, mesh (convex hull = mesh vertices). gjk_dist returns the Euclidean distance (0 if the shapes overlap)."""
import itertools
import numpy as np
import mujoco

_SUBSETS = {n: [s for k in range(1, n + 1) for s in itertools.combinations(range(n), k)] for n in range(1, 5)}


def make_support(m, d, g):
    t = int(m.geom_type[g]); c = d.geom_xpos[g].copy(); R = d.geom_xmat[g].reshape(3, 3).copy(); s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return lambda v: c + R @ (s * np.sign(R.T @ v + 1e-300))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        def sup(v):
            l = R.T @ v; n = np.hypot(l[0], l[1])
            xy = (l[:2] / n * s[0]) if n > 1e-15 else np.zeros(2)
            return c + R @ np.array([xy[0], xy[1], s[1] * (1 if l[2] >= 0 else -1)])
        return sup
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return lambda v: c + s[0] * v / (np.linalg.norm(v) + 1e-300)
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        def sup(v):
            l = R.T @ v
            return c + R @ np.array([0, 0, s[1] * (1 if l[2] >= 0 else -1)]) + s[0] * v / (np.linalg.norm(v) + 1e-300)
        return sup
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        def sup(v):
            l = R.T @ v; w = s * s * l; return c + R @ (w / (np.sqrt(np.sum(s * s * l * l)) + 1e-300))
        return sup
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g]); a = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
        V = (m.mesh_vert[a:a + n].astype(np.float64) @ R.T) + c
        return lambda v: V[int(np.argmax(V @ v))]
    raise NotImplementedError(f"geom type {t}")


def _closest(W):
    P = np.array(W); best = None
    for S in _SUBSETS[len(W)]:
        Q = P[list(S)]
        if len(S) == 1:
            lam = np.array([1.0])
        else:
            D = Q[1:] - Q[0]; G = D @ D.T
            if abs(np.linalg.det(G)) < 1e-24:
                continue
            mu = np.linalg.solve(G, -D @ Q[0]); lam = np.r_[1 - mu.sum(), mu]
            if (lam < -1e-12).any():
                continue
        x = lam @ Q; nx = x @ x
        if best is None or nx < best[0] - 1e-18:
            best = (nx, x, [W[i] for i in S])
    return best[1], best[2]


def gjk_dist(supA, supB, v0, tol=1e-9, maxit=100):
    v = supA(v0) - supB(-v0); W = []
    for _ in range(maxit):
        if v @ v < 1e-20:
            return 0.0
        w = supA(-v) - supB(v)
        if v @ v - v @ w <= tol * tol + 1e-12 * (v @ v):
            return float(np.sqrt(v @ v))
        W.append(w)
        v, W = _closest(W)
        if len(W) == 4:
            return 0.0
    return float(np.sqrt(v @ v))


def _fib(n):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_DIRS = _fib(1500)


def pen_depth(supA, supB):
    """penetration depth of overlapping convex A, B = min_n [h_A(n) + h_B(-n)] (distance from the origin to the boundary
    of A - B), by 1500 Fibonacci directions + Nelder-Mead refinement on the sphere."""
    from scipy.optimize import minimize
    f = lambda n: float(n @ supA(n) - n @ supB(-n))
    vals = np.array([f(n) for n in _DIRS]); best = None
    for k in np.argsort(vals)[:3]:
        n0 = _DIRS[k]; th0, ph0 = np.arctan2(n0[1], n0[0]), np.arccos(np.clip(n0[2], -1, 1))
        g = lambda x: f(np.array([np.cos(x[0]) * np.sin(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])]))
        r = minimize(g, [th0, ph0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8})
        v = min(r.fun, vals[k])
        best = v if best is None else min(best, v)
    return float(best)


def signed_dist(m, d, g1, g2):
    sa, sb = make_support(m, d, g1), make_support(m, d, g2)
    dist = gjk_dist(sa, sb, d.geom_xpos[g2] - d.geom_xpos[g1] + 1e-9)
    if dist > 0:
        return dist
    return -pen_depth(sa, sb)
