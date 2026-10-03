"""Independent convex distance (GJK, numpy) — the T40 A verifier's reference, copied unchanged from
`outputs/verify/T40/A/p/gjk.py` (penetration-depth part dropped; it needs scipy and the tests only need the
sign on overlap). It does not call `mj_geomDistance`: shapes are support functions built from `geom_xpos` /
`geom_xmat` / `geom_size` (box, cylinder, sphere, capsule, ellipsoid, mesh = convex hull of its vertices).
`gjk_dist` returns the Euclidean distance, 0.0 if the shapes overlap. In T40 A it matched the exact box-box
distance to 0.0015 mm."""
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
