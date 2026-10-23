"""Obstacle geometry for the guidance methods: fixed-shape arrays + pure-JAX signed distance / closest point.

Every set is padded to a fixed size with "far" dummies (centre ``FAR``), so a ``guidance_fn`` built from one scene
traces once and its distances are never influenced by padding. All coordinates are in the robot **base** frame.

Primitives (what the original methods use):

* points    ``(P, 3)`` + ``point_radius``  -- depth point clouds (OmniGuide ESDF, NUDGE cuRobo ESDF, Lan-o3dp pcd);
* spheres   ``(S, 3)`` centres + ``(S,)`` radii;
* boxes     ``(B, 3)`` centres + ``(B, 3, 3)`` rotations (local->base) + ``(B, 3)`` half extents;
* cylinders ``(Y, 3)`` centres + ``(Y, 3, 3)`` rotations (axis = local z) + ``(Y,)`` radius + ``(Y,)`` half height.

SafeLIBERO obstacles are MuJoCo sphere / box / cylinder geoms -- the same primitives constrained flow matching
(``ObstacleGeometry``) and AEGIS read from the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAR = 1.0e3  # padding centre (m): distance to padding is ~1e3, never the minimum
_EPS = 1e-12


def _pad(a: np.ndarray, n: int, fill: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    if len(a) >= n:
        return a[:n]
    return np.concatenate([a, np.broadcast_to(fill, (n - len(a),) + fill.shape)], axis=0)


@dataclass
class Obstacles:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    point_radius: float = 0.0
    sphere_c: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sphere_r: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    box_c: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    box_R: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    box_h: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cyl_c: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cyl_R: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    cyl_r: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    cyl_h: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    names: list = field(default_factory=list)

    def counts(self) -> dict:
        return {"points": int(len(self.points)), "spheres": int(len(self.sphere_c)), "boxes": int(len(self.box_c)),
                "cylinders": int(len(self.cyl_c))}

    @property
    def empty(self) -> bool:
        return sum(self.counts().values()) == 0

    def padded(self, n_points: int | None = None, n_prim: int = 4) -> "Obstacles":
        """Fixed shapes (>= 1 of each kind) so that jit sees one signature per padding size."""
        P = max(1, n_points if n_points is not None else len(self.points))
        far3, eye = np.full(3, FAR), np.eye(3)
        S, B, Y = (max(1, n_prim, len(x)) for x in (self.sphere_c, self.box_c, self.cyl_c))
        return Obstacles(points=_pad(self.points.reshape(-1, 3), P, far3), point_radius=float(self.point_radius),
                         sphere_c=_pad(self.sphere_c.reshape(-1, 3), S, far3), sphere_r=_pad(self.sphere_r, S, np.zeros(())),
                         box_c=_pad(self.box_c.reshape(-1, 3), B, far3), box_R=_pad(self.box_R.reshape(-1, 3, 3), B, eye),
                         box_h=_pad(self.box_h.reshape(-1, 3), B, np.full(3, 1e-3)),
                         cyl_c=_pad(self.cyl_c.reshape(-1, 3), Y, far3), cyl_R=_pad(self.cyl_R.reshape(-1, 3, 3), Y, eye),
                         cyl_r=_pad(self.cyl_r, Y, np.full((), 1e-3)), cyl_h=_pad(self.cyl_h, Y, np.full((), 1e-3)),
                         names=list(self.names))

    def as_jax(self, dtype=None) -> dict:
        import jax.numpy as jnp

        dt = dtype or jnp.float32
        keys = ("points", "sphere_c", "sphere_r", "box_c", "box_R", "box_h", "cyl_c", "cyl_R", "cyl_r", "cyl_h")
        out = {k: jnp.asarray(getattr(self, k), dt) for k in keys}
        out["point_radius"] = jnp.asarray(self.point_radius, dt)
        return out

    @staticmethod
    def from_primitives(prims: list[dict], points: np.ndarray | None = None, point_radius: float = 0.0) -> "Obstacles":
        """``prims``: ``{"type": sphere|box|cylinder, "pos": (3,), "rot": (3,3) optional, "size": ...}`` (MuJoCo sizes:
        sphere ``[r]``, box half extents ``[hx, hy, hz]``, cylinder ``[r, half_height]``); base frame."""
        o = Obstacles(points=np.zeros((0, 3)) if points is None else np.asarray(points, np.float64).reshape(-1, 3),
                      point_radius=point_radius)
        sc, sr, bc, bR, bh, yc, yR, yr, yh, names = [], [], [], [], [], [], [], [], [], []
        for p in prims:
            t = str(p["type"]).lower()
            pos = np.asarray(p["pos"], np.float64).reshape(3)
            R = np.asarray(p.get("rot", np.eye(3)), np.float64).reshape(3, 3)
            size = np.atleast_1d(np.asarray(p["size"], np.float64))
            names.append(p.get("name", t))
            if t == "sphere":
                sc.append(pos), sr.append(size[0])
            elif t == "box":
                bc.append(pos), bR.append(R), bh.append(size[:3])
            elif t == "cylinder":
                yc.append(pos), yR.append(R), yr.append(size[0]), yh.append(size[1])
            else:  # mesh / unknown -> bounding sphere if a radius is given
                if "radius" in p:
                    sc.append(pos), sr.append(float(p["radius"]))
                else:
                    raise ValueError(f"unsupported primitive type {t!r} (sphere / box / cylinder)")
        o.sphere_c, o.sphere_r = np.asarray(sc).reshape(-1, 3), np.asarray(sr).reshape(-1)
        o.box_c, o.box_R, o.box_h = np.asarray(bc).reshape(-1, 3), np.asarray(bR).reshape(-1, 3, 3), np.asarray(bh).reshape(-1, 3)
        o.cyl_c, o.cyl_R = np.asarray(yc).reshape(-1, 3), np.asarray(yR).reshape(-1, 3, 3)
        o.cyl_r, o.cyl_h = np.asarray(yr).reshape(-1), np.asarray(yh).reshape(-1)
        o.names = names
        return o


# --- pure JAX queries (x: (3,) one query point; vmap for more) ---------------------------------------------------
def _norm(v, axis=-1):
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(v * v, axis=axis) + _EPS)


def _box_sd_cp(x, c, R, h):
    import jax.numpy as jnp

    l = (x - c) @ R  # local coords (R: local->base, so R^T (x-c))
    q = jnp.abs(l) - h
    outside = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2) + _EPS)
    inside = jnp.minimum(jnp.max(q), 0.0)
    is_out = jnp.max(q) > 0
    sd = jnp.where(is_out, outside, inside)
    cl_out = jnp.clip(l, -h, h)
    ax = jnp.argmax(q)
    cl_in = l.at[ax].set(jnp.sign(l[ax] + 1e-12) * h[ax])
    cl = jnp.where(is_out, cl_out, cl_in)
    return sd, cl @ R.T + c


def _cyl_sd_cp(x, c, R, r, hz):
    import jax.numpy as jnp

    l = (x - c) @ R
    rho = _norm(l[:2])
    dr, dz = rho - r, jnp.abs(l[2]) - hz
    outside = jnp.sqrt(jnp.maximum(dr, 0.0) ** 2 + jnp.maximum(dz, 0.0) ** 2 + _EPS)
    is_out = jnp.maximum(dr, dz) > 0
    sd = jnp.where(is_out, outside, jnp.maximum(dr, dz))
    radial = l[:2] / rho
    xy_out = jnp.where(rho > r, radial * r, l[:2])
    z_out = jnp.clip(l[2], -hz, hz)
    side = dr > dz  # inside: nearer to the side wall than to a cap
    xy_in = jnp.where(side, radial * r, l[:2])
    z_in = jnp.where(side, l[2], jnp.sign(l[2] + 1e-12) * hz)
    cl = jnp.where(is_out, jnp.concatenate([xy_out, z_out[None]]), jnp.concatenate([xy_in, z_in[None]]))
    return sd, cl @ R.T + c


def all_distances_and_closest(x, obs: dict):
    """Per-element ``(sd (E,), closest_surface_point (E, 3))`` of one query ``x (3,)`` (E = every padded element)."""
    import jax
    import jax.numpy as jnp

    dp = _norm(obs["points"] - x[None]) - obs["point_radius"]
    vp = obs["points"] - x[None]
    cpp = obs["points"] - obs["point_radius"] * vp / _norm(vp)[:, None]
    vs = x[None] - obs["sphere_c"]
    ns = _norm(vs)
    ds = ns - obs["sphere_r"]
    cps = obs["sphere_c"] + obs["sphere_r"][:, None] * vs / ns[:, None]
    db, cpb = jax.vmap(_box_sd_cp, in_axes=(None, 0, 0, 0))(x, obs["box_c"], obs["box_R"], obs["box_h"])
    dy, cpy = jax.vmap(_cyl_sd_cp, in_axes=(None, 0, 0, 0, 0))(x, obs["cyl_c"], obs["cyl_R"], obs["cyl_r"], obs["cyl_h"])
    return jnp.concatenate([dp, ds, db, dy]), jnp.concatenate([cpp, cps, cpb, cpy], axis=0)


def signed_distance_and_closest(x, obs: dict):
    """``(sd, closest_surface_point (3,))`` of one query ``x (3,)`` against every obstacle (min over all)."""
    import jax.numpy as jnp

    d, cp = all_distances_and_closest(x, obs)
    i = jnp.argmin(d)
    return d[i], cp[i]


def signed_distance(x, obs: dict):
    """Signed distance of points ``x (..., 3)`` (vmapped over leading dims)."""
    import jax

    flat = x.reshape(-1, 3)
    d = jax.vmap(lambda p: signed_distance_and_closest(p, obs)[0])(flat)
    return d.reshape(x.shape[:-1])


def signed_distance_np(x: np.ndarray, obs: Obstacles) -> np.ndarray:
    """NumPy reference (tests / reporting): runs the JAX function on CPU arrays."""
    import jax.numpy as jnp

    o = obs.padded().as_jax(jnp.float32)
    return np.asarray(signed_distance(jnp.asarray(np.asarray(x, np.float32)), o), np.float64)


__all__ = ["FAR", "Obstacles", "all_distances_and_closest", "signed_distance", "signed_distance_and_closest", "signed_distance_np"]
