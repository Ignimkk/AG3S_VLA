"""Lan-o3dp [15] obstacle-cost guidance, **adapted** to pi0.5 flow matching + LIBERO EE-delta actions.

Not a result of the original method: Lan-o3dp is a per-task DP3-style diffusion policy on RLBench with absolute EE
position actions and no released checkpoint. Only its training-free obstacle guidance is ported, from
``qqianfeng/lan-o3dp@b89c2898`` (MIT):

* obstacles (``real_world/real_data_conversion.py:184-215`` ``preprocess_obstacle_pcd``): one sphere per obstacle
  object point cloud -- centre = mean xy, radius = half the xy bounding-box diagonal (``real_env.py:633-640``
  overrides it with 0.15 m on the real robot);
* gradient field (``policy/diffusion_unet_pcd_policy.py:395-417`` ``gradient_field``): for every predicted waypoint
  **inside** an obstacle circle (xy only), the unit vector pointing away from the centre (the gradient of the xy
  distance), z untouched, summed over obstacles;
* use (``diffusion_unet_pcd_policy.py:155-166`` + ``scheduling_ddim.py:313-316``): the field is evaluated on the
  DDIM ``pred_original_sample`` (unnormalized), normalized, and added to the next sample's position dims as
  ``prev_sample[..., :2] += ngrad[..., :2] * 0.03 / sqrt(alpha_bar_prev)``.

Adaptation (this file):

* DDIM -> flow matching: ``x0_hat`` from :mod:`guidance_common.core`; ``sqrt(alpha_bar_prev)`` (the x0 coefficient
  of the next sample) -> ``1 - t_next`` (clamped at ``min_x0_coef``);
* absolute positions -> EE-delta actions: the waypoint push ``step_m * u_k`` is pulled back to an action change
  with the damped pseudo-inverse of the (constant) action -> path Jacobian of the OSC response model, then scaled
  into normalized units. ``step_m`` replaces the 0.03 normalized constant (default 0.01 m, re-chosen because a
  LIBERO chunk moves <= ~10 cm; 0.03 normalized is ~3 cm on an RLBench workspace);
* obstacle circles come from the scene's obstacle cloud (connected components on a 2 cm xy grid) or from
  privileged primitives (bounding circle).
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.core import GuidanceBase, raw_to_norm_scale
from Baseline.paper_baselines.methods.guidance_common.scene import GuidanceScene

MAX_CIRCLES = 16


def obstacle_circles(scene: GuidanceScene, *, cell: float = 0.02, min_points: int = 3, radius_override: float | None = None,
                     radius_margin: float = 0.0, max_circles: int = MAX_CIRCLES) -> tuple[np.ndarray, np.ndarray]:
    """``(centres_xy (n, 2), radii (n,))``: Lan-o3dp's one-sphere-per-object obstacle model."""
    from scipy import ndimage

    o = scene.obstacles
    cs, rs = [], []
    for c, r in zip(o.sphere_c, o.sphere_r):  # half xy-bbox diagonal, as preprocess_obstacle_pcd computes it
        cs.append(c[:2]), rs.append(np.sqrt(2.0) * r)
    for c, R, h in zip(o.box_c, o.box_R, o.box_h):  # xy bounding circle of the box footprint
        corners = (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h) @ R.T + c
        lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
        cs.append(0.5 * (lo + hi)), rs.append(0.5 * np.linalg.norm(hi - lo))
    for c, r in zip(o.cyl_c, o.cyl_r):  # upright cylinder assumed (SafeLIBERO obstacles stand on the table)
        cs.append(c[:2]), rs.append(np.sqrt(2.0) * r)
    pts = o.points
    if len(pts):
        ij = np.floor(pts[:, :2] / cell).astype(np.int64)
        lo = ij.min(0)
        grid = np.zeros(tuple(ij.max(0) - lo + 1), bool)
        grid[tuple((ij - lo).T)] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        pl = lab[tuple((ij - lo).T)]
        for k in range(1, n + 1):
            p = pts[pl == k]
            if len(p) < min_points:
                continue
            mn, mx = p[:, :2].min(0), p[:, :2].max(0)
            cs.append(p[:, :2].mean(0)), rs.append(0.5 * np.linalg.norm(mx - mn))
    cs = np.asarray(cs, np.float64).reshape(-1, 2)
    rs = np.asarray(rs, np.float64).reshape(-1)
    if radius_override is not None:
        rs = np.full_like(rs, float(radius_override))
    rs = rs + radius_margin
    if len(cs) > max_circles:  # keep the nearest-to-EE circles
        keep = np.argsort(np.linalg.norm(cs - scene.ee_pos[None, :2], axis=1) - rs)[:max_circles]
        cs, rs = cs[keep], rs[keep]
    return cs, rs


class LanO3dpGuidanceAdapted(GuidanceBase):
    name = "lan_o3dp_guidance_adapted"
    paper = {"cite": "[15]", "title": "Lan-o3dp (ICRA 2025)", "arxiv": "2407.00451v3",
             "note": "ADAPTED: only the obstacle-cost guidance is ported onto pi0.5; not a reproduction of Lan-o3dp"}
    source = {"repo": "https://github.com/qqianfeng/lan-o3dp", "sha": "b89c2898d8548f7dfde63c44eeb733a59f0c5222",
              "license": "MIT",
              "files": ["diffusion_policy/policy/diffusion_unet_pcd_policy.py:155-166,395-417",
                        "scheduling_ddim.py:313-316", "diffusion_policy/real_world/real_data_conversion.py:184-215"]}

    @classmethod
    def default_params(cls) -> dict:
        return {"step_m": 0.01, "min_x0_coef": 0.1, "damping": 1e-2, "grid_cell_m": 0.02, "min_cluster_points": 3,
                "radius_override": None, "radius_margin": 0.0}

    def _prepare(self, scene: GuidanceScene, obs, norm) -> dict:
        P = self.params
        cs, rs = obstacle_circles(scene, cell=P["grid_cell_m"], min_points=P["min_cluster_points"],
                                  radius_override=P["radius_override"], radius_margin=P["radius_margin"])
        n = max(1, len(cs))
        cpad = np.full((n, 2), 1e3)
        rpad = np.zeros(n)
        cpad[:len(cs)], rpad[:len(rs)] = cs, rs
        return {"centres": cpad, "radii": rpad, "n_circles": int(len(cs)), "skip": len(cs) == 0,
                "scale_xyz": raw_to_norm_scale(norm)[:3]}

    def pullback(self, H: int, path) -> np.ndarray:
        """``P (3H, 3H)``: damped pseudo-inverse of the action -> path Jacobian (raw action units per metre)."""
        J = path.jacobian_base()
        mu = self.params["damping"] * np.trace(J @ J.T) / J.shape[0]
        return J.T @ np.linalg.inv(J @ J.T + mu * np.eye(J.shape[0]))

    def field(self, p, ctx):
        """Lan-o3dp ``gradient_field``: unit push away from each circle centre for waypoints inside it (xy)."""
        import jax.numpy as jnp

        c = jnp.asarray(ctx["centres"], p.dtype)
        r = jnp.asarray(ctx["radii"], p.dtype)
        d = p[:, None, :2] - c[None]  # (H, n, 2)
        dist = jnp.sqrt(jnp.sum(d * d, -1) + 1e-12)
        inside = dist <= r[None]
        u = jnp.where(inside[..., None], d / dist[..., None], 0.0).sum(1)  # (H, 2)
        return jnp.concatenate([u, jnp.zeros_like(u[:, :1])], axis=-1)

    def _guide_one(self, x_next, x0, t_next, ctx, path):
        import jax.numpy as jnp

        P = self.params
        H = x0.shape[0]
        p = path(self.raw_xyz(x0, ctx["norm"]))
        u = self.field(p, ctx)  # (H, 3)
        dp = P["step_m"] * u
        da_raw = (jnp.asarray(self.pullback(H, path), x0.dtype) @ dp.reshape(-1)).reshape(H, 3)
        da = da_raw * jnp.asarray(ctx["scale_xyz"], x0.dtype)[None]
        coef = jnp.maximum(1.0 - t_next, P["min_x0_coef"])
        return x_next.at[:, :3].add(da / coef)


__all__ = ["LanO3dpGuidanceAdapted", "obstacle_circles"]
