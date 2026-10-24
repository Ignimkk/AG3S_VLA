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

Action range (CLOSED-LOOP-FIX, the port of the original's ``clip_sample``): diffusion_policy's DDIM scheduler clamps
``pred_original_sample`` to the normalized action range ``[-1, 1]`` every step (``scheduling_ddim.py:296-297``,
``clip_sample: True`` in every Lan-o3dp workspace config), so the field is evaluated on a clamped clean trajectory and
the output stays inside the range up to one ``0.03`` push. pi0.5's sampler has no clamp; without one the
``1 / max(1 - t, 0.1)`` push accumulated to ``|a| = 27`` (V3 GPU smoke). Here, with the LIBERO action box
``[-1, 1]`` (raw OSC input range, mapped through the ``pi05_libero`` quantile norm stats) as the range:

* the field is evaluated on ``clip(x0_hat)`` (as on ``pred_original_sample``);
* the push is limited so the implied clean chunk ``x0_hat + push / (1 - t_next)`` stays inside the box -- or, for a
  dimension the network itself already put outside, does not move further out. Zero push -> ``x_next`` unchanged
  (bit-identical). At the last step (``t_next = 0``, ``x0_hat = x_next``) this bounds the served xyz to the box.

Fixed shapes (CLOSED-LOOP-FIX): circles are padded to ``MAX_CIRCLES`` (far centre, radius 0) and padding points of
the obstacle cloud (``geometry.FAR``) are ignored, so one jaxpr serves every call.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.core import GuidanceBase, raw_to_norm_scale
from Baseline.paper_baselines.methods.guidance_common.geometry import FAR
from Baseline.paper_baselines.methods.guidance_common.scene import GuidanceScene
from Baseline.paper_baselines.methods.interface import normalize_actions

MAX_CIRCLES = 16
#: LIBERO OSC_POSE input range (robosuite ``input_max`` / ``input_min``): the raw action box of the clamp
RAW_ACTION_BOX = (-1.0, 1.0)


def action_box_normalized(norm: dict, d: int = 3, box: tuple[float, float] = RAW_ACTION_BOX) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the raw action box in the policy's normalized units (first ``d`` dims)."""
    n = int(np.asarray(norm.get("q01", norm.get("mean"))).shape[-1])
    lo = normalize_actions(np.full(n, float(box[0])), norm, xp=np)[:d]
    hi = normalize_actions(np.full(n, float(box[1])), norm, xp=np)[:d]
    return np.minimum(lo, hi).astype(np.float64), np.maximum(lo, hi).astype(np.float64)


def obstacle_circles(scene: GuidanceScene, *, cell: float = 0.02, min_points: int = 3, radius_override: float | None = None,
                     radius_margin: float = 0.0, max_circles: int = MAX_CIRCLES) -> tuple[np.ndarray, np.ndarray]:
    """``(centres_xy (n, 2), radii (n,))``: Lan-o3dp's one-sphere-per-object obstacle model."""
    from scipy import ndimage

    o = scene.obstacles
    cs, rs = [], []

    def real(c):  # fixed-shape padding (geometry.Obstacles.padded: centre at FAR) is not an object
        return bool(np.all(np.abs(np.asarray(c, np.float64)) < 0.5 * FAR))

    for c, r in zip(o.sphere_c, o.sphere_r):  # half xy-bbox diagonal, as preprocess_obstacle_pcd computes it
        if real(c):
            cs.append(c[:2]), rs.append(np.sqrt(2.0) * r)
    for c, R, h in zip(o.box_c, o.box_R, o.box_h):  # xy bounding circle of the box footprint
        if not real(c):
            continue
        corners = (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h) @ R.T + c
        lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
        cs.append(0.5 * (lo + hi)), rs.append(0.5 * np.linalg.norm(hi - lo))
    for c, r in zip(o.cyl_c, o.cyl_r):  # upright cylinder assumed (SafeLIBERO obstacles stand on the table)
        if real(c):
            cs.append(c[:2]), rs.append(np.sqrt(2.0) * r)
    pts = np.asarray(o.points, np.float64).reshape(-1, 3)
    pts = pts[np.all(np.abs(pts) < 0.5 * FAR, axis=1)]  # padding points, as above
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
                "radius_override": None, "radius_margin": 0.0, "clip_sample": True}

    def _prepare(self, scene: GuidanceScene, obs, norm) -> dict:
        P = self.params
        cs, rs = obstacle_circles(scene, cell=P["grid_cell_m"], min_points=P["min_cluster_points"],
                                  radius_override=P["radius_override"], radius_margin=P["radius_margin"])
        cpad = np.full((MAX_CIRCLES, 2), FAR)  # fixed shape: one jaxpr for every call (padding: never inside)
        rpad = np.zeros(MAX_CIRCLES)
        cpad[:len(cs)], rpad[:len(rs)] = cs, rs
        lo, hi = action_box_normalized(norm)
        return {"centres": cpad, "radii": rpad, "n_circles": int(len(cs)), "skip": len(cs) == 0,
                "scale_xyz": raw_to_norm_scale(norm)[:3], "box_lo": lo, "box_hi": hi,
                "circles": {"centres": np.asarray(cs).round(4).tolist(), "radii": np.asarray(rs).round(4).tolist()}}

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
        if P["clip_sample"]:
            lo = jnp.asarray(ctx["box_lo"], x0.dtype)[None]
            hi = jnp.asarray(ctx["box_hi"], x0.dtype)[None]
            x0f = x0.at[:, :3].set(jnp.clip(x0[:, :3], lo, hi))  # clip_sample: the field sees the clamped x0
        else:
            x0f = x0
        p = path(self.raw_xyz(x0f, ctx["norm"]))
        u = self.field(p, ctx)  # (H, 3)
        dp = P["step_m"] * u
        da_raw = (jnp.asarray(self.pullback(H, path), x0.dtype) @ dp.reshape(-1)).reshape(H, 3)
        da = da_raw * jnp.asarray(ctx["scale_xyz"], x0.dtype)[None]
        coef = jnp.maximum(1.0 - t_next, P["min_x0_coef"])
        push = da / coef
        if P["clip_sample"]:
            # implied clean chunk after the push: x0 + push / k (k = x0 coefficient of x_next; 1 for x0_mode identity)
            k = jnp.maximum(1.0 - t_next, 1e-3) if ctx.get("x0_mode") == "noise" else jnp.ones((), x0.dtype)
            cur = x0[:, :3]
            new = jnp.clip(cur + push / k, jnp.minimum(lo, cur), jnp.maximum(hi, cur))
            push = (new - cur) * k
        return x_next.at[:, :3].add(push)


__all__ = ["LanO3dpGuidanceAdapted", "MAX_CIRCLES", "RAW_ACTION_BOX", "action_box_normalized", "obstacle_circles"]
