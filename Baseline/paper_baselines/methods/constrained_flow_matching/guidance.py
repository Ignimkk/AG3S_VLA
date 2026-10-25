"""Constrained flow matching [5] (English, Zheng, Ewetz, arXiv:2607.01378) as an openpi ``guidance_fn`` (pure JAX).

**Re-implemented from the paper and a reading of the authors' repo** -- the repo
(``Dubascudes/VLA-Safety-ConstrainedFlowMatching@b91153ad``) has *no license* ("has not yet been selected"), so no
code is copied here; only the update rule and the default parameters are restated. Places in the repo that define them
(``software_module/patches/openpi.patch`` applied to ``src/openpi/models_pytorch/pi0_pytorch.py``;
``release-fixes.patch``):

* ``SafetyConfig``: safety_margin 0.01, min_guidance_time 0.0, dt 0.01, eef_ellipsoid_radii (0.06, 0.12, 0.11),
  cbf_gamma 0.9, guidance_schedule "flat", guidance_scale 1.0 (release-fixes; ``serve_policy`` defaults
  ``--safety-margin 0.01 --guidance-scale 1.0``);
* predicted trajectory: ``p_0 = eef_pos``, ``p_j = p_{j-1} + a_j[:3] * dt`` on the post-Euler state ``x_t`` itself
  (normalized model space);
* clearance: obstacle primitive's closest surface point to ``p_j`` -> distance to the axis-aligned EE ellipsoid
  surface (negative inside); ``h_j = clearance_j - margin``;
* discrete CBF constraints ``h_j - (1 - gamma) h_{j-1} >= 0`` (j = 1..H); if none is violated, no correction;
* minimum-norm correction ``min 1/2 ||delta||^2  s.t. constraints(x + delta) >= 0``, ``delta`` on the 3 position dims,
  solved with SciPy SLSQP (maxiter 50) and a greedy fallback; ``x_t <- x_t + guidance_scale * time_scale * delta``
  after every Euler step (post-Euler, the same place as the openpi hook).

Here the SLSQP solve is replaced by a jit-able SQP: ``n_outer`` linearisations of the constraints (``jax.jacfwd``),
each solved exactly for the minimum-norm point with Hildreth's dual coordinate ascent (``n_sweeps`` sweeps). The greedy
fallback is not ported (the last SQP iterate is returned).

Action box (``action_box=True``, BASELINE-BENCH-READY; **not in the original**): the original QP has no bound on
``delta`` (SLSQP without ``bounds``, no clip after the update; the LIBERO OSC clips the executed action to [-1, 1] per
dimension). Its trajectory model integrates the normalized action with ``dt = 0.01`` (one normalized unit = 1 cm of
predicted path), so an EE ellipsoid (6 / 12 / 11 cm) that already overlaps an obstacle (``h_0 < 0``) asks for a
recovery of ``~|h_0| / dt`` normalized units in the first rows, and the post-Euler corrections of the ten steps add up
(V5 pilot: served |xyz| up to 7.28). With the box, the corrected chunk ``x_t + delta`` is clipped into the LIBERO
OSC input range [-1, 1] (normalized with the policy's stats, ``lan_o3dp_guidance_adapted.action_box_normalized``);
during the denoising steps a dimension already outside is not pushed further out (the Lan-o3dp ``clip_sample`` rule
used by ``lan_o3dp_guidance_adapted``); at the last step (the served chunk) every corrected entry is clipped into the
box, so pushes of earlier steps that the network carried along cannot leave it either. No violation -> ``delta = 0``
-> bit-identical to the unguided state, as before.
``action_box=False`` is the V5 behaviour.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.core import GuidanceBase
from Baseline.paper_baselines.methods.guidance_common.geometry import all_distances_and_closest
from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel
from Baseline.paper_baselines.methods.guidance_common.scene import GuidanceScene

SCHEDULES = ("flat", "bellcurve", "posthoc")


def ellipsoid_clearance(center, points, radii):
    """Clearance of each obstacle surface point ``points (E, 3)`` to the EE ellipsoid at ``center`` (negative inside).

    Same construction as the authors' ``_ellipsoid_surface_distance``: radial projection onto the axis-aligned
    ellipsoid surface, Euclidean distance to it, sign from the normalized radius.
    """
    import jax.numpy as jnp

    delta = points - center[None]
    dn = delta / radii[None]
    nd = jnp.sqrt(jnp.sum(dn * dn, axis=-1) + 1e-12)
    surface = center[None] + dn / nd[:, None] * radii[None]
    dist = jnp.sqrt(jnp.sum((points - surface) ** 2, axis=-1) + 1e-12)
    return jnp.where(nd < 1.0, -dist, dist)


class ConstrainedFlowMatchingGuidance(GuidanceBase):
    name = "constrained_flow_matching"
    paper = {"cite": "[5]", "title": "Neuro-Symbolic Safety Guidance for VLA Models via Constrained Flow Matching",
             "arxiv": "2607.01378v1"}
    source = {"repo": "https://github.com/Dubascudes/VLA-Safety-ConstrainedFlowMatching",
              "sha": "b91153adc2f40d0ecfd2ae9ece4b6a2aed31dc7e",
              "license": "none ('has not yet been selected') -- re-implemented, nothing vendored",
              "files": ["software_module/patches/openpi.patch (SafetyConfig, SafetyGuidance._compute_cbf_qp_guidance)",
                        "software_module/patches/release-fixes.patch (guidance_scale)"]}

    def __init__(self, *, x0_mode: str = "identity", path_model: str = "integrator", **kw):
        # the original corrects the post-Euler state itself (no x0 estimate) and integrates the normalized action
        super().__init__(x0_mode=x0_mode, path_model=path_model, **kw)
        if self.params["guidance_schedule"] not in SCHEDULES:
            raise ValueError(f"guidance_schedule must be one of {SCHEDULES}")

    @classmethod
    def default_params(cls) -> dict:
        return {"safety_margin": 0.01, "min_guidance_time": 0.0, "dt": 0.01, "eef_ellipsoid_radii": (0.06, 0.12, 0.11),
                "cbf_gamma": 0.9, "guidance_schedule": "flat", "guidance_scale": 1.0, "bell_peak_tau": 0.45,
                "bell_width": 0.20, "bell_min_tau": 0.05, "bell_max_tau": 0.85, "n_outer": 4, "n_sweeps": 60,
                "action_box": True}

    def path_for(self, H: int, scene: GuidanceScene) -> EEPathModel:
        return EEPathModel(H=H, p0=scene.ee_pos, R_bw=scene.R_bw, model=self.path_model, y_prev=scene.y_prev,
                           dt=float(self.params["dt"]))

    def _prepare(self, scene: GuidanceScene, obs, norm) -> dict:
        ctx = {"obstacles": scene.obstacles.padded().as_jax(), "skip": scene.obstacles.empty,
               "radii": np.asarray(self.params["eef_ellipsoid_radii"], np.float64)}
        if self.params["action_box"]:
            from Baseline.paper_baselines.methods.lan_o3dp_guidance_adapted.guidance import action_box_normalized

            ctx["box_lo"], ctx["box_hi"] = action_box_normalized(norm)
        return ctx

    def time_scale(self, tau):
        """``_guidance_time_scale`` on the flow time *before* the Euler step (``tau = t_next + 1/num_steps``)."""
        import jax.numpy as jnp

        P = self.params
        if P["guidance_schedule"] == "flat":
            return jnp.where(tau > P["min_guidance_time"], 1.0, 0.0)
        if P["guidance_schedule"] == "bellcurve":
            on = (tau >= P["bell_min_tau"]) & (tau <= P["bell_max_tau"])
            return jnp.where(on, jnp.exp(-0.5 * ((tau - P["bell_peak_tau"]) / P["bell_width"]) ** 2), 0.0)
        return jnp.zeros_like(tau)  # posthoc: nothing during denoising (apply posthoc_correct to the final chunk)

    def _pos_input(self, x, ctx):
        """What the trajectory model integrates: normalized ``x[:, :3]`` (original) or raw xyz (``osc_response``)."""
        return x[:, :3] if self.path_model == "integrator" else self.raw_xyz(x, ctx["norm"])

    def constraints(self, x, delta, ctx, path):
        """``g (H,) = h_j - (1-gamma) h_{j-1}`` for the chunk ``x`` with ``delta (H, 3)`` added to its position dims."""
        import jax
        import jax.numpy as jnp

        P = self.params
        xd = x.at[:, :3].add(delta)
        p = path(self._pos_input(xd, ctx))
        traj = jnp.concatenate([jnp.asarray(path.p0, p.dtype)[None], p], axis=0)  # (H+1, 3)
        radii = jnp.asarray(ctx["radii"], p.dtype)

        def clearance(c):
            _, cp = all_distances_and_closest(c, ctx["obstacles"])
            return jnp.min(ellipsoid_clearance(c, cp, radii))

        h = jax.vmap(clearance)(traj) - P["safety_margin"]
        return h[1:] - (1.0 - P["cbf_gamma"]) * h[:-1]

    def correction(self, x, ctx, path):
        """Minimum-norm ``delta (H, 3)`` (zero when no constraint is violated)."""
        import jax
        import jax.numpy as jnp

        P = self.params
        H = x.shape[0]
        g_fn = lambda d: self.constraints(x, d, ctx, path)  # noqa: E731
        delta = jnp.zeros((H, 3), x.dtype)
        violated = jnp.any(g_fn(delta) < 0)
        for _ in range(int(P["n_outer"])):
            g = g_fn(delta)
            J = jax.jacfwd(g_fn)(delta).reshape(H, 3 * H)
            b = J @ delta.reshape(-1) - g  # linearised: J d' >= b
            G = J @ J.T
            diag = jnp.diag(G) + 1e-12

            def sweep(i, lam):
                j = i % H
                lj = jnp.maximum(0.0, lam[j] + (b[j] - G[j] @ lam) / diag[j])
                return lam.at[j].set(lj)

            lam = jax.lax.fori_loop(0, int(P["n_sweeps"]) * H, sweep, jnp.zeros(H, x.dtype))
            delta = (J.T @ lam).reshape(H, 3)
        return jnp.where(violated, delta, jnp.zeros_like(delta)), violated

    def _guide_one(self, x_next, x0, t_next, ctx, path):
        import jax.numpy as jnp

        P = self.params
        delta, _ = self.correction(x0, ctx, path)
        tau = t_next + 1.0 / self.num_steps
        scale = self.time_scale(tau) * P["guidance_scale"]
        push = scale * delta
        if not P["action_box"]:
            return x_next.at[:, :3].add(push)
        lo = jnp.asarray(ctx["box_lo"], x0.dtype)[None]
        hi = jnp.asarray(ctx["box_hi"], x0.dtype)[None]
        cur = x0[:, :3]
        if ctx.get("x0_mode") != "noise":  # original: the post-Euler state itself is the chunk (x0 = x_next)
            xn = x_next[:, :3]
            # intermediate steps: never push a dimension further out of the box than the state already is
            mid = jnp.clip(xn + push, jnp.minimum(lo, xn), jnp.maximum(hi, xn))
            # last step (the served chunk): every corrected entry ends inside the box (pushes of earlier steps that
            # the network carried along are bounded too); uncorrected entries stay bit-identical
            last = jnp.where(push != 0, jnp.clip(xn + push, lo, hi), xn)
            final = t_next < 0.5 / self.num_steps
            return x_next.at[:, :3].set(jnp.where(final, last, mid))
        k = jnp.maximum(1.0 - t_next, 1e-3)  # x0 coefficient of x_next (Lan-o3dp rule, x0_mode="noise" only)
        new = jnp.clip(cur + push / k, jnp.minimum(lo, cur), jnp.maximum(hi, cur))
        return x_next.at[:, :3].add((new - cur) * k)

    def posthoc_correct(self, x, ctx=None, scene=None):
        """``guidance_schedule="posthoc"``: one correction of a final chunk ``x (b, H, D)`` (after sampling)."""
        import jax
        import jax.numpy as jnp

        ctx = ctx or self.last_ctx
        scene = scene or self.last_scene
        path = self.path_for(x.shape[-2], scene)
        x = jnp.asarray(x)
        d = jax.vmap(lambda xb: self.correction(xb, ctx, path)[0])(x)
        if not self.params["action_box"]:
            return x.at[..., :3].add(self.params["guidance_scale"] * d)
        lo, hi = jnp.asarray(ctx["box_lo"], x.dtype), jnp.asarray(ctx["box_hi"], x.dtype)
        cur = x[..., :3]
        new = jnp.clip(cur + self.params["guidance_scale"] * d, jnp.minimum(lo, cur), jnp.maximum(hi, cur))
        return x.at[..., :3].set(new)


__all__ = ["ConstrainedFlowMatchingGuidance", "SCHEDULES", "ellipsoid_clearance"]
