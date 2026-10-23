"""NUDGE [9] (Guo & Kavraki, "Zero-Shot Reactive Obstacle Avoidance for Generative Robot Policies",
arXiv:2609.35231v2) as an openpi ``guidance_fn`` (pure JAX). No public code: implemented from the paper (Sec. III).

Paper method (flow matching, Example III.2 / III.6, Algorithm 1)::

    A0_hat  = A^k - t_k v_theta                                    (clean-chunk estimate)
    C_SDF   = sum_i sum_{l,j} [ d_safe - Phi(T_l(q_{t+i}) c^{l,j}) + r^{l,j} ]_+^2   (robot spheres vs workspace SDF)
    A0_til  = A0_hat - w_k grad C_SDF(A0_hat),   w_k = rho_k / (kappa (1 - rho_k))
    A^{k-1} = (t_{k-1}/t_k) A^k + ((t_k - t_{k-1})/t_k) A0_til

The guided update differs from the unguided Euler step only by ``((t_k - t_{k-1})/t_k) (A0_til - A0_hat)``, so in the
post-Euler hook::

    x_next <- x_next + (dt_abs / t_k) * (clip(A0_hat - w g) - A0_hat),    t_k = t_next + dt_abs

LIBERO-Object setting of the paper: pi0.5 (EE-delta), H = 10, K = 10 Euler steps, rho = 0.85 at all steps, cuRobo
ESDF from one metric-depth view at 2 cm voxels with robot / target / basket / table masked out, 41 Panda spheres +
a 7 cm payload sphere. ``kappa`` (median non-zero l_inf gradient on a calibration set) and ``d_safe`` are not given.

Differences here (README): the "decoder" is the OSC response model with the orientation held (no IK), so only the
spheres that move rigidly with the EE are used (cuRobo ``franka.yml`` panda_hand + finger spheres, + the payload
sphere when ``holding``); ``kappa`` defaults to :data:`KAPPA_SL3_DEV` (calibrated with :func:`calibrate_kappa` on
recorded SL3 dev calls); ``kappa=None`` uses the per-call gradient l_inf instead (every non-zero gradient then gets a
full ``rho/(1-rho)`` step -- overshoots on grazing contacts); the SDF is the exact distance to the scene's obstacle
points / primitives.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.core import GuidanceBase
from Baseline.paper_baselines.methods.guidance_common.geometry import signed_distance
from Baseline.paper_baselines.methods.guidance_common.scene import T_HAND_GRIP, GuidanceScene


#: ``kappa`` calibrated as the paper prescribes (median non-zero l_inf gradient, normalized action coords) on 135
#: recorded calls of the SL3 dev run (episodes 0-9, calls 1, 4, ..., 19; scene obstacles + an injected 3 cm sphere
#: beside the path); ``scripts/smoke_guidance_methods.py --calibrate-kappa`` reproduces it (2026-10-23: 0.00859).
KAPPA_SL3_DEV = 0.0086


class NudgeGuidance(GuidanceBase):
    name = "nudge"
    paper = {"cite": "[9]", "title": "Zero-Shot Reactive Obstacle Avoidance for Generative Robot Policies (NUDGE)",
             "arxiv": "2609.35231v2"}
    source = {"repo": None, "sha": None, "license": None,
              "note": "no public code (survey.md [9]); implemented from the paper Sec. III, Alg. 1, Sec. V-B"}

    @classmethod
    def default_params(cls) -> dict:
        return {"rho": 0.85, "kappa": KAPPA_SL3_DEV, "d_safe": 0.02, "payload_radius": 0.07, "payload_offset_hand": (0.0, 0.0, 0.12),
                "clip_raw": True, "guided_steps": None}

    def _prepare(self, scene: GuidanceScene, obs, norm) -> dict:
        P = self.params
        ex = obs.extras or {}
        c, r = scene.hand_c, scene.hand_r
        if ex.get("holding"):  # payload sphere at a fixed offset from panda_hand
            T_hand = scene.ee_pose @ np.linalg.inv(T_HAND_GRIP)
            pc = T_hand[:3, :3] @ np.asarray(P["payload_offset_hand"], np.float64) + T_hand[:3, 3]
            c, r = np.vstack([c, pc[None]]), np.append(r, P["payload_radius"])
        return {"obstacles": scene.obstacles.padded().as_jax(), "skip": scene.obstacles.empty,
                "sphere_offsets": c - scene.ee_pos[None], "sphere_r": r, "n_spheres": int(len(r))}

    def cost(self, x0, ctx, path):
        """``C_SDF`` of one normalized chunk (hinge over every sphere at every chunk step)."""
        import jax.numpy as jnp

        p = path(self.raw_xyz(x0, ctx["norm"]))  # (H, 3)
        s = p[:, None, :] + jnp.asarray(ctx["sphere_offsets"], p.dtype)[None]  # (H, S, 3)
        phi = signed_distance(s, ctx["obstacles"])
        return jnp.sum(jnp.maximum(self.params["d_safe"] - phi + jnp.asarray(ctx["sphere_r"], p.dtype)[None], 0.0) ** 2)

    def gradient(self, x0, ctx, path):
        import jax

        return jax.grad(lambda x: self.cost(x, ctx, path))(x0)

    def _guide_one(self, x_next, x0, t_next, ctx, path):
        import jax
        import jax.numpy as jnp

        P = self.params
        g = self.gradient(x0, ctx, path)
        ginf = jnp.max(jnp.abs(g))
        kappa = jax.lax.stop_gradient(jnp.maximum(ginf, 1e-12)) if P["kappa"] is None else jnp.asarray(P["kappa"], x0.dtype)
        w = P["rho"] / (kappa * (1.0 - P["rho"]))
        x0_til = x0 - w * g
        if P["clip_raw"]:  # keep the nudged clean chunk inside the executable action box ([-1, 1] raw)
            lo = self.norm_of_raw(-jnp.ones((7,), x0.dtype), ctx["norm"])[:3]
            hi = self.norm_of_raw(jnp.ones((7,), x0.dtype), ctx["norm"])[:3]
            clipped = x0_til.at[:, :3].set(jnp.clip(x0_til[:, :3], lo[None], hi[None]))
            x0_til = jnp.where(g != 0, clipped, x0)  # only entries the nudge moved are clipped
        dt_abs = 1.0 / self.num_steps
        t_k = t_next + dt_abs
        upd = (dt_abs / t_k) * (x0_til - x0)
        if P["guided_steps"] is not None:  # e.g. 2 -> only the last two denoising steps (the paper's UR5 setting)
            upd = jnp.where(t_k <= P["guided_steps"] * dt_abs + 1e-6, upd, 0.0)
        return x_next + upd


def calibrate_kappa(method: NudgeGuidance, chunks_and_ctx: list) -> float:
    """``kappa`` = median non-zero l_inf gradient norm (normalized action coords) over ``[(x0 (H, D), ctx, path), ...]``."""
    import jax.numpy as jnp

    vals = []
    for x0, ctx, path in chunks_and_ctx:
        g = np.asarray(method.gradient(jnp.asarray(x0), ctx, path))
        v = float(np.max(np.abs(g)))
        if v > 0:
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


__all__ = ["KAPPA_SL3_DEV", "NudgeGuidance", "calibrate_kappa"]
