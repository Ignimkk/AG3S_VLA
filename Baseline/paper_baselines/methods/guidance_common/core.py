"""Shared machinery of the sampling-guidance methods (OmniGuide, constrained flow matching, Lan-o3dp adapted, NUDGE).

The openpi hook (``third_party/openpi/src/openpi/models/pi0.py:276-283``) is **post-Euler**::

    x_next = x_t + dt * v_t ; t_next = t + dt ; x_next = guidance_fn(x_next, t_next)     (dt = -1/num_steps)

``guidance_fn`` sees neither ``v_t`` nor ``x_t``. Every original method needs a clean-chunk estimate ``x0_hat``
(OmniGuide ``x_t - t v_t``, NUDGE ``A^k - t_k v``, Lan-o3dp ``pred_original_sample``). Two estimates are offered
(``x0_mode``):

* ``"noise"``    -- the caller passes the same ``noise`` it gives ``sample_actions(noise=...)``; with the linear path
                   ``x_t = t eps + (1-t) x0`` the estimate is ``x0_hat = (x_next - t_next eps) / (1 - t_next)``
                   (exact at ``t_next = 0`` and for a straight flow);
* ``"identity"`` -- ``x0_hat = x_next`` (no noise available; exact only at the last step).

``"auto"`` = noise when given, else identity. Gradients are taken w.r.t. ``x0_hat`` and used as if
``d x0_hat / d x_t = I`` (no back-propagation through the frozen network, which the hook cannot reach).

All functions here are pure JAX and run under ``jax.jit`` inside ``sample_actions``. ``guidance_fn`` built with
``enabled=False`` returns ``x_next`` itself (bit-identical); so does ``guidance_fn=None`` (openpi default).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel
from Baseline.paper_baselines.methods.guidance_common.scene import GuidanceScene, build_scene
from Baseline.paper_baselines.methods.interface import SceneObservation, jsonable, normalize_actions, unnormalize_actions

X0_MODES = ("auto", "noise", "identity")
POS_DIMS = 3  # x[..., :3] = EE position delta (LIBERO OSC_POSE)


def raw_to_norm_scale(norm: dict, d: int = 7) -> np.ndarray:
    """d(normalized)/d(raw) per action dim (affine normalization -> constant)."""
    if "q01" in norm and "q99" in norm:
        q01, q99 = np.asarray(norm["q01"], np.float64)[:d], np.asarray(norm["q99"], np.float64)[:d]
        return 2.0 / (q99 - q01 + 1e-6)
    return 1.0 / (np.asarray(norm["std"], np.float64)[:d] + 1e-6)


def x0_estimate(x_next, t_next, noise, mode: str):
    import jax.numpy as jnp

    if mode == "identity" or noise is None:
        return x_next
    return (x_next - t_next * noise) / jnp.maximum(1.0 - t_next, 1e-3)


def load_norm_stats(path: str, key: str = "actions") -> dict:
    """openpi ``norm_stats.json`` -> ``{"mean","std","q01","q99"}`` arrays (for the ``norm`` argument)."""
    import json

    d = json.loads(open(path).read())
    d = d.get("norm_stats", d)[key]
    return {k: np.asarray(v, np.float32) for k, v in d.items() if v is not None}


class GuidanceBase:
    """Template: subclasses implement :meth:`_prepare` (numpy, once per observation) and :meth:`_guide_one`
    (pure JAX, one chunk ``(H, D)`` normalized) -> corrected ``x_next`` chunk."""

    name = "guidance_base"
    paper: dict = {}
    source: dict = {}
    #: keyword arguments of build_scene
    default_scene_kwargs: dict = {}

    def __init__(self, *, num_steps: int = 10, x0_mode: str = "auto", path_model: str = "osc_response",
                 enabled: bool = True, scene_kwargs: dict | None = None, **params):
        if x0_mode not in X0_MODES:
            raise ValueError(f"x0_mode must be one of {X0_MODES}")
        self.num_steps = int(num_steps)
        self.x0_mode = x0_mode
        self.path_model = path_model
        self.enabled = bool(enabled)
        self.scene_kwargs = {**self.default_scene_kwargs, **(scene_kwargs or {})}
        self.params = {**self.default_params(), **params}
        unknown = set(params) - set(self.default_params())
        if unknown:
            raise TypeError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.last_scene: GuidanceScene | None = None
        self.last_ctx: dict | None = None

    # --- to override --------------------------------------------------------------------------------------------
    @classmethod
    def default_params(cls) -> dict:
        return {}

    def _prepare(self, scene: GuidanceScene, obs: SceneObservation, norm: dict) -> dict:
        return {}

    def _guide_one(self, x_next, x0, t_next, ctx: dict, path: EEPathModel):
        raise NotImplementedError

    # --- interface ----------------------------------------------------------------------------------------------
    def provenance(self) -> dict:
        return {"name": self.name, "family": "sampling_guidance", "paper": self.paper, "source": self.source,
                "hook": "openpi pi0.sample_actions(guidance_fn=...) post-Euler (third_party/openpi/src/openpi/models/pi0.py:276-283)",
                "num_steps": self.num_steps, "x0_mode": self.x0_mode, "path_model": self.path_model,
                "enabled": self.enabled, "params": jsonable(self.params), "scene_kwargs": jsonable(self.scene_kwargs),
                "privileged_inputs": None if self.last_scene is None else self.last_scene.info.get("privileged_inputs"),
                "scene_info": None if self.last_scene is None else jsonable(self.last_scene.info)}

    def reset(self) -> None:
        self.last_scene, self.last_ctx = None, None

    def path_for(self, H: int, scene: GuidanceScene) -> EEPathModel:
        return EEPathModel(H=H, p0=scene.ee_pos, R_bw=scene.R_bw, model=self.path_model, y_prev=scene.y_prev)

    def make_guidance_fn(self, obs: SceneObservation, norm: dict, *, noise=None,
                         scene: GuidanceScene | None = None) -> Callable:
        """``guidance_fn(x_next (b, H, D), t_next) -> x_next'`` for ``pi0.sample_actions``.

        ``noise``: the ``(b, H, D)`` noise passed to ``sample_actions(noise=...)`` (enables ``x0_mode="noise"``).
        """
        if not self.enabled:
            def identity(x_next, t_next):
                return x_next
            self.last_scene = scene
            return identity

        import jax
        import jax.numpy as jnp

        scene = scene if scene is not None else build_scene(obs, **self.scene_kwargs)
        ctx = self._prepare(scene, obs, norm)
        ctx["norm"] = {k: np.asarray(v, np.float32) for k, v in norm.items() if k in ("q01", "q99", "mean", "std")}
        self.last_scene, self.last_ctx = scene, ctx
        mode = self.x0_mode if not (self.x0_mode == "auto") else ("noise" if noise is not None else "identity")
        if mode == "noise" and noise is None:
            raise ValueError("x0_mode='noise' needs the sampling noise (make_guidance_fn(..., noise=noise))")
        self._resolved_x0_mode = mode
        ctx["x0_mode"] = mode  # read at trace time by methods that bound the implied clean chunk (Lan-o3dp clip_sample)
        eps_const = None if mode == "identity" else jnp.asarray(np.asarray(noise), jnp.float32)
        if ctx.get("skip"):  # e.g. OmniGuide freeze zone / nothing to avoid: identity, recorded in ctx["skip"]
            def skipped(x_next, t_next):
                return x_next
            return skipped

        def fn(x_next, t_next):
            H = x_next.shape[-2]
            path = self.path_for(H, scene)
            eps = None if eps_const is None else eps_const.astype(x_next.dtype)
            x0 = x0_estimate(x_next, t_next, eps, mode)

            def one(xn, x0b):
                return self._guide_one(xn, x0b, t_next, ctx, path)

            return jax.vmap(one)(x_next, x0).astype(x_next.dtype)

        return fn

    # --- helpers for subclasses (pure JAX) -------------------------------------------------------------------------
    @staticmethod
    def raw_xyz(x, norm: dict):
        import jax.numpy as jnp

        return unnormalize_actions(x[..., :7], norm, xp=jnp)[..., :POS_DIMS]

    @staticmethod
    def norm_of_raw(a, norm: dict):
        import jax.numpy as jnp

        return normalize_actions(a, norm, xp=jnp)


# --- surrogate sampler ---------------------------------------------------------------------------------------------
def surrogate_sample(x_target, noise, *, num_steps: int = 10, guidance_fn: Callable | None = None, kind: str = "tracking"):
    """The pi0 Euler loop (same ``while_loop`` / time arithmetic as ``pi0.sample_actions``) with a **surrogate**
    velocity field ``v(x_t, t) = (x_t - x_target) / t`` -- the conditional-OT field that transports ``noise`` to
    ``x_target`` in a straight line. Stands in for the frozen network on CPU: with ``guidance_fn=None`` it returns
    ``x_target`` (up to float error), with guidance it returns what the method makes of a policy that "wants"
    ``x_target``. Used by tests and the recorded-frame smoke; the real-model smoke is ``gpu_smoke.py``.

    ``kind`` brackets how much of an intermediate correction survives later steps:

    * ``"tracking"`` -- ``v = (x_t - x_target) / t``: the policy re-aims at its mode from wherever it is, so a deviation
      made at ``t_next`` shrinks by ``t_end / t_next``; only the last call (``t_next = 0``) survives fully (worst case);
    * ``"constant"`` -- ``v = noise - x_target`` (independent of ``x_t``): every correction is carried to the end
      unchanged (best case). A real network lies in between.
    """
    if kind not in ("tracking", "constant"):
        raise ValueError("kind must be 'tracking' or 'constant'")
    import jax
    import jax.numpy as jnp

    dt = -1.0 / num_steps
    x_target = jnp.asarray(x_target, jnp.float32)
    eps = jnp.asarray(noise, jnp.float32)

    def step(carry):
        x_t, time = carry
        v_t = (x_t - x_target) / time if kind == "tracking" else eps - x_target
        x_next = x_t + dt * v_t
        time_next = time + dt
        if guidance_fn is not None:
            x_next = guidance_fn(x_next, time_next)
        return x_next, time_next

    def cond(carry):
        return carry[1] >= -dt / 2

    x0, _ = jax.lax.while_loop(cond, step, (eps, jnp.float32(1.0)))
    return x0


def chunk_to_model_space(raw_chunk: np.ndarray, norm: dict, action_dim: int = 32) -> np.ndarray:
    """Recorded raw LIBERO chunk ``(H, 7)`` -> normalized model-space ``(1, H, action_dim)`` (zero padded)."""
    raw = np.asarray(raw_chunk, np.float64)
    x = normalize_actions(raw[:, :7], norm, xp=np)
    out = np.zeros((1, raw.shape[0], action_dim), np.float32)
    out[0, :, :7] = x
    return out


def model_space_to_chunk(x: Any, norm: dict) -> np.ndarray:
    """Inverse of :func:`chunk_to_model_space` (first batch element, first 7 dims, raw)."""
    x = np.asarray(x, np.float64)[0, :, :7]
    return unnormalize_actions(x, norm, xp=np)


__all__ = ["GuidanceBase", "POS_DIMS", "X0_MODES", "chunk_to_model_space", "load_norm_stats", "model_space_to_chunk",
           "raw_to_norm_scale", "surrogate_sample", "x0_estimate"]
