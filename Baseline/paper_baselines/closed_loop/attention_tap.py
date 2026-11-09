"""KNOWS F4a (2026-11-10): the pi0.5 attention grid of the policy call that produced the nominal chunk.

``AttentionTap`` sits between ``SeededPolicy`` and the openpi ``Policy`` (``PandaVLAOnly(SeededPolicy(AttentionTap(policy)))``).
It **does not change the policy call**: ``policy.infer`` runs exactly as without the tap (same object, same arguments,
same RNG advance), so the served nominal chunk is the one the arm would get without attention. Before the call the tap
reads ``policy._rng`` (``SeededPolicy`` has already re-keyed it); ``Policy.infer`` splits it as
``self._rng, sample_rng = split(self._rng)`` and ``Pi0.sample_actions`` draws ``noise = normal(sample_rng, (1, H, D))``,
so the tap rebuilds that very noise and evaluates **denoising step 0 of the same sample** (same observation through the
policy's own input transform, same noise, same weights) once more with ``return_attn_probs=True``:

    prefix pass (KV cache) -> one suffix pass at t = 1 on x_t = noise -> probs[layer 12, head 3, action queries, 0:256]
    -> mean over the action queries (KNOWS OPEN-Q 10 / P0b: mean) -> (16, 16) agent-view grid

This is the paper's App. 7.2 route (the fused forward is left untouched; the one map is recomputed from the same inputs
after the policy returned). openpi's ``sample_actions`` cannot return the probabilities (``Pi0Config.return_attn_probs``:
"sample_actions is not supported while this is on"), so a literal in-forward tap would replace the compiled sampler and
could move the chunk by XLA rounding; the recompute keeps the chunk bit-identical by construction. The attention-enabled
model shares the policy's parameter arrays (``nnx.merge`` of an attention graphdef with the policy's state; no second
checkpoint copy).

Orientation: the policy image is ``resize_with_pad(raw[::-1, ::-1], 224)``; the head camera's OpenCV image is
``raw[::-1]`` at 256 px, so policy patch ``(r, c)`` = OpenCV block rows ``[16r, 16r+16)``, columns ``[16(15-c), 16(16-c))``
(``AG3S/adapters/panda/attention.py``): the grid handed to the method is ``grid[:, ::-1]``.
"""
from __future__ import annotations

import dataclasses
import time
import traceback

import numpy as np

LAYER, HEAD = 12, 3          # KNOWS Sec. 3.4
AGENT_VIEW_TOKENS = (0, 256)  # base_0_rgb block of the pi05_libero prefix (AG3S/adapters/panda/attention.py)
GRID = 16
DENOISE_STEP = 0              # P0b paper-cell configuration (docs/08: denoise 0, mean over action queries)


class OpenpiAttentionBackend:
    """Step-0 attention of ``policy``'s next sample (see module docstring). ``config_model``: the policy's
    ``Pi0Config`` (``get_config(name).model``)."""

    def __init__(self, policy, config_model, *, layer: int = LAYER, head: int = HEAD, tokens=AGENT_VIEW_TOKENS):
        import flax.nnx as nnx
        import jax

        self._policy, self._nnx, self._jax = policy, nnx, jax
        self.layer, self.head, self.tokens = int(layer), int(head), (int(tokens[0]), int(tokens[1]))
        cfg = dataclasses.replace(config_model, return_attn_probs=True)
        abstract = nnx.eval_shape(lambda: cfg.create(jax.random.key(0)))
        self._graphdef, _ = nnx.split(abstract)
        self._state = nnx.state(policy._model)  # the policy's own arrays (shared, not copied)
        self._ah, self._ad = int(policy._model.action_horizon), int(policy._model.action_dim)
        self._jit = jax.jit(self._grid)

    def _grid(self, state, observation, noise):
        import einops
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)  # = sample_actions
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=make_attn_mask(prefix_mask, prefix_ar_mask),
                                       positions=positions)[1]
        batch = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, noise, jnp.broadcast_to(jnp.asarray(1.0, jnp.float32), (batch,)))  # step 0: t = 1, x_t = noise
        full = jnp.concatenate([einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]),
                                make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        _, _, probs = model.PaliGemma.llm([None, suffix_tokens], mask=full, positions=pos, kv_cache=kv_cache,
                                          adarms_cond=[None, adarms_cond])
        lo, hi = self.tokens
        blk = probs[self.layer, 0, 0, self.head, -self._ah:, lo:hi].astype(jnp.float32)  # [action queries, 256]
        return blk.mean(axis=0)

    def __call__(self, obs: dict, rng, noise=None) -> np.ndarray:
        """``(16, 16)`` grid in the **policy** image orientation for ``obs`` (the dict ``Policy.infer`` received) and
        the policy RNG **before** that call."""
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        if noise is None:
            _, sample_rng = jax.random.split(rng)  # = Policy.infer
            noise = jax.random.normal(sample_rng, (1, self._ah, self._ad))  # = Pi0.sample_actions
        else:
            noise = jnp.asarray(noise)
            noise = noise[None] if noise.ndim == 2 else noise
        g = np.asarray(self._jit(self._state, observation, noise), np.float32)
        n = int(round(np.sqrt(g.size)))  # 16 x 16 SigLIP grid for pi05 (224 px / 14)
        return g.reshape(n, n)

    def describe(self) -> dict:
        return {"backend": "openpi recompute of denoising step 0 (same obs / noise / weights, return_attn_probs)",
                "layer": self.layer, "head": self.head, "tokens": list(self.tokens), "query_agg": "mean",
                "denoise_step": DENOISE_STEP}


class AttentionTap:
    """Pass-through policy wrapper; ``last`` = the attention record of the last call (or ``None`` / an error)."""

    def __init__(self, policy, backend=None):
        self._policy = policy  # ``policy_seed.rng_owner`` walks ``_policy`` down to the openpi Policy
        self.backend = backend
        self.last: dict | None = None
        self.n = 0

    @property
    def metadata(self):
        return getattr(self._policy, "metadata", {})

    def reset(self):
        r = getattr(self._policy, "reset", None)
        if callable(r):
            r()

    def infer(self, obs: dict, **kwargs) -> dict:
        rng = getattr(self._policy, "_rng", None)  # read only; the policy advances it itself
        result = self._policy.infer(obs, **kwargs)
        self.n += 1
        if self.backend is None:
            self.last = {"grid": None, "reason": "no attention backend (fake policy)"}
            return result
        t0 = time.perf_counter()
        try:
            g = self.backend(obs, rng, kwargs.get("noise"))
            self.last = {"grid": np.ascontiguousarray(g[:, ::-1]), "orientation": "opencv (policy grid[:, ::-1])",
                         "ms": (time.perf_counter() - t0) * 1e3, "call": self.n}
        except Exception:  # noqa: BLE001 -- the chunk is already served; the method sees no attention this call
            self.last = {"grid": None, "error": traceback.format_exc(), "ms": (time.perf_counter() - t0) * 1e3}
        return result


__all__ = ["AGENT_VIEW_TOKENS", "AttentionTap", "DENOISE_STEP", "GRID", "HEAD", "LAYER", "OpenpiAttentionBackend"]
