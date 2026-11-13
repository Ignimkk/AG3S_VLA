"""FWDATT (2026-11-13, default off): the step-0 action-query attention of **the sample that is served**.

`AttentionPolicy` (default path) reads attention from a second checkpoint copy, re-run with a fixed noise seed
(`noise_seed=0`) and all ten suffix passes; the attention is therefore not the attention of the sample the arm gets.
This module produces the action chunk **and** the attention from one sampling:

    prefix pass (KV cache)                                    -- the policy's own model (2-tuple, no probs)
    suffix pass at t = 1 on x_t = noise, return_attn_probs    -- step 0, unrolled out of the loop: v_0 and probs
    x_1 = noise + dt * v_0                                    -- the Euler update of `Pi0.sample_actions`
    while time >= -dt / 2: the remaining steps                -- the policy's own model, the `sample_actions` body

and hands back ``x_0`` (the chunk) plus ``probs[layer, 0, 0, head, action tokens, 0:768]`` (float32) of step 0 only.

**Same weights, no copy.** The attention-enabled graphdef (`Pi0Config.return_attn_probs=True`, which adds no
parameters) is merged with the policy's own state (`nnx.state(policy._model)`), as `Baseline/paper_baselines/
closed_loop/attention_tap.py` (KNOWS F4a) does; the prefix pass and steps 1.. use the policy's own graphdef, so probs
are materialized for the one suffix pass that is read.

**Same RNG, same order.** `InpassSampler.infer` follows `openpi.policies.policy.Policy.infer` line for line: input
transform, batch, ``self._rng, sample_rng = split(self._rng)``, sample, ``x[0]``, output transform, ``policy_timing``;
the noise is ``normal(sample_rng, (1, H, D))``, the draw `Pi0.sample_actions` makes from that key (drawn just before
the jitted sampler, as `attention_tap.py` does), so the RNG advances as it would without the flag and the noise is that
call's noise.

**Not bit-identical by construction.** The compiled program differs from `Pi0.sample_actions` (step 0 is outside the
while loop and one suffix pass also emits its probs), so XLA may fuse / pick kernels differently and the chunk can move
by rounding. The unit test (`tests/trajopt/test_fwdatt_inpass.py`) reports the CPU difference; the GPU difference is
the verifier's (FWDATT.impl.md "remaining risks").

**Refused (raise, not ignored):** a guidance hook in the policy's ``sample_kwargs`` (``guidance_fn`` /
``guidance_xv_fn``) or any other key than ``num_steps``; a PyTorch policy; a model whose state does not fit the
attention graphdef.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

__all__ = ["GUIDANCE_KEYS", "InpassSampler", "InpassUnsupported", "N_IMAGE_TOKENS", "check_sample_kwargs"]

#: the hooks `Pi0.sample_actions` takes (openpi SEAM/VLS `guidance_fn`, T93 `guidance_xv_fn`) -- not supported here.
GUIDANCE_KEYS: tuple[str, ...] = ("guidance_fn", "guidance_xv_fn")
#: the `sample_actions` keyword the in-pass loop reproduces.
SUPPORTED_SAMPLE_KWARGS: tuple[str, ...] = ("num_steps",)
#: image columns of the prefix (3 SigLIP 16 x 16 grids), = `AttentionSampler._suffix`'s slice.
N_IMAGE_TOKENS = 768
#: `Pi0.sample_actions`' default.
DEFAULT_NUM_STEPS = 10


class InpassUnsupported(ValueError):
    """The policy is in a form the in-pass sampler does not reproduce (it refuses instead of serving another sample)."""


def check_sample_kwargs(policy) -> int:
    """``num_steps`` of the policy's ``sample_kwargs``; raises `InpassUnsupported` on a guidance hook or any other key."""
    kw = dict(getattr(policy, "_sample_kwargs", None) or {})
    hooks = sorted(k for k in GUIDANCE_KEYS if kw.get(k) is not None)
    if hooks:
        raise InpassUnsupported(f"attention in-pass does not support guidance hooks in the policy's sample_kwargs: {hooks}")
    other = sorted(k for k in kw if k not in SUPPORTED_SAMPLE_KWARGS and k not in GUIDANCE_KEYS)
    if other:
        raise InpassUnsupported(f"attention in-pass does not reproduce sample_kwargs {other} (supported: "
                                f"{list(SUPPORTED_SAMPLE_KWARGS)})")
    n = int(kw.get("num_steps", DEFAULT_NUM_STEPS))
    if n < 1:
        raise InpassUnsupported(f"num_steps must be >= 1 (got {n})")
    return n


class InpassSampler:
    """`Policy.infer` of ``policy`` (a JAX openpi ``Policy``) that also returns the step-0 attention block of one cell.

    ``model_config``: the policy's ``Pi0Config`` (``get_config(name).model``) -- only its graph is used (with
    ``return_attn_probs=True``); the parameters are the policy's. ``layer`` / ``head``: the cell; the returned block is
    ``[action tokens, n_image_tokens]`` float32 (pooling over the action queries is the caller's).
    """

    def __init__(self, policy, model_config, *, layer: int, head: int, n_image_tokens: int = N_IMAGE_TOKENS):
        import flax.nnx as nnx
        import jax

        if getattr(policy, "_is_pytorch_model", False):
            raise InpassUnsupported("attention in-pass needs the JAX openpi policy (this one is PyTorch)")
        for need in ("_model", "_rng", "_input_transform", "_output_transform"):
            if not hasattr(policy, need):
                raise InpassUnsupported(f"attention in-pass needs the openpi Policy itself (no `{need}` on "
                                        f"{type(policy).__name__})")
        self._policy, self._nnx, self._jax = policy, nnx, jax
        self.num_steps = check_sample_kwargs(policy)
        self.layer, self.head, self.n_image_tokens = int(layer), int(head), int(n_image_tokens)
        model = policy._model  # noqa: SLF001
        self._ah, self._ad = int(model.action_horizon), int(model.action_dim)
        # the policy's own graph and arrays (prefix pass and steps 1..), and the attention graph on the same arrays
        self._graphdef, self._state = nnx.split(model)
        cfg = dataclasses.replace(model_config, return_attn_probs=True, return_hidden_states=False)
        abstract = nnx.eval_shape(lambda: cfg.create(jax.random.key(0)))
        self._attn_graphdef, abstract_state = nnx.split(abstract)
        if jax.tree.structure(abstract_state) != jax.tree.structure(self._state):
            raise InpassUnsupported("model_config does not describe the policy's model (state trees differ)")
        depth = int(model.PaliGemma.llm.module.configs[0].depth)
        heads = int(model.PaliGemma.llm.module.configs[0].num_heads)
        if not (0 <= self.layer < depth and 0 <= self.head < heads):
            raise InpassUnsupported(f"cell L{self.layer}H{self.head} is outside {depth} layers x {heads} heads")
        self._jit = jax.jit(self._sample_noise)

    # ------------------------------------------------------------------ the one sampling
    def _suffix(self, model, observation, prefix_mask, kv_cache, x_t, t, batch_size, prefix_len):
        """The body of `Pi0.sample_actions`' ``step`` up to ``v_t`` (+ probs when ``model`` returns them)."""
        import einops
        import jax.numpy as jnp

        from openpi.models.pi0 import make_attn_mask

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(t, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (batch_size, suffix_tokens.shape[1], prefix_len + suffix_tokens.shape[1])
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        out = model.PaliGemma.llm([None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
                                  adarms_cond=[None, adarms_cond])
        prefix_out, suffix_out = out[0]
        assert prefix_out is None
        v_t = model.action_out_proj(suffix_out[:, -self._ah:])
        return v_t, (out[2] if len(out) > 2 else None)

    def _sample_noise(self, state, observation, noise):
        """``(x_0, cell block)`` -- `Pi0.sample_actions` with step 0's probs of (layer, head) kept."""
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        plain = self._nnx.merge(self._graphdef, state)
        attn = self._nnx.merge(self._attn_graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / self.num_steps
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = plain.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = plain.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        prefix_len = prefix_tokens.shape[1]

        # step 0, out of the loop: same arithmetic as the loop body (time is the weak float32 the loop carries)
        t0 = jnp.asarray(1.0)
        v0, probs = self._suffix(attn, observation, prefix_mask, kv_cache, noise, t0, batch_size, prefix_len)
        x1, t1 = noise + dt * v0, t0 + dt
        cell = probs[self.layer, 0, 0, self.head, -self._ah:, :self.n_image_tokens].astype(jnp.float32)

        def step(carry):
            x_t, t = carry
            v_t, _ = self._suffix(plain, observation, prefix_mask, kv_cache, x_t, t, batch_size, prefix_len)
            return x_t + dt * v_t, t + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (x1, t1))
        return x_0, cell

    def sample(self, observation, rng=None, *, noise=None):
        """``(x_0, cell)`` on device for a batched ``Observation``. Without ``noise`` it is drawn from ``rng`` as
        `Pi0.sample_actions` draws it (``normal(rng, (b, H, D))``), **outside** the jitted sampler: drawn inside it, the
        CPU chunk moved by 2.4e-7 against `sample_actions` (a different fusion of the step-0 program), drawn outside it
        the chunk is bit-identical on CPU (tests/trajopt/test_fwdatt_inpass.py)."""
        import jax

        if noise is None:
            noise = jax.random.normal(rng, (observation.state.shape[0], self._ah, self._ad))
        return self._jit(self._state, observation, noise)

    # ------------------------------------------------------------------ Policy.infer
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
        """``(Policy.infer(obs) result, cell block [action tokens, n_image_tokens] float32)``; same keys, dtypes and
        RNG advance as `openpi.policies.policy.Policy.infer`."""
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model

        check_sample_kwargs(self._policy)  # a hook installed after start is refused too
        p = self._policy
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = p._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        p._rng, sample_rng = jax.random.split(p._rng)  # noqa: SLF001
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        x_0, cell = self.sample(observation, sample_rng, noise=noise)
        outputs = {"state": inputs["state"], "actions": x_0}
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = p._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs, np.asarray(cell, np.float32)

    def describe(self) -> dict[str, Any]:
        return {"backend": "in-pass (step 0 of the served sampling, return_attn_probs on that suffix pass only)",
                "layer": self.layer, "head": self.head, "denoise_step": 0, "num_steps": self.num_steps,
                "n_image_tokens": self.n_image_tokens}
