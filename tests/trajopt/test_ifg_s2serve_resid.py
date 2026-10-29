"""IFG-S2SERVE — the prefix layer-8 residual pass (`attention_source.resid_block_impl` / `prefix_resid_loop`), on a tiny
openpi gemma (`gemma.get_config("dummy")`, two experts as in pi0.5) under JAX CPU. Needs jax + openpi (the
`.venv-openpi-live` interpreter); skipped elsewhere.

| # | property |
|---|---|
| 1 | the loop with the slice **inside** one jit (traced layer index) is bit-identical to the held-out collector's loop (`hc_dump.py`: layer params sliced outside, one jit per call) |
| 2 | running all `depth` blocks then the prefix final norm reproduces the openpi `Module` forward (nn.scan + remat) — the loop is the model's own residual stream, so block 8's output is the stream after layer 8 |
| 3 | the loop's intermediate after block k equals the full forward of a k+1-layer truncation (stacked params cut to k+1) |
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("openpi.models.gemma")


def _tiny(depth=4, seq=12):
    import jax.numpy as jnp

    from openpi.models import gemma as G

    cfg = G.get_config("dummy")
    cfgs = [cfg, cfg]
    mod = G.Module(configs=cfgs, embed_dtype="float32")
    import flax.linen as nn

    # `gemma.Module.init` shadows linen's `init` (the openpi convenience method), so call linen's directly
    params = nn.Module.init(mod, jax.random.key(0), [False, True], method=G.Module.init)
    x = jax.random.normal(jax.random.key(1), (1, seq, cfg.width), jnp.float32)
    mask = jnp.ones((1, seq), bool)
    ar = jnp.zeros((1, seq), bool).at[0, seq - 3:].set(True)
    from openpi.models.pi0 import make_attn_mask

    attn = make_attn_mask(mask, ar)
    pos = jnp.cumsum(mask, 1) - 1
    return G, mod, params, cfgs, x, attn, pos


def test_traced_slice_loop_is_bit_identical_to_the_collectors_loop():
    from AG3S.trajopt.attention_source import prefix_resid_loop, resid_block_impl

    G, _, params, cfgs, x, attn, pos = _tiny()
    layers = params["params"]["layers"]
    block = G.Block(configs=tuple(cfgs), return_probs=False)
    mask4 = attn[:, None]
    # the collector (hc_dump.py): params sliced per layer outside the jit
    layer_p = [jax.tree.map(lambda a, l=l: a[l], layers) for l in range(4)]

    @jax.jit
    def layer(pl, x_, p_, m_):
        ys, _ = block.apply({"params": pl}, [x_, None], None, p_, m_, [None, None], True)
        return ys[0]

    ref = x
    for l in range(4):
        ref = layer(layer_p[l], ref, pos, mask4)
    jb = jax.jit(lambda ly, l, x_, p_, m_: resid_block_impl(block, ly, l, x_, p_, m_))
    got = prefix_resid_loop(x, pos, mask4, layers, jb, n_layers=4)
    assert got.dtype == ref.dtype and np.array_equal(np.asarray(got), np.asarray(ref))
    assert not np.array_equal(np.asarray(got), np.asarray(x))


def test_full_depth_loop_plus_final_norm_is_the_model_forward():
    from AG3S.trajopt.attention_source import prefix_resid_loop, resid_block_impl

    G, mod, params, cfgs, x, attn, pos = _tiny()
    (normed, _), _ = mod.apply(params, [x, None], pos, attn), None
    block = G.Block(configs=tuple(cfgs), return_probs=False)
    jb = jax.jit(lambda ly, l, x_, p_, m_: resid_block_impl(block, ly, l, x_, p_, m_))
    stream = prefix_resid_loop(x, pos, attn[:, None], params["params"]["layers"], jb, n_layers=4)
    out, _ = G.RMSNorm().apply({"params": params["params"]["final_norm"]}, stream, None)
    np.testing.assert_allclose(np.asarray(out), np.asarray(normed[0]), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_intermediate_is_the_truncated_models_stream(k):
    from AG3S.trajopt.attention_source import prefix_resid_loop, resid_block_impl

    G, _, params, cfgs, x, attn, pos = _tiny()
    block = G.Block(configs=tuple(cfgs), return_probs=False)
    jb = jax.jit(lambda ly, l, x_, p_, m_: resid_block_impl(block, ly, l, x_, p_, m_))
    stream = prefix_resid_loop(x, pos, attn[:, None], params["params"]["layers"], jb, n_layers=k)
    cut = G.Module(configs=[G.Config(**{**cfgs[0].__dict__, "depth": k})] * 2, embed_dtype="float32")
    p2 = {"params": {**params["params"], "layers": jax.tree.map(lambda a: a[:k], params["params"]["layers"])}}
    normed, _ = cut.apply(p2, [x, None], pos, attn)
    out, _ = G.RMSNorm().apply({"params": params["params"]["final_norm"]}, stream, None)
    np.testing.assert_allclose(np.asarray(out), np.asarray(normed[0]), rtol=2e-5, atol=2e-5)
