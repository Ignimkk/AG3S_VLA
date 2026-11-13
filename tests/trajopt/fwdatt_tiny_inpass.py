"""FWDATT helper: the in-pass sampler on a tiny random pi05_libero model (CPU, .venv-openpi-live).

Run by ``tests/trajopt/test_fwdatt_inpass.py`` in a subprocess with ``JAX_PLATFORMS=cpu`` (no GPU; the factory
replacements below stay in this process). The model is the real ``pi05_libero`` Pi0 class (``pi05=True``) with the
real LIBERO input / output transforms (no norm stats) and two test-only size reductions, as
``tests/benchmark/sl3_tiny_pi05.py``: SigLIP ``So400m/14`` -> ``mu/14`` (still 16 x 16 tokens per 224 image) and gemma
``dummy`` at depth 10 (layer 8 of the V11AB cell must exist). Random init; ``return_attn_probs`` is **off** on the
policy's model (the serving one). Prints one JSON line of measured numbers; the test asserts on them.
"""
import dataclasses
import json
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAYER, HEAD = 8, 2          # attention_policy._LAYER / _HEAD
ALT = (3, 5)                # a second cell (action:mean:L3H5)


def tiny_setup():
    import jax
    import openpi.models.gemma as gemma
    import openpi.models.siglip as siglip
    from openpi import transforms as _t
    from openpi.training import config as _config

    siglip_module, gemma_config = siglip.Module, gemma.get_config

    def tiny_siglip(**kwargs):
        kwargs["variant"] = "mu/14"
        return siglip_module(**kwargs)

    def deep_dummy(variant):
        config = gemma_config(variant)
        return dataclasses.replace(config, depth=10) if variant == "dummy" else config

    siglip.Module, gemma.get_config = tiny_siglip, deep_dummy
    train = _config.get_config("pi05_libero")
    model_config = dataclasses.replace(train.model, paligemma_variant="dummy", action_expert_variant="dummy")
    assert model_config.pi05 and not model_config.return_attn_probs
    model = model_config.create(jax.random.key(0))
    data = train.data.create(pathlib.Path(tempfile.mkdtemp()), model_config)
    # = policy_config.create_trained_policy's transforms, without norm stats
    inputs = [_t.InjectDefaultPrompt(None), *data.data_transforms.inputs,
              _t.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs]
    outputs = [*data.model_transforms.outputs,
               _t.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs]
    return model, model_config, inputs, outputs


def make_policy(model, inputs, outputs, *, seed=7, sample_kwargs=None):
    import jax
    from openpi.policies import policy as _policy

    p = _policy.Policy(model, transforms=inputs, output_transforms=outputs, sample_kwargs=sample_kwargs)
    p._rng = jax.random.key(seed)   # as policy_seed.reseed does (a typed key is not truthy for `rng or ...`)
    return p


def request(seed=0):
    rng = np.random.default_rng(seed)
    return {"observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": rng.normal(size=8),
            "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate"}


def batched(policy, req):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    inputs = policy._input_transform(jax.tree.map(lambda x: x, req))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    return _model.Observation.from_dict(inputs)


def main():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler
    from AG3S.trajopt.attention_policy import AttentionPolicy
    from AG3S.trajopt.inpass_attention import InpassSampler, InpassUnsupported

    model, model_config, tin, tout = tiny_setup()
    out: dict = {"jax": jax.__version__, "backend": jax.default_backend()}

    # --- 1. actions: in-pass vs Pi0.sample_actions (the policy's own jitted call), same rng -------------------------
    pol = make_policy(model, tin, tout)
    ip = InpassSampler(pol, model_config, layer=LAYER, head=HEAD)
    a_diffs, a_equal, keys = [], [], []
    for i in range(3):
        obs = batched(pol, request(i))
        rng = jax.random.fold_in(jax.random.key(11), i)
        ref = np.asarray(pol._sample_actions(rng, obs))           # = Policy.infer's sample call (module_jit)
        mine, cell = ip.sample(obs, rng)
        mine = np.asarray(mine)
        a_diffs.append(float(np.max(np.abs(mine - ref))))
        a_equal.append(bool(np.array_equal(mine, ref)))
        keys.append(i)
    out["actions_max_abs_diff"] = a_diffs
    out["actions_bit_equal"] = a_equal
    out["actions_dtype"] = str(mine.dtype)
    out["actions_ref_scale"] = float(np.max(np.abs(ref)))

    # --- 2. attention: in-pass step-0 cell vs AttentionSampler._jit_prefix / _jit_suffix, same noise ----------------
    from openpi.models import model as _model

    attn_cfg = dataclasses.replace(model_config, return_attn_probs=True)
    attn_graphdef, _ = nnx.split(nnx.eval_shape(lambda: attn_cfg.create(jax.random.key(0))))
    attn_model = nnx.merge(attn_graphdef, nnx.state(model))      # same arrays, attention graph
    sampler = AttentionSampler(attn_model, num_steps=10)
    p_diffs, p_shapes = [], []
    for i in range(3):
        obs = batched(pol, request(i))
        noise = jax.random.normal(jax.random.key(100 + i), (1, ip._ah, ip._ad), jnp.float32)
        _, cell = ip.sample(obs, noise=noise)
        cell = np.asarray(cell)
        pobs = _model.preprocess_observation(None, obs, train=False)
        kv, pmask = sampler._jit_prefix(sampler._state, pobs)
        _, probs = sampler._jit_suffix(sampler._state, pobs, noise, jnp.asarray(1.0, jnp.float32), kv, pmask)
        ref = np.asarray(probs)[LAYER, HEAD]                       # [action tokens, 768]
        p_diffs.append(float(np.max(np.abs(cell - ref))))
        p_shapes.append([list(cell.shape), list(ref.shape)])
    out["probs_max_abs_diff"] = p_diffs
    out["probs_shapes"] = p_shapes
    out["probs_row_sum_768"] = float(np.asarray(cell).sum(-1).max())

    # end-to-end: AttentionPolicy(inpass) with the seed-0 noise == AttentionPolicy default path (noise_seed=0) grids
    ap_on = AttentionPolicy(make_policy(model, tin, tout), None, subtask_probe=None, inpass=True,
                            model_config=model_config)
    ap_off = AttentionPolicy(make_policy(model, tin, tout), attn_model, subtask_probe=None)
    seed0 = np.asarray(jax.random.normal(jax.random.key(0), (1, ip._ah, ip._ad), jnp.float32))[0]
    r_on = ap_on.infer(request(0), noise=seed0)
    r_off = ap_off.infer(request(0), noise=seed0)
    out["ap_keys_on"] = sorted(r_on)
    out["ap_keys_off"] = sorted(r_off)
    out["ap_attention_cameras"] = sorted(r_on["attention"])
    out["ap_attention_shapes_dtypes"] = sorted({(tuple(v.shape), str(v.dtype)) for v in r_on["attention"].values()})
    out["ap_attention_max_abs_diff"] = max(float(np.max(np.abs(r_on["attention"][c] - r_off["attention"][c])))
                                           for c in r_off["attention"])
    out["ap_actions_max_abs_diff_same_noise"] = float(np.max(np.abs(r_on["actions"] - r_off["actions"])))
    # another cell by attention_source
    ap_alt = AttentionPolicy(make_policy(model, tin, tout), None, subtask_probe=None, inpass=True,
                             model_config=model_config, attention_source=f"action:mean:L{ALT[0]}H{ALT[1]}")
    r_alt = ap_alt.infer(request(0), noise=seed0)
    obs0 = batched(pol, request(0))
    blocks = sampler.attention(obs0, [0], noise_seed=0)[0]       # [L, heads, T, 768], default path's call
    from AG3S.trajopt.attention_source import cell_maps

    ref_alt = cell_maps(blocks.mean(axis=-2)[ALT[0], ALT[1]])
    out["alt_cell_max_abs_diff"] = max(float(np.max(np.abs(r_alt["attention"][c] - ref_alt[c]))) for c in ref_alt)
    out["alt_source_record"] = r_alt.get("attention_source")

    # --- 3. RNG advance + result keys / dtypes: two in-pass infers vs two Policy.infer ------------------------------
    pa, pb = make_policy(model, tin, tout, seed=5), make_policy(model, tin, tout, seed=5)
    ib = InpassSampler(pb, model_config, layer=LAYER, head=HEAD)
    ra = [pa.infer(request(3)), pa.infer(request(4))]
    rb = [ib.infer(request(3))[0], ib.infer(request(4))[0]]
    out["rng_equal_after_two"] = bool(np.array_equal(jax.random.key_data(pa._rng), jax.random.key_data(pb._rng)))
    out["infer_keys_equal"] = all(sorted(x) == sorted(y) for x, y in zip(ra, rb))
    out["infer_keys"] = sorted(ra[0])
    out["infer_dtypes_equal"] = all(
        {k: (np.asarray(v).dtype.str, np.shape(v)) for k, v in x.items() if k != "policy_timing"}
        == {k: (np.asarray(v).dtype.str, np.shape(v)) for k, v in y.items() if k != "policy_timing"}
        for x, y in zip(ra, rb))
    out["infer_actions_max_abs_diff"] = [float(np.max(np.abs(x["actions"] - y["actions"]))) for x, y in zip(ra, rb)]
    out["infer_actions_shape"] = list(np.shape(ra[0]["actions"]))
    out["policy_timing_keys"] = sorted(rb[0]["policy_timing"])

    # --- 5. guidance hooks refused ---------------------------------------------------------------------------------
    refusals = {}
    for key in ("guidance_fn", "guidance_xv_fn"):
        g = make_policy(model, tin, tout, sample_kwargs={key: (lambda *a: a[-2] if len(a) > 2 else a[0])})
        try:
            InpassSampler(g, model_config, layer=LAYER, head=HEAD)
            refusals[key] = "accepted"
        except InpassUnsupported as exc:
            refusals[key] = f"refused: {exc}"
        try:
            AttentionPolicy(g, None, subtask_probe=None, inpass=True, model_config=model_config)
            refusals[key + "/AttentionPolicy"] = "accepted"
        except InpassUnsupported as exc:
            refusals[key + "/AttentionPolicy"] = f"refused: {exc}"
    # a hook installed after construction is refused at infer
    late = make_policy(model, tin, tout)
    li = InpassSampler(late, model_config, layer=LAYER, head=HEAD)
    late._sample_kwargs = {"guidance_fn": lambda x, t: x}
    rng_before = np.asarray(jax.random.key_data(late._rng)).tolist()
    try:
        li.infer(request(0))
        refusals["late_hook"] = "accepted"
    except InpassUnsupported as exc:
        refusals["late_hook"] = f"refused: {exc}"
    refusals["late_hook_rng_untouched"] = rng_before == np.asarray(jax.random.key_data(late._rng)).tolist()
    # num_steps is reproduced (not refused)
    p5 = make_policy(model, tin, tout, sample_kwargs={"num_steps": 5})
    i5 = InpassSampler(p5, model_config, layer=LAYER, head=HEAD)
    obs = batched(p5, request(1))
    rng = jax.random.key(3)
    out["num_steps5_max_abs_diff"] = float(np.max(np.abs(np.asarray(i5.sample(obs, rng)[0])
                                                         - np.asarray(p5._sample_actions(rng, obs, num_steps=5)))))
    out["refusals"] = refusals
    print(json.dumps(out))


if __name__ == "__main__":
    main()
