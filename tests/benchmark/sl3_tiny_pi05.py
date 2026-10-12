"""SL3 helper: the Panda attention read-out on a tiny random pi05_libero model (CPU, .venv-openpi-live).

Run by ``tests/benchmark/test_panda_sl3.py::test_attention_reader_on_a_tiny_pi05_libero_model`` in a
subprocess with ``JAX_PLATFORMS=cpu`` (a CPU regression must not touch the shared GPU). The model is the
real ``pi05_libero`` Pi0 class and the real LIBERO input / model transforms (``LiberoInputs``,
``ResizeImages``, ``TokenizePrompt``, ...; no norm stats), with two test-only size reductions applied by
replacing two factory functions in this process: SigLIP ``So400m/14`` -> ``mu/14`` (same 14 px patches,
so still 16x16 tokens per 224 image) and gemma ``dummy`` at depth 10 (V11AB's layer 8 must exist).
Prints one JSON line.
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


def tiny_policy():
    import jax
    import openpi.models.gemma as gemma
    import openpi.models.siglip as siglip
    from openpi import transforms as _t
    from openpi.policies import policy as _policy
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
    model_config = dataclasses.replace(train.model, paligemma_variant="dummy", action_expert_variant="dummy",
                                       return_attn_probs=True)
    model = model_config.create(jax.random.key(0))
    data = train.data.create(pathlib.Path(tempfile.mkdtemp()), model_config)
    # = policy_config.create_trained_policy's input transform, without norm stats.
    transforms = [_t.InjectDefaultPrompt(None), *data.data_transforms.inputs,
                  _t.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs]
    return _policy.Policy(model, transforms=transforms)


def request(seed=0):
    rng = np.random.default_rng(seed)
    return {"observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": rng.normal(size=8),
            "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate"}


def main():
    from AG3S.adapters.panda import attention as pa

    policy = tiny_policy()
    reader = pa.PandaAttentionReader(policy)
    req = request()
    layout = reader.token_layout(req)
    out = reader.read(req)
    ref = pa.v11ab_attention_policy_grids(policy, req)
    diffs, mass = {}, {}
    for block in pa.PI05_LIBERO_TOKEN_BLOCKS:
        mine = out["cell_tokens"][block.tokens[0]:block.tokens[1]].reshape(pa.GRID, pa.GRID)
        diffs[block.model_key] = float(np.max(np.abs(mine - ref[block.v11ab_policy_camera])))
        mass[block.model_key] = float(out["pooled"][..., block.tokens[0]:block.tokens[1]].sum(-1).max())
    grids_equal = all(np.array_equal(out["grids"][b.wire_camera],
                                     out["cell_tokens"][b.tokens[0]:b.tokens[1]].reshape(pa.GRID, pa.GRID))
                      for b in pa.PI05_LIBERO_TOKEN_BLOCKS if b.used)
    no_feature = reader.read(req, full=False, with_feature=False)
    print(json.dumps({"layout": layout, "problems": pa.check_token_layout(layout), "diffs": diffs,
                      "max_block_mass_over_cells": mass, "pooled_shape": list(out["pooled"].shape),
                      "feature_shape": list(np.shape(out["feature"])), "grids_equal": grids_equal,
                      "same_without_feature": bool(np.array_equal(no_feature["cell_tokens"], out["cell_tokens"])),
                      "cameras": sorted(out["grids"])}))


if __name__ == "__main__":
    main()
