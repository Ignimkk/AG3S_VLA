"""SL6 B5 helper: the noun-row audit reader on the tiny random pi05_libero model (CPU, .venv-openpi-live).

Run by ``tests/benchmark/test_sl6_b5_noun_rows.py`` in a subprocess with ``JAX_PLATFORMS=cpu``. Compares
`attention_audit.NounRowReader` (B5 extraction: `NounQueryExtractor`, all layers, sliced outside the jit) with the
served path of a ``noun_target:L<l>H<h>`` source (`attention_source.JaxPrefixBackend.run` -> `prefix_noun_impl`,
the cell sliced inside the jit) for every (layer, head) of the tiny model's layers 2..9. Prints one JSON line.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tests/benchmark")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler
    from AG3S.trajopt.attention_source import AttentionSource, JaxPrefixBackend, NounTokens, noun_lookup
    from benchmark.runners import attention_audit as audit
    from sl3_tiny_pi05 import request, tiny_policy

    policy = tiny_policy()
    tokens = NounTokens.from_config("pi05_libero", rule="libero")
    layers = tuple(range(2, 10))
    reader = audit.NounRowReader(policy, layers=layers, tokens=tokens)
    req = request()
    out = reader.read(req)
    inputs = policy._input_transform(dict(req))  # noqa: SLF001
    look = noun_lookup(inputs, req["prompt"], tokens)
    backend = JaxPrefixBackend(AttentionSampler(policy._model, num_steps=10))  # noqa: SLF001
    max_diff, bit_equal = 0.0, True
    heads = out["image"].shape[1]
    for i, layer in enumerate(layers):
        for head in range(heads):
            served = backend.run(inputs, source=AttentionSource("noun_target", layer, head), rows=look.rows,
                                 want_feature=False, v_layer=None)["cell"]
            d = float(np.max(np.abs(np.asarray(served, np.float32) - out["image"][i, head])))
            max_diff = max(max_diff, d)
            bit_equal &= bool(np.array_equal(np.asarray(served, np.float32), out["image"][i, head]))
    print(json.dumps({"ok": out["ok"], "lookup": out["lookup"], "shape": list(out["image"].shape),
                      "rows": out["rows"].tolist(), "max_abs_diff_vs_served": max_diff, "bit_equal": bit_equal,
                      "image_mass_head_block_max": float(out["image"][..., :256].sum(-1).max()),
                      "right_wrist_mass_max": float(out["image"][..., 512:768].sum(-1).max()),
                      "mass_shape": list(out["mass"]["image_mass"].shape)}))


if __name__ == "__main__":
    main()
