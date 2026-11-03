"""T73 -- `return_hidden` (openpi gemma) and the CPU hidden-state tool `pi05_hidden`.

Checks (T73.task.md section 2):

1. flag off == the pre-T73 gemma.py (BASE_COMMIT), bit for bit (outputs, KV, and the traced jaxpr), and flag on
   changes neither the outputs nor the KV; `Pi0Config.return_hidden_states` adds no parameters.
2. V recomputed from the stored h_l (pre_attention_norm + kv_einsum of block l+1, LoRA included) equals
   the KV cache's V of that block (relative error <= 1e-2 in bfloat16).
3. final_norm(h_L) equals the flag-off prefix output (relative error <= 1e-2).

Small random config on the CPU; needs flax/openpi (`.venv-openpi-live`), skipped elsewhere.
Run with ``JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytest.importorskip("flax")
pytest.importorskip("openpi")

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import openpi.models.gemma as gemma  # noqa: E402
import openpi.models.lora as lora  # noqa: E402

from AG3S.ag3s.experiments.sources import pi05_hidden as ph  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
GEMMA_REL = "third_party/openpi/src/openpi/models/gemma.py"
#: Last commit before T73 -- the reference the flag-off path must match (pinned so the test survives the commit).
BASE_COMMIT = "8ab3c7ef775737b628f5563a74ffdd13db40d023"


def test_imports_from_this_checkout():
    # worktree / env.sh pitfall: make sure the modules under test are this checkout's.
    assert pathlib.Path(gemma.__file__).resolve() == (REPO / GEMMA_REL).resolve()
    assert pathlib.Path(ph.__file__).resolve().is_relative_to(REPO)


@pytest.fixture(scope="module")
def gemma_head(tmp_path_factory):
    """gemma.py as of BASE_COMMIT (before T73), importable side by side."""
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:{GEMMA_REL}"], capture_output=True,
                         text=True, check=True).stdout
    assert "return_hidden" not in src
    path = tmp_path_factory.mktemp("head") / "gemma_head.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("gemma_head_t73", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _configs(mod):
    lc = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
    vlm = mod.Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16, lora_configs=lc)
    act = mod.Config(width=32, depth=4, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16, lora_configs=lc)
    return [vlm, act]


def _module(mod, dtype, **kw):
    return mod.Module(configs=_configs(mod), embed_dtype=dtype, adarms=True, **kw)


def _params(mod, dtype):
    m = _module(mod, dtype)
    # gemma.Module overrides `init` with its own init-method; call linen's init explicitly.
    params = nn.Module.init(m, jax.random.key(0), use_adarms=[False, True], method="init")
    # Perturb every parameter so LoRA (lora_b may init at 0), norm scales and adaRMS are all non-trivial.
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(1), len(leaves))
    leaves = [p + 0.05 * jax.random.normal(k, p.shape, p.dtype) for p, k in zip(leaves, keys, strict=True)]
    return jax.tree.unflatten(tree, leaves)


def _inputs(B=2, T=12, S=5):
    k = jax.random.split(jax.random.key(2), 4)
    x0 = jax.random.normal(k[0], (B, T, 64))
    x1 = jax.random.normal(k[1], (B, S, 32))
    cond = jax.random.normal(k[2], (B, 32))
    valid = jnp.ones((B, T), bool).at[1, -3:].set(False)  # padding on one batch row
    return x0, x1, cond, valid


def _prefix_args(x0, valid):
    from openpi.models.pi0 import make_attn_mask

    mask = make_attn_mask(valid, jnp.zeros(valid.shape[1], bool))
    positions = jnp.cumsum(valid, axis=1) - 1
    return mask, positions


def _bit_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert np.array_equal(np.asarray(x, np.float32), np.asarray(y, np.float32))


# ------------------------------------------------------------------------------ check 1


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_flag_off_is_head_and_flag_on_keeps_outputs(gemma_head, dtype):
    params = _params(gemma, dtype)
    x0, x1, cond, valid = _inputs()
    mask, positions = _prefix_args(x0, valid)

    head = _module(gemma_head, dtype).apply(params, [x0, None], positions, mask)
    off = _module(gemma, dtype).apply(params, [x0, None], positions, mask)
    on = _module(gemma, dtype, return_hidden=True).apply(params, [x0, None], positions, mask)

    assert len(head) == len(off) == 2 and len(on) == 3
    _bit_equal(head, off)  # flag off: outputs and KV identical to the committed module
    _bit_equal(off, on[:2])  # flag on: outputs and KV unchanged
    hidden = on[2]
    assert hidden.shape == (5, 2, 12, 64) and hidden.dtype == jnp.dtype(dtype)
    np.testing.assert_array_equal(np.asarray(hidden[0], np.float32), np.asarray(x0.astype(dtype), np.float32))


def test_flag_off_traces_the_same_program_as_head(gemma_head):
    params = _params(gemma, "bfloat16")
    x0, x1, cond, valid = _inputs()
    mask, positions = _prefix_args(x0, valid)

    def jaxpr(mod):
        m = _module(mod, "bfloat16")
        return str(jax.make_jaxpr(lambda p: m.apply(p, [x0, None], positions, mask))(params))

    assert jaxpr(gemma_head) == jaxpr(gemma)


def test_joint_and_suffix_passes_unchanged(gemma_head):
    """compute_loss-style joint pass and the sample_actions suffix pass (expert 0 absent -> hidden None)."""
    params = _params(gemma, "bfloat16")
    x0, x1, cond, valid = _inputs()
    from openpi.models.pi0 import make_attn_mask

    T, S = x0.shape[1], x1.shape[1]
    full_valid = jnp.concatenate([valid, jnp.ones((2, S), bool)], axis=1)
    ar = jnp.concatenate([jnp.zeros(T, bool), jnp.ones(1, bool), jnp.zeros(S - 1, bool)])
    jmask = make_attn_mask(full_valid, ar)
    jpos = jnp.cumsum(full_valid, axis=1) - 1
    kw = {"adarms_cond": [None, cond]}
    head = _module(gemma_head, "bfloat16").apply(params, [x0, x1], jpos, jmask, **kw)
    on = _module(gemma, "bfloat16", return_hidden=True).apply(params, [x0, x1], jpos, jmask, **kw)
    _bit_equal(head, on[:2])
    assert on[2].shape == (5, 2, T, 64)

    pmask, ppos = _prefix_args(x0, valid)
    _, kv = _module(gemma, "bfloat16").apply(params, [x0, None], ppos, pmask)
    smask = jnp.concatenate([jnp.repeat(valid[:, None, :], S, axis=1),
                             make_attn_mask(jnp.ones((2, S), bool), ar[T:])], axis=-1)
    spos = jnp.sum(valid, axis=-1)[:, None] + jnp.cumsum(jnp.ones((2, S), int), axis=-1) - 1
    skw = {"kv_cache": kv, "adarms_cond": [None, cond]}
    s_head = _module(gemma_head, "bfloat16").apply(params, [None, x1], spos, smask, **skw)
    s_on = _module(gemma, "bfloat16", return_hidden=True).apply(params, [None, x1], spos, smask, **skw)
    _bit_equal(s_head, s_on[:2])
    assert s_on[2] is None


def test_both_flags_order(gemma_head):
    params = _params(gemma, "bfloat16")
    x0, _, _, valid = _inputs()
    mask, positions = _prefix_args(x0, valid)
    probs_only = _module(gemma, "bfloat16", return_probs=True).apply(params, [x0, None], positions, mask)
    both = _module(gemma, "bfloat16", return_probs=True, return_hidden=True).apply(params, [x0, None], positions,
                                                                                   mask)
    assert len(both) == 4
    _bit_equal(probs_only, both[:3])
    assert both[3].shape == (5, 2, 12, 64)


def test_pi0_config_flag_adds_no_parameters():
    import flax.nnx as nnx

    from openpi.models import pi0_config

    def shapes(flag):
        cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                   return_hidden_states=flag)
        model = nnx.eval_shape(cfg.create, jax.random.key(0))
        st = nnx.state(model, nnx.Param).to_pure_dict()
        return {jax.tree_util.keystr(k): (v.shape, v.dtype) for k, v in jax.tree_util.tree_leaves_with_path(st)}

    off, on = shapes(False), shapes(True)
    assert off == on
    assert pi0_config.Pi0Config().return_hidden_states is False


def test_extractor_prefix_shapes_abstract():
    """The tool's jitted prefix function on a dummy-size pi0.5 (abstract: shapes only)."""
    import flax.nnx as nnx

    from openpi.models import pi0_config

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                               return_hidden_states=True)
    model = nnx.eval_shape(cfg.create, jax.random.key(0))
    ex = ph.HiddenExtractor(model)
    obs = cfg.fake_obs()
    prefix_out, kv_v, hidden, mask = jax.eval_shape(ex._prefix, ex._state, obs)  # noqa: SLF001
    T = 3 * 256 + cfg.max_token_len
    assert hidden.shape == (5, 1, T, 64)
    assert kv_v.shape == (4, 1, T, 1, 16)
    assert prefix_out.shape == (1, T, 64) and mask.shape == (1, T)
    llm = ex.llm_params()
    assert "layers" in llm and "final_norm" in llm
    assert llm["layers"]["attn"]["kv_einsum"]["w"].shape == (4, 2, 1, 64, 16)


# ------------------------------------------------------------------------------ checks 2 and 3


@pytest.mark.parametrize("dtype,tol", [("bfloat16", 1e-2), ("float32", 1e-5)])
def test_hidden_reproduces_v_cache_and_final_norm(dtype, tol):
    params = _params(gemma, dtype)
    x0, _, _, valid = _inputs()
    mask, positions = _prefix_args(x0, valid)
    normed, kv, hidden = _module(gemma, dtype, return_hidden=True).apply(params, [x0, None], positions, mask)
    off_normed, _ = _module(gemma, dtype).apply(params, [x0, None], positions, mask)

    vlm = _configs(gemma)[0]
    v_err, fn_err = ph.self_check(params["params"], hidden, kv[1], off_normed[0], vlm)
    assert v_err.shape == (vlm.depth,)
    assert float(v_err.max()) <= tol, v_err
    assert fn_err <= tol, fn_err
    # and the check can fail: a shifted layer index does not reproduce V
    wrong = ph.relerr(ph.recompute_v(params["params"], hidden[1:], vlm)[0], kv[1][0])
    assert wrong > 0.1


# ------------------------------------------------------------------------------ tool logic (numpy)


def test_token_selection_and_pool():
    T = 768 + 200
    idx, grp = ph.token_selection("all", T)
    assert idx.shape == (T,) and np.array_equal(idx, np.arange(T))
    assert np.array_equal(np.bincount(grp), [256, 256, 256, 200])
    idx_i, _ = ph.token_selection("image", T)
    idx_t, grp_t = ph.token_selection("text", T)
    assert idx_i.shape == (768,) and idx_t[0] == 768 and set(grp_t) == {3}

    rng = np.random.default_rng(0)
    h = rng.normal(size=(19, T, 8)).astype(np.float32)
    pmask = np.ones(T, bool)
    pmask[768 + 40:] = False  # text padding
    stored, valid = ph.select_and_pool(h, pmask, [0, 4, 18], "all", "none")
    assert stored.shape == (3, T, 8) and valid.sum() == 768 + 40
    np.testing.assert_array_equal(stored[1], h[4])
    pooled, _ = ph.select_and_pool(h, pmask, [0, 18], "all", "mean")
    assert pooled.shape == (2, 4, 8)
    np.testing.assert_allclose(pooled[1, 1], h[18, 256:512].mean(0), rtol=1e-6)
    np.testing.assert_allclose(pooled[0, 3], h[0, 768:808].mean(0), rtol=1e-6)  # padding excluded
    assert ph.pool_groups("text") == ["text"]


def test_parse_layers():
    assert ph.parse_layers(["all"], 18) == list(range(19))
    assert ph.parse_layers(["18", "4", "4"], 18) == [4, 18]
    with pytest.raises(SystemExit):
        ph.parse_layers(["19"], 18)


def test_read_both_record_formats(tmp_path):
    img = {k: np.full((224, 224, 3), i, np.uint8) for i, k in enumerate(ph.POLICY_CAMERA_NAMES)}
    run = tmp_path / "run_0000"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"prompt": "Put the apple in the basket."}))
    for t in (0, 8):
        np.savez_compressed(run / f"step_{t // 8:05d}.npz", t_step=np.int64(t), state=np.zeros(16),
                            **{f"image_{k}": v for k, v in img.items()})
    kind, files, prompt = ph.list_frames(run)
    assert kind == "ag3s" and len(files) == 2 and prompt.startswith("Put")
    fr = ph.read_frame(kind, files[1], prompt)
    assert fr.t_step == 8 and fr.frame_id == "step_00001@t8"
    assert fr.images["cam_left_wrist"].shape == (224, 224, 3) and fr.images["cam_left_wrist"][0, 0, 0] == 1

    sess = tmp_path / "session_x"
    sess.mkdir()
    for n in (0, 1, 10):
        d = {"inputs/state": np.zeros(16, np.float32), "inputs/prompt": "Place the apple into the basket.",
             **{f"inputs/images/{k}": np.transpose(v, (2, 0, 1)) for k, v in img.items()},
             "outputs/actions": np.zeros((50, 16), np.float32)}
        np.save(sess / f"step_{n}.npy", d, allow_pickle=True)
    kind, files, prompt = ph.list_frames(sess)
    assert kind == "openpi" and [f.stem for f in files] == ["step_0", "step_1", "step_10"] and prompt is None
    fr = ph.read_frame(kind, files[2], None)
    assert fr.t_step == 10 and fr.prompt.startswith("Place")
    assert fr.images["cam_right_wrist"].shape == (224, 224, 3) and fr.images["cam_right_wrist"][0, 0, 0] == 2
