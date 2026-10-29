"""T48a — `kv_L4` feature of (state16, three images, prompt), prefix pass only (GPU stage).

**Same path as serving.** The model is loaded exactly like the live attention model
(`pi05_attention.load_policy` -> `return_attn_probs=True`, used by `attention_policy.
load_attention_model`), the observation is built like `pi05_attention.observation_from_step`
(HWC -> CHW, the policy's own `_input_transform`, batch axis, `Observation.from_dict`), it is
preprocessed once (`AttentionSampler._sample`, `pi05_attention.py:185`) and the prefix pass and the
reduction are the sampler's own jitted executables (`_jit_prefix`, `_jit_kv_feature`,
`pi05_attention.py:89-123`). The suffix / denoise loop is never run — the feature only depends on
the prefix KV cache.

The state string is therefore made by the serving tokenizer path (`AlohaInputs` -> quantile
`Normalize` -> `TokenizePrompt(discrete_state_input=True)`); variants only change the 16D float
state array, never the string.

State is cast to float32 before the transform, as SUBTASK-b did (`extract.py:134`).

Batching (`batch > 1`) uses a batched copy of `_kv_feature` (same reduction per sample). Batch 1
uses the sampler's own reduction. `FeatureExtractor.batch_parity` measures the difference.

Nothing here runs at import time; jax / openpi are imported inside the methods.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t48_hsr_io import CAMS


class FeatureExtractor:
    def __init__(self, config_name: str, checkpoint: str, *, batch: int = 1):
        import jax
        import jax.numpy as jnp

        from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler, load_policy
        from AG3S.ag3s.stages.subtask_probe import KV_HEAD, KV_LAYER, N_IMAGE_TOKENS
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.training import config as _config

        if batch < 1:
            raise ValueError("batch must be >= 1")
        self._jax = jax
        self.batch = int(batch)
        t0 = time.perf_counter()
        self.policy = load_policy(config_name, checkpoint)
        self.load_s = time.perf_counter() - t0
        self.sampler = AttentionSampler(self.policy._model)  # noqa: SLF001
        cfg = _config.get_config(config_name)
        # Decoder only — to record the exact prompt text the model saw (reporting).
        self._tok = PaligemmaTokenizer(cfg.model.max_token_len)

        def kv_batched(kv, prefix_mask):
            v = kv[1][KV_LAYER, :, :, KV_HEAD, :].astype(jnp.float32)          # (B, S, H)
            tm = prefix_mask[:, N_IMAGE_TOKENS:].astype(jnp.float32)            # (B, S_txt)
            txt = (v[:, N_IMAGE_TOKENS:] * tm[..., None]).sum(1) / tm.sum(1, keepdims=True)
            img = v[:, :N_IMAGE_TOKENS].mean(1)
            return jnp.concatenate([txt, img], axis=-1)

        self._kv_batched = jax.jit(kv_batched)
        self.t_prefix: list[float] = []   # seconds per call (prefix + reduction), after warm-up
        self.n_per_call: list[int] = []

    # ------------------------------------------------------------------ inputs

    def inputs(self, state16: np.ndarray, images_hwc: dict, prompt: str) -> dict:
        """The policy's own input transform of one sample (no batch axis)."""
        images = {c: np.transpose(np.asarray(images_hwc[c]), (2, 0, 1)) for c in CAMS}
        return self.policy._input_transform(  # noqa: SLF001
            {"state": np.asarray(state16, np.float32), "images": images, "prompt": str(prompt)})

    def decoded_prompt(self, inputs: dict) -> str:
        toks = np.asarray(inputs["tokenized_prompt"])
        mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
        return self._tok._tokenizer.decode([int(t) for t in toks[mask]])  # noqa: SLF001

    # ------------------------------------------------------------------ forward

    def _obs(self, batch_inputs: Sequence[dict]):
        import jax.numpy as jnp

        from openpi.models import model as _model

        stacked = self._jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *batch_inputs)
        obs = _model.Observation.from_dict(stacked)
        return _model.preprocess_observation(None, obs, train=False)  # exactly once

    def features(self, batch_inputs: Sequence[dict], *, timed: bool = True,
                 force_batched_reduction: bool = False) -> np.ndarray:
        """(len(batch_inputs), 512) float32. A short final batch is padded by repeating its last
        sample so the jitted shapes never change (the padding rows are dropped)."""
        n = len(batch_inputs)
        if n == 0:
            return np.zeros((0, 512), np.float32)
        b = self.batch
        if n > b:
            raise ValueError(f"{n} inputs > batch {b}")
        padded = list(batch_inputs) + [batch_inputs[-1]] * (b - n)
        obs = self._obs(padded)
        t0 = time.perf_counter()
        kv, mask = self.sampler._jit_prefix(self.sampler._state, obs)  # noqa: SLF001
        if b == 1 and not force_batched_reduction:
            f = self.sampler._jit_kv_feature(kv, mask)[None]  # noqa: SLF001 — the serving reduction
        else:
            f = self._kv_batched(kv, mask)
        f = self._jax.block_until_ready(f)
        dt = time.perf_counter() - t0
        if timed:
            self.t_prefix.append(dt)
            self.n_per_call.append(n)
        return np.asarray(f, np.float32)[:n]

    def warmup(self, sample: dict, n: int = 3) -> None:
        for _ in range(max(1, n)):
            self.features([sample] * self.batch, timed=False)

    def batch_parity(self, samples: Sequence[dict]) -> Optional[dict]:
        """With batch > 1: the same samples one by one through the serving batch-1 path vs batched.
        Compiles the batch-1 executables once (a few seconds). None when batch == 1."""
        if self.batch == 1 or not samples:
            return None
        samples = list(samples)[: self.batch]
        got = self.features(samples, timed=False)
        b, self.batch = self.batch, 1
        try:
            ref = np.concatenate([self.features([s], timed=False) for s in samples])
        finally:
            self.batch = b
        rel = np.linalg.norm(got - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-12)
        return {"n": int(len(samples)), "rel_l2_max": float(rel.max()),
                "max_abs": float(np.abs(got - ref).max())}

    # ------------------------------------------------------------------ resources

    def memory(self) -> dict:
        try:
            stats = self._jax.devices()[0].memory_stats() or {}
        except Exception:  # noqa: BLE001 — not every backend reports
            stats = {}
        keep = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        return {k: int(stats[k]) for k in keep if k in stats}

    def timing(self) -> dict:
        t = np.asarray(self.t_prefix, np.float64)
        n = np.asarray(self.n_per_call, np.float64)
        if not len(t):
            return {"n_calls": 0}
        per = t / n
        return {"n_calls": int(len(t)), "n_inputs": int(n.sum()), "batch": self.batch,
                "call_ms_median": float(np.median(t) * 1e3),
                "per_input_ms_median": float(np.median(per) * 1e3),
                "per_input_ms_p90": float(np.percentile(per, 90) * 1e3),
                "prefix_total_s": float(t.sum()), "load_s": float(self.load_s),
                "note": "prefix pass + kv_L4 reduction on device, block_until_ready; excludes "
                        "input transform, video decode, warm-up/JIT"}


def build_cpu_input_transform(config_name: str, checkpoint: str):
    """The policy input transform **without loading the model** (CPU smoke / `--check-tokens`).

    Mirrors `policy_config.create_trained_policy` (`policy_config.py:56-82`): no repack, default
    prompt None, data transforms (`AlohaInputs`), quantile `Normalize` with the checkpoint's norm
    stats, model transforms (`ResizeImages`, `TokenizePrompt(discrete_state_input)`,
    `PadStatesAndActions`). Extraction uses the loaded policy's own `_input_transform`, not this.
    """
    import pathlib

    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint) / "assets", data_config.asset_id)
    fn = transforms.compose([
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    tok = PaligemmaTokenizer(cfg.model.max_token_len)

    def decode(inputs: dict) -> str:
        toks = np.asarray(inputs["tokenized_prompt"])
        mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
        return tok._tokenizer.decode([int(t) for t in toks[mask]])  # noqa: SLF001

    return fn, decode, {"use_quantile_norm": bool(data_config.use_quantile_norm),
                        "asset_id": data_config.asset_id,
                        "discrete_state_input": bool(getattr(cfg.model, "discrete_state_input", False)),
                        "max_token_len": int(cfg.model.max_token_len)}
