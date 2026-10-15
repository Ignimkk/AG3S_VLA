"""Decorates a policy so its `infer()` result also carries a per-camera attention map.

**Why a second model instance.** `Pi0Config.return_attn_probs=True` bakes `return_probs=True` into
`gemma.Module` at construction time, and once that is on, `PaliGemma.llm(...)` returns a 3-tuple
everywhere it's called. `Pi0.sample_actions` (`pi0.py:245` and `:269`) still unpacks a 2-tuple, so
turning the flag on for the policy that serves actions breaks serving outright. Fixing that is an
openpi change, so instead: load a **second** copy of the same checkpoint with the flag on, and use it
only through `AttentionSampler` (`AG3S/ag3s/experiments/sources/pi05_attention.py`), which never calls
`sample_actions` — it drives `PaliGemma.llm` directly with the 3-tuple unpack it already expects.
That sampler is the same one `step-01-attention.json` was scored with; nothing here is reimplemented.

**Why this can't perturb the served actions.** The policy that actually produces `result["actions"]`
is the caller's own, untouched, called exactly as it would be without this wrapper. Attention comes
from a wholly separate model object, a separate forward pass, and is merged into the result dict
afterward. There is no shared state that a bug here could feed back into the action path.

**Subtask label (SUBTASK-c).** The same prefix pass also yields the `kv_L4` feature (prefix KV
cache, no extra forward), and the exported linear probe (`AG3S/ag3s/asset/subtask_probe/`,
`AG3S/ag3s/stages/subtask_probe.py`) turns it into `result["subtask"] =
{"p": {"pick", "place", "home"}, "argmax": str, "probe": "kv_L4_v1"}`. Like the attention, a
failure there is logged and leaves the key out; it never fails the policy call.

**Cost.** Only Euler step 0 is sampled — the (layer, head, agg) cell `step-01-attention.json` scored
best used `denoise=0` — so this is one prefix+suffix pass, not the full ~10-step denoising loop.

**T57-A (default off).** `attention_source` names another cell (`attention_source.parse_attention_source`:
``action:<agg>:L<l>H<h>`` or ``noun_target:L<l>H<h>``) and `task_probe` (a `NounProbe`) adds
`result["task_probe"]` for `SafePolicy`'s HELD gate. With neither given (the default), `infer` takes the
pre-T57 path unchanged — the extended path (`_infer_extended`) is a separate branch. With either given,
`result["attention_source"]` / `result["task_probe"]` record what was used (noun, token, ok, reason).
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

import numpy as np

__all__ = ["AttentionPolicy", "load_attention_model"]

#: `AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json` -> `best`.
_DENOISE_STEP = 0
_AGG = "last"
_LAYER = 8
_HEAD = 2


def load_attention_model(config_name: str, checkpoint: str):
    """A `Pi0` model from the same checkpoint, with `return_attn_probs=True`.

    Returns the bare model (not a `Policy`) — that's all `AttentionSampler` needs.
    """
    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy

    return load_policy(config_name, checkpoint)._model  # noqa: SLF001


class AttentionPolicy:
    """Wraps `policy` so `infer()`'s result carries `result["attention"]`.

    `attention` is `{policy camera name: [16, 16] float32}` — raw softmax mass, not normalized or
    thresholded (`AG3S` does that itself, and it needs the peak on the true scale to do it).

    `subtask_probe`: `"default"` loads the shipped `kv_L4_v1` asset, a path loads that asset, a
    `SubtaskProbe` is used as is, None turns the label off (no `result["subtask"]`). A probe that
    fails to load is logged and turned off — the attention path is unaffected.
    """

    def __init__(self, policy, attn_model, *, num_denoise_steps: int = 10, noise_seed: int = 0,
                 subtask_probe: Any = "default", attention_source: Any = None,
                 noun_tokens: Any = None, task_probe: Any = None, sampler: Any = None,
                 backend: Any = None, default_prompt: Optional[str] = None):
        self._policy = policy
        if sampler is None:
            from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler

            sampler = AttentionSampler(attn_model, num_steps=num_denoise_steps)
        self._sampler = sampler
        self._noise_seed = int(noise_seed)
        self._subtask_probe = _load_probe(subtask_probe)
        # T57-A — begin. Default (both None) = the pre-T57 cell and path (`infer` never reaches the
        # extended branch). `attention_source`: a SPEC string or `AttentionSource`; `task_probe`: a
        # `NounProbe` (T55 prereg); `noun_tokens`: `NounTokens` (needed by a noun source or the probe).
        from AG3S.trajopt.attention_source import AttentionSource, parse_attention_source

        if attention_source is not None and not isinstance(attention_source, AttentionSource):
            attention_source = parse_attention_source(attention_source)
        self._source = attention_source
        self._task_probe = task_probe
        self._extended = attention_source is not None or task_probe is not None
        if self._extended and noun_tokens is None and (task_probe is not None or attention_source.is_noun):
            raise ValueError("a noun_target attention source / the task probe needs the prompt's noun token "
                             "ids (noun_tokens=NounTokens.from_config(config))")
        self._noun_tokens = noun_tokens
        #: the prompt the policy injects when a request carries none (`create_trained_policy(default_prompt)`)
        #: — the noun is read from the prompt that went into the policy.
        self._default_prompt = default_prompt
        self._backend = backend
        if self._extended and self._backend is None:
            from AG3S.trajopt.attention_source import JaxPrefixBackend

            self._backend = JaxPrefixBackend(self._sampler, noise_seed=self._noise_seed)
        # T57-A — end

    @property
    def metadata(self) -> dict[str, Any]:
        meta = getattr(self._policy, "metadata", {})
        if not self._extended:
            return meta
        # T57-A — the keys exist only when a T57 flag is on (the default metadata is the pre-T57 one).
        meta = dict(meta or {})
        if self._source is not None:
            meta["attention_source"] = self._source.label
        if self._task_probe is not None:
            meta["task_probe"] = self._task_probe.record()
        return meta

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    @property
    def subtask_probe(self):
        """The probe in use (`SubtaskProbe`), or None when the label is off."""
        return self._subtask_probe

    @property
    def attention_source(self):
        """The `AttentionSource` given (None = the default cell, no flag)."""
        return self._source

    @property
    def task_probe(self):
        """The `NounProbe` in use, or None (no `result["task_probe"]`)."""
        return self._task_probe

    def infer(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = self._policy.infer(obs, **kwargs)
        if self._extended:
            return self._infer_extended(obs, result)
        try:
            result = dict(result)
            attention, feature = self._attention_and_feature(obs)
            result["attention"] = attention
        except Exception:  # noqa: BLE001 — attention failing must not fail the policy call
            logging.warning("attention extraction failed:\n%s", traceback.format_exc())
            return result
        if feature is not None:
            try:
                result["subtask"] = self._subtask_probe.predict(feature)
            except Exception:  # noqa: BLE001 — the label failing must not fail the policy call
                logging.warning("subtask probe failed:\n%s", traceback.format_exc())
        return result

    # --- T57-A: attention source · task probe (only with a flag) ----------------------------
    def _infer_extended(self, obs: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        """One prefix pass for the cell named by `attention_source` (default cell if only the probe is on),
        the subtask feature from the same pass, and — with the probe — the target-noun V row of the same
        cache. Failures never fail the policy call: the attention key is left out (as the default path
        does), and the probe record says ``ok: False`` (the HELD gate then holds — fail closed)."""
        from AG3S.trajopt.attention_source import (
            DEFAULT_SOURCE, cell_maps, feature_digest, noun_lookup)

        result = dict(result)
        source = self._source or DEFAULT_SOURCE
        probe = self._task_probe
        src_rec = None if self._source is None else source.record()
        probe_rec = None if probe is None else {"p": None, "ok": False, "noun_ok": False, "noun": None,
                                                 "error": "", "feature": probe.feature,
                                                 "probe_md5": probe.md5}
        try:
            inputs = self._policy._input_transform(dict(obs))  # noqa: SLF001
            need_noun = source.is_noun or probe is not None
            prompt = obs.get("prompt") or self._default_prompt
            look = noun_lookup(inputs, prompt, self._noun_tokens) if need_noun else None
            rows = None if look is None or not look.ok else look.rows
            out = self._backend.run(
                inputs, source=source, rows=rows,
                want_feature=self._subtask_probe is not None,
                v_layer=None if probe is None else probe.layer)
        except Exception:  # noqa: BLE001 — attention failing must not fail the policy call
            why = traceback.format_exc()
            logging.warning("attention extraction failed (T57-A path):\n%s", why)
            last = why.strip().splitlines()[-1] if why.strip() else "unknown"
            if src_rec is not None:
                result["attention_source"] = {**src_rec, "attention": "failed", "error": last}
            if probe_rec is not None:
                result["task_probe"] = {**probe_rec, "error": f"extraction failed: {last}"}
            return result

        if look is not None and src_rec is not None and source.is_noun:
            src_rec.update(look.record())
        if out.get("cell") is not None:
            result["attention"] = cell_maps(out["cell"])
            state = "ok"
        else:
            # noun not found: **no synthetic attention** — AG3S runs without a target (`attention_extractor`
            # turns the empty dict into no maps), the reason is in `attention_source.noun_error`.
            result["attention"] = {}
            state = "empty"
        if src_rec is not None:
            src_rec["attention"] = state
            result["attention_source"] = src_rec
        if probe_rec is not None:
            probe_rec.update({"noun": look.noun, "noun_ok": bool(look.ok), "token_index": look.token_index})
            if not look.ok:
                probe_rec["error"] = f"noun not found: {look.error}"
            elif out.get("v") is None:
                probe_rec["error"] = "no V row gathered"
            else:
                try:
                    v = np.asarray(out["v"], np.float32)[0]  # the target row
                    probe_rec.update({"p": probe.p(v), "ok": True, "feature_sha1": feature_digest(v)})
                except Exception:  # noqa: BLE001 — the gate reads ok=False (fail closed)
                    logging.warning("task probe failed:\n%s", traceback.format_exc())
                    probe_rec["error"] = "probe evaluation failed"
            result["task_probe"] = probe_rec
        feature = out.get("feature")
        if feature is not None:
            try:
                result["subtask"] = self._subtask_probe.predict(feature)
            except Exception:  # noqa: BLE001 — the label failing must not fail the policy call
                logging.warning("subtask probe failed:\n%s", traceback.format_exc())
        return result

    def _attention(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self._attention_and_feature(obs, with_feature=False)[0]

    def _attention_and_feature(self, obs: dict[str, Any], *, with_feature: Optional[bool] = None):
        """`(attention dict, kv_L4 feature or None)` from one prefix pass."""
        import jax
        import jax.numpy as jnp

        from AG3S.ag3s.experiments.sources.pi05_attention import AGGREGATIONS
        from AG3S.ag3s.experiments.sources.policy_record import (
            ATTENTION_GRID, CAMERA_BINDINGS, POLICY_CAMERA_NAMES)
        from openpi.models import model as _model

        inputs = self._policy._input_transform(dict(obs))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        model_obs = _model.Observation.from_dict(inputs)

        if with_feature is None:
            with_feature = self._subtask_probe is not None
        if with_feature:
            blocks, feature = self._sampler.attention_and_subtask_feature(
                model_obs, [_DENOISE_STEP], noise_seed=self._noise_seed)
        else:
            blocks = self._sampler.attention(model_obs, [_DENOISE_STEP],
                                             noise_seed=self._noise_seed)
            feature = None
        pooled = AGGREGATIONS[_AGG](blocks[_DENOISE_STEP])  # [L, heads, 768]
        cell = np.asarray(pooled[_LAYER, _HEAD])  # [768]

        out = {}
        for camera in POLICY_CAMERA_NAMES:
            lo, hi = CAMERA_BINDINGS[camera][2]
            out[camera] = cell[lo:hi].reshape(ATTENTION_GRID, ATTENTION_GRID)
        return out, feature


def _load_probe(spec: Any):
    """`AttentionPolicy(subtask_probe=...)` → a `SubtaskProbe` or None (logged, never raised)."""
    if spec is None or spec is False:
        return None
    try:
        from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

        if isinstance(spec, SubtaskProbe):
            return spec
        probe = SubtaskProbe.load(None if spec == "default" else spec)
        logging.info("subtask probe %s loaded (classes %s, dim %d)", probe.name,
                     list(probe.classes), probe.dim)
        return probe
    except Exception:  # noqa: BLE001 — no label is a degraded mode, not a failed server
        logging.warning("subtask probe could not be loaded (%r) — result['subtask'] is off:\n%s",
                        spec, traceback.format_exc())
        return None
