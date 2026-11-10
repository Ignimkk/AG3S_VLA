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

**T82 (default off).** `hsr5_probe` (`hsr5.load_hsr5_probe`, `serve_safe --hsr5`) takes the extended path and asks the
backend for the hidden h_4 feature of the same noun prefix pass (`want_hsr5`; gemma `return_hidden` is turned on for
that pass only, `attention_source.llm_with_hidden`) -> `result["hsr5"] = {"p", "argmax", "probe", "ms"}`. Without it
the backend call has no `want_hsr5` kwarg and the result has no key.
"""

from __future__ import annotations

import logging
import time
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
                 backend: Any = None, default_prompt: Optional[str] = None,
                 prefix_feature: bool = False, hsr6_probe: Any = None, hsr5_probe: Any = None,
                 stage_hidden: bool = False):
        self._policy = policy
        if sampler is None:
            from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler

            sampler = AttentionSampler(attn_model, num_steps=num_denoise_steps)
        self._sampler = sampler
        self._noise_seed = int(noise_seed)
        self._subtask_probe = _load_probe(subtask_probe)
        #: AG3S-CORE-EVAL-HSR6 (`serve_safe --hsr6-*`, default None = off): the 6-phase probe on the same `kv_L4`
        #: feature -> `result["hsr6"]`. None: no key, and the feature is computed exactly when it was before.
        self._hsr6_probe = hsr6_probe
        #: T82 (`serve_safe --hsr5`, default None = off): the 5-stage probe on the hidden h_4 feature of the same noun
        #: prefix pass -> `result["hsr5"]`. None: no key, the backend is called exactly as before (no `want_hsr5`).
        self._hsr5_probe = hsr5_probe
        # --- stage track (P5b 2026-11-10) — begin
        #: `serve_safe --stage-track` (default False = off): one extra jitted prefix pass with the gemma hidden stack after the
        #: attention pass -> `result[stage_track.STAGE_HIDDEN_KEY]` (h_10 pooled, 6144 floats; `SafePolicy` pops it). Off: no
        #: key, no pass, every path exactly as before.
        self._stage_hidden = bool(stage_hidden)
        self._stage_pass = None
        # --- stage track — end
        # T57-A — begin. Default (both None) = the pre-T57 cell and path (`infer` never reaches the
        # extended branch). `attention_source`: a SPEC string or `AttentionSource`; `task_probe`: a
        # `NounProbe` (T55 prereg); `noun_tokens`: `NounTokens` (needed by a noun source or the probe).
        from AG3S.trajopt.attention_source import AttentionSource, parse_attention_source

        # --- T90 — begin: a parsed `HiddenProbeSource` is a source as well
        from AG3S.trajopt.attention_source import HiddenProbeSource

        if attention_source is not None and not isinstance(attention_source, (AttentionSource, HiddenProbeSource)):
            attention_source = parse_attention_source(attention_source)
        # --- T90 — end
        self._source = attention_source
        self._task_probe = task_probe
        #: IFG-S2SERVE (`serve_safe --confidence-gate on`, default False): also carry the prefix layer-8 image
        #: residual (`attention_source.IFG_PREFIX_L8_KEY`) for the confidence gate. Takes the extended path.
        self._prefix_feature = bool(prefix_feature)
        self._extended = (attention_source is not None or task_probe is not None or self._prefix_feature
                          or hsr5_probe is not None)
        if self._extended and noun_tokens is None and (
                task_probe is not None or (attention_source is not None and attention_source.is_noun)):
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
        # --- T90 — begin: `--attention-source hidden_probe[:NPZ]` — load the asset now (a bad file stops the server at start)
        self._hidden_probe = None
        if isinstance(attention_source, HiddenProbeSource):
            from AG3S.trajopt.hidden_probe import load_hidden_probe

            self._hidden_probe = load_hidden_probe(attention_source.asset_path())
            if hasattr(self._backend, "set_hidden_probe"):
                self._backend.set_hidden_probe(self._hidden_probe)
        # --- T90 — end
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
    def hsr5_probe(self):
        """T82 — the 5-stage probe in use (`SubtaskProbe`), or None (no `result["hsr5"]`)."""
        return self._hsr5_probe

    @property
    def hsr6_probe(self):
        """AG3S-CORE-EVAL-HSR6 — the 6-phase probe in use (`SubtaskProbe`), or None (no `result["hsr6"]`)."""
        return self._hsr6_probe

    def _want_feature(self) -> bool:
        return self._subtask_probe is not None or self._hsr6_probe is not None

    def _label_feature(self, result: dict, feature) -> None:
        """`result["subtask"]` (3-class HSR) and, with the HSR6 probe, `result["hsr6"]` from one `kv_L4` feature.
        Each failing is logged and leaves its key out; it never fails the policy call."""
        if self._subtask_probe is not None:
            try:
                result["subtask"] = self._subtask_probe.predict(feature)
            except Exception:  # noqa: BLE001 — the label failing must not fail the policy call
                logging.warning("subtask probe failed:\n%s", traceback.format_exc())
        if self._hsr6_probe is not None:
            try:
                result["hsr6"] = self._hsr6_probe.predict(feature)
            except Exception:  # noqa: BLE001 — HSR6 failing leaves `result["hsr6"]` out (SafePolicy: phase None)
                logging.warning("HSR6 probe failed:\n%s", traceback.format_exc())

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
            self._label_feature(result, feature)
        if self._stage_hidden:   # stage track (P5b) — off: no call
            self._add_stage_hidden(obs, result)
        return result

    # --- stage track (P5b 2026-11-10) — begin ----------------------------------------------
    def _add_stage_hidden(self, obs: dict[str, Any], result: dict[str, Any]) -> None:
        """`result[STAGE_HIDDEN_KEY]` = the h_10 stage feature of this request's observation, or the key left out (logged)
        when the pass fails — `SafePolicy` then has no stage for the request (the evidence latch stays in charge)."""
        from AG3S.trajopt.stage_track import STAGE_HIDDEN_KEY, build_stage_hidden_pass

        try:
            if self._stage_pass is None:
                self._stage_pass = build_stage_hidden_pass(self._sampler)
            t0 = time.perf_counter()
            feat = self._stage_pass(self._policy._input_transform(dict(obs)))  # noqa: SLF001
            result[STAGE_HIDDEN_KEY] = {"feature": feat, "ms": round((time.perf_counter() - t0) * 1000.0, 2)}
        except Exception:  # noqa: BLE001 — the stage feature failing must not fail the policy call
            logging.warning("stage-track hidden pass failed:\n%s", traceback.format_exc())
    # --- stage track — end

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
                want_feature=self._want_feature(),
                v_layer=None if probe is None else probe.layer,
                # IFG-S2SERVE — only with the flag (the kwarg is absent otherwise: the T57-A call as before)
                **({"want_resid": True} if self._prefix_feature else {}),
                # T82 — only with the HSR5 probe (absent otherwise: the call as before)
                **({"want_hsr5": True} if self._hsr5_probe is not None else {}))
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
        # --- T90 — begin: what the hidden probe read / how long it took (only this source adds keys)
        if self._hidden_probe is not None and src_rec is not None:
            src_rec.update(self._hidden_probe_record(out))
        # --- T90 — end
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
        if self._prefix_feature and out.get("resid_l8") is not None:
            # IFG-S2SERVE — the collector's storage rounding (float16 when finite), read back as float32.
            from AG3S.ag3s.runtime.confidence_gate import dump_rounding
            from AG3S.trajopt.attention_source import IFG_PREFIX_L8_KEY

            result[IFG_PREFIX_L8_KEY] = dump_rounding(out["resid_l8"])
        feature = out.get("feature")
        if feature is not None:
            self._label_feature(result, feature)
        if self._hsr5_probe is not None:
            self._label_hsr5(result, out)
        if self._stage_hidden:   # stage track (P5b) — off: no call
            self._add_stage_hidden(obs, result)
        return result

    # --- T90 — begin
    def _hidden_probe_record(self, out: dict) -> dict:
        """T90 — per-request keys of `result["attention_source"]` for the hidden probe: ``ms`` (the pass incl. the probe
        scoring + device-to-host copy, the same prefix forward the noun source runs), ``max_score`` per camera and
        overall (p(target)), ``argmax_patch`` (camera, row, col of the overall maximum). No scores (noun not found / a
        backend without them) → only ``ms``."""
        from AG3S.trajopt.hidden_probe import CAMERAS, GRID, camera_probabilities

        rec: dict = {}
        if out.get("probe_ms") is not None:
            rec["ms"] = float(out["probe_ms"])
        lg = out.get("probe_logits")
        if lg is not None and out.get("cell") is not None:
            p = camera_probabilities(lg).reshape(len(CAMERAS), GRID * GRID)
            rec["max_score"] = {c: round(float(p[i].max()), 6) for i, c in enumerate(CAMERAS)}
            ci = int(np.argmax(p.max(axis=1)))
            pi = int(np.argmax(p[ci]))
            rec["max"] = round(float(p[ci, pi]), 6)
            rec["argmax_patch"] = {"camera": CAMERAS[ci], "row": pi // GRID, "col": pi % GRID}
        return rec
    # --- T90 — end

    def _label_hsr5(self, result: dict, out: dict) -> None:
        """T82 — `result["hsr5"]` from the backend's `hsr5_feature` (`SubtaskProbe.predict` + `ms` of the pass). No
        feature or a failing probe leaves the key out (logged): `SafePolicy` reads that as "no stage"."""
        feat = out.get("hsr5_feature")
        if feat is None:
            logging.warning("HSR5: no feature on this request (%s)", out.get("hsr5_error") or "not returned")
            return
        try:
            rec = self._hsr5_probe.predict(feat)
        except Exception:  # noqa: BLE001 — HSR5 failing leaves `result["hsr5"]` out (SafePolicy: stage None)
            logging.warning("HSR5 probe failed:\n%s", traceback.format_exc())
            return
        if out.get("hsr5_ms") is not None:
            rec["ms"] = out["hsr5_ms"]
        result["hsr5"] = rec

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
            with_feature = self._want_feature()
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
