"""T57-A — which attention feeds AG3S (`--attention-source SPEC`) and the noun-token probe of the HELD gate.

Two serving additions share this module, both **default off** (`serve_safe` without the flags builds
exactly the pre-T57 `AttentionPolicy`, whose path does not touch anything here):

* **Stage 1 — attention source.** `AttentionPolicy` used one hard-coded cell (action query, Euler step 0,
  ``last`` pooling, L8H2 — `attention_policy._DENOISE_STEP/_AGG/_LAYER/_HEAD`). A SPEC names the cell:

  ======================  ==================================================================================
  SPEC                    map fed to AG3S (raw softmax mass, three 16 x 16 camera grids, no smoothing)
  ======================  ==================================================================================
  ``action:last:L8H2``    the serving default (= no flag)
  ``action:mean:L10H7``   T61's pre-registered selection (action query, denoise 0, mean, k 1) — implemented
                          only; adoption waits for the T61 held-out result
  ``noun_target:L10H7``   T57 stage 1 arm **N** = `t54_arms.SOURCES["noun"]` (raw, σ 0, k 1): the prefix
                          attention **row of the prompt's target-noun token**, layer 10 head 7, image columns
  ======================  ==================================================================================

* **Stage 2 — the noun-token probe.** T55's pre-registered noun arm (`T55.prereg.json` ``arms.noun``):
  feature ``v5`` = the layer-5 V cache at the target-noun token's prefix row (kv head 0, 256-d;
  `t53_extract.noun_token_impl` / `t55_study.noun_matrix("v5")`), logistic probe ``noun_probe.npz``
  (C 0.1, `t50_core.LogReg.proba`), τ = 0.3, persistence p = 1, ``noun_ok`` required. The probe's ``p`` rides
  on the policy result (`task_probe`); `SafePolicy` turns it into the HELD gate (`task_not_ready`).

**One lookup, one prefix pass.** The target noun is the single fruit word of the prompt that went into the
policy (T55: "the noun of the prompt actually fed"; regex of `t48_hsr_core.prompt_template`); its token
position is `t49_noun_query_extract.lookup` on the tokenized prompt the policy's own input transform made
(the noun's piece exactly once among the valid tokens, ``basket`` likewise, text token 0 = ``<bos>`` —
else ``noun_ok = False``, never guessed). Both stages read that one row of the prefix pass:

* stage 1 gathers the row's image attention **inside** the jitted prefix pass (`prefix_noun_impl`, the
  pass of `AttentionSampler._prefix` + `t49_noun_query_core.slice_probs`, as `t49_noun_query_extract.
  prefix_rows_impl`);
* stage 2 gathers the row's V vector from the **KV cache of the same pass** (`v_rows_impl`, the gather of
  `t53_extract.noun_token_impl`) — with the action source that is the sampler's own `_jit_prefix` cache
  (`AttentionSampler._sample(kv_hook=...)`), so the served attention is the default path's, computed by the
  same executables.

A noun that is not found gives **no** synthetic attention (an empty dict — AG3S then runs without a target,
the safe direction, as `serve_safe.attention_extractor` documents) and ``noun_ok = False``; the reason is
recorded. Nothing here imports jax at module import (the jitted pieces live in `JaxPrefixBackend`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pathlib
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "AttentionSource", "DEFAULT_SOURCE", "JaxPrefixBackend", "NounProbe", "NounTokens",
    "T55_PROBE_MD5", "T55_PROBE_REL", "TASK_GATE_MODES", "TASK_PROBE_FEATURE", "TASK_PROBE_TAU",
    "TASK_PROBE_PERSIST", "cell_maps", "default_task_probe_path", "load_noun_probe", "noun_lookup",
    "parse_attention_source", "prefix_noun_impl", "prompt_noun", "v_rows_impl",
    "IFG_PREFIX_L8_KEY", "RESID_LAYER", "prefix_resid_loop", "resid_block_impl",
    # --- T90 — begin
    "HiddenProbeSource", "prefix_hidden_impl", "seed_map_source_ok",
    # --- T90 — end
]

#: gemma_2b (pi0.5's PaliGemma LLM): 18 layers, 8 query heads (multi-query: one kv head).
N_LAYERS = 18
N_HEADS = 8
#: Action-query poolings (`pi05_attention.AGGREGATIONS`).
ACTION_AGGS = ("mean", "first", "last")
QUERIES = ("action", "noun_target")

#: T55 prereg (`…/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55/T55.prereg.json`, sha256
#: ``8f62fc76…``) ``arms.noun``: feature, threshold, persistence. Fixed — not flags.
TASK_PROBE_FEATURE = "v5"
TASK_PROBE_TAU = 0.3
TASK_PROBE_PERSIST = 1
#: The prereg's probe file (under ``$AG3S_STORAGE_ROOT``) and its md5 (``arms.noun.probe_md5``).
T55_PROBE_REL = ("results/development/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55/"
                 "noun_probe.npz")
T55_PROBE_MD5 = "648e880d8da8f83df2258005e9060850"
#: `SafePolicy(task_gate=...)` / `serve_safe --task-gate` · `--task-gate-shadow`.
TASK_GATE_MODES = ("off", "on", "shadow")

_SPEC_ACTION = re.compile(r"^action:(?P<agg>[a-z]+):L(?P<layer>\d+)H(?P<head>\d+)$")
_SPEC_NOUN = re.compile(r"^noun_target:L(?P<layer>\d+)H(?P<head>\d+)$")


# ------------------------------------------------------------------------------ SPEC


@dataclasses.dataclass(frozen=True)
class AttentionSource:
    """One attention cell. ``action``: query = the action tokens of Euler step `denoise`, pooled with `agg`;
    ``noun_target``: query = the prompt's target-noun token row of the prefix pass (`agg` / `denoise` unused)."""

    query: str
    layer: int
    head: int
    agg: Optional[str] = None
    denoise: int = 0

    def __post_init__(self):
        if self.query not in QUERIES:
            raise ValueError(f"attention source query {self.query!r} is not one of {QUERIES}")
        if not (0 <= int(self.layer) < N_LAYERS and 0 <= int(self.head) < N_HEADS):
            raise ValueError(f"cell L{self.layer}H{self.head} is outside {N_LAYERS} layers x {N_HEADS} heads")
        if self.query == "action":
            if self.agg not in ACTION_AGGS:
                raise ValueError(f"action pooling {self.agg!r} is not one of {ACTION_AGGS}")
            if int(self.denoise) != 0:
                raise ValueError("only Euler step 0 is sampled for serving (denoise 0)")
        elif self.agg is not None:
            raise ValueError("a noun_target source has no pooling")

    @property
    def label(self) -> str:
        if self.query == "action":
            return f"action:{self.agg}:L{self.layer}H{self.head}"
        return f"noun_target:L{self.layer}H{self.head}"

    @property
    def is_noun(self) -> bool:
        return self.query == "noun_target"

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"spec": self.label, "query": self.query, "layer": int(self.layer),
                               "head": int(self.head)}
        if self.query == "action":
            out.update({"agg": self.agg, "denoise": int(self.denoise)})
        else:
            out.update({"row": "target", "sigma": 0.0, "k": 1})
        return out


#: The pre-T57 hard-coded cell (`step-01-attention.json` best; `attention_policy._AGG/_LAYER/_HEAD`).
DEFAULT_SOURCE = AttentionSource("action", 8, 2, agg="last", denoise=0)


# --- T90 — begin
_SPEC_HIDDEN_PROBE = re.compile(r"^hidden_probe(?::(?P<path>.+))?$")


@dataclasses.dataclass(frozen=True)
class HiddenProbeSource:
    """T90 — ``hidden_probe[:NPZ]``: the T86 hidden image-patch probe (`hidden_probe`) as the ATS target signal instead
    of a noun attention cell. Duck-types `AttentionSource` where serving reads it (``label`` · ``query`` · ``is_noun`` ·
    ``record()``). ``is_noun`` is True because the prompt's single target fruit word still decides **whether there is a
    target** (none → empty attention, as the noun source); the probe itself never reads the prompt text. ``path`` None =
    the shipped asset."""

    path: Optional[str] = None
    query: str = "hidden_probe"

    @property
    def label(self) -> str:
        return "hidden_probe"

    @property
    def is_noun(self) -> bool:
        return True

    @property
    def is_hidden_probe(self) -> bool:
        return True

    def asset_path(self) -> str:
        from AG3S.trajopt.hidden_probe import default_asset_path

        return str(default_asset_path()) if self.path is None else str(self.path)

    def record(self) -> dict[str, Any]:
        from AG3S.trajopt.hidden_probe import describe_asset

        out: dict[str, Any] = {"spec": self.label, "query": self.query}
        out.update(describe_asset(self.asset_path()))
        out["prompt_use"] = "hidden state only; the prompt's fruit word decides whether a target exists (noun lookup)"
        return out


def seed_map_source_ok(src: Any) -> bool:
    """T90 — whether `src` is a seed-map source the attention-surface extent / confidence gate accept: the IFG LOCK's
    ``noun_target:L10H7`` or the T90 hidden probe (which replaces that map). None → False."""
    if src is None:
        return False
    return src.label == "noun_target:L10H7" or getattr(src, "query", None) == "hidden_probe"
# --- T90 — end


def parse_attention_source(spec: str) -> AttentionSource:
    """``action:<mean|first|last>:L<l>H<h>`` or ``noun_target:L<l>H<h>``; anything else raises ValueError."""
    s = str(spec).strip()
    # --- T90 — begin
    m = _SPEC_HIDDEN_PROBE.match(s)
    if m:
        return HiddenProbeSource(path=m["path"])
    # --- T90 — end
    m = _SPEC_ACTION.match(s)
    if m:
        return AttentionSource("action", int(m["layer"]), int(m["head"]), agg=m["agg"], denoise=0)
    m = _SPEC_NOUN.match(s)
    if m:
        return AttentionSource("noun_target", int(m["layer"]), int(m["head"]))
    raise ValueError(f"unknown attention source {spec!r}: expected action:<{'|'.join(ACTION_AGGS)}>:L<l>H<h> "
                     "or noun_target:L<l>H<h> (e.g. action:last:L8H2 = the serving default, "
                     "noun_target:L10H7 = T57 N, action:mean:L10H7 = T61 selection)")


# ------------------------------------------------------------------------------ noun lookup


def prompt_noun(prompt: Optional[str]) -> tuple[Optional[str], str]:
    """``(fruit, "")`` when the prompt names exactly one fruit word (`t48_hsr_core.prompt_template`'s regex),
    else ``(None, why)``."""
    from AG3S.ag3s.experiments.studies.t49_noun_query_core import FRUITS

    if not isinstance(prompt, str) or not prompt.strip():
        return None, "no prompt"
    found = re.findall(r"\b(" + "|".join(FRUITS) + r")\b", prompt)
    if len(found) != 1:
        return None, f"prompt names {len(found)} fruit word(s) {found} (need exactly 1): {prompt!r}"
    return found[0], ""


@dataclasses.dataclass(frozen=True)
class NounTokens:
    """Piece ids the lookup needs: ``{fruit…, "basket": id}`` and ``<bos>`` (`t49_noun_query_extract.
    noun_ids_for` / ``sp.bos_id()``)."""

    ids: Mapping[str, int]
    bos: int

    @classmethod
    def from_config(cls, config_name: str) -> "NounTokens":
        """From the SentencePiece model the policy's `TokenizePrompt` uses. Raises when a noun is not one
        piece (`t49_noun_query_core.piece_ids`) — at server start, not on the first request."""
        from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
        from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

        sp = nx.sentencepiece_of(config_name)
        return cls(ids=dict(nx.noun_ids_for(sp, core.FRUITS)), bos=int(sp.bos_id()))


@dataclasses.dataclass(frozen=True)
class NounLookup:
    noun: Optional[str]
    ok: bool
    error: str
    rows: Optional[np.ndarray]          # (3,) int32 prefix rows, `QUERY_NAMES` order; None unless ok
    token_index: Optional[int]          # text index of the target noun; None unless ok
    n_text_valid: int

    def record(self) -> dict[str, Any]:
        return {"noun": self.noun, "noun_ok": bool(self.ok), "noun_error": self.error,
                "token_index": self.token_index,
                "prefix_row": None if self.rows is None else int(self.rows[0]),
                "n_text_valid": int(self.n_text_valid)}


def noun_lookup(inputs: Mapping[str, Any], prompt: Optional[str], tokens: NounTokens) -> NounLookup:
    """The target-noun row of one request: `prompt_noun`, then `t49_noun_query_extract.lookup` on the
    tokenized prompt of `inputs` (the policy's input transform output, no batch axis) — exactly the T53 /
    T55 ``feature_frame`` call (target, basket, last_text; any error -> not ok)."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    toks = np.asarray(inputs["tokenized_prompt"]).reshape(-1)
    mask = np.asarray(inputs["tokenized_prompt_mask"]).reshape(-1).astype(bool)
    n_valid = int(mask.sum())
    noun, why = prompt_noun(prompt)
    if noun is None:
        return NounLookup(None, False, why, None, None, n_valid)
    if noun not in tokens.ids or core.DESTINATION_NOUN not in tokens.ids:
        return NounLookup(noun, False, f"no piece id for {noun!r} / {core.DESTINATION_NOUN!r}", None, None,
                          n_valid)
    pos, err = nx.lookup(toks, mask, noun, dict(tokens.ids), int(tokens.bos))
    if err:
        return NounLookup(noun, False, "; ".join(err), None, None, n_valid)
    return NounLookup(noun, True, "", core.prefix_rows(pos), int(pos["target"]), n_valid)


# ------------------------------------------------------------------------------ maps


def cell_maps(cell: np.ndarray) -> dict[str, np.ndarray]:
    """``[768]`` image columns -> ``{policy camera: [16, 16]}`` — the loop of
    `AttentionPolicy._attention_and_feature` (`CAMERA_BINDINGS` slices)."""
    from AG3S.ag3s.experiments.sources.policy_record import (
        ATTENTION_GRID, CAMERA_BINDINGS, POLICY_CAMERA_NAMES)

    cell = np.asarray(cell)
    out = {}
    for camera in POLICY_CAMERA_NAMES:
        lo, hi = CAMERA_BINDINGS[camera][2]
        out[camera] = cell[lo:hi].reshape(ATTENTION_GRID, ATTENTION_GRID)
    return out


# ------------------------------------------------------------------------------ probe


def default_task_probe_path(storage_root: Optional[str] = None) -> pathlib.Path:
    """The T55 prereg probe under ``$AG3S_STORAGE_ROOT`` (or `storage_root`)."""
    root = storage_root or os.environ.get("AG3S_STORAGE_ROOT")
    if not root:
        from common.workspace import shared_storage_root

        root = str(shared_storage_root())
    return pathlib.Path(root) / T55_PROBE_REL


def file_md5(path: pathlib.Path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class NounProbe:
    """`t55_study.load_probe` (`t50_core.LogReg`) + the layer its feature reads."""

    model: Any
    feature: str
    layer: int
    path: str
    md5: str

    def p(self, v: np.ndarray) -> float:
        """p(task) of one 256-d feature row — `t55_study.score_test`: ``model.proba(X)[:, 1]``."""
        x = np.asarray(v, np.float64).reshape(1, -1)
        return float(self.model.proba(x)[0, 1])

    def record(self) -> dict[str, Any]:
        return {"feature": self.feature, "layer": int(self.layer), "path": self.path, "md5": self.md5,
                "tau": TASK_PROBE_TAU, "persist": TASK_PROBE_PERSIST}


def load_noun_probe(path, *, expect_md5: Optional[str] = T55_PROBE_MD5,
                    expect_feature: str = TASK_PROBE_FEATURE) -> NounProbe:
    """Load the noun probe (`t55_study.save_probe` format). Raises FileNotFoundError / ValueError when the
    file is missing, is not the pre-registered one (`expect_md5`; None skips the check) or reads another
    feature."""
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"task probe {path} does not exist")
    md5 = file_md5(path)
    if expect_md5 is not None and md5 != expect_md5:
        raise ValueError(f"task probe {path} has md5 {md5}, not the T55 prereg's {expect_md5}")
    with np.load(path, allow_pickle=False) as z:
        model = LogReg(mean=z["mean"], scale=z["scale"], coef=z["coef"], intercept=z["intercept"],
                       C=float(z["C"]), n_iter=int(z["n_iter"]), converged=bool(z["converged"]))
        feature = str(z["feature"])
    if feature != expect_feature:
        raise ValueError(f"task probe {path} reads feature {feature!r}, the gate is defined on {expect_feature!r}")
    m = re.fullmatch(r"v(\d+)", feature)
    if m is None:
        raise ValueError(f"task probe feature {feature!r} is not a V-cache layer (v<l>)")
    if np.asarray(model.coef).shape != (2, 256):
        raise ValueError(f"task probe coef shape {np.asarray(model.coef).shape}, expected (2, 256) binary")
    return NounProbe(model=model, feature=feature, layer=int(m.group(1)), path=str(path), md5=md5)


def feature_digest(v: np.ndarray) -> str:
    """Short sha1 of the float32 feature bytes — shows in the record that each request's p came from a
    newly gathered feature."""
    return hashlib.sha1(np.ascontiguousarray(np.asarray(v, np.float32)).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------ traced bodies


def prefix_noun_impl(model, obs, rows, *, layer: int, head: int, make_attn_mask, xp):
    """Prefix pass (as `AttentionSampler._prefix`: ``embed_prefix -> make_attn_mask -> positions = cumsum(mask)
    - 1 -> PaliGemma.llm([tokens, None])``) returning ``(kv, prefix_mask, rows' image attention at
    (layer, head) [R, 768] float32)``. The rows are sliced from the probabilities with
    `t49_noun_query_core.slice_probs` (T49's extraction), inside the trace. `obs` is preprocessed."""
    from AG3S.ag3s.experiments.studies.t49_noun_query_core import N_IMAGE_TOKENS, slice_probs

    tokens, mask, ar_mask = model.embed_prefix(obs)
    n_image = tokens.shape[1] - obs.tokenized_prompt.shape[1]
    if n_image != N_IMAGE_TOKENS:  # static shapes: checked at trace time
        raise ValueError(f"prefix has {n_image} image tokens, expected {N_IMAGE_TOKENS}")
    attn = make_attn_mask(mask, ar_mask)
    positions = xp.cumsum(mask, axis=1) - 1
    out = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
    if len(out) != 3:
        raise ValueError("PaliGemma.llm returned no attention probabilities — the attention model must be "
                         "loaded with return_attn_probs=True (pi05_attention.load_policy)")
    image = slice_probs(out[2], rows, n_image=N_IMAGE_TOKENS, xp=xp)["image"]  # [L, G, R, 768]
    return out[1], mask, image[int(layer), int(head)]


def llm_with_hidden(llm, *args, **kwargs):
    """T82 — ``llm(*args, **kwargs)`` with openpi's ``gemma.Module.return_hidden`` (T73, default False) turned on
    **for this call only**: the wrapped linen module is cloned with ``return_hidden=True`` and applied to the same
    variables (`ToNNX.__call__` minus the rng plumbing — the prefix pass is deterministic). The flag adds no parameter
    and changes no computation; the call returns the module's tuple plus the stacked expert-0 hidden states
    ``[depth + 1, B, T, D]`` last. The model object (and every other pass on it) keeps its own flags."""
    from flax.nnx.bridge import variables as bv

    variables = bv.nnx_attrs_to_linen_vars({name: getattr(llm, name) for name in llm.linen_attributes})
    return llm.module.clone(return_hidden=True).apply(variables, *args, **kwargs)


def prefix_noun_hidden_impl(model, obs, rows, *, layer: int, head: int, hidden_layer: int, make_attn_mask, xp):
    """T82 — `prefix_noun_impl` with the hidden state of `hidden_layer` from the **same** pass (`llm_with_hidden`).
    Returns ``(kv, prefix_mask, rows' image attention [R, 768] float32 or None when rows is None, h [B, T, D])``.
    `obs` is preprocessed."""
    from AG3S.ag3s.experiments.studies.t49_noun_query_core import N_IMAGE_TOKENS, slice_probs

    tokens, mask, ar_mask = model.embed_prefix(obs)
    n_image = tokens.shape[1] - obs.tokenized_prompt.shape[1]
    if n_image != N_IMAGE_TOKENS:  # static shapes: checked at trace time
        raise ValueError(f"prefix has {n_image} image tokens, expected {N_IMAGE_TOKENS}")
    attn = make_attn_mask(mask, ar_mask)
    positions = xp.cumsum(mask, axis=1) - 1
    out = llm_with_hidden(model.PaliGemma.llm, [tokens, None], mask=attn, positions=positions)
    image = None
    if rows is not None:
        if len(out) != 4:
            raise ValueError("PaliGemma.llm returned no attention probabilities — the attention model must be "
                             "loaded with return_attn_probs=True (pi05_attention.load_policy)")
        image = slice_probs(out[2], rows, n_image=N_IMAGE_TOKENS, xp=xp)["image"][int(layer), int(head)]
    return out[1], mask, image, out[-1][int(hidden_layer)]


# --- T90 — begin
def prefix_hidden_impl(model, obs, *, make_attn_mask, xp):
    """T90 — the prefix pass with the **whole** hidden stack (`llm_with_hidden`, same pass as `prefix_noun_impl` /
    `prefix_noun_hidden_impl`): ``(kv, prefix_mask, hidden [depth + 1, B, T, D])``. No attention row is sliced (the
    hidden probe does not read one), the attention copy's own ``return_attn_probs`` output is simply not used. `obs`
    is preprocessed."""
    from AG3S.ag3s.experiments.studies.t49_noun_query_core import N_IMAGE_TOKENS

    tokens, mask, ar_mask = model.embed_prefix(obs)
    n_image = tokens.shape[1] - obs.tokenized_prompt.shape[1]
    if n_image != N_IMAGE_TOKENS:  # static shapes: checked at trace time
        raise ValueError(f"prefix has {n_image} image tokens, expected {N_IMAGE_TOKENS}")
    attn = make_attn_mask(mask, ar_mask)
    positions = xp.cumsum(mask, axis=1) - 1
    out = llm_with_hidden(model.PaliGemma.llm, [tokens, None], mask=attn, positions=positions)
    return out[1], mask, out[-1]
# --- T90 — end


def v_rows_impl(kv, rows, *, layer: int, xp):
    """V cache ``[layer, batch 0, rows, kv head 0, :]`` -> ``[R, H]`` float32 — `t53_extract.noun_token_impl`'s
    ``v[layers][:, 0][:, rows][:, :, 0, :]`` for one layer."""
    v = kv[1]  # (L, B, S, K, H)
    return v[int(layer)][0][rows][:, 0, :].astype(xp.float32)


# ------------------------------------------------------------------------------ IFG-S2SERVE: prefix layer-8 residual

#: IFG-S2SERVE — the confidence gate's feature: the residual stream **after** prefix block `RESID_LAYER` (blocks
#: 0 … 8 applied), image rows 0–767 — the tensor the held-out collector stored as ``prefix_L8.npz resid_out_image``
#: (`…/20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/p/hc_dump.py`, `dump_step`).
RESID_LAYER = 8
#: `AttentionPolicy` result key that carries it (float32 (768, 2048), float16-rounded as the dump); `SafePolicy`
#: pops it before the response is packed (it never goes on the wire).
IFG_PREFIX_L8_KEY = "ifg_prefix_l8_image"


def resid_block_impl(block, layers, layer, x, positions, mask4):
    """One prefix block, the collector's ``layer`` jit: ``block.apply({"params": layers[l]}, [x, None], None, pos,
    mask4, [None, None], True)[0][0]``. `layers` = the stacked ``nnx.state(PaliGemma.llm).to_pure_dict()["layers"]``;
    the collector sliced layer `l` outside its jit, here the (exact) slice is inside so one executable serves all
    nine layers."""
    import jax

    pl = jax.tree.map(lambda a: a[layer], layers)
    ys, _ = block.apply({"params": pl}, [x, None], None, positions, mask4, [None, None], True)
    return ys[0]


def prefix_resid_loop(x, positions, mask4, layers, block_fn, *, n_layers: int = RESID_LAYER + 1):
    """``for l in range(9): x = layer(layer_p[l], x, pos, mask4)`` — the collector's loop (`block_fn(layers, l, x,
    positions, mask4)` = the jitted `resid_block_impl`). Returns the residual after block ``n_layers − 1``."""
    import jax.numpy as jnp

    for l in range(int(n_layers)):
        x = block_fn(layers, jnp.asarray(l, jnp.int32), x, positions, mask4)
    return x


# ------------------------------------------------------------------------------ backend (jax)


class JaxPrefixBackend:
    """The GPU side of the extended path: one prefix pass per request, sharing `AttentionSampler`'s model
    state and executables. Constructed only when a T57 flag is on (it builds two more jitted functions,
    compiled on first use)."""

    def __init__(self, sampler, *, noise_seed: int = 0):
        import jax

        self._sampler = sampler
        self._jax = jax
        self._noise_seed = int(noise_seed)
        self._jit_noun = jax.jit(self._noun_pass, static_argnames=("layer", "head"))
        self._jit_v = jax.jit(self._v_rows, static_argnames=("layer",))
        #: IFG-S2SERVE — built on first use (`_resid_setup`), only when a request asks for the layer-8 residual.
        self._resid = None
        #: T82 — the noun prefix pass that also returns the HSR5 feature; built on first use (`want_hsr5`) only.
        self._jit_hsr5 = None
        # --- T90 — begin: the hidden-probe pass (`set_hidden_probe` + first `hidden_probe` request)
        self._hidden_probe = None
        self._probe_dev = None
        self._jit_probe = None
        # --- T90 — end

    # --- T82: HSR5 feature from the same prefix pass ---------------------------------------------------
    def _hsr5_pass(self, state, obs, rows, *, layer, head):
        """`prefix_noun_hidden_impl` + the 8192-D `hsr5.hsr5_feature` of batch 0, pooled on the device (only the
        feature leaves it). ``rows`` None = the noun was not found: no attention row, kv and the feature only."""
        import jax.numpy as jnp

        from AG3S.trajopt.hsr5 import HIDDEN_LAYER, hsr5_feature
        from openpi.models.pi0 import make_attn_mask

        s = self._sampler
        model = s._nnx.merge(s._graphdef, state)  # noqa: SLF001
        kv, mask, att, h = prefix_noun_hidden_impl(model, obs, rows, layer=layer, head=head,
                                                   hidden_layer=HIDDEN_LAYER, make_attn_mask=make_attn_mask, xp=jnp)
        return kv, mask, att, hsr5_feature(jnp, h[0].astype(jnp.float32), mask[0])

    def hsr5_pass(self, obs, rows, *, layer: int, head: int):
        """Jitted `_hsr5_pass` on a **preprocessed** batched `obs` → ``(kv, mask, att or None, feature)``."""
        import jax.numpy as jnp

        if self._jit_hsr5 is None:
            self._jit_hsr5 = self._jax.jit(self._hsr5_pass, static_argnames=("layer", "head"))
        r = None if rows is None else jnp.asarray(np.asarray(rows, np.int32))
        return self._jit_hsr5(self._sampler._state, obs, r, layer=int(layer), head=int(head))  # noqa: SLF001

    # --- T90 — begin: the T86 hidden image-patch probe from the same prefix pass --------------------------
    def set_hidden_probe(self, probe) -> None:
        """T90 — the `hidden_probe.HiddenProbe` the ``hidden_probe`` source scores with (device arrays are built on the
        first request). Without it a ``hidden_probe`` request raises (`AttentionPolicy` then records the failure)."""
        self._hidden_probe = probe
        self._probe_dev = None
        self._jit_probe = None

    def _probe_pass(self, state, obs, dev, *, layers, ranges, want_hsr5):
        """`prefix_hidden_impl` + the per-camera probe logits ``[3, 256]`` (`hidden_probe.logits_of_stack`, float32 on the
        device) and, with `want_hsr5`, the 8192-D `hsr5.hsr5_feature` of the same stack. Only those leave the device."""
        import jax.numpy as jnp

        from AG3S.trajopt.hidden_probe import CAMERAS, logits_of_stack
        from openpi.models.pi0 import make_attn_mask

        s = self._sampler
        model = s._nnx.merge(s._graphdef, state)  # noqa: SLF001
        kv, mask, hidden = prefix_hidden_impl(model, obs, make_attn_mask=make_attn_mask, xp=jnp)
        logits = logits_of_stack(jnp, hidden, dev, dict(zip(CAMERAS, layers)), dict(zip(CAMERAS, ranges)))
        feat5 = None
        if want_hsr5:
            from AG3S.trajopt.hsr5 import HIDDEN_LAYER, hsr5_feature

            feat5 = hsr5_feature(jnp, hidden[HIDDEN_LAYER][0].astype(jnp.float32), mask[0])
        return kv, mask, logits, feat5

    def probe_pass(self, obs, *, want_hsr5: bool = False):
        """Jitted `_probe_pass` on a **preprocessed** batched `obs` → ``(kv, mask, logits [3, 256], hsr5 feature or
        None)``."""
        import jax.numpy as jnp

        from AG3S.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS
        from AG3S.trajopt.hidden_probe import CAMERAS

        probe = self._hidden_probe
        if probe is None:
            raise RuntimeError("hidden_probe source: no probe asset set on the backend (set_hidden_probe)")
        if self._probe_dev is None:
            self._probe_dev = {c: {"mean": jnp.asarray(probe.mean[c], jnp.float32),
                                   "scale": jnp.asarray(probe.scale[c], jnp.float32),
                                   "w": jnp.asarray(probe.w(c), jnp.float32), "b": jnp.float32(probe.b(c))}
                               for c in CAMERAS}
            self._jit_probe = self._jax.jit(self._probe_pass, static_argnames=("layers", "ranges", "want_hsr5"))
        layers = tuple(int(probe.layers[c]) for c in CAMERAS)
        ranges = tuple(tuple(int(x) for x in CAMERA_BINDINGS[c][2]) for c in CAMERAS)
        return self._jit_probe(self._sampler._state, obs, self._probe_dev, layers=layers,  # noqa: SLF001
                               ranges=ranges, want_hsr5=bool(want_hsr5))
    # --- T90 — end

    # --- IFG-S2SERVE: the collector's layer-8 residual pass -----------------------------------------
    def _resid_setup(self):
        """`(jit embed, jit block, stacked layer params)` — the collector's pieces (`hc_dump.py`): `Block(configs,
        return_probs=False)`, ``nnx.state(PaliGemma.llm).to_pure_dict()["layers"]`` of the sampler's model state
        (no copy), embed dtype of the LLM module. The served attention executables are not touched."""
        if self._resid is not None:
            return self._resid
        import jax
        import jax.numpy as jnp

        from openpi.models import gemma as G
        from openpi.models.pi0 import make_attn_mask

        s = self._sampler
        nnx = s._nnx  # noqa: SLF001
        graphdef, state = s._graphdef, s._state  # noqa: SLF001
        model = nnx.merge(graphdef, state)
        llm_mod = model.PaliGemma.llm.module
        layers = nnx.state(model.PaliGemma.llm).to_pure_dict()["layers"]
        block = G.Block(configs=tuple(llm_mod.configs), return_probs=False)
        emb_dt = jnp.dtype(llm_mod.embed_dtype)

        def embed(state_, obs):
            m = nnx.merge(graphdef, state_)
            tokens, mask, ar = m.embed_prefix(obs)
            mask4 = make_attn_mask(mask, ar)[:, None]
            pos = jnp.cumsum(mask, 1) - 1
            return tokens.astype(emb_dt), pos, mask4

        jit_block = jax.jit(lambda ly, l, x, pos, mask4: resid_block_impl(block, ly, l, x, pos, mask4))
        self._resid = (jax.jit(embed), jit_block, layers)
        return self._resid

    def resid_l8(self, obs) -> np.ndarray:
        """``x[0, :768]`` float32 after prefix blocks 0 … 8 for a **preprocessed** batched `obs` (the collector's
        ``dump_step``: embed_prefix → x = tokens.astype(embed dtype) → 9 × block → image rows)."""
        from AG3S.ag3s.experiments.studies.t49_noun_query_core import N_IMAGE_TOKENS

        jit_embed, jit_block, layers = self._resid_setup()
        x, pos, mask4 = jit_embed(self._sampler._state, obs)  # noqa: SLF001
        x = prefix_resid_loop(x, pos, mask4, layers, jit_block)
        return np.asarray(x[0, :N_IMAGE_TOKENS], np.float32)

    def _noun_pass(self, state, obs, rows, *, layer, head):
        import jax.numpy as jnp

        from openpi.models.pi0 import make_attn_mask

        s = self._sampler
        model = s._nnx.merge(s._graphdef, state)  # noqa: SLF001
        return prefix_noun_impl(model, obs, rows, layer=layer, head=head, make_attn_mask=make_attn_mask,
                                xp=jnp)

    @staticmethod
    def _v_rows(kv, rows, *, layer):
        import jax.numpy as jnp

        return v_rows_impl(kv, rows, layer=layer, xp=jnp)

    def run(self, inputs: Mapping[str, Any], *, source: AttentionSource, rows: Optional[np.ndarray],
            want_feature: bool, v_layer: Optional[int], want_resid: bool = False,
            want_hsr5: bool = False) -> dict[str, Any]:
        """`inputs` = the policy's input transform output (no batch axis). Returns ``{"cell": [768] or None,
        "feature": kv_L4 (512,) or None, "v": [R, 256] float32 or None, "path": str}``. ``v`` needs `rows`
        and `v_layer`; ``cell`` of a noun source needs `rows`. `want_resid` (IFG-S2SERVE, default False = no key):
        ``"resid_l8"`` = (768, 2048) float32 from `resid_l8` on the same preprocessed observation.
        `want_hsr5` (T82, default False = no key): ``"hsr5_feature"`` = (8192,) float32 from the **same** noun prefix
        pass (`hsr5_pass`, gemma `return_hidden` on for that pass only; kv · attention row from it too) and
        ``"hsr5_ms"``; on an action-query source there is no such pass: ``"hsr5_feature"`` None + ``"hsr5_error"``."""
        import jax.numpy as jnp

        from AG3S.ag3s.experiments.sources.pi05_attention import AGGREGATIONS
        from openpi.models import model as _model

        s = self._sampler
        batched = self._jax.tree.map(lambda x: jnp.asarray(x)[None, ...], dict(inputs))
        model_obs = _model.Observation.from_dict(batched)
        want_v = rows is not None and v_layer is not None
        out: dict[str, Any] = {"cell": None, "feature": None, "v": None}
        if source.query == "action":
            captured: dict[str, Any] = {}
            hook = (lambda kv, mask: captured.update(kv=kv)) if want_v else None
            if want_feature:
                blocks, feature = s.attention_and_subtask_feature(
                    model_obs, [source.denoise], noise_seed=self._noise_seed, kv_hook=hook)
            else:
                blocks = s.attention(model_obs, [source.denoise], noise_seed=self._noise_seed, kv_hook=hook)
                feature = None
            pooled = AGGREGATIONS[source.agg](blocks[source.denoise])  # [L, heads, 768]
            out["cell"] = np.asarray(pooled[source.layer, source.head])
            out["feature"] = feature
            kv = captured.get("kv")
            out["path"] = "sampler prefix + suffix (action query)"
            if want_resid:   # IFG-S2SERVE — the sampler preprocessed its own copy; this is a separate one
                out["resid_l8"] = self.resid_l8(_model.preprocess_observation(None, model_obs, train=False))
            if want_hsr5:    # T82 — no hidden on the sampler's own prefix pass (serve_safe refuses this combination)
                out["hsr5_feature"] = None
                out["hsr5_error"] = "action-query source: the HSR5 hidden feature rides on the noun prefix pass only"
        # --- T90 — begin: the hidden probe's own pass (hidden stack in the same prefix forward)
        elif source.query == "hidden_probe":
            import time as _time

            obs = _model.preprocess_observation(None, model_obs, train=False)  # exactly once
            t0 = _time.perf_counter()
            kv, mask, logits, feat5 = self.probe_pass(obs, want_hsr5=want_hsr5)
            logits = np.asarray(logits, np.float32)       # [3, 256] — blocks until the pass is done
            if want_hsr5:
                out["hsr5_feature"] = np.asarray(feat5, np.float32)
            out["probe_ms"] = round((_time.perf_counter() - t0) * 1000.0, 2)
            if want_hsr5:
                out["hsr5_ms"] = out["probe_ms"]
            out["probe_logits"] = logits
            if rows is None:     # the prompt names no single fruit: no target, as the noun source
                out["path"] = "prefix pass + hidden stack (noun not found: no target)"
            else:
                from AG3S.trajopt.hidden_probe import camera_probabilities

                out["cell"] = camera_probabilities(logits)
                out["path"] = "prefix pass + hidden stack (probe scores per patch)"
            if want_feature:
                out["feature"] = np.asarray(s._jit_kv_feature(kv, mask), np.float32)  # noqa: SLF001
            if want_resid:
                out["resid_l8"] = self.resid_l8(obs)
        # --- T90 — end
        else:
            obs = _model.preprocess_observation(None, model_obs, train=False)  # exactly once
            if want_hsr5:
                # T82 — the same prefix pass with gemma return_hidden on for it: kv, the noun row and h_4 together.
                import time as _time

                t0 = _time.perf_counter()
                kv, mask, att, feat5 = self.hsr5_pass(obs, rows, layer=int(source.layer), head=int(source.head))
                out["hsr5_feature"] = np.asarray(feat5, np.float32)
                out["hsr5_ms"] = round((_time.perf_counter() - t0) * 1000.0, 2)
                if rows is None:
                    out["path"] = "prefix pass + hidden h4 (noun not found)"
                else:
                    out["cell"] = np.asarray(att, np.float32)[0]  # the target row
                    out["path"] = "noun prefix pass + hidden h4 (rows gathered in the jit)"
            elif rows is None:
                kv, mask = s._jit_prefix(s._state, obs)  # noqa: SLF001 — no noun: the kv_L4 pass only
                out["path"] = "sampler prefix (noun not found)"
            else:
                kv, mask, att = self._jit_noun(s._state, obs, jnp.asarray(np.asarray(rows, np.int32)),  # noqa: SLF001
                                               layer=int(source.layer), head=int(source.head))
                out["cell"] = np.asarray(att, np.float32)[0]  # the target row
                out["path"] = "noun prefix pass (rows gathered in the jit)"
            if want_feature:
                out["feature"] = np.asarray(s._jit_kv_feature(kv, mask), np.float32)  # noqa: SLF001
            if want_resid:   # IFG-S2SERVE — the same preprocessed observation as the noun pass
                out["resid_l8"] = self.resid_l8(obs)
        if want_v and kv is not None:
            out["v"] = np.asarray(self._jit_v(kv, jnp.asarray(np.asarray(rows, np.int32)), layer=int(v_layer)),
                                  np.float32)
        return out


def log_source(source: Optional[AttentionSource], probe: Optional[NounProbe]) -> None:
    if source is not None:
        logging.info("attention source: %s (T57-A)", source.label)
    if probe is not None:
        logging.info("task probe: %s (feature %s, layer %d, md5 %s, τ %.2f, p %d)", probe.path, probe.feature,
                     probe.layer, probe.md5, TASK_PROBE_TAU, TASK_PROBE_PERSIST)
