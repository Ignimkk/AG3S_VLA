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
    "LiberoNoun", "NOUN_RULES", "PHRASE_STOP_WORDS", "clean_prompt", "libero_prompt_noun", "token_char_spans",
    "IFG_PREFIX_L8_KEY", "RESID_LAYER", "prefix_resid_loop", "resid_block_impl",
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


def parse_attention_source(spec: str) -> AttentionSource:
    """``action:<mean|first|last>:L<l>H<h>`` or ``noun_target:L<l>H<h>``; anything else raises ValueError."""
    s = str(spec).strip()
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


# ------------------------------------------------------------------------------ SL6 B1: LIBERO target noun
#
# Rule (SL6.task.md §3 B1, fixed by the lead): in ``"pick up the <target phrase> ... and place it <prep> the
# <destination phrase>"`` the **head word** (last noun) of the target phrase is the target noun and the **last
# tokenizer piece** of that word is the query row. Operational reading (no POS tagger in the server):
#   * the sentence must match `_LIBERO_PICK_PLACE` (case-insensitive; the tokenizer's own cleaning first);
#   * the target phrase = the words after "pick up the" up to the first word of `PHRASE_STOP_WORDS` (the
#     "..." of the rule: "between the plate …", "next to …", "on the stove", "in the top drawer …", "from table
#     center"); its head word = the last word of that span; the destination head word is found the same way;
#   * the head word is located **by character span** in the tokenized prompt (pieces decoded, ``▁`` = space),
#     so a destination noun that occurs twice ("… the plate … on the plate") cannot be taken for the target;
#   * the word must start and end on piece boundaries; its last piece's text index -> prefix row ``768 + i``.
# A sentence the rule does not cover keeps today's rule (fruit + basket) with both reasons recorded; no target
# is ever guessed. Only `NounTokens(rule="libero")` reaches this code (default ``"fruit"`` = the T57 lookup
# unchanged — RB-Y1 bit-identical).

#: `NounTokens.rule` values. ``fruit`` = T57-A (default); ``libero`` = SL6 B1 rule first, then the fruit rule.
NOUN_RULES = ("fruit", "libero")
_LIBERO_PICK_PLACE = re.compile(
    r"^\s*pick up the (?P<target>.+?) and place it (?P<prep>[a-z]+(?: [a-z]+)*?) the (?P<dest>.+?)\s*[.!]?\s*$",
    re.IGNORECASE)
#: Words that end a noun phrase (prepositions / relative words). The head word is the word before the first one.
PHRASE_STOP_WORDS = frozenset({
    "between", "next", "from", "on", "in", "inside", "into", "onto", "at", "near", "of", "under", "beside",
    "behind", "above", "below", "over", "to", "with", "by", "and", "that", "which"})
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


@dataclasses.dataclass(frozen=True)
class LiberoNoun:
    """The B1 parse of one prompt (character offsets into the cleaned prompt)."""

    target: Optional[str]
    target_span: Optional[tuple[int, int]]
    destination: Optional[str]
    destination_span: Optional[tuple[int, int]]
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.target is not None and not self.error


def clean_prompt(prompt: str) -> str:
    """`PaligemmaTokenizer.tokenize`'s cleaning (`third_party/openpi/src/openpi/models/tokenizer.py:23`)."""
    return prompt.strip().replace("_", " ").replace("\n", " ")


def _phrase_head(text: str, start: int, end: int) -> tuple[Optional[str], Optional[tuple[int, int]], str]:
    """Head word of the phrase ``text[start:end]``: the last word before the first stop word."""
    words = [(m.group(0), start + m.start(), start + m.end()) for m in _WORD.finditer(text[start:end])]
    if not words:
        return None, None, "empty phrase"
    cut = next((i for i, (w, _, _) in enumerate(words) if w.lower() in PHRASE_STOP_WORDS), len(words))
    if cut == 0:
        return None, None, f"phrase {text[start:end]!r} starts with the stop word {words[0][0]!r}"
    w, a, b = words[cut - 1]
    return w, (a, b), ""


def libero_prompt_noun(prompt: Optional[str]) -> LiberoNoun:
    """SL6 B1 — target / destination head words of a LIBERO pick-and-place prompt (spans in `clean_prompt`)."""
    if not isinstance(prompt, str) or not prompt.strip():
        return LiberoNoun(None, None, None, None, "no prompt")
    text = clean_prompt(prompt)
    m = _LIBERO_PICK_PLACE.match(text)
    if m is None:
        return LiberoNoun(None, None, None, None,
                          f"prompt is not 'pick up the <target> ... and place it <prep> the <destination>': {prompt!r}")
    tw, tspan, terr = _phrase_head(text, m.start("target"), m.end("target"))
    dw, dspan, derr = _phrase_head(text, m.start("dest"), m.end("dest"))
    if terr:
        return LiberoNoun(None, None, dw, dspan, f"target phrase: {terr}")
    if derr:
        return LiberoNoun(tw, tspan, None, None, f"destination phrase: {derr}")
    return LiberoNoun(tw, tspan, dw, dspan, "")


def _piece_text(piece: str) -> str:
    """SentencePiece piece -> the text it stands for (``▁`` -> space, byte pieces ``<0xNN>`` -> the byte)."""
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
    if m:
        return chr(int(m.group(1), 16))
    return piece.replace("▁", " ")


def token_char_spans(token_ids: np.ndarray, mask: np.ndarray, sp) -> tuple[str, list[tuple[int, int]]]:
    """Decoded text of the valid tokens after ``<bos>`` and each token's ``[start, end)`` in it (index 0 =
    ``<bos>`` gets ``(0, 0)``)."""
    ids = np.asarray(token_ids).reshape(-1)
    valid = np.flatnonzero(np.asarray(mask, bool).reshape(-1))
    spans, parts, pos = [], [], 0
    for k, i in enumerate(valid):
        if k == 0:
            spans.append((0, 0))
            continue
        t = _piece_text(str(sp.id_to_piece(int(ids[i]))))
        parts.append(t)
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return "".join(parts), spans


def _word_tokens(spans: Sequence[tuple[int, int]], text: str, a: int, b: int) -> tuple[Optional[list[int]], str]:
    """Token indices covering the word ``text[a:b]`` exactly (first token may carry the leading space)."""
    over = [i for i, (s, e) in enumerate(spans) if e > a and s < b and e > s]
    if not over:
        return None, "no token covers the word"
    s0 = spans[over[0]][0]
    if not (s0 == a or (s0 == a - 1 and text[s0] == " ")):
        return None, f"word starts inside token {over[0]} ({spans[over[0]]} vs word {a})"
    if spans[over[-1]][1] != b:
        return None, f"word ends inside token {over[-1]} ({spans[over[-1]]} vs word end {b})"
    return over, ""


def _libero_lookup(toks: np.ndarray, mask: np.ndarray, prompt: str, tokens: "NounTokens",
                   parsed: LiberoNoun, n_valid: int) -> "NounLookup":
    """B1 rows ``[target last piece, destination last piece, last valid text]`` (`QUERY_NAMES` order)."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

    def bad(why: str) -> NounLookup:
        return NounLookup(parsed.target, False, why, None, None, n_valid, rule="libero",
                          destination=parsed.destination)

    if tokens.sp is None:
        return bad("libero noun rule needs the tokenizer (NounTokens.sp)")
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return bad("no valid text token")
    if not np.array_equal(valid, np.arange(valid.size)):
        return bad("valid text tokens are not a prefix of the sequence")
    if int(toks[0]) != int(tokens.bos):
        return bad(f"text token 0 is {int(toks[0])}, not <bos> {int(tokens.bos)}")
    text, spans = token_char_spans(toks, mask, tokens.sp)
    cleaned = clean_prompt(prompt)
    base = text.find(cleaned)
    if base < 0:
        base = text.find(cleaned.lower())
    if base < 0:
        return bad(f"prompt text not found in the decoded tokens: {text!r}")
    rows_idx = []
    pieces = {}
    for role, span in (("target", parsed.target_span), ("destination", parsed.destination_span)):
        idx, err = _word_tokens(spans, text, base + span[0], base + span[1])
        if err:
            return bad(f"{role} {cleaned[span[0]:span[1]]!r}: {err}")
        rows_idx.append(int(idx[-1]))
        pieces[role] = [str(tokens.sp.id_to_piece(int(toks[i]))) for i in idx]
    pos = {"target": rows_idx[0], "basket": rows_idx[1], "last_text": int(valid[-1])}
    return NounLookup(parsed.target, True, "", core.prefix_rows(pos), int(pos["target"]), n_valid, rule="libero",
                      destination=parsed.destination, pieces=pieces)


@dataclasses.dataclass(frozen=True)
class NounTokens:
    """Piece ids the lookup needs: ``{fruit…, "basket": id}`` and ``<bos>`` (`t49_noun_query_extract.
    noun_ids_for` / ``sp.bos_id()``). SL6 B1: ``rule="libero"`` (default ``"fruit"`` = T57-A unchanged) also
    holds the SentencePiece processor `sp` (anything with ``id_to_piece``) for the LIBERO head-word rule."""

    ids: Mapping[str, int]
    bos: int
    rule: str = "fruit"
    sp: Any = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.rule not in NOUN_RULES:
            raise ValueError(f"noun rule {self.rule!r} is not one of {NOUN_RULES}")

    @classmethod
    def from_config(cls, config_name: str, *, rule: str = "fruit") -> "NounTokens":
        """From the SentencePiece model the policy's `TokenizePrompt` uses. Raises when a noun is not one
        piece (`t49_noun_query_core.piece_ids`) — at server start, not on the first request. ``rule="libero"``
        (SL6 B1, Panda) keeps the processor for the head-word rule; the default builds exactly the T57 object."""
        from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
        from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

        sp = nx.sentencepiece_of(config_name)
        ids = dict(nx.noun_ids_for(sp, core.FRUITS))
        if rule == "fruit":
            return cls(ids=ids, bos=int(sp.bos_id()))
        return cls(ids=ids, bos=int(sp.bos_id()), rule=rule, sp=sp)


@dataclasses.dataclass(frozen=True)
class NounLookup:
    noun: Optional[str]
    ok: bool
    error: str
    rows: Optional[np.ndarray]          # (3,) int32 prefix rows, `QUERY_NAMES` order; None unless ok
    token_index: Optional[int]          # text index of the target noun; None unless ok
    n_text_valid: int
    #: SL6 B1 — which rule produced this lookup; only ``"libero"`` adds keys to `record` (fruit = T57 record).
    rule: str = "fruit"
    destination: Optional[str] = None
    pieces: Optional[Mapping[str, Sequence[str]]] = None

    def record(self) -> dict[str, Any]:
        out = {"noun": self.noun, "noun_ok": bool(self.ok), "noun_error": self.error,
               "token_index": self.token_index,
               "prefix_row": None if self.rows is None else int(self.rows[0]),
               "n_text_valid": int(self.n_text_valid)}
        if self.rule != "fruit":
            out.update({"noun_rule": self.rule, "destination_noun": self.destination,
                        "destination_row": None if self.rows is None else int(self.rows[1]),
                        "noun_pieces": None if self.pieces is None else {k: list(v) for k, v in self.pieces.items()}})
        return out


def noun_lookup(inputs: Mapping[str, Any], prompt: Optional[str], tokens: NounTokens) -> NounLookup:
    """The target-noun row of one request: `prompt_noun`, then `t49_noun_query_extract.lookup` on the
    tokenized prompt of `inputs` (the policy's input transform output, no batch axis) — exactly the T53 /
    T55 ``feature_frame`` call (target, basket, last_text; any error -> not ok). With ``tokens.rule ==
    "libero"`` (SL6 B1) a prompt the LIBERO pick-and-place rule covers is looked up by that rule first."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    toks = np.asarray(inputs["tokenized_prompt"]).reshape(-1)
    mask = np.asarray(inputs["tokenized_prompt_mask"]).reshape(-1).astype(bool)
    n_valid = int(mask.sum())
    libero_why = ""
    if getattr(tokens, "rule", "fruit") == "libero":
        parsed = libero_prompt_noun(prompt)
        if parsed.ok:
            return _libero_lookup(toks, mask, str(prompt), tokens, parsed, n_valid)
        libero_why = f"libero rule: {parsed.error}; "
    # libero mode, sentence outside the B1 rule: the fruit rule, recorded as such (``noun_rule`` "libero:fruit")
    kw = {"rule": "libero:fruit"} if libero_why else {}
    noun, why = prompt_noun(prompt)
    if noun is None:
        return NounLookup(None, False, libero_why + why, None, None, n_valid, **kw)
    if noun not in tokens.ids or core.DESTINATION_NOUN not in tokens.ids:
        return NounLookup(noun, False, f"no piece id for {noun!r} / {core.DESTINATION_NOUN!r}", None, None,
                          n_valid, **kw)
    pos, err = nx.lookup(toks, mask, noun, dict(tokens.ids), int(tokens.bos))
    if err:
        return NounLookup(noun, False, libero_why + "; ".join(err), None, None, n_valid, **kw)
    return NounLookup(noun, True, "", core.prefix_rows(pos), int(pos["target"]), n_valid, **kw)


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
            want_feature: bool, v_layer: Optional[int], want_resid: bool = False) -> dict[str, Any]:
        """`inputs` = the policy's input transform output (no batch axis). Returns ``{"cell": [768] or None,
        "feature": kv_L4 (512,) or None, "v": [R, 256] float32 or None, "path": str}``. ``v`` needs `rows`
        and `v_layer`; ``cell`` of a noun source needs `rows`. `want_resid` (IFG-S2SERVE, default False = no key):
        ``"resid_l8"`` = (768, 2048) float32 from `resid_l8` on the same preprocessed observation."""
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
        else:
            obs = _model.preprocess_observation(None, model_obs, train=False)  # exactly once
            if rows is None:
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
