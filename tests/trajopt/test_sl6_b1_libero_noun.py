"""SL6 B1 — LIBERO target noun for the noun-query attention / noun probe (`attention_source.noun_lookup`).

Pinned (SL6.task.md §3 B1 / VB1):
 1. default ``NounTokens`` (rule "fruit") = the T57-A lookup: RB-Y1 fruit / basket prompts give the rows and the
    record of `t49_noun_query_extract.lookup` (bit-identical), LIBERO prompts stay "no target" as before;
 2. rule "libero": the head word (last word before the first stop word) of "pick up the <target> ... and place it
    <prep> the <destination>" is the target, its **last** piece the row; the destination noun may occur twice;
 3. libero spatial 10 sentences + SafeLIBERO spatial sentences -> bowl / plate (stub tokenizer always; the real
    PaliGemma tokenizer when sentencepiece + the cached model exist, `.venv-openpi-live`);
 4. rule "libero" on RB-Y1 sentences gives the same rows as the fruit rule; uncovered sentences fall back to the fruit
    rule and record both reasons; misaligned / missing pieces -> not ok (never guessed).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from AG3S.trajopt.attention_source import (
    NounTokens,
    clean_prompt,
    libero_prompt_noun,
    noun_lookup,
    token_char_spans,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SP_MODEL = ROOT / ".cache/openpi/big_vision/paligemma_tokenizer.model"
BOS = 2
#: Real PaliGemma ids of the fruit / basket pieces (`tests/trajopt/test_t57a_attention_source.py`).
IDS = {"apple": 15491, "banana": 31985, "basket": 12220, "orange": 10436, "pear": 21338}
PAD = 64


class StubSP:
    """A word-level SentencePiece stand-in: ``▁word`` per word (``ramekin`` -> ``▁rame`` + ``kin``), ``.`` / ``,``
    split off; the fruit / basket pieces get their real ids."""

    SPLIT = {"ramekin": ["▁rame", "kin"], "compartment": ["▁compart", "ment"]}

    def __init__(self):
        self.piece_of = {BOS: "<bos>"}
        self.id_of = {"<bos>": BOS}
        for w, i in IDS.items():
            self._add("▁" + w, i)

    def _add(self, piece, pid=None):
        if piece in self.id_of:
            return self.id_of[piece]
        pid = pid if pid is not None else 1000 + len(self.id_of)
        self.id_of[piece], self.piece_of[pid] = pid, piece
        return pid

    def id_to_piece(self, i):
        return self.piece_of[int(i)]

    def encode(self, text: str) -> list[int]:
        out = []
        for m in re.finditer(r" ?[A-Za-z]+| ?[.,;:]|\n", text):
            tok = m.group(0)
            word = tok.strip()
            lead = tok.startswith(" ")
            if lead and word.lower() in self.SPLIT and word.islower():
                pieces = self.SPLIT[word]
            else:
                pieces = [("▁" if lead else "") + word if word else tok]
            out += [self._add(p) for p in pieces]
        return out


def tokenize(sp, prompt: str, *, pi05: bool = False, pad: int = PAD) -> dict:
    """`PaligemmaTokenizer.tokenize` layout (pi0: <bos> + prompt + "\\n"; pi05: "Task: …, State: …;\\nAction: ")."""
    c = clean_prompt(prompt)
    if pi05:
        ids = [BOS] + sp.encode(f"Task: {c}, State: 1 2;\nAction: ")
    else:
        ids = [BOS] + sp.encode(c) + sp.encode("\n")
    toks = np.zeros(pad, np.int32)
    toks[:len(ids)] = ids
    mask = np.zeros(pad, bool)
    mask[:len(ids)] = True
    return {"tokenized_prompt": toks, "tokenized_prompt_mask": mask}


def libero_spatial_prompts() -> list[str]:
    from benchmark.adapters.libero_panda import task_target as tt

    return [n.replace("_", " ") for n in tt.suite_task_names("libero_spatial")]


def safelibero_spatial_prompts() -> list[str]:
    d = ROOT / "benchmark/SafeLIBERO/safelibero/libero/libero/bddl_files/safelibero_spatial"
    return sorted({p.stem.replace("_", " ") for p in d.glob("*.bddl")})


RBY1 = ["Pick up the apple and place it in the basket.", "Pick up the pear and place it in the basket.",
        "pick up the banana and place it in the basket", "Put the apple in the basket.", "Put the orange in the basket.",
        "put the pear in the basket"]


# =============================================================================================
# 1. default = T57-A (RB-Y1 bit-identical)
# =============================================================================================


def _legacy(inputs, prompt, tokens):
    """The pre-SL6 `noun_lookup` body (T57-A), restated for the comparison."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
    from AG3S.trajopt.attention_source import prompt_noun

    toks = np.asarray(inputs["tokenized_prompt"]).reshape(-1)
    mask = np.asarray(inputs["tokenized_prompt_mask"]).reshape(-1).astype(bool)
    n_valid = int(mask.sum())
    noun, why = prompt_noun(prompt)
    if noun is None:
        return {"noun": None, "noun_ok": False, "noun_error": why, "token_index": None, "prefix_row": None,
                "n_text_valid": n_valid}, None
    pos, err = nx.lookup(toks, mask, noun, dict(tokens.ids), int(tokens.bos))
    if err:
        return {"noun": noun, "noun_ok": False, "noun_error": "; ".join(err), "token_index": None,
                "prefix_row": None, "n_text_valid": n_valid}, None
    rows = core.prefix_rows(pos)
    return {"noun": noun, "noun_ok": True, "noun_error": "", "token_index": int(pos["target"]),
            "prefix_row": int(rows[0]), "n_text_valid": n_valid}, rows


@pytest.mark.parametrize("pi05", [False, True])
def test_default_rule_is_the_t57_lookup_on_rby1_and_libero_prompts(pi05):
    sp = StubSP()
    tokens = NounTokens(ids=IDS, bos=BOS)
    assert tokens.rule == "fruit" and tokens.sp is None
    for prompt in RBY1 + libero_spatial_prompts() + ["pick up the crate and place it in the basket", ""]:
        inputs = tokenize(sp, prompt, pi05=pi05)
        look = noun_lookup(inputs, prompt, tokens)
        rec, rows = _legacy(inputs, prompt, tokens)
        assert look.record() == rec, prompt
        assert (look.rows is None and rows is None) or np.array_equal(look.rows, rows)
        assert set(look.record()) == {"noun", "noun_ok", "noun_error", "token_index", "prefix_row", "n_text_valid"}
    # LIBERO sentences: no fruit -> no target (today's behaviour, unchanged by default)
    assert not noun_lookup(tokenize(sp, libero_spatial_prompts()[0]), libero_spatial_prompts()[0], tokens).ok


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="noun rule"):
        NounTokens(ids=IDS, bos=BOS, rule="bowl")


# =============================================================================================
# 2. the rule
# =============================================================================================


@pytest.mark.parametrize("prompt,target,dest", [
    ("pick up the black bowl between the plate and the ramekin and place it on the plate", "bowl", "plate"),
    ("pick up the black bowl next to the ramekin and place it on the plate", "bowl", "plate"),
    ("pick up the black bowl from table center and place it on the plate", "bowl", "plate"),
    ("pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate", "bowl", "plate"),
    ("pick up the orange juice and place it in the basket", "juice", "basket"),
    ("pick up the cream cheese and place it in the basket", "cheese", "basket"),
    ("pick up the book and place it in the back compartment of the caddy", "book", "compartment"),
    ("pick up the ramekin and place it on the plate", "ramekin", "plate"),
    ("Pick up the pear and place it in the basket.", "pear", "basket"),
])
def test_head_word_of_the_target_and_destination_phrases(prompt, target, dest):
    p = libero_prompt_noun(prompt)
    assert p.ok and p.target == target and p.destination == dest
    text = clean_prompt(prompt)
    assert text[p.target_span[0]:p.target_span[1]] == target
    assert text[p.destination_span[0]:p.destination_span[1]] == dest


@pytest.mark.parametrize("prompt", ["put the bowl on the plate", "open the middle drawer of the cabinet",
                                    "pick up the and place it on the plate", "Put the apple in the basket.", None, ""])
def test_sentences_outside_the_rule(prompt):
    p = libero_prompt_noun(prompt)
    assert not p.ok and p.error


def test_target_row_is_the_last_piece_and_a_repeated_destination_is_harmless():
    sp = StubSP()
    tokens = NounTokens(ids=IDS, bos=BOS, rule="libero", sp=sp)
    prompt = "pick up the ramekin and place it on the plate"
    inputs = tokenize(sp, prompt)
    look = noun_lookup(inputs, prompt, tokens)
    pieces = [sp.id_to_piece(int(i)) for i in inputs["tokenized_prompt"][:int(inputs["tokenized_prompt_mask"].sum())]]
    assert look.ok and look.noun == "ramekin"
    assert look.record()["noun_pieces"]["target"] == ["▁rame", "kin"]
    assert pieces[look.token_index] == "kin" and look.rows[0] == 768 + look.token_index
    # "plate" twice: target row is the bowl, destination row is the plate of the destination phrase (the last one)
    prompt = "pick up the black bowl next to the plate and place it on the plate"
    inputs = tokenize(sp, prompt)
    look = noun_lookup(inputs, prompt, tokens)
    pieces = [sp.id_to_piece(int(i)) for i in inputs["tokenized_prompt"][:int(inputs["tokenized_prompt_mask"].sum())]]
    assert look.ok and pieces[look.token_index] == "▁bowl"
    plates = [i for i, p in enumerate(pieces) if p == "▁plate"]
    assert len(plates) == 2 and look.rows[1] == 768 + plates[1] and look.rows[2] == 768 + len(pieces) - 1
    rec = look.record()
    assert rec["noun_rule"] == "libero" and rec["destination_noun"] == "plate" and rec["destination_row"] == 768 + plates[1]


@pytest.mark.parametrize("pi05", [False, True])
def test_libero_spatial_and_safelibero_sentences_stub(pi05):
    sp = StubSP()
    tokens = NounTokens(ids=IDS, bos=BOS, rule="libero", sp=sp)
    prompts = libero_spatial_prompts() + safelibero_spatial_prompts()
    assert len(libero_spatial_prompts()) == 10 and len(safelibero_spatial_prompts()) == 4
    for prompt in prompts:
        inputs = tokenize(sp, prompt, pi05=pi05)
        look = noun_lookup(inputs, prompt, tokens)
        assert look.ok, (prompt, look.error)
        assert look.noun == "bowl" and look.destination == "plate"
        assert sp.id_to_piece(int(inputs["tokenized_prompt"][look.token_index])) == "▁bowl"


@pytest.mark.parametrize("pi05", [False, True])
def test_libero_rule_on_rby1_sentences_gives_the_fruit_rows(pi05):
    sp = StubSP()
    fruit = NounTokens(ids=IDS, bos=BOS)
    lib = NounTokens(ids=IDS, bos=BOS, rule="libero", sp=sp)
    for prompt in RBY1:
        inputs = tokenize(sp, prompt, pi05=pi05)
        a, b = noun_lookup(inputs, prompt, fruit), noun_lookup(inputs, prompt, lib)
        assert a.ok and b.ok and a.noun == b.noun and np.array_equal(a.rows, b.rows), prompt
        assert b.token_index == a.token_index
    # "Put the apple in the basket." is outside the B1 rule: fruit rule, recorded as the fallback
    look = noun_lookup(tokenize(sp, RBY1[3]), RBY1[3], lib)
    assert look.record()["noun_rule"] == "libero:fruit"


def test_fallback_records_both_reasons_and_failures_are_not_guessed():
    sp = StubSP()
    lib = NounTokens(ids=IDS, bos=BOS, rule="libero", sp=sp)
    look = noun_lookup(tokenize(sp, "put the bowl on the plate"), "put the bowl on the plate", lib)
    assert not look.ok and "libero rule" in look.error and "fruit word" in look.error and look.rows is None
    # the prompt says bowl, the tokens are another sentence -> prompt text not found
    prompt = "pick up the black bowl on the stove and place it on the plate"
    look = noun_lookup(tokenize(sp, "pick up the black bowl on the ramekin and place it on the plate"), prompt, lib)
    assert not look.ok and "not found" in look.error
    # no tokenizer
    look = noun_lookup(tokenize(sp, prompt), prompt, NounTokens(ids=IDS, bos=BOS, rule="libero"))
    assert not look.ok and "tokenizer" in look.error
    # a piece that spans the word boundary ("▁bowlon") -> not aligned
    class Glue(StubSP):
        def encode(self, text):
            return super().encode(text.replace("bowl on", "bowlon"))
    g = Glue()
    look = noun_lookup(tokenize(g, prompt), prompt, NounTokens(ids=IDS, bos=BOS, rule="libero", sp=g))
    assert not look.ok


def test_token_char_spans_decode_pieces():
    sp = StubSP()
    inputs = tokenize(sp, "pick up the ramekin and place it on the plate")
    text, spans = token_char_spans(inputs["tokenized_prompt"], inputs["tokenized_prompt_mask"], sp)
    assert text == "pick up the ramekin and place it on the plate\n" and spans[0] == (0, 0)
    assert [text[a:b] for a, b in spans[1:4]] == ["pick", " up", " the"]


# =============================================================================================
# 3. the real PaliGemma tokenizer (`.venv-openpi-live`)
# =============================================================================================


def _real_sp():
    spm = pytest.importorskip("sentencepiece")
    if not SP_MODEL.is_file():
        pytest.skip(f"{SP_MODEL} not cached")
    return spm.SentencePieceProcessor(model_file=str(SP_MODEL))


class _RealSP:
    def __init__(self, sp):
        self.sp = sp

    def id_to_piece(self, i):
        return self.sp.id_to_piece(int(i))

    def encode(self, text):
        return self.sp.encode(text)


def _real_tokenize(sp, prompt, pi05):
    c = clean_prompt(prompt)
    ids = (sp.encode(f"Task: {c}, State: 1 2;\nAction: ", add_bos=True) if pi05
           else sp.encode(c, add_bos=True) + sp.encode("\n"))
    toks = np.zeros(200, np.int32)
    toks[:len(ids)] = ids
    mask = np.zeros(200, bool)
    mask[:len(ids)] = True
    return {"tokenized_prompt": toks, "tokenized_prompt_mask": mask}


@pytest.mark.parametrize("pi05", [False, True])
def test_real_tokenizer_libero_safelibero_and_rby1(pi05):
    sp = _real_sp()
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    ids = dict(nx.noun_ids_for(sp, core.FRUITS))
    assert ids == IDS
    lib = NounTokens(ids=ids, bos=int(sp.bos_id()), rule="libero", sp=sp)
    fruit = NounTokens(ids=ids, bos=int(sp.bos_id()))
    for prompt in libero_spatial_prompts() + safelibero_spatial_prompts():
        inputs = _real_tokenize(sp, prompt, pi05)
        look = noun_lookup(inputs, prompt, lib)
        assert look.ok and look.noun == "bowl" and look.record()["noun_pieces"] == {"target": ["▁bowl"],
                                                                                   "destination": ["▁plate"]}, prompt
        pieces = [sp.id_to_piece(int(t)) for t in inputs["tokenized_prompt"][:int(inputs["tokenized_prompt_mask"].sum())]]
        assert pieces[look.token_index] == "▁bowl" and pieces.count("▁bowl") == 1
        assert pieces[look.rows[1] - 768] == "▁plate"
        assert not noun_lookup(inputs, prompt, fruit).ok
    for prompt in RBY1:
        inputs = _real_tokenize(sp, prompt, pi05)
        a, b = noun_lookup(inputs, prompt, fruit), noun_lookup(inputs, prompt, lib)
        rec, rows = _legacy(inputs, prompt, fruit)
        assert a.record() == rec and np.array_equal(a.rows, rows)
        assert b.ok and np.array_equal(a.rows, b.rows)
    # a multi-piece head word: ramekin -> "▁rame" + "kin", the row is "kin"
    prompt = "pick up the ramekin and place it on the plate"
    inputs = _real_tokenize(sp, prompt, pi05)
    look = noun_lookup(inputs, prompt, lib)
    assert look.ok and look.record()["noun_pieces"]["target"] == ["▁rame", "kin"]
    assert sp.id_to_piece(int(inputs["tokenized_prompt"][look.token_index])) == "kin"
