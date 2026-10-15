"""T57-A — serving: attention source SPEC (stage 1, `noun_target:L10H7` = T57 N) and the HELD task gate
(stage 2, T55 noun probe). Everything here is CPU (no jax, no checkpoint): the jitted bodies are run with
numpy stand-ins (as `tests/ag3s/test_t53_extract.py` does), `AttentionPolicy` gets a fake backend.

Pinned:
 1. SPEC grammar: default / N / T61 labels; unknown SPECs, cells outside 18 x 8, bad pooling refused.
 2. Noun lookup = `t49_noun_query_extract.lookup` as T53 / T55 `feature_frame` calls it (same rows);
    the prompt noun is its single fruit word; 0 / 2 fruits, a missing / duplicated token → not ok.
 3. Traced bodies: `prefix_noun_impl` gathers `slice_probs` rows (T49 `prefix_rows_impl`'s image at the
    cell) and returns the pass's kv; `v_rows_impl` = `t53_extract.noun_token_impl`'s V for one layer.
 4. Probe: the T55 prereg file loads (md5, v5, layer 5) and its p equals `t55_study`'s offline p on
    stored T55 test features; a wrong md5 / feature / missing file is refused.
 5. `AttentionPolicy`: no flag → the pre-T57 branch, no new keys; noun source → the target row's map;
    noun not found → empty attention + reason; probe → `task_probe` {p, ok, noun_ok, feature_sha1};
    failures fail closed (ok False).
 6. `SafePolicy` gate: exactly the user rule (HELD ∧ ¬probe_pass → `task_not_ready`), not HELD → no
    reason, other reasons untouched, shadow records only, off = no key; refusals.
 7. Consecutive HOLD requests recompute the probe from each new observation; the reason goes away
    exactly on the request whose probe passes (end-to-end through `SafePolicy.infer`).
 8. serve_safe flags: defaults off, wiring helpers, rejections.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.attention_source import (
    DEFAULT_SOURCE,
    NounProbe,
    NounTokens,
    T55_PROBE_MD5,
    TASK_PROBE_TAU,
    default_task_probe_path,
    load_noun_probe,
    noun_lookup,
    parse_attention_source,
    prefix_noun_impl,
    prompt_noun,
    v_rows_impl,
)
from AG3S.trajopt.grasp_latch import GraspPhase

#: Real PaliGemma piece ids (`tests/ag3s/test_t53_extract.py`).
IDS = {"apple": 15491, "banana": 31985, "basket": 12220, "orange": 10436, "pear": 21338}
BOS = 2
TOKENS = NounTokens(ids=IDS, bos=BOS)
PROMPT = "pick up the apple and place it in the basket"
PAD = 48


def tokenize(prompt: str, pad: int = PAD) -> dict:
    """A stand-in `TokenizePrompt`: ``<bos>`` then one piece per word (known nouns get their real ids)."""
    words = prompt.replace(",", " ").split()
    ids = [BOS] + [IDS.get(w, 900 + i) for i, w in enumerate(words)]
    toks = np.zeros(pad, np.int32)
    toks[:len(ids)] = ids
    mask = np.zeros(pad, bool)
    mask[:len(ids)] = True
    return {"tokenized_prompt": toks, "tokenized_prompt_mask": mask}


# =============================================================================================
# 1. SPEC
# =============================================================================================


def test_spec_labels_and_records():
    assert DEFAULT_SOURCE.label == "action:last:L8H2"
    assert parse_attention_source("action:last:L8H2") == DEFAULT_SOURCE
    n = parse_attention_source("noun_target:L10H7")
    assert (n.query, n.layer, n.head, n.agg, n.is_noun) == ("noun_target", 10, 7, None, True)
    assert n.record() == {"spec": "noun_target:L10H7", "query": "noun_target", "layer": 10, "head": 7,
                          "row": "target", "sigma": 0.0, "k": 1}
    t61 = parse_attention_source(" action:mean:L10H7 ")
    assert t61.label == "action:mean:L10H7" and t61.record()["agg"] == "mean" and t61.denoise == 0
    # the T54 / T57 N definition (`t54_arms.SOURCES["noun"]`): query target, one cell, raw σ0
    from AG3S.ag3s.experiments.studies.t54_arms import SOURCES

    assert SOURCES["noun"]["cells"] == ((n.layer, n.head),) and SOURCES["noun"]["sigma"] == 0.0
    assert SOURCES["l8h2"]["layer"] == 8 and SOURCES["l8h2"]["head"] == 2 and SOURCES["l8h2"]["agg"] == "last"


@pytest.mark.parametrize("bad", ["noun_L10H7", "noun_target:L18H0", "noun_target:L10H8", "action:max:L8H2",
                                 "action:L8H2", "action:last:L8", "noun_target:last:L10H7", "", "L8H2",
                                 "noun_basket:L10H7", "action:last:L-1H2"])
def test_unknown_or_out_of_range_spec_is_refused(bad):
    with pytest.raises(ValueError):
        parse_attention_source(bad)


# =============================================================================================
# 2. noun lookup
# =============================================================================================


def test_prompt_noun_needs_exactly_one_fruit_word():
    assert prompt_noun(PROMPT) == ("apple", "")
    assert prompt_noun("Pick up the pear.")[0] == "pear"
    for bad in (None, "", "pick up the crate", "move the apple next to the pear", "pick up the apples"):
        noun, why = prompt_noun(bad)
        assert noun is None and why


def test_lookup_rows_are_the_t53_feature_frame_rows():
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    inputs = tokenize(PROMPT)
    look = noun_lookup(inputs, PROMPT, TOKENS)
    assert look.ok and look.noun == "apple" and look.error == ""
    assert look.token_index == 4 and look.rows.tolist() == [768 + 4, 768 + 10, 768 + 10]
    assert look.record()["prefix_row"] == 772
    calls = []

    def noun_fn(obs, rows):
        calls.append(np.asarray(rows).tolist())
        return {"v": np.zeros((18, len(rows), 4), np.float32), "hidden_final": np.zeros((len(rows), 2))}

    ex.feature_frame(inputs, kv_fn=lambda i: np.zeros(512), obs_fn=lambda i: None, noun_fn=noun_fn,
                     noun="apple", ids=IDS, bos=BOS, n_layers=18)
    assert calls == [look.rows.tolist()]


@pytest.mark.parametrize("prompt,why", [
    ("pick up the crate and place it in the basket", "fruit"),
    ("pick up the apple and place it on the pear", "fruit"),
    ("pick up the apple", "basket"),                      # no destination token -> T53 / T55 not ok
])
def test_lookup_not_found_is_not_guessed(prompt, why):
    look = noun_lookup(tokenize(prompt), prompt, TOKENS)
    assert not look.ok and look.rows is None and look.token_index is None and why in look.error


def test_lookup_token_mismatch_and_duplicate():
    # the prompt names the apple but the tokens do not contain it (e.g. another tokenization)
    inputs = tokenize("pick up the pear and place it in the basket")
    look = noun_lookup(inputs, PROMPT, TOKENS)
    assert not look.ok and "target: 0 matches" in look.error
    inputs = tokenize("apple apple basket")
    look = noun_lookup(inputs, "pick up the apple", TOKENS)
    assert not look.ok and "target: 2 matches" in look.error


# =============================================================================================
# 3. traced bodies (numpy stand-ins)
# =============================================================================================

N_L, N_G, T_TXT = 4, 3, 12


class _FakeLLM:
    """probs[l, 0, 0, g, t, s] = deterministic; kv V[l, 0, s, 0, h] = 1000 l + s + 0.01 h."""

    def __call__(self, embedded, mask, positions):
        tok = np.asarray(embedded[0])
        b, s, _ = tok.shape
        rng = np.random.default_rng(3)
        probs = rng.random((N_L, b, 1, N_G, s, s)).astype(np.float32)
        v = np.zeros((N_L, b, s, 1, 6), np.float32)
        for l in range(N_L):
            v[l, :, :, 0, :] = 1000 * l + np.arange(s)[None, :, None] + 0.01 * np.arange(6)
        return [tok, None], (np.zeros_like(v), v), probs


class _FakeModel:
    def __init__(self, n_image=768):
        self.n_image = n_image
        self.PaliGemma = types.SimpleNamespace(llm=_FakeLLM())

    def embed_prefix(self, obs):
        s = self.n_image + obs.tokenized_prompt.shape[1]
        return (np.zeros((1, s, 2), np.float32), np.ones((1, s), bool), np.zeros((1, s), bool))


def test_prefix_noun_impl_is_t49s_gather_at_the_cell():
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    obs = types.SimpleNamespace(tokenized_prompt=np.zeros((1, T_TXT), np.int32))
    rows = np.asarray([770, 775, 779])
    kv, mask, cell = prefix_noun_impl(_FakeModel(), obs, rows, layer=2, head=1,
                                      make_attn_mask=lambda m, a: None, xp=np)
    ref = nx.prefix_rows_impl(_FakeModel(), obs, rows, make_attn_mask=lambda m, a: None, xp=np)
    assert cell.shape == (3, 768) and cell.dtype == np.float32
    assert np.array_equal(cell, ref["image"][2, 1])
    probs = _FakeLLM()([np.zeros((1, 768 + T_TXT, 2))], None, None)[2]
    assert np.array_equal(cell[1], probs[2, 0, 0, 1, 775, :768])
    assert mask.shape == (1, 768 + T_TXT) and kv[1].shape[0] == N_L
    with pytest.raises(ValueError, match="image tokens"):
        prefix_noun_impl(_FakeModel(n_image=700), obs, rows, layer=2, head=1,
                         make_attn_mask=lambda m, a: None, xp=np)


def test_v_rows_impl_is_t53s_noun_v_at_one_layer():
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    obs = types.SimpleNamespace(tokenized_prompt=np.zeros((1, T_TXT), np.int32))
    rows = np.asarray([770, 773, 779])
    kv = _FakeLLM()([np.zeros((1, 768 + T_TXT, 2), np.float32)], None, None)[1]
    got = v_rows_impl(kv, rows, layer=3, xp=np)
    want = ex.noun_token_impl(_FakeModel(), obs, rows, np.asarray([3]), make_attn_mask=lambda m, a: None,
                              xp=np)["v"][0]
    assert got.dtype == np.float32 and np.array_equal(got, want)
    np.testing.assert_allclose(got[2], 3000 + 779 + 0.01 * np.arange(6), atol=1e-3)


class _JnpLLM:
    """A traceable stand-in (jax.numpy only): probs / V are fixed functions of the shapes."""

    def __call__(self, embedded, mask, positions):
        import jax.numpy as jnp

        tok = embedded[0]
        b, s = tok.shape[0], tok.shape[1]
        grid = jnp.arange(N_L * N_G * s * s, dtype=jnp.float32).reshape(N_L, 1, 1, N_G, s, s)
        probs = jnp.broadcast_to(jnp.sin(grid), (N_L, b, 1, N_G, s, s)) + 0.0 * tok.sum()
        lv = jnp.arange(N_L, dtype=jnp.float32)[:, None, None, None, None]
        sv = jnp.arange(s, dtype=jnp.float32)[None, None, :, None, None]
        hv = jnp.arange(6, dtype=jnp.float32)[None, None, None, None, :]
        v = jnp.broadcast_to(1000 * lv + sv + 0.01 * hv, (N_L, b, s, 1, 6)).astype(jnp.bfloat16)
        return [tok, None], (jnp.zeros_like(v), v), probs


class _JnpModel:
    PaliGemma = types.SimpleNamespace(llm=_JnpLLM())

    @staticmethod
    def embed_prefix(obs):
        import jax.numpy as jnp

        s = 768 + obs.tokenized_prompt.shape[1]
        return (jnp.zeros((1, s, 2), jnp.float32), jnp.ones((1, s), bool), jnp.zeros((1, s), bool))


def test_traced_bodies_jit_under_jax_cpu():
    """The bodies `JaxPrefixBackend` jits: traceable with traced rows and static (layer, head), and equal to
    the T49 / T53 gathers run under the same jit (jax CPU; skipped where jax is absent)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    obs = types.SimpleNamespace(tokenized_prompt=jnp.zeros((1, T_TXT), jnp.int32))
    rows = jnp.asarray([770, 775, 779], jnp.int32)
    mam = lambda m, a: None  # noqa: E731
    f = jax.jit(lambda r: prefix_noun_impl(_JnpModel(), obs, r, layer=2, head=1, make_attn_mask=mam, xp=jnp))
    kv, mask, cell = f(rows)
    ref = jax.jit(lambda r: nx.prefix_rows_impl(_JnpModel(), obs, r, make_attn_mask=mam, xp=jnp))(rows)
    assert cell.shape == (3, 768) and cell.dtype == jnp.float32
    assert np.array_equal(np.asarray(cell), np.asarray(ref["image"][2, 1]))
    v = jax.jit(lambda k, r: v_rows_impl(k, r, layer=3, xp=jnp))(kv, rows)
    want = jax.jit(lambda r: ex.noun_token_impl(_JnpModel(), obs, r, jnp.asarray([3]), make_attn_mask=mam,
                                                xp=jnp)["v"][0])(rows)
    assert v.dtype == jnp.float32 and np.array_equal(np.asarray(v), np.asarray(want))


# =============================================================================================
# 4. probe
# =============================================================================================


def _logreg(w0=1.0, b=0.0, dim=256):
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    coef = np.zeros((2, dim))
    coef[1, 0] = w0
    return LogReg(mean=np.zeros(dim), scale=np.ones(dim), coef=coef, intercept=np.asarray([0.0, b]),
                  C=0.1, n_iter=1, converged=True)


def synthetic_probe(b=0.0) -> NounProbe:
    """p = sigmoid(v[0] + b)."""
    return NounProbe(model=_logreg(b=b), feature="v5", layer=5, path="synthetic", md5="synthetic")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_synthetic_probe_formula():
    pr = synthetic_probe(b=-1.0)
    v = np.zeros(256)
    v[0] = 1.5
    assert pr.p(v) == pytest.approx(_sigmoid(0.5), rel=1e-12)


def _write_probe(path, feature="v5", coef_shape=(2, 256)):
    np.savez(path, mean=np.zeros(256), scale=np.ones(256), coef=np.zeros(coef_shape),
             intercept=np.zeros(2), C=np.asarray(0.1), feature=np.asarray(feature),
             n_iter=np.asarray(3), converged=np.asarray(True))


def test_probe_loader_refusals(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_noun_probe(tmp_path / "none.npz")
    _write_probe(tmp_path / "p.npz")
    with pytest.raises(ValueError, match="md5"):
        load_noun_probe(tmp_path / "p.npz")
    ok = load_noun_probe(tmp_path / "p.npz", expect_md5=None)
    assert ok.layer == 5 and ok.p(np.zeros(256)) == pytest.approx(0.5)
    _write_probe(tmp_path / "h.npz", feature="hidden")
    with pytest.raises(ValueError, match="feature"):
        load_noun_probe(tmp_path / "h.npz", expect_md5=None)
    _write_probe(tmp_path / "v4.npz", feature="v4")
    with pytest.raises(ValueError, match="feature"):
        load_noun_probe(tmp_path / "v4.npz", expect_md5=None)


T55_PROBE = default_task_probe_path("/mnt/dev/AG3S_VLA-storage")
T55_TEST_FEAT = T55_PROBE.parent / "feat" / "test"


@pytest.mark.skipif(not T55_PROBE.is_file(), reason="T55 run not in this storage")
def test_prereg_probe_matches_t55_study_offline_p_on_test_e0_rows():
    """CPU part of the probe parity: on stored T55 **test** E0 features (the feature the GPU gathers), the
    serving probe's p equals `t55_study`'s (`load_probe` + `noun_matrix("v5")` + `proba[:, 1]`) to ~1 ulp."""
    from AG3S.ag3s.experiments.studies import t55_study as st

    probe = load_noun_probe(T55_PROBE)
    assert probe.md5 == T55_PROBE_MD5 and probe.feature == "v5" and probe.layer == 5
    model, feat = st.load_probe(T55_PROBE)
    assert feat == "v5"
    files = sorted(T55_TEST_FEAT.glob("*__E0__orig.npz"))[:3]
    if not files:
        pytest.skip("no T55 test E0 features")
    n = 0
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            feats = {"noun_v": np.asarray(z["noun_v"][:, 0], np.float32),
                     "noun_layers": [int(x) for x in z["noun_layers"]],
                     "noun_ok": np.asarray(z["noun_ok"], bool)}
        X = st.noun_matrix(feats, "v5").astype(np.float64)
        want = model.proba(X)[:, 1]
        for i in np.flatnonzero(feats["noun_ok"])[::7][:6]:
            got = probe.p(feats["noun_v"][i, feats["noun_layers"].index(5)])
            # one row vs the batched matmul: BLAS summation order differs by ~1 ulp, never the threshold side
            assert got == pytest.approx(want[i], rel=1e-12, abs=1e-15)
            assert (got >= TASK_PROBE_TAU) == (want[i] >= TASK_PROBE_TAU)
            n += 1
    assert n > 0


# =============================================================================================
# 5. AttentionPolicy (fake backend)
# =============================================================================================


class _StubPolicy:
    """The served policy, reduced to what `AttentionPolicy` reads: `infer` and `_input_transform`."""

    def __init__(self):
        self.calls = 0

    def infer(self, obs, **_):
        self.calls += 1
        return {"actions": np.zeros((50, wire.ACTION_WIDTH)), "policy_extra": 7}

    @staticmethod
    def _input_transform(obs):
        out = tokenize(obs.get("prompt") or "")
        out["state"] = np.asarray(obs["state"], np.float32)
        return out

    @property
    def metadata(self):
        return {"stub": True}


class _FakeBackend:
    """`JaxPrefixBackend.run` stand-in. cell = state[0] * arange(768); V rows: column 0 = state[1]."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def run(self, inputs, *, source, rows, want_feature, v_layer):
        self.calls.append({"source": source.label, "rows": None if rows is None else np.asarray(rows).tolist(),
                           "want_feature": want_feature, "v_layer": v_layer})
        if self.fail:
            raise RuntimeError("device lost")
        st = np.asarray(inputs["state"], np.float64)
        cell = None
        if source.query == "action" or rows is not None:
            cell = (st[0] * np.arange(768) / 768.0).astype(np.float32)
        v = None
        if rows is not None and v_layer is not None:
            v = np.zeros((len(rows), 256), np.float32)
            v[:, 0] = st[1]
            v[:, 1] = np.arange(len(rows))
        return {"cell": cell, "feature": np.full(512, 0.01, np.float32) if want_feature else None, "v": v}


def _ap(source=None, probe=None, backend=None, subtask_probe=None, **kw):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    return AttentionPolicy(_StubPolicy(), None, sampler=object(), backend=backend or _FakeBackend(),
                           noun_tokens=TOKENS, attention_source=source, task_probe=probe,
                           subtask_probe=subtask_probe, **kw)


def _obs(state0=1.0, state1=0.0, prompt=PROMPT):
    state = np.zeros(wire.ACTION_WIDTH)
    state[0], state[1] = state0, state1
    return {"state": state, "prompt": prompt}


def test_no_flag_takes_the_pre_t57_branch(monkeypatch):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    ap = AttentionPolicy(_StubPolicy(), None, sampler=object(), subtask_probe=None)
    assert ap._extended is False and ap._backend is None                         # noqa: SLF001
    seen = []
    maps = {"cam_high": np.ones((16, 16))}
    monkeypatch.setattr(ap, "_attention_and_feature", lambda obs: (seen.append(1), (maps, None))[1])
    out = ap.infer(_obs())
    assert seen == [1] and out["attention"] is maps
    assert set(out) == {"actions", "policy_extra", "attention"}
    assert ap.metadata == {"stub": True}


def test_noun_source_feeds_the_target_row():
    be = _FakeBackend()
    ap = _ap(source="noun_target:L10H7", backend=be, subtask_probe="default")
    out = ap.infer(_obs(state0=2.0))
    assert be.calls == [{"source": "noun_target:L10H7", "rows": [772, 778, 778], "want_feature": True,
                         "v_layer": None}]
    want = (2.0 * np.arange(768) / 768.0).astype(np.float32)
    assert set(out["attention"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert np.array_equal(out["attention"]["cam_left_wrist"], want[256:512].reshape(16, 16))
    rec = out["attention_source"]
    assert rec["spec"] == "noun_target:L10H7" and rec["noun"] == "apple" and rec["noun_ok"] is True
    assert rec["token_index"] == 4 and rec["prefix_row"] == 772 and rec["attention"] == "ok"
    assert out["subtask"]["probe"] == "kv_L4_v1"
    assert "task_probe" not in out
    assert ap.metadata == {"stub": True, "attention_source": "noun_target:L10H7"}


@pytest.mark.parametrize("prompt", ["pick up the crate and place it in the basket", "apple pear basket", None])
def test_noun_not_found_gives_empty_attention_and_a_reason(prompt):
    be = _FakeBackend()
    ap = _ap(source="noun_target:L10H7", backend=be, subtask_probe="default")
    out = ap.infer(_obs(prompt=prompt))
    assert out["attention"] == {}
    assert be.calls[0]["rows"] is None and be.calls[0]["want_feature"] is True
    rec = out["attention_source"]
    assert rec["noun_ok"] is False and rec["noun_error"] and rec["attention"] == "empty"
    assert "subtask" in out  # the kv_L4 label still rides (same prefix pass)
    # `serve_safe.attention_extractor` turns that into no maps (AG3S runs without a target)
    from AG3S.trajopt.serve_safe import attention_extractor

    assert attention_extractor()(None, out) == {}


def test_default_prompt_is_used_when_the_request_has_none():
    ap = _ap(source="noun_target:L10H7", default_prompt=PROMPT)
    obs = _obs()
    obs.pop("prompt")
    # the stand-in tokenizer has no default-prompt injection: give it the tokens the policy would see
    ap._policy._input_transform = lambda o: {**tokenize(PROMPT), "state": np.asarray(o["state"])}  # noqa: SLF001
    assert ap.infer(obs)["attention_source"]["noun"] == "apple"


def test_action_source_with_the_probe():
    be = _FakeBackend()
    pr = synthetic_probe(b=-1.0)
    ap = _ap(source="action:mean:L10H7", probe=pr, backend=be)
    out = ap.infer(_obs(state0=1.0, state1=1.3))
    assert be.calls == [{"source": "action:mean:L10H7", "rows": [772, 778, 778], "want_feature": False,
                         "v_layer": 5}]
    tp = out["task_probe"]
    assert tp["ok"] is True and tp["noun_ok"] is True and tp["noun"] == "apple" and tp["error"] == ""
    assert tp["p"] == pytest.approx(_sigmoid(0.3)) and tp["probe_md5"] == "synthetic"
    assert len(tp["feature_sha1"]) == 16
    assert out["attention_source"]["spec"] == "action:mean:L10H7" and "noun" not in out["attention_source"]
    assert ap.metadata["task_probe"]["feature"] == "v5"


def test_probe_only_keeps_the_default_cell_and_adds_no_source_record():
    be = _FakeBackend()
    ap = _ap(probe=synthetic_probe(), backend=be)
    out = ap.infer(_obs())
    assert be.calls[0]["source"] == DEFAULT_SOURCE.label and be.calls[0]["v_layer"] == 5
    assert "attention_source" not in out and out["task_probe"]["ok"] is True
    assert set(out["attention"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def test_probe_without_noun_is_not_ok():
    ap = _ap(probe=synthetic_probe(b=10.0))
    tp = ap.infer(_obs(prompt="pick up the crate and place it in the basket"))["task_probe"]
    assert tp["ok"] is False and tp["noun_ok"] is False and tp["p"] is None and "noun not found" in tp["error"]


def test_extraction_failure_fails_closed():
    ap = _ap(source="noun_target:L10H7", probe=synthetic_probe(b=10.0), backend=_FakeBackend(fail=True))
    out = ap.infer(_obs())
    assert "attention" not in out
    assert out["attention_source"]["attention"] == "failed" and "device lost" in out["attention_source"]["error"]
    assert out["task_probe"]["ok"] is False and out["task_probe"]["p"] is None


def test_noun_source_or_probe_needs_noun_tokens():
    from AG3S.trajopt.attention_policy import AttentionPolicy

    for kw in ({"attention_source": "noun_target:L10H7"}, {"task_probe": synthetic_probe()}):
        with pytest.raises(ValueError, match="noun"):
            AttentionPolicy(_StubPolicy(), None, sampler=object(), backend=_FakeBackend(), subtask_probe=None,
                            **kw)
    # an action source alone does not
    AttentionPolicy(_StubPolicy(), None, sampler=object(), backend=_FakeBackend(), subtask_probe=None,
                    attention_source="action:mean:L10H7")
    with pytest.raises(ValueError):
        AttentionPolicy(_StubPolicy(), None, sampler=object(), backend=_FakeBackend(), subtask_probe=None,
                        attention_source="noun_target:L99H7", noun_tokens=TOKENS)


# =============================================================================================
# 6. SafePolicy gate (verdict rig of test_verdict_reasons)
# =============================================================================================

from tests.trajopt.test_verdict_reasons import _cs, _Field, _result, _scene, _wire_up, rig  # noqa: E402,F401


def _sp(task_gate=None, policy=None, **kw):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy as _SP

    model = _robot_model()
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf",
                                      "clustering": {"gripper_max_opening": 0.0715}}),
                robot_model=model, constraint_robot_model=model, attached_parent_links=grasp_parent_links())
    if task_gate is not None:
        kw["task_gate"] = task_gate
    return SafePolicy(policy or _SP(), ag3s=ag3s, **kw)


def _probe(p, ok=True, noun_ok=True):
    return {"p": p, "ok": ok, "noun_ok": noun_ok, "noun": "apple", "error": "", "feature": "v5",
            "probe_md5": "x", "feature_sha1": "abc"}


def _gate_verdict(sp, rig, *, phase, probe, status="optimal", violation=0.0, scene=None, seq=5):
    sp._latch.phase = phase                                                          # noqa: SLF001
    sp._pending = {"scene": {"seq": seq}}                                            # noqa: SLF001
    if sp.task_gate != "off":
        sp._pending["task_probe"] = probe                                            # noqa: SLF001
    scene = scene if scene is not None else _scene(rig, _Field([]))
    _wire_up(sp, rig, scene, _result(status, violation, rig.s.reference), _cs())
    return sp._verdict(None)                                                         # noqa: SLF001


def _kinds(v):
    return [r["kind"] for r in (v.verdict_reasons or [])]


def test_reason_kind_is_registered_as_hold():
    assert "task_not_ready" in wire.REASON_KINDS and wire.GATE_DEFAULT["task_not_ready"] == "hold"
    r = wire.make_reason("task_not_ready", "x", probe_p=0.1)
    assert wire.gate_decision([r])[0] == "hold"
    old = {k: a for k, a in wire.GATE_DEFAULT.items() if k != "task_not_ready"}
    assert wire.gate_decision([r], old)[0] == "hold"  # a client that does not know it holds (fail closed)


def test_constructor_refusals():
    with pytest.raises(ValueError, match="task_gate"):
        _sp(task_gate="maybe")
    with pytest.raises(ValueError, match="legacy"):
        _sp(task_gate="on", verdict_policy="legacy")
    _sp(task_gate="shadow", verdict_policy="legacy")  # shadow only records


def test_off_records_nothing_and_ignores_a_probe(rig):
    sp = _sp()
    assert sp.task_gate == "off" and "task_gate" not in sp.metadata
    v = _gate_verdict(sp, rig, phase=GraspPhase.HELD, probe=_probe(0.0))
    assert v.safe is True and _kinds(v) == []
    assert "task_gate" not in sp._exec_summary() and sp._last_task_gate == {}        # noqa: SLF001


@pytest.mark.parametrize("probe,why", [
    (_probe(0.05), "p 0.0500 < tau 0.3"),
    (_probe(None, ok=False, noun_ok=False) | {"error": "noun not found: x"}, "noun not found"),
    (_probe(0.9, ok=True, noun_ok=False), "noun not found"),
    (_probe(0.9, ok=False) | {"error": "probe evaluation failed"}, "probe evaluation failed"),
    (None, "no task_probe in the policy result"),
    (_probe(float("nan")), "probe gave no finite p"),
])
def test_held_and_not_ready_holds(rig, probe, why):
    sp = _sp("on")
    v = _gate_verdict(sp, rig, phase=GraspPhase.HELD, probe=probe)
    assert v.safe is False and _kinds(v) == ["task_not_ready"]
    r = v.verdict_reasons[0]
    assert r["action"] == "hold" and why in r["detail"]
    ev = r["evidence"]
    assert set(ev) >= {"probe_p", "tau", "noun_ok", "latch_state"}
    assert ev["tau"] == TASK_PROBE_TAU and ev["latch_state"] == "held"
    rec = sp._exec_summary()["task_gate"]
    assert rec["task_ready"] is False and rec["task_gate_applied"] is True and rec["held"] is True
    assert rec["latch_state"] == "held" and rec["seq"] == 5


@pytest.mark.parametrize("p", [TASK_PROBE_TAU, 0.31, 0.99])
def test_held_and_ready_executes(rig, p):
    sp = _sp("on")
    v = _gate_verdict(sp, rig, phase=GraspPhase.HELD, probe=_probe(p))
    assert v.safe is True and _kinds(v) == []
    rec = sp._exec_summary()["task_gate"]
    assert rec["task_ready"] is True and rec["task_gate_applied"] is False and rec["probe_p"] == p


@pytest.mark.parametrize("phase", [GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING,
                                   GraspPhase.PLACED])
def test_not_held_never_gates(rig, phase):
    off = _gate_verdict(_sp(), rig, phase=phase, probe=None)
    sp = _sp("on")
    v = _gate_verdict(sp, rig, phase=phase, probe=_probe(0.0, ok=False, noun_ok=False))
    assert v.to_dict() == off.to_dict()
    rec = sp._exec_summary()["task_gate"]
    assert rec["task_ready"] is False and rec["task_gate_applied"] is False and rec["latch_state"] == phase.value


def test_other_hold_reasons_are_untouched(rig):
    from tests.trajopt.test_t43z6_follow_to import _torso_scene

    scene = _torso_scene(rig, -0.003)  # a real (penetrating) collision row
    off = _gate_verdict(_sp(), rig, phase=GraspPhase.HELD, probe=None, status="violated", violation=0.03,
                        scene=scene)
    assert _kinds(off) == ["collision"]
    ready = _gate_verdict(_sp("on"), rig, phase=GraspPhase.HELD, probe=_probe(0.8), status="violated",
                          violation=0.03, scene=scene)
    assert ready.to_dict() == off.to_dict()  # probe passes: only task_not_ready would differ, and it is absent
    held = _gate_verdict(_sp("on"), rig, phase=GraspPhase.HELD, probe=_probe(0.1), status="violated",
                         violation=0.03, scene=scene)
    assert _kinds(held) == ["collision", "task_not_ready"]
    assert held.verdict_reasons[0] == off.verdict_reasons[0]


def test_follow_to_streak_does_not_count_a_chunk_the_gate_held(rig):
    """T43 Z6 (`--hold-follow-to`, on in the T54c C1a profile): a margin-only collision becomes `follow_to`
    exactly as without the gate; when the gate holds that chunk it was not followed, so the streak is not
    advanced (the record says why)."""
    from tests.trajopt.test_t43z6_follow_to import _sp as _sp_follow
    from tests.trajopt.test_t43z6_follow_to import _torso_scene, _verdict

    def run(probe):
        sp = _sp_follow(limit=3, task_gate="on")
        sp._latch.phase = GraspPhase.HELD                                            # noqa: SLF001
        sp._pending = {"scene": {"seq": 1}, "task_probe": probe}                     # noqa: SLF001
        return sp, _verdict(sp, rig, _torso_scene(rig, 0.02))

    sp, v = run(_probe(0.9))
    assert _kinds(v) == ["follow_to"] and v.safe is True and sp._follow_to_streak == 1  # noqa: SLF001
    sp, v = run(_probe(0.1))
    assert _kinds(v) == ["follow_to", "task_not_ready"] and v.safe is False
    assert v.verdict_reasons[0]["action"] == "execute"           # the follow_to decision itself is unchanged
    assert sp._follow_to_streak == 0                                                 # noqa: SLF001
    rec = sp._exec_summary()["verdict"]["follow_to"]
    assert rec["decision"] == "follow_to" and rec["held_by"] == "task_not_ready" and rec["streak"] == 0


def test_shadow_records_without_gating(rig):
    for phase, p in ((GraspPhase.HELD, 0.1), (GraspPhase.HELD, 0.6), (GraspPhase.LATCHED, 0.1)):
        off = _gate_verdict(_sp(), rig, phase=phase, probe=None)
        sp = _sp("shadow")
        v = _gate_verdict(sp, rig, phase=phase, probe=_probe(p))
        assert v.to_dict() == off.to_dict()
        rec = sp._exec_summary()["task_gate"]
        assert rec["mode"] == "shadow" and rec["task_gate_applied"] is False
        assert rec["task_ready"] is (phase is GraspPhase.HELD and p >= TASK_PROBE_TAU)
        assert rec["probe_p"] == p and rec["noun_ok"] is True
    assert sp.metadata["task_gate"] == {"mode": "shadow", "tau": TASK_PROBE_TAU}


def test_no_scene_chunk_records_the_gate_off_wire(rig):
    sp = _sp("on")
    sp._latch.phase = GraspPhase.HELD                                                # noqa: SLF001
    sp._pending = {"scene": {"seq": 2}, "task_probe": _probe(0.1)}                   # noqa: SLF001
    sp.refiner.last_result = None
    v = sp._verdict(None)                                                            # noqa: SLF001
    assert v.safe is False and v.verdict_reasons is None  # the wire shape of a no-scene chunk is unchanged
    kinds = [r["kind"] for r in sp._last_reasons]                                    # noqa: SLF001
    assert kinds == ["uncertified", "task_not_ready"]
    assert sp._last_reasons[1]["evidence"]["carried_on_wire"] is False               # noqa: SLF001
    assert sp._exec_summary()["task_gate"]["on_wire"] is False


# =============================================================================================
# 7. consecutive HOLD requests: inference and the probe continue on every request
# =============================================================================================


def test_probe_is_recomputed_on_every_held_request_and_the_gate_releases():
    be = _FakeBackend()
    pr = synthetic_probe(b=-1.0)              # p = sigmoid(state[1] - 1)
    ap = _ap(probe=pr, backend=be)
    sp = _sp("on", policy=ap)
    sp._latch.phase = GraspPhase.HELD                                                # noqa: SLF001

    def fake_refine(chunk, context):
        sp.refiner.last_result = _result("optimal", 0.0, None)
        sp._last_constraint_set = _cs()                                              # noqa: SLF001
        return chunk

    sp.refiner.refine = fake_refine
    states1 = [-1.0, 0.0, 0.4, 3.0]           # p: 0.12, 0.27, 0.35, 0.88
    rows = []
    for k, s1 in enumerate(states1):
        state = np.zeros(wire.ACTION_WIDTH)
        state[0], state[1] = 1.0 + k, s1      # a new observation every request
        resp = sp.infer({"state": state, "prompt": PROMPT, wire.PREFIX + "seq": k})
        rec = sp._exec_summary()["task_gate"]
        rows.append((resp, rec))
    assert ap._policy.calls == len(states1) and len(be.calls) == len(states1)       # noqa: SLF001
    ps = [rec["probe_p"] for _, rec in rows]
    assert ps == pytest.approx([_sigmoid(s - 1.0) for s in states1])
    assert len({rec["feature_sha1"] for _, rec in rows}) == len(states1)
    safe = [resp["safe"] for resp, _ in rows]
    kinds = [[r["kind"] for r in resp[wire.VERDICT_REASONS]] for resp, _ in rows]
    assert safe == [False, False, True, True]
    assert kinds == [["task_not_ready"], ["task_not_ready"], [], []]
    for resp, rec in rows:
        # the response carries the request's own probe and gate record
        assert resp["task_probe"]["p"] == rec["probe_p"] and resp["task_gate"] == rec
        assert rec["latch_state"] == "held"
    assert [rec["task_gate_applied"] for _, rec in rows] == [True, True, False, False]
    # the latch was never touched by the gate (attach / detach / phase unchanged)
    assert sp._latch.phase is GraspPhase.HELD                                        # noqa: SLF001


# =============================================================================================
# 8. serve_safe flags
# =============================================================================================

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _args(*extra):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE_ARGV + list(extra))


def test_flags_default_off():
    from AG3S.trajopt import serve_safe

    ap, args = _args()
    assert args.attention_source is None and args.task_gate is False and args.task_gate_shadow is False
    assert args.task_probe is None
    assert serve_safe.task_gate_mode(args) == "off" and serve_safe.t57a_record_meta(args) == {}
    serve_safe.reject_bad_flag_combinations(ap, args)


@pytest.mark.parametrize("extra", [
    ["--attention-source", "noun_L10H7"],
    ["--attention-source", "noun_target:L10H9"],
    ["--attention-source", "noun_target:L10H7", "--no-safe"],
    ["--attention-source", "action:mean:L10H7", "--no-attention"],
    ["--attention-source", "noun_target:L10H7", "--no-perception"],
    ["--task-gate", "--no-attention"],
    ["--task-gate-shadow", "--no-safe"],
    ["--task-probe", "/x.npz"],
    ["--task-gate", "--task-probe", "/nonexistent/noun_probe.npz"],
])
def test_bad_combinations_are_refused(extra):
    from AG3S.trajopt import serve_safe

    ap, args = _args(*extra)
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_gate_and_shadow_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _args("--task-gate", "--task-gate-shadow")


def test_gate_refuses_a_probe_that_is_not_the_prereg(tmp_path):
    from AG3S.trajopt import serve_safe

    _write_probe(tmp_path / "other.npz")
    ap, args = _args("--task-gate", "--task-probe", str(tmp_path / "other.npz"))
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


@pytest.mark.skipif(not T55_PROBE.is_file(), reason="T55 run not in this storage")
def test_flags_accepted_and_record_meta(monkeypatch):
    from AG3S.trajopt import serve_safe

    monkeypatch.setenv("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")
    for extra, mode in ((["--task-gate"], "on"), (["--task-gate-shadow"], "shadow"),
                        (["--task-gate", "--task-probe", str(T55_PROBE)], "on")):
        ap, args = _args("--attention-source", "noun_target:L10H7", *extra)
        serve_safe.reject_bad_flag_combinations(ap, args)
        assert serve_safe.task_gate_mode(args) == mode
        meta = serve_safe.t57a_record_meta(args)
        assert meta["attention_source"]["spec"] == "noun_target:L10H7"
        assert meta["task_gate"] == {"mode": mode, "probe": str(T55_PROBE), "probe_md5": T55_PROBE_MD5,
                                     "feature": "v5", "tau": 0.3, "persist": 1}
    ap, args = _args("--attention-source", "action:mean:L10H7")
    serve_safe.reject_bad_flag_combinations(ap, args)
    assert serve_safe.t57a_record_meta(args) == {
        "attention_source": {"spec": "action:mean:L10H7", "query": "action", "layer": 10, "head": 7,
                             "agg": "mean", "denoise": 0}}
