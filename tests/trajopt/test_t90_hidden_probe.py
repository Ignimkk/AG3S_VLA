"""T90 §1 — the T86 hidden image-patch probe as the ATS target signal (`--attention-source hidden_probe[:NPZ]`).

Pinned (handoff/T90.task.md §1.4):
 1. asset: the shipped npz loads (md5 = its json, class order, layers head 8 / left wrist 12 / right wrist 8); a
    wrong class order / dim / missing camera / missing file is refused.
 2. same hidden -> same score as the offline `t86_probe` (known hidden of the T86 run: the exported asset reproduces
    T86's test maps; skipped when the run is not on this machine) and numpy == jax.numpy == `LogReg.proba`.
 3. serving pass (dummy-size pi0.5, same weights as the T86 extractor model): the jitted `probe_pass` logits equal
    T86's extractor tokens scored offline; kv / mask are the flag-off prefix's; with `want_hsr5` the HSR5 feature is
    the `hsr5_pass` feature of the same stack (one pass, no extra forward).
 4. `AttentionPolicy`: source off = bit-identical (keys, backend call); hidden_probe -> maps, record (spec, layers,
    asset md5, ms, max score); noun not found -> empty attention (no target) and the reason; with / without `--hsr5`.
 5. serve_safe: parse, start-time refusals, record meta, `--hsr5` / attention-surface accept it, noun spec unchanged.
"""

from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

from AG3S.trajopt import attention_source as AS
from AG3S.trajopt import hidden_probe as HP
from tests.trajopt.test_t57a_attention_source import PROMPT, TOKENS, _obs, _StubPolicy

SHIPPED = HP.default_asset_path()
T86_RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261103T000950+0900_8ab3c7ef7757_67e71a9db264")


def _rand_probe(d=HP.HIDDEN_DIM, seed=0, layers=(8, 12, 8)):
    rng = np.random.default_rng(seed)
    kw = {k: {} for k in ("mean", "scale", "coef", "intercept")}
    for c in HP.CAMERAS:
        kw["mean"][c] = rng.normal(size=d)
        kw["scale"][c] = rng.uniform(0.5, 2.0, size=d)
        kw["coef"][c] = rng.normal(size=(2, d)) * 0.05
        kw["intercept"][c] = rng.normal(size=2)
    return HP.HiddenProbe(layers=dict(zip(HP.CAMERAS, layers)), path="mem", md5="0" * 32, meta={"variant": "raw"}, **kw)


# ================================================================================================ 1. asset


def test_shipped_asset_loads_and_matches_its_json():
    probe = HP.load_hidden_probe()
    side = json.loads(SHIPPED.with_suffix(".json").read_text())
    assert probe.md5 == side["md5"] == HP.file_md5(SHIPPED)
    assert probe.layers == {"cam_high": 8, "cam_left_wrist": 12, "cam_right_wrist": 8}     # T86 t86_probe.json selected.raw
    assert {c: side["cameras"][c]["layer"] for c in HP.CAMERAS} == probe.layers
    assert side["classes"] == ["other", "target"] and side["variant"] == "raw"
    for c in HP.CAMERAS:
        assert probe.coef[c].shape == (2, 2048) and probe.mean[c].shape == probe.scale[c].shape == (2048,)
        assert np.all(probe.scale[c] > 0) and side["cameras"][c]["converged"]
    assert probe.layer_set == (8, 12)
    assert HP.load_hidden_probe("default").md5 == probe.md5


def test_asset_selection_is_t86s_when_the_run_is_present():
    sel = T86_RUN / "probe_t86s" / "t86_probe.json"
    if not sel.is_file():
        pytest.skip("T86 run not on this machine")
    doc = json.loads(sel.read_text())
    probe = HP.load_hidden_probe()
    for c in HP.CAMERAS:
        s = doc["cameras"][c]["selected"]["raw"]
        side = json.loads(SHIPPED.with_suffix(".json").read_text())["cameras"][c]
        assert (probe.layers[c], side["C"]) == (int(s["layer"]), float(s["C"]))


def test_save_load_roundtrip_and_refusals(tmp_path):
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    rng = np.random.default_rng(1)
    models = {c: LogReg(mean=rng.normal(size=2048), scale=rng.uniform(0.5, 2, 2048), coef=rng.normal(size=(2, 2048)),
                        intercept=rng.normal(size=2), C=0.01, n_iter=5, converged=True) for c in HP.CAMERAS}
    layers = {"cam_high": 8, "cam_left_wrist": 12, "cam_right_wrist": 8}
    p = HP.save_asset(tmp_path / "a.npz", models, layers, meta={"seed": 0})
    got = HP.load_hidden_probe(p)
    assert got.layers == layers and got.meta["seed"] == 0
    for c in HP.CAMERAS:
        assert np.array_equal(got.coef[c], models[c].coef) and np.array_equal(got.mean[c], models[c].mean)
    with pytest.raises(FileNotFoundError):
        HP.load_hidden_probe(tmp_path / "missing.npz")
    bad = dict(np.load(p))
    bad["class_order"] = np.asarray(["target", "other"])
    np.savez(tmp_path / "order.npz", **bad)
    with pytest.raises(ValueError, match="class order"):
        HP.load_hidden_probe(tmp_path / "order.npz")
    bad = dict(np.load(p))
    bad["cam_high__coef"] = bad["cam_high__coef"][:, :100]
    np.savez(tmp_path / "dim.npz", **bad)
    with pytest.raises(ValueError, match="expected"):
        HP.load_hidden_probe(tmp_path / "dim.npz")
    bad = {k: v for k, v in np.load(p).items() if not k.startswith("cam_left_wrist")}
    np.savez(tmp_path / "cam.npz", **bad)
    with pytest.raises(ValueError, match="cam_left_wrist"):
        HP.load_hidden_probe(tmp_path / "cam.npz")


# ================================================================================================ 2. same hidden -> same score


def test_score_equals_the_two_class_logreg_proba():
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    rng = np.random.default_rng(2)
    d = 64
    m = LogReg(mean=rng.normal(size=d), scale=rng.uniform(0.5, 2, d), coef=rng.normal(size=(2, d)) * 0.3,
               intercept=rng.normal(size=2), C=0.01, n_iter=1, converged=True)
    h = rng.normal(size=(256, d)) * 3
    lg = HP.score_logits(np, h, m.mean, m.scale, m.coef[1] - m.coef[0], float(m.intercept[1] - m.intercept[0]))
    np.testing.assert_allclose(HP.sigmoid(lg), m.proba(h)[:, 1], rtol=1e-12, atol=1e-15)


def test_numpy_and_jax_logits_agree():
    jnp = pytest.importorskip("jax.numpy")
    probe = _rand_probe()
    hidden = np.random.default_rng(3).normal(size=(19, 1, 968, 2048)).astype(np.float32) * 5
    ranges = {"cam_high": (0, 256), "cam_left_wrist": (256, 512), "cam_right_wrist": (512, 768)}
    arrays_np = {c: {"mean": probe.mean[c].astype(np.float32), "scale": probe.scale[c].astype(np.float32),
                     "w": probe.w(c).astype(np.float32), "b": np.float32(probe.b(c))} for c in HP.CAMERAS}
    want = HP.logits_of_stack(np, hidden, arrays_np, probe.layers, ranges)
    arrays_j = {c: {k: jnp.asarray(v) for k, v in a.items()} for c, a in arrays_np.items()}
    got = np.asarray(HP.logits_of_stack(jnp, jnp.asarray(hidden), arrays_j, probe.layers, ranges))
    assert want.shape == got.shape == (3, 256)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)
    # each camera reads its own layer and its own 256 tokens
    c = 1
    h = hidden[probe.layers[HP.CAMERAS[c]], 0, 256:512].astype(np.float64)
    ref = ((h - probe.mean[HP.CAMERAS[c]]) / probe.scale[HP.CAMERAS[c]]) @ probe.w(HP.CAMERAS[c]) + probe.b(HP.CAMERAS[c])
    np.testing.assert_allclose(want[c], ref, rtol=1e-3, atol=2e-2)


def test_probability_map_layout_matches_the_attention_map_layout():
    lg = np.arange(3 * 256, dtype=np.float64).reshape(3, 256) / 100.0 - 3.0
    flat = HP.camera_probabilities(lg)
    assert flat.shape == (768,) and flat.dtype == np.float32
    maps = AS.cell_maps(flat)                       # the very split the attention cell goes through
    pm = HP.probe_map(lg)
    for i, c in enumerate(HP.CAMERAS):
        assert maps[c].shape == (16, 16) and np.array_equal(maps[c], pm[c])
        assert np.argmax(maps[c]) == 255            # token t of camera i = row t // 16, col t % 16
    assert np.all(flat > 0) and np.all(flat < 1)


def test_sigmoid_does_not_tie_where_float32_would_saturate():
    p = HP.sigmoid(np.array([20.0, 21.0, 22.0]))
    assert p[0] < p[1] < p[2] < 1.0


def test_exported_asset_reproduces_t86_test_maps():
    """Known hidden of the T86 run (test scenes the refit never saw): serving's score = T86's B map, to float32."""
    if not (T86_RUN / "probe_t86s" / "test_maps_cam_high.npz").is_file():
        pytest.skip("T86 run not on this machine")
    from AG3S.ag3s.experiments.studies.t86_probe import safe_name

    d = json.loads((T86_RUN / "samples_t020s" / "t86_samples.json").read_text())
    probe = HP.load_hidden_probe()
    rng = np.random.default_rng(0)
    for cam in HP.CAMERAS:
        z = np.load(T86_RUN / "probe_t86s" / f"test_maps_{cam}.npz")
        for i in rng.choice(len(z["B"]), size=4, replace=False):
            ri, req, cf = (int(x) for x in z["meta"][i])
            with np.load(T86_RUN / "hidden_t86s" / f"{safe_name(d['runs'][ri]['label'])}.npz") as pz:
                reqs = [int(r) for r in pz["request"]]
                p = [int(x) for x in pz["cf_request"]].index(reqs.index(req)) if cf else reqs.index(req)
                cams = [str(c) for c in pz["camera_names"]]
                lays = [int(x) for x in pz["layers"]]
                h = np.asarray((pz["cf_patches"] if cf else pz["patches"])[p, cams.index(cam), lays.index(probe.layers[cam])],
                               np.float32)
            lg = HP.score_logits(np, h, probe.mean[cam], probe.scale[cam], probe.w(cam), probe.b(cam))
            np.testing.assert_allclose(HP.sigmoid(lg), z["B"][i], rtol=0, atol=1e-6)


# ================================================================================================ 3. serving pass


def jax_leaves(tree):
    import jax

    return jax.tree.leaves(tree)


@pytest.fixture(scope="module")
def dummy_backend():
    """Dummy-size pi0.5: the serving attention copy (return_attn_probs) and the same params as the T86 extractor model
    (return_hidden_states, `ProbeExtractor`)."""
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import flax.nnx as nnx
    import jax

    from AG3S.ag3s.experiments.sources import pi05_hidden as ph
    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler
    from openpi.models import model as _model
    from openpi.models import pi0_config

    cfg_a = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                 return_attn_probs=True)
    cfg_h = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                 return_hidden_states=True)
    model_a = cfg_a.create(jax.random.key(0))
    model_h = cfg_h.create(jax.random.key(1))
    nnx.update(model_h, nnx.state(model_a, nnx.Param))
    sampler = AttentionSampler(model_a, num_steps=2)
    backend = AS.JaxPrefixBackend(sampler)
    raw = cfg_a.fake_obs()
    obs = _model.preprocess_observation(None, raw, train=False)
    return types.SimpleNamespace(sampler=sampler, backend=backend, obs=obs, raw=raw, model_h=model_h)


def _dummy_probe(d=64, layers=(1, 2, 1)):
    return _rand_probe(d=d, seed=4, layers=layers)


def test_serving_pass_logits_equal_the_t86_extractor_tokens_scored_offline(dummy_backend):
    import jax.numpy as jnp

    from AG3S.ag3s.experiments.sources import pi05_hidden as ph
    from openpi.shared import array_typing as at

    d = dummy_backend
    probe = _dummy_probe()
    d.backend.set_hidden_probe(probe)
    kv, mask, logits, feat = d.backend.probe_pass(d.obs)
    assert feat is None and np.asarray(logits).shape == (3, 256)
    # offline: the T86 extractor's selected-camera tokens at the selected layers (its own jit, return_hidden_states model)
    ext = ph.ProbeExtractor(d.model_h, layers=(1, 2), cameras=HP.CAMERAS)
    with at.disable_typechecking():
        off = ext._pjit(ext._state, d.raw, jnp.asarray(np.array([770], np.int32)))     # noqa: SLF001
    H = np.asarray(off["h"], np.float64)                                              # [3, 2, 256, D]
    want = []
    for ci, c in enumerate(HP.CAMERAS):
        h = H[ci, (1, 2).index(probe.layers[c])]
        want.append(HP.score_logits(np, h, probe.mean[c], probe.scale[c], probe.w(c), probe.b(c)))
    np.testing.assert_allclose(np.asarray(logits), np.asarray(want), rtol=2e-4, atol=2e-4)
    # one pass: kv and mask are the flag-off prefix's (as T82's hsr5 pass)
    kv0, mask0 = d.sampler._jit_prefix(d.sampler._state, d.obs)                       # noqa: SLF001
    for a, b in zip(jax_leaves(kv), jax_leaves(kv0)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(mask), np.asarray(mask0))


def test_serving_pass_with_hsr5_gives_the_same_feature_and_the_same_logits(dummy_backend):
    import jax.numpy as jnp

    d = dummy_backend
    d.backend.set_hidden_probe(_dummy_probe())
    _, _, lg_off, f_off = d.backend.probe_pass(d.obs, want_hsr5=False)
    _, _, lg_on, f_on = d.backend.probe_pass(d.obs, want_hsr5=True)
    assert f_off is None and np.asarray(f_on).shape == (4 * 64,)
    assert np.array_equal(np.asarray(lg_on), np.asarray(lg_off))
    rows = np.array([770, 772], np.int32)
    _, _, _, feat5 = d.backend.hsr5_pass(d.obs, rows, layer=1, head=0)               # T82's own pass, same hidden
    np.testing.assert_allclose(np.asarray(f_on), np.asarray(feat5), rtol=1e-6, atol=1e-6)


def test_probe_pass_without_an_asset_raises(dummy_backend):
    be = AS.JaxPrefixBackend(dummy_backend.sampler)
    with pytest.raises(RuntimeError, match="no probe asset"):
        be.probe_pass(dummy_backend.obs)


def test_attention_model_flags_are_untouched_by_the_probe_pass(dummy_backend):
    dummy_backend.backend.set_hidden_probe(_dummy_probe())
    dummy_backend.backend.probe_pass(dummy_backend.obs)
    llm = dummy_backend.sampler._nnx.merge(dummy_backend.sampler._graphdef,         # noqa: SLF001
                                           dummy_backend.sampler._state).PaliGemma.llm  # noqa: SLF001
    assert llm.module.return_probs is True and llm.module.return_hidden is False


def test_backend_default_has_no_probe_state():
    class _S:
        _nnx = _graphdef = _state = None

    pytest.importorskip("jax")
    be = AS.JaxPrefixBackend(_S())
    assert be._hidden_probe is None and be._jit_probe is None                       # noqa: SLF001


# ================================================================================================ 4. AttentionPolicy


class _ProbeBackend:
    """`JaxPrefixBackend.run` stand-in for the hidden-probe source: logits [3, 256] = state[0] * arange / 768 - 4."""

    def __init__(self):
        self.calls = []
        self.probe = None

    def set_hidden_probe(self, probe):
        self.probe = probe

    def run(self, inputs, *, source, rows, want_feature, v_layer, **kw):
        self.calls.append({"source": source.label, "rows": None if rows is None else np.asarray(rows).tolist(),
                           "want_feature": want_feature, "v_layer": v_layer, **kw})
        st = np.asarray(inputs["state"], np.float64)
        lg = (st[0] * np.arange(768).reshape(3, 256) / 768.0 * 8.0 - 4.0).astype(np.float32)
        out = {"cell": None, "feature": None, "v": None, "probe_logits": lg, "probe_ms": 2.5, "path": "fake"}
        if rows is not None:
            out["cell"] = HP.camera_probabilities(lg)
        if kw.get("want_hsr5"):
            out["hsr5_feature"] = np.zeros(8192, np.float32)
            out["hsr5_ms"] = 2.5
        return out


def _ap(source, backend=None, **kw):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    return AttentionPolicy(_StubPolicy(), None, sampler=object(), backend=backend or _ProbeBackend(),
                           noun_tokens=TOKENS, attention_source=source, subtask_probe=None, **kw)


def test_hidden_probe_feeds_the_attention_slot_and_records_what_it_read():
    be = _ProbeBackend()
    ap = _ap("hidden_probe", be)
    assert be.probe is not None and be.probe.md5 == HP.load_hidden_probe().md5          # asset set at construction
    r = ap.infer(_obs(state0=1.0))
    assert set(r["attention"]) == set(HP.CAMERAS) and all(a.shape == (16, 16) for a in r["attention"].values())
    lg = np.asarray(_ProbeBackend().run(_StubPolicy._input_transform(_obs()), source=ap.attention_source, rows=[1],
                                        want_feature=False, v_layer=None)["probe_logits"])
    want = HP.probe_map(lg)
    for c in HP.CAMERAS:
        assert np.array_equal(r["attention"][c], want[c])
    rec = r["attention_source"]
    assert rec["spec"] == "hidden_probe" and rec["query"] == "hidden_probe" and rec["attention"] == "ok"
    assert rec["layers"] == {"cam_high": 8, "cam_left_wrist": 12, "cam_right_wrist": 8}
    assert rec["asset_md5"] == HP.load_hidden_probe().md5 and rec["ms"] == 2.5 and rec["noun"] == "apple"
    assert rec["max"] == pytest.approx(float(HP.camera_probabilities(lg).max()), abs=1e-6)
    assert set(rec["max_score"]) == set(HP.CAMERAS) and rec["argmax_patch"]["camera"] == "cam_right_wrist"
    assert "hidden state only" in rec["prompt_use"]
    json.dumps(rec)                                                         # the record goes into summary_json
    assert ap.metadata["attention_source"] == "hidden_probe"
    assert be.calls[0]["rows"] is not None and "want_hsr5" not in be.calls[0]


def test_hidden_probe_noun_not_found_is_no_target_like_the_noun_source():
    r = _ap("hidden_probe").infer(_obs(prompt="pick up the thing and place it in the basket"))
    assert r["attention"] == {}
    rec = r["attention_source"]
    assert rec["attention"] == "empty" and rec["noun_ok"] is False and "fruit word" in rec["noun_error"]
    assert "max" not in rec and rec["ms"] == 2.5


def test_hidden_probe_with_hsr5_asks_for_the_stack_feature_once():
    from AG3S.trajopt import hsr5 as H5

    be = _ProbeBackend()
    r = _ap("hidden_probe", be, hsr5_probe=H5.load_hsr5_probe()).infer(_obs())
    assert be.calls[0]["want_hsr5"] is True and len(be.calls) == 1            # one backend call = one prefix forward
    assert "hsr5" in r and r["attention"]


def test_noun_source_is_unchanged_by_the_t90_code():
    """Source off / noun source: same record keys, no probe state, backend called as before (the t57a tests pin the rest)."""
    from tests.trajopt.test_t57a_attention_source import _FakeBackend

    be = _FakeBackend()
    r = _ap("noun_target:L10H7", be).infer(_obs())
    assert set(r["attention_source"]) == {"spec", "query", "layer", "head", "row", "sigma", "k", "noun", "noun_ok",
                                         "noun_error", "token_index", "prefix_row", "n_text_valid", "attention"}
    assert not hasattr(be, "set_hidden_probe") and set(be.calls[0]) == {"source", "rows", "want_feature", "v_layer"}
    from AG3S.trajopt.attention_policy import AttentionPolicy

    plain = AttentionPolicy(_StubPolicy(), None, sampler=object(), subtask_probe=None)
    assert plain._extended is False and plain._hidden_probe is None            # noqa: SLF001


def test_hidden_probe_source_parse_and_record():
    s = AS.parse_attention_source("hidden_probe")
    assert isinstance(s, AS.HiddenProbeSource) and s.path is None and s.is_noun and s.label == "hidden_probe"
    assert s.asset_path() == str(SHIPPED)
    p = AS.parse_attention_source(" hidden_probe:/x/y.npz ")
    assert p.path == "/x/y.npz" and p.asset_path() == "/x/y.npz"
    rec = s.record()
    assert rec["spec"] == "hidden_probe" and rec["asset_md5"] == HP.file_md5(SHIPPED)
    assert AS.seed_map_source_ok(s) and AS.seed_map_source_ok(AS.parse_attention_source("noun_target:L10H7"))
    assert not AS.seed_map_source_ok(AS.parse_attention_source("noun_target:L10H6"))
    assert not AS.seed_map_source_ok(AS.parse_attention_source("action:last:L8H2")) and not AS.seed_map_source_ok(None)
    for bad in ("hidden_probe_x", "hidden_probes", "hidden_probe:"):
        with pytest.raises(ValueError):
            AS.parse_attention_source(bad)


def test_bad_asset_stops_attention_policy_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ap(f"hidden_probe:{tmp_path / 'nope.npz'}")


# ================================================================================================ 5. serve_safe

BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
ATS = ["--target-extent", "attention-surface", "--target-field-policy", "exclude-authorized", "--esdf-backend", "curobo"]


def _parse(argv):
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    a = ap.parse_args(BASE + list(argv))
    serve_safe.reject_bad_flag_combinations(ap, a)
    return a


def test_serve_accepts_hidden_probe_alone_and_with_hsr5_and_attention_surface():
    from AG3S.trajopt import serve_safe

    a = _parse(["--attention-source", "hidden_probe"])
    meta = serve_safe.t57a_record_meta(a)["attention_source"]
    assert meta["spec"] == "hidden_probe" and meta["asset_md5"] == HP.file_md5(SHIPPED)
    _parse(["--attention-source", "hidden_probe", "--hsr5"])
    _parse(["--attention-source", f"hidden_probe:{SHIPPED}", *ATS])


def test_serve_off_and_noun_source_are_unchanged():
    from AG3S.trajopt import serve_safe

    a = _parse([])
    assert a.attention_source is None and serve_safe.t57a_record_meta(a) == {}
    n = _parse(["--attention-source", "noun_target:L10H7", "--hsr5", *ATS])
    assert serve_safe.t57a_record_meta(n) == {"attention_source": {"spec": "noun_target:L10H7", "query": "noun_target",
                                                                    "layer": 10, "head": 7, "row": "target", "sigma": 0.0,
                                                                    "k": 1}}


@pytest.mark.parametrize("argv", [
    ["--attention-source", "hidden_probe:/nonexistent.npz"],                  # asset unreadable: stop at start
    ["--attention-source", "hidden_probe", "--no-attention"],
    ["--attention-source", "hidden_probe", "--no-safe"],
    ["--attention-source", "noun_target:L10H6", *ATS],   # still the L10H7 / probe only
    ["--attention-source", "action:last:L8H2", "--hsr5"]])
def test_serve_refuses(argv):
    with pytest.raises(SystemExit):
        _parse(argv)
