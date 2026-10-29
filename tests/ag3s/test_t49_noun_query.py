"""T49 noun-token query, on synthetic data (no GPU, no checkpoint, no MuJoCo, no storage writes).

Pinned:
  - noun positions by piece id: exactly one valid match, 0 / >= 2 matches and padding handled, BOS
    check, ``last_text``; piece-id lookup rejects multi-piece / unknown nouns;
  - in-jit slicing (`slice_probs`) on a synthetic ``[L, B, K, G, T, S]`` softmax tensor: the image
    block equals ``probs[:, 0, 0, :, rows, :768]``, image + text mass = 1, BOS and self columns;
    the same under ``jax.numpy`` (skipped without jax) and through `prefix_rows_impl` with a
    stand-in model;
  - ``rows_to_maps`` camera order = `pi05_attention` (``CAMERA_BINDINGS`` slices, 16x16 row-major);
  - prompt swap (word boundary, exactly one occurrence), the variant plan;
  - GT-lift cell ranking finds the planted head; T47's label-free selection runs on noun rows;
  - separation and swap frame metrics on hand-made maps; record-level aggregation by phase;
  - the per-frame cache: resume, stamp mismatch, assembly; CLI argument checks and the
    ``AG3S_RUN_DIR`` refusal.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS, POLICY_CAMERA_NAMES
from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t49_noun_query as cli
from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

BOS = 2
IDS = {"apple": 15491, "banana": 31985, "basket": 12220, "orange": 10436, "pear": 21338}


def tokens(*pieces, pad_to=20):
    ids = [BOS, *pieces]
    mask = [True] * len(ids) + [False] * (pad_to - len(ids))
    return np.asarray(ids + [0] * (pad_to - len(ids))), np.asarray(mask)


# ------------------------------------------------------------------------ positions


def test_query_positions_single_match_and_last_text():
    ids, mask = tokens(7, 8, IDS["apple"], 9, IDS["basket"], 10, 11)
    pos, err = core.query_positions(ids, mask, {"target": IDS["apple"], "basket": IDS["basket"]},
                                    bos_id=BOS)
    assert err == []
    assert pos == {"target": 3, "basket": 5, "last_text": 7}
    np.testing.assert_array_equal(core.prefix_rows(pos), [771, 773, 775])


@pytest.mark.parametrize("pieces, why", [
    ((7, 8, 9, IDS["basket"]), "target: 0 matches"),
    ((IDS["apple"], 8, IDS["apple"], IDS["basket"]), "target: 2 matches"),
    ((IDS["apple"], 8, IDS["basket"], IDS["basket"]), "basket: 2 matches"),
])
def test_query_positions_zero_or_two_matches_are_errors(pieces, why):
    ids, mask = tokens(*pieces)
    pos, err = core.query_positions(ids, mask, {"target": IDS["apple"], "basket": IDS["basket"]})
    assert any(e.startswith(why) for e in err), err
    assert why.split(":")[0] not in pos


def test_query_positions_ignore_padding_and_check_bos():
    ids, mask = tokens(IDS["apple"], IDS["basket"], pad_to=10)
    ids[8] = IDS["apple"]  # a stray piece id inside the padding is not a match
    pos, err = core.query_positions(ids, mask, {"target": IDS["apple"], "basket": IDS["basket"]},
                                    bos_id=BOS)
    assert err == [] and pos["target"] == 1 and pos["last_text"] == 2
    ids[0] = 5
    _, err = core.query_positions(ids, mask, {"target": IDS["apple"], "basket": IDS["basket"]},
                                  bos_id=BOS)
    assert any("<bos>" in e for e in err)


class FakeSP:
    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.inv = {v: k for k, v in self.vocab.items()}

    def unk_id(self):
        return 0

    def piece_to_id(self, p):
        return self.vocab.get(p, 0)

    def id_to_piece(self, i):
        return self.inv.get(i, "<unk>")


def test_piece_ids_require_single_known_piece():
    sp = FakeSP({"▁apple": 15491, "▁basket": 12220, "<unk>": 0})
    assert core.piece_ids(sp, ["apple", "basket"]) == {"apple": 15491, "basket": 12220}
    with pytest.raises(ValueError, match="single SentencePiece"):
        core.piece_ids(sp, ["kiwi"])


def test_real_tokenizer_pieces_if_available():
    spm = pytest.importorskip("sentencepiece")
    from common.workspace import PROJECT_ROOT

    path = PROJECT_ROOT / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"
    if not path.exists():
        pytest.skip("tokenizer model not cached")
    sp = spm.SentencePieceProcessor(model_file=str(path))
    ids = core.piece_ids(sp, list(core.FRUITS) + ["basket"])
    assert ids == IDS
    for prompt, noun, t_idx, b_idx in (("Place the apple into the basket.", "apple", 5, 8),
                                       ("Pick up the pear and place it in the basket.", "pear", 6, 12)):
        toks = sp.encode(f"Task: {prompt}, State: 1 2 255;\nAction: ", add_bos=True)
        pos, err = core.query_positions(np.asarray(toks), np.ones(len(toks), bool),
                                        {"target": ids[noun], "basket": ids["basket"]},
                                        bos_id=sp.bos_id())
        assert err == [] and pos["target"] == t_idx and pos["basket"] == b_idx


# ------------------------------------------------------------------------- slicing


def synthetic_probs(L=3, G=2, T=None, S=None, seed=0, n_text=20):
    """Softmax-normalised ``[L, 1, 1, G, T, S]`` over a 768 + n_text prefix."""
    S = S or 768 + n_text
    T = T or S
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(L, 1, 1, G, T, S))
    p = np.exp(logits)
    return p / p.sum(-1, keepdims=True)


def test_slice_probs_numpy():
    probs = synthetic_probs()
    rows = np.asarray([771, 773, 780], np.int32)
    out = core.slice_probs(probs, rows)
    np.testing.assert_allclose(out["image"], probs[:, 0, 0][:, :, rows, :768], rtol=1e-6)
    assert out["image"].shape == (3, 2, 3, 768) and out["image"].dtype == np.float32
    np.testing.assert_allclose(out["image_mass"] + out["text_mass"], 1.0, rtol=1e-5)
    np.testing.assert_allclose(out["bos"], probs[:, 0, 0][:, :, rows, 768], rtol=1e-6)
    want_self = np.stack([probs[:, 0, 0, :, r, r] for r in rows], axis=-1)
    np.testing.assert_allclose(out["self"], want_self, rtol=1e-6)


def test_slice_probs_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    probs = synthetic_probs(seed=1)
    rows = np.asarray([770, 775, 787], np.int32)
    ref = core.slice_probs(probs, rows)
    got = jax.jit(lambda p, r: core.slice_probs(p, r, xp=jnp))(
        jnp.asarray(probs, jnp.float32), jnp.asarray(rows))
    for k in ref:
        np.testing.assert_allclose(np.asarray(got[k]), ref[k], rtol=1e-5, atol=1e-7)


def test_prefix_rows_impl_with_stand_in_model():
    jnp = pytest.importorskip("jax.numpy")
    probs = synthetic_probs(seed=2, n_text=12)
    seen = {}

    class LLM:
        def __call__(self, xs, mask, positions):
            seen["mask"], seen["positions"] = mask, positions
            return [None, None], None, jnp.asarray(probs)

    class Model:
        PaliGemma = types.SimpleNamespace(llm=LLM())

        def embed_prefix(self, obs):
            s = 768 + obs.tokenized_prompt.shape[1]
            return (jnp.zeros((1, s, 4)), jnp.ones((1, s), bool), jnp.zeros((s,), bool))

    obs = types.SimpleNamespace(tokenized_prompt=jnp.zeros((1, 12), jnp.int32))
    rows = jnp.asarray([770, 772, 779])

    def mam(mask, ar):  # bidirectional prefix
        return mask[:, None, :] & mask[:, :, None]

    out = nx.prefix_rows_impl(Model(), obs, rows, make_attn_mask=mam, xp=jnp)
    np.testing.assert_allclose(np.asarray(out["image"]), core.slice_probs(probs, np.asarray(rows))["image"],
                               rtol=1e-5)
    assert int(seen["positions"][0, -1]) == 768 + 12 - 1
    bad = types.SimpleNamespace(tokenized_prompt=jnp.zeros((1, 12), jnp.int32))

    class Short(Model):
        def embed_prefix(self, obs):
            return (jnp.zeros((1, 500, 4)), jnp.ones((1, 500), bool), jnp.zeros((500,), bool))

    with pytest.raises(ValueError, match="image tokens"):
        nx.prefix_rows_impl(Short(), bad, rows, make_attn_mask=mam, xp=jnp)


def test_rows_to_maps_camera_order_matches_pi05_attention():
    L, G, R = 2, 3, 2
    image = np.zeros((L, G, R, 768), np.float32)
    # token t of camera c gets value 1000 * c + t_local; rows / layers / heads offset it
    for cam_i, cam in enumerate(POLICY_CAMERA_NAMES):
        lo, hi = CAMERA_BINDINGS[cam][2]
        image[..., lo:hi] = 1000 * cam_i + np.arange(256)
    image += (np.arange(L)[:, None, None, None] * 1e5 + np.arange(G)[None, :, None, None] * 1e4
              + np.arange(R)[None, None, :, None] * 1e6)
    maps = core.rows_to_maps(image)
    assert maps.shape == (R, L, G, 3, 16, 16)
    for r in range(R):
        for l in range(L):
            for g in range(G):
                for c, cam in enumerate(POLICY_CAMERA_NAMES):
                    lo, hi = CAMERA_BINDINGS[cam][2]
                    # pi05_attention.main: pooled[..., lo:hi].reshape(L, H, 16, 16)
                    np.testing.assert_array_equal(maps[r, l, g, c],
                                                  image[l, g, r, lo:hi].reshape(16, 16))
    assert maps[0, 0, 0, 1, 0, 1] == 1000 + 1  # camera 1, patch (0, 1)
    assert maps[0, 0, 0, 2, 1, 0] == 2000 + 16  # camera 2, patch (1, 0): row-major


def test_fields_from_rows_and_empty_fields():
    probs = synthetic_probs(L=2, G=2, n_text=10)
    rows = np.asarray([770, 772, 777], np.int32)
    out = {k: np.asarray(v) for k, v in core.slice_probs(probs, rows).items()}
    toks = np.arange(10) + 100
    f = nx.fields_from_rows(out, {"target": 2, "basket": 4, "last_text": 9}, toks)
    assert f["maps"].shape == (3, 2, 2, 3, 16, 16) and f["maps"].dtype == np.float16
    np.testing.assert_array_equal(f["prefix_row"], [770, 772, 777])
    np.testing.assert_array_equal(f["token_id"], [102, 104, 109])
    np.testing.assert_allclose(f["image_mass"][1], out["image_mass"][:, :, 1])
    e = nx.empty_fields(2, 2)
    assert set(e) == set(nx.FRAME_FIELDS) and all(e[k].shape == f[k].shape for k in e)


# --------------------------------------------------------------------------- prompts


def test_swap_prompt_and_plan():
    assert core.swap_prompt("Place the apple into the basket.", "apple", "pear") == \
        "Place the pear into the basket."
    with pytest.raises(ValueError):
        core.swap_prompt("the pineapple and the pear", "apple", "pear")  # no whole-word match
    with pytest.raises(ValueError):
        core.swap_prompt("apple then apple", "apple", "pear")
    plan = core.variant_plan("Pick up the banana and place it in the basket.", "banana", swaps=True)
    assert [p["variant"] for p in plan] == ["orig", "swap_apple", "swap_orange", "swap_pear"]
    assert plan[1]["prompt"] == "Pick up the apple and place it in the basket."
    assert plan[1]["noun"] == "apple" and plan[1]["swapped_from"] == "banana"
    assert [p["variant"] for p in core.variant_plan("x apple", "apple", swaps=False)] == ["orig"]


# -------------------------------------------------------------------- cell ranking


def planted(F=4, L=4, H=3, planted_cell=(2, 1), seed=0):
    """Block (F, C, L, H, 16, 16) and coverage (F, C, 7, 16, 16) with the target at rows 2-4."""
    rng = np.random.default_rng(seed)
    block = rng.uniform(0.5, 1.0, (F, 3, L, H, 16, 16)) * 1e-3
    cov = np.zeros((F, 3, 7, 16, 16))
    cov[:, :, 0, 2:5, 3:6] = 1.0
    cov[:, :, 1, 10:13, 10:13] = 1.0
    cov[:, :, 6] = 1.0 - cov[:, :, 0] - cov[:, :, 1]
    l, h = planted_cell
    block[:, :, l, h, 2:5, 3:6] += 0.05
    block[:, :, 1, 0, 10:13, 10:13] += 0.05  # a crate head
    return block, cov


def test_gt_ranking_finds_planted_head_and_destination():
    block, cov = planted()
    keep = np.ones(block.shape[:2], bool)
    met = core.cell_metrics([block], [cov], [keep], target_index=0)
    order = core.rank_by_lift(met["lift"], met["peak_on_target"])
    assert order[0] == (2, 1)
    assert met["peak_on_target"][2, 1] == 1.0
    met_d = core.cell_metrics([block], [cov], [keep], target_index=1)
    assert core.rank_by_lift(met_d["lift"], met_d["peak_on_target"])[0] == (1, 0)
    # keep masks matter: nothing kept -> NaN everywhere, ranking falls back to index order
    none = core.cell_metrics([block], [cov], [np.zeros_like(keep)], target_index=0)
    assert np.isnan(none["lift"]).all() and none["n"] == 0
    assert core.rank_by_lift(none["lift"], none["peak_on_target"])[:2] == [(0, 0), (0, 1)]


def test_rank_by_lift_tie_break_on_peak():
    lift = np.asarray([[1.0, 2.0], [2.0, np.nan]])
    peak = np.asarray([[0.0, 0.2], [0.5, 1.0]])
    assert core.rank_by_lift(lift, peak) == [(1, 0), (0, 1), (0, 0), (1, 1)]


def test_label_free_selection_on_noun_rows():
    """T47's procedure via `_lf_select` picks a compact, image-heavy head and never L0/L1."""
    rng = np.random.default_rng(3)
    F, L, H = 6, 5, 4
    b = rng.uniform(0.9, 1.1, (F, L, H, 3, 16, 16)) * 1e-4  # scattered, low mass
    b[:, 0, 0, :, 4:8, 4:8] += 0.2  # L0: perfect but excluded
    b[:, 3, 2, :, 4:8, 4:8] += 0.1  # the designed head: compact blob, high mass
    args = types.SimpleNamespace(tau_method="kneedle", n_lowest=2, kmax=3)
    res = cli._lf_select({"long": b, "t2": b.copy()}, ["long", "t2"], args)
    top = res["cam"]["common"]["topk"]["1"]
    assert top == [[3, 2]]
    assert all(c[0] >= 2 for c in res["cam"]["common"]["topk"]["3"])
    assert res["cam"]["leave_one_record_out"][0]["overlap_with_full"] == 3


def test_combined_is_t47_combine_maps():
    block, _ = planted()
    from AG3S.ag3s.experiments.studies.t47_head_metrics import combine_maps

    for s in (0.0, 1.0):
        np.testing.assert_array_equal(core.combined(block, [[2, 1], [1, 0]], s),
                                      combine_maps(block, [(2, 1), (1, 0)], s))


# ------------------------------------------------------------------ mass summary (#1)


def test_mass_summary():
    N, L, H = 5, 3, 2
    im = np.full((N, L, H), 0.3)
    im[:, 2] = 0.001
    bos = np.full((N, L, H), 0.5)
    tx = 1.0 - im
    cam = np.stack([im * 0.5, im * 0.25, im * 0.25], axis=-1)
    m = core.mass_summary(im, tx, bos, np.full((N, L, H), 0.1), cam, min_image_mass=0.01)
    pl = m["per_layer_median"]
    np.testing.assert_allclose(pl["image"], [0.3, 0.3, 0.001])
    np.testing.assert_allclose(pl["text_ex_bos"], [0.2, 0.2, 0.499])
    np.testing.assert_allclose(pl["camera_share_of_image"]["cam_high"], [0.5] * 3)
    assert m["layers_image_mass_below"] == [2]
    assert m["image_plus_text_max_abs_dev"] < 1e-12


# ------------------------------------------------------------------- separation (#5)


def frame_gt():
    cov = np.zeros((3, 7, 16, 16))
    cov[0, 0, 2, 3] = 1.0  # target in cam_high patch (2, 3)
    cov[1, 1, 8, 8] = 1.0  # crate in the left wrist patch (8, 8)
    px = np.zeros((3, 7), int)
    px[0, 0], px[1, 1] = 500, 900
    return cov, px


def test_separation_frame_hits_and_mixing():
    cov, px = frame_gt()
    t = np.full((3, 16, 16), 1e-3)
    b = np.full((3, 16, 16), 1e-3)
    t[0, 2, 3] = 1.0
    b[1, 8, 8] = 1.0
    r = core.separation_frame(t, b, cov, px, min_px=200)
    assert r["target_peak_on_target"] and r["basket_peak_on_crate"] and r["both_correct"]
    assert not r["target_peak_on_crate"] and not r["basket_peak_on_target"]
    assert r["target_peak"] == [0, 2, 3] and r["basket_peak"] == [1, 8, 8]
    tot = t.sum()
    assert math.isclose(r["target_map_target_mass"], 1.0 / tot)
    assert math.isclose(r["target_map_crate_mass"], 1e-3 / tot)
    # a basket map that peaks on the target: not separated, and it is counted as cross-talk
    r2 = core.separation_frame(t, t, cov, px, min_px=200)
    assert not r2["both_correct"] and r2["basket_peak_on_target"] and r2["same_peak_patch"]
    # visibility
    px2 = px.copy()
    px2[1, 1] = 50
    r3 = core.separation_frame(t, b, cov, px2, min_px=200)
    assert not r3["crate_visible"] and not r3["both_visible"]


def test_separation_record_level_by_phase():
    rows = []
    for rec, scene in (("headfix", "same_scene"), ("ep1850", "new_scene")):
        for i, (pg, both) in enumerate((("pre_grasp", True), ("pre_grasp", False),
                                        ("post_grasp", True))):
            rows.append({"source": "primary", "record": rec, "scene_group": scene,
                         "phase_group": pg, "target_visible": True, "crate_visible": True,
                         "both_visible": rec == "headfix" or i != 1,
                         "target_peak_on_target": True, "target_peak_on_crate": False,
                         "basket_peak_on_crate": both, "basket_peak_on_target": False,
                         "both_correct": both, "same_peak_patch": False,
                         "target_map_target_mass": 0.5, "target_map_crate_mass": 0.1,
                         "basket_map_crate_mass": 0.4, "basket_map_target_mass": 0.2})
    s = core.summarize_records(rows, core.SEPARATION_RATES, core.SEPARATION_MEANS)["primary"]
    assert math.isclose(s["all"]["per_record"]["headfix"]["both_correct_rate"], 2 / 3)
    assert math.isclose(s["all"]["per_record"]["ep1850"]["both_correct_rate"], 1.0)  # frame 1 ineligible
    assert math.isclose(s["all"]["record_mean"]["both_correct_rate"], (2 / 3 + 1.0) / 2)
    assert math.isclose(s["phase=pre_grasp"]["per_record"]["headfix"]["both_correct_rate"], 0.5)
    assert s["scene=new_scene"]["n_records"] == 1
    assert math.isclose(s["all"]["record_mean"]["basket_map_target_mass"], 0.2)


# ------------------------------------------------------------------------ swap (#7)


def swap_gt():
    """Named bodies crate, apple, banana: apple (orig) at cam 0 (2, 3), banana (new) at cam 0 (9, 9)."""
    cov = np.zeros((3, 3, 16, 16))
    cov[0, 1, 2, 3] = 1.0
    cov[0, 2, 9, 9] = 1.0
    cov[1, 0, 5, 5] = 1.0
    px = np.zeros((3, 3), int)
    px[0, 1], px[0, 2], px[1, 0] = 400, 300, 800
    return cov, px


def test_swap_frame_moved_and_not_moved():
    cov, px = swap_gt()
    orig = np.full((3, 16, 16), 1e-3)
    orig[0, 2, 3] = 1.0
    swap = np.full((3, 16, 16), 1e-3)
    swap[0, 9, 9] = 1.0
    r = core.swap_frame(orig, swap, cov, px, new_index=2, orig_index=1, min_px=200)
    assert r["moved"] and r["swap_peak_on_new"] and not r["swap_peak_on_orig"]
    assert r["orig_peak_on_orig"] and r["peak_patch_changed"]
    assert r["swap_lift_new"] > 100 and r["orig_lift_new"] < 1 and r["lift_new_gain"] > 100
    r2 = core.swap_frame(orig, orig.copy(), cov, px, new_index=2, orig_index=1, min_px=200)
    assert not r2["moved"] and not r2["peak_patch_changed"] and math.isclose(r2["map_corr"], 1.0)
    assert r2["lift_new_gain"] == 0.0
    px_hidden = px.copy()
    px_hidden[0, 2] = 10
    r3 = core.swap_frame(orig, swap, cov, px_hidden, new_index=2, orig_index=1, min_px=200)
    assert not r3["new_visible"] and math.isnan(r3["swap_lift_new"])


def test_map_corr():
    a = np.arange(12.0).reshape(3, 2, 2)
    assert math.isclose(core.map_corr(a, 2 * a + 1), 1.0)
    assert math.isnan(core.map_corr(a, np.ones_like(a)))


# ----------------------------------------------------------------------- frame cache


def fake_steps(n):
    return [types.SimpleNamespace(t_step=8 * i) for i in range(n)]


def test_extract_record_resume_and_assembly(tmp_path):
    stamp = nx.stamp_of(variant="orig", key="ep1850", prompt="p", noun="banana", checkpoint="c",
                        config="cfg", fingerprint="f", n_frames=3)
    calls = []

    def run_frame(step):
        calls.append(step.t_step)
        f = nx.empty_fields(2, 2)
        f["maps"][:] = step.t_step
        ok = step.t_step != 8
        return {"ok": ok, "error": "" if ok else "target: 0 matches", "fields": f,
                "prefix_ms": 1.0, "n_text_valid": 70}

    fdir, final = tmp_path / "x.frames", tmp_path / "x.npz"
    # first run "crashes" after frame 0 -> only that frame file is left behind
    steps = fake_steps(3)
    nx.write_frame(nx.frame_path(fdir, 0), stamp=stamp, t_step=0, ok=True, error="",
                   fields=run_frame(steps[0])["fields"], prefix_ms=1.0, n_text_valid=70)
    calls.clear()
    res = nx.extract_record(frames_dir=fdir, final_path=final, stamp=stamp, steps=steps,
                            run_frame=run_frame, log=lambda s: None)
    assert calls == [8, 16] and res["resumed"] == 1 and res["computed"] == 2
    assert res["n_errors"] == 1 and not fdir.exists()
    ex = nx.load_extract(final)
    assert ex["maps"].shape == (3, 3, 2, 2, 3, 16, 16)
    np.testing.assert_array_equal(ex["ok"], [True, False, True])
    np.testing.assert_array_equal(ex["t_step"], [0, 8, 16])
    assert str(ex["noun"]) == "banana" and ex["stamp_dict"]["record"] == "ep1850"
    assert nx.query_block(ex, "basket").shape == (3, 2, 2, 3, 16, 16)
    assert float(nx.query_block(ex, "target")[2].max()) == 16.0


def test_cached_frame_with_other_stamp_is_refused(tmp_path):
    stamp = nx.stamp_of(variant="orig", key="k", prompt="p", noun="apple", checkpoint="c",
                        config="cfg", fingerprint="f", n_frames=1)
    fp = nx.frame_path(tmp_path, 0)
    nx.write_frame(fp, stamp=stamp, t_step=0, ok=True, error="", fields=nx.empty_fields(1, 1),
                   prefix_ms=1.0, n_text_valid=3)
    with pytest.raises(SystemExit, match="cached frame"):
        nx.read_frame(fp, {**stamp, "prompt": "other"})


# ----------------------------------------------------------------------------- CLI


def test_cli_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        cli.main(["--stage", "select"])


@pytest.mark.parametrize("argv", [
    ["--stage", "e2e", "--workers", "3"],
    ["--stage", "e2e", "--sources", "nope"],
    ["--stage", "select", "--kmax", "2"],
    ["--stage", "select", "--k", "6"],
    ["--stage", "swap", "--sigma", "-1"],
    ["--stage", "extract", "--records", "nope"],
])
def test_cli_argument_checks(argv):
    ap = cli.build_parser()
    with pytest.raises(SystemExit):
        cli.validate_args(ap, ap.parse_args(argv))


def test_e2e_refuses_gt_chosen_sources_on_selection_records(monkeypatch, tmp_path):
    ap = cli.build_parser()
    args = ap.parse_args(["--stage", "e2e", "--records", "long", "--sources", "noun_gt"])
    monkeypatch.setattr(cli, "t47_dir", lambda a: tmp_path)
    monkeypatch.setattr(cli, "load_select", lambda r: {})
    monkeypatch.setattr(cli, "t47_oracle_cells", lambda t: [[8, 3]])
    monkeypatch.setattr(hs, "available_keys", lambda keys, log=print: (list(keys), []))
    with pytest.raises(SystemExit, match="selection"):
        cli.stage_e2e(args, tmp_path)


def test_cellsets_reads_select_json():
    sel = {"queries": {
        "target__vs_target": {"gt_topk": {"3": [[8, 3], [9, 1], [4, 0]]},
                              "label_free": {"cam": {"common": {"topk": {"3": [[4, 1], [2, 0], [5, 5]]}}}}},
        "basket__vs_destination": {"gt_topk": {"3": [[7, 7], [6, 6], [5, 5]]},
                                   "label_free": {"cam": {"common": {"topk": {"3": [[3, 3], [2, 2], [4, 4]]}}}}}}}
    cs = cli.cellsets(sel, 3)
    assert cs["noun_gt"] == [[8, 3], [9, 1], [4, 0]] and cs["basket_lf"][0] == [3, 3]
