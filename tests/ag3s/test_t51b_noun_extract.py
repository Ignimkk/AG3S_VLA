"""T51b noun-row extraction for the extra calibration runs (`t51b_noun_extract`) — no GPU, no params.

Pinned:
  - the plan item is T49's ``orig`` item from the `t51_records` registry (prompt = the step record's
    meta prompt, noun = its target), T47 keys are refused (T49 has them), non-fruit targets refused;
  - the per-frame body equals T49's (`fields_from_rows` on the found noun rows; a frame whose noun is
    not found exactly once is ``ok = False`` with zero maps, the first such frame learns (L, H) from a
    dummy BOS pass; the first successful call is a separate, untimed compile pass);
  - `extract_one` writes T49's record npz (stamp, keys, dtypes), resumes / skips a finished file with
    the same stamp, refuses another stamp, and `limit` makes a parity-length stamp;
  - the format check against a T49-style reference, the record check (t_step vs record / attention npz,
    prompt, failed frames) and the parity numbers;
  - CLI refusals (no run dir, other config, wrong checkpoint step, extract of a T47 key before any GPU
    work); on the real T49 extracts (when storage is present) the format check finds nothing.
"""

from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
from AG3S.ag3s.experiments.studies import t51_records as rc
from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

IDS = {"apple": 15491, "banana": 31985, "basket": 12220, "orange": 10436, "pear": 21338}
BOS = 2
PROMPT = "Put the banana in the basket."


@pytest.fixture(autouse=True)
def _isolated_extra_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_EXTRA_DIR", tmp_path / "T51")


def fake_record(tmp_path, monkeypatch, *, prompt=PROMPT, episode=500, n=3, attention=True,
                n_att=None):
    rec = tmp_path / "rec" / "run_0000"
    rec.mkdir(parents=True)
    argv = ["pi05_infer.py", "--episode-index", str(episode)]
    (rec / "meta.json").write_text(json.dumps({"prompt": prompt, "argv": argv}))
    for i in range(n):
        np.savez(rec / f"step_{i:05d}.npz", t_step=np.asarray(i * 8))
    monkeypatch.setattr(rc, "record_path", lambda key: rec)
    if attention:
        na = n if n_att is None else n_att
        p = rc.npz_path("ep500")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, attention=np.zeros((na, 3, 3, 18, 8, 3, 16, 16), np.float16),
            t_step=np.arange(na, dtype=np.int64) * 8, denoise_steps=np.asarray([0, 4, 9]),
            aggregations=np.asarray(["mean", "first", "last"]),
            cameras=np.asarray(list(hs.POLICY_CAMERAS)), prompt=np.asarray(prompt),
            noise_seeds=np.asarray([0]), records=np.asarray(str(rec.resolve())),
            checkpoint=np.asarray("/s/checkpoints/rby1/" + rc.CHECKPOINT_TAIL))
    return rec


def tokens_for(*pieces, pad_to=16):
    """BOS + pieces (ids from `IDS`, other words -> 900+) + padding, and the mask."""
    ids = [BOS] + [IDS.get(p, 900 + i) for i, p in enumerate(pieces)]
    toks = np.zeros(pad_to, np.int64)
    toks[: len(ids)] = ids
    mask = np.zeros(pad_to, bool)
    mask[: len(ids)] = True
    return toks, mask


class FakeExtractor:
    """`NounQueryExtractor.rows` stand-in: image columns encode the requested prefix rows."""

    def __init__(self, n_l=18, n_h=8):
        self.n_l, self.n_h = n_l, n_h
        self.t_call: list[float] = []
        self.calls: list[tuple[list[int], bool]] = []

    def rows(self, obs, rows, *, timed=True):
        rows = np.asarray(rows)
        self.calls.append((rows.tolist(), timed))
        r = len(rows)
        img = np.zeros((self.n_l, self.n_h, r, core.N_IMAGE_TOKENS), np.float32)
        img += (rows.astype(np.float32) / 1000.0)[None, None, :, None]
        img[..., 7] += 0.5
        if timed:
            self.t_call.append(0.002)
        z = np.ones((self.n_l, self.n_h, r), np.float32)
        return {"image": img, "image_mass": img.sum(-1), "text_mass": 0.5 * z, "bos": 0.1 * z,
                "self": 0.01 * z}


def observe_factory(token_sets):
    """`observe(step, prompt)` returning the tokenized prompt of `token_sets[step.t_step // 8]`."""
    def observe(step, prompt):
        toks, mask = token_sets[int(step.t_step) // 8]
        return types.SimpleNamespace(tokenized_prompt=toks[None], tokenized_prompt_mask=mask[None])
    return observe


GOOD = tokens_for("Put", "the", "banana", "in", "the", "basket", ".,")
BAD = tokens_for("Put", "the", "fruit", "in", "the", "basket", ".,")  # no noun piece


# ---------------------------------------------------------------------------------- plan


def test_plan_item_is_t49_orig_from_registry(tmp_path, monkeypatch):
    fake_record(tmp_path, monkeypatch)
    item = ne.plan_item("ep500")
    assert item == {"key": "ep500", "target": "banana", "variant": "orig", "prompt": PROMPT,
                    "noun": "banana", "swapped_from": None}
    assert core.variant_plan(PROMPT, "banana", swaps=False)[0]["variant"] == ne.VARIANT


def test_plan_item_refuses_unparsable_or_non_fruit_target(tmp_path, monkeypatch):
    fake_record(tmp_path, monkeypatch, prompt="Put the crate in the basket.", attention=False)
    with pytest.raises(SystemExit):
        ne.plan_item("ep500")


def test_extra_keys_refuse_t47_records():
    assert ne.extra_keys(None) == list(rc.EXTRA_KEYS)
    with pytest.raises(SystemExit, match="T49 already extracted"):
        ne.extra_keys(["ep0", "long"])


def test_pairing_with_and_without_attention(tmp_path, monkeypatch):
    fake_record(tmp_path, monkeypatch, attention=False)
    assert ne.pairing("ep500")["attention_npz"] is None
    fake_record(tmp_path / "b", monkeypatch)
    assert ne.pairing("ep500")["scene_id"] == "scene500"
    fake_record(tmp_path / "c", monkeypatch, episode=501, attention=False)
    with pytest.raises(SystemExit, match="episode"):
        ne.pairing("ep500")


# ----------------------------------------------------------------------------- frame body


def test_run_frame_matches_t49_body():
    ex = FakeExtractor()
    state: dict = {}
    item = {"key": "ep500", "prompt": PROMPT, "noun": "banana"}
    rf = ne.make_run_frame(observe=observe_factory([GOOD, BAD]), extractor=ex, ids=IDS, bos=BOS,
                           item=item, state=state)
    res = rf(types.SimpleNamespace(t_step=0))
    pos, err = nx.lookup(GOOD[0], GOOD[1], "banana", IDS, BOS)
    assert not err and res["ok"] and res["error"] == ""
    rows = core.prefix_rows(pos)
    assert ex.calls == [(rows.tolist(), False), (rows.tolist(), True)]  # compile pass, then timed
    assert state["compile_s"] is not None and (state["n_l"], state["n_h"]) == (18, 8)
    want = nx.fields_from_rows(ex.rows(None, rows, timed=False), pos, GOOD[0])
    for k in nx.FRAME_FIELDS:
        np.testing.assert_array_equal(res["fields"][k], want[k])
    assert res["prefix_ms"] == pytest.approx(2.0) and res["n_text_valid"] == int(GOOD[1].sum())
    bad = rf(types.SimpleNamespace(t_step=8))
    assert not bad["ok"] and "target: 0 matches" in bad["error"]
    assert not bad["fields"]["maps"].any() and np.isnan(bad["prefix_ms"])


def test_first_frame_failure_learns_shape_from_bos_pass():
    ex = FakeExtractor(n_l=4, n_h=3)
    state: dict = {}
    rf = ne.make_run_frame(observe=observe_factory([BAD]), extractor=ex, ids=IDS, bos=BOS,
                           item={"key": "ep500", "prompt": PROMPT, "noun": "banana"}, state=state)
    res = rf(types.SimpleNamespace(t_step=0))
    assert ex.calls == [([core.N_IMAGE_TOKENS] * 3, False)]
    assert res["fields"]["maps"].shape == (3, 4, 3, 3, 16, 16) and not res["ok"]


# ----------------------------------------------------------------------------- record npz


def _extract(tmp_path, token_sets, *, variant="orig", limit=0, prompt=PROMPT, force=False):
    steps = [types.SimpleNamespace(t_step=8 * i) for i in range(len(token_sets))]
    item = {"key": "ep500", "target": "banana", "variant": "orig", "prompt": prompt, "noun": "banana",
            "swapped_from": None}
    calls = []
    rf0 = ne.make_run_frame(observe=observe_factory(token_sets), extractor=FakeExtractor(), ids=IDS,
                            bos=BOS, item=item, state={})

    def rf(step):
        calls.append(step.t_step)
        return rf0(step)

    res = ne.extract_one(item=item, record=types.SimpleNamespace(steps=steps), root=tmp_path / "out",
                         variant=variant, checkpoint="/s/checkpoints/rby1/" + rc.CHECKPOINT_TAIL,
                         config=rc.CONFIG, fingerprint="fp", run_frame=rf, limit=limit, force=force,
                         log=lambda s: None)
    return res, calls


def test_extract_one_writes_t49_record_and_skips_cached(tmp_path):
    res, calls = _extract(tmp_path, [GOOD, BAD, GOOD])
    assert calls == [0, 8, 16] and res["n_errors"] == 1
    path = ne.extract_path(tmp_path / "out", "ep500")
    assert path == tmp_path / "out" / "orig" / "ep500.npz"
    ex = nx.load_extract(path)
    assert ex["maps"].shape == (3, 3, 18, 8, 3, 16, 16) and ex["maps"].dtype == np.float16
    assert ex["stamp_dict"] == nx.stamp_of(variant="orig", key="ep500", prompt=PROMPT, noun="banana",
                                           checkpoint="/s/checkpoints/rby1/" + rc.CHECKPOINT_TAIL,
                                           config=rc.CONFIG, fingerprint="fp", n_frames=3)
    np.testing.assert_array_equal(ex["ok"], [True, False, True])
    assert [str(q) for q in ex["query_names"]] == list(core.QUERY_NAMES)
    res2, calls2 = _extract(tmp_path, [GOOD, BAD, GOOD])
    assert res2["cached"] and calls2 == []
    with pytest.raises(SystemExit, match="--force"):
        _extract(tmp_path, [GOOD, BAD, GOOD], prompt="Put the banana into the basket.")
    res3, calls3 = _extract(tmp_path, [GOOD, GOOD, GOOD], force=True)
    assert calls3 == [0, 8, 16] and res3["n_errors"] == 0


def test_extract_one_limit_is_a_parity_stamp(tmp_path):
    _extract(tmp_path, [GOOD, GOOD, GOOD], variant="parity", limit=2)
    ex = nx.load_extract(ne.extract_path(tmp_path / "out", "ep500", "parity"))
    assert ex["stamp_dict"]["n_frames"] == 2 and len(ex["t_step"]) == 2


# ----------------------------------------------------------------------------- checks


def _t49_style_reference(tmp_path, n=2):
    """A reference written by T49's own frame cache path (`extract_record` + `fields_from_rows`)."""
    stamp = nx.stamp_of(variant="orig", key="ep1807", prompt="Pick up the apple and place it in the "
                        "basket.", noun="apple", checkpoint="/s/checkpoints/rby1/" + rc.CHECKPOINT_TAIL,
                        config=rc.CONFIG, fingerprint="x", n_frames=n)
    toks, mask = tokens_for("Pick", "up", "the", "apple", "and", "place", "it", "in", "the", "basket")
    pos, _ = nx.lookup(toks, mask, "apple", IDS, BOS)
    out = FakeExtractor().rows(None, core.prefix_rows(pos), timed=False)

    def run_frame(step):
        return {"ok": True, "error": "", "fields": nx.fields_from_rows(out, pos, toks),
                "prefix_ms": 1.0, "n_text_valid": int(mask.sum())}

    p = tmp_path / "ref" / "ep1807.npz"
    nx.extract_record(frames_dir=tmp_path / "ref" / "f", final_path=p, stamp=stamp,
                      steps=[types.SimpleNamespace(t_step=8 * i) for i in range(n)],
                      run_frame=run_frame, log=lambda s: None)
    return nx.load_extract(p)


def test_format_problems_against_t49_style_reference(tmp_path):
    ref = _t49_style_reference(tmp_path)
    _extract(tmp_path, [GOOD, BAD, GOOD])
    mine = nx.load_extract(ne.extract_path(tmp_path / "out", "ep500"))
    assert ne.format_problems(mine, ref) == []
    bad = dict(mine, maps=mine["maps"].astype(np.float32))
    assert any("maps dtype" in p for p in ne.format_problems(bad, ref))
    bad = dict(mine, stamp_dict={**mine["stamp_dict"], "checkpoint": "/other/29999"})
    assert any("stamp checkpoint" in p for p in ne.format_problems(bad, ref))
    bad = dict(mine, image_mass=mine["image_mass"][:, :2])
    assert any("image_mass shape" in p for p in ne.format_problems(bad, ref))
    bad = {k: v for k, v in mine.items() if k != "bos"}
    assert "missing bos" in ne.format_problems(bad, ref)


def test_record_problems(tmp_path, monkeypatch):
    fake_record(tmp_path, monkeypatch)
    _extract(tmp_path, [GOOD, GOOD, GOOD])
    ex = nx.load_extract(ne.extract_path(tmp_path / "out", "ep500"))
    assert ne.record_t_steps("ep500") == [0, 8, 16]
    assert ne.attention_t_steps("ep500") == [0, 8, 16]
    assert ne.record_problems("ep500", ex, [0, 8, 16], [0, 8, 16]) == []
    assert any("step record" in p for p in ne.record_problems("ep500", ex, [0, 8], None))
    assert any("attention" in p for p in ne.record_problems("ep500", ex, [0, 8, 16], [0, 8, 24]))
    _extract(tmp_path / "e", [GOOD, BAD, GOOD])
    ex_bad = nx.load_extract(ne.extract_path(tmp_path / "e" / "out", "ep500"))
    assert any("ok=False" in p for p in ne.record_problems("ep500", ex_bad, [0, 8, 16], None))
    fake_record(tmp_path / "p", monkeypatch, prompt="Put the banana into the basket.")
    assert any("stamp prompt" in p for p in ne.record_problems("ep500", ex, [0, 8, 16], None))


def test_compare_noun():
    rng = np.random.default_rng(0)
    a = {"t_step": np.arange(2), "token_index": np.ones((2, 3), int), "token_id": np.ones((2, 3), int),
         "ok": np.ones(2, bool), "query_names": np.asarray(core.QUERY_NAMES),
         "maps": rng.random((2, 3, 2, 2, 3, 16, 16)).astype(np.float16),
         **{f: rng.random((2, 3, 2, 2)).astype(np.float32) for f in ("image_mass", "text_mass", "bos",
                                                                     "self")}}
    c = ne.compare_noun(a, dict(a))
    assert c["t_step_equal"] and c["queries"]["target"]["bitwise_equal"]
    assert c["queries"]["target"]["argmax_agree"] == 1.0 and c["image_mass_max_abs"] == 0.0
    b = dict(a, maps=a["maps"].copy())
    b["maps"][0, 0, 0, 0, 0] = 0
    b["maps"][0, 0, 0, 0, 0, 3, 3] = 9
    c = ne.compare_noun(b, a)
    n = 2 * 2 * 2 * 3
    assert c["queries"]["target"]["argmax_agree"] == pytest.approx(1 - 1 / n)
    assert c["queries"]["basket"]["bitwise_equal"] and not c["queries"]["target"]["bitwise_equal"]


# ------------------------------------------------------------------------------- CLI


def test_cli_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        ne.main(["--stage", "check"])


def test_cli_argument_checks():
    ap = ne.build_parser()
    for argv in (["--stage", "extract", "--config", "pi05_other"], ["--stage", "parity",
                                                                   "--parity-frames", "0"],
                 ["--stage", "extract", "--records", "nope"]):
        with pytest.raises(SystemExit):
            ne.validate_args(ap, ap.parse_args(argv))


def test_checkpoint_must_be_the_t49_step(tmp_path):
    a = ne.build_parser().parse_args(["--stage", "extract", "--checkpoint", str(tmp_path)])
    with pytest.raises(SystemExit, match="not step"):
        ne.checkpoint_path(a)


def test_extract_refuses_t47_keys_before_gpu(monkeypatch, tmp_path):
    def boom(args):
        raise AssertionError("GPU session must not start")

    monkeypatch.setattr(ne, "_gpu_session", boom)
    a = ne.build_parser().parse_args(["--stage", "extract", "--records", "ep1807"])
    with pytest.raises(SystemExit, match="T49 already extracted"):
        ne.stage_extract(a, tmp_path)
    a = ne.build_parser().parse_args(["--stage", "parity", "--parity-record", "ep0"])
    with pytest.raises(SystemExit, match="T47 key"):
        ne.stage_parity(a, tmp_path)


def test_real_t49_extracts_have_one_format():
    """On storage: T49's own orig extracts pass the format check against each other (no false
    positives from string widths or frame counts)."""
    t49 = hs.storage_root() / ne.T49_RUN / "verify" / "T49"
    a, b = ne.t49_extract_path(t49, "ep1807"), ne.t49_extract_path(t49, "ep1850")
    if not (a.is_file() and b.is_file()):
        pytest.skip("T49 run not in storage")
    ra, rb = nx.load_extract(a), nx.load_extract(b)
    assert ne.format_problems(rb, ra) == []
    assert ra["stamp_dict"]["checkpoint"].endswith(rc.CHECKPOINT_TAIL)
    assert ra["stamp_dict"]["config"] == rc.CONFIG
