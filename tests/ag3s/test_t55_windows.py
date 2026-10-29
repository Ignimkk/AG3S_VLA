"""T55 auxiliary T60 windows (`t55_windows`): mask loading, window application, window == truncation,
pooled decision rows, parity helpers, and one real full-window parity check on two T53 episodes."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t55_study as st
from AG3S.ag3s.experiments.studies import t55_windows as tw
from AG3S.ag3s.experiments.studies import t60_termination as T

F = 20
CSS = np.arange(F, dtype=np.int64) * 8


def _write_mask(path: pathlib.Path, t_p, t_h, *, tamper: bool = False) -> pathlib.Path:
    kp = T.keep_frames(CSS, t_p)
    kh = T.keep_frames(CSS, t_h)
    if tamper:
        kp = kp.copy()
        kp[0] = not kp[0]
    np.savez(path, protocol=np.asarray(T.PROTOCOL_VERSION), t_P=np.asarray(-1 if t_p is None else t_p),
             t_H=np.asarray(-1 if t_h is None else t_h), chunk_start_steps=CSS, keep_frames_P=kp, keep_frames_H=kh)
    return path


def test_load_mask_rederives_keep(tmp_path):
    m = tw.load_mask(_write_mask(tmp_path / "a.npz", 80, 120))
    assert m["t_P"] == 80 and m["t_H"] == 120
    assert m["P"].sum() == 11 and m["H"].sum() == 16          # t_step 0..80 / 0..120
    none = tw.load_mask(_write_mask(tmp_path / "b.npz", None, None))
    assert none["P"].all() and none["H"].all()                 # no P -> full horizon kept
    with pytest.raises(SystemExit):
        tw.load_mask(_write_mask(tmp_path / "c.npz", 80, 120, tamper=True))


def test_mask_path_requires_exactly_one(tmp_path):
    (tmp_path / "masks").mkdir()
    with pytest.raises(SystemExit):
        tw.mask_path(tmp_path, "ep1")
    _write_mask(tmp_path / "masks" / "T53__C1a_ep1_s11.npz", 80, None)
    assert tw.mask_path(tmp_path, "ep1").name == "T53__C1a_ep1_s11.npz"


# ---------------------------------------------------------------- synthetic traces


def _trace(key: str, tag: str, ptag: str, noun: str, *, ws: int = -1, hold=(5, 13), held_phase=(6, 14),
           crate_from: int = 15) -> st.Trace:
    hb = {"apple": [("right" if hold[0] <= i < hold[1] else None) for i in range(F)], "pear": [None] * F}
    if tag.startswith("empty"):
        hb["apple"] = [None] * F
    phase = np.array(["searching"] * F, dtype="U10")
    phase[held_phase[0] - 1] = "closing"
    phase[held_phase[0]:held_phase[1]] = "held"
    ic = np.arange(F) >= crate_from
    window = np.ones(F, bool) if ws < 0 else np.arange(F) >= ws
    return st.Trace(set_name="test", key=key, tag=tag, prompt_tag=ptag, condition=st.condition_of(tag, ptag),
                    group=f"test:{key}", truth_kind="physics", used_arm="right", noun=noun, window=window,
                    held_by=hb, in_crate=ic,
                    latch={"reground": {"phase": phase, "attach": np.zeros(F, bool), "detach": np.zeros(F, bool)}},
                    gripper_used=np.where((np.arange(F) >= 5) & (np.arange(F) < 14), 0.2, 1.0), meta={})


def _feats(seed: int, n: int = F) -> dict:
    r = np.random.default_rng(seed)
    x = r.uniform(0, 1, size=(n, 2)).astype(np.float32)
    return {"kv_l4": x, "noun_v": x[:, None, :], "noun_hidden_final": x, "noun_layers": [0],
            "noun_ok": np.ones(n, bool)}


class _Probe:
    def proba(self, X):
        p = np.clip(np.asarray(X, np.float64)[:, 0], 0, 1)
        return np.stack([1 - p, p], 1)


PREREG = {"truth": {"K": 2, "persistence": "causal"},
          "arms": {"hsr": {"tau": 0.5, "p": 1}, "noun": {"tau": 0.4, "p": 2, "probe": "x", "feature": "v0"}},
          "bootstrap": {"B": 200, "seed": 5507, "moving_block_frames": 5},
          "tolerance": {"false_reject_rate": 0.05, "latency_frames": 1.0, "undetected_events": 0.0}}


@pytest.fixture()
def fake_sources(monkeypatch):
    monkeypatch.setattr(st, "hsr_place_proba", lambda kv, probe=None: (np.asarray(kv)[:, 1].astype(np.float64),
                                                                       np.asarray(kv)[:, 1] > 0.5))
    monkeypatch.setattr(st, "load_probe", lambda path: (_Probe(), "v0"))


def _episodes():
    trs = []
    for e, key in enumerate(("ep1", "ep2", "ep3")):
        trs += [_trace(key, "E0", "orig", "apple"), _trace(key, "E0", "instr_pear", "pear"),
                _trace(key, "empty_L0", "orig", "apple", ws=4), _trace(key, "slip_A2", "orig", "apple", ws=8)]
    return trs


def _truncate(tr: st.Trace, n: int) -> st.Trace:
    return dataclasses.replace(tr, window=tr.window[:n], held_by={f: v[:n] for f, v in tr.held_by.items()},
                               in_crate=tr.in_crate[:n], gripper_used=tr.gripper_used[:n],
                               latch={m: {k: np.asarray(v)[:n] for k, v in lt.items()} for m, lt in tr.latch.items()})


def _masks(t_p):
    return {k: {"P": T.keep_frames(CSS, t), "H": np.ones(F, bool), "chunk_start_steps": CSS}
            for k, t in zip(("ep1", "ep2", "ep3"), t_p)}


def test_apply_window_same_mask_for_every_trace_and_arm():
    trs = _episodes()
    masks = _masks((96, 72, None))
    win = tw.apply_window(trs, masks, "P")
    for a, b in zip(trs, win):
        keep = masks[a.key]["P"]
        assert np.array_equal(b.window, a.window & keep)       # condition window AND episode mask
        assert b.latch is a.latch and b.held_by is a.held_by    # the arms read the same latch / truth inputs
    # the original traces are untouched; full = identity
    assert all(t.window.sum() in (F, F - 4, F - 8) for t in trs)
    assert all(np.array_equal(a.window, b.window) for a, b in zip(trs, tw.apply_window(trs, masks, "full")))
    with pytest.raises(ValueError):
        tw.episode_keep(masks["ep1"], "X", F)


def test_check_alignment_refuses_other_t_step():
    trs = _episodes()
    masks = _masks((96, 72, None))
    ok = tw.check_alignment(trs, masks, t_steps={t.tid: CSS for t in trs})
    assert ok["traces_checked"] == len(trs)
    bad = {t.tid: CSS for t in trs}
    bad[trs[0].tid] = CSS + 1
    with pytest.raises(SystemExit):
        tw.check_alignment(trs, masks, t_steps=bad)


def test_window_score_equals_truncated_trace(fake_sources):
    """Windows are prefixes and every filter is causal: masking == truncating each episode at t_end."""
    trs = _episodes()
    t_p = (96, 72, None)
    masks = _masks(t_p)
    cache = st.FeatureCache()
    for i, t in enumerate(trs):
        cache.put(t, _feats(i))
    got = tw.score_window(tw.apply_window(trs, masks, "P"), PREREG, cache, ["reground"])["reground"]
    n_keep = {k: int(masks[k]["P"].sum()) for k in masks}
    cut = [_truncate(t, n_keep[t.key]) for t in trs]
    cache2 = st.FeatureCache()
    for i, (t, c) in enumerate(zip(trs, cut)):
        f = _feats(i)
        cache2.put(c, {k: (v[:n_keep[t.key]] if isinstance(v, np.ndarray) else v) for k, v in f.items()})
    want = st.score_test(cut, PREREG, cache2, mode="reground", B=200, seed=5507, block=5)
    assert tw.first_difference(tw.canonical(got), tw.canonical(want)) is None
    # and the window actually removed frames (ep1: 13 of 20 kept, ep2: 10, ep3: all)
    assert n_keep == {"ep1": 13, "ep2": 10, "ep3": 20}
    full = tw.score_window(trs, PREREG, cache, ["reground"])["reground"]
    assert full["by"]["natural"]["p_task"]["strong"]["n_neg"] > got["by"]["natural"]["p_task"]["strong"]["n_neg"]


def test_pooled_decision_sums_trace_counts_over_conditions():
    rows = []
    base = {k: 0.0 for k in st.COUNT_KEYS}
    for g, cond, c in (("test:a", "natural", {"neg": 10, "pos": 5, "fp_s": 2, "fp_a": 0, "fr_s": 0, "fr_a": 1,
                                                  "lat_sum": 1.0, "lat_n": 1, "events": 1}),
                       ("test:a", "instruction", {"neg": 20, "fp_s": 4, "fp_a": 1}),
                       ("test:b", "natural", {"neg": 10, "pos": 5, "fp_s": 1, "fp_a": 1, "lat_n": 1, "events": 1}),
                       ("test:b", "empty", {"neg": 10})):
        for arm in ("hsr", "noun"):
            rows.append({"condition": cond, "predicate": "p_task", "arm": arm, "group": g, **base, **c})
            rows.append({"condition": cond, "predicate": "p_grasp", "arm": arm, "group": g, **base, **c,
                         "fp_a": 99})                                             # never read for p_task
    d = tw.pooled_decision({"per_episode": rows}, PREREG)
    h = d["hsr"]
    assert h["counts"]["neg"] == 50 and h["counts"]["pos"] == 10
    assert h["pooled"]["d_fp_rate"] == pytest.approx((2 - 7) / 50)            # (fp_a − fp_s) / pooled neg
    assert h["pooled"]["d_fr_rate"] == pytest.approx(1 / 10)
    assert h["pooled"]["d_latency_mean"] == pytest.approx(0.5)
    assert h["tolerance_checks"] == {"false_reject": False, "latency": True, "undetected": True}
    assert h["within_tolerance"] is False and h["fewer_false_passes"] is True
    assert h["episode_bootstrap"]["_spec"]["n_groups"] == 2 and h["conditions"] == ["empty", "instruction", "natural"]


def test_parity_helpers():
    assert tw.canonical({"a": float("nan"), "b": np.float64(1.5), "c": np.arange(2)}) == {"a": None, "b": 1.5,
                                                                                           "c": [0, 1]}
    assert tw.first_difference({"x": [1, 2.0]}, {"x": [1, 2.0]}) is None
    assert "x[1]" in tw.first_difference({"x": [1, 2.0]}, {"x": [1, 2.5]})
    assert tw.first_difference({"x": 1}, {"y": 1}).startswith(": keys")


# ---------------------------------------------------------------- real parity (T55 test run, two episodes)

T55_ROOT = pathlib.Path(tw.DEFAULT_T55_RUN) / "verify" / "T55"
STEP1 = pathlib.Path(tw.DEFAULT_T60_STEP1)


@pytest.mark.skipif(not (T55_ROOT / "test_results.json").is_file() or not (STEP1 / "masks").is_dir(),
                    reason="T55 test run / T60 masks not available")
def test_real_full_window_rows_match_test_results():
    prereg = json.loads((T55_ROOT / st.PREREG_NAME).read_text())
    stored = json.loads((T55_ROOT / "test_results.json").read_text())
    keys = ("ep1804", "ep1840")
    traces = [t for t in tw.load_test_traces(T55_ROOT, prereg) if t.key in keys]
    masks = tw.load_masks(STEP1, keys)
    tw.check_alignment(traces, masks)
    res = st.score_test(traces, prereg, st.FeatureCache(), mode="reground", B=10, seed=5507, block=5)
    want = {(r["trace"], r["predicate"], r["arm"]): r for r in stored["results"]["reground"]["per_episode"]
            if r["group"] in {f"test:{k}" for k in keys}}
    got = {(r["trace"], r["predicate"], r["arm"]): r for r in tw.canonical(res["per_episode"])}
    assert got.keys() == want.keys() and len(got) == 2 * 2 * len(traces)
    for k in want:
        assert got[k] == want[k], k
    # the P window of ep1804 (t_P 312) keeps 40 of 75 frames
    assert masks["ep1804"]["t_P"] == 312 and int(masks["ep1804"]["P"].sum()) == 40
    assert masks["ep1840"]["t_P"] is None and masks["ep1840"]["P"].all()
