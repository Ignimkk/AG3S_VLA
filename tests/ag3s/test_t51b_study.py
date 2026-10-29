"""T51b study wiring (`t51b_study`) on synthetic data — no MuJoCo, no GPU, no storage writes.

Pinned:
  - `install` rebinds exactly `PATCHED` in `t51_study` and `uninstall` restores T51's behaviour (the
    noun query is refused for the extra runs again);
  - noun rows for an extra run come from the T51b noun extract and are checked (t_step vs record and
    attention npz, prompt, failed frames); a noun-only Ω never loads the action attention; combined
    returns both, in the requested order;
  - Ω per ``--omega``; t51_study's own ``filter`` and ``score2d`` stages run on the noun-only and the
    combined Ω of two synthetic extra runs (noun cells only / 512 cells, forced cells only when they are
    in Ω);
  - the row cache: T51b first, T51 read-only fallback, writes only under T51b; self label from T51;
  - the replay-worker entry is picklable, rebinds in a fresh process (spawn) before the original
    ``_replay_job`` runs; replay options are checked before T51's rows are reused;
  - (c) helpers: query family, exact McNemar, paired per-frame counts; ``heldout_compare`` and
    ``summary`` end to end on fake ksel / bootstrap / rows; CLI refusals.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import pathlib
import pickle

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
from AG3S.ag3s.experiments.studies import t51_protocol as pr
from AG3S.ag3s.experiments.studies import t51_records as rc
from AG3S.ag3s.experiments.studies import t51_study as st
from AG3S.ag3s.experiments.studies import t51b_study as tb

NOUN = st.NOUN_QUERY
PROMPTS = {"ep500": "Put the banana in the basket.", "ep0": "Pick up the orange and place it in the "
                                                             "basket.",
           "ep1200": "Place the pear into the basket."}
TARGET = {"ep500": "banana", "ep0": "orange", "ep1200": "pear"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_EXTRA_DIR", tmp_path / "T51")
    yield
    tb.uninstall()


def _cfg(tmp_path, omega="noun", **kw) -> tb.Config:
    base = tmp_path / "T51b"
    d = dict(omega=omega, t49_dir=str(tmp_path / "T49"), noun_extra_dir=str(base / "noun_extract"),
             t51_dir=str(tmp_path / "T51"), rows_dir=str(base / "rows"))
    d.update(kw)
    return tb.Config(**d)


def _noun_npz(path, key, t_steps, *, rng, ok=None, prompt=None):
    stamp = nx.stamp_of(variant="orig", key=key, prompt=prompt or PROMPTS[key], noun=TARGET[key],
                        checkpoint="/s/checkpoints/rby1/" + rc.CHECKPOINT_TAIL, config=rc.CONFIG,
                        fingerprint="fp", n_frames=len(t_steps))
    frames = []
    for i, t in enumerate(t_steps):
        f = nx.empty_fields(18, 8)
        m = 0.01 * rng.random(f["maps"].shape)
        m[..., 4:6, 5:7] += 1.0  # a small blob: passes the P1 filter (not broad, not a sink)
        f["maps"][:] = m.astype(np.float16)
        good = True if ok is None else bool(ok[i])
        frames.append({**f, "t_step": t, "ok": good, "error": "" if good else "target: 0 matches",
                       "prefix_ms": 1.0, "n_text_valid": 12})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **nx.assemble_frames(frames, stamp))


def _world(tmp_path, monkeypatch, keys=("ep500",), n=3, *, noun=True, noun_ok=None, seed=0):
    """Synthetic extra runs: step record, T51 attention npz, GT cache, and T51b noun extract."""
    rng = np.random.default_rng(seed)
    recs = {}
    for key in keys:
        rec = tmp_path / "rec" / key / "run_0000"
        rec.mkdir(parents=True)
        argv = ["pi05_infer.py", "--episode-index", str(rc.EXTRA_RECORDS[key].episode)]
        (rec / "meta.json").write_text(json.dumps({"prompt": PROMPTS[key], "argv": argv}))
        for i in range(n):
            np.savez(rec / f"step_{i:05d}.npz", t_step=np.asarray(i * 8))
        recs[key] = rec
        p = rc.npz_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, attention=rng.random((n, 3, 3, 18, 8, 3, 16, 16)).astype(np.float16),
            t_step=np.arange(n, dtype=np.int64) * 8, denoise_steps=np.asarray([0, 4, 9]),
            aggregations=np.asarray(["mean", "first", "last"]),
            cameras=np.asarray(list(hs.POLICY_CAMERAS)), prompt=np.asarray(PROMPTS[key]),
            noise_seeds=np.asarray([0]), records=np.asarray(str(rec.resolve())),
            checkpoint=np.asarray("/s/checkpoints/rby1/" + rc.CHECKPOINT_TAIL))
        cov = np.zeros((n, 3, 4, 16, 16))
        cov[:, :, 0, 4:8, 4:8] = 1.0
        cov[:, :, 3] = 1.0 - cov[:, :, 0]
        px = np.zeros((n, 3, 4), np.int64)
        px[:, :, 0] = 500
        g = rc.gt_cache_path(key)
        g.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(g, class_cov=cov, class_px=px, t_step=np.arange(n) * 8)
        if noun:
            _noun_npz(tb.ne.extract_path(tmp_path / "T51b" / "noun_extract", key), key,
                      list(range(0, 8 * n, 8)), rng=rng, ok=noun_ok)
    monkeypatch.setattr(rc, "record_path", lambda key: recs[key])
    return recs


def _fake_t47(tmp_path) -> pathlib.Path:
    t47 = tmp_path / "T47"
    (t47 / "select").mkdir(parents=True)
    (t47 / "validate").mkdir(parents=True)
    (t47 / "gt_cache").mkdir(parents=True)
    sel = {"settings": {"d0_last": {"tau_source": {"cam": {"common": {"topk": {"3": [[4, 1], [2, 0],
                                                                                   [14, 5]]}}}}}}}
    (t47 / "select" / "selection.json").write_text(json.dumps(sel))
    (t47 / "validate" / "oracle.json").write_text(json.dumps(
        {"settings": {"d0_last": {"top3": [[8, 3], [10, 7], [12, 4]]}}}))
    return t47


# ------------------------------------------------------------------------------ rebinding


def test_install_rebinds_and_uninstall_restores(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    orig = {n: getattr(st, n) for n in tb.PATCHED}
    with tb.installed(_cfg(tmp_path)):
        for n in tb.PATCHED:
            assert getattr(st, n) is not orig[n], n
        assert st.query_blocks is tb.query_blocks
    for n in tb.PATCHED:
        assert getattr(st, n) is orig[n], n
    with pytest.raises(SystemExit, match="T49"):  # T51's behaviour is back
        st.query_blocks("ep500", ["last", NOUN], t49=str(tmp_path))


def test_noun_rows_for_extra_run(tmp_path, monkeypatch):
    _world(tmp_path, monkeypatch)
    want = nx.query_block(nx.load_extract(tb.ne.extract_path(tmp_path / "T51b" / "noun_extract",
                                                             "ep500")), "target")
    with tb.installed(_cfg(tmp_path)):
        out = st.query_blocks("ep500", [NOUN, "last"], None)
        assert list(out) == [NOUN, "last"]
        np.testing.assert_array_equal(out[NOUN], want)
        att, dn, ag, _ = rc.load_attention("ep500")
        np.testing.assert_array_equal(out["last"], att[:, dn.index(0), ag.index("last")])
        monkeypatch.setattr(rc, "load_attention", lambda key: (_ for _ in ()).throw(AssertionError))
        assert list(st.query_blocks("ep500", [NOUN])) == [NOUN]  # noun-only: no attention load
        with pytest.raises(SystemExit, match="unknown query"):
            st.query_blocks("ep500", ["nope"])


def test_noun_dispatch_t47_vs_extra(tmp_path):
    cfg = _cfg(tmp_path)
    assert tb.noun_path("ep1807", cfg) == tmp_path / "T49" / "extract" / "orig" / "ep1807.npz"
    assert tb.noun_path("ep0", cfg) == tmp_path / "T51b" / "noun_extract" / "orig" / "ep0.npz"


@pytest.mark.parametrize("case", ["missing", "failed_frame", "attention_t_step", "prompt"])
def test_noun_rows_refused_when_they_do_not_pair(tmp_path, monkeypatch, case):
    _world(tmp_path, monkeypatch, noun=case not in ("missing",),
           noun_ok=[True, False, True] if case == "failed_frame" else None)
    if case == "attention_t_step":
        p = rc.npz_path("ep500")
        with np.load(p) as d:
            arrs = {k: d[k] for k in d.files}
        arrs["t_step"] = arrs["t_step"] + 1
        np.savez_compressed(p, **arrs)
    if case == "prompt":
        _noun_npz(tb.ne.extract_path(tmp_path / "T51b" / "noun_extract", "ep500"), "ep500", [0, 8, 16],
                  rng=np.random.default_rng(1), prompt="Put the banana into the basket.")
    match = {"missing": "t51b_noun_extract", "failed_frame": "ok=False",
             "attention_t_step": "attention", "prompt": "stamp prompt"}[case]
    with tb.installed(_cfg(tmp_path)):
        with pytest.raises(SystemExit, match=match):
            st.query_blocks("ep500", [NOUN])


def test_queries_per_omega(tmp_path):
    with tb.installed(_cfg(tmp_path, omega="noun")):
        assert st.queries(argparse.Namespace(omega="noun")) == [NOUN]
        assert st.queries(argparse.Namespace(omega="combined")) == ["last", "mean", "first", NOUN]
        assert len(pr.omega(st.queries(argparse.Namespace(omega="combined")))) == 512


def _args(stage, **kw):
    a = tb.build_parser().parse_args(["--stage", stage])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("omega", ["noun", "combined"])
def test_t51_filter_and_score2d_run_on_t51b_omega(tmp_path, monkeypatch, omega):
    _world(tmp_path, monkeypatch, keys=("ep500", "ep0"))
    t47 = _fake_t47(tmp_path)
    root = tmp_path / "T51b" / omega
    monkeypatch.setattr(hs, "git_state", lambda: {})
    with tb.installed(_cfg(tmp_path, omega=omega)):
        a = _args("filter", omega=omega, records=["ep500", "ep0"], t47_dir=str(t47),
                  t49_dir=str(tmp_path / "T49"))
        f = st.stage_filter(a, root)
        assert f["step"] == "T51b" and f["t51b"]["omega"] == omega
        assert set(f["queries"]) == set(tb.OMEGAS[omega])
        n_cells = sum(v["n_cells"] for v in f["queries"].values())
        assert n_cells == (128 if omega == "noun" else 512)
        a = _args("score2d", omega=omega, records=["ep500", "ep0"], t47_dir=str(t47),
                  t49_dir=str(tmp_path / "T49"), label="gt")
        s = st.stage_score2d(a, root)
    assert (root / "filter" / "filter.json").is_file() and (root / "gt" / "score2d" / "score2d.json").is_file()
    cand = s["candidates_3d"]
    if omega == "noun":
        assert cand and all(c.startswith(NOUN + ":") for c in cand)
        assert s["global"]["forced"] == []
    else:
        assert "last:L8H2" in cand  # forced (action) cells stay forced in the combined Ω
    assert not (tmp_path / "T51" / "filter").exists()  # nothing written into T51


# ------------------------------------------------------------------------------ rows / labels


def _row(frame, ident):
    return {"frame": frame, "identity_correct": ident, "on_target": ident, "has_target": True,
            "chosen_cluster_purity": 0.8 if ident else 0.2, "nontarget_seed_loss": 0 if ident else 10,
            "nontarget_seed_loss__other_object": 0, "self_labeled": False,
            "self_identity_correct": None, "self_out_n": None, "phase": "approach"}


def _write(path: pathlib.Path, key, sid, idents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i, g in enumerate(idents):
            fh.write(json.dumps({**_row(i, bool(g)), "source": sid, "record": key,
                                 "scene_group": rc.scene_group(key)}) + "\n")


def test_rows_read_through_t51_and_write_only_t51b(tmp_path):
    cfg = _cfg(tmp_path)
    with tb.installed(cfg):
        t51p = tb.t51_rows_path("ep1850", "last:L8H2")
        _write(t51p, "ep1850", "last:L8H2", [1, 0])
        assert tb.row_origin("ep1850", "last:L8H2") == "t51"
        assert [r["identity_correct"] for r in st.load_rows(tmp_path, "ep1850", "last:L8H2")] == [True, False]
        assert st._complete(tmp_path, "ep1850", "last:L8H2", 2)  # noqa: SLF001
        p = st.rows_path(tmp_path / "anything", "ep1850", "last:L8H2")
        assert p == tmp_path / "T51b" / "rows" / "ep1850" / "last_L8H2.jsonl"
        assert st.rows_path(tmp_path, "ep1850", "last:L8H2", smoke=True).parent.parent.name == "rows_smoke"
        st._write_rows(p, [_row(0, False), _row(1, False)], "ep1850")  # noqa: SLF001
        assert tb.row_origin("ep1850", "last:L8H2") == "t51b"
        assert [r["identity_correct"] for r in st.load_rows(tmp_path, "ep1850", "last:L8H2")] == [False, False]
        assert st.load_rows(tmp_path, "ep1850", "noun_target:L8H2") is None
        assert st.selflabel_path(tmp_path, "ep0") == tmp_path / "T51" / "selflabel" / "ep0.npz"
    t51_rows = [json.loads(x) for x in t51p.read_text().splitlines() if x.strip()]
    assert [r["identity_correct"] for r in t51_rows] == [True, False]  # T51's file untouched
    with tb.installed(_cfg(tmp_path, t51_rows=False)):
        assert tb.row_origin("ep1900", "last:L8H2") == "missing"
        _write(tb.t51_rows_path("ep1900", "last:L8H2"), "ep1900", "last:L8H2", [1])
        assert st.load_rows(tmp_path, "ep1900", "last:L8H2") is None


def test_replay_entry_rebinds_before_original(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    seen = {}

    def fake_original(*args, **kw):
        seen["qb"] = st.query_blocks
        seen["args"] = args
        return "done"

    monkeypatch.setitem(tb._ORIG, "_replay_job", fake_original)  # noqa: SLF001
    tb.uninstall()
    assert tb._replay_entry(cfg, "ep0", "root") == "done"  # noqa: SLF001
    assert seen["qb"] is tb.query_blocks and seen["args"] == ("ep0", "root")
    job = st._replay_job  # noqa: SLF001
    back = pickle.loads(pickle.dumps(job))
    assert back.func is tb._replay_entry and back.args == (cfg,)  # noqa: SLF001


def test_replay_worker_rebinds_in_spawned_process(tmp_path):
    """A spawned worker gets a fresh t51_study; the entry must rebind before ``_replay_job`` runs:
    the noun lookup of an extra run then fails with T51b's hint, not with T51's refusal."""
    cfg = _cfg(tmp_path)
    with tb.installed(cfg):
        job = st._replay_job  # noqa: SLF001
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        fut = ex.submit(job, "ep0", str(tmp_path / "root"), [f"{NOUN}:L8H2"], {}, str(tmp_path),
                        None, 1, 0, False, str(tmp_path / "T51"))
        with pytest.raises(SystemExit, match="t51b_noun_extract"):
            fut.result(timeout=120)


def test_check_replay_opts(tmp_path):
    t51 = tmp_path / "T51"
    args = _args("heldout_compare")
    assert tb.check_replay_opts(t51, args) == ()
    (t51 / "rows").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no T51 payload"):
        tb.check_replay_opts(t51, args)
    p = t51 / "gt" / "score3d" / "score3d.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"args": {k: getattr(args, k) for k in tb.REPLAY_OPTS}}))
    assert tb.check_replay_opts(t51, args) == ("gt/score3d/score3d.json",)
    p.write_text(json.dumps({"args": {**{k: getattr(args, k) for k in tb.REPLAY_OPTS}, "voxel": 0.01}}))
    with pytest.raises(SystemExit, match="voxel"):
        tb.check_replay_opts(t51, args)


# ------------------------------------------------------------------------------ (c) helpers


def test_family_and_mcnemar():
    assert tb.family("last:L8H2") == "action" and tb.family(f"{NOUN}:L5H1") == "noun"
    assert tb.family(f"sum[{NOUN}:L5H1+last:L8H2]") == "action+noun"
    assert tb.family("paper3_s1") == "special" and tb.family("headavg_last") == "special"
    assert tb.mcnemar_exact(0, 0) == 1.0
    assert tb.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert tb.mcnemar_exact(3, 3) == 1.0


def test_paired_identity_counts():
    a = {"ep1850": [_row(0, True), _row(1, True), _row(2, False), _row(3, False)],
         "ep1800": [_row(0, True)]}
    b = {"ep1850": [_row(0, True), _row(1, False), _row(2, True), _row(3, False), _row(4, True)],
         "ep1800": [_row(0, False)]}
    out = tb.paired_identity(a, b, ["ep1850", "ep1800"])
    r = out["per_record"]["ep1850"]
    assert (r["both"], r["only_a"], r["only_b"], r["neither"]) == (1, 1, 1, 1)
    assert r["unmatched_frames"] == 1 and r["a_minus_b"] == 0.0
    assert out["per_record"]["ep1800"]["a_minus_b"] == 1.0
    assert out["pooled"]["all"]["n_frames"] == 5 and out["pooled"]["new_scene"]["n_frames"] == 4
    assert out["pooled"]["same_scene"]["only_a"] == 1
    assert out["record_mean_a_minus_b"] == pytest.approx(0.5)


def _ksel(path: pathlib.Path, combo):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"selected": {"k": len(pr.combo_cells(combo)), "combo": combo,
                                             "cells": [c.id for c in pr.combo_cells(combo)]},
                                "nested_mean": 0.5, "cv_mean_ci95": {}, "single_head_fixed": True,
                                "ranking_all_calibration_top10": [combo]}))


def test_heldout_compare_and_summary_end_to_end(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, omega=None)
    base = tmp_path / "T51b"
    _ksel(tmp_path / "T51" / "gt" / "ksel" / "ksel.json", "last:L9H1")
    _ksel(base / "noun" / "gt" / "ksel" / "ksel.json", f"{NOUN}:L8H5")
    _ksel(base / "combined" / "gt" / "ksel" / "ksel.json", f"{NOUN}:L8H5")
    keys = list(rc.HELDOUT_KEYS)
    vals = {"last:L9H1": [1, 0, 1], f"{NOUN}:L8H5": [1, 1, 1], "last:L8H2": [0, 0, 1]}
    with tb.installed(cfg):
        for k in keys:
            for s, v in vals.items():
                p = tb.t51_rows_path(k, s) if s.startswith("last") else st.rows_path(tmp_path, k, s)
                _write(p, k, s, v)
        monkeypatch.setattr(rc, "check_pairing", lambda k: {"key": k})
        monkeypatch.setattr(rc, "n_frames", lambda k: 3)
        monkeypatch.setattr(st, "t47_dir", lambda args: tmp_path)
        monkeypatch.setattr(hs, "git_state", lambda: {})
        a = _args("heldout_compare", labels=["gt", "self"])
        out = tb.stage_heldout_compare(a, base / "heldout_compare")
        assert out["named"] == {"action_best_gt": "last:L9H1", "noun_best_gt": f"{NOUN}:L8H5",
                                "combined_best_gt": f"{NOUN}:L8H5", "l8h2": "last:L8H2"}
        assert out["family"]["noun_best_gt"] == "noun"
        assert out["record_mean_gt_identity"]["noun_best_gt"] == 1.0
        assert out["record_mean_gt_identity"]["action_best_gt"] == pytest.approx(2 / 3)
        pv = out["paired"]["noun_best_gt__vs__action_best_gt"]
        assert pv["pooled"]["all"]["only_a"] == len(keys) and pv["pooled"]["all"]["only_b"] == 0
        assert out["row_origin"]["last:L9H1"]["ep1850"] == "t51"
        assert out["row_origin"][f"{NOUN}:L8H5"]["ep1850"] == "t51b"
        assert (base / "heldout_compare" / "heldout_compare.json").is_file()
        bs = base / "noun" / "gt" / "bootstrap" / "bootstrap.json"
        bs.parent.mkdir(parents=True)
        bs.write_text(json.dumps({"unit": "scene", "n_units": 5, "full_sample_top1": f"{NOUN}:L8H5",
                                  "full_sample_top1_p_rank1": 0.7, "stable_single_cell": True,
                                  "all": [{"cell": f"{NOUN}:L8H5", "p_rank1": 0.7},
                                          {"cell": f"{NOUN}:L9H5", "p_rank1": 0.3}]}))
        bc = base / "combined" / "gt" / "bootstrap" / "bootstrap.json"
        bc.parent.mkdir(parents=True)
        bc.write_text(json.dumps({"unit": "scene", "full_sample_top1": "last:L8H2",
                                  "all": [{"cell": f"{NOUN}:L8H5", "p_rank1": 0.4},
                                          {"cell": "last:L8H2", "p_rank1": 0.6}]}))
        summ = tb.stage_summary(_args("summary", labels=["gt"]), base / "summary")
    nb = summ["omegas"]["noun"]["gt"]["bootstrap"]
    assert nb["p_rank1_by_family"] == {"noun": pytest.approx(1.0)}
    cb = summ["omegas"]["combined"]["gt"]["bootstrap"]
    assert cb["p_rank1_by_family"] == {"noun": pytest.approx(0.4), "action": pytest.approx(0.6)}
    assert cb["full_sample_top1_family"] == "action"
    assert summ["omegas"]["action_t51"]["gt"]["ksel"]["family"] == "action"
    assert summ["heldout_compare"]["named"]["noun_best_gt"] == f"{NOUN}:L8H5"


def test_heldout_compare_needs_both_selections(tmp_path):
    _ksel(tmp_path / "T51" / "gt" / "ksel" / "ksel.json", "last:L9H1")
    with pytest.raises(SystemExit, match="noun_best_gt"):
        tb.selections(tmp_path / "T51", tmp_path / "T51b", ["gt"])


# ------------------------------------------------------------------------------- CLI


def test_cli_checks(monkeypatch, tmp_path):
    ap = tb.build_parser()
    for argv in (["--stage", "filter"], ["--stage", "heldout_compare", "--omega", "noun"],
                 ["--stage", "score3d", "--omega", "noun", "--workers", "3"]):
        with pytest.raises(SystemExit):
            tb.validate_args(ap, ap.parse_args(argv))
    for bad in (["--stage", "selflabel", "--omega", "noun"], ["--stage", "gt_extra"]):
        with pytest.raises(SystemExit):
            ap.parse_args(bad)
    tb.validate_args(ap, ap.parse_args(["--stage", "ksel", "--omega", "combined", "--label", "self"]))
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        tb.main(["--stage", "summary"])
    import common.run_output as ro

    monkeypatch.setattr(ro, "resolve_output_path",
                        lambda path, *, run_dir, default_name: pathlib.Path(run_dir) / path)
    monkeypatch.setenv("AG3S_RUN_DIR", str(tmp_path / "run"))
    with pytest.raises(SystemExit, match="selflabel"):
        tb.main(["--stage", "summary"])
    t51 = tmp_path / "run" / "verify" / "T51"
    (t51 / "selflabel").mkdir(parents=True)
    (tmp_path / "T49" / "extract" / "orig").mkdir(parents=True)
    monkeypatch.setattr(hs, "git_state", lambda: {})
    (t51 / "rows").mkdir()
    with pytest.raises(SystemExit, match="no T51 payload"):  # T51 rows without recorded options
        tb.main(["--stage", "summary", "--t49-dir", str(tmp_path / "T49")])
    tb.main(["--stage", "summary", "--t49-dir", str(tmp_path / "T49"), "--no-t51-rows"])
    out = json.loads((tmp_path / "run" / "verify" / "T51b" / "summary" / "summary.json").read_text())
    assert out["step"] == "T51b" and out["t51b"]["t51_rows"] is False
    assert out["omegas"]["noun"]["gt"] == {"score3d": None, "bootstrap": None, "ksel": None}
