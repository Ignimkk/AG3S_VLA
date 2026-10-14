"""T56 CLI: dev set, dev stage → prereg, test stage reads the prereg only (synthetic records)."""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t56_selfscore as ss
from AG3S.ag3s.experiments.studies import t56_study as st
from tests.ag3s.test_t56_selfscore import held_event, make_geom

KEYS = ["e0", "e1", "e2"]


def _rd(key, seed, *, event=True):
    g, names = make_geom(seed=seed)
    n_f = len(g.frames)
    gt = {m: np.full((len(names), n_f), np.nan) for m in ("identity", "purity", "coverage", "no_target")}
    base = {"big": 0.6, "good": 0.9, "nbr": 0.1, "none": 0.0}
    for a, nm in enumerate(names):
        gt["identity"][a] = base[nm] + 0.01 * seed
        gt["purity"][a] = base[nm]
        gt["coverage"][a] = base[nm]
        gt["no_target"][a] = 1.0 if nm == "none" else 0.0
    ev = held_event(6, n_f)
    if not event:
        from AG3S.ag3s.experiments.studies import t56_held as hd

        ev = hd.HeldEvent("server_chunk", None, None, ["latched"] * n_f, frames=list(range(n_f)))
    return st.RecordData(key, ev, {("current", "cell"): g, ("hold", "cell"): g},
                         {("current", "cell"): gt, ("hold", "cell"): gt}, None, {"cell": names},
                         {("current", "cell"): "t56_replay"}, {})


@pytest.fixture
def synthetic(monkeypatch):
    rds = {"e0": _rd("e0", 0), "e1": _rd("e1", 1), "e2": _rd("e2", 2, event=False)}
    names = rds["e0"].names["cell"]
    st._CACHE.clear()
    monkeypatch.setattr(st, "resolve_keys", lambda args: list(KEYS))
    monkeypatch.setattr(st, "load_record", lambda args, root, k, names_, conds, kinds: rds[k])
    monkeypatch.setattr(st, "candidates", lambda args: {"cell": names, "arm": [], "cell_source": {"sources": {}}})
    monkeypatch.setattr(st, "gt_frames", lambda args, k: {"phase": ["pre"] * 10, "t_step": list(range(10)),
                                                          "has_target": np.ones(10, bool)})
    monkeypatch.setattr(st.cells_mod, "groups_of", lambda cells: {"cells": sorted(cells)})
    return rds


def _args(*extra):
    return st.build_parser().parse_args(["--run-dir", "x", *extra])


def test_dev_set_defaults_exclude_long_and_ep1800():
    assert "long" not in st.DEV_KEYS and "ep1800" not in st.DEV_KEYS
    assert st.DEV_KEYS == ("t2", "headfix", "ep1807", "ep1850", "ep1900", "ep0", "ep500", "ep1200")
    with pytest.raises(SystemExit):
        st.resolve_keys(_args("--set", "dev", "--stage", "inputs", "--records", "long"))
    assert st.resolve_keys(_args("--set", "dev", "--stage", "inputs")) == list(st.DEV_KEYS)


def test_test_stage_refuses_parameter_flags(tmp_path):
    for flag, val in (("--resolution", "0.02"), ("--regret-tol", "0.1"), ("--s1", "{}")):
        with pytest.raises(SystemExit, match="reads the prereg only"):
            st.stage_test(_args("--set", "test", "--stage", "test", flag, val), tmp_path)
    with pytest.raises(SystemExit):
        st.main(["--run-dir", str(tmp_path), "--set", "test", "--stage", "test", "--seed", "1"])


def test_test_stage_with_arms_needs_the_t54_rows(tmp_path):
    with pytest.raises(SystemExit, match="--t54-run"):
        st.stage_test(_args("--set", "test", "--stage", "test"), tmp_path)


def test_test_stage_needs_an_untampered_prereg(tmp_path):
    with pytest.raises(SystemExit, match="no prereg"):
        st.stage_test(_args("--set", "test", "--stage", "test", "--kinds", "cell"), tmp_path)
    body = {"step": "T56", "resolution": 0.01}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["resolution"] = 0.5  # edited after the dev stage
    (tmp_path / st.PREREG_NAME).write_text(json.dumps(body))
    with pytest.raises(SystemExit, match="sha256"):
        st.stage_test(_args("--set", "test", "--stage", "test", "--kinds", "cell"), tmp_path)


def test_dev_writes_prereg_and_test_reads_it_only(synthetic, tmp_path, monkeypatch):
    dev = _args("--set", "dev", "--stage", "dev", "--kinds", "cell", "--conditions", "current",
                "--n-boot", "50", "--resolution", "0.02", "--regret-tol", "0.04")
    pre = st.stage_dev(dev, tmp_path)
    p = tmp_path / st.PREREG_NAME
    assert p.is_file() and pre["resolution"] == 0.02 and pre["regret_tol"] == 0.04
    assert pre["purity_tol"] == st.DEV_DEFAULTS["purity_tol"]
    assert st.read_prereg(p)["sha256"] == pre["sha256"]
    assert pre["s1"] in [q.as_dict() for q in st.S1_GRID] and pre["s2"] in [q.as_dict() for q in st.S2_GRID]
    assert (tmp_path / "dev" / "dev_summary.json").is_file()
    sp = pre["search_space"]
    assert sp["s1"]["n"] == len(st.S1_GRID) and sp["s2"]["n"] == len(st.S2_GRID)
    assert sp["given_not_searched"]["resolution"] == 0.02 and sp["given_source"]["resolution"] == "cli"
    assert sp["given_source"]["purity_tol"] == "DEV_DEFAULTS"
    assert "2D lift" in pre["tie_break"][-1]
    raw = p.read_bytes()
    mtime = p.stat().st_mtime_ns

    seen = {}
    real = st.analyze

    def spy(rds, conds, kinds, p1, p2, resolution, W, phases):
        seen.update(p1=p1, p2=p2, resolution=resolution, B=W.shape[0], phases=phases)
        return real(rds, conds, kinds, p1, p2, resolution, W, phases)

    monkeypatch.setattr(st, "analyze", spy)
    test = _args("--set", "test", "--stage", "test", "--kinds", "cell", "--conditions", "current")
    out = st.stage_test(test, tmp_path)
    assert seen["p1"] == ss.S1Params(**pre["s1"]) and seen["p2"] == ss.S2Params(**pre["s2"])
    assert seen["resolution"] == 0.02 and seen["B"] == 50 and seen["phases"] is True
    assert p.read_bytes() == raw and p.stat().st_mtime_ns == mtime  # read only
    assert out["prereg_sha256"] == pre["sha256"]
    summ = json.loads((tmp_path / "test" / "analysis" / "t56_summary.json").read_text())
    s1 = summ["results"]["current"]["cells"]["all"]["S1"]
    assert s1["n_event_episodes"] == 2 and s1["episodes_without_event"] == ["e2"]
    # S0 has its own (sim both_contact) event in every synthetic record
    assert summ["results"]["current"]["cells"]["all"]["S0"]["n_event_episodes"] == 3
    crit = summ["criteria_mechanical"]["cells"]
    assert crit["regret_within_tol"]["tol"] == 0.04
    assert "tau_b_note" in crit["vs_S0"]
    assert "paired" in summ["results"]["current"]["cells"]["all"]


def test_tau_undefined_is_reported_not_zero_in_group_results(synthetic):
    rds = [synthetic[k] for k in KEYS]
    res = st.analyze(rds, ["current"], ["cell"], ss.S1Params(), ss.S2Params(), 0.01, None, False)
    s0 = res["current"]["cells"]["all"]["S0"]["summary"]["tau_b:identity"]
    # S0 is binary: good = 1, big / nbr / none = 0 → defined; the no-event episode is undefined (None)
    assert s0["episodes"]["e2"] is not None and s0["n_undefined"] == 0
    s1 = res["current"]["cells"]["all"]["S1"]["summary"]["tau_b:identity"]
    assert s1["episodes"]["e2"] is None and s1["n_undefined"] == 1
    assert s1["n_no_event"] == 1 and s1["episode_status"]["e2"] == "undefined:no_event"
    ties = res["current"]["cells"]["all"]["S1"]["summary"]["episode_ties"]
    assert ties["n_event"] == 2 and ties["per_episode"]["e2"] is None


def test_all_tied_self_score_gives_undefined_tau_and_the_mean_is_not_zero(synthetic):
    rds = [synthetic[k] for k in KEYS]
    # S1 with frame_max normalisation and a 'min' combine of identical candidates would tie; force it:
    # every candidate gets the same S1 frame score → self all tied in every event episode
    real = st.score_record

    def flat(rd, cond, kind, p1, p2):
        fs = real(rd, cond, kind, p1, p2)
        f = fs["frame"]["S1"]
        fs["frame"]["S1"] = np.where(np.isfinite(f), 0.5, np.nan)
        return fs

    import unittest.mock as um

    with um.patch.object(st, "score_record", flat):
        res = st.analyze(rds, ["current"], ["cell"], ss.S1Params(), ss.S2Params(), 0.0, None, False)
    t = res["current"]["cells"]["all"]["S1"]["summary"]["tau_b:identity"]
    assert t["mean_defined"] is None and t["mean_status"] == "undefined"
    assert t["undefined_reasons"] == {"self_all_tied": 2, "no_event": 1}
    summ = res["current"]["cells"]["all"]["S1"]["summary"]
    assert summ["tie_size"] == 4 and summ["episode_ties"]["tie_rate"] == 1.0
    # tie-set mean regret is reported next to the tie-broken one (coverage → name)
    assert summ["all:identity:tie_regret"] is not None


def test_replay_refuses_a_done_job_with_other_candidates(tmp_path, monkeypatch):
    args = _args("--set", "dev", "--stage", "replay", "--records", "t2", "--kinds", "cell",
                 "--conditions", "current", "--cells", "last:L8H2")
    args._t53_cfg = None
    monkeypatch.setattr(st, "record_inputs", lambda set_, k, a: {
        "problems": [], "npz": "n", "noun": "o", "record": "r", "target": "apple", "profile": "t52"})
    monkeypatch.setattr(st, "candidates", lambda a: {"cell": ["last:L8H2"], "arm": []})
    rdir = st.replay_dir(tmp_path, "dev")
    rdir.mkdir(parents=True)
    stem = rdir / st.job_stem("t2", "current", "cell", 0)
    stem.with_suffix(".meta.json").write_text(json.dumps({"names": ["last:L8H2"], "frames": None}))
    specs, skipped = st.plan_jobs(args, tmp_path)
    assert specs == [] and skipped == [stem.name]
    stem.with_suffix(".meta.json").write_text(json.dumps({"names": ["last:L12H4"], "frames": None}))
    with pytest.raises(SystemExit, match="other candidates"):
        st.plan_jobs(args, tmp_path)


def test_t55_resim_label_checks_are_read_from_the_set_dir(tmp_path):
    from tests.ag3s.test_t56_held import resim_npz

    d = tmp_path / "verify" / "T55" / "resim" / "test"
    d.mkdir(parents=True)
    for tag, holder in (("E0", "apple"), ("empty_L0O0_park", None), ("slip_A2D0O0_ghost", "pear")):
        np.savez(d / f"ep1804__{tag}.npz", **resim_npz(holder, tag=tag))
    args = _args("--set", "test", "--stage", "test", "--t55-run", str(tmp_path))
    out = st.t55_label_checks(args, ["ep1804"])
    rec = out["summary"]["record"]
    assert rec["all"]["n_held_events"] == 3 and rec["all"]["n_label_wrong"] == 2
    assert rec["E0"]["held_label_error_rate"] == 0.0
    assert rec["empty"]["verdicts"] == {"empty": 1} and rec["slip"]["verdicts"] == {"wrong_object": 1}
    assert st.t55_label_checks(_args("--set", "test", "--stage", "test"), ["ep1804"]) is None
