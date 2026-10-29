"""T56 auxiliary T60 windows (`t56_windows`) on synthetic records: positions from the episode mask, the same
mask for every candidate / variant, full == the test-stage evaluation, scope "both" == truncated frames, scope
"gt" keeps the full-window self score, mechanical criteria and the parity helpers."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t56_select as sel_mod
from AG3S.ag3s.experiments.studies import t56_study as st
from AG3S.ag3s.experiments.studies import t56_windows as w56
from AG3S.ag3s.experiments.studies import t60_termination as T

NAMES = ["last:L1H0", "last:L2H0", "last:L3H0"]   # sorted; the roles of a, b, c below
NF = 10
TSTEP = np.arange(NF, dtype=np.int64) * 8
T_P, T_H = 40, 56                         # P keeps frames 0..5, H keeps 0..7
PRE = {"regret_tol": 0.03, "purity_tol": 0.05, "coverage_tol": 0.05}


def _mask(t_p=T_P, t_h=T_H):
    return {"chunk_start_steps": TSTEP, "P": T.keep_frames(TSTEP, t_p), "H": T.keep_frames(TSTEP, t_h)}


def _close(a, b, path=""):
    """Structural equality with float tolerance (a NaN-masked mean and a truncated mean differ in the last ulp)."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12, nan_ok=True), path
    else:
        assert a == b, path


def _record(key: str, rng, frames=None, *, event=True):
    frames = list(range(NF)) if frames is None else frames
    n = len(frames)
    pre = np.asarray(frames) <= 5
    # task frames: 'a' is right (GT identity 1) and scores highest; after completion 'b' dominates both
    gt_id = np.where(pre[None, :], np.array([[1.0], [0.3], [0.5]]), np.array([[0.0], [0.7], [0.2]]))
    gt_id = np.clip(gt_id + rng.normal(0, 0.02, gt_id.shape), 0, 1)
    gt = {"identity": gt_id, "purity": gt_id * 0.9, "coverage": gt_id * 0.8, "no_target": 1 - gt_id}
    s = np.where(pre[None, :], np.array([[0.9], [0.4], [0.5]]), np.array([[0.1], [1.0], [0.3]]))
    s = s + rng.normal(0, 0.01, s.shape)
    if not event:
        s = np.full_like(s, np.nan)
    fs = {"frame": {"S0": s * 0.5, "S1": s, "S2": s}, "coverage": {"S1": s * 0.7}, "inclusion": {"S1": s}}
    geom = types.SimpleNamespace(frames=frames)
    rd = st.RecordData(key=key, held=None, geom={("current", "cell"): geom}, gt={("current", "cell"): gt},
                       phase_pos=None, names={"cell": NAMES}, gt_source={}, parity={})
    return rd, fs


def _setup(n_ep=4, events=None):
    rng = np.random.default_rng(3)
    rds, fs = [], {}
    for e in range(n_ep):
        rd, f = _record(f"ep{e}", rng, event=True if events is None else events[e])
        rds.append(rd)
        fs[rd.key] = f
    pos = {rd.key: w56.record_positions(rd, TSTEP, _mask()) for rd in rds}
    return rds, fs, pos


def test_window_positions_follow_t_step_mask():
    m = _mask()
    assert w56.window_positions(list(range(NF)), TSTEP, m, "P") == [0, 1, 2, 3, 4, 5]
    assert w56.window_positions(list(range(NF)), TSTEP, m, "H") == list(range(8))
    # geometry holding a frame subset: positions index the geometry, frames index the mask
    assert w56.window_positions([2, 4, 6, 8], TSTEP, m, "P") == [0, 1]
    assert w56.window_positions(list(range(NF)), TSTEP, _mask(None, None), "P") == list(range(NF))
    with pytest.raises(SystemExit):
        w56.window_positions(list(range(NF)), TSTEP + 1, m, "P")


def test_record_positions_require_same_frames():
    rd, _ = _record("ep0", np.random.default_rng(0))
    rd.geom[("hold", "cell")] = types.SimpleNamespace(frames=list(range(NF - 1)))
    with pytest.raises(SystemExit):
        w56.record_positions(rd, TSTEP, _mask())


def test_full_equals_test_stage_and_windows_change_selection():
    rds, fs, pos = _setup()
    W = sel_mod.episode_draws(len(rds), 200, 5607)
    res = w56.analyze_windows(rds, pos, ["current"], ["cell"], 0.01, W, fs_fn=lambda rd, c, k: fs[rd.key])
    r = res["current"]["cells"]
    keys = [rd.key for rd in rds]
    ref = st.evaluate_group(st.episode_tables(rds, "current", "cell", NAMES, fs), NAMES, keys, 0.01, W)
    assert w56.first_difference(w56.canonical(st.public(r["full"])), w56.canonical(st.public(ref))) is None
    # full: the post-completion frames dominate the mean self score -> 'b'; within P both scopes see 'a' best
    assert r["full"]["S1"]["summary"]["selected"] == NAMES[1]
    assert r["P"]["both"]["S1"]["summary"]["selected"] == NAMES[0]
    assert r["P"]["both"]["S1"]["summary"]["all:identity:regret"] == pytest.approx(0.0, abs=1e-9)
    # scope gt: the full-window choice ('b') graded on task frames only -> large regret
    assert r["P"]["gt"]["S1"]["summary"]["selected"] == NAMES[1]
    assert r["P"]["gt"]["S1"]["summary"]["all:identity:regret"] > 0.5
    # the original records are untouched (phase_pos stays None)
    assert all(rd.phase_pos is None for rd in rds)


def test_both_scope_equals_truncated_records():
    rds, fs, pos = _setup()
    res = w56.analyze_windows(rds, pos, ["current"], ["cell"], 0.01, None, fs_fn=lambda rd, c, k: fs[rd.key])
    keep = 6
    cut_rds, cut_fs = [], {}
    for rd in rds:
        g = rd.gt[("current", "cell")]
        cut_rds.append(dataclasses.replace(rd, gt={("current", "cell"): {m: v[:, :keep] for m, v in g.items()}},
                                           geom={("current", "cell"): types.SimpleNamespace(frames=list(range(keep)))}))
        f = fs[rd.key]
        cut_fs[rd.key] = {"frame": {v: m[:, :keep] for v, m in f["frame"].items()},
                          "coverage": {"S1": f["coverage"]["S1"][:, :keep]},
                          "inclusion": {"S1": f["inclusion"]["S1"][:, :keep]}}
    keys = [rd.key for rd in rds]
    want = st.evaluate_group(st.episode_tables(cut_rds, "current", "cell", NAMES, cut_fs), NAMES, keys, 0.01, None)
    got = res["current"]["cells"]["P"]["both"]
    _close(w56.canonical(st.public(got)), w56.canonical(st.public(want)))
    assert st.public(got)["S1"]["selected"] == st.public(want)["S1"]["selected"] == NAMES[0]


def test_same_mask_for_every_candidate_and_variant():
    rds, fs, pos = _setup()
    rdw = w56.with_positions(rds, pos)
    for rd in rdw:
        tab = st.episode_tables([rd], "current", "cell", NAMES, {rd.key: fs[rd.key]}, phase="H", phase_self=True)
        for v in ("S0", "S1", "S2"):
            want = np.nanmean(fs[rd.key]["frame"][v][:, :8], axis=1)    # frames 0..7 for every candidate
            assert np.allclose(tab["self"][v][0], want)
        assert np.allclose(tab["gt"]["identity"][0], rd.gt[("current", "cell")]["identity"][:, :8].mean(1))


def test_event_less_episode_stays_without_score():
    rds, fs, pos = _setup(events=[True, False, True, True])
    res = w56.analyze_windows(rds, pos, ["current"], ["cell"], 0.01, None, fs_fn=lambda rd, c, k: fs[rd.key])
    for w in ("P", "H"):
        s = res["current"]["cells"][w]["both"]["S1"]["summary"]
        assert s["n_event_episodes"] == 3 and s["episodes_without_event"] == ["ep1"]


def test_criteria_and_parity_helpers():
    rds, fs, pos = _setup()
    res = w56.analyze_windows(rds, pos, ["current"], ["cell"], 0.01, None, fs_fn=lambda rd, c, k: fs[rd.key])
    crit_all = w56.criteria_windows(res, PRE, "current")
    crit = crit_all["cells"]
    assert crit["P"]["both"]["regret_within_tol"]["pass_regret"] is True
    assert crit["full"]["regret_within_tol"]["pass_regret"] is False
    pub = w56.public_windows(res)
    summary = {"results": {"current": {"cells": {"all": w56.canonical(pub["current"]["cells"]["full"])}}},
               "criteria_mechanical": w56.canonical({g: v["full"] for g, v in crit_all.items()})}
    summary["results"]["current"].update({g: {"all": w56.canonical(pub["current"][g]["full"])}
                                          for g in pub["current"]})
    assert w56.parity(pub, crit_all, summary, "current")["exact"] is True
    summary["criteria_mechanical"]["cells"]["regret_within_tol"]["tol"] = 0.04
    assert w56.parity(pub, crit_all, summary, "current")["exact"] is False
    assert w56.first_difference({"x": float("nan")}, {"x": float("nan")}) is None
    rows = w56.table_rows(pub)
    assert {(r["window"], r["scope"]) for r in rows} == {("full", "-"), ("P", "both"), ("P", "gt"), ("H", "both"),
                                                         ("H", "gt")}
