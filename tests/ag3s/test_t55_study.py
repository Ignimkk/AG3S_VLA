"""T55 study (`t55_study`) — a synthetic dev / test pipeline (no GPU, no MuJoCo, no storage).

Pinned (T55.task.md 구현 test):
  - conditions from (tag, prompt tag); instruction traces keep E0's physics and change only the answer's target;
  - episode blocks: the original and every variant of an episode share one group, one CV fold, one bootstrap block;
  - task_ready never alters the latch trace (attach / detach / phase arrays unchanged after evaluation);
  - K rule (least excess of P_grasp onsets over closing attempts, smallest K);
  - combination selection inside the tolerance, fewest false passes; infeasible -> least violation;
  - episode block bootstrap: point = pooled counts, seed-deterministic; moving-block and McNemar shapes;
  - dev writes ``T55.prereg.json`` once (refuses to overwrite); test reads it only (bytes unchanged), refuses a
    prereg of another version and interventions that are not the prereg's.
  - user decisions 2026-10-13: the primary latch is ``reground`` (one per re-sim and prompt — instruction traces
    carry their own); dev baseline = the unintervened re-sim (a non-exact dev E0 is accepted, a wrong baseline
    refused), test interventions need an exact E0; tolerances default to the user-fixed +5 %p / +1 frame / 0 and
    the prereg says so (an override is recorded as such); records without an E0 hold are listed as slip
    exclusions; a trace without the reground latch is refused.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t53_records as rc
from AG3S.ag3s.experiments.studies import t55_resim as rs
from AG3S.ag3s.experiments.studies import t55_study as st

FRUITS = ("apple", "banana", "orange", "pear")
F = 14
PROMPT = "Put the apple in the basket."


def _phase(held_from, held_to, closing_at=3):
    ph = ["searching", "searching"] + ["latched"] * (closing_at - 2)
    ph += ["closing"] * (held_from - closing_at)
    ph += ["held"] * (held_to - held_from)
    ph += ["placed"] * (F - len(ph))
    return np.asarray(ph[:F])


def write_resim(path: pathlib.Path, *, ep: int, holder: str = "apple", held=(4, 9), window_start=-1,
                latch_held=(4, 10), meta=None, record=True, baseline=None):
    ps = np.full((F, 4), "on_table", dtype="U12")
    ph = np.full((F, 4), "", dtype="U8")
    ic = np.zeros((F, 4), bool)
    if holder:
        j = FRUITS.index(holder)
        ps[held[0]:held[1], j] = "held"
        ph[held[0]:held[1], j] = "right"
        ic[held[1] + 1:, j] = holder == "apple"
    state = np.ones((F, 16))
    state[3:held[1] + 1, 15] = 0.6
    phase = _phase(*latch_held) if latch_held else np.asarray(["searching", "searching"] + ["latched"] + ["closing"] * (F - 3))
    attach = np.zeros(F, bool)
    detach = np.zeros(F, bool)
    if latch_held:
        attach[latch_held[0]] = True
        if latch_held[1] < F:
            detach[latch_held[1]] = True
    arrays = dict(fruits=np.asarray(FRUITS), phys_state=ps, phys_hand=ph, in_crate=ic, used_arm=np.asarray("right"),
                  target=np.asarray("apple"), prompt=np.asarray(PROMPT), episode=np.asarray(ep),
                  window_start=np.asarray(window_start), state=state,
                  meta=np.asarray(json.dumps({**(meta or {"parity": {"exact": True}}),
                                              "baseline": baseline or ("record" if record else "unintervened_resim")})))
    for mode in (("oracle", "record") if record else ("oracle",)):
        arrays[f"latch_{mode}__phase"] = phase
        arrays[f"latch_{mode}__attach"] = attach
        arrays[f"latch_{mode}__detach"] = detach
    np.savez(path, **arrays)


def write_perception(path: pathlib.Path, latch_held=(4, 10)):
    """A `t55_perception` output: the ``reground`` latch of one (re-sim, prompt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = _phase(*latch_held) if latch_held else np.asarray(["searching", "searching"] + ["latched"] + ["closing"] * (F - 3))
    attach, detach = np.zeros(F, bool), np.zeros(F, bool)
    if latch_held:
        attach[latch_held[0]] = True
        if latch_held[1] < F:
            detach[latch_held[1]] = True
    np.savez(path, latch_reground__phase=phase, latch_reground__attach=attach, latch_reground__detach=detach)


def write_feat(path: pathlib.Path, p_task: np.ndarray, rng, *, nl=2, h=4, d=6):
    sig = np.asarray(p_task, float)
    nv = rng.normal(scale=0.3, size=(F, 3, nl, h))
    nv[:, 0, :, :] += 2.0 * sig[:, None, None]
    hid = rng.normal(scale=0.3, size=(F, 3, d))
    hid[:, 0, :] += 2.0 * sig[:, None]
    np.savez(path, kv_l4=rng.normal(size=(F, 512)).astype(np.float32), noun_v=nv.astype(np.float32),
             noun_hidden_final=hid.astype(np.float32), noun_layers=np.asarray([0, 4]), noun_ok=np.ones(F, bool))


IV_EMPTY = {"kind": "empty", "method": "park", "lead_frames": 0, "after_frames": 2, "delay_frames": 0,
            "offset_steps": 0, "park_pos": [-20.0, 0.0, -5.0], "ghost_clear_m": 0.005, "ghost_max_steps": 30,
            "tag": "empty_L0O0_park"}


def make_set(root: pathlib.Path, set_name: str, keys, rng, *, wrong=(), empty_tag="empty_L0O0_park", iv=IV_EMPTY):
    rd, fd = root / "resim" / set_name, root / "feat" / set_name
    pd = root / "perception" / set_name / st.DEFAULT_PERCEPTION
    rd.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    meta_ds = {}
    for i, k in enumerate(keys):
        ep = 1800 + i + (100 if set_name == "test" else 0)
        meta_ds[ep] = {"non_target_fruits": ["banana", "orange", "pear"]}
        holder = "banana" if k in wrong else "apple"
        write_resim(rd / f"{k}__E0.npz", ep=ep, holder=holder, record=set_name == "test")
        pt = np.zeros(F, bool)
        if holder == "apple":
            pt[4:9] = True
        write_feat(fd / f"{k}__E0__orig.npz", pt, rng)
        write_perception(pd / f"{k}__E0__orig.npz")
        for f in ("banana", "orange", "pear"):
            ptf = np.zeros(F, bool)
            if holder == f:
                ptf[4:9] = True
            write_feat(fd / f"{k}__E0__instr_{f}.npz", ptf, rng)
            # re-grounded with the changed prompt: AG3S follows the other fruit, the attach is not confirmed
            # except for "banana" (a strong-rule false pass the feature has to catch)
            write_perception(pd / f"{k}__E0__instr_{f}.npz", latch_held=(4, 10) if f == "banana" else None)
        # empty grasp with a strong-rule false pass (the latch still says held)
        write_resim(rd / f"{k}__{empty_tag}.npz", ep=ep, holder="", window_start=4, record=set_name == "test",
                    meta={"intervention": {**iv, "tag": empty_tag}, "e0_exact": True})
        write_feat(fd / f"{k}__{empty_tag}__orig.npz", np.zeros(F, bool), rng)
        write_perception(pd / f"{k}__{empty_tag}__orig.npz")
    return meta_ds


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "out_root", lambda sub="", forbid=(): tmp_path / sub if sub else tmp_path)
    rng = np.random.default_rng(1)
    meta = make_set(tmp_path, "dev", ["d1", "d2", "d3", "d4"], rng, wrong=("d2",))
    meta |= make_set(tmp_path, "test", ["t1", "t2", "t3"], rng, wrong=("t3",))
    monkeypatch.setattr(rc, "load_dataset_meta", lambda *a, **k: meta)
    return tmp_path


DEV_ARGS = ["--stage", "dev", "--C-grid", "1.0", "--noun-cv-layers", "0", "4", "--n-splits", "2",
            "--hsr-tau", "argmax", "0.5", "--noun-tau", "0.5", "0.7", "--feature-persist", "1", "2"]


def test_conditions_and_groups(pipeline):
    tr = st.load_resim_traces(pipeline / "resim" / "dev", pipeline / "feat" / "dev", "dev")
    conds = sorted({(t.key, t.condition) for t in tr if t.key == "d1"})
    assert conds == [("d1", "empty"), ("d1", "instruction"), ("d1", "natural")]
    assert {t.group for t in tr if t.key == "d1"} == {"dev:d1"}
    nat = next(t for t in tr if t.key == "d1" and t.condition == "natural")
    ins = next(t for t in tr if t.key == "d1" and t.prompt_tag == "instr_pear")
    tn, ti = st.truth(nat, 1), st.truth(ins, 1)
    assert np.array_equal(tn["p_grasp"], ti["p_grasp"]) and tn["p_task"].any() and not ti["p_task"].any()
    emp = next(t for t in tr if t.key == "d1" and t.condition == "empty")
    assert emp.window.tolist() == [False] * 4 + [True] * (F - 4)
    with pytest.raises(ValueError):
        st.condition_of("weird", "orig")


def test_evaluate_never_alters_latch(pipeline):
    tr = st.load_resim_traces(pipeline / "resim" / "dev", pipeline / "feat" / "dev", "dev")
    before = {t.tid: {k: v.copy() for k, v in t.latch["oracle"].items()} for t in tr}
    truths = {t.tid: st.truth(t, 1) for t in tr}
    flags = {t.tid: np.zeros(t.n, bool) for t in tr}
    ev = st.evaluate(tr, truths, mode="oracle", flags=flags, p=1)
    for t in tr:
        for k, v in t.latch["oracle"].items():
            assert np.array_equal(v, before[t.tid][k])
    # all-false feature: no false pass left, every true hold rejected
    assert ev["counts"]["fp_a"] == 0 and ev["counts"]["fr_a"] == ev["counts"]["pos"]
    assert ev["counts"]["fp_s"] > 0  # wrong object (d2), instruction swaps, empty with latch held


def test_select_K_and_config():
    rows = [{"config": {"tau": 0.5, "p": 1}, "counts": dict(neg=10, pos=10, fp_s=5, fp_a=1, fr_s=0, fr_a=5, lat_sum=0,
                                                            lat_n=1, und_s=0, und_a=0, events=1)},
            {"config": {"tau": 0.7, "p": 1}, "counts": dict(neg=10, pos=10, fp_s=5, fp_a=2, fr_s=0, fr_a=1, lat_sum=0,
                                                            lat_n=1, und_s=0, und_a=0, events=1)}]
    tol = {"false_reject_rate": 0.2, "latency_frames": 2.0, "undetected_events": 0}
    sel = st.select_config(rows, tol)
    assert sel["feasible"] and sel["chosen"] == {"tau": 0.7, "p": 1}
    sel2 = st.select_config(rows, {**tol, "false_reject_rate": 0.0})
    assert not sel2["feasible"] and sel2["chosen"] == {"tau": 0.7, "p": 1}


def test_bootstrap_point_and_determinism():
    pg = {"a": dict(neg=10, pos=5, fp_s=4, fp_a=1, fr_s=0, fr_a=1, lat_sum=2, lat_n=2, und_s=0, und_a=0, events=2),
          "b": dict(neg=10, pos=5, fp_s=2, fp_a=2, fr_s=1, fr_a=1, lat_sum=0, lat_n=1, und_s=0, und_a=0, events=1)}
    b1 = st.episode_bootstrap(pg, B=500, seed=5507)
    b2 = st.episode_bootstrap(pg, B=500, seed=5507)
    assert b1 == b2
    assert b1["d_fp_rate"]["point"] == pytest.approx((3 - 6) / 20)
    assert b1["d_latency_mean"]["point"] == pytest.approx(2 / 3)
    assert b1["d_fp_rate"]["lo"] <= b1["d_fp_rate"]["point"] <= b1["d_fp_rate"]["hi"]
    X = np.zeros((7, 6))
    X[:, 0] = 1
    X[:3, 2] = 1
    mb = st.moving_block_bootstrap([X], B=200, seed=1)
    assert mb["d_fp_rate"]["hi"] <= 0 and mb["_spec"]["block"] == 5
    assert st.mcnemar(0, 6)["p_exact"] == pytest.approx(2 * 0.5 ** 6)


def test_dev_then_test(pipeline):
    pre = st.main(DEV_ARGS)
    pp = pipeline / st.PREREG_NAME
    assert pp.is_file() and pre["truth"]["K"] in (1, 2, 3)
    assert set(pre["arms"]) == {"strong", "hsr", "noun"} and pre["interventions"] == {"empty_L0O0_park": IV_EMPTY}
    assert pre["bootstrap"] == {"B": 10000, "seed": 5507, "level": 0.95, "unit": "episode block",
                                "moving_block_frames": 5}
    fog = pre["dev_tables"]["noun_cv_spec"]["fold_of_group"]
    assert set(fog) == {"dev:d1", "dev:d2", "dev:d3", "dev:d4"}
    assert (pipeline / "noun_probe.npz").is_file()
    with pytest.raises(SystemExit, match="written once"):
        st.main(DEV_ARGS)
    raw = pp.read_bytes()
    out = st.main(["--stage", "test", "--prereg", str(pp)])
    assert pp.read_bytes() == raw and out["prereg_md5"] == hashlib.md5(raw).hexdigest()
    assert out["primary"] == "reground" and set(out["results"]) == {"reground", "record", "oracle"}
    r = out["results"]["reground"]
    assert set(r["by"]) == {"natural", "instruction", "empty"}
    noun = r["by"]["instruction"]["p_task"]["noun"]
    assert noun["episode_bootstrap"]["_spec"]["n_groups"] == 3
    assert noun["counts"]["fp_a"] <= noun["counts"]["fp_s"]
    assert r["by"]["natural"]["p_grasp"]["strong"]["n_pos"] > 0
    assert "oracle" in out["results"] and set(out["natural_events"]) == {"t1", "t2", "t3"}
    assert out["natural_events"]["t3"]["wrong_object_frames"] > 0


def test_test_refuses_foreign_interventions_and_prereg(pipeline):
    st.main(DEV_ARGS)
    pp = pipeline / st.PREREG_NAME
    rng = np.random.default_rng(2)
    make_set(pipeline, "test", ["t4"], rng, empty_tag="empty_L1O0_park", iv={**IV_EMPTY, "lead_frames": 1})
    out = st.main(["--stage", "test", "--prereg", str(pp)])          # another tag: ignored, listed
    assert out["ignored_intervention_tags"] == ["empty_L1O0_park"]
    rd = pipeline / "resim" / "test"
    write_resim(rd / "t1__empty_L0O0_park.npz", ep=1900, holder="", window_start=4,
                meta={"intervention": {**IV_EMPTY, "offset_steps": 3}, "e0_exact": True})
    with pytest.raises(SystemExit, match="other parameters"):     # the prereg's tag, other parameters
        st.main(["--stage", "test", "--prereg", str(pp)])
    bad = pipeline / "bad.json"
    bad.write_text(json.dumps({"version": "x"}))
    with pytest.raises(SystemExit, match="not a T55 prereg"):
        st.main(["--stage", "test", "--prereg", str(bad)])


def test_dev_baseline_is_the_unintervened_resim(pipeline):
    rd = pipeline / "resim" / "dev"
    # a dev E0 that is not bit-exact to the recording is the baseline itself: accepted
    write_resim(rd / "d1__empty_L0O0_park.npz", ep=1800, holder="", window_start=4, record=False,
                meta={"intervention": IV_EMPTY, "e0_exact": False})
    out = st.main(DEV_ARGS + ["--dry-run"])
    assert out["baseline"] == {"dev": "unintervened_resim", "test": "record"}
    # a dev re-sim made under another baseline is refused
    write_resim(rd / "d1__empty_L0O0_park.npz", ep=1800, holder="", window_start=4, record=False,
                meta={"intervention": IV_EMPTY, "e0_exact": False}, baseline="record")
    with pytest.raises(SystemExit, match="baseline"):
        st.main(DEV_ARGS + ["--dry-run"])


def test_test_interventions_need_exact_e0(pipeline):
    st.main(DEV_ARGS)
    write_resim(pipeline / "resim" / "test" / "t1__empty_L0O0_park.npz", ep=1900, holder="", window_start=4,
                meta={"intervention": IV_EMPTY, "e0_exact": False})
    with pytest.raises(SystemExit, match="exact E0"):
        st.main(["--stage", "test", "--prereg", str(pipeline / st.PREREG_NAME)])


def test_tolerance_user_fixed_and_override(pipeline):
    assert st.USER_TOL == {"false_reject_rate": 0.05, "latency_frames": 1.0, "undetected_events": 0}
    pre = st.main(DEV_ARGS)
    tol = pre["tolerance"]
    assert (tol["false_reject_rate"], tol["latency_frames"], tol["undetected_events"]) == (0.05, 1.0, 0.0)
    assert tol["user_fixed"] and "user decision 2026-10-13" in tol["source"] and "not derived from dev" in tol["source"]
    assert {"hsr", "noun", "note"} <= set(pre["dev_uncertainty"])
    assert pre["dev_uncertainty"]["hsr"]["_spec"]["unit"] == "episode block"
    out = st.main(DEV_ARGS + ["--dry-run", "--tol-latency", "2"])
    assert not out["tolerance"]["user_fixed"] and out["tolerance"]["overridden"] == ["latency_frames"]
    assert "OVERRIDDEN" in out["tolerance"]["source"]


def test_instruction_traces_carry_their_own_reground_latch(pipeline):
    pd = pipeline / "perception" / "dev" / st.DEFAULT_PERCEPTION
    tr = st.load_resim_traces(pipeline / "resim" / "dev", pipeline / "feat" / "dev", "dev", perception_dir=pd)
    by = {(t.key, t.prompt_tag): t for t in tr if t.tag == "E0"}
    assert "held" in by[("d1", "orig")].latch["reground"]["phase"]
    assert "held" in by[("d1", "instr_banana")].latch["reground"]["phase"]
    assert "held" not in by[("d1", "instr_pear")].latch["reground"]["phase"]
    # the sensitivity / auxiliary modes reuse E0's latch for the instruction prompts
    assert np.array_equal(by[("d1", "instr_pear")].latch["oracle"]["phase"], by[("d1", "orig")].latch["oracle"]["phase"])
    assert by[("d1", "instr_pear")].meta["perception"].endswith("d1__E0__instr_pear.npz")


def test_missing_reground_latch_refused(pipeline):
    (pipeline / "perception" / "dev" / st.DEFAULT_PERCEPTION / "d3__E0__instr_orange.npz").unlink()
    with pytest.raises(SystemExit, match="reground"):
        st.main(DEV_ARGS + ["--dry-run"])
    out = st.main(DEV_ARGS + ["--dry-run", "--latch-mode-dev", "oracle"])     # the auxiliary mode still runs
    assert out["latch_mode_dev"] == "oracle"


def test_slip_exclusions_listed(pipeline):
    why = "unintervened re-sim never reaches a hold (no baseline state)"
    assert rs.SLIP_EXCLUSION_REASON == why
    (pipeline / "resim" / "dev" / "d2__slip_A2D0O0_ghost.json").write_text(json.dumps(
        {"variant": "slip_A2D0O0_ghost", "not_applicable": why, "excluded_from_slip_evaluation": True}))
    (pipeline / "resim" / "dev" / "d3__slip_A2D0O0_ghost.json").write_text(json.dumps(
        {"variant": "slip_A2D0O0_ghost", "not_applicable": "something else"}))
    out = st.main(DEV_ARGS + ["--dry-run"])
    assert out["slip_excluded_dev"] == {"d2": {"slip_A2D0O0_ghost": why}}


def test_dev_interventions_filter(pipeline):
    with pytest.raises(SystemExit, match="not among"):
        st.main(DEV_ARGS + ["--interventions", "slip_A2D0O0_ghost", "--dry-run"])
    out = st.main(DEV_ARGS + ["--interventions", "empty_L0O0_park", "--dry-run"])
    assert set(out["interventions"]) == {"empty_L0O0_park"} and not (pipeline / st.PREREG_NAME).exists()


def test_dev_no_features_K_only(pipeline):
    out = st.main(["--stage", "dev", "--no-features"])
    assert "K_selection" in out["truth"] and not (pipeline / st.PREREG_NAME).exists()
    assert (pipeline / "dev_no_features.json").is_file()
