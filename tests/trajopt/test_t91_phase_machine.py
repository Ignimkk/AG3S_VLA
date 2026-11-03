"""T91 — benchmark-agnostic phase machine (`AG3S/trajopt/phase_machine.py`) and its offline evaluation.

Fixed here: the rules (closing on nothing is not carry; co-motion is carry; opening after carry is retreat; place by
destination distance or by the VLA's open command), robot-independent gripper normalisation, causality, that the
machine never sees ground truth, and that threshold selection / evaluation cannot mix train and test scenes.
No serving code is touched, so there is no flag-default test: nothing here is wired into `serve_safe`.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t91_phase_eval as E
from AG3S.trajopt.phase_machine import (STAGES, GripperScale, PhaseConfig, PhaseConfigError, PhaseMachine,
                                        PhaseObservation)

APPLE = np.array([0.50, 0.30, 0.85])


def run_machine(frames, cfg=None, **kw):
    """frames: list of dict(g, ee, tgt, fresh, dst, chunk). Returns the stage list."""
    m = PhaseMachine(cfg or PhaseConfig(**kw))
    out = []
    for f in frames:
        st, _ = m.step(PhaseObservation(
            gripper_raw=f["g"], ee_pos=f.get("ee"), target_pos=f.get("tgt"), target_fresh=f.get("fresh", False),
            destination_pos=f.get("dst"), chunk_gripper=f.get("chunk")))
        out.append(st)
    return out


def approach_frames(n=6, tgt=APPLE):
    """Open hand walks to the target (it is fresh and static)."""
    return [dict(g=1.0, ee=tgt + np.array([0.0, 0.0, 0.30 * (1 - i / (n - 1)) + 0.02]), tgt=tgt, fresh=True,
                 chunk=[1.0] * 8) for i in range(n)]


def close_on(g_hold, tgt=APPLE, n=2):
    ee = tgt + np.array([0, 0, 0.02])
    return [dict(g=g_hold, ee=ee, tgt=tgt, fresh=True, chunk=[0.58] * 8) for _ in range(n)]


def lift(n, start, g_hold, follow=True, fresh=True, step=np.array([0.0, 0.0, 0.03]), dst=None, chunk=None):
    out = []
    for i in range(1, n + 1):
        ee = start + step * i
        out.append(dict(g=g_hold, ee=ee, tgt=(start + step * i) if follow else start, fresh=fresh, dst=dst,
                        chunk=chunk or [g_hold] * 8))
    return out


# --------------------------------------------------------------------------------- rules

def test_empty_hand_close_is_not_carry():
    """The hand closes on nothing and lifts away; the target stays where it was -> never carry."""
    ee0 = APPLE + np.array([0, 0, 0.02])
    fr = approach_frames() + close_on(0.05, n=2) + lift(8, ee0, 0.05, follow=False)
    # target stays put (fresh, static): the empty-hand closure must not become carry, even with the bootstrap on
    for k in (False, True):
        st = run_machine(fr, occluded_bootstrap=k)
        assert "carry" not in st and "place" not in st, st
    assert "grasp" in run_machine(fr)               # near + closed -> grasp (closing on the target's spot)


def test_comotion_is_carry():
    ee0 = APPLE + np.array([0, 0, 0.02])
    fr = approach_frames() + close_on(0.72, n=2) + lift(8, ee0, 0.72, follow=True)
    st = run_machine(fr)
    assert st[: len(approach_frames())] == ["approach"] * len(approach_frames()) or "grasp" in st
    assert st[-1] == "carry" and "carry" in st
    first_carry = st.index("carry")
    assert all(s == "grasp" for s in st[len(approach_frames()):first_carry] if s != "approach")


def test_open_after_carry_is_retreat_and_stays():
    ee0 = APPLE + np.array([0, 0, 0.02])
    fr = approach_frames() + close_on(0.72) + lift(6, ee0, 0.72)
    last = fr[-1]
    fr += [dict(g=1.0, ee=last["ee"] + np.array([0.0, 0.0, 0.03 * i]), tgt=last["tgt"], fresh=True, chunk=[1.0] * 8)
           for i in range(1, 5)]
    st = run_machine(fr)
    assert st[-1] == "retreat" and st[-4:] == ["retreat"] * 4
    assert "carry" in st


def test_place_by_destination_and_by_open_command():
    ee0 = APPLE + np.array([0, 0, 0.02])
    dst = ee0 + np.array([0.0, 0.0, 0.30])
    fr = approach_frames() + close_on(0.72) + lift(5, ee0, 0.72, dst=dst)      # within d_dest (xy 0) -> place
    assert run_machine(fr)[-1] == "place"
    far = ee0 + np.array([0.9, 0.0, 0.0])
    fr = approach_frames() + close_on(0.72) + lift(5, ee0, 0.72, dst=far)
    assert run_machine(fr)[-1] == "carry"                                       # destination far: still carry
    fr += lift(1, fr[-1]["ee"], 0.72, dst=far, chunk=[0.72, 0.72, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert run_machine(fr)[-1] == "place"                                       # the VLA commands opening: place


def test_occlusion_hold_and_bootstrap():
    """Target hidden by the hand: the strict rule never starts carry; the bootstrap does once the hand moved; an
    already-established carry survives occlusion while the hand stays closed on something."""
    ee0 = APPLE + np.array([0, 0, 0.02])
    pre = approach_frames() + close_on(0.72, n=2)
    hidden = lift(8, ee0, 0.72, follow=False, fresh=False)
    assert "carry" not in run_machine(pre + hidden, occluded_bootstrap=False)
    st = run_machine(pre + hidden, occluded_bootstrap=True, bootstrap_hand_mm=30.0)
    assert st[-1] == "carry" and st.index("carry") > len(pre)                  # starts after the hand moved >= 30 mm
    seen = lift(5, ee0, 0.72, follow=True)                                      # co-motion seen first
    after = lift(8, seen[-1]["ee"], 0.72, follow=False, fresh=False)
    assert run_machine(pre + seen + after)[-1] in ("carry", "place")           # no bootstrap needed to keep carry


def test_fully_closed_hand_holds_nothing_even_when_target_hidden():
    ee0 = APPLE + np.array([0, 0, 0.02])
    pre = approach_frames() + close_on(0.02, n=2)
    st = run_machine(pre + lift(8, ee0, 0.02, follow=False, fresh=False), occluded_bootstrap=True)
    assert "carry" not in st                                                    # opening < blocked_min_open


def test_far_target_is_approach_even_with_close_intent():
    fr = [dict(g=1.0, ee=APPLE + np.array([0, 0, 0.5]), tgt=APPLE, fresh=True, chunk=[0.1] * 8) for _ in range(4)]
    assert set(run_machine(fr)) == {"approach"}


def test_stage_names_match_t77():
    from AG3S.ag3s.experiments.studies.t77_labels import PHASES5

    assert tuple(STAGES) == tuple(PHASES5)


# --------------------------------------------------------------------------------- robot independence

def test_gripper_normalisation_is_robot_independent():
    ee0 = APPLE + np.array([0, 0, 0.02])
    fr = approach_frames() + close_on(0.72) + lift(8, ee0, 0.72)
    ref = run_machine(fr)
    for scale in (0.08, 255.0, 0.04):                 # Panda width in m, a 0..255 command, half-width
        fr2 = [{**f, "g": f["g"] * scale} for f in fr]
        assert run_machine(fr2) == ref
    # a robot whose closed limit is not zero (raw 0.2 closed .. 1.0 open): the zero is configurable
    fr3 = [{**f, "g": 0.2 + 0.8 * f["g"]} for f in fr]
    assert run_machine(fr3, closed_limit=0.2) == ref


def test_gripper_scale_reference_from_first_requests_and_clip():
    s = GripperScale(0.0, 3)
    assert s.update(0.04) == pytest.approx(1.0)
    assert s.update(0.08) == pytest.approx(1.0)
    assert s.update(0.06) == pytest.approx(0.75)
    assert s.open_ref == pytest.approx(0.08)
    assert s.update(0.01) == pytest.approx(0.125)     # reference frozen after N requests
    assert s.update(0.20) == pytest.approx(1.5)       # clipped, never wild
    assert s.update(-0.1) == 0.0


def test_rby1_columns_follow_chunk_layout():
    assert PhaseConfig.rby1("left").chunk_gripper_column == 7 and PhaseConfig.rby1("right").chunk_gripper_column == 15
    from AG3S.trajopt.types import ChunkLayout

    names = [f"{s}_arm_{i}" for s in ("left", "right") for i in range(7)] + [f"torso_{i}" for i in range(6)]
    lay = ChunkLayout.rby1(names)
    grip = sorted(set(range(lay.n_valid)) - {a for a, _ in lay.action_to_q})
    assert grip == [7, 15]
    assert E.GRIP_STATE_COL == {"left": 7, "right": 15}
    with pytest.raises(PhaseConfigError):
        PhaseConfig.rby1("middle")


def test_config_validation():
    for bad in (dict(closed_thr=0.95, open_thr=0.9), dict(intent_rows=0), dict(d_near=0.0),
                dict(chunk_open_value=1.0, chunk_closed_value=1.0), dict(open_ref_requests=0)):
        with pytest.raises(PhaseConfigError):
            PhaseConfig(**bad)


def test_missing_inputs_do_not_crash():
    m = PhaseMachine()
    st, ev = m.step(PhaseObservation(gripper_raw=1.0))
    assert st == "approach" and ev["d_target_mm"] is None
    st, _ = m.step(PhaseObservation(gripper_raw=0.7, ee_pos=[np.nan, 0, 0], target_pos=[0.1, 0.1, 0.1]))
    assert st == "approach"


# --------------------------------------------------------------------------------- causality, no ground truth

def synthetic_run(**over):
    ee0 = APPLE + np.array([0, 0, 0.02])
    fr = approach_frames() + close_on(0.72) + lift(8, ee0, 0.72)
    n = len(fr)
    run = dict(request=np.arange(n), t_step=np.arange(n) * 8, grip_raw=np.array([f["g"] for f in fr]),
               ee=np.array([f["ee"] for f in fr]), tgt=np.array([f["tgt"] for f in fr]),
               tgt_fresh=np.array([f["fresh"] for f in fr]), dest=np.full((n, 3), np.nan),
               chunk_grip=np.array([list(f["chunk"]) + [np.nan] * 8 for f in fr]), side="left",
               apple_gt=np.zeros((n, 3)), gt=np.array(["approach"] * n), y=np.zeros(n, int),
               label="x", scene="회귀1", split="train", kit="CE", latch=np.array(["searching"] * n))
    run.update(over)
    return run


def test_machine_is_causal():
    run = synthetic_run()
    full = E.predict_run(run, {})
    for k in (5, 9, 12):
        part = {key: (v[:k] if isinstance(v, np.ndarray) and len(v) == len(full) else v) for key, v in run.items()}
        assert np.array_equal(E.predict_run(part, {}), full[:k])


def test_machine_never_sees_ground_truth():
    run = synthetic_run()
    base = E.predict_run(run, {})
    scr = synthetic_run(apple_gt=np.random.default_rng(0).normal(size=(len(base), 3)),
                        gt=np.array(["carry"] * len(base)), y=np.full(len(base), 2),
                        latch=np.array(["held"] * len(base)))
    assert np.array_equal(E.predict_run(scr, {}), base)
    fields = set(PhaseObservation.__dataclass_fields__)
    assert not ({"apple", "apple_gt", "gt", "held", "phase", "y", "latch"} & fields)


# --------------------------------------------------------------------------------- train / test separation

def test_selection_refuses_test_runs():
    with pytest.raises(AssertionError):
        E.select_thresholds([synthetic_run(split="test")])
    with pytest.raises(AssertionError):
        E.select_thresholds([])


def test_load_runs_filters_split_and_excluded_scene(tmp_path):
    spec = []
    for i, (scene, split) in enumerate((("회귀250", "train"), ("회귀211", "test"), ("허들211", "test"))):
        e = dict(label=f"K/r{i}", scene=scene, split=split, side="left", kit="K")
        spec.append(e)
        r = synthetic_run()
        np.savez(tmp_path / f"{E.safe_name(e['label'])}.npz",
                 **{k: r[k] for k in ("request", "t_step", "grip_raw", "ee", "tgt", "tgt_fresh", "dest", "chunk_grip",
                                      "apple_gt", "gt", "latch")})
    assert [r["scene"] for r in E.load_runs(str(tmp_path), spec, "train")] == ["회귀250"]
    assert [r["scene"] for r in E.load_runs(str(tmp_path), spec, "test")] == ["회귀211"]        # 허들211 dropped
    assert {r["scene"] for r in E.load_runs(str(tmp_path), spec)} == {"회귀250", "회귀211"}


T77_SPEC = pathlib.Path(E.T77) / "smoke" / "t77_spec.json"


@pytest.mark.skipif(not T77_SPEC.exists(), reason="T77 spec not on this machine")
def test_spec_split_is_t77_split_without_leakage():
    from AG3S.ag3s.experiments.studies.t77_labels import TEST_SCENES, check_split

    spec = json.loads(T77_SPEC.read_text())["runs"]
    kept = [e for e in spec if E.keep_run(e)]
    assert all(e["scene"] != "허들211" for e in kept) and len(kept) < len(spec)
    sc = check_split(kept)
    assert set(TEST_SCENES) >= {s for s in sc["test"] if s in TEST_SCENES}
    assert not (set(sc["train"]) & set(TEST_SCENES))


# --------------------------------------------------------------------------------- metrics

def test_danger_counts_and_t77_variant():
    y = np.array([0, 0, 1, 1, 2, 3, 4])
    p = np.array([2, 4, 3, 1, 2, 3, 4])
    assert E.danger_counts(y, p) == (4, 2)                 # approach->carry, grasp->place
    assert E.danger_counts(y, p, with_retreat=True) == (4, 3)   # + approach->retreat (the T77 definition)


def test_scene_mean_is_per_scene_then_mean():
    ra = dict(scene="a", y=np.array([0, 0, 0, 0]))
    rb = dict(scene="b", y=np.array([0, 0]))
    pa, pb = np.array([0, 0, 0, 0]), np.array([1, 1])
    assert E.scene_mean([ra, rb], [pa, pb], E._acc) == pytest.approx(0.5)     # (1.0 + 0.0) / 2, not 4/6


def test_selection_rule_prefers_safe_then_accurate():
    safe_lo = dict(accuracy_scene_mean=0.6, danger_scene_mean=0.005)
    safe_hi = dict(accuracy_scene_mean=0.7, danger_scene_mean=0.009)
    unsafe = dict(accuracy_scene_mean=0.95, danger_scene_mean=0.05)
    assert E.better(safe_lo, None) and E.better(safe_lo, unsafe) and not E.better(unsafe, safe_lo)
    assert E.better(safe_hi, safe_lo) and not E.better(safe_lo, safe_hi)
    assert not E.better(safe_hi, safe_hi)                  # strict: ties keep the earlier candidate
    assert E.better(dict(accuracy_scene_mean=0.1, danger_scene_mean=0.02), unsafe)   # both infeasible: lower danger


def test_verdict_rule():
    a = dict(danger_scene_mean=0.02, accuracy_scene_mean=0.80)
    assert E.verdict(a, dict(danger_scene_mean=0.02, accuracy_scene_mean=0.75))["B_usable"]
    assert not E.verdict(a, dict(danger_scene_mean=0.03, accuracy_scene_mean=0.90))["B_usable"]
    assert not E.verdict(a, dict(danger_scene_mean=0.01, accuracy_scene_mean=0.7499))["B_usable"]


def test_end_to_end_evaluate_on_synthetic(tmp_path):
    """extract-free path: select on train, evaluate on test, A joined from a fake probe json."""
    spec, tr, te = [], [], []
    for i, (scene, split) in enumerate((("회귀250", "train"), ("회귀320", "train"), ("회귀211", "test"))):
        e = dict(label=f"K/r{i}", scene=scene, split=split, side="left", kit="K")
        spec.append(e)
        r = synthetic_run(y=np.array([0] * 6 + [1] * 2 + [2] * 8), gt=np.array(["approach"] * 6 + ["grasp"] * 2 + ["carry"] * 8))
        np.savez(tmp_path / f"{E.safe_name(e['label'])}.npz",
                 **{k: r[k] for k in ("request", "t_step", "grip_raw", "ee", "tgt", "tgt_fresh", "dest", "chunk_grip",
                                      "apple_gt", "gt", "latch")})
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps({"runs": spec}))
    n = len(synthetic_run()["request"])
    probe = {"test_predictions": {"group": ["cl:K/r2"] * n, "t_step": [int(t) for t in synthetic_run()["t_step"]],
                                  "kv_L4": [0] * 6 + [1] * 2 + [2] * 8,
                                  "h4_img3+last": [0] * 6 + [1] * 2 + [2] * 8}}
    (tmp_path / "probe.json").write_text(json.dumps(probe))
    E.SEARCH_GROUPS, saved = ({"d_near": (0.08, 0.12)},), E.SEARCH_GROUPS
    try:
        thr = E.select_thresholds(E.load_runs(str(tmp_path), spec, "train"), log=lambda *_: None)
    finally:
        E.SEARCH_GROUPS = saved
    assert thr["params"]["d_near"] in (0.08, 0.12)
    (tmp_path / "thr.json").write_text(json.dumps(thr))
    import argparse

    E.cmd_evaluate(argparse.Namespace(spec=str(sp), thresholds=str(tmp_path / "thr.json"), feat=str(tmp_path),
                                      probe=str(tmp_path / "probe.json"), out=str(tmp_path / "out.json")))
    res = json.loads((tmp_path / "out.json").read_text())
    assert res["test_scenes"] == ["회귀211"] and res["n_runs"] == 1
    assert res["A_T77_kvL4"]["all"]["accuracy_request"] == 1.0
    assert set(res["verdict"]) >= {"B_usable", "values"}


def test_defaults_are_the_train_selected_thresholds():
    import dataclasses

    d = dataclasses.asdict(PhaseConfig())
    for k, v in E.TRAIN_SELECTED.items():
        assert d[k] == v, k
    assert set(E.TRAIN_SELECTED) == {n for g in E.SEARCH_GROUPS for n in g}      # every searched parameter is frozen
