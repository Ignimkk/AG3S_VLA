"""SUBTASK-f — after PLACED, under a confirmed `home`, "only the crate is in view" is certified too.

SUBTASK-e (gate on, 48 runs) still held 232/984 chunks after B3 released the placed apple, all of
them `uncertified` with grounding `no_admissible`: the only cluster in view was the crate
(`not_graspable`, destination overlap 1.0). The arm stops while it holds, the view does not change,
and four runs never got out (35–41 chunks). After the object is put down and the policy is going
home, having nothing to grasp is the expected scene, not a grounding failure.

The SUBTASK-e predicate (`ag3s.types.subtask_no_target`) is widened by one case:

    gate on ∧ confirmed label `home` ∧ latch reached PLACED (`placed_seen`) ∧ grounding
    `no_admissible`  — with the SUBTASK-e conditions unchanged: status `no_target`, validity
    `valid`, nothing carved (five fields + `metrics["manipulated"]` None + exclusion inactive).

The grounding status is not rewritten (the record keeps `no_admissible`); the reason is still
`subtask_no_target`, with `grounding_status` in its evidence telling the two cases apart.

What is tested (task §1):
  (i)   home ∧ placed_seen ∧ gate on ∧ no_admissible ∧ valid ∧ nothing carved → certified, executed;
  (ii)  placed_seen not set → HOLD;
  (iii) label not home (place · pick · None) → HOLD;
  (iv)  gate off → HOLD, decisions bit-identical to the rule before (SUBTASK-e and pre-SUBTASK-e);
  (v)   validity degraded / incomplete → HOLD;
  (vi)  another grounding (no_seed · lost · low_score · no_cluster · no_geometry · no_attention) → HOLD;
  (vii) something carved → no exception.

Frames come from the real `AG3S` pipeline (planar 2-link fixture). `gripper_max_opening=1 mm` makes
every cluster inadmissible, so the grounding is a genuine `no_admissible` with nothing adopted —
the shape of the SUBTASK-e HOLD frames (nothing held, nothing admissible).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.types import (
    ConstraintValidity,
    GroundingStatus,
    PipelineStatus,
    subtask_no_target,
)
from benchmark.trajopt import safe_policy as sp_mod
from benchmark.trajopt import wire
from benchmark.trajopt.safe_policy import geometry_certified
from benchmark.trajopt.types import TrajOptStatus
from tests.trajopt.test_subtask_no_target import (
    HOME,
    PICK,
    PLACE,
    _inputs,
    _Manip,
    _policy,
    _replace,
    _scene_fn_certified,
    _wire,
)

TINY_OPENING = 0.001  # 1 mm: no cluster of the fixture is graspable → `no_admissible`


def _ag3s(arm, gate, opening=TINY_OPENING):
    from benchmark.ag3s.runtime.pipeline import AG3S

    clustering = {"subtask_gate": gate, "subtask_confirm_frames": 1}
    if opening is not None:
        clustering["gripper_max_opening"] = opening
    return AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05},
                                      "clustering": clustering}), robot_model=arm)


def _frames(gate):
    """Real no_admissible frames under each label / PLACED combination, gate on or off.

    `placed_home`: pick, then PLACED + home  (the SUBTASK-f case)
    `home_not_placed`: home confirmed, PLACED never reported
    `pick` / `placed_place`: label pick (no PLACED) / PLACED + place
    `released`: the SUBTASK-e B3 path (normal opening: pick → adopt → PLACED + home → released) —
                its grounding is `subtask_gated`; tests turn it into `no_admissible` (the frames
                after the release in the field, where only the crate was in view).
    """
    arm, inputs = _inputs()
    out = {}
    a = _ag3s(arm, gate)
    a.set_subtask(PICK)
    out["pick"] = a.process(**inputs)
    a.set_placed(True)
    a.set_subtask(HOME)
    out["placed_home"] = a.process(**inputs)

    b = _ag3s(arm, gate)
    b.set_subtask(HOME)
    out["home_not_placed"] = b.process(**inputs)

    c = _ag3s(arm, gate)
    c.set_subtask(PICK)
    c.process(**inputs)
    c.set_placed(True)
    c.set_subtask(PLACE)
    out["placed_place"] = c.process(**inputs)

    r = _ag3s(arm, gate, opening=None)
    r.set_subtask(PICK)
    r.process(**inputs)
    r.set_placed(True)
    r.set_subtask(HOME)
    out["released"] = r.process(**inputs)
    return out


@pytest.fixture(scope="module")
def on():
    return _frames(True)


@pytest.fixture(scope="module")
def off():
    return _frames(False)


def _released_no_admissible(frames):
    return _replace(frames["released"], grounding_status=GroundingStatus.NO_ADMISSIBLE)


def _with_subtask(cs, **changes):
    metrics = dict(cs.metrics)
    if changes.pop("_drop", False):
        metrics.pop("subtask", None)
    else:
        metrics["subtask"] = dict(metrics["subtask"], **changes)
    return _replace(cs, metrics=metrics)


def _verdict(cs, trajopt_status="violated", policy="reasons"):
    sp = _policy(verdict_policy=policy)
    _wire(sp, cs, trajopt_status)
    return sp, sp._verdict(None)                                                # noqa: SLF001


def _assert_hold(cs, grounding="no_admissible"):
    assert not subtask_no_target(cs) and geometry_certified(cs) is False
    _sp, v = _verdict(cs)
    assert [r["kind"] for r in v.verdict_reasons] == ["uncertified"] and v.safe is False
    assert v.verdict_reasons[0]["evidence"]["grounding_status"] == grounding
    assert wire.gate_decision(v.verdict_reasons)[0] == "hold"


# ------------------------------------------------------------------------- the frames are real


def test_the_frames_are_what_the_tests_say(on, off):
    for frames, gate in ((on, True), (off, False)):
        for name in ("pick", "placed_home", "home_not_placed", "placed_place"):
            cs = frames[name]
            assert cs.status is PipelineStatus.NO_TARGET, name
            assert cs.grounding_status is GroundingStatus.NO_ADMISSIBLE, name
            assert cs.validity is ConstraintValidity.VALID, name
            assert cs.target is None and cs.manipulated is None and cs.attached is None, name
            assert cs.metrics["manipulated"] is None, name
            assert cs.metrics["exclusion"]["source"] == "none", name
            assert cs.metrics["exclusion"]["active"] is False, name
            assert cs.metrics["subtask"]["gate"] is gate, name
        sub = {n: frames[n].metrics["subtask"] for n in frames}
        assert (sub["placed_home"]["label"], sub["placed_home"]["placed_seen"]) == ("home", True)
        assert (sub["home_not_placed"]["label"],
                sub["home_not_placed"]["placed_seen"]) == ("home", False)
        assert (sub["pick"]["label"], sub["pick"]["placed_seen"]) == ("pick", False)
        assert (sub["placed_place"]["label"], sub["placed_place"]["placed_seen"]) == ("place", True)
    assert on["released"].grounding_status is GroundingStatus.SUBTASK_GATED
    assert on["released"].metrics["subtask"]["released"] is True


# ------------------------------------------------------------------------------ (i)


@pytest.mark.parametrize("which", ["placed_home", "released_no_admissible"])
def test_i_placed_home_no_admissible_is_certified(on, which):
    cs = on["placed_home"] if which == "placed_home" else _released_no_admissible(on)
    assert subtask_no_target(cs) and cs.subtask_no_target
    assert geometry_certified(cs) is True
    # the grounding status is not rewritten
    assert cs.grounding_status is GroundingStatus.NO_ADMISSIBLE
    assert cs.status is PipelineStatus.NO_TARGET


@pytest.mark.parametrize("which", ["placed_home", "released_no_admissible"])
def test_i_the_verdict_executes_and_the_evidence_tells_it_apart(on, which):
    cs = on["placed_home"] if which == "placed_home" else _released_no_admissible(on)
    sp, v = _verdict(cs, "optimal")

    assert v.safe is True and v.geometry_certified is True
    assert v.ag3s_status == "no_target"
    assert [r["kind"] for r in v.verdict_reasons] == ["subtask_no_target"]
    r = v.verdict_reasons[0]
    assert r["action"] == "execute"
    ev = r["evidence"]
    assert ev["grounding_status"] == "no_admissible"
    assert ev["validity"] == "valid" and ev["ag3s_status"] == "no_target"
    assert ev["subtask_label"] == "home" and ev["placed_seen"] is True
    assert ev["exclusion_source"] == "none" and ev["exclusion_active"] is False
    assert ev["subtask_released"] is (which == "released_no_admissible")
    assert "nothing admissible is in view after the object was placed" in r["detail"]
    # on the wire, the local gate executes it; the server record keeps it
    out = v.to_dict()
    reasons, source = wire.unpack_verdict_reasons(out)
    assert source == "server" and wire.gate_decision(reasons)[0] == "execute"
    assert out["ag3s_status"] == "no_target"
    assert sp._exec_summary()["verdict"]["reasons"][0]["kind"] == "subtask_no_target"  # noqa: SLF001


def test_i_the_subtask_gated_reason_text_is_unchanged(on):
    """SUBTASK-e's own case keeps its wording; the two are told apart by grounding_status."""
    _sp, v = _verdict(on["released"], "optimal")
    r = v.verdict_reasons[0]
    assert r["kind"] == "subtask_no_target"
    assert r["evidence"]["grounding_status"] == "subtask_gated"
    assert r["detail"].startswith("no target this frame because the subtask gate withheld it")


def test_i_legacy_verdict_policy_agrees(on):
    _sp, v = _verdict(on["placed_home"], "optimal", policy="legacy")
    assert v.safe is True and [r["kind"] for r in v.verdict_reasons] == ["subtask_no_target"]


def test_i_scene_fn_hands_certified_to_the_optimizer(on, monkeypatch):
    assert _scene_fn_certified(on["placed_home"], monkeypatch) is True
    assert _scene_fn_certified(_released_no_admissible(on), monkeypatch) is True
    assert _scene_fn_certified(on["home_not_placed"], monkeypatch) is False
    assert _scene_fn_certified(on["pick"], monkeypatch) is False


@pytest.fixture(scope="module")
def scenario():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=10, penetration=0.03)


def test_i_the_optimizer_status_follows_the_predicate(on, scenario):
    from benchmark.trajopt.sqp import TrajectoryOptimizer

    def solve(cs):
        opt = TrajectoryOptimizer(scenario.robot, scenario.layout, scenario.limits, scenario.config)
        return opt.solve(scenario.reference, scenario.q_now, scenario.scene,
                         geometry_certified=geometry_certified(cs))

    ok = solve(on["placed_home"])
    assert ok.safe and ok.metrics["geometry_certified"] is True
    assert not any("could not certify" in n for n in ok.notes)
    held = solve(on["home_not_placed"])
    assert held.status is TrajOptStatus.VIOLATED
    assert any("could not certify" in n for n in held.notes)


# ------------------------------------------------------------------------------ (ii)


@pytest.mark.parametrize("placed", ["real_not_placed", False, None, "missing", "true_string", 1])
def test_ii_without_placed_seen_it_holds(on, placed):
    if placed == "real_not_placed":
        cs = on["home_not_placed"]
    elif placed == "missing":
        cs = _replace(on["placed_home"], metrics=dict(
            on["placed_home"].metrics,
            subtask={k: v for k, v in on["placed_home"].metrics["subtask"].items()
                     if k != "placed_seen"}))
    elif placed == "true_string":
        cs = _with_subtask(on["placed_home"], placed_seen="true")
    else:
        cs = _with_subtask(on["placed_home"], placed_seen=placed)
    _assert_hold(cs)


def test_ii_no_subtask_record_holds(on):
    _assert_hold(_with_subtask(on["placed_home"], _drop=True))
    _assert_hold(_replace(on["placed_home"],
                          metrics=dict(on["placed_home"].metrics, subtask="home")))


# ------------------------------------------------------------------------------ (iii)


@pytest.mark.parametrize("label", ["real_pick", "real_placed_place", "pick", "place", None, "HOME",
                                   "home "])
def test_iii_label_not_home_holds(on, label):
    if label == "real_pick":
        cs = on["pick"]
    elif label == "real_placed_place":
        cs = on["placed_place"]
    else:
        cs = _with_subtask(on["placed_home"], label=label)
    _assert_hold(cs)


# ------------------------------------------------------------------------------ (iv)


def test_iv_gate_off_real_frames_hold(off):
    for name in ("pick", "placed_home", "home_not_placed", "placed_place"):
        _assert_hold(off[name])
    # the gate-off `released` script never releases (the object is still manipulated); even with
    # every carve field cleared, `gate=False` keeps the no_admissible branch closed
    assert off["released"].grounding_status is not GroundingStatus.SUBTASK_GATED
    assert off["released"].metrics["subtask"]["placed_seen"] is True
    cleared = _replace(_released_no_admissible(off), target=None, manipulated=None,
                       attached=None, target_field_exclude=None, manipulated_link_margin=None,
                       status=PipelineStatus.NO_TARGET,
                       metrics=dict(off["released"].metrics, manipulated=None,
                                    exclusion={"source": "none", "active": False}))
    _assert_hold(cleared)
    assert subtask_no_target(_with_subtask(cleared, gate=True))  # control: only the gate differs


@pytest.mark.parametrize("gate", [False, None, 1, "true"], ids=repr)
def test_iv_gate_flag_must_be_exactly_true(on, gate):
    _assert_hold(_with_subtask(on["placed_home"], gate=gate))


def _rule_before_f(cs):
    """SUBTASK-e's predicate: the widened one restricted to `subtask_gated`."""
    return (getattr(getattr(cs, "grounding_status", None), "value", None) == "subtask_gated"
            and subtask_no_target(cs))


def _snapshot(cs, trajopt_status, policy, monkeypatch, rule):
    sp = _policy(verdict_policy=policy)
    with monkeypatch.context() as m:
        if rule is not None:
            m.setattr(sp_mod, "subtask_no_target", rule)
        _wire(sp, cs, trajopt_status)
        v = sp._verdict(None)                                                   # noqa: SLF001
        return v.to_dict(), sp._last_reasons                                    # noqa: SLF001


@pytest.mark.parametrize("policy", ["reasons", "legacy"])
@pytest.mark.parametrize("trajopt_status", ["optimal", "violated"])
def test_iv_gate_off_verdicts_are_bit_identical_to_the_rules_before(off, policy, trajopt_status,
                                                                   monkeypatch):
    frames = [off[n] for n in ("pick", "placed_home", "home_not_placed", "placed_place",
                               "released")]
    frames += [_replace(off["placed_home"], grounding_status=g) for g in GroundingStatus]
    for cs in frames:
        new = _snapshot(cs, trajopt_status, policy, monkeypatch, None)
        before_f = _snapshot(cs, trajopt_status, policy, monkeypatch, _rule_before_f)
        assert new == before_f, cs.grounding_status
        if cs.grounding_status is not GroundingStatus.SUBTASK_GATED:
            # `subtask_gated` is unreachable with the gate off (SUBTASK-e (iv)); on every reachable
            # gate-off frame the pre-SUBTASK-e rule (`status == ok`) gives the same bytes too
            before_e = _snapshot(cs, trajopt_status, policy, monkeypatch, lambda _cs: False)
            assert new == before_e, cs.grounding_status
    for cs in frames:
        if cs.grounding_status is not GroundingStatus.SUBTASK_GATED:
            assert _scene_fn_certified(cs, monkeypatch) is (cs.status.value == "ok")


def test_iv_control_the_rule_before_held_the_placed_home_frame(on, monkeypatch):
    """Not vacuous: on the gate-on placed-home frame the SUBTASK-e rule is the 232-chunk HOLD."""
    new = _snapshot(on["placed_home"], "violated", "reasons", monkeypatch, None)
    old = _snapshot(on["placed_home"], "violated", "reasons", monkeypatch, _rule_before_f)
    assert new != old
    assert [r["kind"] for r in old[1]] == ["uncertified"]
    assert "grounding=no_admissible, validity=valid" in old[1][0]["detail"]


def test_iv_the_widened_predicate_differs_from_subtask_e_in_exactly_one_cell(on):
    """gate × label × placed_seen × grounding × status × validity: the new predicate equals the
    SUBTASK-e rule everywhere except gate on ∧ home ∧ placed ∧ no_admissible ∧ no_target ∧ valid."""
    base = on["placed_home"]
    n, n_diff = 0, 0
    for gate, label, placed, g, s, v in itertools.product(
            (True, False), ("pick", "place", "home", None), (True, False),
            list(GroundingStatus), list(PipelineStatus), list(ConstraintValidity)):
        cs = _replace(_with_subtask(base, gate=gate, label=label, placed_seen=placed),
                      grounding_status=g, status=s, validity=v)
        widened = (gate and label == "home" and placed and g is GroundingStatus.NO_ADMISSIBLE
                   and s is PipelineStatus.NO_TARGET and v is ConstraintValidity.VALID)
        new = subtask_no_target(cs)
        assert new is (_rule_before_f(cs) or widened), (gate, label, placed, g, s, v)
        assert geometry_certified(cs) is (s is PipelineStatus.OK or new)
        n += 1
        n_diff += new != _rule_before_f(cs)
    assert n == 2 * 4 * 2 * len(GroundingStatus) * len(PipelineStatus) * len(ConstraintValidity)
    assert n_diff == 1


# ------------------------------------------------------------------------------ (v)


@pytest.mark.parametrize("validity,status", [
    (ConstraintValidity.DEGRADED, PipelineStatus.NO_TARGET),
    (ConstraintValidity.DEGRADED, PipelineStatus.DEGRADED),
    (ConstraintValidity.INCOMPLETE, PipelineStatus.NO_TARGET),
    (ConstraintValidity.INCOMPLETE, PipelineStatus.GEOMETRY_INCOMPLETE),
], ids=["degraded_no_target", "degraded", "incomplete_no_target", "incomplete"])
def test_v_not_valid_holds(on, validity, status):
    _assert_hold(_replace(on["placed_home"], validity=validity, status=status))


# ------------------------------------------------------------------------------ (vi)

OTHER = [GroundingStatus.NO_SEED, GroundingStatus.LOST, GroundingStatus.LOW_SCORE,
         GroundingStatus.NO_CLUSTER, GroundingStatus.NO_GEOMETRY, GroundingStatus.NO_ATTENTION]


@pytest.mark.parametrize("grounding", OTHER, ids=lambda g: g.value)
@pytest.mark.parametrize("which", ["placed_home", "released"])
def test_vi_another_grounding_after_placed_home_holds(on, grounding, which):
    _assert_hold(_replace(on[which], grounding_status=grounding), grounding=grounding.value)


# ------------------------------------------------------------------------------ (vii)


@pytest.mark.parametrize("change", [
    {"manipulated": _Manip()},
    {"attached": object()},
    {"target_field_exclude": np.ones(3, bool)},
    {"manipulated_link_margin": np.zeros(3)},
    {"target": object()},
    {"metrics": "manipulated_record"},
    {"metrics": "exclusion_active"},
    {"metrics": "exclusion_source_manipulated"},
    {"metrics": "exclusion_missing"},
], ids=["manipulated", "attached", "target_free_layer", "contact_margin", "target",
        "manipulated_record", "exclusion_active", "exclusion_source", "exclusion_missing"])
def test_vii_anything_carved_keeps_the_exception_off(on, change):
    base = on["placed_home"]
    if "metrics" in change:
        metrics = dict(base.metrics)
        kind = change["metrics"]
        if kind == "manipulated_record":
            metrics["manipulated"] = {"id": 3, "state": "visible", "age_frames": 0}
        elif kind == "exclusion_active":
            metrics["exclusion"] = dict(metrics["exclusion"], active=True)
        elif kind == "exclusion_source_manipulated":
            metrics["exclusion"] = dict(metrics["exclusion"], source="manipulated")
        else:
            metrics.pop("exclusion")
        cs = _replace(base, metrics=metrics)
    else:
        cs = _replace(base, **change)
    assert not subtask_no_target(cs) and geometry_certified(cs) is False


def test_vii_an_invariant_violation_no_admissible_is_not_the_exception(on):
    """`pipeline` also writes `no_admissible` when the exclusion gate refuses a manipulated object
    (T26 invariant_violation). That frame has a live identity record, so it stays uncertified."""
    metrics = dict(on["placed_home"].metrics,
                   manipulated={"id": 4, "state": "visible", "age_frames": 0},
                   invariant_violation={"detail": "exclusion geometry refused"})
    _assert_hold(_replace(on["placed_home"], metrics=metrics))
