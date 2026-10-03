"""SUBTASK-e — a target the subtask gate withheld on purpose is certified geometry.

SUBTASK-d (gate on, 48 runs): after B3 released the placed apple, 984/984 chunks were HOLD as
`uncertified` — `status=no_target, grounding=subtask_gated, validity=valid`, `trajopt_status=violated`
with `max_violation=0.0`. `no_target` meant "grounding could not name the object", and the rule was to
hold. `subtask_gated` is different: the label (place / home) says there is nothing to carve, and the
field that results — nothing carved, every object at full clearance — is the most conservative one.

The certification verdict is made in **two** places and both must agree:
  * `SafePolicy._scene_fn` → `refiner` → `sqp._finish(geometry_certified=…)` (the optimizer status);
  * `SafePolicy._verdict` → `_reasons` → `_certification_reason` (the verdict reasons).
Both now read `safe_policy.geometry_certified` (and `bringup.LivePipeline.scene_fn` too).

What is tested (task §2):
  (i)   subtask_gated + valid + nothing carved → certified, not HOLD, reason `subtask_no_target`;
  (ii)  the same with validity not valid → uncertified (HOLD);
  (iii) any other no_target (`no_seed`, `low_score`, `lost`, …) → uncertified, as before;
  (iv)  gate off: the decisions are bit-identical to the pre-SUBTASK-e rule;
  (v)   something carved (manipulated, attached, exclusion active, …) → the exception does not apply.
"""

from __future__ import annotations

import dataclasses
import types

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

PICK = {"pick": 0.9, "place": 0.08, "home": 0.02}
PLACE = {"pick": 0.05, "place": 0.9, "home": 0.05}
HOME = {"pick": 0.01, "place": 0.09, "home": 0.9}


# ------------------------------------------------------------------------- real AG3S frames


def _inputs():
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    return arm, dict(attention_map=scene.attention_grid, depth=scene.depth,
                     camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                     robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)


def _ag3s(arm, gate):
    from benchmark.ag3s.runtime.pipeline import AG3S

    return AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05},
                                      "clustering": {"subtask_gate": gate,
                                                     "subtask_confirm_frames": 1}}),
                robot_model=arm)


def _frames(gate):
    """The two ways the gate empties the target, run with the gate on or off.

    `first`: place confirmed before anything was adopted (the gate refuses the first target).
    `released`: pick → adopted, then PLACED + home (B3 releases the placed object).
    """
    arm, inputs = _inputs()
    first = _ag3s(arm, gate)
    first.set_subtask(PLACE)
    cs_first = first.process(**inputs)

    rel = _ag3s(arm, gate)
    rel.set_subtask(PICK)
    cs_pick = rel.process(**inputs)
    rel.set_placed(True)
    rel.set_subtask(HOME)
    cs_released = rel.process(**inputs)
    return {"first": cs_first, "pick": cs_pick, "released": cs_released}


@pytest.fixture(scope="module")
def gated():
    return _frames(True)


@pytest.fixture(scope="module")
def ungated():
    return _frames(False)


# ------------------------------------------------------------------------------ (i)


@pytest.mark.parametrize("which", ["first", "released"])
def test_i_a_gated_frame_with_nothing_carved_is_certified(gated, which):
    cs = gated[which]
    assert cs.status is PipelineStatus.NO_TARGET
    assert cs.grounding_status is GroundingStatus.SUBTASK_GATED
    assert cs.validity is ConstraintValidity.VALID
    assert cs.manipulated is None and cs.metrics["manipulated"] is None
    assert cs.metrics["exclusion"]["source"] == "none" and not cs.metrics["exclusion"]["active"]
    assert subtask_no_target(cs) and cs.subtask_no_target
    assert geometry_certified(cs) is True


@pytest.mark.parametrize("which", ["first", "released"])
def test_i_the_verdict_executes_and_says_why(gated, which):
    sp = _policy()
    _wire(sp, gated[which], "optimal")

    v = sp._verdict(None)                                                       # noqa: SLF001

    assert v.safe is True and v.geometry_certified is True
    assert v.ag3s_status == "no_target", "the status is not rewritten — the reason explains it"
    assert [r["kind"] for r in v.verdict_reasons] == ["subtask_no_target"]
    r = v.verdict_reasons[0]
    assert r["action"] == "execute"
    ev = r["evidence"]
    assert ev["grounding_status"] == "subtask_gated" and ev["validity"] == "valid"
    assert ev["exclusion_source"] == "none" and ev["exclusion_active"] is False
    assert ev["subtask_label"] == ("place" if which == "first" else "home")
    assert ev["subtask_released"] is (which == "released")
    # on the wire, and the local gate (every mode) executes it
    out = v.to_dict()
    assert out[wire.VERDICT_REASONS][0]["kind"] == "subtask_no_target"
    reasons, source = wire.unpack_verdict_reasons(out)
    assert source == "server" and wire.gate_decision(reasons)[0] == "execute"
    # the server record carries it too (summary_json.verdict.reasons)
    assert sp._exec_summary()["verdict"]["reasons"][0]["kind"] == "subtask_no_target"  # noqa: SLF001


def test_i_legacy_verdict_policy_agrees_because_the_optimizer_saw_the_same_rule(gated):
    sp = _policy(verdict_policy="legacy")
    _wire(sp, gated["released"], "optimal")
    v = sp._verdict(None)                                                       # noqa: SLF001
    assert v.safe is True and [r["kind"] for r in v.verdict_reasons] == ["subtask_no_target"]


def test_i_scene_fn_hands_the_same_verdict_to_the_optimizer(gated, ungated, monkeypatch):
    """`_scene_fn` → refiner → `sqp._finish(geometry_certified=…)`: the second HOLD site."""
    for cs, expected in ((gated["released"], True), (gated["first"], True),
                         (ungated["pick"], True), (_replace(gated["released"],
                                                            grounding_status=GroundingStatus.NO_SEED), False)):
        assert _scene_fn_certified(cs, monkeypatch) is expected


@pytest.fixture(scope="module")
def scenario():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=10, penetration=0.03)


def test_i_the_optimizer_status_follows_the_predicate(gated, scenario):
    """sqp.py `_finish`: a gated frame is not downgraded to `violated`; a `no_seed` one still is."""
    from benchmark.trajopt.sqp import TrajectoryOptimizer

    def solve(cs):
        opt = TrajectoryOptimizer(scenario.robot, scenario.layout, scenario.limits, scenario.config)
        return opt.solve(scenario.reference, scenario.q_now, scenario.scene,
                         geometry_certified=geometry_certified(cs))

    ok = solve(gated["released"])
    assert ok.safe and ok.metrics["geometry_certified"] is True
    assert not any("could not certify" in n for n in ok.notes)
    seeded = solve(_replace(gated["released"], grounding_status=GroundingStatus.NO_SEED))
    assert seeded.max_violation <= scenario.config.safety.violation_tolerance
    assert seeded.status is TrajOptStatus.VIOLATED
    assert any("could not certify" in n for n in seeded.notes)


def test_i_bringup_scene_fn_uses_the_same_predicate():
    import inspect

    from benchmark.trajopt import bringup

    src = inspect.getsource(bringup.LivePipeline.scene_fn)
    assert "geometry_certified(constraint_set)" in src
    assert 'status.value == "ok"' not in src
    assert bringup.geometry_certified is geometry_certified


# ------------------------------------------------------------------------------ (ii)


@pytest.mark.parametrize("validity,status", [
    (ConstraintValidity.DEGRADED, PipelineStatus.NO_TARGET),
    (ConstraintValidity.DEGRADED, PipelineStatus.DEGRADED),
    (ConstraintValidity.INCOMPLETE, PipelineStatus.GEOMETRY_INCOMPLETE),
], ids=["degraded_no_target", "degraded", "incomplete"])
def test_ii_not_valid_stays_uncertified(gated, validity, status):
    cs = _replace(gated["released"], validity=validity, status=status)
    assert not subtask_no_target(cs) and geometry_certified(cs) is False
    sp = _policy()
    _wire(sp, cs, "violated")
    v = sp._verdict(None)                                                       # noqa: SLF001
    assert [r["kind"] for r in v.verdict_reasons] == ["uncertified"] and v.safe is False


# ------------------------------------------------------------------------------ (iii)

OTHER_GROUNDINGS = [g for g in GroundingStatus
                    if g not in (GroundingStatus.SUBTASK_GATED, GroundingStatus.OK)]


@pytest.mark.parametrize("grounding", OTHER_GROUNDINGS, ids=lambda g: g.value)
def test_iii_any_other_no_target_stays_uncertified(gated, grounding):
    # SUBTASK-f: `no_admissible` on the `released` frame (gate on, home, PLACED seen) is now the
    # widened exception (tests/trajopt/test_subtask_no_admissible_home.py); here it is taken on
    # the `first` frame (label place, nothing placed), where it still holds.
    base = gated["first"] if grounding is GroundingStatus.NO_ADMISSIBLE else gated["released"]
    cs = _replace(base, grounding_status=grounding)
    assert not subtask_no_target(cs) and geometry_certified(cs) is False
    sp = _policy()
    _wire(sp, cs, "violated")
    v = sp._verdict(None)                                                       # noqa: SLF001
    assert [r["kind"] for r in v.verdict_reasons] == ["uncertified"] and v.safe is False
    assert v.verdict_reasons[0]["evidence"]["grounding_status"] == grounding.value


# ------------------------------------------------------------------------------ (iv)


def test_iv_gate_off_never_reaches_the_exception(ungated):
    for name, cs in ungated.items():
        assert cs.grounding_status is not GroundingStatus.SUBTASK_GATED, name
        assert not subtask_no_target(cs), name
        assert geometry_certified(cs) is (cs.status.value == "ok"), name


@pytest.mark.parametrize("grounding", [g for g in GroundingStatus
                                       if g is not GroundingStatus.SUBTASK_GATED],
                         ids=lambda g: g.value)
@pytest.mark.parametrize("status", list(PipelineStatus), ids=lambda s: s.value)
@pytest.mark.parametrize("validity", list(ConstraintValidity), ids=lambda v: v.value)
def test_iv_without_subtask_gated_the_predicate_is_the_old_status_ok(gated, grounding, status,
                                                                    validity):
    cs = _replace(gated["released"], grounding_status=grounding, status=status, validity=validity)
    # SUBTASK-f: this frame is gate on ∧ home ∧ PLACED seen, so `no_admissible` + no_target + valid
    # is the one other combination that is certified (gate off: test_subtask_no_admissible_home).
    placed_home = (grounding is GroundingStatus.NO_ADMISSIBLE
                   and status is PipelineStatus.NO_TARGET and validity is ConstraintValidity.VALID)
    assert geometry_certified(cs) is (status is PipelineStatus.OK or placed_home)


def _verdict_snapshot(cs, trajopt_status, policy, monkeypatch, *, old_rule):
    sp = _policy(verdict_policy=policy)
    with monkeypatch.context() as m:
        if old_rule:
            # the pre-SUBTASK-e rule: `status == "ok"`, and no `subtask_no_target` branch
            m.setattr(sp_mod, "subtask_no_target", lambda _cs: False)
        _wire(sp, cs, trajopt_status)
        v = sp._verdict(None)                                                   # noqa: SLF001
        return v.to_dict(), sp._last_reasons                                    # noqa: SLF001


@pytest.mark.parametrize("policy", ["reasons", "legacy"])
@pytest.mark.parametrize("trajopt_status", ["optimal", "violated"])
def test_iv_gate_off_verdicts_are_identical_to_the_old_rule(ungated, policy, trajopt_status,
                                                            monkeypatch):
    frames = list(ungated.values()) + [
        _replace(ungated["pick"], target=None, status=PipelineStatus.NO_TARGET,
                 grounding_status=g) for g in OTHER_GROUNDINGS]
    for cs in frames:
        new = _verdict_snapshot(cs, trajopt_status, policy, monkeypatch, old_rule=False)
        old = _verdict_snapshot(cs, trajopt_status, policy, monkeypatch, old_rule=True)
        assert new == old, (cs.grounding_status, cs.status)


def test_iv_control_the_old_rule_held_the_gated_frame(gated, monkeypatch):
    """The comparison above is not vacuous: on a *gated* frame the two rules disagree — the old one
    is the SUBTASK-d HOLD (`uncertified`, `status=no_target, grounding=subtask_gated`)."""
    new = _verdict_snapshot(gated["released"], "violated", "reasons", monkeypatch, old_rule=False)
    old = _verdict_snapshot(gated["released"], "violated", "reasons", monkeypatch, old_rule=True)
    assert new != old
    assert old[0]["safe"] is False and old[0]["geometry_certified"] is False
    assert [r["kind"] for r in old[1]] == ["uncertified"]
    assert "grounding=subtask_gated, validity=valid" in old[1][0]["detail"]


def test_iv_gate_off_scene_fn_matches_the_old_rule(ungated, monkeypatch):
    for cs in ungated.values():
        assert _scene_fn_certified(cs, monkeypatch) is (cs.status.value == "ok")


# ------------------------------------------------------------------------------ (v)


class _Manip:
    source = "target"


@pytest.mark.parametrize("change", [
    {"manipulated": _Manip()},
    {"attached": object()},
    {"target_field_exclude": np.ones(3, bool)},
    {"manipulated_link_margin": np.zeros(3)},
    {"metrics": "manipulated_record"},
    {"metrics": "exclusion_active"},
    {"metrics": "exclusion_source_manipulated"},
    {"metrics": "exclusion_missing"},
], ids=["manipulated", "attached", "target_free_layer", "contact_margin", "manipulated_record",
        "exclusion_active", "exclusion_source", "exclusion_missing"])
def test_v_anything_carved_keeps_the_exception_off(gated, change):
    base = gated["released"]
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


def test_v_an_occluded_manipulated_object_still_takes_the_t20_route(gated):
    """subtask_gated with a live (occluded) manipulated object is not this exception — the T20
    `occluded_target` rule decides, unchanged."""
    metrics = dict(gated["released"].metrics,
                   manipulated={"id": 7, "state": "occluded", "age_frames": 2})
    cs = _replace(gated["released"], manipulated=_Manip(), metrics=metrics)
    sp = _policy()
    _wire(sp, cs, "violated")
    v = sp._verdict(None)                                                       # noqa: SLF001
    assert [r["kind"] for r in v.verdict_reasons] == ["occluded_target"]
    assert v.geometry_certified is False


def test_the_kind_is_registered_on_the_execute_side():
    assert "subtask_no_target" in wire.REASON_KINDS
    assert wire.GATE_DEFAULT["subtask_no_target"] == "execute"
    assert wire.make_reason("subtask_no_target")["action"] == "execute"


# ------------------------------------------------------------------------------ helpers


def _replace(cs, **changes):
    return dataclasses.replace(cs, **changes)


def _policy(**kw):
    from tests.trajopt.test_safe_policy import _safe_policy

    return _safe_policy(**kw)


class _Status:
    def __init__(self, value):
        self.value = value

    @property
    def safe(self):
        return self.value in ("optimal", "feasible")


def _wire(sp, cs, trajopt_status):
    sp.refiner.last_result = types.SimpleNamespace(
        status=_Status(trajopt_status), max_violation=0.0, trajectory=None, notes=[],
        metrics={"max_violation_pair": None, "time_budget_hit": False,
                 "max_iterations_hit": False, "sqp_iterations": 2})
    sp._last_constraint_set = cs                                                # noqa: SLF001
    sp._last_snapshot = None                                                    # noqa: SLF001


def _scene_fn_certified(cs, monkeypatch):
    """Run `SafePolicy._scene_fn` with AG3S returning `cs`; return the `certified` it hands the
    refiner (→ `sqp.solve(geometry_certified=…)`)."""
    sp = _policy()
    q = np.zeros(sp.constraint_robot_model.nq)
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q, camera_id="zed_left")]
    with monkeypatch.context() as m:
        m.setattr(wire, "unpack_camera_observations", lambda scene, **kw: obs)
        m.setattr(sp.ag3s, "process_multi_debug", lambda *a, **k: (cs, {}))
        m.setattr(sp, "_advance_grasp", lambda *a, **k: None)
        m.setattr(sp, "_run_latch", lambda *a, **k: None)
        m.setattr(sp, "_finish_grasp_record", lambda *a, **k: None)
        m.setattr(sp_mod, "scene_from_constraint_set", lambda *a, **k: "snapshot")
        sp._pending = {"scene": {}, "exec_feedback": wire.unpack_exec_feedback({})}  # noqa: SLF001
        snapshot, _q, certified = sp._scene_fn(None)                            # noqa: SLF001
    assert snapshot == "snapshot"
    return certified
