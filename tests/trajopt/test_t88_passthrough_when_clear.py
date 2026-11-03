"""T88 — pass the policy chunk through when it is already clear (`serve_safe --passthrough-when-clear`,
`sqp.passthrough_when_clear`, default False). Task: `AG3S/ag3s/docs/handoff/T88.task.md` §1.

What this file pins:
1. Off (default): the config dump and the CLI have no new key; the new functions are never reached (they are replaced
   by ones that raise) on a clear chunk and on a violating chunk, and the result carries no `passthrough_clear`;
   the off results do not move (the golden hashes of `test_e2_multi_seed.py` run in the same suite).
2. On, the reference has no violation and is inside the limits: the output IS the reference (trajectory and chunk,
   gripper columns included), no optimization step ran, the record says applied, `_previous` / `frame_index` as after
   a solve.
3. On, the reference violates: the very same result as off (bit for bit), plus a record saying why not.
4. On, a reference outside the joint limits or an uncertified geometry: not passed through.
5. Look-ahead: the executed window decides (open gate); a closed gate hands the window solve the same rule and the
   tail stays the reference.
6. `multi_seed` / grasp-anchor / path rows composition.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.types import TrajOptStatus

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]


def _opt(sc, on=False, **over):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    o = dict(over)
    if on:
        o["sqp"] = {**o.get("sqp", {}), "passthrough_when_clear": True}
    cfg = sc.config.with_overrides(o) if o else sc.config
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)


def _inside(sc, penetration):
    """The fixture is a hair outside the joint box (left_arm_3 rests at 0, its limit is -0.01; the sweep ends 0.01 rad
    past right_arm_0's). Move the rest pose of left_arm_3 inside and scale the sweep to 80 %, so the reference and the
    robot are inside the limits, and place the obstacle for `penetration` against this reference (negative = clear by
    that much)."""
    from tests.trajopt.fixtures import near_miss_scene

    names = [sc.layout.joint_names[q] for q in sc.layout.q_indices]
    q_now = sc.q_now.copy()
    q_now[sc.layout.q_indices[names.index("left_arm_3")]] = -0.5
    start = q_now[sc.layout.q_indices][:, None]
    ref = start + 0.8 * (sc.reference - sc.q_now[sc.layout.q_indices][:, None])
    chunk = sc.chunk.copy()
    chunk[:, sc.layout.action_columns] = ref.T
    scene = near_miss_scene(sc.linearizer, ref, q_now, penetration=penetration)
    return dataclasses.replace(sc, reference=ref, chunk=chunk, scene=scene, q_now=q_now,
                               reference_violation=-sc.linearizer.full_violation(ref, q_now, scene))


def _clear():
    from tests.trajopt.fixtures import make_scenario

    return _inside(make_scenario(horizon=8), -0.05)     # the obstacle is 5 cm further than the floor requires


def _violating():
    from tests.trajopt.fixtures import make_scenario

    return _inside(make_scenario(horizon=8), 0.03)


def _no_qp(monkeypatch):
    """Any optimization step (a merit evaluation) is an error."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    def boom(*a, **k):
        raise AssertionError("an optimization step ran")

    monkeypatch.setattr(TrajectoryOptimizer, "_merit_terms", boom)
    monkeypatch.setattr(TrajectoryOptimizer, "_stay_start", boom, raising=True)


# ------------------------------------------------------------------------------------------------ 1. off
def test_default_is_off_and_the_dump_has_no_new_key():
    c = TrajOptConfig()
    assert c.sqp.passthrough_when_clear is False
    assert "passthrough_when_clear" not in c.to_dict()["sqp"]
    on = c.with_overrides({"sqp": {"passthrough_when_clear": True}})
    assert on.to_dict()["sqp"]["passthrough_when_clear"] is True
    assert TrajOptConfig.from_dict(on.to_dict()) == on                    # round trip
    assert TrajOptConfig.from_dict(c.to_dict()) == c
    with pytest.raises(TrajOptConfigError, match="passthrough_when_clear"):
        TrajOptConfig.from_dict({"sqp": {"passthrough_when_clear": 1}})


def test_cli_flag_reaches_the_config_and_absent_flag_changes_nothing(capsys):
    from AG3S.trajopt import serve_safe

    plain_args = serve_safe.build_parser().parse_args(BASE)
    assert serve_safe.passthrough_overrides(plain_args) == {}
    plain = serve_safe.trajopt_config_from_args(plain_args)
    assert plain.sqp.passthrough_when_clear is False
    on = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(BASE + ["--passthrough-when-clear"]))
    assert on.sqp.passthrough_when_clear is True
    assert on == plain.with_overrides({"sqp": {"passthrough_when_clear": True}})
    assert on.to_dict()["sqp"].keys() - plain.to_dict()["sqp"].keys() == {"passthrough_when_clear"}


def test_announce_says_on_only_when_on(caplog):
    import logging

    from AG3S.trajopt import serve_safe

    with caplog.at_level(logging.WARNING):
        serve_safe.announce_ta(TrajOptConfig())
        assert "passthrough" not in caplog.text
        serve_safe.announce_ta(TrajOptConfig().with_overrides({"sqp": {"passthrough_when_clear": True}}))
    assert "TO passthrough when clear (T88): ON" in caplog.text


@pytest.mark.parametrize("make", [_clear, _violating])
def test_off_never_reaches_the_new_code(monkeypatch, make):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    sc = make()
    base = _opt(sc).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)

    def boom(*a, **k):
        raise AssertionError("T88 code reached with the switch off")

    monkeypatch.setattr(TrajectoryOptimizer, "_clear_passthrough", boom)
    monkeypatch.setattr(TrajectoryOptimizer, "_finish_passthrough", boom)
    out = _opt(sc).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert "passthrough_clear" not in out.metrics and "passthrough_clear" not in base.metrics
    assert out.trajectory.tobytes() == base.trajectory.tobytes() and out.chunk.tobytes() == base.chunk.tobytes()
    assert set(out.metrics) == set(base.metrics)
    # the explicit False is the default
    off = _opt(sc, sqp={"passthrough_when_clear": False}).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert off.trajectory.tobytes() == base.trajectory.tobytes()


# ------------------------------------------------------------------------------------------------ 2. on, clear
def test_clear_reference_is_returned_unchanged(monkeypatch):
    sc = _clear()
    assert sc.reference_violation < 0.0
    off = _opt(sc).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    _no_qp(monkeypatch)
    opt = _opt(sc, on=True)
    ref = sc.reference.copy()
    out = opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    assert out.trajectory.tobytes() == sc.reference.tobytes()
    assert out.chunk.tobytes() == sc.chunk.tobytes()                       # gripper columns and padding included
    assert ref.tobytes() == sc.reference.tobytes()                         # the caller's array untouched
    assert out.status is TrajOptStatus.OPTIMAL and out.status.safe
    assert out.iterations == 0 and out.reference_deviation == 0.0 and out.max_violation == 0.0
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is True and rec["reference_violation_m"] == 0.0
    assert opt.frame_index == 1 and np.array_equal(opt._previous, sc.reference)
    # the measured keys of an optimized chunk are there (the recorder reads them); the numbers are the reference's
    assert set(off.metrics) <= set(out.metrics)
    assert out.metrics["clearance_m"] == pytest.approx(-sc.reference_violation, abs=1e-9)
    assert out.metrics["limit_overshoot"]["position"] == 0.0


def test_clear_reference_with_off_is_not_marked():
    sc = _clear()
    out = _opt(sc).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert "passthrough_clear" not in out.metrics


# ------------------------------------------------------------------------------------------------ 3. on, violating
def test_violating_reference_takes_the_same_path_as_off():
    sc = _violating()
    assert sc.reference_violation > 0.0
    off = _opt(sc).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    on = _opt(sc, on=True).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert on.trajectory.tobytes() == off.trajectory.tobytes() and on.chunk.tobytes() == off.chunk.tobytes()
    assert on.status is off.status and on.max_violation == off.max_violation and on.cost == off.cost
    rec = on.metrics["passthrough_clear"]
    assert rec["applied"] is False and rec["reason"] == "collision"
    assert rec["reference_violation_m"] == pytest.approx(sc.reference_violation, abs=1e-9)
    assert {k: v for k, v in on.metrics.items() if k != "passthrough_clear"}.keys() == off.metrics.keys()


def test_a_second_chunk_after_a_passthrough_is_optimized_like_any_chunk():
    """An optimizer that passed one chunk through then meets a violating one: the solve sees the passed chunk as its
    previous plan (`_previous`), as after any solve."""
    sc = _clear()
    bad = _violating()
    opt = _opt(sc, on=True)
    first = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert first.metrics["passthrough_clear"]["applied"]
    second = opt.solve(bad.reference, bad.q_now, bad.scene, template=bad.chunk)
    assert second.metrics["passthrough_clear"]["applied"] is False
    assert second.reference_deviation > 0.0 and opt.frame_index == 2


# ------------------------------------------------------------------------------------------------ 4. limits / certification
def test_a_reference_outside_the_limits_is_not_passed_through():
    sc = _clear()
    axis = list(sc.layout.joint_names[q] for _, q in sc.layout.action_to_q).index("right_arm_0")
    ref = sc.reference.copy()
    ref[axis, :] += 10.0                         # far outside the joint box (and the first-step anchor)
    opt = _opt(sc, on=True)
    out = opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is False and rec["reason"] in ("limit", "collision")
    if rec["reason"] == "limit":
        assert rec["reference_violation_m"] == 0.0
    assert out.trajectory.tobytes() != ref.tobytes()
    # the very same call with the switch off
    off = _opt(sc).solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    assert out.trajectory.tobytes() == off.trajectory.tobytes()


def test_the_step0_anchor_counts_as_a_limit():
    """A reference that starts further than `max_step` from the robot is outside what the QP admits: optimized."""
    sc = _clear()
    ref = sc.reference.copy()
    ref[:, :] += 0.5 * np.ones((ref.shape[0], 1))   # a half-radian jump on every joint at step 0
    opt = _opt(sc, on=True)
    assert max(opt._limit_overshoot(ref, sc.q_now).values()) > 1e-6
    out = opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["applied"] is False


def test_uncertified_geometry_is_not_passed_through_when_certification_is_required():
    sc = _clear()
    cfg_on = {"safety": {"require_certified_geometry": True}}
    out = _opt(sc, on=True, **cfg_on).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk,
                                            geometry_certified=False)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is False and rec["reason"] == "geometry not certified"
    off = _opt(sc, **cfg_on).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk, geometry_certified=False)
    assert out.trajectory.tobytes() == off.trajectory.tobytes() and out.status is off.status


def test_collision_off_is_not_passed_through():
    sc = _clear()
    out = _opt(sc, on=True, collision={"enabled": False}).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["applied"] is False
    assert out.metrics["passthrough_clear"]["reason"] == "collision off"


def test_nothing_to_avoid_is_still_the_unconstrained_passthrough():
    from tests.trajopt.fixtures import empty_scene, make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    out = _opt(sc, on=True).solve(sc.reference, sc.q_now, empty_scene(sc.linearizer), template=sc.chunk)
    assert out.status is TrajOptStatus.UNCONSTRAINED and "passthrough_clear" not in out.metrics


# ------------------------------------------------------------------------------------------------ 5. look-ahead
def _la_scene(sc, step, far=False):
    """A field scene whose sphere (radius 0.1) sits on the most-moving robot sphere at `step` of the reference
    (violation there), or 3 m away (`far`: clear everywhere)."""
    from tests.trajopt.test_t43tp_fast_to import _analytic, _rich_scene
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField

    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    moved = np.linalg.norm(centres[-1] - centres[0], axis=1)
    s = int(np.argmax(moved))
    centre = np.array([3.0, 3.0, 3.0]) if far else centres[step, s]
    field = CuroboEsdfField((_analytic(centre, 0.1, 0.04, 1.6), _analytic(centre, 0.1, 0.02, 0.4)),
                            outside_distance=0.6)
    return dataclasses.replace(_rich_scene(sc, held=False), esdf=field)


def _la_cfg(sc, gate, on):
    from tests.trajopt.test_t43tp_fast_to import _a_over

    over = _a_over(gate)
    over.pop("limits", None)
    over.pop("cost", None)
    return sc.config.with_overrides({**over, "sqp": {"passthrough_when_clear": on}})


def _la_opt(sc, cfg):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)


def _la_scenario():
    from tests.trajopt.fixtures import make_scenario

    return _inside(make_scenario(horizon=10), 0.03)


@pytest.mark.parametrize("gate", [0.5, -1.0, None])
def test_lookahead_clear_everywhere_is_passed_through(gate):
    sc = _la_scenario()
    scene = _la_scene(sc, 0, far=True)
    out = _la_opt(sc, _la_cfg(sc, gate, True)).solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert out.trajectory.tobytes() == sc.reference.tobytes() and out.chunk.tobytes() == sc.chunk.tobytes()
    assert out.metrics["passthrough_clear"]["applied"] is True
    assert out.metrics["lookahead"]["gate"] == {0.5: "open", -1.0: "closed", None: None}[gate]
    off = _la_opt(sc, _la_cfg(sc, gate, False)).solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert "passthrough_clear" not in off.metrics


@pytest.mark.parametrize("gate", [0.5, None])
def test_lookahead_open_gate_the_window_decides(gate):
    """Violation only in the tail (step 7 of 10, window = steps 0-3): the executed window is clear → passed through
    (the tail is never certified, with look-ahead on or off the optimizer leaves it to the next chunk)."""
    sc = _la_scenario()
    scene = _la_scene(sc, 7)
    cfg = _la_cfg(sc, gate, True)
    lin = _la_opt(sc, cfg).linearizer
    clearances = lin.step_clearances(sc.reference, sc.q_now, scene)
    k = cfg.horizon.lookahead_from
    assert clearances[:k].min() >= 0.0 and clearances[k:].min() < 0.0
    out = _la_opt(sc, cfg).solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["applied"] is True
    assert out.trajectory.tobytes() == sc.reference.tobytes()
    assert out.metrics["lookahead"]["gate"] == "open" or gate is None


@pytest.mark.parametrize("gate", [0.5, -1.0, None])
def test_lookahead_window_violation_is_optimized_exactly_as_off(gate):
    sc = _la_scenario()
    scene = _la_scene(sc, 2)
    cfg_on, cfg_off = _la_cfg(sc, gate, True), _la_cfg(sc, gate, False)
    k = cfg_on.horizon.lookahead_from
    assert _la_opt(sc, cfg_on).linearizer.step_clearances(sc.reference, sc.q_now, scene)[:k].min() < 0.0
    on = _la_opt(sc, cfg_on).solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    off = _la_opt(sc, cfg_off).solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert on.trajectory.tobytes() == off.trajectory.tobytes() and on.chunk.tobytes() == off.chunk.tobytes()
    assert on.metrics["passthrough_clear"]["applied"] is False
    assert on.metrics["passthrough_clear"]["reference_violation_m"] > 0.0
    assert on.metrics["lookahead"]["gate"] == off.metrics["lookahead"]["gate"]


def test_lookahead_closed_gate_window_passthrough_keeps_the_tail_as_the_reference():
    sc = _la_scenario()
    scene = _la_scene(sc, 0, far=True)
    out = _la_opt(sc, _la_cfg(sc, -1.0, True)).solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    k = int(out.metrics["lookahead"]["from"])
    assert out.metrics["lookahead"]["gate"] == "closed"
    assert out.trajectory.shape == sc.reference.shape
    assert out.trajectory[:, :k].tobytes() == sc.reference[:, :k].tobytes()
    assert out.trajectory[:, k:].tobytes() == sc.reference[:, k:].tobytes()


# ------------------------------------------------------------------------------------------------ 6. composition
def test_multi_seed_does_not_trigger_on_a_passed_chunk_and_still_runs_on_a_violating_one():
    sc = _clear()
    out = _opt(sc, on=True, sqp={"multi_seed": 2}).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["applied"] is True
    assert out.metrics["multi_seed"]["triggered"] is False
    assert out.trajectory.tobytes() == sc.reference.tobytes()
    bad = _violating()
    out = _opt(bad, on=True, sqp={"multi_seed": 2}).solve(bad.reference, bad.q_now, bad.scene, template=bad.chunk)
    assert out.metrics["passthrough_clear"]["applied"] is False
    assert "multi_seed" in out.metrics


def test_the_multi_seed_child_solves_are_never_passed_through(monkeypatch):
    """`initial_iterate` (a seed) is an explicit first iterate: the rule is not applied to it."""
    sc = _violating()
    opt = _opt(sc, on=True)
    first = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    out = opt._solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk, initial_iterate=first.trajectory)
    assert "passthrough_clear" not in out.metrics


def test_grasp_anchor_goal_terms_are_skipped_when_the_reference_is_clear():
    """With an anchor on, a clear reference is still passed through: the anchor's goal / palm terms (which would have
    moved it) do not act. Recorded so the choice is visible, not hidden."""
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK

    sc = _clear()
    p = LinkPointFK(sc.robot, sc.layout, 8, "ee_right")(sc.reference, sc.q_now)[0]
    anc = GraspAnchor("ee_right", p[5] + np.array([0.0, 0.0, 0.01]), 5, 0.0, 2000.0)
    out = _opt(sc, on=True).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk, grasp_anchor=anc)
    assert out.metrics["passthrough_clear"]["applied"] is True
    assert out.trajectory.tobytes() == sc.reference.tobytes()


def test_the_seam_to_the_previous_plan_is_recorded_not_limited():
    sc = _clear()
    out = _opt(sc, on=True).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["previous_plan_seam_rad"] is None            # no previous plan
    prev = sc.reference.copy()
    prev[3, :] += 0.07                                                                   # the plan before is 0.07 rad off
    opt = _opt(sc, on=True)
    out = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk, previous_chunk=prev)
    assert out.metrics["passthrough_clear"]["applied"] is True
    assert out.metrics["passthrough_clear"]["previous_plan_seam_rad"] == pytest.approx(0.07)
    assert out.trajectory.tobytes() == sc.reference.tobytes()                            # not smoothed toward it
    # a not-applied chunk carries the two numbers and the reason only
    bad = _violating()
    rec = _opt(bad, on=True).solve(bad.reference, bad.q_now, bad.scene, template=bad.chunk,
                                   previous_chunk=bad.reference).metrics["passthrough_clear"]
    assert set(rec) == {"applied", "reference_violation_m", "reason"}
