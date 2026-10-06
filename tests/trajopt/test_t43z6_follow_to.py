"""T43 Z6 (2) — follow the TO plan instead of holding (`clustering.hold_follow_to`,
`serve_safe --hold-follow-to N`, verdict kind `follow_to`; default off).

User principle (2026-10-06 07:10): the crate rim and floor stay obstacles with their margin; the TO
plans around them and the system should follow that plan. Rule (`SafePolicy._follow_to`): a chunk
that would HOLD runs the TO refined chunk when

* its only HOLD reasons are `collision`, every collision row is an ESDF row with d − r ≥ 0
  (margin-only, no penetration),
* the minimum clearance over those rows is not below the same query spheres' clearance at the
  current state (stationary plan at q_now, first step),
* the refined arm rows differ from the policy reference in the executed window,
* fewer than N such chunks since the last chunk that certified on its own.

Synthetic rig of `test_verdict_reasons.py` (analytic sphere field). Off = the verdict and the record
are as before.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.trajopt import wire
from tests.trajopt.test_verdict_reasons import (  # noqa: F401 — `rig` is a pytest fixture
    _cs,
    _Field,
    _result,
    _scene,
    _wire_up,
    rig,
)

MARGIN = 0.05


def _sp(limit=None, **kw):
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    clustering = {"gripper_max_opening": 0.0715}
    if limit is not None:
        clustering["hold_follow_to"] = limit
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "clustering": clustering}),
                robot_model=model, constraint_robot_model=model,
                attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s, **kw)


def _beside(rig, query, gap, *, step=0, radius=0.02):
    """An obstacle sphere `gap` (d − r) away from `query`'s sphere at `step`, on the far side of the
    robot's sphere cloud (so no other sphere is nearer)."""
    c = rig.centres[step, query]
    out = c - rig.centres[step].mean(axis=0)
    out /= np.linalg.norm(out)
    r = float(rig.lin.query_radii[query])
    return c + out * (r + radius + gap), radius


def _torso_scene(rig, gap):
    centre, radius = _beside(rig, rig.torso, gap)
    return _scene(rig, _Field([("crate", centre, radius)]), margin=MARGIN, manipulated=False)


def _chunks(rig, moved=0.01):
    policy = np.array(rig.s.chunk, np.float64, copy=True)
    refined = policy.copy()
    refined[:, 0] += moved
    return policy, refined


def _verdict(sp, rig, scene, *, moved=0.01, cs=None, violation=0.03):
    policy, refined = _chunks(rig, moved)
    result = _result("violated", violation, rig.s.reference)
    result.chunk = refined
    _wire_up(sp, rig, scene, result, cs or _cs())
    return sp._verdict(policy)


def test_torso_stays_put_so_plan_and_current_clearance_agree(rig):
    stay = np.tile(np.asarray(rig.s.q_now)[rig.lin.layout.q_indices].reshape(-1, 1),
                   (1, rig.lin.horizon))
    now = rig.lin.sphere_states(stay, rig.s.q_now)[0]
    assert np.allclose(now[0, rig.torso], rig.centres[:, rig.torso])


def test_off_holds_as_before_and_records_nothing_new(rig):
    sp = _sp()
    assert sp.follow_to_limit is None
    v = _verdict(sp, rig, _torso_scene(rig, 0.02))
    assert v.safe is False and [r["kind"] for r in v.verdict_reasons] == ["collision"]
    summary = sp._exec_summary()["verdict"]
    assert "follow_to" not in summary and "rows" not in summary["classification"]


def test_margin_only_not_worse_rows_run_the_refined_chunk(rig):
    off = _sp()
    v_off = _verdict(off, rig, _torso_scene(rig, 0.02))
    sp = _sp(limit=3)
    v = _verdict(sp, rig, _torso_scene(rig, 0.02))
    assert v.safe is True
    kinds = [r["kind"] for r in v.verdict_reasons]
    assert kinds == ["follow_to"]
    follow = v.verdict_reasons[0]
    assert follow["action"] == "execute"
    ev = follow["evidence"]
    assert ev["min_esdf_minus_r_m"] == pytest.approx(0.02, abs=1e-6)
    assert ev["plan_min_clearance_m"] == pytest.approx(0.02 - MARGIN, abs=1e-6)
    assert ev["now_min_clearance_m"] == pytest.approx(ev["plan_min_clearance_m"], abs=1e-9)
    assert ev["streak"] == 1 and ev["limit"] == 3
    assert ev["replaced"] == [dict(r) for r in v_off.verdict_reasons]   # the HOLD it replaced
    assert v.trajopt_status == "violated", "the optimizer's status is not rewritten"
    summary = sp._exec_summary()["verdict"]
    assert summary["follow_to"]["decision"] == "follow_to"
    # the recorded classification is the off one (rows are not recorded)
    assert summary["classification"] == off._exec_summary()["verdict"]["classification"]
    # the client's gate executes it; a client that does not know the kind holds (fail closed)
    assert wire.gate_decision(v.to_dict()[wire.VERDICT_REASONS])[0] == "execute"
    old = {k: a for k, a in wire.GATE_DEFAULT.items() if k != "follow_to"}
    assert wire.gate_decision(v.to_dict()[wire.VERDICT_REASONS], old)[0] == "hold"


def test_penetration_holds(rig):
    sp = _sp(limit=3)
    v = _verdict(sp, rig, _torso_scene(rig, -0.003))
    assert v.safe is False and [r["kind"] for r in v.verdict_reasons] == ["collision"]
    rec = sp._exec_summary()["verdict"]["follow_to"]
    assert rec["refused"] == "penetration" and rec["min_esdf_minus_r_m"] < 0.0


def test_a_plan_that_gets_closer_than_now_holds(rig):
    tip = 33
    centre, radius = _beside(rig, tip, 0.01, step=3, radius=0.005)
    scene = _scene(rig, _Field([("crate", centre, radius)]), margin=MARGIN, manipulated=False)
    sp = _sp(limit=3)
    v = _verdict(sp, rig, scene)
    assert v.safe is False
    rec = sp._exec_summary()["verdict"]["follow_to"]
    assert rec["min_esdf_minus_r_m"] >= 0.0, rec                    # margin-only ...
    assert rec["refused"] == "worsens"                              # ... but deeper than now
    assert rec["plan_min_clearance_m"] < rec["now_min_clearance_m"]


def test_another_hold_kind_keeps_the_hold(rig):
    sp = _sp(limit=3)
    v = _verdict(sp, rig, _torso_scene(rig, 0.02), cs=_cs(status="degraded", validity_ok=False))
    assert v.safe is False
    assert {r["kind"] for r in v.verdict_reasons} == {"uncertified", "collision"}
    assert sp._exec_summary()["verdict"]["follow_to"]["refused"] == "hold_kinds"


def test_the_policy_reference_is_not_what_follow_to_runs(rig):
    sp = _sp(limit=3)
    v = _verdict(sp, rig, _torso_scene(rig, 0.02), moved=0.0)
    assert v.safe is False
    assert sp._exec_summary()["verdict"]["follow_to"]["refused"] == "refined_is_reference"


def test_the_consecutive_limit_and_its_reset_by_a_certified_chunk(rig):
    sp = _sp(limit=2)
    decisions = []
    for _ in range(3):
        v = _verdict(sp, rig, _torso_scene(rig, 0.02))
        decisions.append((v.safe, sp._exec_summary()["verdict"]["follow_to"].get("refused")))
    assert decisions == [(True, None), (True, None), (False, "limit")]
    # a HOLD does not reset the count
    v = _verdict(sp, rig, _torso_scene(rig, 0.02))
    assert v.safe is False and sp._follow_to_streak == 2
    # a chunk that certifies on its own does
    clean = _scene(rig, _Field([("crate", rig.centres[0, rig.torso] + [5.0, 0, 0], 0.02)]),
                   margin=MARGIN, manipulated=False)
    v = _verdict(sp, rig, clean, violation=0.0)
    assert v.safe is True and v.verdict_reasons == []
    assert sp._exec_summary()["verdict"]["follow_to"]["decision"] == "not_needed"
    assert sp._follow_to_streak == 0
    v = _verdict(sp, rig, _torso_scene(rig, 0.02))
    assert v.safe is True and v.verdict_reasons[0]["evidence"]["streak"] == 1
    # an episode boundary resets too
    sp._follow_to_streak = 2
    sp.reset()
    assert sp._follow_to_streak == 0


def test_legacy_verdict_policy_is_untouched(rig):
    sp = _sp(limit=3, verdict_policy="legacy")
    v = _verdict(sp, rig, _torso_scene(rig, 0.02))
    assert v.safe is False and [r["kind"] for r in v.verdict_reasons] == ["collision"]


# --- wiring -----------------------------------------------------------------------------------


def test_config_default_validation_round_trip_and_safe_policy_reads_it():
    from benchmark.ag3s.config import AG3SConfig, AG3SConfigError
    from benchmark.trajopt.safe_policy import hold_follow_to_limit

    assert AG3SConfig().clustering.hold_follow_to is None
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"}).with_overrides(
        {"clustering": {"hold_follow_to": 6}})
    assert cfg.clustering.hold_follow_to == 6 and cfg.collision_backend == "esdf"
    assert AG3SConfig.from_dict(cfg.to_dict()) == cfg
    for bad in (0, -1, 2.5, True, "6"):
        with pytest.raises(AG3SConfigError, match="hold_follow_to"):
            AG3SConfig.from_dict({"clustering": {"hold_follow_to": bad}})
    assert _sp(4).follow_to_limit == 4 and _sp().follow_to_limit is None
    assert hold_follow_to_limit(object()) is None


def test_cli_flag_maps_to_the_clustering_section_and_needs_the_safe_layer():
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.trajopt import serve_safe
    from tests.trajopt.test_t41a_obstacle_margin import BASE_ARGV

    ap = serve_safe.build_parser()
    assert "hold_follow_to" not in serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV))
    sec = serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV + ["--hold-follow-to", "6"]))
    assert sec == {"hold_follow_to": 6}
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"backend": "curobo"}}) \
        .with_overrides({"clustering": sec})
    serve_safe.assert_ag3s_config(cfg, collision_backend="esdf", esdf_backend="curobo", clustering=sec)
    for argv in (BASE_ARGV + ["--hold-follow-to", "6", "--no-safe"],
                 BASE_ARGV + ["--hold-follow-to", "0"]):
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(ap, ap.parse_args(argv))
