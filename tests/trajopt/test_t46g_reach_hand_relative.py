"""T46g (SL5b) — the grasp latch's `reach` of an **occluded** manipulated object rides the hand.

`LatchConfig.reach_hand_relative` / `clustering.latch_reach_hand_relative` / `serve_safe
--latch-reach-hand-relative`, default off. SL5b (`verify/T46g-SL5b/sl5b_diag.md`): the closing hand lifts
the object, the object is occluded, `TargetConfirm` keeps its last observed centroid where it stood, and
`reach` (parent link origin ↔ that stale centroid) refuses the grasp at 128–372 mm while every other
check passes. With the flag on, an unobserved object's centroid is carried with the hold link (palm)
from the state of the frame it was observed in to the current state:

    c_now = T_hold(q_now) · T_hold(q_seen)⁻¹ · c_seen

What this file pins:

1. `carried_point` is a rigid carry (identity for equal states, distance to the link preserved).
2. `AG3S._note_manipulated_seen` / `manipulated_seen_state` — the state is pinned at the last frame
   that observed the object (`last_seen_frame`, kept-anchor frames included; not on occluded frames),
   and survives the 16-frame `_state_by_frame` window.
3. SafePolicy: visible → unchanged; occluded → carried; unknown seen state → unchanged (fail-closed).
4. The SL5b pattern, synthetic: lifted + occluded + stale centroid → reach-only refusal with the flag
   off, attach with it on, every other check identical; the attach gate measures the same point and the
   snapshot is taken at the observation state.
5. Flag off: records identical to a policy without the field set (and no new keys); the wiring
   AG3SConfig → SafePolicy and serve_safe → AG3SConfig.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.grasp_latch import GraspPhase, LatchConfig
from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _target_at

PARENT = "ee_finger_l1"
HOLD_LINK = "ee_left"
LEFT_ARM_1 = 14       # `_robot_model().joint_names` index of left_arm_1


def _sp(on=None, **kw):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    clustering = {"gripper_max_opening": 0.0715}
    if on is not None:
        clustering["latch_reach_hand_relative"] = on
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "clustering": clustering}),
                robot_model=model, constraint_robot_model=model,
                attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s, **kw)


def _ident(centroid, *, state, frame, last_seen_frame, mid=0, geometry=None):
    from AG3S.ag3s.stages.target_grounding import ManipulatedIdentity

    return ManipulatedIdentity(
        id=mid, state=state, geometry=_target_at(centroid) if geometry is None else geometry,
        centroid=np.asarray(centroid, np.float64), age_frames=int(frame - last_seen_frame),
        last_seen_frame=int(last_seen_frame), last_seen_time=1.0, switched_from=None,
        frame=int(frame))


def _poses(sp):
    """q_seen (object observed, hand open beside it) and q_now (the hand lifted/moved ~230 mm)."""
    n = len(sp.constraint_robot_model.joint_names)
    q_seen = np.zeros(n)
    q_now = q_seen.copy()
    q_now[LEFT_ARM_1] = 0.35
    return q_seen, q_now


def _observe_then_occlude(sp, apple, q_seen, *, last_seen_frame=4, frame=7):
    """Frame `last_seen_frame` observed the apple at q_seen (pinned the way `process` pins it); the
    identity SafePolicy then reads is `occluded` with that centroid (stale)."""
    seen = _ident(apple, state="visible", frame=last_seen_frame, last_seen_frame=last_seen_frame)
    sp.ag3s._note_manipulated_seen(seen, q_seen)                                  # noqa: SLF001
    occluded = _ident(apple, state="occluded", frame=frame, last_seen_frame=last_seen_frame,
                      geometry=seen.geometry)
    sp.ag3s._manipulated = occluded                                              # noqa: SLF001
    return occluded


def _run(sp, frames):
    """`frames` = [(q, feedback)]; returns the grasp record of each request (as `_grasp_frames`)."""
    out = []
    for q, fb in frames:
        obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
        sp._pending = {"scene": {}, "exec_feedback": fb}                           # noqa: SLF001
        sp._advance_grasp(obs, ["left"])                                           # noqa: SLF001
        sp._grasp_constraint_version = sp._grasp_state_version                     # noqa: SLF001
        sp._finish_grasp_record(types.SimpleNamespace(attached=sp.ag3s.attached))  # noqa: SLF001
        out.append(dict(sp._last_grasp_record))                                    # noqa: SLF001
    return out


def _sl5b_frames(q_seen, q_now):
    """Open at q_seen; closing (fingers moving) at q_seen; settled on the object **after the lift**
    (q_now) for three requests — the SL5b order: `settled` passes only once the hand has moved."""
    return [(q_seen, _fb(1, [1.0] * 8, 1.0)),
            (q_seen, _fb(2, [0.0] * 8, 0.715)),
            (q_now, _fb(3, [0.0] * 8, 0.715)),
            (q_now, _fb(4, [0.0] * 8, 0.716)),
            (q_now, _fb(5, [0.0] * 8, 0.716))]


# --- 1. the carry ------------------------------------------------------------------------------


def test_carried_point_is_a_rigid_carry_with_the_link():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric, carried_point
    from tests.trajopt.test_safe_policy import _robot_model

    model = _robot_model()
    q0 = np.zeros(len(model.joint_names))
    q1 = q0.copy()
    q1[LEFT_ARM_1], q1[16] = 0.35, -0.4
    T0, T1 = _link_pose_numeric(model, q0, HOLD_LINK), _link_pose_numeric(model, q1, HOLD_LINK)
    p = T0[:3, 3] + np.array([0.04, -0.01, -0.09])
    c = carried_point(model, q0, q1, HOLD_LINK, p)
    # same coordinates in the link frame, same distance to the link origin
    np.testing.assert_allclose(T1[:3, :3].T @ (c - T1[:3, 3]), T0[:3, :3].T @ (p - T0[:3, 3]),
                               atol=1e-12)
    assert np.linalg.norm(c - T1[:3, 3]) == pytest.approx(np.linalg.norm(p - T0[:3, 3]), abs=1e-12)
    assert np.linalg.norm(c - p) > 0.1                                   # it did move with the hand
    np.testing.assert_allclose(carried_point(model, q0, q0, HOLD_LINK, p), p, atol=1e-12)
    np.testing.assert_allclose(carried_point(model, q1, q0, HOLD_LINK, c), p, atol=1e-12)


# --- 2. the pinned observation state -------------------------------------------------------------


def test_the_seen_state_is_pinned_at_the_last_observed_frame():
    sp = _sp()
    ag = sp.ag3s
    q = [np.full(len(sp.constraint_robot_model.joint_names), 0.01 * k) for k in range(40)]
    c0, c1 = np.array([0.4, 0.2, 0.8]), np.array([0.41, 0.2, 0.8])
    assert ag.manipulated_seen_state() is None
    # frame 3 observes c0
    ag._manipulated = _ident(c0, state="visible", frame=3, last_seen_frame=3)   # noqa: SLF001
    ag._note_manipulated_seen(ag._manipulated, q[3])                            # noqa: SLF001
    f, qs = ag.manipulated_seen_state()
    assert f == 3 and np.array_equal(qs, q[3])
    # frame 4 is seen but keeps the anchor (same centroid value): the object was seen at the
    # anchor's place on frame 4 — the pin moves to frame 4 (the frame T32 H2's snapshot uses)
    ag._manipulated = _ident(c0, state="visible", frame=4, last_seen_frame=4)   # noqa: SLF001
    ag._note_manipulated_seen(ag._manipulated, q[4])                            # noqa: SLF001
    assert ag.manipulated_seen_state()[0] == 4
    # frame 5 observes a new centroid
    ag._manipulated = _ident(c1, state="visible", frame=5, last_seen_frame=5)   # noqa: SLF001
    ag._note_manipulated_seen(ag._manipulated, q[5])                            # noqa: SLF001
    assert ag.manipulated_seen_state()[0] == 5
    # occluded for 30 frames (past the 16-frame `_state_by_frame` window): no re-pin, still frame 5
    for k in range(6, 36):
        ag._manipulated = _ident(c1, state="occluded", frame=k, last_seen_frame=5)  # noqa: SLF001
        ag._note_manipulated_seen(ag._manipulated, q[k])                            # noqa: SLF001
    f, qs = ag.manipulated_seen_state()
    assert f == 5 and np.array_equal(qs, q[5])
    # a centroid the pin did not see (e.g. replaced without an observation) or another id: unknown
    ag._manipulated = _ident(c0, state="occluded", frame=36, last_seen_frame=5)  # noqa: SLF001
    assert ag.manipulated_seen_state() is None
    ag._manipulated = _ident(c1, state="occluded", frame=36, last_seen_frame=5, mid=1)  # noqa: SLF001
    assert ag.manipulated_seen_state() is None
    ag.reset()
    assert ag._manip_seen is None and ag.manipulated_seen_state() is None       # noqa: SLF001


def test_process_pins_the_state_it_was_given_for_the_observed_frame():
    """End to end through `AG3S.process` (synthetic scene): the pin is the state of the frame that
    observed the manipulated object."""
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag = AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05}}), robot_model=arm)
    for k in range(3):
        q = np.asarray(DEFAULT_ARM_Q, np.float64) + 0.001 * k
        ag.process(attention_map=scene.attention_grid, depth=scene.depth,
                   camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                   robot_state=q, phase="approach", image_hw=scene.hw)
        m = ag.manipulated
        assert m is not None and m.state == "visible"
        f, qs = ag.manipulated_seen_state()
        assert f == m.last_seen_frame == m.frame and np.array_equal(qs, q)


# --- 3. SafePolicy: which reach -------------------------------------------------------------------


def test_visible_object_keeps_the_observed_reach():
    sp_on, sp_off = _sp(True), _sp()
    recs = []
    for sp in (sp_on, sp_off):
        q_seen, q_now = _poses(sp)
        apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
        sp.ag3s._note_manipulated_seen(_ident(apple, state="visible", frame=4, last_seen_frame=4),  # noqa: SLF001
                                       q_seen)
        sp.ag3s._manipulated = _ident(apple, state="visible", frame=4, last_seen_frame=4)  # noqa: SLF001
        _lock(sp)
        recs.append(_run(sp, _sl5b_frames(q_seen, q_now)))
    on, off = recs
    for a, b in zip(on, off):
        assert a["evidence"]["reach_mm"] == b["evidence"]["reach_mm"]
        assert a["evidence"]["reach_source"] == "observed"
        assert a["evidence"]["reach_carry"]["reason"] == "visible"
        assert a["evidence"]["reach_observed_mm"] == b["evidence"]["reach_mm"]
        assert a["evidence"]["reach_hand_relative_mm"] is None
        assert a["evidence"]["checks"] == b["evidence"]["checks"]
        assert a["attach"] == b["attach"]


def test_unknown_seen_state_keeps_the_observed_reach():
    sp = _sp(True)
    q_seen, q_now = _poses(sp)
    apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
    sp.ag3s._manipulated = _ident(apple, state="occluded", frame=7, last_seen_frame=4)  # noqa: SLF001
    _lock(sp)
    recs = _run(sp, _sl5b_frames(q_seen, q_now))
    assert all(r["evidence"]["reach_source"] == "observed" for r in recs)
    assert all(r["evidence"]["reach_carry"]["reason"] == "seen_state_unknown" for r in recs)
    assert sp.ag3s.attached is None and recs[-1]["evidence"]["checks"]["reach"] is False


# --- 4. the SL5b pattern ----------------------------------------------------------------------


def test_sl5b_lifted_occluded_stale_centroid_refused_off_attached_on():
    out = {}
    for on in (False, True):
        sp = _sp(on)
        q_seen, q_now = _poses(sp)
        apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
        _observe_then_occlude(sp, apple, q_seen)
        _lock(sp)
        out[on] = (sp, _run(sp, _sl5b_frames(q_seen, q_now)), q_seen, q_now, apple)

    sp, off, *_ = out[False]
    # flag off: the SL5b failure — reach is the only failing check once the fingers settle, never attached
    assert sp.ag3s.attached is None and sp._latch.phase is GraspPhase.CLOSING   # noqa: SLF001
    for r in off[2:]:
        assert [k for k, v in r["evidence"]["checks"].items() if not v] == ["reach"]
        assert r["evidence"]["reach_mm"] > 120.0
        assert r["evidence"]["manipulated_state"] == "occluded"
    assert all(k not in r["evidence"] for r in off
               for k in ("reach_source", "reach_observed_mm", "reach_hand_relative_mm", "reach_carry"))

    sp, on, q_seen, q_now, apple = out[True]
    assert [r["attach"] for r in on] == [False, False, True, False, False]
    first = on[2]
    ev = first["evidence"]
    assert ev["reach_source"] == "hand_relative"
    assert ev["reach_observed_mm"] == off[2]["evidence"]["reach_mm"]           # both values recorded
    assert ev["reach_hand_relative_mm"] == ev["reach_mm"] <= 120.0
    # the carried reach equals the finger↔centroid distance at the observation pose (palm-rigid, the
    # finger opening is the same model parameter in both poses here)
    at_seen = np.linalg.norm(apple - _finger_origin(sp, q_seen)) * 1000.0
    assert ev["reach_hand_relative_mm"] == pytest.approx(at_seen, abs=0.01)
    assert ev["reach_carry"]["seen_frame"] == 4 and ev["reach_carry"]["last_seen_frame"] == 4
    assert ev["reach_carry"]["carry_link"] == HOLD_LINK
    assert ev["reach_carry"]["hand_shift_since_seen_mm"] > 100.0
    # every other check is what the flag-off run saw at the same request
    others = {k: v for k, v in off[2]["evidence"]["checks"].items() if k != "reach"}
    assert {k: v for k, v in ev["checks"].items() if k != "reach"} == others
    assert ev["checks"]["reach"] is True and first["rejected"] is None
    # the snapshot was taken at the observation state: the held points sit around the carried centroid
    from AG3S.ag3s.constraints.attached import attached_points_in_base

    pts = attached_points_in_base(sp.ag3s._attached, robot_model=sp.constraint_robot_model,  # noqa: SLF001
                                  robot_state=q_now)
    carried = np.asarray(ev["reach_carry"]["centroid_carried_m"])
    assert np.linalg.norm(pts.mean(axis=0) - carried) < 0.02
    assert sp.ag3s._held_record["snapshot_state"] == "given"                    # noqa: SLF001
    assert sp.ag3s._held_record["hand_shift_since_capture_mm"] > 100.0          # noqa: SLF001


def test_sl5b_flag_on_without_lift_is_identical_to_off_but_for_the_new_keys():
    """Occluded but the hand has not moved: the carried reach is the observed one; the decision and
    every shared key are equal."""
    recs = {}
    for on in (False, True):
        sp = _sp(on)
        q_seen, _ = _poses(sp)
        apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
        _observe_then_occlude(sp, apple, q_seen)
        _lock(sp)
        recs[on] = _run(sp, _sl5b_frames(q_seen, q_seen))
    new = {"reach_source", "reach_observed_mm", "reach_hand_relative_mm", "reach_carry"}

    def same(rec):
        """The record without the attach wall-clock time and the snapshot-state label: on gives the
        observation state explicitly (`given`), off looks it up (`attach_request` here — this test
        runs no `process`, so `_state_by_frame` is empty); the state values are equal (no lift)."""
        out = {k: v for k, v in rec.items() if k != "evidence"}
        if out.get("attached"):
            att = dict(out["attached"])
            att.pop("attached_at", None)
            att["held"] = {k: v for k, v in (att.get("held") or {}).items() if k != "snapshot_state"}
            out["attached"] = att
        return out

    for a, b in zip(recs[True], recs[False]):
        assert {k: v for k, v in a["evidence"].items() if k not in new} == b["evidence"]
        assert same(a) == same(b)
        assert a["evidence"]["reach_source"] == "hand_relative"
        assert a["evidence"]["reach_hand_relative_mm"] == a["evidence"]["reach_observed_mm"]


def test_the_attach_gate_measures_the_carried_point():
    """`AG3S.attach(reach_centroid=)` replaces the geometry centroid in the gate only."""
    from AG3S.ag3s.constraints.attached import AttachRejected

    sp = _sp(True)
    q_seen, q_now = _poses(sp)
    apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
    geom = _target_at(apple)
    with pytest.raises(AttachRejected, match="reach"):
        sp.ag3s.attach(geom, robot_state=q_now, parent_link=PARENT, reach=0.12, hold_link=HOLD_LINK)
    from AG3S.ag3s.constraints.attached import carried_point

    c_now = carried_point(sp.constraint_robot_model, q_seen, q_now, HOLD_LINK, geom.centroid)
    sp.ag3s.attach(geom, robot_state=q_now, parent_link=PARENT, reach=0.12, hold_link=HOLD_LINK,
                   geometry_state=q_seen, reach_centroid=c_now)
    assert sp.ag3s.attached is not None


# --- 5. off = before; wiring ------------------------------------------------------------------


def test_off_by_default_and_explicit_false_records_are_identical():
    assert LatchConfig().reach_hand_relative is False
    recs = []
    for on in (None, False):
        sp = _sp(on)
        assert sp._latch.config.reach_hand_relative is False                    # noqa: SLF001
        q_seen, q_now = _poses(sp)
        apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
        _observe_then_occlude(sp, apple, q_seen)
        _lock(sp)
        recs.append(_run(sp, _sl5b_frames(q_seen, q_now)))
    assert recs[0] == recs[1]


def test_config_field_default_validation_and_round_trip():
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

    assert AG3SConfig().clustering.latch_reach_hand_relative is False
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"}).with_overrides(
        {"clustering": {"latch_reach_hand_relative": True}})
    assert cfg.clustering.latch_reach_hand_relative is True
    assert AG3SConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(AG3SConfigError, match="latch_reach_hand_relative"):
        AG3SConfig.from_dict({"clustering": {"latch_reach_hand_relative": 1}})


def test_safe_policy_turns_the_latch_rule_on_from_the_ag3s_config_only():
    assert _sp()._latch.config.reach_hand_relative is False                    # noqa: SLF001
    assert _sp(True)._latch.config.reach_hand_relative is True                 # noqa: SLF001
    sp = _sp(True, latch=LatchConfig(release_frames=3, slip_detach=True))
    cfg = sp._latch.config                                                      # noqa: SLF001
    assert cfg.reach_hand_relative and cfg.release_frames == 3 and cfg.slip_detach
    assert cfg.reach == pytest.approx(0.12)                                     # the threshold is untouched
    assert _sp(latch=LatchConfig(reach_hand_relative=True))._latch.config.reach_hand_relative  # noqa: SLF001


def test_cli_flag_maps_to_the_clustering_section_and_needs_the_safe_layer():
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.trajopt import serve_safe
    from tests.trajopt.test_t41a_obstacle_margin import BASE_ARGV

    ap = serve_safe.build_parser()
    assert "latch_reach_hand_relative" not in serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV))
    sec = serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV + ["--latch-reach-hand-relative"]))
    assert sec == {"latch_reach_hand_relative": True}
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"backend": "curobo"}}) \
        .with_overrides({"clustering": sec})
    serve_safe.assert_ag3s_config(cfg, collision_backend="esdf", esdf_backend="curobo", clustering=sec)
    with pytest.raises(SystemExit):
        bad = ap.parse_args(BASE_ARGV + ["--no-safe", "--latch-reach-hand-relative"])
        serve_safe.reject_bad_flag_combinations(ap, bad)
