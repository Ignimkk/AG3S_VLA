"""SL16 HSC -- ``--held-support-contact`` (task ``AG3S/ag3s/docs/handoff/SL16.task.md`` §2). CPU, analytic fields.

1. The optimizer rows (``CollisionLinearizer._held_support_contact_margin``) on the synthetic scene of ``test_sl6h_placement_contact``
   (table z = 0, plate, wall, shelf, step): held rows whose nearest surface is beneath (or a support plane) get margin -1 m; the
   grasping hand's finger / palm rows get ``min(margin, -finger_mm)`` under the same test (footprint = held discs ∪ hand discs);
   arm rows, side / above surfaces, rows outside the footprint and rows pressed deeper than ``finger_mm`` stay blocked.
2. Off (``held_support_contact=None``) is bit-identical and leaves no trace on the linearizer.
3. ``SafePolicy._held_support_contact``: holding = HELD (any HSR6 phase, also without HSR6) or CLOSING ∧ closed ∧ blocked; the hand
   rows are the grasping hand's finger / palm links, not the arm; the record has the fixed keys; off = no key in the ``to`` block.
4. The serve_safe flags: default off, ``apply_grasp_contact_flags`` sets the margin only when on, the RB-Y1 server refuses it.
5. End to end through the SQP is covered by the placement-contact rig (same ``_finish`` path); here ``to.metrics.held_support_contact``
   is checked on the returned trajectory's final check.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot
from tests.trajopt.test_sl6h_placement_contact import (  # noqa: F401 (held_rig is a fixture)
    R_FINGER, R_HELD, VOXEL, _Scene3, _held_scene, _refine, held_rig)

N_ROBOT = 3
ROWS = {
    "finger_on_plate": 0,        # hand: 3 mm above the plate (r 5 mm) -> obstacle margin 10 mm applies
    "finger_by_wall": 1,         # hand: 4 mm from the wall (side surface)
    "arm_on_plate": 2,           # NOT a hand row: the same pose as finger_on_plate
    "held_on_plate": 3,          # held sphere 10 mm into the plate
    "held_by_wall": 4,           # held sphere 5 mm into the wall from the side
    "held_under_shelf": 5,       # held sphere 5 mm into a shelf above it
    "held_on_table": 6,          # held sphere 5 mm into the table
    "held_step_off_footprint": 7,  # nearest = the step's top edge, below within 45 degrees, outside the footprint
}
CENTRES = np.array([
    [0.03, 0.0, 0.015 + 0.003],
    [0.20 - 0.004, 0.0, 0.15],
    [0.03, 0.0, 0.015 + 0.003],
    [0.0, 0.0, 0.015 + R_HELD - 0.010],
    [0.20 - R_HELD + 0.005, 0.0, 0.15],
    [-0.35, 0.0, 0.30 - R_HELD + 0.005],
    [0.50, 0.30, R_HELD - 0.005],
    [0.70 - 0.06, 0.0, 0.10 + 0.07],
])
HAND = np.array([True, True, False])
FINGER_M = 0.010


def _lin(n_robot=N_ROBOT, n_total=CENTRES.shape[0]):
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.full(n_robot, R_FINGER)
    lin.query_radii = np.concatenate([lin.robot_radii, np.full(n_total - n_robot, R_HELD)])
    lin.n_spheres = n_robot
    lin._attached_local = None
    return lin


def _spec(**kw):
    return {"cos_min": H6.PLACEMENT_BENEATH_COS, "footprint_tol_m": None, "hand_rows": HAND, "finger_margin_m": FINGER_M, **kw}


def _snapshot(field=None, n_robot=N_ROBOT, **extra):
    kw = {"obstacle_margin": 0.010, "support_planes": np.array([[0.0, 0.0, 1.0, 0.0]]), **extra}
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
        d_safe=np.zeros((n_robot, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
        plane_active=np.zeros(0, bool), robot_radii=np.full(n_robot, R_FINGER), candidate_ids=np.zeros(0, np.int64),
        esdf=_Scene3() if field is None else field, esdf_margin=0.0, **kw)


def _clear(lin, scene, centres=CENTRES):
    return lin._esdf_clearance(np.asarray(centres, np.float64)[None], scene)[0, :, 0]


# ------------------------------------------------------------------------------------------------ 1. the rows
def test_the_rig_is_what_it_says():
    off = _clear(_lin(), _snapshot())
    assert off[ROWS["finger_on_plate"]] == pytest.approx(0.003 - R_FINGER - 0.010, abs=1e-9)     # d - r - obstacle margin
    assert off[ROWS["arm_on_plate"]] == off[ROWS["finger_on_plate"]]
    assert off[ROWS["held_on_plate"]] == pytest.approx(-0.010, abs=1e-9)
    assert off[ROWS["held_on_table"]] == pytest.approx(-0.005, abs=1e-9)


def test_on_held_rows_and_hand_rows_beneath_are_allowed_and_nothing_else():
    lin = _lin()
    off = _clear(lin, _snapshot())
    on = _clear(lin, _snapshot(held_support_contact=_spec()))
    # held rows: margin -1 m (the row cannot bind), exactly the HSR6 5 test
    for name in ("held_on_plate", "held_on_table"):
        assert on[ROWS[name]] == pytest.approx(off[ROWS[name]] + 1.0), name
    # the hand row over the plate: the 10 mm obstacle margin becomes -10 mm (a 20 mm gain)
    assert on[ROWS["finger_on_plate"]] == pytest.approx(off[ROWS["finger_on_plate"]] + 0.010 + FINGER_M, abs=1e-9)
    # unchanged: the hand row beside the wall, the ARM row in the finger's pose, held rows beside / above / off the footprint
    for name in ("finger_by_wall", "arm_on_plate", "held_by_wall", "held_under_shelf", "held_step_off_footprint"):
        assert on[ROWS[name]] == off[ROWS[name]], name
    rec = lin._last_held_support_contact
    assert rec["held_rows_exempt"] == 2 and rec["finger_rows_allowed"] == 1
    assert rec["n_held_rows"] == 5 and rec["n_hand_rows"] == 2 and rec["n_rows"] == 7
    assert rec["footprint_tol_m"] == VOXEL and rec["finger_margin_m"] == FINGER_M


def test_the_finger_overlap_is_limited_to_finger_mm():
    """A finger pressed into the plate by more than ``finger_mm`` stays blocked: the allowance is not -1 m."""
    lin = _lin()
    centres = CENTRES.copy()
    centres[ROWS["finger_on_plate"], 2] = 0.015 - 0.002            # 2 mm inside the plate: d = -2 mm
    on = _clear(lin, _snapshot(held_support_contact=_spec()), centres)
    assert on[ROWS["finger_on_plate"]] == pytest.approx(-0.002 - R_FINGER + FINGER_M, abs=1e-9)   # = +3 mm: allowed
    centres[ROWS["finger_on_plate"], 2] = 0.008                    # 7 mm below the plate's top face (inside the 15 mm plate)
    deep = _clear(lin, _snapshot(held_support_contact=_spec()), centres)
    assert deep[ROWS["finger_on_plate"]] < 0.0                     # -7 mm - 5 mm + 10 mm = -2 mm: still blocked
    assert deep[ROWS["finger_on_plate"]] == pytest.approx(-0.007 - R_FINGER + FINGER_M, abs=1e-9)
    wide = _clear(lin, _snapshot(held_support_contact=_spec(finger_margin_m=0.020)), centres)
    assert wide[ROWS["finger_on_plate"]] == pytest.approx(-0.007 - R_FINGER + 0.020, abs=1e-9)    # finger_mm is the knob


def test_footprint_of_a_hand_row_includes_the_hand_spheres_discs():
    """A hand row whose nearest surface point (the step's edge) lies off its own disc is allowed once another hand sphere
    stands over that edge; for a held row the held discs count and the hand discs do not."""
    lin = _lin()
    k = ROWS["finger_by_wall"]
    centres = CENTRES.copy()
    centres[k] = [0.70 - 0.06, 0.0, 0.10 + 0.07]                   # the step's top edge 60 mm aside, 45-degree cone
    off = _clear(lin, _snapshot(), centres)
    alone = _clear(lin, _snapshot(held_support_contact=_spec()), centres)
    assert alone[k] == off[k]                                       # edge outside every disc
    centres[ROWS["finger_on_plate"]] = [0.70, 0.0, 0.10 + 0.03]   # another HAND sphere right over the edge
    off2 = _clear(lin, _snapshot(), centres)
    both = _clear(lin, _snapshot(held_support_contact=_spec()), centres)
    assert both[k] == pytest.approx(off2[k] + 0.010 + FINGER_M, abs=1e-9)
    # the held row at the same edge: only held discs count, so the hand sphere over the edge does not release it
    assert both[ROWS["held_step_off_footprint"]] == off2[ROWS["held_step_off_footprint"]]
    # an ARM sphere over the edge adds no disc either
    centres[ROWS["finger_on_plate"]] = CENTRES[ROWS["finger_on_plate"]]
    centres[ROWS["arm_on_plate"]] = [0.70, 0.0, 0.10 + 0.03]
    off3 = _clear(lin, _snapshot(), centres)
    arm_over = _clear(lin, _snapshot(held_support_contact=_spec()), centres)
    assert arm_over[k] == off3[k]


def test_support_class_rows_are_allowed_without_the_beneath_test():
    """A flat field has no usable gradient (never 'beneath'), yet a row whose nearest surface is a known support plane is allowed."""
    class _Flat(_Scene3):
        def gradient(self, pts):
            return np.zeros((len(np.asarray(pts).reshape(-1, 3)), 3))

    lin = _lin()
    centres = CENTRES.copy()
    centres[ROWS["finger_by_wall"]] = [0.50, -0.30, R_FINGER - 0.002]       # a hand sphere in the table (support plane z = 0)
    pc = _spec()
    off = _clear(lin, _snapshot(_Flat()), centres)
    on = _clear(lin, _snapshot(_Flat(), held_support_contact=pc), centres)
    k = ROWS["finger_by_wall"]
    assert on[k] == pytest.approx(off[k] + FINGER_M, abs=1e-9)           # a support row has margin 0 (no obstacle margin): 0 -> -10 mm
    assert on[ROWS["held_on_table"]] == pytest.approx(off[ROWS["held_on_table"]] + 1.0)             # the held row on the table too
    for name in ("held_on_plate", "held_by_wall", "held_under_shelf", "held_step_off_footprint", "finger_on_plate", "arm_on_plate"):
        assert on[ROWS[name]] == off[ROWS[name]], name                                              # no gradient, no plane: unchanged
    # without the plane list the same flat field releases nothing
    nop = _clear(lin, _snapshot(_Flat(), held_support_contact=pc, support_planes=None), centres)
    assert np.array_equal(nop, _clear(lin, _snapshot(_Flat(), support_planes=None), centres))


def test_it_is_the_placement_rule_widened_not_a_different_test():
    """On a scene with ⑤ and HSC both, the held rows ⑤ releases are the ones HSC releases (the 'beneath' test is the same);
    HSC adds the hand rows and the support planes."""
    lin = _lin()
    pc = _clear(lin, _snapshot(placement_contact=H6.placement_contact_spec()))
    hsc = _clear(lin, _snapshot(held_support_contact=_spec()))
    both = _clear(lin, _snapshot(placement_contact=H6.placement_contact_spec(), held_support_contact=_spec()))
    held = slice(N_ROBOT, None)
    assert np.array_equal(pc[held], hsc[held]) and np.array_equal(both[held], hsc[held])
    assert np.array_equal(pc[:N_ROBOT], _clear(lin, _snapshot())[:N_ROBOT])                  # 5 never touches robot rows


def test_no_hand_rows_and_no_held_rows():
    lin = _lin()
    only_held = _clear(lin, _snapshot(held_support_contact=_spec(hand_rows=np.zeros(N_ROBOT, bool))))
    off = _clear(lin, _snapshot())
    assert np.array_equal(only_held[:N_ROBOT], off[:N_ROBOT]) and lin._last_held_support_contact["finger_rows_allowed"] == 0
    lin2 = _lin(n_total=N_ROBOT)
    only_hand = _clear(lin2, _snapshot(held_support_contact=_spec()), CENTRES[:N_ROBOT])
    assert only_hand[ROWS["finger_on_plate"]] > _clear(lin2, _snapshot(), CENTRES[:N_ROBOT])[ROWS["finger_on_plate"]]
    assert lin2._last_held_support_contact["held_rows_exempt"] == 0 and lin2._last_held_support_contact["n_held_rows"] == 0
    lin3 = _lin(n_total=N_ROBOT)
    none = _clear(lin3, _snapshot(held_support_contact=_spec(hand_rows=np.zeros(N_ROBOT, bool))), CENTRES[:N_ROBOT])
    assert np.array_equal(none, _clear(lin3, _snapshot(), CENTRES[:N_ROBOT]))
    assert lin3._last_held_support_contact["reason"] == "no rows"


def test_every_step_of_a_horizon_is_judged_on_its_own_footprint():
    lin = _lin()
    c0 = CENTRES.copy()
    c1 = CENTRES.copy()
    c1[ROWS["held_on_plate"]] = [0.0, 0.0, 0.40]                      # step 1: the held sphere is far above everything
    traj = np.stack([c0, c1])
    out = lin._esdf_clearance(traj, _snapshot(held_support_contact=_spec()))[..., 0]
    off = lin._esdf_clearance(traj, _snapshot())[..., 0]
    assert out[0, ROWS["held_on_plate"]] == pytest.approx(off[0, ROWS["held_on_plate"]] + 1.0)
    assert out[1, ROWS["held_on_plate"]] == off[1, ROWS["held_on_plate"]]                    # clear anyway: unchanged
    assert lin._last_held_support_contact["steps"] == 2


def test_sqp_a_plate_beneath_the_held_sphere_does_not_move_the_plan_in_any_phase_and_the_metrics_say_so(held_rig):
    """Through the SQP (same rig as HSR6 5): the held sphere sunk 10 mm into a plate moves the arm with HSC off; with it on the plan is
    the plan of the scene without the plate -- no HSR6 phase is involved. ``to.metrics.held_support_contact`` is the final check."""
    rig = held_rig
    held = rig.held
    zmin = float(held[:, 2].min())
    plate = (tuple(np.r_[held[:, :2].min(0) - 0.05, zmin - rig.r - 0.03]),
             tuple(np.r_[held[:, :2].max(0) + 0.05, zmin - rig.r + 0.010]))
    far = ((9.0, 9.0, 9.0), (9.1, 9.1, 9.1))
    n = rig.sc.linearizer.n_spheres
    spec = {"cos_min": H6.PLACEMENT_BENEATH_COS, "footprint_tol_m": None, "hand_rows": np.zeros(n, bool), "finger_margin_m": 0.010}
    free_out, _, _ = _refine(rig, _held_scene(rig, [far]))
    off_out, off_res, _ = _refine(rig, _held_scene(rig, [plate]))
    on_out, on_res, on_ref = _refine(rig, _held_scene(rig, [plate], held_support_contact=spec))
    assert np.abs(off_out - free_out).max() > 1e-3 and "held_support_contact" not in off_res.metrics
    assert np.array_equal(on_out, free_out) and on_ref < 0.0
    rec = on_res.metrics["held_support_contact"]
    assert rec["held_rows_exempt"] > 0 and rec["finger_rows_allowed"] == 0 and rec["n_rows"] == rec["n_held_rows"] + rec["n_hand_rows"]


# ------------------------------------------------------------------------------------------------ 2. off is bit-identical
def test_off_is_bit_identical_and_leaves_no_trace():
    lin = _lin()
    a = _clear(lin, _snapshot())
    b = _clear(lin, _snapshot(held_support_contact=None))
    assert np.array_equal(a, b)
    assert SceneSnapshot.__dataclass_fields__["held_support_contact"].default is None
    assert "_last_held_support_contact" not in lin.__dict__


def test_on_does_not_mutate_the_input_margin_array_or_the_scene():
    scene = _snapshot(held_support_contact=_spec())
    spec_before = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in scene.held_support_contact.items()}
    lin = _lin()
    _clear(lin, scene)
    _clear(lin, scene)
    assert all(np.array_equal(scene.held_support_contact[k], v) for k, v in spec_before.items())


# ------------------------------------------------------------------------------------------------ 3. SafePolicy
#: the RB-Y1 fixture policy's constraint model has arm spheres only; give it a hand: palm (+ shell) on ee_left, the left fingers,
#: one right finger (the other hand) and arm spheres
SPHERE_LINKS = ["link_left_arm_5"] * 3 + ["ee_left"] * 2 + ["ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "link_right_arm_2"]


def _policy(**kw):
    from tests.trajopt.test_grasp_evidence import _policy as build

    sp = build(**kw)
    sp.constraint_robot_model = types.SimpleNamespace(sphere_link_names=list(SPHERE_LINKS))
    return sp


def _with_latch(sp, phase, checks, hsr6_phase=None):
    from AG3S.trajopt.grasp_latch import LatchEvent

    sp._pending = {}
    if hsr6_phase is not None:
        p = {c: (0.9 if c == hsr6_phase else 0.02) for c in H6.PHASES6}
        sp._hsr6_start({"p": p, "argmax": hsr6_phase, "probe": "test"})
    sp._latch.phase = phase
    sp._latch._last_grasp = LatchEvent(phase=phase, evidence={"checks": dict(checks)})


CB = {"closed": True, "blocked": True, "not_empty": True}


def _field_snapshot(n_spheres):
    return dataclasses.replace(_snapshot(n_robot=n_spheres))


@pytest.mark.parametrize("hsr6_phase,latch,checks,applied", [
    (None, GraspPhase.HELD, {}, True),                      # no HSR6 at all: the phase is not asked
    ("lift", GraspPhase.HELD, {}, True),
    ("transport", GraspPhase.HELD, {}, True),
    ("placement", GraspPhase.HELD, {}, True),
    ("grasp", GraspPhase.HELD, {"closed": False}, True),    # HELD needs no evidence (the opening rule sets HELD without it)
    ("transport", GraspPhase.CLOSING, CB, True),
    ("transport", GraspPhase.CLOSING, {**CB, "blocked": False}, False),
    ("transport", GraspPhase.CLOSING, {**CB, "closed": False}, False),
    ("transport", GraspPhase.CLOSING, {}, False),
    ("transport", GraspPhase.LATCHED, CB, False),
    ("transport", GraspPhase.SEARCHING, {}, False),
    ("transport", GraspPhase.PLACED, {}, False)])
def test_safe_policy_gate(hsr6_phase, latch, checks, applied):
    sp = _policy(hsr6=H6.Hsr6Flags(held_gate=True) if hsr6_phase else None)
    sp.held_support_contact_m = 0.010
    _with_latch(sp, latch, checks, hsr6_phase)
    n = len(sp.constraint_robot_model.sphere_link_names)
    snap = _field_snapshot(n)
    out = sp._held_support_contact(snap, ["left"])
    rec = sp._last_held_support_contact
    assert (out.held_support_contact is not None) is applied and rec["applied"] is applied and rec["latch"] == latch.value
    if not applied:
        assert out is snap and rec["reason"] == "not holding"
        return
    spec = out.held_support_contact
    assert spec["cos_min"] == H6.PLACEMENT_BENEATH_COS and spec["footprint_tol_m"] is None
    assert spec["finger_margin_m"] == 0.010 and spec["hand_rows"].shape == (n,)
    assert rec["finger_margin_m"] == 0.010 and rec["n_hand_spheres"] == int(spec["hand_rows"].sum()) == 4


def test_safe_policy_hand_rows_are_the_grasping_hands_finger_and_palm_links_only():
    sp = _policy()
    sp.held_support_contact_m = 0.010
    _with_latch(sp, GraspPhase.HELD, {})
    names = np.asarray(sp.constraint_robot_model.sphere_link_names)
    out = sp._held_support_contact(_field_snapshot(len(names)), ["left"])
    hand, parent, allowed = sp._grasp_hand(["left"])
    want = {str(parent), str(sp.hold_links.get(hand, parent)), *map(str, allowed)}
    got = set(names[out.held_support_contact["hand_rows"]])
    assert got == want & set(names)                                  # fingers + the hold link (palm, palm shell), nothing of the arm
    assert out.held_support_contact["hand_rows"].sum() < len(names)
    assert set(sp._last_held_support_contact["hand_links"]) == want


def test_safe_policy_no_field_is_not_applied():
    sp = _policy()
    sp.held_support_contact_m = 0.010
    _with_latch(sp, GraspPhase.HELD, {})
    snap = dataclasses.replace(_field_snapshot(3), esdf=None)
    out = sp._held_support_contact(snap, ["left"])
    assert out is snap and sp._last_held_support_contact["applied"] is False and sp._last_held_support_contact["reason"] == "no field"


def test_to_block_record_has_the_fixed_keys_and_is_absent_when_off():
    sp = _policy()
    sp.refiner.last_result = types.SimpleNamespace(metrics={})
    assert "held_support_contact" not in (sp._to_block() or {})                                   # off: no key
    sp.held_support_contact_m = 0.010
    _with_latch(sp, GraspPhase.HELD, {})
    sp._held_support_contact(_field_snapshot(len(sp.constraint_robot_model.sphere_link_names)), ["left"])
    sp.refiner.last_result = types.SimpleNamespace(metrics={"held_support_contact": {
        "held_rows_exempt": 4, "finger_rows_allowed": 9, "n_rows": 120}})
    rec = sp._to_block()["held_support_contact"]
    assert {"applied", "latch", "held_rows_exempt", "finger_rows_allowed", "n_rows"} <= set(rec)
    assert rec["applied"] is True and rec["latch"] == "held"
    assert (rec["held_rows_exempt"], rec["finger_rows_allowed"], rec["n_rows"]) == (4, 9, 120)
    # a request on which the gate did not run (or did not apply): zeros, not the last request's numbers
    sp._last_held_support_contact = {}
    rec = sp._to_block()["held_support_contact"]
    assert rec["applied"] is False and rec["held_rows_exempt"] == rec["finger_rows_allowed"] == rec["n_rows"] == 0


def test_safe_policy_default_is_off():
    assert _policy().held_support_contact_m is None and _policy()._last_held_support_contact == {}


# ------------------------------------------------------------------------------------------------ 4. the flags
B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]


def test_flags_default_off_and_apply_only_when_on():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    off = ap.parse_args(B)
    assert off.held_support_contact is False and off.held_support_contact_finger_mm is None
    safe = _policy()
    out_off = ss.apply_grasp_contact_flags(safe, off)
    assert safe.held_support_contact_m is None and "held_support_contact" not in out_off
    assert set(out_off) == {"latch_phase", "held_hand_exempt", "grasp_zone_allowance_m", "grasp_zone_radius_m"}   # the record as before
    on = ap.parse_args(B + ["--held-support-contact"])
    safe = _policy()
    out = ss.apply_grasp_contact_flags(safe, on)
    assert safe.held_support_contact_m == pytest.approx(0.010) and out["held_support_contact"] == {"finger_margin_m": 0.010}
    four = ap.parse_args(B + ["--held-support-contact", "--held-support-contact-finger-mm", "4"])
    safe = _policy()
    ss.apply_grasp_contact_flags(safe, four)
    assert safe.held_support_contact_m == pytest.approx(0.004)


def test_finger_mm_alone_or_negative_is_refused():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    for extra in (["--held-support-contact-finger-mm", "5"], ["--held-support-contact", "--held-support-contact-finger-mm", "-1"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(B + extra))
    ss.reject_bad_flag_combinations(ap, ap.parse_args(B + ["--held-support-contact", "--held-support-contact-finger-mm", "0"]))


@pytest.mark.parametrize("extra", [["--held-support-contact"], ["--held-support-contact-finger-mm", "10"],
                                   ["--held-support-contact", "--held-support-contact-finger-mm", "10"]])
def test_the_rby1_server_refuses_it_and_the_panda_server_accepts(extra):
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, ap.parse_args(B + extra))                       # default --robot rby1
    ss.reject_panda_only_flags(ap, ap.parse_args(B + extra + ["--robot", "panda"]))


def test_the_rby1_server_without_the_flag_is_not_refused():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    ss.reject_panda_only_flags(ap, ap.parse_args(B))
