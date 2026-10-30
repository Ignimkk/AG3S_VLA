"""SL6-H — HSR6 ⑤ placement contact (`--hsr6-placement-contact`, task `AG3S/ag3s/docs/handoff/SL6.task.md` §11 H1 / H4).

What this file fixes (CPU, analytic fields):

1. the rule (`hsr6.placement_contact_ok`): placement ∧ (HELD ∨ (CLOSING ∧ closed ∧ blocked)); the flag is the fifth
   `Hsr6Flags` field and leaves the four existing names / order alone.
2. the optimizer rows (`CollisionLinearizer._placement_contact_margin`) on a synthetic scene: a held sphere lowered into
   a plate (and one on the table) is released; the same kind of held sphere beside a wall, under a shelf, or above a
   low step outside its footprint keeps its clearance; hand / finger rows never change; `placement_contact=None` is
   bit-identical to a snapshot without the key.
3. end to end through the SQP (`TrajOptChunkRefiner`, RB-Y1 fixture): the plate under the held sphere moves the plan
   with the rule off; with it on the plan is the plan of a scene without the plate; a wall beside the held sphere still
   moves the plan with the rule on. `to.metrics.placement_contact` is the returned trajectory's final check.
4. SafePolicy (`_hsr6_placement_contact` / `_hsr6_placement_contact_result`): the snapshot gets the key only when the
   rule holds and something is attached; every call is recorded in `hsr6.detail.placement_contact`.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot

VOXEL = 0.005
R_HELD = 0.044          # the Panda bowl's held sphere (diagnosis README §2: 44 mm + margin)
R_FINGER = 0.005


# ------------------------------------------------------------------------------------------------ 1. the rule
@pytest.mark.parametrize("phase,latch,checks,want", [
    ("placement", "held", {}, True),
    ("placement", "held", {"closed": False, "blocked": False}, True),        # HELD needs no evidence
    ("placement", "closing", {"closed": True, "blocked": True}, True),
    ("placement", "closing", {"closed": True, "blocked": False}, False),
    ("placement", "closing", {"closed": False, "blocked": True}, False),
    ("placement", "closing", {}, False),
    ("placement", "latched", {"closed": True, "blocked": True}, False),
    ("placement", "searching", {}, False),
    ("placement", "placed", {}, False),
    ("transport", "held", {}, False),
    ("lift", "held", {}, False),
    ("grasp", "held", {}, False),
    ("retreat", "held", {}, False),
    (None, "held", {}, False)])
def test_placement_contact_rule(phase, latch, checks, want):
    assert H6.placement_contact_ok(phase, latch=latch, checks=checks) is want


def test_flag_is_a_fifth_hsr6_use_and_off_by_default():
    assert H6.Hsr6Flags().placement_contact is False and not H6.Hsr6Flags().on
    f = H6.Hsr6Flags(placement_contact=True)
    assert f.on and f.names() == ["placement_contact"]
    four = H6.Hsr6Flags(attach_assist=True, held_gate=True, target_exempt=True, objective=True)
    assert four.names() == ["attach_assist", "held_gate", "target_exempt", "objective"]       # unchanged
    assert dataclasses.replace(four, placement_contact=True).names()[-1] == "placement_contact"
    spec = H6.placement_contact_spec()
    assert spec == {"cos_min": pytest.approx(np.sqrt(0.5)), "footprint_tol_m": None}


def test_serve_safe_flag_parses_into_hsr6_flags():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    assert ap.get_default("hsr6_placement_contact") is False
    off = types.SimpleNamespace(hsr6_attach_assist=True, hsr6_held_gate=False, hsr6_target_exempt=False,
                                hsr6_objective=False)                       # an argv from before SL6-H
    assert ss.hsr6_flags_from_args(off).placement_contact is False
    on = types.SimpleNamespace(**vars(off), hsr6_placement_contact=True)
    assert ss.hsr6_flags_from_args(on).placement_contact is True
    only = types.SimpleNamespace(hsr6_placement_contact=True)
    assert ss.hsr6_flags_from_args(only).names() == ["placement_contact"]
    assert ss.hsr6_record_meta(only)["hsr6"]["flags"] == ["placement_contact"]


# ------------------------------------------------------------------------------------------------ 2. optimizer rows
def _box_sdf(p, lo, hi):
    c, h = (np.asarray(lo) + np.asarray(hi)) / 2, (np.asarray(hi) - np.asarray(lo)) / 2
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


class _Scene3:
    """Table z = 0 + a plate (|x|, |y| ≤ 0.08, z ∈ [0, 0.015]) + a wall (x ∈ [0.20, 0.24], z ≤ 0.30) + a shelf above
    (x ∈ [−0.45, −0.25], z ∈ [0.30, 0.32]) + a low step (x ∈ [0.70, 0.90], z ≤ 0.10). Analytic distance, central-
    difference gradient (no label layer)."""

    has_labels = False
    BOXES = {"plate": ((-0.08, -0.08, 0.0), (0.08, 0.08, 0.015)),
             "wall": ((0.20, -0.10, 0.0), (0.24, 0.10, 0.30)),
             "shelf": ((-0.45, -0.10, 0.30), (-0.25, 0.10, 0.32)),
             "step": ((0.70, -0.05, 0.0), (0.90, 0.05, 0.10))}

    def __init__(self, boxes=None):
        self.boxes = dict(self.BOXES if boxes is None else boxes)
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)

    def distance(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return np.minimum.reduce([p[:, 2]] + [_box_sdf(p, lo, hi) for lo, hi in self.boxes.values()])

    def gradient(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        e = 1e-6
        return np.stack([(self.distance(p + e * np.eye(3)[k]) - self.distance(p - e * np.eye(3)[k])) / (2 * e)
                         for k in range(3)], axis=1)


#: query rows (one step): 0–1 robot (finger) spheres, 2–7 the held object's spheres
ROWS = {
    "finger_on_plate": 0,      # finger 3 mm above the plate (r 5 mm): violates, obstacle class
    "finger_by_wall": 1,       # finger 4 mm from the wall
    "held_on_plate": 2,        # held sphere 10 mm into the plate: the case SL6-H releases
    "held_by_wall": 3,         # held sphere 5 mm into the wall from the side (plate / table farther)
    "held_under_shelf": 4,     # held sphere 5 mm into a shelf above it
    "held_on_table": 5,        # held sphere 5 mm into the table (support beneath)
    "held_step_off_footprint": 6,  # nearest surface = the step's top edge, below within 45°, outside the footprint
    "held_free": 7,            # nothing near
}
CENTRES = np.array([
    [0.03, 0.0, 0.015 + 0.003],
    [0.20 - 0.004, 0.0, 0.15],
    [0.0, 0.0, 0.015 + R_HELD - 0.010],
    [0.20 - R_HELD + 0.005, 0.0, 0.15],
    [-0.35, 0.0, 0.30 - R_HELD + 0.005],
    [0.50, 0.30, R_HELD - 0.005],
    [0.70 - 0.06, 0.0, 0.10 + 0.07],
    [-0.60, 0.40, 0.40],
])
N_ROBOT = 2


def _lin():
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.full(N_ROBOT, R_FINGER)
    lin.query_radii = np.concatenate([lin.robot_radii, np.full(CENTRES.shape[0] - N_ROBOT, R_HELD)])
    lin.n_spheres = N_ROBOT
    lin._attached_local = None
    return lin


def _snapshot(field=None, **extra):
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
        d_safe=np.zeros((N_ROBOT, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
        plane_active=np.zeros(0, bool), robot_radii=np.full(N_ROBOT, R_FINGER), candidate_ids=np.zeros(0, np.int64),
        esdf=_Scene3() if field is None else field, esdf_margin=0.0,
        # the Panda final argv: obstacle margin 10 mm on robot obstacle rows, table plane known
        obstacle_margin=0.010, support_planes=np.array([[0.0, 0.0, 1.0, 0.0]]), **extra)


def _clear(lin, scene, centres=CENTRES):
    return lin._esdf_clearance(np.asarray(centres, np.float64)[None], scene)[0, :, 0]


def test_off_is_bit_identical():
    lin = _lin()
    a = _clear(lin, _snapshot())
    b = _clear(lin, _snapshot(placement_contact=None))
    assert np.array_equal(a, b) and SceneSnapshot.__dataclass_fields__["placement_contact"].default is None
    assert "_last_placement_contact" not in lin.__dict__


def test_the_rig_is_what_it_says():
    off = _clear(_lin(), _snapshot())
    assert off[ROWS["held_on_plate"]] == pytest.approx(-0.010, abs=1e-9)
    assert off[ROWS["held_by_wall"]] == pytest.approx(-0.005, abs=1e-9)
    assert off[ROWS["held_under_shelf"]] == pytest.approx(-0.005, abs=1e-9)
    assert off[ROWS["held_on_table"]] == pytest.approx(-0.005, abs=1e-9)
    assert off[ROWS["finger_on_plate"]] < 0 and off[ROWS["finger_by_wall"]] < 0      # 10 mm obstacle margin
    g = _Scene3().gradient(CENTRES[[ROWS["held_step_off_footprint"]]])[0]
    assert g[2] >= H6.PLACEMENT_BENEATH_COS * np.linalg.norm(g)                        # below, within the cone
    assert _Scene3().distance(CENTRES[[ROWS["held_step_off_footprint"]]])[0] < 0.17    # the step is nearest


def test_on_releases_only_held_rows_on_the_surface_beneath():
    lin = _lin()
    off = _clear(lin, _snapshot())
    on = _clear(lin, _snapshot(placement_contact=H6.placement_contact_spec()))
    for name in ("held_on_plate", "held_on_table"):
        assert on[ROWS[name]] == pytest.approx(off[ROWS[name]] + 1.0), name                  # cannot bind
    for name in ("finger_on_plate", "finger_by_wall", "held_by_wall", "held_under_shelf",
                 "held_step_off_footprint", "held_free"):
        assert on[ROWS[name]] == off[ROWS[name]], name
    rec = lin._last_placement_contact
    assert rec["rows_released"] == 2 and rec["held_rows_released"] == 2 and rec["n_held_rows"] == 6
    assert rec["min_clearance_released_m"] == pytest.approx(-0.010, abs=1e-9)
    assert rec["footprint_tol_m"] == VOXEL                                                   # default: one voxel


def test_hand_rows_never_change_even_on_the_plate():
    """A finger pressed into the plate stays blocked: the rule reads held rows only."""
    lin = _lin()
    centres = CENTRES.copy()
    centres[ROWS["finger_on_plate"], 2] = 0.015 - 0.002                                  # 2 mm inside the plate
    off = _clear(lin, _snapshot(), centres)
    on = _clear(lin, _snapshot(placement_contact=H6.placement_contact_spec()), centres)
    assert np.array_equal(off[:N_ROBOT], on[:N_ROBOT]) and on[ROWS["finger_on_plate"]] < -0.015


def test_held_sphere_beside_a_wall_is_constrained_while_lowered_onto_the_plate():
    """The same held sphere: lowered into the plate it is released; moved beside the wall (above the plate, the wall
    nearer) it is not — at every height on the way down."""
    lin = _lin()
    on_scene = _snapshot(placement_contact=H6.placement_contact_spec())
    for z in (0.20, 0.12, 0.08, 0.06, 0.05):
        c = CENTRES.copy()
        c[ROWS["held_by_wall"]] = [0.20 - R_HELD + 0.005, 0.0, z]
        off = _clear(lin, _snapshot(), c)
        on = _clear(lin, on_scene, c)
        assert on[ROWS["held_by_wall"]] == off[ROWS["held_by_wall"]] < 0, z


def test_wider_footprint_from_other_held_spheres_counts():
    """The footprint is the union of the held spheres' xy discs at the step: a second held sphere over the step's edge
    puts that edge under the held object."""
    lin = _lin()
    c = CENTRES.copy()
    c[ROWS["held_free"]] = [0.70, 0.0, 0.10 + R_HELD + 0.03]                          # over the step's edge
    off = _clear(lin, _snapshot(), c)
    on = _clear(lin, _snapshot(placement_contact=H6.placement_contact_spec()), c)
    assert on[ROWS["held_step_off_footprint"]] == pytest.approx(off[ROWS["held_step_off_footprint"]] + 1.0)


def test_no_held_rows_changes_nothing():
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = lin.query_radii = np.full(N_ROBOT, R_FINGER)
    lin.n_spheres = N_ROBOT
    lin._attached_local = None
    off = _clear(lin, _snapshot(), CENTRES[:N_ROBOT])
    on = _clear(lin, _snapshot(placement_contact=H6.placement_contact_spec()), CENTRES[:N_ROBOT])
    assert np.array_equal(off, on)
    assert lin._last_placement_contact == {"n_held_rows": 0, "steps": 1, "rows_released": 0, "held_rows_released": 0,
                                           "reason": "no held rows"}


def test_flat_field_is_never_called_beneath():
    class _Flat(_Scene3):
        def gradient(self, pts):
            return np.zeros((len(np.asarray(pts).reshape(-1, 3)), 3))

    lin = _lin()
    off = _clear(lin, _snapshot(_Flat()))
    on = _clear(lin, _snapshot(_Flat(), placement_contact=H6.placement_contact_spec()))
    assert np.array_equal(off, on) and lin._last_placement_contact["rows_released"] == 0


# ------------------------------------------------------------------------------------------------ 3. through the SQP
class _Boxes:
    has_labels = False

    def __init__(self, boxes):
        self.boxes = boxes
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)

    def distance(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return np.minimum.reduce([_box_sdf(p, lo, hi) for lo, hi in self.boxes])

    def gradient(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        e = 1e-6
        return np.stack([(self.distance(p + e * np.eye(3)[k]) - self.distance(p - e * np.eye(3)[k])) / (2 * e)
                         for k in range(3)], axis=1)


@pytest.fixture(scope="module")
def held_rig():
    """RB-Y1 fixture arm sweeping; one held sphere (40 mm) rides the link that moves most, 200 mm below its sphere
    (far enough below the hand that a post beside it touches no robot sphere)."""
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8)
    lin = sc.linearizer
    centres, _ = lin.sphere_states(sc.reference, sc.q_now)
    idx = int(np.argmax(np.linalg.norm(centres[-1] - centres[0], axis=1)))
    link = list(lin.robot_model.sphere_link_names)[idx]
    full = sc.layout.full_q(sc.reference, sc.q_now)
    T = _link_pose_numeric(sc.robot, np.asarray(full[:, 0]).reshape(-1), link)
    local = (T[:3, :3].T @ (centres[0, idx] + np.array([0.0, 0.0, -0.20]) - T[:3, 3]))[None]
    lin.set_attached(local, link)
    held = lin.sphere_states(sc.reference, sc.q_now)[0][:, lin.n_spheres]
    lin.set_attached(None, None)
    return types.SimpleNamespace(sc=sc, link=link, local=local, held=held, robot_centres=centres, r=0.040)


def _refine(rig, scene):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sc = rig.sc
    ref = TrajOptChunkRefiner(sc.robot, sc.layout, lambda _c: (scene, sc.q_now, True), sc.config)
    out = np.asarray(ref.refine(sc.chunk, {"t_step": 0, "previous_physical_chunk": None}), np.float64)
    ref_violation = -ref.optimizer.linearizer.full_violation(sc.reference, sc.q_now, scene)
    return out, ref.last_result, ref_violation


def _held_scene(rig, boxes, **extra):
    sc = rig.sc
    return dataclasses.replace(
        sc.scene, candidate_active=np.zeros_like(sc.scene.candidate_active),
        plane_active=np.zeros_like(sc.scene.plane_active), esdf=_Boxes(boxes), esdf_margin=0.0,
        attached_points=rig.local, attached_parent_link=rig.link, attached_radii=np.array([rig.r]), **extra)


def test_sqp_plate_beneath_moves_the_plan_only_without_the_rule(held_rig):
    rig = held_rig
    held = rig.held
    zmin = float(held[:, 2].min())
    plate = (tuple(np.r_[held[:, :2].min(0) - 0.05, zmin - rig.r - 0.03]),
             tuple(np.r_[held[:, :2].max(0) + 0.05, zmin - rig.r + 0.010]))      # 10 mm into the held sphere
    far = ((9.0, 9.0, 9.0), (9.1, 9.1, 9.1))
    free_out, _, _ = _refine(rig, _held_scene(rig, [far]))
    off_out, off_res, off_ref = _refine(rig, _held_scene(rig, [plate]))
    on_out, on_res, on_ref = _refine(rig, _held_scene(rig, [plate], placement_contact=H6.placement_contact_spec()))
    assert off_ref == pytest.approx(0.010, abs=1e-6)                 # the plate blocks the reference descent
    assert np.abs(off_out - free_out).max() > 1e-3                   # off: TrajOpt moves the arm off the plate
    assert on_ref < 0.0                                              # on: the robot rows still clear the plate
    assert np.array_equal(on_out, free_out)                          # on: the plan of the scene without the plate
    assert "placement_contact" not in off_res.metrics
    rec = on_res.metrics["placement_contact"]
    assert rec["rows_released"] > 0 and rec["held_rows_released"] == 1 and on_res.max_violation == 0.0


def test_sqp_wall_beside_the_held_sphere_still_moves_the_plan_with_the_rule(held_rig):
    rig = held_rig
    h = rig.held[rig.held.shape[0] // 2]                         # a post beside the held sphere at mid-chunk
    wall = (tuple(h + np.array([-0.02, rig.r - 0.010, -0.015])), tuple(h + np.array([0.02, rig.r + 0.04, 0.015])))
    robot = rig.robot_centres
    lo, hi = np.asarray(wall[0]), np.asarray(wall[1])
    assert (_box_sdf(robot.reshape(-1, 3), lo, hi).reshape(robot.shape[:2])
            - rig.sc.linearizer.robot_radii[None]).min() > 0.05     # no robot sphere near the post
    far = ((9.0, 9.0, 9.0), (9.1, 9.1, 9.1))
    free_out, _, _ = _refine(rig, _held_scene(rig, [far]))
    on_out, on_res, on_ref = _refine(rig, _held_scene(rig, [wall], placement_contact=H6.placement_contact_spec()))
    assert on_ref == pytest.approx(0.010, abs=1e-6)                  # the wall still binds the held row
    assert np.abs(on_out - free_out).max() > 1e-3
    assert on_res.metrics["placement_contact"]["rows_released"] == 0


# ------------------------------------------------------------------------------------------------ 4. SafePolicy
def _rec(phase):
    p = {c: (0.9 if c == phase else 0.02) for c in H6.PHASES6}
    return {"p": p, "argmax": phase, "probe": "test"}


def _with_latch(sp, phase, checks, hsr6_phase):
    from AG3S.trajopt.grasp_latch import LatchEvent

    sp._pending = {}
    sp._hsr6_start(_rec(hsr6_phase))
    sp._latch.phase = phase
    sp._latch._last_grasp = LatchEvent(phase=phase, evidence={"checks": dict(checks)})


def _policy(**kw):
    from tests.trajopt.test_grasp_evidence import _policy as build

    return build(**kw)


HELD_PTS = np.zeros((3, 3))
CB = {"closed": True, "blocked": True, "not_empty": True}


@pytest.mark.parametrize("hsr6_phase,latch,checks,attached,applied", [
    ("placement", GraspPhase.HELD, {}, True, True),
    ("placement", GraspPhase.CLOSING, CB, True, True),
    ("placement", GraspPhase.CLOSING, {**CB, "blocked": False}, True, False),
    ("placement", GraspPhase.LATCHED, CB, True, False),
    ("transport", GraspPhase.HELD, {}, True, False),
    ("placement", GraspPhase.CLOSING, CB, False, False),     # rule on, nothing attached: nothing to release
    ("placement", GraspPhase.HELD, {}, False, False)])
def test_safe_policy_gate(hsr6_phase, latch, checks, attached, applied):
    sp = _policy(hsr6=H6.Hsr6Flags(placement_contact=True))
    _with_latch(sp, latch, checks, hsr6_phase)
    snap = _snapshot(attached_points=HELD_PTS if attached else None, attached_parent_link="l" if attached else None)
    out = sp._hsr6_placement_contact(snap)
    assert (out.placement_contact is not None) is applied
    if not applied:
        assert out is snap
    rec = sp._last_hsr6["detail"]["placement_contact"]
    on = H6.placement_contact_ok(hsr6_phase, latch=latch.value, checks=checks)
    assert rec["on"] is on and rec["applied"] is applied and rec["attached"] is attached
    assert rec["latch"] == latch.value and rec["phase"] == hsr6_phase
    assert rec["n_held_rows"] == (3 if attached else 0)
    if on and not attached:
        assert rec["reason"].startswith("no attached held object")


def test_safe_policy_result_reads_the_final_check():
    sp = _policy(hsr6=H6.Hsr6Flags(placement_contact=True))
    _with_latch(sp, GraspPhase.HELD, {}, "placement")
    sp._hsr6_placement_contact(_snapshot(attached_points=HELD_PTS, attached_parent_link="l"))
    sp.refiner.last_result = types.SimpleNamespace(metrics={"placement_contact": {
        "rows_released": 7, "held_rows_released": 2, "min_clearance_released_m": -0.012}})
    sp._hsr6_placement_contact_result()
    rec = sp._last_hsr6["detail"]["placement_contact"]
    assert rec["rows_released"] == 7 and rec["held_rows_released"] == 2
    assert rec["min_clearance_released_m"] == pytest.approx(-0.012)
    assert sp._last_hsr6["changed"]["placement_contact"] is True
    # applied but nothing released (or no SQP record): not a change
    _with_latch(sp, GraspPhase.HELD, {}, "placement")
    sp._hsr6_placement_contact(_snapshot(attached_points=HELD_PTS, attached_parent_link="l"))
    sp.refiner.last_result = types.SimpleNamespace(metrics={})
    sp._hsr6_placement_contact_result()
    assert sp._last_hsr6["changed"]["placement_contact"] is False
    assert sp._last_hsr6["detail"]["placement_contact"]["final_check"] is None


def test_record_resets_per_request_and_is_absent_when_off():
    sp = _policy(hsr6=H6.Hsr6Flags(placement_contact=True))
    _with_latch(sp, GraspPhase.HELD, {}, "placement")
    sp._hsr6_placement_contact(_snapshot(attached_points=HELD_PTS, attached_parent_link="l"))
    assert sp._last_placement_contact["applied"] is True
    sp._pending = {}
    sp._hsr6_start(_rec("transport"))                          # next request: the record starts empty
    assert sp._last_placement_contact == {} and "placement_contact" not in sp._last_hsr6["detail"]
    other = _policy(hsr6=H6.Hsr6Flags(held_gate=True))         # HSR6 on, ⑤ off
    assert other.hsr6.placement_contact is False and other._last_placement_contact == {}
    assert _policy()._last_placement_contact == {} and _policy().hsr6 is None
