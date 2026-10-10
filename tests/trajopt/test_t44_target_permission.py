"""T44 I2 — the target exception follows the contact permission (`collision.target_permission_strict`).

The three target tests outside the waypoint margin — the look-ahead tail drop (`_lookahead_keep`),
the obstacle-margin class (`_obstacle_rows`) and the sweep path exclusion (`PathChecker._target_rows`)
— had no link filter: a forearm next to the apple was exempted like a finger, and under
`exclude_authorized` the sweep's `&= ~free` even exempted *only* the unauthorized rows (P1, P2).
Strict: only `SceneSnapshot.target_contact_rows` get the exception. Off (default): bit-identical.

Also pinned here:
* P3 — `obstacle_margin > 0` no longer drops the `target_volume` loaded just before it (flag-free),
  and the V11AB setting (`obstacle_margin` 0.010, `target_volume_exempt` off) is unchanged.
* AG3S `ball_owned` (`stats["target_free"]["exclusion_method"]`): with strict, target membership is
  `d_free − d_main > voxel`, so a neighbouring fruit kept inside the ball is not the target.

Rig: z = 0 table, apple (target), a distractor fruit 7 mm from it, a bar. Query rows (one step):

| row | link | where | permission |
|---|---|---|---|
| 0 | `ee_finger_r1` | 1 mm inside the apple | authorized |
| 1 | `link_right_arm_5` | the same point | **not** authorized |
| 2 | `ee_right` | 8 mm below the bar | authorized |
| 3 | `ee_finger_r2` | 4 mm above the distractor (inside the ball, not E1's apple) | authorized |
| 4 | held | far away | — |
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import types

import numpy as np
import pytest

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import (
    OBSTACLE_CLASSES,
    CollisionLinearizer,
    SceneSnapshot,
    exclusion_method_of,
    scene_from_constraint_set,
    support_planes_of,
    target_contact_rows_of,
)
from benchmark.trajopt.sweep import PathChecker, PathConfigs

VOXEL = 0.005
M = 0.010
APPROACH = 0.02
FULL = 0.05
R_Q = 0.0033
APPLE = np.array([0.0, 0.0, 0.033])
APPLE_R = 0.033
DIST = np.array([0.065, 0.0, 0.033])
DIST_R = 0.025
BAR = np.array([0.3, 0.0, 0.28])
BAR_R = 0.012
NAMES = ["ee_finger_r1", "link_right_arm_5", "ee_right", "ee_finger_r2"]
CONTACT = np.array([True, False, True, True])
FINGER, FOREARM, BARROW, DISTROW, HELD = 0, 1, 2, 3, 4
TARGET = OBSTACLE_CLASSES.index("target")
OBSTACLE = OBSTACLE_CLASSES.index("obstacle")
BALL = np.array([*APPLE, 0.072, -np.inf])     # holds the distractor's top (69.6 mm from the centre)


def _num_grad(f, p, h=1e-5):
    p = np.asarray(p, np.float64).reshape(-1, 3)
    return np.stack([(f(p + h * e) - f(p - h * e)) / (2 * h) for e in np.eye(3)], axis=1)


class _Field:
    """Analytic field. `owned=True`: the target-free layer lost the apple only (AG3S `ball_owned`);
    `owned=False`: it lost the apple and the distractor (the ball)."""

    has_labels = False

    def __init__(self, owned: bool = True):
        self.owned = owned
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)
        self.stats = {}

    @staticmethod
    def _parts(p):
        return {"table": p[:, 2],
                "apple": np.linalg.norm(p - APPLE, axis=1) - APPLE_R,
                "dist": np.linalg.norm(p - DIST, axis=1) - DIST_R,
                "bar": np.linalg.norm(p - BAR, axis=1) - BAR_R}

    def distance(self, pts):
        return np.minimum.reduce(list(self._parts(np.asarray(pts, np.float64).reshape(-1, 3)).values()))

    def target_free_distance(self, pts):
        parts = self._parts(np.asarray(pts, np.float64).reshape(-1, 3))
        keep = ["table", "bar"] + (["dist"] if self.owned else [])
        return np.minimum.reduce([parts[k] for k in keep])

    def gradient(self, pts):
        return _num_grad(self.distance, pts)

    def target_free_gradient(self, pts, return_tier=False):
        g = _num_grad(self.target_free_distance, pts)
        return (g, np.array(["fine"] * len(g))) if return_tier else g


def _apple_points():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4000, 3))
    pts = APPLE + APPLE_R * v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.vstack([pts, APPLE + np.array([0.0, -APPLE_R, 0.0])])   # rows 0/1's nearest point


def _centres(steps=1):
    one = np.stack([
        APPLE + np.array([0.0, -(APPLE_R - 0.001), 0.0]),
        APPLE + np.array([0.0, -(APPLE_R - 0.001), 0.0]),
        BAR + np.array([0.0, 0.0, -(BAR_R + 0.008)]),
        DIST + np.array([0.0, 0.0, DIST_R + 0.004]),
        np.array([0.3, 0.3, 0.3]),
    ])
    return np.repeat(one[None], steps, axis=0)


def _lin():
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.full(4, R_Q)
    lin.query_radii = np.array([R_Q] * 4 + [0.03])
    lin.n_spheres = 4
    lin._attached_local = None
    lin._attached_link = "ee_right"
    lin.robot_model = types.SimpleNamespace(sphere_link_names=NAMES)
    return lin


def _scene(field=None, *, mask=None, strict=False, contact=CONTACT, **extra):
    field = _Field() if field is None else field
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((4, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.full(4, R_Q), candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=0.0, manipulated_points=_apple_points(),
        manipulated_link_margin=np.where(CONTACT, APPROACH, FULL),
        target_free_mask=None if mask is None else np.asarray(mask, bool),
        target_contact_rows=contact, target_permission_strict=strict, **extra)


def _checker(lin, centres):
    pc = object.__new__(PathChecker)
    pc.lin = lin
    pc._floor = np.zeros(centres.shape[1])
    pc.timing = collections.defaultdict(float)
    pc.mode, pc.sweep, pc.max_sweep, pc._state, pc.horizon = "interp", False, 0.0, None, 1
    pc._centres_now = centres[0]
    cfgs = PathConfigs(full=np.zeros((1, 1)), params=None, segment=np.zeros(1, int),
                       where=np.array([0.5]), W=np.zeros((1, 1, 1)), pieces=np.ones(1, int),
                       path_m=np.zeros(1), cap_bound=(), ends_centres=np.stack([centres[0]] * 2))
    pc.configs = lambda trajectory: cfgs
    pc._states = lambda full, params, with_jac: (centres, None)
    return pc


# =============================================================================================
# 1. defaults · permission vector · wiring
# =============================================================================================


def test_defaults_are_off():
    assert TrajOptConfig().collision.target_permission_strict is False
    f = SceneSnapshot.__dataclass_fields__
    assert f["target_contact_rows"].default is None
    assert f["target_permission_strict"].default is False
    assert f["target_membership"].default == "ball"
    assert CollisionLinearizer.target_contact(_scene(), 5) is None
    on = CollisionLinearizer.target_contact(_scene(strict=True), 5)
    assert on.tolist() == [True, False, True, True, False]                 # held padded False
    assert CollisionLinearizer.target_contact(_scene(strict=True, contact=None), 5).tolist() == [False] * 5


@pytest.mark.parametrize("case,expected", [
    ("exclude_authorized", [True, False, True, True]),
    ("exclude_all", [True, False, True, True]),
    ("exclude_all_no_margin", [False] * 4),
    ("relax", [True, False, True, True]),
    ("relax_transit", [False] * 4),
    ("nothing", None),
])
def test_contact_rows_come_from_the_permission_already_held(case, expected):
    margin = np.where(CONTACT, APPROACH, FULL)
    kw = dict(manipulated=True)
    if case == "exclude_authorized":
        got = target_contact_rows_of(CONTACT, None, margin, "exclude_authorized", **kw)
    elif case == "exclude_all":
        got = target_contact_rows_of(np.ones(4, bool), np.ones(4, bool) & CONTACT, margin,
                                     "exclude_all", **kw)
    elif case == "exclude_all_no_margin":
        got = target_contact_rows_of(np.ones(4, bool), None, None, "exclude_all", **kw)
    elif case == "relax":
        got = target_contact_rows_of(None, None, margin, "relax", **kw)
    elif case == "relax_transit":
        got = target_contact_rows_of(None, None, np.full(4, FULL), "relax", **kw)
    else:
        got = target_contact_rows_of(None, None, None, "relax", manipulated=False)
    assert (got is None) if expected is None else got.tolist() == expected


def _cs(field, *, policy="relax", mask=None, attached=None):
    from benchmark.ag3s.types import SupportSurface

    table = SupportSurface(id=1, normal=np.array([0.0, 0.0, 1.0]), offset=0.0,
                           point_indices=np.arange(3), point_count=3, rms_error=0.001,
                           safety_margin=0.01)
    return types.SimpleNamespace(
        constraints=None, esdf=field, support_surfaces=(table,), attached=attached,
        manipulated=types.SimpleNamespace(points=_apple_points(), sphere_centers=None,
                                          sphere_radii=None),
        manipulated_link_margin=np.where(CONTACT, APPROACH, FULL), target_field_exclude=mask,
        target_field_policy=policy, destination_label=None, destination_margin=None)


def _config(**collision):
    return TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": 0.0,
                                                  "use_support_planes": False, **collision}})


_BALL_STATS = {"centre_m": APPLE.tolist(), "radius_mm": 72.0, "z_min_m": None}


def test_scene_from_constraint_set_carries_the_permission_and_the_switches():
    field = _Field()
    field.stats = {"target_free": {"ball": _BALL_STATS, "exclusion_method": "ball_owned"}}
    radii = np.full(4, R_Q)
    off = scene_from_constraint_set(_cs(field), radii, _config())
    assert off.target_contact_rows.tolist() == CONTACT.tolist()               # relax: margin rows
    assert off.target_permission_strict is False and off.target_membership == "ball"
    auth = scene_from_constraint_set(_cs(field, policy="exclude_authorized", mask=CONTACT), radii,
                                     _config(target_permission_strict=True))
    assert auth.target_contact_rows.tolist() == CONTACT.tolist()
    assert auth.target_permission_strict is True and auth.target_membership == "layer_diff"
    field.stats = {"target_free": {"ball": _BALL_STATS}}                        # pre-T44 field
    plain = scene_from_constraint_set(_cs(field), radii, _config(target_permission_strict=True))
    assert plain.target_membership == "ball"
    assert exclusion_method_of(field) is None and exclusion_method_of(None) is None
    field.stats = {"target_free": {"ball": _BALL_STATS, "exclusion_method": "ball_owned"}}
    holding = _cs(field, attached=types.SimpleNamespace(points=np.zeros((1, 3)),
                                                        parent_link="ee_right", primitives=[]))
    held = scene_from_constraint_set(holding, radii, _config(target_permission_strict=True))
    assert held.target_membership == "ball"                                   # nothing held only
    nothing = _cs(field)
    nothing.manipulated, nothing.manipulated_link_margin = None, None
    assert scene_from_constraint_set(nothing, radii, _config()).target_contact_rows is None


def test_config_round_trip_and_validation():
    cfg = _config(target_permission_strict=True)
    assert TrajOptConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.collision.target_permission_strict is True
    with pytest.raises(ValueError, match="target_permission_strict"):
        _config(target_permission_strict="yes")


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_flag():
    from benchmark.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert serve_safe.permission_overrides(args) == {}
    assert serve_safe.trajopt_config_from_args(args).collision.target_permission_strict is False
    args = serve_safe.build_parser().parse_args(BASE_ARGV + ["--target-permission-strict"])
    assert serve_safe.permission_overrides(args) == {"target_permission_strict": True}
    assert serve_safe.trajopt_config_from_args(args).collision.target_permission_strict is True


# =============================================================================================
# 2. off = bit-identical (the permission vector is loaded but nobody reads it)
# =============================================================================================


@pytest.mark.parametrize("mask", [None, CONTACT], ids=["relax", "exclude_authorized"])
@pytest.mark.parametrize("vol", [None, BALL], ids=["no_volume", "volume"])
def test_off_reads_nothing_new(mask, vol):
    centres = _centres(3)
    old = _scene(mask=mask, contact=None, target_volume=vol, obstacle_margin=M)
    new = _scene(mask=mask, target_volume=vol, obstacle_margin=M)
    for lookahead in (None, 1):
        a, b = _lin(), _lin()
        a.lookahead_from = b.lookahead_from = lookahead
        assert np.array_equal(a._esdf_clearance(centres, old), b._esdf_clearance(centres, new))
        assert np.array_equal(a._last_obstacle_class, b._last_obstacle_class)
        assert getattr(a, "_last_lookahead", None) == getattr(b, "_last_lookahead", None)
    ra, rb = _checker(_lin(), centres), _checker(_lin(), centres)
    ta, tb = ra._target_rows(centres, old), rb._target_rows(centres, new)
    assert (ta is None and tb is None) or np.array_equal(ta, tb)
    va, rec_a = ra.violation(np.zeros((1, 1)), old, record=True)
    vb, rec_b = rb.violation(np.zeros((1, 1)), new, record=True)
    assert va == vb and rec_a == rec_b and "target_rows_unauthorized_kept" not in rec_b


# =============================================================================================
# 3. strict — finger keeps the exception, forearm at the same point does not
# =============================================================================================


def test_waypoint_class_forearm_is_an_obstacle_finger_stays_target():
    lin, centres = _lin(), _centres()
    off = lin._esdf_clearance(centres, _scene(obstacle_margin=M))[0, :, 0]
    cls_off = lin._last_obstacle_class[0].copy()
    on = lin._esdf_clearance(centres, _scene(obstacle_margin=M, strict=True))[0, :, 0]
    cls_on = lin._last_obstacle_class[0]
    assert cls_off[FINGER] == TARGET and cls_off[FOREARM] == TARGET
    assert cls_on[FINGER] == TARGET and cls_on[FOREARM] == OBSTACLE
    # the forearm already had the full E1 margin (> M) — its waypoint value does not move
    assert np.array_equal(on, off)


@pytest.mark.parametrize("mask", [None, CONTACT], ids=["relax", "exclude_authorized"])
def test_lookahead_tail_keeps_the_forearm_row(mask):
    centres = _centres(2)
    lin = _lin()
    lin.lookahead_from = 1
    off = lin._esdf_clearance(centres, _scene(mask=mask))[1, :, 0]
    rec_off = dict(lin._last_lookahead)
    on = lin._esdf_clearance(centres, _scene(mask=mask, strict=True))[1, :, 0]
    rec_on = dict(lin._last_lookahead)
    assert np.isinf(off[FOREARM]) and np.isfinite(on[FOREARM]) and on[FOREARM] < 0.0
    if mask is None:
        assert np.isinf(off[FINGER]) and np.isinf(on[FINGER])        # the finger still drops
    assert on[BARROW] == off[BARROW] and np.isfinite(on[BARROW])      # non-target: always kept
    assert "n_tail_target_unauthorized" not in rec_off
    assert rec_on["n_tail_target_unauthorized"] == 1


@pytest.mark.parametrize("mask", [None, CONTACT], ids=["relax", "exclude_authorized"])
def test_sweep_rows_exclude_only_the_finger(mask):
    lin, centres = _lin(), _centres()
    pc = _checker(lin, centres)
    off = pc._target_rows(centres, _scene(mask=mask))
    on = pc._target_rows(centres, _scene(mask=mask, strict=True))
    assert off[0, FOREARM]                                             # P1: the forearm was exempt
    assert on is None or not on[0, FOREARM]
    if mask is None:
        assert off[0, FINGER] and on[0, FINGER]
    else:
        # exclude_authorized: the finger reads the target-free layer (no apple) — no E1 test
        assert not off[0, FINGER] and on is None
    assert pc.target_rows_unauthorized == 1
    for t in (off, on):
        assert t is None or not (t[0, BARROW] or t[0, DISTROW] or t[0, HELD])


@pytest.mark.parametrize("mask", [None, CONTACT], ids=["relax", "exclude_authorized"])
def test_final_verdict_catches_the_forearm_only_on(mask):
    lin, centres = _lin(), _centres()
    pc = _checker(lin, centres)
    w_off, rec_off = pc.violation(np.zeros((1, 1)), _scene(mask=mask), record=True)
    w_on, rec_on = pc.violation(np.zeros((1, 1)), _scene(mask=mask, strict=True), record=True)
    assert w_off > 0.0                                                 # the forearm was not seen
    assert w_on < 0.0 and rec_on["worst"]["link"] == "link_right_arm_5"
    assert rec_on["target_rows_unauthorized_kept"] == 1
    assert "target_rows_unauthorized_kept" not in rec_off


def test_non_target_rows_are_checked_on_and_off():
    lin, centres = _lin(), _centres()
    a = lin._esdf_clearance(centres, _scene(obstacle_margin=M, target_volume=None))[0, :, 0]
    cls_a = lin._last_obstacle_class[0].copy()
    b = lin._esdf_clearance(centres, _scene(obstacle_margin=M, strict=True))[0, :, 0]
    cls_b = lin._last_obstacle_class[0]
    for row in (BARROW, DISTROW):
        assert cls_a[row] == cls_b[row] == OBSTACLE and a[row] == b[row]
    assert a[BARROW] == pytest.approx(0.008 - R_Q - M, abs=1e-9)


def test_target_volume_rows_also_need_permission():
    """The ball names both rows 0/1 (nearest surface = apple); strict keeps only the finger."""
    lin, centres = _lin(), _centres()
    s = dataclasses.replace(_scene(target_volume=BALL, strict=True), manipulated_points=None)
    pc = _checker(lin, centres)
    assert pc._target_rows(centres, s) is None                         # no d_object: as before
    d = np.asarray(s.esdf.distance(centres.reshape(-1, 3))).reshape(1, 5)
    inside = lin.in_target_volume(d, centres.reshape(-1, 3), s)
    assert inside[0, FINGER] and inside[0, FOREARM]
    lin._esdf_clearance(centres, dataclasses.replace(s, obstacle_margin=M))
    cls = lin._last_obstacle_class[0]
    assert cls[FINGER] == TARGET and cls[FOREARM] == OBSTACLE


# =============================================================================================
# 4. P3 — obstacle_margin keeps target_volume (flag-free); V11AB unchanged
# =============================================================================================


def test_p3_target_volume_survives_the_obstacle_margin():
    field = _Field()
    field.stats = {"target_free": {"ball": _BALL_STATS}}
    radii = np.full(4, R_Q)
    s = scene_from_constraint_set(_cs(field), radii,
                                  _config(obstacle_margin=M, target_volume_exempt=True))
    assert s.obstacle_margin == M
    assert s.target_volume is not None and np.allclose(s.target_volume[:4], [*APPLE, 0.072])
    assert np.allclose(s.support_planes, support_planes_of(_cs(field)))


def test_p3_v11ab_setting_is_unchanged():
    """`obstacle_margin` 0.010, `target_volume_exempt` off, look-ahead on (V11AB): no
    `target_volume`, the T41 a keys as before, and the same clearances as the pre-T44 snapshot."""
    field = _Field()
    field.stats = {"target_free": {"ball": _BALL_STATS}}
    radii = np.full(4, R_Q)
    cfg = TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": 0.0, "use_support_planes": False,
                      "obstacle_margin": 0.010},
        "horizon": {"horizon": 4, "execution_length": 1, "plan_horizon": 4,
                    "lookahead": "obstacles"}})
    s = scene_from_constraint_set(_cs(field), radii, cfg)
    assert s.target_volume is None and s.obstacle_margin == 0.010
    assert s.obstacle_margin_support is False and s.held_obstacle_margin is False
    assert s.held_capture_spheres is None
    assert np.array_equal(s.support_planes, support_planes_of(_cs(field)))
    assert s.target_permission_strict is False and s.target_membership == "ball"
    old = dataclasses.replace(s, target_contact_rows=None)               # what pre-T44 carried
    centres = _centres(3)
    for lookahead in (None, 1):
        a, b = _lin(), _lin()
        a.lookahead_from = b.lookahead_from = lookahead
        assert np.array_equal(a._esdf_clearance(centres, old), b._esdf_clearance(centres, s))
        assert getattr(a, "_last_lookahead", None) == getattr(b, "_last_lookahead", None)


# =============================================================================================
# 5. ball_owned membership — a kept neighbour inside the ball is not the target
# =============================================================================================


def test_ball_owned_membership_does_not_call_the_distractor_the_target():
    lin, centres = _lin(), _centres()
    flat = centres.reshape(-1, 3)
    ball = _scene(mask=CONTACT, strict=True, target_volume=BALL)
    owned = dataclasses.replace(ball, target_membership="layer_diff")
    d = np.asarray(ball.esdf.target_free_distance(flat)).reshape(1, 5)
    by_ball = lin.in_target_volume(d, flat, ball)
    by_layer = lin.in_target_volume(d, flat, owned)
    assert by_ball[0, DISTROW]                     # the ball calls the kept neighbour the target
    assert not by_layer[0, DISTROW]                # the layers agree there: not the target
    assert by_layer[0, FINGER] and by_layer[0, FOREARM]          # the apple: only main has it
    assert not by_layer[0, BARROW] and not by_layer[0, HELD]
    lin._esdf_clearance(centres, dataclasses.replace(owned, obstacle_margin=M))
    cls = lin._last_obstacle_class[0]
    assert cls[DISTROW] == OBSTACLE and cls[FINGER] == TARGET and cls[FOREARM] == OBSTACLE
    pc = _checker(lin, centres)
    rows_ball = pc._target_rows(centres, ball)
    rows_owned = pc._target_rows(centres, owned)
    assert rows_ball[0, DISTROW] and (rows_owned is None or not rows_owned[0, DISTROW])


def test_layer_membership_without_the_free_layer_names_nothing():
    class _NoFree:
        has_labels = False
        grid = types.SimpleNamespace(voxel_size=VOXEL)

        def distance(self, pts):
            return _Field().distance(pts)

        def gradient(self, pts):
            return _Field().gradient(pts)

    lin, centres = _lin(), _centres()
    s = dataclasses.replace(_scene(_NoFree(), strict=True, target_volume=BALL),
                            target_membership="layer_diff")
    d = np.zeros((1, 5))
    assert not lin.in_target_volume(d, centres.reshape(-1, 3), s).any()


# =============================================================================================
# 6. recorded chunk (T40 G E3b, when present) — off identical; on moves only unauthorized rows
# =============================================================================================

_RUN = pathlib.Path("/mnt/dev/work/outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/"
                    "server_constraints/run_0000")
_AUTH = ("ee_left", "ee_finger_l1", "ee_finger_l2")


@pytest.mark.skipif(not _RUN.exists(), reason="T40 G E3b 기록이 없다")
@pytest.mark.parametrize("index", [130, 145])
def test_recorded_chunk_strict_moves_only_unauthorized_target_rows(index):
    from tests.trajopt import test_t41a_obstacle_margin as A

    lin, centres, scene, planes, auth = A._replay(index)
    base = scene(M, support_planes=planes)
    off = lin._esdf_clearance(centres, base)[0, :, 0]
    cls_off = lin._last_obstacle_class[0].copy()
    loaded = lin._esdf_clearance(centres, dataclasses.replace(base, target_contact_rows=auth))[0, :, 0]
    assert np.array_equal(off, loaded)
    on = lin._esdf_clearance(centres, dataclasses.replace(
        base, target_contact_rows=auth, target_permission_strict=True))[0, :, 0]
    cls_on = lin._last_obstacle_class[0]
    moved = cls_on != cls_off
    assert np.all(cls_off[moved] == TARGET) and np.all(cls_on[moved] == OBSTACLE)
    assert not moved[auth].any()
    assert np.all(on[auth] == off[auth])
    assert np.all(on <= off + 1e-12)                                   # margins only grow


# =============================================================================================
# 7. T44b I2b — while holding: the holding hand's gripper + N links above it (URDF chain)
# =============================================================================================
#
# Rig: a held apple (sphere H, r 40 mm) whose remnant is still in the field. Every robot row sits
# at the **same point**, 1 mm inside that remnant, so E1's `|d − d_object| ≤ voxel` names every row.
# Carry: `manipulated_link_margin` is the same for every link, so the I2 rule names no row (V-I2-3).
#
# | row | link (URDF names) | left holds, N=1 | left holds, N=0 | right holds, N=1 |
# |---|---|---|---|---|
# | 0-2 | `ee_left`, `ee_finger_l1`, `ee_finger_l2` | permitted | permitted | checked |
# | 3 | `link_left_arm_6` (above `FT_sensor_L`, which has no sphere) | permitted | checked | checked |
# | 4 | `link_left_arm_5` (forearm) | checked | checked | checked |
# | 5-7 | `ee_right`, `ee_finger_r1`, `ee_finger_r2` | checked | checked | permitted |
# | 8 | `link_right_arm_6` | checked | checked | permitted |
# | 9 | `link_right_arm_5` | checked | checked | checked |

HNAMES = ["ee_left", "ee_finger_l1", "ee_finger_l2", "link_left_arm_6", "link_left_arm_5",
          "ee_right", "ee_finger_r1", "ee_finger_r2", "link_right_arm_6", "link_right_arm_5"]
H = np.array([0.0, 0.0, 0.30])
H_R = 0.040
PERMIT = {
    ("left", 1): [True, True, True, True, False, False, False, False, False, False],
    ("left", 0): [True, True, True, False, False, False, False, False, False, False],
    ("right", 1): [False, False, False, False, False, True, True, True, True, False],
    ("right", 0): [False, False, False, False, False, True, True, True, False, False],
}
HOLD = {"left": "ee_left", "right": "ee_right"}


@pytest.fixture(scope="module")
def urdf_model():
    from benchmark.ag3s.robot_models import RBY1_URDF, parse_urdf

    return parse_urdf(RBY1_URDF)


def _builder_ns(urdf_model, names=HNAMES, contact=None):
    from benchmark.ag3s.config import ContactConfig

    return types.SimpleNamespace(
        sphere_link_names=list(names), robot_model=types.SimpleNamespace(model=urdf_model),
        clearance_policy=types.SimpleNamespace(contact=contact or ContactConfig()))


def _held_record(urdf_model, hold_link, **kw):
    from benchmark.ag3s.constraints.to_adapter import _held_permission_record

    return _held_permission_record(_builder_ns(urdf_model, **kw),
                                   types.SimpleNamespace(parent_link=hold_link))


class _HeldField:
    """Table + the held apple's remnant (the field has not let go of it yet)."""

    has_labels = False

    def __init__(self):
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)
        self.stats = {}

    @staticmethod
    def distance(pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return np.minimum(p[:, 2], np.linalg.norm(p - H, axis=1) - H_R)

    def gradient(self, pts):
        return _num_grad(self.distance, pts)


def _held_cs(record, hold_link="ee_left"):
    return types.SimpleNamespace(
        constraints=None, esdf=_HeldField(), support_surfaces=(),
        attached=types.SimpleNamespace(points=np.zeros((1, 3)), parent_link=hold_link,
                                       primitives=[]),
        manipulated=types.SimpleNamespace(points=None, sphere_centers=H[None].copy(),
                                          sphere_radii=np.array([H_R])),
        manipulated_link_margin=np.full(len(HNAMES), FULL), target_field_exclude=None,
        target_field_policy="exclude_authorized", destination_label=None, destination_margin=None,
        metrics={"target_permission": {"held": record}})


def _held_centres(steps=1):
    one = np.repeat((H + np.array([0.0, -(H_R - 0.001), 0.0]))[None], len(HNAMES), axis=0)
    return np.repeat(one[None], steps, axis=0)


def _held_lin():
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.full(len(HNAMES), R_Q)
    lin.query_radii = np.full(len(HNAMES), R_Q)
    lin.n_spheres = len(HNAMES)
    lin._attached_local = None
    lin._attached_link = None
    lin.robot_model = types.SimpleNamespace(sphere_link_names=HNAMES)
    return lin


def _held_scene(urdf_model, side="left", strict=True, n_parent=1, record=True, **collision):
    rec = _held_record(urdf_model, HOLD[side]) if record else None
    cfg = _config(target_permission_strict=strict, held_permission_parent_links=n_parent,
                  **collision)
    return scene_from_constraint_set(_held_cs(rec, HOLD[side]), np.full(len(HNAMES), R_Q), cfg)


def test_held_parent_config_default_round_trip_and_validation():
    assert TrajOptConfig().collision.held_permission_parent_links == 1           # user verdict 2026-10-10
    cfg = _config(target_permission_strict=True, held_permission_parent_links=0)
    assert TrajOptConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.collision.held_permission_parent_links == 0
    for bad in (-1, 1.0, True, "1"):
        with pytest.raises(ValueError, match="held_permission_parent_links"):
            _config(held_permission_parent_links=bad)


def test_hold_chain_is_found_from_the_urdf_parent_relation(urdf_model):
    """`ee_left`'s URDF parent is the fixed `FT_sensor_L` (no sphere) → the first link above the
    gripper with a row is `link_left_arm_6`; then down the arm to `link_left_arm_0`."""
    from benchmark.ag3s.constraints.to_adapter import held_contact_depth, urdf_parent_of

    parent_of = urdf_parent_of(types.SimpleNamespace(model=urdf_model))
    assert parent_of["ee_left"] == "FT_sensor_L" and parent_of["FT_sensor_L"] == "link_left_arm_6"
    assert parent_of["ee_right"] == "FT_sensor_R" and parent_of["FT_sensor_R"] == "link_right_arm_6"
    # URDF names: both left fingers hang from `ee_left` (the MJCF swap of l1/l2 is not in the URDF)
    assert parent_of["ee_finger_l1"] == parent_of["ee_finger_l2"] == "ee_left"
    names = HNAMES + [f"link_left_arm_{i}" for i in (4, 3, 2, 1, 0)] + ["link_torso_5"]
    depth, parents = held_contact_depth(names, "ee_left",
                                        ("ee_left", "ee_finger_l1", "ee_finger_l2"), parent_of)
    assert parents == [f"link_left_arm_{i}" for i in (6, 5, 4, 3, 2, 1, 0)] + ["link_torso_5"]
    assert depth[:10].tolist() == [0, 0, 0, 1, 2, -1, -1, -1, -1, -1]
    assert held_contact_depth(names, "ee_left", (), {})[0].tolist() == [-1] * len(names)


@pytest.mark.parametrize("side", ["left", "right"])
def test_held_record_names_the_holding_side(urdf_model, side):
    rec = _held_record(urdf_model, HOLD[side])
    own = [n for n in HNAMES if n.startswith(f"ee_{side}") or n.startswith(f"ee_finger_{side[0]}")]
    assert rec["hold_link"] == HOLD[side] and rec["side"] == side
    assert rec["contact_links"] == sorted(own)
    assert rec["parent_links"] == [f"link_{side}_arm_6", f"link_{side}_arm_5"]
    assert np.array_equal(np.asarray(rec["depth"]) == 0, [n in own for n in HNAMES])
    json.dumps(rec)


def test_held_record_edge_cases(urdf_model):
    from benchmark.ag3s.constraints.to_adapter import _held_permission_record

    assert _held_permission_record(_builder_ns(urdf_model), None) is None      # nothing held
    # pre-T32 contract: the object rides the finger — still the left hand; the palm is a contact
    # link, so the chain's first counted link is still `link_left_arm_6`
    finger = _held_record(urdf_model, "ee_finger_l1")
    assert finger["side"] == "left" and finger["parent_links"][0] == "link_left_arm_6"
    assert finger["depth"] == _held_record(urdf_model, "ee_left")["depth"]
    # a hold link no side lists: fail closed (no row permitted)
    unknown = _held_record(urdf_model, "link_left_arm_5")
    assert unknown["side"] is None and unknown["depth"] == [-1] * len(HNAMES)
    # a model without a URDF (mock): the gripper rows only
    mock = _builder_ns(urdf_model)
    mock.robot_model = types.SimpleNamespace()
    rec = _held_permission_record(mock, types.SimpleNamespace(parent_link="ee_left"))
    assert rec["parent_links"] == [] and rec["depth"][:5] == [0, 0, 0, -1, -1]


def test_held_contact_rows_of(urdf_model):
    from benchmark.trajopt.linearize import held_contact_rows_of, held_permission_of

    rec = _held_record(urdf_model, "ee_left")
    assert held_contact_rows_of(rec, 1, 10).tolist() == PERMIT[("left", 1)]
    assert held_contact_rows_of(rec, 0, 10).tolist() == PERMIT[("left", 0)]
    assert held_contact_rows_of(rec, 2, 10)[4]                           # 2 reaches the forearm
    assert held_contact_rows_of(None, 1, 10) is None
    with pytest.raises(ValueError, match="share one constraint model"):
        held_contact_rows_of(rec, 1, 11)
    assert held_permission_of(_held_cs(rec)) is rec
    assert held_permission_of(types.SimpleNamespace(metrics={})) is None
    assert held_permission_of(types.SimpleNamespace()) is None


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("n_parent", [0, 1])
def test_scene_held_permission_under_strict(urdf_model, side, n_parent):
    s = _held_scene(urdf_model, side, n_parent=n_parent)
    assert s.target_permission_strict is True
    assert s.target_contact_rows.tolist() == PERMIT[(side, n_parent)]


def test_scene_held_off_and_fallback_keep_the_i2_rule(urdf_model):
    i2 = [False] * len(HNAMES)              # carry: equal margins → the I2 rule names no row
    off = _held_scene(urdf_model, strict=False)
    assert off.target_permission_strict is False and off.target_contact_rows.tolist() == i2
    no_record = _held_scene(urdf_model, record=False)
    assert no_record.target_contact_rows.tolist() == i2
    # the record does not reach a scene that is not holding (approach keeps the mask)
    cs = _cs(_Field(), policy="exclude_authorized", mask=CONTACT)
    cs.metrics = {"target_permission": {"held": {"depth": [0, 0, 0, 0]}}}
    s = scene_from_constraint_set(cs, np.full(4, R_Q), _config(target_permission_strict=True))
    assert s.target_contact_rows.tolist() == CONTACT.tolist()


@pytest.mark.parametrize("side", ["left", "right"])
def test_carry_waypoint_lookahead_sweep_verdict(urdf_model, side):
    """Strict + holding: the holding gripper and the link above it keep the held-remnant
    exception in all four checks; the other hand and the forearm are checked."""
    permit = np.asarray(PERMIT[(side, 1)])
    centres = _held_centres(2)
    off_s = _held_scene(urdf_model, side, strict=False, obstacle_margin=M)
    on_s = _held_scene(urdf_model, side, obstacle_margin=M)
    # waypoint class
    lin = _held_lin()
    lin._esdf_clearance(centres[:1], off_s)
    cls_off = lin._last_obstacle_class[0].copy()
    lin._esdf_clearance(centres[:1], on_s)
    cls_on = lin._last_obstacle_class[0]
    assert np.all(cls_off == TARGET)                                    # off: every row exempt
    assert np.all(cls_on[permit] == TARGET) and np.all(cls_on[~permit] == OBSTACLE)
    # look-ahead tail
    lin = _held_lin()
    lin.lookahead_from = 1
    tail_off = lin._esdf_clearance(centres, off_s)[1, :, 0]
    tail_on = lin._esdf_clearance(centres, on_s)[1, :, 0]
    assert np.all(np.isinf(tail_off))
    assert np.all(np.isinf(tail_on[permit])) and np.all(np.isfinite(tail_on[~permit]))
    assert np.all(tail_on[~permit] < 0.0)
    assert lin._last_lookahead["n_tail_target_unauthorized"] == int((~permit).sum())
    # sweep path rows + final verdict
    lin = _held_lin()
    pc = _checker(lin, centres[:1])
    rows_on = pc._target_rows(centres[:1], on_s)
    assert np.array_equal(rows_on[0], permit)
    assert pc.target_rows_unauthorized == int((~permit).sum())
    w_off, rec_off = _checker(_held_lin(), centres[:1]).violation(np.zeros((1, 1)), off_s,
                                                                  record=True)
    w_on, rec_on = _checker(_held_lin(), centres[:1]).violation(np.zeros((1, 1)), on_s,
                                                                record=True)
    assert w_off > 0.0                                  # off: the remnant is invisible to all rows
    assert w_on < 0.0 and rec_on["worst"]["link"] in {HNAMES[i] for i in np.flatnonzero(~permit)}
    assert rec_on["target_rows_unauthorized_kept"] == int((~permit).sum())


def test_carry_parent_zero_checks_the_wrist(urdf_model):
    centres = _held_centres()
    lin = _held_lin()
    lin._esdf_clearance(centres, _held_scene(urdf_model, n_parent=0, obstacle_margin=M))
    cls = lin._last_obstacle_class[0]
    assert cls[HNAMES.index("link_left_arm_6")] == OBSTACLE
    assert np.all(cls[:3] == TARGET)


def test_carry_off_is_unchanged_by_the_record(urdf_model):
    """Strict off: the held record changes nothing — same snapshot vector as without it, same
    clearances, classes, look-ahead record and sweep record."""
    with_rec = _held_scene(urdf_model, strict=False, obstacle_margin=M)
    without = _held_scene(urdf_model, strict=False, record=False, obstacle_margin=M)
    assert np.array_equal(with_rec.target_contact_rows, without.target_contact_rows)
    centres = _held_centres(3)
    for lookahead in (None, 1):
        a, b = _held_lin(), _held_lin()
        a.lookahead_from = b.lookahead_from = lookahead
        assert np.array_equal(a._esdf_clearance(centres, without), b._esdf_clearance(centres, with_rec))
        assert np.array_equal(a._last_obstacle_class, b._last_obstacle_class)
        assert getattr(a, "_last_lookahead", None) == getattr(b, "_last_lookahead", None)
    va, ra = _checker(_held_lin(), centres[:1]).violation(np.zeros((1, 1)), without, record=True)
    vb, rb = _checker(_held_lin(), centres[:1]).violation(np.zeros((1, 1)), with_rec, record=True)
    assert va == vb and ra == rb


def test_build_constraint_set_records_the_held_hand():
    """Wiring: `build_constraint_set` puts the record under `metrics["target_permission"]["held"]`
    (`None` when nothing is held). The planar test arm has no ContactConfig side → fail closed."""
    pytest.importorskip("casadi")
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.constraints.attached import attach_from_target
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from benchmark.ag3s.constraints.to_adapter import build_constraint_set
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import TargetGeometry
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, FINGERTIP_LINK, PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 3, "max_attached_primitives": 2}})
    builder = ConstraintBuilder(arm, cfg.constraint, cfg.geometry,
                                attached_parent_links=(FINGERTIP_LINK,))
    tip = arm.end_effector_numeric(DEFAULT_ARM_Q)
    pts = tip + np.random.default_rng(0).normal(0.0, 0.013, (60, 3))
    target = TargetGeometry(id=7, points=pts, point_indices=np.arange(60), centroid=pts.mean(0),
                            bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9)
    attached = attach_from_target(target, robot_model=arm, robot_state=DEFAULT_ARM_Q,
                                  parent_link=FINGERTIP_LINK, label="held")
    kw = dict(candidates=[], support_surfaces=[], target=None, robot_state=DEFAULT_ARM_Q,
              phase="grasp")
    held = build_constraint_set(builder, attached=attached, **kw)
    rec = held.metrics["target_permission"]["held"]
    assert rec["hold_link"] == FINGERTIP_LINK and rec["side"] is None
    assert rec["depth"] == [-1] * len(builder.sphere_link_names)
    assert build_constraint_set(builder, **kw).metrics["target_permission"]["held"] is None


_MJCF = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")


@pytest.mark.skipif(not _MJCF.exists(), reason="MuJoCo scene not present")
def test_server_constraint_model_parent_is_arm_6():
    """The server's constraint model (`build_constraint_robot_model`: URDF capsules + MJCF
    gap-filling capsules) — the link right above each gripper with spheres is `link_<side>_arm_6`."""
    mujoco = pytest.importorskip("mujoco")
    from benchmark.ag3s.config import ContactConfig
    from benchmark.ag3s.constraints.to_adapter import _held_permission_record
    from benchmark.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.trajopt.linearize import held_contact_rows_of

    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    scene = TransportScene.attach(model, mujoco.MjData(model))
    cm = build_constraint_robot_model(scene)
    builder = types.SimpleNamespace(sphere_link_names=list(cm.sphere_link_names), robot_model=cm,
                                    clearance_policy=types.SimpleNamespace(contact=ContactConfig()))
    names = np.asarray(cm.sphere_link_names)
    for side, s in (("left", "l"), ("right", "r")):
        rec = _held_permission_record(builder, types.SimpleNamespace(parent_link=f"ee_{side}"))
        assert rec["parent_links"][:2] == [f"link_{side}_arm_6", f"link_{side}_arm_5"]
        rows = held_contact_rows_of(rec, 1, names.size)
        assert set(names[rows]) == {f"ee_{side}", f"ee_finger_{s}1", f"ee_finger_{s}2",
                                    f"link_{side}_arm_6"}
        assert (names == f"link_{side}_arm_6").sum() > 0
