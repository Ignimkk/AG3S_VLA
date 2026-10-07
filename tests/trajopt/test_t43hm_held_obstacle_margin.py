"""T43 HM — the held object's rows get the robot rows' classification and margin
(`collision.held_obstacle_margin`, `--held-obstacle-margin`).

User (2026-10-07): once grasped the apple is part of the robot and has to avoid the hurdle like the
fingers do. T41 a gave every held row class `held` and no obstacle margin ("the held object is the
target") with no measurement behind it.

What this file pins:

1. **Off is bit-identical** — the new scene fields present but the switch off change nothing.
2. On: a held row nearest an obstacle gets `max(margin, obstacle_margin)`; nearest a support plane
   (always, even with `obstacle_margin_support`) or the object's own remnant (surface point inside
   the capture volume + 2 voxels) keeps its margin; robot rows are untouched.
3. Destination: the 20 mm rule stays, and where both apply the larger margin wins.
4. Fail-closed: no capture volume / no gradient ⇒ not a remnant.
5. Records: `worst_row` names the class and the rule; `held_rows_record` · `to.metrics["held_rows"]`.
6. Wiring: config validation · round trip, CLI, `scene_from_constraint_set`, `AG3S.held_capture_spheres`,
   `SafePolicy._held_capture_for_scene`.
7. Recorded V8 chunks (when present): the rows the offline survey (`outputs/impl/T43HM/`) found near
   the hurdle are obstacle rows; the lift rows nearest the apple's own image are remnant rows.
"""

from __future__ import annotations

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
    scene_from_constraint_set,
)

VOXEL = 0.005
M = 0.010
DEST = "destination0"
DEST_MARGIN = 0.020
BAR_R = 0.012
R_H = 0.020                       # held query sphere radius
R_F = 0.004                       # finger sphere radius
CAPTURE = np.array([[0.0, 0.0, 0.035, 0.033]])     # where the apple sat when the grasp closed


class _Field:
    """z = 0 table, a bar (sphere), the apple's top remnant (small sphere inside the capture volume)
    and a crate block carrying the destination label. Analytic distance, numeric gradient."""

    has_labels = True

    def __init__(self):
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)
        self.bar = np.array([0.3, 0.0, 0.30])
        self.remnant = np.array([0.0, 0.0, 0.060])
        self.remnant_r = 0.008
        self.crate_c = np.array([-0.3, 0.0, 0.10])
        self.crate_h = np.array([0.05, 0.05, 0.10])

    def _parts(self, p):
        q = np.maximum(np.abs(p - self.crate_c) - self.crate_h, 0.0)
        return np.stack([
            p[:, 2],
            np.linalg.norm(p - self.bar, axis=1) - BAR_R,
            np.linalg.norm(p - self.remnant, axis=1) - self.remnant_r,
            np.linalg.norm(q, axis=1),
        ], axis=1)

    def distance(self, pts):
        return self._parts(np.asarray(pts, np.float64).reshape(-1, 3)).min(axis=1)

    def gradient(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        e = 1e-6
        return np.stack([(self.distance(p + e * ax) - self.distance(p - e * ax)) / (2 * e)
                         for ax in np.eye(3)], axis=1)

    def is_label(self, pts, name):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return (self._parts(p).argmin(axis=1) == 3) & (name == DEST)


#: query rows: 0 = finger (robot) by the bar; 1–4 = held spheres.
ROWS = {"finger_bar": 0, "held_bar": 1, "held_table": 2, "held_remnant": 3, "held_crate": 4}
PLANES = np.array([[0.0, 0.0, 1.0, 0.0]])


def _rig():
    f = _Field()
    centres = np.stack([
        f.bar - np.array([0.0, 0.0, BAR_R + R_F + 0.008]),            # finger 8 mm under the bar
        f.bar + np.array([0.0, BAR_R + R_H + 0.006, 0.0]),            # held 6 mm beside the bar
        np.array([0.0, 0.25, R_H + 0.004]),                           # held 4 mm above the table
        f.remnant + np.array([0.0, 0.0, f.remnant_r + R_H + 0.003]),  # held 3 mm above the remnant
        np.array([-0.3, 0.0, 0.20 + R_H + 0.006]),                    # held 6 mm above the crate
    ])[None]
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.array([R_F])
    lin.query_radii = np.array([R_F, R_H, R_H, R_H, R_H])
    lin.n_spheres = 1
    lin._attached_local = None
    return lin, centres, f


def _scene(f, *, margin=M, held=None, capture=CAPTURE, dest_margin=DEST_MARGIN, **extra):
    """`held=None` = the T43 HM fields are not passed at all (the pre-HM snapshot)."""
    if held is not None:
        extra = {"held_obstacle_margin": held, "held_capture_spheres": capture, **extra}
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((1, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.array([R_F]), candidate_ids=np.zeros(0, np.int64),
        esdf=f, esdf_margin=0.0, destination_label=DEST, destination_margin=dest_margin,
        obstacle_margin=float(margin), support_planes=PLANES, **extra)


def _clear(lin, centres, scene):
    return lin._esdf_clearance(centres, scene)[0, :, 0]


def _classes(lin):
    return [OBSTACLE_CLASSES[k] for k in lin._last_obstacle_class[0]]


# =============================================================================================
# 1. off
# =============================================================================================


def test_default_off_is_bit_identical():
    lin, centres, f = _rig()
    plain = _clear(lin, centres, _scene(f))
    loaded = _clear(lin, centres, _scene(f, held=False))
    assert plain.tobytes() == loaded.tobytes()
    assert _classes(lin)[1:] == ["held"] * 4
    assert not hasattr(lin, "_last_held_rows")
    assert SceneSnapshot.__dataclass_fields__["held_obstacle_margin"].default is False
    assert SceneSnapshot.__dataclass_fields__["held_capture_spheres"].default is None
    assert TrajOptConfig().collision.held_obstacle_margin is False


# =============================================================================================
# 2. on
# =============================================================================================


def test_on_classifies_the_held_rows_and_margins_only_the_obstacle_one():
    lin, centres, f = _rig()
    off = _clear(lin, centres, _scene(f))
    on = _clear(lin, centres, _scene(f, held=True))
    assert _classes(lin) == ["obstacle", "obstacle", "support", "held", "obstacle"]
    delta = off - on
    assert delta[ROWS["held_bar"]] == pytest.approx(M, abs=1e-12)
    assert on[ROWS["held_bar"]] == pytest.approx(0.006 - M, abs=1e-9)
    for name in ("finger_bar", "held_table", "held_remnant", "held_crate"):
        assert delta[ROWS[name]] == 0.0, name


def test_robot_rows_are_bit_identical_on_and_off():
    lin, centres, f = _rig()
    off = _clear(lin, centres, _scene(f))
    cls_off = lin._last_obstacle_class.copy()
    on = _clear(lin, centres, _scene(f, held=True))
    assert off[:1].tobytes() == on[:1].tobytes()
    assert np.array_equal(cls_off[:, :1], lin._last_obstacle_class[:, :1])


def test_support_is_always_out_for_held_rows_even_with_the_support_switch():
    """T34 J1: the held spheres clear the table only by its field band."""
    lin, centres, f = _rig()
    off = _clear(lin, centres, _scene(f, obstacle_margin_support=True))
    on = _clear(lin, centres, _scene(f, held=True, obstacle_margin_support=True))
    assert off[ROWS["held_table"]] == on[ROWS["held_table"]]
    assert _classes(lin)[ROWS["held_table"]] == "support"


def test_the_remnant_tolerance_is_two_voxels_from_the_capture_volume():
    lin, centres, f = _rig()
    top = f.remnant[2] + f.remnant_r                          # the remnant's nearest point (0, 0, 0.068)
    for gap, expect in ((0.0, "held"), (0.009, "held"), (0.011, "obstacle")):
        cap = np.array([[0.0, 0.0, 0.035, top - 0.035 - gap]])
        _clear(lin, centres, _scene(f, held=True, capture=cap))
        assert _classes(lin)[ROWS["held_remnant"]] == expect, gap


def test_the_remnant_test_is_about_the_nearest_surface_point_not_the_query_point():
    """The query centre is 31 mm above the remnant, outside the capture volume + 10 mm — still own."""
    lin, centres, f = _rig()
    p = centres[0, ROWS["held_remnant"]]
    sd = np.linalg.norm(p - CAPTURE[0, :3]) - CAPTURE[0, 3]
    assert sd > 2 * VOXEL
    _clear(lin, centres, _scene(f, held=True))
    assert _classes(lin)[ROWS["held_remnant"]] == "held"


# =============================================================================================
# 3. destination
# =============================================================================================


@pytest.mark.parametrize("dest_margin,expect", [(DEST_MARGIN, DEST_MARGIN), (0.004, M)])
def test_destination_rows_take_the_larger_margin(dest_margin, expect):
    lin, centres, f = _rig()
    off = _clear(lin, centres, _scene(f, dest_margin=dest_margin))
    on = _clear(lin, centres, _scene(f, held=True, dest_margin=dest_margin))
    assert off[ROWS["held_crate"]] == pytest.approx(0.006 - dest_margin, abs=1e-9)
    assert on[ROWS["held_crate"]] == pytest.approx(0.006 - expect, abs=1e-9)


# =============================================================================================
# 4. fail closed
# =============================================================================================


def test_without_a_capture_volume_the_remnant_is_an_obstacle():
    lin, centres, f = _rig()
    off = _clear(lin, centres, _scene(f, capture=None))
    on = _clear(lin, centres, _scene(f, held=True, capture=None))
    assert _classes(lin)[ROWS["held_remnant"]] == "obstacle"
    assert off[ROWS["held_remnant"]] - on[ROWS["held_remnant"]] == pytest.approx(M, abs=1e-12)


def test_a_row_with_no_gradient_is_not_a_remnant():
    lin, centres, f = _rig()
    f.gradient = lambda pts: np.zeros((len(np.asarray(pts).reshape(-1, 3)), 3))
    _clear(lin, centres, _scene(f, held=True))
    assert _classes(lin)[ROWS["held_remnant"]] == "obstacle"


def test_without_support_planes_the_table_is_an_obstacle_for_held_rows_too():
    lin, centres, f = _rig()
    _clear(lin, centres, dataclasses.replace(_scene(f, held=True), support_planes=None))
    assert _classes(lin)[ROWS["held_table"]] == "obstacle"


# =============================================================================================
# 5. records
# =============================================================================================


def test_worst_row_names_the_held_class_and_the_rule_only_when_on():
    lin, centres, f = _rig()
    lin._esdf_label_name = lambda point, scene: None
    lin._query_name = lambda q: f"q{q}"
    lin._authorized_rows = lambda *a, **k: None
    for held, cls in ((False, "held"), (True, "obstacle")):
        scene = _scene(f, held=held)
        esdf = lin._esdf_clearance(centres, scene)
        out = lin._identify("esdf", (0, ROWS["held_bar"]), centres, scene,
                            float(esdf[0, ROWS["held_bar"], 0]), esdf_block=esdf)
        assert out["obstacle_margin_class"] == cls
        assert ("held_obstacle_margin" in out) is held


def test_held_rows_record_counts_the_classes_and_their_worst_clearance():
    lin, centres, f = _rig()
    assert lin.held_rows_record() is None
    on = _clear(lin, centres, _scene(f, held=True))
    rec = lin.held_rows_record()
    assert {k: v["n"] for k, v in rec.items()} == {"obstacle": 2, "support": 1, "held": 1}
    assert rec["obstacle"]["min_clearance_m"] == pytest.approx(
        min(on[ROWS["held_bar"]], on[ROWS["held_crate"]]))
    assert rec["held"]["min_clearance_m"] == pytest.approx(on[ROWS["held_remnant"]])
    # nothing held: the record says so instead of keeping the last one
    lin.query_radii = np.array([R_F])
    _clear(lin, centres[:, :1], _scene(f, held=True))
    assert lin.held_rows_record() is None


class _FarField:
    has_labels = False
    grid = types.SimpleNamespace(voxel_size=VOXEL)

    @staticmethod
    def distance(pts):
        return np.full(len(np.asarray(pts).reshape(-1, 3)), 1.0)

    @staticmethod
    def gradient(pts):
        return np.zeros((len(np.asarray(pts).reshape(-1, 3)), 3))


@pytest.mark.parametrize("held", [False, True])
def test_to_metrics_carry_held_rows_only_when_on(held):
    from benchmark.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=6)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    record = {"obstacle": {"n": 3, "min_clearance_m": 0.004}}
    opt.linearizer.held_rows_record = lambda: record
    scene = dataclasses.replace(sc.scene, esdf=_FarField(), esdf_margin=0.0,
                                held_obstacle_margin=held)
    result = opt.solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert ("held_rows" in result.metrics) is held
    if held:
        assert result.metrics["held_rows"] == record


# =============================================================================================
# 6. wiring
# =============================================================================================


def test_config_validation_and_round_trip():
    with pytest.raises(ValueError, match="held_obstacle_margin"):
        TrajOptConfig.from_dict({"collision": {"backend": "esdf", "held_obstacle_margin": True}})
    cfg = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": M,
                                                 "held_obstacle_margin": True}})
    assert cfg.collision.held_obstacle_margin is True
    assert TrajOptConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.with_overrides({}) == cfg


def _held_constraint_set(f):
    from benchmark.ag3s.robot_models.held_object import HELD_ROLE
    from benchmark.ag3s.types import (AttachedCollisionGeometry, Primitive, PrimitiveType,
                                      SupportSurface)

    table = SupportSurface(id=1, normal=np.array([0.0, 0.0, 1.0]), offset=0.0,
                           point_indices=np.arange(3), point_count=3, rms_error=0.001,
                           safety_margin=0.01)
    prim = Primitive(type=PrimitiveType.SPHERE, center=np.array([0.0, 0.0, 0.1]),
                     dimensions=np.full(3, 0.039), semantic_role=HELD_ROLE)
    att = AttachedCollisionGeometry(parent_link="ee_left", T_parent_object=np.eye(4),
                                    primitives=[prim], points=np.zeros((1, 3)))
    return types.SimpleNamespace(
        constraints=None, esdf=f, support_surfaces=(table,), attached=att, manipulated=None,
        manipulated_link_margin=None, target_field_exclude=None, target_field_policy="relax",
        destination_label=DEST, destination_margin=DEST_MARGIN)


def _cfg(**collision):
    return TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": 0.0,
                                                  "use_support_planes": False, **collision}})


def test_scene_from_constraint_set_carries_the_switch_and_the_capture_only_when_on():
    _, _, f = _rig()
    cs = _held_constraint_set(f)
    radii = np.array([R_F])
    off = scene_from_constraint_set(cs, radii, _cfg(obstacle_margin=M), held_capture=CAPTURE)
    assert off.held_obstacle_margin is False and off.held_capture_spheres is None
    on = scene_from_constraint_set(cs, radii, _cfg(obstacle_margin=M, held_obstacle_margin=True),
                                   held_capture=CAPTURE)
    assert on.held_obstacle_margin is True
    assert np.allclose(on.held_capture_spheres, CAPTURE)
    blind = scene_from_constraint_set(cs, radii, _cfg(obstacle_margin=M, held_obstacle_margin=True))
    assert blind.held_obstacle_margin is True and blind.held_capture_spheres is None
    # nothing held: the capture volume is not carried even if handed over
    free = scene_from_constraint_set(types.SimpleNamespace(**{**vars(cs), "attached": None}),
                                     radii, _cfg(obstacle_margin=M, held_obstacle_margin=True),
                                     held_capture=CAPTURE)
    assert free.held_capture_spheres is None


def test_cli_flag_sets_the_config_and_needs_the_obstacle_margin():
    from benchmark.trajopt import serve_safe
    from tests.trajopt.test_t41a_obstacle_margin import BASE_ARGV

    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--obstacle-margin", "0.01"])
    assert "held_obstacle_margin" not in serve_safe.obstacle_overrides(args)
    assert serve_safe.trajopt_config_from_args(args).collision.held_obstacle_margin is False
    args = ap.parse_args(BASE_ARGV + ["--obstacle-margin", "0.01", "--held-obstacle-margin"])
    assert serve_safe.obstacle_overrides(args)["held_obstacle_margin"] is True
    assert serve_safe.trajopt_config_from_args(args).collision.held_obstacle_margin is True
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(ap.parse_args(BASE_ARGV + ["--held-obstacle-margin"]))


def test_ag3s_held_capture_spheres_is_the_first_free_volume_while_holding():
    from benchmark.ag3s.runtime.pipeline import AG3S

    ag = object.__new__(AG3S)
    ag._attached = None
    ag._held_first_free = [(np.array([0.1, 0.2, 0.85]), 0.038)]
    assert ag.held_capture_spheres is None
    ag._attached = object()
    assert np.allclose(ag.held_capture_spheres, [[0.1, 0.2, 0.85, 0.038]])
    ag._held_first_free = []
    assert ag.held_capture_spheres is None


def test_safe_policy_hands_the_capture_volume_over_only_when_on():
    from benchmark.trajopt.safe_policy import SafePolicy

    sp = object.__new__(SafePolicy)
    sp.ag3s = types.SimpleNamespace(held_capture_spheres=CAPTURE)
    sp.to_config = _cfg(obstacle_margin=M)
    assert sp._held_capture_for_scene() is None
    sp.to_config = _cfg(obstacle_margin=M, held_obstacle_margin=True)
    assert np.array_equal(sp._held_capture_for_scene(), CAPTURE)
    sp.ag3s = types.SimpleNamespace()                    # a pipeline without the property
    assert sp._held_capture_for_scene() is None


# =============================================================================================
# 7. recorded V8 chunks (offline survey of T43 HM)
# =============================================================================================

_SURVEY = pathlib.Path("/mnt/dev/work/outputs/impl/T43HM/raw/survey_V8.json")
_V8 = pathlib.Path("/mnt/dev/work/outputs/verify/T43/V8/analysis/analysis.json")
#: (run, seq, k, expected class of the nearest-to-violation held row)
_CASES = [
    ("V8/H/V8_ep1807_s18071", 26, 5, "obstacle"),     # 9.9 mm from the bar (field), true 17.9 mm
    ("V8/H/V8_ep1807_s18071", 27, 0, "obstacle"),     # 9.0 mm
    ("V8/H/V8_ep1995_s19952", 18, 0, "held"),         # lift: the apple's own image, 7.3 mm
    ("V8/H/V8_ep1834_s18343", 17, 0, "held"),         # lift (right arm): 6.1 mm
]


@pytest.mark.skipif(not (_SURVEY.exists() and _V8.exists()), reason="T43 HM survey / V8 records absent")
@pytest.mark.parametrize("key,seq,k,expect", _CASES)
def test_recorded_held_rows_classify_as_the_survey_found(key, seq, k, expect):
    from benchmark.trajopt.experiments.fixed_input_ablation import build_field

    run = json.loads(_SURVEY.read_text())["runs"][key]
    ch = next(c for c in run["chunks"] if c["seq"] == seq)
    row = next(r for r in ch["rows"] if r["k"] == k)["plan"]
    seg = json.loads(_V8.read_text())["runs"][key]["server_segment"]
    first = pathlib.Path(seg["first"])
    i0 = int(first.stem.split("_")[1])
    path = next(first.parent / f"chunk_{i0 + n:05d}.npz" for n in range(int(seg["n"]))
                if int(np.load(first.parent / f"chunk_{i0 + n:05d}.npz")["t_step"]) == seq)
    z = np.load(path, allow_pickle=True)
    field = build_field(z, outside_distance=1.0)
    P = np.asarray(row["P_base"], np.float64)
    r = np.asarray(row["r_mm"], np.float64) / 1000.0
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.array([0.005])
    lin.query_radii = np.concatenate([[0.005], r])
    lin.n_spheres = 1
    lin._attached_local = None
    centres = np.vstack([[P[0] + np.array([0.0, 0.0, 0.5])], P])[None]    # a far robot sphere
    scene = SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((1, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.array([0.005]), candidate_ids=np.zeros(0, np.int64), esdf=field,
        esdf_margin=0.0, obstacle_margin=M, support_planes=np.asarray([ch["plane"]]),
        held_obstacle_margin=True, held_capture_spheres=np.asarray(ch["fit_spheres"]))
    clear = lin._esdf_clearance(centres, scene)[0, 1:, 0]
    j = int(np.argmin(np.asarray(row["d_minus_r_mm"])))
    assert clear[j] * 1000 == pytest.approx(row["d_minus_r_mm"][j] - (10.0 if expect == "obstacle"
                                                                      else 0.0), abs=0.05)
    assert OBSTACLE_CLASSES[int(lin._last_obstacle_class[0, 1 + j])] == expect
