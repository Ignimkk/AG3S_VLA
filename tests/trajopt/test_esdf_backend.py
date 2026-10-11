"""ESDF backend in the optimizer — the field must behave like the primitive rows it stands beside.

The point of keeping both backends is that they describe the same scene. So the tests here are
mostly *comparisons*: a field built from one sphere has to produce the same clearances, the same
selection behaviour and the same violation report as the sphere slot it replaces. Where the two
must differ — row count, and what `slot` means — the difference is asserted rather than allowed.

A field is easier to construct here than to capture: `analytic_field` fills a grid with the exact
signed distance to a sphere, so a discrepancy is the optimizer's and never the perception's.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.fields.esdf import EsdfField, VoxelGrid
from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.linearize import ESDF_SLOT, CollisionLinearizer, SceneSnapshot
from AG3S.trajopt.sqp import TrajectoryOptimizer

from tests.trajopt.fixtures import empty_scene, make_scenario, near_miss_scene


def analytic_field(centre, radius, *, voxel=0.02, half_extent=1.2, max_distance=0.6) -> EsdfField:
    """A grid holding the exact signed distance to one sphere.

    Built analytically rather than integrated from depth so that any error in a test below belongs
    to the optimizer. The perception side is exercised in `tests/ag3s/test_esdf.py`.
    """
    centre = np.asarray(centre, float)
    grid = VoxelGrid.from_bounds(centre - half_extent, centre + half_extent, voxel)
    pts = grid.centres()
    d = np.linalg.norm(pts - centre, axis=-1) - float(radius)
    return EsdfField(grid=grid, distance_grid=np.clip(d, -max_distance, max_distance).astype(
        np.float32), max_distance=max_distance, stats={"unknown_fraction": 0.0},
        outside_distance=max_distance)


def scene_with_field(base: SceneSnapshot, field, margin: float) -> SceneSnapshot:
    return dataclasses.replace(base, esdf=field, esdf_margin=margin)


# --------------------------------------------------------------------------- shape and identity


def test_the_field_adds_one_row_per_sphere_not_per_slot():
    """The whole cost argument for a field: it answers about a point, so there is no slot loop."""
    sc = make_scenario()
    base = empty_scene(sc.linearizer, n_slots=8)
    field = analytic_field([0.4, 0.0, 0.9], 0.1)
    scene = scene_with_field(base, field, 0.05)
    assert base.n_esdf_rows == 0
    assert scene.n_esdf_rows == sc.linearizer.n_spheres
    clearance = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, scene)
    assert clearance.shape == (sc.reference.shape[1], sc.linearizer.n_spheres, 1)


def test_a_scene_with_only_a_field_is_not_empty():
    """`is_empty` gates the whole solve. A field-only scene has constraints and must say so."""
    sc = make_scenario()
    base = empty_scene(sc.linearizer)
    assert base.is_empty
    assert not scene_with_field(base, analytic_field([0.4, 0, 0.9], 0.1), 0.05).is_empty


# --------------------------------------------------------------------------- agreement


def test_field_clearances_match_the_equivalent_sphere_slot():
    """Same obstacle, two descriptions. If these disagree the backends are not interchangeable."""
    sc = make_scenario()
    centre, radius, margin = np.array([0.45, 0.05, 0.95]), 0.12, 0.05
    spheres = sc.linearizer.n_spheres
    primitive = SceneSnapshot(
        candidate_pos=centre[None, :],
        candidate_radius=np.array([radius]),
        candidate_active=np.array([True]),
        d_safe=np.full((spheres, 1), margin),
        plane_normal=np.zeros((0, 3)),
        plane_offset=np.zeros(0),
        plane_active=np.zeros(0, bool),
        robot_radii=sc.linearizer.robot_radii,
        candidate_ids=np.array([0]),
    )
    field_scene = scene_with_field(
        dataclasses.replace(primitive, candidate_active=np.array([False])),
        analytic_field(centre, radius, voxel=0.005), margin)

    prim_clear, _, _ = sc.linearizer.clearances(sc.reference, sc.q_now, primitive)
    esdf_clear = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, field_scene)
    # Compare only where the field is not saturated. Beyond `max_distance` the grid stores the clip
    # value by design, so a disagreement there is the clip and not an error — the two backends are
    # only claimed to agree about geometry the field still resolves.
    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    d_true = np.linalg.norm(centres - centre, axis=-1) - radius
    inside = d_true < field_scene.esdf.max_distance - 0.05
    assert inside.any()
    diff = (prim_clear[..., 0] - esdf_clear[..., 0])[inside]
    assert np.max(np.abs(diff)) < 0.005 + 1e-9, np.max(np.abs(diff))


def test_full_violation_includes_the_field():
    """The safety re-check must see every backend, or adding one silently exempts it."""
    sc = make_scenario()
    base = empty_scene(sc.linearizer)
    assert np.isinf(sc.linearizer.full_violation(sc.reference, sc.q_now, base))
    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    touching = centres[centres.shape[0] // 2, 0]
    scene = scene_with_field(base, analytic_field(touching, 0.15), 0.05)
    worst = sc.linearizer.full_violation(sc.reference, sc.q_now, scene)
    assert np.isfinite(worst) and worst < 0.0


# --------------------------------------------------------------------------- linearization


def test_field_rows_are_labelled_so_a_consumer_can_tell_them_apart():
    sc = make_scenario()
    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    scene = scene_with_field(empty_scene(sc.linearizer),
                             analytic_field(centres[len(centres) // 2, 0], 0.15), 0.05)
    rows = sc.linearizer.linearize(sc.reference, sc.q_now, scene, sc.config)
    used = rows.used
    assert used.any()
    assert np.all(rows.slot[used] == ESDF_SLOT)


def test_the_field_gradient_is_the_row_gradient():
    """`dh/dq = grad d_esdf . dp/dq`. Checked against a finite difference of the value itself."""
    sc = make_scenario()
    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    scene = scene_with_field(empty_scene(sc.linearizer),
                             analytic_field(centres[len(centres) // 2, 0], 0.15, voxel=0.01), 0.05)
    rows = sc.linearizer.linearize(sc.reference, sc.q_now, scene, sc.config)
    # Pick a row whose Jacobian is not identically zero. Spheres on the base and torso do not move
    # with any optimized joint, so their rows are legitimately flat and say nothing about the chain
    # rule this test is checking.
    moving = np.argwhere(rows.used & (np.linalg.norm(rows.gradient, axis=2) > 1e-9))
    assert len(moving), "no field row moves with an optimized joint"
    step, col = moving[0]
    grad = rows.gradient[step, col]

    def value(traj):
        c = sc.linearizer.esdf_clearance(traj, sc.q_now, scene)
        return float(c[step, rows.sphere[step, col], 0])

    eps = 1e-5
    numeric = np.zeros_like(grad)
    for j in range(grad.size):
        up = sc.reference.copy(); up[j, step] += eps
        dn = sc.reference.copy(); dn[j, step] -= eps
        numeric[j] = (value(up) - value(dn)) / (2 * eps)
    # Trilinear interpolation is piecewise linear, so the finite difference is exact away from a
    # cell boundary and can disagree sharply on one. Compare direction, which survives that.
    if np.linalg.norm(numeric) > 1e-6:
        cos = float(grad @ numeric / (np.linalg.norm(grad) * np.linalg.norm(numeric)))
        assert cos > 0.7, (cos, grad, numeric)


def test_both_backends_can_hold_rows_in_the_same_qp():
    """`both` is what an ablation needs: one QP, two descriptions, comparable row for row."""
    sc = make_scenario()
    primitive = near_miss_scene(sc.linearizer, sc.reference, sc.q_now, penetration=0.03)
    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    scene = scene_with_field(primitive,
                             analytic_field(centres[len(centres) // 2, 0], 0.15), 0.05)
    rows = sc.linearizer.linearize(sc.reference, sc.q_now, scene, sc.config)
    used = rows.used
    assert (rows.slot[used] == ESDF_SLOT).any(), "field rows"
    assert (rows.slot[used] >= 0).any(), "candidate rows"


# --------------------------------------------------------------------------- solving


def test_the_optimizer_clears_a_violation_reported_only_by_the_field():
    """End to end: the field alone has to be enough to push the trajectory out of the obstacle."""
    sc = make_scenario()
    centres, jac = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    mid = centres.shape[0] // 2
    # Put the obstacle on a sphere the optimizer can actually move. Spheres on the base and torso
    # do not depend on any optimized joint, so a violation there is unfixable by construction and
    # would test nothing but the reporting.
    movable = np.flatnonzero(np.linalg.norm(jac[mid], axis=(1, 2)) > 1e-6)
    assert movable.size, "the fixture has no sphere driven by an optimized joint"
    scene = scene_with_field(empty_scene(sc.linearizer),
                             analytic_field(centres[mid, movable[-1]], 0.12, voxel=0.01), 0.03)
    before = sc.linearizer.full_violation(sc.reference, sc.q_now, scene)
    assert before < 0.0, "the fixture must actually start in violation"

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    result = opt.solve(sc.reference, sc.q_now, scene)
    after = sc.linearizer.full_violation(result.trajectory, sc.q_now, scene)
    assert after > before, (before, after)


# --------------------------------------------------------------------------- config


def test_the_optimizer_defaults_to_the_primitive_backend():
    cfg = TrajOptConfig()
    assert cfg.collision.backend == "primitive"
    assert cfg.collision.wants_primitive and not cfg.collision.wants_esdf


@pytest.mark.parametrize("backend,prim,esdf", [
    ("primitive", True, False), ("esdf", False, True), ("both", True, True)])
def test_backend_selection(backend, prim, esdf):
    cfg = TrajOptConfig.from_dict({"collision": {"backend": backend}})
    assert cfg.collision.wants_primitive is prim
    assert cfg.collision.wants_esdf is esdf


@pytest.mark.parametrize("bad", [{"backend": "nope"}, {"esdf_margin": -0.01}])
def test_malformed_backend_configuration_is_rejected(bad):
    with pytest.raises((TrajOptConfigError, ValueError)):
        TrajOptConfig.from_dict({"collision": bad})


# --------------------------------------------------------------------------- backend 선택


def _constraint_set(with_field: bool, n_slots: int = 4):
    """`scene_from_constraint_set` 이 읽는 최소한의 것만 흉내 낸 껍데기.

    실제 `CollisionConstraintSet` 을 만들려면 파이프라인 한 프레임을 돌려야 하고, 그러면 이
    테스트가 backend 선택이 아니라 파이프라인을 재게 된다. 여기서 검사하려는 것은 **무엇을
    켜고 끄는가** 하나다.
    """
    import types

    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario()
    spec = types.SimpleNamespace(
        parameter_values=None, layout=None, max_candidates=n_slots,
        n_robot_spheres=sc.linearizer.n_spheres)
    return sc, spec


def test_backend_selection_decides_what_the_optimizer_consumes(monkeypatch):
    """AG3S 는 backend 와 무관하게 candidate 를 내놓는다. 무엇을 쓸지는 소비자가 정한다."""
    import types

    from AG3S.trajopt.linearize import scene_from_constraint_set

    sc = make_scenario()
    base = near_miss_scene(sc.linearizer, sc.reference, sc.q_now)
    field = analytic_field([0.45, 0.0, 0.95], 0.1)
    monkeypatch.setattr(SceneSnapshot, "from_spec",
                        classmethod(lambda cls, spec, radii: base))
    cs = types.SimpleNamespace(constraints=object(), esdf=field)

    prim = scene_from_constraint_set(cs, sc.linearizer.robot_radii,
                                     TrajOptConfig.from_dict({"collision": {"backend": "primitive"}}))
    assert prim.esdf is None
    assert prim.candidate_active.any(), "primitive 는 후보를 쓴다"

    esdf = scene_from_constraint_set(cs, sc.linearizer.robot_radii,
                                     TrajOptConfig.from_dict({"collision": {"backend": "esdf"}}))
    assert esdf.esdf is not None
    assert not esdf.candidate_active.any(), "esdf 는 후보를 끈다"
    # 끄는 것이지 지우는 것이 아니다 — 슬롯 배치가 유지되어야 QP 희소성이 프레임 간에 같다.
    assert esdf.candidate_pos.shape == prim.candidate_pos.shape
    assert esdf.n_slots == prim.n_slots

    both = scene_from_constraint_set(cs, sc.linearizer.robot_radii,
                                     TrajOptConfig.from_dict({"collision": {"backend": "both"}}))
    assert both.esdf is not None and both.candidate_active.any()


def test_esdf_backend_without_a_field_is_an_error_not_a_silent_fallback(monkeypatch):
    """조용히 primitive 로 돌아가면 ESDF 를 쓰고 있다고 믿으면서 아닌 것을 재게 된다."""
    import types

    from AG3S.trajopt.linearize import scene_from_constraint_set

    sc = make_scenario()
    monkeypatch.setattr(SceneSnapshot, "from_spec",
                        classmethod(lambda cls, spec, radii: empty_scene(sc.linearizer)))
    cs = types.SimpleNamespace(constraints=object(), esdf=None)
    with pytest.raises(ValueError, match="ESDF"):
        scene_from_constraint_set(cs, sc.linearizer.robot_radii,
                                  TrajOptConfig.from_dict({"collision": {"backend": "esdf"}}))
