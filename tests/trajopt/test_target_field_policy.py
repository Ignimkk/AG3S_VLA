"""T8b — 아직 쥐지 않은 target 을 거리장에서 빼는 길 (`constraint.target_field_policy`).

**마진으로는 닿지 않는 자리가 있다.** 제약은 `d − r ≥ m` 이고 `m ≥ 0` 이므로 표면 안쪽(음수
여유)을 허용할 수 없는데, 성공한 shadow 궤적의 손끝 구는 사과 표면을 17.96 mm 관통한다
(T7a 실측). 그래서 `m` 이 아니라 `d` 를 바꾼다 — 권한 있는 질의점만 **target 이 빠진 계층**에
거리를 묻는다.

이 파일이 지키는 것은 넷이다.

1. **기본값은 지금 동작이다.** flag 를 하나도 안 주면 정책은 `"relax"` 이고, 그때 새 코드가
   하는 일은 하나도 없다 — 마스크는 `None`, 계층은 0 겹, 시작 로그는 조용하다.
2. **빼는 것은 행이 아니라 계층이다.** ESDF 의 한 행은 최근접 표면까지의 거리 하나뿐이므로
   행을 끄면 그 질의점은 table 에 대해서도 보호를 잃는다. target 없는 계층은 table 을 그대로
   담고 있고, 그 사실을 여기서 수치로 박는다.
3. **권한은 이름으로 준다** (E1 재발 방지). `exclude_authorized` 는 권한 있는 link 의 구만
   그 계층에 묻고, 나머지 구의 거리는 **한 값도** 바뀌지 않는다.
4. **조용한 실패가 없다.** 정책을 켰는데 계층이 없으면 죽고, `exclude_all` 은 시작 로그가
   E1 을 이름으로 부르며 크게 말한다.

거리장은 해석적으로 만든다 (평면과 구의 정확한 signed distance). 그러면 아래 어떤 실패도
cuRobo 적분이 아니라 이 모듈의 것이다 — 지각 쪽은 `tests/ag3s/test_esdf.py` 가 본다.
"""

from __future__ import annotations

import dataclasses
import logging
import types

import numpy as np
import pytest

from benchmark.ag3s.config import TARGET_FIELD_POLICIES, AG3SConfig, ConstraintConfig
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
from benchmark.ag3s.fields.esdf import VoxelGrid
from benchmark.ag3s.types import TARGET_LABEL, CollisionConstraintSet, PipelineStatus
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import SceneSnapshot, scene_from_constraint_set

from tests.trajopt.fixtures import make_scenario, near_miss_scene

# --------------------------------------------------------------------- 해석적 씬 하나
# 사과 하나, 테이블 하나, 벽 하나. 전부 정확한 signed distance 다.
APPLE = np.array([0.50, 0.00, 0.040])
APPLE_R = 0.038
WALL_X = 0.68  # 미세 창 안에서도 보이지만 **미세 계층에는 seed 가 없는** 표면
FINE_HALF = 0.12


def _table(p: np.ndarray) -> np.ndarray:
    return p[:, 2]


def _apple(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p - APPLE[None, :], axis=1) - APPLE_R


def _wall(p: np.ndarray) -> np.ndarray:
    return WALL_X - p[:, 0]


def _layer(fn, lower, upper, voxel, *, label_fn=None):
    """`fn(points) -> (N,)` 을 격자에 구워 한 계층으로. `label_fn` 이 있으면 라벨 층도 붙인다."""
    grid = VoxelGrid.from_bounds(np.asarray(lower, float), np.asarray(upper, float), voxel)
    pts = grid.centres().reshape(-1, 3)
    values = np.asarray(fn(pts), np.float32).reshape(grid.shape)
    layer = layer_from_arrays(values, grid.origin, voxel)
    if label_fn is not None:
        layer.label_grid = np.asarray(label_fn(pts), np.int32).reshape(grid.shape)
        layer.label_names = (TARGET_LABEL,)
    return layer


def _with_target(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(_table(p), _apple(p)), _wall(p))


def _target_is_nearest(p: np.ndarray) -> np.ndarray:
    """라벨 층 — 가장 가까운 표면이 사과면 `0`(= `TARGET_LABEL`), 아니면 `-1`."""
    apple = _apple(p)
    return np.where(apple <= np.minimum(_table(p), _wall(p)), 0, -1)


def apple_scene(*, labelled: bool = True, target_free: bool = True) -> CuroboEsdfField:
    """coarse(20 mm) + fine(5 mm) 합성 필드, 그리고 사과가 빠진 미세 계층 한 겹.

    미세 계층에는 **벽의 seed 가 없다** (창 밖 표면이라 실제 builder 에서도 그렇다). 그래서
    "거친 계층만 아는 표면" 이 이 씬에 실제로 존재하고, `target_free_distance` 의 경계항
    (`max(d, b)`)이 그것을 구해 낸다.
    """
    coarse = _layer(_with_target, (0.20, -0.30, -0.05), (0.80, 0.30, 0.50), 0.02,
                    label_fn=_target_is_nearest if labelled else None)
    fine_lower, fine_upper = APPLE - FINE_HALF, APPLE + FINE_HALF
    fine = _layer(lambda p: np.minimum(_table(p), _apple(p)), fine_lower, fine_upper, 0.005,
                  label_fn=_target_is_nearest if labelled else None)
    free = ((_layer(_table, fine_lower, fine_upper, 0.005),) if target_free else ())
    return CuroboEsdfField((coarse, fine), outside_distance=0.5,
                           target_free_layers=free,
                           target_label=TARGET_LABEL if (labelled and target_free) else None)


# ============================================================ 1. 기본값 — 지금 동작 그대로


def test_the_default_policy_is_relax_and_it_asks_for_nothing():
    """기본 실행에서 이 기능의 비용은 0 이어야 한다 — 계층도, 마스크도, 로그도 없다."""
    assert TARGET_FIELD_POLICIES[0] == "relax"
    assert ConstraintConfig().target_field_policy == "relax"
    assert ConstraintConfig().excludes_target_from_field is False
    assert AG3SConfig.from_dict({}).constraint.target_field_policy == "relax"


def test_the_constraint_set_carries_no_mask_by_default():
    cs = CollisionConstraintSet(
        constraints=None, candidates=[], target=None, support_surfaces=[],
        robot_state=np.zeros(2), phase="transit", timestamp=0.0, frame_id="base",
        status=PipelineStatus.NO_GEOMETRY)
    assert cs.target_field_exclude is None
    assert cs.target_field_policy == "relax"


def test_the_scene_snapshot_carries_no_mask_by_default():
    sc = make_scenario(horizon=3)
    scene = near_miss_scene(sc.linearizer, sc.reference, sc.q_now)
    assert scene.target_free_mask is None


def test_a_field_without_the_extra_layer_answers_exactly_as_before():
    """`target_free_distance` 가 계층 없이 불려도 지금 답과 **같은 값**이어야 한다."""
    field = apple_scene(target_free=False)
    assert field.has_target_free is False
    assert field.target_free_layers == ()
    pts = np.array([[0.50, 0.0, 0.045], [0.30, 0.0, 0.01], [0.61, 0.0, 0.10]])
    assert np.array_equal(field.target_free_distance(pts), field.distance(pts))


def test_an_unknown_policy_name_is_rejected_at_config_time():
    with pytest.raises(Exception, match="target_field_policy"):
        AG3SConfig.from_dict({"constraint": {"target_field_policy": "exclude"}})
    with pytest.raises(Exception, match="target_field_policy"):
        AG3SConfig.from_dict({"constraint": {"target_field_policy": "exclude-all"}})


@pytest.mark.parametrize("policy", ["exclude_authorized", "exclude_all"])
def test_every_policy_name_is_reachable_from_config(policy):
    cfg = AG3SConfig.from_dict({"constraint": {"target_field_policy": policy}})
    assert cfg.constraint.target_field_policy == policy
    assert cfg.constraint.excludes_target_from_field is True


# ================================================ 2. 계층 — 사과만 사라지고 table 은 남는다


def test_the_finger_may_pass_through_the_apple():
    """마진으로는 못 하는 일. `d` 자체가 사과를 안 보게 된다."""
    field = apple_scene()
    inside = np.array([[0.50, 0.0, 0.045]])  # 사과 안쪽 33 mm
    assert field.distance(inside)[0] < -0.02, "이 점은 사과 안이어야 시험이 뜻을 갖는다"
    assert field.target_free_distance(inside)[0] == pytest.approx(0.045, abs=1e-3)


def test_the_table_is_still_there_after_the_apple_is_gone():
    """**행을 끈 것이 아니다.** 끄면 이 점은 테이블 위 4 mm 에서도 자유공간이 된다."""
    field = apple_scene()
    low = np.array([[0.50, 0.0, 0.004]])  # 사과 밑면 바로 안, 테이블 위 4 mm
    free = field.target_free_distance(low)[0]
    assert free == pytest.approx(0.004, abs=1e-3), "테이블까지의 거리를 그대로 답해야 한다"
    assert free < 0.01, "테이블이 사라지면 이 값이 커진다 — 그것이 '행을 끄는' 실패다"
    assert field.distance(low)[0] < free, "사과가 든 답은 더 작아야 한다 (보수적)"


def test_a_surface_only_the_coarse_layer_knows_is_not_lost():
    """창 밖 표면은 `max(d, b)` 가 지킨다 — 벽은 미세 계층에 seed 가 없다.

    이 점은 창 경계에서 10 mm 안쪽이고, 벽은 그 경계 바로 밖 70 mm 지점에 있다. target 없는
    계층만 믿으면 벽이 사라져 100 mm(테이블까지)로 답하게 된다. `max(d, b)` 가 그것을 막는다.
    """
    field = apple_scene()
    p = np.array([[0.61, 0.0, 0.10]])
    free_layer_alone = field.target_free_layers[0].distance(p)[0]
    assert free_layer_alone == pytest.approx(0.10, abs=1e-3), "계층 혼자면 벽을 모른다"
    assert field.distance(p)[0] == pytest.approx(WALL_X - 0.61, abs=2e-3)
    assert field.target_free_distance(p)[0] == pytest.approx(WALL_X - 0.61, abs=2e-3)


def test_the_composition_is_never_more_conservative_than_today():
    """`f >= d` 이므로 `min(f, max(d, b)) >= d` — 이 경로가 없던 위반을 만들 수 없다.

    창 경계에서 `b` 가 0 에 가까워지는데, `max(d, b)` 가 아니라 `b` 를 그대로 쓰면 그 자리에
    유령 장애물이 서고 **해가 없는 행**이 생긴다. 경계를 촘촘히 훑어 그 일이 없음을 본다.
    """
    field = apple_scene()
    grid = field.target_free_layers[0].grid
    last = grid.origin + (np.asarray(grid.shape) - 1) * grid.voxel_size
    rng = np.random.default_rng(3)
    pts = rng.uniform(grid.origin - 0.02, last + 0.02, size=(4000, 3))
    free = field.target_free_distance(pts)
    plain = field.distance(pts)
    assert (free >= plain - 1e-9).all(), "target 을 빼는 것이 여유를 줄일 수는 없다"
    assert (free > plain + 1e-6).any(), "그런데 어딘가에서는 실제로 풀려야 한다"


def test_outside_the_window_the_answer_is_the_conservative_one():
    """창 밖은 target 이 든 답으로 되돌아간다 (지금 동작)."""
    field = apple_scene()
    outside = np.array([[0.25, 0.0, 0.30], [0.75, 0.20, 0.40]])
    for p in outside:
        assert not field._covers(field.target_free_layers[0].grid, p[None, :])[0]
    assert np.array_equal(field.target_free_distance(outside), field.distance(outside))


def test_the_extra_layer_does_not_touch_the_layers_that_answer_everyone_else():
    """본 계층은 한 복셀도 바뀌지 않는다 — 그것이 파내기(E1)와 다른 점 전부다."""
    with_free = apple_scene()
    without = apple_scene(target_free=False)
    pts = np.array([[0.50, 0.0, 0.045], [0.50, 0.0, 0.004], [0.61, 0.0, 0.10],
                    [0.30, 0.0, 0.02]])
    assert np.array_equal(with_free.distance(pts), without.distance(pts))
    assert np.array_equal(with_free.label(pts), without.label(pts))
    assert with_free.grid.voxel_size == without.grid.voxel_size


def test_the_extra_layer_is_not_in_the_composite_min():
    """`layers` 에 넣으면 모든 질의점에서 사과가 사라진다 — 그것이 E1 이다."""
    field = apple_scene()
    assert len(field.layers) == 2
    assert all(layer is not field.target_free_layers[0] for layer in field.layers)
    # 합성에 들어갔다면 이 값이 양수가 됐을 것이다.
    assert field.distance(np.array([[0.50, 0.0, 0.045]]))[0] < 0.0


def test_the_target_label_rides_along_for_diagnostics():
    """라벨은 판정에 쓰지 않지만 **실려는 있어야 한다** — T8a 가 묻는 질문이 그것이다."""
    field = apple_scene()
    assert field.target_label == TARGET_LABEL
    assert field.has_labels
    assert field.is_label(np.array([[0.50, 0.0, 0.060]]), TARGET_LABEL)[0]
    assert not field.is_label(np.array([[0.30, 0.0, 0.010]]), TARGET_LABEL)[0]


def test_the_answer_does_not_depend_on_the_label_layer():
    """같은 씬을 라벨 없이 만들어도 **같은 값**이어야 한다.

    라벨로 갈랐던 첫 구현이 사과 밑면(테이블 위 4 mm, 파지하는 자리)에서 완화를 껐다. 거친
    격자에서 값은 삼선형이고 라벨은 최근접 격자점이라 둘이 어긋나기 때문이다. 지금 합성은
    기하적 하한(`max(d, b)`)만 쓰므로 그 어긋남이 판정에 닿지 않는다.
    """
    pts = np.array([[0.50, 0.0, 0.045], [0.50, 0.0, 0.004], [0.61, 0.0, 0.10],
                    [0.30, 0.0, 0.020], [0.46, 0.03, 0.030]])
    assert np.array_equal(apple_scene().target_free_distance(pts),
                          apple_scene(labelled=False).target_free_distance(pts))


# ================================================ 3. 마스크 — 누가 그 계층에 묻는가


@pytest.fixture(scope="module")
def robot_scene():
    """실제 RB-Y1 제약 모델 위의 장면 하나. 구 위치는 FK 가 정한다."""
    sc = make_scenario(horizon=4)
    centres, _ = sc.linearizer.sphere_states(sc.reference, sc.q_now)
    return sc, centres


def ball_field(centres: np.ndarray, *, radius: float = 0.12, target_free: bool = True):
    """로봇 구 하나를 감싼 공(= target)과 바닥 평면. 공이 빠진 계층을 함께 낸다."""
    hit = np.asarray(centres[0, 0], float)
    # 바닥은 격자 경계보다 **가까이** 둔다. 멀면 `max(d, b)` 의 경계항이 답이 되어
    # 이 시험이 재려는 것(바닥이 그대로 막는다)이 아니라 경계를 재게 된다.
    floor_z = float(np.min(centres[..., 2])) - 0.15
    lower = np.min(centres.reshape(-1, 3), axis=0) - 0.4
    upper = np.max(centres.reshape(-1, 3), axis=0) + 0.4

    def floor(p):
        return p[:, 2] - floor_z

    def ball(p):
        return np.linalg.norm(p - hit[None, :], axis=1) - radius

    both = _layer(lambda p: np.minimum(floor(p), ball(p)), lower, upper, 0.05,
                  label_fn=lambda p: np.where(ball(p) <= floor(p), 0, -1))
    free = (_layer(floor, lower, upper, 0.05),) if target_free else ()
    return CuroboEsdfField((both,), outside_distance=0.5, target_free_layers=free,
                           target_label=TARGET_LABEL if target_free else None), hit, floor_z


def esdf_scene(sc, field, mask=None, margin: float = 0.01) -> SceneSnapshot:
    base = near_miss_scene(sc.linearizer, sc.reference, sc.q_now)
    return dataclasses.replace(base, esdf=field, esdf_margin=margin, target_free_mask=mask)


def test_only_the_masked_spheres_ask_the_target_free_layer(robot_scene):
    """권한 없는 구의 거리는 **한 값도** 바뀌지 않는다 (E1 재발 방지의 수치적 내용)."""
    sc, centres = robot_scene
    field, hit, floor_z = ball_field(centres)
    n = sc.linearizer.n_spheres

    mask = np.zeros(n, bool)
    mask[0] = True
    relaxed = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, mask))
    plain = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, None))

    assert np.array_equal(relaxed[:, 1:, 0], plain[:, 1:, 0]), "권한 없는 구가 움직였다"
    assert relaxed[0, 0, 0] > plain[0, 0, 0], "공 안에 있던 구가 풀려야 한다"
    # 그리고 바닥은 그대로 막는다: 값이 '공이 없는 필드' = 바닥까지의 거리에서 온다.
    expected = centres[0, 0, 2] - floor_z - sc.linearizer.query_radii[0] - 0.01
    assert relaxed[0, 0, 0] == pytest.approx(expected, abs=5e-3)


def test_no_mask_is_the_old_path_exactly(robot_scene):
    """`None` 마스크와 '전부 False' 마스크가 같은 답이어야 한다 — 기본이 갈리지 않는다."""
    sc, centres = robot_scene
    field, _, _ = ball_field(centres)
    none = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, None))
    empty = sc.linearizer.esdf_clearance(
        sc.reference, sc.q_now,
        esdf_scene(sc, field, np.zeros(sc.linearizer.n_spheres, bool)))
    assert np.array_equal(none, empty)


def test_exclude_all_moves_every_row_and_that_is_e1(robot_scene):
    """`exclude_all` 은 전신에서 공을 지운다. 그 사실이 시험에 남아 있어야 한다."""
    sc, centres = robot_scene
    field, _, _ = ball_field(centres)
    everyone = sc.linearizer.esdf_clearance(
        sc.reference, sc.q_now,
        esdf_scene(sc, field, np.ones(sc.linearizer.n_spheres, bool)))
    plain = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, None))
    assert (everyone[..., 0] >= plain[..., 0] - 1e-9).all(), "빼는 것이 여유를 줄일 수는 없다"
    assert (everyone[..., 0] > plain[..., 0] + 1e-9).any()


def test_a_mask_without_a_capable_field_is_an_error_not_a_silent_no_op(robot_scene):
    """정책을 켰다고 믿은 채 아무 일도 안 일어나는 것이 가장 나쁜 실패다."""
    sc, centres = robot_scene
    plain_field, _, _ = ball_field(centres, target_free=False)
    assert plain_field.has_target_free is False
    mask = np.ones(sc.linearizer.n_spheres, bool)
    # `CuroboEsdfField` 는 메서드를 늘 갖고 있으므로, 그 메서드가 아예 없는 필드로 시험한다.
    naked = types.SimpleNamespace(
        distance=plain_field.distance, grid=plain_field.grid, has_labels=False)
    with pytest.raises(ValueError, match="target_free_distance"):
        sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, naked, mask))


def test_a_capable_field_without_the_layer_is_a_no_op_not_an_error(robot_scene):
    """계층이 0 겹이면 답이 지금과 같다 — 메서드가 있는 필드는 죽지 않는다."""
    sc, centres = robot_scene
    field, _, _ = ball_field(centres, target_free=False)
    mask = np.ones(sc.linearizer.n_spheres, bool)
    with_mask = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, mask))
    plain = sc.linearizer.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, None))
    assert np.array_equal(with_mask, plain)


def test_the_held_objects_own_points_are_never_relaxed(robot_scene):
    """쥔 물체의 점은 권한 대상이 아니다 — 짧은 마스크는 `False` 로 채워진다 (fail closed)."""
    sc, centres = robot_scene
    lin = sc.linearizer
    held = np.random.default_rng(0).normal(scale=0.02, size=(9, 3))
    lin.set_attached(held, "ee_finger_l1")
    try:
        with_attached, _ = lin.sphere_states(sc.reference, sc.q_now)
        field, _, _ = ball_field(with_attached)
        n = lin.n_spheres
        assert with_attached.shape[1] > n
        mask = np.ones(n, bool)  # 로봇 구 전부, 쥔 물체 점은 안 적었다
        relaxed = lin.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, mask))
        plain = lin.esdf_clearance(sc.reference, sc.q_now, esdf_scene(sc, field, None))
        assert np.array_equal(relaxed[:, n:, 0], plain[:, n:, 0])
    finally:
        lin.set_attached(None, None)


def test_full_violation_sees_the_relaxed_rows(robot_scene):
    """보고되는 최악 값이 이 축을 빠뜨리면 완화가 기록에 안 나타난다."""
    sc, centres = robot_scene
    field, _, _ = ball_field(centres)
    mask = np.ones(sc.linearizer.n_spheres, bool)
    worst_plain = sc.linearizer.full_violation(sc.reference, sc.q_now, esdf_scene(sc, field, None))
    worst_free = sc.linearizer.full_violation(sc.reference, sc.q_now, esdf_scene(sc, field, mask))
    assert worst_free > worst_plain


# ================================================ 4. 배선 — AG3S → optimizer


def test_the_adapter_carries_the_mask_only_for_the_esdf_backends(monkeypatch, robot_scene):
    sc, centres = robot_scene
    field, _, _ = ball_field(centres)
    base = near_miss_scene(sc.linearizer, sc.reference, sc.q_now)
    monkeypatch.setattr(SceneSnapshot, "from_spec", classmethod(lambda cls, spec, radii: base))
    mask = np.zeros(sc.linearizer.n_spheres, bool)
    mask[2] = True
    cs = types.SimpleNamespace(constraints=object(), esdf=field, target_field_exclude=mask)

    prim = scene_from_constraint_set(
        cs, sc.linearizer.robot_radii,
        TrajOptConfig.from_dict({"collision": {"backend": "primitive"}}))
    assert prim.target_free_mask is None, "필드가 없는 backend 에는 실을 곳이 없다"

    for backend in ("esdf", "both"):
        scene = scene_from_constraint_set(
            cs, sc.linearizer.robot_radii,
            TrajOptConfig.from_dict({"collision": {"backend": backend}}))
        assert np.array_equal(scene.target_free_mask, mask)


def test_a_mask_of_the_wrong_length_stops_the_frame(monkeypatch, robot_scene):
    """길이가 다르면 AG3S 와 optimizer 가 다른 제약 모델을 보고 있다는 뜻이다."""
    sc, centres = robot_scene
    field, _, _ = ball_field(centres)
    base = near_miss_scene(sc.linearizer, sc.reference, sc.q_now)
    monkeypatch.setattr(SceneSnapshot, "from_spec", classmethod(lambda cls, spec, radii: base))
    cs = types.SimpleNamespace(constraints=object(), esdf=field,
                               target_field_exclude=np.ones(3, bool))
    with pytest.raises(ValueError, match="target_field_exclude"):
        scene_from_constraint_set(cs, sc.linearizer.robot_radii,
                                  TrajOptConfig.from_dict({"collision": {"backend": "esdf"}}))


def test_an_old_constraint_set_without_the_field_still_works(monkeypatch, robot_scene):
    """옛 기록을 다시 돌릴 때 이 축이 없는 객체가 온다. `getattr` 기본값이 그것을 받는다."""
    sc, centres = robot_scene
    field, _, _ = ball_field(centres)
    base = near_miss_scene(sc.linearizer, sc.reference, sc.q_now)
    monkeypatch.setattr(SceneSnapshot, "from_spec", classmethod(lambda cls, spec, radii: base))
    cs = types.SimpleNamespace(constraints=object(), esdf=field)
    scene = scene_from_constraint_set(cs, sc.linearizer.robot_radii,
                                      TrajOptConfig.from_dict({"collision": {"backend": "esdf"}}))
    assert scene.target_free_mask is None


# ---------------------------------------------------------------- AG3S 쪽 마스크 결정


def _adapter_setup(policy: str, *, has_target_free: bool = True):
    """두 링크 mock 팔 + 손끝만 권한이 있는 정책 + target.

    `tests/ag3s/test_to_contract.py` 의 `_grasp_setup` 과 같은 구성이고, 다른 것은
    `target_field_policy` 하나다 — 그 하나가 마스크를 만드는지 아닌지를 본다.
    """
    from benchmark.ag3s.constraints.clearance import ClearancePolicy
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import PointCloud, TargetGeometry
    from tests.ag3s.fixtures import FINGERTIP_LINK, PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig.from_dict({
        "contact": {"contact_links": {"right": [FINGERTIP_LINK]}},
        "constraint": {"horizon": 3, "target_field_policy": policy},
    })
    builder = ConstraintBuilder(
        arm, cfg.constraint, cfg.geometry,
        clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry))
    q = np.array([-0.60, -0.60])
    tip = arm.end_effector_numeric(q)
    cloud = PointCloud(tip + np.random.default_rng(0).normal(0.0, 0.01, (60, 3)))
    target = TargetGeometry(
        id=0, points=cloud.points, point_indices=np.arange(len(cloud.points)),
        centroid=cloud.points.mean(axis=0), bounding_geometry=fit_sphere(cloud.points),
        attention_score=0.9, confidence=0.9)
    field = types.SimpleNamespace(stats={"n_occupied": 10}, has_target_free=has_target_free)
    return arm, builder, q, target, field


def _build(builder, q, target, field, **kw):
    from benchmark.ag3s.constraints.to_adapter import build_constraint_set
    from benchmark.ag3s.types import ContactPolicyContext, Manipulator, Phase

    return build_constraint_set(
        builder, candidates=[], support_surfaces=[], target=target, robot_state=q,
        phase=Phase.GRASP, esdf=field, build_spec=False,
        contact_context=ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT]),
        **kw)


def test_relax_asks_for_no_mask_even_when_the_layer_is_there():
    """기본 정책은 계층이 있어도 쓰지 않는다 — 켜는 것은 명시적이어야 한다."""
    arm, builder, q, target, field = _adapter_setup("relax")
    cs = _build(builder, q, target, field)
    assert cs.target_field_exclude is None
    assert cs.target_field_policy == "relax"
    assert cs.manipulated_link_margin is not None, "마진 완화는 그대로 있어야 한다"


def test_exclude_authorized_names_exactly_the_authorized_spheres():
    """마진 완화가 쓰는 것과 **같은** 권한 집합이어야 한다 — 두 곳이면 갈라진다."""
    from tests.ag3s.fixtures import FINGERTIP_LINK

    arm, builder, q, target, field = _adapter_setup("exclude_authorized")
    cs = _build(builder, q, target, field)
    mask = cs.target_field_exclude
    assert mask is not None and mask.dtype == bool
    expected = np.asarray([n == FINGERTIP_LINK for n in builder.sphere_link_names], bool)
    assert np.array_equal(mask, expected)
    assert mask.sum() == 1, "이 mock 은 손끝 구가 하나다"
    # 그 구는 마진 완화도 함께 받는다. 두 기전이 같은 구를 고른다는 것이 요점이다.
    relaxed = cs.manipulated_link_margin < builder.clearance_policy.safety_margin
    assert np.array_equal(relaxed, expected)


def test_the_authorized_set_is_changed_from_config_not_from_code():
    """T8a 가 '어느 link 이 필요한가' 를 재고 있다. 답은 config 한 줄로 들어와야 한다."""
    from benchmark.ag3s.constraints.clearance import ClearancePolicy
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from tests.ag3s.fixtures import PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig.from_dict({
        "contact": {"contact_links": {"right": ["link_1"]}},  # 손끝이 아니라 전완
        "constraint": {"horizon": 3, "target_field_policy": "exclude_authorized"},
    })
    builder = ConstraintBuilder(
        arm, cfg.constraint, cfg.geometry,
        clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry))
    _, _, q, target, field = _adapter_setup("exclude_authorized")
    cs = _build(builder, q, target, field)
    expected = np.asarray([n == "link_1" for n in builder.sphere_link_names], bool)
    assert np.array_equal(cs.target_field_exclude, expected)
    assert expected.sum() >= 1


def test_no_authorized_link_means_no_mask():
    """권한 집합이 비면 `exclude_authorized` 는 아무 일도 하지 않는다 (fail closed)."""
    from benchmark.ag3s.constraints.clearance import ClearancePolicy
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from tests.ag3s.fixtures import PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig.from_dict({
        "contact": {"contact_links": {"right": ["nobody_link"]}},
        "constraint": {"horizon": 3, "target_field_policy": "exclude_authorized"},
    })
    builder = ConstraintBuilder(
        arm, cfg.constraint, cfg.geometry,
        clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry))
    _, _, q, target, field = _adapter_setup("exclude_authorized")
    assert _build(builder, q, target, field).target_field_exclude is None


def test_exclude_all_takes_every_sphere():
    arm, builder, q, target, field = _adapter_setup("exclude_all")
    cs = _build(builder, q, target, field)
    assert cs.target_field_exclude is not None
    assert cs.target_field_exclude.all()
    assert cs.target_field_exclude.shape == (builder.n_robot_spheres,)


def test_nothing_is_excluded_once_something_is_held():
    """쥔 뒤의 target 은 목적지(crate)다. 그것을 빼면 안으로 넣을 근거가 사라진다 (규칙 3)."""
    from benchmark.ag3s.types import AttachedCollisionGeometry, Primitive, PrimitiveType

    arm, builder, q, target, field = _adapter_setup("exclude_all")
    held = AttachedCollisionGeometry(
        parent_link=builder.sphere_link_names[-1], T_parent_object=np.eye(4),
        primitives=[Primitive(type=PrimitiveType.SPHERE, center=np.zeros(3),
                              dimensions=np.full(3, 0.03))],
        points=np.zeros((4, 3)))
    cs = _build(builder, q, target, field, attached=held)
    assert cs.target_field_exclude is None


def test_nothing_is_excluded_without_a_grounded_target():
    arm, builder, q, target, field = _adapter_setup("exclude_authorized")
    assert _build(builder, q, None, field).target_field_exclude is None


def test_nothing_is_excluded_when_the_field_has_no_such_layer():
    """계층이 없으면 마스크를 만들지 않는다 — optimizer 가 죽는 것보다 앞에서 막는다."""
    arm, builder, q, target, field = _adapter_setup("exclude_authorized",
                                                    has_target_free=False)
    cs = _build(builder, q, target, field)
    assert cs.target_field_exclude is None
    assert cs.target_field_policy == "exclude_authorized", "정책은 기록에 남아야 한다"


def test_the_builder_shouts_when_the_policy_is_not_the_default(caplog):
    """`serve_safe` 를 거치지 않는 호출자(실험 스크립트·테스트)도 반드시 듣는다."""
    with caplog.at_level(logging.WARNING):
        _adapter_setup("relax")
    assert caplog.text == "", "기본 실행은 아무 말도 하지 않는다"

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _adapter_setup("exclude_authorized")
    assert "AUTHORIZED LINKS" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _adapter_setup("exclude_all")
    assert "EVERY" in caplog.text
    assert "E1" in caplog.text, "E1 이 되살아난다는 것을 이름으로 불러야 한다"


# ================================================ 5. 서버 flag


def test_the_server_flag_defaults_to_relax_and_says_nothing(caplog):
    from benchmark.trajopt.serve_safe import (announce_diagnostic_scope, build_parser,
                                              resolve_target_field_policy,
                                              target_field_policy_choices)

    args = build_parser().parse_args(
        ["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml"])
    assert args.target_field_policy == "relax"
    assert resolve_target_field_policy(args.target_field_policy) == "relax"
    assert target_field_policy_choices() == ("relax", "exclude-authorized", "exclude-all")
    with caplog.at_level(logging.WARNING):
        announce_diagnostic_scope(links="arms", self_collision=True, link_scales={})
    assert caplog.text == ""


@pytest.mark.parametrize("flag,expected", [
    ("exclude-authorized", "exclude_authorized"), ("exclude-all", "exclude_all")])
def test_the_server_flag_reaches_the_config_name(flag, expected):
    from benchmark.trajopt.serve_safe import build_parser, resolve_target_field_policy

    args = build_parser().parse_args(
        ["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml",
         "--target-field-policy", flag])
    assert resolve_target_field_policy(args.target_field_policy) == expected
    cfg = AG3SConfig.from_dict({"constraint": {"target_field_policy": expected}})
    assert cfg.constraint.target_field_policy == expected


def test_an_unknown_policy_is_refused_by_the_parser_and_the_resolver():
    from benchmark.trajopt.serve_safe import build_parser, resolve_target_field_policy

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml",
             "--target-field-policy", "exclude_authorized"])  # 밑줄은 CLI 이름이 아니다
    with pytest.raises(ValueError, match="target-field-policy"):
        resolve_target_field_policy("exclude")


@pytest.mark.parametrize("policy,needle", [
    ("exclude_authorized", "AUTHORIZED LINKS"), ("exclude_all", "EVERY")])
def test_a_non_default_policy_is_announced_loudly(caplog, policy, needle):
    from benchmark.trajopt.serve_safe import announce_diagnostic_scope

    with caplog.at_level(logging.WARNING):
        announce_diagnostic_scope(links="arms", self_collision=True, link_scales={},
                                  target_field_policy=policy,
                                  authorized_links=["ee_finger_r1"])
    assert "!!!" in caplog.text and needle in caplog.text
    assert "TARGET" in caplog.text


def test_the_policy_cannot_be_combined_with_no_safe():
    from benchmark.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml",
                          "--target-field-policy", "exclude-all", "--no-safe"])
    with pytest.raises(SystemExit):
        reject_bad_flag_combinations(ap, args)


def test_the_default_flag_combination_is_still_accepted():
    from benchmark.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml"])
    reject_bad_flag_combinations(ap, args)  # 조용히 통과해야 한다


def test_a_single_tier_field_cannot_carry_the_policy_and_says_so():
    """미세 계층이 없으면 이 정책은 아무 일도 못 한다 — 조용히 0 겹으로 지나가지 않는다."""
    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder

    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "voxel_size": 0.02}}).esdf
    builder = CuroboFieldBuilder(cfg)
    assert builder.fine_voxel is None
    with pytest.raises(ValueError, match="fine_voxel_size"):
        builder.update([], target_points=APPLE[None, :],
                       target_free_points=APPLE[None, :])

    two_tier = AG3SConfig.from_dict(
        {"esdf": {"backend": "curobo", "voxel_size": 0.02, "fine_voxel_size": 0.005}}).esdf
    with pytest.raises(ValueError, match="target_points"):
        # 창의 중심이 target 무게중심이다 — target 이 없으면 놓을 자리가 없다.
        CuroboFieldBuilder(two_tier).update([], target_free_points=APPLE[None, :])


@pytest.mark.parametrize("extra,needle", [
    (["--esdf-backend", "legacy"], "curobo"),
    (["--esdf-backend", "curobo", "--fine-voxel", "0"], "미세 계층"),
])
def test_a_setup_that_cannot_carry_the_policy_is_refused_at_startup(extra, needle, capsys):
    """첫 프레임에서 죽으면 그때는 체크포인트 두 벌이 이미 GPU 에 올라간 뒤다."""
    from benchmark.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml",
                          "--target-field-policy", "exclude-authorized", *extra])
    with pytest.raises(SystemExit):
        reject_bad_flag_combinations(ap, args)
    assert needle in capsys.readouterr().err


def test_the_curobo_two_tier_default_carries_the_policy_fine():
    """권장 실행은 그대로 통과해야 한다 — 가드가 정상 조합을 막으면 쓸 수 없다."""
    from benchmark.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    args = ap.parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml",
                          "--target-field-policy", "exclude-authorized",
                          "--esdf-backend", "curobo"])
    reject_bad_flag_combinations(ap, args)
    assert args.fine_voxel == 0.005
