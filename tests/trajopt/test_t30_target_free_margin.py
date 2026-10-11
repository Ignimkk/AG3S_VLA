"""T30 F1 — target 없는 계층에 묻는 권한 행에는 조작 대상 마진(phase 여유)을 붙이지 않는다.

`_esdf_clearance` 는 행마다 `|d − d_object| ≤ voxel` 이면 "이 행의 최근접 장애물은 조작 대상" 으로 보고
`manipulated_link_margin` 을 쓴다. 그런데 권한 행의 `d` 가 **target 없는 계층**(사과를 뺀 층)에서 오면 그
검사는 뜻이 없다 — 그 층에는 사과가 없고, 참이 되는 것은 테이블이 우연히 사과와 같은 거리에 있을 때다.
T28 E3b chunk t=88 이 정확히 그것이었다: target-free +20.0 mm · 사과 점 20.5 mm · 손가락 구 r 2.6 mm,
client 가 `--safe-phase approach` 를 고정해 권한 마진 = 0.05 × 0.4 = 20 mm → TO −2.6 mm (기록 −2.25 mm),
판정 `collision` (`answer_excludes_target`) → HOLD.

이 파일이 고정하는 것:

1. **측정 사례 그대로** — 새 규칙 +17.4 mm, 옛 규칙(면제 없음) −2.6 mm.
2. **권한 없는 행 · `relax` 는 한 값도 안 바뀐다.** `exclude_all` 에서도 권한 없는 행은 옛 규칙.
3. **판정은 느슨해지지 않는다** — target 없는 계층에서의 진짜 위반(손가락 ↔ 테이블)은 여전히
   `collision` · `answer_excludes_target` 이고, 그 값은 이제 phase 여유 없이 잰 참 관통량이다.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot, scene_from_constraint_set
from tests.trajopt.test_verdict_reasons import (  # noqa: F401 — `rig` 는 fixture
    APPLE_R,
    TARGET,
    _classify,
    _Field,
    _surface,
    rig,
)

VOXEL = 0.005
#: E3b chunk t=88 (T30.task.md D1).
D_TABLE = 0.0200           # target-free 계층의 답 (테이블)
D_APPLE = 0.0205           # 가장 가까운 사과 점
R_FINGER = 0.0026          # 손가락 구 반지름
APPROACH_MARGIN = 0.05 * 0.4   # safety_margin × APPROACH margin_scale (권한 link)
FULL_MARGIN = 0.05             # 권한 없는 link 의 TARGET 마진
UNIT_APPLE_R = 0.033


# =============================================================================================
# 1. `_esdf_clearance` 한 줄 — 측정 사례
# =============================================================================================


class _PlaneAndApple:
    """z = −D_TABLE 의 테이블 평면 + 사과 구. target 없는 계층 = 테이블만. 정확한 해석적 거리."""

    has_labels = False

    def __init__(self, apple_centre):
        self.apple = np.asarray(apple_centre, np.float64)
        self.grid = types.SimpleNamespace(voxel_size=VOXEL)

    def _table(self, pts):
        return pts[:, 2] + D_TABLE

    def distance(self, pts):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        apple = np.linalg.norm(pts - self.apple, axis=1) - UNIT_APPLE_R
        return np.minimum(self._table(pts), apple)

    def target_free_distance(self, pts):
        return self._table(np.asarray(pts, np.float64).reshape(-1, 3))


def _unit_rig():
    """질의점 셋 (step 1 개):

    * 0 — 권한 손가락, 원점. 테이블 20.0 mm, 사과 20.5 mm (측정 사례).
    * 1 — 권한 없는 구, 사과 바로 위 30 mm (최근접이 **진짜** 사과).
    * 2 — 권한 없는 구, 원점 (0 과 같은 자리 — 우연의 일치를 권한 없는 행에서 본다).
    """
    apple = np.array([0.0, D_APPLE + UNIT_APPLE_R, 0.0])
    torso = apple + np.array([0.0, 0.0, UNIT_APPLE_R + 0.030])
    centres = np.stack([np.zeros(3), torso, np.zeros(3)])[None]           # (1, 3, 3)
    # 관측 점군: 구 표면 + 각 질의점의 정확한 최근접 표면점 (KD-tree 거리가 정확하도록).
    nearest = [apple + UNIT_APPLE_R * (p - apple) / np.linalg.norm(p - apple) for p in centres[0]]
    points = np.vstack([_surface(apple, UNIT_APPLE_R, 4000), nearest])
    lin = object.__new__(CollisionLinearizer)
    lin.query_radii = np.array([R_FINGER, 0.02, R_FINGER])
    lin.n_spheres = 3
    return lin, centres, _PlaneAndApple(apple), points


def _snapshot(field, points, *, mask, exempt=None, margin=0.0):
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((3, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.array([R_FINGER, 0.02, R_FINGER]), candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=float(margin), manipulated_points=points,
        manipulated_link_margin=np.array([APPROACH_MARGIN, FULL_MARGIN, FULL_MARGIN]),
        target_free_mask=None if mask is None else np.asarray(mask, bool),
        target_free_exempt=None if exempt is None else np.asarray(exempt, bool))


def test_the_measured_e3b_case_gets_esdf_margin_only():
    """E3b t=88: target-free 20.0 mm · 사과 20.5 mm · r 2.6 mm · APPROACH → +17.4 mm (옛 −2.6 mm)."""
    lin, centres, field, points = _unit_rig()
    mask = [True, False, False]                                        # exclude_authorized
    new = lin._esdf_clearance(centres, _snapshot(field, points, mask=mask))[0, :, 0]
    # 옛 규칙 = 면제 없음. 같은 씬에서 면제 행만 비우면 T30 전 값이 그대로 나온다.
    old = lin._esdf_clearance(
        centres, _snapshot(field, points, mask=mask, exempt=[False] * 3))[0, :, 0]

    assert new[0] == pytest.approx(D_TABLE - R_FINGER, abs=1e-9)           # +17.4 mm
    assert new[0] * 1000 == pytest.approx(17.4, abs=1e-6)
    assert old[0] == pytest.approx(D_TABLE - R_FINGER - APPROACH_MARGIN, abs=1e-9)   # −2.6 mm
    # 권한 없는 행은 한 값도 안 바뀐다: 1 = 진짜 사과 30 mm → full margin, 2 = 우연의 일치 → full margin.
    assert new[1] == old[1] == pytest.approx(0.030 - 0.02 - FULL_MARGIN, abs=1e-9)
    assert new[2] == old[2] == pytest.approx(D_TABLE - R_FINGER - FULL_MARGIN, abs=1e-9)


def test_relax_has_no_target_free_rows_and_is_unchanged():
    """`relax` (마스크 없음) 은 면제가 없다 — 권한 행은 사과가 든 답에 phase 여유를 그대로 쓴다."""
    lin, centres, field, points = _unit_rig()
    relax = lin._esdf_clearance(centres, _snapshot(field, points, mask=None))[0, :, 0]
    stray = lin._esdf_clearance(                               # 면제만 있고 마스크가 없으면 무시
        centres, _snapshot(field, points, mask=None, exempt=[True] * 3))[0, :, 0]
    assert relax[0] == pytest.approx(D_TABLE - R_FINGER - APPROACH_MARGIN, abs=1e-9)
    assert np.array_equal(relax, stray)
    assert CollisionLinearizer._target_free_exempt_rows(
        _snapshot(field, points, mask=None, exempt=[True] * 3), 3) is None


def test_the_destination_rule_still_applies_to_exempt_rows():
    """면제된 행의 여유는 `esdf_margin` 이다 — 전역 여유가 있으면 그것이 붙는다 (0 이 아니다)."""
    lin, centres, field, points = _unit_rig()
    new = lin._esdf_clearance(
        centres, _snapshot(field, points, mask=[True, False, False], margin=0.01))[0, :, 0]
    assert new[0] == pytest.approx(D_TABLE - R_FINGER - 0.01, abs=1e-9)


def test_padding_never_exempts_the_held_objects_own_points():
    lin, centres, field, points = _unit_rig()
    scene = _snapshot(field, points, mask=[True, False, False])
    rows = CollisionLinearizer._target_free_exempt_rows(scene, 5)
    assert rows.tolist() == [True, False, False, False, False]
    scene = _snapshot(field, points, mask=[True, True, True], exempt=[True, False])
    assert CollisionLinearizer._target_free_exempt_rows(scene, 4).tolist() == [
        True, False, False, False]


# =============================================================================================
# 2. `scene_from_constraint_set` — 어느 행이 면제인가
# =============================================================================================


def _constraint_set(field, points, *, policy, mask, margins):
    return types.SimpleNamespace(
        constraints=None, esdf=field, support_surfaces=(), attached=None,
        manipulated=types.SimpleNamespace(points=points, sphere_centers=None, sphere_radii=None),
        manipulated_link_margin=np.asarray(margins, np.float64),
        target_field_exclude=np.asarray(mask, bool), target_field_policy=policy,
        destination_label=None, destination_margin=None)


def _config(margin=0.0):
    return TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": margin}})


def test_exclude_authorized_exempts_exactly_its_mask():
    _, _, field, points = _unit_rig()
    cs = _constraint_set(field, points, policy="exclude_authorized", mask=[True, False, False],
                         margins=[APPROACH_MARGIN, FULL_MARGIN, FULL_MARGIN])
    scene = scene_from_constraint_set(cs, np.array([R_FINGER, 0.02, R_FINGER]), _config())
    assert scene.target_free_exempt is None                       # None = 마스크 그대로
    assert CollisionLinearizer._target_free_exempt_rows(scene, 3).tolist() == [True, False, False]


def test_exclude_all_exempts_only_the_authorized_rows():
    """`exclude_all` 은 전신을 마스크에 넣는다. 권한 없는 행은 T30 전 규칙 그대로여야 한다."""
    lin, centres, field, points = _unit_rig()
    radii = np.array([R_FINGER, 0.02, R_FINGER])
    cs = _constraint_set(field, points, policy="exclude_all", mask=[True, True, True],
                         margins=[APPROACH_MARGIN, FULL_MARGIN, FULL_MARGIN])
    scene = scene_from_constraint_set(cs, radii, _config())
    assert scene.target_free_exempt.tolist() == [True, False, False]

    import dataclasses

    new = lin._esdf_clearance(centres, scene)[0, :, 0]
    old = lin._esdf_clearance(
        centres, dataclasses.replace(scene, target_free_exempt=np.zeros(3, bool)))[0, :, 0]
    assert new[0] == pytest.approx(D_TABLE - R_FINGER, abs=1e-9)
    assert new[1] == old[1] and new[2] == old[2]
    # 2 는 권한 없고 target 없는 계층에 묻는다 — 우연의 일치에 full margin (T30 전과 같다).
    assert new[2] == pytest.approx(D_TABLE - R_FINGER - FULL_MARGIN, abs=1e-9)


def test_exclude_all_in_transit_cannot_tell_the_rows_apart_and_exempts_nothing():
    """TRANSIT (margin_scale 1.0) 에서는 권한 행과 아닌 행의 마진이 같다 — 가를 수 없으면 면제 없음."""
    _, _, field, points = _unit_rig()
    cs = _constraint_set(field, points, policy="exclude_all", mask=[True, True, True],
                         margins=[FULL_MARGIN] * 3)
    scene = scene_from_constraint_set(cs, np.array([R_FINGER, 0.02, R_FINGER]), _config())
    assert scene.target_free_exempt.tolist() == [False, False, False]


# =============================================================================================
# 3. 판정 — 실제 RB-Y1 구 모델 위에서 (`test_verdict_reasons.rig`)
# =============================================================================================


def _tip_scene(rig, gap, *, exempt=None):
    """손끝 구 아래에 테이블(구 r 0.5), 옆에 사과. 둘 다 손끝 **표면**에서 `gap` (사과는 +0.5 mm)."""
    r = float(rig.lin.query_radii[rig.tip])
    tip = rig.centres[rig.step, rig.tip]
    down = np.array([0.0, 0.0, -1.0])
    side = np.cross(rig.axis, down)
    side /= np.linalg.norm(side)
    table = tip + down * (r + 0.5 + gap)
    apple = tip + side * (r + APPLE_R + gap + 0.0005)
    field = _Field([(TARGET, apple, APPLE_R), ("table", table, 0.5)])
    authorized = np.asarray([n in rig.authorized for n in rig.names])
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((rig.lin.n_spheres, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=rig.lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=0.0, manipulated_points=_surface(apple, APPLE_R),
        manipulated_link_margin=np.where(authorized, APPROACH_MARGIN, FULL_MARGIN),
        target_free_mask=authorized,
        target_free_exempt=None if exempt is None else np.asarray(exempt, bool))


def _tip_row(cls, rig):
    return next((r for r in cls["rows"] if r["step"] == rig.step and r["query"] == rig.tip), None)


def test_the_verdict_no_longer_holds_on_the_phase_margin(rig):
    """테이블 10 mm · 사과 10.5 mm 떨어진 손끝: 옛 규칙은 −10 mm `collision` (E3b 의 HOLD), 새 규칙은 위반 없음."""
    old = _classify(rig, _tip_scene(rig, 0.010, exempt=np.zeros(rig.lin.n_spheres, bool)),
                    keep_rows=True, manipulated={"id": 0, "state": "visible"})
    row = _tip_row(old, rig)
    assert row is not None and row["class"] == "collision"
    assert row["why"] == "answer_excludes_target" and row["tier"] == "target_free"
    assert row["clearance_m"] == pytest.approx(0.010 - APPROACH_MARGIN, abs=1.5e-3)

    new = _classify(rig, _tip_scene(rig, 0.010), keep_rows=True,
                    manipulated={"id": 0, "state": "visible"})
    assert _tip_row(new, rig) is None
    clearance = rig.lin.esdf_clearance(rig.s.reference, rig.s.q_now, _tip_scene(rig, 0.010))
    assert clearance[rig.step, rig.tip, 0] == pytest.approx(0.010, abs=1e-9)


def test_a_real_violation_in_the_target_free_layer_is_still_a_collision(rig):
    """판정은 느슨해지지 않는다: 손끝이 테이블을 4 mm 파고들면 (사과 4.5 mm) 여전히 `collision`.
    값은 phase 여유 없이 잰 참 관통량 −4 mm 다 (옛 규칙이면 −24 mm)."""
    cls = _classify(rig, _tip_scene(rig, -0.004), keep_rows=True,
                    manipulated={"id": 0, "state": "visible"})
    row = _tip_row(cls, rig)
    assert row is not None
    assert row["class"] == "collision" and row["why"] == "answer_excludes_target"
    assert row["clearance_m"] == pytest.approx(-0.004, abs=1e-9)
    assert cls["n_allowed"] == 0 and cls["n_collision"] >= 1


# =============================================================================================
# 4. F1b — target-free 행은 방향도 target-free 층에서 (값과 같은 층)
# =============================================================================================
#
# `linearize` 는 ESDF 행의 방향을 `scene.esdf.gradient` (사과가 든 본 합성)에서 받았다. 값은
# `target_free_distance` 인데 방향은 사과 법선이면, 손끝이 사과 안 · 테이블 위에 있을 때 SQP 가
# 테이블 행을 풀려고 손끝을 **사과 밖으로** 민다 — F1 뒤에도 E3a 증상이 남는 길이다.

APPLE_R_F1B = 0.04


@pytest.fixture(scope="module")
def tip_rig():
    """`test_verdict_reasons.rig` 와 같은 손끝 (`link_right_arm_5` 의 가장 낮은 구, step 3)."""
    from tests.trajopt.fixtures import make_scenario

    s = make_scenario(horizon=6)
    lin = s.linearizer
    centres, jac = lin.sphere_states(s.reference, s.q_now)
    names = [lin._query_name(q) for q in range(lin.n_spheres)]
    step = 3
    tips = [q for q in range(lin.n_spheres) if names[q] == "link_right_arm_5"]
    tip = min(tips, key=lambda q: centres[step, q, 2])
    # 순수한 +z 밀기가 운동학적으로 끌고 가는 수평 방향 (J Jᵀ ẑ 의 수평 성분). 사과를 그와
    # **직교한** 쪽에 두면, 테이블 행만 푸는 올바른 step 은 사과까지의 거리를 바꾸지 않는다 —
    # 그래서 "사과에서 밀려났나" 가 방향 선택 하나로 갈린다.
    pull = jac[step, tip] @ jac[step, tip].T @ np.array([0.0, 0.0, 1.0])
    side = np.array([pull[1], -pull[0], 0.0])
    side /= np.linalg.norm(side)
    mask = np.asarray([n == "link_right_arm_5" for n in names])
    return types.SimpleNamespace(s=s, lin=lin, centres=centres, step=step, tip=tip, side=side,
                                 mask=mask, p=centres[step, tip].copy(),
                                 r=float(lin.query_radii[tip]))


def _table_and_apple(p, r, side):
    """손끝 구가 테이블을 4 mm, 사과를 15 mm 파고든다 — 본 합성의 최근접은 사과, target-free 는 테이블."""
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField
    from tests.trajopt.test_target_field_policy import _layer

    z_table = p[2] - r + 0.004
    apple = p + side * (APPLE_R_F1B + r - 0.015)

    def table(x):
        return x[:, 2] - z_table

    def both(x):
        return np.minimum(table(x), np.linalg.norm(x - apple, axis=1) - APPLE_R_F1B)

    coarse = _layer(both, p - 0.6, p + 0.6, 0.02)
    fine = _layer(both, p - 0.15, p + 0.15, 0.005)
    free = _layer(table, p - 0.15, p + 0.15, 0.005)
    return CuroboEsdfField((coarse, fine), outside_distance=0.5, target_free_layers=(free,)), apple


def _tip_after_solve(tip_rig, field):
    import dataclasses

    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import empty_scene

    s = tip_rig.s
    scene = dataclasses.replace(empty_scene(tip_rig.lin), esdf=field, esdf_margin=0.0,
                                target_free_mask=tip_rig.mask)
    opt = TrajectoryOptimizer(s.robot, s.layout, s.limits, s.config)
    result = opt.solve(s.reference, s.q_now, scene)
    after = opt.linearizer.sphere_states(result.trajectory, s.q_now)[0][tip_rig.step, tip_rig.tip]
    return after - tip_rig.p, opt.linearizer.last_target_free_tiers


def test_the_target_free_gradient_points_away_from_the_table_not_the_apple(tip_rig):
    field, apple = _table_and_apple(tip_rig.p, tip_rig.r, tip_rig.side)
    p = tip_rig.p[None]
    grad, tier = field.target_free_gradient(p, return_tier=True)
    assert tier.tolist() == ["target_free"]
    assert grad[0] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)            # 테이블 법선
    main = field.gradient(p)[0]
    assert main @ (-tip_rig.side) > 0.5                                     # 본 합성 = 사과 법선 (옛 방향)
    assert field.target_free_distance(p)[0] == pytest.approx(tip_rig.r - 0.004, abs=1e-6)


def test_the_linearized_row_uses_the_layer_that_gave_the_value(tip_rig):
    import dataclasses

    from tests.trajopt.fixtures import empty_scene

    field, _ = _table_and_apple(tip_rig.p, tip_rig.r, tip_rig.side)
    lin, s = tip_rig.lin, tip_rig.s
    scene = dataclasses.replace(empty_scene(lin), esdf=field, esdf_margin=0.0,
                                target_free_mask=tip_rig.mask)
    rows = lin.linearize(s.reference, s.q_now, scene, s.config)
    _, jac = lin.sphere_states(s.reference, s.q_now)
    hit = np.argwhere(rows.used & (rows.sphere == tip_rig.tip))
    hit = [(k, j) for k, j in hit if k == tip_rig.step]
    assert hit, "사전 조건: 손끝 행이 선택됐다"
    k, j = hit[0]
    expected = np.array([0.0, 0.0, 1.0]) @ jac[tip_rig.step, tip_rig.tip]
    assert rows.gradient[k, j] == pytest.approx(expected, abs=1e-6)
    assert lin.last_target_free_tiers.get("target_free", 0) >= 1          # 어느 층이 답했나의 기록


def test_the_qp_step_lifts_off_the_table_without_pushing_off_the_apple(tip_rig):
    """새 규칙: 손끝은 테이블에서 올라가고 사과까지의 거리는 (≤ 1 mm) 그대로.
    옛 규칙(본 합성 기울기): 같은 씬에서 사과 밖으로 ≥ 5 mm 밀린다 — E3a 증상."""
    field, _ = _table_and_apple(tip_rig.p, tip_rig.r, tip_rig.side)
    step_new, tiers = _tip_after_solve(tip_rig, field)
    away = -tip_rig.side                                                    # 사과에서 멀어지는 방향
    assert step_new[2] > 0.002                                              # 테이블에서 올라간다
    assert abs(step_new @ away) < 0.001
    assert tiers.get("target_free", 0) >= 1

    old_field, _ = _table_and_apple(tip_rig.p, tip_rig.r, tip_rig.side)
    old_field.target_free_gradient = (                                     # F1b 전: 본 합성의 기울기
        lambda pts, return_tier=False: (old_field.gradient(pts), np.full(len(pts), "fine"))
        if return_tier else old_field.gradient(pts))
    step_old, _ = _tip_after_solve(tip_rig, old_field)
    assert step_old @ away > 0.005


def test_outside_the_window_and_at_its_edge_value_and_direction_share_a_layer(tip_rig):
    """창 밖은 본 합성이 답하므로 본 합성의 기울기, 창 경계(`window_edge`)는 경계의 기울기.
    tier 는 `answer_tier` 와 언제나 같다."""
    field, _ = _table_and_apple(tip_rig.p, tip_rig.r, tip_rig.side)
    rng = np.random.default_rng(0)
    pts = tip_rig.p + rng.uniform(-0.3, 0.3, (400, 3))
    grad, tier = field.target_free_gradient(pts, return_tier=True)
    assert tier.tolist() == field.answer_tier(pts, target_free=True).tolist()
    outside = ~np.isin(tier, ("target_free", "window_edge", "static"))
    assert outside.any()
    assert np.allclose(grad[outside], field.gradient(pts[outside]))
    assert (tier == "target_free").any()
    # 창 경계: 사과(본 합성에만 있다)가 가깝고 창 면도 가까운 점 — 경계 하한 `b` 가 답한다.
    # 기울기는 가장 가까운 창 면의 안쪽 축 법선 (크기 1, 한 축).
    near = tip_rig.p + rng.uniform(-0.149, 0.149, (4000, 3))
    grad, tier = field.target_free_gradient(near, return_tier=True)
    edge = tier == "window_edge"
    assert edge.any(), "사전 조건: 창 경계가 답하는 점이 있다"
    assert np.allclose(np.abs(grad[edge]).sum(axis=1), 1.0)
    assert np.allclose(np.abs(grad[edge]).max(axis=1), 1.0)
    assert np.allclose(grad[edge], field._boundary_gradient(field.target_free_layers[0].grid,
                                                            near[edge]))


def test_a_mask_without_a_matching_gradient_is_an_error_not_a_silent_fallback(tip_rig):
    import dataclasses

    from tests.trajopt.fixtures import empty_scene

    field, _ = _table_and_apple(tip_rig.p, tip_rig.r, tip_rig.side)
    naked = types.SimpleNamespace(distance=field.distance, gradient=field.gradient,
                                  target_free_distance=field.target_free_distance,
                                  grid=field.grid, has_labels=False)
    lin, s = tip_rig.lin, tip_rig.s
    scene = dataclasses.replace(empty_scene(lin), esdf=naked, esdf_margin=0.0,
                                target_free_mask=tip_rig.mask)
    with pytest.raises(ValueError, match="target_free_gradient"):
        lin.linearize(s.reference, s.q_now, scene, s.config)
