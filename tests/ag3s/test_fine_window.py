"""T21 — fine / target-free 창이 **손의 실행 경로**를 덮는다.

## 무엇이었나

T24 가 T17 접근 구간(t=72–120)의 손끝 변위 중앙값 51 mm · 최대 195 mm 의 원인을 특정했다: QP 가 받은
행은 전부 `ee_finger_l1/l2` 의 **coarse 20 mm** 행이었다. 창이 target(→ crate) 무게중심에 놓여 손가락이
그 밖에 있었다 (T14 seq 12–21: 손가락 질의점이 창 안에 든 chunk 0/8).

## 이 파일이 박는 것

1. **배치 규칙 하나** (`place_fine_window`): 손 swept AABB ∪ manipulated AABB 가 들어가면 그것, 안 들어가면
   지금 손 → 실행 구간 → 물체 순으로 담고 넘친 것을 기록한다. 손 정보가 없으면 물체 중심 (옛 규칙).
2. **어느 계층이 답했나** (`CuroboEsdfField.answer_tier`): `distance`·`target_free_distance` 와 같은 합성을
   따라가며 이긴 쪽에 이름을 붙인다. 값과 이름이 어긋나지 않는지 성질로 확인한다.
3. **pipeline 배선**: 권한 link 만 창을 끌고, 실행 구간이 오면 그것도 들어가고, 제외 기하(`_target_ball`)는
   손과 무관하게 manipulated 만 정한다 (창 확대 ≠ 제외 확대). 창 밖 권한 질의는 노트로 남는다.
4. **guard 반경 규칙** (사용자 판정 2026-09-28): 설정값이 0 이 아니면 그 값, 0 이면 manipulated 반지름 +
   미세 복셀 1 칸.

cuRobo·torch 없이 numpy 로 돈다 (`.venv-ag3s`). 실제 cuRobo 적분은 verifier 의 몫이다.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, EsdfConfig
from AG3S.ag3s.fields.curobo_builder import (
    FineWindowRequest,
    WINDOW_BASES,
    WINDOW_PLACEMENTS,
    place_fine_window,
)
from AG3S.ag3s.fields.curobo_field import (
    ANSWER_TIERS,
    RESOLVED_TIERS,
    CuroboEsdfField,
    count_tiers,
    layer_from_arrays,
)
from AG3S.ag3s.fields.esdf import StaticBox
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.types import ContactPolicyContext, SupportSurface

SHAPE = (128, 128, 128)
FINE = 0.005
L = (SHAPE[0] - 1) * FINE          # 보간 범위 0.635 m


# ============================================================================ 1. 배치 규칙


def _hand(centres, radius=0.02):
    """`(T, S, 3)` 을 그대로 쓰는 request 재료."""
    c = np.asarray(centres, np.float64)
    if c.ndim == 2:
        c = c[None]
    return c, np.full(c.shape[1], radius)


def _box_inside(win, lo, hi):
    return bool(np.all(lo >= win.lower - 1e-9) and np.all(hi <= win.upper + 1e-9))


def test_when_hand_and_object_fit_the_window_holds_both_and_is_centred_on_their_union():
    c, r = _hand([[[0.50, 0.30, 0.95], [0.52, 0.30, 0.95]],
                  [[0.53, 0.31, 0.92], [0.55, 0.31, 0.92]]])
    apple = np.random.default_rng(0).normal([0.56, 0.32, 0.87], 0.02, (80, 3))
    req = FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.06,
                            manipulated_points=apple, manipulated_pad=0.01)
    win = place_fine_window(req, shape=SHAPE, voxel_size=FINE)

    assert win.basis == "hand_swept" and win.overflow is None
    for lo, hi in req.step_boxes():
        assert _box_inside(win, lo, hi)
    assert win.manipulated_covered is True
    assert win.n_steps_covered == win.n_steps == 2
    np.testing.assert_allclose(win.centre, 0.5 * (win.required_lower + win.required_upper))
    np.testing.assert_allclose(win.upper - win.lower, L)


def test_the_padded_box_is_centre_plus_radius_plus_pad_per_sphere():
    c, _ = _hand([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]])
    req = FineWindowRequest(hand_centres=c, hand_radii=np.array([0.01, 0.03]), hand_pad=0.05)
    (lo, hi), = req.step_boxes()
    np.testing.assert_allclose(lo, [-0.06, -0.08, 0.92])
    np.testing.assert_allclose(hi, [0.18, 0.08, 1.08])


def test_the_window_uses_the_voxel_centre_convention_of_the_fine_tier():
    """`lower` = 첫 복셀 중심 = `CuroboFieldBuilder._origin_of` (중심 − n·v/2 + v/2)."""
    c, r = _hand([[[0.4, 0.1, 0.9]]])
    win = place_fine_window(FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.05),
                            shape=SHAPE, voxel_size=FINE)
    first_voxel_centre = win.centre - 0.5 * np.asarray(SHAPE) * FINE + 0.5 * FINE
    np.testing.assert_allclose(win.lower, first_voxel_centre)
    # 같은 경계를 `CuroboEsdfField._covers` 가 쓴다.
    layer = layer_from_arrays(np.zeros(SHAPE, np.float32), win.lower, FINE)
    probe = np.array([win.lower, win.upper, win.upper + 1e-6])
    np.testing.assert_array_equal(CuroboEsdfField._covers(layer.grid, probe), [True, True, False])


def test_without_a_hand_the_window_is_the_old_rule_the_object_centroid():
    apple = np.random.default_rng(1).normal([0.56, 0.32, 0.87], 0.02, (50, 3))
    win = place_fine_window(FineWindowRequest(manipulated_points=apple), shape=SHAPE,
                            voxel_size=FINE)
    assert win.basis == "manipulated"
    np.testing.assert_array_equal(win.centre, apple.mean(axis=0))


def test_target_centroid_placement_ignores_the_hand_entirely():
    apple = np.random.default_rng(2).normal([0.56, 0.32, 0.87], 0.02, (50, 3))
    c, r = _hand([[[1.5, -0.5, 1.2]]])
    win = place_fine_window(FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.06,
                                              manipulated_points=apple,
                                              placement="target_centroid"),
                            shape=SHAPE, voxel_size=FINE)
    assert win.basis == "manipulated" and win.placement == "target_centroid"
    np.testing.assert_array_equal(win.centre, apple.mean(axis=0))
    assert win.n_steps_covered == 0     # 기록은 여전히 손이 창 밖임을 말한다


def test_nothing_to_place_means_no_fine_window():
    assert place_fine_window(FineWindowRequest(), shape=SHAPE, voxel_size=FINE) is None


def test_hand_only_fits_and_says_so():
    c, r = _hand([[[0.5, 0.3, 0.95]]])
    win = place_fine_window(FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.06),
                            shape=SHAPE, voxel_size=FINE)
    assert win.basis == "hand_swept" and win.manipulated_covered is None


def test_a_far_object_overflows_and_the_hand_wins_but_the_window_leans_toward_the_object():
    """T14 의 crate 처럼 물체가 멀다: 손은 통째로, 물체는 되는 만큼 (창이 물체 쪽으로 기운다)."""
    c, r = _hand([[[0.50, 0.30, 0.95]], [[0.52, 0.30, 0.95]]])
    crate = np.array([[0.95, 0.30, 0.95], [1.25, 0.35, 1.00]])
    req = FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.06,
                            manipulated_points=crate, manipulated_pad=0.01)
    win = place_fine_window(req, shape=SHAPE, voxel_size=FINE)

    assert win.basis == "overflow" and win.overflow == "manipulated"
    assert win.n_steps_covered == 2 and win.manipulated_covered is False
    hand_lo = min(lo[0] for lo, _ in req.step_boxes())
    # 물체 쪽(+x)으로 최대한 밀었다: 창의 −x 경계가 손의 −x 경계에 붙는다.
    assert win.lower[0] == pytest.approx(hand_lo)
    assert win.upper[0] > crate[0, 0]                     # 물체의 일부는 들어온다


def test_an_unreachable_object_leaves_the_window_centred_on_the_hand():
    """물체를 한 조각도 담을 수 없으면 손 가운데 — 기울여 봐야 여유만 잃는다."""
    c, r = _hand([[[0.50, 0.30, 0.95]]])
    req = FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.06,
                            manipulated_points=np.array([[1.8, 0.3, 0.95], [1.9, 0.3, 1.0]]))
    win = place_fine_window(req, shape=SHAPE, voxel_size=FINE)
    assert win.overflow == "manipulated"
    np.testing.assert_allclose(win.centre[0], 0.50)


def test_a_long_path_keeps_the_current_hand_and_the_earliest_steps():
    """우선순위는 지금 손 → 실행 구간 (step 순). 넘친 step 수가 기록된다."""
    xs = 0.40 + 0.12 * np.arange(8)                      # 8 step 에 0.84 m
    c = np.stack([np.array([[x, 0.3, 0.95]]) for x in xs])
    req = FineWindowRequest(hand_centres=c, hand_radii=np.array([0.02]), hand_pad=0.06)
    win = place_fine_window(req, shape=SHAPE, voxel_size=FINE)

    boxes = req.step_boxes()
    assert win.basis == "overflow" and win.overflow == "hand_path"
    assert _box_inside(win, *boxes[0])                    # 지금 자세는 반드시
    covered = [_box_inside(win, lo, hi) for lo, hi in boxes]
    k = covered.index(False)
    assert all(covered[:k]) and not any(covered[k:])      # 앞에서부터 연속
    assert win.n_steps_covered == k and 1 <= k < 8


def test_a_hand_bigger_than_the_window_is_centred_on_itself_and_recorded():
    c, r = _hand([[[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]])
    win = place_fine_window(FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.3),
                            shape=SHAPE, voxel_size=FINE)
    assert win.basis == "overflow" and win.overflow == "hand_now"
    np.testing.assert_allclose(win.centre, [0.25, 0.0, 1.0])


def test_summary_is_json_friendly_and_names_the_basis():
    import json

    c, r = _hand([[[0.5, 0.3, 0.95]]])
    win = place_fine_window(FineWindowRequest(hand_centres=c, hand_radii=r, hand_pad=0.06,
                                              hand_source="q_now", links=("ee_finger_l1",)),
                            shape=SHAPE, voxel_size=FINE)
    s = win.summary()
    json.dumps(s)
    assert s["basis"] in WINDOW_BASES and s["hand_source"] == "q_now"
    assert s["links"] == ["ee_finger_l1"] and s["extent_m"] == pytest.approx([L] * 3)


def test_request_validation():
    with pytest.raises(ValueError):
        FineWindowRequest(placement="attention")
    with pytest.raises(ValueError):
        FineWindowRequest(hand_centres=np.zeros((1, 2, 3)), hand_radii=np.zeros(3))
    with pytest.raises(ValueError):
        FineWindowRequest(hand_pad=-0.01)
    assert FineWindowRequest(hand_centres=np.zeros((0, 2, 3))).has_hand is False


# ============================================================================ 2. 어느 계층이 답했나

APPLE = np.array([0.55, 0.30, 0.87])
APPLE_R = 0.04
TABLE_Z = 0.83


def _grid_points(origin, shape, vs):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def _truth(p, *, apple=True):
    d = p[..., 2] - TABLE_Z
    if apple:
        d = np.minimum(d, np.linalg.norm(p - APPLE, axis=-1) - APPLE_R)
    return d


def _field(fine_centre, *, fine_shape=(40, 40, 40), target_free=True, static=()):
    """거친 20 mm (작업공간) + 미세 5 mm (창) + target 없는 미세 (같은 창, 사과 없음).

    거친 계층은 +4 mm 과대평가한다 — 문서가 말하는 이산화 편향의 방향이다 (그래서 창 안에서는 미세가 이긴다).
    """
    c_origin, c_shape, c_vs = np.array([-0.2, -0.6, 0.4]), (80, 70, 50), 0.02
    coarse = layer_from_arrays(_truth(_grid_points(c_origin, c_shape, c_vs)) + 0.004,
                               c_origin, c_vs)
    f_origin = np.asarray(fine_centre) - 0.5 * np.asarray(fine_shape) * FINE + 0.5 * FINE
    pts = _grid_points(f_origin, fine_shape, FINE)
    fine = layer_from_arrays(_truth(pts), f_origin, FINE)
    free = (layer_from_arrays(_truth(pts, apple=False), f_origin, FINE),) if target_free else ()
    field = CuroboEsdfField((coarse, fine), outside_distance=0.5, target_free_layers=free,
                            stats={})
    field.static_shapes = tuple(static)
    return field


def test_the_tiers_are_named_by_what_answered():
    field = _field(APPLE + [0.0, 0.0, 0.03])
    fine_grid = field.layers[1].grid
    in_fine = APPLE + [0.0, 0.0, 0.06]                  # 사과 위 2 cm, 창 안
    in_coarse = np.array([0.2, -0.3, 1.0])              # 창 밖, 작업공간 안
    outside = np.array([3.0, 3.0, 3.0])                 # 작업공간 밖
    tiers = field.answer_tier(np.stack([in_fine, in_coarse, outside]))
    assert list(tiers) == ["fine", "coarse", "outside"]
    assert CuroboEsdfField._covers(fine_grid, in_fine[None])[0]


def test_target_free_rows_are_named_target_free_and_the_value_is_the_free_layers():
    field = _field(APPLE + [0.0, 0.0, 0.03], fine_shape=(60, 60, 60))
    p = APPLE + [0.0, 0.0, 0.045]                       # 사과 표면 5 mm 위, 창 경계는 130 mm
    tier = field.answer_tier(p[None], target_free=True)[0]
    assert tier == "target_free"
    assert field.target_free_distance(p[None])[0] == pytest.approx(
        field.target_free_layers[0].distance(p[None])[0])
    # 같은 점을 권한 없이 물으면 사과가 든 미세 계층이다.
    assert field.answer_tier(p[None], target_free=False)[0] == "fine"


def test_a_window_bound_answer_is_window_edge_not_target_free():
    """창 경계 거리 `b` 가 답하면 하한이다 — 안전 쪽이지만 풀린 답이 아니다."""
    # 사과 옆 10 mm (d), 창 +x 경계까지 25 mm (b), 테이블까지 40 mm (f): 답 = min(f, max(d, b)) = b.
    p = APPLE + [APPLE_R + 0.01, 0.0, 0.0]
    centre = p + [0.025 - 0.0975, 0.0, 0.0]
    field = _field(centre, fine_shape=(40, 40, 40))
    grid = field.target_free_layers[0].grid
    b = CuroboEsdfField._boundary_distance(grid, p[None])[0]
    assert b == pytest.approx(0.025)
    assert field.answer_tier(p[None], target_free=True)[0] == "window_edge"
    assert field.target_free_distance(p[None])[0] == pytest.approx(b)


def test_static_geometry_that_wins_is_named_static():
    box = StaticBox(center=np.array([0.2, -0.3, 1.02]), half_extents=np.array([0.05, 0.05, 0.01]))
    field = _field(APPLE, static=(box,))
    p = np.array([0.2, -0.3, 1.04])
    assert field.answer_tier(p[None])[0] == "static"


def test_tier_names_agree_with_the_values_everywhere():
    """성질: 이름이 `target_free` 면 값이 free 층의 값, `window_edge` 면 경계 거리, 나머지는 `distance`."""
    field = _field(APPLE + [0.05, 0.0, 0.03])
    rng = np.random.default_rng(3)
    pts = APPLE + rng.uniform(-0.25, 0.25, (4000, 3))
    flags = rng.random(4000) < 0.5
    tiers = field.answer_tier(pts, target_free=flags)
    d = field.distance(pts)
    tf = field.target_free_distance(pts)
    free_layer = field.target_free_layers[0]
    grid = free_layer.grid
    answered = np.where(flags, tf, d)
    for i in np.flatnonzero(tiers == "target_free"):
        assert answered[i] == pytest.approx(free_layer.distance(pts[i:i + 1])[0])
    for i in np.flatnonzero(tiers == "window_edge"):
        assert answered[i] == pytest.approx(
            CuroboEsdfField._boundary_distance(grid, pts[i:i + 1])[0])
    plain = np.isin(tiers, ("coarse", "fine", "outside"))
    np.testing.assert_allclose(answered[plain], d[plain])
    assert set(np.unique(tiers)) <= set(ANSWER_TIERS)
    assert {"target_free", "fine", "coarse"} <= set(np.unique(tiers))


def test_a_window_edge_row_is_settled_only_when_its_bound_already_clears():
    """`window_edge` 의 값은 하한이다: 하한만으로 여유 ≥ 0 이면 참 거리도 그렇다 → 풀린 행."""
    from AG3S.ag3s.fields.curobo_field import unresolved_rows

    tiers = np.array(["window_edge", "window_edge", "coarse", "outside", "fine", "target_free"])
    clearance = np.array([0.01, -0.002, 0.3, 0.3, -0.01, -0.01])
    unresolved, bounded = unresolved_rows(tiers, clearance)
    np.testing.assert_array_equal(bounded, [True, False, False, False, False, False])
    # coarse·outside 는 여유가 커도 미해결 — 창 밖을 자유 공간으로 읽지 않는다.
    np.testing.assert_array_equal(unresolved, [False, True, True, True, False, False])


def test_count_tiers_always_has_every_key():
    c = count_tiers(np.array(["fine", "fine", "coarse"]))
    assert set(c) == set(ANSWER_TIERS) and c["fine"] == 2 and c["outside"] == 0
    assert set(RESOLVED_TIERS) == {"fine", "target_free", "static"}


# ============================================================================ 3. pipeline 배선


class _HandRobot:
    """RB-Y1 이름을 가진 장난감: 몸통 구 1 · 왼손 3 · 오른손 3. `q` 는 왼손·오른손의 x 이동."""

    sphere_link_names = ("torso", "ee_left", "ee_finger_l1", "ee_finger_l2",
                         "ee_right", "ee_finger_r1", "ee_finger_r2")
    nq = 2
    n_spheres = 7

    BASE = np.array([[0.0, 0.0, 1.0],
                     [0.45, 0.30, 0.95], [0.48, 0.32, 0.93], [0.48, 0.28, 0.93],
                     [0.45, -0.30, 0.95], [0.48, -0.28, 0.93], [0.48, -0.32, 0.93]])
    RADII = np.array([0.15, 0.03, 0.012, 0.012, 0.03, 0.012, 0.012])
    #: 구마다 x 를 미는 관절 (-1 = 몸통, 고정).
    JOINT = (-1, 0, 0, 0, 1, 1, 1)

    def sphere_centers_numeric(self, q):
        q = np.asarray(q, np.float64).reshape(-1)
        centres = self.BASE.copy()
        for i, j in enumerate(self.JOINT):
            if j >= 0:
                centres[i, 0] += q[j]
        return centres, self.RADII.copy()

    def sphere_centers_symbolic(self, q):
        import casadi as ca

        out = []
        for i, j in enumerate(self.JOINT):
            c = ca.DM(self.BASE[i]).reshape((3, 1))
            if j >= 0:
                c = c + ca.vertcat(q[j], 0, 0)
            out.append((c, float(self.RADII[i])))
        return out


class _FakeBuilder:
    """`CuroboFieldBuilder.update` 의 자리. 받은 인자를 적고, 진짜 배치 함수로 창을 놓아 필드를 낸다."""

    def __init__(self, fine_shape=SHAPE):
        self.calls = []
        self.fine_shape = fine_shape

    def update(self, cameras, **kw):
        self.calls.append(kw)
        req = kw.get("fine_window") or FineWindowRequest(
            manipulated_points=kw.get("target_points"), placement="target_centroid")
        win = place_fine_window(req, shape=self.fine_shape, voxel_size=FINE)
        field = _field(win.centre, fine_shape=self.fine_shape,
                       target_free=kw.get("target_free_points") is not None)
        field.stats["window"] = win.summary()
        return field


def _apple_geometry(n=120, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    pts = APPLE + APPLE_R * v / np.linalg.norm(v, axis=1, keepdims=True)
    return types.SimpleNamespace(points=pts[pts[:, 2] > TABLE_Z + 0.002])


TABLE = SupportSurface(id=0, normal=np.array([0.0, 0.0, 1.0]), offset=TABLE_Z,
                       point_indices=np.arange(3), point_count=3, rms_error=0.001)


def _ag3s(monkeypatch, **esdf):
    from AG3S.ag3s.fields import esdf as esdf_mod

    monkeypatch.setattr(esdf_mod.EsdfBuilder, "instances_created", 0)
    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "esdf": {"backend": "curobo", "voxel_size": 0.02, "fine_voxel_size": FINE, **esdf},
        "constraint": {"target_field_policy": "exclude_authorized"},
    })
    ag3s = AG3S(cfg, robot_model=_HandRobot())
    ag3s._esdf_builder = _FakeBuilder()
    return ag3s


LEFT = ContactPolicyContext.make("approach", ["left"])


def _build(ag3s, **kw):
    kw.setdefault("robot_state", np.zeros(2))
    kw.setdefault("contact_context", LEFT)
    return ag3s._build_esdf([object()], _apple_geometry(), support_surfaces=[TABLE], **kw)


def test_the_window_follows_the_authorized_hand_only(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    field, _ = _build(ag3s)
    req = ag3s._esdf_builder.calls[-1]["fine_window"]
    assert req.links == ("ee_left", "ee_finger_l1", "ee_finger_l2")
    centres, _ = _HandRobot().sphere_centers_numeric(np.zeros(2))
    np.testing.assert_array_equal(req.hand_centres[0], centres[1:4])
    assert req.hand_source == "q_now" and req.hand_centres.shape == (1, 3, 3)
    assert field.stats["window"]["basis"] == "hand_swept"


def test_the_execution_path_is_part_of_the_swept_box(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    path = np.stack([[0.02 * k, 0.0] for k in range(1, 9)])
    field, _ = _build(ag3s, execution_path=path)
    req = ag3s._esdf_builder.calls[-1]["fine_window"]
    assert req.hand_source == "q_now+path" and req.hand_centres.shape[0] == 9
    np.testing.assert_allclose(req.hand_centres[-1, 0], [0.45 + 0.16, 0.30, 0.95])
    w = field.stats["window"]
    assert w["n_steps"] == 9 and w["n_steps_covered"] == 9


def test_a_path_of_the_wrong_width_is_an_error_not_a_silent_drop(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    with pytest.raises(ValueError, match="nq=2"):
        _build(ag3s, execution_path=np.zeros((8, 3)))


def test_the_pad_is_margin_plus_interpolation_voxels_from_config(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    _build(ag3s)
    req = ag3s._esdf_builder.calls[-1]["fine_window"]
    assert req.hand_pad == pytest.approx(ag3s.config.geometry.safety_margin + 2 * FINE)
    assert req.manipulated_pad == pytest.approx(2 * FINE)

    ag3s = _ag3s(monkeypatch, fine_window_margin=0.02, fine_window_pad_voxels=3)
    _build(ag3s)
    req = ag3s._esdf_builder.calls[-1]["fine_window"]
    assert req.hand_pad == pytest.approx(0.02 + 3 * FINE)


def test_nobody_authorized_falls_back_to_the_object_and_says_so(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    field, _ = _build(ag3s, contact_context=ContactPolicyContext.make("approach", []))
    req = ag3s._esdf_builder.calls[-1]["fine_window"]
    assert req.hand_source == "none" and not req.has_hand
    assert field.stats["window"]["basis"] == "manipulated"


def test_the_old_rule_is_one_config_key_away(monkeypatch):
    ag3s = _ag3s(monkeypatch, fine_window_placement="target_centroid")
    field, _ = _build(ag3s)
    req = ag3s._esdf_builder.calls[-1]["fine_window"]
    assert req.placement == "target_centroid"
    np.testing.assert_array_equal(field.stats["window"]["centre_m"],
                                  np.round(_apple_geometry().points.mean(axis=0), 6))


def test_the_exclusion_ball_does_not_grow_with_the_window(monkeypatch):
    """지침 §5.3-3: 창을 넓히는 것과 제외 영역을 넓히는 것은 다른 결정이다."""
    a = _ag3s(monkeypatch)
    _build(a)
    b = _ag3s(monkeypatch)
    _build(b, execution_path=np.stack([[0.03 * k, 0.0] for k in range(1, 9)]))
    ball_a = a._esdf_builder.calls[-1]["target_free_ball"]
    ball_b = b._esdf_builder.calls[-1]["target_free_ball"]
    np.testing.assert_array_equal(ball_a.centre, ball_b.centre)
    assert ball_a.radius == ball_b.radius
    # T32 S1: the centre is the sphere fitted to the observed surface — here the true apple centre
    # (the points are an exact sphere), not their centroid (which sits on the seen side).
    np.testing.assert_allclose(ball_a.centre, APPLE, atol=1e-6)


def test_authorized_queries_are_counted_by_tier_and_resolved_when_the_window_covers(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    field, notes = _build(ag3s)
    q = field.stats["window"]["authorized_query"]
    assert q["n_points"] == 3 and q["target_free"] is True
    assert q["fraction_in_window"] == 1.0 and q["n_unresolved"] == 0
    assert q["by_tier"]["target_free"] + q["by_tier"]["static"] == 3
    assert not [n for n in notes if "unverified" in n]


def test_a_window_away_from_the_hand_is_noted_as_unverified(monkeypatch):
    """T14 seq 12–21 의 축소판: 창이 1 m 떨어진 물체에 가 있으면 손가락 질의는 coarse 가 답한다."""
    ag3s = _ag3s(monkeypatch, fine_window_placement="target_centroid")
    far = types.SimpleNamespace(points=_apple_geometry().points + [0.6, -0.9, 0.0])
    field, notes = ag3s._build_esdf([object()], far, support_surfaces=[TABLE],
                                    robot_state=np.zeros(2), contact_context=LEFT)
    q = field.stats["window"]["authorized_query"]
    assert q["fraction_in_window"] == 0.0
    assert q["n_unresolved"] == 3 and q["by_tier"]["coarse"] == 3
    [note] = [n for n in notes if "unverified" in n]
    assert "coarse" in note and "not a contact permission" in note


def test_overflow_is_a_note(monkeypatch):
    ag3s = _ag3s(monkeypatch)
    far = types.SimpleNamespace(points=_apple_geometry().points + [1.2, 0.0, 0.0])
    _, notes = ag3s._build_esdf([object()], far, support_surfaces=[TABLE],
                                robot_state=np.zeros(2), contact_context=LEFT)
    assert any(n.startswith("fine window overflow (manipulated)") for n in notes)


def test_run_forwards_the_execution_path_and_the_contact_context(monkeypatch):
    """`process*(execution_path=...)` → `_run` → `_build_esdf`. 없으면 `None`."""
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"})
    ag3s = AG3S(cfg, robot_model=arm)
    seen = []

    def spy(cameras, target, **kw):
        seen.append(kw)
        return None, []

    ag3s._build_esdf = spy
    inputs = dict(attention_map=scene.attention_grid, depth=scene.depth,
                  camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                  robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw,
                  active_manipulators=["left"])
    ag3s.process(**inputs)
    path = np.tile(DEFAULT_ARM_Q, (8, 1))
    ag3s.process(**inputs, execution_path=path)
    assert seen[0]["execution_path"] is None
    assert seen[1]["execution_path"] is path
    assert {m.value for m in seen[1]["contact_context"].active_manipulators} == {"left"}


def test_config_keys_and_validation():
    e = EsdfConfig()
    assert e.fine_window_placement == "hand_swept"
    assert e.fine_window_margin is None and e.fine_window_pad_voxels == 2.0
    assert tuple(WINDOW_PLACEMENTS) == EsdfConfig.FINE_WINDOW_PLACEMENTS
    for bad in ({"fine_window_placement": "attention"}, {"fine_window_margin": -0.01},
                {"fine_window_pad_voxels": -1}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": bad})
    cfg = AG3SConfig.from_dict({"esdf": {"fine_window_margin": 0.03}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf.fine_window_margin == 0.03


# ============================================================================ 4. guard 반경


def _manipulated(radius, centroid=(0.5, 0.3, 0.9)):
    geom = types.SimpleNamespace(
        bounding_geometry=types.SimpleNamespace(dimensions=np.array([radius, 0, 0])))
    return types.SimpleNamespace(geometry=geom, radius=float(radius),
                                 centroid=np.asarray(centroid, np.float64))


def _guard_ag3s(**pc):
    esdf = {"fine_voxel_size": FINE, "voxel_size": 0.02}
    return AG3S(AG3SConfig.from_dict({"pointcloud": pc, "esdf": esdf}))


def test_a_configured_guard_radius_is_used_as_is():
    ag3s = _guard_ag3s(self_filter_target_guard_radius=0.08)
    assert ag3s._guard_radius_for(_manipulated(0.04)) == 0.08
    assert ag3s._guard_radius_for(None) == 0.08
    assert ag3s._config_with_guard_radius(0.08) is ag3s.config


def test_zero_means_the_objects_radius_plus_one_fine_voxel():
    ag3s = _guard_ag3s()
    assert ag3s._guard_radius_for(_manipulated(0.04)) == pytest.approx(0.04 + FINE)
    assert ag3s._guard_radius_for(_manipulated(0.10)) == pytest.approx(0.10 + FINE)
    # 단일 계층이면 그 복셀 (`_target_ball` 과 같은 복셀).
    single = AG3S(AG3SConfig.from_dict({"esdf": {"voxel_size": 0.01}}))
    assert single._guard_radius_for(_manipulated(0.04)) == pytest.approx(0.05)


def test_no_object_no_guard_and_the_config_is_untouched():
    ag3s = _guard_ag3s()
    assert ag3s._guard_radius_for(None) == 0.0
    assert ag3s._guard_radius_for(types.SimpleNamespace(geometry=None, radius=0.0)) == 0.0
    assert ag3s._config_with_guard_radius(0.0) is ag3s.config


def test_the_frame_config_differs_only_in_the_guard_radius():
    ag3s = _guard_ag3s(self_filter_inflation=0.03)
    frame = ag3s._config_with_guard_radius(0.045)
    assert frame.pointcloud.self_filter_target_guard_radius == 0.045
    assert ag3s.config.pointcloud.self_filter_target_guard_radius == 0.0
    a = dataclasses.asdict(frame)
    b = dataclasses.asdict(ag3s.config)
    a["pointcloud"].pop("self_filter_target_guard_radius")
    b["pointcloud"].pop("self_filter_target_guard_radius")
    assert a == b


def test_settings_say_which_guard_rule_is_in_force():
    assert _guard_ag3s().self_filter_settings()["target_guard_rule"] == \
        "manipulated_radius_plus_voxel"
    s = _guard_ag3s(self_filter_target_guard_radius=0.08).self_filter_settings()
    assert s["target_guard_rule"] == "fixed" and s["target_guard_pad_m"] == FINE


def test_the_second_frame_filters_with_the_objects_radius(monkeypatch):
    """파이프라인 끝까지: 프레임 1 이 물체를 확정하면 프레임 2 의 cloud 필터와 depth mask 가 둘 다
    `반지름 + 복셀` 을 받는다 (설정값 0)."""
    from AG3S.ag3s.stages import robot_filter
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = AG3S(AG3SConfig.from_dict({"esdf": {"fine_voxel_size": FINE, "voxel_size": 0.02}}),
                robot_model=arm)
    inputs = dict(attention_map=scene.attention_grid, depth=scene.depth,
                  camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                  robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)
    cs1 = ag3s.process(**inputs)
    assert cs1.metrics["self_filter"]["guard_active"] is False
    manipulated = ag3s.manipulated
    if manipulated is None or manipulated.geometry is None:
        pytest.skip("grounding declined on this scene; nothing to guard")

    calls = []
    real = robot_filter.robot_sphere_mask

    def spy(points, centers, radii, inflation=0.0, **kw):
        calls.append(kw.get("guard_radius"))
        return real(points, centers, radii, inflation, **kw)

    monkeypatch.setattr(robot_filter, "robot_sphere_mask", spy)
    cs2 = ag3s.process(**inputs)
    ag3s._robot_mask_for(scene.depth, scene.camera_intrinsics, scene.T_base_cam, DEFAULT_ARM_Q)
    expected = manipulated.radius + FINE
    assert calls and all(r == pytest.approx(expected) for r in calls)
    sf = cs2.metrics["self_filter"]
    assert sf["guard_active"] is True and sf["guard_radius"] == pytest.approx(expected)
    assert sf["guard_radius_rule"] == "manipulated_radius_plus_voxel"
