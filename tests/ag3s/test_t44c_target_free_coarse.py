"""T44c A5 — target 을 지운 거친 계층 (`coarse_no_target`) 과 target-free 질의의 새 합성.

## 무엇을 박나

| # | 성질 (사용자 판정 1 의 조건) |
|---|---|
| 1 | distractor 내부: 새 질의 < 0 (예전 > 0) — 거친 계층의 음수가 살아난다 |
| 2 | target 내부 · 표면: 새 질의 ≈ 예전 (target 이 장애물로 되살아나지 않음, 조건 6) |
| 3 | 모든 점에서 `min(예전, d) ≤ 새 ≤ 예전` — 예전보다 덜 막지 않고, 새 항 (`max(c, d)`) 이 본 합성보다 target 을 더 막지 않는다 |
| 4 | 값-기울기 같은 branch: 계층 이름마다 값이 그 항과 같고, `c` 가 답한 곳의 기울기 = 그 계층 · 유한차분 (조건 3) |
| 5 | 정적 도형 유지 · 창 경계 (`b`) · 창 밖 = `d` · 거친 격자 밖 = 예전 식 (조건 2) |
| 6 | off (`target_free_coarse_layer=None`) = 예전 식 비트 동일 (조건 5) |
| 7 | m 환산 (`coarse_exclusion_params`) — 미세 voxel 을 넣으면 미세 계층 값 그대로 (조건 1) |
| 8 | 거친 제외: 공유 voxel 보존 (`n_shared_kept`) · 지원면 guard (거친 띠) · `ball_owned` 수 맞음 · 부호 교정 집합에서 공유 영역 제외 (조건 4) |
| 9 | `_build_tiers(only="coarse", coarse_free=…)`: 거친 TSDF · distractor 내부 음수 유지 · target 내부 양수 · 공 밖 = main coarse |
| 10 | config 왕복 · validation · CLI · pipeline 배선 (`ball` 에서도 non-target 점) · 기록 키 |
| 11 | (GPU) 실제 cuRobo integration → seed → 제외 → propagate → 부호 → 질의 smoke |

거친 계층의 값은 해석적 SDF 를 격자에 찍은 `layer_from_arrays` (1–6), 가짜 integrator (8–9) 로 만든다.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_field import (
    ANSWER_TIERS,
    CuroboEsdfField,
    layer_from_arrays,
    unresolved_rows,
)
from AG3S.ag3s.fields.esdf import StaticBox

# ============================================================================ 합성 장면 (1–6)

VC, VF = 0.02, 0.005
OC = np.zeros(3)                                  # 거친 첫 voxel 중심
NC = (20, 20, 20)                                 # 0 … 0.38 m
OF = np.array([0.10, 0.10, 0.10])                 # 미세 창 첫 voxel 중심
NF = (33, 33, 33)                                 # 0.10 … 0.26 m
T = np.array([0.17, 0.18, 0.18])                  # target
D = np.array([0.23, 0.18, 0.18])                  # distractor (표면 간격 5 mm)
RT, RD = 0.030, 0.025
TABLE = 0.12


def _grid(origin, shape, vs):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def _sd_t(p):
    return np.linalg.norm(p - T, axis=-1) - RT


def _sd_d(p):
    return np.linalg.norm(p - D, axis=-1) - RD


def _table(p):
    return p[..., 2] - TABLE


def _main(p):
    return np.minimum(np.minimum(_sd_t(p), _sd_d(p)), _table(p))


def _fine_free(p):
    # 미세 TSDF 가 distractor 내부를 못 봤다 → 내부도 양수 (`T44E2b` (a)).
    return np.minimum(np.abs(_sd_d(p)), _table(p))


def _coarse_free(p):
    return np.minimum(_sd_d(p), _table(p))


def _field(*, coarse=True, static=(), coarse_fn=_coarse_free, coarse_grid=(OC, NC)):
    gc, gf = _grid(OC, NC, VC), _grid(OF, NF, VF)
    layers = (layer_from_arrays(_main(gc), OC, VC), layer_from_arrays(_main(gf), OF, VF))
    free = (layer_from_arrays(_fine_free(gf), OF, VF),)
    cl = None
    if coarse:
        o, n = coarse_grid
        cl = layer_from_arrays(coarse_fn(_grid(o, n, VC)), o, VC)
    f = CuroboEsdfField(layers, outside_distance=0.5, target_free_layers=free,
                        target_free_coarse_layer=cl)
    f.static_shapes = tuple(static)
    return f


def _old_formula(field, pts):
    """T44c 전의 식 그대로 — `min(f, max(d, b))` (창 안), `d` (창 밖)."""
    d = np.asarray(field.distance(pts), np.float64).copy()
    layer = field.target_free_layers[0]
    inside = field._covers(layer.grid, pts)
    here = pts[inside]
    free = np.asarray(layer.distance(here), np.float64)
    if field.static_shapes:
        from AG3S.ag3s.fields.esdf import analytic_distance
        free = np.minimum(free, analytic_distance(here, field.static_shapes))
    b = field._boundary_distance(layer.grid, here)
    d[inside] = np.minimum(free, np.maximum(d[inside], b))
    return d


def _window_points(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(OF + 1e-6, OF + (np.asarray(NF) - 1) * VF - 1e-6, (n, 3))


# ---------------------------------------------------------------------------- 1 · 2 · 3


def test_distractor_interior_turns_negative_where_the_old_query_was_positive():
    new, old = _field(), _field(coarse=False)
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = D + v * rng.uniform(0.0, 0.6 * RD, (200, 1))           # distractor 깊은 내부
    o, n = old.target_free_distance(pts), new.target_free_distance(pts)
    assert np.all(o > 0.0), "예전 식은 미세 계층의 미관측 내부 (양수) 를 답한다"
    assert np.all(n < 0.0), "새 식은 target 없는 거친 계층의 음수를 들인다"
    tiers = new.answer_tier(pts, target_free=True)
    assert set(np.unique(tiers)) <= {"target_free_coarse", "fine", "coarse"}
    assert np.count_nonzero(tiers == "target_free_coarse") > 0


def test_target_interior_and_surface_stay_at_the_old_level():
    new, old = _field(), _field(coarse=False)
    rng = np.random.default_rng(2)
    v = rng.normal(size=(400, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    away = v[v[:, 0] < -0.2]                                    # distractor 반대쪽 반구
    pts = np.vstack([T + v[:100] * rng.uniform(0, 0.8 * RT, (100, 1)),       # 내부
                     T + away * (RT + 0.002)])                                 # 표면 2 mm 밖
    pts = pts[pts[:, 2] > TABLE + 0.02]
    o, n = old.target_free_distance(pts), new.target_free_distance(pts)
    d = new.distance(pts)
    assert np.all(n > 0.0), "target 내부 · 표면이 새 식에서 장애물로 살아나면 안 된다"
    assert np.all(n > d + 0.001), "target 은 본 합성보다 덜 막는다 (target 이 빠졌다)"
    # 거친 계층의 이산화 (20 mm) 안에서 예전과 같은 수준.
    assert np.max(o - n) <= VC + 1e-9
    assert np.median(o - n) <= 0.005


def test_the_new_value_is_bracketed_by_main_and_old_everywhere():
    new, old = _field(), _field(coarse=False)
    pts = _window_points()
    d, o, n = new.distance(pts), old.target_free_distance(pts), new.target_free_distance(pts)
    assert np.all(np.minimum(o, d) <= n + 1e-12) and np.all(n <= o + 1e-12)
    # 이 장면에서는 f ≥ d 라 d ≤ 새 가 그대로 선다.
    assert np.all(d <= n + 1e-12)


def test_a_coarse_term_below_main_cannot_revive_the_target_below_main():
    """`c` 가 `d` 보다 작은 자리 (seed 를 지운 뒤 부호가 남은 target 내부) → 그 항은 `d` 로 묶인다."""
    revived = lambda p: np.minimum(_coarse_free(p), _sd_t(p) - 0.02)      # noqa: E731 — 깊은 가짜 음수
    f = _field(coarse_fn=revived)
    rng = np.random.default_rng(8)
    v = rng.normal(size=(200, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = T + v * rng.uniform(0, 0.8 * RT, (200, 1))
    pts = pts[pts[:, 0] < T[0]]                                      # distractor 반대쪽
    n, d = f.target_free_distance(pts), f.distance(pts)
    assert np.all(n >= d - 1e-12)
    tier = f.answer_tier(pts, target_free=True)
    assert "target_free_coarse" not in set(tier[n < 0])


# ---------------------------------------------------------------------------- 4 값-기울기


def test_each_tier_value_is_its_own_term():
    box = StaticBox(center=np.array([0.17, 0.12, 0.24]), half_extents=np.array([0.02, 0.01, 0.01]))
    f = _field(static=(box,))
    pts = np.vstack([_window_points(3000, 3),
                     OF + np.array([0.001, 0.08, 0.08]),            # 창 경계 근처
                     np.array([[0.05, 0.05, 0.30]])])               # 창 밖
    d, tier, _ = f._answer(pts, True, want_gradient=False)
    np.testing.assert_array_equal(d, f.target_free_distance(pts))
    layer = f.target_free_layers[0]
    from AG3S.ag3s.fields.esdf import analytic_distance

    terms = {"target_free": lambda p: layer.distance(p),
             "target_free_coarse": lambda p: f.target_free_coarse_layer.distance(p),
             "static": lambda p: analytic_distance(p, f.static_shapes),
             "window_edge": lambda p: f._boundary_distance(layer.grid, p)}
    seen = set()
    for name, fn in terms.items():
        sel = tier == name
        if sel.any():
            seen.add(name)
            np.testing.assert_allclose(d[sel], fn(pts[sel]), atol=1e-12)
    main = np.isin(tier, ("fine", "coarse", "outside"))
    np.testing.assert_allclose(d[main], f.distance(pts[main]), atol=1e-12)
    assert {"target_free", "target_free_coarse", "static"} <= seen


def _linear_rig():
    """`c` 가 선형 (기울기가 정확) · `f` 는 상수 0.03 · 본 합성 = min(c − 10 mm, 25 mm) — c < 0.03 에서 c 가 답."""
    n = np.array([0.3, -0.2, 0.9327379])
    n /= np.linalg.norm(n)
    lin = lambda p: (p - np.array([0.18, 0.18, 0.18])) @ n          # noqa: E731
    gc, gf = _grid(OC, NC, VC), _grid(OF, NF, VF)
    # 본 합성 ≤ f (표면을 빼서 거리가 줄지 않는다) — 0.045 에서 자른다.
    layers = (layer_from_arrays(np.minimum(lin(gc) - 0.01, 0.025), OC, VC),
              layer_from_arrays(np.full(NF, 0.5), OF, VF))
    free = (layer_from_arrays(np.full(NF, 0.03), OF, VF),)
    cl = layer_from_arrays(lin(gc), OC, VC)
    return CuroboEsdfField(layers, outside_distance=0.5, target_free_layers=free,
                           target_free_coarse_layer=cl), n, lin


def test_the_coarse_branch_gradient_is_that_layer_and_matches_finite_differences():
    f, n, lin = _linear_rig()
    rng = np.random.default_rng(4)
    pts = rng.uniform(OF + 0.03, OF + 0.13, (500, 3))
    v = f.target_free_distance(pts)
    g, tier = f.target_free_gradient(pts, return_tier=True)
    h = 1e-4
    same = np.ones(len(pts), bool)        # 유한차분이 branch 경계를 넘지 않는 점만 (smooth 영역)
    for e in np.eye(3):
        for s in (h, -h):
            same &= f.answer_tier(pts + s * e, target_free=True) == tier
    sel = (tier == "target_free_coarse") & same
    assert sel.sum() > 50
    np.testing.assert_allclose(v[sel], lin(pts[sel]), atol=1e-9)
    np.testing.assert_allclose(g[sel], np.broadcast_to(n, (sel.sum(), 3)), atol=1e-5)
    fd = np.stack([(f.target_free_distance(pts[sel] + h * e) - f.target_free_distance(pts[sel] - h * e))
                   / (2 * h) for e in np.eye(3)], axis=1)
    np.testing.assert_allclose(fd, g[sel], atol=1e-5)
    # f 가 답한 곳은 그 계층 (상수 → 기울기 0) — 본 합성의 기울기 (n) 가 섞이지 않는다.
    q = _window_points(6000, 7)
    q = q[(lin(q) > 0.04) & (f._boundary_distance(f.target_free_layers[0].grid, q) > 0.035)]
    gq, tq = f.target_free_gradient(q, return_tier=True)
    fsel = tq == "target_free"
    assert fsel.sum() > 50
    np.testing.assert_allclose(gq[fsel], 0.0, atol=1e-9)


def test_the_clamp_answers_with_the_main_field_and_its_gradient():
    """`f`·`c` 가 `d` 아래 (seed 를 지운 뒤 부호가 남은 자리) → 바깥 `max(d, ·)` 가 `d` 와 그 기울기를 준다."""
    gc, gf = _grid(OC, NC, VC), _grid(OF, NF, VF)
    main_c = gc[..., 0] * 0.0 + 0.02 + 0.5 * (gc[..., 0] - 0.18)
    layers = (layer_from_arrays(main_c, OC, VC), layer_from_arrays(np.full(NF, 0.5), OF, VF))
    free = (layer_from_arrays(np.full(NF, 0.3), OF, VF),)
    cl = layer_from_arrays(np.full(NC, -0.04), OC, VC)
    f = CuroboEsdfField(layers, outside_distance=0.5, target_free_layers=free,
                        target_free_coarse_layer=cl)
    pts = np.array([[0.18, 0.18, 0.18], [0.16, 0.17, 0.19]])
    np.testing.assert_allclose(f.target_free_distance(pts), f.distance(pts), atol=1e-12)
    g, tier = f.target_free_gradient(pts, return_tier=True)
    assert set(tier) == {"coarse"}
    np.testing.assert_allclose(g, f.gradient(pts), atol=1e-12)


# ---------------------------------------------------------------------------- 5 정적 · 창 · 격자 밖


def test_static_shapes_still_block_inside_the_window():
    box = StaticBox(center=np.array([0.12, 0.24, 0.22]), half_extents=np.array([0.01, 0.01, 0.01]))
    f = _field(static=(box,))
    p = np.array([[0.12, 0.24, 0.22], [0.12, 0.24, 0.2345]])
    from AG3S.ag3s.fields.esdf import analytic_distance

    np.testing.assert_allclose(f.target_free_distance(p), analytic_distance(p, (box,)), atol=1e-12)
    g, tier = f.target_free_gradient(p, return_tier=True)
    assert list(tier) == ["static", "static"]
    np.testing.assert_allclose(g[1], [0, 0, 1], atol=1e-12)


def test_window_edge_is_kept_and_continuous_and_outside_is_main():
    f = _field()
    # 창의 x 하한면으로 다가가는 선 — b → 0 에서 값이 d 로 이어진다.
    xs = OF[0] + np.array([0.0, 1e-4, 1e-3, 5e-3])
    pts = np.stack([xs, np.full(4, 0.20), np.full(4, 0.15)], axis=1)
    v, d = f.target_free_distance(pts), f.distance(pts)
    assert v[0] == pytest.approx(d[0], abs=1e-12)
    assert np.all(v >= d - 1e-12)
    out = np.array([[0.05, 0.05, 0.30], [0.30, 0.30, 0.30], [T[0], T[1], 0.30]])
    np.testing.assert_array_equal(f.target_free_distance(out), f.distance(out))


def test_points_outside_the_coarse_free_grid_fall_back_to_the_old_formula():
    # 거친 target-free 격자를 창의 절반만 덮게 둔다 — 덮지 않는 쪽은 예전 식.
    f = _field(coarse_grid=(OC, (10, 20, 20)))      # x ≤ 0.18
    old = _field(coarse=False)
    pts = _window_points(2000, 5)
    far = pts[:, 0] > 0.18 + 1e-9
    np.testing.assert_array_equal(f.target_free_distance(pts[far]),
                                  old.target_free_distance(pts[far]))
    assert "target_free_coarse" not in set(f.answer_tier(pts[far], target_free=True))


# ---------------------------------------------------------------------------- 6 off


def test_off_is_the_old_formula_bit_for_bit():
    box = StaticBox(center=np.array([0.12, 0.24, 0.22]), half_extents=np.array([0.01, 0.01, 0.01]))
    f = _field(coarse=False, static=(box,))
    pts = np.vstack([_window_points(3000, 6), [[0.05, 0.05, 0.30]]])
    np.testing.assert_array_equal(f.target_free_distance(pts), _old_formula(f, pts))
    d, tier, g = f._answer(pts, True, want_gradient=True)
    np.testing.assert_array_equal(d, _old_formula(f, pts))
    assert "target_free_coarse" not in set(tier)


def test_the_new_tier_is_listed_and_unresolved():
    assert "target_free_coarse" in ANSWER_TIERS
    u, b = unresolved_rows(np.array(["target_free_coarse", "target_free", "coarse"]),
                           np.array([0.1, 0.1, 0.1]))
    assert list(u) == [True, False, True] and not b.any()


def test_mirror_to_host_includes_the_coarse_free_layer():
    calls = []
    f = _field()
    f.target_free_coarse_layer.mirror_to_host = lambda: calls.append(1)
    assert f.mirror_to_host() == 1 and calls == [1]


# ============================================================================ 7 · 8 · 9 builder

torch = pytest.importorskip("torch")

from AG3S.ag3s.fields.curobo_builder import (  # noqa: E402
    CoarseFreeSpec,
    CuroboFieldBuilder,
    OwnedEvidence,
    TargetBall,
    coarse_exclusion_params,
)

BC = np.array([0.30, 0.30, 0.80])                  # 거친 가짜 격자의 첫 voxel 중심
BN = (24, 24, 24)                                  # 0.30 … 0.76
BT = np.array([0.52, 0.50, 0.893])                 # target (r 30 mm)
BD = np.array([0.585, 0.50, 0.893])                # distractor (표면 간격 5 mm)
BR = 0.030
BTABLE = 0.862


def _bsdf(p):
    a = np.linalg.norm(p - BT, axis=-1) - BR
    b = np.linalg.norm(p - BD, axis=-1) - BR
    return np.minimum(np.minimum(a, b), p[..., 2] - BTABLE)


class _VG:
    def __init__(self, t):
        self.feature_tensor = t


class _CoarseItg:
    """cuRobo 의 seed (`|sdf| ≤ 0.9 vs`) → propagate (남은 seed 최근접) → 부호 (그 voxel 의 sdf) 흉내."""

    def __init__(self, origin=BC, shape=BN, vs=VC):
        self.shape, self.vs = shape, vs
        self._esdf_voxel_size = torch.zeros(1)
        self._origin = torch.tensor(origin + 0.5 * (np.asarray(shape) - 1) * vs, dtype=torch.float32)
        self._last_esdf_origin = torch.zeros(3)
        self._site_index = torch.full(shape, -1, dtype=torch.int32)
        self._dist = torch.zeros(shape, dtype=torch.float32)

    def _grid(self):
        vs = float(self._esdf_voxel_size[0])
        c = self._last_esdf_origin.numpy().astype(np.float64)
        return _grid(c - 0.5 * np.asarray(self.shape) * vs + 0.5 * vs, self.shape, vs)

    def _seed_esdf_impl(self, origin, vs):
        v = float(self._esdf_voxel_size[0])
        seed = np.abs(_bsdf(self._grid())) <= 0.9 * v
        ii, jj, kk = np.nonzero(seed)
        packed = np.full(self.shape, -1, np.int64)
        packed[ii, jj, kk] = (kk << 20) | (jj << 10) | ii
        self._site_index.copy_(torch.as_tensor(packed.astype(np.int32)))

    def _propagate_and_distance_impl(self, origin, vs):
        from scipy.spatial import cKDTree

        g = self._grid()
        seeds = np.argwhere(self._site_index.numpy() >= 0)
        d, _ = cKDTree(g[seeds[:, 0], seeds[:, 1], seeds[:, 2]]).query(g.reshape(-1, 3), k=1)
        val = np.where(_bsdf(g).reshape(-1) < 0.0, -d, d).reshape(self.shape)
        self._dist.copy_(torch.as_tensor(val.astype(np.float32)))

    def get_voxel_grid(self):
        return _VG(self._dist)


def _bbuilder(**esdf):
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "voxel_size": VC,
                                         "fine_voxel_size": VF, **esdf}}).esdf
    return CuroboFieldBuilder(cfg)


def _bball():
    return TargetBall(centre=BT, radius=0.045, z_min=BTABLE + 0.005,
                      provenance={"support_plane_z_m": BTABLE, "pad_mm": 5.0})


def _sphere_pts(c, n, seed, *, upper_only=False):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(4 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if upper_only:
        v = v[v[:, 2] > -0.3]
    return c + BR * v[:n]


def _spec(b, *, owned=None, nontarget="default"):
    nt = _sphere_pts(BD, 400, 1) if isinstance(nontarget, str) else nontarget
    return b._coarse_free_spec(_bball(), owned, nt)


def _excl(b, spec):
    itg = _CoarseItg()
    itg._esdf_voxel_size.copy_(torch.tensor([VC]))
    itg._last_esdf_origin.copy_(itg._origin)
    itg._seed_esdf_impl(None, None)
    seed_before = itg._site_index.numpy().copy() >= 0
    out = b._coarse_ball_exclusion(itg, _bball(), BC, VC, BN, spec)
    return itg, seed_before, out


# ---------------------------------------------------------------------------- 7 m 환산


def test_metre_conversion_reproduces_the_fine_values_on_the_fine_voxel():
    b = _bbuilder(owned_tie_voxels=1.0, owned_max_dist_voxels=3.0)
    spec = _spec(b)
    assert spec.tie_m == pytest.approx(0.005) and spec.max_dist_m == pytest.approx(0.015)
    assert spec.sign_threshold_m == pytest.approx(1.5 * VF)
    fine = coarse_exclusion_params(_bball(), VF, spec)
    assert fine["z_min_m"] == pytest.approx(_bball().z_min)          # 미세: 지금 값 그대로
    assert fine["tie_voxels"] == pytest.approx(1.0)
    assert fine["max_dist_voxels"] == pytest.approx(3.0)
    assert fine["sign_threshold_voxels"] == pytest.approx(1.5)
    coarse = coarse_exclusion_params(_bball(), VC, spec)
    assert coarse["band_m"] == pytest.approx(0.018)
    assert coarse["z_min_m"] == pytest.approx(BTABLE + VC)            # 지원면이 거친 voxel 을 심는 띠 위
    assert coarse["tie_voxels"] == pytest.approx(0.25)                # 5 mm 그대로
    assert coarse["max_dist_m_effective"] == pytest.approx(0.018)     # 15 mm < 띠 18 mm → 띠
    assert coarse["sign_threshold_voxels"] == pytest.approx(0.375)    # 7.5 mm 그대로
    no_plane = coarse_exclusion_params(
        TargetBall(centre=BT, radius=0.045, z_min=0.9), VC, CoarseFreeSpec())
    assert no_plane["z_min_m"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------- 8 거친 제외


def test_shared_coarse_voxels_keep_their_seeds_and_are_counted():
    b = _bbuilder()
    itg, before, (n, sign_idx, ex) = _excl(b, _spec(b))
    c = ex["counts"]
    assert c["exclusion_method"] == "ball"
    assert c["n_shared_kept"] > 0, "distractor 와 공유한 거친 voxel 이 공 안에 있다"
    assert c["n_removed"] + c["n_shared_kept"] == c["n_seeds_in_ball"] == n + c["n_shared_kept"]
    kept = ex["trace"]["shared_kept_ijk"]
    removed = ex["trace"]["removed_ijk"]
    assert len(kept) == c["n_shared_kept"] and len(removed) == n
    after = itg._site_index.numpy() >= 0
    assert np.all(after[tuple(kept.T)]), "공유 seed 는 남는다"
    assert not np.any(after[tuple(removed.T)])
    # 지운 seed 는 어느 non-target 점에서도 띠 (18 mm) 밖이고, 보존한 것은 안이다.
    from scipy.spatial import cKDTree

    tree = cKDTree(_sphere_pts(BD, 400, 1))
    assert np.all(tree.query(BC + removed * VC)[0] > 0.018)
    assert np.all(tree.query(BC + kept * VC)[0] <= 0.018)
    # 부호 교정 집합에 공유 영역 (seed 든 아니든) 이 없다.
    assert np.all(tree.query(BC + sign_idx * VC)[0] > 0.018)
    assert c["n_shared_cell"] <= c["n_seeds_shared"]


def test_without_nontarget_evidence_every_ball_seed_goes_like_the_fine_ball():
    b = _bbuilder()
    _, before, (n, sign_idx, ex) = _excl(b, _spec(b, nontarget=None))
    c = ex["counts"]
    assert c["n_shared_kept"] == 0 and n == c["n_seeds_in_ball"] > 0
    cand = b._ball_candidates(TargetBall(centre=BT, radius=0.045, z_min=BTABLE + VC), BC, VC, BN)
    assert len(sign_idx) == len(cand)                  # `ball`: 공 안 전부 (공유 영역이 없으면)


def test_the_support_guard_uses_the_coarse_seed_band():
    b = _bbuilder()
    _, before, (n, sign_idx, ex) = _excl(b, _spec(b, nontarget=None))
    z = (BC + ex["trace"]["removed_ijk"] * VC)[:, 2]
    assert np.all(z >= BTABLE + VC - 1e-9), "테이블이 거친 voxel 을 심을 수 있는 높이는 지우지 않는다"
    zs = (BC + sign_idx * VC)[:, 2]
    assert np.all(zs >= BTABLE + VC - 1e-9)
    # 미세 z_min (테이블 + 5 mm) 이었다면 지웠을 테이블 띠 seed 가 실제로 있다 (guard 가 일을 한다).
    g = _grid(BC, BN, VC)
    band = (np.abs(_bsdf(g)) <= 0.9 * VC) & (g[..., 2] >= BTABLE + 0.005) & (g[..., 2] < BTABLE + VC) \
        & (np.linalg.norm(g - BT, axis=-1) <= 0.045)
    assert band.any()


def test_ball_owned_counts_add_up_on_the_coarse_tier():
    b = _bbuilder(target_exclusion="ball_owned", owned_max_dist_voxels=3.0)
    owned = OwnedEvidence(target_points=_sphere_pts(BT, 200, 0, upper_only=True),
                          nontarget_points=_sphere_pts(BD, 400, 1),
                          tie_voxels=1.0, max_dist_voxels=3.0)
    _, before, (n, sign_idx, ex) = _excl(b, _spec(b, owned=owned))
    c = ex["counts"]
    assert c["exclusion_method"] == "ball_owned"
    assert c["params"]["max_dist_m_effective"] == pytest.approx(0.018)
    assert (c["n_removed"] + c["n_seeds_shared"] + c["n_kept_nontarget"] + c["n_ambiguous"]
            + c["n_unowned_kept"]) == c["n_seeds_in_ball"]
    assert c["n_removed"] == c["n_target_owned"] + c["n_unowned_removed"]
    assert c["n_removed"] > 0


def test_coarse_tier_values_distractor_negative_target_positive_far_unchanged():
    b = _bbuilder()
    itg = _CoarseItg()
    (main,) = b._build_tiers(itg, None, "cpu", torch, only="coarse")
    (free,) = b._build_tiers(itg, None, "cpu", torch, only="coarse", tier_name="coarse_no_target",
                             ball=_bball(), coarse_free=_spec(b))
    assert main.name == "coarse" and free.name == "coarse_no_target"
    assert free.extra["exclusion"]["n_shared_kept"] > 0
    assert free.extra["threshold_voxels"] == pytest.approx(0.375)
    assert free.extra["tsdf_truncation_m"] == pytest.approx(b.coarse_truncation)
    lm = layer_from_arrays(main.values.numpy(), main.origin, VC)
    lf = layer_from_arrays(free.values.numpy(), free.origin, VC)
    # distractor 중심 (공 밖 쪽): 거친 target-free 도 음수.
    p_d = np.array([BD + [0.01, 0.0, 0.0]])
    assert lf.distance(p_d)[0] < 0.0
    # target 중심: main 은 음수, target-free 는 양수 (부호 교정 · seed 제거).
    p_t = np.array([BT])
    assert lm.distance(p_t)[0] < 0.0 < lf.distance(p_t)[0]
    # 공에서 먼 곳 (테이블 위 다른 자리): 같은 값.
    far = np.array([[0.34, 0.34, 0.90], [0.70, 0.70, 0.95]])
    np.testing.assert_array_equal(lm.distance(far), lf.distance(far))


def test_tier_specs_and_itg_choice_are_unchanged_for_existing_calls():
    b = _bbuilder()
    assert [s[0] for s in b._tier_specs(np.zeros(3))] == ["coarse", "fine"]
    assert [s[0] for s in b._tier_specs(np.zeros(3), only="fine", tier_name="fine_no_target")] \
        == ["fine_no_target"]
    assert [s[0] for s in b._tier_specs(None, only="coarse")] == ["coarse"]
    assert [s[0] for s in b._tier_specs(None, only="coarse", tier_name="coarse_no_target")] \
        == ["coarse_no_target"]


# ============================================================================ 10 배선


def test_config_round_trip_and_validation():
    cfg = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True}})
    assert cfg.esdf.target_free_coarse is True
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf.target_free_coarse is True
    assert AG3SConfig.from_dict({}).esdf.target_free_coarse is False
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"target_free_coarse": "yes"}})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml",
                                      *argv])


def test_cli_flag_maps_to_the_esdf_key_and_defaults_to_nothing():
    from AG3S.trajopt.serve_safe import target_free_coarse_section

    assert target_free_coarse_section(_parse()) == {}
    sec = target_free_coarse_section(_parse("--target-free-coarse"))
    assert sec == {"target_free_coarse": True}
    assert AG3SConfig.from_dict({"esdf": sec}).esdf.target_free_coarse is True


def test_cli_rejects_the_flag_with_relax(monkeypatch, capsys):
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", "--esdf-backend", "curobo",
                                      "--target-free-coarse"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert "--target-free-coarse" in capsys.readouterr().err


def test_pipeline_passes_nontarget_points_with_target_free_coarse_in_ball_mode(monkeypatch):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    nt = {"points": np.zeros((4, 3)), "source": "indices", "n_cloud": 9,
          "n_target_dropped": 3, "n_support_dropped": 2}
    ag3s = fw._ag3s(monkeypatch, target_free_coarse=True)
    fw._build(ag3s, nontarget=nt)
    kw = ag3s._esdf_builder.calls[-1]
    assert kw["target_free_ball"] is not None
    assert kw["nontarget_points"] is nt["points"]
    ag3s = fw._ag3s(monkeypatch)
    fw._build(ag3s, nontarget=nt)
    assert "nontarget_points" not in ag3s._esdf_builder.calls[-1], "off 는 예전 호출 그대로"


def test_the_record_writes_the_free_coarse_layer_and_trace(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    f = _field()
    f.stats = {"target_free": {"mode": "ball", "exclusion_method": "ball",
                               "coarse": {"mode": "ball", "n_shared_kept": 3}}}
    f.exclusion_trace = {"tier": "fine_no_target", "method": "ball", "origin": OF,
                         "voxel_size": VF, "removed_ijk": np.zeros((2, 3), np.int64)}
    f.coarse_exclusion_trace = {"tier": "coarse_no_target", "method": "ball", "origin": OC,
                                "voxel_size": VC, "removed_ijk": np.ones((5, 3), np.int64),
                                "shared_kept_ijk": np.ones((3, 3), np.int64)}
    names = [n for n, _ in ConstraintRecordWriter.field_layers(f)]
    assert names == ["", "coarse", "free", "free_coarse"]
    payload: dict = {}
    info = ConstraintRecordWriter._write_exclusion_trace(f, payload)
    assert payload["target_free_coarse_removed_ijk"].shape == (5, 3)
    assert payload["target_free_coarse_shared_kept_ijk"].shape == (3, 3)
    assert float(payload["target_free_coarse_trace_voxel_size"]) == VC
    assert info["target_exclusion"]["coarse"]["n_shared_kept"] == 3
    json.dumps(info)
    off = _field(coarse=False)
    assert [n for n, _ in ConstraintRecordWriter.field_layers(off)] == ["", "coarse", "free"]


# ============================================================================ 11 GPU (real cuRobo; skipped on CPU)


def test_curobo_target_free_coarse_smoke():
    """Real cuRobo integration → seed → coarse exclusion → propagate → sign → query, one scene.

    The coarse tier is built **after** the main and target-free fine tiers on the same buffers
    (pitfall 4) — the main layers and the fine target-free layer must equal the flag-off build."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from AG3S.ag3s.fields.curobo_builder import FineWindowRequest

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    ball = TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825,
                      provenance={"support_plane_z_m": 0.82, "pad_mm": 5.0})
    kw = dict(target_points=apple, target_free_points=apple, target_free_ball=ball,
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other)
    fields = {}
    for name, over in (("off", {}), ("on", {"target_free_coarse": True})):
        b = CuroboFieldBuilder(G._config(**over))
        for _ in range(2):
            fields[name] = b.update(cams, **kw)
    off, on = fields["off"], fields["on"]
    assert off.target_free_coarse_layer is None and on.target_free_coarse_layer is not None
    for la, lb in zip(off.layers, on.layers):
        assert torch.equal(la.values_device, lb.values_device)
    assert torch.equal(off.target_free_layers[0].values_device, on.target_free_layers[0].values_device)
    c = on.stats["target_free"]["coarse"]
    assert off.stats["target_free"]["coarse"] == {"mode": "off"}
    assert c["mode"] == "ball" and c["n_seeds_in_ball"] > 0
    assert c["n_removed"] + c["n_shared_kept"] == c["n_seeds_in_ball"]
    assert on.coarse_exclusion_trace["tier"] == "coarse_no_target"
    pts = np.random.default_rng(5).normal([0.51, 0.0, 0.86], 0.04, (2000, 3))
    d, n, o = on.distance(pts), on.target_free_distance(pts), off.target_free_distance(pts)
    assert np.all(np.minimum(o, d) <= n + 1e-9) and np.all(n <= o + 1e-9)
    tier = on.answer_tier(pts, target_free=True)
    assert set(np.unique(tier)) <= set(ANSWER_TIERS)
