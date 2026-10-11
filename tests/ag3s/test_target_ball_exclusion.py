"""T13 — target 을 표본점이 아니라 **기하 범위**로 지운다 (중심 + 반지름, 테이블 면 위만).

## 무엇이었나 (실측, `chunk_00014.npz`)

`exclude_authorized` 가 사과를 **한 voxel도 지우지 못했다.** 손끝 질의점에서 주 계층과
`target_free` 계층이 `17.319 mm` 로 **같았고**, 그 표면은 사과였다.

| | voxel |
|---|---|
| 사과 표면 (중심 45 mm 안) | **2,203** |
| 두 계층이 다른 voxel | 578 |
| 그중 사과 45 mm 안 | **0** |

원인 셋이 겹쳤다 — grounding 표본점이 **74 개**뿐이고(그중 seed 60 개), seed 제외가 팽창하지
않고, **거리장이 성긴 제거에 꿈쩍하지 않는다** (표면 2,203 중 60 을 지워도 지워진 자리마다
5 mm 옆에 이웃 seed 가 살아 있다).

## 이 파일이 박는 것

씬을 **그 기록의 수치로** 만든다 (5 mm voxel · 사과 centroid `[451.6, 329.2, 863.4] mm` ·
표본점 74 개 · 반지름 39.3 mm · 밑면 832.2 mm). 그 위에서 셋을 박는다.

1. **점으로는 못 지운다** — 실패를 재현한다. 지워지는 voxel 이 표본점 수 이하다.
2. **공으로는 지운다** — 사과 45 mm 안에서 지워진 voxel 이 **0 이 아니다** (task 의 합격선).
3. **테이블에는 구멍이 안 난다** — 절단면 아래에서 지워진 voxel 이 **0 이다**.

그리고 부호 판정이 퇴화하는지 수치로 답한다 (task 3 번).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, TargetBall
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.types import SupportSurface

# --- 기록에서 그대로 옮긴 수치 (`chunk_00014.npz`, t=104) -------------------------------
VOXEL = 0.005
APPLE_CENTRE = np.array([0.45161013, 0.32924498, 0.86337266])
APPLE_R = 0.03932          # centroid 까지 거리의 최대값
TABLE_Z = 0.83218          # 표본점의 최저 z = 사과가 테이블에 닿은 곳
N_TARGET_POINTS = 74
#: task 가 합격선으로 쓴 반경.
APPLE_NEIGHBOURHOOD = 0.045


class _FakeSiteIndex:
    """`itg._site_index` 자리. numpy 배열 하나에 `.device`/`.shape` 만 얹는다."""

    def __init__(self, array):
        self.array = array
        self.device = "cpu"

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value


class _FakeIntegrator:
    def __init__(self, site):
        self._site_index = _FakeSiteIndex(site)


@pytest.fixture(autouse=True)
def _torch_stub(monkeypatch):
    """`_clear_seeds` 가 쓰는 torch 표면만 흉내 낸다 (`.venv-ag3s` 에는 torch 가 없다).

    `as_tensor` 를 numpy 로 두면 그 뒤의 fancy indexing 이 **같은 의미**로 돈다 — 이 함수가
    torch 에 요구하는 것이 그것뿐이다. 합집합 인덱스 계산·범위 판정은 전부 numpy 다.
    """
    if "torch" in sys.modules:  # pragma: no cover - GPU 머신에서는 진짜를 쓴다
        return
    stub = types.SimpleNamespace(
        as_tensor=lambda x, device=None, dtype=None: np.asarray(x, np.int64),
        long=np.int64)
    monkeypatch.setitem(sys.modules, "torch", stub)


# --- 씬 ---------------------------------------------------------------------------------


def _scene(margin=0.06):
    """사과 하나 + 테이블 평면. `(origin, shape, site_index, 좌표 격자)`.

    격자는 사과를 넉넉히 감싸는 작은 상자다 — 실제 미세 계층은 128³ 이지만, 이 시험이 묻는 것은
    "공 안이 지워지고 테이블 면은 남는가" 이므로 사과 주변만 있으면 답이 같다.
    """
    lower = APPLE_CENTRE - np.array([margin, margin, margin])
    n = int(np.ceil(2 * margin / VOXEL)) + 1
    shape = (n, n, n)
    ax = [lower[i] + np.arange(n) * VOXEL for i in range(3)]
    grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    d_apple = np.linalg.norm(grid - APPLE_CENTRE, axis=-1) - APPLE_R
    d_table = grid[..., 2] - TABLE_Z
    surface = (np.abs(d_apple) < VOXEL) | (np.abs(d_table) < VOXEL)
    site = np.where(surface, 0, -1).astype(np.int64)
    return lower, shape, site, grid


def _builder():
    return CuroboFieldBuilder(AG3SConfig().esdf)


def _ball(quantile=1.0, pad_voxels=1.0, surfaces=None):
    """`AG3S._target_ball` 을 config 만 든 stub 으로 부른다 — 이 메서드가 그것만 읽는다."""
    cfg = AG3SConfig.from_dict({"esdf": {"target_ball_quantile": quantile,
                                         "target_ball_pad_voxels": pad_voxels}})
    rng = np.random.default_rng(0)
    # 실제 표본점처럼: 사과 표면의 일부만 보인다 (74 개, 카메라가 보는 쪽).
    direction = rng.normal(size=(N_TARGET_POINTS, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = APPLE_CENTRE + direction * APPLE_R
    if surfaces is None:
        surfaces = [SupportSurface(id=0, normal=np.array([0.0, 0.0, 1.0]), offset=TABLE_Z,
                                   point_indices=np.arange(3), point_count=3, rms_error=0.001)]
    # `config` 와 `_support_plane_z` 만 있으면 된다 — 이 메서드가 `self` 에서 읽는 것이 그 둘이다.
    stub = types.SimpleNamespace(config=cfg, _support_plane_z=AG3S._support_plane_z)
    return points, AG3S._target_ball(stub, points, surfaces, voxel=VOXEL)


def _cleared(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """지워진 voxel 의 bool 마스크 — seed 였다가 `-1` 이 된 것."""
    return (before >= 0) & (after < 0)


# --- 1. 점으로는 못 지운다 (실패 재현) --------------------------------------------------


def test_the_point_path_cannot_clear_the_apple():
    """**실패의 재현이다.** 표본점 74 개로는 표면 voxel 수백 개를 지울 수 없다."""
    lower, shape, site, grid = _scene()
    points, _ = _ball()
    itg = _FakeIntegrator(site.copy())
    n, idx = _builder()._exclude_seed_points(itg, points, lower, VOXEL, shape)

    surface_in_apple = int(((np.abs(np.linalg.norm(grid - APPLE_CENTRE, axis=-1) - APPLE_R)
                             < VOXEL) & (site >= 0)).sum())
    assert surface_in_apple > 300, "이 씬의 사과 표면 voxel 이 충분히 많아야 시험이 뜻을 갖는다"
    assert n <= N_TARGET_POINTS, "점 하나가 voxel 하나를 지운다 (팽창하지 않는다)"
    assert n < surface_in_apple / 3, (
        f"점 경로가 표면의 {n}/{surface_in_apple} 만 지웠다 — 이것이 T13 의 원인이다")


# --- 2. 공으로는 지운다 -----------------------------------------------------------------


def test_the_ball_path_clears_the_apple(caplog):
    """**task 의 합격선**: 사과 45 mm 안에서 지워진 voxel 이 0 이 아니다."""
    lower, shape, site, grid = _scene()
    _, ball = _ball()
    assert ball is not None
    before = site.copy()
    itg = _FakeIntegrator(site.copy())
    n, idx = _builder()._exclude_ball_seeds(itg, ball, lower, VOXEL, shape)
    cleared = _cleared(before, itg._site_index.array)

    near_apple = np.linalg.norm(grid - APPLE_CENTRE, axis=-1) <= APPLE_NEIGHBOURHOOD
    n_near = int((cleared & near_apple).sum())
    assert n_near > 0, "사과 45 mm 안에서 지워진 voxel 이 0 이면 T13 이 아무 일도 안 한 것이다"
    assert n_near > 300, f"사과 표면을 대부분 지워야 한다 (지금 {n_near})"
    assert n == int(cleared.sum()), "보고한 수와 실제 지운 수가 같아야 한다"
    assert idx is not None and len(idx) >= n


def test_the_ball_path_does_not_punch_a_hole_in_the_table():
    """**절단면 아래는 한 voxel도 안 지운다.** 조용히 테이블을 지우는 것이 최악이다."""
    lower, shape, site, grid = _scene()
    _, ball = _ball()
    before = site.copy()
    itg = _FakeIntegrator(site.copy())
    _builder()._exclude_ball_seeds(itg, ball, lower, VOXEL, shape)
    cleared = _cleared(before, itg._site_index.array)

    assert int((cleared & (grid[..., 2] < ball.z_min)).sum()) == 0
    # 테이블 상판 자체 (평면 ±한 칸) 의 seed 가 살아 있어야 한다.
    table_band = np.abs(grid[..., 2] - TABLE_Z) < VOXEL
    assert int((cleared & table_band).sum()) == 0, "테이블 면 seed 가 지워졌다"
    assert int(((before >= 0) & table_band).sum()) > 0, "이 씬에 테이블 seed 가 있어야 한다"


def test_the_ball_only_reaches_where_the_apple_is():
    """공 밖은 안 지운다 — 반지름이 측정값이므로 사과보다 한두 칸 넘어서는 데서 멈춘다."""
    lower, shape, site, grid = _scene()
    _, ball = _ball()
    before = site.copy()
    itg = _FakeIntegrator(site.copy())
    _builder()._exclude_ball_seeds(itg, ball, lower, VOXEL, shape)
    cleared = _cleared(before, itg._site_index.array)
    far = np.linalg.norm(grid - APPLE_CENTRE, axis=-1) > ball.radius + 1e-9
    assert int((cleared & far).sum()) == 0


# --- 3. 공을 재는 규칙 ------------------------------------------------------------------


def test_the_ball_is_measured_not_hardcoded():
    points, ball = _ball()
    # T32 S1: the centre is the sphere fitted to the observed surface (here the points lie exactly on
    # the apple sphere, so it is `APPLE_CENTRE`), the radius covers the observed points and the
    # fitted sphere, + one voxel.
    from AG3S.ag3s.robot_models.held_object import fit_surface_sphere

    fit = fit_surface_sphere(points)
    assert fit.accepted
    assert np.allclose(ball.centre, fit.centre) and np.allclose(ball.centre, APPLE_CENTRE, atol=1e-6)
    spread = np.linalg.norm(points - ball.centre, axis=1)
    assert ball.radius == pytest.approx(max(float(spread.max()), fit.radius) + VOXEL)
    assert ball.summary()["centre_method"] == "surface_fit"
    assert ball.z_min == pytest.approx(TABLE_Z + VOXEL)
    # 무엇을 재서 나온 값인지 기록에 남는다.
    summary = ball.summary()
    assert summary["n_points"] == N_TARGET_POINTS
    assert summary["radius_quantile"] == 1.0
    assert summary["support_plane_z_m"] == pytest.approx(TABLE_Z)
    assert summary["voxel_mm"] == pytest.approx(VOXEL * 1000.0)


def test_a_lower_quantile_makes_a_smaller_ball():
    """이상점(반사·잎사귀)이 공을 부풀리는 씬을 위한 손잡이. 작아지면 표면이 **남는다**."""
    _, full = _ball(quantile=1.0)
    _, tight = _ball(quantile=0.8)
    assert tight.radius < full.radius
    assert tight.summary()["radius_quantile"] == 0.8


def test_the_pad_is_added_to_both_the_radius_and_the_cut():
    _, no_pad = _ball(pad_voxels=0.0)
    _, padded = _ball(pad_voxels=2.0)
    assert padded.radius == pytest.approx(no_pad.radius + 2.0 * VOXEL)
    assert padded.z_min == pytest.approx(no_pad.z_min + 2.0 * VOXEL)


# --- 4. 지원면을 못 얻으면 물러난다 (구멍을 내지 않는다) --------------------------------


def test_no_support_surface_means_no_ball():
    """**물러나는 쪽이 안전한 방향이다.** 사과는 안 지워지지만 테이블에 구멍이 안 난다."""
    _, ball = _ball(surfaces=[])
    assert ball is None


def test_a_vertical_plane_is_not_a_support_surface():
    """벽이나 선반 옆면을 받침으로 쓰면 절단면이 엉뚱한 곳에 생긴다."""
    wall = SupportSurface(id=1, normal=np.array([1.0, 0.0, 0.0]), offset=0.2,
                          point_indices=np.arange(3), point_count=3, rms_error=0.001)
    _, ball = _ball(surfaces=[wall])
    assert ball is None


def test_a_plane_above_the_target_is_not_its_support():
    above = SupportSurface(id=2, normal=np.array([0.0, 0.0, 1.0]),
                           offset=APPLE_CENTRE[2] + 0.2, point_indices=np.arange(3),
                           point_count=3, rms_error=0.001)
    _, ball = _ball(surfaces=[above])
    assert ball is None


def test_the_highest_plane_below_the_target_wins():
    """테이블 위에 놓인 물건의 받침은 바닥이 아니라 테이블이다."""
    floor = SupportSurface(id=3, normal=np.array([0.0, 0.0, 1.0]), offset=0.0,
                           point_indices=np.arange(3), point_count=3, rms_error=0.001)
    table = SupportSurface(id=4, normal=np.array([0.0, 0.0, 1.0]), offset=TABLE_Z,
                           point_indices=np.arange(3), point_count=3, rms_error=0.001)
    _, ball = _ball(surfaces=[floor, table])
    assert ball.z_min == pytest.approx(TABLE_Z + VOXEL)


def test_a_tilted_support_plane_is_evaluated_at_the_target_xy():
    """기울어진 평면은 x·y 에 따라 높이가 다르다. centroid 자리에서 풀어야 한다."""
    n = np.array([0.05, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    offset = float(n @ np.array([APPLE_CENTRE[0], APPLE_CENTRE[1], TABLE_Z]))
    tilted = SupportSurface(id=5, normal=n, offset=offset, point_indices=np.arange(3),
                            point_count=3, rms_error=0.001)
    points, ball = _ball(surfaces=[tilted])
    # **공 중심 자리**에서 푼 값이어야 한다. T32 S1 전에는 그 중심이 표본점 centroid 였고
    # (`APPLE_CENTRE` 와 0.107 mm 차이가 z 로 번졌다), 이제는 표면에 맞춘 구의 중심이다.
    c = ball.centre
    expected = (offset - n[0] * c[0] - n[1] * c[1]) / n[2]
    assert ball.z_min == pytest.approx(expected + VOXEL, abs=1e-9)
    tilt_at_centroid = (offset - n[0] * points.mean(0)[0] - n[1] * points.mean(0)[1]) / n[2]
    assert abs(expected - tilt_at_centroid) > 1e-7 or np.allclose(c, points.mean(0))


# --- 5. 부호 판정이 퇴화하는지 (task 3 번) ----------------------------------------------


def test_the_sign_judgement_does_not_degenerate_on_the_ball_layer():
    """**task 3 번에 대한 수치 답.** 공 제외는 판정을 무력화하지 않는다.

    `_force_sign_on_pure_voxels` 는 제외 **뒤**의 `|d|` 가 복셀 반 칸(× 임계)보다 크면 그 자리에
    다른 표면이 없다고 보고 부호를 양수로 돌린다. 공을 지우면 사과 속 voxel 은 남은 표면
    (테이블)까지 멀어 **순수**로 판정되고, 테이블 접촉 고리 근처는 가까워 **공유**로 남는다.

    퇴화란 공유가 **0 개**가 되는 것이다 (판정 B → A). 이 시험은 공유가 0 이 아님을 박는다 —
    즉 접촉면 쪽 관통은 여전히 그대로 보고된다.
    """
    lower, shape, site, grid = _scene()
    _, ball = _ball()
    # 공을 지운 **뒤**의 거리장: 남은 표면은 테이블뿐이다.
    d_after = grid[..., 2] - TABLE_Z
    # 사과 내부는 TSDF 부호가 음수로 남아 있다 (cuRobo 가 부호를 질의 복셀의 TSDF 에서 가져온다).
    inside_apple = np.linalg.norm(grid - APPLE_CENTRE, axis=-1) < APPLE_R
    values = np.where(inside_apple, -np.abs(d_after), d_after).astype(np.float32)

    itg = _FakeIntegrator(site.copy())
    _, idx = _builder()._exclude_ball_seeds(itg, ball, lower, VOXEL, shape)
    sign = CuroboFieldBuilder._force_sign_on_pure_voxels(
        values, idx, VOXEL, threshold_voxels=1.5)

    assert sign["n_attached_voxels"] == len(idx)
    assert sign["n_sign_forced"] > 0, "공 안쪽은 부호가 돌아가야 한다"
    assert sign["n_shared_left_alone"] > 0, (
        "공유 voxel 이 0 이면 판정이 퇴화한 것이다 (판정 B → A). 테이블 접촉 고리 근처는 "
        "남은 표면이 반 칸 안에 있으므로 손대지 않아야 한다")
    # 어느 쪽도 전체를 먹지 않는다 — 그것이 "판정이 살아 있다" 의 뜻이다.
    assert 0 < sign["n_sign_forced"] < sign["n_attached_voxels"]


def test_the_forced_voxels_are_the_ones_far_from_the_remaining_surface():
    """판정의 내용 자체를 박는다 — 남은 표면에서 먼 voxel 만 부호가 돌아간다."""
    lower, shape, site, grid = _scene()
    _, ball = _ball()
    d_after = grid[..., 2] - TABLE_Z
    inside = np.linalg.norm(grid - APPLE_CENTRE, axis=-1) < APPLE_R
    values = np.where(inside, -np.abs(d_after), d_after).astype(np.float32)
    itg = _FakeIntegrator(site.copy())
    _, idx = _builder()._exclude_ball_seeds(itg, ball, lower, VOXEL, shape)

    before = values.copy()
    CuroboFieldBuilder._force_sign_on_pure_voxels(values, idx, VOXEL, threshold_voxels=1.5)
    changed = values != before
    assert np.all(np.abs(before[changed]) > 1.5 * VOXEL - 1e-9)
    assert np.all(before[changed] < 0.0), "부호가 돌아간 것은 원래 음수였던 자리다"
