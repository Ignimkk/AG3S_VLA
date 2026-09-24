"""Frustum-aware decay — 잊는 장치, 그리고 **잊지 말아야 할 것** (F20).

우리 TSDF 가중치는 `min(w + 1, max_weight)` 로 단조 증가한다. 옛 관측이 절대 흐려지지 않으므로
사라진 물체의 잔상이 남는다 — 실측(`run_0004`): 사과가 원래 자리에서 **302 mm 떠난 뒤에도 그
자리가 -6.7 mm 로 점유**였고 8 프레임 내내 자유가 되지 않았다.

cuRoboV2 §5.3 의 감쇠가 그것을 푼다. 다만 **실제 장애물도 함께 잊는다** — 그래서 이 파일은
"잔상이 지워지는가" 만큼 "표면이 살아남는가" 를 함께 고정한다. 실측 요약:

    설정              잔상이 자유가 된 프레임   실제 표면 최악   점유 복셀 변화
    끔                       0/9                +20.0 mm        +2.7 %
    a_t 0.99                 0/9                +20.0 mm        +2.7 %   (시간 감쇠만으론 안 된다)
    a_t 0.99 · a_f 0.5       7/9                +74.6 mm       -19.2 %   <- 실제 기하가 침식된다
    a_t 0.99 · a_f 0.8       2/9                +25.4 mm        +2.5 %   <- 채택 권고

**잔상은 보수적인 오류이고 침식은 위험한 오류다.** 그래서 잔상 제거 속도를 침식과 맞바꾸지
않는다. 기본값은 끔이고, 켤 때의 값은 위 측정에서 나온다.
"""

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError
from benchmark.ag3s.fields.esdf import FREE, OCCUPIED, UNKNOWN, TsdfVolume, VoxelGrid

VOXEL = 0.05


@pytest.fixture
def volume():
    grid = VoxelGrid.from_bounds([-0.2, -0.2, -0.2], [0.2, 0.2, 0.2], VOXEL)
    vol = TsdfVolume(grid, truncation=3 * VOXEL)
    vol.weight[:] = 10.0
    vol.tsdf[:] = 0.5
    return vol


def test_decay_off_changes_nothing(volume):
    before = volume.weight.copy()
    stats = volume.decay([], time_decay=1.0, frustum_decay=1.0)
    assert stats["decayed"] is False
    assert np.array_equal(volume.weight, before)


def test_time_decay_applies_everywhere(volume):
    volume.decay([], time_decay=0.5, frustum_decay=1.0)
    assert np.allclose(volume.weight, 5.0)


def test_frustum_decay_applies_only_inside_the_mask(volume):
    """비대칭이 요점이다 — 보이는 곳은 빨리 잊고, 안 보이는 곳은 천천히 잊는다."""
    mask = np.zeros(volume.grid.shape, bool)
    mask[0, :, :] = True
    volume.decay([mask], time_decay=1.0, frustum_decay=0.5)
    assert np.allclose(volume.weight[0], 5.0)     # 절두체 안
    assert np.allclose(volume.weight[1:], 10.0)   # 밖은 그대로


def test_both_factors_compose(volume):
    mask = np.zeros(volume.grid.shape, bool)
    mask[0] = True
    volume.decay([mask], time_decay=0.5, frustum_decay=0.5)
    assert np.allclose(volume.weight[0], 2.5)     # 0.5 * 0.5
    assert np.allclose(volume.weight[1:], 5.0)


def test_min_weight_returns_a_faded_voxel_to_unknown(volume):
    """'잊는다' 가 실제로 뜻하는 것 — 값이 남아 있어도 근거가 사라지면 근거 없음으로."""
    volume.tsdf[:] = 0.0                          # 전부 표면
    assert (volume.occupancy(surface_band=VOXEL) == OCCUPIED).all()
    volume.decay([], time_decay=0.01, frustum_decay=1.0)   # 10.0 -> 0.1
    assert (volume.occupancy(surface_band=VOXEL, min_weight=0.5) == UNKNOWN).all()
    # 문턱이 낮으면 여전히 관측으로 친다 — 문턱이 정하는 것이지 감쇠가 정하는 것이 아니다.
    assert (volume.occupancy(surface_band=VOXEL, min_weight=0.0) == OCCUPIED).all()


def test_frustum_mask_selects_what_the_camera_sees(volume):
    # 초점거리를 길게 잡아 시야를 좁힌다 — 격자 일부만 들어와야 "고른다" 가 뜻을 갖는다.
    K = np.array([[800.0, 0.0, 64.0], [0.0, 800.0, 64.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, 3] = np.array([0.0, 0.0, -1.0])         # 카메라가 +z 를 본다
    mask = volume.frustum_mask(K, T, (128, 128), depth_min=0.1, depth_max=3.0)
    assert mask.any() and not mask.all(), "좁은 시야인데 전부 보인다면 투영이 틀렸다"

    # 깊이 범위 밖은 빠진다 — 절두체는 각도만이 아니라 거리로도 잘린다.
    near_only = volume.frustum_mask(K, T, (128, 128), depth_min=0.1, depth_max=0.5)
    assert near_only.sum() < mask.sum()


def test_config_rejects_out_of_range_decay():
    for bad in ({"time_decay": 1.5}, {"time_decay": 0.0}, {"frustum_decay": -0.1},
                {"min_weight": -1.0}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": bad})


def test_decay_is_off_by_default():
    """기본값이 끔인 것은 판정이다 — 침식이 잔상보다 위험하기 때문."""
    cfg = AG3SConfig.from_dict({}).esdf
    assert cfg.time_decay == 1.0 and cfg.frustum_decay == 1.0 and cfg.min_weight == 0.0
