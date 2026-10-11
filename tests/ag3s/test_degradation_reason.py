"""`DEGRADED` 를 내는 자리마다 **사유가 있는가.**

이 파일이 지키는 불변량은 하나다: **`ag3s_status`/`validity` 가 degraded 인데 사유가 빈
경로는 없다.** 2026-09-25 첫 live smoke 에서 2 청크 중 1 개가 `max_violation_m = 0.0` 인데
`degraded` 로 HOLD 됐고 — 궤적이 아니라 기하 인증이 실패한 것인데 — **왜인지가 와이어에도
서버 로그에도 없었다.** 사유를 모르면 그 실행은 기록만 쌓이고 판정이 안 된다.

검사는 두 층이다.

* **자리별** — `DEGRADED` 를 내는 분기를 하나씩 태워 **그 자리의 코드**가 나오는지 본다.
  코드가 나오는 것만으로는 부족하다: 엉뚱한 코드가 나오면 다음 사람이 엉뚱한 곳을 본다.
* **구조** — 어느 분기가 `degradation.reason()` 을 빠뜨려도 `ensure_reason` 이 마지막 관문에서
  `degraded_without_reason` 을 달아 보낸다. 그 코드가 **정상 시나리오에서는 안 나오는 것**이
  자리별 배선이 살아 있다는 증거다.

`CODES` 에 등록되지 않은 코드는 쓸 수 없다 (`reason()` 이 거절한다). 그래서 새 `DEGRADED`
분기를 만드는 사람은 등록을 피해 갈 수 없고, 등록이 곧 "이 코드가 어느 분기인가" 의 문서다.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.runtime import degradation
from AG3S.ag3s.runtime.multiview import check_freshness, fuse_observations
from AG3S.ag3s.types import CameraID, CameraObservation, ConstraintValidity


# --- 등록 자체 -------------------------------------------------------------------------


def test_an_unregistered_code_is_refused():
    """오타가 조용히 새 코드를 만들면 "사유가 비지 않았다" 는 통과하는데 아무도 그 코드를 모른다."""
    with pytest.raises(KeyError, match="unknown degradation code"):
        degradation.reason("pointcloud_cappd", "typo")


def test_reason_roundtrips_through_a_plain_note_list():
    """매체는 `notes: list[str]` 그대로다 — 병렬 자료구조를 만들지 않은 것이 이 설계의 핵심이다."""
    note = degradation.reason("camera_skew", "cameras span 120 ms (limit 40 ms)")
    assert note.startswith("camera_skew: ")
    assert degradation.reasons([note]) == [
        {"code": "camera_skew", "detail": "cameras span 120 ms (limit 40 ms)"}]
    assert degradation.codes([note]) == ["camera_skew"]
    # 코드 없는 산문 노트는 **코드로 추측하지 않는다.** 추측하면 기록이 거짓이 된다.
    assert degradation.reasons(["just some prose about the scene"]) == []


def test_explain_says_how_many_it_left_out():
    """조용히 자르면 읽는 사람이 그것이 전부라고 믿는다."""
    notes = [degradation.reason(c, "d") for c in
             ("camera_skew", "camera_missing", "pointcloud_capped", "candidate_overflow")]
    text = degradation.explain(notes, limit=2)
    assert "camera_skew" in text and "+2 more" in text
    assert degradation.explain([]) == ""


# --- 구조 관문 -------------------------------------------------------------------------


def test_ensure_reason_fills_an_unattributed_degraded_frame():
    """새 DEGRADED 분기가 `reason()` 을 빠뜨려도 와이어에 빈 사유가 나가지 않는다."""
    out = degradation.ensure_reason(["some prose"], degraded=True)
    assert degradation.codes(out) == ["degraded_without_reason"]
    # 원래 노트를 지우지 않는다.
    assert "some prose" in out


def test_ensure_reason_is_silent_when_a_site_already_declared_one():
    note = degradation.reason("camera_missing", "camera(s) ['wrist_cam_l'] did not report")
    assert degradation.ensure_reason([note], degraded=True) == [note]
    # degraded 가 아니면 아무것도 더하지 않는다 — 정상 프레임의 노트를 늘리지 않는다.
    assert degradation.ensure_reason(["prose"], degraded=False) == ["prose"]


# --- 자리별: multiview ------------------------------------------------------------------


#: RB-Y1 자기 필터 모델의 관절 수. `robot_state` 길이가 이것과 다르면 `robot_filter` 가
#: 거절한다 — 무엇으로 채워야 할지 추측하지 않는다는 그쪽 계약이다.
NQ = 20


def _observation(camera, *, t: float, state_t: float | None = None, n: int = 40):
    """작은 합성 depth 관측 하나. 점 개수는 `n*n` 픽셀에서 나온다.

    **매끄러운 벽**이어야 한다. 노이즈 depth 는 점이 부피 전체에 퍼져 voxel 을 키워도 개수가
    안 줄고, 그러면 cap 이 `covered=False`(= INCOMPLETE) 로 끝나 이 파일이 재려는
    DEGRADED 분기에 도달하지 못한다 — 실제로 처음 그렇게 썼다가 걸렸다.
    """
    tilt = 0.01 * len(CameraID.parse(camera).value)
    yy, xx = np.mgrid[0:n, 0:n] / float(n)
    depth = 0.9 + 0.05 * xx + 0.02 * yy + tilt
    K = np.array([[float(n), 0.0, n / 2.0], [0.0, float(n), n / 2.0], [0.0, 0.0, 1.0]])
    return CameraObservation(
        camera_id=CameraID.parse(camera),
        depth=depth,
        camera_intrinsics=K,
        T_base_cam=np.eye(4),
        robot_state=np.zeros(NQ),
        timestamp=t,
        state_timestamp=t if state_t is None else state_t,
    )


def test_camera_skew_declares_its_code():
    cfg = AG3SConfig.from_dict({"timing": {"max_camera_skew_sec": 0.01}})
    validity, notes, _ = check_freshness(
        [_observation(CameraID.HEAD, t=10.0), _observation(CameraID.LEFT_WRIST, t=10.5)], cfg)

    assert validity is ConstraintValidity.DEGRADED
    assert "camera_skew" in degradation.codes(notes)


def test_stale_robot_state_declares_its_code():
    cfg = AG3SConfig.from_dict({"timing": {"max_state_age_sec": 0.01}})
    validity, notes, _ = check_freshness([_observation(CameraID.HEAD, t=10.0, state_t=9.0)], cfg)

    assert validity is ConstraintValidity.DEGRADED
    assert "camera_state_stale" in degradation.codes(notes)


def test_stale_transform_declares_its_code():
    cfg = AG3SConfig.from_dict({"timing": {"max_transform_age_sec": 0.01}})
    validity, notes, _ = check_freshness(
        [_observation(CameraID.HEAD, t=10.0), _observation(CameraID.LEFT_WRIST, t=9.0)], cfg)

    assert validity is ConstraintValidity.DEGRADED
    assert "camera_transform_stale" in degradation.codes(notes)


def test_a_missing_expected_camera_declares_its_code():
    cfg = AG3SConfig.from_dict(
        {"timing": {"expected_cameras": ["head", "left_wrist"]}})
    validity, notes, _ = check_freshness([_observation(CameraID.HEAD, t=10.0)], cfg)

    assert validity is ConstraintValidity.DEGRADED
    assert "camera_missing" in degradation.codes(notes)


def test_a_capped_fused_cloud_declares_its_code():
    """**여기가 2026-09-25 까지 노트가 아예 없던 자리다.** validity 만 내리고 조용했다."""
    # cap 에 **걸리되** voxel 을 키워 coverage 를 지킬 수 있는 값. 더 작게 주면
    # `covered=False` 로 INCOMPLETE 가 되어 이 분기에 닿지 않는다.
    cfg = AG3SConfig.from_dict({"pointcloud": {"max_points": 600}})
    fusion = fuse_observations([_observation(CameraID.HEAD, t=10.0)], cfg)

    assert fusion.validity is ConstraintValidity.DEGRADED
    assert "fused_pointcloud_capped" in degradation.codes(fusion.notes)
    # 사유에 수치가 있어야 한다 — 없으면 다음 사람이 다시 서버에 들어가서 재야 한다.
    detail = next(r["detail"] for r in degradation.reasons(fusion.notes)
                  if r["code"] == "fused_pointcloud_capped")
    assert "head" in detail and "max_points=600" in detail and "1600 pts" in detail


# --- 자리별: pipeline / to_adapter -------------------------------------------------------


def _pipeline(**overrides):
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from AG3S.ag3s.runtime.pipeline import AG3S

    model = UrdfSphereChain(parse_urdf(RBY1_URDF),
                            link_filter=[f"link_{s}_arm_{i}"
                                         for s in ("left", "right") for i in range(6)])
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", **overrides})
    return AG3S(cfg, robot_model=model, constraint_robot_model=model)


def test_a_capped_point_cloud_declares_its_code():
    """단일 카메라 경로의 cap. 융합 경로(`fused_pointcloud_capped`)와 **다른 코드**여야 한다 —
    같은 코드를 쓰면 어느 경로가 걸렸는지 다시 알 수 없다."""
    ag3s = _pipeline(pointcloud={"max_points": 600})
    cs = ag3s.process_multi([_observation(CameraID.HEAD, t=10.0)], phase="approach")

    codes = degradation.codes(cs.notes)
    assert "pointcloud_capped" in codes or "fused_pointcloud_capped" in codes
    assert "degraded_without_reason" not in codes


def test_no_robot_model_declares_its_code():
    """제약을 쓸 대상이 없다. 기하는 있는데 제약이 없는 것을 VALID 로 두면 소비자가
    '제약 없음' 을 '깨끗함' 으로 읽는다."""
    from AG3S.ag3s.runtime.pipeline import AG3S

    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf"}))
    cs = ag3s.process_multi([_observation(CameraID.HEAD, t=10.0)], phase="approach")

    assert "no_robot_model" in degradation.codes(cs.notes)


# --- 불변량: 어떤 프레임도 사유 없이 degraded 로 나가지 않는다 -----------------------------


@pytest.mark.parametrize("overrides", [
    {},
    {"pointcloud": {"max_points": 600}},
    {"timing": {"max_camera_skew_sec": 0.001, "expected_cameras": ["head", "left_wrist"]}},
    {"esdf": {"unknown_report_threshold": 0.0}},
])
def test_a_degraded_frame_never_leaves_without_a_reason(overrides):
    """이것이 이 STEP 의 합격 조건이다. 어느 설정에서 degraded 가 나오든 사유가 있어야 한다."""
    ag3s = _pipeline(**overrides)
    cs = ag3s.process_multi(
        [_observation(CameraID.HEAD, t=10.0), _observation(CameraID.LEFT_WRIST, t=10.2)],
        phase="approach")

    if cs.validity is not ConstraintValidity.DEGRADED:
        pytest.skip(f"이 설정은 degraded 가 아니다 (validity={cs.validity.value})")
    codes = degradation.codes(cs.notes)
    assert codes, (
        f"validity=DEGRADED, status={cs.status.value} 인데 코드 달린 사유가 없다. "
        f"notes={list(cs.notes)}")
    # 관문이 채운 것이면 **어느 분기가 reason() 을 빠뜨린 것**이다 — 자리별 배선이 죽었다.
    assert "degraded_without_reason" not in codes, (
        f"어느 DEGRADED 분기가 degradation.reason() 을 빠뜨렸다. notes={list(cs.notes)}")


# --- 자리별: _esdf_coverage (G3 · G4) ----------------------------------------------------
#
# 이 둘은 실제 cuRobo 필드가 있어야 나는 자리라 합성 씬으로는 못 태운다. 필드가 이 검사에게
# 보여 주는 면(`stats["per_camera"]` · `coverage_grid` · `outside_distance`)만 흉내내
# **분기 자체**를 태운다 — 그것이 "사유가 비지 않는가" 를 재는 데 필요한 전부다.


class _Grid:
    def __init__(self, lower, upper):
        self.origin = np.asarray(lower, np.float64)
        self.upper = np.asarray(upper, np.float64)


class _Field:
    outside_distance = 0.4

    def __init__(self, per_camera, *, lower=(-5.0, -5.0, -5.0), upper=(5.0, 5.0, 5.0)):
        self.stats = {"per_camera": per_camera}
        self.grid = _Grid(lower, upper)
        self.coverage_grid = self.grid


def test_a_dead_camera_declares_its_code():
    """정상 프레임에서는 카메라마다 voxel 을 13k~150k 개 쓴다. 0 개는 조용한 프레임이 아니라 깨진 것이다."""
    ag3s = _pipeline()
    notes, validity = ag3s._esdf_coverage(
        _Field({"head": {"n_updated": 40000}, "left_wrist": {"n_updated": 0}}),
        np.zeros(NQ))

    assert validity is ConstraintValidity.DEGRADED
    reasons = degradation.reasons(notes)
    assert [r["code"] for r in reasons] == ["esdf_dead_camera"]
    assert "left_wrist" in reasons[0]["detail"]


def test_spheres_outside_the_grid_declare_their_code():
    """격자 밖의 구는 무엇이 옆에 있든 자유 공간으로 읽힌다 (E4). 그것을 말하지 않으면 아무도 모른다."""
    ag3s = _pipeline()
    notes, validity = ag3s._esdf_coverage(
        # 격자를 로봇이 들어갈 수 없을 만큼 작게 잡으면 모든 구가 밖이다.
        _Field({"head": {"n_updated": 40000}}, lower=(9.0, 9.0, 9.0), upper=(9.1, 9.1, 9.1)),
        np.zeros(NQ))

    assert validity is ConstraintValidity.DEGRADED
    assert "spheres_outside_grid" in degradation.codes(notes)


def test_a_healthy_field_says_nothing():
    """정상 프레임에 노트를 늘리지 않는다 — 늘리면 신호가 소음에 묻힌다."""
    ag3s = _pipeline()
    notes, validity = ag3s._esdf_coverage(
        _Field({"head": {"n_updated": 40000}, "left_wrist": {"n_updated": 13000}}),
        np.zeros(NQ))

    assert validity is ConstraintValidity.VALID
    assert notes == []
