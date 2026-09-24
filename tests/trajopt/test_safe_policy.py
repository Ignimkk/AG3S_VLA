"""서버측 SafePolicy 의 계약. 합성 씬만 쓰므로 GPU 도 체크포인트도 필요 없다.

여기서 검사하는 것은 최적화 품질이 아니라 **계약**이다: 무엇이 왕복하는가, 실패했을 때 무엇이
나가는가, 그리퍼가 그대로인가. 그 셋이 깨지면 로봇이 잘못된 것을 실행하고, 세 가지 다 조용히
깨질 수 있는 종류다.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.runtime.pipeline import AG3S
from benchmark.trajopt import wire
from benchmark.trajopt.safe_policy import SafePolicy, grasp_parent_links


H, DOF = 50, wire.ACTION_WIDTH


class _StubPolicy:
    """`[H, wire.ACTION_WIDTH]` 를 돌려주는 최소 정책. reset 호출 여부를 센다."""

    def __init__(self, chunk=None):
        self.chunk = chunk if chunk is not None else np.zeros((H, DOF))
        self.resets = 0
        self.seen_keys = None

    def infer(self, obs, **_):
        self.seen_keys = set(obs)
        return {"actions": self.chunk.copy(), "policy_extra": 7}

    def reset(self):
        self.resets += 1

    @property
    def metadata(self):
        return {"stub": True}


def _robot_model():
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=[f"link_{s}_arm_{i}"
                                        for s in ("left", "right") for i in range(6)])


def _safe_policy(policy=None, **kw):
    model = _robot_model()
    # **attached 슬롯을 예약한다.** 실제 경로와 같아야 한다 — 슬롯은 AG3S 생성 시점에 고정되고
    # `attach()` 가 나중에 늘릴 수 없다. 이것 없이 지으면 `SafePolicy` 가 생성자에서 거절한다
    # (2026-09-18: 예약을 빠뜨리면 파지 다음 프레임부터 파이프라인이 조용히 멈췄다).
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf"}),
                robot_model=model, constraint_robot_model=model,
                attached_parent_links=grasp_parent_links())
    return SafePolicy(policy or _StubPolicy(), ag3s=ag3s, **kw)


def test_missing_attached_slots_is_refused_at_construction():
    """파지 **전에** 죽어야 한다 — 파지 뒤에 죽으면 조용하다."""
    model = _robot_model()
    bare = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf"}),
                robot_model=model, constraint_robot_model=model)
    with pytest.raises(ValueError, match="attached_parent_links"):
        SafePolicy(_StubPolicy(), ag3s=bare)


def test_certification_has_exactly_one_switch():
    """같은 뜻의 스위치가 둘이면 반쪽만 작동한다 — 실측으로 확인한 실패다."""
    sp = _safe_policy()
    assert not hasattr(sp, "require_certified")
    assert sp.to_config.safety.require_certified_geometry is True


# --- 요청 왕복 -----------------------------------------------------------------------------


def test_pack_strip_roundtrip_keeps_policy_obs_clean():
    """정책은 `ag3s/` 키를 한 번도 보면 안 된다 — 입력 변환이 모르는 키에 깨진다."""
    obs = {"state": np.zeros(DOF), "prompt": "pick"}
    request = wire.pack_request(
        obs, cameras=["zed_left"], depth={"zed_left": np.full((4, 4), 1.5)},
        intrinsics={"zed_left": np.eye(3)}, extrinsics={"zed_left": np.eye(4)},
        robot_state={"zed_left": np.zeros(12)}, stamps={"zed_left": 3.0},
        phase="approach", reset=True, seq=9)
    policy_obs, scene = wire.strip_request(request)

    assert set(policy_obs) == {"state", "prompt"}
    assert scene["seq"] == 9 and scene["reset"] is True
    assert scene["cameras"] == ["zed_left"]
    # depth 는 uint16 밀리미터로 오가고 되돌렸을 때 1 mm 안쪽이어야 한다.
    assert scene["depth/zed_left"].dtype == np.uint16
    assert np.allclose(scene["depth/zed_left"] / wire.DEPTH_SCALE_MM, 1.5, atol=1e-3)


def test_missing_camera_is_dropped_not_raised():
    """카메라 한 대가 빠지는 것은 정상이다. 예외를 내면 지각 결함이 정책을 죽인다."""
    scene = {"cameras": ["zed_left", "wrist_cam_l"],
             "depth/zed_left": np.ones((4, 4), np.uint16),
             "K/zed_left": np.eye(3), "T_base_cam/zed_left": np.eye(4),
             "robot_state/zed_left": np.zeros(12), "stamp/zed_left": 1.0}
    out = wire.unpack_camera_observations(scene, robot_model=None)
    assert len(out) == 1


# --- 응답 계약 -----------------------------------------------------------------------------


def test_shape_and_grippers_are_invariant():
    """`[H, ACTION_WIDTH]` 와 **그리퍼 열**은 무슨 일이 있어도 그대로다.

    열 번호를 박지 않는다 — 14D 는 (6, 13), 16D 는 (7, 15) 다. 박아 두면 차원을 바꿀 때
    이 테스트가 손목 관절을 그리퍼로 보호하라고 요구한다.
    """
    rng = np.random.default_rng(0)
    chunk = rng.normal(size=(H, DOF)) * 0.1
    policy = _StubPolicy(chunk)
    sp = _safe_policy(policy)
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})

    assert out["actions"].shape == (H, DOF)
    for col in wire.GRIPPER_COLUMNS:
        np.testing.assert_allclose(out["actions"][:, col], chunk[:, col], rtol=0, atol=1e-6)


def test_no_camera_is_reported_unsafe_with_the_chunk_still_present():
    """씬을 못 얻으면 실행 불가 판정이되, 무엇이 제안됐는지는 보여야 한다."""
    sp = _safe_policy()
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": [], wire.PREFIX + "seq": 4})

    assert out["safe"] is False
    assert out["geometry_certified"] is False
    assert out["seq"] == 4
    assert out["actions"].shape == (H, DOF)          # 청크는 빠지지 않는다
    assert out["max_violation_m"] == float("inf")
    assert any("unverified" in n for n in out["notes"])


def test_timing_and_policy_extras_survive():
    sp = _safe_policy()
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})
    assert set(out["timing_ms"]) >= {"infer", "total"}
    assert out["policy_extra"] == 7                   # 정책이 낸 다른 키는 지워지지 않는다


# --- 리셋 -----------------------------------------------------------------------------------


def test_reset_flag_clears_every_layer():
    """SEAM·AG3S·TO 중 하나라도 남으면 다음 에피소드가 지난 씬을 warm-start 로 받는다."""
    policy = _StubPolicy()
    sp = _safe_policy(policy)
    sp._previous_chunk = np.ones((H, DOF))
    sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": [], wire.PREFIX + "reset": True})

    assert policy.resets == 1
    # 리셋 후 첫 청크가 연속성 기준이 되므로 이전 청크는 반드시 버려져야 한다.
    assert sp._previous_chunk is not None            # 이번 프레임 결과로 다시 채워진 것
    sp.reset()
    assert sp._previous_chunk is None


def test_metadata_reports_the_constraint_model():
    sp = _safe_policy()
    meta = sp.metadata
    assert meta["safe_policy"] is True
    assert meta["stub"] is True                       # 감싼 정책의 메타를 지우지 않는다
    assert meta["constraint_spheres"] > 0


# --- 정책 실패 --------------------------------------------------------------------------------


def test_policy_failure_propagates():
    """정책이 죽으면 감춰서는 안 된다. 안전 계층이 조용히 대신 답하면 아무도 모른다."""

    class _Broken(_StubPolicy):
        def infer(self, obs, **_):
            raise RuntimeError("checkpoint gone")

    sp = _safe_policy(_Broken())
    with pytest.raises(RuntimeError, match="checkpoint gone"):
        sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})
