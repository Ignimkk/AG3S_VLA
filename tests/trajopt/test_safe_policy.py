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


# --- shadow (T5) — 서버는 전부 돌고, 응답에 원본 청크를 **선택 키**로 더한다 ------------------
#
# shadow 가 지켜야 하는 두 가지가 여기서 갈린다. (a) **켰을 때**: 로컬이 원본을 실행할 수 있도록
# `actions_reference` 가 실리고, `actions` 는 **여전히 refined** 다 (서버가 거짓말하지 않는다).
# (b) **안 켰을 때**: 응답이 T0 때와 키 하나도 다르지 않다 — 회귀 기준선과 T0 기록이 그것에
# 달려 있다.

#: 모든 응답에 있는 키. 여기에 무엇이 더 생기면 T0 기록과 갈라진다.
#:
#: **2026-09-26 (T9): `shadow` 가 들어왔다.** 모드가 `actions_reference` 의 있음/없음으로
#: 추론되던 자리다. 그 겸직 때문에 closed loop 이 원본 청크를 실을 수 없었고 (실으면 로컬이
#: "서버가 shadow" 로 읽고 즉시 죽는다), *"TO 가 청크를 얼마나 바꿨나"* 를 닫힌 고리에서 잴 수
#: 없었다 — 같은 결핍에 다섯 번 막혔다. 지금은 모드가 이 키로 **명시**된다.
BASE_RESPONSE_KEYS = {"actions", "seq", "timing_ms", "field", "ag3s_status",
                      "geometry_certified", "trajopt_status", "max_violation_m",
                      "safe", "notes", "shadow"}


def _verdict(safe=True):
    return wire.SafetyVerdict(ag3s_status="ok", geometry_certified=True,
                              trajopt_status="optimal", max_violation_m=0.0, safe=safe)


def test_response_without_a_reference_is_what_it_always_was():
    out = wire.pack_response(np.zeros((H, DOF)), _verdict(), seq=3, timing_ms={"total": 1.0})

    assert set(out) == BASE_RESPONSE_KEYS
    # 없음은 **데이터의 부재**다 (T9). 모드는 `shadow` 키가 따로 말한다.
    assert wire.unpack_actions_reference(out) is None
    assert wire.unpack_shadow(out) is False


def test_reference_roundtrips_and_actions_stay_refined():
    refined, reference = np.full((H, DOF), 2.0), np.full((H, DOF), 1.0)
    out = wire.pack_response(refined, _verdict(), seq=3, timing_ms={},
                             actions_reference=reference)

    assert set(out) == BASE_RESPONSE_KEYS | {wire.ACTIONS_REFERENCE}
    np.testing.assert_allclose(out["actions"], refined)       # 서버는 자기가 계산한 것을 말한다
    np.testing.assert_allclose(wire.unpack_actions_reference(out), reference)
    # 전송은 msgpack_numpy 다 — `actions` 와 같은 float32 ndarray 여야 그대로 건너간다.
    assert isinstance(out[wire.ACTIONS_REFERENCE], np.ndarray)
    assert out[wire.ACTIONS_REFERENCE].dtype == np.float32


def test_a_reference_of_a_different_shape_is_refused_at_pack_time():
    """로컬이 둘 중 하나를 골라 실행한다. 모양이 어긋나면 한 칸씩 밀린 청크가 실행된다."""
    with pytest.raises(ValueError, match="actions_reference"):
        wire.pack_response(np.zeros((H, DOF)), _verdict(), seq=0, timing_ms={},
                           actions_reference=np.zeros((H - 1, DOF)))


def _shadow_policy(chunk, refine_offset=0.5, **kw):
    """`refined != reference` 인 SafePolicy. 합성 씬에서는 refiner 가 청크를 그대로 통과시켜
    둘이 같아지므로, 그 상태로는 "무엇이 실렸나" 를 구별할 수 없다."""
    sp = _safe_policy(_StubPolicy(chunk), **kw)
    sp.refiner.refine = lambda c, context=None: np.asarray(c, np.float64) + refine_offset
    return sp


def test_shadow_carries_the_policy_chunk_next_to_the_refined_one():
    rng = np.random.default_rng(1)
    chunk = rng.normal(size=(H, DOF)) * 0.1
    sp = _shadow_policy(chunk, shadow=True)
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})

    assert sp.shadow is True
    reference = wire.unpack_actions_reference(out)
    np.testing.assert_allclose(reference, chunk, rtol=0, atol=1e-6)
    # `actions` 는 여전히 refined 다 — 그리퍼 열만 정책 값으로 되돌려진다.
    for col in range(DOF):
        if col in wire.GRIPPER_COLUMNS:
            continue
        np.testing.assert_allclose(out["actions"][:, col], chunk[:, col] + 0.5,
                                   rtol=0, atol=1e-6)


def test_closed_loop_also_carries_the_reference_but_declares_itself_not_shadow():
    """**T9 에서 뒤집힌 동작이다.** 예전에는 이 프레임에 `actions_reference` 가 없었다.

    닫힌 고리에서도 원본 청크가 실린다 — 충돌 제약이 하나도 활성이 아닌 판에서도 TO 가 청크를
    고치므로(실행 창 안 중앙값 2.6°, 최대 8.8°), 남은 변형이 목적함수에서 온 것인지 재려면
    `actions`(refined) 옆에 원본이 있어야 한다. **모드는 따로 말한다**: `shadow` 가 `False`
    이므로 로컬은 refined 를 실행한다.

    키 집합이 `BASE + policy_extra + ag3s + actions_reference` 인 것은 이 합성 씬의
    `ag3s_status` 가 `ok` 가 아니기 때문이다 (카메라가 없으므로 `no_geometry`).
    """
    sp = _shadow_policy(np.zeros((H, DOF)))
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})

    assert sp.shadow is False
    assert wire.unpack_shadow(out) is False, "모드는 명시돼야 한다"
    assert wire.unpack_actions_reference(out) is not None, "원본이 기록을 위해 실린다"
    # 정책이 낸 다른 키(`policy_extra`)는 예전처럼 그대로 실린다.
    assert set(out) == BASE_RESPONSE_KEYS | {
        "policy_extra", wire.AG3S_BLOCK, wire.ACTIONS_REFERENCE}


def test_metadata_advertises_shadow_only_when_it_is_on():
    """로컬이 접속 즉시 짝을 볼 수 있어야 한다. 꺼졌을 때 키가 없는 것은 메타데이터도
    T0 때와 같게 두기 위해서다."""
    assert "shadow" not in _safe_policy().metadata
    assert _safe_policy(shadow=True).metadata["shadow"] is True


def test_continuity_anchors_on_the_chunk_the_robot_actually_flew():
    """`_continuity_reference` 는 "앞 스텝은 이미 실행됐다" 를 전제로 꼬리를 쓴다. shadow 에서
    실행된 것은 reference 이므로 그것을 물려줘야 한다 — refined 를 물려주면 SQP 가 날아간 적
    없는 궤적에서 이어지는 것으로 계획하고, 그 오차는 어디에도 안 찍힌다."""
    chunk = np.full((H, DOF), 0.2)
    obs = {"state": np.zeros(DOF), wire.PREFIX + "cameras": []}

    closed = _shadow_policy(chunk)
    closed.infer(obs)
    np.testing.assert_allclose(closed._previous_chunk, chunk + 0.5)

    shadow = _shadow_policy(chunk, shadow=True)
    shadow.infer(obs)
    np.testing.assert_allclose(shadow._previous_chunk, chunk)


# --- T5b: `degraded` 의 사유가 와이어에 실린다 --------------------------------------------
#
# 첫 live smoke 에서 2 청크 중 1 개가 `max_violation_m = 0.0` 인데 `degraded` 로 HOLD 됐고,
# **왜인지가 응답에도 서버 로그에도 없었다.** 사유는 줄곧 `CollisionConstraintSet.notes` 에
# 있었고 나가는 길이 없었을 뿐이다 (응답의 `notes` 는 최적화기 쪽이다).


def test_an_ok_frame_carries_no_ag3s_block():
    """**요구사항 1.** 정상 프레임의 응답이 T0 때와 한 바이트도 달라지면 기준선이 재현되지 않는다."""
    out = wire.pack_response(np.zeros((H, DOF)), _verdict(), seq=1, timing_ms={}, ag3s=None)

    assert set(out) == BASE_RESPONSE_KEYS
    assert wire.AG3S_BLOCK not in out
    assert wire.unpack_ag3s(out) == {}


def test_the_ag3s_block_roundtrips_with_its_reasons():
    from benchmark.ag3s.runtime import degradation

    block = {"status": "degraded", "validity": "degraded", "grounding_status": "ok",
             "reasons": degradation.reasons(
                 [degradation.reason("camera_missing", "camera(s) ['left_wrist'] did not report")]),
             "notes": ["camera_missing: camera(s) ['left_wrist'] did not report"]}
    out = wire.pack_response(np.zeros((H, DOF)), _verdict(safe=False), seq=1, timing_ms={},
                             ag3s=block)

    assert set(out) == BASE_RESPONSE_KEYS | {wire.AG3S_BLOCK}
    got = wire.unpack_ag3s(out)
    assert got["reasons"] == [{"code": "camera_missing",
                               "detail": "camera(s) ['left_wrist'] did not report"}]
    assert got["grounding_status"] == "ok"


def test_the_ag3s_block_and_the_shadow_reference_do_not_collide():
    """T5a 와 T5b 의 선택 키 둘이 같은 응답에 함께 실릴 수 있어야 한다 — shadow 실행의 unsafe
    프레임이 정확히 그 조합이고, 그것이 T5 에서 보려는 프레임이다."""
    out = wire.pack_response(
        np.full((H, DOF), 2.0), _verdict(safe=False), seq=1, timing_ms={},
        actions_reference=np.full((H, DOF), 1.0),
        ag3s={"status": "degraded", "validity": "degraded", "grounding_status": "ok",
              "reasons": [], "notes": []})

    assert set(out) == BASE_RESPONSE_KEYS | {wire.ACTIONS_REFERENCE, wire.AG3S_BLOCK}
    np.testing.assert_allclose(wire.unpack_actions_reference(out), 1.0)
    assert wire.unpack_ag3s(out)["status"] == "degraded"


def test_a_non_ok_frame_explains_itself_on_the_wire():
    """서버 밖에서 `degraded` 의 이유를 알 수 있어야 한다. 이 STEP 의 합격 조건이다."""
    sp = _safe_policy()
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": [], wire.PREFIX + "seq": 2})

    assert out["ag3s_status"] != "ok"
    block = wire.unpack_ag3s(out)
    assert block["status"] == out["ag3s_status"]        # 블록만 보고도 짝을 확인할 수 있다
    # 카메라가 하나도 없으면 제약 집합 자체가 없다. 그 사실과 **왜 지각이 안 돌았는지**를
    # 말해야 한다 — `status: no_geometry` 만 나가면 이유는 다시 서버 안에 남는다.
    assert block["notes"] and "no constraint set" in block["notes"][0]
