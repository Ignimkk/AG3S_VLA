"""로컬 클라이언트의 안전 게이트. 검증 목록의 여섯 시나리오를 여기서 고정한다.

이 파일이 지키는 불변량은 하나다: **확인된 것이 없으면 실행하지 않는다.** unsafe·timeout·
오래된 응답·서버 오류 네 갈래가 전부 같은 결과(hold)로 수렴해야 하고, 그중 하나라도 빠지면
그 경로는 "검증되지 않은 청크를 실행" 으로 끝난다.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.client import SafeRemoteClient


#: 차원은 `wire` 에서 가져온다. 14 를 박으면 16D 전환에서 이 테스트가 옛 계약을
#: 지키라고 요구하게 된다 — 2026-09-24 에 실제로 그랬다.
H, DOF = 50, wire.ACTION_WIDTH


class _Frame:
    def __init__(self, q):
        self.depth = np.full((8, 8), 1.2)
        self.camera_intrinsics = np.eye(3)
        self.T_base_cam = np.eye(4)
        self.robot_state = np.asarray(q, np.float64)


class _Scene:
    """`TransportScene` 중 클라이언트가 실제로 쓰는 것만."""

    def __init__(self):
        self.q = np.arange(DOF, dtype=np.float64) * 0.01
        self.captures = 0

    def capture(self, _camera):
        self.captures += 1
        return _Frame(self.q)

    def robot_state(self):
        return self.q


class _Server:
    """`WebsocketClientPolicy` 자리. 응답을 시나리오별로 흉내낸다.

    `reference` 를 주면 서버가 `--shadow` 로 떠 있는 것을 흉내낸다 — 응답에
    `actions_reference` 가 실린다. `metadata` 를 주면 `get_server_metadata` 가 생겨
    접속 시점 검사까지 흉내낸다 (실제 `WebsocketClientPolicy` 는 그 메서드를 갖는다).
    """

    def __init__(self, *, safe=True, seq_offset=0, delay=0.0, raises=None,
                 actions=None, reference=None, metadata=None, ag3s=None, pair=None,
                 **verdict):
        self.safe, self.seq_offset, self.delay, self.raises = safe, seq_offset, delay, raises
        self.actions = actions
        self.reference = reference
        self.metadata = metadata
        self.ag3s = ag3s
        #: `max_violation_pair` — 위반을 만든 행의 신원. `None` 이면 키가 안 실린다 (옛 서버).
        self.pair = pair
        self.verdict = verdict
        self.requests = []
        if metadata is not None:
            self.get_server_metadata = lambda: dict(metadata)

    def infer(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.delay:
            time.sleep(self.delay)
        actions = self.actions if self.actions is not None else np.zeros((H, DOF))
        out = {
            "actions": np.asarray(actions),
            "seq": int(request[wire.PREFIX + "seq"]) + self.seq_offset,
            "safe": self.safe,
            "ag3s_status": self.verdict.get("ag3s_status", "ok"),
            "geometry_certified": self.verdict.get("geometry_certified", True),
            "trajopt_status": self.verdict.get("trajopt_status", "optimal"),
            "max_violation_m": self.verdict.get("max_violation_m", 0.0),
            "timing_ms": {"total": 12.0},
            "notes": [],
        }
        if self.reference is not None:
            out[wire.ACTIONS_REFERENCE] = np.asarray(self.reference)
        if self.ag3s is not None:
            out[wire.AG3S_BLOCK] = dict(self.ag3s)
        if self.pair is not None:
            out[wire.VIOLATION_PAIR] = dict(self.pair)
        return out


def _client(server, **kw):
    return SafeRemoteClient(policy=server, scene=_Scene(),
                            cameras=("zed_left", "wrist_cam_l"), **kw)


# --- 1. 정상 장면 ----------------------------------------------------------------------------


def test_safe_response_is_executed():
    client = _client(_Server(safe=True))
    out = client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is True
    assert out["actions"].shape == (H, DOF)
    assert client.stats["safe"] == 1


def test_request_carries_every_camera_field():
    """카메라마다 자기 robot_state 와 촬영 시각이 실려야 한다 — 손목은 팔과 함께 움직인다."""
    server = _Server()
    client = _client(server)
    client.infer({"state": np.zeros(DOF)}, reset=True)
    req = server.requests[0]

    assert req[wire.PREFIX + "cameras"] == ["zed_left", "wrist_cam_l"]
    assert req[wire.PREFIX + "reset"] is True
    for cam in ("zed_left", "wrist_cam_l"):
        for field in ("depth", "K", "T_base_cam", "robot_state", "stamp"):
            assert f"{wire.PREFIX}{field}/{cam}" in req


# --- 2. TO 실패 / geometry incomplete ---------------------------------------------------------


def test_unsafe_verdict_holds():
    client = _client(_Server(safe=False, trajopt_status="violated", max_violation_m=0.031))
    client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    assert "violated" in client.last_reason and "31.0 mm" in client.last_reason
    assert client.stats["unsafe"] == 1


def test_uncertified_geometry_explains_itself_differently():
    """`violated` 에는 두 갈래가 있다. 이유 문장이 그것을 구분해야 한다."""
    client = _client(_Server(safe=False, geometry_certified=False, ag3s_status="degraded",
                             trajopt_status="violated"))
    client.infer({"state": np.zeros(DOF)})

    assert "could not certify" in client.last_reason
    assert "nobody saw" in client.last_reason


# --- 3. timeout ------------------------------------------------------------------------------


def test_late_response_holds_even_when_safe():
    """늦은 응답은 지나간 자세를 위한 계획이다. safe 라고 해도 실행하지 않는다."""
    client = _client(_Server(safe=True, delay=0.05), timeout_s=0.01)
    client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    assert client.stats["timeout"] == 1
    assert "limit" in client.last_reason


# --- 4. 오래된 응답 --------------------------------------------------------------------------


def test_stale_sequence_holds():
    client = _client(_Server(safe=True, seq_offset=-1))
    client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    assert client.stats["stale"] == 1


# --- 5. 서버 오류 ----------------------------------------------------------------------------


def test_server_exception_holds_and_returns_a_hold_chunk():
    """예외가 밖으로 나가면 제어 루프가 죽고, 팔은 마지막 ctrl 에 매달린다."""
    scene = _Scene()
    client = SafeRemoteClient(policy=_Server(raises=ConnectionError("socket closed")),
                              scene=scene, cameras=("zed_left",))
    out = client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    assert client.stats["error"] == 1
    assert out["actions"].shape == (H, DOF)
    # hold 청크의 모든 행이 현재 자세여야 한다 — 정지가 아니라 유지다.
    np.testing.assert_allclose(out["actions"], np.tile(scene.q, (H, 1)))


def test_bad_action_shape_holds():
    client = _client(_Server(safe=True, actions=np.zeros((H, 7))))
    client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    assert f"expected [H, {wire.ACTION_WIDTH}]" in client.last_reason


# --- 6. 카메라 누락 --------------------------------------------------------------------------


def test_camera_capture_failure_holds():
    class _BrokenScene(_Scene):
        def capture(self, camera):
            raise RuntimeError("renderer died")

    client = SafeRemoteClient(policy=_Server(), scene=_BrokenScene(), cameras=("zed_left",))
    out = client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    assert "could not capture" in client.last_reason
    assert out["actions"].shape == (H, DOF)


# --- 상태 유지 -------------------------------------------------------------------------------


def test_last_safe_starts_false():
    """첫 응답이 오기 전에도 이 값이 읽힌다. 기본이 True 면 그 프레임이 무검증으로 실행된다."""
    client = _client(_Server())
    assert client.last_safe is False


def test_one_unsafe_frame_does_not_poison_the_next():
    server = _Server(safe=False)
    client = _client(server)
    client.infer({"state": np.zeros(DOF)})
    assert client.last_safe is False

    server.safe = True
    client.infer({"state": np.zeros(DOF)})
    assert client.last_safe is True


# --- 7. shadow (T5) ---------------------------------------------------------------------------
#
# shadow 는 *"전부 계산하되 수정된 청크를 로봇에 보내지 않는"* 실행이다. 여기서 고정하는 것은
# 셋이다: (a) 판정이 unsafe 여도 멈추지 않지만 **판정 자체는 그대로 남는다**, (b) 실행되는 것은
# reference 이고 그 사실이 기록에 남는다, (c) **짝이 안 맞으면 즉시 죽는다** — 조용히 refined 를
# 실행하면 shadow 가 아닌데 shadow 라고 기록된다.

#: reference 와 refined 를 구별할 수 있게 값을 다르게 둔다. 같은 배열로 시험하면 "reference 를
#: 실행했다" 와 "refined 를 실행했다" 가 통과에서 구별되지 않는다.
REFINED = np.full((H, DOF), 2.0)
REFERENCE = np.full((H, DOF), 1.0)


def _shadow_server(**kw):
    kw.setdefault("actions", REFINED)
    kw.setdefault("reference", REFERENCE)
    return _Server(**kw)


def test_shadow_executes_the_reference_not_the_refined_chunk():
    client = _client(_shadow_server(safe=True), shadow=True)
    out = client.infer({"state": np.zeros(DOF)})

    np.testing.assert_allclose(out["actions"], REFERENCE)
    # refined 를 조용히 버리지 않는다 — 호출부가 무엇을 받았는지 알 수 있어야 한다.
    np.testing.assert_allclose(out["actions_refined"], REFINED)
    assert out["executed_chunk"] == "reference"
    assert client.last_executed_chunk == "reference"
    assert client.should_execute is True
    assert client.last_safe is True


def test_shadow_does_not_hold_on_an_unsafe_verdict_but_keeps_the_verdict():
    """멈추면 에피소드가 서고 볼 것이 없어진다. 그래도 판정은 **거짓말하지 않는다.**"""
    client = _client(_shadow_server(safe=False, trajopt_status="violated",
                                    max_violation_m=0.031), shadow=True)
    out = client.infer({"state": np.zeros(DOF)})

    assert client.should_execute is True            # 실행된다
    np.testing.assert_allclose(out["actions"], REFERENCE)
    assert client.last_safe is False                # 판정은 그대로 unsafe
    assert "violated" in client.last_reason and "31.0 mm" in client.last_reason
    assert client.last_ipc == "unsafe"
    assert client.stats["unsafe"] == 1
    # 프레임을 셀 수 있어야 한다 — 몇 프레임이 unsafe 인데 움직였나.
    assert client.stats["shadow_override"] == 1


def test_shadow_without_a_server_reference_fails_immediately():
    """가장 비싼 버그를 막는 검사. 조용히 refined 를 실행하면 기록이 거짓이 된다."""
    client = _client(_Server(safe=True), shadow=True)
    with pytest.raises(RuntimeError, match="Restart the server with --shadow"):
        client.infer({"state": np.zeros(DOF)})


def test_a_shadow_server_without_a_shadow_client_fails_immediately():
    """반대 방향도 기록을 거짓으로 만든다 — 로봇은 닫힌 고리인데 서버 기록은 shadow 다."""
    client = _client(_shadow_server(safe=True), shadow=False)
    with pytest.raises(RuntimeError, match="--safe-shadow is off"):
        client.infer({"state": np.zeros(DOF)})


def test_mismatched_pair_is_caught_at_construction_when_metadata_exists():
    """실제 전송 계층은 접속 즉시 메타데이터를 준다. 그때 죽으면 로봇이 아직 안 움직였다."""
    with pytest.raises(RuntimeError, match="Restart the server with --shadow"):
        _client(_Server(metadata={"safe_policy": True}), shadow=True)
    with pytest.raises(RuntimeError, match="--safe-shadow is off"):
        _client(_Server(metadata={"safe_policy": True, "shadow": True}), shadow=False)
    # 짝이 맞으면 조용히 지나간다.
    _client(_Server(metadata={"safe_policy": True, "shadow": True}), shadow=True)
    _client(_Server(metadata={"safe_policy": True}), shadow=False)


@pytest.mark.parametrize("kw,kind", [
    ({"delay": 0.05}, "timeout"),
    ({"seq_offset": -1}, "stale"),
    ({"raises": ConnectionError("socket closed")}, "error"),
])
def test_shadow_still_holds_on_every_ipc_failure(kw, kind):
    """이 넷은 "위험하다" 가 아니라 "신뢰할 근거가 없다" 다 — 원본도 근거가 없다."""
    client = _client(_shadow_server(**kw), shadow=True, timeout_s=0.01)
    client.infer({"state": np.zeros(DOF)})

    assert client.should_execute is False
    assert client.last_executed_chunk == "none"
    assert client.stats[kind] == 1


def test_shadow_holds_when_the_two_chunks_disagree_on_shape():
    """어느 쪽이 어느 스텝인지 알 수 없다. 형태는 맞고 뜻은 틀린 실패를 실행하지 않는다."""
    client = _client(_shadow_server(reference=np.zeros((H - 1, DOF))), shadow=True)
    client.infer({"state": np.zeros(DOF)})

    assert client.should_execute is False
    assert wire.ACTIONS_REFERENCE in client.last_reason


def test_shadow_starts_held_like_everything_else():
    client = _client(_shadow_server(), shadow=True)
    assert client.last_executed_chunk == "none" and client.should_execute is False


# --- 기본 동작이 바뀌지 않았다 -----------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    {},                                              # safe
    {"safe": False},                                 # unsafe
    {"safe": False, "geometry_certified": False},     # uncertified
    {"delay": 0.05},                                 # timeout
    {"seq_offset": -1},                              # stale
    {"raises": ConnectionError("gone")},              # error
    {"actions": np.zeros((H, 7))},                   # 차원 불일치
])
def test_without_shadow_should_execute_is_exactly_last_safe(kw):
    """새 게이트가 옛 게이트와 한 프레임도 다르면 안 된다 — 회귀 기준선이 여기에 달려 있다."""
    client = _client(_Server(**kw), timeout_s=0.01)
    client.infer({"state": np.zeros(DOF)})

    assert client.should_execute == client.last_safe
    assert client.last_executed_chunk == ("refined" if client.last_safe else "none")


def test_without_shadow_the_response_is_returned_untouched():
    """shadow 가 아니면 응답 객체를 **그대로** 돌려준다 — 키 하나도 더하거나 빼지 않는다."""
    server = _Server(safe=True)
    client = _client(server)
    sent = {}

    original_infer = server.infer

    def _capture(request):
        out = original_infer(request)
        sent["keys"] = set(out)
        sent["obj"] = out
        return out

    server.infer = _capture
    out = client.infer({"state": np.zeros(DOF)})

    assert out is sent["obj"]                     # 복사도 하지 않는다
    assert set(out) == sent["keys"]
    assert wire.ACTIONS_REFERENCE not in out
    assert "actions_refined" not in out and "executed_chunk" not in out
    assert "shadow_override" not in client.stats


# --- 8. T5b: HOLD 가 **왜** 인지를 같이 말한다 ---------------------------------------------
#
# 첫 live smoke 의 HOLD 한 줄은 `"AG3S could not certify the geometry (status=degraded)"` 까지만
# 나왔다. 거기서 멈추면 읽는 사람이 서버에 다시 들어가야 하는데 **서버 로그에도 없었다.**


def _degraded_server(**kw):
    from AG3S.ag3s.runtime import degradation

    note = degradation.reason(
        "spheres_outside_grid",
        "25 of 120 constraint sphere(s) lie outside the ESDF grid (base, wheel_l)")
    kw.setdefault("ag3s", {
        "status": "degraded", "validity": "degraded", "grounding_status": "low_score",
        "reasons": degradation.reasons([note]), "notes": [note]})
    return _Server(safe=False, geometry_certified=False, ag3s_status="degraded",
                   trajopt_status="violated", max_violation_m=0.0, **kw)


def test_the_hold_reason_names_the_degradation_site():
    client = _client(_degraded_server())
    client.infer({"state": np.zeros(DOF)})

    assert client.last_safe is False
    # 어느 분기인지 — 코드가 그것을 말한다.
    assert "spheres_outside_grid" in client.last_reason
    # 수치도 — 없으면 다시 서버에 들어가서 재야 한다.
    assert "25 of 120" in client.last_reason
    # grounding 상태도. `degraded` 만으로는 target 을 못 잡은 것인지 알 수 없다.
    assert "low_score" in client.last_reason
    # 옛 문장의 뜻은 지운 것이 아니라 덧붙인 것이다.
    assert "nobody saw" in client.last_reason


def test_the_block_is_kept_for_the_record_even_on_a_hold():
    """control frame 이 왜 멈췄는지를 적으려면 그 프레임의 지각 사유가 필요하다."""
    client = _client(_degraded_server())
    client.infer({"state": np.zeros(DOF)})

    assert client.last_ag3s["grounding_status"] == "low_score"
    assert [r["code"] for r in client.last_ag3s["reasons"]] == ["spheres_outside_grid"]
    # `last_verdict` 의 키 집합은 건드리지 않는다 — T0 기록의 `verdict` 가 그 모양이다.
    assert set(client.last_verdict) == {"ag3s_status", "geometry_certified", "trajopt_status",
                                        "max_violation_m", "timing_ms", "notes"}


def test_an_old_server_without_the_block_says_so_rather_than_going_quiet():
    """빈칸으로 두면 "사유 없는 degraded" 와 "블록을 모르는 서버" 가 구별되지 않는다."""
    client = _client(_degraded_server(ag3s=None))
    client.infer({"state": np.zeros(DOF)})

    assert "predates the `ag3s` block" in client.last_reason
    assert client.last_ag3s == {}


def test_a_certified_frame_keeps_the_block_empty():
    """정상 프레임에는 키가 안 온다 — 로컬 상태도 비어 있어야 한다."""
    client = _client(_Server(safe=True))
    client.infer({"state": np.zeros(DOF)})

    assert client.last_ag3s == {}


def test_a_response_with_no_block_does_not_inherit_the_previous_frame_reason():
    """지난 프레임의 사유가 남으면 이 프레임이 그 사유로 멈춘 것처럼 읽힌다."""
    server = _degraded_server()
    client = _client(server)
    client.infer({"state": np.zeros(DOF)})
    assert client.last_ag3s

    # 다음 프레임은 응답이 아예 없다 (서버 오류).
    server.raises = ConnectionError("socket closed")
    client.infer({"state": np.zeros(DOF)})
    assert client.last_ag3s == {}


# --- 8. T6a — 두 청크와 위반 행의 신원을 기록에 넘긴다 ----------------------------------------
#
# `T5c` 는 `refined` 대 `reference` 여유거리 비교를, `T5f` 는 *"violated 가 어느 제약인가"* 를
# 각각 미측정으로 닫았다. 둘 다 클라이언트가 그 값을 붙들지 않아 기록에 넘길 것이 없었기
# 때문이다. 여기서 고정하는 것은 **모드가 달라도 `last_actions_refined` 의 뜻이 하나**라는
# 것이다 — shadow 에서 `infer` 의 반환값은 `actions` 자리에 reference 를 싣는다.

_PAIR = {"block": "esdf", "clearance_m": -0.031, "step": 12, "query": 5,
         "link": "ee_finger_r1", "slot": None, "candidate_id": None,
         "obstacle": "apple", "point_m": [0.51, 0.09, 0.84]}


def test_closed_loop_carries_the_refined_chunk_and_no_reference():
    refined = np.full((H, DOF), 0.25)
    client = _client(_Server(safe=True, actions=refined, pair=_PAIR))
    client.infer({"state": np.zeros(DOF)})

    np.testing.assert_allclose(client.last_actions_refined, refined)
    # reference 없음은 **shadow 가 아니다** 라는 뜻이다.
    assert client.last_actions_reference is None
    assert client.last_violation_pair == _PAIR


def test_shadow_keeps_both_chunks_apart():
    """`infer` 의 반환값은 reference 를 실행하지만, 기록에는 둘이 따로 남아야 한다."""
    refined = np.full((H, DOF), 0.25)
    reference = np.full((H, DOF), 0.75)
    client = _client(_Server(safe=True, actions=refined, reference=reference,
                             metadata={"shadow": True}, pair=_PAIR), shadow=True)
    out = client.infer({"state": np.zeros(DOF)})

    # 로봇이 받는 것은 reference 다 (기존 계약).
    np.testing.assert_allclose(out["actions"], reference)
    # 기록이 받는 것은 둘 다이고, `last_actions_refined` 는 **서버가 계산한** 쪽이다.
    np.testing.assert_allclose(client.last_actions_refined, refined)
    np.testing.assert_allclose(client.last_actions_reference, reference)
    assert client.last_executed_chunk == "reference"


def test_an_old_server_without_the_pair_leaves_it_none():
    """키 없음은 "신원을 낼 수 없는 서버" 다. 지어내지 않는다."""
    client = _client(_Server(safe=False, trajopt_status="violated", max_violation_m=0.031))
    client.infer({"state": np.zeros(DOF)})

    assert client.last_violation_pair is None
    # 숫자는 그대로 온다 — 신원이 없어도 판정은 바뀌지 않는다.
    assert client.last_verdict["max_violation_m"] == 0.031


def test_a_hold_with_a_response_still_records_what_the_server_computed():
    """왜 거부됐는지는 무엇이 제안됐는지 없이 읽을 수 없다 (timeout 도 마찬가지다)."""
    refined = np.full((H, DOF), 0.25)
    client = _client(_Server(safe=True, actions=refined, delay=0.05, pair=_PAIR),
                     timeout_s=0.01)
    client.infer({"state": np.zeros(DOF)})

    assert client.last_ipc == "timeout"
    np.testing.assert_allclose(client.last_actions_refined, refined)
    assert client.last_violation_pair == _PAIR


def test_no_response_clears_the_chunks_and_the_pair():
    """지난 프레임 것을 남기면 이 프레임이 그것 때문에 멈춘 것처럼 읽힌다."""
    server = _Server(safe=True, actions=np.full((H, DOF), 0.25), pair=_PAIR)
    client = _client(server)
    client.infer({"state": np.zeros(DOF)})
    assert client.last_actions_refined is not None

    server.raises = ConnectionError("socket closed")
    client.infer({"state": np.zeros(DOF)})
    assert client.last_actions_refined is None
    assert client.last_actions_reference is None
    assert client.last_violation_pair is None


def test_the_verdict_key_set_is_untouched_by_the_pair():
    """신원은 `last_verdict` 에 들어가지 않는다 — 그 키 집합이 T0 기록의 `verdict` 다."""
    client = _client(_Server(safe=True, pair=_PAIR))
    client.infer({"state": np.zeros(DOF)})

    assert set(client.last_verdict) == {"ag3s_status", "geometry_certified", "trajopt_status",
                                        "max_violation_m", "timing_ms", "notes"}
