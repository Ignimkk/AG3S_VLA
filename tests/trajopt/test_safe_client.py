"""로컬 클라이언트의 안전 게이트. 검증 목록의 여섯 시나리오를 여기서 고정한다.

이 파일이 지키는 불변량은 하나다: **확인된 것이 없으면 실행하지 않는다.** unsafe·timeout·
오래된 응답·서버 오류 네 갈래가 전부 같은 결과(hold)로 수렴해야 하고, 그중 하나라도 빠지면
그 경로는 "검증되지 않은 청크를 실행" 으로 끝난다.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from benchmark.trajopt import wire
from benchmark.trajopt.client import SafeRemoteClient


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
    """`WebsocketClientPolicy` 자리. 응답을 시나리오별로 흉내낸다."""

    def __init__(self, *, safe=True, seq_offset=0, delay=0.0, raises=None,
                 actions=None, **verdict):
        self.safe, self.seq_offset, self.delay, self.raises = safe, seq_offset, delay, raises
        self.actions = actions
        self.verdict = verdict
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.delay:
            time.sleep(self.delay)
        actions = self.actions if self.actions is not None else np.zeros((H, DOF))
        return {
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
