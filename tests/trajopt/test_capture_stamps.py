"""T30c (F2) — 시뮬레이션 관측 하나에는 **촬영 순간이 하나**다.

T28 E3b 의 `uncertified` HOLD 125 청크가 `camera_transform_stale` 이었다. 옛 `_pack` 은 카메라마다
**렌더가 끝난 순간**을 찍었고 osmesa 는 세 카메라를 차례로 렌더하므로 head 가 wrist 보다 102–142 ms
앞섰다. 그동안 시뮬레이션은 정지해 있다 — 씬은 한 순간이다.

이 파일이 박는 것:

1. `capture_time` 을 주면 한 관측의 `ag3s/stamp/<cam>` 이 **모두 같다** (그 순간). 렌더 벽시계는
   `ag3s/render_stamp/<cam>` 에 **실제 퍼짐 그대로** 남는다 (`stamp_mode = sim_frozen`).
2. 주지 않으면 T30c 전 동작 그대로 (`render_end`, stamps = render_stamps) — 실기용 기본값.
3. 서버의 검사(`check_freshness`)는 **그대로**이고, 같은 렌더 퍼짐에서 옛 스탬프는 stale 을, 새
   스탬프는 아무 사유도 내지 않는다.
4. 미래의 `capture_time` 은 hold (조용히 싣지 않는다).
5. `FrameRecorder.observation` 의 새 키 (주었을 때만).
6. `pi05_infer.py` 가 `build_obs` **직전**에 정지 순간을 재고 그것을 `infer` 에 넘긴다 (소스).
"""

from __future__ import annotations

import json
import pathlib
import re
import time

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.runtime.degradation import reasons
from benchmark.ag3s.runtime.frame_record import FrameRecorder
from benchmark.ag3s.runtime.multiview import check_freshness
from benchmark.trajopt import wire
from benchmark.trajopt.client import SafeRemoteClient
from tests.trajopt.test_safe_client import DOF, _Server

PI05_INFER = pathlib.Path(__file__).resolve().parents[2] / "pi05_TO_hybrid" / "rby1_bringup" \
    / "pi05_infer.py"

CAMERAS = wire.DEFAULT_CAMERAS
#: 카메라 하나의 렌더 시간. 세 대면 head ↔ wrist_r 퍼짐이 2 × 이 값 ≥ 120 ms — E3b 중앙값(121 ms)
#: 이고 `timing.max_transform_age_sec` (100 ms) 를 넘는다. `sleep` 은 **적어도** 이만큼 자므로 하한은
#: 흔들리지 않는다.
RENDER_SEC = 0.060


class _Frame:
    def __init__(self, q):
        self.depth = np.full((8, 8), 1.2)
        self.camera_intrinsics = np.eye(3)
        self.T_base_cam = np.eye(4)
        self.robot_state = np.asarray(q, np.float64)


class _SlowScene:
    """정지한 씬을 **차례로** 렌더한다 (osmesa 처럼). 카메라 하나에 `RENDER_SEC`."""

    def __init__(self):
        self.q = np.arange(DOF, dtype=np.float64) * 0.01
        self.captured = []

    def capture(self, camera):
        time.sleep(RENDER_SEC)
        self.captured.append(camera)
        return _Frame(self.q)

    def robot_state(self):
        return self.q


def _client(server=None):
    server = server or _Server(safe=True)
    return SafeRemoteClient(policy=server, scene=_SlowScene(), cameras=CAMERAS), server


def _scene_of(request):
    return wire.strip_request(request)[1]


def _codes(notes):
    return [r["code"] for r in reasons(notes)]


# --- 1. 한 관측 = 한 순간 ----------------------------------------------------------------------


def test_one_observation_has_one_capture_instant_and_keeps_the_render_spread():
    client, server = _client()
    frozen_at = time.monotonic()
    client.infer({"state": np.zeros(DOF)}, reset=True, capture_time=frozen_at)

    scene = _scene_of(server.requests[-1])
    got = wire.unpack_stamps(scene)
    assert got["mode"] == "sim_frozen"
    # 판정용 스탬프: 세 카메라가 **정확히** 같은 값 = 정지 순간.
    assert set(got["stamps"]) == set(CAMERAS)
    assert set(got["stamps"].values()) == {frozen_at}
    assert got["skew_sec"] == 0.0
    # 진단용 렌더 벽시계: 실제 순차 렌더의 퍼짐 그대로 (head 먼저, wrist_r 마지막).
    render = got["render_stamps"]
    assert list(render) == list(CAMERAS)
    assert render["zed_left"] < render["wrist_cam_l"] < render["wrist_cam_r"]
    assert got["render_spread_sec"] >= 2 * RENDER_SEC
    # 렌더는 정지 순간 **뒤**에 끝났다.
    assert min(render.values()) >= frozen_at + RENDER_SEC
    # 클라이언트가 붙드는 값 = 실어 보낸 값 (로컬 observation 기록이 이것을 읽는다).
    assert client.last_stamps == got["stamps"]
    assert client.last_render_stamps == render
    assert client.last_stamp_mode == "sim_frozen"


def test_state_and_extrinsics_ride_the_same_instant():
    """자세 · 외부 파라미터는 카메라별로 실리지만 `state_timestamp` 가 없으므로 그 카메라의 스탬프 =
    정지 순간이다. 서버가 읽는 관측에서 셋이 한 순간인지 본다."""
    client, server = _client()
    frozen_at = time.monotonic()
    client.infer({"state": np.zeros(DOF)}, capture_time=frozen_at)

    obs = wire.unpack_camera_observations(_scene_of(server.requests[-1]), robot_model=None)
    assert len(obs) == len(CAMERAS)
    assert {o.timestamp for o in obs} == {frozen_at}
    assert {o.state_time for o in obs} == {frozen_at}
    for o in obs:
        np.testing.assert_array_equal(o.robot_state, obs[0].robot_state)
        np.testing.assert_array_equal(o.T_base_cam, obs[0].T_base_cam)


def test_without_capture_time_the_old_per_camera_render_stamps_stay():
    """실기 기본값 — 순간을 주장하지 않는 호출자는 T30c 전과 같다."""
    client, server = _client()
    client.infer({"state": np.zeros(DOF)})

    got = wire.unpack_stamps(_scene_of(server.requests[-1]))
    assert got["mode"] == "render_end"
    assert got["stamps"] == got["render_stamps"]
    assert got["skew_sec"] >= 2 * RENDER_SEC
    assert client.last_stamp_mode == "render_end"


# --- 2. 서버 검사는 그대로, 거짓 stale 만 사라진다 -----------------------------------------------


def test_server_freshness_check_is_unchanged_and_the_false_stale_disappears():
    cfg = AG3SConfig.from_dict({})
    limit = cfg.timing.max_transform_age_sec
    assert limit == pytest.approx(0.1)      # 검사의 한도는 건드리지 않았다

    old_client, old_server = _client()
    old_client.infer({"state": np.zeros(DOF)})
    old_obs = wire.unpack_camera_observations(_scene_of(old_server.requests[-1]),
                                              robot_model=None)
    _, old_notes, old_metrics = check_freshness(old_obs, cfg)
    # 옛 스탬프: 같은 정지 씬인데 head 가 `camera_transform_stale` (E3b 의 거짓 사유).
    assert "camera_transform_stale" in _codes(old_notes)
    assert old_metrics["camera_skew_sec"] > limit and old_metrics["n_stale_transform"] >= 1

    new_client, new_server = _client()
    new_client.infer({"state": np.zeros(DOF)}, capture_time=time.monotonic())
    new_obs = wire.unpack_camera_observations(_scene_of(new_server.requests[-1]),
                                              robot_model=None)
    validity, new_notes, new_metrics = check_freshness(new_obs, cfg)
    assert _codes(new_notes) == []
    assert new_metrics["camera_skew_sec"] == 0.0 and new_metrics["n_stale_transform"] == 0
    assert validity.value == "valid"


def test_the_check_still_catches_a_real_skew():
    """실기에서 카메라가 정말로 150 ms 벌어지면 검사는 여전히 잡는다 — 검사를 약하게 하지 않았다."""
    cfg = AG3SConfig.from_dict({})
    request = wire.pack_request(
        {"state": np.zeros(DOF)}, cameras=CAMERAS,
        depth={c: np.full((8, 8), 1.2) for c in CAMERAS},
        intrinsics={c: np.eye(3) for c in CAMERAS}, extrinsics={c: np.eye(4) for c in CAMERAS},
        robot_state={c: np.zeros(DOF) for c in CAMERAS},
        stamps={"zed_left": 10.0, "wrist_cam_l": 10.15, "wrist_cam_r": 10.15},
        phase="approach")
    obs = wire.unpack_camera_observations(_scene_of(request), robot_model=None)
    _, notes, _ = check_freshness(obs, cfg)
    assert "camera_transform_stale" in _codes(notes)


# --- 3. 잘못된 순간 ------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_capture_time_holds(bad):
    client, server = _client()
    client.infer({"state": np.zeros(DOF)}, capture_time=bad)
    assert server.requests == []
    assert client.last_ipc == "error" and client.should_execute is False
    assert "capture_time" in client.last_reason
    assert client.last_stamps == {} and client.last_stamp_mode == "none"


def test_a_future_capture_time_holds_without_rendering():
    client, server = _client()
    client.infer({"state": np.zeros(DOF)}, capture_time=time.monotonic() + 5.0)
    assert server.requests == [] and client._scene.captured == []
    assert client.last_ipc == "error" and "capture_time" in client.last_reason
    # 실행 피드백은 캡처 전에 붙든다 (T18 규약 그대로).
    assert client.last_exec_feedback["available"] is False


# --- 4. 와이어 ----------------------------------------------------------------------------------


def _pack(**kw):
    return wire.pack_request(
        {"state": np.zeros(DOF)}, cameras=("zed_left",), depth={"zed_left": np.ones((2, 2))},
        intrinsics={"zed_left": np.eye(3)}, extrinsics={"zed_left": np.eye(4)},
        robot_state={"zed_left": np.zeros(DOF)}, stamps={"zed_left": 1.0}, phase="approach", **kw)


def test_old_callers_keep_the_old_key_set():
    old = _pack()
    assert not any(k.startswith(wire.PREFIX + "render_stamp/") for k in old)
    assert wire.PREFIX + wire.STAMP_MODE not in old
    got = wire.unpack_stamps(_scene_of(old))
    assert got["mode"] is None and got["render_stamps"] == {} and got["render_spread_sec"] is None

    new = _pack(render_stamps={"zed_left": 1.25}, stamp_mode="sim_frozen")
    assert set(new) - set(old) == {wire.PREFIX + "render_stamp/zed_left",
                                   wire.PREFIX + wire.STAMP_MODE}


def test_unknown_stamp_mode_is_refused():
    with pytest.raises(ValueError, match="stamp_mode"):
        _pack(stamp_mode="frozen")


# --- 5. 로컬 observation 기록 ---------------------------------------------------------------------


def test_frame_record_keeps_both_clocks(tmp_path):
    rec = FrameRecorder(str(tmp_path), manifest={"timing": {}}, expected_observation_frames=2)
    rec.observation(t_step=0, stamps={c: 5.0 for c in CAMERAS},
                    render_stamps={"zed_left": 5.06, "wrist_cam_l": 5.12, "wrist_cam_r": 5.18},
                    stamp_mode="sim_frozen")
    rec.observation(t_step=8, stamps={"zed_left": 6.0})
    rec.close()

    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    obs = [r for r in rows if r["kind"] == "observation"]
    assert obs[0]["camera_skew_sec"] == 0.0
    assert obs[0]["render_spread_sec"] == pytest.approx(0.12)
    assert obs[0]["stamp_mode"] == "sim_frozen"
    assert obs[0]["camera_render_stamps"]["wrist_cam_r"] == pytest.approx(5.18)
    # 주지 않으면 키가 생기지 않는다 (옛 기록과 같은 키 집합).
    assert not {"camera_render_stamps", "render_spread_sec", "stamp_mode"} & set(obs[1])


# --- 6. pi05_infer.py (소스) ---------------------------------------------------------------------


def test_pi05_infer_freezes_the_instant_before_build_obs_and_passes_it():
    src = PI05_INFER.read_text(encoding="utf-8")
    loop = src[src.index("for t_step in range(0, args.max_steps"):]
    frozen = loop.index('sim_clock["frozen_at"] = time.monotonic()')
    built = loop.index("obs = build_obs(")
    sent = loop.index("result = safe_client.infer(")
    stepped = loop.index("mujoco.mj_step(m, d)")
    # 정지 순간 → 정책 렌더 → AG3S 캡처(`infer`) → 다음 `mj_step`.
    assert frozen < built < sent < stepped
    call = loop[sent:loop.index(")", loop.index("capture_time=", sent)) + 1]
    assert 'capture_time=sim_clock["frozen_at"]' in call
    # 기록에 렌더 벽시계와 모드가 실린다.
    assert re.search(r"render_stamps=\(dict\(safe_client\.last_render_stamps\)", loop)
    assert "stamp_mode=(safe_client.last_stamp_mode" in loop
    # in-process 경로도 카메라마다 `time.monotonic()` 을 찍지 않는다.
    assert "timestamp=time.monotonic()" not in src
    assert "timestamp=frozen_at" in src
