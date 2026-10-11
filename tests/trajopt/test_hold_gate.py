"""T23 — 로컬 게이트와 HOLD: 사유별로 실행/HOLD, HOLD 는 멈추되 명령을 삼키지 않는다.

T16 의 HOLD 는 매 스텝 `action = rby1_state()` 였다 — 측정 관절과 측정 gripper 개도를 목표로 다시
줬다. 그래서 (1) HOLD 청크의 닫힘 명령이 사라졌고 (t=176 · 208), (2) 팔이 0.10–0.17°/step 흘렀다
(t=72–92, 손가락 23 mm). 이 파일이 고정하는 것:

1. `SafeRemoteClient` 게이트 — 서버 `safe` 와 사유 표가 둘 다 실행일 때만 실행. 통신 실패는 `comms`.
2. `HoldController` — `q_hold` 는 HOLD 진입 때 **명령한** 팔 목표로 한 번, gripper 는 마지막 명령,
   HOLD 가 청크를 넘어도 같은 값. 연속 HOLD 한도를 넘으면 중단 사유.
3. `SafeRemoteClient._hold` 의 대체 청크 = 마지막 명령.
4. control frame 의 `hold` 키.

드리프트의 **실측**(MuJoCo, mm)은 verifier 의 몫이다. 여기의 1 차 servo 모형은 "측정값을 목표로 다시
주면 정착 오차가 쌓이고 고정 목표면 쌓이지 않는다" 는 기전만 박는다 — 숫자의 크기는 모형의 것이다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.client import HoldController, SafeRemoteClient, action_row
from tests.trajopt.test_safe_client import DOF, H, _client, _Scene, _Server

N = wire.ARM_JOINT_DIM


class _ReasonServer(_Server):
    """`verdict_reasons` 를 싣는 서버 (T23). `reasons=None` 이면 옛 서버와 같다."""

    def __init__(self, *, reasons=None, **kw):
        super().__init__(**kw)
        self.reasons = reasons

    def infer(self, request):
        out = super().infer(request)
        if self.reasons is not None:
            out[wire.VERDICT_REASONS] = [dict(r) for r in self.reasons]
        return out


ALLOWED = wire.make_reason("allowed_contact", "ee_finger_l1 ↔ manipulated id=0, -6.2 mm")
COLLISION = wire.make_reason("collision", "1 row(s): link_torso_0 ↔ table -5.0 mm",
                             row={"link": "link_torso_0"})


# =============================================================================================
# 1. 게이트
# =============================================================================================


def test_allowed_contact_executes_even_though_the_optimizer_said_violated():
    client = _client(_ReasonServer(safe=True, trajopt_status="violated", max_violation_m=0.006,
                                   reasons=[ALLOWED]))
    client.infer({"state": np.zeros(DOF)})

    assert client.should_execute and client.last_safe is True
    assert client.last_reasons == [ALLOWED] and client.last_reasons_source == "server"
    assert client.last_gate == {"mode": "reasons", "action": "execute", "source": "server",
                                "server_safe": True, "hold_kinds": []}
    assert client.reason_stats == {"allowed_contact": 1}


def test_a_collision_reason_holds_and_the_line_names_it():
    client = _client(_ReasonServer(safe=False, trajopt_status="violated", max_violation_m=0.012,
                                   reasons=[ALLOWED, COLLISION]))
    client.infer({"state": np.zeros(DOF)})

    assert not client.should_execute and client.last_ipc == "unsafe"
    assert client.last_gate["hold_kinds"] == ["collision"]
    assert client.last_reason.startswith("[collision] ")
    # 옛 문장(상태 · mm)은 지운 것이 아니라 덧붙인 것이다.
    assert "violated" in client.last_reason and "12.0 mm" in client.last_reason
    assert "link_torso_0" in client.last_reason


def test_a_stricter_local_table_holds_what_the_server_would_execute():
    strict = dict(wire.GATE_DEFAULT, allowed_contact="hold")
    client = _client(_ReasonServer(safe=True, trajopt_status="violated", reasons=[ALLOWED]),
                     gate_table=strict)
    client.infer({"state": np.zeros(DOF)})

    assert not client.should_execute
    assert client.last_gate["hold_kinds"] == ["allowed_contact"]


def test_legacy_gate_is_the_pre_t23_decision():
    """`--safe-gate legacy`: `trajopt_status` 가 optimal/feasible 일 때만 — 허용 접촉도 HOLD."""
    client = _client(_ReasonServer(safe=True, trajopt_status="violated", reasons=[ALLOWED]),
                     gate="legacy")
    client.infer({"state": np.zeros(DOF)})
    assert not client.should_execute

    client = _client(_ReasonServer(safe=True, trajopt_status="feasible",
                                   reasons=[wire.make_reason("budget_only")]), gate="legacy")
    client.infer({"state": np.zeros(DOF)})
    assert client.should_execute


def test_gate_mode_is_validated():
    with pytest.raises(ValueError, match="gate"):
        _client(_Server(), gate="lenient")


def test_the_server_wins_when_it_says_unsafe():
    """서버가 `legacy` 판정으로 떠서 `safe=False` 인데 사유는 실행 쪽 — 로컬은 더 너그러워지지 않는다."""
    client = _client(_ReasonServer(safe=False, trajopt_status="violated", reasons=[ALLOWED]))
    client.infer({"state": np.zeros(DOF)})

    assert not client.should_execute
    assert client.last_gate["hold_kinds"] == ["server_unsafe"]


def test_an_old_server_is_gated_exactly_as_before():
    client = _client(_Server(safe=False, geometry_certified=False, ag3s_status="degraded",
                             trajopt_status="violated"))
    client.infer({"state": np.zeros(DOF)})
    assert not client.should_execute
    assert client.last_reasons_source == "derived"
    assert [r["kind"] for r in client.last_reasons] == ["uncertified"]

    client = _client(_Server(safe=True))
    client.infer({"state": np.zeros(DOF)})
    assert client.should_execute and client.last_reasons == []


@pytest.mark.parametrize("server,kw,sub", [
    (dict(safe=True, delay=0.05), dict(timeout_s=0.01), "timeout"),
    (dict(safe=True, seq_offset=-1), {}, "stale"),
    (dict(safe=True, actions=np.zeros((H, 7))), {}, "dimension"),
    (dict(raises=ConnectionError("socket closed")), {}, "error"),
], ids=["timeout", "stale", "dimension", "error"])
def test_comms_failures_hold_with_a_comms_reason(server, kw, sub):
    client = _client(_ReasonServer(reasons=[ALLOWED], **server), **kw)
    client.infer({"state": np.zeros(DOF)})

    assert not client.should_execute
    assert [r["kind"] for r in client.last_reasons] == ["comms"]
    assert client.last_reasons[0]["evidence"]["ipc"] == sub
    assert client.last_reasons_source == "client"
    assert client.last_gate["hold_kinds"] == ["comms"]
    # T18 의 `HOLD_KINDS` 는 그대로다 — `dimension` 은 `last_ipc` 로는 `error`.
    assert client.last_ipc in wire.HOLD_KINDS


def test_reasons_do_not_leak_into_the_next_chunk():
    server = _ReasonServer(safe=False, trajopt_status="violated", reasons=[COLLISION])
    client = _client(server)
    client.infer({"state": np.zeros(DOF)})
    server.safe, server.reasons = True, []
    client.infer({"state": np.zeros(DOF)})

    assert client.should_execute and client.last_reasons == []
    assert client.reason_stats == {"collision": 1}


def test_the_response_object_is_still_returned_untouched():
    """T9 계약 — shadow 가 아니면 응답에 키 하나 더하지 않는다. 사유는 속성으로만."""
    server = _ReasonServer(safe=True, reasons=[ALLOWED])
    client = _client(server)
    out = client.infer({"state": np.zeros(DOF)})
    assert set(out) == set(server.infer({wire.PREFIX + "seq": 2}).keys())


# =============================================================================================
# 2. `_hold` 의 대체 청크 = 마지막 명령
# =============================================================================================


def test_the_replacement_chunk_is_the_last_command_not_the_measured_pose():
    scene = _Scene()
    client = SafeRemoteClient(policy=_Server(raises=ConnectionError("down")), scene=scene,
                              cameras=("zed_left",))
    arm = np.linspace(-0.5, 0.5, 2 * N)
    client.note_command({"arm": arm, "gripper": np.array([0.0, 1.0])})   # 왼손 닫힘 명령
    out = client.infer({"state": np.zeros(DOF)})

    expected = action_row(arm, [0.0, 1.0])
    np.testing.assert_allclose(out["actions"], np.tile(expected, (50, 1)))
    left_grip, _ = wire.gripper_columns()
    assert out["actions"][0, left_grip] == 0.0, "닫힘 명령이 열린 측정값으로 덮이지 않는다"


def test_reset_forgets_the_last_command():
    scene = _Scene()
    client = SafeRemoteClient(policy=_Server(raises=ConnectionError("down")), scene=scene,
                              cameras=("zed_left",))
    client.note_command({"arm": np.ones(2 * N), "gripper": np.zeros(2)})
    out = client.infer({"state": np.zeros(DOF)}, reset=True)
    np.testing.assert_allclose(out["actions"], np.tile(scene.q, (50, 1)))


def test_action_row_is_the_apply_action_order():
    row = action_row(np.arange(2 * N), [0.25, 0.75])
    left, right = wire.gripper_columns()
    assert row.shape == (wire.ACTION_WIDTH,)
    assert row[left] == 0.25 and row[right] == 0.75
    np.testing.assert_allclose(row[:N], np.arange(N))
    np.testing.assert_allclose(row[left + 1:right], np.arange(N, 2 * N))
    with pytest.raises(ValueError):
        action_row(np.arange(2 * N + 1), [0, 0])


# =============================================================================================
# 3. HoldController
# =============================================================================================


def _measured(arm, grip):
    return action_row(arm, grip)


def test_q_hold_is_the_commanded_target_captured_once():
    hold = HoldController()
    command = {"arm": np.full(2 * N, 0.3), "gripper": np.array([0.2, 1.0])}
    hold.begin_chunk(executed=False, reason="TO says violated", kinds=["collision"])

    first = hold.hold_action(last_command=command, measured=_measured(np.full(2 * N, 0.29),
                                                                       [0.9, 1.0]))
    # 다음 스텝의 "직전 명령" 은 HOLD 자신이고, 측정값은 흘렀다 — 그래도 같은 목표.
    drifted = _measured(np.full(2 * N, 0.25), [0.95, 1.0])
    for _ in range(20):
        row = hold.hold_action(last_command={"arm": np.zeros(2 * N), "gripper": np.ones(2)},
                               measured=drifted)
        np.testing.assert_allclose(row, first)
    np.testing.assert_allclose(first, action_row(command["arm"], command["gripper"]))
    assert hold.source == "command" and hold.age == 20
    rec = hold.record(measured=drifted)
    assert rec["q_hold"] == pytest.approx([0.3] * (2 * N))
    assert rec["gripper_hold"] == pytest.approx([0.2, 1.0])
    assert rec["entry_kinds"] == ["collision"] and rec["age"] == 20
    assert rec["max_abs_dev_rad"] == pytest.approx(0.05)
    json.dumps(rec)


def test_the_hold_survives_consecutive_hold_chunks_and_releases_on_execution():
    hold = HoldController()
    cmd = {"arm": np.full(2 * N, 0.1), "gripper": np.array([1.0, 1.0])}
    hold.begin_chunk(executed=False)
    a = hold.hold_action(last_command=cmd, measured=_measured(np.zeros(2 * N), [1, 1]))
    hold.begin_chunk(executed=False)             # 두 번째 HOLD 청크
    b = hold.hold_action(last_command={"arm": np.full(2 * N, 9.0), "gripper": np.zeros(2)},
                         measured=_measured(np.zeros(2 * N), [1, 1]))
    np.testing.assert_allclose(a, b)
    assert hold.consecutive == 2

    hold.begin_chunk(executed=True)
    hold.release()
    assert not hold.holding and hold.record() is None and hold.consecutive == 0
    hold.begin_chunk(executed=False)
    c = hold.hold_action(last_command={"arm": np.full(2 * N, 0.7), "gripper": np.zeros(2)},
                         measured=_measured(np.zeros(2 * N), [1, 1]))
    np.testing.assert_allclose(c, action_row(np.full(2 * N, 0.7), [0.0, 0.0]))
    assert hold.stats["holds"] == 2 and hold.stats["max_consecutive"] == 2


def test_without_a_known_command_the_hold_is_taken_from_the_measurement_once():
    hold = HoldController()
    hold.begin_chunk(executed=False)
    m0 = _measured(np.full(2 * N, 0.4), [0.8, 1.0])
    a = hold.hold_action(last_command=None, measured=m0)
    b = hold.hold_action(last_command=None, measured=_measured(np.full(2 * N, 0.0), [0, 0]))
    np.testing.assert_allclose(a, m0)
    np.testing.assert_allclose(b, m0)
    assert hold.source == "measured"


def test_legacy_mode_is_the_measured_state_every_step():
    hold = HoldController(mode="legacy")
    hold.begin_chunk(executed=False)
    cmd = {"arm": np.full(2 * N, 0.3), "gripper": np.array([0.0, 0.0])}
    for k in range(3):
        m = _measured(np.full(2 * N, 0.1 * k), [0.9, 1.0])
        np.testing.assert_allclose(hold.hold_action(last_command=cmd, measured=m), m)
    assert hold.record()["mode"] == "legacy"


def test_the_consecutive_hold_limit_reports_instead_of_looping_forever():
    unlimited = HoldController()
    assert all(unlimited.begin_chunk(executed=False) is None for _ in range(1000))

    hold = HoldController(max_hold_chunks=2)
    assert hold.begin_chunk(executed=False, reason="r1", kinds=["collision"]) is None
    assert hold.begin_chunk(executed=False, reason="r2", kinds=["collision"]) is None
    abort = hold.begin_chunk(executed=False, reason="r3", kinds=["uncertified"])
    assert abort == {"hold_chunks": 3, "limit": 2, "reason": "r3", "kinds": ["uncertified"],
                     "entry_reason": "", "entry_kinds": []}
    # 실행된 청크가 끼면 다시 센다.
    hold = HoldController(max_hold_chunks=1)
    hold.begin_chunk(executed=False)
    hold.begin_chunk(executed=True)
    assert hold.begin_chunk(executed=False) is None

    assert HoldController(max_hold_chunks=0).begin_chunk(executed=False) is not None
    with pytest.raises(ValueError):
        HoldController(max_hold_chunks=-1)
    with pytest.raises(ValueError):
        HoldController(mode="freeze")


def test_the_gripper_hold_neither_opens_nor_closes():
    """지침 §8.4 — 열린 채 취소된 파지는 열린 채, 쥔 채 HOLD 는 쥔 명령 그대로."""
    for commanded in (1.0, 0.0, 0.57):
        hold = HoldController()
        hold.begin_chunk(executed=False)
        row = hold.hold_action(last_command={"arm": np.zeros(2 * N),
                                             "gripper": np.array([commanded, 1.0])},
                               measured=_measured(np.zeros(2 * N), [0.72, 1.0]))
        assert row[wire.gripper_columns()[0]] == commanded


def test_a_fixed_target_does_not_accumulate_servo_error_the_way_re_commanding_does():
    """기전만: 1 차 servo `q += k (target − q) + g` (g = 정착 편향). 33 스텝 뒤 편차."""
    k, g, steps = 0.3, 0.002, 33

    def run(mode):
        hold = HoldController(mode=mode)
        hold.begin_chunk(executed=False)
        q = np.zeros(2 * N)
        cmd = {"arm": q.copy(), "gripper": np.ones(2)}
        for _ in range(steps):
            row = hold.hold_action(last_command=cmd, measured=action_row(q, [1.0, 1.0]))
            target = np.concatenate([row[:N], row[N + 1:2 * N + 1]])
            q = q + k * (target - q) + g
        return float(np.max(np.abs(q)))

    fixed, legacy = run("fixed"), run("legacy")
    assert legacy == pytest.approx(steps * g)          # 매 스텝 재명령 — 편향이 그대로 쌓인다
    assert fixed == pytest.approx(g / k, rel=0.05)     # 고정 목표 — 정상상태 오차에서 멈춘다
    assert fixed < legacy / 5


# =============================================================================================
# 4. control frame 의 `hold`
# =============================================================================================


def test_the_control_frame_carries_hold_only_when_given(tmp_path):
    from AG3S.ag3s.runtime.frame_record import FrameRecorder

    hold = HoldController()
    hold.begin_chunk(executed=False, reason="TO says violated", kinds=["collision"])
    hold.hold_action(last_command={"arm": np.full(2 * N, 0.2), "gripper": np.array([0.0, 1.0])},
                     measured=action_row(np.full(2 * N, 0.21), [0.9, 1.0]))
    record = hold.record(measured=action_row(np.full(2 * N, 0.21), [0.9, 1.0]))
    with FrameRecorder(str(tmp_path), manifest={"timing": {}}) as rec:
        rec.control(t_step=1, chunk_seq=1, step_in_chunk=0, now=0.0, field=None,
                    executed=False, qpos=[0.0], hold=record)
        rec.control(t_step=2, chunk_seq=2, step_in_chunk=0, now=0.0, field=None,
                    executed=True, qpos=[0.0])
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    controls = [r for r in rows if r["kind"] == "control"]

    assert controls[0]["hold"]["q_hold"] == pytest.approx([0.2] * (2 * N))
    assert controls[0]["hold"]["gripper_hold"] == [0.0, 1.0]
    assert controls[0]["hold"]["entry_kinds"] == ["collision"]
    assert controls[0]["hold"]["max_abs_dev_rad"] == pytest.approx(0.01)
    assert "hold" not in controls[1]
