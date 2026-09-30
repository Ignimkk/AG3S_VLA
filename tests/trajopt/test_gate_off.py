"""T27 — 로컬 `--safe-gate off`: 판정은 기록만, 도착한 청크는 언제나 **refined** 를 실행한다.

사용자 지시 (2026-09-28): *"이 실험에서는 ESDF 판정이나 hold 가 있으면 안 된다."* T16 은 refined ≡
reference 0.00° 인데도 HOLD 25/34 로 실패했다 — 판정과 HOLD 가 여전히 돌았기 때문이다.

이 파일이 박는 것:

1. 서버가 무엇이라고 하든(`collision` · `uncertified` · `safe=False` · 옛 서버) refined 를 실행한다.
   `last_gate` 에 `{mode: off, would_hold, kinds, would_hold_kinds}` — 게이트가 켜져 있었다면의 결정.
2. `last_safe` · `last_ipc` · `stats` 는 **판정 그대로** (shadow 와 같은 규율) — 실행은
   `last_executed_chunk` 가 따로 말한다.
3. 청크가 **도착하지 않으면**(timeout · stale · error · dimension) fixed HOLD — `comms_holds` 로 세고
   크게 찍는다.
4. shadow 와 같이 쓸 수 없다 (shadow 는 reference, gate off 는 refined).
5. `pi05_infer.py` 의 flag · 거절 · 시작 배너.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from benchmark.trajopt import wire
from benchmark.trajopt.client import GATE_MODES, ExecutionLog, SafeRemoteClient, gate_banner
from tests.trajopt.test_hold_gate import ALLOWED, COLLISION, _ReasonServer
from tests.trajopt.test_safe_client import DOF, H, _client, _Server

PI05_INFER = pathlib.Path(__file__).resolve().parents[2] / "pi05_TO_hybrid" / "rby1_bringup" \
    / "pi05_infer.py"

REFINED = np.full((H, DOF), 0.25)
REFERENCE = np.full((H, DOF), -0.5)


def _obs():
    return {"state": np.zeros(DOF)}


# =============================================================================================
# 1. 도착한 청크는 언제나 refined
# =============================================================================================


def test_off_is_a_registered_gate_mode_and_not_the_default():
    assert "off" in GATE_MODES and GATE_MODES[0] == "reasons"
    import inspect
    assert inspect.signature(SafeRemoteClient.__init__).parameters["gate"].default == "reasons"


def test_a_collision_verdict_is_recorded_but_the_refined_chunk_executes():
    client = _client(_ReasonServer(safe=False, trajopt_status="violated", max_violation_m=0.012,
                                   reasons=[ALLOWED, COLLISION], actions=REFINED), gate="off")
    out = client.infer(_obs())

    assert client.should_execute and client.last_executed_chunk == "refined"
    assert np.array_equal(out["actions"], REFINED)
    assert client.last_gate == {
        "mode": "off", "action": "execute", "source": "server", "server_safe": False,
        "would_hold": True, "kinds": ["allowed_contact", "collision"],
        "would_hold_kinds": ["collision"], "hold_kinds": []}
    # 판정은 그대로다 — 실행 사실과 섞지 않는다.
    assert client.last_safe is False and client.last_ipc == "unsafe"
    assert client.last_reason.startswith("[collision] ")
    assert client.stats["unsafe"] == 1 and client.stats["gate_off_override"] == 1
    assert client.would_hold == 1 and client.would_hold_kinds == {"collision": 1}
    assert client.comms_holds == 0


def test_a_clean_chunk_is_would_not_hold():
    client = _client(_ReasonServer(safe=True, reasons=[], actions=REFINED), gate="off")
    client.infer(_obs())
    assert client.should_execute and client.last_safe is True and client.last_ipc == "ok"
    assert client.last_gate["would_hold"] is False and client.last_gate["kinds"] == []
    assert client.stats["safe"] == 1 and "gate_off_override" not in client.stats


def test_an_uncertified_old_server_verdict_is_derived_and_still_executes():
    """옛 서버(사유 키 없음) · `safe=False` · 미인증 → 유도 사유 `uncertified`, 그래도 실행한다."""
    client = _client(_Server(safe=False, geometry_certified=False, ag3s_status="degraded",
                             actions=REFINED), gate="off")
    out = client.infer(_obs())
    assert client.should_execute and np.array_equal(out["actions"], REFINED)
    assert client.last_reasons_source == "derived"
    assert client.last_gate["would_hold_kinds"] == ["uncertified"]


def test_server_safe_false_without_a_hold_reason_still_counts_as_would_hold():
    client = _client(_ReasonServer(safe=False, reasons=[ALLOWED], actions=REFINED), gate="off")
    client.infer(_obs())
    assert client.should_execute
    assert client.last_gate["would_hold"] is True
    assert client.last_gate["would_hold_kinds"] == ["server_unsafe"]


def test_the_refined_chunk_is_executed_even_when_the_server_also_sends_the_reference():
    """closed-loop 서버는 T9 부터 reference 도 싣는다. gate off 가 고르는 것은 **refined** 다."""
    client = _client(_ReasonServer(safe=False, reasons=[COLLISION], actions=REFINED,
                                   reference=None), gate="off")
    server = client._policy
    orig = server.infer

    def with_reference(request):
        out = orig(request)
        out[wire.ACTIONS_REFERENCE] = REFERENCE
        out[wire.SHADOW] = False
        return out

    server.infer = with_reference
    out = client.infer(_obs())
    assert np.array_equal(out["actions"], REFINED)
    assert np.array_equal(client.last_actions_reference, REFERENCE)


def test_the_execution_log_records_refined_and_no_hold_kind():
    client = _client(_ReasonServer(safe=False, reasons=[COLLISION], actions=REFINED), gate="off")
    log = ExecutionLog()
    client.infer(_obs())
    log.open_from(client, t_step=0)
    for _ in range(3):
        log.step(executed=client.should_execute, planned_row=REFINED[0],
                 applied_arm=np.zeros(2 * wire.ARM_JOINT_DIM), applied_gripper=np.ones(2))
    fb = log.close(measured_gripper=(1.0, 1.0))
    assert fb["executed_chunk"] == "refined" and fb["n_exec"] == 3 and fb["hold_kind"] is None


# =============================================================================================
# 2. 도착하지 않은 청크만 HOLD — 크게, 세서
# =============================================================================================


@pytest.mark.parametrize("server,ipc", [
    (_Server(delay=0.05), "timeout"),
    (_Server(seq_offset=1), "stale"),
    (_Server(raises=ConnectionError("boom")), "error"),
    (_Server(actions=np.zeros((H, DOF - 2))), "dimension"),
])
def test_a_chunk_that_never_arrived_holds_and_is_counted_loudly(server, ipc, capsys):
    client = _client(server, gate="off", timeout_s=0.01 if ipc == "timeout" else 2.0)
    client.infer(_obs())
    assert not client.should_execute and client.last_executed_chunk == "none"
    assert client.comms_holds == 1
    assert client.last_gate["hold_kinds"] == ["comms"]
    assert client.last_reasons[0]["evidence"]["ipc"] == ipc
    printed = capsys.readouterr().out
    assert "GATE OFF but chunk seq 1 did NOT arrive" in printed and "fixed HOLD #1" in printed


def test_close_summarises_executed_would_hold_and_comms(capsys):
    client = _client(_ReasonServer(safe=False, reasons=[COLLISION], actions=REFINED), gate="off")
    client.infer(_obs())
    client.infer(_obs())
    client.close()
    out = capsys.readouterr().out
    assert "GATE OFF: executed the refined chunk on 2/2 chunks" in out
    assert "would have held 2 (collision 2)" in out and "comms HOLD 0" in out
    assert "must be 0" not in out


# =============================================================================================
# 3. shadow 와 같이 쓸 수 없다 · 배너
# =============================================================================================


def test_gate_off_and_shadow_are_refused_together():
    with pytest.raises(ValueError, match="REFINED.*REFERENCE"):
        _client(_Server(), gate="off", shadow=True)


def test_the_banner_is_loud_only_when_the_gate_is_off():
    assert gate_banner("reasons") is None and gate_banner("legacy") is None
    text = gate_banner("off")
    assert "GATE OFF" in text and "판정은 기록만" in text and "refined" in text
    assert "comms" in text and "would_hold" in text


# =============================================================================================
# 4. pi05_infer.py (서버 사본) — flag 와 거절
# =============================================================================================


@pytest.fixture(scope="module")
def pi05_src():
    if not PI05_INFER.exists():  # pragma: no cover - 서버 사본이 없는 체크아웃
        pytest.skip(f"not present: {PI05_INFER}")
    return PI05_INFER.read_text()


def test_pi05_infer_offers_gate_off_and_refuses_it_with_shadow(pi05_src):
    assert 'choices=("reasons", "legacy", "off")' in pi05_src
    assert 'args.safe_gate == "off" and args.safe_shadow' in pi05_src
    assert 'args.safe_gate == "off" and not args.safe_remote' in pi05_src
    # 시작 배너는 클라이언트 모듈의 문장 하나를 찍는다 — 두 곳에 적지 않는다.
    assert "print(gate_banner(args.safe_gate))" in pi05_src
    # planning frame 의 gate 기록은 `last_gate` 를 그대로 싣는다 (would_hold · kinds 가 여기 탄다).
    assert '"gate": dict(safe_client.last_gate)' in pi05_src
