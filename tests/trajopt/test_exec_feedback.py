"""T18 — 실행 피드백 계약: 서버가 "제안" 이 아니라 "실행된 것" 을 안다.

T16 에서 정책이 t=176 · t=208 에 gripper 닫힘을 제안했지만 그 청크는 HOLD 로 실행되지 않았고,
서버는 그것을 몰랐다. 이 파일이 고정하는 것:

1. **왕복** — `make_exec_feedback` → `pack_request` → (msgpack) → `strip_request` →
   `unpack_exec_feedback` 이 모든 키를 보존한다. 없음과 `available=False` 와 모양 틀림이 구별된다.
2. **로컬 수집** — `ExecutionLog` 가 HOLD 청크의 "planned 0.02 / applied 1.0" 을 **둘 다** 남긴다.
3. **서버 사용** — 연속성 참조가 `n_exec` 로 정렬되고 HOLD 뒤에는 없어진다. latch 는 측정 개도를
   먼저 본다. 기록의 `summary_json` 에 `exec_feedback` 이 **언제나** 실린다.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.client import ExecutionLog, SafeRemoteClient

H, DOF = 50, wire.ACTION_WIDTH
N = wire.ARM_JOINT_DIM
LEFT, RIGHT = wire.GRIPPER_COLUMNS


def _feedback(*, seq=3, executed=(False,) * 8, executed_chunk="none", ipc="unsafe",
              hold_kind="unsafe", hold_reason="TO says violated", planned=0.02, applied=1.0,
              measured=(0.99, 1.0)):
    n = len(executed)
    return wire.make_exec_feedback(
        seq=seq, t_step_start=176, executed=executed, executed_chunk=executed_chunk,
        ipc=ipc, hold_kind=hold_kind, hold_reason=hold_reason,
        planned_gripper=np.column_stack([np.full(n, planned), np.ones(n)]),
        applied_gripper=np.column_stack([np.full(n, applied), np.ones(n)]),
        applied_arm=np.arange(n * 2 * N, dtype=np.float64).reshape(n, 2 * N) * 0.01,
        measured_gripper=measured)


def _request(feedback):
    return wire.pack_request(
        {"state": np.zeros(DOF), "prompt": "pick"}, cameras=["zed_left"],
        depth={"zed_left": np.full((4, 4), 1.5)}, intrinsics={"zed_left": np.eye(3)},
        extrinsics={"zed_left": np.eye(4)}, robot_state={"zed_left": np.zeros(12)},
        stamps={"zed_left": 3.0}, phase="approach", seq=4, exec_feedback=feedback)


# --- 1. 왕복 ---------------------------------------------------------------------------------


def _assert_same(got, sent):
    assert set(got) == set(sent)
    for key, value in sent.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(got[key], value, rtol=0, atol=1e-6, err_msg=key)
        else:
            assert got[key] == value, key


def test_pack_strip_unpack_keeps_every_key():
    sent = _feedback()
    policy_obs, scene = wire.strip_request(_request(sent))

    assert set(policy_obs) == {"state", "prompt"}          # 정책은 피드백을 보지 않는다
    _assert_same(wire.unpack_exec_feedback(scene), sent)


def _msgpack_numpy():
    pytest.importorskip("msgpack")
    path = (pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi" /
            "packages" / "openpi-client" / "src" / "openpi_client" / "msgpack_numpy.py")
    if not path.exists():
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location("_t18_msgpack_numpy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_feedback_survives_the_real_transport():
    """실제 전송(msgpack_numpy)을 지나도 같다. 튜플은 리스트가 되고 배열은 읽기 전용이 된다."""
    mp = _msgpack_numpy()
    sent = _feedback(executed=(True, True, True, False, False, False, False, False),
                     executed_chunk="refined", ipc="ok", hold_kind=None, hold_reason=None)
    request = mp.unpackb(mp.packb(_request(sent)))
    _, scene = wire.strip_request(request)
    _assert_same(wire.unpack_exec_feedback(scene), sent)


def test_absence_and_unavailable_and_malformed_are_three_different_things():
    _, old = wire.strip_request(_request(None))
    assert wire.EXEC_FEEDBACK not in old
    missing = wire.unpack_exec_feedback(old)
    assert missing == {"available": False, "reason": missing["reason"]}
    assert "predates T18" in missing["reason"]

    _, first = wire.strip_request(_request(wire.no_exec_feedback("first chunk")))
    assert wire.unpack_exec_feedback(first) == {"available": False, "reason": "first chunk"}

    bad = dict(_feedback())
    bad["applied_gripper"] = np.zeros((3, 2))              # 스텝 수가 어긋났다
    got = wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: bad})   # 예외가 아니다
    assert got["available"] is False and got["reason"].startswith("malformed")

    lying = dict(_feedback())
    lying["n_exec"] = 5                                    # `executed` 는 0 개라고 말한다
    got = wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: lying})
    assert got["available"] is False and "disagrees" in got["reason"]


def test_make_refuses_misaligned_or_unregistered_values():
    with pytest.raises(ValueError, match="planned_gripper"):
        wire.make_exec_feedback(
            seq=1, t_step_start=0, executed=[True] * 8, executed_chunk="refined", ipc="ok",
            hold_kind=None, hold_reason=None, planned_gripper=np.zeros((7, 2)),
            applied_gripper=np.zeros((8, 2)), applied_arm=np.zeros((8, 14)),
            measured_gripper=(1, 1))
    with pytest.raises(ValueError, match="hold_kind"):
        _feedback(hold_kind="bored")
    with pytest.raises(ValueError, match="executed_chunk"):
        _feedback(executed_chunk="policy")


def test_the_payload_stays_small():
    """와이어에 스텝별 팔 ctrl 까지 싣는 결정의 근거 (task §되돌아올 지점)."""
    fb = _feedback()
    nbytes = sum(v.nbytes for v in fb.values() if isinstance(v, np.ndarray))
    assert nbytes < 1024


# --- 2. 로컬 수집 (`ExecutionLog`) · 클라이언트 ------------------------------------------------


class _Decided:
    """`SafeRemoteClient` 가 청크 하나에 내린 결정만."""

    def __init__(self, seq, executed_chunk, ipc, reason=""):
        self._seq, self.last_executed_chunk, self.last_ipc, self.last_reason = (
            seq, executed_chunk, ipc, reason)


def _planned_row(gripper):
    row = np.zeros(DOF)
    row[LEFT] = gripper
    row[RIGHT] = 1.0
    return row


def test_a_held_close_is_recorded_as_proposed_but_not_applied():
    """T16 t=176 의 모양: 정책이 0.02 로 닫으라 했고, HOLD 가 1.0(현재 개도)을 재명령했다."""
    log = ExecutionLog()
    log.open_from(_Decided(22, "none", "unsafe", "TO says violated"), t_step=176)
    for _ in range(8):
        log.step(executed=False, planned_row=_planned_row(0.02),
                 applied_arm=np.zeros(2 * N), applied_gripper=(1.0, 1.0))
    fb = log.close(measured_gripper=(1.0, 1.0))

    assert fb["available"] and fb["seq"] == 22 and fb["t_step_start"] == 176
    assert (fb["n_steps"], fb["n_exec"], fb["n_hold"]) == (8, 0, 8)
    assert fb["executed_chunk"] == "none"
    assert fb["hold_kind"] == "unsafe" and fb["hold_reason"] == "TO says violated"
    np.testing.assert_allclose(fb["planned_gripper"][:, 0], 0.02, atol=1e-6)
    np.testing.assert_allclose(fb["applied_gripper"][:, 0], 1.0, atol=1e-6)
    # 닫힌 청크는 다시 내지 않는다 — 같은 사실이 두 요청에 실리면 서버가 두 번 센다.
    assert log.close(measured_gripper=(1, 1))["available"] is False


def test_an_executed_chunk_has_no_hold_kind_and_nothing_open_is_not_invented():
    log = ExecutionLog()
    assert log.close(measured_gripper=(1, 1)) == wire.no_exec_feedback(
        "no chunk has run yet in this episode")
    log.open_from(_Decided(5, "refined", "ok"), t_step=32)
    assert log.close(measured_gripper=(1, 1))["available"] is False   # 0 스텝
    log.open_from(_Decided(6, "refined", "ok"), t_step=40)
    for _ in range(8):
        log.step(executed=True, planned_row=_planned_row(0.5),
                 applied_arm=np.zeros(2 * N), applied_gripper=(0.5, 1.0))
    fb = log.close(measured_gripper=(0.8, 1.0))
    assert fb["n_exec"] == 8 and fb["hold_kind"] is None and fb["hold_reason"] is None


def test_a_hold_without_a_registered_ipc_is_called_an_error():
    log = ExecutionLog()
    log.open(seq=1, t_step=0, executed_chunk="none", ipc="ok")
    log.step(executed=False, planned_row=_planned_row(1.0), applied_arm=np.zeros(2 * N),
             applied_gripper=(1, 1))
    assert log.close(measured_gripper=(1, 1))["hold_kind"] == "error"


class _Frame:
    def __init__(self):
        self.depth = np.full((8, 8), 1.2)
        self.camera_intrinsics = np.eye(3)
        self.T_base_cam = np.eye(4)
        self.robot_state = np.zeros(DOF)


class _Scene:
    def capture(self, _camera):
        return _Frame()

    def robot_state(self):
        return np.zeros(DOF)


class _Server:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return {"actions": np.zeros((H, DOF)), "seq": int(request[wire.PREFIX + "seq"]),
                "safe": True, "ag3s_status": "ok", "geometry_certified": True,
                "trajopt_status": "optimal", "max_violation_m": 0.0, "timing_ms": {},
                "notes": [], "shadow": False}


def test_the_client_always_carries_the_key():
    """새 클라이언트는 낼 사실이 없어도 키를 싣는다 — 빼면 서버가 옛 클라이언트로 읽는다."""
    server = _Server()
    client = SafeRemoteClient(policy=server, scene=_Scene(), cameras=("zed_left",))

    client.infer({"state": np.zeros(DOF)}, reset=True)
    first = wire.unpack_exec_feedback(wire.strip_request(server.requests[-1])[1])
    assert first["available"] is False and "first chunk" in first["reason"]

    client.infer({"state": np.zeros(DOF)})
    bare = wire.unpack_exec_feedback(wire.strip_request(server.requests[-1])[1])
    assert bare["available"] is False and "passed no exec_feedback" in bare["reason"]

    sent = _feedback(seq=2)
    client.infer({"state": np.zeros(DOF)}, exec_feedback=sent)
    _assert_same(wire.unpack_exec_feedback(wire.strip_request(server.requests[-1])[1]), sent)
    assert client.last_exec_feedback["seq"] == 2           # 로컬 planning 기록이 읽는 값


# --- 3. 서버 사용 ----------------------------------------------------------------------------


def _safe_policy(chunk=None, **kw):
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy as build

    return build(_StubPolicy(chunk), **kw)


def _spy(sp, offset=0.5):
    """refiner 를 흉내내되 **받은 context 를 붙든다** — 연속성 입력을 직접 본다."""
    seen = []

    def refine(c, context=None):
        seen.append(dict(context or {}))
        sp.refiner.last_continuity = {"used": True, "source": "spy", "aligned_from_step": None}
        return np.asarray(c, np.float64) + offset

    sp.refiner.refine = refine
    return seen


def _obs(seq, feedback=None):
    out = {"state": np.zeros(DOF), wire.PREFIX + "cameras": [], wire.PREFIX + "seq": seq}
    if feedback is not None:
        out[wire.PREFIX + wire.EXEC_FEEDBACK] = feedback
    return out


def test_without_feedback_the_old_assumption_stays():
    sp = _safe_policy(np.full((H, DOF), 0.2))
    seen = _spy(sp)
    sp.infer(_obs(1))
    out = sp.infer(_obs(2))

    np.testing.assert_allclose(seen[1]["previous_physical_chunk"], 0.7)   # refined 를 가정
    assert "previous_executed_steps" not in seen[1]
    assert any(n.startswith("피드백 없음") for n in out["notes"])


def test_after_a_hold_the_continuity_reference_is_gone():
    sp = _safe_policy(np.full((H, DOF), 0.2))
    seen = _spy(sp)
    sp.infer(_obs(1))
    out = sp.infer(_obs(2, _feedback(seq=1)))                # 8 스텝 전부 HOLD

    assert seen[1]["previous_executed_steps"] == 0
    assert sp._last_continuity["previous_chunk"] is None
    assert "피드백 수신: seq 1, exec 0/8 (HOLD unsafe: 8 steps)" in out["notes"]


@pytest.mark.parametrize("shadow, which, expected", [(False, "refined", 0.7),
                                                     (True, "reference", 0.2)])
def test_partial_execution_aligns_to_what_ran(shadow, which, expected):
    sp = _safe_policy(np.full((H, DOF), 0.2), shadow=shadow)
    seen = _spy(sp)
    sp.infer(_obs(1))
    ran = (True, True, True) + (False,) * 5
    sp.infer(_obs(2, _feedback(seq=1, executed=ran, executed_chunk=which, ipc="ok",
                               hold_kind=None, hold_reason=None)))

    assert seen[1]["previous_executed_steps"] == 3
    np.testing.assert_allclose(seen[1]["previous_physical_chunk"], expected)
    assert sp._last_continuity["previous_chunk"] == which


def test_feedback_about_another_plan_is_not_trusted():
    sp = _safe_policy()
    seen = _spy(sp)
    sp.infer(_obs(1))
    out = sp.infer(_obs(2, _feedback(seq=7, executed=(True,) * 8, executed_chunk="refined",
                                     ipc="ok", hold_kind=None, hold_reason=None)))
    assert seen[1]["previous_physical_chunk"] is None
    assert any("seq 7" in n and "seq 1" in n for n in out["notes"])


def test_the_refiner_drops_or_aligns_by_the_executed_count():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=12, penetration=0.03)
    refiner = TrajOptChunkRefiner(sc.robot, sc.layout,
                                  lambda ctx: (sc.scene, sc.q_now, True), sc.config)
    previous = np.zeros_like(sc.chunk)
    for row in range(previous.shape[0]):
        previous[row, sc.layout.action_columns] = row           # 행 번호가 값이다

    ctx = {"previous_physical_chunk": previous}
    K = sc.config.horizon.execution_length
    assumed = refiner._continuity_reference(ctx, sc.horizon)
    assert np.allclose(assumed[:, 0], K) and refiner.last_continuity["source"] == "assumed"

    assert refiner._continuity_reference({**ctx, "previous_executed_steps": 0}, sc.horizon) is None
    assert refiner.last_continuity["used"] is False

    partial = refiner._continuity_reference({**ctx, "previous_executed_steps": 3}, sc.horizon)
    assert np.allclose(partial[:, 0], 3) and refiner.last_continuity["source"] == "feedback"


def test_latch_reads_the_measured_opening_first():
    sp = _safe_policy()
    chunk = np.ones((H, DOF))
    chunk[0, LEFT] = 0.02                                    # 계획은 닫힘

    sp._pending = {"chunk": chunk, "exec_feedback": wire.unpack_exec_feedback({})}
    assert sp._latch_gripper_signal("left") == pytest.approx(0.02)
    assert sp._last_latch_signal["note"] == "gripper signal = planned (no feedback)"
    assert "gripper signal = planned (no feedback)" in sp._feedback_notes()

    fb = wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: _feedback(seq=4, measured=(0.97, 1.0))})
    sp._pending = {"chunk": chunk, "exec_feedback": fb}
    assert sp._latch_gripper_signal("left") == pytest.approx(0.97)   # 손은 열려 있다
    assert sp._last_latch_signal["source"] == "measured"
    assert sp._last_latch_signal["planned_first_row"] == pytest.approx(0.02)


def test_the_constraint_record_always_carries_exec_feedback(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import (
        ConstraintRecordWriter, load_constraint_run)

    sp = _safe_policy(np.full((H, DOF), 0.2))
    sp.recorder = ConstraintRecordWriter(tmp_path, esdf_mode="none")
    _spy(sp)
    sp.infer(_obs(1))                                        # 피드백 없음
    sp.infer(_obs(2, _feedback(seq=1)))                      # HOLD 한 청크의 사실

    summaries = [json.loads(str(np.load(p)["summary_json"]))
                 for p in sorted(sp.recorder.run_dir.glob("chunk_*.npz"))]
    assert summaries[0]["exec_feedback"]["available"] is False
    assert summaries[0]["exec_feedback"]["reason"]
    held = summaries[1]["exec_feedback"]
    assert held["available"] and held["seq"] == 1 and held["n_exec"] == 0
    assert held["planned_gripper"][0][0] == pytest.approx(0.02)
    assert held["applied_gripper"][0][0] == pytest.approx(1.0)
    assert summaries[1]["exec_continuity"]["previous_chunk"] is None
    assert "exec_latch_signal" in summaries[1]
    assert load_constraint_run  # 읽는 쪽 import 가 깨지지 않았다


def test_summary_extra_never_overwrites_what_the_writer_knows(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    writer = ConstraintRecordWriter(tmp_path, esdf_mode="none")
    path = writer.record(t_step=9, chunk_index=0, constraint_set=None, debug={},
                         summary_extra={"t_step": -1, "exec_feedback": {"available": False}})
    summary = json.loads(str(np.load(path)["summary_json"]))
    assert summary["t_step"] == 9
    assert summary["exec_feedback"] == {"available": False}


# --- 4. 로컬 frame 기록 ----------------------------------------------------------------------


def test_frame_rows_carry_feedback_and_applied_only_when_given(tmp_path):
    from AG3S.ag3s.runtime.frame_record import FrameRecorder

    with FrameRecorder(str(tmp_path), manifest={"timing": {}}) as rec:
        rec.planning(seq=2, t_step=8, field=None, verdict={}, timing_ms={},
                     exec_feedback=_feedback(seq=1))
        rec.control(t_step=8, chunk_seq=2, step_in_chunk=0, now=1.0, field=None,
                    executed=False, applied_ctrl={"arm": np.zeros(2 * N),
                                             "gripper": np.array([1.0, 1.0])})
        rec.control(t_step=9, chunk_seq=2, step_in_chunk=1, now=1.1, field=None,
                    executed=False)
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    plan, ctrl, bare = rows
    assert plan["exec_feedback"]["n_hold"] == 8
    assert plan["exec_feedback"]["planned_gripper"][0][0] == pytest.approx(0.02)
    assert ctrl["applied_ctrl"]["gripper"] == [1.0, 1.0]
    assert "applied_ctrl" not in bare
