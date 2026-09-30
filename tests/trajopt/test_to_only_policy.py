"""T27 — `serve_safe --no-perception`: π0.5 → **TO 만** (`trajopt/to_only_policy.py`).

사용자 사다리 실험 E1·E2 의 서버다. 이 파일이 박는 것:

1. **빈 세계** — `EmptyWorldField` 를 실은 씬은 비지 않았다(SQP 가 돈다)가 여유거리는 `+inf` 이고
   최악 행 신원은 `None` 이다 (없는 신원을 지어내지 않는다).
2. **같은 와이어** — `actions`=refined · `actions_reference` · `to` · `safe=True` ·
   `verdict_reasons=[no_perception]` · `field` unavailable · `shadow=False`.
3. **E1 판정 기준 #1** — 추적 항만 + `--no-limits` 면 refined ≡ reference (청크 전체, 비트 단위).
4. **AG3S 를 만들지 않는다** — attention 두 번째 사본 · `build_ag3s` · 거리장 기록기를 부르지 않는다.
5. 기록 (`--record-constraints`) · 연속성 (T18 표) · 서버 flag 조합.
6. 클라이언트 왕복 — 사유 게이트는 HOLD (`no_perception`), `gate="off"` 는 refined 를 실행.
"""

from __future__ import annotations

import inspect
import json
import logging

import numpy as np
import pytest

from benchmark.ag3s.robot_models import load_rby1
from benchmark.trajopt import serve_safe, wire
from benchmark.trajopt.client import SafeRemoteClient
from benchmark.trajopt.config import NO_LIMITS, TrajOptConfig
from benchmark.trajopt.to_only_policy import (NO_PERCEPTION_STATUS, EmptyWorldField,
                                              ToOnlyPolicy, ToOnlyRecorder, empty_world_scene,
                                              q_now_from_request)
from benchmark.trajopt.types import ChunkLayout

from tests.trajopt.fixtures import Q_HOME, sweep_reference

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
H = 50
E1 = {"limits": NO_LIMITS, "cost": {"w_smooth": 0.0, "w_continuity": 0.0}}
SERVER_CFG = {"collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False}}


@pytest.fixture(scope="module")
def robot():
    return load_rby1()


@pytest.fixture(scope="module")
def chunk(robot):
    layout = ChunkLayout.rby1(robot.joint_names)
    ref = sweep_reference(layout, H)
    out = np.zeros((H, wire.ACTION_WIDTH))
    out[:, layout.action_columns] = ref.T
    left, right = wire.gripper_columns(layout.nq_opt // 2)
    out[:, left], out[:, right] = 0.7, 0.2
    return out


class _Policy:
    metadata = {"policy": "fake"}

    def __init__(self, chunk):
        self.chunk = np.asarray(chunk)
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": self.chunk.copy(), "policy_timing": {"infer_ms": 1.0}}


def _request(seq, *, reset=False, q=Q_HOME, cameras=("zed_left", "wrist_cam_l"), feedback=None):
    return wire.pack_request(
        {"state": np.zeros(wire.ACTION_WIDTH)}, cameras=cameras,
        depth={c: np.zeros((4, 4)) for c in cameras},
        intrinsics={c: np.eye(3) for c in cameras},
        extrinsics={c: np.eye(4) for c in cameras},
        robot_state={c: np.asarray(q, np.float64) for c in cameras},
        stamps={c: 1.0 + i for i, c in enumerate(cameras)},
        phase="approach", reset=reset, seq=seq, exec_feedback=feedback)


def _policy(robot, chunk, over=None, recorder=None):
    cfg = TrajOptConfig.from_dict({**SERVER_CFG, **(over or {})})
    return ToOnlyPolicy(_Policy(chunk), robot_model=robot, to_config=cfg, recorder=recorder)


# =============================================================================================
# 1. 빈 세계
# =============================================================================================


def test_the_empty_world_is_not_an_empty_scene_but_answers_infinity(robot):
    scene = empty_world_scene(np.full(5, 0.05))
    assert not scene.is_empty, "비면 sqp 가 SQP 를 건너뛰고 UNCONSTRAINED 로 통과시킨다"
    assert not scene.candidate_active.any() and not scene.plane_active.any()
    assert np.all(np.isposinf(EmptyWorldField().distance(np.zeros((7, 3)))))


def test_the_optimizer_runs_and_reports_no_invented_worst_row(robot, chunk):
    pol = _policy(robot, chunk)
    pol.infer(_request(1, reset=True))
    r = pol.refiner.last_result
    assert r.status.value in ("optimal", "feasible")
    assert r.iterations >= 1, "SQP 가 돌아야 'TO only' 다 (UNCONSTRAINED 통과가 아니다)"
    assert r.max_violation == 0.0 and r.metrics["max_violation_pair"] is None
    assert r.metrics["collision_enabled"] is False


# =============================================================================================
# 2. 같은 와이어
# =============================================================================================


def test_the_response_is_the_same_wire_the_safe_client_reads(robot, chunk):
    out = _policy(robot, chunk).infer(_request(1, reset=True))
    assert out["seq"] == 1
    assert out["actions"].shape == out[wire.ACTIONS_REFERENCE].shape == chunk.shape
    assert np.allclose(out[wire.ACTIONS_REFERENCE], chunk.astype(np.float32))
    assert out[wire.SHADOW] is False
    assert out["safe"] is True
    assert out["ag3s_status"] == NO_PERCEPTION_STATUS and out["geometry_certified"] is False
    assert out["trajopt_status"] in ("optimal", "feasible")
    reasons, source = wire.unpack_verdict_reasons(out)
    assert source == "server" and [r["kind"] for r in reasons] == ["no_perception"]
    assert wire.gate_decision(reasons)[0] == "hold", "사유 게이트는 검사되지 않은 청크를 실행하지 않는다"
    assert wire.unpack_field(out).state == "unavailable"
    assert wire.unpack_ag3s(out)["status"] == NO_PERCEPTION_STATUS
    to = wire.unpack_to(out)
    for key in ("status", "iterations", "solve_ms", "sqp_iterations", "qp_iterations",
                "max_iterations", "time_budget_ms", "time_budget_hit", "max_iterations_hit",
                "collision_enabled", "reference_deviation"):
        assert key in to, f"SafePolicy._to_block 과 같은 키여야 한다: {key}"
    assert to["perception"] is False and to["collision_enabled"] is False
    assert to["limits_enforced"] == {"position": True, "velocity": True, "acceleration": True,
                                     "anchor": True}
    # 정책의 다른 키도 실린다 (SafePolicy 와 같은 규약).
    assert out["policy_timing"] == {"infer_ms": 1.0}


def test_metadata_does_not_claim_shadow(robot, chunk):
    """클라이언트의 짝 검사(`_check_server_metadata`)는 `shadow` 키만 본다 — 없어야 closed loop 다."""
    meta = _policy(robot, chunk).metadata
    assert "shadow" not in meta
    assert meta["to_only"] is True and meta["perception"] is False


def test_grippers_are_the_policys_own_values(robot, chunk):
    out = _policy(robot, chunk).infer(_request(1, reset=True))
    for col in wire.gripper_columns():
        assert np.array_equal(out["actions"][:, col], chunk[:, col].astype(np.float32))


# =============================================================================================
# 3. E1 · E2
# =============================================================================================


def test_e1_track_only_no_limits_returns_the_reference_bit_for_bit(robot, chunk):
    """**판정 기준 #1** (T27/T28): `--no-perception --no-limits --w-smooth 0 --w-continuity 0`."""
    pol = _policy(robot, chunk, E1)
    for seq in (1, 2, 3):
        out = pol.infer(_request(seq, reset=(seq == 1)))
        assert np.array_equal(out["actions"], out[wire.ACTIONS_REFERENCE])
        assert wire.unpack_to(out)["returned"] == "initial"
        assert wire.unpack_to(out)["reference_deviation"] == 0.0
    assert wire.unpack_to(out)["limits_enforced"] == {
        "position": False, "velocity": False, "acceleration": False, "anchor": True}


def test_e2_default_objective_and_limits_do_change_the_chunk(robot, chunk):
    """대조군. 기본 목적함수(평활 · 연속성)와 limit 이 있으면 SQP 가 청크를 바꾼다 — 이 fixture 의
    reference 는 joint box 밖이기도 하다 (T15 의 2.547 mm)."""
    pol = _policy(robot, chunk)
    out = None
    for seq in (1, 2):
        out = pol.infer(_request(seq, reset=(seq == 1)))
    assert np.abs(out["actions"] - out[wire.ACTIONS_REFERENCE]).max() > 1e-3
    assert wire.unpack_to(out)["sqp_iterations"] >= 1


def test_the_tail_beyond_the_planned_window_is_the_policys(robot, chunk):
    pol = _policy(robot, chunk)
    out = pol.infer(_request(1, reset=True))
    planned = pol.to_config.horizon.planned
    assert np.array_equal(out["actions"][planned:], chunk[planned:].astype(np.float32))


# =============================================================================================
# 4. q_now · 실패
# =============================================================================================


def test_q_now_is_the_latest_camera_state():
    later = Q_HOME + 0.1
    req = _request(1, cameras=("zed_left", "wrist_cam_l"))
    _, scene = wire.strip_request(req)
    scene["robot_state/wrist_cam_l"] = later       # stamp 2.0 > 1.0
    assert np.array_equal(q_now_from_request(scene), later)
    with pytest.raises(ValueError, match="robot_state"):
        q_now_from_request({"cameras": []})


def test_without_robot_state_the_chunk_passes_through_and_is_not_called_safe(robot, chunk):
    pol = _policy(robot, chunk)
    out = pol.infer(_request(1, reset=True, cameras=()))
    assert np.array_equal(out["actions"], chunk.astype(np.float32))
    assert out["safe"] is False and out["trajopt_status"] == "solver_failed"
    reasons, _ = wire.unpack_verdict_reasons(out)
    assert [r["kind"] for r in reasons] == ["unverified"]
    assert wire.TO_BLOCK not in out


# =============================================================================================
# 5. 연속성 (T18 표) · reset
# =============================================================================================


def _feedback(seq, n_exec, n_steps=8, chunk_name="refined"):
    flags = [True] * n_exec + [False] * (n_steps - n_exec)
    return wire.make_exec_feedback(
        seq=seq, t_step_start=0, executed=flags, executed_chunk=chunk_name,
        ipc="ok", hold_kind=None if n_exec == n_steps else "timeout",
        hold_reason=None, planned_gripper=np.ones((n_steps, 2)),
        applied_gripper=np.ones((n_steps, 2)),
        applied_arm=np.zeros((n_steps, 2 * wire.ARM_JOINT_DIM)),
        measured_gripper=np.ones(2))


def test_continuity_follows_the_feedback_table(robot, chunk):
    pol = _policy(robot, chunk)
    pol.infer(_request(1, reset=True))
    assert pol._last_continuity["mode"] == "no cached plan (first chunk after reset or server restart)" \
        or pol._last_continuity["mode"] == "assumed (no feedback)"
    pol.infer(_request(2, feedback=_feedback(1, 8)))
    assert pol._last_continuity["mode"] == "feedback"
    assert pol._last_continuity["previous_chunk"] == "refined"
    pol.infer(_request(3, feedback=_feedback(1, 8)))       # 캐시는 seq 2
    assert "which plan executed is unknown" in pol._last_continuity["mode"]
    pol.infer(_request(4, feedback=_feedback(3, 0)))       # HOLD: 한 스텝도 안 날았다
    assert pol._last_continuity["previous_chunk"] is None


def test_reset_drops_the_cached_plan(robot, chunk):
    pol = _policy(robot, chunk)
    pol.infer(_request(1, reset=True))
    assert pol._previous_plan is not None
    pol.infer(_request(2, reset=True))
    assert pol._previous_plan["seq"] == 2 and pol._last_continuity["plan_seq"] is None


# =============================================================================================
# 6. 기록
# =============================================================================================


def test_the_recorder_writes_reference_refined_deviation_limits_and_sqp(robot, chunk, tmp_path):
    rec = ToOnlyRecorder(tmp_path, meta={"links": "gripper", "limits": "off"})
    pol = _policy(robot, chunk, recorder=rec)
    for seq in (1, 2):
        pol.infer(_request(seq, reset=(seq == 1)))
    run = rec.run_dir
    assert json.loads((run / "meta.json").read_text())["kind"] == "to_only"
    lines = (run / "chunks.jsonl").read_text().splitlines()
    assert len(lines) == 2 and rec.n_records == 2
    assert "Infinity" not in lines[0] and "NaN" not in lines[0]
    row = json.loads(lines[-1])
    for key in ("seq", "max_abs_deviation_deg", "per_joint_max_abs_deviation_deg",
                "limit_overshoot_reference", "limit_overshoot_refined", "to", "sqp", "notes",
                "exec_feedback", "continuity"):
        assert key in row
    # 로봇 한계 기준 — 이 fixture 의 reference 는 box 밖 2.5 mm (T15).
    assert row["limit_overshoot_reference"]["position"] > 0.0
    assert set(row["per_joint_max_abs_deviation_deg"]) >= {"left_arm_0", "right_arm_0"}
    assert row["sqp"]["returned"] in ("initial", "accepted", "best_unaccepted")
    npz = np.load(run / "chunk_0001.npz")
    assert npz["reference"].shape == npz["refined"].shape == chunk.shape
    planned = pol.to_config.horizon.planned
    assert npz["deviation_rad"].shape == (pol.layout.nq_opt, planned)
    assert np.degrees(np.abs(npz["deviation_rad"]).max()) == pytest.approx(
        row["max_abs_deviation_deg"], rel=1e-6, abs=1e-9)
    assert npz["q_now"].shape == (len(robot.joint_names),)


def test_limit_overshoot_is_measured_against_the_robots_limits_even_with_no_limits(robot, chunk,
                                                                                   tmp_path):
    rec = ToOnlyRecorder(tmp_path)
    pol = _policy(robot, chunk, E1, recorder=rec)
    pol.infer(_request(1, reset=True))
    row = json.loads((rec.run_dir / "chunks.jsonl").read_text().splitlines()[0])
    assert row["max_abs_deviation_deg"] == 0.0
    assert row["limit_overshoot_refined"]["position"] > 0.0, (
        "--no-limits 여도 기록은 로봇 자신의 한계로 잰다 — 강제하지 않은 값이 보여야 한다")


# =============================================================================================
# 7. 서버 flag · AG3S 를 만들지 않는다
# =============================================================================================


def test_no_perception_is_off_by_default_and_reaches_a_collision_free_config():
    ap = serve_safe.build_parser()
    assert ap.parse_args(BASE_ARGV).no_perception is False
    args = ap.parse_args(BASE_ARGV + ["--no-perception"])
    serve_safe.reject_bad_flag_combinations(ap, args)
    assert serve_safe.trajopt_config_from_args(args).collision.enabled is False


@pytest.mark.parametrize("extra", [
    ["--no-safe"], ["--shadow"], ["--exclude-links", "ee_left"], ["--no-self-collision"],
    ["--target-field-policy", "exclude-authorized", "--esdf-backend", "curobo"],
    ["--esdf-backend", "curobo"], ["--static-geometry", "auto"], ["--sphere-spacing", "0.5"],
    ["--self-filter-target-guard", "0.05"], ["--max-field-age-sec", "0.5"],
])
def test_flags_that_do_nothing_without_perception_are_refused(extra):
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-perception", *extra])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


@pytest.mark.parametrize("extra", [
    ["--no-limits", "--w-smooth", "0", "--w-continuity", "0"], ["--links", "arms"],
    ["--no-collision"], ["--no-attention"], ["--record-constraints", "/tmp/x"],
    ["--plan-horizon", "full"],
])
def test_the_ladder_flags_are_accepted(extra):
    ap = serve_safe.build_parser()
    serve_safe.reject_bad_flag_combinations(ap, ap.parse_args(BASE_ARGV + ["--no-perception",
                                                                           *extra]))


def test_the_no_perception_path_builds_no_ag3s_and_no_attention_copy():
    """**두 번째 체크포인트 사본을 올리지 않는다** (task §2). 소스로 박는다 — 실제 서버는 GPU 가 필요하다."""
    src = inspect.getsource(serve_safe.build_to_only_policy)
    for forbidden in ("load_attention_model", "AttentionPolicy", "build_ag3s", "SafePolicy",
                      "ConstraintRecordWriter", "load_static_geometry"):
        assert forbidden not in src
    main = inspect.getsource(serve_safe.main)
    branch = main.index("elif args.no_perception")
    assert branch < main.index("load_attention_model"), "지각 없는 분기가 attention 로드보다 먼저다"
    robot_src = inspect.getsource(serve_safe.build_to_only_robot_model)
    assert "AG3S(" not in robot_src and "build_robot_model(" not in robot_src


def test_the_startup_log_shouts_no_perception_and_names_the_client_pairing(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_no_perception()
    text = caplog.text
    assert "NO PERCEPTION" in text and "--no-collision" in text
    assert "--safe-gate off" in text


# =============================================================================================
# 8. 클라이언트 왕복 — 기존 --safe-remote 클라이언트가 그대로 받는다
# =============================================================================================


class _Frame:
    def __init__(self, q):
        self.depth = np.full((4, 4), 1.0)
        self.camera_intrinsics = np.eye(3)
        self.T_base_cam = np.eye(4)
        self.robot_state = np.asarray(q, np.float64)


class _ModelScene:
    """클라이언트가 보는 씬. `robot_state` 는 **모델 q (20)** — 서버가 FK 에 쓴다."""

    def capture(self, _cam):
        return _Frame(Q_HOME)

    def robot_state(self):
        return Q_HOME


def _client(pol, gate):
    return SafeRemoteClient(policy=pol, scene=_ModelScene(),
                            cameras=("zed_left", "wrist_cam_l"), gate=gate)


def test_the_reasons_gate_holds_a_no_perception_chunk(robot, chunk):
    client = _client(_policy(robot, chunk, E1), "reasons")
    client.infer({"state": np.zeros(wire.ACTION_WIDTH)}, reset=True)
    assert not client.should_execute
    assert client.last_gate["hold_kinds"] == ["no_perception"]
    assert client.last_ipc == "unsafe"


def test_gate_off_executes_the_refined_chunk_from_a_no_perception_server(robot, chunk):
    client = _client(_policy(robot, chunk, E1), "off")
    out = client.infer({"state": np.zeros(wire.ACTION_WIDTH)}, reset=True)
    assert client.should_execute and client.last_executed_chunk == "refined"
    assert np.array_equal(out["actions"], client.last_actions_refined)
    assert np.array_equal(client.last_actions_refined, client.last_actions_reference)
    assert client.last_gate["mode"] == "off" and client.last_gate["would_hold"] is True
    assert client.last_gate["kinds"] == ["no_perception"]
    assert client.last_to["perception"] is False
    assert client.comms_holds == 0
