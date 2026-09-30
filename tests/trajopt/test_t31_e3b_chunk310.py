"""T31 — E3b chunk 310 (ep1808 r1, local seq 11 · t_step 80) 재현: MJCF 한계면 refined ≡ reference.

기록: 정책은 `left_arm_6` 을 −2.72 → −2.84 rad 로 명령했다. TO 는 URDF 범위(−2.705 + 0.02 =
−2.685)로 보고 초기 iterate 를 부적격으로 두었고 (`initial_limit_overshoot.position` 0.152),
50 ms 예산 안의 1 반복에서 거절된 후보(`best_unaccepted`, merit 0.194 vs 초기 0.018)를 반환했다 —
손목 8.66°, 다른 관절 최대 0.94° (서버 기록 `refined − reference`).

입력은 T24 도구(`fixed_input_ablation`)로 복원한다: reference · 그 실행의 sphere option 으로 다시
지은 모델 · 로컬 planning 행(실제 actions 로 짝지음)의 `q_now` · ESDF 세 계층 · 접촉 정책 · 손가락
관절(T29, 기록의 `finger_joints.opening_norm` + 명령 열). 재구성 충실도는 기록 clearance 대비 < 1 mm.
기록이 이 머신에 없으면 건너뛴다.

**이 파일의 chunk 번호/시각:** 서버 기록 `chunk_00310.npz` 의 `t_step` 은 11 (서버의 프레임 번호)이고,
로컬 planning 행은 seq 11 · `t_step` 80 이다 (T30 `E3b_analysis.json` 과 같은 짝).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

REC = pathlib.Path("/mnt/dev/work/outputs/verify/T30/E3b_rec")
RUN = REC / "E3b_ep1808_r1"
CHUNK = REC / "server_8206/server_constraints/run_0000/chunk_00310.npz"
MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")

pytestmark = pytest.mark.skipif(
    not (CHUNK.exists() and (RUN / "frames.jsonl").exists() and MODEL.exists()),
    reason="E3b records / model not present")


@pytest.fixture(scope="module")
def restored():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from benchmark.ag3s.robot_models.gripper_state import FingerJointMap
    from benchmark.trajopt.experiments import fixed_input_ablation as T
    from benchmark.trajopt.limits import build_limits
    from benchmark.trajopt.types import ChunkLayout

    args = T.build_parser().parse_args([
        "--constraints", str(CHUNK.parent), "--frames", str(RUN / "frames.jsonl"),
        "--server-log", str(REC / "serve_safe_8206.log"), "--out", "/dev/null"])
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log),
                                  json.loads((RUN / "manifest.json").read_text()))
    robot, qadr, _ = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False,
                                          has_free=False, attached=False)
    z = np.load(CHUNK)
    rows = [json.loads(line) for line in (RUN / "frames.jsonl").read_text().splitlines() if line]
    row = next(r for r in rows if r.get("kind") == "planning"
               and np.abs(np.asarray(r["actions"]) - z["refined_chunk"]).max() <= 1e-5)
    summary = json.loads(str(z["summary_json"]))
    ref_chunk = np.asarray(z["reference_chunk"], np.float64)

    def config(**limits):
        cfg = T.to_config(settings, collision=True, cost=settings.cost,
                          sqp={"time_budget_ms": T.NO_CLOCK_MS})
        return cfg.with_overrides({"limits": limits}) if limits else cfg

    base = config()
    meter = T.Meter(robot, layout, base.horizon.planned, settings.hand,
                    build_limits(robot, layout, dt=base.horizon.dt, config=base.limits))
    fingers, _ = FingerJointMap(robot).path(
        (summary.get("finger_joints") or {}).get("opening_norm") or {},
        {"left": ref_chunk[:, 7], "right": ref_chunk[:, 15]}, meter.lin.horizon,
        default=robot.param_vector())
    meter.lin.set_joint_parameter_path(fingers)
    sh = T.Shared(args=args, settings=settings, robot=robot, qadr=qadr, layout=layout,
                  meter=meter, plan_rows={int(z["chunk_index"]) + 1: row}, variant_configs={},
                  active_rows_config=None, outside_distance=maxd if unk == "free" else -maxd,
                  recorded_iters={})
    rec, inp = T.prepare_chunk(sh, CHUNK)
    assert inp is not None, rec.get("skipped_reason")
    previous = np.asarray(np.load(CHUNK.with_name("chunk_00309.npz"))["refined_chunk"], np.float64)

    def solve(**sections):
        limits = sections.pop("limits", {})
        cfg = config(**limits).with_overrides(sections) if sections else config(**limits)
        runner = T.Instrumented(robot, layout, cfg, {"value": (inp.scene, inp.q_now, inp.certified)})
        runner.refiner.optimizer.linearizer.set_joint_parameter_path(fingers)
        result, _ = runner.run(inp.reference_chunk,
                               {"t_step": inp.seq, "previous_physical_chunk": previous})
        return result, runner.refiner.limits

    return {"rec": rec, "inp": inp, "row": row, "summary": summary, "solve": solve,
            "layout": layout, "z": z}


def test_the_reconstruction_is_faithful_and_paired(restored):
    assert restored["row"]["seq"] == 11 and restored["row"]["t_step"] == 80
    fid = restored["rec"]["fidelity"]
    assert fid["within_tolerance"] and fid["record_clearance"]["max_abs_diff_mm"] < 1e-3
    to = restored["summary"]["to"]["metrics"]
    assert to["returned"] == "best_unaccepted" and to["initial_eligible"] is False


def test_with_mjcf_limits_the_overshoot_is_zero_and_refined_equals_reference(restored):
    """**T31 의 재현 판정.** 기본 config (MJCF 한계 · 3 반복) — 초기 iterate 가 적격이고 반환된다."""
    from benchmark.trajopt.limits import limit_report

    result, limits = restored["solve"]()
    inp = restored["inp"]
    assert limit_report(inp.reference, limits)["position"] == 0.0
    m = result.metrics
    assert m["initial_eligible"] is True
    assert m["initial_limit_overshoot"] == {"position": 0.0, "velocity": 0.0,
                                            "acceleration": 0.0, "anchor": 0.0}
    assert m["sqp_iterations"] == 3 and m["min_iterations"] == 3
    assert m["returned"] == "initial"
    np.testing.assert_array_equal(result.trajectory, inp.reference)
    assert result.max_violation == 0.0


def test_with_urdf_limits_the_wrist_alone_is_projected(restored):
    """URDF 로 되돌려도 (G1 없이) G2-ii 가 거절된 후보 대신 최소 투영을 낸다 — 손목만 움직인다."""
    result, limits = restored["solve"](limits={"source": "urdf"})
    inp, layout = restored["inp"], restored["layout"]
    m = result.metrics
    assert m["initial_eligible"] is False
    assert m["initial_limit_overshoot"]["position"] == pytest.approx(0.15216, abs=1e-4)
    assert m["returned"] == "projection"
    moved = np.flatnonzero(np.abs(result.trajectory - inp.reference).max(axis=1) > 0)
    names = [layout.joint_names[layout.q_indices[i]] for i in moved]
    assert names == ["left_arm_6"]
    assert np.degrees(np.abs(result.trajectory - inp.reference).max()) == pytest.approx(8.718,
                                                                                         abs=0.01)


def test_the_recorded_rule_moved_every_joint(restored):
    """대조: URDF + T24 규칙 + 1 반복 (서버가 돈 그대로) — 거절된 후보가 나가고 손목 밖도 움직인다."""
    result, _ = restored["solve"](limits={"source": "urdf"},
                                  sqp={"max_iterations": 1, "min_iterations": 1,
                                       "limit_projection_fallback": False})
    inp = restored["inp"]
    m = result.metrics
    assert m["returned"] == "best_unaccepted" and m["sqp_iterations"] == 1
    dev = np.degrees(np.abs(result.trajectory - inp.reference).max(axis=1))
    assert dev.max() > 8.0
    assert (dev > 0.1).sum() > 1, "손목 밖의 관절도 움직였다"
