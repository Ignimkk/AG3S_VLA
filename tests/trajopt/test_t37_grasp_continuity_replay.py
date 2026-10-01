"""T37 (K1) — T34 E3b 기록의 청크를 **고정 입력**으로 다시 풀어 grasp continuity gate 를 본다.

입력 복원은 T35.diag `b_ablation.py` 와 같다 (`fixed_input_ablation` 함수들): 서버 로그 설정 · 청크 npz 의
거리장(coarse + fine + target-free 층) · 짝 지은 로컬 planning 행의 qpos → `q_now` · contact policy 마스크 ·
측정 개도의 손가락 관절 경로 · 직전 청크의 기록된 refined 청크(연속성 기준, K 스텝 실행 가정) · 시계 없음 ·
`max_iterations` = 기록된 sqp 반복 수. 상태는 그 청크의 `summary.grasp.state` (live latch 가 낸 값) 를
`grasp_continuity_off_reason` 에 넣어 정한다 — `SafePolicy` 와 같은 판단.

| run | t | chunk | 상태 | gate 없이 손끝 편차 | gate |
|---|---|---|---|---|---|
| E3b ep1808 r1 | 120 | 00315 | closing | 19.6 mm (T35 replay) | ≈ 0 |
| E3b ep1808 r2 | 112 | 00389 | closing | 34.7 mm (T35 replay) | ≈ 0 |
| E3b ep1808 r1 | 40 | 00305 | latched | 12.9 mm | 불변 (궤적 동일) |
| E3b ep1808 r2 | 56 | 00382 | latched | 10.2 mm | 불변 (궤적 동일) |

기록이 이 머신에 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

V = pathlib.Path("/mnt/dev/work/outputs/verify/T34")
REC = V / "E3b_rec"
CH = REC / "server_8206/server_constraints/run_0000"
MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")

pytestmark = pytest.mark.skipif(
    not ((REC / "serve_safe_8206.log").exists() and (CH / "chunk_00315.npz").exists()
         and (CH / "chunk_00389.npz").exists() and (REC / "E3b_ep1808_r1/frames.jsonl").exists()
         and MODEL.exists()),
    reason="T34 E3b records / model not present")

_SETUP: dict = {}


def _setup(run: str):
    if run in _SETUP:
        return _SETUP[run]
    from benchmark.ag3s.robot_models.gripper_state import FingerJointMap
    from benchmark.trajopt.experiments import fixed_input_ablation as T
    from benchmark.trajopt.limits import build_limits
    from benchmark.trajopt.types import ChunkLayout

    run_dir = REC / run
    args = T.build_parser().parse_args([
        "--constraints", str(CH), "--frames", str(run_dir / "frames.jsonl"),
        "--server-log", str(REC / "serve_safe_8206.log"), "--out", "/dev/null"])
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log),
                                  json.loads((run_dir / "manifest.json").read_text()))
    robot, qadr, _ = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False,
                                          has_free=False, attached=False)
    base = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
    limits = build_limits(robot, layout, dt=base.horizon.dt, config=base.limits)
    rows = {int(x["seq"]): x for x in (json.loads(line) for line in
                                       (run_dir / "frames.jsonl").read_text().splitlines()
                                       if line.strip()) if x.get("kind") == "planning"}
    _SETUP[run] = dict(T=T, settings=settings, robot=robot, qadr=qadr, layout=layout,
                       outside=maxd if unk == "free" else -maxd, rows=rows,
                       meter=T.Meter(robot, layout, base.horizon.planned, settings.hand, limits),
                       fmap=FingerJointMap(robot))
    return _SETUP[run]


def _replay(run: str, seq: int, chunk: int, prev: int, *, gate: bool, switch: bool = True):
    """청크 하나를 새 refiner 로 푼다. `gate` 면 기록 상태로 `continuity_off` 를 정한다."""
    from benchmark.trajopt.safe_policy import grasp_continuity_off_reason

    sh = _setup(run)
    T, st, robot, layout, meter = sh["T"], sh["settings"], sh["robot"], sh["layout"], sh["meter"]
    z = np.load(CH / f"chunk_{chunk:05d}.npz", allow_pickle=False)
    zp = np.load(CH / f"chunk_{prev:05d}.npz", allow_pickle=False)
    s = json.loads(str(z["summary_json"]))
    state = (s.get("grasp") or {}).get("state")
    tm = (s.get("to") or {}).get("metrics") or {}
    q_now = T.q_now_from_qpos(sh["rows"][seq]["qpos"], sh["qadr"])
    field = T.build_field(z, outside_distance=sh["outside"])
    tp = z["target_points"] if "target_points" in z.files else None
    margin, mask, _, _, _ = T.contact_policy(
        st, robot.sphere_link_names, has_target=bool(s.get("has_target")) and tp is not None,
        has_free=bool(field.target_free_layers), attached=False)
    lin = meter.lin
    scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp,
                         margin=margin, mask=mask)
    refc = np.asarray(z["reference_chunk"], np.float64)
    meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
    path, _ = sh["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon,
                              default=robot.param_vector())
    cfg = T.to_config(st, collision=True, cost={**dict(st.cost), "grasp_continuity_off": switch},
                      sqp={"time_budget_ms": T.NO_CLOCK_MS,
                           "max_iterations": max(1, int(tm.get("sqp_iterations") or 1))})
    runner = T.Instrumented(robot, layout, cfg, {"value": (scene, q_now, s.get("status") == "ok")})
    runner.refiner.optimizer.linearizer.set_joint_parameter_path(path)
    ctx = {"t_step": seq, "previous_physical_chunk": np.asarray(zp["refined_chunk"], np.float64)}
    off = grasp_continuity_off_reason(state, cfg.cost.grasp_continuity_off) if gate else None
    if off:
        ctx["continuity_off"] = off
    result, _ = runner.run(refc, ctx)
    reference = layout.chunk_to_trajectory(refc)[:, :lin.horizon]
    cmp = meter.compare(result.trajectory, reference, q_now)
    return {"state": state, "off": off, "tip_mm": cmp["fingertip_disp_mm_max"],
            "dev_deg": cmp["joint_dev_max_deg"], "metrics": result.metrics,
            "trajectory": result.trajectory}


@pytest.mark.parametrize("run, seq, chunk, prev, before_mm", [
    ("E3b_ep1808_r1", 16, 315, 314, 19.6),   # t=120
    ("E3b_ep1808_r2", 15, 389, 388, 34.7),   # t=112
])
def test_closing_chunks_lose_the_fingertip_deviation(run, seq, chunk, prev, before_mm):
    base = _replay(run, seq, chunk, prev, gate=False)
    gated = _replay(run, seq, chunk, prev, gate=True)
    assert base["state"] == gated["state"] == "closing"
    # gate 없이는 T35 replay 값 그대로 (입력 복원이 맞다는 근거).
    assert base["tip_mm"] == pytest.approx(before_mm, abs=0.1)
    assert base["metrics"]["continuity_active"] is True
    # gate: 그 청크의 continuity 항만 빠지고, 손끝 편차가 사라진다.
    assert gated["off"] == "grasp latch closing (cost.grasp_continuity_off)"
    assert gated["metrics"]["continuity_active"] is False
    assert gated["metrics"]["continuity_off_by"] == gated["off"]
    assert gated["tip_mm"] < 0.5
    assert gated["dev_deg"] < 0.05


@pytest.mark.parametrize("run, seq, chunk, prev, tip_mm", [
    ("E3b_ep1808_r1", 6, 305, 304, 12.9),    # t=40, latched
    ("E3b_ep1808_r2", 8, 382, 381, 10.2),    # t=56, latched
])
def test_chunks_outside_closing_and_held_are_unchanged(run, seq, chunk, prev, tip_mm):
    base = _replay(run, seq, chunk, prev, gate=False)
    gated = _replay(run, seq, chunk, prev, gate=True)
    assert gated["state"] == "latched" and gated["off"] is None
    np.testing.assert_array_equal(gated["trajectory"], base["trajectory"])
    assert gated["metrics"]["continuity_active"] is True
    # continuity 가 실제로 무언가를 하고 있던 청크다 (T35: driver = no_continuity).
    assert gated["tip_mm"] == pytest.approx(tip_mm, abs=0.1)


def test_the_config_switch_restores_the_old_chunk():
    """`grasp_continuity_off: false` 면 closing 청크도 예전 궤적 그대로."""
    base = _replay("E3b_ep1808_r1", 16, 315, 314, gate=False)
    off = _replay("E3b_ep1808_r1", 16, 315, 314, gate=True, switch=False)
    assert off["off"] is None and off["metrics"]["continuity_active"] is True
    np.testing.assert_array_equal(off["trajectory"], base["trajectory"])
