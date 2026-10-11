"""T31 G1 — TO 의 joint **position** 범위는 제어 대상 로봇(MJCF)에서 온다.

E3 에서 정책은 `left_arm_6` 을 2.86 rad 까지 명령했고 (MuJoCo 범위 ±2.967), TO 는 URDF 범위
(±2.705 − margin 0.02 = 2.685) 로 그것을 잘랐다 — ep1800·1808 의 파지 실패. 이 파일이 박는 것:

| 절 | 무엇 |
|---|---|
| 1 | 기본 출처는 `model_xml` 이고, 비면 시뮬레이션의 로봇(`model_transport.xml`)이다 |
| 2 | MJCF 의 `jnt_range` 가 쓰이고 actuator `ctrlrange` 와 일치한다 (RB-Y1: `inheritrange=1`) |
| 3 | `urdf` · `config` 출처 — 예전 동작과 실기용 표 |
| 4 | 속도·가속도는 셋 모두 URDF 다 (MJCF 에 없다) |
| 5 | 시작 표 — 다르면 크게 |
| 6 | 서버 flag · 기록 meta · 오프라인 도구가 같은 출처를 본다 |
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import logging
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
from AG3S.trajopt import serve_safe
from AG3S.trajopt.config import (
    DEFAULT_LIMITS_MODEL_XML, LIMIT_SOURCES, NO_LIMITS, LimitsConfig, TrajOptConfig,
    TrajOptConfigError)
from AG3S.trajopt.limits import (
    _mjcf_range, build_limits, format_position_limit_table, joint_ranges_from_mj_model,
    limits_model_xml, mismatch_with_mj_model, mjcf_joint_ranges, position_limit_table,
    resolve_position_ranges)
from AG3S.trajopt.types import ChunkLayout

pytest.importorskip("mujoco")

DT = 1.0 / 15.0
MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'))
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", str(MODEL)]

#: `model_transport.xml` (→ `rby1.xml`) 의 값 — task 표와 같은 숫자. 손으로 옮겨 적은 **기대값**이고,
#: 코드는 이 표를 읽지 않는다 (MJCF 를 MuJoCo 로 컴파일해 읽는다).
MJCF_ARM = {
    "arm_0": (-2.35619, 2.35619), "arm_2": (-2.0944, 2.0944), "arm_4": (-6.28319, 6.28319),
    "arm_5": (-1.74533, 2.00713), "arm_6": (-2.96706, 2.96706),
}
URDF_ARM_6 = 2.705260340


@pytest.fixture(scope="module")
def robot():
    return load_rby1()


@pytest.fixture(scope="module")
def layout():
    return ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)


def _index(layout, name):
    names = [layout.joint_names[i] for i in layout.q_indices]
    return names.index(name)


# --- 1. 기본 출처 --------------------------------------------------------------------------


def test_the_default_source_is_the_simulated_robot():
    cfg = LimitsConfig()
    assert cfg.source == "model_xml" and cfg.model_xml is None
    assert LIMIT_SOURCES == ("model_xml", "urdf", "config")
    assert TrajOptConfig().limits.source == "model_xml"
    assert limits_model_xml(cfg) == DEFAULT_LIMITS_MODEL_XML
    assert limits_model_xml(LimitsConfig(model_xml="/x.xml")) == "/x.xml"


def test_the_default_model_is_the_transport_scene_the_simulator_loads():
    """기본 경로는 `mujoco_source.TRANSPORT_MODEL` 과 **같은 파일**이다 — 두 곳에 적힌 경로가
    갈라지면 한계와 시뮬레이터가 다른 로봇을 본다."""
    from AG3S.ag3s.experiments.sources.mujoco_source import TRANSPORT_MODEL

    assert pathlib.PurePath(DEFAULT_LIMITS_MODEL_XML) == pathlib.PurePath(TRANSPORT_MODEL)


# --- 2. MJCF 의 범위 --------------------------------------------------------------------------


def test_build_limits_uses_the_mjcf_range_with_the_margin(robot, layout):
    """`left_arm_6` 상한: MJCF 2.96706 − 0.02 (T31) — URDF 2.70526 − 0.02 = 2.685 가 아니다."""
    limits = build_limits(robot, layout, dt=DT)
    for side in ("left", "right"):
        for short, (lo, hi) in MJCF_ARM.items():
            i = _index(layout, f"{side}_{short}")
            assert limits.lower[i] == pytest.approx(lo + 0.02, abs=1e-5), f"{side}_{short}"
            assert limits.upper[i] == pytest.approx(hi - 0.02, abs=1e-5), f"{side}_{short}"
    i6 = _index(layout, "left_arm_6")
    assert limits.upper[i6] > URDF_ARM_6, "E3 의 2.86 rad 명령이 box 안이어야 한다"
    assert limits.upper[i6] > 2.86


def test_every_optimized_joint_agrees_with_its_position_actuator_ctrlrange(robot, layout):
    """RB-Y1 MJCF 는 `inheritrange="1"` — `jnt_range` 와 `ctrlrange` 가 같아야 한다."""
    table = position_limit_table(robot, layout)
    optimized = [r for r in table["rows"] if r["optimized"]]
    assert len(optimized) == layout.nq_opt == 14
    assert all(r["ctrlrange_agrees"] is True for r in optimized), [
        (r["joint"], r["note"]) for r in optimized if r["ctrlrange_agrees"] is not True]
    assert table["n_ctrl_mismatch"] == 0
    raw = mjcf_joint_ranges(DEFAULT_LIMITS_MODEL_XML)
    for r in optimized:
        assert tuple(r["used"]) == pytest.approx(raw[r["joint"]]["range"])


def test_ctrlrange_disagreement_uses_the_intersection_and_says_so():
    entry = {"limited": True, "range": (-2.0, 2.0),
             "actuators": [{"name": "a", "position": True, "ctrllimited": True,
                            "ctrlrange": (-1.5, 2.5)}]}
    lo, hi, note, agrees = _mjcf_range("j", entry)
    assert (lo, hi) == (-1.5, 2.0) and agrees is False and "intersection" in note


def test_a_velocity_actuator_ctrlrange_is_not_read_as_a_position_range():
    entry = {"limited": True, "range": (-2.0, 2.0),
             "actuators": [{"name": "v", "position": False, "ctrllimited": True,
                            "ctrlrange": (-20.0, 20.0)}]}
    lo, hi, _, agrees = _mjcf_range("j", entry)
    assert (lo, hi) == (-2.0, 2.0) and agrees is None


def test_an_unlimited_joint_without_a_position_ctrlrange_is_refused():
    """범위를 **지어내지 않는다** — `limited=false` 이고 position actuator 도 없으면 예외."""
    with pytest.raises(ValueError, match="no position range"):
        _mjcf_range("j", {"limited": False, "range": (0.0, 0.0), "actuators": []})
    lo, hi, _, _ = _mjcf_range("j", {"limited": False, "range": (0.0, 0.0), "actuators": [
        {"name": "p", "position": True, "ctrllimited": True, "ctrlrange": (-1.0, 1.0)}]})
    assert (lo, hi) == (-1.0, 1.0)


def test_a_joint_missing_from_the_mjcf_is_refused(layout):
    class Other:
        nq = 2
        joint_names = ("not_a_joint", "left_arm_0")

        def joint_limits(self):
            return np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    lay = ChunkLayout(action_dim=2, n_valid=2, action_to_q=((0, 0), (1, 1)), fixed_q=(),
                      passthrough=(), nq_model=2)
    with pytest.raises(ValueError, match="not 1-DoF joints"):
        resolve_position_ranges(Other(), LimitsConfig(), index=lay.q_indices)


def test_a_model_without_joint_names_needs_the_urdf_source():
    class Anonymous:
        nq = 1

        def joint_limits(self):
            return np.array([-1.0]), np.array([1.0])

    with pytest.raises(ValueError, match="joint name"):
        resolve_position_ranges(Anonymous(), LimitsConfig())
    lo, hi, _ = resolve_position_ranges(Anonymous(), LimitsConfig(source="urdf"))
    assert (lo[0], hi[0]) == (-1.0, 1.0)


def test_the_running_mj_model_matches_the_default_source(robot, layout):
    """bringup 검사: 컴파일된 모델과 설정된 출처를 **값으로** 대조한다."""
    import mujoco

    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert mismatch_with_mj_model(robot, layout, LimitsConfig(), mj_model) == []
    bad = mismatch_with_mj_model(robot, layout, LimitsConfig(source="urdf"), mj_model)
    assert {b["joint"] for b in bad} == {f"{s}_arm_{k}" for s in ("left", "right")
                                          for k in range(7)}
    assert set(joint_ranges_from_mj_model(mj_model)) >= set(DEFAULT_RBY1_JOINTS)


# --- 3. urdf · config 출처 --------------------------------------------------------------------


def test_the_urdf_source_is_the_pre_t31_behaviour(robot, layout):
    lower_all, upper_all = robot.joint_limits()
    limits = build_limits(robot, layout, dt=DT, config=LimitsConfig(source="urdf"))
    index = layout.q_indices
    np.testing.assert_allclose(limits.lower, lower_all[index] + 0.02)
    np.testing.assert_allclose(limits.upper, upper_all[index] - 0.02)
    assert limits.upper[_index(layout, "left_arm_6")] == pytest.approx(URDF_ARM_6 - 0.02)


def _ranges(layout, **override):
    names = [layout.joint_names[i] for i in layout.q_indices]
    table = {n: [-1.0, 1.0] for n in names}
    table.update(override)
    return table


def test_the_config_source_reads_the_written_table(robot, layout):
    table = _ranges(layout, left_arm_6=[-2.5, 2.9])
    limits = build_limits(robot, layout, dt=DT,
                          config=LimitsConfig(source="config", position_ranges=table))
    i = _index(layout, "left_arm_6")
    assert (limits.lower[i], limits.upper[i]) == pytest.approx((-2.48, 2.88))
    assert limits.upper[_index(layout, "left_arm_0")] == pytest.approx(0.98)


def test_the_config_source_must_name_every_optimized_joint(robot, layout):
    table = _ranges(layout)
    table.pop("right_arm_3")
    with pytest.raises(ValueError, match="right_arm_3"):
        build_limits(robot, layout, dt=DT,
                     config=LimitsConfig(source="config", position_ranges=table))
    with pytest.raises(ValueError, match="does not have"):
        build_limits(robot, layout, dt=DT, config=LimitsConfig(
            source="config", position_ranges={**_ranges(layout), "elbow_x": [0, 1]}))


@pytest.mark.parametrize("data, match", [
    ({"source": "mjcf"}, "limits.source"),
    ({"source": "config"}, "position_ranges"),
    ({"source": "config", "position_ranges": {"a": [1.0, -1.0]}}, "lower < upper"),
    ({"source": "config", "position_ranges": {"a": [1.0]}}, r"\[lower, upper\]"),
    ({"source": "urdf", "position_ranges": {"a": [-1.0, 1.0]}}, "only read"),
])
def test_bad_limit_sources_are_refused(data, match):
    with pytest.raises(TrajOptConfigError, match=match):
        TrajOptConfig.from_dict({"limits": data})


# --- 4. 속도 · 가속도는 URDF 그대로 ------------------------------------------------------------


@pytest.mark.parametrize("source", ["model_xml", "urdf"])
def test_velocity_and_acceleration_still_come_from_the_urdf(robot, layout, source):
    """MJCF 에는 관절 속도·가속도 한계가 없다 (position actuator 와 `jnt_range` 뿐)."""
    limits = build_limits(robot, layout, dt=DT, config=LimitsConfig(source=source))
    index = layout.q_indices
    np.testing.assert_allclose(limits.max_step, robot.velocity_limits()[index] * 0.9 * DT)
    np.testing.assert_allclose(limits.max_step_change,
                               robot.acceleration_limits()[index] * 0.9 * DT * DT)


def test_no_limits_still_opens_the_box_whatever_the_source(robot, layout):
    off = build_limits(robot, layout, dt=DT, config=LimitsConfig(**NO_LIMITS))
    assert np.all(np.isinf(off.lower)) and np.all(np.isinf(off.upper))


# --- 5. 시작 표 ---------------------------------------------------------------------------------


def test_the_table_marks_every_optimized_joint_that_differs(robot, layout):
    table = position_limit_table(robot, layout)
    assert table["source"] == "model_xml" and table["model_xml"].endswith("model_transport.xml")
    assert table["n_differs"] == 14  # 팔 14 관절 전부 URDF 와 다르다 (arm_1/arm_3 는 수 mrad)
    text = format_position_limit_table(table)
    row = next(line for line in text.splitlines() if line.strip().startswith("left_arm_6"))
    assert "<< DIFFERS" in row and "+2.9671" in row and "+2.7053" in row
    bounds = {r["joint"]: r["bound"] for r in table["rows"]}
    assert bounds["left_arm_6"] == pytest.approx([-2.94706, 2.94706], abs=1e-5)
    assert bounds["torso_0"] is None  # 움직이지 않는 관절은 box 에 없다


def test_a_urdf_run_prints_a_quiet_table(robot, layout):
    table = position_limit_table(robot, layout, LimitsConfig(source="urdf"))
    assert table["n_differs"] == 0
    assert "DIFFERS" not in format_position_limit_table(table)


def test_the_server_shouts_when_the_limits_differ(robot, layout, caplog):
    with caplog.at_level(logging.INFO, logger=serve_safe.__name__):
        serve_safe.announce_joint_limits(robot, layout, LimitsConfig())
    loud = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert loud and "DIFFER FROM THE URDF ON 14" in loud[0].getMessage()
    assert "left_arm_6" in loud[0].getMessage()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=serve_safe.__name__):
        serve_safe.announce_joint_limits(robot, layout, LimitsConfig(source="urdf"))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("source: urdf" in r.getMessage() for r in caplog.records)


# --- 6. 서버 · 기록 · 오프라인 도구 ----------------------------------------------------------------


def test_the_server_reads_the_limits_from_its_own_model_xml():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert args.limits_source == "model_xml"
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits.source == "model_xml" and cfg.limits.model_xml == str(MODEL)
    meta = serve_safe.joint_limits_meta(cfg.limits)
    assert meta == {"source": "model_xml", "position_margin": 0.02,
                    "velocity_acceleration": "urdf", "model_xml": str(MODEL)}


def test_the_server_urdf_and_config_sources(tmp_path, layout):
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--limits-source", "urdf"])
    serve_safe.reject_bad_flag_combinations(ap, args)
    assert serve_safe.trajopt_config_from_args(args).limits.source == "urdf"

    path = tmp_path / "ranges.yaml"
    path.write_text(json.dumps(_ranges(layout, left_arm_6=[-2.9, 2.9])))
    args = ap.parse_args(BASE_ARGV + ["--limits-source", "config", "--joint-ranges", str(path)])
    serve_safe.reject_bad_flag_combinations(ap, args)
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits.source == "config"
    assert cfg.limits.position_ranges["left_arm_6"] == [-2.9, 2.9]
    assert serve_safe.joint_limits_meta(cfg.limits)["position_ranges"]["left_arm_6"] == [-2.9, 2.9]


@pytest.mark.parametrize("extra", [
    ["--joint-ranges", "/r.yaml"],                      # 표를 줬는데 출처가 config 가 아님
    ["--limits-source", "config"],                      # config 인데 표가 없음
    ["--limits-source", "urdf", "--no-safe"],           # TO 가 없는데 TO 의 flag
    ["--sqp-min-iterations", "2", "--no-safe"],
    ["--sqp-min-iterations", "0"],
])
def test_inconsistent_limit_flags_are_refused(extra):
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + extra)
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_no_limits_keeps_the_model_xml_source():
    args = serve_safe.build_parser().parse_args(BASE_ARGV + ["--no-limits"])
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits == LimitsConfig(**NO_LIMITS, model_xml=str(MODEL))


def test_the_record_meta_always_names_the_limit_source():
    """기본값이 바뀌었으므로 (URDF → MJCF) **항상** 남긴다 — 소스로 확인한다."""
    src = pathlib.Path(serve_safe.__file__).read_text()
    assert src.count('"joint_limits": joint_limits_meta(') == 2  # SafePolicy · ToOnly 기록기
    assert src.count('"sqp": sqp_meta(') == 2
    assert src.count("announce_joint_limits(served.") == 2


def test_the_offline_tools_use_the_same_source():
    """`esdf_rollout` 은 재생하는 기록의 MJCF, `bringup` 은 `mj_model` 과 값으로 대조한다."""
    from AG3S.trajopt import bringup
    from AG3S.trajopt.experiments import esdf_rollout

    rollout = pathlib.Path(esdf_rollout.__file__).read_text()
    assert '"model_xml": str(run.model_xml)' in rollout
    assert "position_limit_table(robot, layout, to_cfg.limits)" in rollout
    live = pathlib.Path(bringup.__file__).read_text()
    assert "mismatch_with_mj_model(constraint_robot, layout, to_cfg.limits, mj_model)" in live
