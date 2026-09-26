"""기록 형식 — **옛 키가 그대로이고 새 키가 실린다.**

`T6a` 가 planning/control 기록에 청크와 참값을 더했다. 그 변경이 지켜야 하는 것은 형식이다:
`plot_t0.py:37` 이 `frames.jsonl` 을 읽고 completeness 표의 항목들이 지금 그대로 나와야 한다.
그래서 이 파일은 **키 집합을 문자 그대로 고정한다** — 옛 키가 하나라도 사라지거나 이름이
바뀌면 여기서 깨진다.

`_jsonable` 을 따로 고정하는 이유는 그 실패가 조용하기 때문이다. `json.dumps(default=str)` 는
`ndarray` 를 만나도 예외를 내지 않고 `"[[0.1 ... 0.9]]"` 를 적는다 — **기록은 있는데 값이
없다.** T6a 가 청크와 `qpos` 를 싣기 시작하면서 그 경로가 실제로 열렸다.
"""

from __future__ import annotations

import json

import numpy as np

from benchmark.ag3s.runtime.frame_record import FrameRecorder, collect_manifest

#: T0 이 낸 기록의 키. **이 목록을 줄이는 변경은 회귀다.**
OBSERVATION_KEYS = {
    "kind", "t_step", "camera_stamps", "newest_stamp", "camera_skew_sec",
    "ag3s_status", "grounding_status", "validity", "n_points",
    "timestamp_reversed", "notes",
}
PLANNING_KEYS_T0 = {
    "kind", "seq", "t_step", "duplicate_sequence", "ipc", "field", "verdict", "timing_ms",
}
#: T6a 가 더한 것. `actions_reference` 는 shadow 일 때만 실리므로 여기 없다.
PLANNING_KEYS_T6A = {"actions", "qpos", "object_poses", "max_violation_pair"}
CONTROL_KEYS_T0 = {
    "kind", "t_step", "chunk_seq", "step_in_chunk", "now", "executed", "field",
    "last_field_update",
}
CONTROL_KEYS_T6A = {"qpos"}

COMPLETENESS_KEYS = {
    "expected_observation_frames", "captured_observation_frames", "planning_records",
    "control_records", "third_person_frames", "planned_missing_frames",
    "duplicate_sequence_ids", "timestamp_reversals", "unexplained_carried_or_stale",
    "field_state_counts", "ipc_outcomes", "staleness_limit_sec", "staleness_checked",
    "pass",
}


def _manifest() -> dict:
    return collect_manifest(seed=7, scene={"model": "rby1_16d"}, policy={}, esdf={},
                            timing={"max_field_age_sec": None}, cameras=("zed_left",),
                            render={})


def _recorder(tmp_path, **kwargs) -> FrameRecorder:
    return FrameRecorder(str(tmp_path / "ep"), manifest=_manifest(), **kwargs)


def _rows(tmp_path) -> list[dict]:
    text = (tmp_path / "ep" / "frames.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- 옛 키가 그대로다 -------------------------------------------------------------------
def test_old_keys_survive(tmp_path):
    """새 인자를 하나도 주지 않은 호출이 T0 의 키 집합 + T6a 의 `null` 키를 낸다."""
    rec = _recorder(tmp_path, expected_observation_frames=1)
    rec.observation(t_step=0, stamps={"zed_left": 1.0})
    rec.planning(seq=1, t_step=0, field=None, verdict={"safe": True},
                 timing_ms={"policy_infer": 12.0})
    rec.control(t_step=0, chunk_seq=1, step_in_chunk=0, now=2.0, field=None, executed=True)
    rec.third_person(t_step=0, path="x.png")
    rec.close()

    obs, plan, ctrl, third = _rows(tmp_path)
    assert set(obs) == OBSERVATION_KEYS
    assert set(plan) == PLANNING_KEYS_T0 | PLANNING_KEYS_T6A
    assert set(ctrl) == CONTROL_KEYS_T0 | CONTROL_KEYS_T6A
    assert set(third) == {"kind", "t_step", "path"}
    # 값을 안 준 새 키는 **키는 있고 값이 없다.** A2 가 프레임을 세로로 세므로 있음/없음이
    # 섞이면 파서가 두 갈래로 갈린다.
    assert all(plan[k] is None for k in PLANNING_KEYS_T6A)
    assert ctrl["qpos"] is None
    # `actions_reference` 는 shadow 일 때만이다 — 규약이 응답 쪽과 같다.
    assert "actions_reference" not in plan


def test_completeness_table_is_unchanged(tmp_path):
    """completeness 표의 항목과 통과 조건. `verify_t0`·`plot_t0` 가 이 이름들을 읽는다."""
    rec = _recorder(tmp_path, expected_observation_frames=2)
    rec.observation(t_step=0, stamps={"zed_left": 1.0})
    rec.observation(t_step=8, stamps={"zed_left": 2.0})
    rec.planning(seq=1, t_step=0, field=None, verdict={}, timing_ms={},
                 qpos=[0.1], object_poses={"apple": {"pos": [0, 0, 0]}})
    rec.control(t_step=0, chunk_seq=1, step_in_chunk=0, now=2.0, field=None,
                executed=True, qpos=[0.1])
    table = rec.close()

    assert set(table) == COMPLETENESS_KEYS
    assert table["expected_observation_frames"] == 2
    assert table["captured_observation_frames"] == 2
    assert table["planning_records"] == 1
    assert table["control_records"] == 1
    assert table["planned_missing_frames"] == 0
    assert table["unexplained_carried_or_stale"] == 0
    assert table["staleness_checked"] is False
    assert table["pass"] is True
    # 파일로도 같은 표가 나가야 한다 — `plot_t0.load` 가 그 파일을 읽는다.
    on_disk = json.loads((tmp_path / "ep" / "completeness.json").read_text())
    assert set(on_disk) == COMPLETENESS_KEYS


# --- 새 키가 실린다 ---------------------------------------------------------------------
def test_planning_carries_both_chunks(tmp_path):
    """`actions`(refined) 와 `actions_reference`(원본)가 **값으로** 실린다.

    `T5c` 가 `refined` 대 `reference` 여유거리 비교를 미측정으로 닫은 것이 이 두 배열이
    기록에 없었기 때문이다.
    """
    refined = np.arange(50 * 16, dtype=np.float32).reshape(50, 16) * 0.001
    reference = refined + 0.5
    rec = _recorder(tmp_path)
    rec.planning(seq=3, t_step=8, field=None, verdict={}, timing_ms={},
                 actions=refined, actions_reference=reference)
    rec.close()

    row = _rows(tmp_path)[0]
    got = np.asarray(row["actions"], np.float32)
    assert got.shape == (50, 16)
    np.testing.assert_allclose(got, refined)
    np.testing.assert_allclose(np.asarray(row["actions_reference"], np.float32), reference)
    # 되읽을 수 있는 숫자여야 한다 — 문자열 repr 이 아니다.
    assert isinstance(row["actions"][0][0], float)


def test_planning_carries_qpos_object_poses_and_pair(tmp_path):
    qpos = np.linspace(-1.0, 1.0, 66)
    poses = {"apple": {"pos": [0.5, 0.1, 0.85], "quat": [1.0, 0.0, 0.0, 0.0]},
             "crate": {"pos": [0.7, -0.2, 0.83], "quat": [1.0, 0.0, 0.0, 0.0]}}
    pair = {"block": "esdf", "clearance_m": -0.033, "step": 12,
            "link": "ee_finger_r1", "obstacle": "apple"}
    rec = _recorder(tmp_path)
    rec.planning(seq=4, t_step=16, field=None, verdict={}, timing_ms={},
                 qpos=qpos, object_poses=poses, max_violation_pair=pair)
    rec.close()

    row = _rows(tmp_path)[0]
    assert len(row["qpos"]) == 66
    np.testing.assert_allclose(np.asarray(row["qpos"]), qpos)
    assert row["object_poses"] == poses
    assert row["max_violation_pair"] == pair


def test_control_carries_qpos(tmp_path):
    """control frame 의 `qpos` — 멈춘 구간에서 팔이 정말 안 움직였는지가 여기서 나온다."""
    rec = _recorder(tmp_path)
    for step in range(3):
        rec.control(t_step=step, chunk_seq=1, step_in_chunk=step, now=float(step),
                    field=None, executed=False, qpos=np.full(66, 0.25 + step))
    rec.close()

    rows = _rows(tmp_path)
    assert [r["qpos"][0] for r in rows] == [0.25, 1.25, 2.25]
    assert all(len(r["qpos"]) == 66 for r in rows)


# --- 배열이 조용히 문자열로 적히지 않는다 ----------------------------------------------
def test_numpy_never_degrades_to_a_repr_string(tmp_path):
    """`json.dumps(default=str)` 의 조용한 실패를 막는다.

    큰 배열을 `str()` 로 적으면 numpy 가 줄임표를 넣어 **되읽을 수 없는** 값이 남는데,
    예외가 나지 않으므로 기록이 망가진 것을 아무도 모른다.
    """
    big = np.arange(2000, dtype=np.float64).reshape(50, 40)
    rec = _recorder(tmp_path)
    rec.planning(seq=1, t_step=0, field=None, verdict={}, timing_ms={},
                 qpos=big, extra={"nested": {"inner": np.arange(3)},
                                  "scalar": np.float32(1.5),
                                  "tupled": (np.int64(2), np.int64(3))})
    rec.close()

    text = (tmp_path / "ep" / "frames.jsonl").read_text()
    assert "..." not in text
    row = json.loads(text)
    assert np.asarray(row["qpos"]).shape == (50, 40)
    assert row["nested"] == {"inner": [0, 1, 2]}
    assert row["scalar"] == 1.5
    assert row["tupled"] == [2, 3]
