"""T11 후속 — 기록기가 cuRobo 필드를 싣고, hold 청크가 **이름으로** 관절을 찾는다.

두 결함 다 **조용했다**는 것이 공통점이다.

1. `--record-constraints` 가 cuRobo backend 에서 청크마다 죽었는데 (`'CuroboEsdfField' object
   has no attribute 'max_distance'`) 실패가 `print` 한 줄이라 75 chunk 를 돌리고도 실행이 끝까지
   갔다. 끝나고 남은 것은 **빈 디렉토리**였다.
2. `SafeRemoteClient._current_state()` 가 20-vector(torso 6 · 오른팔 7 · 왼팔 7)를 `state[:14]`
   로 잘라 "양팔 14" 로 읽었다. **14 개 열 전부 엉뚱한 관절**이고, 그리퍼 자리에 넣은 `0.0` 은
   주석이 말하는 "열림" 이 아니라 **닫힘**이다 (`ctrl = norm × −0.045`).

둘 다 *"틀린 값이 형태는 맞아서 그대로 지나간다"* 는 같은 종류의 실패다. 그래서 이 파일은
**값**을 박는다 — 어느 관절이 어느 열에 오는지, 어느 계층이 어느 키로 실리는지.
"""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

from benchmark.ag3s.experiments.sources.constraint_record import (
    SUPPORTED_ESDF_BACKENDS,
    ConstraintRecordWriter,
    load_constraint_run,
)
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
from benchmark.ag3s.fields.esdf import VoxelGrid
from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS
from benchmark.trajopt import wire
from benchmark.trajopt.client import HOLD_GRIPPER_NORM, SafeRemoteClient


# --- 기록기: 계층이 여럿인 필드 ---------------------------------------------------------


def _layer(lower, upper, voxel, fn):
    grid = VoxelGrid.from_bounds(np.asarray(lower, float), np.asarray(upper, float), voxel)
    return layer_from_arrays(fn(grid.centres()).astype(np.float32), grid.origin, voxel)


def _composite(*, with_free=True):
    """coarse(20 mm) + fine(5 mm) + target 없는 계층 — 실제 서버가 내놓는 모양."""
    coarse = _layer((0.0, -0.3, 0.0), (1.0, 0.3, 0.5), 0.02, lambda p: p[..., 2])
    fine = _layer((0.4, -0.1, -0.05), (0.6, 0.1, 0.15), 0.005, lambda p: p[..., 2])
    free = _layer((0.4, -0.1, -0.05), (0.6, 0.1, 0.15), 0.005, lambda p: p[..., 2] + 0.01)
    field = CuroboEsdfField((coarse, fine), outside_distance=0.5,
                            target_free_layers=(free,) if with_free else ())
    field.stats = {"backend": "curobo", "n_occupied": 123}
    return field


def _constraint_set(field):
    return types.SimpleNamespace(
        esdf=field, target=None, candidates=[], support_surfaces=[], status=None,
        validity=None, notes=[], phase=None, frame_index=0, grounding_status=None,
        metrics={}, contact_context=None, robot_state=np.zeros(20), attached=None,
        timestamp=0.0, frame_id="base", profile={}, constraints=None, manipulated=None,
        destination_label=None)


def _record(tmp_path, field, *, mode="full", occupancy=None):
    writer = ConstraintRecordWriter(tmp_path, esdf_mode=mode)
    writer.record(t_step=1, chunk_index=0, constraint_set=_constraint_set(field), debug={},
                  occupancy=occupancy)
    writer.close()
    meta, chunks = load_constraint_run(writer.run_dir)
    assert len(chunks) == 1, "기록이 하나는 남아야 한다 — 빈 디렉토리가 이 시험의 반대다"
    return chunks[0]


def test_a_curobo_field_records_instead_of_raising(tmp_path):
    """**이 시험이 회귀다.** 예전에는 `max_distance` 에서 `AttributeError` 로 죽었다."""
    chunk = _record(tmp_path, _composite())
    assert chunk["summary"]["esdf_max_distance"] > 0.0
    assert chunk["summary"]["esdf_stats"]["backend"] == "curobo"


def test_every_layer_is_recorded_with_its_own_grid(tmp_path):
    """계층마다 격자가 다르다. 하나만 실으면 나머지 계층의 값을 어느 좌표로도 놓을 수 없다."""
    chunk = _record(tmp_path, _composite())
    layers = {row["name"]: row for row in chunk["summary"]["esdf_layers"]}
    assert set(layers) == {"main", "coarse", "free"}
    # 주 격자 = **가장 미세한 계층** (질의의 허용오차가 그것을 쓴다).
    assert layers["main"]["voxel_size"] == pytest.approx(0.005)
    assert layers["coarse"]["voxel_size"] == pytest.approx(0.02)
    for prefix in ("esdf_", "esdf_coarse_", "esdf_free_"):
        assert prefix + "origin" in chunk and prefix + "shape" in chunk
        assert prefix + "voxel_size" in chunk
        assert prefix + "distance" in chunk, f"{prefix}: full 모드는 거리장을 싣는다"
    # 주 계층의 격자 메타가 옛 키 이름 그대로여야 옛 판독 코드가 돈다.
    assert tuple(chunk["esdf_shape"]) == tuple(layers["main"]["shape"])


def test_the_target_free_layer_is_the_one_the_fingertips_query(tmp_path):
    """유령을 찾을 때 봐야 하는 계층이다 (T10a). 값이 주 계층과 **다르다**는 것을 박는다."""
    chunk = _record(tmp_path, _composite())
    main = np.asarray(chunk["esdf_distance"], np.float64)
    free = np.asarray(chunk["esdf_free_distance"], np.float64)
    assert main.shape == free.shape
    assert not np.allclose(main, free), "두 계층이 같은 값이면 기록이 한쪽을 덮어쓴 것이다"


def test_a_field_without_the_free_layer_records_only_two(tmp_path):
    chunk = _record(tmp_path, _composite(with_free=False))
    assert {row["name"] for row in chunk["summary"]["esdf_layers"]} == {"main", "coarse"}
    assert "esdf_free_distance" not in chunk


def test_a_legacy_single_layer_field_records_exactly_as_before(tmp_path):
    """옛 경로가 한 바이트도 안 바뀌어야 한다 — 그쪽 기록으로 이미 낸 결과가 있다."""
    single = _layer((0.0, -0.3, 0.0), (1.0, 0.3, 0.5), 0.02, lambda p: p[..., 2])
    single.stats = {"backend": "legacy"}
    chunk = _record(tmp_path, single)
    assert [row["name"] for row in chunk["summary"]["esdf_layers"]] == ["main"]
    assert "esdf_distance" in chunk
    assert not [k for k in chunk if k.startswith(("esdf_coarse", "esdf_free"))]


@pytest.mark.parametrize("mode,has_distance", [("none", False), ("occupancy", False),
                                               ("full", True)])
def test_the_mode_decides_what_is_written_but_never_whether_it_writes(tmp_path, mode,
                                                                     has_distance):
    chunk = _record(tmp_path, _composite(), mode=mode)
    assert ("esdf_distance" in chunk) is has_distance
    # 어느 모드에서든 격자 메타는 남는다 — 그게 이 기록의 좌표계다.
    assert "esdf_origin" in chunk and chunk["summary"]["esdf_layers"]


def test_a_missing_occupancy_array_says_so_instead_of_looking_empty(tmp_path):
    """cuRobo 에는 3 상태 점유 배열이 없다. 없음을 안 적으면 "점유가 비었던 프레임" 으로 읽힌다."""
    chunk = _record(tmp_path, _composite(), mode="occupancy")
    assert "esdf_occupancy" not in chunk
    assert chunk["summary"]["esdf_occupancy_available"] is False
    assert "full" in chunk["summary"]["esdf_occupancy_reason"]
    # 배열이 오면 그 키는 아예 안 생긴다 (있음이 기본이므로 적을 필요가 없다).
    got = _record(tmp_path, _composite(), mode="occupancy",
                  occupancy=np.zeros((4, 4, 4), np.uint8))
    assert "esdf_occupancy" in got
    assert "esdf_occupancy_available" not in got["summary"]


def test_an_unknown_field_type_is_refused_not_silently_skipped():
    """조용히 빈 목록을 돌려주면 **격자 없는 기록**이 남고, 그것은 없는 기록보다 나쁘다."""
    with pytest.raises(TypeError, match="거리장 인터페이스"):
        ConstraintRecordWriter.field_layers(object())
    assert ConstraintRecordWriter.supports_field(object()) is False
    assert ConstraintRecordWriter.supports_field(_composite()) is True


def test_the_supported_backends_are_named_so_the_server_can_refuse_at_startup():
    assert set(SUPPORTED_ESDF_BACKENDS) == {"legacy", "curobo"}


def test_the_server_refuses_to_record_an_unsupported_backend():
    """**시작할 때** 죽어야 한다. 끝나고 빈 디렉토리를 보는 것이 최악이다."""
    import inspect

    from benchmark.trajopt import serve_safe

    src = inspect.getsource(serve_safe.main)
    assert "SUPPORTED_ESDF_BACKENDS" in src
    assert "record_constraints" in src


def test_a_failing_record_is_shouted_once_not_printed_seventy_five_times(caplog):
    """첫 실패는 크게, 그 뒤는 한 줄씩. 같은 문장을 75 번 읽으면 로그를 건너뛰기 시작한다."""
    from benchmark.trajopt import safe_policy as sp_module
    from tests.trajopt.test_safe_policy import _safe_policy

    sp = _safe_policy()
    sp.recorder = types.SimpleNamespace(
        record=lambda **kw: (_ for _ in ()).throw(AttributeError("no max_distance")),
        run_dir="/tmp/nowhere")
    with caplog.at_level(logging.WARNING, logger=sp_module.__name__):
        sp._record(1, np.zeros((5, wire.ACTION_WIDTH)), np.zeros((5, wire.ACTION_WIDTH)))
        first = caplog.text
        sp._record(2, np.zeros((5, wire.ACTION_WIDTH)), np.zeros((5, wire.ACTION_WIDTH)))
    assert "THIS RUN WILL HAVE NO RECORD" in first
    assert "AttributeError" in first
    assert sp._record_failures == 2
    assert caplog.text.count("THIS RUN WILL HAVE NO RECORD") == 1, "두 번째는 한 줄이다"


# --- hold 청크: 이름으로 찾는다 ---------------------------------------------------------


class _Scene:
    """`TransportScene` 중 hold 경로가 쓰는 것만. `robot_state` 는 **20-vector** 다."""

    def __init__(self, state, gripper=None):
        self.state = np.asarray(state, np.float64)
        self._gripper = gripper

    def robot_state(self):
        return self.state

    def capture(self, _camera):
        raise RuntimeError("이 시험은 카메라를 찍지 않는다 (hold 경로만 본다)")

    if True:  # `gripper_norm` 은 있을 때만 쓰인다 — 아래에서 동적으로 붙인다
        pass


class _DeadServer:
    def infer(self, _request):
        raise ConnectionError("socket closed")


def _hold_chunk(state, *, gripper=None):
    scene = _Scene(state)
    if gripper is not None:
        scene.gripper_norm = lambda: gripper
    client = SafeRemoteClient(policy=_DeadServer(), scene=scene, cameras=())
    out = client.infer({"state": np.zeros(wire.ACTION_WIDTH)})
    assert out["safe"] is False, "서버가 죽었으므로 hold 여야 한다"
    return np.asarray(out["actions"], np.float64)


def _q_by_name():
    return {name: i for i, name in enumerate(DEFAULT_RBY1_JOINTS)}


def test_the_hold_chunk_takes_each_arm_joint_by_name():
    """**이 시험이 회귀다.** 위치로 자르면 왼팔 열에 `torso_0..5` 가 들어갔다."""
    q = np.arange(len(DEFAULT_RBY1_JOINTS), dtype=np.float64)  # 값 = 그 관절의 인덱스
    hold = _hold_chunk(q)
    assert hold.shape == (50, wire.ACTION_WIDTH)
    names = _q_by_name()
    n = wire.ARM_JOINT_DIM
    for i in range(n):
        assert hold[0, i] == names[f"left_arm_{i}"], f"왼팔 {i} 열"
        assert hold[0, n + 1 + i] == names[f"right_arm_{i}"], f"오른팔 {i} 열"
    # torso 는 action 형식이 가리키지 못한다 — 어느 열에도 없어야 한다.
    torso = {names[f"torso_{i}"] for i in range(6)}
    arm_columns = [c for c in range(wire.ACTION_WIDTH) if c not in wire.gripper_columns(n)]
    assert not (torso & {hold[0, c] for c in arm_columns}), "torso 가 팔 열에 새면 안 된다"


def test_every_row_of_the_hold_chunk_is_the_same_pose():
    """hold 는 정지가 아니라 **유지**다 — 50 행이 전부 같아야 한다."""
    hold = _hold_chunk(np.arange(len(DEFAULT_RBY1_JOINTS), dtype=np.float64))
    assert np.allclose(hold, hold[0][None, :])


def test_the_gripper_column_is_the_open_value_and_open_is_one():
    """`ctrl = norm × RBY1_GRIPPER_OPEN(−0.045)` 이므로 **1.0 이 열림, 0.0 이 닫힘**이다.

    예전 코드는 `0.0` 을 넣고 주석에 "열림" 이라고 적었다 — 값과 주석이 반대였고, 그 값은
    그리퍼를 **닫는다.**
    """
    assert HOLD_GRIPPER_NORM == 1.0
    hold = _hold_chunk(np.zeros(len(DEFAULT_RBY1_JOINTS)))
    for column in wire.gripper_columns(wire.ARM_JOINT_DIM):
        assert hold[0, column] == HOLD_GRIPPER_NORM
        assert hold[0, column] != 0.0, "0.0 은 닫힘이다"


def test_a_scene_that_knows_its_grippers_is_read_instead_of_guessed():
    hold = _hold_chunk(np.zeros(len(DEFAULT_RBY1_JOINTS)), gripper=(0.25, 0.75))
    left, right = wire.gripper_columns(wire.ARM_JOINT_DIM)
    assert hold[0, left] == pytest.approx(0.25)
    assert hold[0, right] == pytest.approx(0.75)


def test_a_scene_whose_gripper_reader_raises_falls_back_to_the_constant():
    """읽기 실패로 hold 를 못 만들면 안 된다 — hold 는 마지막 안전망이다."""
    scene = _Scene(np.zeros(len(DEFAULT_RBY1_JOINTS)))
    scene.gripper_norm = lambda: (_ for _ in ()).throw(RuntimeError("no sensor"))
    client = SafeRemoteClient(policy=_DeadServer(), scene=scene, cameras=())
    hold = np.asarray(client.infer({"state": np.zeros(wire.ACTION_WIDTH)})["actions"])
    for column in wire.gripper_columns(wire.ARM_JOINT_DIM):
        assert hold[0, column] == HOLD_GRIPPER_NORM


def test_an_action_shaped_state_is_passed_through():
    """이미 action 레이아웃이면 손대지 않는다 (14D/16D 를 직접 주는 호출부가 있다)."""
    state = np.arange(wire.ACTION_WIDTH, dtype=np.float64)
    hold = _hold_chunk(state)
    assert np.allclose(hold[0], state)


def test_an_arms_only_state_is_still_read_positionally():
    """씬이 팔 관절만 주면 그때는 위치로 읽는 것이 맞다 — 그 경우만."""
    n = wire.ARM_JOINT_DIM
    state = np.arange(2 * n, dtype=np.float64) + 100.0
    hold = _hold_chunk(state)
    assert np.allclose(hold[0, :n], state[:n])
    assert np.allclose(hold[0, n + 1:2 * n + 1], state[n:])


def test_a_state_of_an_unknown_length_is_refused_not_guessed():
    """추측하면 팔이 엉뚱한 관절로 지령된다. 그것이 이 버그의 원래 모양이었다."""
    client = SafeRemoteClient(policy=_DeadServer(), scene=_Scene(np.zeros(9)), cameras=())
    with pytest.raises(ValueError, match="추측하지 않습니다"):
        client._current_state()


def test_the_mapping_comes_from_the_same_table_the_refiner_uses():
    """매핑이 두 곳에 따로 있으면 한쪽이 조용히 한 칸 밀린다."""
    from benchmark.trajopt.types import ChunkLayout

    client = SafeRemoteClient(policy=_DeadServer(),
                             scene=_Scene(np.zeros(len(DEFAULT_RBY1_JOINTS))), cameras=())
    layout = client._chunk_layout()
    reference = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS, action_dim=wire.ACTION_WIDTH,
                                 arm_joint_dim=wire.ARM_JOINT_DIM)
    assert layout.action_to_q == reference.action_to_q
