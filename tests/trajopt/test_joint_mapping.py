"""chunk 열 ↔ optimizer 결정변수 ↔ 다시 chunk 열. **TO 안쪽의 관절 매핑** (T6f 확인할 것 4).

client 쪽은 이미 대조됐다 (`(chunk_seq, step_in_chunk)` 행과 다음 control `qpos` 599 쌍,
mae ≤ 0.011 rad). 검증되지 않은 것이 이 안쪽이고, 이 프로젝트는 같은 종류의 index 오류를 이미
두 번 밟았다 — `EE_BODY_L` 대 `ee_left`, 그리고 gripper 열 6 대 7. 둘 다 조용했다.

**왕복 동일성만으로는 부족하다.** `chunk_to_trajectory` 와 `trajectory_to_chunk` 는 같은 열
목록을 쓰므로 왼팔과 오른팔이 뒤바뀌어 있어도 왕복은 항등이다. 그래서 여기서는 세 가지를 따로
본다: **열 ↔ 관절 이름표**, **왕복 동일성**, 그리고 **FK 국소성**(왼팔 열을 흔들면 오른팔 구는
안 움직인다). 세 번째가 좌우 교환을 잡는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
from AG3S.trajopt import wire
from AG3S.trajopt.types import ChunkLayout

casadi = pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def layout():
    """서버가 쓰는 것과 같은 레이아웃 — `SafePolicy` 는 `ChunkLayout.rby1(model.joint_names)` 다."""
    return ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)


# --- 열 ↔ 관절 이름표 ----------------------------------------------------------------


def test_the_16d_column_map_is_pinned(layout):
    """A2 가 client 쪽에서 확인한 배치와 **글자 그대로** 같은 표를 서버 쪽에도 박는다:
    `left_arm_0..6 = 0..6 · left_gripper = 7 · right_arm_0..6 = 8..14 · right_gripper = 15`."""
    names = list(layout.joint_names)
    got = {column: names[q] for column, q in layout.action_to_q}
    expected = {i: f"left_arm_{i}" for i in range(7)}
    expected.update({8 + i: f"right_arm_{i}" for i in range(7)})
    assert got == expected
    assert wire.gripper_columns(7) == (7, 15)
    # 그리퍼 열은 결정 변수가 아니다 — 통과 열이다.
    assert set(wire.gripper_columns(7)) <= set(layout.passthrough)


def test_the_14d_layout_keeps_its_own_map():
    """은퇴한 배치지만 아직 받는다. 두 배치가 같은 상수를 쓰면 하나가 조용히 틀린다."""
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS, arm_joint_dim=6, n_valid=14)
    names = list(layout.joint_names)
    got = {column: names[q] for column, q in layout.action_to_q}
    expected = {i: f"left_arm_{i}" for i in range(6)}
    expected.update({7 + i: f"right_arm_{i}" for i in range(6)})
    assert got == expected
    assert wire.gripper_columns(6) == (6, 13)
    # 14D 에서 손목은 정책이 지령하지 않으므로 **고정 관절**이다 (FK 는 여전히 필요하다).
    fixed = {names[q] for q in layout.fixed_q}
    assert {"left_arm_6", "right_arm_6"} <= fixed


def test_the_q_index_comes_from_the_joint_name_not_from_the_column(layout):
    """`DEFAULT_RBY1_JOINTS` 는 torso 6 · right 7 · left 7 순서다. 열 0 이 q 0 이라고 믿으면
    왼팔 명령이 torso 로 간다."""
    names = list(layout.joint_names)
    assert names.index("left_arm_0") == 13 and names.index("right_arm_0") == 6
    assert dict(layout.action_to_q)[0] == 13
    assert dict(layout.action_to_q)[8] == 6


# --- 왕복 ----------------------------------------------------------------------------


def test_the_round_trip_is_the_identity_on_the_whole_chunk(layout):
    rng = np.random.default_rng(0)
    chunk = rng.normal(size=(9, layout.action_dim))
    trajectory = layout.chunk_to_trajectory(chunk)
    assert trajectory.shape == (layout.nq_opt, 9)
    out = layout.trajectory_to_chunk(trajectory, chunk)
    assert np.array_equal(out, chunk)


def test_the_round_trip_is_the_identity_on_the_trajectory_too(layout):
    """반대 방향도 본다 — 열 순서가 섞이면 한 방향만으로는 안 보인다."""
    rng = np.random.default_rng(1)
    trajectory = rng.normal(size=(layout.nq_opt, 9))
    template = np.zeros((9, layout.action_dim))
    back = layout.chunk_to_trajectory(layout.trajectory_to_chunk(trajectory, template))
    assert np.array_equal(back, trajectory)


def test_a_short_trajectory_writes_only_its_own_steps(layout):
    """실행 창만 다듬을 때 나머지 행은 정책의 답 그대로여야 한다 (T6f)."""
    rng = np.random.default_rng(2)
    chunk = rng.normal(size=(12, layout.action_dim))
    out = layout.trajectory_to_chunk(np.zeros((layout.nq_opt, 4)), chunk)
    assert np.array_equal(out[4:], chunk[4:])
    assert np.all(out[:4, layout.action_columns] == 0.0)


def test_full_q_puts_each_column_at_its_own_joint(layout):
    """결정 변수 → 모델 q. 여기서 한 칸 밀리면 FK 가 다른 팔을 움직인다."""
    names = list(layout.joint_names)
    q_now = np.arange(layout.nq_model, dtype=np.float64) * 0.01
    chunk = np.zeros((3, layout.action_dim))
    chunk[:, 2] = 0.5   # left_arm_2
    chunk[:, 10] = -0.5  # right_arm_2
    full = layout.full_q(layout.chunk_to_trajectory(chunk), q_now)

    assert full[names.index("left_arm_2")].tolist() == [0.5] * 3
    assert full[names.index("right_arm_2")].tolist() == [-0.5] * 3
    # 고정 관절은 `q_now` 를 든다 — 빼면 로봇이 똑바로 서 있는 것처럼 FK 가 계산된다.
    for q in layout.fixed_q:
        assert np.allclose(full[q], q_now[q])


# --- FK 국소성: 좌우가 안 바뀌었다 ---------------------------------------------------


@pytest.fixture(scope="module")
def robot():
    return load_rby1()


def _moved_links(robot, layout, column, delta=0.4):
    q_now = np.zeros(layout.nq_model)
    base, _ = robot.sphere_centers_numeric(q_now)
    chunk = np.zeros((1, layout.action_dim))
    chunk[0, column] = delta
    moved_q = layout.full_q(layout.chunk_to_trajectory(chunk), q_now)[:, 0]
    after, _ = robot.sphere_centers_numeric(moved_q)
    shift = np.linalg.norm(after - base, axis=1)
    return {link for link, d in zip(robot.sphere_link_names, shift) if d > 1e-6}


def test_a_left_arm_column_moves_only_the_left_arm(robot, layout):
    links = _moved_links(robot, layout, column=1)  # left_arm_1
    assert links, "열 1 을 흔들었는데 아무 구도 안 움직였다 — 매핑이 끊긴 것이다"
    assert all("left" in link for link in links), links
    assert not any("right" in link for link in links), links


def test_a_right_arm_column_moves_only_the_right_arm(robot, layout):
    links = _moved_links(robot, layout, column=9)  # right_arm_1
    assert links
    assert all("right" in link for link in links), links
    assert not any("left" in link for link in links), links


def test_a_gripper_column_moves_nothing(robot, layout):
    """그리퍼 열은 결정 변수가 아니고 관절도 아니다. 무언가 움직이면 그 열이 관절로 읽힌 것이다."""
    for column in wire.gripper_columns(7):
        assert _moved_links(robot, layout, column=column) == set()


def test_the_linearizer_orders_its_jacobian_columns_like_the_decision_vector(robot, layout):
    """`linearize` 는 `q[layout.q_indices]` 로 야코비안을 뽑는다. 그 순서가 결정 변수의 순서와
    다르면 QP 가 다른 관절을 밀면서 값은 맞게 떨어진다 — 가장 조용한 실패다."""
    from AG3S.trajopt.linearize import CollisionLinearizer

    linearizer = CollisionLinearizer(robot, layout, 1)
    q_now = np.zeros(layout.nq_model)
    trajectory = np.zeros((layout.nq_opt, 1))
    _, jac = linearizer.sphere_states(trajectory, q_now)

    names = list(layout.joint_names)
    axis = [column for column, q in layout.action_to_q if names[q] == "left_arm_1"][0]
    reduced_index = [c for c, _ in layout.action_to_q].index(axis)
    sensitivity = np.linalg.norm(jac[0, :, :, reduced_index], axis=1)
    left = [i for i, link in enumerate(robot.sphere_link_names) if "left" in link]
    right = [i for i, link in enumerate(robot.sphere_link_names) if "right" in link]
    assert sensitivity[left].max() > 1e-6
    assert sensitivity[right].max() < 1e-9


# --- 그리퍼 열: 서버가 읽는 자리 -----------------------------------------------------


def test_safe_policy_derives_the_gripper_columns_from_the_layout():
    from tests.trajopt.test_safe_policy import _safe_policy

    sp = _safe_policy()
    assert sp.gripper_columns == (7, 15)
    assert sp.gripper_columns == wire.gripper_columns(sp.layout.nq_opt // 2)


def test_the_latch_reads_the_gripper_column_and_not_the_wrist():
    """**T6f 에서 고친 별건.** 전에는 `column = 6 if left else 13` 으로 박혀 있었고, 16D 에서
    열 6 은 `left_arm_6`(손목) · 13 은 `right_arm_5` 다. 잠금은 값이 0.85 아래면 "닫혔다" 로
    읽으므로, 손목 각도가 그 아래인 프레임은 손이 열려 있는데도 파지로 보였다 — 증상이 궤적
    오차가 아니라 **권한이 엉뚱한 물체에 붙는 것**으로 나오므로 조용하다."""
    import types

    from tests.trajopt.test_safe_policy import _safe_policy

    sp = _safe_policy()
    left_col, right_col = sp.gripper_columns
    chunk = np.zeros((2, wire.ACTION_WIDTH))
    chunk[:, left_col] = 0.25   # 왼 그리퍼: 닫힘
    chunk[:, right_col] = 0.75  # 오른 그리퍼: 조금 덜 닫힘
    chunk[:, 6] = 1.9           # left_arm_6 손목 — 옛 코드가 읽던 열
    chunk[:, 13] = 1.9          # right_arm_5 — 옛 코드가 읽던 열

    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(attach=False, detach=False, destination=None)

    sp._latch.update = _spy  # noqa: SLF001
    sp._pending = {"chunk": chunk, "scene": {}}  # noqa: SLF001

    sp._run_latch(None, (), ["left"])  # noqa: SLF001
    assert seen["gripper"] == pytest.approx(0.25)
    sp._run_latch(None, (), ["right"])  # noqa: SLF001
    assert seen["gripper"] == pytest.approx(0.75)


def test_the_refined_chunk_keeps_the_policys_gripper_values():
    from tests.trajopt.test_safe_policy import _safe_policy

    sp = _safe_policy()
    rng = np.random.default_rng(3)
    original = rng.normal(size=(5, wire.ACTION_WIDTH))
    refined = rng.normal(size=(5, wire.ACTION_WIDTH))
    out = sp._preserve_grippers(original, refined)  # noqa: SLF001
    for column in sp.gripper_columns:
        assert np.array_equal(out[:, column], original[:, column])
    # 관절 열은 refined 그대로여야 한다 — 전부 되돌리면 TO 가 아무 일도 안 한 것이다.
    for column in sp.layout.action_columns:
        assert np.array_equal(out[:, column], refined[:, column])
