"""쥔 물체와 **상대 자세가 고정인** 로봇 구 — self-collision 허용목록을 기구학에서 뽑는다 (N1).

쥔 물체를 로봇 자신과도 충돌 검사하려면 (cuRoboV2 §6.2) 어떤 쌍을 빼야 하는지부터 정해야 한다.
이름으로 쓴 허용목록(`self_collision_mask` 의 손끝 둘)으로는 **부족하다** — RB-Y1 실측에서 쥔
사과가 `link_left_arm_5` 의 구 안에 -56.1 mm 들어가 있고 그 값이 파지 구간 9 프레임 내내
변하지 않는다. 손목 전체가 물체와 함께 움직이기 때문이다.

그런 쌍에 제약을 걸면 **최적화가 영원히 못 푸는 행**이 된다. 그래서 허용목록을 손으로 쓰지 않고
"최적화가 이 거리를 바꿀 수 있는가" 로 유도한다.
"""

import numpy as np
import pytest

from AG3S.ag3s.constraints.attached import base_frame_points, rigid_spheres, self_collision_mask
from AG3S.ag3s.robot_models.urdf_sphere_chain import load_rby1
from AG3S.ag3s.types import AttachedCollisionGeometry
from AG3S.trajopt.types import ChunkLayout

PARENT = "ee_finger_l1"
FINGERS = ("ee_finger_l1", "ee_finger_l2")


@pytest.fixture(scope="module")
def setup():
    robot = load_rby1()
    layout = ChunkLayout.rby1(robot.joint_names)
    q = np.zeros(robot.nq)
    rng = np.random.default_rng(0)
    attached = AttachedCollisionGeometry(
        parent_link=PARENT, T_parent_object=np.eye(4), primitives=[],
        points=rng.normal(scale=0.03, size=(40, 3)),
        allowed_contact_links=frozenset(FINGERS),
    )
    return robot, layout, q, attached


def test_rigid_spheres_are_genuinely_invariant(setup):
    """뽑아낸 구들의 거리가 최적화 자유도로 정말 안 변하는지 직접 확인한다."""
    robot, layout, q, attached = setup
    rigid = rigid_spheres(attached, robot, q, layout.q_indices)
    assert rigid.any(), "손끝에 붙은 물체라면 최소한 손끝 구는 고정이어야 한다"

    free = np.asarray(layout.q_indices, np.int64)
    rng = np.random.default_rng(7)   # 판정에 쓴 것과 **다른** 씨앗

    def gaps(qq):
        centres, radii = robot.sphere_centers_numeric(qq)
        pts = base_frame_points(attached, robot, qq)
        return np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2).min(axis=0) - radii

    base = gaps(q)
    for _ in range(5):
        moved = q.copy()
        moved[free] += rng.normal(scale=0.5, size=free.size)
        assert np.allclose(gaps(moved)[rigid], base[rigid], atol=1e-9)


def test_the_name_based_allowlist_is_not_enough(setup):
    """이 파일이 있는 이유 — 이름 목록이 손목을 놓친다."""
    robot, layout, q, attached = setup
    rigid = rigid_spheres(attached, robot, q, layout.q_indices)
    by_name = ~self_collision_mask(attached, robot.sphere_link_names).astype(bool)
    missed = rigid & ~by_name
    assert missed.any(), "이름 목록이 전부 잡는다면 이 헬퍼가 필요 없다"
    # 놓친 것은 손끝이 아닌 링크다 — 손끝은 이름으로 이미 빠져 있다.
    names = np.asarray(robot.sphere_link_names)
    assert not set(names[missed]) & set(FINGERS)


def test_fixed_joints_must_not_be_perturbed(setup):
    """고정 관절까지 흔들면 움직일 수 없는 쌍을 움직인다고 잘못 판정한다.

    처음에 그렇게 재서 `link_left_arm_5` 를 "움직인다" 로 분류했고, 실제 롤아웃에서 그 거리가
    9 프레임 내내 상수 -56.1 mm 였던 것이 오류를 잡아냈다.
    """
    robot, layout, q, attached = setup
    only_free = rigid_spheres(attached, robot, q, layout.q_indices)
    everything = rigid_spheres(attached, robot, q, list(range(robot.nq)))
    # 전부 흔들면 고정으로 판정되는 구가 더 적거나 같다 — 즉 허용목록이 더 좁아진다.
    assert int(everything.sum()) <= int(only_free.sum())


def test_no_points_means_no_rigid_spheres(setup):
    robot, layout, q, _ = setup
    empty = AttachedCollisionGeometry(
        parent_link=PARENT, T_parent_object=np.eye(4), primitives=[], points=None)
    assert not rigid_spheres(empty, robot, q, layout.q_indices).any()


def test_no_free_joints_means_nothing_can_be_constrained(setup):
    """자유도가 없으면 모든 거리가 고정이다 — 전부 허용목록에 들어가야 한다."""
    robot, layout, q, attached = setup
    assert rigid_spheres(attached, robot, q, []).sum() == 0  # 자유도 0 이면 판정 자체를 안 한다
