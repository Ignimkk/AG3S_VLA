"""Small, fast, fully-known scenes for the trajectory-optimization tests.

Everything runs against the real RB-Y1 collision model rather than a mock, because the things most
likely to break — the chunk-to-joint mapping, the Jacobian's sparsity, the reshape conventions
between CasADi and numpy — are exactly the things a mock would define away. What is scaled down is
the *horizon*, not the fidelity: a 10-step chunk exercises every code path a 50-step one does and
builds in a fraction of the time.

The obstacle is placed by construction rather than by guessing: `near_miss_scene` offsets it
perpendicular to the swept path so the reference penetrates by a requested number of millimetres.
A test that asserts "the optimizer fixed the violation" is only meaningful if the violation was there
to begin with, and was of a size the optimizer could plausibly fix.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.limits import build_limits
from benchmark.trajopt.linearize import CollisionLinearizer, SceneSnapshot
from benchmark.trajopt.types import ChunkLayout

#: A configuration with the right arm folded in front of the torso — away from its joint limits, so
#: a test that moves it is measuring the optimizer rather than the box constraints.
Q_HOME = np.zeros(20)
Q_HOME[7] = -0.6  # right_arm_1
Q_HOME[9] = -1.0  # right_arm_3

#: How far `right_arm_0` sweeps across the reference. Large enough that the arm covers real distance,
#: small enough to stay inside the velocity limit at 15 Hz over a short horizon.
SWEEP = 1.2


def small_config(horizon: int = 10, **overrides) -> TrajOptConfig:
    """A config sized for tests: short horizon, plans all of it, converges rather than times out."""
    base = {
        "horizon": {"horizon": horizon, "execution_length": min(4, horizon), "plan_horizon": None},
        "sqp": {"max_iterations": 6, "time_budget_ms": 1e6},
        "reduction": {"rows_per_step": 24},
    }
    for section, values in overrides.items():
        base[section] = {**base.get(section, {}), **values}
    return TrajOptConfig.from_dict(base)


@dataclasses.dataclass
class Scenario:
    """A complete, solvable problem plus the numbers a test needs to judge the answer."""

    robot: object
    layout: ChunkLayout
    config: TrajOptConfig
    limits: object
    linearizer: CollisionLinearizer
    reference: np.ndarray  # (nq_opt, H)
    q_now: np.ndarray  # (nq_model,)
    scene: SceneSnapshot
    chunk: np.ndarray  # (H, action_dim) with recognisable gripper and padding values
    reference_violation: float  # metres of penetration in the reference

    @property
    def horizon(self) -> int:
        return int(self.reference.shape[1])


def sweep_reference(layout: ChunkLayout, horizon: int, q_now: np.ndarray = Q_HOME,
                    joint: str = "right_arm_0") -> np.ndarray:
    """A straight-line joint-space sweep, which is what a smoothed policy chunk looks like.

    **어느 관절을 쓸지 이름으로 찾는다.** 전에는 축소 벡터의 인덱스 6 을 `right_arm_0` 이라고
    박아 두었는데, 그것은 `arm_joint_dim=6`(14D)에서만 맞다 — 16D 에서 인덱스 6 은
    `left_arm_6`(손목)이고, 손목을 쓸면 팔이 훨씬 덜 움직여 fixture 가 만들려던 관통이 생기지
    않는다. 2026-09-24 의 16D 전환에서 실제로 그렇게 깨졌고, `ChunkLayout.rby1` 자신이
    지키는 원칙(q 를 이름으로 찾는다)을 이 fixture 가 어기고 있었던 것이다.
    """
    q_names = list(layout.joint_names)
    reduced = [q_names[q] for _, q in layout.action_to_q]
    if joint not in reduced:
        raise ValueError(
            f"{joint!r} 은 이 레이아웃이 지령할 수 있는 관절이 아닙니다. 가능한 것: {reduced}")
    axis = reduced.index(joint)
    start = np.asarray(q_now, np.float64)[layout.q_indices].copy()
    goal = start.copy()
    goal[axis] = SWEEP
    return start[:, None] + (goal - start)[:, None] * np.linspace(0.0, 1.0, horizon)[None, :]


def empty_scene(linearizer: CollisionLinearizer, n_slots: int = 8, n_planes: int = 2) -> SceneSnapshot:
    """Correct shapes, nothing active. The optimizer should hand the reference straight back."""
    spheres = linearizer.n_spheres
    return SceneSnapshot(
        candidate_pos=np.zeros((n_slots, 3)),
        candidate_radius=np.zeros(n_slots),
        candidate_active=np.zeros(n_slots, bool),
        d_safe=np.full((spheres, n_slots), 0.05),
        plane_normal=np.tile(np.array([0.0, 0.0, 1.0]), (n_planes, 1)),
        plane_offset=np.zeros(n_planes),
        plane_active=np.zeros(n_planes, bool),
        robot_radii=linearizer.robot_radii,
        candidate_ids=np.arange(n_slots),
    )


def near_miss_scene(
    linearizer: CollisionLinearizer,
    reference: np.ndarray,
    q_now: np.ndarray,
    *,
    penetration: float = 0.03,
    radius: float = 0.10,
    d_safe: float = 0.05,
    n_slots: int = 8,
    floor: bool = True,
) -> SceneSnapshot:
    """An obstacle placed so the reference penetrates it by exactly `penetration` metres.

    Offset **perpendicular** to the sweep so the arm has somewhere to go. An obstacle on the path
    itself would demand a detour the size of its own radius, which tests the trust region rather than
    the collision handling.
    """
    centres, _ = linearizer.sphere_states(reference, q_now)
    middle = centres.shape[0] // 2
    moved = np.linalg.norm(centres[middle] - centres[0], axis=1)
    sphere = int(np.argmax(moved))
    point = centres[middle, sphere]

    direction = centres[min(middle + 1, centres.shape[0] - 1), sphere] - centres[max(middle - 1, 0), sphere]
    direction /= max(np.linalg.norm(direction), 1e-12)
    perpendicular = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    perpendicular /= max(np.linalg.norm(perpendicular), 1e-12)

    required = float(linearizer.robot_radii[sphere] + radius + d_safe)
    obstacle = point + perpendicular * (required - penetration)

    spheres = linearizer.n_spheres
    active = np.zeros(n_slots, bool)
    active[0] = True
    return SceneSnapshot(
        candidate_pos=np.tile(obstacle, (n_slots, 1)),
        candidate_radius=np.full(n_slots, radius),
        candidate_active=active,
        d_safe=np.full((spheres, n_slots), d_safe),
        plane_normal=np.tile(np.array([0.0, 0.0, 1.0]), (2, 1)),
        # A floor well below the robot: active, satisfied, and therefore proof that an *inactive*
        # plane and a *cleared* plane are handled differently.
        plane_offset=np.array([-1.0, 0.0]),
        plane_active=np.array([floor, False]),
        robot_radii=linearizer.robot_radii,
        candidate_ids=np.arange(n_slots),
    )


def make_scenario(horizon: int = 10, penetration: float = 0.03, **config_overrides) -> Scenario:
    """The standard test problem: a sweeping arm and one obstacle it currently clips."""
    robot = load_rby1()
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    config = small_config(horizon, **config_overrides)
    limits = build_limits(robot, layout, dt=config.horizon.dt, config=config.limits)
    linearizer = CollisionLinearizer(robot, layout, horizon)
    reference = sweep_reference(layout, horizon)
    scene = near_miss_scene(linearizer, reference, Q_HOME, penetration=penetration)

    # Recognisable values in every column the optimizer must not touch.
    chunk = np.zeros((horizon, layout.action_dim))
    chunk[:, layout.action_columns] = reference.T
    chunk[:, 6] = 0.7  # left gripper
    chunk[:, 13] = -0.3  # right gripper
    chunk[:, layout.n_valid :] = np.arange(horizon)[:, None] * 0.01  # padding

    return Scenario(
        robot=robot,
        layout=layout,
        config=config,
        limits=limits,
        linearizer=linearizer,
        reference=reference,
        q_now=Q_HOME.copy(),
        scene=scene,
        chunk=chunk,
        reference_violation=-linearizer.full_violation(reference, Q_HOME, scene),
    )


__all__ = [
    "Q_HOME",
    "SWEEP",
    "Scenario",
    "empty_scene",
    "make_scenario",
    "near_miss_scene",
    "small_config",
    "sweep_reference",
]
