"""Joint bounds for the optimized subset, converted once into the optimizer's own units.

The URDF speaks in radians, rad/s and rad/s². The optimizer speaks in *steps*: its decision variable
is a sequence of absolute joint targets one control period apart, so a velocity limit is a bound on
consecutive differences and an acceleration limit is a bound on second differences. Doing that
conversion here, once, is what keeps a stray `dt` from appearing in one constraint and not another —
a mistake that produces a trajectory obeying limits nobody wrote down.

Everything is taken from the robot model rather than configured. A hand-typed limit table for a
20-joint robot is a table with one wrong number in it, and the joint it is wrong on is the one nobody
checked.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from AG3S.trajopt.config import LimitsConfig
from AG3S.trajopt.types import ChunkLayout, JointLimits


@dataclasses.dataclass(frozen=True)
class RelaxedJointLimits(JointLimits):
    """`JointLimits` + **스텝 사이 속도 행을 걸지 않는다**는 표시 (T27, `--no-limits`).

    `max_step` 은 **유한하게 남는다.** 그 값이 첫 스텝 anchor ``|Q[:, 0] - q_now| <= max_step`` 의
    근거이고 (`problem.build_problem` 의 box 블록), `sqp._limit_overshoot` 도 같은 값으로 초기 iterate
    의 적격성을 잰다 (T24). anchor 만 남기고 속도 행을 빼려면 두 곳이 같은 값을 봐야 하므로 값을
    `inf` 로 만드는 대신 이 표시를 둔다.

    `velocity_rows=False` 를 읽는 곳은 둘이다 — `build_problem` (속도 행을 만들지 않는다) 과
    `limit_report` (속도 초과를 한계 위반으로 세지 않는다: 걸지 않은 한계다). 둘 다 `getattr` 로
    읽으므로 plain `JointLimits` 는 예전 그대로다.
    """

    velocity_rows: bool = True


def velocity_rows(limits: JointLimits) -> bool:
    """이 한계가 **스텝 사이 속도 행**을 거는가. `max_step` 이 유한하고 표시가 끄지 않았을 때."""
    return bool(getattr(limits, "velocity_rows", True)) and bool(
        np.all(np.isfinite(np.asarray(limits.max_step, np.float64))))


def build_limits(
    robot_model,
    layout: ChunkLayout,
    *,
    dt: float,
    config: Optional[LimitsConfig] = None,
) -> JointLimits:
    """Position/velocity/acceleration bounds for `layout`'s optimized joints.

    Args:
        robot_model: anything exposing `joint_limits()` and, for velocity/acceleration,
            `velocity_limits()` / `acceleration_limits()` — `UrdfSphereChain` does.
        layout: which joints are free, and in what order the decision vector holds them.
        dt: control period in seconds.

    `enforce_velocity=False` / `enforce_acceleration=False` do not remove the bound; they widen it to
    something the optimizer can never reach. Keeping the constraint row present means the ablation
    changes one number rather than the problem's structure, which is what makes the comparison mean
    anything.
    """
    cfg = config or LimitsConfig()
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    index = layout.q_indices
    lower_all, upper_all = robot_model.joint_limits()
    lower = np.asarray(lower_all, np.float64)[index] + cfg.position_margin
    upper = np.asarray(upper_all, np.float64)[index] - cfg.position_margin
    if np.any(lower > upper):
        bad = [layout.joint_names[index[i]] if layout.joint_names else int(index[i])
               for i in np.flatnonzero(lower > upper)]
        raise ValueError(
            f"limits.position_margin={cfg.position_margin} rad closes the range of joint(s) {bad}"
        )

    velocity = _limit_vector(robot_model, "velocity_limits", cfg.default_velocity)[index]
    acceleration = _limit_vector(
        robot_model, "acceleration_limits", cfg.default_acceleration
    )[index]

    # `v_max * dt` is how far a joint may move between two consecutive chunk entries, and
    # `a_max * dt^2` how much that displacement may itself change. Both are plain bounds on
    # differences of the decision variable, which is the whole reason for converting here.
    max_step = velocity * cfg.velocity_scale * dt
    max_step_change = acceleration * cfg.acceleration_scale * dt * dt
    # **anchor 만 남기는 판** (T27): 속도 행은 빼되 `max_step` 은 유한하게 둔다 — anchor 와
    # 초기 iterate 적격성 검사가 그 값을 본다 (`RelaxedJointLimits`).
    anchor_only = (not cfg.enforce_velocity) and cfg.keep_anchor
    if not cfg.enforce_velocity and not anchor_only:
        max_step = np.full_like(max_step, np.inf)
    if not cfg.enforce_acceleration:
        max_step_change = np.full_like(max_step_change, np.inf)
    if not cfg.enforce_position:
        # position box 를 뺀다 (T27). box 행에는 trust region 과 anchor 만 남는다.
        lower = np.full_like(lower, -np.inf)
        upper = np.full_like(upper, np.inf)

    if anchor_only:
        return RelaxedJointLimits(
            lower=lower,
            upper=upper,
            max_step=max_step,
            max_step_change=max_step_change,
            dt=float(dt),
            velocity_rows=False,
        )
    return JointLimits(
        lower=lower,
        upper=upper,
        max_step=max_step,
        max_step_change=max_step_change,
        dt=float(dt),
    )


def _limit_vector(robot_model, method: str, default: Optional[float]) -> np.ndarray:
    getter = getattr(robot_model, method, None)
    if getter is None:
        if default is None:
            raise ValueError(
                f"the robot model has no {method}() and no default was configured. Set "
                f"limits.default_{method.split('_')[0]} explicitly rather than letting a bound be "
                "invented for a real robot."
            )
        return np.full(int(robot_model.nq), float(default), np.float64)
    return np.asarray(getter(default), np.float64)


def clamp_to_limits(trajectory: np.ndarray, limits: JointLimits) -> np.ndarray:
    """Project a trajectory onto the position box. Diagnostics and fallbacks only.

    Deliberately *not* used to enforce velocity or acceleration: clamping those after the fact
    changes the trajectory's shape in ways the optimizer never saw, so the result would satisfy the
    limits while no longer satisfying the collision constraints that were solved around it.
    """
    return np.clip(np.asarray(trajectory, np.float64), limits.lower[:, None], limits.upper[:, None])


def limit_report(trajectory: np.ndarray, limits: JointLimits) -> dict[str, float]:
    """How far a trajectory is outside each bound, in the bound's own units. Zero means compliant.

    Reported rather than asserted, because the interesting question during development is not
    *whether* a limit is violated but *by how much* — a micron of overshoot from the QP's tolerance
    and a joint driven past its stop are the same boolean and very different problems.
    """
    Q = np.asarray(trajectory, np.float64)
    if Q.shape[1] < 1:
        return {"position": 0.0, "velocity": 0.0, "acceleration": 0.0}
    below = float(np.max(limits.lower[:, None] - Q, initial=0.0))
    above = float(np.max(Q - limits.upper[:, None], initial=0.0))

    step = np.diff(Q, axis=1)
    # 속도 행을 걸지 않은 한계(`RelaxedJointLimits`, T27)는 속도 초과를 세지 않는다 — 걸지 않은
    # 한계를 위반으로 세면 `sqp` 가 초기 iterate 를 부적격으로 보고 reference 를 끌어당긴다.
    velocity = (
        float(np.max(np.abs(step) - limits.max_step[:, None], initial=0.0))
        if step.shape[1] and velocity_rows(limits) else 0.0
    )
    change = np.diff(Q, n=2, axis=1)
    acceleration = (
        float(np.max(np.abs(change) - limits.max_step_change[:, None], initial=0.0))
        if change.shape[1] and np.all(np.isfinite(limits.max_step_change)) else 0.0
    )
    return {
        "position": max(below, above, 0.0),
        "velocity": max(velocity, 0.0),
        "acceleration": max(acceleration, 0.0),
    }


__all__ = ["RelaxedJointLimits", "build_limits", "clamp_to_limits", "limit_report",
           "velocity_rows"]
