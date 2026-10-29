"""AG3S-PRE-CORE-TO E2 — multi-seed TrajOpt: seeds and the selection rule.

`serve_safe --to-multi-seed N` (config `sqp.multi_seed`, default 0 = off). When a chunk's standard solve (today's SQP,
E1-C included when it is on) leaves a violation above `safety.violation_tolerance`, `TrajectoryOptimizer` solves the
same problem again from N other first iterates with the same SQP (same settings, same iteration cap and budget per
seed) and keeps the best of the N + 1 results. Every other chunk is the standard solve, untouched.

Why (task §2): IFG-APPLE-TOA2 ep211 — three SQP iterations from the policy plan could not leave a pose where the
collision rows and the tracking cost balance (a local minimum, ~50 chunks of HOLD); without the palm-path term the same
chunk was solved by a 54 mm translation. A different first iterate is the cheapest way out of such a basin.

Seeds, in order (`seed_specs`; deterministic: the noise of slot i comes from `chunk_rng(chunk index, i)` and the random
direction of fill draw j from `chunk_rng(chunk index, RANDOM_DIRECTION_STREAM + j)`; the chunk index is the request's
`t_step`, else a count of this optimizer's solves):

| # | kind | first iterate |
|---|---|---|
| 0 | `policy` | the policy plan (the standard solve starts from its warm-start blend or from `stay`) |
| 1 | `stay` | ``Q[:, k] = q_now`` (T43 Q `_stay_start`'s plan) |
| 2-7 | `detour` | the palm path pushed 5 / 10 cm along `away` (the field gradient at the deepest violating sphere of the standard result), `e1` (horizontal, perpendicular to `away`) and `e2` (perpendicular to both, pointing up) |
| 8-9 | `noise` | the policy plan + N(0, σ²) per joint and row, σ 0.02 / 0.05 rad, row 0 kept |
| 10- | fill | repeating: two detours (5 / 10 cm) along the next direction of `FILL_DIRECTIONS` (then random directions with ``d · away >= 0``), two noise seeds (σ 0.02 / 0.05, new draws) |

A detour is a smooth bump over the rows (`bump`: 0 at the first and the last row, 1 at the deepest violating step),
mapped to the joints row by row through the damped pseudo-inverse of the palm Jacobian at the policy plan
(`detour_seed`). The palm is the hold link (`ee_left` / `ee_right`, or the grasp anchor's link) whose Jacobian shares
the most joints with the deepest violating sphere's. Only the joints of that chain move; the noise seeds perturb the same
joints. Every seed is clipped to the position limits. Without a direction (no violating row, zero gradient) the detour
slots become noise seeds (`detail.no_direction`).

Selection (`selection_key`, lexicographic, the smallest wins, ties go to the earlier candidate — the standard solve is
candidate 0, so the result can never be worse than it on the first key):

1. max violation (m), 0 at or below the tolerance, else in 0.1 mm steps;
2. goal error (mm, 0.1 mm steps): Euclidean max over the goal rows of the palm vs the frozen grasp goal — only when the
   solve carried a grasp anchor with goal rows on this horizon (E1-C / E1-B / IFG-TOA), else 0 for everyone;
3. joint path length (rad): ``sum_k ||Q_k - Q_{k-1}||`` with ``Q_{-1} = q_now``;
4. sum of squared accelerations: ``sum_k ||Q_{k+1} - 2 Q_k + Q_{k-1}||^2`` over the same sequence.

**Applied rule (user, 2026-10-26): `select_moving_first`.** The candidates that move (all but `stay`) and resolve the
violation are ranked by (goal error, deviation from the policy reference); the key above decides only when none of
them resolves. The key's third entry (path length) made `stay` win almost every offline chunk (see the function).

A seed result further outside the joint limits (position / velocity / acceleration / the step-0 anchor,
`_limit_overshoot`) than max(the standard result's overshoot, `LIMIT_SLACK`) is not eligible (the solve can return a
far-off first iterate, or a QP step that has not yet pulled it back inside the soft velocity / acceleration rows).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

#: Detour magnitudes (m) — task §6 E2 "5 · 10 cm".
DETOUR_MAGNITUDES: tuple[float, ...] = (0.05, 0.10)
#: Joint-noise standard deviations (rad) — task §6 E2 "σ 0.02 · 0.05 rad".
NOISE_SIGMAS: tuple[float, ...] = (0.02, 0.05)
#: The RNG of a chunk: `np.random.default_rng([RNG_SALT, chunk index])`.
RNG_SALT = 0xE2
#: Stream offset of the random fill directions in `chunk_rng` (seed slots use their own index, < this).
RANDOM_DIRECTION_STREAM = 100_000
#: Damping of the palm Jacobian pseudo-inverse, ``J' (J J' + λ² I)^-1`` (m/rad scale; the arm's J is ~0.3-0.8 m/rad).
PINV_DAMPING = 0.03
#: Directions of the fill detours after `-e1`, as combinations of (away, e1, e2); then random ones.
FILL_DIRECTIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("-e1", (0.0, -1.0, 0.0)),
    ("away+e2", (1.0, 0.0, 1.0)),
    ("away+e1", (1.0, 1.0, 0.0)),
    ("away-e1", (1.0, -1.0, 0.0)),
    ("-e2", (0.0, 0.0, -1.0)),
)
#: A seed result is eligible only if its joint-limit overshoot (`limit_report` + the step-0 anchor, in each bound's
#: units: rad, rad/row, rad/row²) is at most max(the standard result's, LIMIT_SLACK). The QP keeps the velocity and
#: acceleration rows soft, so a returned solve carries ~1e-5 overshoot; a seed whose SQP could not pull a far-off first
#: iterate back inside the limits (or returned it unchanged) carries 1e-2 and more.
LIMIT_SLACK = 1e-3
#: Hold links tried as "the palm" when no grasp anchor names one (`safe_policy.DEFAULT_HOLD_LINKS`).
PALM_LINKS: tuple[str, ...] = ("ee_left", "ee_right")


@dataclasses.dataclass(frozen=True)
class SeedSpec:
    """One seed slot: `kind` (`policy` · `stay` · `detour` · `noise`) and its parameters."""

    kind: str
    direction: Optional[str] = None                     # detour: "away" · "e1" · "e2" · a fill name · "random<i>"
    weights: Optional[tuple[float, float, float]] = None  # detour: (away, e1, e2) combination; None = random
    magnitude: float = 0.0                              # detour (m)
    sigma: float = 0.0                                  # noise (rad)
    draw: int = 0                                       # noise / random direction: which draw of the chunk's RNG


def seed_specs(n: int) -> list[SeedSpec]:
    """The first `n` seed slots (see the module table). Independent of the geometry."""
    base: list[SeedSpec] = [SeedSpec("policy"), SeedSpec("stay")]
    for name, w in (("away", (1.0, 0.0, 0.0)), ("e1", (0.0, 1.0, 0.0)), ("e2", (0.0, 0.0, 1.0))):
        for m in DETOUR_MAGNITUDES:
            base.append(SeedSpec("detour", name, w, m))
    for s in NOISE_SIGMAS:
        base.append(SeedSpec("noise", sigma=s, draw=0))
    out = base[:max(int(n), 0)]
    i, draw, n_random = 0, 1, 0
    while len(out) < n:
        if i < len(FILL_DIRECTIONS):
            name, w = FILL_DIRECTIONS[i]
        else:
            name, w = f"random{n_random}", None
            n_random += 1
        for m in DETOUR_MAGNITUDES:
            out.append(SeedSpec("detour", name, w, m, draw=n_random - 1 if w is None else 0))
        for s in NOISE_SIGMAS:
            out.append(SeedSpec("noise", sigma=s, draw=draw))
        draw += 1
        i += 1
    return out[:n]


def chunk_rng(index: int, *stream: int) -> np.random.Generator:
    """The chunk's RNG for one stream (a seed slot, or a random direction's draw): `default_rng([RNG_SALT, index,
    *stream])` — the same chunk index and slot give the same numbers, whatever else is solved."""
    return np.random.default_rng([RNG_SALT, abs(int(index)), *(abs(int(s)) for s in stream)])


def perpendicular_frame(away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(e1, e2)`: `e1` horizontal and perpendicular to `away`, `e2 = away × e1` flipped to point up (z >= 0).

    With `away` (nearly) vertical, `e1` is the world x axis projected off `away`."""
    a = np.asarray(away, np.float64) / max(float(np.linalg.norm(away)), 1e-12)
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), a)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0]) - a * a[0]
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(a, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)
    if e2[2] < 0.0:
        e2 = -e2
    return e1, e2


def bump(horizon: int, peak: int) -> np.ndarray:
    """`(H,)` smooth bump: 0 at row 0 and row H-1, 1 at `peak` (clamped into [1, H-2]); sin² on each side."""
    H = int(horizon)
    b = np.zeros(H)
    if H < 3:
        return b
    p = int(min(max(int(peak), 1), H - 2))
    k = np.arange(H, dtype=np.float64)
    left = np.sin(0.5 * np.pi * k / p) ** 2
    right = np.sin(0.5 * np.pi * (H - 1 - k) / (H - 1 - p)) ** 2
    b = np.where(k <= p, left, right)
    b[0] = b[-1] = 0.0
    return b


def damped_pinv(J: np.ndarray, damping: float = PINV_DAMPING) -> np.ndarray:
    """``J' (J J' + λ² I)^-1`` for a `(3, n)` Jacobian."""
    J = np.asarray(J, np.float64)
    return J.T @ np.linalg.inv(J @ J.T + float(damping) ** 2 * np.eye(J.shape[0]))


def detour_seed(reference: np.ndarray, palm_jac: np.ndarray, direction: np.ndarray, magnitude: float,
                profile: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """The policy plan with the palm pushed `magnitude * profile[k]` along `direction` on row k, through the damped
    pseudo-inverse of the palm Jacobian `palm_jac[k]` (`(H, 3, nq)`, at the policy plan) restricted to `joints`."""
    Q = np.asarray(reference, np.float64).copy()
    d = np.asarray(direction, np.float64) / max(float(np.linalg.norm(direction)), 1e-12)
    joints = np.asarray(joints, int)
    for k in range(Q.shape[1]):
        if profile[k] == 0.0:
            continue
        J = np.asarray(palm_jac[k], np.float64)[:, joints]
        Q[joints, k] += damped_pinv(J) @ (float(magnitude) * float(profile[k]) * d)
    return Q


def noise_seed(reference: np.ndarray, sigma: float, rng: np.random.Generator, joints: np.ndarray) -> np.ndarray:
    """The policy plan + N(0, σ²) on `joints`, every row but row 0."""
    Q = np.asarray(reference, np.float64).copy()
    joints = np.asarray(joints, int)
    eps = rng.standard_normal((joints.size, Q.shape[1]))
    eps[:, 0] = 0.0
    Q[joints] += float(sigma) * eps
    return Q


def random_direction(away: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random unit vector with ``d · away >= 0``."""
    d = rng.standard_normal(3)
    d /= max(float(np.linalg.norm(d)), 1e-12)
    return -d if float(d @ away) < 0.0 else d


def path_length(trajectory: np.ndarray, q0: np.ndarray) -> float:
    """``sum_k ||Q_k - Q_{k-1}||`` (rad), ``Q_{-1} = q0``."""
    Q = np.concatenate([np.asarray(q0, np.float64).reshape(-1, 1), np.asarray(trajectory, np.float64)], axis=1)
    return float(np.linalg.norm(np.diff(Q, axis=1), axis=0).sum())


def accel_sq(trajectory: np.ndarray, q0: np.ndarray) -> float:
    """``sum_k ||Q_{k+1} - 2 Q_k + Q_{k-1}||^2`` (rad²), ``Q_{-1} = q0``."""
    Q = np.concatenate([np.asarray(q0, np.float64).reshape(-1, 1), np.asarray(trajectory, np.float64)], axis=1)
    if Q.shape[1] < 3:
        return 0.0
    return float(np.sum(np.diff(Q, n=2, axis=1) ** 2))


def selection_key(violation_m: float, tolerance_m: float, goal_err_mm: Optional[float], length: float,
                  accel: float) -> tuple:
    """The lexicographic key of a candidate (smaller is better; see the module docstring)."""
    v = 0 if float(violation_m) <= float(tolerance_m) else int(round(float(violation_m) * 1e4))
    g = 0 if goal_err_mm is None else int(round(float(goal_err_mm) * 10.0))
    return (v, g, round(float(length), 6), round(float(accel), 8))


def select(keys: list[Optional[tuple]]) -> int:
    """Index of the smallest key; `None` = not eligible; ties go to the earlier index (0 = the standard solve)."""
    best, best_key = 0, keys[0]
    for i, k in enumerate(keys):
        if k is not None and (best_key is None or k < best_key):
            best, best_key = i, k
    return best


def select_moving_first(kinds: list[str], keys: list[Optional[tuple]], violation_mm: list[float], tolerance_mm: float,
                        goal_err_mm: list[Optional[float]], reference_deviation: list[float]) -> int:
    """The chosen candidate under the user's rule (2026-10-26, "stay is the last resort").

    Among the eligible candidates that move (every kind but `stay`) and resolve the violation (<= tolerance), the
    smallest (goal error, deviation from the policy reference) wins, ties to the earlier index. Only when no moving
    candidate resolves does `select` decide over all of them, `stay` included. Measured offline (S-OCC, 68 chunks the
    standard solve left violated): the path-length rule chose `stay` — a HOLD-like stop in closed loop — on 67-68 of
    them although a moving candidate resolved every one.
    """
    moving = [i for i, k in enumerate(keys)
              if k is not None and kinds[i] != "stay" and float(violation_mm[i]) <= float(tolerance_mm)]
    if not moving:
        return select(keys)
    return min(moving, key=lambda i: (0 if goal_err_mm[i] is None else int(round(float(goal_err_mm[i]) * 10.0)),
                                      round(float(reference_deviation[i]), 6), i))


__all__ = ["DETOUR_MAGNITUDES", "NOISE_SIGMAS", "RNG_SALT", "PALM_LINKS", "SeedSpec", "seed_specs", "chunk_rng",
           "perpendicular_frame", "bump", "damped_pinv", "detour_seed", "noise_seed", "random_direction",
           "path_length", "accel_sq", "selection_key", "select", "select_moving_first"]
