"""T63 step 2 — numpy reference of the TO objective terms that B1-B3 add to cuRobo (pure numpy).

Every value function here is the term of the **baseline** TO objective (`AG3S/trajopt/problem.py`
of snapshot `f0d8046`; the working-tree file is byte-identical, checked 2026-10-21), written per
term so that each can be switched on alone, plus its analytic gradient. The torch terms in
`costs.py` are tested against these, and these are tested against `problem.objective` /
`deflection_rate_cost` / `servo_accel_overshoot` (`tests/trajopt/test_t63_curobo_v2_terms.py`).

Shapes follow `problem.py`: a trajectory is `Q (nq, H)` — column `k` is chunk row `k` (time
`(k+1)/15 s` after the capture), joints in the chunk-column order. Units are the TO's own: rad,
rad/row, rad/row^2 — **no time scaling**.

| term | value (`problem.py`) | where |
|---|---|---|
| tracking | `w_track · Σ_k decay^k · ‖Q_k − R_k‖²` | `objective`, `build_problem` (cost) |
| continuity | `w_continuity · Σ_{k<L} ‖Q_k − P_k‖²`, `L = min(P.shape[1], H)` | same |
| smoothness | `w_smooth · Σ_k ‖Q_{k+2} − 2Q_{k+1} + Q_k‖²` | same (not in B1-B4; kept for completeness) |
| deflection rate (T43 TA B) | `w · (Σ_k ‖D_{k+1} − D_k‖² + ‖D_0 − D_prev‖²)`, `D = Q − R` | `deflection_rate_cost` |
| servo accel (T43 TA C) | `w_slack · servo_accel_weight · Σ max(0, |A_q vec Q + c| − b)` + `w_slack · 10 · Σ max(0, |Q_0 − u_{-1}| − v_max)` | `objective` (the L1 price of the C rows and the boundary velocity row) |
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# =================================================================================================
# values
# =================================================================================================

def tracking_value(Q: np.ndarray, R: np.ndarray, w_track: float, track_decay: float = 1.0) -> float:
    Q, R = np.asarray(Q, np.float64), np.asarray(R, np.float64)
    decay = float(track_decay) ** np.arange(Q.shape[1], dtype=np.float64)
    return float(w_track * np.sum(decay * np.sum((Q - R) ** 2, axis=0)))


def continuity_value(Q: np.ndarray, P: Optional[np.ndarray], w_continuity: float) -> float:
    if P is None or w_continuity <= 0.0:
        return 0.0
    Q, P = np.asarray(Q, np.float64), np.asarray(P, np.float64)
    L = min(P.shape[1], Q.shape[1])
    return float(w_continuity * np.sum((Q[:, :L] - P[:, :L]) ** 2))


def smooth_value(Q: np.ndarray, w_smooth: float) -> float:
    Q = np.asarray(Q, np.float64)
    if w_smooth <= 0.0 or Q.shape[1] <= 2:
        return 0.0
    return float(w_smooth * np.sum(np.diff(Q, n=2, axis=1) ** 2))


def deflection_rate_value(Q: np.ndarray, R: np.ndarray, w: float,
                          deflection_history: Optional[np.ndarray] = None) -> float:
    if w <= 0.0:
        return 0.0
    D = np.asarray(Q, np.float64) - np.asarray(R, np.float64)
    value = float(np.sum(np.diff(D, axis=1) ** 2)) if D.shape[1] > 1 else 0.0
    if deflection_history is not None:
        value += float(np.sum((D[:, 0] - np.asarray(deflection_history, np.float64)) ** 2))
    return float(w) * value


def servo_accel_rows(R: np.ndarray, accel_bound: np.ndarray, mode: str,
                     max_step_change: Optional[np.ndarray] = None,
                     command_history: Optional[np.ndarray] = None):
    """The rows of `problem.servo_accel_rows` in dense, joint-separable form.

    Row `i` is the second difference `x_i − 2 x_{i+1} + x_{i+2}` of the padded sequence
    `[u_{-2}, u_{-1}, Q_0, Q_1, …]` (no padding without `command_history`). Returns
    `A (n_rows, H)` — the stencil weights on the columns of `Q` (the same for every joint),
    `const (n_rows, nq)` — the history part, `bound (n_rows, nq)`; the row values are
    `A @ Q.T + const`. Same rows, same order and same bounds as the sparse operator in
    `problem.py` (which interleaves joints step-major).
    """
    R = np.asarray(R, np.float64)
    nq, H = R.shape
    hist = None if command_history is None else np.asarray(command_history, np.float64)
    n_hist = 0 if hist is None else int(hist.shape[1])
    n_rows = H + n_hist - 2
    A = np.zeros((max(n_rows, 0), H))
    const = np.zeros((max(n_rows, 0), nq))
    for i in range(max(n_rows, 0)):
        for off, c in zip(range(3), (1.0, -2.0, 1.0)):
            col = i + off - n_hist  # index into Q; negative = history column
            if col >= 0:
                A[i, col] += c
            else:
                const[i] += c * hist[:, n_hist + col]
    a_c = np.broadcast_to(np.asarray(accel_bound, np.float64).reshape(1, -1), (max(n_rows, 0), nq))
    if mode == "relaxed":
        own = np.abs(A @ R.T + const)
        bound = np.maximum(a_c, own)
    elif mode == "strict":
        bound = np.array(a_c, copy=True)
    else:
        raise ValueError(f"servo_accel mode must be 'strict' or 'relaxed', got {mode!r}")
    if max_step_change is not None and np.all(np.isfinite(max_step_change)):
        bound = np.minimum(bound, np.asarray(max_step_change, np.float64).reshape(1, -1))
    return A, const, bound


def servo_accel_value(Q: np.ndarray, R: np.ndarray, accel_bound: np.ndarray, mode: str, *,
                      w_slack: float, servo_accel_weight: float,
                      max_step_change: Optional[np.ndarray] = None,
                      command_history: Optional[np.ndarray] = None,
                      max_step: Optional[np.ndarray] = None) -> float:
    """`objective`'s servo-accel block: L1 overshoot of the C rows + the boundary velocity row."""
    Q = np.asarray(Q, np.float64)
    A, const, bound = servo_accel_rows(R, accel_bound, mode, max_step_change, command_history)
    over = np.maximum(0.0, np.abs(A @ Q.T + const) - bound)
    value = float(w_slack * servo_accel_weight * np.sum(over))
    if command_history is not None and max_step is not None and np.all(np.isfinite(max_step)):
        u_last = np.asarray(command_history, np.float64)[:, -1]
        bv = np.maximum(0.0, np.abs(Q[:, 0] - u_last) - np.asarray(max_step, np.float64))
        value += float(w_slack * 10.0 * np.sum(bv))
    return value


# =================================================================================================
# analytic gradients (d value / d Q), same shape as Q
# =================================================================================================

def tracking_grad(Q, R, w_track, track_decay=1.0) -> np.ndarray:
    Q, R = np.asarray(Q, np.float64), np.asarray(R, np.float64)
    decay = float(track_decay) ** np.arange(Q.shape[1], dtype=np.float64)
    return 2.0 * w_track * decay[None, :] * (Q - R)


def continuity_grad(Q, P, w_continuity) -> np.ndarray:
    Q = np.asarray(Q, np.float64)
    g = np.zeros_like(Q)
    if P is None or w_continuity <= 0.0:
        return g
    P = np.asarray(P, np.float64)
    L = min(P.shape[1], Q.shape[1])
    g[:, :L] = 2.0 * w_continuity * (Q[:, :L] - P[:, :L])
    return g


def smooth_grad(Q, w_smooth) -> np.ndarray:
    Q = np.asarray(Q, np.float64)
    g = np.zeros_like(Q)
    if w_smooth <= 0.0 or Q.shape[1] <= 2:
        return g
    a = np.diff(Q, n=2, axis=1)
    for c, off in ((1.0, 0), (-2.0, 1), (1.0, 2)):
        g[:, off:off + a.shape[1]] += 2.0 * w_smooth * c * a
    return g


def deflection_rate_grad(Q, R, w, deflection_history=None) -> np.ndarray:
    D = np.asarray(Q, np.float64) - np.asarray(R, np.float64)
    g = np.zeros_like(D)
    if w <= 0.0:
        return g
    if D.shape[1] > 1:
        dD = np.diff(D, axis=1)
        g[:, 1:] += 2.0 * w * dD
        g[:, :-1] -= 2.0 * w * dD
    if deflection_history is not None:
        g[:, 0] += 2.0 * w * (D[:, 0] - np.asarray(deflection_history, np.float64))
    return g


def servo_accel_grad(Q, R, accel_bound, mode, *, w_slack, servo_accel_weight,
                     max_step_change=None, command_history=None, max_step=None) -> np.ndarray:
    """Subgradient: `sign(row)` where the row is over its bound (0 at and below the bound)."""
    Q = np.asarray(Q, np.float64)
    A, const, bound = servo_accel_rows(R, accel_bound, mode, max_step_change, command_history)
    rows = A @ Q.T + const
    active = (np.abs(rows) - bound) > 0.0
    s = np.where(active, np.sign(rows), 0.0) * (w_slack * servo_accel_weight)
    g = (A.T @ s).T
    if command_history is not None and max_step is not None and np.all(np.isfinite(max_step)):
        u_last = np.asarray(command_history, np.float64)[:, -1]
        e = Q[:, 0] - u_last
        act = (np.abs(e) - np.asarray(max_step, np.float64)) > 0.0
        g[:, 0] += w_slack * 10.0 * np.where(act, np.sign(e), 0.0)
    return g


__all__ = [
    "continuity_grad", "continuity_value", "deflection_rate_grad", "deflection_rate_value",
    "servo_accel_grad", "servo_accel_rows", "servo_accel_value", "smooth_grad", "smooth_value",
    "tracking_grad", "tracking_value",
]
