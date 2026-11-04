"""SL10 §2.2 -- the controller's action limit as rows of the TrajOpt QP (``serve_safe --to-action-box``, Panda only).

**Why.** The Panda adapter (``AG3S/adapters/panda/action.py``, reverse conversion) turns a refined joint chunk into the
OSC action chunk the robot executes. With ``Dy_k`` = (refined chained EE step) - (reference chained EE step) it adds
``Dx_k / scale`` to the policy's action, ``Dx_k = (Dy_k - alpha Dy_{k-1}) / beta`` (the controller's first-order lag
``y_t = alpha y_{t-1} + beta x_t``, ``robots/panda/description/osc_response_fit.json``), and clips to ``[-1, 1]``.
``1 / beta`` is 14 (position) and 32 (rotation): a TrajOpt correction of a few millimetres becomes a full-range action
and the clip then removes part of it. TrajOpt did not know the limit (``outputs/2026-11-04_SL10_...``: the executed
action's saturated components are 0.05-0.08 before the grasp and 0.12-0.52 while grasping and after).

**What.** Rows on the same variables TrajOpt already optimises, so that the action conversion C would return is inside
``[-1 + margin, 1 - margin]`` per row and component (position and rotation separately, world frame)::

    raw_k = a_k + g . (dy_k - alpha . dy_{k-1}),      g = 1 / (beta * scale)      (per component)
    dy_k  = R_wb [ (p_k - p_{k-1}) - (pr_k - pr_{k-1}) ;  rotvec(R_k R_{k-1}^T) - rotvec(Rr_k Rr_{k-1}^T) ]
    lo_k <= raw_k <= hi_k,    hi = max(1 - margin, a_k),   lo = min(-(1 - margin), a_k)

``p_k, R_k`` = the EE (``panda_grip_site``) pose of joint row ``Q_k`` (row ``-1`` = ``q_now``), ``pr, Rr`` the same for
the reference. ``a_k`` is the policy's own action, clipped to ``[-1, 1]`` (what the forward conversion predicted from).
**The bound is widened to ``a_k`` when the policy itself is outside ``1 - margin``** so that the reference (``Dy = 0``)
is always inside: the box never asks TrajOpt to change a chunk it has no reason to touch, and it never *adds* to a
component that is already beyond the margin. (The task text writes ``|raw| <= 1 - margin`` for every row; the two
differ only on policy components in ``(1 - margin, 1]``.)

**Linearisation** (the SQP's own, at the iterate ``X``): ``p_k(Q) ~ p_k(X) + Jp_k (Q_k - X_k)``;
``rotvec(R_k R_{k-1}^T) ~ w0 + Jl^-1(w0) (Jw_k dQ_k - S0 Jw_{k-1} dQ_{k-1})`` with ``S0 = R_k(X) R_{k-1}(X)^T``,
``w0 = rotvec(S0)``, ``Jw`` the spatial angular-velocity Jacobian (``grasp_anchor.LinkPoseFK``), ``Jl^-1`` the inverse
left Jacobian of SO(3). Row ``k`` therefore touches ``Q_k, Q_{k-1}, Q_{k-2}`` (``Q_{-1} = q_now`` is fixed, so row 0 has
no earlier term -- ``Dy_{-1} = 0`` exactly as ``action.joint_chunk_to_delta_chunk`` starts it).

**Soft, priced far above a collision** (``ACTION_BOX_SLACK_SCALE x w_slack`` per action unit): the rows must not make
the QP infeasible when the first iterate (a warm start) is outside the box or the linearisation is off, and a collision
that cannot be avoided inside the box has to stay a *collision violation* (the existing HOLD / violated rules decide),
not turn into an action the controller clips. The price is ~10x the largest dual a row can have (a unit of action moves
a rotation row's spheres < 0.1 m), so the box wins over a collision slack whenever one is possible.

The exact (non-linear) value of the same quantity at the returned trajectory is what ``record`` reports.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

#: ``--to-action-box-margin`` default (task SL10 §2.2).
ACTION_BOX_MARGIN_DEFAULT = 0.05
#: price of the action slack, in units of ``cost.w_slack`` (per action unit). See the module docstring.
ACTION_BOX_SLACK_SCALE = 10.0
#: a row is "active" when the returned trajectory puts its action this close (action units) to the bound it moved toward.
ACTIVE_TOL = 1e-3
#: components per row: 3 position, 3 rotation.
N_COMP = 6


@dataclasses.dataclass(frozen=True)
class ActionBox:
    """Everything one chunk's rows need, handed from the Panda adapter (``action.action_box_from_conversion``) through
    ``SafePolicy`` (``context["action_box"]``) and the refiner to ``TrajectoryOptimizer.solve(action_box=...)``."""

    link: str                 # EE link of the controller (``panda_grip_site``)
    actions: np.ndarray       # (H, 6) the policy's OSC action a[:6], clipped to [-input_clip, input_clip]
    R_world_base: np.ndarray  # (3, 3) rotation of the base frame in the world frame
    q_now: np.ndarray         # (nq,) the joint state the conversion started from (checked against the solver's)
    alpha: np.ndarray         # (6,) lag coefficient per component (position x3, rotation x3)
    beta: np.ndarray          # (6,) input gain per component
    scale: np.ndarray         # (6,) OSC output scale per component (m / rad per unit action)
    margin: float = ACTION_BOX_MARGIN_DEFAULT
    n_rows: int = 5           # rows constrained (the executed ones); the rest of the chunk is not executed
    input_clip: float = 1.0
    mode: str = "C"

    @property
    def gain(self) -> np.ndarray:
        """``g = 1 / (beta * scale)`` per component: action units per unit of ``(dy_k - alpha dy_{k-1})``."""
        return 1.0 / (np.asarray(self.beta, np.float64) * np.asarray(self.scale, np.float64))

    def rows(self, horizon: int) -> int:
        return int(min(self.n_rows, horizon, self.actions.shape[0]))

    def allowed(self, rows: int) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` ``(rows, 6)`` of the raw action ``a_k + correction`` (see the module docstring)."""
        a = np.asarray(self.actions, np.float64)[:rows, :N_COMP]
        bound = float(self.input_clip) - float(self.margin)
        return np.minimum(-bound, a), np.maximum(bound, a)


def _hat(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def left_jacobian_inv(w: np.ndarray) -> np.ndarray:
    """``Jl^-1(w)``: ``rotvec(exp(d) exp(w)) ~ w + Jl^-1(w) d`` for small ``d``."""
    w = np.asarray(w, np.float64)
    th = float(np.linalg.norm(w))
    E = _hat(w)
    if th < 1e-6:
        return np.eye(3) - 0.5 * E + E @ E / 12.0
    return np.eye(3) - 0.5 * E + (1.0 / th ** 2 - (1.0 + math.cos(th)) / (2.0 * th * math.sin(th))) * (E @ E)


def _so3_log(R: np.ndarray) -> np.ndarray:
    from AG3S.ag3s.grasp.ik import so3_log

    return so3_log(R)


def chain_steps(p: np.ndarray, R: np.ndarray, p_now: np.ndarray, R_now: np.ndarray, rows: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(dp (rows, 3), dr (rows, 3))`` base-frame position and rotation-vector steps of the chained path
    ``now -> row 0 -> row 1 ...`` (the FK chain of ``action.joint_chunk_to_delta_chunk``)."""
    dp = np.zeros((rows, 3))
    dr = np.zeros((rows, 3))
    prev_p, prev_R = np.asarray(p_now, np.float64), np.asarray(R_now, np.float64)
    for k in range(rows):
        dp[k] = p[k] - prev_p
        dr[k] = _so3_log(np.asarray(R[k], np.float64) @ prev_R.T)
        prev_p, prev_R = np.asarray(p[k], np.float64), np.asarray(R[k], np.float64)
    return dp, dr


def exact_correction(box: ActionBox, p: np.ndarray, R: np.ndarray, p_now: np.ndarray, R_now: np.ndarray,
                     ref_steps: tuple[np.ndarray, np.ndarray], rows: int) -> np.ndarray:
    """``(rows, 6)`` exact ``g . (dy_k - alpha . dy_{k-1})`` of the trajectory with FK poses ``p, R`` (not linearised)."""
    dp, dr = chain_steps(p, R, p_now, R_now, rows)
    R_wb = np.asarray(box.R_world_base, np.float64)
    dy = np.concatenate([(dp - ref_steps[0][:rows]) @ R_wb.T, (dr - ref_steps[1][:rows]) @ R_wb.T], axis=1)
    prev = np.vstack([np.zeros((1, N_COMP)), dy[:-1]]) if rows else dy
    return box.gain[None, :] * (dy - box.alpha[None, :] * prev)


def linear_pieces(box: ActionBox, pose: tuple, now_pose: tuple, ref_steps: tuple, rows: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(c (rows, 6), A (rows, 6, nq), B (rows, 6, nq))``: the chained step deviation of row ``k`` linearised at the
    pose ``pose = (p, Jp, R, Jw)`` of the iterate (``LinkPoseFK.pose``) is ``dy_k ~ c_k + A_k dQ_k + B_k dQ_{k-1}``
    (``dQ = Q - X``; ``B_0 = 0`` because ``Q_{-1} = q_now`` is fixed). ``now_pose = (p_now, R_now)``, ``ref_steps`` the
    reference's chained steps (``chain_steps``)."""
    p, Jp, R, Jw = pose
    p_now, R_now = now_pose
    nq = np.asarray(Jp).shape[2]
    R_wb = np.asarray(box.R_world_base, np.float64)
    c = np.zeros((rows, N_COMP))
    A = np.zeros((rows, N_COMP, nq))
    B = np.zeros((rows, N_COMP, nq))
    prev_p, prev_R = np.asarray(p_now, np.float64), np.asarray(R_now, np.float64)
    for k in range(rows):
        S0 = np.asarray(R[k], np.float64) @ prev_R.T
        w0 = _so3_log(S0)
        Jli = left_jacobian_inv(w0)
        c[k, :3] = R_wb @ ((p[k] - prev_p) - ref_steps[0][k])
        c[k, 3:] = R_wb @ (w0 - ref_steps[1][k])
        A[k, :3] = R_wb @ np.asarray(Jp[k], np.float64)
        A[k, 3:] = R_wb @ Jli @ np.asarray(Jw[k], np.float64)
        if k > 0:
            B[k, :3] = -R_wb @ np.asarray(Jp[k - 1], np.float64)
            B[k, 3:] = -R_wb @ Jli @ S0 @ np.asarray(Jw[k - 1], np.float64)
        prev_p, prev_R = np.asarray(p[k], np.float64), np.asarray(R[k], np.float64)
    return c, A, B


def linear_rows(box: ActionBox, iterate: np.ndarray, pose: tuple, now_pose: tuple, ref_steps: tuple, rows: int
                ) -> tuple[sp.csr_matrix, np.ndarray]:
    """``(G (6 rows, nq H), e (6 rows,))`` with ``correction(Q) ~ G vec(Q) + e`` (``vec`` step-major, as the QP).
    Row ``r = 6 k + i``; the sparsity pattern is fixed (explicit zeros kept)."""
    nq, H = np.asarray(iterate).shape
    g, alpha = box.gain, np.asarray(box.alpha, np.float64)
    c, A, B = linear_pieces(box, pose, now_pose, ref_steps, rows)
    X = np.asarray(iterate, np.float64)
    G = np.zeros((rows * N_COMP, nq * H))
    value = np.zeros(rows * N_COMP)
    for k in range(rows):
        for i in range(N_COMP):
            r = k * N_COMP + i
            G[r, k * nq:(k + 1) * nq] += g[i] * A[k, i]
            value[r] = g[i] * c[k, i]
            if k > 0:
                G[r, (k - 1) * nq:k * nq] += g[i] * (B[k, i] - alpha[i] * A[k - 1, i])
                value[r] -= g[i] * alpha[i] * c[k - 1, i]
            if k > 1:
                G[r, (k - 2) * nq:(k - 1) * nq] += -g[i] * alpha[i] * B[k - 1, i]
    e = value - G @ X.T.reshape(-1)
    return _fixed_pattern(G, rows, nq), e


def _fixed_pattern(G: np.ndarray, rows: int, nq: int) -> sp.csr_matrix:
    """``G`` as CSR with the band pattern (row ``(k, i)`` holds the blocks of ``Q_k, Q_{k-1}, Q_{k-2}``) whether or not
    an entry is zero: the QP backend updates ``A.data`` in place and needs the same pattern at every iterate."""
    ri, ci = [], []
    for k in range(rows):
        cols = np.concatenate([np.arange(j * nq, (j + 1) * nq) for j in range(max(k - 2, 0), k + 1)])
        for i in range(N_COMP):
            ri.append(np.full(cols.size, k * N_COMP + i))
            ci.append(cols)
    ri, ci = np.concatenate(ri), np.concatenate(ci)
    return sp.csr_matrix((G[ri, ci], (ri, ci)), shape=G.shape)


def add_action_box(problem, G: sp.csr_matrix, e: np.ndarray, lo: np.ndarray, hi: np.ndarray, weight: float):
    """A copy of ``problem`` with the rows ``lo <= G Q + e <= hi`` softened by one slack ``s >= 0`` per row
    (``G Q + e - s <= hi``, ``G Q + e + s >= lo``), the slacks new columns after every existing variable
    (``QpProblem.n_action_slack``), priced linearly at ``weight`` per action unit. Nothing else in ``problem`` changes."""
    m = int(G.shape[0])
    if m == 0:
        return problem
    n_old = problem.P.shape[0]
    n_q = G.shape[1]
    Gq = sp.hstack([G, sp.csr_matrix((m, n_old - n_q))], format="csr")
    eye = sp.identity(m, format="csr")
    A_old = sp.hstack([problem.A, sp.csr_matrix((problem.A.shape[0], m))], format="csr")
    A = sp.vstack([A_old,
                   sp.hstack([Gq, -eye], format="csr"),                 # upper:  G Q - s <= hi - e
                   sp.hstack([Gq, eye], format="csr"),                  # lower:  G Q + s >= lo - e
                   sp.hstack([sp.csr_matrix((m, n_old)), eye], format="csr")],   # s >= 0
                  format="csc")
    start = problem.A.shape[0]
    row_blocks = dict(problem.row_blocks)
    row_blocks["action_box"] = (start, start + m)
    row_blocks["action_box_lower"] = (start + m, start + 2 * m)
    row_blocks["action_box_slack"] = (start + 2 * m, start + 3 * m)
    P = sp.block_diag([problem.P, sp.csc_matrix((m, m))], format="csc")
    q = np.concatenate([np.asarray(problem.q, np.float64), np.full(m, float(weight))])
    l = np.concatenate([problem.l, np.full(m, -np.inf), np.asarray(lo, np.float64) - e, np.zeros(m)])
    u = np.concatenate([problem.u, np.asarray(hi, np.float64) - e, np.full(m, np.inf), np.full(m, np.inf)])
    return dataclasses.replace(problem, P=P, q=q, A=A, l=l, u=u, row_blocks=row_blocks,
                               n_action_slack=int(getattr(problem, "n_action_slack", 0)) + m)


def record(box: ActionBox, correction: np.ndarray, rows: int, *, margin_slack: Optional[float] = None) -> dict[str, Any]:
    """``to.action_box`` for a returned trajectory whose exact correction is ``correction`` ``(rows, 6)``.

    ``on`` -- the rows were in the QP; ``n_rows`` -- constrained (step, component) pairs (``6 x rows``; two-sided);
    ``active_rows`` -- pairs the trajectory moved off the reference action and left within ``ACTIVE_TOL`` of the bound
    it moved toward; ``max_pred_abs_action`` -- the largest ``|a + correction|`` over the pairs (what conversion C
    would send before the clip). ``max_excess`` -- the largest amount beyond the allowed interval (0 = inside)."""
    lo, hi = box.allowed(rows)
    a = np.asarray(box.actions, np.float64)[:rows, :N_COMP]
    raw = a + correction
    moved = np.abs(correction) > 1e-9
    active = moved & ((raw >= hi - ACTIVE_TOL) | (raw <= lo + ACTIVE_TOL))
    excess = np.maximum(np.maximum(raw - hi, lo - raw), 0.0)
    return {"on": True, "n_rows": int(rows * N_COMP), "active_rows": int(active.sum()),
            "max_pred_abs_action": float(np.abs(raw).max()) if raw.size else 0.0,
            "max_excess": float(excess.max()) if excess.size else 0.0, "margin": float(box.margin),
            "rows": int(rows), "mode": str(box.mode)}


__all__ = ["ACTION_BOX_MARGIN_DEFAULT", "ACTION_BOX_SLACK_SCALE", "ActionBox", "add_action_box", "chain_steps",
           "exact_correction", "left_jacobian_inv", "linear_rows", "record"]
