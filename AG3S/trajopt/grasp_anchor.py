"""IFG-TOA — grasp-goal anchor: avoid by changing the pose, not by moving the grasp point.

IFG ep954 (2026-10-20): approaching a static apple next to a banana, TrajOpt kept the 5 mm neighbour clearance by
translating the hand 6-19 mm (rotation <= 1.7 deg); the policy re-planned from the pushed hand and its intended grasp
point followed it (16 mm forward, 23 mm sideways at the close) -- the grasp missed. The joint-space tracking cost has
no notion of where the hand must end up, so the cheapest way out of a collision is a translation.

This module adds, on chunks the caller (`SafePolicy`) marks as pre-grasp near a frozen grasp goal, two Cartesian
terms on the hold link (palm) origin ``p_k(Q)``:

    w_tcp  * sum_k            || p_k(Q) - p_k(R) ||^2      (stay on the policy's hand path: translation is expensive,
                                                              rotation / null-space motion of every joint is not)
    w_goal * sum_{c<=k<c+G}   || p_k(Q) - g ||^2           (the G rows from the policy's close row c: the palm at the
                                                              goal g frozen when the policy first planned the close, while
                                                              the fingers close; c = 0 -- a chunk that starts closed -- has
                                                              no goal rows: the rows after the close lift the object)

Both are linearised at the SQP iterate (palm Jacobian, `LinkPointFK`) into the QP's ``P`` / ``q`` and evaluated exactly
in the merit (`anchor_cost`), so step acceptance sees what the QP optimises. Every joint stays free within its
limits; nothing here is a constraint, so collision rows still win when no pose satisfies both.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp


@dataclasses.dataclass(frozen=True)
class GraspAnchor:
    """One chunk's anchor, as `SafePolicy` hands it to the refiner (`context["grasp_anchor"]`).

    `close_row` is the first row of the policy chunk whose gripper command for the hand closes (`None` = the chunk does
    not close); the optimizer builds its row masks from it for its own horizon. `goal` (world, m) is the frozen palm
    goal (`None` = no goal term, tracking only).
    """

    link: str
    goal: Optional[np.ndarray]
    close_row: Optional[int]
    w_tcp: float
    w_goal: float
    #: rows of the goal term from `close_row` on (IFG smoke ep56: goal rows over every row after the close also held
    #: the lift rows of a closing chunk at the goal, 117.7 mm off).
    goal_rows: int = 3

    def masks(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """`(track_rows, goal_rows)` bool `(horizon,)`."""
        track = np.full(horizon, self.w_tcp > 0.0)
        goal = np.zeros(horizon, bool)
        if (self.goal is not None and self.w_goal > 0.0 and self.close_row is not None
                and 0 < int(self.close_row) < horizon):
            goal[int(self.close_row):int(self.close_row) + max(int(self.goal_rows), 0)] = True
        return track, goal


class LinkPointFK:
    """Origin of `link` and its Jacobian w.r.t. the optimized joints, along a chunk (CasADi, mapped over the horizon)."""

    def __init__(self, robot_model, layout, horizon: int, link: str):
        import casadi as ca

        self.layout = layout
        self.horizon = int(horizon)
        self.nq_opt = int(layout.nq_opt)
        q = ca.SX.sym("q", int(robot_model.nq), 1)
        T = robot_model.link_pose_symbolic(q, link)
        pos = T[:3, 3]
        free = [int(i) for i in layout.q_indices]
        jac = ca.jacobian(pos, q[free])                          # 3 x nq_opt
        self._map = ca.Function(f"pt_{link}", [q], [pos, jac]).map(self.horizon)

    def __call__(self, trajectory: np.ndarray, q_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(p[H, 3], J[H, 3, nq_opt])` at `trajectory` (`(nq_opt, H)`), the other joints at `q_now`."""
        full = self.layout.full_q(np.asarray(trajectory, np.float64), q_now)
        pos, jac = self._map(full)
        p = np.asarray(pos, np.float64).T.reshape(self.horizon, 3)
        J = np.asarray(jac, np.float64).reshape(3, self.horizon, self.nq_opt).transpose(1, 0, 2)
        return p, J


def anchor_targets(anchor: GraspAnchor, p_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(weights[H, n_terms], targets[H, n_terms, 3])` — the tracking and goal targets per row (weight 0 = off)."""
    H = p_ref.shape[0]
    track, goal = anchor.masks(H)
    w = np.stack([np.where(track, anchor.w_tcp, 0.0), np.where(goal, anchor.w_goal, 0.0)], axis=1)
    g = np.zeros((H, 3)) if anchor.goal is None else np.tile(np.asarray(anchor.goal, np.float64).reshape(1, 3), (H, 1))
    return w, np.stack([p_ref, g], axis=1)


def anchor_cost(p: np.ndarray, weights: np.ndarray, targets: np.ndarray) -> float:
    """Exact value of the anchor terms for palm positions `p[H, 3]`."""
    d = p[:, None, :] - targets
    return float(np.sum(weights * np.sum(d * d, axis=2)))


def anchor_qp_terms(iterate: np.ndarray, p_it: np.ndarray, J_it: np.ndarray, weights: np.ndarray,
                    targets: np.ndarray) -> tuple[sp.csc_matrix, np.ndarray]:
    """`(P_add[nq*H, nq*H], q_add[nq*H])` of the anchor linearised at `iterate`, in `build_problem`'s layout
    (step-major ``z[k*nq + j]``, cost ``1/2 z'Pz + q'z``).

    ``p_k(Q) ~ p_k(X) + J_k (Q_k - X_k)``; per term ``w || J_k Q_k + c ||^2`` with ``c = p_k(X) - J_k X_k - target``
    gives ``P += 2 w J'J`` and ``q += 2 w J'c``.
    """
    nq, H = iterate.shape
    blocks, q_add = [], np.zeros(nq * H)
    for k in range(H):
        w_k = weights[k]
        if not np.any(w_k > 0.0):
            blocks.append(sp.csc_matrix((nq, nq)))
            continue
        J = J_it[k]
        JtJ = J.T @ J
        P_k = 2.0 * float(np.sum(w_k)) * JtJ
        lin = np.zeros(nq)
        for wt, tgt in zip(w_k, targets[k]):
            if wt > 0.0:
                c = p_it[k] - J @ iterate[:, k] - tgt
                lin += 2.0 * wt * (J.T @ c)
        blocks.append(sp.csc_matrix(P_k))
        q_add[k * nq:(k + 1) * nq] = lin
    return sp.block_diag(blocks, format="csc"), q_add


def add_to_problem(problem, P_add: sp.csc_matrix, q_add: np.ndarray):
    """A copy of `problem` with the anchor added to its joint block (slack columns untouched)."""
    n = problem.P.shape[0]
    pad = n - P_add.shape[0]
    P = (problem.P + sp.block_diag([P_add, sp.csc_matrix((pad, pad))], format="csc")).tocsc()
    q = np.asarray(problem.q, np.float64).copy()
    q[:q_add.shape[0]] += q_add
    return dataclasses.replace(problem, P=P, q=q)


def anchor_record(anchor: GraspAnchor, p_ref: np.ndarray, p_out: np.ndarray) -> dict[str, Any]:
    """What `TrajOptResult.metrics["grasp_anchor"]` carries: weights, rows, and the palm errors of reference vs result."""
    track, goal = anchor.masks(p_ref.shape[0])
    rec: dict[str, Any] = {"link": anchor.link, "w_tcp": anchor.w_tcp, "w_goal": anchor.w_goal,
                           "close_row": anchor.close_row, "n_track_rows": int(track.sum()), "n_goal_rows": int(goal.sum()),
                           "goal_m": None if anchor.goal is None else [round(float(v), 5) for v in anchor.goal],
                           "palm_dev_from_reference_max_mm": round(float(np.max(np.linalg.norm(p_out - p_ref, axis=1)) * 1000), 2)}
    if goal.any():
        g = np.asarray(anchor.goal, np.float64).reshape(1, 3)
        rec["goal_err_reference_mm"] = round(float(np.max(np.linalg.norm(p_ref[goal] - g, axis=1)) * 1000), 2)
        rec["goal_err_result_mm"] = round(float(np.max(np.linalg.norm(p_out[goal] - g, axis=1)) * 1000), 2)
    return rec


__all__ = ["GraspAnchor", "LinkPointFK", "anchor_targets", "anchor_cost", "anchor_qp_terms", "add_to_problem",
           "anchor_record"]
