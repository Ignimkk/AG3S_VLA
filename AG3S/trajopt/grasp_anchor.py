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

**AG3S-PRE-CORE-TO E1** (default off) adds two variants on the same gate and the same frozen goal:

* **E1-C** (`w_tcp = 0`, `serve_safe --grasp-anchor W --grasp-anchor-path-weight 0`): the goal term alone -- no palm
  path-tracking term, so a translation away from the policy's path is no longer priced (IFG-APPLE-TOA2 ep211 / 1668:
  the path term blocked the translation the obstacle needed).
* **AG3S-GEO-GRASP** (`goal_R` set, `serve_safe --geo-grasp`): the goal is a 6-D hold-link pose from the geometric
  grasp candidates (`AG3S.ag3s.grasp`), not the policy's frozen close position. On the goal rows the position term
  ``w_goal ||p_k - g||^2`` plus a rotation term ``w_rot ||Log(R_g^T R_k)||^2``; linearised with
  ``d Log(R_g^T R) = Jr^-1(e) R^T J_w dq`` (`rotation_qp_terms`), evaluated exactly in the merit (`rotation_cost`) and
  recorded with the same definition (`anchor_record`). No path term (`w_tcp = 0`).
* **E1-B** (`goal_box = r > 0`, `serve_safe --grasp-goal-box MM`): no palm cost at all; the goal rows carry a soft
  per-axis box ``g - r <= p_k(X) + J_k (Q_k - X_k) <= g + r`` in the QP (`add_goal_box`), one slack per row and axis
  shared by its two sides, priced at ``w_slack / 10`` per m (`GOAL_BOX_SLACK_SCALE`) -- a collision metre costs ten
  goal-box metres, so collision rows win. The merit carries the same exact excess (`goal_box_excess`) at that price.

**AG3S-PRE-CORE-TO-GGW** (`serve_safe --grasp-goal-window widen`, `goal_window = "widen"`, default `"close"` = the rule
above): on the same anchored chunks the goal also gets rows when the close row c gives none on the optimised horizon H
-- c is None or c >= H: the last G rows of H; c = 0 (the chunk starts closed): the first G rows (the palm at the goal
while the fingers close). 0 < c < H is unchanged. Every consumer reads `goal_mask` (the SQP terms, the E1-B box, E2's
goal probe, E3-b's GPU seed terms via `anchor_targets`), so the widened rows apply on whichever horizon is solved.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

#: E1-B: the goal-box slack is priced at `cost.w_slack * GOAL_BOX_SLACK_SCALE` per m (collision first).
GOAL_BOX_SLACK_SCALE = 0.1


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
    #: E1-B — half-width (m) of the per-axis goal box on the goal rows; `0.0` (default) = no box (TOA / E1-C).
    goal_box: float = 0.0
    #: AG3S-PRE-CORE-TO-GGW — `"close"` (default: rows from `close_row` only when 0 < c < H) or `"widen"`.
    goal_window: str = "close"
    #: AG3S-GEO-GRASP — goal rotation of the hold link (3x3, base frame) and its weight (per rad^2); None = position only.
    goal_R: Optional[np.ndarray] = None
    w_rot: float = 0.0
    #: AG3S-GEO-GRASP — explicit goal rows (step indices) instead of `goal_rows` rows from `close_row` (the approach
    #: phase puts the pre-grasp pose on the last planned row only). None = from `close_row`.
    explicit_rows: Optional[tuple] = None
    #: AG3S-GEO-GRASP — what the goal is ("grasp" / "pre-grasp") and the acceptance tolerances (record only).
    goal_kind: str = "grasp"
    tol_pos: float = 0.010
    tol_rot: float = math.radians(15.0)
    #: AG3S-CORE-EVAL-G1 (option (a)) — the variant of the anchor this geometric one replaced on this chunk (the E1-C
    #: grasp-goal term, "E1-C"); None = nothing replaced (record key only when set).
    replaced_variant: Optional[str] = None

    @property
    def variant(self) -> str:
        """`"GEO"` (6-D geometric goal), `"E1-B"` (goal box), `"E1-C"` (goal term without the path term) or `"TOA"`."""
        if self.goal_R is not None:
            return "GEO"
        if self.goal_box > 0.0:
            return "E1-B"
        if self.w_tcp <= 0.0 and self.w_goal > 0.0:
            return "E1-C"
        return "TOA"

    def goal_mask(self, horizon: int) -> np.ndarray:
        """The rows the frozen goal applies to, whatever the weights: `goal_rows` rows from `close_row` (0 < c < H)."""
        goal = np.zeros(horizon, bool)
        if self.goal is not None and self.explicit_rows is not None:
            for k in self.explicit_rows:
                if 0 < int(k) < horizon:
                    goal[int(k)] = True
            return goal
        if self.goal is not None and self.goal_window == "widen":   # AG3S-PRE-CORE-TO-GGW
            n = max(int(self.goal_rows), 0)
            c = self.close_row
            if c is None or int(c) >= horizon:
                goal[max(horizon - n, 0):horizon] = bool(n)
            elif int(c) <= 0:
                goal[:n] = True
            else:
                goal[int(c):int(c) + n] = True
            return goal
        if self.goal is not None and self.close_row is not None and 0 < int(self.close_row) < horizon:
            goal[int(self.close_row):int(self.close_row) + max(int(self.goal_rows), 0)] = True
        return goal

    def goal_window_rule(self, horizon: int) -> str:
        """AG3S-PRE-CORE-TO-GGW — which rule of `goal_mask` gave this horizon's goal rows (for the records)."""
        c = self.close_row
        if self.goal is None:
            return "no goal"
        if c is not None and 0 < int(c) < horizon:
            return "close row inside the horizon: rows from the close row"
        if self.goal_window != "widen":
            return ("no close planned" if c is None else "starts closed (close row 0)" if int(c) <= 0
                    else "close row beyond this horizon")
        return ("no close planned: last rows" if c is None else "starts closed (close row 0): first rows"
                if int(c) <= 0 else "close row beyond this horizon: last rows")

    def masks(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """`(track_rows, goal_rows)` bool `(horizon,)`. The goal rows are those of the goal term (`w_goal > 0`) or of
        the goal box (`goal_box > 0`)."""
        track = np.full(horizon, self.w_tcp > 0.0)
        goal = self.goal_mask(horizon) if (self.w_goal > 0.0 or self.goal_box > 0.0) else np.zeros(horizon, bool)
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


class LinkPoseFK(LinkPointFK):
    """`LinkPointFK` plus the link rotation and its spatial angular-velocity Jacobian (AG3S-GEO-GRASP).
    Calling it returns `(p, J_p)` like `LinkPointFK`; `pose` returns `(p, J_p, R, J_w)`."""

    def __init__(self, robot_model, layout, horizon: int, link: str):
        import casadi as ca

        super().__init__(robot_model, layout, horizon, link)
        q = ca.SX.sym("q", int(robot_model.nq), 1)
        T = robot_model.link_pose_symbolic(q, link)
        R = T[:3, :3]
        free = [int(i) for i in layout.q_indices]
        cols = []
        for k in free:
            dR = ca.reshape(ca.jacobian(ca.reshape(R, 9, 1), q[k]), 3, 3)
            S = ca.mtimes(dR, R.T)
            cols.append(ca.vertcat(S[2, 1], S[0, 2], S[1, 0]))
        self._pose_map = ca.Function(f"pose_{link}", [q], [T[:3, 3], ca.jacobian(T[:3, 3], q[free]),
                                                           ca.reshape(R, 9, 1), ca.horzcat(*cols)]).map(self.horizon)

    def pose(self, trajectory: np.ndarray, q_now: np.ndarray):
        """`(p[H,3], J_p[H,3,n], R[H,3,3], J_w[H,3,n])` at `trajectory` (`(nq_opt, H)`)."""
        full = self.layout.full_q(np.asarray(trajectory, np.float64), q_now)
        pos, jp, rr, jw = self._pose_map(full)
        H, n = self.horizon, self.nq_opt
        p = np.asarray(pos, np.float64).T.reshape(H, 3)
        Jp = np.asarray(jp, np.float64).reshape(3, H, n).transpose(1, 0, 2)
        # casadi reshape is column-major: R(:) stacks columns
        R = np.asarray(rr, np.float64).T.reshape(H, 3, 3).transpose(0, 2, 1)
        Jw = np.asarray(jw, np.float64).reshape(3, H, n).transpose(1, 0, 2)
        return p, Jp, R, Jw


def rotation_errors(R: np.ndarray, rows: np.ndarray, R_goal: np.ndarray) -> np.ndarray:
    """`(n_rows, 3)` rotation vectors ``Log(R_goal^T R_k)`` on `rows`."""
    from AG3S.ag3s.grasp.ik import so3_log

    Rg = np.asarray(R_goal, np.float64)
    return np.array([so3_log(Rg.T @ np.asarray(R[k], np.float64)) for k in np.asarray(rows, int)]).reshape(-1, 3)


def rotation_cost(R: np.ndarray, rows: np.ndarray, R_goal: np.ndarray, w_rot: float) -> float:
    """Exact ``w_rot * sum_k ||Log(R_goal^T R_k)||^2`` over `rows`."""
    e = rotation_errors(R, rows, R_goal)
    return float(w_rot) * float(np.sum(e * e))


def rotation_qp_terms(iterate: np.ndarray, R_it: np.ndarray, Jw_it: np.ndarray, rows: np.ndarray, R_goal: np.ndarray,
                      w_rot: float) -> tuple[sp.csc_matrix, np.ndarray]:
    """The rotation term linearised at `iterate`: ``e_k(Q) ~ e_k(X) + A_k (Q_k - X_k)``, ``A_k = Jr^-1(e) R_k^T J_w``;
    per row ``w ||A_k Q_k + c_k||^2`` with ``c_k = e_k(X) - A_k X_k``, in `build_problem`'s layout."""
    from AG3S.ag3s.grasp.ik import so3_log, so3_right_jacobian_inv

    nq, H = iterate.shape
    Rg = np.asarray(R_goal, np.float64)
    blocks, q_add = [], np.zeros(nq * H)
    rows = set(int(k) for k in np.asarray(rows, int))
    for k in range(H):
        if k not in rows or w_rot <= 0.0:
            blocks.append(sp.csc_matrix((nq, nq)))
            continue
        Rk = np.asarray(R_it[k], np.float64)
        e = so3_log(Rg.T @ Rk)
        A = so3_right_jacobian_inv(e) @ Rk.T @ np.asarray(Jw_it[k], np.float64)
        c = e - A @ iterate[:, k]
        blocks.append(sp.csc_matrix(2.0 * w_rot * (A.T @ A)))
        q_add[k * nq:(k + 1) * nq] = 2.0 * w_rot * (A.T @ c)
    return sp.block_diag(blocks, format="csc"), q_add


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


def goal_box_excess(p: np.ndarray, rows: np.ndarray, goal: np.ndarray, half_width: float) -> np.ndarray:
    """E1-B — `(n_rows, 3)` per-axis excess (m, >= 0) of the palm positions `p[rows]` outside ``goal ± half_width``:
    exactly the value each goal-box slack takes at `p` (the merit's goal-box term is `weight * excess.sum()`)."""
    d = np.abs(np.asarray(p, np.float64)[np.asarray(rows, int)] - np.asarray(goal, np.float64).reshape(1, 3))
    return np.maximum(d - float(half_width), 0.0)


def add_goal_box(problem, iterate: np.ndarray, p_it: np.ndarray, J_it: np.ndarray, rows: np.ndarray,
                 goal: np.ndarray, half_width: float, weight: float):
    """E1-B — a copy of `problem` with the soft goal box on the palm rows `rows` (step indices), linearised at `iterate`.

    Per row k and axis a, with ``c = p_k(X) - J_k X_k`` (``p_k(Q) ~ J_k Q_k + c``) and one slack ``s >= 0``:

        J_k[a] Q_k - s <= g_a + r - c_a        J_k[a] Q_k + s >= g_a - r - c_a

    The slacks are new columns after every existing variable (`QpProblem.n_goal_slack`), priced linearly at `weight`
    per m in ``q`` (L1, exact above a finite weight, as the collision slack). Nothing else in `problem` changes.
    """
    rows = np.asarray(rows, int)
    nq, H = np.asarray(iterate).shape
    n_old = problem.P.shape[0]
    m = 3 * rows.size
    if m == 0:
        return problem
    g = np.asarray(goal, np.float64).reshape(3)
    r = float(half_width)
    data, ri, ci, lo, hi = [], [], [], [], []
    for i, k in enumerate(rows):
        J = np.asarray(J_it[k], np.float64)
        c = np.asarray(p_it[k], np.float64) - J @ np.asarray(iterate, np.float64)[:, k]
        for a in range(3):
            row = 3 * i + a
            ri.extend([row] * nq)
            ci.extend(range(k * nq, (k + 1) * nq))
            data.extend(J[a].tolist())
            lo.append(g[a] - r - c[a])
            hi.append(g[a] + r - c[a])
    # explicit zeros are kept (`coo -> csr` does not drop them): the pattern is the same at every iterate
    JQ = sp.csr_matrix((np.asarray(data, np.float64), (np.asarray(ri), np.asarray(ci))), shape=(m, n_old))
    eye = sp.identity(m, format="csr")
    zeros_m = sp.csr_matrix((m, n_old))
    A_old = sp.hstack([problem.A, sp.csr_matrix((problem.A.shape[0], m))], format="csr")
    A = sp.vstack([A_old,
                   sp.hstack([JQ, -eye], format="csr"),     # upper side:  J Q - s <= g + r - c
                   sp.hstack([JQ, eye], format="csr"),      # lower side:  J Q + s >= g - r - c
                   sp.hstack([zeros_m, eye], format="csr")],  # s >= 0
                  format="csc")
    start = problem.A.shape[0]
    row_blocks = dict(problem.row_blocks)
    row_blocks["goal_box"] = (start, start + m)
    row_blocks["goal_box_lower"] = (start + m, start + 2 * m)
    row_blocks["goal_box_slack"] = (start + 2 * m, start + 3 * m)
    lo, hi = np.asarray(lo), np.asarray(hi)
    P = sp.block_diag([problem.P, sp.csc_matrix((m, m))], format="csc")
    q = np.concatenate([np.asarray(problem.q, np.float64), np.full(m, float(weight))])
    l = np.concatenate([problem.l, np.full(m, -np.inf), lo, np.zeros(m)])
    u = np.concatenate([problem.u, hi, np.full(m, np.inf), np.full(m, np.inf)])
    return dataclasses.replace(problem, P=P, q=q, A=A, l=l, u=u, row_blocks=row_blocks,
                               n_goal_slack=int(getattr(problem, "n_goal_slack", 0)) + m)


def anchor_record(anchor: GraspAnchor, p_ref: np.ndarray, p_out: np.ndarray, R_ref: Optional[np.ndarray] = None,
                  R_out: Optional[np.ndarray] = None) -> dict[str, Any]:
    """What `TrajOptResult.metrics["grasp_anchor"]` carries: weights, rows, and the palm errors of reference vs result.

    AG3S-PRE-CORE-TO E1: also the variant, the goal rows, the per-axis goal error and (E1-B) the box half-width and
    the box excess (= the goal-box slack at the returned trajectory)."""
    track, goal = anchor.masks(p_ref.shape[0])
    rec: dict[str, Any] = {"variant": anchor.variant,
                           "link": anchor.link, "w_tcp": anchor.w_tcp, "w_goal": anchor.w_goal,
                           "close_row": anchor.close_row, "n_track_rows": int(track.sum()), "n_goal_rows": int(goal.sum()),
                           "goal_rows": [int(k) for k in np.flatnonzero(goal)],
                           "goal_m": None if anchor.goal is None else [round(float(v), 5) for v in anchor.goal],
                           "palm_dev_from_reference_max_mm": round(float(np.max(np.linalg.norm(p_out - p_ref, axis=1)) * 1000), 2)}
    if anchor.goal_box > 0.0:
        rec["goal_box_mm"] = round(float(anchor.goal_box) * 1000.0, 3)
    if getattr(anchor, "replaced_variant", None) is not None:   # AG3S-CORE-EVAL-G1: G1 filled the A4 slot
        rec["replaced_by_geo"] = True
        rec["replaced_variant"] = str(anchor.replaced_variant)
    if anchor.goal_window != "close":   # AG3S-PRE-CORE-TO-GGW (close: no key, the record as before)
        rec.update(goal_window=anchor.goal_window, goal_window_rule=anchor.goal_window_rule(p_ref.shape[0]),
                   horizon=int(p_ref.shape[0]))
    if goal.any():
        g = np.asarray(anchor.goal, np.float64).reshape(1, 3)
        rec["goal_err_reference_mm"] = round(float(np.max(np.linalg.norm(p_ref[goal] - g, axis=1)) * 1000), 2)
        rec["goal_err_result_mm"] = round(float(np.max(np.linalg.norm(p_out[goal] - g, axis=1)) * 1000), 2)
        rec["goal_err_axis_max_reference_mm"] = round(float(np.max(np.abs(p_ref[goal] - g))) * 1000, 2)
        rec["goal_err_axis_max_result_mm"] = round(float(np.max(np.abs(p_out[goal] - g))) * 1000, 2)
        if anchor.goal_R is not None and R_out is not None:
            rows = np.flatnonzero(goal)
            er_out = np.linalg.norm(rotation_errors(R_out, rows, anchor.goal_R), axis=1)
            rec.update(goal_kind=anchor.goal_kind, w_rot=anchor.w_rot,
                       goal_rot_err_result_deg=round(math.degrees(float(er_out.max())), 2),
                       tol_pos_mm=round(anchor.tol_pos * 1000, 1), tol_rot_deg=round(math.degrees(anchor.tol_rot), 1),
                       within_tolerance=bool(rec["goal_err_result_mm"] <= anchor.tol_pos * 1000
                                             and er_out.max() <= anchor.tol_rot))
            if R_ref is not None:
                er_ref = np.linalg.norm(rotation_errors(R_ref, rows, anchor.goal_R), axis=1)
                rec["goal_rot_err_reference_deg"] = round(math.degrees(float(er_ref.max())), 2)
        if anchor.goal_box > 0.0:
            ex = goal_box_excess(p_out, np.flatnonzero(goal), anchor.goal, anchor.goal_box)
            rec["goal_box_slack_sum_mm"] = round(float(ex.sum()) * 1000, 3)
            rec["goal_box_excess_max_mm"] = round(float(ex.max(initial=0.0)) * 1000, 3)
    return rec


__all__ = ["GOAL_BOX_SLACK_SCALE", "GraspAnchor", "LinkPointFK", "LinkPoseFK", "rotation_errors", "rotation_cost",
           "rotation_qp_terms", "anchor_targets", "anchor_cost", "anchor_qp_terms",
           "add_to_problem", "add_goal_box", "goal_box_excess", "anchor_record"]
