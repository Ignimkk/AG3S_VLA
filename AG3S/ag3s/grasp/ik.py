"""AG3S-GEO-GRASP — inverse kinematics of one link's 6-D pose over a subset of joints (damped least squares).

A reachability pre-filter for grasp candidates, not a planner: it answers "is there a joint configuration within the
limits that puts the hold link at this pose", starting from a seed (the policy's close-row configuration or the
current one). The error is ``[p - p_goal ; Log(R_goal^T R)]`` and the step uses the matching Jacobian
``[J_p ; Jr^-1(e) R^T J_w]`` — the same rotation error the TrajOpt goal term uses (`AG3S.trajopt.grasp_anchor`).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of `R` (rad), stable near 0 and pi."""
    R = np.asarray(R, np.float64)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    if math.pi - th < 1e-4:
        # near pi: axis from the diagonal
        A = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / math.sqrt(max(A[k, k], 1e-12))
        if float(np.dot(ax, w)) < 0.0:
            ax = -ax
        return th * ax / np.linalg.norm(ax)
    return th / (2.0 * math.sin(th)) * w


def _hat(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_right_jacobian_inv(e: np.ndarray) -> np.ndarray:
    """``Jr^-1(e)``: ``d Log(R0^T R) = Jr^-1(e) (R^T dR)^v`` at ``e = Log(R0^T R)``."""
    e = np.asarray(e, np.float64)
    th = float(np.linalg.norm(e))
    E = _hat(e)
    if th < 1e-6:
        return np.eye(3) + 0.5 * E + E @ E / 12.0
    return np.eye(3) + 0.5 * E + (1.0 / th**2 - (1.0 + math.cos(th)) / (2.0 * th * math.sin(th))) * (E @ E)


class LinkPoseIK:
    """IK of `link` over the joints `joint_indices` (indices into the model's full q) of `robot_model`
    (`UrdfSphereChain`), the other joints fixed at the `q_full` given per call."""

    def __init__(self, robot_model, link: str, joint_indices: Sequence[int], *, damping: float = 1e-3,
                 max_iterations: int = 100, pos_tol: float = 0.002, rot_tol_deg: float = 2.0, rot_weight: float = 0.2,
                 limits: Optional[tuple[np.ndarray, np.ndarray]] = None):
        import casadi as ca

        self.link = str(link)
        self.idx = np.asarray(list(joint_indices), int)
        self.nq = int(robot_model.nq)
        if limits is None:
            lo, hi = robot_model.joint_limits()
            self.lo, self.hi = np.asarray(lo)[self.idx], np.asarray(hi)[self.idx]
        else:
            # the controlled robot's position limits (MJCF `jnt_range`, `AG3S.trajopt.limits`): the URDF's are
            # narrower on the wrist (arm_6 +-2.705 vs +-2.967) and the policy uses that range
            self.lo, self.hi = (np.asarray(v, np.float64).reshape(-1) for v in limits)
        q = ca.SX.sym("q", self.nq, 1)
        T = robot_model.link_pose_symbolic(q, link)
        p = T[:3, 3]
        Jp = ca.jacobian(p, q[self.idx.tolist()])
        # spatial angular velocity Jacobian from dR R^T
        R = T[:3, :3]
        cols = []
        for k in self.idx.tolist():
            dR = ca.jacobian(ca.reshape(R, 9, 1), q[k])
            dR = ca.reshape(dR, 3, 3)
            S = ca.mtimes(dR, R.T)
            cols.append(ca.vertcat(S[2, 1], S[0, 2], S[1, 0]))
        Jw = ca.horzcat(*cols)
        self._f = ca.Function(f"ik_{link}", [q], [T, Jp, Jw])
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.pos_tol = float(pos_tol)
        self.rot_tol = math.radians(float(rot_tol_deg))
        self.rot_weight = float(rot_weight)
        #: give up a seed still `abort_pos` (m) away after `abort_after` iterations.
        self.abort_after = 25
        self.abort_pos = 0.02

    def pose_and_jacobians(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T, Jp, Jw = self._f(np.asarray(q_full, np.float64))
        return np.asarray(T), np.asarray(Jp), np.asarray(Jw)

    def solve(self, T_goal: np.ndarray, q_full: np.ndarray, seed: Optional[np.ndarray] = None
              ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        """`(q_sub or None, info)` from one seed (default: the solved joints of `q_full`). Levenberg-Marquardt: the
        damping shrinks after a step that lowers the weighted error and grows (step rejected) otherwise."""
        T_goal = np.asarray(T_goal, np.float64)
        q = np.asarray(q_full, np.float64).copy()
        if seed is not None:
            q[self.idx] = np.clip(np.asarray(seed, np.float64).reshape(-1), self.lo, self.hi)
        Rg = T_goal[:3, :3]
        w = self.rot_weight
        lam = self.damping

        def residual(qq):
            T, Jp, Jw = self.pose_and_jacobians(qq)
            ep = T[:3, 3] - T_goal[:3, 3]
            er = so3_log(Rg.T @ T[:3, :3])
            return T, Jp, Jw, ep, er

        T, Jp, Jw, ep, er = residual(q)
        cost = float(ep @ ep + w * w * er @ er)
        info: dict[str, Any] = {}
        for it in range(self.max_iterations):
            info = {"iterations": it, "pos_err_mm": round(float(np.linalg.norm(ep)) * 1000, 3),
                    "rot_err_deg": round(math.degrees(float(np.linalg.norm(er))), 3)}
            if np.linalg.norm(ep) <= self.pos_tol and np.linalg.norm(er) <= self.rot_tol:
                info["converged"] = True
                return q[self.idx].copy(), info
            J = np.vstack([Jp, w * (so3_right_jacobian_inv(er) @ T[:3, :3].T @ Jw)])
            e = np.concatenate([ep, w * er])
            dq = -J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > 0.4:
                dq *= 0.4 / step
            q_try = q.copy()
            q_try[self.idx] = np.clip(q[self.idx] + dq, self.lo, self.hi)
            T2, Jp2, Jw2, ep2, er2 = residual(q_try)
            c2 = float(ep2 @ ep2 + w * w * er2 @ er2)
            if it == self.abort_after and float(np.linalg.norm(ep)) > self.abort_pos:
                break                                   # far from converging: leave the budget to the next seed
            if c2 < cost:
                q, T, Jp, Jw, ep, er, cost = q_try, T2, Jp2, Jw2, ep2, er2, c2
                lam = max(lam * 0.3, 1e-6)
            else:
                lam = min(lam * 10.0, 1e2)
        info["converged"] = False
        return None, info

    def solve_seeds(self, T_goal: np.ndarray, q_full: np.ndarray, seeds: Sequence[np.ndarray]
                    ) -> tuple[Optional[np.ndarray], dict[str, Any]]:
        """`solve` from each seed in order; the first converged answer (info carries `seed_index`, `n_seeds_tried`)."""
        last: dict[str, Any] = {}
        for i, sd in enumerate(seeds):
            q, info = self.solve(T_goal, q_full, sd)
            last = dict(info, seed_index=i, n_seeds_tried=i + 1)
            if q is not None:
                return q, last
        return None, last

    def wrist_seeds(self, seed: np.ndarray, extra: Sequence[np.ndarray] = (), n_random: int = 0,
                    turns: Sequence[float] = (0.5 * math.pi, -0.5 * math.pi)) -> list[np.ndarray]:
        """`seed`, then `seed` with the last solved joint (the wrist roll about the approach axis) turned by
        `turns` (default +-90 deg, when inside its limits), then `extra`, then `n_random` fixed pseudo-random
        configurations within the limits (same every call — deterministic)."""
        out = [np.asarray(seed, np.float64).copy()]
        for d in turns:
            s = out[0].copy()
            s[-1] = s[-1] + d
            if self.lo[-1] <= s[-1] <= self.hi[-1]:
                out.append(s)
        rnd = np.random.default_rng(0).uniform(self.lo, self.hi, size=(int(n_random), len(self.lo)))
        return out + [np.asarray(e, np.float64).copy() for e in extra] + list(rnd)


def mjcf_limits(model_xml: str, joint_names) -> tuple[np.ndarray, np.ndarray]:
    """`(lo, hi)` of `joint_names` from the MJCF `jnt_range` (the controlled robot's limits)."""
    import mujoco

    import os

    m = mujoco.MjModel.from_xml_path(os.path.abspath(str(model_xml)))   # relative mesh dirs resolve from the file
    r = np.array([m.joint(n).range for n in joint_names], np.float64)
    return r[:, 0], r[:, 1]


__all__ = ["so3_log", "so3_right_jacobian_inv", "LinkPoseIK", "mjcf_limits"]
