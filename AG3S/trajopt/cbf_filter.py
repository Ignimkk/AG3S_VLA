"""T94q — one-step discrete-time CBF-QP safety filter on AG3S's own constraint rows (default off).

Replaces only the trajectory optimizer: the same `SceneSnapshot` (ESDF layers, target exclusion,
support planes, held object) and the same robot spheres, linearized by the same
`CollisionLinearizer.linearize`, so the comparison against TrajOpt changes *one* thing — how the
rows are used.

Per executed step k (state ``q_k``, policy waypoint ``r_k``)::

    u* = argmin ||u - (r_k - q_k)||^2 + w_s ||s||^2
         s.t.  grad h_i(q_k) . u + s_i >= -gamma * h_i(q_k)     (every active row i)
               s >= 0,  |u| <= max_step,  lower <= q_k + u <= upper
    q_{k+1} = q_k + u*

``h(q_{k+1}) >= (1 - gamma) h(q_k)`` linearized at ``q_k`` (KNOWS Eq. (8) form, joint space as
KNOWS D7, soft rows as KNOWS D3). No horizon, no smoothness, no continuity term: the filter is
myopic by construction — that is what a CBF-QP is. Every planned step is filtered in sequence; with
look-ahead on, only the certified window is, and the tail is the policy's own tail shifted by the last
deflection (never executed; it only feeds the record).

The result goes through `TrajectoryOptimizer._finish`, so status / clearance / records are measured
exactly as for TrajOpt, and `SafePolicy` treats a VIOLATED chunk the same way.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import osqp
import scipy.sparse as sp

from AG3S.trajopt.sqp import TrajectoryOptimizer
from AG3S.trajopt.types import TrajOptResult, TrajOptStatus

#: Defaults (T94q). gamma = fraction of the current barrier value the step may consume.
CBF_GAMMA = 0.3
CBF_SLACK_WEIGHT = 1.0e4


class CbfSafetyFilter(TrajectoryOptimizer):
    """`TrajectoryOptimizer` with `solve` replaced by the per-step CBF-QP above."""

    gamma: float = CBF_GAMMA
    slack_weight: float = CBF_SLACK_WEIGHT

    def _qp(self, h: np.ndarray, G: np.ndarray, u_nom: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        """Solve the step QP. Variables x = [u (n), s (m)]."""
        n, m = u_nom.size, h.size
        if m == 0:
            return np.clip(u_nom, lo, hi), 0.0, "no-rows"
        P = sp.diags(np.concatenate([np.full(n, 2.0), np.full(m, 2.0 * self.slack_weight)])).tocsc()
        q = np.concatenate([-2.0 * u_nom, np.zeros(m)])
        A = sp.vstack([
            sp.hstack([sp.csr_matrix(G), sp.eye(m)]),                       # G u + s >= -gamma h
            sp.hstack([sp.csr_matrix((m, n)), sp.eye(m)]),                  # s >= 0
            sp.hstack([sp.eye(n), sp.csr_matrix((n, m))]),                  # lo <= u <= hi
        ]).tocsc()
        l = np.concatenate([-self.gamma * h, np.zeros(m), lo])
        u = np.concatenate([np.full(m, np.inf), np.full(m, np.inf), hi])
        solver = osqp.OSQP(algebra="builtin")
        solver.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polish=True)
        res = solver.solve()
        x = getattr(res, "x", None)
        status = str(getattr(res.info, "status", "?"))
        if x is None or not np.all(np.isfinite(x[:n])):
            return np.clip(u_nom, lo, hi), float("nan"), status
        return np.clip(x[:n], lo, hi), float(np.linalg.norm(x[n:])), status

    def solve(self, reference, q_now, scene, **kwargs) -> TrajOptResult:
        started = time.perf_counter()
        reference = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        template = kwargs.get("template")
        previous_chunk = kwargs.get("previous_chunk")
        certified = bool(kwargs.get("geometry_certified", True))
        notes: list[str] = [f"cbf-qp (T94q): gamma={self.gamma:g}, slack_weight={self.slack_weight:g}"]
        # `_finish` reads these solve-scoped attributes; a CBF solve sets none of them.
        self._history = {}
        self._anchor_state = None
        self._gate_record = None
        self._track_scale = None
        if scene is None or scene.is_empty:
            return self._passthrough(reference, q_now, template, TrajOptStatus.UNCONSTRAINED,
                                     notes + ["AG3S emitted no geometry to avoid; the reference passes through"],
                                     started)
        cfg = self.config
        H = reference.shape[1]
        # every planned step is filtered (the certified window is all of them unless look-ahead is on)
        E = H if self.lookahead_from is None else int(min(max(self.lookahead_from, cfg.horizon.execution_length), H))
        q = q_now[self.layout.q_indices].copy()
        lim = self.limits
        traj = reference.copy()
        steps: list[dict[str, Any]] = []
        slack_total = 0.0
        for k in range(E):
            probe = traj.copy()
            probe[:, k] = q                      # rows at step k evaluated at the current state
            rows = self.linearizer.linearize(probe, q_now, scene, cfg)
            used = rows.used[k]
            h = np.asarray(rows.value[k][used], np.float64)
            G = np.asarray(rows.gradient[k][used], np.float64)
            u_nom = reference[:, k] - q
            lo = np.maximum(-lim.max_step, lim.lower - q)
            hi = np.minimum(lim.max_step, lim.upper - q)
            u, slack, status = self._qp(h, G, u_nom, lo, hi)
            slack_total += 0.0 if not np.isfinite(slack) else slack ** 2
            steps.append({"k": k, "rows": int(used.sum()),
                          "h_min_mm": (float(h.min()) * 1000.0 if h.size else None),
                          "deflection_rad": float(np.linalg.norm(u - u_nom)),
                          "slack": slack, "qp_status": status})
            q = q + u
            traj[:, k] = q
        if E < H:
            traj[:, E:] = reference[:, E:] + (q - reference[:, E - 1])[:, None]
            traj[:, E:] = np.clip(traj[:, E:], lim.lower[:, None], lim.upper[:, None])
        steps_cert = self.lookahead_from if self.lookahead_from is not None else None
        ref_clear, _ = self.linearizer.worst_row(reference, q_now, scene, steps=steps_cert)
        reference_violation = max(0.0, -float(ref_clear))
        result = self._finish(traj, reference, q_now, scene, template, E, float(np.sqrt(slack_total)),
                              reference_violation, previous_chunk, notes, started, certified)
        result.metrics["cbf"] = {"gamma": self.gamma, "slack_weight": self.slack_weight,
                                 "executed_steps": E, "steps": steps}
        return result


def apply_cbf_overrides(gamma: Optional[float] = None, slack_weight: Optional[float] = None) -> None:
    """Server start: set the class-level parameters (one filter per server)."""
    if gamma is not None:
        if not 0.0 < float(gamma) <= 1.0:
            raise SystemExit(f"--cbf-gamma must be in (0, 1]: {gamma}")
        CbfSafetyFilter.gamma = float(gamma)
    if slack_weight is not None:
        CbfSafetyFilter.slack_weight = float(slack_weight)
