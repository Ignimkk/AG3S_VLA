"""T63 — register our own per-step cost terms with cuRobo v2 **without editing third_party/curobo**.

cuRobo's `RobotCostManager.compute_costs` evaluates a fixed set of names (tool pose, c-space,
self / scene collision); a cost registered under any other name is never evaluated. The hook is
the rollout's `cost_manager_config_instance_type` (`TrajOptSolverCfg.create(...,
cost_manager_config_instance_type=...)`): every rollout builds its cost manager from
`cfg.class_type`. `extensible_cost_manager_cfg(terms)` returns a `RobotCostManagerCfg` subclass
whose managers evaluate cuRobo's own costs **and then** each `ExtraCostTerm`.

B0 (basic cuRobo) passes no terms and therefore uses cuRobo's own `RobotCostManagerCfg` — this
module is not on the B0 path at all. B1-B3 add the TO's objective terms below (`TrackTerm`,
`ContinuityTerm`, `DeflectionRateTerm`, `ServoAccelTerm`; `SmoothTerm` exists but no variant
uses it); B4's link-specific field is a world change (`composed_world.py`), not a cost term.

Rules for a term (all rollouts share the same term objects):
* `forward(state, goal) -> (batch, horizon, 1)` tensor (cuRobo concatenates cost values on the
  last axis and sums), built from torch ops only, on tensors whose **shape and storage do not
  change** between solves — rollouts are captured in CUDA graphs, so per-chunk data (the reference
  chunk, the previous chunk) must be copied *into* preallocated buffers (`set_*` methods), never
  re-bound, and a term that is off for one chunk is switched off by a **weight tensor**, not by a
  Python branch;
* `setup_batch_tensors(batch, horizon)` is called when the rollout's batch shape changes;
* **weights are fixed at construction.** They enter the captured graph as Python constants;
  changing `term.w_*` after the first solve leaves the graph with the old value while uncaptured
  paths (line search bookkeeping, metrics) see the new one (measured: an inconsistent solve).

## How the TO terms are put on cuRobo's trajectory

cuRobo optimizes B-spline knots; the cost managers see the **state grid** `state.joint_state.
position (batch, H, 14)`, state 0 = the start (`q_now`), spaced `span / (H-1)` with the span fixed
to the chunk's 50/15 s (`solver.FixedDtTrajectoryOptimizer`). Our TO's terms are defined on the
**15 Hz chunk rows** (row `k` at `(k+1)/15 s`). Each term reads its rows with the fixed matrix
`spec.chunk_row_weights` (`exact`: a pick of every 2nd state; `resample16`: linear weights), so the
value is **the TO's own expression in the TO's own units** (rad, rad/row, rad/row^2) — no time
rescaling, no weight rescaling. `objective_ref.py` is the numpy reference of each term; the
unit tests check value and gradient against it, and it against `problem.py`.

`rows` = how many chunk rows a term covers: 50 (`--term-window chunk`, every row cuRobo
optimizes) or 16 (`--term-window plan`, the TO's plan window).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ExtraCostTerm(Protocol):
    name: str
    enabled: bool

    def setup_batch_tensors(self, batch_size: int, horizon: int) -> None: ...

    def forward(self, state: Any, goal: Optional[Any]) -> Any: ...


def extensible_cost_manager_cfg(terms: Sequence[ExtraCostTerm]):
    """`RobotCostManagerCfg` subclass whose cost managers also evaluate `terms`."""
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg

    terms = tuple(terms)
    names = [t.name for t in terms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate extra cost names {names}")
    reserved = {"tool_pose", "cspace", "self_collision", "scene_collision", "start_cspace_dist",
                "target_cspace_dist"}
    clash = reserved & set(names)
    if clash:
        raise ValueError(f"extra cost names clash with cuRobo's own: {sorted(clash)}")

    class ExtensibleRobotCostManager(RobotCostManager):
        extra_terms = terms
        #: The rollout builds its cost, constraint, hybrid and convergence managers from the
        #: same config type, so the terms are off until `activate_extra_terms` switches them on
        #: for the **cost** managers only (otherwise they would also be summed as constraints).
        extra_active = False

        def setup_batch_tensors(self, batch_size: int, horizon: int) -> None:
            changed = self._batch_size != batch_size or self._horizon != horizon
            super().setup_batch_tensors(batch_size, horizon)
            if changed:
                for term in self.extra_terms:
                    term.setup_batch_tensors(batch_size, horizon)

        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            collection = super().compute_costs(state, cost_collection, goal, **kwargs)
            if not self.extra_active:
                return collection
            for term in self.extra_terms:
                if term.enabled:
                    collection.add(term.forward(state, goal), term.name)
            return collection

    @dataclasses.dataclass
    class ExtensibleRobotCostManagerCfg(RobotCostManagerCfg):
        def __post_init__(self):
            self.class_type = ExtensibleRobotCostManager

        @staticmethod
        def create(data_dict, scene_collision_checker=None, device_cfg=None, **kw):
            base = (RobotCostManagerCfg.create(data_dict, scene_collision_checker)
                    if device_cfg is None else
                    RobotCostManagerCfg.create(data_dict, scene_collision_checker, device_cfg))
            fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)
                      if f.name != "class_type"}
            return ExtensibleRobotCostManagerCfg(**fields)

    ExtensibleRobotCostManagerCfg.extra_terms = terms
    ExtensibleRobotCostManagerCfg.manager_type = ExtensibleRobotCostManager
    return ExtensibleRobotCostManagerCfg


def activate_extra_terms(rollouts) -> int:
    """Switch the extra terms on for each rollout's `cost_manager` / `metrics_cost_manager` (not
    the constraint / convergence managers). Call before the first solve (CUDA graph capture)."""
    n = 0
    for r in rollouts:
        for attr in ("cost_manager", "metrics_cost_manager"):
            m = getattr(r, attr, None)
            if m is not None and hasattr(m, "extra_terms"):
                m.extra_active = True
                n += 1
    return n


# =================================================================================================
# TO objective terms on cuRobo's state grid (torch; no curobo import needed)
# =================================================================================================

class ChunkRowTerm:
    """Base: a TO term evaluated on chunk rows read off the optimizer state grid.

    Subclasses implement `row_costs(Q) -> (batch, rows)` with `Q (batch, rows, nq)`."""

    enabled: bool = True

    def __init__(self, name: str, *, rows: int, nq: int = 14, chunk_dt: float = 1.0 / 15.0,
                 span_s: float = 50.0 / 15.0, device: Any = "cpu", dtype: Any = None):
        import torch

        self.name = str(name)
        self.rows = int(rows)
        self.nq = int(nq)
        self.chunk_dt = float(chunk_dt)
        self.span_s = float(span_s)
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self._maps: dict[int, tuple[Any, Any]] = {}

    def _buf(self, *shape):
        import torch

        return torch.zeros(shape, device=self.device, dtype=self.dtype)

    def _map(self, horizon: int):
        """`(i0, i1, w0, w1, owner)` for `horizon` states: row `k` = `w0 x[i0] + w1 x[i1]`.

        Gather + elementwise, **not** a matmul: cuRobo enables TF32 matmul globally
        (`seed_ik_solver.py`), and a TF32 product of 1 rad positions is off by ~1e-4 rad."""
        m = self._maps.get(int(horizon))
        if m is None:
            import numpy as np
            import torch

            from AG3S.trajopt.curobo_v2.spec import chunk_row_weights

            W, owner = chunk_row_weights(int(horizon), self.rows, self.chunk_dt, self.span_s)
            k = np.arange(self.rows)
            i0 = (W > 0.0).argmax(1)  # first state with weight
            i1 = np.minimum(i0 + 1, int(horizon) - 1)
            w0 = W[k, i0]
            w1 = np.where(i1 != i0, W[k, i1], 0.0)
            if not np.allclose(w0 + w1, 1.0):
                raise ValueError("chunk_row_weights is not a two-point interpolation")
            if len(set(owner.tolist())) != self.rows:
                raise ValueError("two chunk rows share an owner state")
            t = lambda a, dt=None: torch.as_tensor(a, device=self.device, dtype=dt)  # noqa: E731
            m = (t(i0, torch.long), t(i1, torch.long), t(w0, self.dtype).view(1, -1, 1),
                 t(w1, self.dtype).view(1, -1, 1), t(owner, torch.long))
            self._maps[int(horizon)] = m
        return m

    def setup_batch_tensors(self, batch_size: int, horizon: int) -> None:
        self._map(horizon)

    def chunk_rows(self, position):
        """`(batch, H, nq)` states -> `(batch, rows, nq)` chunk rows."""
        i0, i1, w0, w1, _ = self._map(position.shape[-2])
        return w0 * position.index_select(-2, i0) + w1 * position.index_select(-2, i1)

    def forward(self, state: Any, goal: Optional[Any] = None):
        position = state.joint_state.position
        return self.forward_position(position)

    def forward_position(self, position):
        import torch

        *_, owner = self._map(position.shape[-2])
        per_row = self.row_costs(self.chunk_rows(position))
        out = torch.zeros(position.shape[:-1], device=position.device, dtype=per_row.dtype)
        return out.index_copy(-1, owner, per_row).unsqueeze(-1)

    def row_costs(self, Q):  # pragma: no cover - abstract
        raise NotImplementedError


class TrackTerm(ChunkRowTerm):
    """`w_track · Σ_k track_decay^k ‖Q_k − R_k‖²` (`problem.objective`, tracking)."""

    def __init__(self, w_track: float, track_decay: float = 1.0, **kw):
        import torch

        super().__init__(kw.pop("name", "to_track"), **kw)
        self.w_track, self.track_decay = float(w_track), float(track_decay)
        self.reference = self._buf(self.rows, self.nq)
        self.decay = torch.as_tensor([self.track_decay ** k for k in range(self.rows)],
                                     device=self.device, dtype=self.dtype)

    def set_reference(self, ref_rows) -> None:
        import torch

        self.reference.copy_(torch.as_tensor(ref_rows[: self.rows], dtype=self.dtype))

    def row_costs(self, Q):
        return self.w_track * self.decay * ((Q - self.reference) ** 2).sum(-1)


class ContinuityTerm(ChunkRowTerm):
    """`w_continuity · Σ_{k<L} ‖Q_k − P_k‖²`, `P` = the previous physical chunk's tail aligned at
    the executed step (`refiner._continuity_reference`), `L = min(len(P), rows)`. A chunk without
    a continuity reference (`None`) gets weight 0 (the TO drops the term)."""

    def __init__(self, w_continuity: float, **kw):
        super().__init__(kw.pop("name", "to_continuity"), **kw)
        self.w_continuity = float(w_continuity)
        self.previous = self._buf(self.rows, self.nq)
        self.mask = self._buf(self.rows)

    def set_previous(self, tail_rows) -> int:
        """`tail_rows (L, nq)` or `None`. Returns the overlap used."""
        import torch

        self.mask.zero_()
        self.previous.zero_()
        if tail_rows is None or len(tail_rows) == 0:
            return 0
        n = min(len(tail_rows), self.rows)
        self.previous[:n].copy_(torch.as_tensor(tail_rows[:n], dtype=self.dtype))
        self.mask[:n] = 1.0
        return n

    def row_costs(self, Q):
        return self.w_continuity * self.mask * ((Q - self.previous) ** 2).sum(-1)


class SmoothTerm(ChunkRowTerm):
    """`w_smooth · Σ_k ‖Q_{k+2} − 2Q_{k+1} + Q_k‖²` (reported at row `k+2`). Not in B1-B4."""

    def __init__(self, w_smooth: float, **kw):
        super().__init__(kw.pop("name", "to_smooth"), **kw)
        self.w_smooth = float(w_smooth)

    def row_costs(self, Q):
        import torch

        a = Q[:, 2:] - 2.0 * Q[:, 1:-1] + Q[:, :-2]
        out = torch.zeros(Q.shape[:2], device=Q.device, dtype=Q.dtype)
        return torch.cat([out[:, :2], self.w_smooth * (a ** 2).sum(-1)], dim=1)


class DeflectionRateTerm(ChunkRowTerm):
    """`w · (Σ_k ‖D_{k+1} − D_k‖² + ‖D_0 − D_prev‖²)`, `D = Q − R` (`problem.deflection_rate_cost`);
    the boundary part only when the executed history exists (`set_history(None)` drops it)."""

    def __init__(self, w_deflection_rate: float, **kw):
        super().__init__(kw.pop("name", "to_deflection_rate"), **kw)
        self.w = float(w_deflection_rate)
        self.reference = self._buf(self.rows, self.nq)
        self.d_prev = self._buf(self.nq)
        self.boundary = self._buf(1)

    def set_reference(self, ref_rows) -> None:
        import torch

        self.reference.copy_(torch.as_tensor(ref_rows[: self.rows], dtype=self.dtype))

    def set_history(self, deflection_history) -> bool:
        import torch

        if deflection_history is None:
            self.d_prev.zero_()
            self.boundary.zero_()
            return False
        self.d_prev.copy_(torch.as_tensor(deflection_history, dtype=self.dtype).reshape(-1))
        self.boundary.fill_(1.0)
        return True

    def row_costs(self, Q):
        import torch

        D = Q - self.reference
        first = self.boundary * ((D[:, 0] - self.d_prev) ** 2).sum(-1)
        rate = ((D[:, 1:] - D[:, :-1]) ** 2).sum(-1)
        return self.w * torch.cat([first.unsqueeze(1), rate], dim=1)


class ServoAccelTerm(ChunkRowTerm):
    """The TO's servo-accel price (T43 TA C, `problem.objective`):
    `w_slack · servo_accel_weight · Σ max(0, |row| − b)` over the second differences of
    `[u_{-2}, u_{-1}, Q_0, …]`, plus `w_slack · 10 · Σ max(0, |Q_0 − u_{-1}| − v_max)`.

    Row `i` here is the stencil on padded entries `i, i+1, i+2`; rows 0 and 1 touch the history and
    are masked off without it (then the remaining rows are exactly `problem.servo_accel_rows`'s
    in-chunk rows). Bounds come from `objective_ref.servo_accel_rows` (strict / relaxed, capped by
    the robot's `max_step_change`)."""

    def __init__(self, w_slack: float, servo_accel_weight: float, **kw):
        super().__init__(kw.pop("name", "to_servo_accel"), **kw)
        self.w_rows = float(w_slack) * float(servo_accel_weight)
        self.w_bvel = float(w_slack) * 10.0
        self.history = self._buf(2, self.nq)
        self.row_mask = self._buf(self.rows)
        self.bound = self._buf(self.rows, self.nq)
        self.has_bvel = self._buf(1)
        self.max_step = self._buf(self.nq)
        self.active = self._buf(1)

    def set_chunk(self, ref_rows, accel_bound, mode: str, *, max_step_change=None,
                  command_history=None, max_step=None) -> None:
        """Host-side bounds (numpy, `objective_ref`) copied into the buffers. `accel_bound=None`
        switches the term off for this chunk (`limits.servo_accel = off`)."""
        import numpy as np
        import torch

        from AG3S.trajopt.curobo_v2.objective_ref import servo_accel_rows

        for t in (self.history, self.row_mask, self.bound, self.has_bvel, self.max_step,
                  self.active):
            t.zero_()
        if accel_bound is None:
            return
        R = np.asarray(ref_rows, np.float64)[: self.rows].T
        _, _, bound = servo_accel_rows(R, accel_bound, mode, max_step_change, command_history)
        full = np.zeros((self.rows, self.nq))
        mask = np.zeros(self.rows)
        if command_history is not None:
            full[: bound.shape[0]] = bound
            mask[: bound.shape[0]] = 1.0
            self.history.copy_(torch.as_tensor(np.asarray(command_history, np.float64).T,
                                               dtype=self.dtype))
            if max_step is not None and np.all(np.isfinite(max_step)):
                self.has_bvel.fill_(1.0)
                self.max_step.copy_(torch.as_tensor(np.asarray(max_step, np.float64),
                                                    dtype=self.dtype))
        else:
            full[2: 2 + bound.shape[0]] = bound
            mask[2: 2 + bound.shape[0]] = 1.0
        self.bound.copy_(torch.as_tensor(full, dtype=self.dtype))
        self.row_mask.copy_(torch.as_tensor(mask, dtype=self.dtype))
        self.active.fill_(1.0)

    def row_costs(self, Q):
        import torch

        hist = self.history.unsqueeze(0).expand(Q.shape[0], 2, self.nq)
        X = torch.cat([hist, Q], dim=1)  # (b, rows + 2, nq)
        a = X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]  # row i = padded i..i+2, (b, rows, nq)
        over = torch.clamp(a.abs() - self.bound, min=0.0).sum(-1) * self.row_mask
        bvel = torch.clamp((Q[:, 0] - hist[:, 1]).abs() - self.max_step, min=0.0).sum(-1)
        out = self.w_rows * over
        out = torch.cat([out[:, :1] + self.w_bvel * self.has_bvel * bvel.unsqueeze(1), out[:, 1:]],
                        dim=1)
        return self.active * out


__all__ = ["ChunkRowTerm", "ContinuityTerm", "DeflectionRateTerm", "ExtraCostTerm",
           "ServoAccelTerm", "SmoothTerm", "TrackTerm", "activate_extra_terms",
           "extensible_cost_manager_cfg"]
