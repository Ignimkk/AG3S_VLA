"""The real-time iteration loop: linearize, solve, accept, re-check.

Collision avoidance is not convex — ``||p(q) - c|| >= R`` carves a hole out of the configuration
space — so the problem is solved as a sequence of convex ones. Each iteration linearizes the
clearances at the current trajectory, solves a QP inside a trust region, and either accepts the step
or shrinks the region and tries again.

**The budget is a deadline, not a suggestion.** `sqp.max_iterations` and `sqp.time_budget_ms` are
both hard caps, and whichever binds first ends the loop with the best iterate so far. That is the
real-time iteration scheme's central idea: at 15 Hz the optimizer runs once per chunk while the
control loop waits, so an algorithm that runs until it converges cannot be used at all. What makes
the truncation safe is not that the answer is optimal — it is that the answer is *checked*.

**T31 — iterations first.** The deadline is checked only after `sqp.min_iterations` iterations
(default 3). With the ESDF backend the first linearization on a new chunk's field costs ~100 ms
(later ones ~3 ms), so a 50 ms deadline checked after the first iteration meant *one QP per chunk,
always* — a rejected first step was returned without the retry the trust region exists to give it. The simulation loop is synchronous; a real
robot that needs the old wall-clock behaviour sets `min_iterations: 1`.

**A limit-only initial iterate falls back to its projection, not to a rejected step** (T31). When the
reference breaks only the joint limits (no collision violation) and no QP candidate is accepted, the
returned trajectory is the reference clipped onto the position box and the step-0 anchor — provided
that clip satisfies every limit and passes the full-resolution collision check. The rejected QP
candidate it replaces moved every joint (measured on E3b chunk 310: up to 8.7° on the wrist that
was out of range, and 0.4–0.9° on joints that were not).

**Every returned trajectory is checked at full resolution.** `linearize` may hand the QP a small
subset of the rows; `_finish` re-evaluates all of them. That is the whole basis on which the
reduction is allowed to exist: pruning may cost convergence, and it may not hide a collision. The
check costs about a twelfth of a solve, because it needs no Jacobian.

**A trajectory that could not be made safe is still returned.** `TrajOptStatus.VIOLATED` carries the
best effort together with the worst remaining violation in metres. Raising instead would leave the
controller with nothing to send at the exact moment it most needs something, which is how a safety
layer ends up switched off in production. Deciding between stopping, holding and re-planning belongs
to whoever owns the task — the same division of labour AG3S keeps with `ConstraintValidity`.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import numpy as np

from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.linearize import (
    CollisionBlock,
    CollisionLinearizer,
    SceneSnapshot,
    append_collision_rows,
)
from AG3S.trajopt.limits import (
    limit_report,
    project_to_limits,
    rest_start_envelope,
    velocity_rows,
)
from AG3S.trajopt.problem import build_problem, objective, rebox_problem
from AG3S.trajopt.qp import QpSolver
from AG3S.trajopt.types import ChunkLayout, JointLimits, TrajOptResult, TrajOptStatus


#: 초기 iterate 가 "한계 안" 인가의 문턱 (rad 또는 rad/step). `_finish` 가 limit overshoot 를
#: 노트에 적는 문턱과 같다 — 두 곳이 다르면 "한계 안" 의 뜻이 갈라진다.
_LIMIT_TOLERANCE = 1e-6


class TrajectoryOptimizer:
    """Stateful across chunks, because that is where the warm start comes from.

    The QP backend keeps its factorization, and the previous solution seeds the next chunk's first
    iterate. Both are only valid while the problem's sparsity pattern holds, which is why the row
    budget is per step and fixed rather than per scene — see `linearize`.
    """

    def __init__(
        self,
        robot_model,
        layout: ChunkLayout,
        limits: JointLimits,
        config: Optional[TrajOptConfig] = None,
    ):
        self.config = config or TrajOptConfig()
        self.layout = layout
        self.limits = limits
        self.horizon = self.config.horizon.horizon
        self.linearizer = CollisionLinearizer(robot_model, layout, self.horizon)
        #: T43 TP — `sqp.fast_eval`: buffered FK + per-solve memo (bit-identical). Off = as before.
        self.fast_eval = bool(getattr(self.config.sqp, "fast_eval", False))
        if self.fast_eval:
            self.linearizer.fast_eval = True
        self.block = CollisionBlock(
            self.horizon, self.config.reduction.rows_per_step, layout.nq_opt
        )
        self.solver = QpSolver(self.config.qp)
        self._previous: Optional[np.ndarray] = None
        self.frame_index = 0
        #: T43 Q — servo model (`limits.servo_model`) and the rows between waypoints
        #: (`reduction.sweep_check`). Both `None` when both are off: nothing below reads them then.
        self.servo = None
        self.observer = None
        self.path = None
        if (getattr(self.config.limits, "servo_model", False)
                or getattr(self.config.reduction, "sweep_check", False)):
            self._build_path(robot_model)
        #: T43 TA (A) — first look-ahead step, or None (every planned step certified, as before).
        self.lookahead_from = self.config.horizon.lookahead_from
        self.linearizer.lookahead_from = self.lookahead_from
        if self.lookahead_from is not None and int(
                getattr(self.config.horizon, "lookahead_stride", 1)) > 1:
            # T43 TP — the tail's field rows on every `stride`-th step only.
            self.linearizer.lookahead_stride = int(self.config.horizon.lookahead_stride)
        #: T43 TA (A) — the gate's own optimizer: the executed window alone, every other setting
        #: as this one (`horizon.lookahead_gate`). `None` without look-ahead or without a gate.
        self.window_optimizer: Optional["TrajectoryOptimizer"] = None
        self.lookahead_gate = (None if self.lookahead_from is None
                               else self.config.horizon.lookahead_gate)
        if self.lookahead_gate is not None:
            k = int(self.lookahead_from)
            window_cfg = self.config.with_overrides(
                {"horizon": {"horizon": k, "execution_length": k, "plan_horizon": None,
                             "lookahead": "off"}})
            self.window_optimizer = TrajectoryOptimizer(robot_model, layout, limits, window_cfg)
            # AG3S-PRE-CORE-TO E2: the window optimizer runs its multi-seed only when this one hands it a chunk index.
            self.window_optimizer._ms_is_window = True
        #: T43 TA (C) — `a_C` per optimized joint (rad/row²), or None when `limits.servo_accel` is off.
        self.servo_accel_bound = None
        self.servo_accel_record = None
        if getattr(self.config.limits, "servo_accel", "off") != "off":
            self._build_servo_accel(robot_model)
        #: T43 TA (B, C) — this solve's history (`solve(deflection_history=, command_history=)`).
        self._history: dict[str, Any] = {}
        #: IFG-TOA — this solve's grasp-anchor state (`_anchor_setup`), `None` outside a solve or without an anchor.
        self._anchor_state: Optional[dict[str, Any]] = None
        #: AG3S-PRE-CORE-TO E2 — `sqp.multi_seed` (0 = off: `solve` is the standard solve, nothing below is read).
        self.multi_seed = int(getattr(self.config.sqp, "multi_seed", 0) or 0)
        #: AG3S-PRE-CORE-TO E3-b — `sqp.gpu_seeds` ("off" = E2's seeds on the CPU SQP; "curobo" = `gpu_seeds.py`).
        self.gpu_seeds = str(getattr(self.config.sqp, "gpu_seeds", "off") or "off")
        #: E3-b — `sqp.gpu_seeds_check` ("all" = every GPU result re-checked; "first-pass" = in the choosing order).
        self.gpu_seeds_check = str(getattr(self.config.sqp, "gpu_seeds_check", "all") or "all")

    def _build_path(self, robot_model) -> None:
        """T43 Q — the servo (from the controlled MJCF) and the path checker. Only when on."""
        from AG3S.trajopt.sweep import PathChecker

        cfg = self.config
        if getattr(cfg.limits, "servo_model", False):
            from AG3S.trajopt.limits import limits_model_xml
            from AG3S.trajopt.servo import ServoModel, ServoObserver, servo_params_from_xml

            names = [str(robot_model.joint_names[int(i)]) for i in self.layout.q_indices]
            params = servo_params_from_xml(limits_model_xml(cfg.limits), names,
                                           control_hz=cfg.horizon.control_hz)
            self.servo = ServoModel(params)
            self.observer = ServoObserver(self.servo)
        self.path = PathChecker(self.linearizer, self.layout, cfg, self.servo)
        self.path.warm_seconds = self.path.warm()

    def _build_servo_accel(self, robot_model) -> None:
        """T43 TA (C) — `a_C = ε Δt² / κ` from the controlled MJCF (`servo.ServoParams.accel_bound`)."""
        from AG3S.trajopt.limits import limits_model_xml
        from AG3S.trajopt.servo import servo_params_from_xml

        cfg = self.config
        names = [str(robot_model.joint_names[int(i)]) for i in self.layout.q_indices]
        params = servo_params_from_xml(limits_model_xml(cfg.limits), names,
                                       control_hz=cfg.horizon.control_hz)
        tol = float(cfg.limits.servo_accel_tolerance)
        bound = params.accel_bound(tol, row_seconds=cfg.horizon.dt)
        self.servo_accel_bound = np.asarray(bound, np.float64)
        self.servo_accel_record = {
            "mode": str(cfg.limits.servo_accel), "tolerance_rad": tol,
            "row_seconds": float(cfg.horizon.dt),
            "shape_coefficient_s2": [round(float(v), 6) for v in params.shape_coefficient()],
            "a_c_rad_per_row2": [round(float(v), 6) for v in bound],
            "joint_names": names, "weight": float(cfg.limits.servo_accel_weight)}

    def _max_step_change(self) -> Optional[np.ndarray]:
        step = np.asarray(self.limits.max_step_change, np.float64)
        return step if np.all(np.isfinite(step)) else None

    def _objective(self, trajectory, reference, previous_chunk) -> float:
        """`problem.objective` with this solve's T43 TA history (identical call when TA is off)."""
        extra: dict[str, Any] = {}
        if self._history.get("deflection") is not None:
            extra["deflection_history"] = self._history["deflection"]
        if self.__dict__.get("_track_scale") is not None:
            extra["joint_track_scale"] = self._track_scale   # IFG-TOW
        if self.servo_accel_bound is not None:
            step = np.asarray(self.limits.max_step, np.float64)
            extra.update(servo_accel_bound=self.servo_accel_bound,
                         command_history=self._history.get("command"),
                         max_step_change=self._max_step_change(),
                         max_step=step if velocity_rows(self.limits) else None)
        value = objective(trajectory, reference, self.config, previous_chunk=previous_chunk, **extra)
        anchor = self.__dict__.get("_anchor_state")
        if anchor is not None and np.asarray(trajectory).shape[1] == anchor["horizon"]:
            # IFG-TOA: the anchor terms, exactly (palm FK), so step acceptance sees what the QP optimises.
            from AG3S.trajopt.grasp_anchor import anchor_cost, goal_box_excess

            rot = anchor.get("rot")
            if rot is not None:
                # AG3S-GEO-GRASP: position and rotation of the hold link, exactly (same Log map as the QP terms)
                from AG3S.trajopt.grasp_anchor import rotation_cost

                p, _, R, _ = anchor["fk"].pose(trajectory, anchor["q_now"])
                value += rotation_cost(R, rot["rows"], rot["Rg"], rot["w"])
            else:
                p = anchor["fk"](trajectory, anchor["q_now"])[0]
            if anchor["cost_on"]:
                value += anchor_cost(p, anchor["weights"], anchor["targets"])
            box = anchor.get("box")
            if box is not None:
                # E1-B: the goal-box slack's exact value at this trajectory, at the QP's price (w_slack / 10 per m).
                value += box["weight"] * float(goal_box_excess(p, box["rows"], box["goal"], box["r"]).sum())
        return value

    def reset(self) -> None:
        """Clear cross-chunk state. Between episodes, never mid-episode."""
        self.solver.reset()
        self._previous = None
        self.frame_index = 0
        if self.observer is not None:
            self.observer.reset()
        if self.window_optimizer is not None:
            self.window_optimizer.reset()

    # ------------------------------------------------------------------------------------
    #: AG3S-PRE-CORE-TO E2 — multi-seed bookkeeping (class defaults; only read with `sqp.multi_seed > 0`).
    #: `_ms_is_window`: this is a look-ahead window optimizer (multi-seed only with a chunk index from its owner);
    #: `_ms_depth`: nesting of `solve` (the IFG-TOA safety-first re-solve calls it again); `_ms_count`: top-level
    #: solves so far (the chunk index when the caller gives none); `_ms_child_index`: the index the current top-level
    #: solve hands its window optimizer (`_solve_window`), `None` inside the IFG-TOA3 safety-first re-solve.
    _ms_is_window = False
    _ms_depth = 0
    _ms_count = 0
    _ms_child_index: Optional[int] = None

    def solve(self, reference, q_now, scene, **kwargs) -> TrajOptResult:
        """The standard solve (`_solve_standard`) and, with `sqp.multi_seed = N > 0` (AG3S-PRE-CORE-TO E2), the
        multi-seed re-solve when it leaves a violation (`_multi_seed`). Off (default), this is `_solve_standard`.

        `multi_seed_index` (keyword, optional): the chunk index the seeds' RNG uses (the refiner passes the request's
        `t_step`); without it a count of this optimizer's solves. Ignored when off."""
        if self.multi_seed <= 0:
            kwargs.pop("multi_seed_index", None)
            return self._solve_standard(reference, q_now, scene, **kwargs)
        index = kwargs.pop("multi_seed_index", None)
        run = self._ms_depth == 0 and (index is not None or not self._ms_is_window)
        if run:
            if index is None:
                index = self._ms_count
            self._ms_count += 1
        saved_child = self._ms_child_index
        # a nested solve keeps the index (E1's `_solve_plain_recorded` nests the standard solve itself); the IFG-TOA3
        # safety-first re-solve clears it (`_solve_standard`), so a window optimizer runs its seeds once per chunk
        self._ms_child_index = int(index) if run else saved_child
        self._ms_depth += 1
        try:
            result = self._solve_standard(reference, q_now, scene, **kwargs)
            if run:
                result = self._multi_seed(result, int(index), reference, q_now, scene, kwargs)
            return result
        finally:
            self._ms_depth -= 1
            self._ms_child_index = saved_child

    def _solve_standard(self, reference, q_now, scene, **kwargs) -> TrajOptResult:
        """See `_solve`. With `sqp.fast_eval` (T43 TP) the linearizer's per-solve memo lives exactly
        as long as this call; off, this is `_solve` itself."""
        if getattr(self.config.collision, "esdf_host_mirror", False):
            # T43 TP: one host copy of the GPU field's grids per chunk (cached on the field object).
            mirror = getattr(getattr(scene, "esdf", None), "mirror_to_host", None)
            if mirror is not None:
                mirror()
        anchor = kwargs.get("grasp_anchor")
        self._anchor_state = self._anchor_setup(anchor, reference, q_now)
        if (anchor is not None and self._anchor_state is None
                and getattr(anchor, "variant", "TOA") != "TOA"):
            # E1-B / E1-C: an anchor was given but nothing applies on this horizon (no goal rows) — the solve is
            # the plain one; say so in the record (IFG-TOA keeps its earlier record: none).
            return self._solve_plain_recorded(anchor, reference, q_now, scene, **kwargs)
        scale = kwargs.pop("joint_track_scale", None)
        self._track_scale = None if scale is None else np.asarray(scale, np.float64).reshape(-1)   # IFG-TOW
        if scale is not None and self.window_optimizer is not None:
            kwargs["window_track_scale"] = self._track_scale
        try:
            if not self.fast_eval:
                result = self._solve(reference, q_now, scene, **kwargs)
            else:
                self.linearizer.begin_memo()
                try:
                    result = self._solve(reference, q_now, scene, **kwargs)
                finally:
                    self.linearizer.end_memo()
            st = self._anchor_state
            if st is not None and np.asarray(result.trajectory).shape[1] >= st["horizon"]:
                from AG3S.trajopt.grasp_anchor import anchor_record

                p_out, R_out = self._anchor_pose(st, result.trajectory)
                rec = anchor_record(anchor, st["p_ref"], p_out, st.get("R_ref"), R_out)
                tol = float(self.config.safety.violation_tolerance)
                inner_rec = result.metrics.get("grasp_anchor")
                inner = (inner_rec or {}).get("fallback")
                if getattr(anchor, "goal_window", "close") != "close":
                    # AG3S-PRE-CORE-TO-GGW: the rows actually optimised — the window solve's (closed look-ahead gate:
                    # only the execution window was solved, with its own goal rows) or this horizon's.
                    rec.update(self._applied_goal_rows(rec, inner_rec, result, st["horizon"]))
                if anchor.variant != "TOA":
                    # E1: whether the safety-first re-solve below ran and won (IFG-TOA keeps its earlier record).
                    rec["fallback_used"] = bool(inner is not None and inner.get("used") == "plain")
                if inner is None and float(result.max_violation) > tol:
                    # IFG-TOA safety first: the anchored solve left a violation (IFG-APPLE-TOA2 ep211: the anchor priced
                    # the translation the obstacle needs, every candidate was rejected, HOLD for ~50 chunks). Solve
                    # again without the anchor and keep the result with the smaller violation (the anchored one on a tie).
                    kw = {k: v for k, v in kwargs.items() if k != "grasp_anchor"}
                    self._anchor_state = None
                    child, self._ms_child_index = self._ms_child_index, None   # E2: no seeds inside the re-solve
                    try:
                        plain = self.solve(reference, q_now, scene, **kw)
                    finally:
                        self._ms_child_index = child
                    use_plain = float(plain.max_violation) < float(result.max_violation)
                    fb = {"anchored_violation_mm": round(float(result.max_violation) * 1000.0, 2),
                          "plain_violation_mm": round(float(plain.max_violation) * 1000.0, 2),
                          "used": "plain" if use_plain else "anchored"}
                    rec["fallback"] = fb
                    if use_plain:
                        p_plain, R_plain = self._anchor_pose(st, plain.trajectory)
                        rec = anchor_record(anchor, st["p_ref"], p_plain, st.get("R_ref"), R_plain)
                        rec["fallback"] = fb
                        if getattr(anchor, "goal_window", "close") != "close":
                            rec.update(applied_horizon=int(st["horizon"]), goal_rows_applied=[])   # GGW: plain won
                        if anchor.variant != "TOA":
                            rec["fallback_used"] = True
                        plain.metrics["grasp_anchor"] = rec
                        return plain
                elif inner is not None:
                    rec["fallback"] = inner
                result.metrics["grasp_anchor"] = rec
            return result
        finally:
            self._anchor_state = None

    @staticmethod
    def _anchor_pose(st: dict[str, Any], trajectory) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Hold-link positions (and rotations for a 6-D goal, AG3S-GEO-GRASP) of `trajectory`'s first rows."""
        traj = np.asarray(trajectory, np.float64)[:, :st["horizon"]]
        if st.get("R_ref") is not None:
            p, _, R, _ = st["fk"].pose(traj, st["q_now"])
            return p, R
        return st["fk"](traj, st["q_now"])[0], None

    def _anchor_setup(self, anchor, reference, q_now) -> Optional[dict[str, Any]]:
        """IFG-TOA — this solve's anchor state (palm FK, per-row weights and targets), or `None` (no anchor given, or
        nothing switched on). The FK is built once per link.

        AG3S-PRE-CORE-TO E1-B: with `anchor.goal_box > 0` the state also carries `box` (goal rows, goal, half-width,
        slack price `w_slack * GOAL_BOX_SLACK_SCALE`); `cost_on` says whether any palm cost term is on."""
        if anchor is None:
            return None
        reference = np.asarray(reference, np.float64)
        H = int(reference.shape[1])
        box_r = float(getattr(anchor, "goal_box", 0.0) or 0.0)
        if H != self.horizon or not (anchor.w_tcp > 0.0 or anchor.w_goal > 0.0 or box_r > 0.0):
            return None
        from AG3S.trajopt.grasp_anchor import GOAL_BOX_SLACK_SCALE, LinkPointFK, LinkPoseFK, anchor_targets

        cache = self.__dict__.setdefault("_anchor_fk", {})
        six_d = getattr(anchor, "goal_R", None) is not None
        key = (anchor.link, "pose") if six_d else anchor.link
        fk = cache.get(key)
        if fk is None:
            fk = cache[key] = (LinkPoseFK if six_d else LinkPointFK)(self.linearizer.robot_model, self.layout, H,
                                                                     anchor.link)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        R_ref = None
        if six_d:
            p_ref, _, R_ref, _ = fk.pose(reference, q_now)
        else:
            p_ref = fk(reference, q_now)[0]
        weights, targets = anchor_targets(anchor, p_ref)
        cost_on = bool(np.any(weights > 0.0))
        rot = None
        if six_d and float(anchor.w_rot) > 0.0:
            rows = np.flatnonzero(anchor.goal_mask(H))
            if rows.size:
                rot = {"rows": rows, "Rg": np.asarray(anchor.goal_R, np.float64), "w": float(anchor.w_rot)}
        box = None
        if box_r > 0.0:
            rows = np.flatnonzero(anchor.goal_mask(H))
            if rows.size:
                box = {"rows": rows, "goal": np.asarray(anchor.goal, np.float64).reshape(3), "r": box_r,
                       "weight": float(self.config.cost.w_slack) * GOAL_BOX_SLACK_SCALE}
        if not cost_on and box is None and rot is None:
            return None
        return {"fk": fk, "q_now": q_now, "p_ref": p_ref, "weights": weights, "targets": targets, "horizon": H,
                "cost_on": cost_on, "box": box, "rot": rot, "R_ref": R_ref}

    @staticmethod
    def _applied_goal_rows(rec, inner_rec, result, horizon: int) -> dict[str, Any]:
        """AG3S-PRE-CORE-TO-GGW — `{applied_horizon, goal_rows_applied[, window]}` for the anchor record: with a closed
        look-ahead gate the window optimizer solved the first rows only, with its own anchor record (`window`)."""
        gate = (result.metrics.get("lookahead") or {}).get("gate")
        if gate == "closed" and isinstance(inner_rec, dict):
            applied = inner_rec.get("goal_rows_applied", inner_rec.get("goal_rows"))   # [] when its plain re-solve won
            return {"applied_horizon": inner_rec.get("horizon"), "goal_rows_applied": list(applied or []),
                    "window": dict(inner_rec)}
        return {"applied_horizon": int(horizon), "goal_rows_applied": list(rec.get("goal_rows") or [])}

    def _solve_plain_recorded(self, anchor, reference, q_now, scene, **kwargs) -> TrajOptResult:
        """E1 — an anchor with nothing to apply on this horizon: the plain solve plus a record saying so."""
        kw = {k: v for k, v in kwargs.items() if k != "grasp_anchor"}
        result = self.solve(reference, q_now, scene, **kw)
        H = int(np.asarray(reference).shape[1])
        result.metrics["grasp_anchor"] = {
            "variant": anchor.variant, "link": anchor.link, "close_row": anchor.close_row, "n_goal_rows": 0,
            "goal_rows": [], "applied": False,
            "reason": ("no goal" if anchor.goal is None else "no close planned" if anchor.close_row is None
                       else "starts closed (close row 0)" if int(anchor.close_row) <= 0
                       else "close row beyond this horizon" if int(anchor.close_row) >= H else "no goal rows"),
            **({"goal_box_mm": round(float(anchor.goal_box) * 1000.0, 3)} if anchor.goal_box > 0.0 else {}),
            "fallback_used": False}
        if getattr(anchor, "goal_window", "close") != "close":   # AG3S-PRE-CORE-TO-GGW (close: the record as before)
            result.metrics["grasp_anchor"].update(goal_window=anchor.goal_window,
                                                  goal_window_rule=anchor.goal_window_rule(H), horizon=H,
                                                  applied_horizon=H, goal_rows_applied=[])
        return result

    # --- AG3S-PRE-CORE-TO E2: multi-seed -------------------------------------------------
    def _multi_seed(self, result: TrajOptResult, index: int, reference, q_now, scene,
                    kwargs: dict[str, Any]) -> TrajOptResult:
        """E2 — re-solve from `sqp.multi_seed` other first iterates when the standard `result` leaves a violation, and
        return the best of the standard and the seeds (`multi_seed.select`). Records `metrics["multi_seed"]`.

        Not triggered (the standard result returned as it is, plus `{"triggered": False}`): violation within
        `safety.violation_tolerance`, nothing to avoid, or a closed look-ahead gate (the window optimizer's solve is
        the standard one and runs its own multi-seed). The seeds use their own QP solver object, so the standard
        solver's factorization and warm start carry to the next chunk as without E2; `_previous` and `frame_index`
        end as if only the chosen trajectory had been solved."""
        if "multi_seed" in result.metrics:          # the window optimizer's (gate closed)
            return result
        tol = float(self.config.safety.violation_tolerance)
        n = int(self.multi_seed)
        record: dict[str, Any] = {"triggered": False, "n_seeds": n, "index": int(index),
                                  "standard_violation_mm": round(float(result.max_violation) * 1000.0, 3),
                                  "tolerance_mm": round(tol * 1000.0, 3)}
        gate = (result.metrics.get("lookahead") or {}).get("gate")
        if gate == "closed":
            record["reason"] = "look-ahead gate closed: the window solve is the standard one"
        elif result.status == TrajOptStatus.UNCONSTRAINED:
            record["reason"] = "nothing to avoid"
        elif float(result.max_violation) <= tol:
            record["reason"] = "standard solve within tolerance"
        else:
            return self._run_seeds(result, index, reference, q_now, scene, kwargs, record)
        result.metrics["multi_seed"] = record
        return result

    def _run_seeds(self, result, index, reference, q_now, scene, kwargs, record) -> TrajOptResult:
        from AG3S.trajopt import multi_seed as MS

        started = time.perf_counter()
        reference = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        q0 = q_now[self.layout.q_indices]
        H = int(reference.shape[1])
        tol = float(self.config.safety.violation_tolerance)
        anchor = kwargs.get("grasp_anchor")
        scale = kwargs.get("joint_track_scale")
        kw = {k: v for k, v in kwargs.items() if k not in ("grasp_anchor", "joint_track_scale")}
        envelope = (rest_start_envelope(self.limits, self.horizon,
                                        getattr(self.config.limits, "rest_start_tolerance", None))
                    if kw.get("start_at_rest") else None)
        geometry = self._seed_geometry(result, reference, q_now, scene, anchor)
        seeds = self._build_seeds(MS.seed_specs(int(self.multi_seed)), index, reference, q0, geometry)
        goal = self._goal_probe(anchor, H)
        record.update(triggered=True, geometry=geometry["record"],
                      goal_rows=None if goal is None else [int(k) for k in goal["rows"]])

        limit_bound = [MS.LIMIT_SLACK]

        def describe(kind, detail, res, ms, is_seed: bool):
            traj = np.asarray(res.trajectory, np.float64)
            g = None if goal is None else self._goal_error_mm(goal, traj, q_now)
            length, acc = MS.path_length(traj, q0), MS.accel_sq(traj, q0)
            over = float(max(self._limit_overshoot(traj, q_now, envelope).values()))
            if not is_seed:
                limit_bound[0] = max(MS.LIMIT_SLACK, over)
            # a seed may not end further outside the joint limits than the standard solve (or LIMIT_SLACK)
            eligible = bool(not is_seed or (over <= limit_bound[0] and res.status != TrajOptStatus.SOLVER_FAILED))
            rec = {"kind": kind, **({"detail": detail} if detail else {}),
                   "max_violation_mm": round(float(res.max_violation) * 1000.0, 3),
                   "goal_err_mm": None if g is None else round(g, 2),
                   "status": res.status.value, "eligible": eligible,
                   "path_length_rad": round(length, 5), "accel_sq": round(acc, 7),
                   "limit_overshoot": round(over, 6), "cost": round(float(res.cost), 6),
                   "reference_deviation": round(float(res.reference_deviation), 6),
                   "solve_ms": round(float(ms), 2), "sqp_iterations": int(res.metrics.get("sqp_iterations", 0) or 0)}
            key = MS.selection_key(res.max_violation, tol, g, length, acc) if eligible else None
            return rec, key

        std_rec, std_key = describe("standard", None, result, float(result.solve_time_ms), False)
        records, keys, results = [std_rec], [std_key], [result]
        frame_after_standard = self.frame_index
        main_solver = self.solver
        seed_solver = self.__dict__.get("_ms_solver")
        if seed_solver is None:
            seed_solver = self._ms_solver = QpSolver(self.config.qp)
        self.solver = seed_solver
        gpu = None
        try:
            if self.gpu_seeds != "off":
                # AG3S-PRE-CORE-TO E3-b: the same seeds in one batched cuRobo L-BFGS (`gpu_seeds.py`), each result
                # re-checked by `_finish` (cost with this chunk's anchor / IFG-TOW scale, as a seed solve has it).
                from AG3S.trajopt import gpu_seeds as GS

                self._anchor_state = self._anchor_setup(anchor, reference, q_now) if anchor is not None else None
                self._track_scale = None if scale is None else np.asarray(scale, np.float64).reshape(-1)
                first_pass = None
                if self.gpu_seeds_check == "first-pass":
                    # `select_moving_first`'s order read without the check: (goal error, reference deviation, index)
                    # with `describe`'s rounding, over the moving results inside the joint-limit bound. The first
                    # checked one that resolves is the one the full check would choose.
                    def order_key(i, Q):
                        if seeds[i][0] == "stay":
                            return None
                        traj = np.asarray(Q, np.float64)
                        if float(max(self._limit_overshoot(traj, q_now, envelope).values())) > limit_bound[0]:
                            return None
                        g = None if goal is None else round(self._goal_error_mm(goal, traj, q_now), 2)
                        return (0 if g is None else int(round(float(g) * 10.0)),
                                round(float(np.linalg.norm(traj - reference)), 6), i + 1)

                    def resolves(res):
                        return (res.status != TrajOptStatus.SOLVER_FAILED
                                and round(float(res.max_violation) * 1000.0, 3) <= tol * 1000.0)

                    first_pass = (order_key, resolves)
                try:
                    gpu = GS.solve_seeds(self, seeds, reference, q_now, scene, kw, anchor, scale, envelope, result,
                                         goal, first_pass=first_pass)
                finally:
                    self._anchor_state = None
                record.update(backend=gpu["backend"], gpu=gpu["record"])
                if getattr(anchor, "goal_R", None) is not None:
                    # AG3S-GEO-GRASP: said, not silently dropped — the batched cuRobo cost has no rotation term
                    record["gpu_goal_note"] = ("6-D goal: the cuRobo seed cost carries the position term only (rotation "
                                               "not supported there); every seed is re-checked with the rotation in the "
                                               "CPU merit")
                for (kind, detail, _), item in zip(seeds, gpu["results"] or ()):
                    if item is None:            # first-pass: not checked (a resolving result came first)
                        records.append({"kind": kind, **({"detail": detail} if detail else {}), "checked": False,
                                        "max_violation_mm": None, "goal_err_mm": None, "status": None,
                                        "eligible": False, "reference_deviation": None})
                        keys.append(None)
                        results.append(None)
                        continue
                    res, ms = item
                    rec, key = describe(kind, detail, res, ms, True)
                    records.append(rec)
                    keys.append(key)
                    results.append(res)
            # E2's CPU seeds (always without E3-b; with it only when the GPU path failed — `record["gpu"]["error"]`)
            for kind, detail, Q0 in (seeds if gpu is None or gpu["results"] is None else ()):
                t0 = time.perf_counter()
                self._anchor_state = self._anchor_setup(anchor, reference, q_now) if anchor is not None else None
                self._track_scale = None if scale is None else np.asarray(scale, np.float64).reshape(-1)
                try:
                    if self.fast_eval:
                        self.linearizer.begin_memo()
                    try:
                        res = self._solve(reference, q_now, scene, initial_iterate=Q0, **kw)
                    finally:
                        if self.fast_eval:
                            self.linearizer.end_memo()
                finally:
                    self._anchor_state = None
                rec, key = describe(kind, detail, res, (time.perf_counter() - t0) * 1000.0, True)
                records.append(rec)
                keys.append(key)
                results.append(res)
        finally:
            self.solver = main_solver
        chosen = MS.select_moving_first([r["kind"] for r in records], keys,
                                        [r["max_violation_mm"] for r in records], tol * 1000.0,
                                        [r["goal_err_mm"] for r in records],
                                        [r["reference_deviation"] for r in records])
        total_ms = (time.perf_counter() - started) * 1000.0
        out = results[chosen]
        metrics = dict(out.metrics)
        if chosen > 0:
            if anchor is not None and "grasp_anchor" in result.metrics:
                # the anchor record of the returned trajectory (the standard's record described the standard one)
                metrics["grasp_anchor"] = self._seed_anchor_record(anchor, reference, q_now, out.trajectory,
                                                                   result.metrics["grasp_anchor"])
            notes = list(out.notes) + [
                f"multi-seed (E2): seed {chosen} ({records[chosen]['kind']}) of {len(seeds)} replaced the standard "
                f"solve ({record['standard_violation_mm']:.2f} mm -> {records[chosen]['max_violation_mm']:.2f} mm)"]
        else:
            notes = list(out.notes)
        self.frame_index = frame_after_standard
        metrics["frame_index"] = frame_after_standard
        self._previous = np.asarray(out.trajectory, np.float64)
        record.update(candidates=records, chosen=int(chosen), chosen_kind=records[chosen]["kind"],
                      chosen_violation_mm=records[chosen]["max_violation_mm"],
                      chosen_goal_err_mm=records[chosen]["goal_err_mm"],
                      n_resolved=int(sum(1 for r in records if r["eligible"]
                                         and r["max_violation_mm"] <= tol * 1000.0)),
                      standard_ms=round(float(result.solve_time_ms), 2), total_ms=round(total_ms, 2))
        metrics["multi_seed"] = record
        return dataclasses.replace(out, metrics=metrics, notes=notes,
                                   solve_time_ms=float(result.solve_time_ms) + total_ms)

    def _seed_geometry(self, result, reference, q_now, scene, anchor) -> dict[str, Any]:
        """E2 — the deepest violating sphere of the standard result, the direction away from what it violates, its
        perpendicular frame, and the palm (hold link) to push: `{"away", "e1", "e2", "peak", "palm", "joints",
        "palm_jac", "record"}`; `away` is `None` when no direction can be read (the detour slots become noise)."""
        from AG3S.trajopt import multi_seed as MS
        from AG3S.trajopt.grasp_anchor import LinkPointFK

        H = int(reference.shape[1])
        pair = result.metrics.get("max_violation_pair") or {}
        out: dict[str, Any] = {"away": None, "e1": None, "e2": None, "peak": H // 2, "palm": None,
                               "joints": np.arange(self.layout.nq_opt), "palm_jac": None}
        rec: dict[str, Any] = {"block": pair.get("block"), "link": pair.get("link"), "step": pair.get("step")}
        traj = np.asarray(result.trajectory, np.float64)[:, :H]
        sphere_cols = None
        away = None
        try:
            if pair and pair.get("point_m") is not None:
                point = np.asarray(pair["point_m"], np.float64)
                query, step = int(pair["query"]), int(pair["step"])
                out["peak"] = step
                states = self.linearizer.sphere_states(traj, q_now)
                sphere_cols = np.flatnonzero(np.abs(np.asarray(states[1][step, query])).sum(axis=0) > 1e-12)
                if pair.get("block") == "esdf" and getattr(scene, "esdf", None) is not None:
                    away = self.linearizer._esdf_directions(point.reshape(1, 3), np.array([query]), scene,
                                                            int(states[0].shape[1]))[0]
                elif pair.get("block") == "candidate" and pair.get("slot") is not None:
                    away = point - np.asarray(scene.candidate_pos, np.float64)[int(pair["slot"])]
                elif pair.get("block") == "plane" and pair.get("slot") is not None:
                    away = np.asarray(scene.plane_normal, np.float64)[int(pair["slot"])]
        except Exception as exc:  # noqa: BLE001 - a seed direction is a heuristic; the noise seeds remain
            rec["direction_error"] = f"{type(exc).__name__}: {exc}"
            away = None
        if away is not None and np.all(np.isfinite(away)) and float(np.linalg.norm(away)) > 1e-9:
            away = np.asarray(away, np.float64) / float(np.linalg.norm(away))
        else:
            away = None
        # the palm: the hold link whose chain shares the most joints with the violating sphere's (the anchor's first)
        names = ([anchor.link] if anchor is not None else []) + [l for l in MS.PALM_LINKS
                                                                  if anchor is None or l != anchor.link]
        cache = self.__dict__.setdefault("_anchor_fk", {})
        best = None
        for name in names:
            try:
                fk = cache.get(name)
                if fk is None:
                    fk = cache[name] = LinkPointFK(self.linearizer.robot_model, self.layout, H, name)
                if fk.horizon != H:
                    continue
                _, J = fk(reference, q_now)
            except Exception:  # noqa: BLE001 - a model without this link
                continue
            cols = np.flatnonzero(np.abs(J).sum(axis=(0, 1)) > 1e-12)
            score = 0 if sphere_cols is None else int(np.intersect1d(cols, sphere_cols).size)
            if best is None or score > best[0]:
                best = (score, name, cols, J)
        if best is not None and sphere_cols is not None and best[0] > 0:
            out.update(palm=best[1], joints=best[2], palm_jac=best[3])
        if away is not None and out["palm_jac"] is not None:
            e1, e2 = MS.perpendicular_frame(away)
            out.update(away=away, e1=e1, e2=e2)
        rec.update(palm=out["palm"], peak_step=int(out["peak"]), n_joints=int(np.asarray(out["joints"]).size),
                   away=None if out["away"] is None else [round(float(v), 4) for v in out["away"]],
                   e1=None if out["e1"] is None else [round(float(v), 4) for v in out["e1"]],
                   e2=None if out["e2"] is None else [round(float(v), 4) for v in out["e2"]])
        out["record"] = rec
        return out

    def _build_seeds(self, specs, index, reference, q0, geometry) -> list[tuple[str, dict, np.ndarray]]:
        """E2 — `[(kind, detail, first iterate)]` for `specs`, clipped to the position limits."""
        from AG3S.trajopt import multi_seed as MS

        H = int(reference.shape[1])
        lo = np.asarray(self.limits.lower, np.float64)[:, None]
        hi = np.asarray(self.limits.upper, np.float64)[:, None]
        joints = np.asarray(geometry["joints"], int)
        profile = MS.bump(H, int(geometry["peak"]))
        away = geometry["away"]
        out = []
        for slot, spec in enumerate(specs):
            detail: dict[str, Any] = {}
            kind = spec.kind
            if kind == "policy":
                Q = reference.copy()
            elif kind == "stay":
                Q = np.tile(np.asarray(q0, np.float64)[:, None], (1, H))
            elif kind == "detour" and away is not None:
                if spec.weights is None:
                    d = MS.random_direction(away, MS.chunk_rng(index, MS.RANDOM_DIRECTION_STREAM + spec.draw))
                else:
                    w = spec.weights
                    d = w[0] * away + w[1] * geometry["e1"] + w[2] * geometry["e2"]
                d = d / max(float(np.linalg.norm(d)), 1e-12)
                Q = MS.detour_seed(reference, geometry["palm_jac"], d, spec.magnitude, profile, joints)
                detail = {"direction": spec.direction, "cm": round(spec.magnitude * 100.0, 1),
                          "d": [round(float(v), 4) for v in d]}
            else:
                # noise, or a detour slot without a direction (the noise of the same slot instead)
                sigma = spec.sigma if kind == "noise" else MS.NOISE_SIGMAS[slot % len(MS.NOISE_SIGMAS)]
                if kind != "noise":
                    detail["no_direction"] = spec.direction
                kind = "noise"
                Q = MS.noise_seed(reference, sigma, MS.chunk_rng(index, slot), joints)
                detail["sigma_rad"] = sigma
            out.append((kind, detail, np.clip(Q, lo, hi)))
        return out

    def _goal_probe(self, anchor, H: int) -> Optional[dict[str, Any]]:
        """E2 — the frozen grasp goal and its rows on this horizon, or `None` (no anchor, no goal, no goal rows)."""
        if anchor is None or getattr(anchor, "goal", None) is None:
            return None
        rows = np.flatnonzero(anchor.goal_mask(H))
        if not rows.size:
            return None
        from AG3S.trajopt.grasp_anchor import LinkPointFK

        cache = self.__dict__.setdefault("_anchor_fk", {})
        fk = cache.get(anchor.link)
        if fk is None:
            fk = cache[anchor.link] = LinkPointFK(self.linearizer.robot_model, self.layout, H, anchor.link)
        return {"rows": rows, "goal": np.asarray(anchor.goal, np.float64).reshape(1, 3), "fk": fk}

    @staticmethod
    def _goal_error_mm(goal: dict[str, Any], trajectory: np.ndarray, q_now: np.ndarray) -> float:
        p = goal["fk"](np.asarray(trajectory, np.float64)[:, :goal["fk"].horizon], q_now)[0]
        return float(np.max(np.linalg.norm(p[goal["rows"]] - goal["goal"], axis=1)) * 1000.0)

    def _seed_anchor_record(self, anchor, reference, q_now, trajectory, standard_record) -> dict[str, Any]:
        from AG3S.trajopt.grasp_anchor import anchor_record

        st = self._anchor_setup(anchor, reference, q_now)
        if st is None:
            return {**standard_record, "multi_seed": "the standard solve's record (no anchor terms on the seeds)"}
        p_out, R_out = self._anchor_pose(st, trajectory)
        rec = anchor_record(anchor, st["p_ref"], p_out, st.get("R_ref"), R_out)
        if anchor.variant != "TOA":
            rec["fallback_used"] = False
        rec["multi_seed"] = "seed result (no safety-first re-solve on seeds)"
        if getattr(anchor, "goal_window", "close") != "close":   # AG3S-PRE-CORE-TO-GGW: the seeds carry this horizon's rows
            rec.update(applied_horizon=int(st["horizon"]), goal_rows_applied=list(rec.get("goal_rows") or []))
        return rec

    def _solve(
        self,
        reference: np.ndarray,
        q_now: np.ndarray,
        scene: Optional[SceneSnapshot],
        *,
        previous_chunk: Optional[np.ndarray] = None,
        template: Optional[np.ndarray] = None,
        geometry_certified: bool = True,
        start_at_rest: bool = False,
        servo_feedback: Optional[dict[str, Any]] = None,
        deflection_history: Optional[np.ndarray] = None,
        command_history: Optional[np.ndarray] = None,
        grasp_anchor: Optional[Any] = None,
        window_track_scale: Optional[np.ndarray] = None,
        initial_iterate: Optional[np.ndarray] = None,
    ) -> TrajOptResult:
        """Make `reference` safe against `scene`, or report how far short it fell.

        Args:
            reference: ``(nq_opt, H)`` — SEAM's chunk in joint space.
            q_now: ``(nq_model,)`` the robot's current configuration. Positions the fixed joints and
                anchors the first step.
            scene: AG3S's decoded constraint set, or None when there is nothing to avoid.
            previous_chunk: the overlapping tail of the last refined chunk, for continuity.
            template: the original ``(H, action_dim)`` chunk, so grippers and padding pass through
                untouched. Defaults to zeros, which is only right when the caller has no chunk.
            geometry_certified: AG3S's own verdict on the scene. A frame it could not certify may
                still be optimized against; the result simply cannot be called safe.
            start_at_rest: T41 b. `True` = the robot is known to be at rest at `q_now` (the previous
                chunk was held throughout). The step-0 anchor becomes the reach-from-rest envelope
                (`limits.rest_start_envelope`, `config.limits.rest_start_tolerance`) on every step —
                in the QP box, in the eligibility of the initial iterate and in the limit projection.
                `False` (default) leaves every number as it was. Who decides: `SafePolicy`, only when
                `limits.rest_start` is on.
            servo_feedback: T43 Q. ``{"feedback": exec_feedback, "seq": request seq}`` — what the
                servo observer propagates to estimate the arm's velocity at `q_now`
                (`servo.ServoObserver`). Read only when `limits.servo_model` is on; without it the
                arm is taken to be at rest and the record says so.
            deflection_history: T43 TA (B). ``(nq_opt,)`` deflection of the last executed command
                (`cost.w_deflection_rate`); `None` = no boundary term.
            command_history: T43 TA (C). ``(nq_opt, 2)`` the two last executed commands, oldest
                first (`limits.servo_accel`); `None` = the C rows stay inside the chunk.
            initial_iterate: AG3S-PRE-CORE-TO E2. ``(nq_opt, H)`` a multi-seed first iterate
                (`_multi_seed`): it replaces the reference / warm-start blend as the first iterate, the
                stay start is not tried, and the servo observer is not stepped again (the standard
                solve of this chunk already did). The cost still tracks `reference`. `None` (default)
                = as before.
        """
        started = time.perf_counter()
        cfg = self.config
        reference = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        notes: list[str] = []
        #: T43 TP — the continuity reference before the look-ahead cut (`warm_start_lookahead`).
        continuity_full = previous_chunk
        if self.lookahead_from is not None and previous_chunk is not None:
            # T43 TA (A): continuity only over the executed window, as without look-ahead. Pulling
            # the tail toward the previous chunk's stale prediction moved chunks that collide with
            # nothing (offline V8: 8 of 35 closing-set chunks the base left untouched).
            previous_chunk = np.asarray(previous_chunk, np.float64)[:, :self.lookahead_from]
        # T43 TA — only what is switched on is kept; with B and C off this stays empty.
        self._history = {}
        if deflection_history is not None and float(cfg.cost.w_deflection_rate) > 0.0:
            self._history["deflection"] = np.asarray(deflection_history, np.float64).reshape(-1)
        if command_history is not None and self.servo_accel_bound is not None:
            self._history["command"] = np.asarray(command_history, np.float64).reshape(
                reference.shape[0], -1)
        #: T41 b — `None` 이면 예전 anchor (첫 스텝 `max_step`) 그대로.
        envelope = (rest_start_envelope(self.limits, self.horizon,
                                        getattr(cfg.limits, "rest_start_tolerance", None))
                    if start_at_rest else None)

        servo_state = None
        if self.observer is not None:
            if initial_iterate is not None and self.__dict__.get("_ms_servo_state") is not None:
                # E2: a seed of this chunk — the observer was stepped by the standard solve (not idempotent).
                servo_state = self._ms_servo_state
            else:
                fb = (servo_feedback or {}).get("feedback")
                seq = (servo_feedback or {}).get("seq")
                servo_state = self.observer.update(q_now[self.layout.q_indices], fb, seq)
                if self.multi_seed > 0:
                    self._ms_servo_state = servo_state

        if scene is None or scene.is_empty:
            return self._passthrough(
                reference, q_now, template, TrajOptStatus.UNCONSTRAINED,
                notes + ["AG3S emitted no geometry to avoid; the reference passes through unchanged"],
                started,
            )

        self._gate_record = None
        if self.window_optimizer is not None:
            gate = self._lookahead_gate(reference, q_now, scene)
            self._gate_record = {x: v for x, v in gate.items() if x != "open"}
            if not gate["open"]:
                return self._solve_window(
                    gate, reference, q_now, scene, previous_chunk=previous_chunk,
                    template=template, geometry_certified=geometry_certified,
                    start_at_rest=start_at_rest, servo_feedback=servo_feedback,
                    grasp_anchor=grasp_anchor, window_track_scale=window_track_scale,
                    deflection_history=deflection_history, command_history=command_history)

        n_path = self.path.n_slack if self.path is not None else 0
        n_slack = self.block.n_rows + n_path
        seeded = initial_iterate is not None   # E2: a multi-seed first iterate (False = as before)
        iterate = reference.copy() if not seeded else np.asarray(initial_iterate, np.float64).copy()
        warm_lookahead = (self.lookahead_from is not None
                          and getattr(cfg.sqp, "warm_start_lookahead", False)
                          and self._previous is not None
                          and self._seed_is_aligned(continuity_full))
        if (not seeded and cfg.sqp.warm_start and self._previous is not None
                and cfg.horizon.execution_length < self.horizon
                and (self.lookahead_from is None or warm_lookahead)):
            # Seed from the previous chunk shifted by the steps that have executed since. The tail is
            # held rather than extrapolated: extrapolating a trajectory past its horizon invents
            # motion the policy never proposed.
            #
            # **Skipped when the whole planned window has executed** (`execution_length >= horizon`,
            # which is what `PLAN_EXECUTION_WINDOW` produces): there is then no overlap between the
            # last plan and this one, and the shift would blend in `self._previous[:, -1]` held
            # constant — a pose from before the executed window, pulling the first iterate backwards.
            # With no overlap the reference *is* the best available start.
            k = min(cfg.horizon.execution_length, self.horizon - 1)
            shifted = np.concatenate(
                [self._previous[:, k:], np.tile(self._previous[:, -1:], (1, k))], axis=1
            )
            iterate = 0.5 * (iterate + shifted)

        # 쥔 물체를 질의점으로 올린다. 매 청크 다시 거는 것은 파지가 청크 사이에 시작되고 끝나기
        # 때문이고, 값이 같으면 하는 일이 없다 — 심볼릭 그래프는 링크마다 한 번만 만들어진다.
        self.linearizer.set_attached(
            getattr(scene, "attached_points", None), getattr(scene, "attached_parent_link", None)
        )
        if self.lookahead_from is not None:
            # T43 TA (A): what the reference does in the executed window — the certified span.
            reference_violation = -float(self.linearizer.step_clearances(
                reference, q_now, scene)[:self.lookahead_from].min(initial=np.inf))
        else:
            reference_violation = -self.linearizer.full_violation(reference, q_now, scene)
        if self.path is not None:
            # T43 Q — floors at `q_now` and the servo state, once per chunk (before any merit).
            self.path.begin(q_now, scene, servo_state)
        # Forward kinematics for a whole chunk is the most expensive thing left in the loop, so each
        # trajectory's sphere states are computed once and carried: the iterate's states feed both
        # the linearization and its merit, and an accepted candidate hands its states to the next
        # iteration rather than being recomputed there.
        iterate_states = self.linearizer.sphere_states(iterate, q_now)
        iterate_merit, iterate_violation = self._merit_terms(
            iterate, reference, previous_chunk, q_now, scene, states=iterate_states
        )
        # **초기 iterate 도 반환 후보다** (T24, 지침 §7.3). 예전에는 `best_merit = np.inf` 로
        # 시작해서, 첫 QP 후보가 초기 궤적보다 merit 이 나빠 반복점으로 **거절**되더라도 반환용
        # best 로는 저장됐다. 예산으로 끝나면 (T14 30/30 · T17 75/75 chunk 가 그랬다) 그 거절된
        # 후보가 나갔다. 이제 초기 iterate 와 그 merit 이 기준이고, 그보다 나쁜 후보는 반환되지
        # 않는다.
        #
        # **단, 초기 iterate 가 로봇 한계 안에 있을 때만이다.** merit 은 비용 + 충돌 위반이고
        # **관절 한계를 보지 않는다** — 한계는 QP 의 행(box·anchor 는 hard, 속도·가속도는 soft)이
        # 지킨다. QP 후보는 그 행을 거쳐 나오지만 초기 iterate 는 정책이 낸 그대로다. 그래서
        # reference 가 joint box 밖이면 box 안으로 끌어들인 후보가 추적 비용만큼 merit 이 나빠
        # **언제나 진다** — merit 만으로 고르면 한계를 넘는 reference 가 그대로 나간다. 두 후보의
        # 적격성이 같지 않은 것이다 (지침 §7.3 "모든 후보의 적격성이 같다고 가정하지 않는다").
        # 한계를 넘는 초기 iterate 는 비교 대상에서 빠지고 (예전 규칙 그대로), QP 가 하나도 못
        # 풀었을 때의 마지막 대비로만 남는다. **T31**: 그 초기 iterate 가 limit **만** 어겼고
        # 받아들여진 후보가 없으면, 거절된 후보 대신 reference 의 최소 투영이 나간다
        # (`_limit_projection`, 루프 뒤).
        #
        # 적격한 초기화 아래에서 best 는 언제나 수락된 반복점이다: 후보의 merit 이 `best_merit`
        # 보다 작으면 `best_merit <= iterate_merit` 이므로 수락 조건도 만족한다. 그래서
        # `best_unaccepted` 는 **초기 iterate 가 부적격일 때만** 나온다. `returned` 는 이 성질을
        # 가정하지 않고 후보 기록에서 읽어 싣는다.
        start_record = None
        if self.path is not None and seeded:
            start_record = {"source": "multi-seed", "stay_merit": None, "stay_violation_m": None}
        elif self.path is not None:
            # T43 Q — **a second starting point: stay where the robot is.** With the path rows a
            # reference that runs through an obstacle is a poor place to linearize: the waypoint rows
            # push the commands across the obstacle while the lagging path rows pull the robot back,
            # and three SQP iterations from there end VIOLATED (T40 C1807 t=216, margin 10 mm). The
            # plan `Q[:, k] = q_now` is the one the path rows always admit for a robot at rest (every
            # row sits at its floor) and the one a HOLD would execute. It replaces the reference as
            # the start only when its merit is lower — i.e. only when the reference violates.
            iterate, iterate_states, iterate_merit, iterate_violation, start_record = (
                self._stay_start(iterate, iterate_states, iterate_merit, iterate_violation,
                                 reference, previous_chunk, q_now, scene, envelope))
        initial_merit, initial_violation = iterate_merit, iterate_violation
        initial_overshoot = self._limit_overshoot(iterate, q_now, envelope)
        initial_eligible = max(initial_overshoot.values()) <= _LIMIT_TOLERANCE
        best = iterate
        best_states = iterate_states
        best_merit = iterate_merit if initial_eligible else np.inf
        #: `-1` = 초기 iterate. 그 밖은 `candidates` 의 인덱스.
        best_index = -1
        #: QP 가 풀린 모든 후보의 `(merit, 위반, 수락)`. 반환된 것이 무엇이었는지와 그 옆에 무엇이
        #: 있었는지를 기록으로 남긴다 — 후보 선택 규칙을 나중에 다시 실행하지 않고 검산할 수 있다.
        candidates: list[dict[str, Any]] = []
        qp_failures = 0
        # Per-stage timing, accumulated rather than sampled. Where the milliseconds go is the single
        # most useful thing to know about a real-time optimizer, and it changes with the scene.
        timing = {"linearize": 0.0, "assemble": 0.0, "qp": 0.0, "check": 0.0}
        best_slack = 0.0
        iterations = 0
        qp_iterations = 0
        radius = cfg.sqp.trust_radius
        budget_bound = False
        time_budget_hit = False

        # **반복 수 우선** (T31 G2-i): 벽시계 예산은 `min_iterations` 만큼 돈 뒤부터 검사한다.
        # `max_iterations` 가 더 작으면 그것이 이긴다 — 상한은 상한이다.
        min_iterations = max(1, min(int(getattr(cfg.sqp, "min_iterations", 1)),
                                    int(cfg.sqp.max_iterations)))
        #: T43 TP — this solve's `build_problem` result (`sqp.fast_eval`) and QPs solved so far.
        qp_template = None
        n_qp = 0
        while iterations < cfg.sqp.max_iterations:
            # Checked *after* the first iteration, never before it. A budget so tight that the
            # set-up alone exhausts it would otherwise return the reference untouched while
            # reporting success — the one outcome a safety layer must not produce. One step is the
            # real-time iteration scheme's minimum unit of work, and it is always taken.
            #
            # T31: "the first iteration" became "the first `min_iterations`". In E3 the first
            # iteration costs ~105 ms (the first linearize on a new field ~100 ms; later ones ~3 ms)
            # against a 50 ms budget, so the old check ended 1124/1125 chunks after a single QP — a
            # rejected first step never got its retry.
            if (iterations >= min_iterations
                    and (time.perf_counter() - started) * 1000.0 >= cfg.sqp.time_budget_ms):
                notes.append(
                    f"time budget {cfg.sqp.time_budget_ms:.0f} ms reached after {iterations} "
                    "iteration(s); returning the best iterate so far"
                )
                time_budget_hit = True
                break
            iterations += 1

            mark = time.perf_counter()
            rows = self.linearizer.linearize(iterate, q_now, scene, cfg, states=iterate_states)
            timing["linearize"] += (time.perf_counter() - mark) * 1000.0
            mark = time.perf_counter()
            budget_bound = budget_bound or bool(rows.budget_bound_steps)
            if self.fast_eval and qp_template is not None:
                # T43 TP: only the box rows depend on the iterate and the trust radius.
                problem = rebox_problem(
                    qp_template, self.limits, iterate, radius, q_now=q_now[self.layout.q_indices],
                    anchor_envelope=envelope)
            else:
                problem = build_problem(
                    reference, iterate, self.limits, cfg,
                    q_now=q_now[self.layout.q_indices], previous_chunk=previous_chunk,
                    n_slack=n_slack, trust_radius=radius,
                    **({"anchor_envelope": envelope} if envelope is not None else {}),
                    **self._problem_history(),
                )
                if self.fast_eval:
                    qp_template = problem
            problem = append_collision_rows(
                problem, rows, iterate, self.block, cfg.reduction.linearization_backoff,
                **({"extra_slack": n_path} if n_path else {})
            )
            timing["assemble"] += (time.perf_counter() - mark) * 1000.0
            if self.path is not None:
                mark = time.perf_counter()
                problem = self._append_path(problem, iterate, scene)
                timing["path"] = timing.get("path", 0.0) + (time.perf_counter() - mark) * 1000.0
            st = self.__dict__.get("_anchor_state")
            if st is not None:
                # IFG-TOA: the palm terms linearised at this iterate, on the joint block of P / q.
                from AG3S.trajopt.grasp_anchor import add_goal_box, add_to_problem, anchor_qp_terms

                mark = time.perf_counter()
                if st.get("rot") is not None:
                    # AG3S-GEO-GRASP: the rotation term on the goal rows, linearised at this iterate
                    from AG3S.trajopt.grasp_anchor import rotation_qp_terms

                    p_it, J_it, R_it, Jw_it = st["fk"].pose(iterate, st["q_now"])
                    r = st["rot"]
                    problem = add_to_problem(problem, *rotation_qp_terms(iterate, R_it, Jw_it, r["rows"], r["Rg"], r["w"]))
                else:
                    p_it, J_it = st["fk"](iterate, st["q_now"])
                if st["cost_on"]:
                    problem = add_to_problem(problem, *anchor_qp_terms(iterate, p_it, J_it, st["weights"],
                                                                       st["targets"]))
                if st["box"] is not None:
                    # E1-B: the soft per-axis goal box on the goal rows (its own slack, after every other variable).
                    b = st["box"]
                    problem = add_goal_box(problem, iterate, p_it, J_it, b["rows"], b["goal"], b["r"], b["weight"])
                timing["anchor"] = timing.get("anchor", 0.0) + (time.perf_counter() - mark) * 1000.0

            mark = time.perf_counter()
            qp_extra: dict[str, Any] = {}
            if getattr(cfg.qp, "warm_start_iterate", False) and not n_qp:
                # T43 TP: the first QP of this solve starts at the iterate, not at the last chunk.
                qp_extra["x0"] = problem.flatten(iterate)
            if getattr(cfg.qp, "eps_after_first", None) is not None and n_qp:
                qp_extra["eps"] = float(cfg.qp.eps_after_first)
            solution = self.solver.solve(problem, warm_start=cfg.sqp.warm_start, **qp_extra)
            n_qp += 1
            timing["qp"] += (time.perf_counter() - mark) * 1000.0
            qp_iterations += solution.iterations
            if not solution.success:
                notes.append(f"QP failed at iteration {iterations}: {solution.status}")
                qp_failures += 1
                radius *= cfg.sqp.trust_shrink
                if radius < cfg.sqp.trust_radius_min:
                    break
                continue

            mark = time.perf_counter()
            candidate, slack = problem.split(solution.x)
            candidate_states = self.linearizer.sphere_states(candidate, q_now)
            merit, violation = self._merit_terms(
                candidate, reference, previous_chunk, q_now, scene, states=candidate_states
            )
            timing["check"] += (time.perf_counter() - mark) * 1000.0
            step_size = float(np.abs(candidate - iterate).max())
            accepted = bool(merit <= iterate_merit)
            candidates.append({
                "iteration": int(iterations),
                "merit": float(merit),
                "violation_m": None if violation is None else float(violation),
                "accepted": accepted,
                "step_size_rad": step_size,
                "trust_radius_rad": float(radius),
            })

            if merit < best_merit:
                best, best_merit = candidate, merit
                best_slack = float(np.sum(np.abs(slack)))
                best_states = candidate_states
                best_index = len(candidates) - 1
            # A step that improved the merit earns a wider trust region; one that did not gets a
            # narrower one and another attempt from the same iterate. `iterate_merit` is carried
            # rather than recomputed: it costs a full-resolution clearance sweep, which after the QP
            # became fast is the most expensive thing left in the loop.
            if accepted:
                iterate, iterate_merit, iterate_states = candidate, merit, candidate_states
                radius = min(radius * cfg.sqp.trust_expand, cfg.sqp.trust_radius_max)
            else:
                radius = max(radius * cfg.sqp.trust_shrink, cfg.sqp.trust_radius_min)

            if step_size < cfg.sqp.step_tolerance:
                break

        # **limit 만 어긴 초기 iterate 의 대비는 reference 의 최소 투영이다** (T31 G2-ii).
        projection = None
        if not initial_eligible and not any(c["accepted"] for c in candidates):
            projection = self._limit_projection(reference, q_now, scene, initial_violation,
                                                envelope)
            if projection["used"]:
                best, best_slack = projection.pop("_trajectory"), 0.0
                best_states = projection.pop("_states")
                best_index = _PROJECTION_INDEX
                notes.append(
                    "the initial iterate was outside the joint limits only (collision violation "
                    + ("not enforced" if initial_violation is None
                       else f"{initial_violation * 1000:.2f} mm")
                    + ") and no QP candidate was accepted; returning the minimal projection of the "
                    f"reference onto the limits (max change {projection['max_change_rad']:.4f} rad)"
                    " instead of the rejected QP step")
            projection.pop("_trajectory", None)
            projection.pop("_states", None)

        if budget_bound:
            notes.append(
                f"reduction.rows_per_step={cfg.reduction.rows_per_step} bound on at least one step; "
                "some near-active constraints were not shown to the QP (the final check still sees "
                "all of them)"
            )
        selection = {
            "returned": _returned_label(best_index, candidates),
            "returned_index": int(best_index),
            "initial_merit": float(initial_merit),
            "initial_violation_m": (None if initial_violation is None
                                    else float(initial_violation)),
            "initial_eligible": bool(initial_eligible),
            "initial_limit_overshoot": {k: float(v) for k, v in initial_overshoot.items()},
            "candidates": candidates,
            "n_candidates": len(candidates),
            "n_accepted": int(sum(1 for c in candidates if c["accepted"])),
            "qp_failures": int(qp_failures),
            "min_iterations": int(min_iterations),
            "projection": projection,
            # T43 Q — only with the path rows on.
            **({"initial_source": start_record} if start_record is not None else {}),
        }
        result = self._finish(
            best, reference, q_now, scene, template, iterations, best_slack,
            reference_violation, previous_chunk, notes, started, geometry_certified,
            states=best_states, timing=timing, qp_iterations=qp_iterations,
            time_budget_hit=time_budget_hit, selection=selection,
        )
        if envelope is not None:
            result.metrics["rest_start"] = self._rest_start_record(
                best, reference, q_now, envelope)
        return result

    def _seed_is_aligned(self, continuity) -> bool:
        """T43 TP (`warm_start_lookahead`) — is shifting the last plan by `execution_length` this
        chunk's seed? Yes only when the caller's continuity reference (the previous refined chunk,
        aligned by the steps that actually executed — `refiner._continuity_reference`) starts with
        exactly that shifted plan: the previous chunk was this optimizer's own look-ahead plan, it
        executed `execution_length` steps, and nothing came between. A closed gate (the plan was the
        window optimizer's), a HOLD (no reference), a skipped or passed-through chunk all fail it."""
        if continuity is None:
            return False
        k = min(int(self.config.horizon.execution_length), self.horizon - 1)
        n = self.horizon - k
        cont = np.asarray(continuity, np.float64)
        prev = np.asarray(self._previous, np.float64)
        return (cont.ndim == 2 and cont.shape[0] == prev.shape[0] and cont.shape[1] >= n
                and np.array_equal(cont[:, :n], prev[:, k:]))

    def _lookahead_gate(self, reference, q_now, scene) -> dict[str, Any]:
        """T43 TA (A) — does the reference come within `lookahead_gate` of an obstacle in the tail?

        The tail's rows are the look-ahead ones (obstacle class only, `_lookahead_keep`); the value is
        their worst TO clearance (`d − r − margin`) over the reference's tail steps.
        """
        self.linearizer.set_attached(
            getattr(scene, "attached_points", None), getattr(scene, "attached_parent_link", None))
        k = int(self.lookahead_from)
        per_step = self.linearizer.step_clearances(reference, q_now, scene)
        tail = per_step[k:]
        tail_min = float(tail.min(initial=np.inf))
        gate = float(self.lookahead_gate)
        return {"open": bool(tail_min < gate), "gate_m": gate,
                "tail_reference_clearance_m": tail_min if np.isfinite(tail_min) else None,
                "tail_reference_worst_step": (int(k + int(np.argmin(tail)))
                                              if tail.size and np.isfinite(tail_min) else None)}

    def _solve_window(self, gate, reference, q_now, scene, *, previous_chunk, template,
                      geometry_certified, start_at_rest, servo_feedback, deflection_history,
                      command_history, grasp_anchor=None, window_track_scale=None) -> TrajOptResult:
        """T43 TA (A) — gate closed: solve exactly as without look-ahead (`execution_length` steps,
        `self.window_optimizer`), and hand back the reference's tail untouched."""
        k = int(self.lookahead_from)
        inner = self.window_optimizer
        path = getattr(self.linearizer, "_param_path", None)
        inner.linearizer.set_joint_parameter_path(None if path is None else path[:, :k])
        result = inner.solve(
            reference[:, :k], q_now, scene,
            previous_chunk=None if previous_chunk is None else previous_chunk[:, :k],
            template=template, geometry_certified=geometry_certified,
            start_at_rest=start_at_rest, servo_feedback=servo_feedback,
            deflection_history=deflection_history, command_history=command_history,
            **({} if grasp_anchor is None else {"grasp_anchor": grasp_anchor}),
            **({} if window_track_scale is None else {"joint_track_scale": window_track_scale}),
            # E2: the window solve is this chunk's standard solve — its multi-seed runs there (top-level solves only).
            **({"multi_seed_index": self._ms_child_index}
               if self.multi_seed > 0 and self._ms_child_index is not None else {}))
        trajectory = np.concatenate([np.asarray(result.trajectory, np.float64),
                                     np.asarray(reference, np.float64)[:, k:]], axis=1)
        result = dataclasses.replace(result, trajectory=trajectory)
        result.metrics["lookahead"] = {"from": k, "planned": int(reference.shape[1]),
                                       "gate": "closed", **{x: v for x, v in gate.items()
                                                            if x != "open"}}
        return result

    def _problem_history(self) -> dict[str, Any]:
        """T43 TA — the `build_problem` keywords for B and C. `{}` when both are off."""
        out: dict[str, Any] = {}
        if self._history.get("deflection") is not None:
            out["deflection_history"] = self._history["deflection"]
        if self.servo_accel_bound is not None:
            out["servo_accel_bound"] = self.servo_accel_bound
            if self._history.get("command") is not None:
                out["command_history"] = self._history["command"]
        if self.__dict__.get("_track_scale") is not None:
            out["joint_track_scale"] = self._track_scale   # IFG-TOW
        return out

    def _append_path(self, problem, iterate, scene):
        """T43 Q — the rows between waypoints, below the collision block (their own slacks)."""
        from AG3S.trajopt.sweep import append_path_rows

        if not getattr(self.config.collision, "enabled", True):
            # collision off (T15): the same shape, every row unused — as `LinearizedRows.all_unused`.
            from AG3S.trajopt.sweep import PathRows

            H, R, nq = self.horizon, self.path.rows_per_step, self.layout.nq_opt
            rows = PathRows(np.full((H, R), np.inf), np.zeros((H, R, H, nq)), np.zeros((H, R), bool),
                            np.zeros((H, R), int), np.zeros((H, R), int), 0)
        else:
            rows = self.path.linearize(iterate, scene)
        return append_path_rows(problem, rows, iterate, self.path.block, self.block.n_rows,
                                self.config.reduction.linearization_backoff)

    def _stay_start(self, iterate, states, merit, violation, reference, previous_chunk, q_now,
                    scene, envelope):
        """T43 Q — keep the given start, or switch to ``Q[:, k] = q_now`` if that has lower merit.

        Only looked at when the given start violates (`violation > violation_tolerance`); the stay
        plan must also be inside the limits (`_limit_overshoot`), as every other candidate.
        """
        tol = self.config.safety.violation_tolerance
        record = {"source": "reference", "stay_merit": None, "stay_violation_m": None}
        if violation is None or violation <= tol:
            return iterate, states, merit, violation, record
        stay = np.tile(np.asarray(q_now, np.float64)[self.layout.q_indices][:, None],
                       (1, iterate.shape[1]))
        if max(self._limit_overshoot(stay, q_now, envelope).values()) > _LIMIT_TOLERANCE:
            record["source"] = "reference (stay outside the limits)"
            return iterate, states, merit, violation, record
        stay_states = self.linearizer.sphere_states(stay, q_now)
        stay_merit, stay_violation = self._merit_terms(stay, reference, previous_chunk, q_now,
                                                       scene, states=stay_states)
        record.update(stay_merit=float(stay_merit),
                      stay_violation_m=None if stay_violation is None else float(stay_violation))
        if stay_merit < merit:
            record["source"] = "stay"
            return stay, stay_states, stay_merit, stay_violation, record
        return iterate, states, merit, violation, record

    def _path_violation(self, trajectory, scene) -> float:
        """T43 Q — penetration (m, ≥ 0) of the rows between waypoints, below their floors."""
        worst, _ = self.path.violation(trajectory, scene)
        return max(0.0, -worst)

    def _rest_start_record(self, trajectory, reference, q_now, envelope) -> dict[str, Any]:
        """T41 b — 정지 출발 범위가 무엇이었고 반환된 계획이 어디서 시작하나 (기록용, rad)."""
        q0 = np.asarray(q_now, np.float64).reshape(-1)[self.layout.q_indices][:, None]
        traj = np.asarray(trajectory, np.float64)
        ref = np.asarray(reference, np.float64)
        n = min(traj.shape[1], envelope.shape[1])
        off = np.abs(traj[:, :n] - q0)
        return {
            "applied": True,
            "tolerance_rad": (None if getattr(self.config.limits, "rest_start_tolerance", None)
                              is None else float(self.config.limits.rest_start_tolerance)),
            "envelope_rad_max_per_step": [float(v) for v in envelope[:, :n].max(axis=0)],
            "step0_offset_rad": float(off[:, 0].max(initial=0.0)) if n else 0.0,
            "reference_step0_offset_rad": (float(np.abs(ref[:, 0] - q0[:, 0]).max(initial=0.0))
                                           if ref.shape[1] else 0.0),
            "envelope_overshoot_rad": float(np.max(off - envelope[:, :n], initial=0.0)),
        }

    # --- helpers --------------------------------------------------------------------------
    def _merit(self, trajectory, reference, previous_chunk, q_now, scene, states=None) -> float:
        """Cost plus the true constraint violation, both at full resolution.

        Deliberately *not* computed from the QP's own linearized rows. A merit function built by the
        same code that builds the subproblem agrees with it even when both are wrong; this one
        re-evaluates the geometry and so can disagree, which is the only way the step-acceptance test
        can catch a bad linearization.
        """
        return self._merit_terms(trajectory, reference, previous_chunk, q_now, scene, states)[0]

    def _merit_terms(
        self, trajectory, reference, previous_chunk, q_now, scene, states=None
    ) -> tuple[float, Optional[float]]:
        """`(merit, violation_m)` — `_merit` 과 **같은 계산**이고, 위반량을 버리지 않고 돌려줄 뿐이다.

        후보 기록(T24)이 `(merit, 위반, 수락)` 을 싣기 위해 있다. 위반을 따로 한 번 더 재면
        후보마다 clearance sweep 이 하나 늘고, 그것은 예산 안의 반복 수를 바꾼다 — 기록하려고
        동작을 바꾸는 것이 된다.

        충돌이 꺼진 판에서는 `violation_m` 이 `None` 이다: merit 이 위반을 **재지 않으므로**
        (T15), 여기서 재면 역시 루프 비용이 는다. "0" 이 아니라 "재지 않음" 이다. 반환된 궤적의
        위반은 `_finish` 가 스위치와 무관하게 잰다.
        """
        cost = self._objective(trajectory, reference, previous_chunk)
        if not self.config.collision.enabled:
            # **꺼진 판에서는 벌점도 없어야 한다** (T15). 행만 비우고 merit 에 위반을 남기면
            # 충돌이 여전히 해를 밀고, 그러면 "충돌을 껐다" 가 거짓이 된다 — QP 가 아니라 step
            # 수락 기준을 통해 미는 것뿐이다. 측정(`worst_row`)은 이것과 무관하게 계속 돈다.
            return cost, None
        if self.lookahead_from is not None:
            # T43 TA (A): the window's and the tail's worst violations, summed — neither masks
            # the other in step acceptance (a max would let a stuck tail veto a window fix).
            per_step = self.linearizer.step_clearances(trajectory, q_now, scene, states)
            k = self.lookahead_from
            violation = (max(0.0, -float(per_step[:k].min(initial=np.inf)))
                         + max(0.0, -float(per_step[k:].min(initial=np.inf))))
            return cost + self.config.cost.w_slack * violation, violation
        violation = max(0.0, -self.linearizer.full_violation(trajectory, q_now, scene, states))
        if self.path is not None:
            violation = max(violation, self._path_violation(trajectory, scene))
        return cost + self.config.cost.w_slack * violation, violation

    def _limit_overshoot(self, trajectory: np.ndarray, q_now: np.ndarray,
                         envelope: Optional[np.ndarray] = None) -> dict[str, float]:
        """`limit_report` + **anchor** — QP 가 hard 로 지키는 첫 스텝 조건까지.

        `build_problem` 은 ``|Q[:, 0] - q_now| <= max_step`` 을 box 에 넣어 hard 로 지킨다. 초기
        iterate 가 그것을 넘으면 QP 후보와 같은 자격이 아니므로 여기서 함께 잰다.

        T41 b: `envelope` (정지 출발 범위) 가 있으면 anchor 는 **모든 스텝**의
        ``|Q[:, k] - q_now| - envelope[:, k]`` 최대값이다 — QP box 가 지키는 것과 같은 조건.
        """
        report = dict(limit_report(trajectory, self.limits))
        anchor = 0.0
        step = np.asarray(self.limits.max_step, np.float64)
        if trajectory.shape[1] and np.all(np.isfinite(step)):
            q0 = np.asarray(q_now, np.float64).reshape(-1)[self.layout.q_indices]
            anchor = float(np.max(np.abs(trajectory[:, 0] - q0) - step, initial=0.0))
        if envelope is not None and trajectory.shape[1]:
            q0 = np.asarray(q_now, np.float64).reshape(-1)[self.layout.q_indices][:, None]
            n = min(trajectory.shape[1], envelope.shape[1])
            anchor = max(anchor, float(np.max(np.abs(trajectory[:, :n] - q0) - envelope[:, :n],
                                              initial=0.0)))
        report["anchor"] = max(anchor, 0.0)
        return report

    def _limit_projection(self, reference, q_now, scene, initial_violation,
                          envelope: Optional[np.ndarray] = None) -> dict[str, Any]:
        """T31 G2-ii — **reference 의 최소 투영**을 반환할 수 있는가, 그리고 그 궤적.

        | 조건 | 아니면 |
        |---|---|
        | `sqp.limit_projection_fallback` 이 켜져 있다 | `reason: disabled` |
        | 초기 iterate 의 충돌 위반 ≤ `safety.violation_tolerance` (= limit **만** 어겼다) | `reason: initial_collision_violation` |
        | 투영이 존재한다 (box ∩ anchor 가 비지 않음) | `reason: no_projection` |
        | 투영이 limit 을 전부 지킨다 (`_limit_overshoot` — position·velocity·acceleration·anchor) | `reason: projection_outside_limits` |
        | 투영이 충돌 검사를 통과한다 (전 해상도 `full_violation` ≤ tolerance) | `reason: projection_collides` |

        충돌이 꺼진 판(`collision.enabled=False`)에서는 둘째·다섯째 줄을 보지 않는다 — merit 도
        충돌을 보지 않으므로 (T15), 여기서만 보면 "껐다" 가 거짓이 된다. 반환된 궤적의 위반은
        `_finish` 가 스위치와 무관하게 잰다.
        """
        cfg = self.config
        out: dict[str, Any] = {"tried": True, "used": False, "reason": None,
                               "max_change_rad": None, "violation_m": None,
                               "limit_overshoot": None}
        if not getattr(cfg.sqp, "limit_projection_fallback", True):
            out.update(tried=False, reason="disabled")
            return out
        collision = bool(cfg.collision.enabled)
        tolerance = cfg.safety.violation_tolerance
        if collision and (initial_violation is None or initial_violation > tolerance):
            out.update(tried=False, reason="initial_collision_violation")
            return out
        projected = project_to_limits(reference, self.limits,
                                      np.asarray(q_now, np.float64)[self.layout.q_indices],
                                      **({"envelope": envelope} if envelope is not None else {}))
        if projected is None:
            out["reason"] = "no_projection"
            return out
        out["max_change_rad"] = float(np.abs(projected - reference).max(initial=0.0))
        overshoot = self._limit_overshoot(projected, q_now, envelope)
        out["limit_overshoot"] = {k: float(v) for k, v in overshoot.items()}
        if max(overshoot.values()) > _LIMIT_TOLERANCE:
            out["reason"] = "projection_outside_limits"
            return out
        states = None
        if collision:
            # FK 는 한 번 — `_finish` 가 같은 상태로 최종 검증을 한다.
            states = self.linearizer.sphere_states(projected, q_now)
            violation = max(0.0, -self.linearizer.full_violation(projected, q_now, scene, states))
            if self.path is not None:
                violation = max(violation, self._path_violation(projected, scene))
            out["violation_m"] = float(violation)
            if violation > tolerance:
                out["reason"] = "projection_collides"
                return out
        out.update(used=True, reason="limit_only", _trajectory=projected, _states=states)
        return out

    def _passthrough(self, reference, q_now, template, status, notes, started) -> TrajOptResult:
        chunk = self._chunk_from(reference, template)
        return TrajOptResult(
            chunk=chunk,
            trajectory=reference,
            status=status,
            iterations=0,
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            max_violation=0.0,
            reference_violation=0.0,
            slack_norm=0.0,
            cost=0.0,
            reference_deviation=0.0,
            notes=list(notes),
            # 제약이 없었으므로 최악 행도 없다. **키는 둔다** — 없음이 `None` 으로 적히는 것과
            # 키가 빠지는 것은 읽는 쪽에 다른 뜻이다.
            metrics={"n_rows": 0, "max_violation_pair": None,
                     # 키 집합을 `_finish` 와 같게 둔다 — 읽는 쪽이 `.get` 두 갈래로 갈리지 않는다.
                     "sqp_iterations": 0, "max_iterations": int(self.config.sqp.max_iterations),
                     "time_budget_ms": float(self.config.sqp.time_budget_ms),
                     "time_budget_hit": False, "max_iterations_hit": False,
                     "collision_enabled": bool(self.config.collision.enabled),
                     # 후보 선택 키도 같게 둔다 (T24). 풀지 않았으므로 후보는 없고, 나간 것은
                     # reference 그 자체다.
                     **_no_selection(),
                     **self._scm_record()},
        )

    def _chunk_from(self, trajectory: np.ndarray, template: Optional[np.ndarray]) -> np.ndarray:
        if template is None:
            template = np.zeros((trajectory.shape[1], self.layout.action_dim))
        return self.layout.trajectory_to_chunk(trajectory, template)

    def _finish(
        self, trajectory, reference, q_now, scene, template, iterations, slack_norm,
        reference_violation, previous_chunk, notes, started, geometry_certified, states=None,
        timing=None, qp_iterations=0, time_budget_hit=False, selection=None,
    ) -> TrajOptResult:
        cfg = self.config
        # `full_violation` 이 아니라 `worst_row` 다. **같은 값을 내면서 그 값을 만든 행의 신원까지
        # 돌려준다** — clearance sweep 도 forward kinematics 도 늘지 않고 argmin 만 더 돈다.
        # `T5f` 가 *"violated 가 어느 제약인가"* 에서 막힌 것이 이 숫자에 이름이 없었기 때문이다.
        lookahead_record = None
        if self.lookahead_from is not None:
            # T43 TA (A): certify the executed window; the tail is recorded, never certified.
            k = self.lookahead_from
            clearance, worst_pair = self.linearizer.worst_row(trajectory, q_now, scene, states,
                                                              steps=k)
            per_step = self.linearizer.step_clearances(trajectory, q_now, scene, states)
            tail = per_step[k:]
            tail_min = float(tail.min(initial=np.inf))
            lookahead_record = {
                "from": int(k), "planned": int(trajectory.shape[1]),
                **({"gate": "open", **(getattr(self, "_gate_record", None) or {})}
                   if self.lookahead_gate is not None else {"gate": None}),
                "window_clearance_m": float(clearance),
                "tail_clearance_m": tail_min if np.isfinite(tail_min) else None,
                "tail_violation_m": max(0.0, -tail_min) if np.isfinite(tail_min) else 0.0,
                "tail_worst_step": (int(k + int(np.argmin(tail))) if tail.size
                                    and np.isfinite(tail_min) else None),
                **{key: v for key, v in (getattr(self.linearizer, "_last_lookahead", None)
                                         or {}).items() if key != "from"}}
        else:
            clearance, worst_pair = self.linearizer.worst_row(trajectory, q_now, scene, states)
        # AG3S-CORE-EVAL-A6 — the relaxed palm rows of what is returned (read before the path check re-evaluates the
        # field). `None` unless the scene carries the palm allowance this chunk: the metrics keys as before.
        palm_record = (dict(getattr(self.linearizer, "_last_neighbour_palm_contact", None) or {})
                       if getattr(scene, "neighbour_palm_contact", None) is not None
                       and getattr(scene, "esdf", None) is not None else None)
        # T43 HM — the held rows' classes on what is returned (read before the path check
        # re-evaluates the field). `None` unless `held_obstacle_margin` is on and something is held.
        held_rows = (self.linearizer.held_rows_record()
                     if getattr(scene, "held_obstacle_margin", False)
                     and getattr(scene, "esdf", None) is not None
                     and hasattr(self.linearizer, "held_rows_record") else None)
        violation = max(0.0, -clearance)
        tolerance = cfg.safety.violation_tolerance
        path_record = None
        if self.path is not None:
            # T43 Q — the rows between waypoints, at full resolution, on what is returned. A
            # violation there makes the chunk VIOLATED like any other row; `clearance_m` and
            # `max_violation_pair` stay the waypoint numbers, the path's are in `path_check`.
            path_worst, path_record = self.path.violation(trajectory, scene, record=True)
            path_record["timing_ms"] = {k: round(v, 3) for k, v in self.path.timing.items()}
            path_violation = max(0.0, -path_worst)
            if path_violation > tolerance:
                notes.append(
                    f"path check ({self.path.mode}): {path_violation * 1000:.1f} mm below the floor "
                    f"at step {path_record['worst']['step']} "
                    f"({path_record['worst']['at_kind']} {path_record['worst']['at']:g}, "
                    f"{path_record['worst']['link']}) — between the waypoints")
            violation = max(violation, path_violation)

        if violation > tolerance:
            status = TrajOptStatus.VIOLATED
            notes.append(
                f"best effort: {violation * 1000:.1f} mm of penetration remains "
                f"(was {reference_violation * 1000:.1f} mm). The caller decides whether to execute."
            )
        elif iterations >= cfg.sqp.max_iterations or any("budget" in n for n in notes):
            status = TrajOptStatus.FEASIBLE
        else:
            status = TrajOptStatus.OPTIMAL

        if status.safe and cfg.safety.require_certified_geometry and not geometry_certified:
            # The trajectory clears everything AG3S described. AG3S said it could not account for
            # everything it saw, so "clear" is a statement about the model, not about the world.
            status = TrajOptStatus.VIOLATED
            notes.append(
                "the trajectory clears every constraint, but AG3S could not certify the geometry "
                "behind them; safety cannot be claimed for a scene that was not fully accounted for"
            )

        if not cfg.collision.enabled:
            # **이 기록이 "위반 0" 으로 읽히면 안 된다.** 위 `violation` 은 여전히 참값이지만
            # (측정은 이 스위치와 무관하다), 최적화기는 그것을 보지 않았다. 그 구별이 기록에
            # 없으면 이 실행의 궤적이 "충돌을 피한 궤적" 으로 읽힌다.
            notes.append(
                "COLLISION CONSTRAINTS WERE OFF (collision.enabled=False): the optimizer saw no "
                "collision rows and no violation penalty. max_violation below is still measured "
                "from the real geometry — it is what the trajectory does, not what was enforced."
            )
        limits = limit_report(trajectory, self.limits)
        if max(limits.values()) > _LIMIT_TOLERANCE:
            notes.append(
                f"limit overshoot (m or m/step): {  {k: round(v, 6) for k, v in limits.items()} }"
            )

        self._previous = trajectory
        self.frame_index += 1
        return TrajOptResult(
            chunk=self._chunk_from(trajectory, template),
            trajectory=trajectory,
            status=status,
            iterations=iterations,
            solve_time_ms=(time.perf_counter() - started) * 1000.0,
            max_violation=violation,
            reference_violation=max(0.0, reference_violation),
            slack_norm=float(slack_norm),
            cost=self._objective(trajectory, reference, previous_chunk),
            reference_deviation=float(np.linalg.norm(trajectory - reference)),
            notes=notes,
            metrics={
                "clearance_m": float(clearance),
                # **최악 행의 신원.** 활성 제약이 하나도 없으면 `None` 이다 — 없는 신원을
                # 지어내지 않는다. 키는 항상 있어서 소비 쪽이 `.get` 두 갈래로 갈리지 않는다.
                "max_violation_pair": worst_pair,
                "limit_overshoot": limits,
                "frame_index": self.frame_index,
                "geometry_certified": bool(geometry_certified),
                "qp_solver": cfg.qp.solver,
                "qp_iterations": int(qp_iterations),
                # **구조화된 값으로 싣는다** (T15). 예전에는 `notes` 의 문장
                # (`"time budget 50 ms reached after 1 iteration(s)"`) 뿐이라 **셀 수 없었다** —
                # "몇 번 돌았나" 가 다음 질문인데 그때마다 실행을 다시 해야 했다.
                "sqp_iterations": int(iterations),
                "max_iterations": int(cfg.sqp.max_iterations),
                "time_budget_ms": float(cfg.sqp.time_budget_ms),
                "time_budget_hit": bool(time_budget_hit),
                "max_iterations_hit": bool(iterations >= cfg.sqp.max_iterations),
                # 충돌이 꺼졌는지. `notes` 와 **둘 다** 싣는다 — 기계가 세는 쪽과 사람이 읽는 쪽.
                "collision_enabled": bool(cfg.collision.enabled),
                "timing_ms": {k: round(v, 3) for k, v in (timing or {}).items()},
                # **무엇이 반환됐나** (T24, 지침 §7.3). `returned` 는 `"initial"` · `"accepted"` ·
                # `"best_unaccepted"` 중 하나이고, 그 판정의 근거인 후보 목록이 옆에 실린다.
                # 위의 `clearance_m`·`limit_overshoot`·상태는 **반환된 그 궤적**을 최종 기하와
                # 실제 한계로 다시 잰 값이다 — 후보의 `violation_m` 은 루프 안의 merit 가 본 값이다.
                **(selection if selection is not None else _no_selection()),
                **({"path_check": path_record} if path_record is not None else {}),
                **({"held_rows": held_rows} if held_rows is not None else {}),
                **({"neighbour_palm_contact": palm_record} if palm_record is not None else {}),
                **({"lookahead": lookahead_record} if lookahead_record is not None else {}),
                **self._ta_record(trajectory, reference),
                **self._scm_record(),
            },
        )

    def _scm_record(self) -> dict[str, Any]:
        """AG3S-PRE-CORE-TO SCM — `{"palm_shell_margin_m": m, "obstacle_margin_m": m}` when
        `collision.palm_shell_margin` is set, `{}` otherwise (the metrics keys as before)."""
        col = self.config.collision
        m = getattr(col, "palm_shell_margin", None)
        if m is None:
            return {}
        return {"palm_shell_margin_m": float(m), "obstacle_margin_m": float(col.obstacle_margin)}

    def _ta_record(self, trajectory, reference) -> dict[str, Any]:
        """T43 TA (B, C) — what the returned trajectory does to the new terms. `{}` when off."""
        out: dict[str, Any] = {}
        w = float(getattr(self.config.cost, "w_deflection_rate", 0.0) or 0.0)
        if w > 0.0:
            from AG3S.trajopt.problem import deflection_rate_cost

            D = np.asarray(trajectory) - np.asarray(reference)
            out["deflection_rate"] = {
                "weight": w, "boundary": self._history.get("deflection") is not None,
                "cost": deflection_rate_cost(trajectory, reference, w,
                                             self._history.get("deflection")),
                "max_step_change_rad": (float(np.abs(np.diff(D, axis=1)).max())
                                        if D.shape[1] > 1 else 0.0)}
        if self.servo_accel_bound is not None:
            from AG3S.trajopt.problem import servo_accel_overshoot

            over = servo_accel_overshoot(trajectory, reference, self.servo_accel_bound,
                                         str(self.config.limits.servo_accel),
                                         self._max_step_change(), self._history.get("command"))
            from AG3S.trajopt.problem import boundary_velocity_overshoot

            hist = self._history.get("command")
            step = np.asarray(self.limits.max_step, np.float64)
            bvel = (boundary_velocity_overshoot(trajectory, hist, step)
                    if velocity_rows(self.limits) else np.zeros(0))
            out["servo_accel"] = {**(self.servo_accel_record or {}),
                                  "boundary": hist is not None,
                                  "overshoot_max_rad": float(over.max(initial=0.0)),
                                  "n_over": int((over > 1e-9).sum()),
                                  "boundary_velocity_ratio": (
                                      None if hist is None else float(np.max(
                                          np.abs(np.asarray(trajectory)[:, 0] - hist[:, -1])
                                          / step))),
                                  "boundary_velocity_overshoot_rad": float(
                                      bvel.max(initial=0.0))}
        return out


#: `best_index` 의 표시 — reference 의 최소 투영이 반환됐다 (T31 G2-ii). `-1` 은 초기 iterate.
_PROJECTION_INDEX = -2


def _returned_label(index: int, candidates: list[dict[str, Any]]) -> str:
    """`best_index` → `"initial"` · `"accepted"` · `"best_unaccepted"` · `"projection"` (T31).

    수락 여부를 **후보 기록에서 읽는다.** 적격한 초기 iterate 로 시작하면 `best_unaccepted` 는
    나오지 않는다 (`solve` 의 주석) — 나오는 것은 초기 iterate 가 한계 밖이라 비교에서 빠졌을
    때뿐이다. 그 성질을 여기서 가정하면 그것이 깨진 날 기록이 거짓말을 한다.
    """
    if index == _PROJECTION_INDEX:
        return "projection"
    if index < 0:
        return "initial"
    return "accepted" if candidates[index]["accepted"] else "best_unaccepted"


def _no_selection() -> dict[str, Any]:
    """풀지 않은 결과의 후보 선택 키 — `_finish` 와 키 집합을 같게 둔다."""
    return {"returned": "initial", "returned_index": -1, "initial_merit": None,
            "initial_violation_m": None, "initial_eligible": None,
            "initial_limit_overshoot": None, "candidates": [], "n_candidates": 0,
            "n_accepted": 0, "qp_failures": 0, "min_iterations": None, "projection": None}


__all__ = ["TrajectoryOptimizer"]
