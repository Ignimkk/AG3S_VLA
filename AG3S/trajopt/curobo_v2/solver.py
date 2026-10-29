"""T63 — cuRobo v2 `TrajectoryOptimizer` on one recorded chunk (B0 = basic cuRobo).

B0 = cuRobo's own trajopt (`trajopt/lbfgs_bspline_trajopt.yml`, `metrics_base.yml`,
`trajopt/transition_bspline_trajopt.yml`, all cost weights as shipped) with:
start = the chunk's `q_now`, goal = `reference_chunk[-1]` as a **c-space goal** (`solve_cspace`),
one seed = the reference chunk, world = the record's target-excluded layer (`esdf_free_*`) for
every sphere.

Every setting that differs from cuRobo's defaults is listed in `B0_CHANGES` (and copied into the
output record). B1-B4 will add `costs.ExtraCostTerm`s through `extra_terms=`; B0 passes none.

Timing grid. Our chunk is 50 steps at 15 Hz after `q_now` (t = k/15, k = 1..50). cuRobo's
optimizer state grid is `horizon = (n_knots + 4) x 4 + 1` states spaced `dt`; after every pass
cuRobo re-times the trajectory to its velocity / acceleration / jerk limits
(`compute_trajectory_dt`, time-optimal). B0 **fixes** dt instead: `dt = (50/15) / (horizon - 1)`
for all passes (`FixedDtTrajectoryOptimizer`, `finetune_attempts=0`, `finetune_dt_scale=1`), so
the spline spans exactly the chunk's 3.333 s and ends at the goal. The output is read from the
optimizer's **state grid** (`js_solution`), not from cuRobo's dense interpolation: that
interpolator rounds the samples per knot (`calculate_traj_steps`, `nearest_int`) and labels them
`interpolation_dt`, which re-times the spline. Timing `exact` (default) uses 21 knots so the
state grid (1/30 s) contains the 15 Hz grid; `resample16` keeps cuRobo's 16 knots and linearly
resamples the 1/24 s states.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
from curobo.types import DeviceCfg, JointState

from AG3S.trajopt.curobo_v2 import spec as S
from AG3S.trajopt.curobo_v2.spec import knot_seed, sample_chunk_grid
from AG3S.trajopt.curobo_v2 import world as W
from AG3S.trajopt.curobo_v2.costs import activate_extra_terms, extensible_cost_manager_cfg
from AG3S.trajopt.curobo_v2.robot_world import (
    build_robot_cfg, device_cfg, joint_limits_now, kinematics_params_in, set_finger_openings)

CHUNK_DT = 1.0 / 15.0
CHUNK_STEPS = 50

#: Every B0 setting changed from cuRobo v2's defaults, and why. (Copied into each output.)
B0_CHANGES: tuple[dict[str, str], ...] = (
    {"setting": "robot model", "value": "served RB-Y1 constraint model: 270 spheres on 6 gripper "
     "links from robot_spec.json (no cuRobo sphere fit); base_link link_torso_5; fingers locked at "
     "0 and their spheres shifted by the measured opening per chunk",
     "why": "same URDF, spheres and optimized joints as our TO (task step 0.1)"},
    {"setting": "joint limits", "value": "position = model_xml +- 0.02 rad margin, velocity / "
     "acceleration = URDF x 0.9 (TO build_limits); jerk = cuRobo default 500 rad/s^3",
     "why": "same robot limits as the TO; URDF limits are what cuRobo would read by default"},
    {"setting": "world", "value": "one voxel grid = record esdf_free_* (target-excluded fine "
     "layer), posed in link_torso_5 frame per chunk; outside the grid = no obstacle "
     "(max_esdf_distance); a chunk recorded without esdf_free (no target to exclude) gets the "
     "main fine layer esdf_*", "why": "task step 0.2: target-excluded layer for every sphere"},
    {"setting": "self_collision_check", "value": "False, and the self-collision constraint "
     "removed from the metrics rollout (metrics_base.yml) that decides success",
     "why": "our TO has no robot-robot pairs (serve --no-self-collision); no cuRobo self-collision "
     "ignore table exists for this sphere model"},
    {"setting": "goal", "value": "solve_cspace(goal = reference_chunk[-1] arm joints)",
     "why": "user decision 2026-10-20 (c-space goal, no pose goal)"},
    {"setting": "seed", "value": "seed_traj = reference chunk sampled at the knot times "
     "(k/(n_knots-1) of the span, as cuRobo's own linear seed); remaining seeds cuRobo default",
     "why": "task step 1 (seed = reference_chunk)"},
    {"setting": "dt / knots", "value": "fixed: state dt = (50/15)/(horizon-1) for every pass "
     "(compute_trajectory_dt overridden, finetune_attempts=0, finetune_dt_scale=1.0); timing "
     "'exact' = n_knots 21 (default 16) so 101 states at 1/30 s contain the 15 Hz grid; "
     "'resample16' = default 16 knots, states at 1/24 s linearly resampled",
     "why": "output on our 15 Hz, 50-step chunk grid without re-timing. cuRobo's interpolated "
     "output cannot be used: its samples per knot are rounded (calculate_traj_steps, "
     "nearest_int), so 16 knots at 1/6 s give 61 samples labelled 1/15 s for a 3.33 s spline"},
    {"setting": "start state", "value": "q_now, velocity = acceleration = 0",
     "why": "the record has no joint velocity at capture; cuRobo's default for a missing velocity"},
)


#: The `world` row of `B0_CHANGES`, per world mode (`CuroboChunkSolver(world=...)`).
WORLD_CHANGES: dict[str, dict[str, str]] = {
    "free_layer": {"value": B0_CHANGES[2]["value"], "why": B0_CHANGES[2]["why"]},
    "b0": {"value": "composed per sphere and step (composed_world.ComposedSceneCollision, mode "
           "b0): inside the esdf_free lattice the target-excluded esdf_free layer, outside it the "
           "coarse esdf_coarse layer (20 mm). A chunk recorded without esdf_free (no target to "
           "exclude) uses our main field there: min(coarse, fine) inside the fine lattice + coarse "
           "outside (tagged box_layer=main_fine; T63 step 3, user decision 3 lead default — step 2 "
           "used the fine layer alone inside the lattice). Each layer is queried by cuRobo's own kernel (sweep, speed "
           "metric, activation, weight as configured); outside the coarse lattice = no obstacle",
           "why": "user decision 2 (2026-10-21): same world information as our TO outside the "
           "box; esdf_free alone left the other hand unprotected (97.8 % of its sphere-steps "
           "outside the box)"},
    "b0m": {"value": "the b0 world with our TO's obstacle margin (composed_world, mode b0m; "
            "T63b user decision 2): each sphere and step is queried at radius r + margin, margin = "
            "the server's obstacle_margin (0.010 m) when our TO's _obstacle_rows rule (b0m.py) "
            "calls the row 'obstacle' on the b0 distance, else 0 — target rows (link may touch the "
            "target, |d - d_obj| <= finest voxel) and support rows (|d - s_k| <= 2 finest voxels) "
            "keep 0. Rule inputs per chunk: the scene our TO certifies against "
            "(b0m.rules_from_scene: contact rows, support planes, manipulated points)",
            "why": "T63b user decision 2 (2026-10-21): B0 did not know the 10 mm margin our TO is "
                   "certified with; information only, no objective term"},
    "lte": {"value": "B4 per-link world (composed_world, mode lte): rows with "
            "target_field_exclude -> the b0 rule; other rows -> our main field (in the fine lattice "
            "min(coarse, fine), outside coarse); rows with a target margin (contact policy "
            "manipulated_link_margin, rows not exempt) also see the target point cloud at radius "
            "r + margin", "why": "T63 step 2 B4 (LTE), mirroring fixed_input_ablation.contact_policy"},
}


def term_description(term: Any) -> str:
    keys = ("w_track", "track_decay", "w_continuity", "w", "w_smooth", "w_rows", "w_bvel", "rows")
    return ", ".join(f"{k}={getattr(term, k)}" for k in keys if hasattr(term, k))


class FixedDtTrajectoryOptimizer(TrajectoryOptimizer):
    """`TrajectoryOptimizer` whose re-timing returns a fixed state dt (inheritance only)."""

    fixed_dt: Optional[float] = None

    def compute_trajectory_dt(self, trajectory, epsilon: float = 1e-3, scale_dt: bool = True):
        if self.fixed_dt is None:
            return super().compute_trajectory_dt(trajectory, epsilon=epsilon, scale_dt=scale_dt)
        shape = trajectory.position.shape[:-2]
        return torch.full(shape, float(self.fixed_dt), device=trajectory.position.device,
                          dtype=trajectory.position.dtype)


#: knots per timing mode. `exact`: 21 knots x 4 interpolation steps -> 101 states at 1/30 s, so
#: every 2nd optimizer state is a 15 Hz chunk step (no resampling; cuRobo default is 16 knots).
#: `resample16`: cuRobo's 16 knots -> 81 states at 1/24 s, linearly resampled at k/15.
TIMING_KNOTS = {"exact": 21, "resample16": 16}


def transition_model_dict(n_knots: int) -> dict[str, Any]:
    """cuRobo's own `trajopt/transition_bspline_trajopt.yml` with `n_knots` replaced."""
    from curobo._src.util.config_io import load_yaml
    from curobo.content import get_task_configs_path

    d = load_yaml(str(get_task_configs_path() / "trajopt" / "transition_bspline_trajopt.yml"))
    d["transition_model_cfg"]["n_knots"] = int(n_knots)
    return d


def metrics_rollout_dict_without_self_collision() -> dict[str, Any]:
    """cuRobo's own `metrics_base.yml` without the self-collision constraint.

    `self_collision_check=False` only reaches the optimizer rollouts; the metrics rollout that
    decides `success` keeps its self-collision constraint, and with this sphere model (no
    self-collision ignore table) the finger spheres always overlap the palm spheres — every
    result would be "infeasible" for a reason the optimizer was told to ignore."""
    from curobo._src.util.config_io import load_yaml
    from curobo.content import get_task_configs_path

    d = load_yaml(str(get_task_configs_path() / "metrics_base.yml"))
    d["rollout"]["constraint_cfg"].pop("self_collision_cfg", None)
    return d


def result_diagnostics(solver, res, num_seeds: int) -> dict[str, Any]:
    """Why cuRobo calls the returned trajectory (in)feasible: the metrics rollout re-evaluated on
    the returned knots (the returned result carries no metrics). Outside the timed region."""
    out: dict[str, Any] = {}
    for key in ("position_error", "rotation_error"):
        v = getattr(res, key, None)
        if v is not None:
            out[f"curobo_{key}"] = float(np.asarray(v.detach().cpu()).reshape(-1)[0])
    sol = res.solution
    act = sol.reshape(-1, sol.shape[-2], sol.shape[-1])[:1].repeat(num_seeds, 1, 1).contiguous()
    with torch.no_grad():
        m = solver.metrics_rollout.compute_metrics_from_action(act)
    cons = {}
    for group in ("constraints", "hybrid_costs_constraints"):
        col = getattr(m.costs_and_constraints, group)
        for name, val in zip(col.names, col.values):
            a = np.asarray(val.detach().cpu())
            a0 = a.reshape(a.shape[0], a.shape[1], -1)[0]
            cons[name] = {"sum": float(a0.sum()), "n_steps_positive": int((a0 > 0).any(-1).sum())}
    out["curobo_constraints"] = cons
    out["curobo_costs_at_solution"] = cost_breakdown(m)
    return out


def cost_breakdown(metrics) -> dict[str, float]:
    """`{name: sum over the horizon}` of every cost / constraint value in a metrics result (seed 0),
    i.e. the terms whose sum cuRobo minimizes, in cuRobo's own scale."""
    out: dict[str, float] = {}
    for group in ("costs", "constraints", "hybrid_costs_constraints"):
        col = getattr(metrics.costs_and_constraints, group)
        for name, val in zip(col.names, col.values):
            a = np.asarray(val.detach().cpu())
            out[f"{group}/{name}"] = float(a.reshape(a.shape[0], -1)[0].sum())
    return out


class CuroboChunkSolver:
    """B0 (and later B1-B4) on one chunk at a time; the world and fingers change per chunk."""

    def __init__(self, spec: Mapping[str, Any], *, dev: Optional[DeviceCfg] = None,
                 extra_terms: Sequence[Any] = (), voxel_shape=(128, 128, 128),
                 voxel_size: float = 0.005, num_seeds: Optional[int] = None,
                 use_cuda_graph: bool = True, timing: str = "exact",
                 world: str = "free_layer", term_lambda: Optional[float] = None):
        """`world`: `free_layer` = step 1's B0 world (one target-excluded layer, nothing outside
        its box); `b0` = user decision 2 (box: target-excluded layer, outside: coarse);
        `b0m` = `b0` + our TO's obstacle margin (T63b, `b0m.py`);
        `lte` = B4's per-link world (`composed_world.py`)."""
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg

        if world not in WORLD_CHANGES:
            raise ValueError(f"world must be one of {sorted(WORLD_CHANGES)}")
        self.world = world
        self.extra_terms = tuple(extra_terms)
        #: the multiplier already folded into the extra terms' weights (`t63_curobo_compare.
        #: scaled_weights`), recorded only; `None` = not a lambda run (B0, step 1-2)
        self.term_lambda = None if term_lambda is None else float(term_lambda)
        self.spec = spec
        self.dev = dev or device_cfg()
        self.robot_cfg = build_robot_cfg(spec, self.dev)
        cost_type = (extensible_cost_manager_cfg(extra_terms) if extra_terms
                     else RobotCostManagerCfg)
        if timing not in TIMING_KNOTS:
            raise ValueError(f"timing must be one of {sorted(TIMING_KNOTS)}")
        self.timing = timing
        kwargs: dict[str, Any] = dict(
            robot=self.robot_cfg, self_collision_check=False, device_cfg=self.dev,
            use_cuda_graph=use_cuda_graph, cost_manager_config_instance_type=cost_type,
            transition_model=transition_model_dict(TIMING_KNOTS[timing]),
            metrics_rollout=metrics_rollout_dict_without_self_collision())
        if num_seeds is not None:
            kwargs["num_seeds"] = int(num_seeds)
        self.cfg = TrajectoryOptimizerCfg.create(**kwargs)
        if world == "free_layer":
            cache = {"voxel": {"layers": 1, "dims": [float(n) * voxel_size for n in voxel_shape],
                               "voxel_size": float(voxel_size)}}
            self.scene = SceneCollision.from_config(
                SceneCollisionCfg(device_cfg=self.dev, scene_model=None, cache=cache))
        else:
            from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision

            self.scene = ComposedSceneCollision.build(world, int(spec["n_spheres"]), self.dev,
                                                      capacity_shape=voxel_shape,
                                                      voxel_size=voxel_size)
        self.solver = FixedDtTrajectoryOptimizer(self.cfg, scene_collision_checker=self.scene)
        self.horizon = int(self.solver.horizon)
        self.n_knots = int(self.solver.action_horizon)
        self.state_dt = CHUNK_STEPS * CHUNK_DT / (self.horizon - 1)
        ratio = CHUNK_DT / self.state_dt
        #: the 15 Hz chunk grid lies on the optimizer's state grid (no resampling) iff integral
        self.grid_stride = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else None
        self.solver.fixed_dt = self.state_dt
        rollouts = list(self.solver.get_all_rollout_instances(include_optimizer_rollouts=True))
        rollouts += list(self.solver.additional_metrics_rollouts.values())
        self.kparams = kinematics_params_in([self.solver.kinematics, *rollouts])
        self.n_extra_managers = activate_extra_terms(rollouts) if extra_terms else 0
        self.n_solves = 0

    def settings_record(self) -> dict[str, Any]:
        changes = [dict(c) for c in B0_CHANGES]
        for c in changes:
            if c["setting"] == "world":
                c.update(WORLD_CHANGES[self.world])
        for t in self.extra_terms:
            changes.append({"setting": f"extra cost term {t.name}",
                            "value": term_description(t),
                            "why": "T63 step 2 (B1-B3): the TO's objective term, TO weights"
                                   + (f" x lambda {self.term_lambda:g} (user decision 3)"
                                      if self.term_lambda is not None else "")})
        return {"changes_from_default": changes, "world": self.world,
                "extra_terms": [t.name for t in self.extra_terms],
                "term_lambda": self.term_lambda,
                "timing": self.timing, "horizon_states": self.horizon, "n_knots": self.n_knots,
                "state_dt_s": self.state_dt, "chunk_grid_stride": self.grid_stride,
                "output": ("optimizer state grid, every chunk_grid_stride-th state (exact)"
                           if self.grid_stride else
                           "optimizer state grid linearly resampled at t = k/15"),
                "interpolation_dt_s": float(self.cfg.interpolation_dt),
                "num_seeds": int(self.cfg.num_seeds),
                "n_kinematics_params_updated_for_fingers": len(self.kparams),
                "joint_limits": joint_limits_now(self.robot_cfg),
                "self_collision_check": bool(self.cfg.self_collision_check),
                "optimizer_collision_activation_distance":
                    float(self.cfg.optimizer_collision_activation_distance),
                "position_tolerance": float(self.cfg.position_tolerance)}

    def set_world(self, z: Mapping[str, np.ndarray], T_base_frame: np.ndarray,
                  layer: str = "free", **lte) -> Any:
        """`free_layer`: load `layer`. `b0` / `b0m` / `lte`: `ComposedSceneCollision.load_chunk(z,
        T, **lte)` (`lte` takes `excl_curobo`, `target_margin_curobo`, `target_points`; `b0m`
        takes `b0m_rules`)."""
        if self.world != "free_layer":
            return self.scene.load_chunk(z, T_base_frame, **lte)
        from AG3S.trajopt.curobo_v2.robot_world import VoxelWorld

        grid = VoxelWorld.voxel_grid(layer, W.layer_arrays(z, layer), T_base_frame, self.dev)
        self.scene.data.voxels.load_batch([grid], env_idx=0)
        return {"layer": layer}

    def optimizer_breakdown(self, q_now14: np.ndarray, reference_chunk: np.ndarray
                            ) -> dict[str, dict[str, float]]:
        """The **optimizer's** objective by term (`cost_breakdown` of the first rollout that has a
        cost manager — cuRobo's c-space regularization, its scene-collision term at the optimizer
        weight, and our extra terms) at the seed (reference path knots) and at the last solution.
        Same batch shape as the optimizer (`num_seeds`), so no rollout buffer is reallocated
        (checked: solve -> breakdown -> solve gives bit-identical trajectories)."""
        rollouts = list(self.solver.get_all_rollout_instances(include_optimizer_rollouts=True))
        opt = next(r for r in rollouts if getattr(r, "cost_manager", None) is not None)
        ref = S.chunk_to_cspace(reference_chunk, self.spec)
        d = dict(dtype=self.dev.dtype, device=self.dev.device)
        n = int(self.cfg.num_seeds)
        seed = torch.as_tensor(knot_seed(q_now14, ref, self.n_knots)[None], **d)
        sol = self.last_result.solution.reshape(-1, self.n_knots, seed.shape[-1])[:1]
        out = {}
        for name, act in (("seed", seed), ("solution", sol)):
            with torch.no_grad():
                m = opt.compute_metrics_from_action(act.repeat(n, 1, 1).contiguous())
            out[name] = cost_breakdown(m)
        return out

    def curobo_costs_at(self, q_now14: np.ndarray, reference_chunk: np.ndarray) -> dict[str, float]:
        """cuRobo's cost breakdown (`cost_breakdown`) of the **seed** knots (the reference path) —
        the scale the extra terms compete with before any optimization."""
        ref = S.chunk_to_cspace(reference_chunk, self.spec)
        d = dict(dtype=self.dev.dtype, device=self.dev.device)
        seed = torch.as_tensor(knot_seed(q_now14, ref, self.n_knots)[None], **d)
        act = seed.repeat(int(self.cfg.num_seeds), 1, 1).contiguous()
        with torch.no_grad():
            m = self.solver.metrics_rollout.compute_metrics_from_action(act)
        return cost_breakdown(m)

    def set_fingers(self, values: Mapping[str, float]) -> None:
        set_finger_openings(self.kparams, self.spec, values)

    def solve(self, q_now14: np.ndarray, reference_chunk: np.ndarray) -> dict[str, Any]:
        ref = S.chunk_to_cspace(reference_chunk, self.spec)
        if ref.shape[0] != CHUNK_STEPS:
            raise ValueError(f"reference chunk has {ref.shape[0]} steps, expected {CHUNK_STEPS}")
        d = dict(dtype=self.dev.dtype, device=self.dev.device)
        q0 = torch.as_tensor(np.asarray(q_now14)[None], **d)
        start = JointState.from_position(q0, joint_names=self.solver.joint_names)
        start.velocity = torch.zeros_like(q0)
        start.acceleration = torch.zeros_like(q0)
        goal = JointState.from_position(torch.as_tensor(ref[-1][None], **d),
                                        joint_names=self.solver.joint_names)
        seed = torch.as_tensor(knot_seed(q_now14, ref, self.n_knots)[None, None], **d)
        num_seeds = int(self.cfg.num_seeds)
        dt = torch.full((1, num_seeds), self.state_dt, **d)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = self.solver.solve_cspace(goal, start, seed_traj=seed, dt=dt, finetune_attempts=0,
                                       finetune_dt_scale=1.0)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1e3
        self.n_solves += 1
        self.last_result = res
        js = res.js_solution
        names = list(getattr(js, "joint_names", None) or [])
        pos = js.position.detach().cpu().numpy()
        pos = pos.reshape(-1, pos.shape[-2], pos.shape[-1])[0]
        if pos.shape[-1] != len(self.solver.joint_names):
            # cuRobo appends the locked finger joints; keep the c-space joints by name
            pos = pos[:, [names.index(n) for n in self.solver.joint_names]]
        pos = np.ascontiguousarray(pos, np.float64)
        out_dt = float(np.asarray(js.dt.detach().cpu()).reshape(-1)[0]) \
            if getattr(js, "dt", None) is not None else float("nan")
        traj = sample_chunk_grid(pos, out_dt, CHUNK_DT, CHUNK_STEPS)
        success = bool(np.asarray(res.success.detach().cpu()).reshape(-1)[0])
        diag = result_diagnostics(self.solver, res, num_seeds)
        out = {
            "success": success,
            "chunk": None if traj is None else S.cspace_to_chunk(traj, reference_chunk),
            "cspace": traj,
            "n_states": int(pos.shape[0]), "solution_state_dt_s": out_dt,
            "resampled": self.grid_stride is None, **diag,
            # CudaEventTimer returns seconds (0.0 when curobo.runtime.cuda_event_timers is off)
            "solve_time_ms": float(res.solve_time) * 1e3,
            "total_time_ms": float(res.total_time) * 1e3,
            "wall_ms": wall_ms, "cold": self.n_solves == 1,
            "start_error_rad": float(np.abs(pos[0] - np.asarray(q_now14)).max()),
            "goal_error_rad": (None if traj is None
                               else float(np.abs(traj[-1] - ref[-1]).max())),
            "state_end_goal_error_rad": float(np.abs(pos[-1] - ref[-1]).max()),
            "finite": bool(np.all(np.isfinite(pos))),
        }
        return out


__all__ = ["B0_CHANGES", "CHUNK_DT", "CHUNK_STEPS", "CuroboChunkSolver",
           "FixedDtTrajectoryOptimizer", "WORLD_CHANGES", "cost_breakdown", "knot_seed"]
