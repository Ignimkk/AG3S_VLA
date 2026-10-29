"""AG3S-PRE-CORE-TO E3-b — E2's seeds solved in one batched cuRobo L-BFGS on the GPU.

`serve_safe --to-gpu-seeds curobo` (config `sqp.gpu_seeds`, default `"off"`), always with `--to-multi-seed N`. When
E2 triggers (the chunk's standard solve leaves a violation above `safety.violation_tolerance`), the **same N seeds**
(`multi_seed.seed_specs`, built by `TrajectoryOptimizer._build_seeds`) are optimized together — one problem per seed,
one batch — by cuRobo's own L-BFGS (`curobo.optim.LBFGSOpt`, line search and two-loop recursion on the GPU, CUDA
graphs) instead of N CPU SQP solves. Each of the N GPU results is then clipped onto the position box and the step-0
anchor (`limits.project_to_limits`, the T31 projection) and **re-measured by our own full-resolution check**
(`TrajectoryOptimizer._finish`, the same check every returned trajectory goes through); E2's `select_moving_first`
picks among the standard solve (candidate 0, untouched) and the N checked results. Nothing else changes: the chunks
where E2 does not trigger are the standard solve, and with `"off"` this module is not imported.

**Why (task §6 E3).** E2 resolved every chunk the standard solve left violated but took ~4 s per triggered chunk on
the CPU (16 SQP solves). E3-b asks whether the same seeds give the same result inside the 533 ms execution window
when solved in one batch on the GPU.

**T63 / T63b (task §2): cuRobo as the serving TO without a VLA-tracking term gave 0/10.** Here the tracking term is
the first cost term and has exactly the form of our `w_track` term; cuRobo's own c-space goal, regularisation and
re-timing are not used at all (only its kinematics, its collision kernels through T63's composed world, and its
L-BFGS).

## The problem solved per seed (decision variables = the H plan rows of the 14 optimized joints, as our SQP)

| term | value (per seed; `D = Q − R`, `R` = the policy reference) | weight |
|---|---|---|
| tracking | ``Σ_k decay^k Σ_j s_j D_kj²`` — `problem.objective`'s term, IFG-TOW scale `s` included | `cost.w_track` |
| smoothness | ``Σ_k ‖Q_{k+2} − 2Q_{k+1} + Q_k‖²`` | `cost.w_smooth` |
| continuity | ``Σ_{k<L} ‖Q_k − P_k‖²`` (previous refined chunk, cut to the look-ahead window as `_solve` does) | `cost.w_continuity` |
| deflection rate (T43 TA B) | ``Σ_k ‖D_{k+1} − D_k‖² + ‖D_0 − D_prev‖²`` (boundary only with the history) | `cost.w_deflection_rate` |
| joint limits | squared hinges: position box (inset `POSITION_INSET_RAD`), ``|Q_k − q_now|`` ≤ the step-0 anchor / rest envelope, velocity and acceleration ≤ `LIMIT_SHRINK` × the bound | `LIMIT_WEIGHT` |
| collision | cuRobo's sphere cost ``w · act(r + e + η − sdf)`` per sphere and row on the composed world below (`act`: 0 → quadratic over η → linear) | `cost.w_slack × COLLISION_WEIGHT_SCALE` |
| support planes | the same activation on our plane rows ``n·p − o − r`` (window rows only, as `_clearances_from`) | same |
| grasp goal (only when the anchor has goal rows on this horizon) | the anchor's own terms (`grasp_anchor.anchor_targets`): ``w_goal ‖p_k − g‖²`` on the goal rows (+ ``w_tcp ‖p_k − p_ref,k‖²`` when IFG-TOA's path term is on), `p` = the anchor link origin | the anchor's `w_goal` / `w_tcp` |

Not on the GPU (our check measures them; they can only make a GPU result lose the selection): the servo-accel
price (T43 TA C), E1-B's goal box, the path rows between waypoints (T43 Q), held / attached query points, candidate
(primitive) rows.

## World (T63's `composed_world`, mode `b0m`, loaded per chunk from the live field)

Inside the target-excluded layer's lattice the target-excluded layer (`esdf_free`), outside it the coarse layer; a chunk
without a target-excluded layer uses our main field ``min(coarse, fine)`` inside the fine lattice. Each sphere is
queried at radius ``r + esdf_margin + m``, ``m`` = `obstacle_margin − esdf_margin` (≥ 0) on the rows our `_obstacle_rows`
calls `obstacle` (`b0m.rules_from_scene` on the same `SceneSnapshot`; target and support rows get 0), so an obstacle row
clears ``max(esdf_margin, obstacle_margin)`` and every other row `esdf_margin` — our rule. **Gaps** (recorded in every
GPU record, `WORLD_GAPS`): every robot row sees the target-excluded layer inside its lattice (our TO gives it only to
the rows `target_free_mask` authorizes; a non-authorized link is not repelled by the target on the GPU); no
`manipulated_link_margin`; one finger opening per chunk (step 0 of the TO's parameter path, `serving.finger_values`)
instead of the per-row envelope; the destination-label and neighbour-contact relaxations are absent.

Every cuRobo import and build saves and restores `torch.backends.cuda.matmul.allow_tf32` (cuRobo switches TF32 on
globally on import; the rest of the server keeps its own setting).
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Mapping, Optional

import numpy as np

BACKENDS = ("off", "curobo")
#: Our optimized joints must be cuRobo's c-space in this order (`ChunkLayout.rby1`, chunk columns).
CUROBO_BASE_LINK = "link_torso_5"
#: Tool frames whose origin the goal term can read (cuRobo returns the poses of these links).
TOOL_FRAMES = ("ee_left", "ee_right")
#: cuRobo's L-BFGS settings: `trajopt/lbfgs_bspline_trajopt.yml` (line search, step scale, epsilon) with the history
#: and the iteration count sized for our 14 × H variables; `fixed_iters` so every chunk does the same work.
LBFGS_SETTINGS: dict[str, Any] = {
    "num_iters": 100, "inner_iters": 25, "history": 15, "epsilon": 0.01,
    "line_search_scale": [0.0, 0.1, 0.5, 1.0], "line_search_type": "approx_wolfe",
    "step_scale": 0.98, "initial_step_scale": 0.001, "fixed_iters": True, "cost_convergence": 0.0,
    "cost_delta_threshold": 0.0, "cost_relative_threshold": 0.0, "return_best_action": True,
    "store_debug": False, "sync_cuda_time": True, "use_cuda_kernel_line_search": True,
    "use_cuda_kernel_step_direction": True, "use_cuda_kernel_shared_buffers": True, "stable_mode": True,
}
#: CUDA graphs for the L-BFGS iterations (cuRobo's `GraphExecutor`). Per-chunk data is copied into fixed buffers.
CUDA_GRAPH = True
#: Collision weight = `cost.w_slack` × this (per metre in the linear part). Our merit prices a violation at w_slack per m.
COLLISION_WEIGHT_SCALE = 10.0
#: cuRobo's collision activation distance η (m): the cost starts η before the margin, so the optimum clears it.
ACTIVATION_M = 0.005
#: Weight of the squared joint-limit hinges (per rad²).
LIMIT_WEIGHT = 1.0e6
#: Velocity / acceleration / step-0 anchor bounds are enforced at this fraction (the hinges are soft).
LIMIT_SHRINK = 0.97
#: Inset of the position box (rad).
POSITION_INSET_RAD = 1.0e-3
#: A bound that is absent (non-finite) is replaced by this (rad), so the hinges stay finite.
BIG = 1.0e3
#: Support planes the buffer holds (our scenes carry one table plane).
MAX_PLANES = 8
#: Voxel cache of each world channel (`composed_world`): cells and voxel size (m) — the recorded layers are 128³.
WORLD_CAPACITY = (128, 128, 128)
WORLD_VOXEL_M = 0.005
#: What the GPU problem does not know (see the module docstring). Copied into every record.
WORLD_GAPS = (
    "target-excluded layer for every robot row inside its lattice (ours: only target_free_mask rows)",
    "no manipulated_link_margin (per-link target margin)",
    "one finger opening per chunk (TO parameter path step 0), not the per-row envelope",
    "no destination-label / neighbour-contact relaxation, no held / attached query points",
    "no servo-accel price, no E1-B goal box, no path rows between waypoints",
)


# =====================================================================================================
# robot spec of the live constraint model (T63 `export_inputs.robot_spec`, without the server argv)
# =====================================================================================================

def robot_spec(robot, layout, limits, dt: float) -> dict[str, Any]:
    """The T63 `robot_spec` dict for `curobo_v2.spec` / `robot_world`, read from the live `UrdfSphereChain` and the
    TO's own `JointLimits` (no model rebuild from argv: the spheres are the served ones whatever flags built them)."""
    names = [str(robot.joint_names[int(i)]) for i in layout.q_indices]
    spheres = [{"index": i, "link": str(link), "center": [float(v) for v in centre], "radius": float(radius)}
               for i, (link, centre, radius) in enumerate(robot._local_spheres)]
    fingers = {}
    sphere_links = {str(link) for link, _, _ in robot._local_spheres}
    for name in tuple(getattr(robot, "param_joint_names", ()) or ()):
        j = robot.model.joint_by_name(name)
        if str(j.child) not in sphere_links:
            continue        # a finger without spheres is not in cuRobo's tree (nothing to lock, nothing to shift)
        fingers[name] = {"type": j.type, "axis": [float(v) for v in j.axis], "parent": j.parent,
                         "child": j.child, "lower": j.lower, "upper": j.upper}
    import pathlib

    urdf = getattr(robot.model, "path", None)
    if urdf is None:
        # `UrdfModel` keeps no path; every constraint model here is built on the shipped RB-Y1 URDF (as the T63 exporter)
        from AG3S.ag3s.robot_models.urdf_sphere_chain import RBY1_URDF
        from AG3S.ag3s.runtime.asset_path import resolve_asset

        urdf = resolve_asset(RBY1_URDF, what="URDF")

    def finite(a, fill):
        a = np.asarray(a, np.float64)
        return [float(v) if np.isfinite(v) else float(fill) for v in a]

    return {
        "urdf_path": str(pathlib.Path(urdf).resolve()),
        "fk_root_link": "base", "curobo_base_link": CUROBO_BASE_LINK,
        "model_joint_names": [str(n) for n in robot.joint_names],
        "arm_joint_names": names,
        "finger_joints": fingers,
        "n_spheres": int(robot.n_spheres),
        "sphere_link_names": [str(n) for n in robot.sphere_link_names],
        "spheres": spheres,
        "limits": {"joint_names": names, "lower": finite(limits.lower, -BIG), "upper": finite(limits.upper, BIG),
                   "velocity": finite(np.asarray(limits.max_step) / dt, BIG),
                   "acceleration": finite(np.asarray(limits.max_step_change) / dt / dt, BIG), "dt": float(dt)},
    }


def available() -> tuple[bool, str]:
    """`(ok, why not)`: torch with a CUDA device and an importable cuRobo (`serve_safe --to-gpu-seeds` checks this
    before the checkpoint loads). TF32 matmul is left as it was."""
    try:
        import torch
    except ImportError as exc:
        return False, f"torch is not importable ({exc})"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        import curobo  # noqa: F401
        from curobo.optim import LBFGSOpt  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"cuRobo is not importable ({type(exc).__name__}: {exc})"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return True, ""


def _act(p, eta):
    """cuRobo's collision activation of a penetration-like `p = r + η − sdf`: 0, quadratic over (0, η], linear."""
    import torch

    return torch.where(p > eta, p - 0.5 * eta, torch.where(p > 0.0, p * p / (2.0 * eta), torch.zeros_like(p)))


# =====================================================================================================
# the rollout (cuRobo's rollout protocol as `LBFGSOpt` uses it; action = (B, H, nq) joint positions)
# =====================================================================================================

class _Rollout:
    """One cost evaluator. Each instance has its own cuRobo `Kinematics` (buffers sized by its batch)."""

    sum_horizon = True

    def __init__(self, batch: "CuroboSeedBatch", name: str):
        from curobo.kinematics import Kinematics

        self.b = batch
        self.name = name
        self.kin = Kinematics(batch.robot_cfg.kinematics)
        self._batch_size = 1
        self._key = object()          # this rollout's collision buffers in the composed world

    # -- protocol --
    @property
    def action_dim(self) -> int:
        return self.b.nq

    @property
    def action_horizon(self) -> int:
        return self.b.horizon

    @property
    def horizon(self) -> int:
        return self.b.horizon

    @property
    def action_bound_lows(self):
        return self.b.lo_box

    @property
    def action_bound_highs(self):
        return self.b.hi_box

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def update_batch_size(self, batch_size: int) -> None:
        self._batch_size = int(batch_size)

    def update_params(self, *args, **kwargs) -> bool:
        return True

    def reset_shape(self) -> bool:
        return True

    def reset_cuda_graph(self) -> bool:
        return True

    def evaluate_action(self, act_seq, **kwargs):
        from curobo._src.rollout.metrics import CostsAndConstraints, RolloutResult
        from curobo.types import JointState

        terms = self.terms(act_seq)
        cc = CostsAndConstraints()
        for name, value in terms.items():
            cc.costs.add(value.unsqueeze(-1), name)
        return RolloutResult(actions=act_seq, state=JointState.from_position(act_seq),
                             costs_and_constraints=cc)

    # -- the terms, `(B, H)` each --
    def terms(self, Q) -> dict[str, Any]:
        import torch
        from curobo.types import JointState

        b = self.b
        zeros = torch.zeros(Q.shape[:2], device=Q.device, dtype=Q.dtype)
        D = Q - b.ref
        out: dict[str, Any] = {"track": (b.w_track_rows * D * D).sum(-1)}
        if b.w_smooth > 0.0 and b.horizon > 2:
            a = Q[:, 2:] - 2.0 * Q[:, 1:-1] + Q[:, :-2]
            out["smooth"] = torch.cat([zeros[:, :2], b.w_smooth * (a * a).sum(-1)], dim=1)
        if b.w_continuity > 0.0:
            out["continuity"] = b.w_continuity * b.prev_mask * ((Q - b.prev) ** 2).sum(-1)
        if b.w_rate > 0.0:
            first = b.has_dprev * ((D[:, 0] - b.dprev) ** 2).sum(-1)
            rate = ((D[:, 1:] - D[:, :-1]) ** 2).sum(-1)
            out["deflection_rate"] = b.w_rate * torch.cat([first.unsqueeze(1), rate], dim=1)
        relu = torch.relu
        lim = relu(Q - b.hi) ** 2 + relu(b.lo - Q) ** 2 + relu((Q - b.q0).abs() - b.env) ** 2
        lim = lim.sum(-1)
        v = (relu((Q[:, 1:] - Q[:, :-1]).abs() - b.vel) ** 2).sum(-1)
        lim = lim + torch.cat([zeros[:, :1], v], dim=1)
        if b.horizon > 2:
            acc = (relu((Q[:, 2:] - 2.0 * Q[:, 1:-1] + Q[:, :-2]).abs() - b.accel) ** 2).sum(-1)
            lim = lim + torch.cat([zeros[:, :2], acc], dim=1)
        out["limits"] = LIMIT_WEIGHT * lim
        state = self.kin.compute_kinematics(JointState.from_position(Q.contiguous(), joint_names=b.joint_names))
        sph = state.robot_spheres.view(Q.shape[0], Q.shape[1], -1, 4)
        r = sph[..., 3:4]
        on = r > 0.0
        q_sph = torch.cat([sph[..., :3], torch.where(on, r + b.inflate, r)], dim=-1)
        c = b.world.get_sphere_distance_raw(q_sph, self._key, b.w_coll, b.eta, None, True)
        out["collision"] = c.view(Q.shape[0], Q.shape[1], -1).sum(-1) * b.coll_rows
        # support planes, robot base frame (elementwise, not a TF32 matmul)
        xyz = sph[..., :3]
        base = (xyz.unsqueeze(-2) * b.R).sum(-1) + b.t                     # (B, H, S, 3)
        s = (base.unsqueeze(-2) * b.plane_n).sum(-1) - b.plane_o - r        # (B, H, S, P)
        pen = _act(b.eta - s, b.eta) * b.plane_on * on.to(Q.dtype)
        out["planes"] = b.w_coll * pen.sum(dim=(-1, -2)) * b.plane_rows
        # grasp goal / palm path on the anchor link origin (one-hot over the tool frames)
        p = (state.tool_poses.position.view(Q.shape[0], Q.shape[1], -1, 3) * b.tool_onehot).sum(2)
        pb = (p.unsqueeze(-2) * b.R).sum(-1) + b.t                          # (B, H, 3)
        d2 = ((pb.unsqueeze(2) - b.anchor_targets) ** 2).sum(-1)            # (B, H, 2)
        out["grasp_anchor"] = (b.anchor_weights * d2).sum(-1)
        return out


# =====================================================================================================
# the batched solver of one optimizer (one horizon, one seed count)
# =====================================================================================================

class CuroboSeedBatch:
    """N seeds × H rows on the GPU for one `TrajectoryOptimizer` (its horizon and its config)."""

    def __init__(self, optimizer, n_seeds: int, horizon: int, *, device: str = "cuda:0"):
        import torch

        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        t0 = time.perf_counter()
        try:
            self._build(optimizer, int(n_seeds), int(horizon), device)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        self.build_ms = (time.perf_counter() - t0) * 1e3
        self.n_solves = 0

    def _build(self, optimizer, n: int, H: int, device: str) -> None:
        import torch

        from curobo.optim import LBFGSOpt, LBFGSOptCfg

        from AG3S.trajopt.curobo_v2 import spec as S
        from AG3S.trajopt.curobo_v2.composed_world import ComposedSceneCollision
        from AG3S.trajopt.curobo_v2.robot_world import device_cfg, kinematics_params_in

        cfg = optimizer.config
        robot = optimizer.linearizer.robot_model
        self.layout = optimizer.layout
        self.limits = optimizer.limits
        self.n, self.horizon, self.nq = n, H, int(optimizer.layout.nq_opt)
        self.lookahead_from = optimizer.lookahead_from
        self.spec = robot_spec(robot, optimizer.layout, optimizer.limits, float(cfg.horizon.dt))
        outside = [s["link"] for s in self.spec["spheres"]
                   if not _descends(robot, s["link"], CUROBO_BASE_LINK)]
        if outside:
            raise ValueError(f"spheres on links outside the {CUROBO_BASE_LINK} subtree: {sorted(set(outside))}")
        self.dev = device_cfg(device)
        # `robot_world.build_robot_cfg`, with both palms as tool frames even when they carry no sphere (the goal term
        # reads their origin; T63's dict names them only when they have spheres)
        from curobo._src.types.robot import RobotCfg

        from AG3S.trajopt.curobo_v2.robot_world import apply_limits

        cfg_dict = S.robot_cfg_dict(self.spec)
        urdf_links = set(getattr(robot.model, "links", ()) or ())
        cfg_dict["robot_cfg"]["kinematics"]["tool_frames"] = [f for f in TOOL_FRAMES if f in urdf_links] or \
            cfg_dict["robot_cfg"]["kinematics"]["tool_frames"]
        self.robot_cfg = RobotCfg.create(cfg_dict, self.dev)
        apply_limits(self.robot_cfg, self.spec)
        self.perm = S.sphere_permutation(self.spec)
        self.n_spheres = int(self.spec["n_spheres"])
        self.world = ComposedSceneCollision.build("b0m", self.n_spheres, self.dev, capacity_shape=WORLD_CAPACITY,
                                                  voxel_size=WORLD_VOXEL_M)
        self.tool_frames = list(cfg_dict["robot_cfg"]["kinematics"]["tool_frames"])
        d = dict(device=self.dev.device, dtype=self.dev.dtype)
        z = lambda *shape: torch.zeros(shape, **d)  # noqa: E731
        c = cfg.cost
        # per-chunk buffers (copied into, never re-bound: the CUDA graphs read these storages)
        self.ref, self.w_track_rows, self.prev = z(1, H, self.nq), z(1, H, self.nq), z(1, H, self.nq)
        self.prev_mask, self.dprev, self.has_dprev = z(1, H), z(1, self.nq), z(1)
        self.q0, self.env = z(1, 1, self.nq), z(1, H, self.nq)
        self.lo, self.hi, self.vel, self.accel = z(self.nq), z(self.nq), z(self.nq), z(self.nq)
        self.lo_box, self.hi_box = z(self.nq), z(self.nq)
        self.coll_rows, self.plane_rows = z(1, H), z(1, H)
        self.R, self.t = z(3, 3), z(3)
        self.plane_n, self.plane_o, self.plane_on = z(MAX_PLANES, 3), z(MAX_PLANES), z(MAX_PLANES)
        self.inflate = z(1)
        self.tool_onehot = z(1, 1, max(len(self.tool_frames), 1), 1)
        self.anchor_weights, self.anchor_targets = z(1, H, 2), z(1, H, 2, 3)
        self.w_coll = torch.full((1,), float(c.w_slack) * COLLISION_WEIGHT_SCALE, **d)
        self.eta = torch.full((1,), ACTIVATION_M, **d)
        # constants of this optimizer's config (Python constants in the captured graph)
        self.w_smooth = float(c.w_smooth)
        self.w_continuity = float(c.w_continuity)
        self.w_rate = float(getattr(c, "w_deflection_rate", 0.0) or 0.0)
        self.joint_names = list(self.spec["arm_joint_names"])
        self._set_limits()
        self.rollouts = [_Rollout(self, "main"), _Rollout(self, "initial")]
        self.eval_rollout = _Rollout(self, "eval")
        if self.rollouts[0].kin.joint_names != self.joint_names:
            raise ValueError(f"cuRobo c-space {self.rollouts[0].kin.joint_names} != our joints {self.joint_names}")
        tool_ok = list(self.rollouts[0].kin.tool_frames)
        self.tool_index = {f: tool_ok.index(f) for f in self.tool_frames if f in tool_ok}
        self.kparams = kinematics_params_in([r.kin for r in self.rollouts] + [self.eval_rollout.kin])
        ocfg = LBFGSOptCfg(**LBFGSOptCfg.create_data_dict(dict(LBFGS_SETTINGS), self.dev))
        self.opt = LBFGSOpt(ocfg, self.rollouts, use_cuda_graph=CUDA_GRAPH)
        self.opt.update_num_problems(n)
        self.num_particles = int(ocfg.num_particles)

    def _set_limits(self) -> None:
        import torch

        lim = self.limits
        lo = np.asarray(lim.lower, np.float64)
        hi = np.asarray(lim.upper, np.float64)
        lo_c = np.where(np.isfinite(lo), lo + POSITION_INSET_RAD, -BIG)
        hi_c = np.where(np.isfinite(hi), hi - POSITION_INSET_RAD, BIG)
        from AG3S.trajopt.limits import velocity_rows

        step = np.asarray(lim.max_step, np.float64)
        vel = (np.where(np.isfinite(step), step * LIMIT_SHRINK, BIG) if velocity_rows(lim)
               else np.full(self.nq, BIG))
        acc = np.asarray(lim.max_step_change, np.float64)
        acc = np.where(np.isfinite(acc), acc * LIMIT_SHRINK, BIG) if np.all(np.isfinite(acc)) else np.full(self.nq, BIG)
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), dtype=self.dev.dtype)  # noqa: E731
        self.lo.copy_(t(lo_c))
        self.hi.copy_(t(hi_c))
        self.vel.copy_(t(vel))
        self.accel.copy_(t(acc))
        # the line search's step scale only (cuRobo `ActionBounds`), not a constraint
        self.lo_box.copy_(t(np.where(np.isfinite(lo), lo, -np.pi)))
        self.hi_box.copy_(t(np.where(np.isfinite(hi), hi, np.pi)))

    # -- per chunk ------------------------------------------------------------------------------------
    def load(self, **kw) -> dict[str, Any]:
        """Copy one chunk's data into the buffers and the world (`_load`). Returns what was loaded (record)."""
        import torch

        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        try:
            return self._load(**kw)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def _load(self, *, reference, q_now, scene, previous_chunk, deflection_history, joint_track_scale,
              anchor_terms, envelope, robot, linearizer, config) -> dict[str, Any]:
        import torch

        from AG3S.trajopt.curobo_v2 import b0m as B0M
        from AG3S.trajopt.curobo_v2.robot_world import set_finger_openings
        from AG3S.trajopt.curobo_v2.serving import finger_values, live_layers

        H, nq = self.horizon, self.nq
        t = lambda a: torch.as_tensor(np.asarray(a, np.float64), dtype=self.dev.dtype)  # noqa: E731
        R = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        q0 = q_now[np.asarray(self.layout.q_indices)]
        c = config.cost
        decay = float(c.track_decay) ** np.arange(H, dtype=np.float64)
        scale = (np.ones(nq) if joint_track_scale is None
                 else np.asarray(joint_track_scale, np.float64).reshape(-1))
        self.ref.copy_(t(R.T[None]))
        self.w_track_rows.copy_(t((float(c.w_track) * decay[:, None] * scale[None, :])[None]))
        prev = np.zeros((H, nq))
        mask = np.zeros(H)
        if previous_chunk is not None and float(c.w_continuity) > 0.0:
            P = np.asarray(previous_chunk, np.float64)
            L = min(P.shape[1], H)
            prev[:L] = P[:, :L].T
            mask[:L] = 1.0
        self.prev.copy_(t(prev[None]))
        self.prev_mask.copy_(t(mask[None]))
        if deflection_history is not None and self.w_rate > 0.0:
            self.dprev.copy_(t(np.asarray(deflection_history, np.float64).reshape(1, -1)))
            self.has_dprev.fill_(1.0)
        else:
            self.dprev.zero_()
            self.has_dprev.zero_()
        self.q0.copy_(t(q0.reshape(1, 1, -1)))
        env = np.full((H, nq), BIG)
        step = np.asarray(self.limits.max_step, np.float64)
        if np.all(np.isfinite(step)):
            env[0] = step * LIMIT_SHRINK
        if envelope is not None:
            e = np.asarray(envelope, np.float64)[:, :H].T
            env[:e.shape[0]] = np.minimum(env[:e.shape[0]], e * LIMIT_SHRINK)
        self.env.copy_(t(env[None]))
        rows = np.ones(H)
        self.coll_rows.copy_(t(rows[None]))
        prow = np.ones(H)
        if self.lookahead_from is not None and 0 < int(self.lookahead_from) < H:
            prow[int(self.lookahead_from):] = 0.0   # a support plane is never a look-ahead row (`_clearances_from`)
        self.plane_rows.copy_(t(prow[None]))
        # frame: cuRobo's base = link_torso_5; the world, the planes and the goal are in our base frame
        T = np.asarray(robot.link_pose(q_now, CUROBO_BASE_LINK), np.float64)
        self.R.copy_(t(T[:3, :3]))
        self.t.copy_(t(T[:3, 3]))
        n = np.zeros((MAX_PLANES, 3))
        o = np.zeros(MAX_PLANES)
        on = np.zeros(MAX_PLANES)
        act = np.asarray(getattr(scene, "plane_active", np.zeros(0)), bool).reshape(-1)
        idx = np.flatnonzero(act)
        if idx.size > MAX_PLANES:
            raise ValueError(f"{idx.size} active support planes > {MAX_PLANES}")
        n[:idx.size] = np.asarray(scene.plane_normal, np.float64)[idx]
        o[:idx.size] = np.asarray(scene.plane_offset, np.float64)[idx]
        on[:idx.size] = 1.0
        self.plane_n.copy_(t(n))
        self.plane_o.copy_(t(o))
        self.plane_on.copy_(t(on))
        # margins: r + esdf_margin everywhere, + (obstacle_margin - esdf_margin) on obstacle rows (b0m)
        esdf_m = float(getattr(scene, "esdf_margin", 0.0) or 0.0)
        self.inflate.fill_(esdf_m)
        field = getattr(scene, "esdf", None)
        field = getattr(field, "_field", field)     # offline: `ta_offline._DestinationLabels` wraps the field
        if field is None:
            raise ValueError("scene has no ESDF field")
        rules = B0M.rules_from_scene(scene, self.n_spheres)
        om = float(rules["obstacle_margin"])
        rules["obstacle_margin"] = max(0.0, om - esdf_m)
        fingers, fsrc = finger_values(linearizer, robot)
        # only the fingers cuRobo carries (with spheres), clipped into their URDF range (`finger_sphere_table` refuses
        # a value outside it; a measured opening can sit a hair beyond the stop)
        clipped = {}
        for name, info in self.spec["finger_joints"].items():
            v = float(fingers.get(name, 0.0))
            lo_f = -np.inf if info.get("lower") is None else float(info["lower"])
            hi_f = np.inf if info.get("upper") is None else float(info["upper"])
            clipped[name] = float(min(max(v, lo_f), hi_f))
        set_finger_openings(self.kparams, self.spec, clipped)
        info = self.world.load_layers(live_layers(field), T, b0m_rules=B0M.to_curobo(rules, self.perm))
        # anchor terms (`grasp_anchor.anchor_targets`), on the anchor link's origin
        w = np.zeros((H, 2))
        tg = np.zeros((H, 2, 3))
        oh = np.zeros(max(len(self.tool_frames), 1))
        goal_note = None
        if anchor_terms is not None:
            link = anchor_terms["link"]
            if link in self.tool_index:
                w, tg = anchor_terms["weights"], anchor_terms["targets"]
                oh[self.tool_index[link]] = 1.0
            else:
                goal_note = f"anchor link {link} is not a cuRobo tool frame {self.tool_frames}: no goal term"
        self.anchor_weights.copy_(t(np.asarray(w, np.float64)[None]))
        self.anchor_targets.copy_(t(np.asarray(tg, np.float64)[None]))
        self.tool_onehot.copy_(t(oh.reshape(1, 1, -1, 1)))
        return {"world": info, "finger_values": clipped, "finger_source": fsrc,
                "obstacle_margin_m": om, "esdf_margin_m": esdf_m,
                "goal_term": bool(anchor_terms is not None and goal_note is None),
                **({"goal_note": goal_note} if goal_note else {}),
                # AG3S-PRE-CORE-TO SCM — only when on: the B0m world has one obstacle margin for every row, so the
                # palm-shell rows are seeded at `obstacle_margin` (our check / selection reads `palm_shell_margin`).
                **({"palm_shell_margin_gap": {
                    "palm_shell_margin_m": float(scene.palm_shell_margin), "gpu_shell_rows_margin_m": om}}
                   if getattr(scene, "palm_shell_margin", None) is not None else {})}

    def solve(self, seeds: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """`seeds (N, nq, H)` → the L-BFGS best actions `(N, nq, H)` (float64) and timing / cost record."""
        import torch

        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        try:
            x0 = torch.as_tensor(np.ascontiguousarray(np.transpose(np.asarray(seeds, np.float64), (0, 2, 1))),
                                 dtype=self.dev.dtype, device=self.dev.device).contiguous()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.opt.reinitialize(x0)
            out = self.opt.optimize(x0)
            torch.cuda.synchronize()
            opt_ms = (time.perf_counter() - t0) * 1e3
            with torch.no_grad():
                terms = self.eval_rollout.terms(torch.cat([x0, out], dim=0))
            res = out.detach().to(torch.float64).cpu().numpy().transpose(0, 2, 1).copy()
            costs = {k: v.sum(-1).detach().cpu().numpy() for k, v in terms.items()}
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        self.n_solves += 1
        n = self.n
        rec = {"optimize_ms": round(opt_ms, 2), "cold": self.n_solves == 1,
               "cost_seed": [round(float(sum(v[i] for v in costs.values())), 4) for i in range(n)],
               "cost_out": [round(float(sum(v[n + i] for v in costs.values())), 4) for i in range(n)],
               "collision_out": [round(float(costs["collision"][n + i]), 5) for i in range(n)],
               "limits_out": [round(float(costs["limits"][n + i]), 6) for i in range(n)],
               "finite": bool(np.all(np.isfinite(res)))}
        return res, rec

    def settings_record(self) -> dict[str, Any]:
        return {"optimizer": "cuRobo LBFGSOpt", **{k: LBFGS_SETTINGS[k] for k in (
                    "num_iters", "inner_iters", "history", "line_search_scale", "step_scale")},
                "cuda_graph": CUDA_GRAPH, "n_seeds": self.n, "horizon": self.horizon,
                "num_particles": self.num_particles, "world": "composed_world b0m",
                "collision_weight": float(self.w_coll[0]), "activation_m": ACTIVATION_M,
                "limit_weight": LIMIT_WEIGHT, "limit_shrink": LIMIT_SHRINK, "tool_frames": self.tool_frames,
                "n_spheres": self.n_spheres, "build_ms": round(self.build_ms, 1), "gaps": list(WORLD_GAPS)}


def _descends(robot, link: str, root: str) -> bool:
    """Is `link` `root` or below it in the URDF tree (`robot.model` joints, child -> parent)?"""
    parent = {}
    for j in getattr(robot.model, "joints", ()):  # UrdfModel.joints: UrdfJoint(parent, child, ...)
        parent[str(j.child)] = str(j.parent)
    cur, seen = str(link), set()
    while cur not in seen:
        if cur == root:
            return True
        seen.add(cur)
        if cur not in parent:
            return False
        cur = parent[cur]
    return False


# =====================================================================================================
# entry points
# =====================================================================================================

def batch_for(optimizer, n_seeds: int, horizon: int) -> CuroboSeedBatch:
    """The optimizer's batch solver for (N, H), built once (cached on the optimizer)."""
    cache = optimizer.__dict__.setdefault("_gpu_batches", {})
    key = (int(n_seeds), int(horizon))
    if key not in cache:
        try:
            cache[key] = CuroboSeedBatch(optimizer, n_seeds, horizon)
        except Exception as exc:
            cache[key] = exc          # a build that failed is not retried on every chunk
    if isinstance(cache[key], Exception):
        raise RuntimeError(f"cuRobo seed batch could not be built: {type(cache[key]).__name__}: {cache[key]}")
    return cache[key]


def anchor_terms_for(optimizer, anchor, reference, q_now, goal) -> Optional[dict[str, Any]]:
    """The anchor's weights / targets on this horizon (`grasp_anchor.anchor_targets`), only when it has goal rows."""
    if anchor is None or goal is None:
        return None
    from AG3S.trajopt.grasp_anchor import anchor_targets

    p_ref = goal["fk"](np.asarray(reference, np.float64)[:, :goal["fk"].horizon], q_now)[0]
    w, tg = anchor_targets(anchor, p_ref)
    return {"link": str(anchor.link), "weights": np.asarray(w, np.float64), "targets": np.asarray(tg, np.float64)}


def solve_seeds(optimizer, seeds, reference, q_now, scene, kw: Mapping[str, Any], anchor, scale, envelope,
                standard, goal, first_pass=None) -> dict[str, Any]:
    """E3-b — the N seeds on the GPU, each result clipped and re-checked by `optimizer._finish`.

    Returns `{"backend", "results": [(TrajOptResult, check_ms)] or None, "record"}`; `results is None` = the GPU path
    failed (the caller solves the seeds on the CPU, as E2) and `record["error"]` says why.

    `first_pass = (order_key, resolves)` (`sqp.gpu_seeds_check = "first-pass"`): the clipped results with a non-`None`
    `order_key(i, Q)` are checked in that key's order until `resolves(result)`; the others are `None` in `results`.
    When none resolves, every result not yet checked is checked too. `record["first_pass"]` lists what was checked."""
    from AG3S.trajopt.limits import project_to_limits

    started = time.perf_counter()
    reference = np.asarray(reference, np.float64)
    q_now = np.asarray(q_now, np.float64).reshape(-1)
    H = int(reference.shape[1])
    rec: dict[str, Any] = {"backend": "curobo"}
    try:
        cold = (int(len(seeds)), H) not in optimizer.__dict__.get("_gpu_batches", {})
        t0 = time.perf_counter()
        batch = batch_for(optimizer, len(seeds), H)
        rec["build_ms"] = round((time.perf_counter() - t0) * 1e3, 1) if cold else 0.0
        prev = kw.get("previous_chunk")
        if prev is not None and optimizer.lookahead_from is not None:
            prev = np.asarray(prev, np.float64)[:, :optimizer.lookahead_from]   # as `_solve`
        t0 = time.perf_counter()
        loaded = batch.load(reference=reference, q_now=q_now, scene=scene, previous_chunk=prev,
                            deflection_history=kw.get("deflection_history"), joint_track_scale=scale,
                            anchor_terms=anchor_terms_for(optimizer, anchor, reference, q_now, goal),
                            envelope=envelope, robot=optimizer.linearizer.robot_model,
                            linearizer=optimizer.linearizer, config=optimizer.config)
        load_ms = (time.perf_counter() - t0) * 1e3
        out, srec = batch.solve(np.stack([np.asarray(Q0, np.float64) for _, _, Q0 in seeds]))
        gpu_ms = (time.perf_counter() - t0) * 1e3
        if not srec["finite"]:
            raise FloatingPointError("cuRobo returned a non-finite trajectory")
        rec.update(load_ms=round(load_ms, 2), gpu_ms=round(gpu_ms, 2), **srec,
                   **{k: v for k, v in loaded.items() if k != "world"}, world=loaded["world"],
                   settings=batch.settings_record())
    except Exception as exc:  # noqa: BLE001 - a GPU fault falls back to E2's CPU seeds, never kills the server
        rec.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=6))
        rec["total_ms"] = round((time.perf_counter() - started) * 1e3, 2)
        return {"backend": "sqp (cuRobo failed)", "results": None, "record": rec}
    q0 = q_now[optimizer.layout.q_indices]
    n = int(out.shape[0])
    results: list = [None] * n
    clip = []
    clipped = []
    clip_ms = []
    t_check = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        Q = out[i]
        P = project_to_limits(Q, optimizer.limits, q0, envelope=envelope)
        if P is not None:
            clip.append(round(float(np.abs(P - Q).max(initial=0.0)), 6))
            Q = P
        else:
            clip.append(None)
        clipped.append(Q)
        clip_ms.append((time.perf_counter() - t0) * 1e3)

    def run(i):
        t0 = time.perf_counter()
        res = check(optimizer, clipped[i], reference, q_now, scene, kw, standard, envelope,
                    notes=[f"E3-b: cuRobo L-BFGS result of seed {i + 1}, re-checked at full resolution"])
        results[i] = (res, clip_ms[i] + (time.perf_counter() - t0) * 1e3)
        return res

    if first_pass is None:
        for i in range(n):
            run(i)
    else:
        order_key, resolves = first_pass
        keyed = [(order_key(i, clipped[i]), i) for i in range(n)]
        order = [i for k, i in sorted(e for e in keyed if e[0] is not None)]
        hit = None
        for i in order:
            if resolves(run(i)):
                hit = i
                break
        if hit is None:
            for i in range(n):
                if results[i] is None:
                    run(i)
        rec["first_pass"] = {"order": [i + 1 for i in order], "hit": None if hit is None else hit + 1,
                             "n_checked": int(sum(r is not None for r in results))}
    rec.update(check_ms=round((time.perf_counter() - t_check) * 1e3, 2), clip_rad=clip,
               total_ms=round((time.perf_counter() - started) * 1e3, 2))
    return {"backend": "curobo", "results": results, "record": rec}


def check(optimizer, trajectory, reference, q_now, scene, kw: Mapping[str, Any], standard, envelope, *,
          notes: Optional[list[str]] = None):
    """Our full-resolution check of an external plan (`TrajectoryOptimizer._finish`, `iterations = 0`), in the state
    the chunk's standard solve left (attached points, path floors, T43 TA history are those of this chunk)."""
    prev = kw.get("previous_chunk")
    if prev is not None and optimizer.lookahead_from is not None:
        prev = np.asarray(prev, np.float64)[:, :optimizer.lookahead_from]
    started = time.perf_counter()
    res = optimizer._finish(np.asarray(trajectory, np.float64), reference, q_now, scene, kw.get("template"), 0, 0.0,
                            float(standard.reference_violation), prev, list(notes or []), started,
                            bool(kw.get("geometry_certified", True)), states=None, timing={}, qp_iterations=0,
                            time_budget_hit=False, selection=None)
    if envelope is not None:
        res.metrics["rest_start"] = optimizer._rest_start_record(trajectory, reference, q_now, envelope)
    return res


def warm_up(optimizer, n_seeds: Optional[int] = None) -> list[dict[str, Any]]:
    """Build the batch solvers of `optimizer` (and of its look-ahead window optimizer) and run one synthetic solve each,
    so the CUDA graphs are captured before the first real chunk. Returns one record per horizon."""
    out = []
    for opt in (optimizer, getattr(optimizer, "window_optimizer", None)):
        if opt is None or int(getattr(opt.config.sqp, "multi_seed", 0) or 0) <= 0:
            continue
        n = int(n_seeds or opt.config.sqp.multi_seed)
        H = int(opt.horizon)
        t0 = time.perf_counter()
        batch = batch_for(opt, n, H)
        out.append({"horizon": H, "n_seeds": n, **_synthetic_solve(opt, batch),
                    "warm_ms": round((time.perf_counter() - t0) * 1e3, 1), "build_ms": round(batch.build_ms, 1)})
    return out


def _synthetic_solve(optimizer, batch: CuroboSeedBatch) -> dict[str, Any]:
    """One solve on an obstacle-free synthetic world (as `serving.CuroboB0Backend.warm_up`)."""
    import types

    lim = optimizer.limits
    lo, hi = np.asarray(lim.lower, np.float64), np.asarray(lim.upper, np.float64)
    q = np.clip(np.zeros(batch.nq), np.where(np.isfinite(lo), lo + 0.05, -1.0), np.where(np.isfinite(hi), hi - 0.05, 1.0))
    robot = optimizer.linearizer.robot_model
    q_model = np.zeros(len(robot.joint_names))
    q_model[np.asarray(optimizer.layout.q_indices)] = q
    T = np.asarray(robot.link_pose(q_model, CUROBO_BASE_LINK), np.float64)

    def layer(shape, vs, origin):
        return types.SimpleNamespace(grid=types.SimpleNamespace(origin=np.asarray(origin, np.float64), voxel_size=vs,
                                                                shape=shape),
                                     distance_grid=np.full(shape, 0.5, np.float32))

    fine = layer((64, 64, 64), 0.005, T[:3, 3] + np.array([0.3, -0.16, -0.16]))
    coarse = layer((48, 48, 48), 0.02, T[:3, 3] + np.array([-0.2, -0.48, -0.48]))
    field = types.SimpleNamespace(layers=(coarse, fine), target_free_layers=(fine,), grid=fine.grid)
    # every branch of the per-chunk load: a target cloud far from the arm (the b0m target channel and its grid), a
    # support plane 1 m below the torso, one active plane row (as `serving.CuroboB0Backend.warm_up`)
    far = T[:3, 3] + np.array([0.6, 0.0, 0.6])
    plane = np.array([[0.0, 0.0, 1.0, float(T[2, 3]) - 1.0]])
    scene = types.SimpleNamespace(esdf=field, esdf_margin=0.0, obstacle_margin=0.01, plane_active=np.ones(1, bool),
                                  plane_normal=plane[:, :3], plane_offset=plane[:, 3],
                                  manipulated_points=far[None] + 0.01 * np.eye(3), support_planes=plane)
    ref = np.tile(q[:, None], (1, batch.horizon))
    batch.load(reference=ref, q_now=q_model, scene=scene, previous_chunk=None, deflection_history=None,
               joint_track_scale=None, anchor_terms=None, envelope=None, robot=robot,
               linearizer=optimizer.linearizer, config=optimizer.config)
    _, rec = batch.solve(np.tile(ref[None], (batch.n, 1, 1)))
    return {"optimize_ms": rec["optimize_ms"], "finite": rec["finite"]}


__all__ = ["BACKENDS", "CuroboSeedBatch", "available", "LBFGS_SETTINGS", "WORLD_GAPS", "anchor_terms_for", "batch_for", "check",
           "robot_spec", "solve_seeds", "warm_up"]
