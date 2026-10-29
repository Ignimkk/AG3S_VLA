"""T63b — cuRobo v2 B0 in the serving path (`serve_safe --to-backend curobo-b0`, default `sqp`).

`--to-backend curobo-b0m` (T63b user decision 2) is the same with the world `b0m`: B0 + our TO's
`obstacle_margin` on the rows our TO's `_obstacle_rows` calls `obstacle` (`b0m.py`). Its rule inputs
— contact rows, support planes, the manipulated points / held spheres, the margin and tolerances —
are read each chunk from the **same live `SceneSnapshot`** our certification then checks against
(`b0m.rules_from_scene`), so the margin comes from the server's own config (`--obstacle-margin`).

Only the optimizer step is swapped. `TrajOptChunkRefiner.refine` gets its scene, `q_now`, finger
state, continuity and history exactly as before; with a backend attached it calls
`CuroboB0Backend.refine` instead of `TrajectoryOptimizer.solve`. That method

1. runs the T63 offline B0 (`solver.CuroboChunkSolver`, world `b0`, timing `exact`) on the **live**
   inputs of this chunk (table below) and takes cuRobo's 50 chunk steps as the refined chunk;
2. hands the plan window of that chunk to **our** certification — `certify`, which is the prologue of
   `TrajectoryOptimizer._solve` (continuity cut to the look-ahead window, T43 TA history, empty-scene
   passthrough, attached points, reference violation, servo / path state) followed by the
   optimizer's own `_finish` (full-row clearance with the same contact policy, status, notes,
   metrics). The SQP loop and the look-ahead gate are the only parts of `_solve` not run: the gate
   chooses *what the SQP optimizes*, not how the result is checked.

Everything after the refiner — verdict reasons, HOLD, follow-TO, records — is the existing
`SafePolicy` code. A cuRobo failure (`success = False`) is *not* a HOLD by itself; its trajectory is
certified like any other. A non-finite or missing cuRobo trajectory, or an exception, returns the
reference with status `solver_failed` (verdict `unverified` -> HOLD) and says why.

| offline B0 input (T63, from the record) | live input here (what the record stores for the same chunk) |
|---|---|
| `reference_chunk` (50, 16) | the policy chunk `refine` receives (`SafePolicy.infer` -> `chunk`) |
| `q_now` (20) from `frames.jsonl` `qpos` | `scene_fn` -> `q_now` (newest observation's `robot_state`) |
| `T_base_torso5` = baseline FK at `q_now` | `constraint_robot_model.link_pose(q_now, "link_torso_5")` (same model) |
| finger value = `finger_path_plan[:, 0]` (`FingerJointMap.path`, step 0 = measured) | the TO linearizer's parameter path step 0 (`SafePolicy._apply_finger_state` sets the same `FingerJointMap.path`); model values when no path is set |
| `esdf_coarse_*` / `esdf_*` / `esdf_free_*` float16 arrays | the live field's layers: `layers[0]` / `layers[-1]` / `target_free_layers[0]` (`ConstraintRecordWriter.field_layers` rule), GPU float32 rounded to float16 on the device |
| `robot_spec.json` (exporter, snapshot builder) | `export_inputs.robot_spec` on the served constraint model and the server's own argv (`model_options_from_argv`), limits checked against the refiner's |

Imports nothing heavy at module level (the arg check runs before the checkpoint loads).
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import traceback
from typing import Any, Mapping, Optional

import numpy as np

TO_BACKENDS = ("sqp", "curobo-b0", "curobo-b0m")
#: backend -> `CuroboChunkSolver` world
BACKEND_WORLD = {"curobo-b0": "b0", "curobo-b0m": "b0m"}
TORSO_LINK = "link_torso_5"
LOG = logging.getLogger("AG3S.trajopt.curobo_v2.serving")


# =================================================================================================
# CLI
# =================================================================================================

def check_to_backend_args(args) -> Optional[str]:
    """`None` when the flags allow `--to-backend`, else why not (serve_safe exits with it)."""
    backend = getattr(args, "to_backend", "sqp")
    if backend not in TO_BACKENDS:
        return f"--to-backend must be one of {TO_BACKENDS}"
    if backend == "sqp":
        return None
    if getattr(args, "no_safe", False) or getattr(args, "no_perception", False):
        return f"--to-backend {backend} needs the SafePolicy path (not --no-safe / --no-perception)"
    if getattr(args, "esdf_backend", None) != "curobo":
        return f"--to-backend {backend} needs --esdf-backend curobo (its world is the cuRobo field)"
    if getattr(args, "shadow", False):
        return f"--to-backend {backend} is not defined for --shadow"
    if backend == "curobo-b0m" and not float(getattr(args, "obstacle_margin", 0.0) or 0.0) > 0.0:
        return "--to-backend curobo-b0m needs --obstacle-margin > 0 (B0m = B0 + that margin)"
    return None


# =================================================================================================
# live inputs
# =================================================================================================

def _layer_dict(layer) -> dict[str, Any]:
    grid = layer.grid
    values = getattr(layer, "values_device", None)
    if values is None:
        values = np.asarray(layer.distance_grid)
    return {"distance": values, "origin": np.asarray(grid.origin, np.float64).reshape(3),
            "voxel_size": float(grid.voxel_size)}


def live_layers(field) -> dict[str, dict[str, Any]]:
    """`{"coarse", "main", ["free"]}` of the live cuRobo field, by the recorder's naming rule
    (`ConstraintRecordWriter.field_layers`: main = finest = `layers[-1]`, coarse = the other one,
    free = `target_free_layers[0]`). B0 needs exactly two tiers, as recorded in T63."""
    layers = tuple(getattr(field, "layers", None) or ())
    if len(layers) != 2:
        raise ValueError(f"curobo-b0 needs a two-tier field (coarse + fine), got {len(layers)} "
                         f"layer(s) on {type(field).__name__}")
    out = {"coarse": _layer_dict(layers[0]), "main": _layer_dict(layers[-1])}
    free = tuple(getattr(field, "target_free_layers", None) or ())
    if free:
        out["free"] = _layer_dict(free[0])
    return out


def finger_values(linearizer, robot) -> tuple[dict[str, float], str]:
    """`({finger joint: value}, source)` — step 0 of the TO's parameter path (the measured opening),
    or the model's current values when no path is set (what the linearizer then uses)."""
    names = list(getattr(robot, "param_joint_names", ()) or ())
    path = getattr(linearizer, "_param_path", None)
    if path is not None:
        P = np.asarray(path, np.float64)
        return {n: float(P[i, 0]) for i, n in enumerate(names)}, "param_path[:, 0]"
    vec = np.asarray(robot.param_vector(), np.float64).reshape(-1)
    return {n: float(vec[i]) for i, n in enumerate(names)}, "model param_vector"


def failure_reasons(diag: Mapping[str, Any]) -> list[str]:
    """cuRobo metrics-rollout constraints that are active on the returned trajectory (T63 rule)."""
    cons = diag.get("curobo_constraints") or {}
    return sorted(name for name, v in cons.items() if int(v.get("n_steps_positive", 0)) > 0)


# =================================================================================================
# certification = `TrajectoryOptimizer._solve` without its SQP loop and look-ahead gate
# =================================================================================================

def certify(optimizer, trajectory: np.ndarray, reference: np.ndarray, q_now: np.ndarray, scene, *,
            template: np.ndarray, previous_chunk: Optional[np.ndarray] = None,
            geometry_certified: bool = True, start_at_rest: bool = False,
            servo_feedback: Optional[dict[str, Any]] = None,
            deflection_history: Optional[np.ndarray] = None,
            command_history: Optional[np.ndarray] = None, notes: Optional[list[str]] = None,
            started: Optional[float] = None, timing: Optional[dict[str, float]] = None):
    """Our certification of an externally optimized plan window `trajectory (nq_opt, H)`.

    Line for line the prologue of the snapshot's `TrajectoryOptimizer._solve` (and the memo / host
    mirror of `solve`), then `optimizer._finish` on `trajectory` with `iterations = 0`. An empty
    scene returns `_passthrough(reference, ..., UNCONSTRAINED)` exactly as `_solve` does — the
    external trajectory is then not used.
    """
    from AG3S.trajopt.limits import rest_start_envelope
    from AG3S.trajopt.types import TrajOptStatus

    started = time.perf_counter() if started is None else started
    notes = list(notes or [])
    if getattr(optimizer.config.collision, "esdf_host_mirror", False):
        mirror = getattr(getattr(scene, "esdf", None), "mirror_to_host", None)
        if mirror is not None:
            mirror()
    memo = bool(optimizer.fast_eval)
    if memo:
        optimizer.linearizer.begin_memo()
    try:
        cfg = optimizer.config
        reference = np.asarray(reference, np.float64)
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        trajectory = np.asarray(trajectory, np.float64)
        if trajectory.shape != reference.shape:
            raise ValueError(f"trajectory {trajectory.shape} != reference {reference.shape}")
        if optimizer.lookahead_from is not None and previous_chunk is not None:
            previous_chunk = np.asarray(previous_chunk, np.float64)[:, :optimizer.lookahead_from]
        optimizer._history = {}
        if deflection_history is not None and float(cfg.cost.w_deflection_rate) > 0.0:
            optimizer._history["deflection"] = np.asarray(deflection_history,
                                                          np.float64).reshape(-1)
        if command_history is not None and optimizer.servo_accel_bound is not None:
            optimizer._history["command"] = np.asarray(command_history, np.float64).reshape(
                reference.shape[0], -1)
        envelope = (rest_start_envelope(optimizer.limits, optimizer.horizon,
                                        getattr(cfg.limits, "rest_start_tolerance", None))
                    if start_at_rest else None)
        servo_state = None
        if optimizer.observer is not None:
            fb = (servo_feedback or {}).get("feedback")
            seq = (servo_feedback or {}).get("seq")
            servo_state = optimizer.observer.update(q_now[optimizer.layout.q_indices], fb, seq)
        if scene is None or scene.is_empty:
            return optimizer._passthrough(
                reference, q_now, template, TrajOptStatus.UNCONSTRAINED,
                notes + ["AG3S emitted no geometry to avoid; the reference passes through unchanged"],
                started)
        optimizer._gate_record = None
        optimizer.linearizer.set_attached(getattr(scene, "attached_points", None),
                                          getattr(scene, "attached_parent_link", None))
        if optimizer.lookahead_from is not None:
            reference_violation = -float(optimizer.linearizer.step_clearances(
                reference, q_now, scene)[:optimizer.lookahead_from].min(initial=np.inf))
        else:
            reference_violation = -optimizer.linearizer.full_violation(reference, q_now, scene)
        if optimizer.path is not None:
            optimizer.path.begin(q_now, scene, servo_state)
        result = optimizer._finish(
            trajectory, reference, q_now, scene, template, 0, 0.0, reference_violation,
            previous_chunk, notes, started, geometry_certified, states=None,
            timing=timing or {}, qp_iterations=0, time_budget_hit=False, selection=None)
        if envelope is not None:
            result.metrics["rest_start"] = optimizer._rest_start_record(
                trajectory, reference, q_now, envelope)
        look = result.metrics.get("lookahead")
        if isinstance(look, dict) and optimizer.lookahead_gate is not None:
            # `_finish` writes "open" whenever a gate exists; no gate ran here.
            look["gate"] = "not run (external optimizer)"
        return result
    finally:
        if memo:
            optimizer.linearizer.end_memo()


# =================================================================================================
# the backend
# =================================================================================================

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


class CuroboB0Backend:
    """One `CuroboChunkSolver` (world `b0` or `b0m`, timing `exact`) for the life of the server."""

    name = "curobo-b0"

    def __init__(self, spec: Mapping[str, Any], robot, layout, *,
                 log_dir: Optional[pathlib.Path] = None, solver=None, backend: str = "curobo-b0"):
        if backend not in BACKEND_WORLD:
            raise ValueError(f"backend must be one of {sorted(BACKEND_WORLD)}")
        self.name = backend
        self.world = BACKEND_WORLD[backend]
        self.spec = spec
        self.robot = robot
        self.layout = layout
        names = [str(robot.joint_names[i]) for i in layout.q_indices]
        if names != list(spec["arm_joint_names"]):
            raise ValueError(f"layout joints {names} != cuRobo c-space {spec['arm_joint_names']}")
        if solver is None:
            from AG3S.trajopt.curobo_v2.solver import CuroboChunkSolver

            solver = CuroboChunkSolver(spec, world=self.world, timing="exact")
        self.solver = solver
        self.perm = None
        if self.world == "b0m":
            from AG3S.trajopt.curobo_v2 import spec as S

            self.perm = S.sphere_permutation(spec)
        self.log_dir = None if log_dir is None else pathlib.Path(log_dir)
        self.n_chunks = 0
        self.n_episode_chunks = 0
        self.n_episodes = 0
        self.warmup: dict[str, Any] = {}
        self.last_record: Optional[dict[str, Any]] = None

    # -- lifecycle ---------------------------------------------------------------------------
    def reset(self) -> None:
        """Episode boundary. The solver (and its CUDA graphs) is kept — task T63b item 5."""
        self.n_episode_chunks = 0
        self.n_episodes += 1

    def warm_up(self, n: int = 2) -> dict[str, Any]:
        """CUDA graph capture before any episode: `n` solves on a synthetic obstacle-free world
        (both box rules: with and without a target-excluded layer) at a pose inside the limits."""
        lim = self.spec["limits"]
        lo, hi = np.asarray(lim["lower"]), np.asarray(lim["upper"])
        q14 = np.clip(np.zeros(len(lo)), lo + 0.05, hi - 0.05)
        q_model = np.zeros(len(self.robot.joint_names))
        q_model[np.asarray(self.layout.q_indices)] = q14
        T = np.asarray(self.robot.link_pose(q_model, TORSO_LINK), np.float64)
        ref = q14[None] + np.linspace(0.0, 0.05, 50)[:, None] * np.sign(hi - q14 - 0.1)[None]
        chunk = np.zeros((50, 16))
        chunk[:, [*range(0, 7), *range(8, 15)]] = ref
        free = {"distance": np.full((64, 64, 64), 0.5, np.float32), "voxel_size": 0.005,
                "origin": np.asarray(T[:3, 3]) + np.array([0.3, -0.16, -0.16])}
        layers = {"coarse": {"distance": np.full((48, 48, 48), 0.5, np.float32),
                             "voxel_size": 0.02,
                             "origin": np.asarray(T[:3, 3]) + np.array([-0.2, -0.48, -0.48])},
                  "main": free}
        out = []
        self.solver.set_fingers({n: 0.0 for n in self.spec["finger_joints"]})
        rules = None
        if self.world == "b0m":
            # a synthetic rule exercising every branch: one contact row, a plane 1 m below the
            # torso, target points far from the arm
            ns = int(self.spec["n_spheres"])
            far = np.asarray(T[:3, 3]) + np.array([0.6, 0.0, 0.6])
            rules = {"obstacle_margin": 0.010, "target_tol": 0.005, "support_tol": 0.010,
                     "contact": np.arange(ns) == 0,
                     "target": {"kind": "points", "points": far[None] + 0.01 * np.eye(3)},
                     "planes": np.array([[0.0, 0.0, 1.0, float(T[2, 3]) - 1.0]]),
                     "support_exempt": True, "strict": True}
            rules = self._rules_curobo(rules)
        for i in range(int(n)):
            lay = dict(layers, free=free) if i % 2 == 0 else layers
            t0 = time.perf_counter()
            info = (self.solver.scene.load_layers(lay, T, b0m_rules=rules) if rules is not None
                    else self.solver.scene.load_layers(lay, T))
            r = self.solver.solve(q14, chunk)
            out.append({"i": i, "box_layer": info.get("box_layer"), "success": r["success"],
                        "wall_ms": r["wall_ms"], "gpu_solve_ms": r["solve_time_ms"],
                        "total_ms": (time.perf_counter() - t0) * 1e3, "cold": r["cold"],
                        "finite": r["finite"]})
        self.warmup = {"backend": self.name, "solves": out,
                       "settings": self.solver.settings_record()}
        self._write("curobo_b0_startup.json", self.warmup, mode="json")
        return self.warmup

    def _rules_curobo(self, rules: Mapping[str, Any]) -> dict[str, Any]:
        from AG3S.trajopt.curobo_v2 import b0m as B0M

        return B0M.to_curobo(rules, self.perm)

    # -- per chunk ---------------------------------------------------------------------------
    def solve_chunk(self, chunk: np.ndarray, q_now: np.ndarray, field, finger: Mapping[str, float],
                    b0m_rules: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """cuRobo B0 / B0m on this chunk's live inputs. `b0m_rules` (`b0m.rules_from_scene`, our
        sphere order) is required for B0m and refused for B0. Returns the solver's dict + world /
        timing info."""
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        q14 = q_now[np.asarray(self.layout.q_indices)]
        T = np.asarray(self.robot.link_pose(q_now, TORSO_LINK), np.float64)
        t0 = time.perf_counter()
        if self.world == "b0m":
            if b0m_rules is None:
                raise ValueError("curobo-b0m: no margin rule for this chunk")
            info = self.solver.scene.load_layers(live_layers(field), T,
                                                 b0m_rules=self._rules_curobo(b0m_rules))
        else:
            if b0m_rules is not None:
                raise ValueError("curobo-b0 got a B0m margin rule")
            info = self.solver.scene.load_layers(live_layers(field), T)
        self.solver.set_fingers(finger)
        t_world = (time.perf_counter() - t0) * 1e3
        out = self.solver.solve(q14, np.asarray(chunk, np.float64))
        out["world"] = info
        out["world_load_ms"] = t_world
        out["T_base_torso5"] = T
        return out

    def refine(self, optimizer, chunk: np.ndarray, reference: np.ndarray, q_now: np.ndarray,
               scene, *, previous_chunk=None, geometry_certified: bool = True,
               start_at_rest: bool = False, servo_feedback=None, deflection_history=None,
               command_history=None, t_step=None):
        """The refiner's `optimizer.solve(...)` call, with cuRobo B0 as the optimizer."""
        from AG3S.trajopt.types import TrajOptStatus

        started = time.perf_counter()
        chunk = np.asarray(chunk, np.float64)
        planned = int(reference.shape[1])
        kw = dict(previous_chunk=previous_chunk, geometry_certified=geometry_certified,
                  start_at_rest=start_at_rest, servo_feedback=servo_feedback,
                  deflection_history=deflection_history, command_history=command_history)
        self.n_chunks += 1
        self.n_episode_chunks += 1
        rec: dict[str, Any] = {"backend": self.name, "t_step": t_step, "chunk_seq": self.n_chunks,
                               "episode": self.n_episodes, "episode_chunk": self.n_episode_chunks}
        if scene is None or scene.is_empty:
            # Same as `_solve`: nothing to avoid -> the reference passes through (no cuRobo).
            rec.update(ran=False, reason="empty scene (UNCONSTRAINED, as the SQP path)")
            result = certify(optimizer, reference, reference, q_now, scene, template=chunk, **kw)
            return self._finish_record(result, rec, started)
        failure = None
        out: dict[str, Any] = {}
        try:
            if getattr(scene, "esdf", None) is None:
                raise ValueError("scene has no ESDF field (curobo-b0 world)")
            fingers, fsrc = finger_values(optimizer.linearizer, self.robot)
            rec.update(finger_values=fingers, finger_source=fsrc)
            rules = None
            if self.world == "b0m":
                from AG3S.trajopt.curobo_v2 import b0m as B0M

                rules = B0M.rules_from_scene(scene, int(self.spec["n_spheres"]))
                cfg_m = float(getattr(optimizer.config.collision, "obstacle_margin", 0.0) or 0.0)
                rec["b0m_rule"] = {
                    "obstacle_margin_m": rules["obstacle_margin"], "config_obstacle_margin_m": cfg_m,
                    "target_tol_m": rules["target_tol"], "support_tol_m": rules["support_tol"],
                    "n_contact_rows": int(np.sum(rules["contact"])),
                    "n_support_planes": int(rules["planes"].shape[0]),
                    "target": rules["target"]["kind"], "strict": rules["strict"],
                    "support_exempt": rules["support_exempt"]}
                if abs(rules["obstacle_margin"] - cfg_m) > 1e-12:
                    raise ValueError(f"scene obstacle_margin {rules['obstacle_margin']} != server "
                                     f"config {cfg_m}")
            out = self.solve_chunk(chunk, q_now, scene.esdf, fingers, b0m_rules=rules)
            if out.get("chunk") is None or not out.get("finite", False):
                failure = "cuRobo returned no finite 50-step trajectory"
        except Exception as exc:  # noqa: BLE001 - a backend fault must not kill the server
            failure = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc(limit=8)
            LOG.exception("%s failed at t_step=%s", self.name, t_step)
        rec.update(ran=True, **{k: out[k] for k in (
            "success", "wall_ms", "solve_time_ms", "total_time_ms", "cold", "finite",
            "start_error_rad", "goal_error_rad", "state_end_goal_error_rad", "n_states",
            "solution_state_dt_s", "world_load_ms", "curobo_position_error") if k in out})
        if "world" in out:
            rec["world"] = out["world"]
        if "curobo_constraints" in out:
            rec["curobo_constraints"] = out["curobo_constraints"]
            rec["failure_reasons"] = failure_reasons(out)
        if failure is not None:
            rec["fallback"] = failure
            result = optimizer._passthrough(
                reference, q_now, chunk, TrajOptStatus.SOLVER_FAILED,
                [f"{self.name}: {failure}; the reference is returned UNVERIFIED (verdict HOLD)"],
                started)
            return self._finish_record(result, rec, started)
        cu_chunk = np.asarray(out["chunk"], np.float64)
        trajectory = self.layout.chunk_to_trajectory(cu_chunk)[:, :planned]
        t_cert = time.perf_counter()
        notes = [f"{self.name}: success={bool(out['success'])}"
                 + ("" if out["success"] else f" (active: {', '.join(rec['failure_reasons'])})")
                 + f", {out['wall_ms']:.1f} ms; certified by our TO's check"]
        result = certify(optimizer, trajectory, reference, q_now, scene, template=cu_chunk,
                         notes=notes, started=started,
                         timing={"curobo_wall": float(out["wall_ms"]),
                                 "curobo_world_load": float(out["world_load_ms"])}, **kw)
        rec["certify_ms"] = (time.perf_counter() - t_cert) * 1e3
        return self._finish_record(result, rec, started)

    # -- records -----------------------------------------------------------------------------
    def _finish_record(self, result, rec: dict[str, Any], started: float):
        rec["status"] = getattr(result.status, "value", None)
        rec["max_violation_m"] = float(result.max_violation)
        rec["clearance_m"] = result.metrics.get("clearance_m")
        rec["total_ms"] = (time.perf_counter() - started) * 1e3
        compact = {k: rec.get(k) for k in (
            "backend", "ran", "success", "failure_reasons", "fallback", "wall_ms", "solve_time_ms",
            "world_load_ms", "certify_ms", "total_ms", "cold", "finite", "goal_error_rad",
            "finger_source", "chunk_seq", "episode_chunk")}
        compact["box_layer"] = (rec.get("world") or {}).get("box_layer")
        if "b0m_rule" in rec:
            compact["b0m_rule"] = {k: rec["b0m_rule"][k] for k in (
                "obstacle_margin_m", "n_contact_rows", "n_support_planes", "target")}
        # `summary_json.to.metrics` stringifies dicts (`constraint_record._scalar`): store JSON
        result.metrics["to_backend"] = self.name
        result.metrics["curobo_b0"] = json.dumps(_jsonable(compact))
        self.last_record = rec
        self._write("curobo_b0_chunks.jsonl", rec)
        LOG.info("%s t_step=%s ep_chunk=%d ran=%s success=%s wall=%s ms gpu=%s ms "
                 "box=%s reasons=%s -> status=%s clearance=%s mm%s", self.name, rec.get("t_step"),
                 self.n_episode_chunks, rec.get("ran"), rec.get("success"),
                 _fmt(rec.get("wall_ms")), _fmt(rec.get("solve_time_ms")),
                 compact["box_layer"], rec.get("failure_reasons"), rec["status"],
                 _fmt(None if rec["clearance_m"] is None else 1e3 * rec["clearance_m"]),
                 f" FALLBACK {rec['fallback']}" if rec.get("fallback") else "")
        return result

    def _write(self, name: str, obj: Mapping[str, Any], mode: str = "jsonl") -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / name
        if mode == "json":
            path.write_text(json.dumps(_jsonable(obj), indent=1))
            return
        with path.open("a") as fh:
            fh.write(json.dumps(_jsonable(dict(obj))) + "\n")


def _fmt(v) -> str:
    return "-" if v is None else f"{float(v):.1f}"


# =================================================================================================
# server wiring
# =================================================================================================

def build_spec(robot, layout, limits, argv: list[str], dt: float) -> dict[str, Any]:
    """The T63 `robot_spec` of the **served** constraint model (same function as the exporter,
    options parsed from this server's argv), checked against the refiner's limits."""
    from AG3S.trajopt.curobo_v2 import export_inputs as X

    opts = X.model_options_from_argv(list(argv))
    spec = X.robot_spec(robot, opts, dt)
    lim = spec["limits"]
    checks = {
        "n_spheres": int(spec["n_spheres"]),
        "lower": float(np.abs(np.asarray(lim["lower"]) - np.asarray(limits.lower)).max()),
        "upper": float(np.abs(np.asarray(lim["upper"]) - np.asarray(limits.upper)).max()),
        "max_step": float(np.abs(np.asarray(lim["velocity"]) * dt
                                 - np.asarray(limits.max_step)).max()),
        "max_step_change": float(np.abs(np.asarray(lim["acceleration"]) * dt * dt
                                        - np.asarray(limits.max_step_change)).max()),
    }
    if max(checks[k] for k in ("lower", "upper", "max_step", "max_step_change")) > 1e-9:
        raise SystemExit(f"curobo-b0: spec limits differ from the refiner's: {checks}")
    spec["serving_checks"] = checks
    return spec


def attach_to_backend(served, args, argv: list[str]) -> Optional[CuroboB0Backend]:
    """`serve_safe --to-backend curobo-b0 | curobo-b0m`: build the backend on the served model, warm
    it up, and hand it to the refiner. `sqp` (default) does nothing and returns None."""
    if getattr(args, "to_backend", "sqp") == "sqp":
        return None
    import torch

    tf32_before = bool(torch.backends.cuda.matmul.allow_tf32)
    refiner = served.refiner
    robot = served.constraint_robot_model
    dt = float(refiner.config.horizon.dt)
    spec = build_spec(robot, refiner.layout, refiner.limits, argv, dt)
    log_dir = pathlib.Path(args.record_constraints or args.run_dir or ".")
    t0 = time.perf_counter()
    backend = CuroboB0Backend(spec, robot, refiner.layout, log_dir=log_dir,
                              backend=str(args.to_backend))
    build_s = time.perf_counter() - t0
    (log_dir / "curobo_b0_robot_spec.json").parent.mkdir(parents=True, exist_ok=True)
    (log_dir / "curobo_b0_robot_spec.json").write_text(json.dumps(_jsonable(spec), indent=1))
    t0 = time.perf_counter()
    warm = backend.warm_up()
    warm.update(build_s=build_s, warmup_s=time.perf_counter() - t0,
                tf32_matmul_before_curobo=tf32_before,
                tf32_matmul_after_curobo=bool(torch.backends.cuda.matmul.allow_tf32))
    backend._write("curobo_b0_startup.json", warm, mode="json")
    refiner.backend = backend
    label = {"curobo-b0": "curobo-b0 (T63b) — cuRobo v2 B0",
             "curobo-b0m": "curobo-b0m (T63b B0m) — cuRobo v2 B0 + our TO's obstacle margin "
                           f"({float(refiner.config.collision.obstacle_margin) * 1e3:.1f} mm on "
                           "obstacle rows)"}[backend.name]
    LOG.warning("TO backend: %s replaces the SQP step; our TO's "
                "certification, HOLD and status paths are unchanged. %d spheres, build %.1f s, "
                "warm-up %s, TF32 matmul %s -> %s. Per-chunk log: %s", label, spec["n_spheres"],
                build_s, [(w["wall_ms"], w["success"]) for w in warm["solves"]], tf32_before,
                warm["tf32_matmul_after_curobo"], log_dir / "curobo_b0_chunks.jsonl")
    return backend


__all__ = ["CuroboB0Backend", "TO_BACKENDS", "attach_to_backend", "build_spec", "certify",
           "check_to_backend_args", "failure_reasons", "finger_values", "live_layers"]
