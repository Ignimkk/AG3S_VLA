"""The quadratic subproblem, and which solver runs it.

Four backends sit behind one call because the right choice is a measurement, not a preference, and
because the measurement should be repeatable on the machine that will run the robot. `--benchmark`
times every available backend on a problem the same size as the real one and prints the table the
README quotes.

**Why not an interior-point NLP solver for the whole thing.** IPOPT is already wired into this
repository (`tests/ag3s/test_to_contract.py`) and it converges, so the objection is not that it fails
— it is that it cannot be given a deadline.

* Interior-point iterates follow a central path parameterised by a barrier weight. A good
  collision-free trajectory hugs its safety margin, which is exactly where the previous solution sits
  *close to the constraint boundary* — and that is a poor starting point for the next barrier
  subproblem, so the solver re-anneals rather than resuming. Active-set and operator-splitting
  methods have no such pathology; warm-starting them is just handing them the previous answer.
* The iteration count is not bounded. IPOPT runs to a tolerance and may enter a restoration phase.
  Real-time control needs a fixed budget per cycle, which is what the real-time iteration scheme
  provides: take one or a few Newton-type steps and let the feedback loop finish the job.

**What the measurement said.** On the RB-Y1 subproblem — 1,884 variables, 4,116 rows, 16,980
nonzeros, H=32 with 24 rows per step:

| backend | first solve | re-solve | verdict                                     |
|---------|-------------|----------|---------------------------------------------|
| `osqp`  | 96.7 ms     | 2.7 ms   | **default**                                  |
| `highs` | 337.7 ms    | 332.8 ms | correct, 120x slower, no warm-start benefit  |
| `proxqp`| 5,723 ms    | 5,739 ms | correct and unusable here                    |
| `qrqp`  | > 200 s     | —        | excluded                                     |
| `ipqp`  | —           | —        | **aborted the interpreter** (heap corruption)|

The last row is why `ipqp` is not in `QpConfig.SOLVERS`. A backend that can take the control loop
down with it is not a candidate whatever its timing would have been.

The re-solve column is the optimistic case: the *same* problem handed back, where the warm start
lands on the answer. Inside the SQP loop each iteration linearizes somewhere new, and the honest
figure is the per-chunk total in the README (28.4 ms), not this column.

**Why qpOASES is absent from the list.** Not preference — structure. It is a dense online active-set
method whose cost grows superlinearly with the constraint count, and this problem has thousands of
rows. It would be the right answer for a QP an order of magnitude smaller.

**Why not acados, HPIPM or cuRobo.** They are not installed, and the repository's dependency policy
is to use what is present. acados needs C code generation and a build toolchain; cuRobo needs CUDA.
If the measured budget is missed after the reductions in `linearize`, that is the moment to revisit
them — with a number to justify the dependency.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import os
import pathlib
import sys
import time
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from AG3S.trajopt.config import QpConfig
from AG3S.trajopt.problem import QpProblem

#: Bounds this large are treated as infinite. OSQP wants a finite-ish number; CasADi does not care.
_INF = 1e20

# --- AG3S-PRE-CORE-TO E3-a: OSQP with the CUDA algebra (`qp.solver = "osqp-cuda"`) -------------------
#
# **Where the CUDA algebra lives.** `osqp-cu12 1.1.3` ships one extension module, `osqp_cuda`, that the
# `osqp` package finds by name. Installed into a venv, osqp 1.1.3 then makes CUDA the *default* algebra
# of every bare `osqp.OSQP()` in that venv (priority cuda > mkl > builtin) — which would silently move
# every other process that shares the venv onto the GPU. So it is *not* installed into the venv: it sits
# in its own directory (`pip install --no-deps --target <checkout>/.cache/osqp-cu12 osqp-cu12==1.1.3`,
# override with `$AG3S_OSQP_CUDA_DIR`) that is appended to `sys.path` only when this backend is used.
# Even then, the process-wide default algebra is pinned back to `builtin` (`_load_osqp_cuda`), and this
# module always names its algebra explicitly.
#
# **Settings.** The configured `eps_abs` · `eps_rel` · `max_iter` · `polish` · `adaptive_rho` ·
# `warm_starting` go to both builds as they are. The CUDA build also ships *different defaults* for
# ADMM settings the CPU build fixes (`check_dualgap` 0 vs 1, `check_termination` 5 vs 25,
# `adaptive_rho_tolerance` 2 vs 5, scalar vs vector rho). Left as shipped, the CUDA answer on the
# RB-Y1 fixture QP stops at "solved" with an objective of -2226 against -29.66 at the true optimum
# (CPU at the same eps: -29.47); with the CPU values it is -29.71, 0.035 rad from the optimum against
# the CPU's 0.080 rad. So `_CUDA_ALIGNED` copies the CPU build's defaults onto the CUDA solver at setup.
# What cannot be the same: the CUDA build is single precision (float32) and its linear system is solved
# by preconditioned CG (indirect), not QDLDL; OSQP cannot polish with the indirect solver
# (`QpConfig.validate` refuses `polish` with this backend).
#
# **Fallback.** A CUDA solve that the CPU path would call a failure (status other than "solved" /
# "solved inaccurate" / an accepted max-iter iterate, non-finite x, or an exception from the CUDA
# library) is solved again by a CPU OSQP solver kept beside it, warm-started from the same last
# solution. Every attempt is counted process-wide (`cuda_mark` / `cuda_record_since`; the refiner puts
# the per-chunk difference in `to.metrics.qp_backend`).

#: `$AG3S_OSQP_CUDA_DIR` overrides; default `<checkout>/.cache/osqp-cu12` (git-ignored).
OSQP_CUDA_DIR_ENV = "AG3S_OSQP_CUDA_DIR"
_DEFAULT_OSQP_CUDA_DIR = pathlib.Path(__file__).resolve().parents[2] / ".cache" / "osqp-cu12"
#: ADMM settings whose CUDA-build default differs from (or may differ from) the CPU build's; set on the
#: CUDA solver to the CPU build's default (`_cpu_defaults`). Not copied: `linsys_solver` (CUDA has only
#: the indirect one), `device`, the `cg_*` settings (used only by the indirect solver).
_CUDA_ALIGNED = ("rho", "sigma", "alpha", "delta", "scaling", "adaptive_rho_interval", "adaptive_rho_tolerance",
                 "adaptive_rho_fraction", "check_termination", "check_dualgap", "rho_is_vec", "eps_prim_inf",
                 "eps_dual_inf", "scaled_termination", "time_limit", "polish_refine_iter")
#: Process-wide counts of the CUDA backend (every `QpSolver` in the process adds to them).
_CUDA_KEYS = ("qp_solves", "cuda_solved", "cuda_failed", "cuda_errors", "cuda_setups", "cpu_fallback_solved",
              "cpu_fallback_failed")
_CUDA_COUNTS: collections.Counter = collections.Counter()
_CUDA_FAILED_STATUS: collections.Counter = collections.Counter()


def osqp_cuda_dir() -> pathlib.Path:
    """The directory that holds `osqp_cuda` (see the E3-a note above)."""
    return pathlib.Path(os.environ.get(OSQP_CUDA_DIR_ENV) or _DEFAULT_OSQP_CUDA_DIR)


def _load_osqp_cuda():
    """Import `osqp` with the CUDA algebra reachable; returns the `osqp` module. Raises `ImportError`
    with the install command when the extension is not there. Pins the process-wide default algebra to
    `builtin` so that any bare `osqp.OSQP()` elsewhere in this process keeps the CPU build."""
    import importlib

    import osqp
    import osqp.interface as interface

    try:
        importlib.import_module("osqp_cuda")
    except ImportError:
        path = str(osqp_cuda_dir())
        if path not in sys.path:
            sys.path.append(path)
        try:
            importlib.import_module("osqp_cuda")
        except ImportError as exc:
            raise ImportError(
                f"qp.solver 'osqp-cuda' needs the osqp-cu12 extension in {path} (install: pip install --no-deps "
                f"--target {path} osqp-cu12==1.1.3; or set ${OSQP_CUDA_DIR_ENV}): {exc}") from exc
    if getattr(interface, "OSQP_ALGEBRA_BACKEND", None) is None:
        interface.OSQP_ALGEBRA_BACKEND = "builtin"
    return osqp


def osqp_cuda_available() -> tuple[bool, str]:
    """`(True, dir)` when the CUDA algebra can be imported (no GPU work is done), else `(False, why)`."""
    try:
        _load_osqp_cuda()
    except ImportError as exc:
        return False, str(exc)
    return True, str(osqp_cuda_dir())


def _cpu_defaults() -> dict[str, Any]:
    """The CPU (builtin) build's defaults of `_CUDA_ALIGNED`, read from the build itself."""
    import osqp.ext_builtin as ext

    settings = ext.OSQPSettings()
    ext.osqp_set_default_settings(settings)
    return {k: getattr(settings, k) for k in _CUDA_ALIGNED if hasattr(settings, k)}


def cuda_mark() -> tuple[dict, dict]:
    """A snapshot of the process-wide CUDA-backend counts, for `cuda_record_since`."""
    return dict(_CUDA_COUNTS), dict(_CUDA_FAILED_STATUS)


def cuda_record_since(mark: tuple[dict, dict]) -> dict[str, Any]:
    """What the CUDA backend did since `mark`: QP solves, CUDA solved / failed / raised, CUDA setups
    (new sparsity pattern or first use), CPU fallbacks solved / failed, and the failed CUDA statuses."""
    counts, statuses = mark
    out: dict[str, Any] = {"backend": "osqp-cuda"}
    out.update({k: int(_CUDA_COUNTS[k] - counts.get(k, 0)) for k in _CUDA_KEYS})
    out["failed_statuses"] = {s: int(n - statuses.get(s, 0)) for s, n in _CUDA_FAILED_STATUS.items()
                              if n - statuses.get(s, 0) > 0}
    return out


@dataclasses.dataclass(frozen=True)
class QpSolution:
    """One QP solve. `x` is meaningful only when `success` is True."""

    x: np.ndarray
    y: Optional[np.ndarray]  # dual variables, for warm-starting the next solve
    success: bool
    status: str
    iterations: int
    solve_time_ms: float

    @property
    def n_var(self) -> int:
        return int(self.x.shape[0])


class QpSolver:
    """A QP backend, kept alive across solves so it can reuse whatever it is able to reuse.

    Instances are stateful on purpose. OSQP factorizes its KKT matrix once and reuses it while the
    sparsity pattern holds; a fresh object per iteration would throw that away and turn the single
    biggest advantage of the method into overhead.
    """

    def __init__(self, config: Optional[QpConfig] = None):
        self.config = config or QpConfig()
        self._backend = None
        self._signature: Optional[tuple] = None
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Drop the cached factorization. Call between episodes, not between iterations."""
        self._backend = None
        self._signature = None
        self._last_x = None
        self._last_y = None
        self.__dict__.pop("_eps_now", None)
        fallback = self.__dict__.get("_cpu_fallback")      # E3-a (`osqp-cuda` only)
        if fallback is not None:
            fallback.reset()

    def solve(self, problem: QpProblem, *, warm_start: bool = True,
              x0: Optional[np.ndarray] = None, eps: Optional[float] = None) -> QpSolution:
        """`x0` (T43 TP, `qp.warm_start_iterate`): the primal warm start to use instead of the last
        solution (the last duals are kept). `eps` (T43 TP, `qp.eps_after_first`): OSQP `eps_abs` =
        `eps_rel` for this solve. Both `None` (default) = as before. OSQP only."""
        started = time.perf_counter()
        name = self.config.solver
        if name == "osqp":
            if x0 is None and eps is None:
                solution = self._solve_osqp(problem, warm_start)
            else:
                solution = self._solve_osqp(problem, warm_start, x0=x0, eps=eps)
        elif name == "osqp-cuda":
            solution = self._solve_osqp_cuda(problem, warm_start, x0=x0, eps=eps)
        else:
            solution = self._solve_casadi(problem, name, warm_start)
        elapsed = (time.perf_counter() - started) * 1000.0
        if solution.success:
            self._last_x, self._last_y = solution.x, solution.y
        return dataclasses.replace(solution, solve_time_ms=elapsed)

    # --- OSQP -----------------------------------------------------------------------------
    def _solve_osqp(self, problem: QpProblem, warm_start: bool, *, x0: Optional[np.ndarray] = None,
                    eps: Optional[float] = None, algebra: str = "builtin") -> QpSolution:
        if algebra == "cuda":
            osqp = _load_osqp_cuda()
        else:
            import osqp

        signature = _signature(problem)
        l = np.clip(problem.l, -_INF, _INF)
        u = np.clip(problem.u, -_INF, _INF)

        if self._backend is None or self._signature != signature:
            # The algebra is always named (E3-a): a bare `osqp.OSQP()` takes whichever build has the
            # highest priority among those importable, and the CUDA one outranks the CPU one.
            self._backend = osqp.OSQP(algebra=algebra)
            if algebra == "cuda":
                _CUDA_COUNTS["cuda_setups"] += 1
            self._backend.setup(
                **(_cpu_defaults() if algebra == "cuda" else {}),
                P=sp.triu(problem.P, format="csc"),
                q=problem.q,
                A=problem.A,
                l=l,
                u=u,
                verbose=self.config.verbose,
                eps_abs=self.config.eps_abs,
                eps_rel=self.config.eps_rel,
                max_iter=self.config.max_iter,
                polish=self.config.polish,
                adaptive_rho=self.config.adaptive_rho,
                warm_starting=warm_start,
            )
            self._signature = signature
            self.__dict__.pop("_eps_now", None)      # a fresh backend has the configured accuracy
        else:
            # Same pattern: only the numbers move. This is the path the whole fixed-sparsity design
            # exists to reach — no re-factorization, no re-setup.
            self._backend.update(
                Px=sp.triu(problem.P, format="csc").data,
                Ax=problem.A.data,
                q=problem.q,
                l=l,
                u=u,
            )
            if warm_start and self._last_x is not None and self._last_x.shape[0] == problem.n_var:
                if x0 is not None and np.shape(x0) == self._last_x.shape:
                    self._backend.warm_start(x=np.asarray(x0, np.float64), y=self._last_y)
                else:
                    self._backend.warm_start(x=self._last_x, y=self._last_y)
        # T43 TP (`qp.eps_after_first`): this solve's accuracy. The backend keeps what it was last
        # told, so it is set back to the configured one when `eps` is None again. Never asked
        # (default), the backend is never touched here.
        want = ((float(self.config.eps_abs), float(self.config.eps_rel)) if eps is None
                else (float(eps), float(eps)))
        now = self.__dict__.get("_eps_now")
        if now is not None and now != want or now is None and eps is not None:
            self._backend.update_settings(eps_abs=want[0], eps_rel=want[1])
            self.__dict__["_eps_now"] = want

        result = self._backend.solve()
        status = str(getattr(result.info, "status", "unknown"))
        success = status in ("solved", "solved inaccurate")
        if (not success and getattr(self.config, "accept_max_iter", False)
                and "maximum iterations" in status and result.x is not None
                and np.all(np.isfinite(result.x))):
            # T43 TP (`qp.accept_max_iter`): the iterate at the cap, as an inaccurate solution.
            status = "maximum iterations reached (accepted)"
            success = True
        return QpSolution(
            x=np.asarray(result.x, np.float64) if success else np.zeros(problem.n_var),
            y=np.asarray(result.y, np.float64) if success else None,
            # `bool(...)` rather than the numpy scalar `np.all` returns: a `np.bool_` leaking into
            # the public result makes `isinstance(x, bool)` false for a caller who reasonably assumed
            # a flag is a flag.
            success=bool(success and np.all(np.isfinite(result.x))),
            status=status,
            iterations=int(getattr(result.info, "iter", 0)),
            solve_time_ms=0.0,
        )

    # --- OSQP, CUDA algebra (AG3S-PRE-CORE-TO E3-a) ------------------------------------------
    def _solve_osqp_cuda(self, problem: QpProblem, warm_start: bool, *, x0: Optional[np.ndarray] = None,
                         eps: Optional[float] = None) -> QpSolution:
        """The same OSQP call on the CUDA algebra; a failed CUDA solve is solved again on the CPU
        (see the E3-a note at the top of this module). `iterations` of a fallback = CUDA + CPU."""
        _CUDA_COUNTS["qp_solves"] += 1
        try:
            cuda = self._solve_osqp(problem, warm_start, x0=x0, eps=eps, algebra="cuda")
        except ImportError:
            raise                                   # not installed: a configuration error, not a QP failure
        except Exception as exc:                    # the CUDA library raised (device, memory, setup)
            _CUDA_COUNTS["cuda_errors"] += 1
            self._backend, self._signature = None, None      # set up again on the next solve
            self.__dict__.pop("_eps_now", None)
            cuda = QpSolution(np.zeros(problem.n_var), None, False,
                              f"cuda error: {type(exc).__name__}: {exc}", 0, 0.0)
        if cuda.success:
            _CUDA_COUNTS["cuda_solved"] += 1
            return cuda
        _CUDA_COUNTS["cuda_failed"] += 1
        _CUDA_FAILED_STATUS[cuda.status] += 1
        fallback = self.__dict__.get("_cpu_fallback")
        if fallback is None:
            fallback = self._cpu_fallback = QpSolver(dataclasses.replace(self.config, solver="osqp"))
        # Same warm start as the CUDA attempt had: this solver's last solution (the fallback's own
        # factorization is reused while the sparsity pattern holds).
        fallback._last_x, fallback._last_y = self._last_x, self._last_y
        cpu = fallback.solve(problem, warm_start=warm_start, x0=x0, eps=eps)
        _CUDA_COUNTS["cpu_fallback_solved" if cpu.success else "cpu_fallback_failed"] += 1
        return dataclasses.replace(cpu, status=f"{cpu.status} (cpu fallback; cuda: {cuda.status})",
                                   iterations=int(cpu.iterations) + int(cuda.iterations))

    # --- CasADi conic backends -------------------------------------------------------------
    def _solve_casadi(self, problem: QpProblem, name: str, warm_start: bool) -> QpSolution:
        import casadi as ca

        signature = _signature(problem)
        P = _to_dm(problem.P)
        A = _to_dm(problem.A)
        if self._backend is None or self._signature != signature:
            options: dict[str, Any] = {"print_time": False}
            if not self.config.verbose:
                options.update(_quiet_options(name))
            self._backend = ca.conic(
                f"trajopt_{name}", name, {"h": P.sparsity(), "a": A.sparsity()}, options
            )
            self._signature = signature

        kwargs: dict[str, Any] = {
            "h": P,
            "g": problem.q,
            "a": A,
            "lba": np.clip(problem.l, -np.inf, np.inf),
            "uba": np.clip(problem.u, -np.inf, np.inf),
        }
        if warm_start and self._last_x is not None and self._last_x.shape[0] == problem.n_var:
            kwargs["x0"] = self._last_x
        try:
            result = self._backend(**kwargs)
        except RuntimeError as exc:
            return QpSolution(np.zeros(problem.n_var), None, False, f"error: {exc}", 0, 0.0)

        x = np.asarray(result["x"], np.float64).reshape(-1)
        stats = self._backend.stats()
        success = bool(stats.get("success", True)) and bool(np.all(np.isfinite(x)))
        return QpSolution(
            x=x,
            y=np.asarray(result.get("lam_a", np.zeros(0)), np.float64).reshape(-1),
            success=success,
            status=str(stats.get("return_status", "solved" if success else "failed")),
            iterations=int(stats.get("iter_count", 0)),
            solve_time_ms=0.0,
        )


def _signature(problem: QpProblem) -> tuple:
    """What must stay the same for a solver to reuse its setup: shapes and sparsity, never values."""
    return (
        problem.P.shape,
        problem.A.shape,
        problem.P.nnz,
        problem.A.nnz,
        problem.P.indptr.tobytes(),
        problem.P.indices.tobytes(),
        problem.A.indptr.tobytes(),
        problem.A.indices.tobytes(),
    )


def _to_dm(matrix: sp.spmatrix):
    import casadi as ca

    coo = matrix.tocoo()
    return ca.DM(
        ca.Sparsity.triplet(coo.shape[0], coo.shape[1], coo.row.tolist(), coo.col.tolist()),
        coo.data.tolist(),
    )


def _quiet_options(name: str) -> dict[str, Any]:
    return {
        "proxqp": {"proxqp": {"verbose": False}},
        "qrqp": {"print_iter": False, "print_header": False},
        "ipqp": {"print_iter": False, "print_header": False},
        "highs": {"highs": {"output_flag": False}},
    }.get(name, {})


def available_solvers() -> list[str]:
    """Which backends this build can actually construct. Probed, not assumed."""
    import casadi as ca

    found: list[str] = []
    try:
        import osqp  # noqa: F401

        found.append("osqp")
    except ImportError:
        pass
    probe_h = ca.DM(np.eye(2)).sparsity()
    probe_a = ca.DM(np.ones((1, 2))).sparsity()
    for name in ("proxqp", "highs"):
        try:
            # Several backends announce themselves on construction regardless of their options, so
            # the probe is muted at the file-descriptor level — CasADi prints from C++, where a
            # Python `redirect_stdout` does not reach.
            with _muted():
                ca.conic("probe", name, {"h": probe_h, "a": probe_a},
                         {"print_time": False, **_quiet_options(name)})
            found.append(name)
        except Exception:
            continue
    return found


@contextlib.contextmanager
def _muted():
    """Silence C++-level stdout for the duration of the block."""
    import os

    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _bench_one(name: str, horizon: int, rows_per_step: int, repeats: int) -> None:
    """Time one backend and print a single line. Runs as its own process — see `main`."""
    import dataclasses as dc
    import time

    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.linearize import append_collision_rows
    from AG3S.trajopt.problem import build_problem
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import ChunkLayout
    from tests.trajopt.fixtures import Q_HOME, near_miss_scene, sweep_reference

    config = TrajOptConfig.from_dict({
        "horizon": {"horizon": horizon, "execution_length": 8, "plan_horizon": None},
        "reduction": {"rows_per_step": rows_per_step},
    })
    robot = load_rby1()
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    limits = build_limits(robot, layout, dt=config.horizon.dt, config=config.limits)
    optimizer = TrajectoryOptimizer(robot, layout, limits, config)
    reference = sweep_reference(layout, horizon)
    scene = near_miss_scene(optimizer.linearizer, reference, Q_HOME)
    rows = optimizer.linearizer.linearize(reference, Q_HOME, scene, config)
    problem = append_collision_rows(
        build_problem(reference, reference, limits, config,
                      q_now=Q_HOME[layout.q_indices], n_slack=optimizer.block.n_rows),
        rows, reference, optimizer.block, config.reduction.linearization_backoff,
    )
    solver = QpSolver(dc.replace(QpConfig(), solver=name))
    started = time.perf_counter()
    first = solver.solve(problem)
    elapsed = (time.perf_counter() - started) * 1000.0
    if not first.success:
        print(f"BENCH {name} - - - {first.status}")
        return
    warm = [solver.solve(problem).solve_time_ms for _ in range(repeats)]
    checksum = float(np.abs(problem.split(first.x)[0]).sum())
    print(f"BENCH {name} {elapsed:.1f} {float(np.median(warm)):.1f} {first.iterations} "
          f"{first.status} {checksum:.6f}")


def main(argv: Optional[list[str]] = None) -> int:
    """Time every available backend on a problem the size of the real one.

    The point is that the default is a measurement taken on the machine that will run the robot, not
    a number copied out of a paper.

    Each backend runs in **its own process**. Two of the CasADi conic plugins abort this problem at
    the C level (`double free or corruption`), and a benchmark that a candidate can crash is a
    benchmark that only ever reports the survivors. A crash is a result, and it is reported as one.
    """
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(description="QP backend benchmark for the RB-Y1 subproblem")
    parser.add_argument("--benchmark", action="store_true", help="run the comparison")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--rows-per-step", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per backend")
    parser.add_argument("--only", default=None, help=argparse.SUPPRESS)  # used by the child process
    args = parser.parse_args(argv)

    if args.only:
        _bench_one(args.only, args.horizon, args.rows_per_step, args.repeats)
        return 0
    if not args.benchmark:
        parser.print_help()
        return 0

    print(f"{'solver':>8s} {'first(ms)':>10s} {'warm(ms)':>9s} {'iters':>7s}  status")
    reference = None
    for name in available_solvers():
        command = [
            sys.executable, "-m", "AG3S.trajopt.qp", "--only", name,
            "--horizon", str(args.horizon), "--rows-per-step", str(args.rows_per_step),
            "--repeats", str(args.repeats),
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=args.timeout, check=False
            )
        except subprocess.TimeoutExpired:
            print(f"{name:>8s} {'-':>10s} {'-':>9s} {'-':>7s}  excluded: over {args.timeout:.0f} s")
            continue
        line = next(
            (l for l in done.stdout.splitlines() if l.startswith("BENCH ")), None
        )
        if line is None:
            reason = "crashed" if done.returncode != 0 else "no result"
            print(f"{name:>8s} {'-':>10s} {'-':>9s} {'-':>7s}  excluded: {reason} "
                  f"(exit {done.returncode})")
            continue
        _, _, first, warm, iterations, status, *rest = line.split()
        note = ""
        if rest:
            # Sum of |Q| over every joint and step. Reported as a difference rather than judged: at a
            # solver tolerance of 1e-3 across ~400 entries a spread of this order is expected, and
            # labelling it a disagreement would be reading precision into the third decimal of a
            # number the solvers were never asked to agree on. A large gap here means something real.
            checksum = float(rest[0])
            if reference is None:
                reference = checksum
            else:
                per_entry = abs(checksum - reference) / max(args.horizon * 12, 1)
                note = f"  checksum d={per_entry:.1e}/entry"
        print(f"{name:>8s} {first:>10s} {warm:>9s} {iterations:>7s}  {status}{note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["QpSolution", "QpSolver", "available_solvers", "cuda_mark", "cuda_record_since", "main",
           "osqp_cuda_available", "osqp_cuda_dir"]
