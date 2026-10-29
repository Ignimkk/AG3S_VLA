"""AG3S-PRE-CORE-TO E3-a — the OSQP CUDA backend (`serve_safe --qp-backend osqp-cuda`, `qp.solver = "osqp-cuda"`).

Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 "E3 구현 명세", row E3-a.

What this file pins:
1. Default (`osqp`) is bit-identical to HEAD b3b69311 (the E2 golden SHA-1s of the fixture solves, measured on a
   `git archive HEAD` copy), the backend is the builtin algebra *by name*, and that stays so after the CUDA extension
   has been loaded into the same process (osqp 1.1.3 ranks cuda above builtin for a bare `osqp.OSQP()`).
2. On a GPU: the CUDA backend solves a small QP to the CPU solution within OSQP tolerance, the RB-Y1 fixture QP to an
   objective and constraint residual close to the CPU's, and the ADMM settings the CPU build fixes are on the CUDA
   solver. Skipped without a CUDA device.
3. Fall-back: a CUDA solve that fails (non-solved status, or the library raising) is solved again by CPU OSQP with the
   same warm start; the result is the CPU solution and the counts (`cuda_mark` / `cuda_record_since`) record it.
4. Wiring (no GPU needed, CUDA call replaced by the builtin one): every QP of the standard solve and of the multi-seed
   seeds goes through the CUDA path; the refiner records `to.metrics.qp_backend` only with the CUDA backend.
5. Config / CLI validation, the offline settings, the summary.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from AG3S.trajopt.config import QpConfig, TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

from tests.trajopt.test_e2_multi_seed import BASE_ARGV, HEAD_SHA, _sha, _stuck  # noqa: E402

CUDA = {"qp": {"solver": "osqp-cuda"}}


@functools.lru_cache(maxsize=1)
def _cuda_device() -> bool:
    """The CUDA algebra imports and can set up a 2x2 QP (a device is visible)."""
    from AG3S.trajopt import qp as Q

    ok, _ = Q.osqp_cuda_available()
    if not ok:
        return False
    import osqp

    try:
        s = osqp.OSQP(algebra="cuda")
        s.setup(P=sp.csc_matrix(np.eye(2)), q=np.ones(2), A=sp.csc_matrix(np.eye(2)), l=-np.ones(2), u=np.ones(2),
                verbose=False)
        return s.solve(raise_error=False).info.status == "solved"
    except Exception:
        return False


needs_cuda = pytest.mark.skipif("not _cuda_device()", reason="no CUDA device / osqp-cu12 extension")


@pytest.fixture
def fake_cuda(monkeypatch):
    """Route the CUDA attempt to the builtin algebra (counts still go through the CUDA path)."""
    from AG3S.trajopt import qp as Q

    calls = []
    real = Q.QpSolver._solve_osqp

    def solve(self, problem, warm_start, *, x0=None, eps=None, algebra="builtin"):
        if algebra == "cuda":
            calls.append(algebra)
        return real(self, problem, warm_start, x0=x0, eps=eps, algebra="builtin")

    monkeypatch.setattr(Q.QpSolver, "_solve_osqp", solve)
    return calls


def _small_qp(n=20, m=30, seed=0):
    """A strongly convex QP with box and general rows, as a `QpProblem` (one 'joint' per variable, H=1)."""
    from AG3S.trajopt.problem import QpProblem

    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    P = sp.csc_matrix(M @ M.T / n + np.eye(n))
    A = sp.csc_matrix(np.vstack([rng.normal(size=(m - n, n)), np.eye(n)]))
    q = rng.normal(size=n)
    return QpProblem(P=P, q=q, A=A, l=-np.ones(m), u=np.ones(m), n_q=n, n_slack=0, nq=n, horizon=1)


def _fixture_problem():
    from AG3S.trajopt.linearize import append_collision_rows
    from AG3S.trajopt.problem import build_problem
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    rows = opt.linearizer.linearize(sc.reference, sc.q_now, sc.scene, sc.config)
    pb = build_problem(sc.reference, sc.reference, sc.limits, sc.config, q_now=sc.q_now[sc.layout.q_indices],
                       n_slack=opt.block.n_rows)
    return append_collision_rows(pb, rows, sc.reference, opt.block, sc.config.reduction.linearization_backoff)


def _residual(problem, x):
    Ax = problem.A @ x
    return float(max(0.0, np.max(Ax - problem.u), np.max(problem.l - Ax)))


# ------------------------------------------------------------------------------------------------ 1. default
def _default_shas() -> tuple[dict, Any]:
    """SHA-1[:16] of the E2 fixture solves with the default config (keys of `HEAD_SHA`), and the last refiner."""
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    out = {}
    sc = make_scenario(horizon=8, penetration=0.03)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    r = opt.solve(sc.reference, sc.q_now, sc.scene)
    out["std_off"] = _sha(r.trajectory)
    assert r.metrics["qp_solver"] == "osqp" and opt.solver._backend.algebra == "builtin"
    st, scene = _stuck()
    o = TrajectoryOptimizer(st.robot, st.layout, st.limits, st.config)
    out["stuck_off_1"] = _sha(o.solve(st.reference, st.q_now, scene).trajectory)
    out["stuck_off_2"] = _sha(o.solve(st.reference, st.q_now, scene).trajectory)
    ref = TrajOptChunkRefiner(st.robot, st.layout, lambda c: (scene, st.q_now, True),
                              st.config.with_overrides({"horizon": {"plan_horizon": None}}))
    c1 = ref.refine(st.chunk, {"t_step": 3, "previous_physical_chunk": None})
    c2 = ref.refine(st.chunk, {"t_step": 4, "previous_physical_chunk": c1})
    out["refiner_off_1"], out["refiner_off_2"] = _sha(c1), _sha(c2)
    return out, ref


def test_default_backend_is_bit_identical_and_named_builtin(monkeypatch):
    """HEAD built its solver with a bare `osqp.OSQP()`; this build names `algebra="builtin"`. The same fixtures give
    the same SHAs both ways (any venv), and the E2 golden SHAs of HEAD in `.venv-ag3s` where they were measured."""
    import osqp

    from AG3S.trajopt import qp as Q

    assert QpConfig().solver == "osqp" and TrajOptConfig().qp.solver == "osqp"
    real = osqp.OSQP
    algebras = []

    def spy(*a, **k):
        algebras.append(k.get("algebra"))
        return real(*a, **k)

    def bare(*a, **k):                     # HEAD's call: no algebra argument
        k.pop("algebra", None)
        return real(*a, **k)

    monkeypatch.setattr(osqp, "OSQP", bare)
    head, _ = _default_shas()
    if "venv-ag3s" in sys.prefix:
        assert head == {k: HEAD_SHA[k] for k in head}
    monkeypatch.setattr(osqp, "OSQP", spy)
    now, ref = _default_shas()
    assert now == head and "qp_backend" not in ref.last_result.metrics
    assert algebras and set(algebras) == {"builtin"}
    # osqp ranks the CUDA algebra first among those importable; once it is importable in this process the default
    # path is still the CPU build, and so is a bare `osqp.OSQP()` elsewhere in the process
    if Q.osqp_cuda_available()[0]:
        assert real().algebra == "builtin"
        again, _ = _default_shas()
        assert again == head


# ------------------------------------------------------------------------------------------------ 2. on a GPU
@needs_cuda
def test_cuda_solves_a_small_qp_to_the_cpu_solution():
    from AG3S.trajopt.qp import QpSolver, cuda_mark, cuda_record_since

    problem = _small_qp()
    P, q = problem.P, problem.q
    cfg = dataclasses.replace(QpConfig(), eps_abs=1e-5, eps_rel=1e-5)
    mark = cuda_mark()
    cpu = QpSolver(cfg).solve(problem)
    gpu_solver = QpSolver(dataclasses.replace(cfg, solver="osqp-cuda"))
    gpu = gpu_solver.solve(problem)
    rec = cuda_record_since(mark)
    assert cpu.success and gpu.success and gpu.status == "solved"            # solved on the GPU, no fallback
    assert rec["cuda_solved"] == 1 and rec["cpu_fallback_solved"] == 0 and rec["cuda_setups"] == 1
    assert gpu_solver._backend.algebra == "cuda" and gpu.x.dtype == np.float64
    assert np.abs(gpu.x - cpu.x).max() <= 1e-3                                 # float32 ADMM at eps 1e-5
    obj = lambda x: 0.5 * x @ (P @ x) + q @ x                                    # noqa: E731
    assert abs(obj(gpu.x) - obj(cpu.x)) <= 1e-4 * max(1.0, abs(obj(cpu.x)))
    # warm re-solve after a value-only update keeps the setup
    problem2 = dataclasses.replace(problem, q=q * 1.01)
    again = gpu_solver.solve(problem2)
    assert again.success and cuda_record_since(mark)["cuda_setups"] == 1


@needs_cuda
def test_cuda_on_the_trajopt_fixture_qp_and_the_aligned_settings():
    from AG3S.trajopt import qp as Q

    problem = _fixture_problem()
    cpu = Q.QpSolver(QpConfig()).solve(problem)
    s = Q.QpSolver(dataclasses.replace(QpConfig(), solver="osqp-cuda"))
    gpu = s.solve(problem)
    assert cpu.success and gpu.success
    st, want = s._backend.settings, Q._cpu_defaults()
    for k, v in want.items():
        assert float(getattr(st, k)) == pytest.approx(float(v), rel=1e-6), k   # float32 copies of the CPU defaults
    assert float(st.eps_abs) == pytest.approx(1e-3, rel=1e-6) and int(st.max_iter) == 4000
    assert bool(st.adaptive_rho) and bool(st.warm_starting) and not bool(st.polishing)
    obj = lambda x: 0.5 * x @ (problem.P @ x) + problem.q @ x                    # noqa: E731
    assert abs(obj(gpu.x) - obj(cpu.x)) <= 0.05 * abs(obj(cpu.x))
    assert _residual(problem, gpu.x) <= 5e-3 and _residual(problem, cpu.x) <= 5e-3


@needs_cuda
def test_cuda_optimizer_solve_matches_the_cpu_outcome():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    cpu = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config).solve(sc.reference, sc.q_now, sc.scene)
    gpu = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config.with_overrides(CUDA)).solve(
        sc.reference, sc.q_now, sc.scene)
    assert gpu.status == cpu.status and gpu.metrics["qp_solver"] == "osqp-cuda"
    tol = float(sc.config.safety.violation_tolerance)
    assert cpu.max_violation <= tol and gpu.max_violation <= tol
    assert np.abs(gpu.trajectory - cpu.trajectory).max() < 0.05


# ------------------------------------------------------------------------------------------------ 3. fall-back
def test_failed_cuda_solve_is_resolved_on_the_cpu_and_recorded(monkeypatch):
    from AG3S.trajopt import qp as Q

    real = Q.QpSolver._solve_osqp
    mode = {"m": "status"}

    def solve(self, problem, warm_start, *, x0=None, eps=None, algebra="builtin"):
        if algebra == "cuda":
            if mode["m"] == "raise":
                raise RuntimeError("device lost")
            return Q.QpSolution(np.zeros(problem.n_var), None, False, "primal infeasible", 7, 0.0)
        return real(self, problem, warm_start, x0=x0, eps=eps, algebra=algebra)

    monkeypatch.setattr(Q.QpSolver, "_solve_osqp", solve)
    problem = _fixture_problem()
    cpu_solver = Q.QpSolver(QpConfig())
    cpu = [cpu_solver.solve(problem), cpu_solver.solve(problem)]
    mark = Q.cuda_mark()
    s = Q.QpSolver(dataclasses.replace(QpConfig(), solver="osqp-cuda"))
    a = s.solve(problem)
    rec = Q.cuda_record_since(mark)
    assert a.success and np.array_equal(a.x, cpu[0].x)                       # the CPU solution, same (cold) start
    assert a.status == "solved (cpu fallback; cuda: primal infeasible)" and a.iterations == cpu[0].iterations + 7
    assert rec == {"backend": "osqp-cuda", "qp_solves": 1, "cuda_solved": 0, "cuda_failed": 1, "cuda_errors": 0,
                   "cuda_setups": 0, "cpu_fallback_solved": 1, "cpu_fallback_failed": 0,
                   "failed_statuses": {"primal infeasible": 1}}
    assert s._cpu_fallback.config.solver == "osqp" and np.array_equal(s._last_x, a.x)
    mode["m"] = "raise"
    b = s.solve(problem)                                                     # warm-started like the CPU's 2nd solve
    assert b.success and np.array_equal(b.x, cpu[1].x)
    rec = Q.cuda_record_since(mark)
    assert rec["qp_solves"] == 2 and rec["cuda_errors"] == 1 and rec["cuda_failed"] == 2
    assert rec["failed_statuses"] == {"primal infeasible": 1, "cuda error: RuntimeError: device lost": 1}
    assert s._backend is None                                                # set up again next time
    s.reset()
    assert s._cpu_fallback._backend is None


def test_no_device_falls_back_without_crashing(monkeypatch):
    """A real CUDA library error (no visible device, when there is none) also falls back."""
    from AG3S.trajopt import qp as Q

    if not Q.osqp_cuda_available()[0]:
        pytest.skip("osqp-cu12 extension not installed")
    if _cuda_device():
        pytest.skip("a CUDA device is visible (the GPU tests cover this path)")
    mark = Q.cuda_mark()
    r = Q.QpSolver(dataclasses.replace(QpConfig(), solver="osqp-cuda")).solve(_fixture_problem())
    rec = Q.cuda_record_since(mark)
    assert r.success and "cpu fallback" in r.status and rec["cuda_errors"] == 1 and rec["cpu_fallback_solved"] == 1


def test_missing_extension_is_a_configuration_error(tmp_path):
    """In a fresh process (the extension may already be imported in this one): a missing extension is an
    `ImportError` naming the install command, never a silent CPU run."""
    import os
    import subprocess

    code = (
        "import dataclasses, sys\n"
        "from AG3S.trajopt import qp as Q\n"
        "from AG3S.trajopt.config import QpConfig\n"
        "from tests.trajopt.test_e3a_gpu_qp import _small_qp\n"
        "ok, why = Q.osqp_cuda_available()\n"
        "assert not ok and 'pip install --no-deps --target' in why, why\n"
        "try:\n"
        "    Q.QpSolver(dataclasses.replace(QpConfig(), solver='osqp-cuda')).solve(_small_qp())\n"
        "except ImportError:\n"
        "    print('IMPORT_ERROR')\n"
        "import osqp\n"
        "print(osqp.OSQP().algebra)\n")
    env = {**os.environ, "AG3S_OSQP_CUDA_DIR": str(tmp_path / "nowhere"), "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
                          cwd=os.getcwd())
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["IMPORT_ERROR", "builtin"]


# ------------------------------------------------------------------------------------------------ 4. wiring
def test_standard_and_seed_solves_all_go_through_the_cuda_path(fake_cuda):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sc, scene = _stuck()
    base = sc.config.with_overrides({"horizon": {"plan_horizon": None}, "sqp": {"multi_seed": 16}})
    out = {}
    for name, cfg in (("cpu", base), ("cuda", base.with_overrides(CUDA))):
        ref = TrajOptChunkRefiner(sc.robot, sc.layout, lambda c: (scene, sc.q_now, True), cfg)
        n0 = len(fake_cuda)
        chunk = ref.refine(sc.chunk, {"t_step": 3, "previous_physical_chunk": None})
        out[name] = (chunk, ref.last_result.metrics, len(fake_cuda) - n0, ref)
    (c_cpu, m_cpu, n_cpu, _), (c_gpu, m_gpu, n_gpu, ref) = out["cpu"], out["cuda"]
    assert m_cpu["multi_seed"]["triggered"] and n_cpu == 0 and "qp_backend" not in m_cpu
    assert np.array_equal(c_gpu, c_cpu)                     # the "CUDA" call is the builtin one: same result
    rec = m_gpu["qp_backend"]
    # 1 standard + 16 seeds, one SQP iteration each. A seed QP the warm solver fails is failed by the "CUDA" call too
    # (it is the same builtin call); the fall-back's fresh setup starts cold and may solve it
    assert rec["qp_solves"] == n_gpu == 17 and rec["cuda_solved"] + rec["cuda_failed"] == n_gpu
    assert rec["cpu_fallback_solved"] + rec["cpu_fallback_failed"] == rec["cuda_failed"] and rec["cuda_errors"] == 0
    assert ref.optimizer._ms_solver.config.solver == "osqp-cuda" and ref.optimizer.solver.config.solver == "osqp-cuda"
    seeds_iters = sum(c["sqp_iterations"] for c in m_gpu["multi_seed"]["candidates"])
    assert n_gpu >= seeds_iters                             # every seed QP was counted
    assert set(m_gpu) - set(m_cpu) == {"qp_backend"} and set(m_cpu) <= set(m_gpu)


# ------------------------------------------------------------------------------------------------ 5. config / CLI
def test_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import qp as Q
    from AG3S.trajopt import serve_safe

    assert "osqp-cuda" in QpConfig.SOLVERS and "qpoases" not in QpConfig.SOLVERS
    assert TrajOptConfig.from_dict(CUDA).qp.solver == "osqp-cuda"
    with pytest.raises(TrajOptConfigError, match="polish"):
        TrajOptConfig.from_dict({"qp": {"solver": "osqp-cuda", "polish": True}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"qp": {"solver": "cuda"}})
    assert set(TrajOptConfig.from_dict(CUDA).to_dict()["qp"]) == set(TrajOptConfig().to_dict()["qp"])   # no new key
    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert parse().qp_backend == "osqp"
    assert serve_safe.qp_overrides(parse()) == {} and serve_safe.qp_overrides(parse("--qp-backend", "osqp")) == {}
    assert serve_safe.trajopt_config_from_args(parse()).to_dict() == \
        serve_safe.trajopt_config_from_args(parse("--qp-backend", "osqp")).to_dict()
    assert serve_safe.qp_overrides(parse("--qp-backend", "osqp-cuda")) == {"solver": "osqp-cuda"}
    assert serve_safe.qp_overrides(parse("--qp-backend", "osqp-cuda", "--qp-eps", "1e-4")) == \
        {"eps_abs": 1e-4, "eps_rel": 1e-4, "solver": "osqp-cuda"}
    assert serve_safe.trajopt_config_from_args(parse("--qp-backend", "osqp-cuda")).qp.solver == "osqp-cuda"
    with pytest.raises(SystemExit):
        parse("--qp-backend", "cuda")
    capsys.readouterr()
    monkeypatch.setattr(Q, "osqp_cuda_available", lambda: (False, "extension missing here"))
    for extra, needle in ((["--no-safe", "--qp-backend", "osqp-cuda"], "--qp-backend"),
                          (["--qp-backend", "osqp-cuda"], "extension missing here")):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err


def test_offline_settings_are_registered_and_valid():
    from AG3S.trajopt.experiments import e2_multi_seed as E2
    from AG3S.trajopt.experiments import e3a_gpu_qp as E3

    S = E3.SETTINGS
    assert set(S) >= set(E2.SETTINGS) | {"E3a_0", "E3a_16", "SMOKE_IT1_E3a_0", "SMOKE_IT1_E3a_16"}
    base = TrajOptConfig()
    e0, e2, a0, a16 = (base.with_overrides(S[n]) for n in ("E0", "E2_16", "E3a_0", "E3a_16"))
    assert a0.qp.solver == a16.qp.solver == "osqp-cuda" and e0.qp.solver == e2.qp.solver == "osqp"
    assert dataclasses.replace(a0.qp, solver="osqp") == e0.qp and a0.sqp == e0.sqp and a0.sqp.multi_seed == 0
    assert dataclasses.replace(a16.qp, solver="osqp") == e2.qp and a16.sqp == e2.sqp and a16.sqp.multi_seed == 16
    assert E2.SETTINGS["E2_16"] == {"sqp": {"multi_seed": 16}}                       # untouched
    s1 = base.with_overrides(S["SMOKE_IT1_E3a_16"])
    assert s1.sqp.max_iterations == 1 and s1.sqp.multi_seed == 16 and s1.qp.solver == "osqp-cuda"


def test_summarize_is_e2_plus_the_backend_counts(tmp_path):
    from AG3S.trajopt.experiments import e2_multi_seed as E2
    from AG3S.trajopt.experiments import e3a_gpu_qp as E3

    def var(viol, kind=None, qb=None):
        e = {"max_violation_mm": viol, "solve_ms": 30.0, "status": "feasible"}
        if kind is not None:
            e["multi_seed"] = {"triggered": True, "chosen_kind": kind, "total_ms": 300.0}
        if qb is not None:
            e["qp_backend"] = {"backend": "osqp-cuda", "qp_solves": qb[0], "cuda_solved": qb[0] - qb[1],
                               "cuda_failed": qb[1], "cuda_errors": 0, "cuda_setups": 1,
                               "cpu_fallback_solved": qb[1], "cpu_fallback_failed": 0,
                               "failed_statuses": {"maximum iterations reached": qb[1]} if qb[1] else {}}
        return e

    chunks = [{"seq": 1, "variants": {"E0": var(0.0), "E2_16": var(0.0), "E3a_16": var(0.0, qb=(3, 0))}},
              {"seq": 2, "variants": {"E0": var(5.0), "E2_16": var(0.0, "detour"), "E3a_16": var(0.0, "detour", (40, 2))}},
              {"seq": 3, "variants": {"E0": var(3.0), "E2_16": var(0.0, "stay"), "E3a_16": var(1.0, "standard", (50, 0))}}]
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"key": "K", "chunks": chunks}))
    out = E3.summarize([str(p)])
    ref = E2.summarize([str(p)])
    assert out["per_setting"] == ref["per_setting"] and out["per_run"] == ref["per_run"]
    assert out["per_setting"]["E3a_16"]["n_resolved"] == 1 and out["per_setting"]["E2_16"]["n_resolved"] == 2
    qb = out["qp_backend"]
    assert set(qb) == {"E3a_16"}
    assert qb["E3a_16"]["all"]["qp_solves"] == 93 and qb["E3a_16"]["all"]["chunks"] == 3
    assert qb["E3a_16"]["base_violated"]["qp_solves"] == 90 and qb["E3a_16"]["base_violated"]["cpu_fallback_solved"] == 2
    assert qb["E3a_16"]["all"]["failed_statuses"] == {"maximum iterations reached": 2}
