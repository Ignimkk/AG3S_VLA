"""T63 step 2 — the TO objective terms put on cuRobo's trajectory (`curobo_v2/objective_ref.py`,
`curobo_v2/costs.py`), the chunk-row matrix (`spec.chunk_row_weights`), the composed-world rules
(`world.compose_box` / `compose_lte`) and the record-context rebuild (`record_context.py`).

Chain of references: `problem.py` (the TO, byte-identical to snapshot f0d8046) -> `objective_ref`
(numpy, value + analytic gradient) -> `costs.*Term` (torch, on the optimizer state grid).
torch tests skip without torch (`.venv-ag3s`); the `problem.py` tests skip without scipy."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from AG3S.trajopt.curobo_v2 import objective_ref as O
from AG3S.trajopt.curobo_v2 import spec as S
from AG3S.trajopt.curobo_v2 import world as W

NQ = 14


def _rand(rng, *shape, scale=0.3):
    return rng.normal(scale=scale, size=shape)


def _cfg(w_track=1.0, w_smooth=0.0, w_continuity=0.0, w_slack=1000.0, track_decay=1.0,
         w_deflection_rate=0.0, servo_accel="off", servo_accel_weight=0.01):
    cost = types.SimpleNamespace(w_track=w_track, w_smooth=w_smooth, w_continuity=w_continuity,
                                 w_slack=w_slack, track_decay=track_decay,
                                 w_deflection_rate=w_deflection_rate)
    limits = types.SimpleNamespace(servo_accel=servo_accel, servo_accel_weight=servo_accel_weight)
    return types.SimpleNamespace(cost=cost, limits=limits)


# =================================================================================================
# objective_ref == problem.py
# =================================================================================================

@pytest.fixture(scope="module")
def problem():
    pytest.importorskip("scipy")
    from AG3S.trajopt import problem as P

    return P


@pytest.mark.parametrize("H", [16, 50])
def test_tracking_continuity_smooth_equal_problem_objective(problem, H):
    rng = np.random.default_rng(1)
    Q, R = _rand(rng, NQ, H), _rand(rng, NQ, H)
    prev = _rand(rng, NQ, 11)  # shorter than H: overlap rule
    for kw in (dict(w_track=1.0, track_decay=0.9), dict(w_track=1e-9, w_continuity=0.5),
               dict(w_track=1e-9, w_smooth=0.05)):
        cfg = _cfg(**kw)
        want = problem.objective(Q, R, cfg, previous_chunk=prev)
        got = (O.tracking_value(Q, R, cfg.cost.w_track, cfg.cost.track_decay)
               + O.continuity_value(Q, prev, cfg.cost.w_continuity)
               + O.smooth_value(Q, cfg.cost.w_smooth))
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("with_history", [False, True])
def test_deflection_rate_equals_problem(problem, with_history):
    rng = np.random.default_rng(2)
    Q, R = _rand(rng, NQ, 16), _rand(rng, NQ, 16)
    d = _rand(rng, NQ) if with_history else None
    want = problem.deflection_rate_cost(Q, R, 20.0, d)
    assert O.deflection_rate_value(Q, R, 20.0, d) == pytest.approx(want, rel=1e-12)
    # and through `objective` (w_track tiny so the deflection term dominates the comparison)
    cfg = _cfg(w_track=1e-9, w_deflection_rate=20.0)
    obj = problem.objective(Q, R, cfg, deflection_history=d)
    assert obj - O.tracking_value(Q, R, 1e-9) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
@pytest.mark.parametrize("with_history", [False, True])
def test_servo_accel_rows_and_value_equal_problem(problem, mode, with_history):
    rng = np.random.default_rng(3)
    H = 16
    R = np.cumsum(_rand(rng, NQ, H, scale=0.02), axis=1)
    Q = R + _rand(rng, NQ, H, scale=0.01)
    bound = np.full(NQ, 0.004)
    msc = np.full(NQ, 0.02)
    hist = _rand(rng, NQ, 2, scale=0.05) + R[:, :1] if with_history else None
    over_problem = problem.servo_accel_overshoot(Q, R, bound, mode, msc, hist)
    A, const, b = O.servo_accel_rows(R, bound, mode, msc, hist)
    over_ref = np.maximum(0.0, np.abs(A @ Q.T + const) - b).reshape(-1)  # row-major = step-major
    np.testing.assert_allclose(over_ref, over_problem, rtol=0, atol=1e-15)
    max_step = np.full(NQ, 0.01)
    cfg = _cfg(w_track=1e-9, servo_accel=mode)
    obj = problem.objective(Q, R, cfg, servo_accel_bound=bound, max_step_change=msc,
                            command_history=hist, max_step=max_step)
    got = O.servo_accel_value(Q, R, bound, mode, w_slack=1000.0, servo_accel_weight=0.01,
                              max_step_change=msc, command_history=hist, max_step=max_step)
    assert obj - O.tracking_value(Q, R, 1e-9) == pytest.approx(got, rel=1e-9, abs=1e-12)


# =================================================================================================
# analytic gradients of objective_ref (finite differences)
# =================================================================================================

def _fd(f, Q, eps=1e-6):
    g = np.zeros_like(Q)
    for idx in np.ndindex(Q.shape):
        Qp, Qm = Q.copy(), Q.copy()
        Qp[idx] += eps
        Qm[idx] -= eps
        g[idx] = (f(Qp) - f(Qm)) / (2 * eps)
    return g


def test_reference_gradients_match_finite_differences():
    rng = np.random.default_rng(4)
    nq, H = 3, 9
    Q, R, P, d = _rand(rng, nq, H), _rand(rng, nq, H), _rand(rng, nq, 6), _rand(rng, nq)
    cases = [
        (lambda X: O.tracking_value(X, R, 1.3, 0.9), O.tracking_grad(Q, R, 1.3, 0.9)),
        (lambda X: O.continuity_value(X, P, 0.5), O.continuity_grad(Q, P, 0.5)),
        (lambda X: O.smooth_value(X, 0.05), O.smooth_grad(Q, 0.05)),
        (lambda X: O.deflection_rate_value(X, R, 20.0, d), O.deflection_rate_grad(Q, R, 20.0, d)),
        (lambda X: O.deflection_rate_value(X, R, 20.0), O.deflection_rate_grad(Q, R, 20.0)),
    ]
    for f, g in cases:
        np.testing.assert_allclose(g, _fd(f, Q), rtol=1e-6, atol=1e-6)
    # servo: L1 hinge — check away from the kinks
    Rs = np.cumsum(_rand(rng, nq, H, scale=0.02), axis=1)
    Qs = Rs + _rand(rng, nq, H, scale=0.02)
    hist = Rs[:, :2] + _rand(rng, nq, 2, scale=0.03)
    kw = dict(w_slack=1000.0, servo_accel_weight=0.01, max_step_change=np.full(nq, 0.05),
              command_history=hist, max_step=np.full(nq, 0.01))
    A, c, b = O.servo_accel_rows(Rs, np.full(nq, 0.004), "strict", kw["max_step_change"], hist)
    assert np.min(np.abs(np.abs(A @ Qs.T + c) - b)) > 1e-4  # no kink within eps
    g = O.servo_accel_grad(Qs, Rs, np.full(nq, 0.004), "strict", **kw)
    fd = _fd(lambda X: O.servo_accel_value(X, Rs, np.full(nq, 0.004), "strict", **kw), Qs, 1e-7)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-5)


# =================================================================================================
# chunk rows on the state grid
# =================================================================================================

@pytest.mark.parametrize("horizon,exact", [(101, True), (81, False)])
def test_chunk_row_weights_equal_sample_chunk_grid(horizon, exact):
    rng = np.random.default_rng(5)
    span = 50.0 / 15.0
    pos = np.cumsum(_rand(rng, horizon, NQ, scale=0.01), axis=0)
    Wm, owner = S.chunk_row_weights(horizon, 50, 1.0 / 15.0, span)
    want = S.sample_chunk_grid(pos, span / (horizon - 1), 1.0 / 15.0, 50)
    np.testing.assert_allclose(Wm @ pos, want, rtol=0, atol=1e-12)
    assert np.allclose(Wm.sum(1), 1.0)
    if exact:
        assert set(np.unique(Wm)) <= {0.0, 1.0}
        np.testing.assert_array_equal(owner, 2 * np.arange(1, 51))
    W16, _ = S.chunk_row_weights(horizon, 16, 1.0 / 15.0, span)
    np.testing.assert_array_equal(W16, Wm[:16])


# =================================================================================================
# torch terms == objective_ref (value and gradient), on the optimizer state grid
# =================================================================================================

def _torch_case(term, pos_np, value_fn, grad_fn, rows):
    torch = pytest.importorskip("torch")
    pos = torch.tensor(pos_np, dtype=torch.float64, requires_grad=True)
    state = types.SimpleNamespace(joint_state=types.SimpleNamespace(position=pos))
    out = term.forward(state, None)
    assert tuple(out.shape) == (pos.shape[0], pos.shape[1], 1)
    out.sum().backward()
    Wm, _ = S.chunk_row_weights(pos.shape[1], rows, 1.0 / 15.0, 50.0 / 15.0)
    for b in range(pos.shape[0]):
        Q = (Wm @ pos_np[b]).T  # (nq, rows)
        assert float(out[b].sum()) == pytest.approx(value_fn(Q), rel=1e-10, abs=1e-12)
        g_state = Wm.T @ grad_fn(Q).T  # (H, nq)
        np.testing.assert_allclose(pos.grad[b].numpy(), g_state, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("horizon", [101, 81])
@pytest.mark.parametrize("rows", [50, 16])
def test_torch_terms_match_reference(horizon, rows):
    torch = pytest.importorskip("torch")
    from AG3S.trajopt.curobo_v2 import costs as C

    rng = np.random.default_rng(6)
    pos = np.cumsum(_rand(rng, 2, horizon, NQ, scale=0.01), axis=1)
    ref = np.cumsum(_rand(rng, 50, NQ, scale=0.01), axis=0)
    R = ref[:rows].T
    kw = dict(rows=rows, dtype=torch.float64)

    t = C.TrackTerm(1.0, 0.95, **kw)
    t.set_reference(ref)
    _torch_case(t, pos, lambda Q: O.tracking_value(Q, R, 1.0, 0.95),
                lambda Q: O.tracking_grad(Q, R, 1.0, 0.95), rows)

    tail = np.cumsum(_rand(rng, 42, NQ, scale=0.01), axis=0)
    c = C.ContinuityTerm(0.5, **kw)
    assert c.set_previous(tail[:rows]) == min(42, rows)
    P = tail[:rows].T
    _torch_case(c, pos, lambda Q: O.continuity_value(Q, P, 0.5),
                lambda Q: O.continuity_grad(Q, P, 0.5), rows)
    assert c.set_previous(None) == 0
    _torch_case(c, pos, lambda Q: 0.0, lambda Q: np.zeros_like(Q), rows)

    s = C.SmoothTerm(0.05, **kw)
    _torch_case(s, pos, lambda Q: O.smooth_value(Q, 0.05), lambda Q: O.smooth_grad(Q, 0.05), rows)

    d_prev = _rand(rng, NQ, scale=0.01)
    d = C.DeflectionRateTerm(20.0, **kw)
    d.set_reference(ref)
    for hist in (d_prev, None):
        d.set_history(hist)
        _torch_case(d, pos, lambda Q: O.deflection_rate_value(Q, R, 20.0, hist),
                    lambda Q: O.deflection_rate_grad(Q, R, 20.0, hist), rows)


@pytest.mark.parametrize("mode", ["strict", "relaxed"])
@pytest.mark.parametrize("with_history", [False, True])
def test_torch_servo_accel_matches_reference(mode, with_history):
    torch = pytest.importorskip("torch")
    from AG3S.trajopt.curobo_v2 import costs as C

    rng = np.random.default_rng(7)
    rows, horizon = 16, 101
    ref = np.cumsum(_rand(rng, 50, NQ, scale=0.02), axis=0)
    # a state grid near the reference path (state 0 = start), plus noise
    path = np.vstack([ref[:1] - 0.01, ref])
    t_path = np.arange(51) / 15.0
    t_state = np.arange(horizon) * (50.0 / 15.0) / (horizon - 1)
    base = np.stack([np.interp(t_state, t_path, path[:, j]) for j in range(NQ)], axis=1)
    pos = base[None] + _rand(rng, 2, horizon, NQ, scale=0.02)
    R = ref[:rows].T
    bound, msc, max_step = np.full(NQ, 0.004), np.full(NQ, 0.05), np.full(NQ, 0.01)
    hist = (ref[:2] + _rand(rng, 2, NQ, scale=0.03)).T if with_history else None
    kw = dict(w_slack=1000.0, servo_accel_weight=0.01, max_step_change=msc,
              command_history=hist, max_step=max_step)
    term = C.ServoAccelTerm(1000.0, 0.01, rows=rows, dtype=torch.float64)
    term.set_chunk(ref, bound, mode, max_step_change=msc, command_history=hist, max_step=max_step)
    Wm, _ = S.chunk_row_weights(horizon, rows, 1.0 / 15.0, 50.0 / 15.0)
    for b in range(2):
        Q = (Wm @ pos[b]).T
        A, cst, bd = O.servo_accel_rows(R, bound, mode, msc, hist)
        assert np.min(np.abs(np.abs(A @ Q.T + cst) - bd)) > 1e-6  # away from kinks
    _torch_case(term, pos, lambda Q: O.servo_accel_value(Q, R, bound, mode, **kw),
                lambda Q: O.servo_accel_grad(Q, R, bound, mode, **kw), rows)
    term.set_chunk(ref, None, mode)  # servo_accel = off for this chunk
    _torch_case(term, pos, lambda Q: 0.0, lambda Q: np.zeros_like(Q), rows)


def test_terms_have_names_usable_by_the_hook():
    pytest.importorskip("torch")
    pytest.importorskip("curobo")
    from AG3S.trajopt.curobo_v2 import costs as C

    terms = [C.TrackTerm(1.0, rows=50), C.ContinuityTerm(0.5, rows=50),
             C.DeflectionRateTerm(20.0, rows=50)]
    cfg = C.extensible_cost_manager_cfg(terms)
    assert [t.name for t in cfg.extra_terms] == ["to_track", "to_continuity", "to_deflection_rate"]


# =================================================================================================
# composed-world rules (numpy reference of composed_world.py)
# =================================================================================================

def test_compose_box_and_lte():
    coarse = np.array([0.10, 0.02, 0.30, 0.05])
    fine = np.array([0.08, 0.04, 0.01, 0.03])
    free = np.array([0.09, 0.06, 0.20, 0.04])
    in_box = np.array([True, True, False, True])
    np.testing.assert_allclose(W.compose_box(free, in_box, coarse), [0.09, 0.06, 0.30, 0.04])
    excl = np.array([True, False, True, False])
    want = [0.09, min(0.02, 0.04), 0.30, min(0.05, 0.03)]
    np.testing.assert_allclose(W.compose_lte(coarse, fine, in_box, free, in_box, excl), want)


def test_min_composition_is_max_of_curobo_costs():
    """`act` is non-increasing in sdf, so the cost of `min(sdf_a, sdf_b)` is `max(cost_a, cost_b)`
    — what `ComposedSceneCollision` uses to compose layers in cost space."""
    def act(sdf, r, eta):
        p = r + eta - sdf
        return np.where(p > eta, p - eta / 2, np.where(p > 0, p * p / (2 * eta), 0.0))

    rng = np.random.default_rng(8)
    a, b = rng.uniform(-0.05, 0.1, 1000), rng.uniform(-0.05, 0.1, 1000)
    r, eta = 0.006, 0.0025
    np.testing.assert_allclose(act(np.minimum(a, b), r, eta),
                               np.maximum(act(a, r, eta), act(b, r, eta)))


# =================================================================================================
# record context
# =================================================================================================

def _record(path, k, t_step, refined, reference, summary):
    np.savez(path / f"chunk_{k:05d}.npz", chunk_index=np.asarray(k), t_step=np.asarray(t_step),
             refined_chunk=refined.astype(np.float32), reference_chunk=reference.astype(np.float32),
             summary_json=np.asarray(json.dumps(summary)))


def test_record_context_rebuilds_the_server_inputs(tmp_path):
    from AG3S.trajopt.curobo_v2.record_context import chunk_record_context

    rng = np.random.default_rng(9)
    prev_ref, prev_ref_fin = _rand(rng, 50, 16), _rand(rng, 50, 16)
    cols = [*range(7), *range(8, 15)]
    applied = prev_ref_fin[:8][:, cols]
    d = applied[-1] - prev_ref.astype(np.float32).astype(np.float64)[7][cols]
    _record(tmp_path, 9, 5, prev_ref_fin, prev_ref, {})
    _record(tmp_path, 10, 6, _rand(rng, 50, 16), _rand(rng, 50, 16), {
        "exec_continuity": {"used": True, "previous_chunk": "refined", "aligned_from_step": 8,
                            "plan_seq": 5},
        "exec_feedback": {"available": True, "seq": 5, "executed": [True] * 8,
                          "applied_arm": applied.tolist()},
        "ta_history": {"deflection": {"row": 7, "max_abs_rad": float(np.abs(d).max())}}})
    ctx = chunk_record_context(tmp_path / "chunk_00010.npz")
    np.testing.assert_allclose(ctx["continuity_tail"],
                               prev_ref_fin.astype(np.float32)[8:][:, cols], atol=0)
    assert ctx["continuity_tail"].shape == (42, 14)
    np.testing.assert_allclose(ctx["deflection_history"], d, atol=1e-6)
    assert ctx["deflection_check_abs_err_rad"] < 1e-6
    np.testing.assert_allclose(ctx["command_history"], applied[-2:].T)

    # not used / no history -> None with a reason
    _record(tmp_path, 11, 7, _rand(rng, 50, 16), _rand(rng, 50, 16), {
        "exec_continuity": {"used": False, "mode": "feedback", "off_by": "latch"},
        "ta_history": {"deflection": None, "deflection_note": "last row was a HOLD"}})
    ctx = chunk_record_context(tmp_path / "chunk_00011.npz")
    assert ctx["continuity_tail"] is None and ctx["deflection_history"] is None
    assert "latch" in ctx["notes"]["continuity"] and "HOLD" in ctx["notes"]["deflection"]


def test_chunk_rows_are_not_tf32_on_cuda():
    """cuRobo turns TF32 matmul on globally (`seed_ik_solver.py`). The terms read chunk rows by
    gather + elementwise ops, so rows of ~1 rad positions stay float32-exact (a TF32 matmul is
    off by ~1e-4 rad). CUDA only."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from AG3S.trajopt.curobo_v2 import costs as C

    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        rng = np.random.default_rng(10)
        for horizon in (101, 81):
            pos = 1.0 + _rand(rng, 4, horizon, NQ, scale=0.5)
            term = C.TrackTerm(1.0, rows=50, device="cuda")
            got = term.chunk_rows(torch.as_tensor(pos, dtype=torch.float32, device="cuda"))
            Wm, _ = S.chunk_row_weights(horizon, 50, 1.0 / 15.0, 50.0 / 15.0)
            want = np.einsum("kh,bhj->bkj", Wm, pos)
            assert np.abs(got.cpu().numpy() - want).max() < 1e-6
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
