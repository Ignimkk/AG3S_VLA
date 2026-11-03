"""T93 — steer the VLA's flow-matching sampler with the TrajOpt collision cost, behind `serve_safe --fm-guide` (default off).

Task: `AG3S/ag3s/docs/handoff/T93.task.md`. Report: `T93.impl.md`.

CPU part (`.venv-ag3s`, no JAX needed):
1. off = as before: no `noise` kwarg to the policy, no `fm_resample` in the refiner context, no `to.metrics.fm_guide`, no kwarg to
   `SafePolicy`; the new function is replaced by one that raises and the off path never calls it.
2. violation <= 0: the VLA is not asked again (the resampler raises if it is) and the refined chunk is bit-identical.
3. the output transform is per-dimension affine: finite difference of the real transform = the diagonal scale used for the chain
   rule; a transform that is not affine stops the build.
4. `guidance_update`: chain rule, clip, padding / zero-scale columns, and a synthetic Euler flow whose final cost drops.
5. `guide_reference`: used only when the violation is lower, gripper / padding columns stay, a failing resampler keeps the chunk,
   the linearizer's per-call records are restored.
6. `serve_safe`: parse, refusals, kwargs, defaults.

JAX part (`.venv-openpi-live`, `JAX_PLATFORMS=cpu`): `Pi0.sample_actions` is unchanged without the new hook (SEAM's `guidance_fn`
path included), the Python-loop sampler reproduces it, and `FlowGuide.resample` on a small random Pi0 lowers the violation of a
real RB-Y1 scene while keeping the gripper columns.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from AG3S.trajopt import fm_guide as FG
from AG3S.trajopt import serve_safe
from AG3S.trajopt import wire
from AG3S.trajopt.refiner import TrajOptChunkRefiner
from AG3S.trajopt.safe_policy import SafePolicy

pytest.importorskip("casadi")
pytest.importorskip("osqp")
from tests.trajopt.fixtures import make_scenario  # noqa: E402

BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(argv):
    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE + list(argv))


def _check(argv):
    ap, a = _parse(argv)
    serve_safe.reject_bad_flag_combinations(ap, a)
    return a


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=12, penetration=0.03)


@pytest.fixture(scope="module")
def clear_scenario():
    # the obstacle sits 5 cm beyond the required clearance: violation < 0
    return make_scenario(horizon=12, penetration=-0.05)


def _refiner(sc):
    return TrajOptChunkRefiner(sc.robot, sc.layout, lambda ctx: (sc.scene, sc.q_now, True), sc.config)


class _Guide:
    """What `BoundResample.guide` is for `guide_reference`: `.config` and `.sweep` (empty)."""
    config = FG.FmGuideConfig()
    sweep = ()


class _Resample:
    """A resampler double: `fn(oracle, original) -> chunk`; counts calls."""

    def __init__(self, fn=None, skip=None):
        self.fn, self.skip, self.calls = fn, skip, 0
        self.guide = _Guide()

    def __call__(self, oracle, original):
        self.calls += 1
        if self.fn is None:
            raise AssertionError("the VLA was asked again although the chunk does not violate")
        return FG.GuideOutcome(chunk=self.fn(oracle, original), info={"steps": 3, "cost_per_step": [2.0, 1.0, 0.5]})


def _descend(oracle, chunk, lam=40.0, iters=30):
    """A synthetic 'guided resampler': plain gradient descent of the oracle's cost on the chunk."""
    x = np.array(chunk, np.float64)
    for _ in range(iters):
        _, g = oracle.cost_grad(x)
        x = x - lam * g
    return x


# ------------------------------------------------------------------------------------------------ 1. off = before
def test_off_refiner_never_calls_guide_reference(scenario, monkeypatch):
    monkeypatch.setattr(FG, "guide_reference", lambda *a, **k: pytest.fail("guide_reference called with the flag off"))
    r = _refiner(scenario)
    out = r.refine(scenario.chunk)
    assert "fm_guide" not in r.last_result.metrics
    assert out.shape == scenario.chunk.shape
    out2 = _refiner(scenario).refine(scenario.chunk, {"t_step": 3})        # a context without the key: same
    assert np.array_equal(out, out2)


def test_off_safe_policy_passes_no_noise_and_no_context_key():
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    stub = _StubPolicy()
    seen = {}
    orig = stub.infer

    def infer(obs, **kw):
        seen["kwargs"] = dict(kw)
        return orig(obs, **kw)

    stub.infer = infer
    sp = _safe_policy(stub)
    assert sp.fm_guide is None
    ctx = {}
    real = sp.refiner.refine
    sp.refiner.refine = lambda chunk, context=None: (ctx.update(context or {}), real(chunk, context))[1]
    sp.infer({"state": np.zeros(wire.ACTION_WIDTH), wire.PREFIX + "cameras": [], wire.PREFIX + "seq": 1})
    assert seen["kwargs"] == {}
    assert "fm_resample" not in ctx


def test_on_safe_policy_reads_the_noise_before_infer_and_calls_infer_unchanged():
    """On: the policy is called exactly as with the flag off (no `noise=`: the same compiled graph, bit-identical chunk); the
    noise was read BEFORE the call (infer advances the key) and rides to the refiner through the context."""
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    order, seen, ctx = [], {}, {}
    noise = np.arange(6.0).reshape(1, 2, 3)

    class _FakeGuide:
        def peek_noise(self):
            order.append("peek")
            return noise

        def bind(self, policy_obs, n, *, skip=None):
            return ("bound", policy_obs, n, skip)

    stub = _StubPolicy()
    orig = stub.infer

    def infer(obs, **kw):
        order.append("infer")
        seen["kwargs"] = dict(kw)
        return orig(obs, **kw)

    stub.infer = infer
    sp = _safe_policy(stub, fm_guide=_FakeGuide())
    real = sp.refiner.refine
    sp.refiner.refine = lambda chunk, context=None: (ctx.update(context or {}), real(chunk, context))[1]
    sp.infer({"state": np.zeros(wire.ACTION_WIDTH), wire.PREFIX + "cameras": [], wire.PREFIX + "seq": 1})
    assert order == ["peek", "infer"]
    assert seen["kwargs"] == {}
    tag, policy_obs, n, skip = ctx["fm_resample"]
    assert tag == "bound" and n is noise and skip is None and set(policy_obs) == {"state"}


def test_off_server_side_has_no_kwarg():
    _, a = _parse([])
    assert a.fm_guide is False
    assert serve_safe.fm_guide_config_from_args(a) is None
    assert serve_safe.fm_guide_build(a, None) == {}
    src = inspect.getsource(serve_safe.main)
    assert "**fm_guide_build(args" in src
    assert inspect.signature(SafePolicy.__init__).parameters["fm_guide"].default is None


def test_record_meta_only_when_on():
    _, a = _parse([])
    assert serve_safe.fm_guide_record_meta(a) == {}
    _, a = _parse(["--fm-guide", "--fm-guide-scale", "7"])
    assert serve_safe.fm_guide_record_meta(a) == {"fm_guide": {"scale": 7.0, "steps": FG.DEFAULT_STEPS,
                                                                "max_step_rad": FG.DEFAULT_MAX_STEP_RAD, "num_steps": None}}


def test_off_flag_changes_no_other_argument():
    _, base = _parse([])
    _, on = _parse(["--fm-guide"])
    assert {k for k in vars(on) if vars(on)[k] != vars(base).get(k)} == {"fm_guide"}


# ------------------------------------------------------------------------------------------------ 2. violation <= 0
def test_no_violation_means_no_resample_and_a_bit_identical_output(clear_scenario):
    sc = clear_scenario
    assert sc.reference_violation < 0
    base = _refiner(sc)
    out_base = base.refine(sc.chunk)
    rs = _Resample()                                   # raises if called
    on = _refiner(sc)
    out_on = on.refine(sc.chunk, {"fm_resample": rs})
    assert rs.calls == 0
    assert np.array_equal(out_on, out_base)
    rec = on.last_result.metrics["fm_guide"]
    assert rec["applied"] is False and rec["used"] is False and rec["reason"] == "no violation"
    assert rec["violation_before_m"] == 0.0
    # the rest of the solve record is the one without the flag
    a, b = dict(base.last_result.metrics), dict(on.last_result.metrics)
    b.pop("fm_guide")
    assert a.keys() == b.keys()


def test_skip_reason_keeps_the_chunk(scenario):
    r = _refiner(scenario)
    out_off = _refiner(scenario).refine(scenario.chunk)
    out = r.refine(scenario.chunk, {"fm_resample": _Resample(skip="recovery retreat")})
    assert np.array_equal(out, out_off)
    assert r.last_result.metrics["fm_guide"]["reason"] == "recovery retreat"
    assert r.last_result.metrics["fm_guide"]["applied"] is False


# ------------------------------------------------------------------------------------------------ 3. affine output transform
def _toy_transform(q01, q99, mask_n, state_dims=16, adapt=False):
    """Quantile unnormalize (affine) -> delta -> absolute on the joint dims -> slice: the RB-Y1 output path in numpy."""
    def f(data):
        x, st = np.asarray(data["actions"], np.float64), np.asarray(data["state"], np.float64)
        n = q01.shape[-1]
        u = np.concatenate([(x[..., :n] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., n:]], axis=-1)
        s = np.concatenate([(st[..., :n] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, st[..., n:]], axis=-1)
        a = u[..., :state_dims].copy()
        a[..., mask_n] += s[..., mask_n][None, :]
        return {"actions": a}
    return f


def test_output_affine_diagonal_scale_matches_finite_difference():
    rng = np.random.default_rng(1)
    q01 = -rng.uniform(0.2, 1.5, 16)
    q99 = rng.uniform(0.2, 1.5, 16)
    mask = np.array([i for i in range(16) if i not in (7, 15)])
    f = _toy_transform(q01, q99, mask)
    state = rng.uniform(-1, 1, 32)
    aff = FG.OutputAffine.measure(f, state, horizon=6, action_dim=32)
    # central finite difference of the transform, column by column, at a random point
    x = rng.standard_normal((6, 32))
    eps = 1e-3
    for j in range(16):
        e = np.zeros_like(x)
        e[:, j] = eps
        fd = (f({"state": state, "actions": x + e})["actions"] - f({"state": state, "actions": x - e})["actions"]) / (2 * eps)
        assert np.allclose(fd[:, j], aff.scale[j], rtol=1e-6, atol=1e-8)
        others = np.delete(fd, j, axis=1)
        assert np.abs(others).max() < 1e-8                      # diagonal: no column couples to another
    assert np.allclose(aff.scale, (q99 - q01 + 1e-6) / 2.0)
    assert np.allclose(aff.to_physical(x), f({"state": state, "actions": x})["actions"])
    # padding columns (16..31) do not enter at all
    x2 = x.copy()
    x2[:, 16:] += 5.0
    assert np.allclose(aff.to_physical(x2), aff.to_physical(x))


def test_output_affine_refuses_what_is_not_per_dimension_affine():
    state = np.zeros(32)

    def nonlinear(d):
        return {"actions": np.tanh(np.asarray(d["actions"], np.float64))[:, :16]}

    def coupled(d):
        a = np.asarray(d["actions"], np.float64)[:, :16].copy()
        a[:, 1] += 0.3 * a[:, 0]
        return {"actions": a}

    with pytest.raises(FG.FmGuideError):
        FG.OutputAffine.measure(nonlinear, state, 5, 32)
    with pytest.raises(FG.FmGuideError):
        FG.OutputAffine.measure(coupled, state, 5, 32)


# ------------------------------------------------------------------------------------------------ 4. one flow step
class _QuadOracle:
    """C(phys) = 1/2 sum (phys - goal)^2 on the first 16 columns; gradient = phys - goal."""

    def __init__(self, goal):
        self.goal = goal

    def cost_grad(self, phys):
        d = phys - self.goal
        return 0.5 * float(np.sum(d * d)), d


def _affine(scale):
    return FG.OutputAffine(offset=np.zeros((5, scale.size)), scale=np.asarray(scale, np.float64))


def test_guidance_update_is_the_chain_rule_and_descends():
    scale = np.linspace(0.3, 1.2, 16)
    aff = _affine(scale)
    goal = np.full((5, 16), 0.4)
    x = np.random.default_rng(0).standard_normal((5, 32)) * 0.2
    orc = _QuadOracle(goal)
    dx, cost, _ = FG.guidance_update(x, orc, aff, scale=0.1, max_step_rad=10.0)
    phys = aff.to_physical(x)
    g_x = scale[None, :] * (phys - goal)                       # dC/dx by the chain rule
    assert np.allclose(dx[:, :16], -0.1 * g_x)
    assert np.all(dx[:, 16:] == 0.0)                           # padding columns: no gradient
    # a numerical gradient of C(x) agrees with g_x
    eps = 1e-6
    num = np.zeros_like(g_x)
    for t, j in [(0, 0), (2, 5), (4, 15)]:
        xp, xm = x.copy(), x.copy()
        xp[t, j] += eps
        xm[t, j] -= eps
        num[t, j] = (orc.cost_grad(aff.to_physical(xp))[0] - orc.cost_grad(aff.to_physical(xm))[0]) / (2 * eps)
        assert num[t, j] == pytest.approx(g_x[t, j], rel=1e-5, abs=1e-8)
    assert orc.cost_grad(aff.to_physical(x + dx))[0] < cost


def test_guidance_update_clip_is_in_physical_radians_and_zero_scale_is_safe():
    scale = np.linspace(0.3, 1.2, 16)
    scale[3] = 0.0
    aff = _affine(scale)
    x = np.zeros((5, 32))
    orc = _QuadOracle(np.full((5, 16), 3.0))
    dx, _, change = FG.guidance_update(x, orc, aff, scale=100.0, max_step_rad=0.05)
    assert change <= 0.05 + 1e-12
    assert np.abs(aff.scale[None, :] * dx[:, :16]).max() <= 0.05 + 1e-12
    assert np.all(dx[:, 3] == 0.0)
    assert np.isfinite(dx).all()


def test_synthetic_euler_flow_final_cost_drops_with_guidance():
    """A toy flow x' = (x - a_data)/t ... i.e. v_t = noise - a (openpi: x_t = t*noise + (1-t)*a), integrated with the
    hook of the real code path (`guidance_update` on x_hat0 = x_t - t*v_t, applied to x_next)."""
    rng = np.random.default_rng(3)
    T, D, N = 5, 32, 10
    a_data = np.zeros((T, D))
    a_data[:, :16] = 0.5
    noise = rng.standard_normal((T, D))
    scale = np.full(16, 0.8)
    aff = _affine(scale)
    goal = np.zeros((T, 16))
    orc = _QuadOracle(goal)

    def run(lam):
        x, t, dt, costs = noise.copy(), 1.0, -1.0 / N, []
        for _ in range(N):
            v = noise - a_data                          # exact conditional velocity of the straight path
            x_next = x + dt * v
            if lam > 0:
                dx, c, _ = FG.guidance_update(x - t * v, orc, aff, scale=lam, max_step_rad=10.0)
                x_next = x_next + dx
                costs.append(c)
            x, t = x_next, t + dt
        return x, orc.cost_grad(aff.to_physical(x))[0]

    x_off, c_off = run(0.0)
    x_on, c_on = run(0.4)
    assert np.allclose(x_off[:, :16], a_data[:, :16], atol=1e-9)             # unguided: the straight flow lands on the data
    assert c_on < 0.5 * c_off


# ------------------------------------------------------------------------------------------------ 5. guide_reference
def test_guided_chunk_with_lower_violation_is_used_and_only_joint_columns_change(scenario):
    sc = scenario
    junk = {}

    def fn(oracle, original):
        out = _descend(oracle, original)
        out[:, list(wire.GRIPPER_COLUMNS)] = 9.9                               # a resampler that tries to touch the grippers
        out[:, sc.layout.n_valid:] = -7.0
        junk["x"] = out
        return out

    r = _refiner(sc)
    rs = _Resample(fn)
    out = r.refine(sc.chunk, {"fm_resample": rs})
    rec = r.last_result.metrics["fm_guide"]
    assert rs.calls == 1
    assert rec["applied"] is True and rec["used"] is True
    assert rec["violation_after_m"] < rec["violation_before_m"]
    assert rec["violation_before_m"] == pytest.approx(sc.reference_violation, rel=1e-9)
    assert rec["max_joint_change_rad"] > 0.0
    assert set(rec) >= {"applied", "violation_before_m", "violation_after_m", "used", "max_joint_change_rad", "guide_ms", "steps"}
    cols = list(wire.GRIPPER_COLUMNS)
    assert np.array_equal(out[:, cols], sc.chunk[:, cols])                    # grippers: the original chunk's
    padding = list(range(sc.layout.n_valid, sc.layout.action_dim))
    assert np.array_equal(out[:, padding], sc.chunk[:, padding])
    # TrajOpt started from the guided reference: its reference violation is the guided one
    assert r.last_result.reference_violation == pytest.approx(rec["violation_after_m"], rel=1e-9, abs=1e-12)


def test_guided_chunk_with_higher_violation_is_not_used(scenario):
    sc = scenario

    def worse(oracle, original):
        x = np.array(original, np.float64)
        _, g = oracle.cost_grad(x)
        return x + 3.0 * g                                                      # one step up the cost: deeper into the obstacle

    base = _refiner(sc)
    out_base = base.refine(sc.chunk)
    r = _refiner(sc)
    out = r.refine(sc.chunk, {"fm_resample": _Resample(worse)})
    rec = r.last_result.metrics["fm_guide"]
    assert rec["applied"] is True and rec["used"] is False
    assert rec["violation_after_m"] >= rec["violation_before_m"]
    assert np.array_equal(out, out_base)                                         # the original reference: same TrajOpt answer


def test_failing_resampler_keeps_the_chunk_and_says_so(scenario):
    sc = scenario

    def boom(oracle, original):
        raise RuntimeError("cuda out of memory")

    out_base = _refiner(sc).refine(sc.chunk)
    r = _refiner(sc)
    out = r.refine(sc.chunk, {"fm_resample": _Resample(boom)})
    rec = r.last_result.metrics["fm_guide"]
    assert rec["used"] is False and "cuda out of memory" in rec["error"]
    assert np.array_equal(out, out_base)


def test_oracle_restores_the_linearizer_records(scenario):
    sc = scenario
    r = _refiner(sc)
    lin = r.optimizer.linearizer
    from AG3S.trajopt.linearize import _CLEARANCE_SIDE_ATTRS, _MISSING

    before = {a: lin.__dict__.get(a, _MISSING) for a in _CLEARANCE_SIDE_ATTRS}
    traj = sc.layout.chunk_to_trajectory(sc.chunk)
    with FG.CollisionOracle(r.optimizer, sc.layout, sc.horizon, sc.q_now, sc.scene) as orc:
        v = orc.violation(sc.chunk)
        c, g = orc.cost_grad(sc.chunk)
    after = {a: lin.__dict__.get(a, _MISSING) for a in _CLEARANCE_SIDE_ATTRS}
    assert all(after[a] is before[a] or (isinstance(before[a], np.ndarray) and np.array_equal(after[a], before[a]))
               or after[a] == before[a] for a in before)
    assert v == pytest.approx(sc.reference_violation, rel=1e-9)
    assert c > 0.0 and g.shape == sc.chunk.shape
    assert np.all(g[:, list(wire.GRIPPER_COLUMNS)] == 0.0)                     # no gradient on grippers
    assert np.all(g[:, sc.layout.n_valid:] == 0.0)                             # nor on padding
    assert traj.shape[0] == sc.layout.nq_opt


def test_cost_gradient_matches_finite_difference_where_rows_are_stable(scenario):
    """dC/dq of the oracle is the Jacobian^T (-clearance)+ of the rows the solve uses: a small step down the gradient lowers
    both C and the violation (first-order check on the real RB-Y1 scene)."""
    sc = scenario
    r = _refiner(sc)
    with FG.CollisionOracle(r.optimizer, sc.layout, sc.horizon, sc.q_now, sc.scene) as orc:
        c0, g = orc.cost_grad(sc.chunk)
        v0 = orc.violation(sc.chunk)
        step = 1e-3 / max(float(np.abs(g).max()), 1e-12)
        c1, _ = orc.cost_grad(sc.chunk - step * g)
        v1 = orc.violation(sc.chunk - step * g)
    assert c1 < c0
    assert v1 < v0
    # directional derivative: (C(x - eps g) - C(x)) / eps ~= -|g|^2
    eps = step * 1e-2
    with FG.CollisionOracle(r.optimizer, sc.layout, sc.horizon, sc.q_now, sc.scene) as orc:
        cm, _ = orc.cost_grad(sc.chunk - eps * g)
    assert (cm - c0) / eps == pytest.approx(-float(np.sum(g * g)), rel=0.05)


# ------------------------------------------------------------------------------------------------ 6. serve_safe
def test_flags_parse_and_defaults():
    a = _check(["--fm-guide"])
    cfg = serve_safe.fm_guide_config_from_args(a)
    assert (cfg.scale, cfg.steps, cfg.max_step_rad) == (FG.DEFAULT_SCALE, FG.DEFAULT_STEPS, FG.DEFAULT_MAX_STEP_RAD)
    a = _check(["--fm-guide", "--fm-guide-scale", "3", "--fm-guide-steps", "4", "--fm-guide-max-step-rad", "0.1"])
    cfg = serve_safe.fm_guide_config_from_args(a)
    assert (cfg.scale, cfg.steps, cfg.max_step_rad) == (3.0, 4, 0.1)


@pytest.mark.parametrize("argv", [
    ["--fm-guide-scale", "2"], ["--fm-guide-steps", "3"], ["--fm-guide-max-step-rad", "0.1"],       # tuning without the flag
    ["--fm-guide", "--no-safe"], ["--fm-guide", "--no-perception"],
    ["--fm-guide", "--fm-guide-steps", "0"], ["--fm-guide", "--fm-guide-scale", "-1"],
    ["--fm-guide", "--fm-guide-max-step-rad", "0"],
])
def test_flag_refusals(argv):
    with pytest.raises(SystemExit):
        _check(argv)


def test_p0_server_argv_plus_flag_is_accepted():
    import pathlib
    import shlex

    root = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
    path = root / "results/development/20261103T002419+0900_8ab3c7ef7757_82e2790bab11/verify/T88/kit/server_flags_p0.txt"
    if not path.exists():
        pytest.skip("P0 argv file not on this machine")
    a = _check(shlex.split(path.read_text()) + ["--fm-guide"])
    assert a.fm_guide is True


def test_config_validation():
    with pytest.raises(ValueError):
        FG.FmGuideConfig(scale=-1.0)
    with pytest.raises(ValueError):
        FG.FmGuideConfig(steps=0)
    with pytest.raises(ValueError):
        FG.FmGuideConfig(max_step_rad=0.0)
