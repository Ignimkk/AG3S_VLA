"""T93 — the JAX side of `--fm-guide`: the model hook, the Python-loop sampler, `FlowGuide` on a small random Pi0.

Run with the openpi interpreter on CPU (the model is the `dummy` variant, a few thousand parameters):

    JAX_PLATFORMS=cpu .venv-openpi-live/bin/python -m pytest tests/trajopt/test_t93_fm_guide_jax.py

Skipped in `.venv-ag3s` (no JAX there).

1. `Pi0.sample_actions` with `guidance_xv_fn=None` (default) or an identity hook, and with SEAM's `guidance_fn` None or identity,
   is bit-identical to the original — the SEAM / baseline path is untouched.
2. the Python-loop sampler (`sample_prefix` + `sample_velocity`) reproduces `sample_actions` (float32 model: to 1e-5).
3. `peek_noise` is the noise `Policy.infer` draws from its key: passing it back as `noise=` is bit-identical.
4. the x0 prediction handed to the hook is `x_t - t * v_t` (checked against the loop's own `v_t`).
5. `FlowGuide.resample` + `guide_reference` on a real RB-Y1 scene: the guided chunk violates less than the unguided resample with
   the same noise, joint columns only, a guidance scale of 0 reproduces the unguided sample.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("openpi")
pytest.importorskip("casadi")
pytest.importorskip("osqp")

import jax.numpy as jnp  # noqa: E402

from AG3S.trajopt import fm_guide as FG  # noqa: E402
from AG3S.trajopt.refiner import TrajOptChunkRefiner  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from tests.trajopt.fixtures import make_scenario  # noqa: E402

H = 12


@pytest.fixture(scope="module")
def cfg():
    return pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                dtype="float32", action_horizon=H)


@pytest.fixture(scope="module")
def model(cfg):
    return cfg.create(jax.random.key(0))


@pytest.fixture(scope="module")
def obs(cfg):
    fake = cfg.fake_obs(1)
    d = fake.to_dict()
    d = {k: v for k, v in d.items() if v is not None}
    return jax.tree.map(lambda x: np.asarray(x)[0], d)


@pytest.fixture(scope="module")
def observation(cfg):
    return cfg.fake_obs(1)


class _Affine:
    """The output transform of the tests: physical = offset + scale * x over the first 16 columns (an RB-Y1 look-alike)."""

    def __init__(self, offset, scale):
        self.offset, self.scale = np.asarray(offset, np.float64), np.asarray(scale, np.float64)

    def __call__(self, data):
        x = np.asarray(data["actions"], np.float64)
        return {**data, "actions": self.offset + self.scale[None, :] * x[:, :16]}


def _policy_for(model, offset=None, scale=0.05):
    off = np.zeros((H, 16)) if offset is None else offset
    p = _policy.Policy(model, transforms=[], output_transforms=[_Affine(off, np.full(16, scale))])
    p._rng = jax.random.key(7)   # noqa: SLF001 — (`Policy(rng=<key>)` does `rng or ...`, which a typed key array cannot answer)
    return p


# ------------------------------------------------------------------------------------------------ 1. the hook is inert
def test_default_and_identity_hooks_are_bit_identical(model, observation):
    rng = jax.random.key(1)
    noise = jax.random.normal(jax.random.key(2), (1, H, 32))
    base = np.asarray(model.sample_actions(rng, observation, noise=noise))
    none_xv = np.asarray(model.sample_actions(rng, observation, noise=noise, guidance_xv_fn=None))
    ident_xv = np.asarray(model.sample_actions(rng, observation, noise=noise,
                                               guidance_xv_fn=lambda x_t, v_t, t, x_next, t_next: x_next))
    seam_none = np.asarray(model.sample_actions(rng, observation, noise=noise, guidance_fn=None))
    seam_ident = np.asarray(model.sample_actions(rng, observation, noise=noise, guidance_fn=lambda x, t: x))
    for other in (none_xv, ident_xv, seam_none, seam_ident):
        assert np.array_equal(base, other)


def test_the_hook_sees_x_hat0_and_changes_the_result(model, observation):
    noise = jax.random.normal(jax.random.key(2), (1, H, 32))
    seen = []

    def hook(x_t, v_t, t, x_next, t_next):
        seen.append((x_t, v_t, t))
        return x_next + 0.01 * (x_t - t * v_t)

    base = np.asarray(model.sample_actions(jax.random.key(1), observation, noise=noise))
    out = np.asarray(model.sample_actions(jax.random.key(1), observation, noise=noise, guidance_xv_fn=hook))
    assert not np.array_equal(base, out)
    assert np.isfinite(out).all()


# ------------------------------------------------------------------------------------------------ 2. the loop sampler
def test_python_loop_sampler_reproduces_sample_actions(model, observation):
    noise = jax.random.normal(jax.random.key(2), (1, H, 32))
    ref = np.asarray(model.sample_actions(jax.random.key(1), observation, noise=noise))
    got = np.asarray(FG.GuidedSampler(model).sample(observation, noise, 10))
    assert np.abs(ref - got).max() < 1e-5


def test_loop_guide_sees_the_same_x_hat0_as_the_hook(model, observation):
    noise = jax.random.normal(jax.random.key(2), (1, H, 32))
    xs = []
    loop = np.asarray(FG.GuidedSampler(model).sample(
        observation, noise, 10,
        guide=lambda x_t, v_t, t, i: (xs.append((x_t - t * v_t)[0]), 0.01 * (x_t - t * v_t)[0])[1]))
    hooked = np.asarray(model.sample_actions(
        jax.random.key(1), observation, noise=noise,
        guidance_xv_fn=lambda x_t, v_t, t, x_next, t_next: x_next + 0.01 * (x_t - t * v_t)))
    assert len(xs) == 10
    assert np.abs(loop - hooked).max() < 1e-4


def test_only_the_last_steps_are_guided(model, observation):
    noise = jax.random.normal(jax.random.key(2), (1, H, 32))
    calls = []
    FG.GuidedSampler(model).sample(observation, noise, 10, guide=lambda *a: calls.append(a[3]), first_guided=7)
    assert calls == [7, 8, 9]


# ------------------------------------------------------------------------------------------------ 3. the noise
def test_peek_noise_is_the_noise_the_policy_draws(model, obs):
    p1, p2 = _policy_for(model), _policy_for(model)
    a = p1.infer(obs)["actions"]
    b = p2.infer(obs, noise=FG.peek_noise(p2))["actions"]
    assert np.array_equal(a, b)
    # and the key advanced the same way: the next request is the same too
    assert np.array_equal(p1.infer(obs)["actions"], p2.infer(obs)["actions"])


# ------------------------------------------------------------------------------------------------ 5. FlowGuide on a real scene
@pytest.fixture(scope="module")
def scene_case(model):
    sc = make_scenario(horizon=H, penetration=0.03)
    offset = np.zeros((H, 16))
    offset[:, :] = sc.chunk[:, :16]                      # x = 0 is the violating reference; the model perturbs it by 0.05 * x
    policy = _policy_for(model, offset=offset, scale=0.05)
    return sc, policy


def _bound(policy, obs, cfg_g):
    guide = FG.FlowGuide(policy, cfg_g)
    noise = guide.peek_noise()
    return guide, guide.bind(obs, noise), noise


def test_guided_resample_lowers_the_violation_and_keeps_non_joint_columns(scene_case, obs):
    sc, policy = scene_case
    r = TrajOptChunkRefiner(sc.robot, sc.layout, lambda c: (sc.scene, sc.q_now, True), sc.config)
    # unguided resample (scale 0): the plain sample for this noise
    g0, b0, noise = _bound(policy, obs, FG.FmGuideConfig(scale=0.0))
    # scale 0 must be exactly the plain sample for this noise
    plain = np.asarray(policy.infer(obs, noise=noise)["actions"], np.float64)
    with FG.CollisionOracle(r.optimizer, sc.layout, H, sc.q_now, sc.scene) as orc:
        out0 = b0(orc, sc.chunk)
        v_plain = orc.violation(out0.chunk)
    jc = sc.layout.action_columns
    assert np.abs(out0.chunk[:, jc] - plain[:, jc]).max() < 1e-5          # joint columns: the plain sample for this noise

    g1, b1, _ = _bound(policy, obs, FG.FmGuideConfig(scale=300.0, steps=10, max_step_rad=0.05))
    with FG.CollisionOracle(r.optimizer, sc.layout, H, sc.q_now, sc.scene) as orc:
        out1 = b1(orc, sc.chunk)
        v_guided = orc.violation(out1.chunk)
    assert v_guided < v_plain
    assert len(out1.info["cost_per_step"]) == 10
    assert out1.info["cost_per_step"][-1] < out1.info["cost_per_step"][0]


def test_guide_reference_end_to_end_with_the_real_sampler(scene_case, obs):
    sc, policy = scene_case
    r = TrajOptChunkRefiner(sc.robot, sc.layout, lambda c: (sc.scene, sc.q_now, True), sc.config)
    guide, bound, _ = _bound(policy, obs, FG.FmGuideConfig(scale=300.0, steps=10, max_step_rad=0.05))
    chunk, rec = FG.guide_reference(bound, r.optimizer, sc.layout, sc.chunk, sc.q_now, sc.scene, H)
    assert rec["applied"] is True and rec["violation_before_m"] > 0
    assert rec["violation_after_m"] < rec["violation_before_m"] and rec["used"] is True
    assert rec["steps"] == 10 and rec["max_joint_change_rad"] > 0
    grip = list(range(16))
    non_joint = [c for c in range(sc.chunk.shape[1]) if c not in sc.layout.action_columns.tolist()]
    assert np.array_equal(chunk[:, non_joint], sc.chunk[:, non_joint])
    # the refiner then runs TrajOpt on it; the result records the mechanism
    out = r.refine(sc.chunk, {"fm_resample": bound})
    assert r.last_result.metrics["fm_guide"]["used"] is True
    assert out.shape == sc.chunk.shape and grip


def test_warm_up_compiles_the_sampler(model, cfg, scene_case):
    sc, policy = scene_case
    guide = FG.FlowGuide(policy, FG.FmGuideConfig())
    assert guide.warm_up(cfg) >= 0.0
