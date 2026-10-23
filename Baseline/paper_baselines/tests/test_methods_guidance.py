"""Unit tests of the sampling-guidance methods (PAPER-METHODS, Implementer D).

Run (CPU only; JAX from the openpi venv, nothing installed)::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \\
        .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/tests/test_methods_guidance.py

Synthetic scene: EE at (0.45, 0, 0.30), straight +x chunk (raw dx = 1.0 for 10 rows, ~11 cm under the OSC response
model), one obstacle sphere r = 3 cm centred at (0.52, 0.02, 0.30) -- slightly to +y of the path so a repeller has a
lateral direction. The frozen network is replaced by the surrogate straight-flow velocity (``core.surrogate_sample``,
the same Euler loop as ``pi0.sample_actions``).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from Baseline.paper_baselines.methods.guidance_common.core import (  # noqa: E402
    chunk_to_model_space, model_space_to_chunk, surrogate_sample)
from Baseline.paper_baselines.methods.guidance_common.geometry import Obstacles, signed_distance_np  # noqa: E402
from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel, load_response  # noqa: E402
from Baseline.paper_baselines.methods.interface import SamplingGuidance, SceneObservation, get_method  # noqa: E402

GUIDANCE = ("omniguide", "constrained_flow_matching", "lan_o3dp_guidance_adapted", "nudge")
# pi05_libero quantile stats (assets/physical-intelligence/libero/norm_stats.json, first 7 dims)
NORM = {"q01": np.array([-0.7474, -0.7961, -0.9375, -0.1158, -0.1694, -0.1945, -1.0], np.float32),
        "q99": np.array([0.9371, 0.8595, 0.9371, 0.1402, 0.181, 0.3115, 0.9996], np.float32)}
H = 10
OBST_C = np.array([0.52, 0.02, 0.30])
OBST_R = 0.03


def _ee():
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # gripper pointing down
    T[:3, 3] = [0.45, 0.0, 0.30]
    return T


def _obs(kind: str = "prim", target=None, ee=None, **extra) -> SceneObservation:
    ex = dict(extra)
    if kind == "prim":
        ex["privileged/obstacle_geoms"] = [{"type": "sphere", "pos": OBST_C, "size": [OBST_R]}]
    elif kind == "points":
        v = np.random.default_rng(1).standard_normal((400, 3))
        ex["obstacle_points"] = OBST_C + OBST_R * v / np.linalg.norm(v, axis=1, keepdims=True)
    if target is not None:
        ex["target_points"] = np.asarray(target, float).reshape(-1, 3)
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7),
                            ee_pose=_ee() if ee is None else ee, gripper=0.04, instruction="test", extras=ex)


def _chunk():
    raw = np.zeros((H, 7))
    raw[:, 0] = 1.0
    raw[:, 6] = -1.0
    return raw


def _noise(b=1):
    return np.random.default_rng(0).standard_normal((b, H, 32)).astype(np.float32)


def _path(raw):
    return EEPathModel(H=H, p0=_ee()[:3, 3]).numpy(np.clip(raw[:, :3], -1, 1))


def _clearance(p):
    return float(np.min(np.linalg.norm(p - OBST_C[None], axis=1)) - OBST_R)


def _run(name, obs, b=1, **params):
    m = get_method(name, **params)
    x = np.repeat(chunk_to_model_space(_chunk(), NORM), b, axis=0)
    noise = _noise(b)
    fn = m.make_guidance_fn(obs, NORM, noise=noise)
    base = jax.jit(lambda: surrogate_sample(x, noise))()
    out = jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn))()
    return m, np.asarray(base), np.asarray(out)


# --- registry / interface ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", GUIDANCE)
def test_registry_and_protocol(name):
    m = get_method(name)
    assert isinstance(m, SamplingGuidance)
    assert m.name == name
    prov = m.provenance()
    assert prov["family"] == "sampling_guidance" and prov["source"] is not None


# --- identity ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", GUIDANCE)
def test_disabled_is_bit_identical(name):
    """enabled=False -> the hook returns its input; the sampled chunk equals guidance_fn=None bit for bit."""
    m = get_method(name, enabled=False)
    fn = m.make_guidance_fn(_obs(), NORM, noise=_noise())
    x = jnp.asarray(_noise() * 0.3)
    assert fn(x, jnp.float32(0.5)) is x
    xt = chunk_to_model_space(_chunk(), NORM)
    a = np.asarray(jax.jit(lambda: surrogate_sample(xt, _noise()))())
    b = np.asarray(jax.jit(lambda: surrogate_sample(xt, _noise(), guidance_fn=fn))())
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name", GUIDANCE)
def test_no_obstacle_far_target_is_identity(name):
    """Nothing to avoid (and OmniGuide attraction off) -> bit-identical output."""
    params = {"guidance_types": ("collision",)} if name == "omniguide" else {}
    _, base, out = _run(name, _obs(kind="none"), **params)
    assert base.tobytes() == out.tobytes()


def test_surrogate_reproduces_target():
    xt = chunk_to_model_space(_chunk(), NORM)
    out = np.asarray(surrogate_sample(xt, _noise()))
    np.testing.assert_allclose(out, xt, atol=1e-5)


# --- repeller -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["prim", "points"])
@pytest.mark.parametrize("name", GUIDANCE)
def test_repeller_increases_clearance(name, kind):
    if name == "omniguide" and kind == "points":
        pytest.skip("OmniGuide drops cloud points within 5 cm of the EE (EE-clearance filter); covered with prims")
    params = {"guidance_types": ("collision",)} if name == "omniguide" else {}
    _, base, out = _run(name, _obs(kind), **params)
    rb, rg = model_space_to_chunk(base, NORM), model_space_to_chunk(out, NORM)
    cb, cg = _clearance(_path(rb)), _clearance(_path(rg))
    assert cb < 0, "baseline path must cut the obstacle"
    assert cg > cb + 0.005, f"{name}/{kind}: clearance {cb:.4f} -> {cg:.4f}"


@pytest.mark.parametrize("name", ["omniguide", "constrained_flow_matching", "nudge"])
def test_repeller_pushes_waypoints_away(name):
    """The waypoint nearest the obstacle moves away from the obstacle centre (lateral y < 0 since the obstacle is +y)."""
    params = {"guidance_types": ("collision",)} if name == "omniguide" else {}
    _, base, out = _run(name, _obs("prim"), **params)
    pb = _path(model_space_to_chunk(base, NORM))
    pg = _path(model_space_to_chunk(out, NORM))
    k = int(np.argmin(np.linalg.norm(pb - OBST_C, axis=1)))
    assert np.linalg.norm(pg[k] - OBST_C) > np.linalg.norm(pb[k] - OBST_C) + 0.003
    assert pg[-1, 1] < pb[-1, 1] - 0.005


@pytest.mark.parametrize("name", GUIDANCE)
def test_batch_rows_independent_and_equal(name):
    params = {"guidance_types": ("collision",)} if name == "omniguide" else {}
    _, base, out = _run(name, _obs("prim"), b=2, **params)
    assert out.shape == (2, H, 32) and np.all(np.isfinite(out))
    assert not np.allclose(out, base)


# --- method specifics -----------------------------------------------------------------------------------------------
def test_omniguide_attractor_moves_toward_target():
    """Direction test. With the original factor 0.4 the pull is ~0.3 mm here: LIBERO's action -> displacement gain
    (~0.0125 m per unit at steady state) is ~10x below DROID joint velocity, the setting the factor was tuned for
    ([9] reports OmniGuide ~= unguided on LIBERO-Object). factor 20 makes the direction measurable."""
    tgt = np.array([[0.50, -0.10, 0.25]])
    _, base, out = _run("omniguide", _obs("none", target=tgt), guidance_types=("attract",), freeze_zone=False, factor=20.0)
    pb, pg = _path(model_space_to_chunk(base, NORM)), _path(model_space_to_chunk(out, NORM))
    assert np.linalg.norm(pg[-1] - tgt[0]) < np.linalg.norm(pb[-1] - tgt[0]) - 0.01


def test_omniguide_freeze_zone_is_identity_near_target():
    tgt = _ee()[:3, 3][None] + np.array([[0.05, 0.0, 0.0]])  # 5 cm < 0.1 m freeze threshold
    m, base, out = _run("omniguide", _obs("prim", target=tgt))
    assert m.last_ctx["frozen"] and base.tobytes() == out.tobytes()


def test_cfm_correction_satisfies_cbf_constraints():
    m = get_method("constrained_flow_matching")
    fn = m.make_guidance_fn(_obs("prim"), NORM)  # x0_mode identity (original)
    del fn
    x = jnp.asarray(chunk_to_model_space(_chunk(), NORM)[0])
    path = m.path_for(H, m.last_scene)
    g0 = np.asarray(m.constraints(x, jnp.zeros((H, 3)), m.last_ctx, path))
    delta, violated = m.correction(x, m.last_ctx, path)
    g1 = np.asarray(m.constraints(x, delta, m.last_ctx, path))
    assert bool(violated) and g0.min() < 0
    assert g1.min() > -1e-3, g1
    # minimum norm: a random feasible-looking bigger step is not smaller
    assert np.linalg.norm(np.asarray(delta)) < 10.0


def test_cfm_no_violation_no_correction():
    far = _ee()
    far[:3, 3] = [0.10, -0.40, 0.60]
    m = get_method("constrained_flow_matching")
    m.make_guidance_fn(_obs("prim", ee=far), NORM)
    x = jnp.asarray(chunk_to_model_space(_chunk(), NORM)[0])
    delta, violated = m.correction(x, m.last_ctx, m.path_for(H, m.last_scene))
    assert not bool(violated) and float(jnp.abs(delta).max()) == 0.0


def test_nudge_update_rule_matches_paper_flow_form():
    """x_next' - x_next == (dt/t_k) (A0_til - A0_hat) with A0_til = clip(A0_hat - w g), w = rho/(kappa(1-rho))."""
    m = get_method("nudge", kappa=2.0, clip_raw=False)
    m.make_guidance_fn(_obs("prim"), NORM)
    ctx = m.last_ctx
    path = m.path_for(H, m.last_scene)
    x0 = jnp.asarray(chunk_to_model_space(_chunk(), NORM)[0])
    xn = x0 + 0.1
    t_next = jnp.float32(0.3)
    out = m._guide_one(xn, x0, t_next, ctx, path)
    g = m.gradient(x0, ctx, path)
    w = 0.85 / (2.0 * 0.15)
    expect = xn + (0.1 / 0.4) * (-w * g)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expect), rtol=1e-5, atol=1e-6)
    assert float(jnp.abs(g).max()) > 0


def test_lan_circles_from_point_cloud():
    from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
    from Baseline.paper_baselines.methods.lan_o3dp_guidance_adapted.guidance import obstacle_circles

    sc = build_scene(_obs("points"), voxel=0.005)
    cs, rs = obstacle_circles(sc)
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0], OBST_C[:2], atol=0.005)
    assert 0.03 < rs[0] < 0.05  # half xy-bbox diagonal of a 3 cm sphere = 4.2 cm


# --- shared geometry / kinematics -----------------------------------------------------------------------------------
def test_signed_distance_primitives():
    R = np.eye(3)
    o = Obstacles.from_primitives([
        {"type": "sphere", "pos": [0, 0, 0], "size": [0.1]},
        {"type": "box", "pos": [1, 0, 0], "rot": R, "size": [0.1, 0.2, 0.3]},
        {"type": "cylinder", "pos": [0, 1, 0], "rot": R, "size": [0.05, 0.2]},
    ])
    q = np.array([[0.3, 0, 0], [1.0, 0.5, 0], [1.0, 0.0, 0.0], [0, 1.2, 0], [0, 1, 0.5]])
    d = signed_distance_np(q, o)
    np.testing.assert_allclose(d, [0.2, 0.3, -0.1, 0.15, 0.3], atol=1e-5)


def test_path_model_matches_response_recursion():
    alpha, beta, _ = load_response()
    a = np.random.default_rng(3).uniform(-1, 1, (H, 3))
    y_prev = np.array([0.001, -0.002, 0.0005])
    R_wb = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    m = EEPathModel(H=H, p0=np.array([0.1, 0.2, 0.3]), R_bw=R_wb.T, y_prev=y_prev)
    p, y, out = np.array([0.1, 0.2, 0.3]), y_prev.copy(), []
    for k in range(H):  # AG3S/adapters/panda/action.py predict_ee_path
        y = alpha * y + beta * 0.05 * a[k]
        p = p + R_wb.T @ y
        out.append(p.copy())
    np.testing.assert_allclose(m.numpy(a), np.asarray(out), atol=1e-12)
    np.testing.assert_allclose(np.asarray(m(jnp.asarray(a, jnp.float32))), np.asarray(out), atol=1e-6)
