"""BASELINE-BENCH-READY L: constrained flow matching keeps the corrected chunk inside the LIBERO action box.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD:$PWD/third_party/openpi/src .venv-openpi-live/bin/python \\
        -m pytest -q Baseline/paper_baselines/methods/constrained_flow_matching/tests

Scenes (base frame, gripper down at EE (0.45, 0, 0.30), straight +x chunk): an obstacle cloud 5 cm beside the EE
(inside the 6 / 12 / 11 cm EE ellipsoid: ``h_0 < 0``, the V5 situation), a 2 cm sphere 14 cm ahead (mild
violation late in the chunk), and nothing. The network is the surrogate flow (``tracking`` / ``constant``).
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
from Baseline.paper_baselines.methods.guidance_common.scene import build_scene  # noqa: E402
from Baseline.paper_baselines.methods.interface import SceneObservation, get_method  # noqa: E402

NORM = {"q01": np.array([-0.7474, -0.7961, -0.9375, -0.1158, -0.1694, -0.1945, -1.0], np.float32),
        "q99": np.array([0.9371, 0.8595, 0.9371, 0.1402, 0.181, 0.3115, 0.9996], np.float32)}
H = 10
EE = np.array([0.45, 0.0, 0.30])
BOX_TOL = 1e-5


def _ee():
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = EE
    return T


def _sphere_points(c, r, n=400, seed=1):
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return np.asarray(c) + r * v / np.linalg.norm(v, axis=1, keepdims=True)


def _obs(points=None):
    ex = {"finger_opening_m": 0.04}
    if points is not None:
        ex["obstacle_points"] = np.asarray(points, float).reshape(-1, 3)
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=_ee(),
                            gripper=0.04, instruction="test", extras=ex)


def _scene(obs):
    sc = build_scene(obs, obstacle_source="points", voxel=0.02, max_points=1024, workspace_radius=0.6,
                     remove_table=False)
    sc.obstacles = sc.obstacles.padded(1024, 4)  # as serve_method._guidance_call
    return sc


def _chunk(dx=0.6):
    raw = np.zeros((H, 7))
    raw[:, 0] = dx
    raw[:, 6] = -1.0
    return raw


def _sample(method, obs, kind):
    x = chunk_to_model_space(_chunk(), NORM)
    noise = np.random.default_rng(0).standard_normal((1, H, 32)).astype(np.float32)
    fn = method.make_guidance_fn(obs, NORM, noise=noise, scene=_scene(obs))
    base = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, kind=kind))())
    out = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn, kind=kind))())
    return base, out


OVERLAP = _sphere_points(EE + [0.0, 0.05, -0.02], 0.02)  # inside the EE ellipsoid at the measured pose
AHEAD = _sphere_points(EE + [0.14, 0.05, 0.0], 0.02, n=200)  # mild violation late in the chunk


@pytest.mark.parametrize("kind", ["tracking", "constant"])
def test_overlap_at_start_blows_up_without_the_box_and_stays_inside_with_it(kind):
    obs = _obs(OVERLAP)
    m0 = get_method("constrained_flow_matching", action_box=False)  # = V5
    base, out0 = _sample(m0, obs, kind)
    # the mechanism: h_0 < 0 at the measured EE, recovery asked within a row of dt = 0.01 per normalized unit
    x = jnp.asarray(chunk_to_model_space(_chunk(), NORM)[0])
    path = m0.path_for(H, m0.last_scene)
    g = np.asarray(m0.constraints(x, jnp.zeros((H, 3)), m0.last_ctx, path))
    assert g.min() < -0.005
    raw0 = model_space_to_chunk(out0, NORM)
    assert np.abs(raw0[:, :3]).max() > 1.5, np.abs(raw0[:, :3]).max()
    m1 = get_method("constrained_flow_matching")  # action_box=True (default)
    assert m1.params["action_box"] is True
    _, out1 = _sample(m1, obs, kind)
    raw1 = model_space_to_chunk(out1, NORM)
    assert np.abs(raw1[:, :3]).max() <= 1.0 + BOX_TOL, np.abs(raw1[:, :3]).max()
    assert np.abs(out1 - base).max() > 1e-2  # still guided
    assert np.array_equal(out1[..., 3:], out0[..., 3:])  # only the position dims are corrected


@pytest.mark.parametrize("kind", ["tracking", "constant"])
def test_correction_inside_the_box_is_bit_identical_to_v5(kind):
    obs = _obs(AHEAD)
    base, out0 = _sample(get_method("constrained_flow_matching", action_box=False), obs, kind)
    _, out1 = _sample(get_method("constrained_flow_matching"), obs, kind)
    raw0 = model_space_to_chunk(out0, NORM)
    assert np.abs(out0 - base).max() > 1e-3 and np.abs(raw0[:, :3]).max() < 1.0
    assert out1.tobytes() == out0.tobytes()


def test_nothing_to_avoid_is_bit_identical():
    for m in (get_method("constrained_flow_matching"), get_method("constrained_flow_matching", action_box=False)):
        base, out = _sample(m, _obs(_sphere_points(EE + [0.0, 0.5, 0.2], 0.02)), "constant")
        assert out.tobytes() == base.tobytes()


def test_network_outside_the_box_is_not_pushed_further_out_during_denoising():
    """Intermediate step: a dimension the (noisy) state already has outside the box is not pushed outward;
    last step: every corrected entry is inside the box."""
    m = get_method("constrained_flow_matching")
    obs = _obs(OVERLAP)
    m.make_guidance_fn(obs, NORM, scene=_scene(obs))
    lo, hi = m.last_ctx["box_lo"], m.last_ctx["box_hi"]
    x = np.asarray(chunk_to_model_space(_chunk(), NORM)[0], np.float32)
    x[:, 1] = hi[1] + 0.5  # every row beyond +y
    path = m.path_for(H, m.last_scene)
    y = np.asarray(m._guide_one(jnp.asarray(x), jnp.asarray(x), jnp.float32(0.5), m.last_ctx, path))
    assert np.all(y[:, 1] <= x[:, 1] + 1e-6)
    assert np.all(y[:, :3] >= np.minimum(lo, x[:, :3]) - 1e-6) and np.all(y[:, :3] <= np.maximum(hi, x[:, :3]) + 1e-6)
    assert np.abs(y - x).max() > 1e-3
    z = np.asarray(m._guide_one(jnp.asarray(x), jnp.asarray(x), jnp.float32(0.0), m.last_ctx, path))
    moved = z[:, :3] != x[:, :3]
    assert moved.any()
    assert np.all((z[:, :3] >= lo - 1e-6)[moved]) and np.all((z[:, :3] <= hi + 1e-6)[moved])
    assert np.array_equal(z[:, 3:], x[:, 3:])


def test_provenance_records_the_box():
    m = get_method("constrained_flow_matching")
    assert m.provenance()["params"]["action_box"] is True
    assert "action_box" in get_method("constrained_flow_matching", action_box=False).provenance()["params"]
