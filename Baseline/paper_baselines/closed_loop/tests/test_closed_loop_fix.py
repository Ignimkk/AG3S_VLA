"""CLOSED-LOOP-FIX (Implementer K): action range of Lan-o3dp, one guidance compile per server, OmniGuide target
memory / freeze zone, latency keys of the method server.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD:$PWD/third_party/openpi/src \\
        .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_closed_loop_fix.py

Synthetic scene of ``tests/test_methods_guidance.py`` (EE at (0.45, 0, 0.30), straight +x chunk, 3 cm sphere on the
path); the frozen network is the surrogate straight flow (``core.surrogate_sample``), ``constant`` = every push kept
to the end (the worst case for accumulation).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from Baseline.paper_baselines.methods.guidance_common.core import (  # noqa: E402
    chunk_to_model_space, model_space_to_chunk, surrogate_sample)
from Baseline.paper_baselines.methods.guidance_common.geometry import FAR  # noqa: E402
from Baseline.paper_baselines.methods.guidance_common.scene import build_scene  # noqa: E402
from Baseline.paper_baselines.methods.interface import SceneObservation, get_method  # noqa: E402

NORM = {"q01": np.array([-0.7474, -0.7961, -0.9375, -0.1158, -0.1694, -0.1945, -1.0], np.float32),
        "q99": np.array([0.9371, 0.8595, 0.9371, 0.1402, 0.181, 0.3115, 0.9996], np.float32)}
H = 10
EE = np.array([0.45, 0.0, 0.30])
OBST_C = np.array([0.52, 0.02, 0.30])
OBST_R = 0.03
BOX_TOL = 1e-5  # raw units: float32 round-off of the last step


def _ee(p=EE):
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = p
    return T


def _sphere_points(c, r, n=400, seed=1):
    v = np.random.default_rng(seed).standard_normal((n, 3))
    return np.asarray(c) + r * v / np.linalg.norm(v, axis=1, keepdims=True)


def _obs(points=None, prims=None, target=None, ee=EE, opening=0.04):
    ex = {"finger_opening_m": opening}
    if points is not None:
        ex["obstacle_points"] = np.asarray(points, float).reshape(-1, 3)
    if prims is not None:
        ex["privileged/obstacle_geoms"] = prims
    if target is not None:
        ex["target_points"] = np.asarray(target, float).reshape(-1, 3)
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=_ee(ee),
                            gripper=0.04, instruction="test", extras=ex)


def _scene(obs, padded=True):
    """What ``serve_method._guidance_call`` hands the method: obstacle cloud padded to 1024 points."""
    sc = build_scene(obs, obstacle_source="points", voxel=0.02, max_points=1024, workspace_radius=0.6,
                     remove_table=False)
    if padded:
        sc.obstacles = sc.obstacles.padded(1024, 4)
    return sc


def _chunk(dx=1.0):
    raw = np.zeros((H, 7))
    raw[:, 0] = dx
    raw[:, 6] = -1.0
    return raw


def _noise():
    return np.random.default_rng(0).standard_normal((1, H, 32)).astype(np.float32)


def _sample(method, obs, kind, scene=None, raw=None):
    x = chunk_to_model_space(_chunk() if raw is None else raw, NORM)
    noise = _noise()
    fn = method.make_guidance_fn(obs, NORM, noise=noise, scene=scene)
    base = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, kind=kind))())
    out = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn, kind=kind))())
    return base, out


# --- 1. Lan-o3dp: action range ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["tracking", "constant"])
@pytest.mark.parametrize("src", ["points", "prim"])
def test_lan_guided_chunk_stays_in_the_libero_action_box(kind, src):
    obs = (_obs(points=_sphere_points(OBST_C, OBST_R)) if src == "points"
           else _obs(prims=[{"type": "sphere", "pos": OBST_C, "size": [OBST_R]}]))
    m = get_method("lan_o3dp_guidance_adapted")
    base, out = _sample(m, obs, kind)
    raw = model_space_to_chunk(out, NORM)
    assert np.abs(raw[:, :3]).max() <= 1.0 + BOX_TOL, np.abs(raw[:, :3]).max()
    assert np.abs(out - base).max() > 1e-3  # the guidance still acts
    # without the clip_sample port the same scene leaves the box (the V3 |a| = 27 mechanism)
    m0 = get_method("lan_o3dp_guidance_adapted", clip_sample=False)
    _, out0 = _sample(m0, obs, kind)
    assert np.abs(model_space_to_chunk(out0, NORM)[:, :3]).max() > 1.5


def test_lan_obstacle_off_the_path_is_bit_identical():
    """Circles exist (no skip) but no waypoint is inside one: zero push -> x_next unchanged, bit for bit."""
    obs = _obs(points=_sphere_points([0.10, 0.45, 0.30], OBST_R))
    m = get_method("lan_o3dp_guidance_adapted")
    base, out = _sample(m, obs, "constant", scene=_scene(obs))
    assert not m.last_ctx["skip"] and m.last_ctx["n_circles"] == 1
    assert base.tobytes() == out.tobytes()


def test_lan_no_obstacle_is_identity():
    obs = _obs()
    m = get_method("lan_o3dp_guidance_adapted")
    base, out = _sample(m, obs, "constant", scene=_scene(obs))
    assert m.last_ctx["skip"] and base.tobytes() == out.tobytes()


def test_lan_circles_ignore_the_far_padding():
    from Baseline.paper_baselines.methods.lan_o3dp_guidance_adapted.guidance import obstacle_circles

    obs = _obs(points=_sphere_points(OBST_C, OBST_R))
    cs_p, rs_p = obstacle_circles(_scene(obs, padded=True))
    cs_u, rs_u = obstacle_circles(_scene(obs, padded=False))
    assert len(cs_p) == 1 and np.array_equal(cs_p, cs_u) and np.array_equal(rs_p, rs_u)
    assert np.all(np.abs(cs_p) < 0.5 * FAR)


# --- 2. one guidance compile per server --------------------------------------------------------------------------------
def _calls():
    """Consecutive calls whose scenes differ the way closed-loop calls do (cloud size, circle count, target on / off,
    points inside the OmniGuide EE clearance, gripper closed)."""
    a = _sphere_points(OBST_C, OBST_R, 300)
    b = _sphere_points([0.40, 0.15, 0.25], 0.04, 500, seed=2)
    near = _sphere_points(EE + [0.0, 0.0, -0.03], 0.01, 60, seed=3)  # inside the 5 / 10 cm EE clearance
    tgt = _sphere_points([0.60, -0.10, 0.20], 0.02, 50, seed=4)
    return [_obs(points=a, target=tgt), _obs(points=np.concatenate([a, b]), target=None),
            _obs(points=np.concatenate([a, near]), target=tgt, opening=0.01),
            _obs(points=np.concatenate([a, b, near]), target=tgt, ee=EE + [0.01, -0.02, 0.0])]


@pytest.mark.parametrize("name", ["lan_o3dp_guidance_adapted", "omniguide"])
def test_guidance_compiles_once_over_consecutive_calls(name):
    from Baseline.paper_baselines.closed_loop.guided import GuidanceSlot, hoist

    kw = {"freeze_zone": False} if name == "omniguide" else {}
    m = get_method(name, **kw)
    slot = GuidanceSlot()
    keys = []
    for obs in _calls():
        fn = m.make_guidance_fn(obs, NORM, noise=_noise(), scene=_scene(obs))
        h = hoist(fn, (1, H, 32), np.float32)
        assert not h.is_identity
        slot.set(h)
        slot.host(chunk_to_model_space(_chunk(), NORM), np.float32(0.5))  # runs the compiled guidance once
        keys.append(h.key)
    assert len(set(keys)) == 1 and slot.stats == {"compiles": 1, "hits": len(keys) - 1}, (keys, slot.stats)


# --- 3. OmniGuide: target memory and freeze zone ---------------------------------------------------------------------------
def test_omniguide_target_is_cached_per_episode():
    t0 = _sphere_points([0.60, -0.10, 0.20], 0.02, 50)
    t1 = t0 + [0.0, 0.2, 0.0]
    m = get_method("omniguide", freeze_zone=False)
    m.make_guidance_fn(_obs(points=_sphere_points(OBST_C, OBST_R), target=t0), NORM, noise=_noise())
    assert m.last_ctx["target_source"] == "cached_now"
    m.make_guidance_fn(_obs(points=_sphere_points(OBST_C, OBST_R), target=t1), NORM, noise=_noise())
    assert m.last_ctx["target_source"] == "cached"
    np.testing.assert_allclose(m.last_ctx["centroid"], t0.mean(0))
    m.reset()
    m.make_guidance_fn(_obs(points=_sphere_points(OBST_C, OBST_R), target=t1), NORM, noise=_noise())
    np.testing.assert_allclose(m.last_ctx["centroid"], t1.mean(0))
    # target_cache="call": the previous per-call behaviour
    mc = get_method("omniguide", freeze_zone=False, target_cache="call")
    for t in (t0, t1):
        mc.make_guidance_fn(_obs(points=_sphere_points(OBST_C, OBST_R), target=t), NORM, noise=_noise())
    np.testing.assert_allclose(mc.last_ctx["centroid"], t1.mean(0))


def test_omniguide_freeze_zone_exits_when_the_held_target_leaves_its_cached_place():
    """Grasp: EE within 0.1 m of the target -> frozen (identity). Transport: the detection moves with the gripper
    (held bowl, 5 cm below the EE); the original exit test reads the cloud cached at entry, so the freeze zone ends
    once the EE is 0.1 m away from the grasp place. A per-call target (held, 5 cm away) re-enters every other call."""
    pts = _sphere_points(OBST_C + [0.0, 0.3, 0.0], OBST_R)
    grasp = EE + [0.0, 0.0, -0.05]
    m = get_method("omniguide")
    m.make_guidance_fn(_obs(points=pts, target=grasp), NORM, noise=_noise())
    assert m.last_ctx["frozen"] and m.last_ctx["skip"]
    ee2 = EE + [0.0, 0.0, 0.20]
    m.make_guidance_fn(_obs(points=pts, target=ee2 + [0.0, 0.0, -0.05], ee=ee2), NORM, noise=_noise())
    assert not m.last_ctx["frozen"] and not m.last_ctx["skip"]
    frozen = {}
    for mode in ("episode", "call"):
        mm = get_method("omniguide", target_cache=mode)
        seq = []
        for dz in (0.0, 0.20, 0.35, 0.50):  # grasp, then transport upwards with the bowl held 5 cm below the EE
            ee = EE + [0.0, 0.0, dz]
            mm.make_guidance_fn(_obs(points=pts, target=ee + [0.0, 0.0, -0.05], ee=ee), NORM, noise=_noise())
            seq.append(bool(mm.last_ctx["frozen"]))
        frozen[mode] = seq
    assert frozen["episode"] == [True, False, False, False]  # original: the cached grasp place is left behind
    assert frozen["call"] == [True, False, True, False]  # per-call target re-enters the zone on every other call


def test_omniguide_clearance_points_moved_to_far_not_dropped():
    pts = np.concatenate([_sphere_points(OBST_C, OBST_R, 100), _sphere_points(EE, 0.01, 20, seed=5)])
    obs = _obs(points=pts)
    sc = _scene(obs)
    m = get_method("omniguide", freeze_zone=False)
    m.make_guidance_fn(obs, NORM, noise=_noise(), scene=sc)
    assert m.last_ctx["n_clearance_dropped"] > 0
    assert m.last_ctx["obstacles"]["points"].shape == (1024, 3)


# --- 4. method server latency keys ------------------------------------------------------------------------------------------
def test_latency_keys_are_consistent():
    from Baseline.paper_baselines.closed_loop import serve_method as S

    assert S.LATENCY_KEYS == ("total", "diagnostics", "method", "serve")
    assert not S.NO_PADDING
