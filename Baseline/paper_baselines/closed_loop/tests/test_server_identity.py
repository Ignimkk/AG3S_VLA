"""``--method none`` = the VLA-only server, guidance off = baseline, guidance on runs inside the real pi0 sampler.

Uses a **tiny pi0.5** built on CPU (openpi ``Pi0`` with the ``dummy`` Gemma variants; SigLIP So400m/14 swapped for
``mu/16`` *in this test process only*, openpi files untouched) so the real ``Policy.infer`` / ``Pi0.sample_actions``
code runs without the checkpoint or a GPU. ``.venv-openpi-live``::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD:$PWD/third_party/openpi/src \
      .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_server_identity.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

from Baseline.paper_baselines.closed_loop.replay import readonly

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "third_party/openpi/src"))
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage"))
SL5C_LI_VLA = STORAGE / "results/benchmark/20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only"


@pytest.fixture(scope="module")
def tiny():
    import openpi.models.pi0 as pi0mod
    from openpi.models import pi0_config

    orig = pi0mod._siglip.Module

    def small(**kw):
        kw["variant"] = "mu/16"
        return orig(**kw)

    pi0mod._siglip.Module = small
    try:
        cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                                   action_horizon=10, max_token_len=48, dtype="float32")
        model = cfg.create(jax.random.key(0))
    finally:
        pi0mod._siglip.Module = orig
    obs = jax.tree.map(lambda x: np.asarray(x)[0], cfg.fake_obs(1).to_dict())
    return model, obs


def make_policy(model):
    from openpi.policies import policy as _policy

    p = _policy.Policy(model)
    p._rng = jax.random.key(0)  # = a freshly started server
    return p


def requests(obs, n=3, seed=700000):
    out = []
    for k in range(n):
        r = dict(obs)
        r.update({"ag3s/seq": k, "ag3s/reset": k == 0, "ag3s/phase": "approach", "panda/robot_state": np.zeros(9)})
        if k == 0:
            r["ag3s/policy_seed"] = seed
        out.append(r)
    return out


def test_method_none_is_bit_identical_to_the_vla_only_server(tiny):
    from AG3S.adapters.panda.server import PandaVLAOnly
    from AG3S.trajopt.policy_seed import SeededPolicy

    from Baseline.paper_baselines.closed_loop import arms
    from Baseline.paper_baselines.closed_loop.serve_method import MethodServer

    model, obs = tiny
    ref = PandaVLAOnly(SeededPolicy(make_policy(model)))  # serve_safe --robot panda --no-safe (server.py:main)
    srv = MethodServer(make_policy(model), arms.resolve("none"))
    for episode_seed in (700000, 701002):  # two episodes, each re-seeded by its first request
        for req in requests(obs, 3, episode_seed):
            a = ref.infer(dict(req))
            b = srv.infer(dict(req))
            assert a["actions"].dtype == b["actions"].dtype
            assert a["actions"].tobytes() == b["actions"].tobytes()
            assert a.get("policy_seed") == b.get("policy_seed")
    assert srv.n_calls == 6 and srv.episode == 1


def _scene_obs(ee=(0.45, 0.0, 0.30)):
    from Baseline.paper_baselines.methods.interface import SceneObservation

    T = np.eye(4)
    T[:3, 3] = ee
    rng = np.random.default_rng(0)
    ball = rng.normal(size=(400, 3))
    ball = 0.03 * ball / np.linalg.norm(ball, axis=1, keepdims=True) + np.array([0.50, 0.0, 0.30])
    return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=T, gripper=0.08,
                            instruction="pick up the bowl", extras={"obstacle_points": ball, "T_world_base": np.eye(4)})


NORM = {"q01": -np.ones(7, np.float32), "q99": np.ones(7, np.float32)}


def _guided(model, method, exec_mode="callback"):
    from Baseline.paper_baselines.closed_loop.guided import GuidanceSlot, GuidedPolicy, OpenpiBackend

    inner = make_policy(model)
    backend = OpenpiBackend(inner, GuidanceSlot(), exec_mode=exec_mode)
    return inner, GuidedPolicy(inner, backend, method, NORM)


def test_guidance_off_matches_policy_infer(tiny):
    """``enabled=False`` -> identity guidance_fn -> the plain sampler with the same rng split and noise."""
    from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
    from Baseline.paper_baselines.methods.interface import get_method

    model, obs = tiny
    ref = make_policy(model)
    _inner, g = _guided(model, get_method("nudge", enabled=False))
    so = _scene_obs()
    for _ in range(3):
        a = np.asarray(ref.infer(dict(obs))["actions"])
        g.set_call(so, build_scene(so, obstacle_source="points"))
        b = np.asarray(g.infer(dict(obs))["actions"])
        assert g.last["guided"] is False
        # Policy.infer: module_jit(sample_actions)(rng, obs) draws the noise inside; GuidedPolicy draws it with the
        # same key and passes it in. Equal to float32 round-off of the two XLA programs (not required bit-equal).
        assert np.allclose(a, b, atol=1e-5, rtol=0), np.abs(a - b).max()
        assert np.array_equal(np.asarray(g.last["nominal_actions"]), b)


@pytest.mark.parametrize("exec_mode", ["callback", "inline"])
def test_guidance_on_runs_in_the_real_sampler(tiny, exec_mode):
    from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
    from Baseline.paper_baselines.methods.interface import get_method

    model, obs = tiny
    _inner, g = _guided(model, get_method("nudge", kappa=0.0086), exec_mode)
    so = _scene_obs()
    sc = build_scene(so, obstacle_source="points")
    sc.obstacles = sc.obstacles.padded(1024, 4)
    outs = []
    for k in range(2):
        g.set_call(so, sc)
        outs.append(np.asarray(g.infer(dict(obs))["actions"]))
        assert g.last["guided"] is True and g.last["error"] is None
        assert np.all(np.isfinite(outs[-1]))
        if exec_mode == "callback":
            assert len(g.last["denoise_steps"]) == 10  # one host call per Euler step
    assert g.slot.stats["compiles"] == 1 and g.slot.stats["hits"] == 1  # same structure -> one compile


def test_callback_and_inline_agree(tiny):
    from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
    from Baseline.paper_baselines.methods.interface import get_method

    model, obs = tiny
    res = {}
    for mode in ("callback", "inline"):
        _inner, g = _guided(model, get_method("constrained_flow_matching"), mode)
        so = _scene_obs()
        sc = build_scene(so, obstacle_source="points")
        sc.obstacles = sc.obstacles.padded(1024, 4)
        g.set_call(so, sc)
        res[mode] = (np.asarray(g.infer(dict(obs))["actions"]), np.asarray(g.last["nominal_actions"]))
    (a, an), (b, bn) = res["callback"], res["inline"]
    assert np.array_equal(an, bn)
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()


def test_seeded_policy_rekeys_through_guided_policy(tiny):
    """``SeededPolicy`` finds the openpi policy's ``_rng`` under ``GuidedPolicy`` (``_policy`` chain)."""
    from AG3S.trajopt.policy_seed import rng_owner

    from Baseline.paper_baselines.methods.interface import get_method

    model, _obs = tiny
    inner, g = _guided(model, get_method("nudge", enabled=False))
    assert rng_owner(g) is inner


# --- wire level, fake policy ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recorded():
    if not SL5C_LI_VLA.is_dir():
        pytest.skip("SL5c VLA-only run not available")
    pytest.importorskip("openpi_client")
    from Baseline.paper_baselines.closed_loop.replay import RecordedEpisode

    return RecordedEpisode(SL5C_LI_VLA, "safelibero_spatial_t0_e0")


def _server(method, *extra, tmp=None):
    from Baseline.paper_baselines.closed_loop import serve_method as S

    args = S.build_parser().parse_args(["--method", method, "--port", "0", "--fake-policy", "recorded",
                                        "--fake-source-run", str(SL5C_LI_VLA), *extra])
    return S.build_server(args, record_root=tmp)


def test_wire_none_replays_the_recorded_vla_only_chunks(recorded, tmp_path):
    srv, info = _server("none", tmp=tmp_path)
    for k in range(4):
        out = srv.infer(readonly(recorded.request(k)))
        assert np.array_equal(np.asarray(out["actions"]), recorded.calls["actions_response"][k])
        assert out["panda/mode"] == "method:none"
    srv.close()
    rows = (tmp_path / "method_calls/calls.jsonl").read_text().splitlines()
    assert len(rows) == 4 and (tmp_path / "method_calls" / "ep000_seed700000.npz").exists()


def test_wire_guidance_round_trip_with_the_surrogate(recorded, tmp_path):
    import json

    srv, info = _server("nudge", "--target", "none", tmp=tmp_path)
    for k in range(3):
        out = srv.infer(readonly(recorded.request(k)))
        a = np.asarray(out["actions"])
        assert a.shape == (10, 7) and np.all(np.isfinite(a))
    srv.close()
    rows = [json.loads(l) for l in (tmp_path / "method_calls/calls.jsonl").read_text().splitlines()]
    assert all(r["fallback"] is None for r in rows), [r.get("error") for r in rows]
    assert all(r["guidance"]["guided"] for r in rows)
    assert {"da_xyz_l2_max", "intervened", "h_nominal", "h_served", "perception"} <= set(rows[-1])
    assert rows[0]["perception"]["n_robot_removed"] > 0 and rows[0]["perception"]["scene"]["privileged_inputs"] == []
