"""F2 (``--filter-rate step``) and KNOWS F4a (attention tap) on the server side, and the arm / launcher records.

* step mode: the policy is called once per 5 env steps with the same seeds (nominal chunks and the policy RNG equal
  chunk mode, bit for bit), row ``k`` of the cached chunk is filtered at step ``k`` (one sidecar call per env step), the
  records keep one row per step and ``metrics.method_calls`` counts calls once;
* attention tap: the chunk with the tap equals the chunk without it (bit for bit, same RNG), and the tap's grid is the
  denoising-step-0 attention of the very sample that produced the chunk (an attention-enabled manual denoising loop from
  the same noise reproduces the policy's ``sample_actions`` output and its step-0 map equals the tap's).

Tiny pi0.5 on CPU (as ``test_server_identity.py``). ``.venv-openpi-live``::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD:$PWD/third_party/openpi/src \
      .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_filter_rate_attention.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "third_party/openpi/src"))

from Baseline.paper_baselines.closed_loop.tests.test_server_identity import make_policy, tiny  # noqa: E402,F401


class StubSidecar:
    """Records every filter call; 'filters' by adding 0.01 to x of every row."""

    def __init__(self):
        self.calls = []

    def call(self, op, **kw):
        if op == "reset":
            self.calls.append(("reset", None))
            return {}
        assert op == "filter"
        A = np.asarray(kw["actions"], np.float64)
        self.calls.append(("filter", A.copy(), kw["obs"]["extras"].get("knows/attention")))
        B = A.copy()
        B[:, 0] += 0.01
        return {"actions": B, "intervened": np.ones(len(A), bool), "ms": 0.1,
                "info": {"steps": [{"min_h": 0.1, "status": "solved"} for _ in A], "obstacles": {}, "perception": {}}}

    def close(self):
        return 0


def stub_scene_obs(server):
    from Baseline.paper_baselines.methods.interface import SceneObservation

    def f(obs, seq, row=None):
        return SceneObservation(rgb={}, depth={}, intrinsics={}, T_base_cam={}, q=np.zeros(7), ee_pose=np.eye(4),
                                gripper=0.0, instruction="", extras={"seq": seq})
    server._scene_obs = f
    return server


def requests_step(obs, n_calls=3, replan=5, seed=700000):
    out = []
    for c in range(n_calls):
        for k in range(replan):
            r = dict(obs)
            r.update({"ag3s/seq": c, "ag3s/reset": c == 0 and k == 0, "ag3s/phase": "approach",
                      "panda/robot_state": np.zeros(9), "ag3s/filter_step": k})
            if c == 0 and k == 0:
                r["ag3s/policy_seed"] = seed
            out.append(r)
    return out


def rng_bytes(policy):
    return np.asarray(jax.random.key_data(policy._rng)).tobytes()


def make_server(model, rate, tmp, tap_backend=None, arm=None):
    from Baseline.paper_baselines.closed_loop import arms
    from Baseline.paper_baselines.closed_loop.attention_tap import AttentionTap
    from Baseline.paper_baselines.closed_loop.serve_method import CallRecorder, MethodServer

    pol = make_policy(model)
    tap = AttentionTap(pol, tap_backend) if tap_backend is not None else None
    srv = MethodServer(pol, arm or arms.resolve("aegis", filter_rate=rate), sidecar=StubSidecar(),
                       recorder=CallRecorder(tmp), filter_rate=rate, attention_tap=tap)
    return stub_scene_obs(srv), pol


def test_step_mode_same_policy_calls_one_filter_row_per_step(tiny, tmp_path):
    from Baseline.paper_baselines.closed_loop import metrics
    from Baseline.paper_baselines.closed_loop.tests.test_server_identity import requests

    model, obs = tiny
    chunk_srv, chunk_pol = make_server(model, "chunk", tmp_path / "chunk" / "method_calls")
    step_srv, step_pol = make_server(model, "step", tmp_path / "step" / "method_calls")
    chunk_out = [chunk_srv.infer(r) for r in requests(obs, 3)]
    step_out = [step_srv.infer(r) for r in requests_step(obs, 3)]
    assert rng_bytes(chunk_pol) == rng_bytes(step_pol)  # the policy RNG advanced exactly as in chunk mode
    # nominal chunks: the stub saw them (chunk mode: whole chunk; step mode: row k at step k)
    noms = [c[1] for c in chunk_srv.sidecar.calls if c[0] == "filter"]
    rows = [c[1] for c in step_srv.sidecar.calls if c[0] == "filter"]
    D = noms[0].shape[1]  # tiny policy: no output transform, D = action_dim
    assert len(noms) == 3 and all(n.shape == (10, D) for n in noms)
    assert len(rows) == 15 and all(r.shape == (1, D) for r in rows)
    for i, r in enumerate(rows):
        assert np.array_equal(r[0], noms[i // 5][i % 5])  # row k of the same nominal chunk, bit for bit
    # responses: policy step = chunk with row 0 filtered; filter step = the filtered row
    for i, out in enumerate(step_out):
        k, c = i % 5, i // 5
        a = np.asarray(out["actions"], np.float64)
        if k == 0:
            assert a.shape == (10, D) and np.allclose(a[0, 0], noms[c][0, 0] + 0.01)
            assert np.array_equal(a[1:], noms[c][1:].astype(a.dtype))
            if c == 0:
                assert out["policy_seed"] == 700000
        else:
            assert a.shape == (1, D) and np.allclose(a[0, 0], noms[c][k, 0] + 0.01)
        assert out["panda/filter_step"] == k
    for c in range(3):  # chunk mode served the whole filtered chunk; step mode's executed rows are the same rows
        executed = np.concatenate([np.asarray(step_out[5 * c + k]["actions"])[:1] for k in range(5)])
        assert np.allclose(np.asarray(chunk_out[c]["actions"])[:5], executed)
    # metadata: step mode announces itself, chunk mode is the policy's own metadata
    assert step_srv.metadata.get("ag3s_filter_rate") == "step" and "ag3s_filter_rate" not in (chunk_srv.metadata or {})
    step_srv.recorder.close()
    chunk_srv.recorder.close()
    rows_j = [json.loads(l) for l in (tmp_path / "step/method_calls/calls.jsonl").read_text().splitlines()]
    assert [r["call_kind"] for r in rows_j] == (["policy"] + ["filter"] * 4) * 3 and [r["k"] for r in rows_j][:5] == [0, 1, 2, 3, 4]
    with np.load(sorted((tmp_path / "step/method_calls").glob("ep*.npz"))[0]) as z:
        assert list(z["seq"]) == [0, 1, 2] and z["nominal"].shape == (3, 10, D) and z["step_served"].shape == (15, D)
        assert list(z["step_k"]) == [0, 1, 2, 3, 4] * 3
        assert np.array_equal(z["nominal"], np.stack(noms))
    mc = metrics.method_calls(tmp_path / "step")
    assert mc["n_calls"] == 3 and mc["steps"]["n_steps"] == 15 and mc["executed_rows_intervened"] == 15
    assert mc["intervention_rate_calls"] == 1.0 and mc["intervention_rate_rows"] == 1.0
    assert mc["steps"]["by_kind"]["policy"]["n"] == 3 and mc["steps"]["by_kind"]["filter"]["n"] == 12
    mcc = metrics.method_calls(tmp_path / "chunk")
    assert "steps" not in mcc and mcc["n_calls"] == 3


def test_step_and_chunk_requests_are_refused_by_the_other_mode(tiny, tmp_path):
    from Baseline.paper_baselines.closed_loop.tests.test_server_identity import requests

    model, obs = tiny
    chunk_srv, _ = make_server(model, "chunk", None)
    step_srv, _ = make_server(model, "step", None)
    with pytest.raises(ValueError):
        chunk_srv.infer(requests_step(obs, 1)[0])
    with pytest.raises(ValueError):
        step_srv.infer(requests(obs, 1)[0])
    step_srv.infer(requests_step(obs, 1)[0])
    bad = dict(requests_step(obs, 1)[1], **{"ag3s/seq": 4})
    with pytest.raises(ValueError):  # a filter step of a chunk the server never produced
        step_srv.infer(bad)


# --- attention tap -------------------------------------------------------------------------------------------------
def tiny_backend(model):
    from openpi.models import pi0_config

    import openpi.models.pi0 as pi0mod
    from Baseline.paper_baselines.closed_loop.attention_tap import OpenpiAttentionBackend

    orig = pi0mod._siglip.Module

    def small(**kw):
        kw["variant"] = "mu/16"
        return orig(**kw)

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                               action_horizon=10, max_token_len=48, dtype="float32")

    class Lazy:  # the backend builds its graphdef on the first policy it sees (needs the same patched SigLIP)
        def __call__(self, policy):
            pi0mod._siglip.Module = small
            try:
                return OpenpiAttentionBackend(policy, cfg, layer=2, head=3, tokens=(0, 196))
            finally:
                pi0mod._siglip.Module = orig
    return Lazy()


def test_tap_keeps_the_chunk_and_reads_step0_of_the_same_sample(tiny):
    import einops
    import jax.numpy as jnp
    from AG3S.trajopt.policy_seed import SeededPolicy

    from Baseline.paper_baselines.closed_loop.attention_tap import AttentionTap
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    model, obs = tiny
    ref_pol, tap_pol = make_policy(model), make_policy(model)
    backend = tiny_backend(model)(tap_pol)
    ref, tap = SeededPolicy(ref_pol), AttentionTap(tap_pol, backend)
    tapped = SeededPolicy(tap)
    for i, seed in enumerate((700000, None, None, 701002)):
        o = dict(obs)
        if seed is not None:
            o["policy_seed"] = seed
        rng_before = tap_pol._rng if seed is None else jax.random.key(seed)
        a, b = ref.infer(dict(o)), tapped.infer(dict(o))
        assert np.array_equal(np.asarray(a["actions"]), np.asarray(b["actions"]))  # chunk bit-identical
        assert rng_bytes(ref_pol) == rng_bytes(tap_pol)
        g = tap.last["grid"]
        assert g is not None and g.shape == (14, 14) and np.all(np.isfinite(g)), tap.last
        # the same sample, replayed with the attention model: noise from the policy's split, 10 Euler steps
        o2 = {k: v for k, v in o.items() if k != "policy_seed"}
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], tap_pol._input_transform(jax.tree.map(lambda x: x, o2)))
        ob = _model.Observation.from_dict(inputs)
        _, srng = jax.random.split(rng_before)
        x0_policy = np.asarray(tap_pol._sample_actions(srng, ob))
        noise = jax.random.normal(srng, (1, 10, model.action_dim))
        am = backend._nnx.merge(backend._graphdef, backend._state)
        pob = _model.preprocess_observation(None, ob, train=False)
        pt, pm, par = am.embed_prefix(pob)
        kv = am.PaliGemma.llm([pt, None], mask=make_attn_mask(pm, par), positions=jnp.cumsum(pm, axis=1) - 1)[1]
        x, t, step0 = noise, 1.0, None
        for s in range(10):
            st, sm, sar, ad = am.embed_suffix(pob, x, jnp.broadcast_to(jnp.asarray(t, jnp.float32), (1,)))
            full = jnp.concatenate([einops.repeat(pm, "b p -> b s p", s=st.shape[1]), make_attn_mask(sm, sar)], -1)
            pos = jnp.sum(pm, -1)[:, None] + jnp.cumsum(sm, -1) - 1
            (_, so), _, probs = am.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv, adarms_cond=[None, ad])
            if s == 0:
                step0 = np.asarray(probs[2, 0, 0, 3, -10:, 0:196].astype(jnp.float32)).mean(0).reshape(14, 14)
            x, t = x + (-0.1) * am.action_out_proj(so[:, -10:]), t - 0.1
        assert np.allclose(np.asarray(x), x0_policy, atol=1e-4)  # the replay IS the policy's sample
        assert np.allclose(g, step0[:, ::-1], atol=1e-6)  # tap grid (OpenCV-oriented) = its step-0 map


def test_knows_attention_arm_hands_the_grid_to_the_method(tiny, tmp_path):
    from Baseline.paper_baselines.closed_loop import arms

    model, obs = tiny
    arm = arms.resolve("knows")
    assert arm["target"] == "attention"
    pol = make_policy(model)
    backend = tiny_backend(model)(pol)
    from Baseline.paper_baselines.closed_loop.attention_tap import AttentionTap
    from Baseline.paper_baselines.closed_loop.serve_method import CallRecorder, MethodServer

    srv = stub_scene_obs(MethodServer(pol, arm, sidecar=StubSidecar(), recorder=CallRecorder(None), filter_rate="step",
                                      attention_tap=AttentionTap(pol, backend)))
    for r in requests_step(obs, 2):
        srv.infer(r)
    atts = [c[2] for c in srv.sidecar.calls if c[0] == "filter"]
    assert len(atts) == 10 and all(a is not None for a in atts)
    assert [a["seq"] for a in atts] == [1] * 5 + [2] * 5  # the grid of the call that produced the chunk, held 5 steps
    assert np.array_equal(atts[0]["grid"], atts[4]["grid"]) and not np.array_equal(atts[0]["grid"], atts[5]["grid"])


# --- arms / launcher --------------------------------------------------------------------------------------------------
def test_arm_records_per_filter_rate_and_target():
    from Baseline.paper_baselines.closed_loop import arms

    for name in ("aegis", "multilink_cbf", "knows"):
        c, s = arms.resolve(name), arms.resolve(name, filter_rate="step")
        assert arms.COMMON_FILTER_DEVIATIONS[0] in c["deviation_from_paper"] and "filter_rate" not in c
        assert arms.STEP_FILTER_DEVIATION in s["deviation_from_paper"] and s["filter_rate"] == "step"
        assert arms.COMMON_FILTER_DEVIATIONS[0] not in s["deviation_from_paper"]
        g = arms.resolve(name, gt="default", filter_rate="step")
        assert arms.STEP_FILTER_DEVIATION in g["deviation_from_paper"]
    assert arms.resolve("knows")["variant"] == "knows:obst-detector:target-attention"
    old = arms.resolve("knows", target="detector")
    assert old["variant"] == "knows:obst-detector:target-detector" and old["target"] == "detector"
    assert any(d.startswith("target exclusion by the Grounding DINO target box") for d in old["deviation_from_paper"])
    assert arms.resolve("aegis")["target"] == "detector"  # AEGIS / multi-link keep their target handling
    with pytest.raises(ValueError):
        arms.resolve("aegis", target="attention")
    with pytest.raises(ValueError):
        arms.resolve("none", filter_rate="step")
    with pytest.raises(ValueError):
        arms.resolve("omniguide", filter_rate="step")


def test_launcher_argv_default_unchanged_and_step_flags():
    out = subprocess.run([str(ROOT / ".venv-ag3s/bin/python"), "-c", """
import json, sys
sys.path.insert(0, %r)
from Baseline.paper_baselines.closed_loop import launch_method as lm
base = ["--method", "aegis", "--task-ids", "0", "--episode-ids", "0", "--run-id", "x", "--safety-level", "I"]
res = {}
for extra in ([], ["--filter-rate", "chunk"], ["--filter-rate", "step"]):
    a = lm.build_parser().parse_args(base + extra)
    a.data_split = "safelibero_spatial_LI_dev"
    p = lm.paired_namespace(a)
    res[" ".join(extra) or "default"] = {"server": lm.server_argv(a, 1, None), "client": lm.method_client_argv(a, "/r", 1, p),
                                          "arm": lm.resolve_arm(a)}
print(json.dumps(res))
""" % str(ROOT)], capture_output=True, text=True, cwd=str(ROOT), env={"PATH": "/usr/bin:/bin", "AG3S_PROJECT_ROOT": str(ROOT)})
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.loads(out.stdout.strip().splitlines()[-1])
    d, c, s = res["default"], res["--filter-rate chunk"], res["--filter-rate step"]
    assert d == c and "--filter-rate" not in d["server"] and "--filter-every-step" not in d["client"]
    assert s["server"] == d["server"] + ["--filter-rate", "step"] and s["client"] == d["client"] + ["--filter-every-step"]
    assert s["arm"]["filter_rate"] == "step"
