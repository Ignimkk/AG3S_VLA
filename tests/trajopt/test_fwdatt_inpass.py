"""FWDATT (2026-11-13): action-query attention read from the served sampling (`AG3S/trajopt/inpass_attention.py`).

| # | what | where |
|---|---|---|
| 1 | in-pass actions vs `Pi0.sample_actions`, same rng (target max abs diff <= 1e-5) | tiny pi05_libero, subprocess |
| 2 | in-pass step-0 cell probs vs `AttentionSampler._jit_prefix` / `_jit_suffix` step 0, same noise | tiny, subprocess |
| 3 | `inner._rng` after two in-pass infers == after two `Policy.infer`; same result keys / dtypes | tiny, subprocess |
| 4 | flag off: `AttentionPolicy` takes the earlier path, result keys unchanged, nothing in-pass built | stub, in process |
| 5 | guidance hooks (`guidance_fn`, `guidance_xv_fn`) refused, at construction and at infer | both |
| + | `--attention-inpass`: parser default off, RB-Y1 refuses, Panda refuses the contradicting flags | in process |

The tiny-model checks run in a subprocess with ``JAX_PLATFORMS=cpu`` (no GPU; the helper replaces two factory
functions of openpi in its own process only).
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _has(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


# --- 1, 2, 3, 5 on a tiny random pi05_libero model -----------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    done = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests/trajopt/fwdatt_tiny_inpass.py")],
                          cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=1200)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    print("FWDATT tiny-model numbers:", json.dumps({k: v for k, v in out.items() if k != "refusals"}))
    return out


needs_jax = pytest.mark.skipif(not _has("jax", "flax", "openpi"), reason="needs jax/flax/openpi (.venv-openpi-live)")


@needs_jax
def test_1_actions_match_sample_actions(tiny):
    assert tiny["backend"] == "cpu"
    assert max(tiny["actions_max_abs_diff"]) <= 1e-5, tiny["actions_max_abs_diff"]
    assert max(tiny["infer_actions_max_abs_diff"]) <= 1e-5   # after the output transform, through Policy.infer order
    assert tiny["num_steps5_max_abs_diff"] <= 1e-5           # `num_steps` in sample_kwargs is reproduced
    assert tiny["ap_actions_max_abs_diff_same_noise"] <= 1e-5
    assert tiny["actions_dtype"] == "float32"


@needs_jax
def test_2_step0_probs_match_attention_sampler(tiny):
    assert max(tiny["probs_max_abs_diff"]) <= 1e-6, tiny["probs_max_abs_diff"]
    assert all(a == b == [10, 768] for a, b in tiny["probs_shapes"])
    # the default-path format and values (AttentionPolicy with the seed-0 noise both ways)
    assert tiny["ap_attention_cameras"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert tiny["ap_attention_shapes_dtypes"] == [[[16, 16], "float32"]]
    assert tiny["ap_attention_max_abs_diff"] <= 1e-6
    assert tiny["ap_keys_on"] == tiny["ap_keys_off"] == ["actions", "attention", "policy_timing"]
    # an `action:` attention_source names another cell
    assert tiny["alt_cell_max_abs_diff"] <= 1e-6
    assert tiny["alt_source_record"] == {"spec": "action:mean:L3H5", "query": "action", "layer": 3, "head": 5,
                                         "agg": "mean", "denoise": 0, "attention": "ok"}


@needs_jax
def test_3_rng_advance_and_result_keys(tiny):
    assert tiny["rng_equal_after_two"] is True
    assert tiny["infer_keys_equal"] and tiny["infer_dtypes_equal"]
    assert tiny["infer_keys"] == ["actions", "policy_timing"] and tiny["policy_timing_keys"] == ["infer_ms"]


@needs_jax
def test_5_guidance_hooks_refused(tiny):
    r = tiny["refusals"]
    for key in ("guidance_fn", "guidance_xv_fn", "guidance_fn/AttentionPolicy", "guidance_xv_fn/AttentionPolicy",
                "late_hook"):
        assert r[key].startswith("refused:"), (key, r[key])
    assert r["late_hook_rng_untouched"] is True   # refused before the RNG is split


# --- 5 (pure python part) ------------------------------------------------------------------------------------------
class _P:
    def __init__(self, kw):
        self._sample_kwargs = kw


@pytest.mark.parametrize("kw", [{"guidance_fn": lambda x, t: x}, {"guidance_xv_fn": lambda *a: a[3]},
                                {"num_steps": 10, "guidance_fn": lambda x, t: x}, {"noise_schedule": "x"},
                                {"num_steps": 0}])
def test_5_check_sample_kwargs_refuses(kw):
    from AG3S.trajopt.inpass_attention import InpassUnsupported, check_sample_kwargs

    with pytest.raises(InpassUnsupported):
        check_sample_kwargs(_P(kw))


def test_check_sample_kwargs_accepts_plain():
    from AG3S.trajopt.inpass_attention import check_sample_kwargs

    assert check_sample_kwargs(_P({})) == 10 and check_sample_kwargs(_P(None)) == 10
    assert check_sample_kwargs(_P({"num_steps": 5})) == 5
    assert check_sample_kwargs(_P({"guidance_fn": None, "guidance_xv_fn": None})) == 10


# --- 4. flag off ---------------------------------------------------------------------------------------------------
class _StubPolicy:
    def infer(self, obs, **_):
        return {"actions": np.zeros((10, 7)), "policy_timing": {"infer_ms": 1.0}}


def test_4_flag_off_default_and_unchanged_path(monkeypatch):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    sig = inspect.signature(AttentionPolicy.__init__).parameters
    assert sig["inpass"].default is False and sig["model_config"].default is None
    ap = AttentionPolicy(_StubPolicy(), None, sampler=object(), subtask_probe=None)
    assert ap.inpass is None and ap._extended is False                     # noqa: SLF001
    maps = {"cam_high": np.ones((16, 16), np.float32)}
    calls = []
    monkeypatch.setattr(ap, "_attention_and_feature", lambda obs: (calls.append(1), (maps, None))[1])
    out = ap.infer({"x": 1})
    assert calls == [1] and out["attention"] is maps
    assert set(out) == {"actions", "policy_timing", "attention"}


def test_4_flag_off_extended_path_unchanged():
    from AG3S.trajopt.attention_policy import AttentionPolicy

    class _Backend:
        def __init__(self):
            self.calls = []

        def run(self, inputs, **kw):
            self.calls.append(kw)
            return {"cell": np.zeros(768, np.float32)}

    class _P2(_StubPolicy):
        @staticmethod
        def _input_transform(obs):
            return obs

    be = _Backend()
    ap = AttentionPolicy(_P2(), None, sampler=object(), backend=be, subtask_probe=None,
                         attention_source="action:last:L12H3")
    assert ap.inpass is None and ap._extended is True                      # noqa: SLF001
    out = ap.infer({"prompt": "x"})
    assert len(be.calls) == 1 and set(out) == {"actions", "policy_timing", "attention", "attention_source"}


@pytest.mark.parametrize("kw", [
    {"attention_source": "noun_target:L8H2"}, {"task_probe": object()}, {"prefix_feature": True},
    {"hsr6_probe": object()}, {"hsr5_probe": object()}, {"subtask_probe": "default"}, {"sampler": object()},
    {"backend": object()}])
def test_inpass_refuses_what_it_does_not_cover(kw):
    from AG3S.trajopt.attention_policy import AttentionPolicy

    kw = {"subtask_probe": None, **kw}
    with pytest.raises(ValueError, match="inpass=True"):
        AttentionPolicy(_StubPolicy(), None, inpass=True, model_config=object(), **kw)


def test_inpass_needs_model_config():
    from AG3S.trajopt.attention_policy import AttentionPolicy

    with pytest.raises(ValueError, match="model_config"):
        AttentionPolicy(_StubPolicy(), None, subtask_probe=None, inpass=True)


# --- server flag ---------------------------------------------------------------------------------------------------
def test_parser_flag_default_off_and_rby1_refuses():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    assert ap.get_default("attention_inpass") is False
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, argparse.Namespace(robot="rby1", attention_inpass=True))
    ss.reject_panda_only_flags(ap, argparse.Namespace(robot="panda", attention_inpass=True))


@pytest.mark.parametrize("extra, ok", [
    ({}, True), ({"attention_source": "action:last:L12H3"}, True),
    ({"no_attention": True}, False), ({"no_safe": True}, False), ({"attention_source": "noun_target:L8H2"}, False),
    ({"hsr6_probe": "x.npz", "hsr6_flags": True}, False)])
def test_panda_inpass_flag_checks(extra, ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    extra = dict(extra)
    hsr6 = extra.pop("hsr6_flags", False)
    monkeypatch.setattr(ss, "hsr6_flags_from_args", lambda args: object() if hsr6 else None)
    args = argparse.Namespace(**{"attention_inpass": True, "no_attention": False, "no_safe": False,
                                 "attention_source": None, "hsr6_probe": None, **extra})
    if ok:
        ps.check_inpass_flags(args, None)
    else:
        with pytest.raises(ValueError, match="--attention-inpass"):
            ps.check_inpass_flags(args, None)
    ps.check_inpass_flags(argparse.Namespace(attention_inpass=False, no_attention=True, no_safe=True,
                                             attention_source="noun_target:L8H2"), None)   # off: no check


def test_panda_record_key_only_when_on():
    from AG3S.adapters.panda import server as ps

    assert ps.attention_inpass_record(argparse.Namespace(attention_inpass=False)) == {}
    assert ps.attention_inpass_record(argparse.Namespace()) == {}
    assert ps.attention_inpass_record(argparse.Namespace(attention_inpass=True)) == {"attention_inpass": True}
