#!/usr/bin/env python3
"""GPU smoke (prepared for the verifier, **not run by the implementer**): the guidance methods on the real pi05_libero
network, recorded SL3 dev observations, same noise with / without ``guidance_fn``.

Per call: the recorded observation is rebuilt exactly as ``benchmark/runners/libero_pi05_client.py:357-375`` sends it
(``raw[::-1, ::-1]`` -> ``resize_with_pad`` 224, recorded ``policy_state``, task prompt), passed through the policy's
own input transforms, then ``Pi0.sample_actions(rng, observation, noise=noise, guidance_fn=fn)`` is jitted the
way ``Baseline/seam_vla/integration/openpi_jax.py`` does (``nnx.split`` -> merge inside the jit) and the output goes
through the policy's output transforms (raw LIBERO actions). The guidance closure captures the scene as constants, so
each call re-traces (fine for a smoke).

Reported per method / call: raw action chunks (unguided, guided), max |delta|, EE / hand min SDF of the decoded path
against the method's obstacle set, guided-vs-unguided wall time. Output: ``--out-dir`` (default
``outputs/paper_baselines_smoke/methods/guidance_gpu``).

Command (GPU lock + env as CLAUDE.md; see methods/*/README.md)::

    source scripts/setup/env.sh
    export OPENPI_DATA_HOME=$PWD/.cache/openpi XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    ( flock -w 3600 9 || exit 1
      PYTHONPATH=$PWD .venv-openpi-live/bin/python -m Baseline.paper_baselines.methods.guidance_common.gpu_smoke \\
          --methods omniguide constrained_flow_matching lan_o3dp_guidance_adapted nudge --episode 0 --calls 1 2
    ) 9>/tmp/ag3s-vla-gpu-validation.lock
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]


def storage_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage"))


def policy_inputs(npz_path: str, call: int, prompt: str, resize: int = 224) -> dict:
    from openpi_client import image_tools

    with np.load(npz_path, allow_pickle=False) as z:
        t = int(z["chunk_t"][call])
        img = np.ascontiguousarray(z["obs_agentview_image"][t][::-1, ::-1])
        wrist = np.ascontiguousarray(z["obs_robot0_eye_in_hand_image"][t][::-1, ::-1])
        state = np.asarray(z["policy_state"][call], np.float64)
    return {"observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize)),
            "observation/state": state, "prompt": prompt}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", default=["omniguide", "constrained_flow_matching", "lan_o3dp_guidance_adapted", "nudge"])
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--calls", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default=None, help="default: $AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject-sphere", action="store_true", help="also run with a privileged 3 cm sphere beside the path")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs/paper_baselines_smoke/methods/guidance_gpu"))
    a = ap.parse_args(argv)

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    from Baseline.paper_baselines.methods.guidance_common.core import load_norm_stats
    from Baseline.paper_baselines.methods.guidance_common.recorded import SL3_DEV_RUN, load_recorded_call
    from Baseline.paper_baselines.methods.interface import get_method, jsonable
    from Baseline.paper_baselines.methods.guidance_common.evaluation import SCENE_KW, injected_obs, path_metrics

    ckpt = pathlib.Path(a.checkpoint) if a.checkpoint else storage_root() / "checkpoints/panda/pi05_libero"
    train_cfg = _config.get_config(a.config)
    policy = _policy_config.create_trained_policy(train_cfg, ckpt)
    model = policy._model
    graphdef, state = nnx.split(model)
    norm = load_norm_stats(str(ckpt / "assets/physical-intelligence/libero/norm_stats.json"))
    H, D = model.action_horizon, model.action_dim

    def sample(rng, observation, noise, fn):
        def body(state, rng, observation, noise):
            m = nnx.merge(graphdef, state)
            return m.sample_actions(rng, observation, noise=noise, guidance_fn=fn)
        return jax.jit(body)(state, rng, observation, noise)

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for call in a.calls:
        obs, chunk_rec, meta = load_recorded_call(SL3_DEV_RUN, a.episode, call)
        inputs = policy._input_transform(jax.tree.map(lambda x: x, policy_inputs(meta["npz"], call, obs.instruction)))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        rng = jax.random.key(a.seed)
        noise = np.asarray(jax.random.normal(jax.random.key(a.seed + 1), (1, H, D)), np.float32)

        def to_raw(x):
            out = policy._output_transform({"state": np.asarray(inputs["state"][0]), "actions": np.asarray(x[0])})
            return np.asarray(out["actions"], np.float64)

        t0 = time.perf_counter()
        x_base = np.asarray(sample(rng, observation, jnp.asarray(noise), None))
        t_base = time.perf_counter() - t0
        raw_base = to_raw(x_base)
        variants = {"scene": obs}
        if a.inject_sphere:
            from Baseline.paper_baselines.methods.guidance_common.scene import build_scene

            variants["injected"] = injected_obs(obs, build_scene(obs, **SCENE_KW), raw_base)[0]
        for vname, o in variants.items():
            for name in a.methods:
                m = get_method(name, scene_kwargs=SCENE_KW)
                fn = m.make_guidance_fn(o, norm, noise=noise)
                t0 = time.perf_counter()
                x_g = np.asarray(sample(rng, observation, jnp.asarray(noise), fn))
                t_g = time.perf_counter() - t0
                raw_g = to_raw(x_g)
                mb, mg = path_metrics(raw_base, m.last_scene, raw_base.shape[0]), path_metrics(raw_g, m.last_scene, raw_g.shape[0])
                rec = {"call": call, "variant": vname, "method": name, "meta": meta,
                       "max_abs_delta_raw": float(np.abs(raw_g - raw_base).max()),
                       "ee_min_sdf_base_m": mb["ee_min_sdf_m"], "ee_min_sdf_guided_m": mg["ee_min_sdf_m"],
                       "hand_min_sdf_base_m": mb["hand_min_sdf_m"], "hand_min_sdf_guided_m": mg["hand_min_sdf_m"],
                       "wall_s_base_incl_compile": t_base, "wall_s_guided_incl_compile": t_g,
                       "raw_base": raw_base.tolist(), "raw_guided": raw_g.tolist(), "recorded_chunk": chunk_rec.tolist(),
                       "provenance": m.provenance()}
                records.append(rec)
                print(json.dumps({k: rec[k] for k in ("call", "variant", "method", "max_abs_delta_raw", "ee_min_sdf_base_m",
                                                      "ee_min_sdf_guided_m", "hand_min_sdf_base_m", "hand_min_sdf_guided_m")}),
                      flush=True)
    (out_dir / "gpu_smoke.json").write_text(json.dumps(jsonable({"checkpoint": str(ckpt), "config": a.config,
                                                                  "seed": a.seed, "records": records}), indent=1))
    print("wrote", out_dir / "gpu_smoke.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
