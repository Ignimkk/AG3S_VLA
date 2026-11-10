#!/usr/bin/env python3
"""P3 (제약 단계 추종, 2026-11-10 KST): per-request pi0.5 prefix features of the pe123458 dev-split episodes (GPU, offline).

No policy_records exist for the AG3S closed-loop runs, so each request is rebuilt from the episode npz exactly as the client
sends it (`attention_audit.client_images`: 180 deg rotation + resize_with_pad 224; state = `policy_state[k]`; prompt =
`task_description`) and put through `policy._input_transform`. One jitted prefix pass per request (`pi05_hidden.HiddenExtractor`
with K added), pooled on the device:

  hidden [19, 4, 2048] f16   Gemma residual stream h_0 (input embedding; image rows = SigLIP projection) .. h_18
  K, V   [18, 4, 256]  f32   KV cache (gemma_2b is multi-query: one KV head per layer)
  rows (axis 1): 0 agentview-camera token mean, 1 wrist-camera token mean, 2 valid-text mean, 3 last valid text token
  kv_L4  [512]         f32   the served probe feature (text mean | mean over all 768 image tokens of V layer 4)

Before extracting, the path is validated against the 11-04 attention audit on this machine
(`nb1104-probe-attention-audit-dev-e1-4`, `kv_feature` per call, built from policy_records of `nb1104-probe-dev-pi05only`):
max relative error of kv_L4 over the first `--validate-episodes` episodes must be < 1e-2, else the run stops.

usage (GPU, shared lock — offline extraction):
  source scripts/setup/env.sh
  python3 scripts/setup/gpu_lock.py shared --est-gib 24 --tag p3-hidden -- \
      .venv-openpi-live/bin/python experiments/segmentation/stage_track/p3_extract_hidden.py outputs/2026-11-10_제약단계_추종
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path("/mnt/dev/AG3S_VLA")
sys.path.insert(0, str(ROOT))
RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
CHECKPOINT = "/mnt/dev/AG3S_VLA-storage/checkpoints/panda/pi05_libero"
CONFIG = "pi05_libero"
AUDIT_RUN = RESULTS / "20261104T055314_fe3abf215dc4_nb1104-probe-attention-audit-dev-e1-4"
SOURCE_RUN = RESULTS / "20261104T051534_fe3abf215dc4_nb1104-probe-dev-pi05only-t0-9e1-4"
VSUFFIX = "-vla_v11ab-AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-C"
N_IMG, CAM = 768, ((0, 256), (256, 512))           # agentview, wrist (the third block is the padded right wrist)
KV_LAYER = 4


def build_extractor(model):
    import jax
    import jax.numpy as jnp

    from AG3S.ag3s.experiments.sources.pi05_hidden import HiddenExtractor

    class Extractor(HiddenExtractor):
        def _kvh(self, state, obs):
            from openpi.models import model as _model
            from openpi.models.pi0 import make_attn_mask

            m = self._nnx.merge(self._graphdef, state)
            obs = _model.preprocess_observation(None, obs, train=False)
            tokens, mask, ar_mask = m.embed_prefix(obs)
            attn = make_attn_mask(mask, ar_mask)
            positions = jnp.cumsum(mask, axis=1) - 1
            (_, _), kv, hidden = m.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
            return kv, hidden, mask

        def _feat(self, state, obs):
            kv, hidden, mask = self._kvh(state, obs)
            k_all, v_all = kv                                  # (L, B, T, 1, 256)
            mf = mask[0].astype(jnp.float32)
            tm = mf[N_IMG:]
            last = mask.shape[1] - 1 - jnp.argmax(mask[0][::-1].astype(jnp.int32))

            def pool(x):                                       # x [L, T, D] f32 -> [L, 4, D]
                rows = [x[:, a:b].mean(1) for a, b in CAM]
                rows.append((x[:, N_IMG:] * tm[None, :, None]).sum(1) / tm.sum())
                rows.append(x[:, last])
                return jnp.stack(rows, 1)

            h = hidden[:, 0].astype(jnp.float32)
            kk = k_all[:, 0, :, 0, :].astype(jnp.float32)
            vv = v_all[:, 0, :, 0, :].astype(jnp.float32)
            v4 = vv[KV_LAYER]
            kv4 = jnp.concatenate([(v4[N_IMG:] * tm[:, None]).sum(0) / tm.sum(), v4[:N_IMG].mean(0)])
            return pool(h).astype(jnp.float16), pool(kk), pool(vv), kv4, jnp.abs(h).max(axis=(1, 2))

        def features(self, obs):
            from openpi.shared import array_typing as at

            with at.disable_typechecking():
                if getattr(self, "_fc", None) is None:
                    t0 = time.perf_counter()
                    self._fc = jax.jit(self._feat).lower(self._state, obs).compile()
                    self.feat_compile_s = time.perf_counter() - t0
                out = self._fc(self._state, obs)
            return [np.asarray(x) for x in jax.block_until_ready(out)]

    return Extractor(model)


def observation(policy, request):
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    inputs = policy._input_transform(dict(request))  # noqa: SLF001
    return _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs))


def requests_of(ep_npz: pathlib.Path, prompt: str):
    from benchmark.runners.attention_audit import client_images

    with np.load(ep_npz) as z:
        chunk_t, state = z["chunk_t"].astype(int), z["policy_state"]
        a, w = z["obs_agentview_image"], z["obs_robot0_eye_in_hand_image"]
        for k, t in enumerate(chunk_t):
            img, wrist = client_images(a[t], w[t], 224)
            yield k, int(t), {"observation/image": img, "observation/wrist_image": wrist,
                              "observation/state": np.asarray(state[k]), "prompt": prompt}


def episodes_meta(run: pathlib.Path) -> dict:
    out = {}
    for l in (run / "episodes.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            out[f"{r['suite']}_t{r['task_id']}_e{r['episode_idx']}"] = r
    return out


def validate(policy, ex, n_eps: int) -> dict:
    meta = episodes_meta(SOURCE_RUN)
    files = sorted((AUDIT_RUN / "attention").glob("*_attn.npz"))[:n_eps]
    errs, n = [], 0
    for f in files:
        stem = f.name[: -len("_attn.npz")]
        with np.load(f) as z:
            ref, ck = z["kv_feature"], z["call_k"]
        for k, t, req in requests_of(SOURCE_RUN / "episodes" / f"{stem}.npz", meta[stem]["task_description"]):
            if k not in set(int(x) for x in ck):
                continue
            kv4 = ex.features(observation(policy, req))[3]
            r = ref[list(ck).index(k)]
            errs.append(float(np.abs(kv4 - r).max() / (np.abs(r).max() + 1e-12)))
            n += 1
    return {"episodes": [f.name for f in files], "requests": n, "max_rel_err": max(errs), "median_rel_err": float(np.median(errs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--validate-episodes", type=int, default=2)
    ap.add_argument("--limit-episodes", type=int, default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.out_root)
    out = root / "p3"
    (out / "features").mkdir(parents=True, exist_ok=True)

    import jax

    from AG3S.ag3s.experiments.sources.pi05_hidden import load_hidden_policy

    rec = {"started": time.strftime("%F %T %Z"), "argv": sys.argv, "jax_devices": [str(d) for d in jax.devices()],
           "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "gpu_lock_mode": os.environ.get("AG3S_GPU_LOCK_MODE"), "checkpoint": CHECKPOINT, "config": CONFIG}
    if jax.default_backend() != "gpu":
        raise SystemExit(f"expected a GPU backend, got {jax.devices()}")
    t0 = time.perf_counter()
    policy, _ = load_hidden_policy(CONFIG, CHECKPOINT)
    ex = build_extractor(policy._model)  # noqa: SLF001
    rec["load_s"] = time.perf_counter() - t0

    rec["validation"] = validate(policy, ex, args.validate_episodes)
    rec["feat_compile_s"] = getattr(ex, "feat_compile_s", None)
    print(json.dumps(rec["validation"]), flush=True)
    (out / "extract_record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    if rec["validation"]["max_rel_err"] >= 1e-2:
        raise SystemExit("validation failed: kv_L4 differs from the 11-04 audit (>= 1e-2) -- stop")

    eps = json.loads((root / "p2" / "p2_episodes.json").read_text())
    if args.limit_episodes:
        eps = eps[: args.limit_episodes]
    secs, n_req = [], 0
    for i, e in enumerate(eps):
        tag, stem = e["tag"], e["stem"]
        run = sorted(RESULTS.glob(f"*_{tag}{VSUFFIX}"))[-1]
        target = out / "features" / tag / f"{stem}.npz"
        if target.exists():
            continue
        prompt = episodes_meta(run)[stem]["task_description"]
        cols = {k: [] for k in ("hidden", "K", "V", "kv_L4", "absmax", "obs_index", "call_k")}
        for k, t, req in requests_of(run / "episodes" / f"{stem}.npz", prompt):
            s = time.perf_counter()
            h, kk, vv, kv4, am = ex.features(observation(policy, req))
            secs.append(time.perf_counter() - s)
            for key, val in zip(("hidden", "K", "V", "kv_L4", "absmax", "obs_index", "call_k"), (h, kk, vv, kv4, am, t, k)):
                cols[key].append(val)
        if not np.isfinite(np.stack(cols["hidden"]).astype(np.float32)).all():
            raise SystemExit(f"{tag}/{stem}: non-finite hidden (float16 overflow?)")
        target.parent.mkdir(exist_ok=True)
        with open(target, "xb") as f:
            np.savez(f, **{k: np.asarray(v) for k, v in cols.items()})
        n_req += len(cols["call_k"])
        print(f"[{i + 1}/{len(eps)}] {tag} {stem} {len(cols['call_k'])} req, {np.median(secs) * 1e3:.1f} ms/req", flush=True)
    rec.update({"finished": time.strftime("%F %T %Z"), "episodes": len(eps), "requests": n_req,
                "ms_per_request_median": float(np.median(secs) * 1e3) if secs else None})
    (out / "extract_record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
