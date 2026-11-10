#!/usr/bin/env python3
"""P5b GPU check (제약 단계 추종, 2026-11-10): the server's stage-track hidden pass equals the P3 features it was trained on.

The server runs `stage_track.build_stage_hidden_pass` on the **attention model** (`attention_policy.load_attention_model`,
`return_attn_probs=True`) with the serving policy's `_input_transform`; P3 extracted with `pi05_hidden.HiddenExtractor` on the
policy model. Same checkpoint, same prefix computation -> the h_10 rows should agree up to float16 storage. For the first
`--episodes` P2 episodes: max |server - P3| over every request, the probe's stage on both (agreement), and ms per request.

usage (GPU, shared lock, offline — never concurrently with a closed-loop run):
  source scripts/setup/env.sh
  python3 scripts/setup/gpu_lock.py shared --est-gib 24 --tag p5b-check -- \
      .venv-openpi-live/bin/python experiments/segmentation/stage_track/p5b_gpu_check.py outputs/2026-11-10_제약단계_추종
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import p3_extract_hidden as p3  # noqa: E402  (inserts the main checkout root; put this checkout first again)

sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    root = pathlib.Path(args.out_root)
    out = root / "p5b"
    out.mkdir(exist_ok=True)

    import jax

    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler, load_policy
    from AG3S.trajopt import stage_track as st
    from AG3S.trajopt.attention_policy import load_attention_model

    if jax.default_backend() != "gpu":
        raise SystemExit(f"expected a GPU backend, got {jax.devices()}")
    t0 = time.perf_counter()
    policy = load_policy(p3.CONFIG, p3.CHECKPOINT)
    attn = load_attention_model(p3.CONFIG, p3.CHECKPOINT)
    run_pass = st.build_stage_hidden_pass(AttentionSampler(attn, num_steps=10))
    probe = st.StageTrackProbe.load(st.default_asset_path())
    rec = {"started": time.strftime("%F %T %Z"), "load_s": round(time.perf_counter() - t0, 1), "module": st.__file__,
           "asset_md5": probe.md5, "episodes": []}
    eps = json.loads((root / "p2" / "p2_episodes.json").read_text())
    # one episode per group first (LIBERO, SafeLIBERO11, E1a ...), then the rest in order
    seen, order = set(), []
    for e in eps:
        if e["group"] not in seen:
            seen.add(e["group"])
            order.append(e)
    order += [e for e in eps if e not in order]
    secs, worst = [], 0.0
    for e in order[: args.episodes]:
        run = sorted(p3.RESULTS.glob(f"*_{e['tag']}{p3.VSUFFIX}"))[-1]
        F = np.load(root / "p3" / "features" / e["tag"] / f"{e['stem']}.npz")
        ref = F["hidden"][:, st.HIDDEN_LAYER, :3, :].reshape(len(F["obs_index"]), -1).astype(np.float32)
        prompt = p3.episodes_meta(run)[e["stem"]]["task_description"]
        got = []
        for k, t, req in p3.requests_of(run / "episodes" / f"{e['stem']}.npz", prompt):
            inputs = policy._input_transform(dict(req))  # noqa: SLF001
            s = time.perf_counter()
            got.append(run_pass(inputs))
            secs.append(time.perf_counter() - s)
        got = np.stack(got)
        d = float(np.abs(got - ref).max())
        rel = float(d / (np.abs(ref).max() + 1e-12))
        worst = max(worst, rel)
        grip = np.zeros((len(got), st.GRIP_DIM))     # stage agreement on the hidden part only (same GRIP on both sides)
        same = float((probe.proba(got, grip).argmax(1) == probe.proba(ref, grip).argmax(1)).mean())
        rec["episodes"].append({"tag": e["tag"], "stem": e["stem"], "requests": len(got), "max_abs": d, "max_rel": rel,
                                "stage_agreement_hidden_only": same})
        print(json.dumps(rec["episodes"][-1]), flush=True)
    rec.update({"max_rel_all": worst, "ms_per_request_median": float(np.median(secs[1:]) * 1e3) if len(secs) > 1 else None,
                "first_call_s": round(secs[0], 2) if secs else None, "finished": time.strftime("%F %T %Z")})
    (out / "p5b_gpu_check.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("max_rel_all", "ms_per_request_median", "first_call_s")}))


if __name__ == "__main__":
    main()
