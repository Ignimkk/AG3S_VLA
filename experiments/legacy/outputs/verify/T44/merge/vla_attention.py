"""T44b D-merge (verifier-owned): the real VLA attention for the 64 E2-demo frames, offline, one model.

Same cell and path as the live server (`trajopt/attention_policy.py` AttentionPolicy._attention_and_feature):
pi0.5 16D checkpoint loaded once with `return_attn_probs=True` (`pi05_attention.load_policy`), the policy's own
input transform on {state16, 3 policy images (CHW), episode prompt}, AttentionSampler denoise step 0,
aggregation `last`, layer 8, head 2 -> {cam_high, cam_left_wrist, cam_right_wrist: 16x16 raw softmax mass}.
Inputs: the policy images rendered by dump_frames.py (299x224 -> 224x224, reflections off = pi05_infer render_cam).
No server. usage: vla_attention.py --raw DIR --out F.npz
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, json, pathlib, sys, time
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"
CKPT = (_ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from AG3S.ag3s.experiments.sources.pi05_attention import AGGREGATIONS, AttentionSampler, load_policy
    from AG3S.ag3s.experiments.sources.policy_record import ATTENTION_GRID, CAMERA_BINDINGS, POLICY_CAMERA_NAMES
    from AG3S.trajopt import attention_policy as AP
    from openpi.models import model as _model

    t0 = time.time()
    policy = load_policy(CONFIG, CKPT)
    sampler = AttentionSampler(policy._model, num_steps=10)  # noqa: SLF001  (AttentionPolicy default)
    print(f"model loaded {time.time() - t0:.1f}s", flush=True)
    keys, maps, prompts = [], [], []
    for fj in sorted(pathlib.Path(args.raw).glob("demo_*.json")):
        rec = json.loads(fj.read_text())
        z = np.load(fj.with_suffix(".npz"))
        obs = {"state": np.asarray(rec["state16"], np.float64),
               "images": {pk: np.transpose(z[f"pol_{pk}"], (2, 0, 1)) for pk in POLICY_CAMERA_NAMES},
               "prompt": rec["prompt"]}
        inputs = policy._input_transform(dict(obs))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        mobs = _model.Observation.from_dict(inputs)
        blocks = sampler.attention(mobs, [AP._DENOISE_STEP], noise_seed=0)
        pooled = AGGREGATIONS[AP._AGG](blocks[AP._DENOISE_STEP])
        cell = np.asarray(pooled[AP._LAYER, AP._HEAD])
        maps.append(np.stack([cell[CAMERA_BINDINGS[c][2][0]:CAMERA_BINDINGS[c][2][1]].reshape(ATTENTION_GRID, ATTENTION_GRID)
                              for c in POLICY_CAMERA_NAMES]))
        keys.append(rec["key"]); prompts.append(rec["prompt"])
        print(rec["key"], f"{time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(args.out, attention=np.asarray(maps, np.float32), keys=np.asarray(keys),
                        cameras=np.asarray(POLICY_CAMERA_NAMES), prompts=np.asarray(prompts),
                        cell=np.asarray([AP._DENOISE_STEP, AP._LAYER, AP._HEAD]), agg=np.asarray(AP._AGG),
                        config=np.asarray(CONFIG), checkpoint=np.asarray(CKPT))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
