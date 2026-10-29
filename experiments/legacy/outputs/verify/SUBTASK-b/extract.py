"""SUBTASK-b feature extraction: hidden states of the 16D checkpoint (29999, LoRA on), read-only.

Per frame, two conditions:
  normal : state as recorded
  masked : state[7] = state[15] = 1.0 (open gripper constant) BEFORE the policy input transform,
           so the discretised State string in the prompt carries no gripper information.

Features (all computed inside jit, reduced on device):
  prefix_last  Gemma final (normed) hidden at the last valid prompt token          (2048)
  prefix_img   Gemma final (normed) hidden, mean over the 768 image tokens         (2048)
  kv_L{l}      prefix KV-cache V at layer l, [mean over valid text | mean over image] (256+256)
  ae_final     action-expert final hidden (suffix_out), mean over action tokens,
               denoise step 0 (time=1.0), x_t = N(0,1) with jax.random.key(0)       (1024)
  attn_L8      layer-8 attention probs of the LAST action token (step 0), 8 heads x
               [cam_high, cam_left_wrist, cam_right_wrist, text] mass               (32)

Usage: extract.py --set {closed,val,train} [--ep-stride N] [--frame-stride 6]
Per-episode npz under feat/<set>/, existing files skipped (resumable).
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import dataclasses
import glob
import json
import pathlib
import sys
import time

import av
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, _ag3s_workspace_path(''))

from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config as _pc  # noqa: E402
from openpi.training import config as _config  # noqa: E402

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-b'))
CKPT = (_ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'))
CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"
DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'))
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
KV_LAYERS = (0, 4, 8, 12, 17)
ATTN_LAYER = 8
N_IMG = 768
RUNS = {
    "ep1807_apple": _ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000'),
    "ep0_orange": _ag3s_workspace_path('outputs/live_test/20260925_train/ep0/run_0000'),
    "ep500_banana": _ag3s_workspace_path('outputs/live_test/20260925_train/ep500/run_0000'),
    "ep1200_pear": _ag3s_workspace_path('outputs/live_test/20260925_train/ep1200/run_0000'),
}
FRUIT_QPOS = {"apple": 38, "banana": 45, "orange": 52, "pear": 59}


def closed_loop_gt(d, fruit):
    """Same rule as outputs/verify/subtask_probe/subtask_episode.py:ground_truth()."""
    q, s = d["qpos"], d["state"]
    a = FRUIT_QPOS[fruit]
    obj, crate = q[a:a + 3], q[31:34]
    closed = bool(min(s[7], s[15]) < 0.9)
    in_crate = bool(np.linalg.norm(obj[:2] - crate[:2]) < 0.15 and obj[2] < 1.0)
    return "held" if closed else ("released" if in_crate else "reach")


class Extractor:
    def __init__(self):
        cfg = _config.get_config(CONFIG)
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, return_attn_probs=True))
        self.policy = _pc.create_trained_policy(cfg, CKPT)
        model = self.policy._model  # noqa: SLF001
        self.ah, self.ad = int(model.action_horizon), int(model.action_dim)
        self.graphdef, self.state = nnx.split(model)
        self._pre = jax.jit(self._prefix)
        self._suf = jax.jit(self._suffix)
        self.x0 = jax.random.normal(jax.random.key(0), (1, self.ah, self.ad), jnp.float32)
        self.t_pre, self.t_suf = [], []

    def _prefix(self, state, obs):
        model = nnx.merge(self.graphdef, state)
        tokens, mask, ar = model.embed_prefix(obs)
        attn = make_attn_mask(mask, ar)
        pos = jnp.cumsum(mask, axis=1) - 1
        res = model.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
        (out, _), kv = res[0], res[1]
        h = out[0].astype(jnp.float32)  # (S, 2048)
        m = mask[0]
        last = jnp.sum(m) - 1  # images all valid, text mask contiguous
        prefix_last = h[last]
        prefix_img = h[:N_IMG].mean(0)
        tm = m[N_IMG:].astype(jnp.float32)
        v = kv[1]  # (L, B, S, K, H)
        kvf = {}
        for l in KV_LAYERS:
            vl = v[l, 0, :, 0, :].astype(jnp.float32)  # (S, 256)
            txt = (vl[N_IMG:] * tm[:, None]).sum(0) / tm.sum()
            img = vl[:N_IMG].mean(0)
            kvf[f"kv_L{l}"] = jnp.concatenate([txt, img])
        return kv, mask, dict(prefix_last=prefix_last, prefix_img=prefix_img, **kvf)

    def _suffix(self, state, obs, x_t, kv, prefix_mask):
        import einops
        model = nnx.merge(self.graphdef, state)
        time_ = jnp.ones((1,), jnp.float32)
        st, sm, sar, cond = model.embed_suffix(obs, x_t, time_)
        sam = make_attn_mask(sm, sar)
        pre = einops.repeat(prefix_mask, "b p -> b s p", s=st.shape[1])
        full = jnp.concatenate([pre, sam], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
        (_, so), _, probs = model.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv,
                                                adarms_cond=[None, cond])
        ae = so[0, -self.ah:].astype(jnp.float32).mean(0)
        p = probs[ATTN_LAYER, 0, 0, :, -1, :].astype(jnp.float32)  # (heads, S_total), last action token
        n_pre = prefix_mask.shape[1]
        cams = [p[:, i * 256:(i + 1) * 256].sum(-1) for i in range(3)]
        txt = p[:, N_IMG:n_pre].sum(-1)
        attn = jnp.stack(cams + [txt], -1).reshape(-1)  # head-major: h0[c0,c1,c2,txt], h1[...]
        return dict(ae_final=ae, attn_L8=attn)

    def obs(self, state16, images_hwc, prompt):
        images = {k: np.transpose(images_hwc[k], (2, 0, 1)) for k in CAMS}
        inputs = self.policy._input_transform({"state": state16, "images": images, "prompt": prompt})  # noqa: SLF001
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        return _model.preprocess_observation(None, _model.Observation.from_dict(batched), train=False)

    def features(self, state16, images_hwc, prompt):
        obs = self.obs(np.asarray(state16, np.float32), images_hwc, prompt)
        t0 = time.perf_counter()
        kv, pm, f1 = self._pre(self.state, obs)
        jax.block_until_ready(f1)
        t1 = time.perf_counter()
        f2 = self._suf(self.state, obs, self.x0, kv, pm)
        jax.block_until_ready(f2)
        t2 = time.perf_counter()
        self.t_pre.append(t1 - t0)
        self.t_suf.append(t2 - t1)
        return {k: np.asarray(v) for k, v in {**f1, **f2}.items()}

    def both(self, state16, images_hwc, prompt):
        s = np.asarray(state16, np.float64).copy()
        normal = self.features(s, images_hwc, prompt)
        sm = s.copy()
        sm[7] = 1.0
        sm[15] = 1.0
        masked = self.features(sm, images_hwc, prompt)
        return normal, masked


def decode(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def save_episode(path, recs, meta):
    keys = recs[0]["normal"].keys()
    arrs = {}
    for cond in ("normal", "masked"):
        for k in keys:
            arrs[f"{cond}__{k}"] = np.stack([r[cond][k] for r in recs]).astype(np.float32)
    for k in recs[0]["info"]:
        arrs[k] = np.asarray([r["info"][k] for r in recs])
    for k, v in meta.items():
        arrs[k] = np.asarray(v)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    tmp.rename(path)


def run_dataset(ex, split, ep_stride, frame_stride):
    lo, hi = (0, 1600) if split == "train" else (1600, 1800)
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(DS / "meta/tasks.jsonl")}
    d = OUT / "feat" / split
    d.mkdir(parents=True, exist_ok=True)
    eps = list(range(lo, hi, ep_stride))
    t0 = time.time()
    for n, ep in enumerate(eps):
        path = d / f"ep_{ep:06d}.npz"
        if path.exists():
            continue
        ch = ep // 1000
        df = pd.read_parquet(DS / f"data/chunk-{ch:03d}/episode_{ep:06d}.parquet")
        vids = {c: decode(DS / f"videos/chunk-{ch:03d}/observation.images.{c}/episode_{ep:06d}.mp4") for c in CAMS}
        nvid = min(len(v) for v in vids.values())
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        fidx = df["frame_index"].to_numpy()
        phase = df["phase_index"].to_numpy()
        prompt = tasks[int(df["task_index"].iloc[0])]
        recs = []
        for i in range(len(df)):
            fi = int(fidx[i])
            if fi % frame_stride:
                continue
            if fi >= nvid:
                print(f"  ep{ep}: frame {fi} beyond video length {nvid}", flush=True)
                continue
            imgs = {c: vids[c][fi] for c in CAMS}
            normal, masked = ex.both(states[i], imgs, prompt)
            # rule history: closed at any frame <= this one (full parquet, every frame)
            closed_hist = bool((np.minimum(states[: i + 1, 7], states[: i + 1, 15]) < 0.9).any())
            recs.append(dict(normal=normal, masked=masked, info=dict(
                frame_index=fi, phase_index=int(phase[i]), state=states[i], ever_closed=closed_hist)))
        save_episode(path, recs, dict(episode=ep, prompt=prompt, n_parquet=len(df), n_video=nvid))
        if n % 5 == 0:
            el = time.time() - t0
            print(f"[{split} {n + 1}/{len(eps)} ep{ep}] frames={len(recs)} ({el:.0f}s) "
                  f"pre_med={np.median(ex.t_pre[-50:]) * 1e3:.1f}ms suf_med={np.median(ex.t_suf[-50:]) * 1e3:.1f}ms",
                  flush=True)


def run_closed(ex):
    d = OUT / "feat" / "closed"
    d.mkdir(parents=True, exist_ok=True)
    for rk, run in RUNS.items():
        path = d / f"{rk}.npz"
        if path.exists():
            continue
        prompt = json.load(open(f"{run}/meta.json"))["prompt"]
        fruit = next(f for f in FRUIT_QPOS if f in prompt.lower())
        files = sorted(glob.glob(f"{run}/step_*.npz"))
        recs = []
        ever = False
        for j, f in enumerate(files):
            z = np.load(f)
            s = z["state"].astype(np.float64)
            ever = ever or bool(min(s[7], s[15]) < 0.9)
            imgs = {c: z[f"image_{c}"] for c in CAMS}
            normal, masked = ex.both(s, imgs, prompt)
            recs.append(dict(normal=normal, masked=masked, info=dict(
                frame_index=int(z["t_step"]), file_index=j, gt=closed_loop_gt(z, fruit), state=s,
                ever_closed=ever, stride2=(j % 2 == 0) if len(files) > 80 else True)))
        save_episode(path, recs, dict(episode=rk, prompt=prompt, run=run))
        print(f"[closed {rk}] frames={len(recs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["closed", "val", "train"])
    ap.add_argument("--ep-stride", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    ex = Extractor()
    # warmup / JIT on a closed-loop frame; excluded from timing
    z = np.load(sorted(glob.glob(f"{RUNS['ep1807_apple']}/step_*.npz"))[0])
    for _ in range(args.warmup):
        ex.both(z["state"], {c: z[f"image_{c}"] for c in CAMS}, "warmup")
    ex.t_pre.clear()
    ex.t_suf.clear()
    t0 = time.time()
    if args.set == "closed":
        run_closed(ex)
    else:
        stride = args.ep_stride or (8 if args.set == "train" else 4)
        run_dataset(ex, args.set, stride, args.frame_stride)
    tp, ts = np.asarray(ex.t_pre) * 1e3, np.asarray(ex.t_suf) * 1e3
    timing = dict(set=args.set, n_forward=len(tp), wall_s=time.time() - t0,
                  prefix_ms_median=float(np.median(tp)) if len(tp) else None,
                  prefix_ms_p90=float(np.percentile(tp, 90)) if len(tp) else None,
                  suffix_ms_median=float(np.median(ts)) if len(ts) else None,
                  suffix_ms_p90=float(np.percentile(ts, 90)) if len(ts) else None,
                  note="per forward (one condition), batch 1, H200 bf16, after warmup JIT; "
                       "includes on-device feature reduction; excludes input transform & video decode")
    json.dump(timing, open(OUT / f"timing_{args.set}.json", "w"), indent=1)
    np.save(OUT / f"timing_{args.set}_raw.npy", np.stack([tp, ts]) if len(tp) else np.zeros((2, 0)))
    print("timing", timing, flush=True)


if __name__ == "__main__":
    main()
