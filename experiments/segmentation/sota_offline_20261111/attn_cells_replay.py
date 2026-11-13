#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) GPU, VLA only: re-read the pi0.5 attention of every pre-grasp policy call of the frozen version's
SafeLIBERO runs (the 184 episodes of attn_vs_target.py) for ALL [layer, head] cells, so another cell (e.g. KNOWS L12H3)
can be scored offline without a closed loop.

The request is rebuilt from the episode npz (official client preprocessing: flip both axes, resize_with_pad 224,
policy_state, task_description), the read-out is PandaAttentionReader (denoise step 0, aggregation `last`, noise seed 0
= the served V11AB path). Join check: the reproduced L8H2 head / wrist grid vs the grid the server recorded in the same
chunk (attention_head / attention_left_wrist, float16).

Writes OUT/<run_short>/<stem>.npz: pooled_f16 [n, L, H, 768], seq [n], chunk [n] (file name), l8h2_maxdiff [n, 2].
usage (GPU lock held by the caller):
  source /mnt/dev/AG3S_VLA/scripts/setup/env.sh; PYTHONPATH=$WT /mnt/dev/AG3S_VLA/.venv-openpi-live/bin/python \
      attn_cells_replay.py OUT [--limit-episodes N] [--budget-s S]
"""
import glob, json, os, sys, time
import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
PATTERNS = ("2*_sub-pkgFrssdm-e1a-spatial-LI-t*", "2*_ms-pkgFrssdm-*", "2*_ms-tab10sdm-*-i19-*")  # same set; spatial L-I first (resume order)
CKPT = "/mnt/dev/AG3S_VLA-storage/checkpoints/panda/pi05_libero"


def eps_of(D):
    ch = sorted(glob.glob(D + "/server_constraints/run_0000/chunk_*.npz")); eps = []; cur = []; last = -1
    for f in ch:
        ts = int(np.load(f, allow_pickle=True)["t_step"])
        if ts <= last and cur:
            eps.append(cur); cur = []
        cur.append(f); last = ts
    if cur:
        eps.append(cur)
    return eps


def client_images(raw_agentview, raw_wrist, resize=224):
    from openpi_client import image_tools
    out = []
    for raw in (raw_agentview, raw_wrist):
        img = np.ascontiguousarray(raw[::-1, ::-1])
        out.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize)))
    return out


def jobs():
    for pat in PATTERNS:
        for D in sorted(glob.glob(B + pat)):
            if not os.path.isfile(D + "/episodes.jsonl") or os.path.getsize(D + "/episodes.jsonl") == 0:
                continue
            E = [json.loads(l) for l in open(D + "/episodes.jsonl") if l.strip()]
            chunks = eps_of(D)
            if len(chunks) != len(E):
                print("skip (episode/chunk count)", os.path.basename(D)[:60], flush=True); continue
            for e, files in zip(E, chunks):
                yield D, e, files


def main():
    out = sys.argv[1]
    lim = int(sys.argv[sys.argv.index("--limit-episodes") + 1]) if "--limit-episodes" in sys.argv else None
    budget = float(sys.argv[sys.argv.index("--budget-s") + 1]) if "--budget-s" in sys.argv else None
    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy
    from AG3S.adapters.panda import attention as pa
    t0 = time.monotonic()
    policy = load_policy("pi05_libero", CKPT)
    reader = pa.PandaAttentionReader(policy, num_denoise_steps=10, noise_seed=0)
    print("loaded", round(time.monotonic() - t0, 1), "s", flush=True)
    n_ep = n_call = 0; worst = 0.0
    for D, e, files in jobs():
        if lim is not None and n_ep >= lim:
            break
        if budget is not None and time.monotonic() - t0 > budget:
            print("budget reached; rerun resumes (finished episodes are skipped)", flush=True); break
        t, i = int(e["task_id"]), int(e["init_state_index"]); suite = e["suite"]
        stem = f"{suite}_t{t}_e{i}"
        dst = os.path.join(out, os.path.basename(D)[:15] + "_" + os.path.basename(D).split("_", 2)[2][:40], stem + ".npz")
        if os.path.exists(dst):
            n_ep += 1; continue
        cf = glob.glob(D + f"/episodes/*_t{t}_e{i}_calls.jsonl")[0]; cs = [json.loads(x) for x in open(cf)]
        ep = np.load(cf.replace("_calls.jsonl", ".npz"))
        chunk_t = [int(x) for x in ep["chunk_t"]]
        rows = {"pooled_f16": [], "seq": [], "chunk": [], "l8h2_maxdiff": []}
        for f in files:
            z = np.load(f, allow_pickle=True); sq = int(z["t_step"]); sm = json.loads(str(z["summary_json"]))
            if ((sm.get("grasp") or {}).get("state")) in ("held", "closing"):
                break
            if sq >= len(cs) or cs[sq]["obs_index"] not in chunk_t:
                continue
            k = chunk_t.index(cs[sq]["obs_index"]); ob = chunk_t[k]
            img, wrist = client_images(ep["obs_agentview_image"][ob], ep["obs_robot0_eye_in_hand_image"][ob])
            req = {"observation/image": img, "observation/wrist_image": wrist,
                   "observation/state": np.asarray(ep["policy_state"][k], np.float64), "prompt": e["task_description"]}
            res = reader.read(req, full=True, with_feature=False)
            # recorded grid orientation is not asserted here: the smaller of policy / un-mirrored orientation
            d = [min(float(np.max(np.abs(g - np.asarray(z[key], np.float32))))
                     for g in (res["grids"][c].astype(np.float32), pa.unmirror_grid(res["grids"][c].astype(np.float32))))
                 if key in z.files else np.nan
                 for c, key in (("head", "attention_head"), ("left_wrist", "attention_left_wrist"))]
            worst = max(worst, max(x for x in d if x == x) if any(x == x for x in d) else 0.0)
            rows["pooled_f16"].append(res["pooled"].astype(np.float16)); rows["seq"].append(sq)
            rows["chunk"].append(os.path.basename(f)); rows["l8h2_maxdiff"].append(d)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if rows["seq"]:
            np.savez_compressed(dst, pooled_f16=np.stack(rows["pooled_f16"]), seq=np.array(rows["seq"]),
                                chunk=np.array(rows["chunk"]), l8h2_maxdiff=np.array(rows["l8h2_maxdiff"], np.float32))
        else:
            np.savez_compressed(dst, seq=np.zeros(0, int))
        n_ep += 1; n_call += len(rows["seq"])
        print(f"{n_ep} {stem} calls {len(rows['seq'])} l8h2_maxdiff_worst {worst:.2e} t {time.monotonic() - t0:.0f}s",
              flush=True)
    print(json.dumps({"episodes": n_ep, "calls": n_call, "l8h2_worst_maxdiff": worst,
                      "seconds": time.monotonic() - t0}), flush=True)


if __name__ == "__main__":
    main()
