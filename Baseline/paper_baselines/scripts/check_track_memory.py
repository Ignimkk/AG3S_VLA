"""Real SAM 2.1 check of the bounded ``grounded_sam2:track`` memory on ONE recorded LIBERO episode (CPU or GPU).

The video predictor is prompted with the GT target mask of the first frame where the target is visible (no
detector: the check is about the streaming state, not about detection), then fed every later recorded frame in
time order -- the same ``CausalTrack`` calls the provider makes. Per frame it records the state footprint
(``state_footprint``: tensors reachable from the state, shared storages counted once), the process RSS and, on
CUDA, ``torch.cuda.memory_allocated``.

* ``bounded``   -- ``CausalTrack(bounded_memory=True)`` (the provider default), all ``--frames`` frames;
* ``unbounded`` -- the pinned code's state (``bounded_memory=False``) for the first ``--compare-frames`` frames;
  the masks of the two must be identical frame by frame (``masks_identical``).

Writes ``<out>/track_memory_<camera>.json``. Example (CPU, smallest weights)::

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python Baseline/paper_baselines/scripts/check_track_memory.py \\
        --weight sam2.1-hiera-tiny --device cpu --frames 120 --compare-frames 40 --out outputs/paper_baselines_smoke/G_track_memory
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np

from Baseline.paper_baselines import weights as W
from Baseline.paper_baselines.eval import offline_grounding as OG
from Baseline.paper_baselines.providers.grounded_sam2 import CausalTrack, read_window, state_footprint

S = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
SOURCE = S / "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
SCENE = pathlib.Path("outputs/verify/SL4/raw/scene")


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_track(vp, ep, camera: str, frames: list[int], bounded: bool, torch) -> dict:
    tr = CausalTrack(vp, bounded_memory=bounded)
    rec = {"bounded": bounded, "frames": [], "masks": []}
    for i, t in enumerate(frames):
        img = OG.L.opencv_rgb_from_raw(ep.raw[camera][t])
        t0 = time.perf_counter()
        if i == 0:
            gt = OG.seg_labels(ep.seg[camera][ep.row(t)], ep.lut) == OG.M.CLASSES.index("target")
            m = tr.start(img, gt)
        else:
            m = tr.step(img)
        dt = time.perf_counter() - t0
        fp = state_footprint(tr.state)
        row = {"obs_index": int(t), "track_frames": tr.n_frames, "state_tensors": fp["n_tensors"],
               "state_bytes": fp["bytes"], "rss_mb": round(rss_mb(), 1), "seconds": round(dt, 3),
               "mask_pixels": int(m.sum()),
               "non_cond_outputs": len(tr.state["output_dict"]["non_cond_frame_outputs"]),
               "cached_features": len(tr.state["cached_features"]),
               "images_frames": int(tr.state["images"].shape[0])}
        if torch.cuda.is_available():
            row["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
            row["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 2 ** 20, 1)
        rec["frames"].append(row)
        rec["masks"].append(np.packbits(m))
        print(f"[track_memory] {camera} bounded={bounded} obs {t}: {row}", flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weight", default="sam2.1-hiera-tiny", choices=["sam2.1-hiera-tiny", "sam2.1-hiera-large"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episode", default="libero_spatial_t0_e0")
    ap.add_argument("--cameras", nargs="+", default=["head"])
    ap.add_argument("--frames", type=int, default=120, help="frames fed to the bounded track")
    ap.add_argument("--compare-frames", type=int, default=40, help="frames also run unbounded (masks compared)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    torch.set_num_threads(args.threads)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = next(e for e in OG.load_episodes(SOURCE) if e["stem"] == args.episode)
    ep = OG.Episode(meta, OG.CAMERAS, SCENE.resolve())
    vp = build_sam2_video_predictor(W.WEIGHTS[args.weight]["config"], str(W.weight_file(args.weight)), device=args.device)
    for cam in args.cameras:
        n_rec = len(ep.raw[cam])
        have = set(int(x) for x in ep.obs_index)
        visible = [t for t in range(n_rec) if t in have and
                   (OG.seg_labels(ep.seg[cam][ep.row(t)], ep.lut) == OG.M.CLASSES.index("target")).any()]
        t_first = visible[0]
        frames = list(range(t_first, min(n_rec, t_first + args.frames)))
        bounded = run_track(vp, ep, cam, frames, True, torch)
        unbounded = run_track(vp, ep, cam, frames[: args.compare_frames], False, torch)
        n = len(unbounded["masks"])
        same = [bool(np.array_equal(a, b)) for a, b in zip(bounded["masks"][:n], unbounded["masks"])]
        rec = {"episode": args.episode, "camera": cam, "weight": args.weight, "weight_sha256": W.provenance(args.weight)["sha256"],
               "device": args.device, "threads": args.threads, "read_window": read_window(vp),
               "first_frame": t_first, "n_frames_bounded": len(frames), "n_frames_compared": n,
               "masks_identical": all(same), "n_masks_identical": int(sum(same)),
               "bounded": bounded["frames"], "unbounded": unbounded["frames"], "argv": sys.argv}
        (out / f"track_memory_{cam}.json").write_text(json.dumps(rec, indent=1) + "\n")
        b, u = bounded["frames"], unbounded["frames"]
        print(f"[track_memory] {cam}: masks identical {sum(same)}/{n}; bounded state bytes first/last "
              f"{b[0]['state_bytes']}/{b[-1]['state_bytes']} (max {max(r['state_bytes'] for r in b)}); "
              f"unbounded first/last {u[0]['state_bytes']}/{u[-1]['state_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
