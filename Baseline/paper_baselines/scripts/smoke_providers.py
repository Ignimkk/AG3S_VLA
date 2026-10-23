"""CPU wiring smoke of the paper-baseline providers on ONE recorded LIBERO frame (smallest weights; no numbers judged).

Writes ``outputs/paper_baselines_smoke/smoke_<provider>.json`` and ``smoke_<provider>.png`` (RGB + mask overlay +
GT outline, both cameras). ``grounded_sam2:track`` additionally runs a causality check on 3 consecutive frames:
the masks of frames t0, t0+1 must be identical whether the third frame is the recorded one or a different frame.

Run one provider per process (keeps RSS small)::

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python Baseline/paper_baselines/scripts/smoke_providers.py \\
        --provider grounded_sam   # | grounded_sam2:per_frame | grounded_sam2:track | yoloe | gt_oracle | tokencut | localization_heads:sam_point
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import time

import numpy as np

from Baseline.paper_baselines import lift as L
from Baseline.paper_baselines.eval import offline_grounding as OG
from Baseline.paper_baselines.eval.metrics import CLASSES, mask_metrics_2d
from Baseline.paper_baselines.interface import TargetQuery, _jsonable, get_provider, validate_results

S = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
SOURCE = S / "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
AUDIT = S / "20261012T202459_309e4c6da477_sl3-attention-audit-dev"
STEM = "libero_spatial_t0_e0"
OUT = pathlib.Path("outputs/paper_baselines_smoke")


def load_episode():
    ep = next(e for e in OG.load_episodes(SOURCE) if e["stem"] == STEM)
    return OG.Episode(ep, OG.CAMERAS, pathlib.Path("outputs/verify/SL4/raw/scene").resolve())


def make_provider(spec: str):
    name, _, mode = spec.partition(":")
    if name == "gt_oracle":
        return get_provider("gt_oracle")
    if name == "tokencut":
        return get_provider("tokencut")  # official DINO ViT-S/16 from TORCH_HOME/hub/checkpoints
    if name == "localization_heads":
        from Baseline.paper_baselines.providers.sam_backend import SamV1Backend

        sel = pathlib.Path("Baseline/paper_baselines/providers/frozen/localization_heads_pi05_libero_spatial_dev_init0-4_official_code.json")
        return get_provider("localization_heads", selection=sel, mask_mode=mode or "threshold",
                            sam=SamV1Backend("sam-vit-base", "cpu"))
    kw = {"size": "smoke", "device": "cpu"}
    if mode:
        kw["mode"] = mode
    return get_provider(name, **kw)


def aux_for(spec: str, ep, fr, t: int) -> dict:
    aux = {"segmentation": fr["seg"], "target_geom_ids": ep.target_geoms.tolist()}
    if spec.startswith("localization_heads"):
        from Baseline.paper_baselines.providers import localization_heads as lh

        row = next(json.loads(x) for x in (AUDIT / "attention_index.jsonl").read_text().splitlines() if STEM in x)
        with np.load(AUDIT / row["attn_npz"]) as z:
            z = {k: z[k] for k in z.files}
        k = int(np.flatnonzero(z["obs_index"] == t)[0])
        aux.update(lh.aux_from_attention_npz(z, k))
    return aux


def overlay_png(path, fr, res, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 4.4))
    for ax, (c, r) in zip(np.atleast_1d(axes), res.items()):
        img = fr["rgb"][c].astype(float) / 255
        over = img.copy()
        over[r.mask] = 0.45 * over[r.mask] + 0.55 * np.array([0.0, 0.45, 0.85])
        ax.imshow(over)
        gt = fr["labels"][c] == CLASSES.index("target")
        edge = gt & ~ndimage.binary_erosion(gt)
        ys, xs = np.nonzero(edge)
        ax.scatter(xs, ys, s=0.3, c="#e8590c", lw=0)
        if r.boxes is not None:
            for b in r.boxes:
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#2b8a3e", lw=0.8))
        ax.set_title(f"{c}: found={r.found} score={'-' if r.score is None else f'{r.score:.2f}'}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title + "\nblue = predicted mask, orange = GT target outline, green = candidate boxes (OpenCV frame)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--obs-index", type=int, default=10)
    ap.add_argument("--prompt-mode", default="phrase")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    ep = load_episode()
    t = args.obs_index
    fr = ep.frame(t)
    q = OG.query_for(ep, args.prompt_mode, {})
    t0 = time.time()
    prov = make_provider(args.provider)
    load_s = time.time() - t0
    rec = {"provider": args.provider, "source_run": SOURCE.name, "stem": STEM, "obs_index": t, "prompt": q.prompt,
           "query": {"instruction": q.instruction, "target_phrase": q.target_phrase, "prompt_mode": q.prompt_mode},
           "load_s": round(load_s, 1), "device": "cpu (CUDA_VISIBLE_DEVICES='')"}
    if args.provider == "grounded_sam2:track":
        frames = [t - 2, t - 1, t]
        alt = t + 30
        seqA, seqB = [], []
        prov.reset()
        for f in frames:
            fx = ep.frame(f)
            seqA.append(prov.predict(fx["rgb"], q, aux=aux_for(args.provider, ep, fx, f)))
        prov.reset()
        for f in frames[:-1] + [alt]:
            fx = ep.frame(f)
            seqB.append(prov.predict(fx["rgb"], q, aux=aux_for(args.provider, ep, fx, f)))
        same = {f"frame{frames[i]}/{c}": bool(np.array_equal(seqA[i][c].mask, seqB[i][c].mask))
                for i in range(len(frames) - 1) for c in OG.CAMERAS}
        rec["causality_check"] = {"seqA": frames, "seqB": frames[:-1] + [alt], "past_masks_identical": same,
                                  "last_frame_differs": {c: bool(not np.array_equal(seqA[-1][c].mask, seqB[-1][c].mask))
                                                         for c in OG.CAMERAS},
                                  "sources": [{c: r[c].extras.get("source") for c in OG.CAMERAS} for r in seqA]}
        res = seqA[-1]
    else:
        prov.reset()
        res = prov.predict(fr["rgb"], q, aux=aux_for(args.provider, ep, fr, t))
    validate_results(res, fr["rgb"])
    rec["results"] = {}
    for c, r in res.items():
        gt = fr["labels"][c] == CLASSES.index("target")
        pts = L.lift_mask(r.mask, fr["depth_m"][c], fr["K"][c], fr["T"][c])
        rec["results"][c] = {**r.summary(), "2d_vs_gt": mask_metrics_2d(r.mask, gt), "lifted_points": int(len(pts))}
    try:
        rec["provenance"] = prov.provenance()
    except Exception as exc:  # noqa: BLE001 -- report, do not hide
        rec["provenance_error"] = f"{type(exc).__name__}: {exc}"
    rec["maxrss_MB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    tag = args.provider.replace(":", "-")
    (OUT / f"smoke_{tag}.json").write_text(json.dumps(_jsonable(rec), indent=1, default=str) + "\n")
    overlay_png(OUT / f"smoke_{tag}.png", fr, res, f"{args.provider} | {STEM} obs {t} | prompt '{q.prompt}' | smoke weights, CPU")
    print(json.dumps({k: rec[k] for k in ("provider", "prompt", "load_s", "maxrss_MB")}),
          json.dumps({c: {k: v for k, v in x.items() if k in ("found", "score", "mask_pixels", "latency_ms", "lifted_points")}
                      for c, x in rec["results"].items()}), json.dumps(rec.get("causality_check", {})))


if __name__ == "__main__":
    main()
