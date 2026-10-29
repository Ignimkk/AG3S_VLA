"""Extract the raster crops used by the draft F01-F03 CVPR-Figure specs.

Every crop comes from an existing recording or research figure; nothing is
generated. Inputs are read-only. Output: paper/figures/cvprfig_assets/*.png and
assets.json (source path, sha256, crop box per asset).

Usage (any Python with numpy + Pillow; ffmpeg on PATH):
    python paper/tools/plotting/cvprfig_assets.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
STORAGE = Path("/mnt/dev/AG3S_VLA-storage")
OUT = REPO / "paper/figures/cvprfig_assets"

ROLLOUT = STORAGE / "results/rby1/rollouts/ep1807_apple.mp4"
HEADFIX = STORAGE / "results/workspace/live_test/20260925_headfix16d/records/run_0000/run_0000"
T47_OVERLAY = REPO / "AG3S/ag3s/docs/figures/t47/t47-scene-overlay.png"
T47B_IDENTITY = REPO / "AG3S/ag3s/docs/figures/t47/t47b-scene-identity.png"
T31_SLICES = REPO / "AG3S/ag3s/docs/figures/t31/t31-d4-slices-E3b_ep1807_r1-t240.png"

# rollout frame index -> asset name (15 fps third-person render)
ROLLOUT_FRAMES = {0: "demo_start", 120: "demo_reach", 150: "demo_lift", 210: "demo_place"}
ROLLOUT_CROP = (150, 0, 550, 420)  # x0, y0, x1, y1 in 640x480

# crop boxes in original-pixel coordinates of the research figures
FIGURE_CROPS = {
    # T47 row 2 (headfix frame 5, t=40, approach), column 4 (L8H2, cam_left_wrist)
    "attn_left_wrist": (T47_OVERLAY, (722, 349, 917, 543)),
    # T47b bottom row, x-y top view: target (orange) merged with touching distractor (blue)
    "merged_cluster_topview": (T47B_IDENTITY, (950, 880, 1350, 1280)),
    # T31 E3b ep1807 t=240, fine 5 mm x-z slice plot area
    "esdf_fine_xz": (T31_SLICES, (86, 1250, 558, 1662)),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rollout_frame(index: int, dst: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "f.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(ROLLOUT),
             "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(raw)],
            check=True,
        )
        Image.open(raw).convert("RGB").crop(ROLLOUT_CROP).save(dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}

    for index, name in ROLLOUT_FRAMES.items():
        dst = OUT / f"{name}.png"
        rollout_frame(index, dst)
        meta[name] = {"source": str(ROLLOUT), "source_sha256": sha256(ROLLOUT),
                      "frame_index": index, "crop_xyxy": ROLLOUT_CROP,
                      "note": "third-person render; controller configuration of this rollout not registered"}

    step = HEADFIX / "step_00005.npz"
    rec = np.load(step, allow_pickle=False)
    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        dst = OUT / f"policy_{cam}.png"
        Image.fromarray(rec[f"image_{cam}"]).save(dst)
        meta[f"policy_{cam}"] = {"source": str(step), "source_sha256": sha256(step),
                                 "key": f"image_{cam}", "t_step": int(rec["t_step"]),
                                 "note": "224x224 policy input (headfix record, held-out, apple)"}

    for name, (src, box) in FIGURE_CROPS.items():
        dst = OUT / f"{name}.png"
        Image.open(src).convert("RGB").crop(box).save(dst)
        meta[name] = {"source": str(src.relative_to(REPO)), "source_sha256": sha256(src),
                      "crop_xyxy": box}

    for name, m in meta.items():
        m["sha256"] = sha256(OUT / f"{name}.png")
    (OUT / "assets.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(meta)} assets to {OUT}")


if __name__ == "__main__":
    main()
