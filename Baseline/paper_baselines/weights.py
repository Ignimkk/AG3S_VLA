"""Pinned model weights of the paper baselines: id, source revision, local path, sha256.

All weights live in this checkout's cache (``.cache/paper_baselines``, ``setup/paper_baselines_env.sh``), never in
shared storage or ``~``. Hugging Face models are pinned by **commit revision**; GitHub release assets by release tag
+ file name. Every file's sha256 is computed once and kept next to it (``<file>.sha256``) so a provider's
``provenance()`` can list it without re-hashing GBs each run.

``default`` = the large model of each family (measurement); ``smoke`` = the smallest (CPU wiring check).
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request

from Baseline.paper_baselines.interface import sha256_file

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = pathlib.Path(os.environ.get("PAPER_BASELINES_CACHE", PROJECT_ROOT / ".cache/paper_baselines"))
WEIGHTS_DIR = pathlib.Path(os.environ.get("PAPER_BASELINES_WEIGHTS", CACHE / "weights"))
#: HF hub cache used for every snapshot (explicit ``cache_dir``: independent of whatever HF_HOME the shell has).
HF_HUB_CACHE = CACHE / "huggingface" / "hub"

ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

#: name -> spec. ``hf``: repo + revision (+ files to hash); ``url``: direct file.
WEIGHTS: dict[str, dict] = {
    # Grounding DINO (HF transformers port of the official Swin-T / Swin-B OGC checkpoints)
    "grounding-dino-tiny": {"kind": "hf", "repo": "IDEA-Research/grounding-dino-tiny",
                            "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71", "files": ["model.safetensors"],
                            "license": "apache-2.0"},
    "grounding-dino-base": {"kind": "hf", "repo": "IDEA-Research/grounding-dino-base",
                            "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", "files": ["model.safetensors"],
                            "license": "apache-2.0"},
    # SAM v1 (HF port of the official ViT-B / ViT-H checkpoints)
    "sam-vit-base": {"kind": "hf", "repo": "facebook/sam-vit-base", "revision": "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
                     "files": ["model.safetensors"], "license": "apache-2.0"},
    "sam-vit-huge": {"kind": "hf", "repo": "facebook/sam-vit-huge", "revision": "87aecf0df4ce6b30cd7de76e87673c49644bdf67",
                     "files": ["model.safetensors"], "license": "apache-2.0"},
    # SAM 2.1 (official .pt checkpoints as published on the HF hub; config from the pinned Grounded-SAM-2 clone)
    "sam2.1-hiera-tiny": {"kind": "hf", "repo": "facebook/sam2.1-hiera-tiny", "revision": "de431c4043854a71d8101e17995dfe596bf101a5",
                          "files": ["sam2.1_hiera_tiny.pt"], "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                          "license": "apache-2.0"},
    "sam2.1-hiera-large": {"kind": "hf", "repo": "facebook/sam2.1-hiera-large", "revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
                           "files": ["sam2.1_hiera_large.pt"], "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
                           "license": "apache-2.0"},
    # YOLOE (ultralytics release assets; text encoder MobileCLIP-B(LT) TorchScript used by set_classes)
    "yoloe-11s-seg": {"kind": "url", "url": f"{ULTRALYTICS_RELEASE}/yoloe-11s-seg.pt", "file": "yoloe-11s-seg.pt",
                      "license": "AGPL-3.0"},
    "yoloe-11l-seg": {"kind": "url", "url": f"{ULTRALYTICS_RELEASE}/yoloe-11l-seg.pt", "file": "yoloe-11l-seg.pt",
                      "license": "AGPL-3.0"},
    "mobileclip_blt": {"kind": "url", "url": f"{ULTRALYTICS_RELEASE}/mobileclip_blt.ts", "file": "mobileclip_blt.ts",
                       "license": "Apple sample code license (ml-mobileclip); redistributed by ultralytics"},
}

PRESETS = {
    "grounded_sam": {"default": {"detector": "grounding-dino-base", "sam": "sam-vit-huge"},
                     "smoke": {"detector": "grounding-dino-tiny", "sam": "sam-vit-base"}},
    "grounded_sam2": {"default": {"detector": "grounding-dino-base", "sam2": "sam2.1-hiera-large"},
                      "smoke": {"detector": "grounding-dino-tiny", "sam2": "sam2.1-hiera-tiny"}},
    "yoloe": {"default": {"model": "yoloe-11l-seg"}, "smoke": {"model": "yoloe-11s-seg"}},
}


def _sha256_cached(path: pathlib.Path) -> str:
    side = path.with_name(path.name + ".sha256")
    st = path.stat()
    if side.exists():
        rec = json.loads(side.read_text())
        if rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
            return rec["sha256"]
    digest = sha256_file(path)
    try:
        side.write_text(json.dumps({"sha256": digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns}))
    except OSError:  # read-only cache (e.g. HF blob dir) -> just return
        pass
    return digest


def fetch(name: str, *, offline: bool = False) -> pathlib.Path:
    """Local path of weight ``name`` (HF: snapshot directory; url: the file). Downloads if missing."""
    spec = WEIGHTS[name]
    if spec["kind"] == "hf":
        from huggingface_hub import snapshot_download

        return pathlib.Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_files_only=offline,
                                              cache_dir=str(HF_HUB_CACHE),
                                              allow_patterns=["*.json", "*.txt"] + spec["files"]))
    target = WEIGHTS_DIR / spec["file"]
    if not target.exists():
        if offline:
            raise FileNotFoundError(target)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        urllib.request.urlretrieve(spec["url"], tmp)  # noqa: S310 (pinned https URL)
        tmp.replace(target)
    return target


def weight_file(name: str) -> pathlib.Path:
    """The hashed file of ``name`` (HF: first entry of ``files`` inside the snapshot)."""
    spec = WEIGHTS[name]
    root = fetch(name)
    return root / spec["files"][0] if spec["kind"] == "hf" else root


def provenance(name: str) -> dict:
    spec = WEIGHTS[name]
    f = weight_file(name)
    rec = {"name": name, "kind": spec["kind"], "license": spec.get("license"), "file": f.name,
           "sha256": _sha256_cached(f.resolve()), "bytes": f.resolve().stat().st_size}
    if spec["kind"] == "hf":
        rec.update(repo=spec["repo"], revision=spec["revision"])
    else:
        rec.update(url=spec["url"])
    return rec


def preset(provider: str, size: str) -> dict:
    if size not in ("default", "smoke"):
        raise ValueError(f"size must be 'default' or 'smoke', got {size!r}")
    return dict(PRESETS[provider][size])


__all__ = ["CACHE", "HF_HUB_CACHE", "PRESETS", "WEIGHTS", "WEIGHTS_DIR", "fetch", "preset", "provenance", "weight_file"]
