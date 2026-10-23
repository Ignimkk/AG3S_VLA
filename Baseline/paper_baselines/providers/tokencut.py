"""TokenCut (Wang et al., CVPR 2022) normalized cut -- re-implementation + parity with the pinned official code.

Paper: Y. Wang, X. Shen, S. X. Hu, Y. Yuan, J. L. Crowley, D. Vaufreydaz, "Self-supervised transformers for
unsupervised object discovery using normalized cut", CVPR 2022 (paper ref. [26]). Code:
YangtaoWANG95/TokenCut @ ``fed52cd5b60891baefd8ec7110dafa73be816ee1`` (MIT), fetched by
``external/fetch_sources.py`` into ``external/TokenCut``. :func:`ncut` restates ``object_discovery.py:12-60``
(``ncut``) and ``:62-92`` (``detect_box``) in numpy; ``tests/test_tokencut.py`` runs the official function on
the same random features and requires the same mask, seed and eigenvector.

Algorithm (official ``object_discovery.ncut``, defaults of ``main_tokencut.py``)
-------------------------------------------------------------------------------
1. patch features (CLS token dropped), L2-normalised per token; ``A = F F^T`` (cosine affinity)
2. binary graph: ``A > tau`` -> 1, else ``eps`` (``--tau 0.2``, ``--eps 1e-5``); ``no_binary_graph`` keeps the
   similarity and sets ``A < tau`` to ``eps`` (ablation)
3. ``D = diag(A 1)``; generalized eigenproblem ``(D - A) y = lambda D y``; second smallest eigenvector
   (``scipy.linalg.eigh(D - A, D, subset_by_index=[1, 2])[:, 0]``)
4. bipartition ``y > mean(y)``; seed ``= argmax |y|``; the side containing the seed is foreground
5. mask = the **connected component (4-connected, ``ndimage.label`` default) of the foreground that contains
   the seed** -- note: not "the largest component"; the official code keeps the seed's component
6. upsample the patch mask to pixels with nearest neighbour (``F.interpolate(mode='nearest')``)

Features: ``feature_source="dino_vits16"`` = official default (``--arch vit_small --patch_size 16
--which_features k``): keys of the last block's qkv of DINO ViT-S/16 (``dino_deitsmall16_pretrain.pth``),
ImageNet normalisation, image zero-padded to a multiple of 16 (``main_tokencut.py:185-203``). The model code is
loaded from the pinned clone (``external/TokenCut/dino/vision_transformer.py``). ``feature_source="aux"`` takes
``aux["patch_features"][camera]`` = ``(rows, cols, D)`` patch features (e.g. pi0.5 SigLIP patch tokens) with
``aux["patch_features_orientation"]`` in ``{"image", "libero_policy"}``. No recorded producer of per-patch pi0.5
features exists yet (the attention audit stores a pooled 512-d ``kv_feature`` only).

Target selection -- **TokenCut uses no language.**
* ``target_select="foreground"`` (default): the official foreground component. The target phrase is ignored.
* ``target_select="attention_overlap"``: **not part of TokenCut** -- our grounding variant. Every connected
  component of both bipartition sides is a candidate; the one with the largest above-mean attention mass
  ``sum relu(att - mean(att))`` (``aux["attention_map"][camera]``, any resolution, resampled bilinearly to the
  RGB size) is the target. Ties go to the official seed component. A constant / missing attention map gives
  ``found=False``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage
from scipy.linalg import eigh

SOURCE = {
    "paper": "Wang et al., Self-supervised transformers for unsupervised object discovery using normalized cut, "
             "CVPR 2022",
    "repo": "https://github.com/YangtaoWANG95/TokenCut",
    "sha": "fed52cd5b60891baefd8ec7110dafa73be816ee1",
    "files": ["object_discovery.py (ncut, detect_box)", "main_tokencut.py (defaults, feature extraction)",
              "networks.py (DINO weights URL)", "dino/vision_transformer.py (loaded, not copied)"],
    "license": "MIT",
    "implementation": "re-impl of ncut (numpy) + parity test against the pinned official function",
}
DINO_WEIGHTS = {
    "vit_small16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external" / "TokenCut"
TARGET_SELECT = ("foreground", "attention_overlap")
FEATURE_SOURCES = ("dino_vits16", "aux")


@dataclasses.dataclass(frozen=True)
class TokenCutConfig:
    tau: float = 0.2              # main_tokencut.py --tau
    eps: float = 1e-5             # main_tokencut.py --eps
    binary_graph: bool = True     # main_tokencut.py --no-binary-graph (default False -> binary graph)
    patch_size: int = 16          # --patch_size
    which_features: str = "k"     # --which_features


@dataclasses.dataclass
class NcutResult:
    mask: np.ndarray          # bool (rows, cols): the seed's connected component of the foreground side
    bipartition: np.ndarray   # bool (rows, cols): foreground side (sign fixed so the seed is in it)
    seed: int                 # flat patch index, argmax |eigvec|
    eigvec: np.ndarray        # (rows, cols) second smallest generalized eigenvector, sign-fixed like official
    labels: np.ndarray        # int (rows, cols) components of the foreground side (ndimage.label default)


# ----------------------------------------------------------------------------------------------------------
# ncut (object_discovery.py:12-92)
# ----------------------------------------------------------------------------------------------------------
def ncut(patch_feats: np.ndarray, dims: tuple[int, int], *, tau: float = 0.2, eps: float = 1e-5,
         binary_graph: bool = True) -> NcutResult:
    """``patch_feats``: ``(rows*cols, D)`` patch tokens (no CLS), row-major over ``dims = (rows, cols)``."""
    f = np.asarray(patch_feats, dtype=np.float32)
    rows, cols = int(dims[0]), int(dims[1])
    if f.ndim != 2 or f.shape[0] != rows * cols:
        raise ValueError(f"patch_feats {f.shape} does not match dims {dims}")
    if not np.all(np.isfinite(f)):
        raise ValueError("non-finite features")
    norm = np.sqrt((f * f).sum(axis=1, keepdims=True, dtype=np.float32))
    f = f / np.maximum(norm, np.float32(1e-12))           # F.normalize(p=2, dim=1, eps=1e-12)
    A = f @ f.T                                          # float32, like the torch matmul
    if binary_graph:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    else:
        A = A.astype(np.float32)
        A[A < tau] = eps
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second = eigenvectors[:, 0]
    avg = np.sum(second) / len(second)
    bipartition = second > avg
    seed = int(np.argmax(np.abs(second)))
    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bip = bipartition.reshape(rows, cols)
    labels, _ = ndimage.label(bip.astype(float))
    cc = labels[np.unravel_index(seed, (rows, cols))]
    return NcutResult(mask=labels == cc, bipartition=bip, seed=seed, eigvec=eigenvec.reshape(rows, cols),
                      labels=labels)


# ----------------------------------------------------------------------------------------------------------
# resampling helpers
# ----------------------------------------------------------------------------------------------------------
def nearest_resize(grid: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """``F.interpolate(mode='nearest')``: ``src = floor(dst * in / out)``."""
    g = np.asarray(grid)
    ri = np.minimum((np.arange(out_hw[0]) * (g.shape[0] / out_hw[0])).astype(np.int64), g.shape[0] - 1)
    ci = np.minimum((np.arange(out_hw[1]) * (g.shape[1] / out_hw[1])).astype(np.int64), g.shape[1] - 1)
    return g[ri][:, ci]


def patch_mask_to_pixels(mask: np.ndarray, out_hw: tuple[int, int], patch_size: int | None) -> np.ndarray:
    """DINO path: repeat each patch ``patch_size`` times (the padded image) and crop to ``out_hw``; aux path
    (``patch_size=None``): nearest resize to ``out_hw``."""
    if patch_size:
        up = np.repeat(np.repeat(np.asarray(mask), patch_size, axis=0), patch_size, axis=1)
        return up[: out_hw[0], : out_hw[1]]
    return nearest_resize(mask, out_hw)


def _bilinear(grid: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    from Baseline.paper_baselines.providers.localization_heads import upscale_bilinear

    return upscale_bilinear(grid, out_hw)


def select_attention_overlap(result: NcutResult, attention_px: np.ndarray, out_hw: tuple[int, int],
                             patch_size: int | None) -> tuple[np.ndarray | None, dict]:
    """**Not TokenCut.** Candidate = every 4-connected component of either bipartition side; pick the one with
    the largest ``sum relu(att - mean(att))`` at pixel resolution. Returns ``(patch mask or None, info)``."""
    att = np.asarray(attention_px, dtype=np.float32)
    if not np.all(np.isfinite(att)):
        return None, {"not_found_reason": "non-finite attention map"}
    w = np.maximum(att - att.mean(), 0.0)
    if float(w.sum()) <= 0.0:
        return None, {"not_found_reason": "constant attention map"}
    fg_labels = result.labels
    bg_labels, _ = ndimage.label(np.logical_not(result.bipartition).astype(float))
    seed_rc = np.unravel_index(result.seed, result.bipartition.shape)
    candidates = []
    for side, labels in (("foreground", fg_labels), ("background", bg_labels)):
        for c in range(1, int(labels.max()) + 1):
            comp = labels == c
            px = patch_mask_to_pixels(comp, out_hw, patch_size)
            candidates.append({"side": side, "label": c, "patches": int(comp.sum()),
                               "overlap_mass": float(w[px].sum()),
                               "is_seed_component": bool(side == "foreground" and labels[seed_rc] == c),
                               "_mask": comp})
    total = float(w.sum())
    best = max(candidates, key=lambda r: (r["overlap_mass"], r["is_seed_component"]))
    info = {"candidates": [{k: v for k, v in r.items() if k != "_mask"} | {"overlap_frac": r["overlap_mass"] / total}
                           for r in candidates],
            "chosen": {k: v for k, v in best.items() if k != "_mask"}, "overlap_frac": best["overlap_mass"] / total}
    return best["_mask"], info


# ----------------------------------------------------------------------------------------------------------
# DINO ViT-S/16 features (main_tokencut.py:185-266)
# ----------------------------------------------------------------------------------------------------------
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_module(relpath: str, name: str, external_dir: Path | None = None):
    """Import a file of the pinned TokenCut clone under a private module name (path scoped to the import)."""
    root = Path(external_dir or os.environ.get("PAPER_BASELINES_TOKENCUT_DIR", EXTERNAL_DIR))
    path = root / relpath
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run Baseline/paper_baselines/external/fetch_sources.py "
                                "--only TokenCut")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    added = [str(path.parent)] if str(path.parent) not in sys.path else []
    sys.path[:0] = added
    had_utils = "utils" in sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        for p in added:
            sys.path.remove(p)
        if not had_utils and "utils" in sys.modules and \
                str(getattr(sys.modules["utils"], "__file__", "")).startswith(str(root)):
            del sys.modules["utils"]  # dino/vision_transformer.py does `from utils import trunc_normal_`
    return module


class DinoKeyExtractor:
    """Last-block keys of DINO ViT-S/16 as ``main_tokencut.py`` computes them (CPU by default)."""

    def __init__(self, *, weights_path: str | None = None, device: str = "cpu", external_dir: Path | None = None,
                 patch_size: int = 16, which_features: str = "k"):
        import torch

        vits = load_official_module("dino/vision_transformer.py", "_tokencut_dino_vits", external_dir)
        self.model = vits.vit_small(patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        url = DINO_WEIGHTS["vit_small16"]
        if weights_path is None:
            hub = Path(torch.hub.get_dir()) / "checkpoints"
            weights_path = str(hub / Path(url).name)
            if not Path(weights_path).exists():
                torch.hub.load_state_dict_from_url(url, model_dir=str(hub), map_location="cpu")
        state = torch.load(weights_path, map_location="cpu")
        msg = self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        self.device = device
        self.patch_size = patch_size
        self.which = which_features
        self.weights = {"url": url, "path": str(weights_path), "sha256": sha256_file(weights_path),
                        "load_msg": str(msg)}
        self._feat = {}
        self.model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
            lambda m, i, o: self._feat.__setitem__("qkv", o))

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        import torch

        img = torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = (img - mean) / std
        ps = self.patch_size
        size_im = (3, int(math.ceil(img.shape[1] / ps) * ps), int(math.ceil(img.shape[2] / ps) * ps))
        padded = torch.zeros(size_im)
        padded[:, : img.shape[1], : img.shape[2]] = img
        dims = (size_im[1] // ps, size_im[2] // ps)
        with torch.no_grad():
            attentions = self.model.get_last_selfattention(padded[None].to(self.device))
            nb_im, nh, nb_tokens = attentions.shape[0], attentions.shape[1], attentions.shape[2]
            qkv = self._feat["qkv"].reshape(nb_im, nb_tokens, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
            feats = {"q": qkv[0], "k": qkv[1], "v": qkv[2]}[self.which]
            feats = feats.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        return feats[0, 1:].cpu().numpy().astype(np.float32), dims


# ----------------------------------------------------------------------------------------------------------
# provider
# ----------------------------------------------------------------------------------------------------------
def _interface():
    from Baseline.paper_baselines import interface  # Implementer A

    return interface


def _to_image_orientation(grid: np.ndarray, orientation: str) -> np.ndarray:
    from Baseline.paper_baselines.providers.localization_heads import to_image_orientation

    return to_image_orientation(grid, orientation)


class TokenCutProvider:
    """TokenCut target mask (``TargetMaskProvider``). See the module docstring for the two target rules."""

    def __init__(self, *, feature_source: str = "dino_vits16", target_select: str = "foreground",
                 cfg: TokenCutConfig = TokenCutConfig(), device: str = "cpu", weights_path: str | None = None,
                 external_dir: Path | None = None, extractor: Any = None):
        if feature_source not in FEATURE_SOURCES:
            raise ValueError(f"feature_source must be one of {FEATURE_SOURCES}")
        if target_select not in TARGET_SELECT:
            raise ValueError(f"target_select must be one of {TARGET_SELECT}")
        self.feature_source = feature_source
        self.target_select = target_select
        self.cfg = cfg
        self.device = device
        self.weights_path = weights_path
        self.external_dir = external_dir
        self._extractor = extractor
        self.name = "tokencut" if target_select == "foreground" else "tokencut_attention_overlap"
        if feature_source == "aux":
            self.name += "_auxfeat"

    def _get_extractor(self):
        if self._extractor is None:
            self._extractor = DinoKeyExtractor(weights_path=self.weights_path, device=self.device,
                                               external_dir=self.external_dir, patch_size=self.cfg.patch_size,
                                               which_features=self.cfg.which_features)
        return self._extractor

    def provenance(self) -> dict:
        weights = getattr(self._extractor, "weights", None) if self.feature_source == "dino_vits16" else None
        return {"provider": self.name, "source": SOURCE, "config": dataclasses.asdict(self.cfg),
                "feature_source": self.feature_source,
                "dino_weights": weights or ({"url": DINO_WEIGHTS["vit_small16"], "sha256": None,
                                             "note": "not loaded yet"} if self.feature_source == "dino_vits16"
                                            else None),
                "target_select": self.target_select,
                "target_select_is_original_method": self.target_select == "foreground",
                "uses_query_text": False,
                "candidate_rule": ("official seed component" if self.target_select == "foreground" else
                                   "max above-mean attention mass over all components of both sides (NOT TokenCut)")}

    def reset(self) -> None:
        return None

    def _features(self, camera: str, image: np.ndarray, aux: Mapping | None):
        if self.feature_source == "dino_vits16":
            feats, dims = self._get_extractor()(image)
            return feats, dims, self.cfg.patch_size
        if aux is None or camera not in aux.get("patch_features", {}):
            return None, None, None
        grid = np.asarray(aux["patch_features"][camera], dtype=np.float32)  # (rows, cols, D)
        orient = aux.get("patch_features_orientation")
        if orient not in ("image", "libero_policy"):
            raise ValueError("aux['patch_features_orientation'] must be 'image' or 'libero_policy'")
        # (rows, cols, D) -> (D, rows, cols) so the column flip of `unmirror_grid` acts on the patch columns
        grid = np.ascontiguousarray(_to_image_orientation(grid.transpose(2, 0, 1), orient).transpose(1, 2, 0))
        return grid.reshape(-1, grid.shape[-1]), grid.shape[:2], None

    def predict(self, rgb: Mapping[str, np.ndarray], query, *, aux: Mapping | None = None) -> dict:
        iface = _interface()
        out = {}
        for camera, image in rgb.items():
            t0 = time.perf_counter()
            image = iface.check_rgb(image)
            hw = (int(image.shape[0]), int(image.shape[1]))
            mask = np.zeros(hw, dtype=bool)
            extras: dict = {"target_select": self.target_select, "feature_source": self.feature_source}
            found, score, boxes = False, None, None
            feats, dims, ps = self._features(camera, image, aux)
            if feats is None:
                extras["not_found_reason"] = f"no aux patch features for camera {camera!r}"
            else:
                try:
                    res = ncut(feats, dims, tau=self.cfg.tau, eps=self.cfg.eps, binary_graph=self.cfg.binary_graph)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    res, extras["not_found_reason"] = None, f"ncut failed: {exc}"
                if res is not None:
                    extras.update({"feature_grid": list(dims), "seed": res.seed,
                                   "foreground_patches": int(res.bipartition.sum())})
                    patch_mask = res.mask
                    if self.target_select == "attention_overlap":
                        att = None if aux is None else aux.get("attention_map", {}).get(camera)
                        if att is None:
                            patch_mask, info = None, {"not_found_reason": "no aux attention_map for camera"}
                        else:
                            att = np.asarray(att, dtype=np.float32)
                            orient = aux.get("attention_orientation")
                            if orient not in ("image", "libero_policy"):
                                raise ValueError("aux['attention_orientation'] must be 'image' or 'libero_policy'")
                            if not np.all(np.isfinite(att)):
                                patch_mask, info = None, {"not_found_reason": "non-finite attention map"}
                            elif float(att.max()) <= float(att.min()):
                                # checked before resampling: bilinear weights turn a constant float32 map
                                # into +-1 ulp ripples that would look like attention
                                patch_mask, info = None, {"not_found_reason": "constant attention map"}
                            else:
                                att_px = _bilinear(_to_image_orientation(att, orient), hw)
                                patch_mask, info = select_attention_overlap(res, att_px, hw, ps)
                        extras.update(info)
                        if patch_mask is not None:
                            score = float(info["overlap_frac"])
                    if patch_mask is not None:
                        mask = patch_mask_to_pixels(patch_mask, hw, ps).astype(bool)
                        found = bool(mask.any())
                        ys, xs = np.nonzero(mask)
                        if found:
                            boxes = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
            extras["box_convention"] = "pixel edges [x0, y0, x1+1, y1+1]"
            out[camera] = iface.TargetMaskResult(camera=camera, mask=mask, found=found, score=score, boxes=boxes,
                                                 phrases=[], latency_ms=(time.perf_counter() - t0) * 1e3,
                                                 extras=extras)
        return out
