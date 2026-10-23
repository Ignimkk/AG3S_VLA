"""Localization heads (Kang et al., CVPR 2025) applied to pi0.5's recorded attention -- partial re-impl.

Paper: S. Kang, J. Kim, J. Kim, S. J. Hwang, "Your large vision-language model only needs a few attention
heads for visual grounding", arXiv:2503.06287 (CVPR 2025). Code: seilk/LocalizationHeads @
``9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a`` (fetched by ``external/fetch_sources.py`` into
``external/LocalizationHeads``). The pinned checkout has **no LICENSE file** although its README says MIT;
nothing of it is vendored here -- the functions below are re-written from it and cite it line by line.

What is ported, and from where
------------------------------
* ``elbow_chord``                 = ``analyze.py:45-63``  (criterion 1 threshold: max distance to the chord
                                    of the ascending-sorted head sums; the paper's "maximum curvature")
* ``spatial_entropy_code``        = ``analyze.py:20-42``  (criterion 2 as the code computes it:
                                    ``relu(S - 2 mean) > 1e-3``, 8-connected components, **mass**-weighted p)
* ``spatial_entropy_paper``       = paper Sec. 4.1 / Eq. 3 / App. B.1 (binarise ``> mean``, 8-connected,
                                    **area**-weighted p). Only used by ``protocol="paper"``
* ``analyze_heads_code``          = ``analyze.py:66-119`` (per-sample chord threshold on head sums, entropy for
                                    heads above it, drop bottom-row-focus heads and layers 0-1, sort by entropy)
* frequency aggregation           = paper Sec. 4.2: a head is counted each time it is among the
                                    ``10``-lowest-entropy heads of a sample; heads are ranked by that count and
                                    the top ``k`` kept (``k = 3`` in the paper's main experiments; the code's demo
                                    yaml uses ``top_k: 5``). The pinned code has no cross-sample aggregation.
* ``combine_heads``               = ``bbox.py:9-22``  (Gaussian sigma 1.0 per head map, summed; scipy's default
                                    ``truncate=4`` i.e. a 9-tap kernel -- the paper's App. B.2 says kernel 7)
* ``binarize_mean_relu``          = ``bbox.py:25-28`` (``M > mean(M)``)
* ``upscale_bilinear``            = ``bbox.py:40-46`` (``F.interpolate(bilinear, align_corners=False) > 0.5``),
                                    re-written in numpy; ``tests/test_localization_heads.py`` checks it against
                                    torch when torch is importable

What is different from the paper (also in ``docs/catalog_vla_internal.md``)
---------------------------------------------------------------------------
* Model: pi0.5 (PaliGemma Gemma-2B prefix, 18 layers x 8 heads, SigLIP 16x16 grid per camera), not LLaVA.
* Query token: the paper reads the **last text token** of a referring-expression prompt. The recorded pi0.5
  grids (``benchmark/runners/attention_audit.py`` -> ``pooled_f16``) are the attention of the **last action
  token of the action expert at Euler denoise step 0** (``AGGREGATIONS["last"]``) over the 768 image tokens of
  the prefix; the softmax runs over prefix + suffix keys, so the image part sums to < 1. There is no separate
  noun query: the whole instruction is in the prompt the policy was run with; ``TargetQuery`` text is not used
  at predict time.
* Two cameras per call (``head`` 0-256, ``left_wrist`` 256-512; ``right_wrist`` 512-768 is masked out). Each
  camera's 16x16 block is treated as its own P x P image, and heads are selected **per camera**.
* Samples: LIBERO dev policy calls (init states 0-4 only), not 1,000 RefCOCO images.
* ``sam_point`` prompts SAM with the attention peak point; the paper prompts SAM with the box. ``sam_box`` is
  the paper's recipe. Both need Implementer A's SAM wrapper (``grounded_sam.py``).

Pixel frame: pi05_libero grids are in **policy orientation** (mirrored left-right relative to the adapter's
OpenCV/depth image, ``AG3S_BENCHMARK_LOG.md`` SL1/SL3). ``aux["attention_orientation"] = "libero_policy"``
un-mirrors with :func:`AG3S.adapters.panda.attention.unmirror_grid` (``grid[:, ::-1]``; the 224->256 scale is
absorbed by resampling the grid to the RGB size). ``"image"`` means the grid is already in the RGB frame.

Usage
-----
Select heads on dev runs (CPU, reads ``attention/*.npz`` episode by episode)::

    .venv-ag3s/bin/python -m Baseline.paper_baselines.providers.localization_heads select \\
        --attention-run <sl3 audit run> --attention-run <sl4b audit run> --out <frozen.json>

Predict (``aux`` from one recorded call)::

    prov = LocalizationHeadsProvider(<frozen.json>, mask_mode="threshold")
    aux = aux_from_attention_npz(npz, call_index)
    prov.predict({"head": rgb_head, "left_wrist": rgb_wrist}, query, aux=aux)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, label

SOURCE = {
    "paper": "Kang et al., Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding, "
             "arXiv:2503.06287, CVPR 2025",
    "repo": "https://github.com/seilk/LocalizationHeads",
    "sha": "9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a",
    "files": ["analyze.py", "bbox.py", "conf/logic/selection_v1.yaml"],
    "license": "UNKNOWN (no LICENSE file in the pinned checkout; README states MIT)",
    "implementation": "partial re-impl (numpy); no code vendored",
}

SCHEMA = "paper_baselines.localization_heads.selection/v1"
#: pi05_libero prefix image token blocks per wire camera (`AG3S/adapters/panda/attention.py` PI05_LIBERO_TOKEN_BLOCKS).
PI05_LIBERO_BLOCKS: dict[str, tuple[int, int]] = {"head": (0, 256), "left_wrist": (256, 512)}
#: LIBERO dev split (AG3S_BENCHMARK_LOG.md decision "안 A": init states 0-4 dev, 5-49 test).
DEV_INIT_STATES = (0, 1, 2, 3, 4)
ORIENTATIONS = ("libero_policy", "image")
MASK_MODES = ("threshold", "sam_point", "sam_box")


# ----------------------------------------------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SelectionConfig:
    """Head-selection settings. Defaults: pinned code (``conf/logic/selection_v1.yaml``, ``analyze.py``) for the
    per-sample criteria, paper Sec. 4.2 for the cross-sample frequency and ``k``."""

    protocol: str = "official_code"          # "official_code" | "paper"
    entropy_binarize_threshold: float = 1e-3  # selection_v1.yaml entropy.binarize_threshold
    min_keep: int = 1                         # selection_v1.yaml threshold.min_keep
    bottom_row_value: float = 0.05            # analyze.py:98
    min_layer: int = 2                        # analyze.py:112 `layer > 1`; paper: first two layers excluded
    top_n_per_sample: int = 10                # paper Sec. 4.2: "10-lowest spatial entropy"
    top_k: int = 3                            # paper: k = 3 for all models (code demo yaml: top_k 5)

    def __post_init__(self):
        if self.protocol not in ("official_code", "paper"):
            raise ValueError(f"protocol must be 'official_code' or 'paper', got {self.protocol!r}")


@dataclasses.dataclass(frozen=True)
class MaskConfig:
    smoothing_sigma: float = 1.0      # selection_v1.yaml smoothing.sigma, bbox.py combine_heads
    smoothing_truncate: float = 4.0   # scipy default used by bbox.py (9-tap); paper App. B.2 kernel 7 = truncate 3
    upscale_threshold: float = 0.5    # bbox.py upscale_mask


# ----------------------------------------------------------------------------------------------------------
# ported criteria (numpy)
# ----------------------------------------------------------------------------------------------------------
def elbow_chord(values: Sequence[float]) -> float:
    """``analyze.py:45-63``: threshold value (not index) at the max distance to the chord of sorted values."""
    if len(values) <= 2:
        return min(values) if len(values) else 0.0
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return float(y[0])
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    return float(y[int(np.argmax(d))])


_EIGHT = np.ones((3, 3), dtype=np.int32)


def spatial_entropy_code(a2d: np.ndarray, threshold: float = 1e-3) -> dict:
    """``analyze.py:20-42`` exactly: ``B = relu(S - 2 mean)``, components of ``B > threshold`` (8-conn),
    ``p_i = mass_i / sum(B)``. Returns ``inf`` when ``sum(B) <= 0``."""
    s = np.asarray(a2d, dtype=np.float32)
    b = np.maximum(s - s.mean(dtype=np.float32) * np.float32(2.0), np.float32(0.0))
    binary = (b > threshold).astype(np.int32)
    labeled, num = label(binary, structure=_EIGHT)
    total = float(b.sum())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}
    probs = []
    for i in range(1, num + 1):
        comp_sum = b[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)
    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}


def spatial_entropy_paper(a2d: np.ndarray) -> dict:
    """Paper Sec. 4.1 Eq. 3 / App. B.1: binarise ``> mean``, 8-connected components, ``p_i = |C_i| / sum |C_j|``."""
    s = np.asarray(a2d, dtype=np.float32)
    binary = (s > s.mean(dtype=np.float32)).astype(np.int32)
    labeled, num = label(binary, structure=_EIGHT)
    if num == 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}
    sizes = np.bincount(labeled.ravel())[1:].astype(np.float64)
    p = sizes / sizes.sum()
    return {"spatial_entropy": float(-(p * np.log(p)).sum()), "labeled_array": labeled, "num_components": int(num)}


def analyze_heads_code(attn: np.ndarray, P: int, cfg: SelectionConfig = SelectionConfig()) -> list[dict]:
    """``analyze.py:66-119`` for one sample. ``attn``: ``(L, H, P*P)`` (the official ``[L, H, 1, V]`` without the
    query axis). Returns the kept heads sorted by spatial entropy (ascending; stable, so ties keep (l, h) order)."""
    attn = np.asarray(attn, dtype=np.float32)
    L, H, V = attn.shape
    if V != P * P:
        raise ValueError(f"attention has {V} image tokens, expected P*P = {P * P}")
    sums = [float(attn[l, h].sum()) for l in range(L) for h in range(H)]
    thr_val = elbow_chord(sums)
    results: list[dict] = []
    idx = 0
    for l in range(L):
        for h in range(H):
            s = sums[idx]
            idx += 1
            if s < thr_val:
                se, bottom_row_focus, n_comp = float("inf"), False, 0
            else:
                a2d = attn[l, h].reshape(P, P)
                se_res = spatial_entropy_code(a2d, cfg.entropy_binarize_threshold)
                bottom_row_focus = bool((a2d.shape[0] > 0) and (a2d[-1, :] > cfg.bottom_row_value).any())
                se = float(se_res["spatial_entropy"])
                n_comp = int(se_res["num_components"])
            results.append({"layer": l, "head": h, "attn_sum": s, "spatial_entropy": se,
                            "bottom_row_focus": bottom_row_focus, "num_components": n_comp})
    kept = [r for r in results if np.isfinite(r["spatial_entropy"]) and r["attn_sum"] >= thr_val
            and not r["bottom_row_focus"] and r["layer"] >= cfg.min_layer]
    if len(kept) < cfg.min_keep:
        by_sum = sorted(results, key=lambda x: x["attn_sum"], reverse=True)
        kept = [x for x in by_sum if not x["bottom_row_focus"]][: cfg.min_keep]
    kept.sort(key=lambda x: x["spatial_entropy"])
    return kept


# ----------------------------------------------------------------------------------------------------------
# cross-sample selection (paper Sec. 4.2 frequency)
# ----------------------------------------------------------------------------------------------------------
def _rank_by_frequency(freq: Mapping, se_sum: Mapping, sum_s: Mapping, n_samples: int, L: int, H: int,
                       cfg: SelectionConfig, extra: dict) -> dict:
    rows = []
    for l in range(L):
        for h in range(H):
            f = int(freq.get((l, h), 0))
            mean_se = (se_sum[(l, h)] / f) if f else None
            rows.append({"layer": l, "head": h, "frequency": f,
                         "frequency_rate": f / n_samples if n_samples else 0.0,
                         "mean_spatial_entropy_when_counted": mean_se,
                         "mean_image_attention_sum": sum_s[(l, h)] / n_samples if n_samples else 0.0})
    rows.sort(key=lambda r: (-r["frequency"],
                             r["mean_spatial_entropy_when_counted"] if r["frequency"] else math.inf,
                             r["layer"], r["head"]))
    selected = [r for r in rows if r["frequency"] > 0][: cfg.top_k]
    return {"protocol": cfg.protocol, "n_samples": n_samples, "num_layers": L, "num_heads": H,
            "selected": selected, "ranking_top20": rows[:20], **extra}


def select_heads(sample_source: Callable[[], Iterable[np.ndarray]], P: int,
                 cfg: SelectionConfig = SelectionConfig()) -> dict:
    """Select localization heads for one camera.

    ``sample_source()`` returns a fresh iterable of ``(L, H, P*P)`` arrays (one per dev sample); ``"paper"``
    iterates it twice (dataset-mean criterion 1, then entropy). Deterministic: no randomness, ties broken by
    (mean entropy, layer, head).
    """
    freq: dict = defaultdict(int)
    se_sum: dict = defaultdict(float)
    sum_s: dict = defaultdict(float)
    n = 0
    L = H = None
    if cfg.protocol == "official_code":
        taus = []
        for attn in sample_source():
            attn = np.asarray(attn, dtype=np.float32)
            L, H, _ = attn.shape
            n += 1
            totals = attn.sum(-1)
            for l in range(L):
                for h in range(H):
                    sum_s[(l, h)] += float(totals[l, h])
            taus.append(elbow_chord([float(totals[l, h]) for l in range(L) for h in range(H)]))
            for r in analyze_heads_code(attn, P, cfg)[: cfg.top_n_per_sample]:
                freq[(r["layer"], r["head"])] += 1
                if np.isfinite(r["spatial_entropy"]):
                    se_sum[(r["layer"], r["head"])] += r["spatial_entropy"]
        if n == 0:
            raise ValueError("no samples")
        extra = {"criterion1": "per-sample chord elbow on head image-attention sums (analyze.py:82)",
                 "criterion2": "code spatial entropy (analyze.py:20-42), bottom-row filter, layer >= min_layer",
                 "tau_per_sample": {"min": float(np.min(taus)), "median": float(np.median(taus)),
                                    "max": float(np.max(taus))}}
        return _rank_by_frequency(freq, se_sum, sum_s, n, L, H, cfg, extra)

    # protocol == "paper": pass 1 -- dataset-mean S_img per head, chord elbow.
    for attn in sample_source():
        attn = np.asarray(attn, dtype=np.float32)
        L, H, _ = attn.shape
        n += 1
        totals = attn.sum(-1)
        for l in range(L):
            for h in range(H):
                sum_s[(l, h)] += float(totals[l, h])
    if n == 0:
        raise ValueError("no samples")
    mean_s = [sum_s[(l, h)] / n for l in range(L) for h in range(H)]
    tau = elbow_chord(mean_s)
    eligible = [(l, h) for l in range(L) for h in range(H)
                if sum_s[(l, h)] / n >= tau and l >= cfg.min_layer]
    # pass 2 -- per-sample paper entropy for eligible heads, count the 10 lowest.
    for attn in sample_source():
        attn = np.asarray(attn, dtype=np.float32)
        ents = []
        for (l, h) in eligible:
            se = spatial_entropy_paper(attn[l, h].reshape(P, P))["spatial_entropy"]
            if np.isfinite(se):
                ents.append((se, l, h))
        ents.sort()
        for se, l, h in ents[: cfg.top_n_per_sample]:
            freq[(l, h)] += 1
            se_sum[(l, h)] += se
    extra = {"criterion1": "chord elbow on dataset-mean head image-attention sums (paper Sec. 4.1)",
             "criterion2": "paper spatial entropy (> mean, area-weighted, 8-conn), layer >= min_layer",
             "tau": tau, "n_eligible": len(eligible)}
    return _rank_by_frequency(freq, se_sum, sum_s, n, L, H, cfg, extra)


# ----------------------------------------------------------------------------------------------------------
# recorded attention-audit runs (benchmark/runners/attention_audit.py)
# ----------------------------------------------------------------------------------------------------------
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_attention_runs(run_dirs: Sequence[str | Path], dev_init_states: Sequence[int] = DEV_INIT_STATES) -> dict:
    """Episode rows of ``attention_index.jsonl`` split into dev (used) and excluded (non-dev init state)."""
    used, excluded = [], []
    for run in run_dirs:
        run = Path(run)
        with open(run / "attention_index.jsonl") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                init = row.get("init_state_index", row.get("episode_idx"))
                item = {"run": run.name, "run_dir": str(run), "attn_npz": row["attn_npz"],
                        "suite": row.get("suite"), "task_id": row.get("task_id"),
                        "init_state_index": None if init is None else int(init),
                        "calls": row.get("calls")}
                (used if item["init_state_index"] in set(int(i) for i in dev_init_states) else excluded).append(item)
    used.sort(key=lambda r: (r["suite"] or "", r["task_id"] if r["task_id"] is not None else -1,
                             r["init_state_index"], r["run"]))
    return {"used": used, "excluded": excluded}


def _iter_camera_samples(episodes: Sequence[dict], block: tuple[int, int]) -> Iterator[np.ndarray]:
    lo, hi = block
    for ep in episodes:
        with np.load(str(Path(ep["run_dir"]) / ep["attn_npz"])) as z:
            pooled = z["pooled_f16"]  # (calls, L, H, 768) float16, policy orientation
            for k in range(pooled.shape[0]):
                yield pooled[k, :, :, lo:hi].astype(np.float32)


def _canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_selection(run_dirs: Sequence[str | Path], *, cameras: Sequence[str] = ("head", "left_wrist"),
                    blocks: Mapping[str, tuple[int, int]] = PI05_LIBERO_BLOCKS,
                    cfg: SelectionConfig = SelectionConfig(), grid: int = 16,
                    dev_init_states: Sequence[int] = DEV_INIT_STATES, hash_inputs: bool = True) -> dict:
    """Frozen selection document (no timestamp inside; ``selection_sha256`` covers everything but
    ``created_at``)."""
    idx = index_attention_runs(run_dirs, dev_init_states)
    if not idx["used"]:
        raise ValueError("no dev episodes in the given attention runs")
    inputs = []
    for ep in idx["used"]:
        item = {k: ep[k] for k in ("run", "attn_npz", "suite", "task_id", "init_state_index", "calls")}
        if hash_inputs:
            item["sha256"] = sha256_file(Path(ep["run_dir"]) / ep["attn_npz"])
        inputs.append(item)
    per_camera = {}
    for cam in cameras:
        block = tuple(int(v) for v in blocks[cam])
        per_camera[cam] = {"token_block": list(block), "grid": grid,
                           **select_heads(lambda b=block: _iter_camera_samples(idx["used"], b), grid, cfg)}
    doc = {
        "schema": SCHEMA,
        "method": "localization_heads",
        "source": SOURCE,
        "model": "pi05_libero",
        "attention_definition": {
            "producer": "benchmark/runners/attention_audit.py (key pooled_f16, shape (calls, L, H, 768))",
            "query": "last action token of the action expert, Euler denoise step 0 (AGGREGATIONS['last'])",
            "keys": "768 prefix image tokens; softmax over prefix + suffix keys (image part sums to < 1)",
            "orientation": "libero_policy (mirrored left-right vs OpenCV/depth image)",
            "per_camera": "each 16x16 camera block is one P x P image; heads selected per camera",
        },
        "config": dataclasses.asdict(cfg),
        "dev_split": {"init_states": list(dev_init_states),
                      "rule": "AG3S_BENCHMARK_LOG.md: LIBERO init 0-4 dev, 5-49 test; selection on dev only",
                      "excluded_non_dev": [{k: e[k] for k in ("run", "attn_npz", "init_state_index")}
                                           for e in idx["excluded"]]},
        "inputs": inputs,
        "selection": per_camera,
    }
    doc["selection_sha256"] = _canonical_sha(doc)
    return doc


def load_selection(selection: str | Path | Mapping) -> dict:
    doc = json.loads(Path(selection).read_text()) if not isinstance(selection, Mapping) else dict(selection)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"not a localization-heads selection ({doc.get('schema')!r})")
    body = {k: v for k, v in doc.items() if k not in ("selection_sha256", "created_at")}
    if doc.get("selection_sha256") != _canonical_sha(body):
        raise ValueError("selection_sha256 does not match the document (edited after freezing?)")
    return doc


def aux_from_attention_npz(npz: Mapping[str, np.ndarray], call_index: int) -> dict:
    """``aux`` for :meth:`LocalizationHeadsProvider.predict` from one call of an attention-audit episode npz."""
    return {"attention_pooled": np.asarray(npz["pooled_f16"][call_index], dtype=np.float32),
            "attention_blocks": dict(PI05_LIBERO_BLOCKS), "attention_orientation": "libero_policy",
            "attention_grid": 16}


# ----------------------------------------------------------------------------------------------------------
# mask generation (bbox.py)
# ----------------------------------------------------------------------------------------------------------
def combine_heads(grids: Sequence[np.ndarray], sigma: float = 1.0, truncate: float = 4.0) -> np.ndarray:
    """``bbox.py:9-22``: per-head Gaussian smoothing (scipy, mode 'reflect'), summed in float32."""
    M = None
    for g in grids:
        a = np.asarray(g, dtype=np.float32)
        if sigma and sigma > 0:
            a = gaussian_filter(a, sigma=sigma, truncate=truncate)
        M = a.astype(np.float32) if M is None else M + a.astype(np.float32)
    if M is None:
        raise ValueError("no heads to combine")
    return M


def binarize_mean_relu(M: np.ndarray) -> np.ndarray:
    """``bbox.py:25-28``: ``relu(M - mean) > 0``."""
    return (np.maximum(M - M.mean(), 0.0) > 0)


def _bilinear_weights(n_in: int, n_out: int) -> np.ndarray:
    """Row-stochastic (n_out, n_in) matrix of torch ``F.interpolate(bilinear, align_corners=False)``."""
    scale = n_in / n_out
    w = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        src = max((i + 0.5) * scale - 0.5, 0.0)
        i0 = min(int(math.floor(src)), n_in - 1)
        i1 = i0 + 1 if i0 < n_in - 1 else i0
        lam = src - i0
        w[i, i0] += 1.0 - lam
        w[i, i1] += lam
    return w


def upscale_bilinear(grid: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """``F.interpolate(grid[None, None], size=out_hw, mode='bilinear', align_corners=False)[0, 0]`` in numpy."""
    g = np.asarray(grid, dtype=np.float64)
    wy = _bilinear_weights(g.shape[0], int(out_hw[0]))
    wx = _bilinear_weights(g.shape[1], int(out_hw[1]))
    return (wy @ g @ wx.T).astype(np.float32)


def bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    """Pixel-edge box ``[x0, y0, x1 + 1, y1 + 1]`` of a bool mask, or ``None`` when empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def to_image_orientation(grid: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "image":
        return np.asarray(grid)
    if orientation == "libero_policy":
        from AG3S.adapters.panda.attention import unmirror_grid  # single source of the LIBERO mirror rule

        return np.ascontiguousarray(unmirror_grid(np.asarray(grid)))
    raise ValueError(f"attention_orientation must be one of {ORIENTATIONS}, got {orientation!r}")


# ----------------------------------------------------------------------------------------------------------
# provider
# ----------------------------------------------------------------------------------------------------------
def _interface():
    from Baseline.paper_baselines import interface  # Implementer A

    return interface


class LocalizationHeadsProvider:
    """Target mask from the frozen localization heads of pi0.5 (``TargetMaskProvider``).

    ``mask_mode``: ``"threshold"`` (bbox.py recipe, upsampled to the RGB size), ``"sam_point"`` (SAM prompted
    with the positive attention-peak point) or ``"sam_box"`` (SAM prompted with the threshold-mask box; the
    paper's recipe). ``sam``: an object with ``predict_points(rgb, points_xy (N, 2), labels (N,))`` and
    ``predict_box(rgb, box_xyxy)`` -> ``(mask bool (H, W), score float)`` (Implementer A's
    ``providers.sam_backend.SamV1Backend``, the default), or ``segment(rgb, *, points_xy=, point_labels=, box_xyxy=)``.
    """

    name = "localization_heads"

    def __init__(self, selection: str | Path | Mapping, *, mask_mode: str = "threshold",
                 mask_cfg: MaskConfig = MaskConfig(), sam: Any = None):
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
        self.selection = load_selection(selection)
        self.selection_path = None if isinstance(selection, Mapping) else str(selection)
        self.mask_mode = mask_mode
        self.mask_cfg = mask_cfg
        self._sam = sam
        if mask_mode != "threshold":
            self.name = f"localization_heads_{mask_mode}"

    def provenance(self) -> dict:
        sam_prov = None
        if self._sam is not None and hasattr(self._sam, "provenance"):
            sam_prov = self._sam.provenance()
        return {"provider": self.name, "source": SOURCE, "mask_mode": self.mask_mode,
                "mask_config": dataclasses.asdict(self.mask_cfg),
                "selection_path": self.selection_path,
                "selection_sha256": self.selection["selection_sha256"],
                "selection_config": self.selection["config"],
                "heads": {cam: [(r["layer"], r["head"]) for r in v["selected"]]
                          for cam, v in self.selection["selection"].items()},
                "uses_query_text": False,
                "candidate_rule": "single candidate per camera (no box ranking)",
                "sam": sam_prov}

    def reset(self) -> None:  # stateless across frames
        return None

    def _sam_segmenter(self):
        if self._sam is None:
            try:  # Implementer A's SAM v1 wrapper (lazy: torch only when a SAM mode is used)
                from Baseline.paper_baselines.providers.sam_backend import SamV1Backend
            except ImportError as exc:
                raise NotImplementedError(
                    "sam_point / sam_box need a SAM segmenter: Baseline.paper_baselines.providers.sam_backend is not "
                    f"importable ({exc}); pass sam=<object with predict_points / predict_box>") from exc
            self._sam = SamV1Backend()
        return self._sam

    def _sam_points(self, image, points_xy, labels):
        sam = self._sam_segmenter()
        if hasattr(sam, "predict_points"):  # sam_backend.SamV1Backend
            return sam.predict_points(image, points_xy, labels)
        return sam.segment(image, points_xy=points_xy, point_labels=labels)

    def _sam_box(self, image, box_xyxy):
        sam = self._sam_segmenter()
        if hasattr(sam, "predict_box"):
            return sam.predict_box(image, box_xyxy)
        return sam.segment(image, box_xyxy=box_xyxy)

    def combined_map(self, aux: Mapping, camera: str) -> tuple[np.ndarray | None, str | None, list]:
        """Combined smoothed map of the selected heads in **image orientation** (grid resolution), or
        ``(None, reason, heads)``."""
        sel = self.selection["selection"].get(camera)
        if sel is None or not sel["selected"]:
            return None, f"no frozen heads for camera {camera!r}", []
        heads = [(int(r["layer"]), int(r["head"])) for r in sel["selected"]]
        blocks = aux.get("attention_blocks", PI05_LIBERO_BLOCKS)
        if camera not in blocks:
            return None, f"no attention token block for camera {camera!r}", heads
        lo, hi = (int(v) for v in blocks[camera])
        P = int(aux.get("attention_grid", sel.get("grid", 16)))
        pooled = np.asarray(aux["attention_pooled"], dtype=np.float32)
        grids = [pooled[l, h, lo:hi].reshape(P, P) for (l, h) in heads]
        M = combine_heads(grids, self.mask_cfg.smoothing_sigma, self.mask_cfg.smoothing_truncate)
        if not np.all(np.isfinite(M)):
            return None, "non-finite attention", heads
        if float(M.max()) <= float(M.min()):
            return None, "constant attention map (no localisation)", heads
        return to_image_orientation(M, aux["attention_orientation"]), None, heads

    def predict(self, rgb: Mapping[str, np.ndarray], query, *, aux: Mapping | None = None) -> dict:
        iface = _interface()
        if aux is None or "attention_pooled" not in aux:
            raise ValueError("localization_heads needs aux['attention_pooled'] (L, H, n_image_tokens)")
        if aux.get("attention_orientation") not in ORIENTATIONS:
            raise ValueError(f"aux['attention_orientation'] must be one of {ORIENTATIONS} (pixel-frame safety)")
        out = {}
        for camera, image in rgb.items():
            t0 = time.perf_counter()
            image = iface.check_rgb(image)
            Hh, Ww = int(image.shape[0]), int(image.shape[1])
            M, reason, heads = self.combined_map(aux, camera)
            extras: dict = {"heads": heads, "mask_mode": self.mask_mode,
                            "attention_orientation": aux.get("attention_orientation"),
                            "selection_sha256": self.selection["selection_sha256"]}
            mask = np.zeros((Hh, Ww), dtype=bool)
            score, boxes, found = None, None, False
            if M is None:
                extras["not_found_reason"] = reason
            else:
                grid_mask = binarize_mean_relu(M)
                thr_mask = upscale_bilinear(grid_mask.astype(np.float32), (Hh, Ww)) > self.mask_cfg.upscale_threshold
                dense = upscale_bilinear(M, (Hh, Ww))
                py, px = np.unravel_index(int(np.argmax(dense)), dense.shape)
                extras.update({"peak_xy": [int(px), int(py)], "map_max": float(M.max()), "map_mean": float(M.mean()),
                               "grid_mask_patches": int(grid_mask.sum())})
                score = float(M.max())
                if self.mask_mode == "threshold":
                    mask = thr_mask
                elif self.mask_mode == "sam_point":
                    m, s = self._sam_points(np.asarray(image), np.array([[px, py]], np.float32),
                                            np.array([1], np.int64))
                    mask, extras["sam_score"] = np.asarray(m, dtype=bool), (None if s is None else float(s))
                else:  # sam_box
                    box = bbox_xyxy(thr_mask)
                    if box is not None:
                        m, s = self._sam_box(np.asarray(image), box)
                        mask, extras["sam_score"] = np.asarray(m, dtype=bool), (None if s is None else float(s))
                        extras["prompt_box_xyxy"] = box.tolist()
                found = bool(mask.any())
                if not found:
                    extras["not_found_reason"] = "empty mask"
                    score = None
                else:
                    box = bbox_xyxy(mask)
                    boxes = box[None, :]
            extras["box_convention"] = "pixel edges [x0, y0, x1+1, y1+1]"
            out[camera] = iface.TargetMaskResult(
                camera=camera, mask=mask, found=found, score=score, boxes=boxes, phrases=[],
                latency_ms=(time.perf_counter() - t0) * 1e3, extras=extras)
        return out


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="select heads on dev attention-audit runs and freeze them to json")
    s.add_argument("--attention-run", action="append", required=True, help="attention_audit.py run dir (repeat)")
    s.add_argument("--out", required=True)
    s.add_argument("--protocol", default="official_code", choices=("official_code", "paper"))
    s.add_argument("--top-k", type=int, default=SelectionConfig.top_k)
    s.add_argument("--cameras", nargs="+", default=["head", "left_wrist"])
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "select":
        cfg = SelectionConfig(protocol=args.protocol, top_k=args.top_k)
        doc = build_selection(args.attention_run, cameras=args.cameras, cfg=cfg)
        from datetime import datetime, timedelta, timezone

        doc["created_at"] = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise SystemExit(f"{out} exists; frozen selections are not overwritten")
        out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
        for cam, v in doc["selection"].items():
            print(cam, "n_samples", v["n_samples"], "selected",
                  [(r["layer"], r["head"], r["frequency"]) for r in v["selected"]])
        print("wrote", out, "selection_sha256", doc["selection_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
