"""T47b measurement #5 — sink diagnostic: is a head a fixed-patch sink or a broad single blob?

T47's label-free top-3 (L4H1 · L2H0 · L14H5) won the selection by **low spatial entropy** but put
1-2 % of their mass on the target. Low entropy has two very different causes:

* a **fixed-patch sink** — the head always puts most of its mass on the same patch, whatever the
  scene (high modal-argmax fraction, high top-1 share, one tiny component);
* a **broad single blob** — mass spread over a large connected region (low top-1 share, one
  component covering much of the above-mean area / grid).

Per map (one frame x cell x camera; T47b.task.md "범위 변경" 3):

(a) ``argmax_patch``  flat index ``r * 16 + c`` of the hottest patch; per (cell, camera) the
                      **modal-argmax fraction** = share of frames whose argmax is the most frequent
                      patch (ties between patches -> the lowest index; the patch is reported).
(b) ``top1_share``    max patch value / the map's mass ``S_cam`` (NaN for a zero map).
(c) ``n_components`` / ``largest_component_px`` after the paper's mean binarisation
                      (`t47_head_metrics.binarize_above_mean`, 8-connectivity — the same binarisation
                      as the spatial entropy); ``largest_share_of_binarized`` = largest / above-mean
                      pixels and ``largest_share_of_grid`` = largest / 256. A constant map has no
                      component (0, NaN).
(d) ``entropy``       `t47_head_metrics.spatial_entropy_batch` (raw nats, NaN without a component).
(e) ``S_cam``         the 16x16 sum (`t47_head_metrics.image_attention_sum`).

Pure numpy; the CLI stage ``sink`` (`t47b_study`) feeds it every record and frame at denoise 0 ·
agg last, cells L2-17 x 8 heads.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t47_head_metrics import (
    GRID, binarize_above_mean, spatial_entropy_batch)


def component_stats(maps: np.ndarray, *, chunk: int = 20000) -> dict[str, np.ndarray]:
    """``(..., 16, 16)`` -> ``n_components``, ``largest_component_px``, ``n_above_mean`` (each ``(...)``)."""
    from scipy import ndimage

    a = np.asarray(maps, np.float64)
    if a.ndim < 2 or a.shape[-2:] != (GRID, GRID):
        raise ValueError(f"expected (..., {GRID}, {GRID}) maps, got {a.shape}")
    lead = a.shape[:-2]
    flat = a.reshape((-1, GRID, GRID))
    n = flat.shape[0]
    ncomp = np.zeros(n, np.int64)
    largest = np.zeros(n, np.int64)
    above = np.zeros(n, np.int64)
    structure = np.zeros((3, 3, 3), bool)
    structure[1] = True  # 8-connectivity within a map, nothing across the stack
    for s in range(0, n, chunk):
        block = binarize_above_mean(flat[s:s + chunk])
        above[s:s + len(block)] = block.sum(axis=(1, 2))
        labels, n_lab = ndimage.label(block, structure=structure)
        if n_lab == 0:
            continue
        fl = labels.ravel()
        sizes = np.bincount(fl, minlength=n_lab + 1)[1:]
        ids, first = np.unique(fl, return_index=True)
        owner = first[ids > 0] // (GRID * GRID)
        ncomp[s:s + len(block)] = np.bincount(owner, minlength=len(block))
        big = np.zeros(len(block), np.int64)
        np.maximum.at(big, owner, sizes)
        largest[s:s + len(block)] = big
    return {"n_components": ncomp.reshape(lead), "largest_component_px": largest.reshape(lead),
            "n_above_mean": above.reshape(lead)}


def map_stats(maps: np.ndarray) -> dict[str, np.ndarray]:
    """Per-map (a)-(e) of ``(..., 16, 16)`` maps (module docstring). Every output is ``(...)``."""
    a = np.asarray(maps, np.float64)
    flat = a.reshape(a.shape[:-2] + (GRID * GRID,))
    s_cam = flat.sum(axis=-1)
    top = flat.max(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        top1 = np.where(s_cam > 0, top / np.where(s_cam > 0, s_cam, 1.0), np.nan)
    cs = component_stats(a)
    ent, _ = spatial_entropy_batch(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = np.where(cs["n_above_mean"] > 0,
                      cs["largest_component_px"] / np.maximum(cs["n_above_mean"], 1), np.nan)
        lg = np.where(cs["n_components"] > 0, cs["largest_component_px"] / float(GRID * GRID),
                      np.nan)
    return {"argmax_patch": flat.argmax(axis=-1).astype(np.int16), "top1_share": top1,
            "n_components": cs["n_components"], "largest_component_px": cs["largest_component_px"],
            "largest_share_of_binarized": lb, "largest_share_of_grid": lg,
            "entropy": ent, "S_cam": s_cam}


def modal_fraction(argmax_patch: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Along `axis` (frames): ``(fraction of frames at the modal patch, modal patch index)``.

    Ties between patches resolve to the lowest patch index (`np.argmax` over the counts).
    """
    p = np.moveaxis(np.asarray(argmax_patch, np.int64), axis, -1)
    n = p.shape[-1]
    if n == 0:
        raise ValueError("no frames")
    flat = p.reshape(-1, n)
    counts = np.zeros((flat.shape[0], GRID * GRID), np.int64)
    rows = np.repeat(np.arange(flat.shape[0]), n)
    np.add.at(counts, (rows, flat.ravel()), 1)
    mode = counts.argmax(axis=1)
    frac = counts[np.arange(flat.shape[0]), mode] / float(n)
    return frac.reshape(p.shape[:-1]), mode.reshape(p.shape[:-1])


def summarize_cells(per_frame: dict[str, np.ndarray], *, frame_axis: int = 0) -> dict[str, np.ndarray]:
    """Per (cell, camera) summaries over frames of `map_stats` output (frame axis first).

    ``modal_fraction`` / ``modal_patch`` (a); means over frames of ``top1_share`` (b),
    ``n_components``, ``largest_share_of_binarized``, ``largest_share_of_grid`` (c), ``entropy`` (d)
    and ``S_cam`` (e) — NaN frames skipped (``entropy_nan_frames`` counts the component-less ones).
    """
    frac, mode = modal_fraction(per_frame["argmax_patch"], axis=frame_axis)
    out = {"modal_fraction": frac, "modal_patch": mode.astype(np.int16),
           "modal_patch_rc": np.stack([mode // GRID, mode % GRID], axis=-1).astype(np.int16)}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cell -> NaN, on purpose
        for k in ("top1_share", "n_components", "largest_share_of_binarized",
                  "largest_share_of_grid", "entropy", "S_cam"):
            v = np.asarray(per_frame[k], np.float64)
            out[f"{k}_mean"] = np.nanmean(v, axis=frame_axis)
        out["top1_share_median"] = np.nanmedian(np.asarray(per_frame["top1_share"], np.float64),
                                                axis=frame_axis)
    out["entropy_nan_frames"] = np.isnan(np.asarray(per_frame["entropy"], np.float64)).sum(
        axis=frame_axis)
    return out


def classify_cell(modal_fraction: float, top1_share: float, largest_share_of_grid: float, *,
                  sink_modal: float = 0.8, sink_top1: float = 0.3, blob_grid: float = 0.15) -> str:
    """A descriptive label only (not used to select anything; thresholds are arguments, reported).

    ``fixed_patch_sink``  modal fraction >= `sink_modal` and top-1 share >= `sink_top1`;
    ``broad_single_blob`` otherwise, when the largest component covers >= `blob_grid` of the grid;
    ``other``             the rest.
    """
    if modal_fraction >= sink_modal and top1_share >= sink_top1:
        return "fixed_patch_sink"
    if np.isfinite(largest_share_of_grid) and largest_share_of_grid >= blob_grid:
        return "broad_single_blob"
    return "other"


def mark_cells(cells: Sequence[Sequence[int]], *, first_layer: int = 2, n_heads: int = 8
               ) -> list[int]:
    """``(layer, head)`` -> row index into an L2-17 x 8 cell axis (layer-major)."""
    return [(int(l) - first_layer) * n_heads + int(h) for l, h in cells]


__all__ = ["classify_cell", "component_stats", "map_stats", "mark_cells", "modal_fraction",
           "summarize_cells"]
