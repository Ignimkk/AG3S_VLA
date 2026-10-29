"""T47b B1 — how the selected heads are summed: raw sum vs per-head mass-normalised sums.

T47 sums the top-3 heads' raw maps (the paper's element-wise sum, `t47_head_metrics.combine_maps`).
A raw sum lets the head with the largest image mass dominate the combined map, and because the
mass is split unevenly over the three cameras it also changes which camera's values are largest —
which matters downstream, where the cameras are fused by a raw ``max`` (`AG3S/ag3s/runtime/
multiview.py:384`) before the fused values are percentile-normalised (`:399`).

Three ways to combine the same head set (T47b.task.md B1):

``raw_sum``               T47 as is (the reference): smooth each head's raw map, sum.
``mass_norm_joint``       divide each head's maps by **that head's mass over all three cameras**
                          (768 tokens), then smooth and sum. Every head carries mass 1; the head's own
                          camera-to-camera ratios are kept. **Pre-declared secondary comparison.**
``mass_norm_per_camera``  divide each (head, camera) 16x16 map by its own mass, then smooth and sum.
                          Every (head, camera) carries mass 1; camera-to-camera ratios are gone.
                          Exploratory.

Normalising before or after the Gaussian smoothing gives the same map up to float rounding: the
smoothing (`t47_head_metrics.gaussian_smooth`, reflect boundary) preserves each map's mass. We
normalise the raw maps first (the mass is the paper's ``S_img`` / ``S_cam`` criterion quantity).

A head (or head x camera) with zero mass contributes zeros — it cannot be normalised and has nothing
to say. `zero_mass_counts` reports how often that happened; negative attention is refused (softmax
mass is never negative).

Pure numpy; pinned by `tests/ag3s/test_t47b_aggregation_metrics.py`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t47_head_metrics import GRID, combine_maps, gaussian_smooth

AGGREGATIONS = ("raw_sum", "mass_norm_joint", "mass_norm_per_camera")
#: T47.task.md pre-registration: raw sum is the paper's (reference) sum; T47b.task.md B1 + "범위 변경"
#: (2026-10-12): joint mass-normalisation is the pre-declared secondary comparison **on the GT-oracle
#: top-3**; per-camera mass-normalisation is exploratory.
AGGREGATION_ROLE = {"raw_sum": "reference (T47 / paper sum)",
                    "mass_norm_joint": "secondary, pre-declared on oracle top-3 (T47b B1 scope change)",
                    "mass_norm_per_camera": "exploratory"}
#: Axis layout every function here expects: ``(..., C, L, H, 16, 16)`` — `t47_head_selection.
#: source_maps`' ``block_c`` (frame, camera, layer, head, grid, grid).
CAMERA_AXIS = -5


def _check_block(block_c: np.ndarray) -> np.ndarray:
    a = np.asarray(block_c)
    if a.ndim < 5 or a.shape[-2:] != (GRID, GRID):
        raise ValueError(f"expected (..., C, L, H, {GRID}, {GRID}) maps, got {a.shape}")
    return a


def _cells(cells: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    out = [(int(c[0]), int(c[1])) for c in cells]
    if not out:
        raise ValueError("cells must not be empty")
    return out


def head_maps(block_c: np.ndarray, cells: Sequence[Sequence[int]]) -> np.ndarray:
    """``(..., k, C, 16, 16)`` float64 raw maps of the `cells` (``(layer, head)``), in that order."""
    a = _check_block(block_c)
    return np.stack([np.asarray(a[..., l, h, :, :], np.float64) for l, h in _cells(cells)],
                    axis=-4)


def head_masses(block_c: np.ndarray, cells: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    """Per-head image mass of the raw maps.

    ``per_camera`` ``(..., k, C)`` = the 16x16 sum (`S_cam`); ``joint`` ``(..., k)`` = the sum over the
    three cameras (`S_img`); ``share_of_sum`` ``(..., k)`` = each head's share of the raw sum's total
    mass (``joint / sum_k joint``) — 1/k when the heads are equally heavy; ``dominance`` ``(...)`` =
    the largest share (how much one head owns the raw sum); ``camera_share`` ``(..., k, C)`` = each
    head's split of its own mass over the cameras.
    """
    m = head_maps(block_c, cells)
    per_cam = m.sum(axis=(-2, -1))
    joint = per_cam.sum(axis=-1)
    total = joint.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, joint / np.where(total > 0, total, 1.0), np.nan)
        cam_share = np.where(joint[..., None] > 0,
                             per_cam / np.where(joint[..., None] > 0, joint[..., None], 1.0), np.nan)
    finite = np.isfinite(share)
    dominance = np.where(finite.any(axis=-1), np.where(finite, share, -np.inf).max(axis=-1),
                         np.nan)
    return {"per_camera": per_cam, "joint": joint, "share_of_sum": share,
            "dominance": dominance, "camera_share": cam_share}


def zero_mass_counts(block_c: np.ndarray, cells: Sequence[Sequence[int]]) -> dict[str, int]:
    """How many (sample, head) [joint] and (sample, head, camera) [per camera] maps have zero mass."""
    m = head_maps(block_c, cells)
    per_cam = m.sum(axis=(-2, -1))
    return {"joint": int((per_cam.sum(axis=-1) <= 0).sum()), "per_camera": int((per_cam <= 0).sum())}


def _normalise(m: np.ndarray, mode: str) -> np.ndarray:
    """`m` is one head's ``(..., C, 16, 16)`` raw maps -> divided by the mode's mass (0 -> zeros)."""
    if mode == "mass_norm_joint":
        mass = m.sum(axis=(-3, -2, -1), keepdims=True)
    elif mode == "mass_norm_per_camera":
        mass = m.sum(axis=(-2, -1), keepdims=True)
    else:
        raise ValueError(mode)
    safe = np.where(mass > 0, mass, 1.0)
    return np.where(mass > 0, m / safe, 0.0)


def combine_heads(block_c: np.ndarray, cells: Sequence[Sequence[int]], sigma: float = 0.0, *,
                  mode: str = "raw_sum") -> np.ndarray:
    """Combine the `cells` of ``(..., C, L, H, 16, 16)`` maps into ``(..., C, 16, 16)``.

    ``raw_sum`` **is** `t47_head_metrics.combine_maps` (called, not re-implemented, so it stays the
    T47 value bit for bit). The two mass-normalised modes divide each head first (module docstring),
    then smooth by `sigma` grid cells (`gaussian_smooth`, reflect) and sum element-wise.
    """
    if mode not in AGGREGATIONS:
        raise ValueError(f"mode must be one of {AGGREGATIONS}, got {mode!r}")
    a = _check_block(block_c)
    cells = _cells(cells)
    if mode == "raw_sum":
        return combine_maps(a, cells, sigma)
    out = None
    for l, h in cells:
        m = np.asarray(a[..., l, h, :, :], np.float64)
        if np.any(m < 0):
            raise ValueError(f"cell ({l}, {h}) has negative attention — not softmax mass")
        n = gaussian_smooth(_normalise(m, mode), sigma)
        out = n if out is None else out + n
    return out


def combine_heads_per_camera(block_c: np.ndarray, cells_by_camera: Sequence[Sequence[Sequence[int]]],
                             sigma: float = 0.0, *, mode: str = "raw_sum") -> np.ndarray:
    """Camera c gets its own head set ``cells_by_camera[c]`` (T47's per-camera top-k).

    Each camera's map is `combine_heads(block_c, cells_by_camera[c], sigma, mode)[..., c, :, :]`, so
    ``mass_norm_joint`` still divides a head by **its own mass over all three cameras** (the head's
    768-token mass), even though only camera c's map of it is used. ``raw_sum`` equals
    `t47_head_metrics.combine_maps(block_c[:, c], cells, sigma)` (T47 `topk_camera`).
    """
    a = _check_block(block_c)
    n_c = a.shape[CAMERA_AXIS]
    if len(cells_by_camera) != n_c:
        raise ValueError(f"{len(cells_by_camera)} head sets for {n_c} cameras")
    return np.stack([combine_heads(a, cells_by_camera[c], sigma, mode=mode)[..., c, :, :]
                     for c in range(n_c)], axis=-3)


def head_average_variant(block_c: np.ndarray, sigma: float = 0.0, *, mode: str = "raw_sum",
                         excluded_layers: Sequence[int] = (0, 1)) -> np.ndarray:
    """Mean over every (layer, head) except `excluded_layers` (L2-17 = 128 cells), per `mode`.

    ``raw_sum`` -> `t47_head_metrics.head_average` (T47's head-average, bit for bit at sigma 0), then
    smoothed. ``mass_norm_joint`` / ``mass_norm_per_camera`` -> every head (or head x camera) first
    divided by its own mass, so every head weighs the same in the mean (T47b "범위 변경" 1: "모든
    head 의 mass 가 같아지도록 정규화한 평균"), then smoothed. The mean is the sum / 128 — a constant
    factor the pipeline does not see (B3).
    """
    from AG3S.ag3s.experiments.studies.t47_head_metrics import head_average

    if mode not in AGGREGATIONS:
        raise ValueError(f"mode must be one of {AGGREGATIONS}, got {mode!r}")
    a = _check_block(block_c)
    if mode == "raw_sum":
        return gaussian_smooth(head_average(np.asarray(a, np.float64),
                                            excluded_layers=excluded_layers), sigma)
    ex = set(int(l) for l in excluded_layers)
    cells = [(l, h) for l in range(a.shape[-4]) if l not in ex for h in range(a.shape[-3])]
    out = None
    for l, h in cells:
        m = np.asarray(a[..., l, h, :, :], np.float64)
        if np.any(m < 0):
            raise ValueError(f"cell ({l}, {h}) has negative attention — not softmax mass")
        n = _normalise(m, mode)
        out = n if out is None else out + n
    return gaussian_smooth(out / len(cells), sigma)


def top_camera(maps: np.ndarray) -> np.ndarray:
    """``(..., C, 16, 16)`` -> ``(...)`` index of the camera holding the largest single value.

    The 2D stand-in for which camera wins the raw ``max`` fusion at the hottest point
    (`multiview.py:384` takes a per-voxel max of the lifted raw values). Ties -> lowest index.
    """
    a = np.asarray(maps, np.float64)
    return a.reshape(a.shape[:-2] + (-1,)).max(axis=-1).argmax(axis=-1)


def camera_mass_share(maps: np.ndarray) -> np.ndarray:
    """``(..., C, 16, 16)`` -> ``(..., C)`` each camera's share of the combined map's mass."""
    a = np.asarray(maps, np.float64)
    per = a.sum(axis=(-2, -1))
    tot = per.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, per / np.where(tot > 0, tot, 1.0), np.nan)


__all__ = ["AGGREGATIONS", "AGGREGATION_ROLE", "CAMERA_AXIS", "camera_mass_share", "combine_heads",
           "combine_heads_per_camera", "head_average_variant", "head_maps", "head_masses", "top_camera", "zero_mass_counts"]
