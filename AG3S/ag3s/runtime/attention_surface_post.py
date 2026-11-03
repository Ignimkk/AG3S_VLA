"""T78 갈래 A — post-processing of the attention-surface region (`esdf.target_extent == "attention_surface"`).

`runtime.attention_surface.select_target_surface` gives one request's code images (1 = the selected surface region,
2 = every other non-robot surface, 0 = robot / invalid). This module works **only on those code images, the request's
depth and its attention maps** — not on the graph internals — so it is the same on the CPU and the GPU selection
(T78 갈래 B) and can be replayed offline from the recorded chunks (`attention_surface_codes_*`).

| step | flag (`esdf.*`, default off) | what |
|---|---|---|
| A1 | `surface_split` | cut a merged region with the existing attention split rule (`stages.attention_split.split_by_attention`: trigger narrowest ≤ max opening < largest, `clustering.attention_split_radius` re-cluster, raw-attention ratio `clustering.attention_split_ratio`, + the attention partition when `clustering.attention_split_partition`) and keep the target component. Not triggered / not split → the region is left whole |
| A2 | `surface_target_check` | the region (after A1) must be this request's ATS target (`target_check`: ≥ 50 % of the region (pixel-weighted) within 10 mm of an ATS point) — else the request abstains (`why = "not_ats_target"`) |
| A3 · A4 · A5 | `surface_keep_last` · `surface_grasp_hold` · `surface_no_ball` | pipeline side (`runtime.pipeline.AG3S`) |

## Region nodes (`region_from_codes`)

The code-1 pixels are lifted with the same pinhole as `stages.reconstruction.backproject` and binned into the graph's
voxel (`GraphConfig.voxel`, 5 mm, absolute `floor(p / voxel)` keys — the graph's own voxels, since every pixel of a
selected node is code 1 and no other pixel is). Node position = mean of its pixels (= the graph's node centroid), node
attention = mean over its pixels of the raw attention of the pixel's patch (`(v·16 // H)·16 + u·16 // W`, camera
order `CAMERA_ORDER`) — the graph's `node_patch_mean` of the seed map.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.runtime.oracle_segmentation import CODE_STATIC, CODE_TARGET

GRID = 16


@dataclasses.dataclass
class Region:
    """The region's graph nodes: `pos` (M, 3) centroid, `att` (M,) raw attention, `n_px` (M,) pixel count, and per
    camera `(name, v, u, node)` of its code-1 pixels (`node` indexes `pos`)."""

    pos: np.ndarray
    att: np.ndarray
    n_px: np.ndarray
    pixels: list

    def __len__(self) -> int:
        return int(len(self.pos))


def region_from_codes(views: Sequence[Any], codes: Mapping[str, np.ndarray], *, voxel: float,
                      attention: Optional[np.ndarray] = None) -> Region:
    """`Region` of the code-1 pixels (module doc). `views` = `SurfaceView`s in `CAMERA_ORDER` (depth in m, `K`,
    `T_base_cam`, `attention` 16 × 16); `attention` = the (768,) seed map (default: the views' maps concatenated)."""
    from AG3S.ag3s.runtime.attention_surface import attention_vector

    if attention is None:
        attention, _ = attention_vector(views)
    smap = None if attention is None else np.asarray(attention, np.float64).reshape(-1)
    P, A, cams = [], [], []
    for c, v in enumerate(views):
        img = codes.get(v.name)
        if img is None:
            continue
        rows, cols = np.nonzero(np.asarray(img) == CODE_TARGET)
        if not len(rows):
            continue
        d = np.asarray(v.depth_m, np.float64)
        K = np.asarray(v.K, np.float64)
        T = np.asarray(v.T_base_cam, np.float64)
        z = d[rows, cols]
        pc = np.empty((len(rows), 3), np.float64)
        pc[:, 0] = (cols - K[0, 2]) * z / K[0, 0]
        pc[:, 1] = (rows - K[1, 2]) * z / K[1, 1]
        pc[:, 2] = z
        P.append(pc @ T[:3, :3].T + T[:3, 3])
        H, W = d.shape
        patch = (rows * GRID // H) * GRID + (cols * GRID // W) + 256 * c
        A.append(np.zeros(len(rows)) if smap is None else smap[patch])
        cams.append((v.name, rows, cols))
    if not P:
        return Region(np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64), [])
    Pc = np.vstack(P)
    Ac = np.concatenate(A)
    c = np.floor(Pc / float(voxel)).astype(np.int64) + (1 << 20)          # ±2^20 voxels per axis (±5 km at 5 mm)
    keys = (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]
    _, inv = np.unique(keys, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    m = int(inv.max()) + 1
    n = np.bincount(inv, minlength=m).astype(np.float64)
    pos = np.stack([np.bincount(inv, weights=Pc[:, k], minlength=m) for k in range(3)], axis=1) / n[:, None]
    att = np.bincount(inv, weights=Ac, minlength=m) / n
    pixels, off = [], 0
    for name, rows, cols in cams:
        pixels.append((name, rows, cols, inv[off:off + len(rows)]))
        off += len(rows)
    return Region(pos, att, n.astype(np.int64), pixels)


def split_region(region: Region, *, max_opening: Optional[float], radius: float, min_points: int,
                 min_ratio: float, partition: bool = False, partition_min_drop: float = 0.03,
                 reattach_hull: Optional[float] = None) -> tuple[np.ndarray, dict]:
    """A1 — `(keep (M,) bool, record)`. The cluster split rule (`split_by_attention`) on the region's nodes with the
    raw node attention as both the mass and the ratio attention. Not triggered (`None`) or not accepted → keep all.

    T78b (`reattach_hull` m, `esdf.surface_reattach_hull_mm`; None = the T78 A1 rule as is): after an accepted split,
    every other component whose centroid lies inside the target component's convex hull dilated by `reattach_hull`
    is kept with the target (`reattach_components`; record `reattach`)."""
    from AG3S.ag3s.stages.attention_split import split_by_attention

    keep = np.ones(len(region), bool)
    rec: dict[str, Any] = {"applied": False, "n_nodes_before": int(len(region))}
    if len(region) < 2:
        rec["reason"] = "too_few_nodes"
        return keep, rec
    dec = split_by_attention(region.pos, region.att, np.arange(len(region)), max_opening=max_opening,
                             radius=float(radius), min_points=int(min_points), min_ratio=float(min_ratio),
                             ratio_attention=region.att, partition=bool(partition),
                             partition_min_drop=float(partition_min_drop))
    if dec is None:
        rec["reason"] = "not_triggered" if max_opening is not None else "max_opening_unknown"
        return keep, rec
    rec["decision"] = dec.record()
    rec["reason"] = dec.reason
    if not dec.accepted:
        return keep, rec
    keep = np.zeros(len(region), bool)
    keep[np.asarray(dec.target.indices, np.int64)] = True
    if reattach_hull is not None:   # T78b
        keep, rec["reattach"] = reattach_components(region, dec.components, float(reattach_hull))
    rec["applied"] = True
    rec["n_nodes_after"] = int(keep.sum())
    return keep, rec


# --- T78b — begin: convex hull of the accepted region (re-attach · hull fill) -------------------------------------
def convex_hull(points: np.ndarray) -> tuple[Optional[dict], str]:
    """`({"equations" (F, 4), "volume" m³, "lo", "hi", "centroid", "n_points", "n_vertices"}, "ok")` of `points` —
    `scipy.spatial.ConvexHull` (qhull), facet `n · x + b ≤ 0` inside with `|n| = 1` — or `(None, why)` for fewer than
    4 points / a flat or degenerate set (`hull_degenerate`)."""
    from scipy.spatial import ConvexHull, QhullError

    p = np.asarray(points, np.float64).reshape(-1, 3)
    if len(p) < 4:
        return None, "too_few_points"
    try:
        h = ConvexHull(p)
    except (QhullError, ValueError):
        return None, "hull_degenerate"
    if not float(h.volume) > 0.0:
        return None, "hull_degenerate"
    return {"equations": np.asarray(h.equations, np.float64), "volume": float(h.volume),
            "lo": p.min(axis=0), "hi": p.max(axis=0), "centroid": p.mean(axis=0), "n_points": int(len(p)),
            "n_vertices": int(len(h.vertices))}, "ok"


def hull_offset_distance(equations: np.ndarray, q: np.ndarray) -> np.ndarray:
    """`max_f (n_f · q + b_f)` (m) of each query point: ≤ 0 inside the hull; `≤ r` ⇔ inside the hull with every facet
    pushed out by `r` (the offset polytope — the Minkowski sum with a ball of radius `r`, slightly larger at the
    vertices). Outside, a lower bound of the Euclidean distance (equal off a facet's interior)."""
    eq = np.asarray(equations, np.float64).reshape(-1, 4)
    q = np.asarray(q, np.float64).reshape(-1, 3)
    if not len(q):
        return np.zeros(0)
    return (q @ eq[:, :3].T + eq[:, 3][None, :]).max(axis=1)


def reattach_components(region: Region, components, margin: float) -> tuple[np.ndarray, dict]:
    """T78b re-attach — `(keep (M,) bool, record)` after an accepted split. `components[0]` is the target (its nodes are
    kept); each other component is kept too when its centroid lies inside the target component's convex hull dilated
    by `margin` m (`hull_offset_distance(centroid) ≤ margin`). A component outside — another object beside the target
    (the banana) — stays dropped. Target hull degenerate → nothing is re-attached (record `why`)."""
    keep = np.zeros(len(region), bool)
    tgt = np.asarray(components[0].indices, np.int64)
    keep[tgt] = True
    hull, why = convex_hull(region.pos[tgt])
    rec: dict[str, Any] = {"margin_mm": round(float(margin) * 1e3, 3), "applied": hull is not None,
                           "components": []}
    if hull is None:
        rec["why"] = why
    else:
        rec["target_hull"] = {"n_vertices": hull["n_vertices"], "volume_cm3": round(hull["volume"] * 1e6, 3)}
    for c in components[1:]:
        idx = np.asarray(c.indices, np.int64)
        cen = region.pos[idx].mean(axis=0)
        row: dict[str, Any] = {"n_nodes": int(len(idx)), "centroid": [round(float(v), 5) for v in cen],
                               "centroid_to_target_mm": round(float(np.linalg.norm(
                                   cen - region.pos[tgt].mean(axis=0))) * 1e3, 1)}
        if hull is None:
            row.update({"attached": False, "hull_dist_mm": None})
        else:
            d = float(hull_offset_distance(hull["equations"], cen[None, :])[0])
            row.update({"attached": bool(d <= float(margin)), "hull_dist_mm": round(d * 1e3, 2)})
            if row["attached"]:
                keep[idx] = True
        rec["components"].append(row)
    rec["n_attached"] = int(sum(r["attached"] for r in rec["components"]))
    rec["n_dropped"] = int(len(rec["components"]) - rec["n_attached"])
    rec["n_nodes_attached"] = int(sum(r["n_nodes"] for r in rec["components"] if r["attached"]))
    return keep, rec


def pixel_points(views: Sequence[Any], region: Region, node_mask: np.ndarray) -> np.ndarray:
    """(K, 3) base-frame points of the region's code-1 pixels on `node_mask` nodes (the pinhole of
    `region_from_codes`) — the samples of the components a split dropped (the hull fill protects their voxels)."""
    by = {v.name: v for v in views}
    node_mask = np.asarray(node_mask, bool)
    out = []
    for name, rows, cols, node in region.pixels:
        m = node_mask[node]
        v = by.get(name)
        if v is None or not m.any():
            continue
        r, c = rows[m], cols[m]
        K = np.asarray(v.K, np.float64)
        T = np.asarray(v.T_base_cam, np.float64)
        z = np.asarray(v.depth_m, np.float64)[r, c]
        pc = np.stack([(c - K[0, 2]) * z / K[0, 0], (r - K[1, 2]) * z / K[1, 1], z], axis=1)
        out.append(pc @ T[:3, :3].T + T[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))
# --- T78b — end


def restrict_codes(codes: Mapping[str, np.ndarray], region: Region, keep: np.ndarray) -> dict[str, np.ndarray]:
    """Copies of `codes` whose code-1 pixels on dropped nodes become `CODE_STATIC` (they are another surface)."""
    out = {k: np.asarray(v).copy() for k, v in codes.items()}
    keep = np.asarray(keep, bool)
    for name, rows, cols, node in region.pixels:
        drop = ~keep[node]
        if drop.any():
            out[name][rows[drop], cols[drop]] = CODE_STATIC
    return out


def subset(region: Region, keep: np.ndarray) -> Region:
    """The region restricted to `keep` nodes (node ids re-numbered)."""
    keep = np.asarray(keep, bool)
    new = np.full(len(region), -1, np.int64)
    new[keep] = np.arange(int(keep.sum()))
    pix = []
    for name, rows, cols, node in region.pixels:
        k = keep[node]
        pix.append((name, rows[k], cols[k], new[node[k]]))
    return Region(region.pos[keep], region.att[keep], region.n_px[keep], pix)


#: A2 overlap rule (T78A impl.md "A2", fixed on the T76 SURF records before the replay): a region node is **on the ATS
#: target** when an ATS point is within this distance (m) …
OVERLAP_RADIUS = 0.010
#: … and the region is the ATS target when at least this pixel-weighted fraction of it is on the ATS target (a majority,
#: as `OracleMemory.instance_of`). T76: correct regions (≥ 50 % apple) with the ATS on the apple, 229 requests — min
#: 0.587, 5th pct 0.662; wrong regions (< 20 % apple), 143 — max 0.175.
OVERLAP_MIN_FRACTION = 0.5


def target_check(region: Region, ats_points: Optional[np.ndarray], *, radius: float = OVERLAP_RADIUS,
                 min_fraction: float = OVERLAP_MIN_FRACTION) -> tuple[bool, dict]:
    """A2 — `(is the ATS target, record)`: the pixel-weighted fraction of region nodes with an ATS target point (the
    manipulated geometry of this request) within `radius` is at least `min_fraction`. The region side is measured,
    not the ATS side: the ATS cluster is fused from all cameras and covers more of the object than one request's
    region (a correct region from the wrist camera alone holds 16–50 % of the ATS points), while a wrong region lies
    off the ATS points whatever its size. No ATS point / empty region → not confirmed.

    Stale ATS points (the manipulated object `occluded`: last observed geometry) are used as they are. They still pass
    a correct region while the object has not moved (T76: 13 / 13 latched-occluded correct regions); they fail it when
    the object was moved after its last observation (T75 211) — then this request removes the last accepted samples
    (A3), never the old place. Keeping the ATS on a moved object is the ATS tracker's job, not this check's."""
    from scipy.spatial import cKDTree

    p = np.zeros((0, 3)) if ats_points is None else np.asarray(ats_points, np.float64).reshape(-1, 3)
    rec: dict[str, Any] = {"rule": "region_on_ats", "radius_mm": round(float(radius) * 1e3, 3),
                           "min_fraction": float(min_fraction), "n_ats_points": int(len(p)),
                           "n_region_nodes": int(len(region))}
    if not len(p):
        rec.update({"passed": False, "why": "no_ats_target"})
        return False, rec
    if not len(region):
        rec.update({"passed": False, "why": "empty_region"})
        return False, rec
    d, _ = cKDTree(p).query(region.pos, k=1)
    w = np.asarray(region.n_px, np.float64)
    w = w / max(float(w.sum()), 1e-12)
    frac = float(w[d <= float(radius)].sum())
    ok = frac >= float(min_fraction)
    rec.update({"fraction": round(frac, 4), "median_mm": round(float(np.median(d)) * 1e3, 2), "passed": bool(ok),
                "centroid_mm": round(float(np.linalg.norm(p.mean(0) - region.pos.mean(0))) * 1e3, 2)})
    if not ok:
        rec["why"] = "no_overlap"
    return ok, rec


def extent_record(region: Region) -> dict:
    """Centroid, AABB extent and principal extents (mm) of the region nodes."""
    from AG3S.ag3s.stages.admissibility import principal_extents

    if not len(region):
        return {"n_nodes": 0}
    p = region.pos
    return {"n_nodes": int(len(p)), "centroid": [round(float(x), 4) for x in p.mean(0)],
            "extent_m": [round(float(x), 4) for x in (p.max(0) - p.min(0))],
            "principal_extents_mm": [round(float(x) * 1e3, 1) for x in principal_extents(p)]}


__all__ = ["OVERLAP_MIN_FRACTION", "OVERLAP_RADIUS", "Region", "convex_hull", "extent_record", "hull_offset_distance",
           "pixel_points", "reattach_components", "region_from_codes", "restrict_codes", "split_region", "subset",
           "target_check"]
