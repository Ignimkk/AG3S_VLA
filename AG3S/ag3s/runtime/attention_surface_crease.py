"""T78d — concave-crease edge cut on a 2.5 mm attention-surface graph (`esdf.surface_crease_cut`, default off), and
the small-component re-attach (`esdf.surface_crease_reattach_ratio`, default None).

`runtime.attention_surface.select_target_surface(..., crease=CreaseParams(...))` (CPU) and
`runtime.attention_surface_gpu.select(..., crease=...)` (GPU) run the locked selection (`ifg_round_ns.g_and_top1`) on a
graph that differs from the LOCK graph in two places only — the study `verify/T78c/t78c_crease.py`
`run_variant(voxel=0.0025, cut=True)` (T78c §8, offline):

| step | LOCK (off) | crease (on) |
|---|---|---|
| node size | 5 mm (`GraphConfig.voxel`) | `voxel_mm` (2.5 mm) — the whole graph (`build_geometry` with `voxel` replaced) |
| edges | same-surface 4-neighbour pairs | the same, minus the **concave creases inside the seed ROI** (below) |
| seed · ROI · solve · region | `g_and_top1` | the same function on the cut graph |

Crease cut (inside the seed ROI, `roi_nodes(seed_unary(top-5 seeds), roi_radius)` — the ROI `g_and_top1` solves on):

1. normal of an ROI node = smallest-eigenvalue eigenvector of the covariance of the ROI node centroids within
   `r = max(2 · voxel, 7.5 mm)` (itself included; `cKDTree.query_pairs(r)`); NaN when fewer than 5 points or degenerate
   (`w[1] < 1e-10`). These normals only decide the cut — the solve's edge weights keep the graph's 1-ring normals.
2. oriented toward the observing cameras: sign of `n · Σ_pixels unit(camera centre − node)`.
3. an ROI edge (i, j) with both normals is cut iff concave, `(n_i − n_j) · (x_i − x_j) < 0`, and
   `angle(n_i, n_j) ≥ angle_deg` (20°).

Re-attach (`reattach_ratio` R, after the top-1 region is chosen; one pass): components of the cut ROI graph over the
non-support nodes that hold no selected node and touch the selected region through a cut edge are added whole when
their node count ≤ R × the selected node count. Intent: a stem · leaf · handle that meets the body in a concave fold.
Limit: a touching neighbour smaller than R × the region is attached too.

T97 — protrusion re-attach (`protrusion`, refused with `reattach_ratio`; `protrude`): the same candidates, attached
only when they rise above the region's top on its body — median height above the ROI support plane (serving RANSAC
on the ROI support nodes) ≥ the region's top-10 % height boundary, projected area ≤ `protrusion_area` ρ × the
region's, and projection overlapping the region's projected convex hull. A side-by-side touching neighbour of the
same height fails the height condition, whatever its size; a side handle (not above the top) stays an obstacle.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional, Sequence

import numpy as np

#: study `t78c_crease.py`: PCA radius floor (m) and the minimum neighbourhood size of a normal
NORMAL_RADIUS_MIN_M = 0.0075
NORMAL_MIN_POINTS = 5
#: re-attach record: at most this many component entries (largest first); the counts cover all
REATTACH_RECORD_MAX = 32
# --- T97 — begin
#: protrusion re-attach (T97 §1.1): default ρ (`surface_crease_protrusion_area`), the region height quantile h_R is
#: taken at (top 10 % boundary), and the fewest ROI support nodes the local support plane is fitted on
PROTRUSION_AREA_DEFAULT = 0.15
PROTRUSION_REGION_QUANTILE = 0.90
PROTRUSION_PLANE_MIN_INLIERS = 30
# --- T97 — end


def _positive_finite(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0.0:
        raise ValueError(f"{name} 는 0 보다 큰 유한한 수여야 합니다: {v!r}")
    return float(v)


@dataclasses.dataclass(frozen=True)
class CreaseParams:
    """The T78d graph change. `voxel_mm` node size, `angle_deg` crease angle, `reattach_ratio` R (None = off)."""

    voxel_mm: float = 2.5
    angle_deg: float = 20.0
    reattach_ratio: Optional[float] = None
    # --- T97 — begin: the protrusion re-attach (False = off; the record has no new key then)
    protrusion: bool = False
    protrusion_area: float = PROTRUSION_AREA_DEFAULT
    # --- T97 — end

    def __post_init__(self):
        _positive_finite("surface_crease_voxel_mm", self.voxel_mm)
        a = _positive_finite("surface_crease_angle_deg", self.angle_deg)
        if a >= 180.0:
            raise ValueError(f"surface_crease_angle_deg 는 180 보다 작아야 합니다: {self.angle_deg!r}")
        if self.reattach_ratio is not None:
            _positive_finite("surface_crease_reattach_ratio", self.reattach_ratio)
        # --- T97 — begin
        if not isinstance(self.protrusion, bool):
            raise ValueError(f"surface_crease_protrusion 는 true 또는 false 여야 합니다: {self.protrusion!r}")
        _positive_finite("surface_crease_protrusion_area", self.protrusion_area)
        if self.protrusion and self.reattach_ratio is not None:
            raise ValueError("surface_crease_protrusion 과 surface_crease_reattach_ratio 는 함께 켤 수 없습니다 "
                             "(두 다시 붙이기 규칙의 효과를 가른다, T97 §1.1)")
        # --- T97 — end

    @property
    def voxel_m(self) -> float:
        return float(self.voxel_mm) * 1e-3

    @property
    def angle_rad(self) -> float:
        return float(np.deg2rad(float(self.angle_deg)))

    @property
    def normal_radius_m(self) -> float:
        return max(2.0 * self.voxel_m, NORMAL_RADIUS_MIN_M)

    def graph_config(self, cfg):
        """`cfg` with the node size replaced (everything else of the LOCK graph unchanged)."""
        return dataclasses.replace(cfg, voxel=self.voxel_m)

    def record(self) -> dict[str, Any]:
        out = {"voxel_mm": float(self.voxel_mm), "angle_deg": float(self.angle_deg),
               "normal_radius_mm": round(self.normal_radius_m * 1e3, 4),
               "reattach_ratio": None if self.reattach_ratio is None else float(self.reattach_ratio)}
        # --- T97 — begin (only when on: the T78d record is unchanged otherwise)
        if self.protrusion:
            out["protrusion_area"] = float(self.protrusion_area)
        # --- T97 — end
        return out


def from_esdf(esdf) -> Optional[CreaseParams]:
    """`CreaseParams` of an `EsdfConfig` (`surface_crease_*`), or None when `surface_crease_cut` is off."""
    if not bool(getattr(esdf, "surface_crease_cut", False)):
        return None
    return CreaseParams(voxel_mm=float(getattr(esdf, "surface_crease_voxel_mm", 2.5)),
                        angle_deg=float(getattr(esdf, "surface_crease_angle_deg", 20.0)),
                        reattach_ratio=getattr(esdf, "surface_crease_reattach_ratio", None),
                        # --- T97 — begin
                        protrusion=bool(getattr(esdf, "surface_crease_protrusion", False)),
                        protrusion_area=float(getattr(esdf, "surface_crease_protrusion_area",
                                                      PROTRUSION_AREA_DEFAULT)))
                        # --- T97 — end


# ------------------------------------------------------------------------------------------------- CPU (study copy)


def normals_pca(pos: np.ndarray, r: float) -> np.ndarray:
    """study `t78c_crease.normals_pca`: PCA normal over the points within `r` (itself included); NaN when < 5 points
    or degenerate."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    n = len(pos)
    if n == 0:
        return np.zeros((0, 3))
    pairs = cKDTree(pos).query_pairs(r, output_type="ndarray")
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    A = (A + A.T + sp.identity(n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    mean = (A @ pos) / deg[:, None]
    cov = (A @ (pos[:, :, None] * pos[:, None, :]).reshape(n, 9)).reshape(n, 3, 3) / deg[:, None, None] \
        - mean[:, :, None] * mean[:, None, :]
    w, vec = np.linalg.eigh(cov)
    nrm = vec[:, :, 0].copy()
    nrm[(deg < NORMAL_MIN_POINTS) | (w[:, 1] < 1e-10)] = np.nan
    return nrm


def orient(nrm: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Flip each normal toward `view` (sum of unit node → camera vectors); NaN normals stay NaN."""
    s = np.sign(np.einsum("ij,ij->i", nrm, view))
    s[s == 0] = 1
    return nrm * s[:, None]


def cut_edges(edges: np.ndarray, pos: np.ndarray, nrm: np.ndarray, in_roi: np.ndarray, angle_rad: float):
    """`(cut (E,) bool, considered (E,) bool)`: ROI edges with both normals; cut = concave and angle ≥ `angle_rad`."""
    a, b = edges[:, 0], edges[:, 1]
    inroi = in_roi[a] & in_roi[b] & np.isfinite(nrm[a, 0]) & np.isfinite(nrm[b, 0])
    with np.errstate(invalid="ignore"):
        conv = np.einsum("ij,ij->i", nrm[a] - nrm[b], pos[a] - pos[b])
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", nrm[a], nrm[b]), -1, 1))
        cut = inroi & (conv < 0) & (ang >= angle_rad)
    return cut, inroi


def cut_cpu(views: Sequence[Any], geo, cfg, smap: np.ndarray, vp: np.ndarray, params: CreaseParams):
    """CPU crease cut (study `run_variant(cut=True)` lines). `geo` = the `build_geometry` of `cfg` (voxel already
    `params.voxel_m`), `vp` = `ifg_round_ns.eligible(geo, cfg, True)`. Returns `(geo with the cut edges removed,
    state)`; `state["cut"]` = the removed edges (graph node ids), `state["roi"]` the ROI mask, `state["record"]`."""
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_s1 as S1
    from AG3S.ag3s.runtime.attention_surface import _pixel_nodes

    N = len(geo.pos)
    rec: dict[str, Any] = {"n_roi_nodes": 0, "n_edges_roi": 0, "n_cut": 0, "n_normal_nan": 0}
    state = {"cut": np.zeros((0, 2), np.int64), "roi": np.zeros(N, bool), "record": rec}
    six = S1.top_k(smap, S1.SEED_K, vp) if N else np.zeros(0, np.int64)
    if not six.size:
        return geo, state
    sw = smap[six] / smap[six].sum()
    roi = G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius)
    idx = np.flatnonzero(roi)
    nrm = np.full((N, 3), np.nan)
    nrm[idx] = normals_pca(geo.pos[idx], params.normal_radius_m)
    pix, _ = _pixel_nodes(views, cfg, geo.pos)   # orient toward the observing cameras
    view = np.zeros((N, 3))
    for v, (u, vv, node) in zip(views, pix):
        ok = node >= 0
        c = np.asarray(v.T_base_cam, float)[:3, 3]
        d = c[None] - geo.pos[node[ok]]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        np.add.at(view, node[ok], d)
    nrm = orient(nrm, view)
    cut, inroi = cut_edges(geo.edges, geo.pos, nrm, roi, params.angle_rad)
    rec.update({"n_roi_nodes": int(idx.size), "n_edges_roi": int(inroi.sum()), "n_cut": int(cut.sum()),
                "n_normal_nan": int((~np.isfinite(nrm[idx, 0])).sum())})
    state.update({"cut": geo.edges[cut], "roi": roi})
    return dataclasses.replace(geo, edges=geo.edges[~cut]), state


# ------------------------------------------------------------------------------------------------- re-attach (host)


def reattach(n: int, edges: np.ndarray, cut: np.ndarray, support: np.ndarray, selected: np.ndarray,
             pos: np.ndarray, ratio: float) -> tuple[np.ndarray, dict]:
    """`(add (n,) bool, record)` on the ROI sub-graph (local ids). `edges` = the kept (uncut) sub-graph edges,
    `cut` = the cut edges (both ends in the ROI), `selected` = the top-1 region (m ≥ 0.5). See the module doc."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    add = np.zeros(n, bool)
    n_sel = int(selected.sum())
    rec: dict[str, Any] = {"ratio": float(ratio), "n_selected_before": n_sel, "n_candidates": 0,
                           "n_attached_components": 0, "n_attached_nodes": 0, "components": []}
    if n == 0 or n_sel == 0 or len(cut) == 0:
        return add, rec
    ns = ~np.asarray(support, bool)
    e = edges[ns[edges[:, 0]] & ns[edges[:, 1]]] if len(edges) else np.zeros((0, 2), np.int64)
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(W, directed=False)
    sel_comp = np.zeros(int(comp.max()) + 1, bool)
    sel_comp[comp[selected]] = True
    a, b = cut[:, 0], cut[:, 1]
    other = np.concatenate([b[selected[a] & ~selected[b]], a[selected[b] & ~selected[a]]])
    other = other[ns[other] & ~sel_comp[comp[other]]]
    cand, n_touch = np.unique(comp[other], return_counts=True)
    if not cand.size:
        return add, rec
    size = np.bincount(comp, minlength=len(sel_comp))[cand]
    limit = float(ratio) * n_sel
    attach = size <= limit
    for c in cand[attach]:
        add |= comp == c
    rec.update({"n_candidates": int(cand.size), "n_attached_components": int(attach.sum()),
                "n_attached_nodes": int(add.sum())})
    order = np.argsort(-size, kind="stable")[:REATTACH_RECORD_MAX]
    for k in order:
        m = comp == cand[k]
        rec["components"].append({"n_nodes": int(size[k]), "ratio": round(float(size[k]) / n_sel, 4),
                                   "n_cut_edges": int(n_touch[k]), "attached": bool(attach[k]),
                                   "centroid": [round(float(x), 4) for x in pos[m].mean(0)]})
    return add, rec


# --- T97 — begin: protrusion re-attach (`esdf.surface_crease_protrusion`, host, CPU · GPU paths alike)


def _cut_candidates(n: int, edges: np.ndarray, cut: np.ndarray, support: np.ndarray, selected: np.ndarray):
    """`(comp (n,) component ids, cand (K,) candidate component ids, n_touch (K,) cut edges into the region)` —
    exactly `reattach`'s candidates: components of the kept non-support ROI graph that hold no selected node and
    touch the selected region through a cut edge."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    ns = ~np.asarray(support, bool)
    e = edges[ns[edges[:, 0]] & ns[edges[:, 1]]] if len(edges) else np.zeros((0, 2), np.int64)
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(W, directed=False)
    sel_comp = np.zeros(int(comp.max()) + 1, bool)
    sel_comp[comp[selected]] = True
    a, b = cut[:, 0], cut[:, 1]
    other = np.concatenate([b[selected[a] & ~selected[b]], a[selected[b] & ~selected[a]]])
    other = other[ns[other] & ~sel_comp[comp[other]]]
    cand, n_touch = np.unique(comp[other], return_counts=True)
    return comp, cand, n_touch


def support_plane(pos: np.ndarray, support: np.ndarray):
    """`(normal (3,), offset, n_inliers)` of the support plane under the region — `stages.support_surface.
    fit_plane_ransac` (the serving RANSAC: same seed · hypotheses · inlier threshold · +z orientation · SVD refit) on
    the ROI's support nodes, size gate `PROTRUSION_PLANE_MIN_INLIERS`. `(None, nan, 0)` when no plane qualifies."""
    from AG3S.ag3s.config import SupportSurfaceConfig
    from AG3S.ag3s.stages.support_surface import fit_plane_ransac

    pts = np.asarray(pos, np.float64)[np.asarray(support, bool)]
    if len(pts) < PROTRUSION_PLANE_MIN_INLIERS:
        return None, float("nan"), 0
    nrm, off, inl = fit_plane_ransac(pts, SupportSurfaceConfig(), min_inliers=PROTRUSION_PLANE_MIN_INLIERS)
    if nrm is None:
        return None, float("nan"), 0
    return np.asarray(nrm, np.float64), float(off), int(len(inl))


def _plane_basis(nrm: np.ndarray) -> np.ndarray:
    """(3, 2) orthonormal in-plane axes of the unit normal `nrm`."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(nrm[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(nrm, ref)
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(nrm, u)], axis=1)


def protrude(n: int, edges: np.ndarray, cut: np.ndarray, support: np.ndarray, selected: np.ndarray,
             pos: np.ndarray, area_ratio: float, voxel_m: float) -> tuple[np.ndarray, dict]:
    """T97 §1.1 — `(add (n,) bool, record)` on the ROI sub-graph (local ids, the same inputs as `reattach`).

    For every `reattach` candidate C (non-support, touches the selected region R only through cut edges):
    heights above the support plane (`support_plane`, normal direction) — `h_C` = median over C's nodes, `h_R` = the
    `PROTRUSION_REGION_QUANTILE` (0.9) quantile over R's nodes (top-10 % boundary); projected areas on that plane —
    occupied `voxel_m` cells × `voxel_m²` (`a_C`, `a_R`); overlap = fraction of C's projected nodes inside R's
    projected convex hull. C is added whole iff `h_C ≥ h_R` ∧ `a_C ≤ area_ratio · a_R` ∧ overlap > 0 (one pass).
    No support plane or a degenerate R hull → nothing is added (record `why`)."""
    add = np.zeros(n, bool)
    selected = np.asarray(selected, bool)
    n_sel = int(selected.sum())
    rec: dict[str, Any] = {"area_ratio": float(area_ratio), "region_quantile": PROTRUSION_REGION_QUANTILE,
                           "n_selected_before": n_sel, "n_candidates": 0, "n_attached_components": 0,
                           "n_attached_nodes": 0, "components": []}
    if n == 0 or n_sel == 0 or len(cut) == 0:
        rec["why"] = "no_region" if n_sel == 0 else "no_cut_edge"
        return add, rec
    comp, cand, n_touch = _cut_candidates(n, edges, np.asarray(cut, np.int64).reshape(-1, 2), support, selected)
    rec["n_candidates"] = int(cand.size)
    if not cand.size:
        rec["why"] = "no_candidate"
        return add, rec
    nrm, off, n_inl = support_plane(pos, support)
    if nrm is None:
        rec.update({"plane": None, "why": "no_support_plane"})
        return add, rec
    rec["plane"] = {"normal": [round(float(x), 5) for x in nrm], "offset_m": round(off, 5), "n_inliers": n_inl,
                    "n_support_roi": int(np.asarray(support, bool).sum())}
    h = pos @ nrm - off
    h_R = float(np.quantile(h[selected], PROTRUSION_REGION_QUANTILE))
    uv = pos @ _plane_basis(nrm)
    cell = np.floor(uv / float(voxel_m)).astype(np.int64)
    a_R = float(len(np.unique(cell[selected], axis=0))) * float(voxel_m) ** 2
    rec.update({"h_region_m": round(h_R, 5), "area_region_cm2": round(a_R * 1e4, 4)})
    # R's projected convex hull (2-D); a degenerate hull (collinear / < 3 points) overlaps nothing
    hull = None
    try:
        from scipy.spatial import ConvexHull, Delaunay

        ruv = uv[selected]
        hull = Delaunay(ruv[ConvexHull(ruv).vertices])
    except Exception as exc:  # noqa: BLE001 — scipy QhullError (degenerate) / too few points: recorded, no overlap
        rec["hull_why"] = f"{type(exc).__name__}"
    # per candidate (nodes grouped by component)
    is_cand = np.zeros(int(comp.max()) + 1, np.int64) - 1
    is_cand[cand] = np.arange(cand.size)
    k_of = is_cand[comp]
    idx = np.flatnonzero(k_of >= 0)
    kk = k_of[idx]
    order = np.argsort(kk, kind="stable")
    idx, kk = idx[order], kk[order]
    bounds = np.searchsorted(kk, np.arange(cand.size + 1))
    inside = (np.zeros(len(idx), bool) if hull is None else hull.find_simplex(uv[idx]) >= 0)
    size = np.diff(bounds)
    h_C = np.empty(cand.size)
    a_C = np.empty(cand.size)
    ov = np.empty(cand.size)
    for k in range(cand.size):
        s_ = slice(bounds[k], bounds[k + 1])
        h_C[k] = float(np.median(h[idx[s_]]))
        a_C[k] = float(len(np.unique(cell[idx[s_]], axis=0))) * float(voxel_m) ** 2
        ov[k] = float(inside[s_].mean())
    attach = (h_C >= h_R) & (a_C <= float(area_ratio) * a_R) & (ov > 0.0)
    for k in np.flatnonzero(attach):
        add[idx[bounds[k]:bounds[k + 1]]] = True
    rec.update({"n_attached_components": int(attach.sum()), "n_attached_nodes": int(add.sum())})
    for k in np.argsort(-size, kind="stable")[:REATTACH_RECORD_MAX]:
        nodes = idx[bounds[k]:bounds[k + 1]]
        rec["components"].append({
            "n_nodes": int(size[k]), "n_cut_edges": int(n_touch[k]),
            "dh_mm": round(float(h_C[k] - h_R) * 1e3, 2),
            "area_ratio": round(float(a_C[k] / a_R), 4) if a_R > 0 else None,
            "overlap": round(float(ov[k]), 4), "attached": bool(attach[k]),
            "rule": {"above": bool(h_C[k] >= h_R), "small": bool(a_C[k] <= float(area_ratio) * a_R),
                     "on_body": bool(ov[k] > 0.0)},
            "centroid": [round(float(x), 4) for x in pos[nodes].mean(0)]})
    return add, rec
# --- T97 — end


__all__ = ["CreaseParams", "cut_cpu", "cut_edges", "from_esdf", "normals_pca", "orient", "protrude", "reattach",
           "support_plane"]
