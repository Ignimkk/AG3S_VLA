"""IFG S2/S3 — observed-surface graph, geometry-only (G) and feature+geometry (FG) membership.

Protocol v1.1 §6–7, §12. One frame, three cameras, fused in the robot base frame.

Representation
  pixels   serving `backproject` (integer pixel coordinates, `PointCloudConfig` depth gate) of the
           replay depth; `robot_mask` = the serving FK self-filter decided per pixel (S0).
  surfaces 4-neighbour pixel pairs whose z-depth differs by <= `jump_gate` are on one surface;
           no edge ever crosses a depth jump (protocol §6.3).
  nodes    5 mm base-frame voxels of the non-robot pixels (fused over cameras: a voxel seen by two
           cameras is one node, so multiview does not multiply area). Evaluation uses the voxels of
           *all* valid pixels (robot pixels included) — the same grid, occupancy weight 1 each.
  edges    a node pair is adjacent iff some camera has a same-surface 4-neighbour pixel pair across
           the two voxels and their centroids are within `radius_mult * voxel` (§6.6). No pure 3D
           proximity edge: touching objects connect only where the depth surface is continuous.
  normals  PCA over the node and its 1-ring centroids; unavailable when < 4 points or degenerate.
  features node feature = normalised pixel-count-weighted mix of its patches' unit features; a
           patch feature copied to many nodes adds no new evidence (it is only an edge affinity).

Energy (§7):  min_{0<=m<=1} (m-s)^T Λ (m-s) + λ m^T L m,  L = D - W, solved per connected
component as (Λ + λL) m = Λ s (an M-matrix system; with s in [0,1] the solution stays in [0,1]).
A component with no unary (ΣΛ = 0) is `unknown` (m = NaN).

Unary terms (all bounded in [0,1], provenance kept):
  seed     s=1 on the nodes under the k seed patches; each seed patch's reliability mass
           (w_p / max w) is split over the nodes by their pixel share of the patch (mass conserved;
           a mixed patch spreads its evidence over every surface it covers instead of hard-labelling
           them all).
  support  s=0, Λ=kappa_support on nodes of planes found by the serving RANSAC estimator
           (`fit_support_surfaces`, no GT plane).
  proto    (FG unary arms only) s = clip((cos - c0) / (c1 - c0), 0, 1) with c1 = mean cosine of the
           seed patches, c0 = median cosine of all depth-supported patches of the frame; Λ=kappa_proto.

Arms on one seed (A/N/P) x one feature:
  G       geometry edges (position, normal), unary seed + support
  F       feature-only: node m = proto unary mapping, no graph
  FGe     G + feature affinity in the edges
  FGu     G edges, unary seed + support + proto
  FG      feature affinity in the edges and proto unary (the full S3 method)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

GRID = 16


@dataclasses.dataclass(frozen=True)
class GraphConfig:
    voxel: float = 0.005
    range_max: float = 2.0  # serving `--range-max` default (T53 server_proc); radial, metres
    jump_gate: float = 0.010  # max(10 mm, 3 sigma); sigma uncalibrated (simulated depth)
    radius_mult: float = 3.0
    roi_radius: float = 0.15  # seed-surface ROI; None = full scene (dev control)
    sigma_p_mult: float = 2.0
    sigma_n: float = 0.2
    sigma_f: float = 0.5
    lam: float = 1.0
    kappa_support: float = 1.0
    kappa_proto: float = 0.1
    min_seed_patch_px: int = 60  # a seed patch needs >= 5 % of its 1200 px as non-robot depth
    unknown_low: float = 0.3
    thresholds: tuple = (0.5, 0.7, 0.9)


@dataclasses.dataclass
class FrameGeometry:
    """Pixels → nodes of one frame; arrays over *algorithm* nodes unless named eval_*."""

    pos: np.ndarray  # (N,3) node centroid (base)
    normal: np.ndarray  # (N,3), NaN when unavailable
    patch_px: sp.csr_matrix  # (N,768) pixel counts per patch
    patch_alg_px: np.ndarray  # (768,) non-robot depth pixels per patch
    edges: np.ndarray  # (E,2) i<j
    support: np.ndarray  # (N,) bool
    node_eval: np.ndarray  # (N,) index into eval voxels
    eval_target: np.ndarray  # (V,) target pixel fraction
    eval_class_frac: np.ndarray  # (V,C)
    eval_robot_removed: np.ndarray  # (V,) True if the voxel has no algorithm node
    classes: list
    stats: dict
    eval_pos: np.ndarray = None  # (V,3) centroid of all pixels in the eval voxel


def _cam_pixels(depth, K, T, robot_mask, cam_index, range_max):
    from AG3S.ag3s.config import PointCloudConfig
    from AG3S.ag3s.stages.reconstruction import backproject

    cfg = PointCloudConfig(range_max=range_max)
    pc = backproject(np.asarray(depth, np.float64), K, T, cfg)
    u, v = pc.uv[:, 0].astype(np.int64), pc.uv[:, 1].astype(np.int64)
    H, W = depth.shape
    patch = (v * GRID // H) * GRID + (u * GRID // W) + 256 * cam_index
    return pc.points, u, v, patch, robot_mask[v, u]


def build_geometry(depth, K, T, robot_mask, class_map, classes, cfg: GraphConfig) -> FrameGeometry:
    t0 = time.perf_counter()
    P, U, V, PATCH, ROB, CAM, LAB = [], [], [], [], [], [], []
    for c in range(len(depth)):
        p, u, v, patch, rob = _cam_pixels(depth[c], K[c], T[c], robot_mask[c], c, cfg.range_max)
        P.append(p); U.append(u); V.append(v); PATCH.append(patch); ROB.append(rob)
        CAM.append(np.full(len(p), c, np.int64)); LAB.append(class_map[c][v, u].astype(np.int64))
    P = np.concatenate(P); PATCH = np.concatenate(PATCH); ROB = np.concatenate(ROB)
    CAM = np.concatenate(CAM); LAB = np.concatenate(LAB)
    keys = np.floor(P / cfg.voxel).astype(np.int64)
    keys -= keys.min(0)
    ext = keys.max(0) + 1
    packed = (keys[:, 0] * ext[1] + keys[:, 1]) * ext[2] + keys[:, 2]
    # ---- evaluation voxels: all valid pixels
    ev_keys, ev_inv = np.unique(packed, return_inverse=True)
    nV, C = len(ev_keys), len(classes)
    cnt = np.bincount(ev_inv, minlength=nV).astype(np.float64)
    cls_frac = np.zeros((nV, C))
    np.add.at(cls_frac, (ev_inv, LAB), 1.0)
    cls_frac /= cnt[:, None]
    eval_pos = np.zeros((nV, 3))
    np.add.at(eval_pos, ev_inv, P)
    eval_pos /= cnt[:, None]
    # ---- algorithm nodes: non-robot pixels
    alg = ~ROB
    nd_keys, nd_inv_alg = np.unique(packed[alg], return_inverse=True)
    N = len(nd_keys)
    node_of_px = np.full(len(P), -1, np.int64)
    node_of_px[alg] = nd_inv_alg
    node_eval = np.searchsorted(ev_keys, nd_keys)
    ncnt = np.bincount(nd_inv_alg, minlength=N).astype(np.float64)
    pos = np.zeros((N, 3))
    np.add.at(pos, nd_inv_alg, P[alg])
    pos /= ncnt[:, None]
    patch_px = sp.csr_matrix((np.ones(alg.sum()), (nd_inv_alg, PATCH[alg])), shape=(N, 768))
    patch_px.sum_duplicates()
    patch_alg_px = np.bincount(PATCH[alg], minlength=768).astype(np.float64)
    # ---- edges from same-surface 4-neighbour pixel pairs, per camera
    E = []
    off = 0
    jumps = 0
    for c in range(len(depth)):
        H, W = depth[c].shape
        img = np.full((H, W), -1, np.int64)
        n_c = len(U[c])
        img[V[c], U[c]] = node_of_px[off:off + n_c]
        z = np.asarray(depth[c], np.float64)
        for (a_sl, b_sl) in (((slice(None), slice(0, W - 1)), (slice(None), slice(1, W))),
                             ((slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))):
            na, nb = img[a_sl].ravel(), img[b_sl].ravel()
            ok = (na >= 0) & (nb >= 0) & (na != nb)
            dz = np.abs(z[a_sl].ravel() - z[b_sl].ravel())
            jumps += int((ok & (dz > cfg.jump_gate)).sum())
            ok &= dz <= cfg.jump_gate
            E.append(np.stack([np.minimum(na[ok], nb[ok]), np.maximum(na[ok], nb[ok])], 1))
        off += n_c
    E = np.unique(np.concatenate(E), axis=0) if E else np.zeros((0, 2), np.int64)
    d = np.linalg.norm(pos[E[:, 0]] - pos[E[:, 1]], axis=1)
    E = E[d <= cfg.radius_mult * cfg.voxel]
    # ---- normals: PCA over node + 1-ring
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(N, N))
    A = (A + A.T + sp.identity(N)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    s1 = A @ pos
    s2 = A @ (pos[:, :, None] * pos[:, None, :]).reshape(N, 9)
    mean = s1 / deg[:, None]
    cov = s2.reshape(N, 3, 3) / deg[:, None, None] - mean[:, :, None] * mean[:, None, :]
    w, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0].copy()
    bad = (deg < 4) | (w[:, 1] < 1e-10)
    normal[bad] = np.nan
    # ---- support planes (serving RANSAC estimator, no GT)
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import PointCloud

    _, support = fit_support_surfaces(PointCloud(pos))
    eval_has_node = np.zeros(nV, bool)
    eval_has_node[node_eval] = True
    stats = dict(n_px=int(len(P)), n_px_robot=int(ROB.sum()), n_eval_voxels=int(nV), n_nodes=int(N),
                 n_edges=int(len(E)), n_depth_jump_pairs=jumps, n_normal_unavailable=int(bad.sum()),
                 n_support_nodes=int(support.sum()), ms_build=(time.perf_counter() - t0) * 1000)
    return FrameGeometry(pos, normal, patch_px, patch_alg_px, E, support, node_eval,
                         cls_frac[:, classes.index("target")], cls_frac, ~eval_has_node, list(classes), stats,
                         eval_pos)


def subset(geo: FrameGeometry, keep: np.ndarray) -> FrameGeometry:
    """The ROI sub-graph: nodes `keep`, edges with both ends kept; eval voxels unchanged."""
    idx = np.flatnonzero(keep)
    loc = np.full(len(geo.pos), -1, np.int64)
    loc[idx] = np.arange(idx.size)
    e = geo.edges[keep[geo.edges[:, 0]] & keep[geo.edges[:, 1]]]
    return dataclasses.replace(geo, pos=geo.pos[idx], normal=geo.normal[idx], patch_px=geo.patch_px[idx],
                               edges=loc[e], support=geo.support[idx], node_eval=geo.node_eval[idx])


# ---------------------------------------------------------------------------- evidence


def node_features(geo: FrameGeometry, patch_feat_unit: np.ndarray) -> np.ndarray:
    f = geo.patch_px @ np.asarray(patch_feat_unit, np.float32)
    return (f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def node_patch_mean(geo: FrameGeometry, patch_score: np.ndarray) -> np.ndarray:
    num = geo.patch_px @ np.asarray(patch_score, np.float64)
    return num / np.maximum(np.asarray(geo.patch_px.sum(1)).ravel(), 1e-12)


def seed_unary(geo: FrameGeometry, seeds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-node reliability of s=1: each seed patch's mass (w/max w) split by pixel share."""
    lam = np.zeros(len(geo.pos))
    if seeds.size == 0:
        return lam
    wn = np.asarray(weights, np.float64) / np.max(weights)
    col = geo.patch_px[:, seeds].toarray()  # (N,k)
    share = col / np.maximum(geo.patch_alg_px[seeds], 1e-12)
    return share @ wn


def valid_seed_patches(geo: FrameGeometry, cfg: GraphConfig) -> np.ndarray:
    return geo.patch_alg_px >= cfg.min_seed_patch_px


def proto_unary(cos_patch: np.ndarray, seeds: np.ndarray, valid_patch: np.ndarray):
    c1 = float(np.mean(cos_patch[seeds]))
    c0 = float(np.median(cos_patch[valid_patch]))
    span = max(c1 - c0, 1e-6)
    return np.clip((cos_patch - c0) / span, 0.0, 1.0), dict(c0=c0, c1=c1)


def roi_nodes(geo: FrameGeometry, seed_lam: np.ndarray, radius: Optional[float]) -> np.ndarray:
    if radius is None:
        return np.ones(len(geo.pos), bool)
    sn = seed_lam > 0
    if not sn.any():
        return np.zeros(len(geo.pos), bool)
    d, _ = cKDTree(geo.pos[sn]).query(geo.pos, distance_upper_bound=radius)
    return np.isfinite(d)


def edge_weights(geo: FrameGeometry, cfg: GraphConfig, feat: Optional[np.ndarray]) -> np.ndarray:
    i, j = geo.edges[:, 0], geo.edges[:, 1]
    sp_ = cfg.sigma_p_mult * cfg.voxel
    e = -np.sum((geo.pos[i] - geo.pos[j]) ** 2, 1) / sp_ ** 2
    nd = np.abs(np.sum(geo.normal[i] * geo.normal[j], 1))
    e = e - np.where(np.isfinite(nd), (1.0 - nd) / cfg.sigma_n, 0.0)
    if feat is not None:  # unit rows: ||fi - fj||^2 = 2 - 2 fi.fj
        e = e - np.maximum(2.0 - 2.0 * np.einsum("ij,ij->i", feat[i], feat[j]).astype(np.float64), 0.0) / cfg.sigma_f ** 2
    return np.exp(e)


def solve(geo: FrameGeometry, roi: np.ndarray, w: np.ndarray, lam_u: np.ndarray, s: np.ndarray,
          lam: float) -> tuple[np.ndarray, dict]:
    """Box-constrained Dirichlet energy per component of the ROI graph.
    Returns m in [0,1]; NaN = unknown (component without unary); -1 = outside the ROI (not target)."""
    t0 = time.perf_counter()
    N = len(geo.pos)
    m = np.full(N, -1.0)
    idx = np.flatnonzero(roi)
    if idx.size == 0:
        return m, dict(n_roi=0, n_unknown_components=0, residual=0.0, ms_solve=0.0, box_violation=0.0)
    loc = np.full(N, -1, np.int64)
    loc[idx] = np.arange(idx.size)
    keep = roi[geo.edges[:, 0]] & roi[geo.edges[:, 1]]
    ei, ej, we = loc[geo.edges[keep, 0]], loc[geo.edges[keep, 1]], w[keep]
    n = idx.size
    W = sp.coo_matrix((np.r_[we, we], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    lu, su = lam_u[idx], s[idx]
    ncomp, comp = connected_components(W, directed=False)
    has_u = np.bincount(comp, weights=lu, minlength=ncomp) > 0
    ok = has_u[comp]
    res = 0.0
    viol = 0.0
    if ok.any():
        sel = np.flatnonzero(ok)
        A = (sp.diags(lu[sel]) + lam * L[sel][:, sel]).tocsc()
        b = lu[sel] * su[sel]
        x = spsolve(A, b)
        res = float(np.abs(A @ x - b).max()) if x.size else 0.0
        viol = float(max(0.0, -x.min(), x.max() - 1.0)) if x.size else 0.0
        mm = np.full(n, np.nan)
        mm[sel] = np.clip(x, 0.0, 1.0)
        m[idx] = mm
    return m, dict(n_roi=int(n), n_edges_roi=int(keep.sum()), n_components=int(ncomp),
                   n_unknown_components=int((~has_u).sum()), n_unknown_nodes=int((~ok).sum()),
                   residual=res, box_violation=viol, ms_solve=(time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------- evaluation


NEIGHBOR = ("other_object", "destination")
ROBOT = ("robot", "robot_gripper")


def evaluate(geo: FrameGeometry, m: np.ndarray, tau: float, cfg: GraphConfig) -> dict:
    """Visible-surface metrics on eval voxels (occupancy weight 1, soft target fraction)."""
    V = len(geo.eval_target)
    pred = np.zeros(V, bool)
    unk = np.zeros(V, bool)
    sel = np.isfinite(m) & (m >= tau)
    pred[geo.node_eval[sel]] = True
    u = np.isnan(m) | ((m >= cfg.unknown_low) & (m < tau))
    unk[geo.node_eval[u]] = True
    t = geo.eval_target
    tp = float(t[pred].sum())
    fp = float((1 - t[pred]).sum())
    fn = float(t[~pred].sum())
    tot = float(t.sum())
    cf = geo.eval_class_frac[pred]
    ci = {c: i for i, c in enumerate(geo.classes)}
    out = dict(TP=tp, FP=fp, FN=fn, target_voxels=tot, pred_voxels=int(pred.sum()),
               precision=tp / (tp + fp) if pred.any() else None,
               recall=tp / tot if tot > 0 else None,
               iou=tp / (tp + fp + fn) if tot > 0 else None,
               neighbor_fp=float(cf[:, [ci[c] for c in NEIGHBOR]].sum()),
               robot_fp=float(cf[:, [ci[c] for c in ROBOT]].sum()),
               support_fp=float(cf[:, ci["support"]].sum()),
               background_fp=float(cf[:, ci["background"]].sum()),
               unknown_weight=float(t[unk].sum()),
               target_lost_to_robot_filter=float(t[geo.eval_robot_removed].sum()))
    if tot <= 0:
        out["missing_reason"] = "target_not_visible"
    return out

