"""SL-PE1 — the object under the VLA's own grasp point (2026-11-08, dev-30).

`clustering.close_seed_grounding` (off by default). The policy chunk says where the hand will close: the finger-tip
midpoint at the first row whose gripper command is below the latch threshold (`safe_policy.fingertip_alignment`'s
`plan_close`, recorded as `grasp.tip_alignment.plan_close` since T43). On the GT-free B5 records (gt24att, LIBERO 27 +
SafeLIBERO 9) the nearest true object to that point two calls before the close is the target in 34 / 36 episodes, while the
attention grounding's first identity was another object in 12 of the 16 failures (dev-e5/dev-38 quantification, G102).

The attention grounding loses the target in those scenes because the 3-D partition merges it with the fixture it stands on
(cabinet top, drawer floor + walls: one 15 mm component, too large to grasp, so the bowl is never a candidate). This module
cuts the object out locally around the seed, with no learned part:

1. non-support points within `crop` m of the seed;
2. the local support = among bands of upward-facing points (PCA normal within ~25 deg of vertical, 15 mm thick) whose top
   lies below the seed + `above`, the one covering the most 1 cm x-y cells (>= `plane_cells`, spanning >= `plane_span`) —
   cabinet top / drawer floor / box lid; everything at or below its top (+ `plane_tol`) is dropped;
3. 10 mm-grid 26-connected components; the component nearest to the seed (<= `reach`);
4. if its longest side exceeds `grasp_ext` (object still stuck to a fixture), large flat vertical planes (RANSAC, the
   drawer / cabinet walls) are removed and step 3 is repeated; above `max_ext` only its points within `ball` of the seed
   are kept.

Pure functions, deterministic (fixed RNG seed). Offline evaluation: `outputs/2026-11-08_PE_경로끝점_identity/`.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class CloseSeedParams:
    crop: float = 0.15
    above: float = 0.02
    plane_span: float = 0.10
    plane_cells: int = 40
    plane_tol: float = 0.004
    reach: float = 0.06
    grasp_ext: float = 0.15
    max_ext: float = 0.20
    ball: float = 0.08
    grid: float = 0.010
    min_points: int = 15


def upward_mask(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Points whose local PCA normal is within ~25 deg of vertical (|n_z| >= 0.9)."""
    from scipy.spatial import cKDTree

    P = np.asarray(points, np.float64).reshape(-1, 3)
    if len(P) < k:
        return np.zeros(len(P), bool)
    _, nb = cKDTree(P).query(P, k=k)
    X = P[nb] - P[nb].mean(axis=1, keepdims=True)
    _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", X, X))
    return np.abs(v[:, 2, 0]) >= 0.9


def _xy_cells(q: np.ndarray) -> int:
    return int(len(np.unique(np.floor(q[:, :2] / 0.01).astype(np.int64), axis=0)))


def vertical_walls(P: np.ndarray, *, n_iter: int = 200, tol: float = 0.004, min_inliers: int = 40,
                   min_span: float = 0.10, max_planes: int = 3, seed: int = 0,
                   min_fill: float = 0.4) -> np.ndarray:
    """Mask of points on large flat vertical planes: RANSAC on two points + the vertical axis; a plane counts when it
    spans >= `min_span` horizontally and its inliers fill >= `min_fill` of its 1 cm (along-plane, z) cells."""
    rng = np.random.default_rng(seed)
    wall = np.zeros(len(P), bool)
    for _ in range(max_planes):
        cand = np.flatnonzero(~wall)
        if len(cand) < min_inliers:
            break
        best = None
        for _ in range(n_iter):
            a, b = P[rng.choice(cand, 2, replace=False)]
            t = (b - a)[:2]
            nt = float(np.linalg.norm(t))
            if nt < 0.02:
                continue
            nrm = np.array([-t[1], t[0], 0.0]) / nt
            inl = cand[np.abs((P[cand] - a) @ nrm) <= tol]
            if best is None or len(inl) > len(best[0]):
                best = (inl, t / nt)
        if best is None or len(best[0]) < min_inliers:
            break
        inl, t = best
        u = P[inl][:, :2] @ t
        if float(np.ptp(u)) < min_span:
            break
        # a wall fills its plane; a plane through a bowl only meets two wall lines and a floor stripe
        cells = len(np.unique(np.c_[np.floor(u / 0.01), np.floor(P[inl][:, 2] / 0.01)].astype(np.int64), axis=0))
        area = (np.floor(np.ptp(u) / 0.01) + 1) * (np.floor(np.ptp(P[inl][:, 2]) / 0.01) + 1)
        if cells < min_fill * area:
            break
        wall[inl] = True
    return wall


def _components(P: np.ndarray, grid: float) -> np.ndarray:
    from scipy import ndimage

    lo = P.min(axis=0)
    ijk = np.floor((P - lo) / grid).astype(np.int64)
    occ = np.zeros(tuple(int(x) for x in ijk.max(axis=0) + 1), bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    return lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def split_at_seed(points: np.ndarray, seed, exclude_mask: Optional[np.ndarray] = None,
                  params: CloseSeedParams = CloseSeedParams()) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """Indices (into `points`) of the object under `seed`, or None; and a JSON-able record.

    `exclude_mask`: points never considered (the support-plane points of this frame)."""
    p = params
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    s = np.asarray(seed, np.float64).reshape(3)
    rec: dict[str, Any] = {"seed_m": [round(float(v), 4) for v in s]}
    ok = np.ones(len(pts), bool)
    if exclude_mask is not None and len(np.asarray(exclude_mask).reshape(-1)) == len(pts):
        ok = ~np.asarray(exclude_mask, bool).reshape(-1)
    near = np.flatnonzero(ok & (np.linalg.norm(pts - s, axis=1) <= p.crop))
    rec["n_crop"] = int(len(near))
    if len(near) < p.min_points:
        rec["why"] = "nothing near the seed"
        return None, rec
    P = pts[near]
    up = upward_mask(P)
    zb = np.floor(P[:, 2] / 0.005).astype(np.int64)
    plane_top, best = None, p.plane_cells - 1
    for b in sorted(set(zb[up].tolist())):
        if (b + 2) * 0.005 > s[2] + p.above:
            break
        q = P[up & (np.abs(zb - b) <= 1)]
        cells = _xy_cells(q)
        if cells > best and (np.ptp(q[:, 0]) >= p.plane_span or np.ptp(q[:, 1]) >= p.plane_span):
            plane_top, best = (b + 2) * 0.005, cells
    rec["local_support_top_m"] = None if plane_top is None else round(float(plane_top), 4)
    keep = P[:, 2] > plane_top + p.plane_tol if plane_top is not None else np.ones(len(P), bool)
    idx, P = near[keep], P[keep]
    if len(P) < p.min_points:
        rec["why"] = "nothing above the local support"
        return None, rec
    comp = _components(P, p.grid)
    d = np.linalg.norm(P - s, axis=1)
    j = int(np.argmin(d))
    rec["seed_gap_mm"] = round(float(d[j]) * 1000.0, 1)
    if d[j] > p.reach:
        rec["why"] = "no point within reach of the seed"
        return None, rec
    sel = comp == comp[j]
    rec["n_walls"] = 0
    if float(np.ptp(P[sel], axis=0).max()) > p.grasp_ext:
        wall = vertical_walls(P)
        if wall.any() and int((~wall).sum()) >= p.min_points:
            rec["n_walls"] = int(wall.sum())
            idx, P, d = idx[~wall], P[~wall], d[~wall]
            comp = _components(P, p.grid)
            j = int(np.argmin(d))
            if d[j] > p.reach:
                rec["why"] = "no point within reach of the seed after the walls"
                return None, rec
            sel = comp == comp[j]
    if float(np.ptp(P[sel], axis=0).max()) > p.max_ext:
        sel &= d <= p.ball
        rec["ball_cropped"] = True
    out = idx[sel]
    rec["n_points"] = int(len(out))
    if len(out) < p.min_points:
        rec["why"] = "component too small"
        return None, rec
    rec["extents_mm"] = [round(float(v) * 1000.0, 1) for v in np.ptp(pts[out], axis=0)]
    rec["why"] = None
    return out, rec
