"""T78 B — the attention-surface target extent (`runtime.attention_surface.select_target_surface`) on the GPU.

`select_target_surface(..., device="cuda")` (or `config.ATTENTION_SURFACE_DEVICE = "cuda"`, serve_safe
`--attention-surface-device cuda`) runs this module instead of the locked CPU chain. Default ``"cpu"`` = the locked
chain, bit for bit (this module is never imported).

## What runs where (one request, three cameras)

| step | locked CPU code (`experiments/studies/ifg_*.py`) | here |
|---|---|---|
| back-projection | `ifg_graph._cam_pixels` → `reconstruction.backproject` | `device_recon.backproject_device` (same bits: T38 B4 kernels) |
| voxel keys · nodes | `np.floor(P / voxel)`, `np.unique(packed[alg])` | same ops on the device (`torch.unique`, sorted) — same node order |
| node centroid | `np.add.at` (pixel order) / count | sort by node, `segment_reduce` (sum) / count — **not** the same summation order (≈ 1 ulp) |
| surface edges | 4-neighbour pairs, `|dz| ≤ jump_gate`, `np.unique`, centroid distance ≤ radius | same, packed `min · N + max` keys, `torch.unique` (same lexicographic order) |
| support planes | `fit_support_surfaces(PointCloud(pos))` (numpy RNG hypotheses, inlier counts, SVD refit) | same RNG · same hypotheses (numpy, from the device positions), counts on the device (`dot3`), refit by the 3 × 3 scatter matrix's smallest eigenvector (`np.linalg.eigh`) |
| seed eligibility | `ifg_round_ns.eligible(geo, cfg, True)` | per-patch count of non-support node pixels (integers) |
| seeds | `ifg_s1.top_k` | the same function (host, 768 values) |
| seed unary · ROI | `ifg_graph.seed_unary`, `roi_nodes` (cKDTree, `distance_upper_bound`) | per-pixel shares summed per node; squared distance to the seed nodes `< roi_radius²` |
| normals | PCA over node + 1-ring, all nodes | the same sums for the ROI nodes only (the only ones the solve reads), `np.linalg.eigh` on the host |
| edge weights · solve · region | `ifg_graph.edge_weights`, `ifg_graph.solve` (`spsolve`), `ifg_round.region_gate(top1_only)` | **the same locked functions**, on the host, on the ROI sub-graph |
| code images | `attention_surface._pixel_nodes` (re-derives the pixel → node map) | the build's own pixel → node map, written on the device |

The differences against the CPU chain are therefore floating-point summation order only (centroids, 1-ring sums,
seed shares, refit plane). Node ids are the same when the voxel keys are (back-projection is bit-identical), so a
selected-node set can be compared node by node. `tests/ag3s/test_t78b_attention_surface_gpu.py` and the T76 replay
(`verify/T78B`) measure the agreement (IoU of the selected node sets, code-image pixel agreement).

`keep_debug=True` (the IFG-CL2 hybrid and the IFG-S2SERVE gate read the full `FrameGeometry`) and an evaluator GT
(`class_map`) stay on the CPU chain; `select_target_surface` routes them there and says so in the record.

## Record

The same keys as the CPU record (`ms` = build · solve · codes · total) plus `device = "cuda"` and
`ms_detail` (sub-stages). `pixel_node_keys = "build"` (the map is the build's own, not re-derived).

## T78d — crease (`select(..., crease=CreaseParams)`, default None = everything above unchanged)

The graph is built at `crease.voxel_m` (2.5 mm). After the ROI: radius-PCA normals of the ROI nodes on the device
(`_radius_normals`: voxel-key neighbour lookup, `torch.linalg.eigh`), oriented to the cameras, concave ROI edges with
angle ≥ `crease.angle_deg` cut (`_crease_cut`), and the linear solve on the device (`_solve_device`, Jacobi PCG to a
relative residual of 1e-10; the host `ifg_graph.solve` when it does not converge) — the 2.5 mm ROI is ~3× the 5 mm
one and the host `spsolve` alone took 50–110 ms. The re-attach (`attention_surface_crease.reattach`) runs on the host.
Record: `crease` (+ `crease.solve` = `cg_iter` · `residual`), `ms_detail.crease`.
"""

from __future__ import annotations

import math
import time
import types
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.runtime.oracle_segmentation import CODE_NONE, CODE_STATIC, CODE_TARGET

#: Elements of one (candidate × seed) squared-distance block in the ROI query (float64: 8 B each → 128 MiB).
ROI_BLOCK_ELEMS = 1 << 24


def cuda_device():
    """The CUDA device, or raise (the flag asked for the GPU explicitly)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("attention-surface device 'cuda' but torch sees no CUDA device")
    return torch.device("cuda", torch.cuda.current_device())


def _sync(dev) -> None:
    import torch

    torch.cuda.synchronize(dev)


# ------------------------------------------------------------------------------------------- support planes


def _plane_counts(pts, normals: np.ndarray, offsets: np.ndarray, threshold: float, dev):
    """`support_surface._hypothesis_counts` on resident points: ``#{i : |p_i · n_j − d_j| ≤ threshold}`` (M,)."""
    import torch

    from AG3S.ag3s.stages import device_recon

    dot3 = device_recon.kernel("dot3")
    Nm = torch.as_tensor(np.ascontiguousarray(normals), device=dev)
    off = torch.as_tensor(np.ascontiguousarray(offsets), device=dev)
    m = int(Nm.shape[0])
    counts = torch.empty(m, dtype=torch.int64, device=dev)
    px, py, pz = (pts[:, k:k + 1] for k in range(3))
    for start in range(0, m, 64):
        stop = min(m, start + 64)
        nb = Nm[start:stop]
        d = dot3(px, py, pz, nb[:, 0][None], nb[:, 1][None], nb[:, 2][None])   # (N, b)
        d.sub_(off[start:stop][None])
        d.abs_()
        counts[start:stop] = (d <= threshold).sum(dim=0)
    return counts.cpu().numpy()


def _plane_inliers(pts, normal: np.ndarray, offset: float, threshold: float, dev):
    """``|p · n − d| ≤ threshold`` (N,) bool on the device."""
    import torch

    from AG3S.ag3s.stages import device_recon

    dot3 = device_recon.kernel("dot3")
    n = torch.as_tensor(np.asarray(normal, np.float64).reshape(1, 3), device=dev)
    d = dot3(pts[:, 0], pts[:, 1], pts[:, 2], n[:, 0], n[:, 1], n[:, 2])
    return torch.abs(d - float(offset)) <= float(threshold)


def _refit(pts_in, dev) -> tuple[np.ndarray, float]:
    """`support_surface._refit` (total least squares) via the 3 × 3 scatter matrix: the smallest eigenvector of
    ``(X − c)ᵀ (X − c)`` is the SVD's last right singular vector (up to sign; `_orient` and the |·| inlier test do
    not see the sign)."""
    c = pts_in.mean(dim=0)
    X = pts_in - c
    C = (X.T @ X).cpu().numpy()
    centroid = c.cpu().numpy()
    _, vec = np.linalg.eigh(C)
    normal = vec[:, 0]
    return normal, float(normal @ centroid)


def _fit_plane(pts, scfg, seed: int, min_inliers: int, dev):
    """`support_surface.fit_plane_ransac` on resident points → local inlier mask (device bool) or None."""
    from AG3S.ag3s.stages import support_surface as SS

    n = int(pts.shape[0])
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    tri = rng.integers(0, n, size=(scfg.max_iterations, 3))
    distinct = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[distinct]
    if tri.shape[0] == 0:
        return None
    import torch

    tp = pts[torch.as_tensor(tri.reshape(-1), device=dev)].cpu().numpy().reshape(-1, 3, 3)
    normals, offsets = SS._plane_from_points(tp)
    reference = np.asarray(scfg.normal_reference, np.float64)
    reference = reference / max(float(np.linalg.norm(reference)), SS._EPS)
    cos_limit = float(np.cos(np.deg2rad(scfg.max_normal_angle_deg)))
    aligned = np.abs(normals @ reference) >= cos_limit
    normals, offsets = normals[aligned], offsets[aligned]
    if normals.shape[0] == 0:
        return None
    counts = _plane_counts(pts, normals, offsets, float(scfg.distance_threshold), dev)
    best = int(np.argmax(counts))
    if int(counts[best]) < int(min_inliers):
        return None
    inl = _plane_inliers(pts, normals[best], float(offsets[best]), float(scfg.distance_threshold), dev)
    normal, offset = _refit(pts[inl], dev)
    return _plane_inliers(pts, normal, offset, float(scfg.distance_threshold), dev)


def support_mask(pos, dev):
    """`fit_support_surfaces(PointCloud(pos))[1]` (default `SupportSurfaceConfig`) on the device: (N,) bool."""
    import torch

    from AG3S.ag3s.config import SupportSurfaceConfig

    scfg = SupportSurfaceConfig()
    N = int(pos.shape[0])
    mask = torch.zeros(N, dtype=torch.bool, device=dev)
    if not scfg.enabled or N == 0:
        return mask
    size_gate = max(scfg.min_inliers, int(scfg.min_inlier_ratio * N))
    remaining = torch.arange(N, device=dev)
    for plane_id in range(scfg.max_planes):
        if int(remaining.shape[0]) < 3:
            break
        local = _fit_plane(pos[remaining], scfg, scfg.seed + plane_id, size_gate, dev)
        if local is None or not bool(local.any()):
            break
        mask[remaining[local]] = True
        remaining = remaining[~local]
    return mask


# ------------------------------------------------------------------------------------------- the graph


def _segment_sum(values, inverse, counts):
    """Per-segment sums of `values` (rows) grouped by `inverse` (0..S-1, `counts` rows each) — deterministic
    (stable sort + `segment_reduce`), not `np.add.at`'s sequential order."""
    import torch

    order = torch.argsort(inverse, stable=True)
    return torch.segment_reduce(values[order], "sum", lengths=counts, axis=0, unsafe=True)


def build(views, cfg, dev):
    """The surface graph of `ifg_graph.build_geometry` (algorithm nodes only) on the device.

    Returns a namespace (device tensors unless noted) or raises ValueError when no pixel is valid."""
    import torch

    from AG3S.ag3s.config import PointCloudConfig
    from AG3S.ag3s.stages import device_recon
    from AG3S.ag3s.stages.reconstruction import backproject

    GRID = 16
    pcfg = PointCloudConfig(range_max=cfg.range_max)
    t = time.perf_counter()
    P, U, V, PATCH, ROB, depth_dev, hw = [], [], [], [], [], [], []
    for c, v in enumerate(views):
        d = np.asarray(v.depth_m, np.float64)
        H, W = d.shape
        dd = device_recon.upload_depth(d, dev)
        rm = torch.as_tensor(np.ascontiguousarray(np.asarray(v.robot_mask, bool)), device=dev)
        pc = device_recon.backproject_device(dd, np.asarray(v.K, np.float64), np.asarray(v.T_base_cam, np.float64),
                                             pcfg)
        if pc is None:   # < 2 points: numpy (a 1-point cloud is rotated by gemv there)
            host = backproject(d, np.asarray(v.K, np.float64), np.asarray(v.T_base_cam, np.float64), pcfg)
            pts = torch.as_tensor(np.asarray(host.points, np.float64).reshape(-1, 3), device=dev)
            uv = np.asarray(host.uv if host.uv is not None else np.zeros((0, 2)), np.int64).reshape(-1, 2)
            u = torch.as_tensor(uv[:, 0], device=dev)
            vv = torch.as_tensor(uv[:, 1], device=dev)
        else:
            pts, u, vv = pc.points, pc.cols, pc.rows
        P.append(pts); U.append(u); V.append(vv)
        PATCH.append((vv * GRID // H) * GRID + (u * GRID // W) + 256 * c)
        ROB.append(rm[vv, u])
        depth_dev.append(dd); hw.append((H, W))
    sizes = [int(u.shape[0]) for u in U]
    P = torch.cat(P); PATCH = torch.cat(PATCH); ROB = torch.cat(ROB)
    if int(P.shape[0]) == 0:
        raise ValueError("zero-size array to reduction operation minimum which has no identity")
    vs = torch.tensor(float(cfg.voxel), dtype=torch.float64, device=dev)
    keys = torch.floor(P / vs).to(torch.int64)
    keys -= keys.min(dim=0).values
    ext = (keys.max(dim=0).values + 1).tolist()
    packed = (keys[:, 0] * int(ext[1]) + keys[:, 1]) * int(ext[2]) + keys[:, 2]
    alg = ~ROB
    nd_keys, inv = torch.unique(packed[alg], sorted=True, return_inverse=True)
    N = int(nd_keys.shape[0])
    node_of_px = torch.full((int(P.shape[0]),), -1, dtype=torch.int64, device=dev)
    node_of_px[alg] = inv
    ncnt = torch.bincount(inv, minlength=N)
    pos = _segment_sum(P[alg], inv, ncnt) / ncnt.to(torch.float64)[:, None] if N else P[:0]
    patch_alg = PATCH[alg]
    patch_alg_px = torch.bincount(patch_alg, minlength=768)
    _sync(dev)
    ms = {"nodes": (time.perf_counter() - t) * 1e3}
    # ---- edges from same-surface 4-neighbour pixel pairs, per camera
    t = time.perf_counter()
    E, off = [], 0
    for c in range(len(views)):
        H, W = hw[c]
        img = torch.full((H, W), -1, dtype=torch.int64, device=dev)
        img[V[c], U[c]] = node_of_px[off:off + sizes[c]]
        z = depth_dev[c]
        for na, nb, za, zb in ((img[:, :-1], img[:, 1:], z[:, :-1], z[:, 1:]),
                               (img[:-1, :], img[1:, :], z[:-1, :], z[1:, :])):
            ok = (na >= 0) & (nb >= 0) & (na != nb) & (torch.abs(za - zb) <= cfg.jump_gate)
            a, b = na[ok], nb[ok]
            E.append(torch.minimum(a, b) * N + torch.maximum(a, b))
        off += sizes[c]
    Ek = torch.unique(torch.cat(E), sorted=True) if E else torch.zeros(0, dtype=torch.int64, device=dev)
    ei, ej = Ek // max(N, 1), Ek % max(N, 1)
    dvec = pos[ei] - pos[ej]
    s = dvec[:, 0] * dvec[:, 0]
    s += dvec[:, 1] * dvec[:, 1]
    s += dvec[:, 2] * dvec[:, 2]
    keep = torch.sqrt(s) <= cfg.radius_mult * cfg.voxel
    ei, ej = ei[keep], ej[keep]
    _sync(dev)
    ms["edges"] = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    support = support_mask(pos, dev)
    _sync(dev)
    ms["support"] = (time.perf_counter() - t) * 1e3
    return types.SimpleNamespace(N=N, pos=pos, ncnt=ncnt, node_of_px=node_of_px, inv=inv, alg=alg,
                                 patch_alg=patch_alg, patch_alg_px=patch_alg_px, ei=ei, ej=ej, support=support,
                                 U=U, V=V, sizes=sizes, hw=hw, n_px=int(P.shape[0]), n_px_robot=int(ROB.sum()),
                                 ms=ms, nd_keys=nd_keys, ext=ext)   # nd_keys · ext: T78d (radius neighbours)


def _roi(pos, seed_nodes, radius: float, dev):
    """`ifg_graph.roi_nodes`: nodes with a seed node closer than `radius` (cKDTree `distance_upper_bound`)."""
    import torch

    N = int(pos.shape[0])
    roi = torch.zeros(N, dtype=torch.bool, device=dev)
    S = pos[seed_nodes]
    if int(S.shape[0]) == 0:
        return roi
    lo, hi = S.min(dim=0).values - radius, S.max(dim=0).values + radius
    cand = torch.nonzero(((pos >= lo) & (pos <= hi)).all(dim=1)).reshape(-1)
    if int(cand.shape[0]) == 0:
        return roi
    r2 = float(radius) * float(radius)
    C = pos[cand]
    step = max(1, ROI_BLOCK_ELEMS // int(S.shape[0]))
    hit = torch.zeros(int(C.shape[0]), dtype=torch.bool, device=dev)
    for a in range(0, int(C.shape[0]), step):
        blk = C[a:a + step]
        d2 = torch.zeros((int(blk.shape[0]), int(S.shape[0])), dtype=torch.float64, device=dev)
        for k in range(3):
            diff = blk[:, k:k + 1] - S[:, k][None]
            d2 += diff * diff
        hit[a:a + step] = (d2 < r2).any(dim=1)
    roi[cand[hit]] = True
    return roi


def _roi_normals(g, roi_idx, dev, eigh_device: bool = False):
    """`build_geometry`'s PCA normals (node + 1-ring over the radius-filtered surface edges) for the ROI nodes.
    `eigh_device` (T78d crease path only; default False = the host `np.linalg.eigh` as before): `torch.linalg.eigh`
    on the device — the 2.5 mm ROI has ~3× the nodes and the host eigh took 16–27 ms."""
    import torch

    N, n = g.N, int(roi_idx.shape[0])
    loc = torch.full((N,), -1, dtype=torch.int64, device=dev)
    loc[roi_idx] = torch.arange(n, device=dev)
    dst = torch.cat([g.ei, g.ej]); src = torch.cat([g.ej, g.ei])
    m = loc[dst] >= 0
    dl, sp_ = loc[dst[m]], src[m]
    p = g.pos[roi_idx]
    deg = torch.ones(n, dtype=torch.float64, device=dev).index_add_(0, dl, torch.ones(int(dl.shape[0]),
                                                                                       dtype=torch.float64, device=dev))
    s1 = p.clone().index_add_(0, dl, g.pos[sp_])
    outer = lambda q: (q[:, :, None] * q[:, None, :]).reshape(-1, 9)
    s2 = outer(p).index_add_(0, dl, outer(g.pos[sp_]))
    mean = s1 / deg[:, None]
    cov = s2.reshape(n, 3, 3) / deg[:, None, None] - mean[:, :, None] * mean[:, None, :]
    if eigh_device:   # T78d
        w_d, vec_d = torch.linalg.eigh(cov)
        normal_d = vec_d[:, :, 0].clone()
        normal_d[(deg < 4) | (w_d[:, 1] < 1e-10)] = float("nan")
        return normal_d.cpu().numpy(), loc
    cov_h, deg_h = cov.cpu().numpy(), deg.cpu().numpy()
    w, vec = np.linalg.eigh(cov_h)
    normal = vec[:, :, 0].copy()
    bad = (deg_h < 4) | (w[:, 1] < 1e-10)
    normal[bad] = np.nan
    return normal, loc


# --- T78d — begin: concave-crease cut on the device (`attention_surface_crease`, study `t78c_crease.run_variant`)
#: offsets per radius-neighbour query block (× ROI nodes int64 keys)
CREASE_OFFSET_BLOCK = 49


def _radius_normals(g, roi_idx, r: float, voxel: float, dev):
    """`attention_surface_crease.normals_pca(pos[roi], r)` on the device: the ROI node centroids within `r` of each
    ROI node (itself included) found through the nodes' voxel keys (|Δkey| ≤ ceil(r / voxel) per axis), covariance
    sums by `index_add_`, `torch.linalg.eigh`. NaN when < 5 points or degenerate. Returns (n, 3) float64 (device)."""
    import torch

    from AG3S.ag3s.runtime.attention_surface_crease import NORMAL_MIN_POINTS

    n = int(roi_idx.shape[0])
    if n == 0:
        return torch.zeros((0, 3), dtype=torch.float64, device=dev)
    e1, e2 = int(g.ext[1]), int(g.ext[2])
    nk = g.nd_keys[roi_idx]
    k = torch.stack([nk // (e1 * e2), (nk // e2) % e1, nk % e2], dim=1)
    M = max(1, int(math.ceil(r / voxel - 1e-9)))
    k = k - k.min(dim=0).values + M
    E = (k.max(dim=0).values + M + 1).tolist()
    key = (k[:, 0] * int(E[1]) + k[:, 1]) * int(E[2]) + k[:, 2]
    order = None
    if n > 1 and not bool((key[1:] > key[:-1]).all()):   # roi_idx ascending keeps the lexicographic order
        key, order = torch.sort(key)
    p = g.pos[roi_idx] if order is None else g.pos[roi_idx][order]
    rng = torch.arange(-M, M + 1, device=dev)
    off = torch.cartesian_prod(rng, rng, rng)
    off = (off[:, 0] * int(E[1]) + off[:, 1]) * int(E[2]) + off[:, 2]
    r2 = float(r) * float(r)
    deg = torch.zeros(n, dtype=torch.float64, device=dev)
    s1 = torch.zeros((n, 3), dtype=torch.float64, device=dev)
    s2 = torch.zeros((n, 9), dtype=torch.float64, device=dev)
    rows = torch.arange(n, device=dev)
    for a in range(0, int(off.shape[0]), CREASE_OFFSET_BLOCK):
        q = (key[:, None] + off[None, a:a + CREASE_OFFSET_BLOCK]).reshape(-1)
        j = torch.searchsorted(key, q).clamp_(max=n - 1)
        hit = key[j] == q
        i = rows.repeat_interleave(int(min(CREASE_OFFSET_BLOCK, int(off.shape[0]) - a)))[hit]
        j = j[hit]
        d = p[i] - p[j]
        ok = (d * d).sum(dim=1) <= r2
        i, j = i[ok], j[ok]
        pj = p[j]
        deg.index_add_(0, i, torch.ones(int(i.shape[0]), dtype=torch.float64, device=dev))
        s1.index_add_(0, i, pj)
        s2.index_add_(0, i, (pj[:, :, None] * pj[:, None, :]).reshape(-1, 9))
    mean = s1 / deg[:, None]
    cov = s2.reshape(n, 3, 3) / deg[:, None, None] - mean[:, :, None] * mean[:, None, :]
    w, vec = torch.linalg.eigh(cov)
    nrm = vec[:, :, 0].clone()
    nrm[(deg < NORMAL_MIN_POINTS) | (w[:, 1] < 1e-10)] = float("nan")
    if order is not None:
        out = torch.empty_like(nrm)
        out[order] = nrm
        nrm = out
    return nrm


def _crease_cut(g, views, roi_idx, loc, ek, crease, dev):
    """`(keep (E_roi,) bool device, rec)` over the ROI edges `g.ei[ek]`, `g.ej[ek]` — the study's cut on the device:
    radius-PCA normals (`_radius_normals`) oriented to the cameras (Σ over cameras of the node's pixel count × the unit
    node → camera vector), cut = concave and angle ≥ `crease.angle_deg`."""
    import torch

    n = int(roi_idx.shape[0])
    nrm = _radius_normals(g, roi_idx, crease.normal_radius_m, crease.voxel_m, dev)
    p = g.pos[roi_idx]
    view = torch.zeros((n, 3), dtype=torch.float64, device=dev)
    off = 0
    for c, v in enumerate(views):
        node = g.node_of_px[off:off + g.sizes[c]]
        off += g.sizes[c]
        cnt = torch.bincount(node[node >= 0], minlength=g.N)[roi_idx].to(torch.float64)
        cc = torch.as_tensor(np.asarray(v.T_base_cam, np.float64)[:3, 3], device=dev)
        d = cc[None] - p
        d = d / torch.linalg.norm(d, dim=1, keepdim=True)
        view += cnt[:, None] * d
    s = torch.sign((nrm * view).sum(dim=1))
    s[s == 0] = 1
    nrm = nrm * s[:, None]
    a, b = loc[g.ei[ek]], loc[g.ej[ek]]
    fin = torch.isfinite(nrm[:, 0])
    considered = fin[a] & fin[b]
    conv = ((nrm[a] - nrm[b]) * (p[a] - p[b])).sum(dim=1)
    ang = torch.arccos(torch.clamp((nrm[a] * nrm[b]).sum(dim=1), -1, 1))
    cut = considered & (conv < 0) & (ang >= crease.angle_rad)
    cnt = torch.stack([considered.sum(), cut.sum(), (~fin).sum()]).tolist()
    return ~cut, {"n_roi_nodes": n, "n_edges_roi": int(cnt[0]), "n_cut": int(cnt[1]), "n_normal_nan": int(cnt[2])}


#: T78d solve on the device (`_solve_device`): Jacobi-preconditioned CG to ‖r‖ ≤ CG_RTOL ‖b‖, convergence read every
#: CG_BLOCK iterations, host `spsolve` (the locked `ifg_graph.solve`) when CG_MAX_ITER is reached.
CG_RTOL = 1e-10
CG_BLOCK = 32
CG_MAX_ITER = 20000


def _pcg(A, dinv, b, dev):
    """Jacobi PCG for `A x = b` (A = the full CSR system, `dinv` = 1 / its diagonal) from x = 0. On CUDA one block of
    `CG_BLOCK` iterations is captured as a CUDA graph and replayed (the per-op launch cost is most of an eager
    iteration); elsewhere it runs eagerly. Returns `(x, iterations, converged)`."""
    import torch

    k = int(b.shape[0])
    x = torch.zeros(k, dtype=torch.float64, device=dev)
    bn = float(torch.linalg.norm(b))
    if bn == 0.0:
        return x, 0, True
    r, z, p = b.clone(), torch.empty_like(b), torch.empty_like(b)
    rz, rz2 = (torch.empty(1, dtype=torch.float64, device=dev) for _ in range(2))

    def reset():
        x.zero_()
        r.copy_(b)
        torch.mul(r, dinv, out=z)
        p.copy_(z)
        rz.copy_(torch.dot(r, z))

    def step():
        Ap = A @ p
        al = rz / torch.dot(p, Ap)
        x.addcmul_(al, p)
        r.addcmul_(al, Ap, value=-1.0)
        torch.mul(r, dinv, out=z)
        rz2.copy_(torch.dot(r, z))
        p.mul_(rz2 / rz).add_(z)
        rz.copy_(rz2)

    graph = None
    if dev.type == "cuda":
        reset()
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        graph = torch.cuda.CUDAGraph()
        # `capture_begin` / `capture_end` directly: the `torch.cuda.graph` context also runs `gc.collect()` and
        # `empty_cache()` on entry (tens of ms per request in a server process).
        with torch.cuda.stream(side):
            for _ in range(3):   # warm the kernels (cuSPARSE workspace) before the capture
                step()
            graph.capture_begin()
            try:
                for _ in range(CG_BLOCK):
                    step()
            finally:
                graph.capture_end()
        torch.cuda.current_stream(dev).wait_stream(side)
    reset()
    it, conv = 0, False
    while not conv and it < CG_MAX_ITER:
        if graph is not None:
            graph.replay()
        else:
            for _ in range(CG_BLOCK):
                step()
        it += CG_BLOCK
        conv = float(torch.linalg.norm(r)) <= CG_RTOL * bn
    return x, it, conv


def _solve_device(sub, w: np.ndarray, lam_u: np.ndarray, s: np.ndarray, lam: float, dev):
    """`ifg_graph.solve(sub, all-ROI, w, lam_u, s, lam)` with the linear solve on the device: the same components
    (host `connected_components`), the same system `(Λ + λL) m = Λ s` on the components with a unary, NaN on the
    others, clip to [0, 1] — but solved by PCG instead of SuperLU (equal to the tolerance, not bit for bit).
    Returns `(m, info)` with `info["cg_iter"]` (or `info["cg_fallback"]`)."""
    import scipy.sparse as sp
    import torch
    from scipy.sparse.csgraph import connected_components

    from AG3S.ag3s.experiments.studies import ifg_graph as G

    t0 = time.perf_counter()
    n = len(sub.pos)
    m = np.full(n, -1.0)
    ei, ej = sub.edges[:, 0], sub.edges[:, 1]
    W = sp.coo_matrix((np.r_[w, w], (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()
    ncomp, comp = connected_components(W, directed=False)
    has_u = np.bincount(comp, weights=lam_u, minlength=ncomp) > 0
    ok = has_u[comp]
    info = dict(n_roi=int(n), n_edges_roi=int(len(ei)), n_components=int(ncomp),
                n_unknown_components=int((~has_u).sum()), n_unknown_nodes=int((~ok).sum()), residual=0.0,
                box_violation=0.0, cg_iter=0)
    sel = np.flatnonzero(ok)
    if sel.size:
        k = sel.size
        loc = np.full(n, -1, np.int64)
        loc[sel] = np.arange(k)
        Wk = W[sel][:, sel].tocsr() if k < n else W
        Wk.sort_indices()
        deg = np.asarray(Wk.sum(1)).ravel()
        diag = lam_u[sel] + lam * deg
        b_h = lam_u[sel] * s[sel]
        f64 = dict(dtype=torch.float64, device=dev)
        import warnings

        Ak = (sp.diags(diag) - lam * Wk).tocsr()   # = (Λ + λL) on the components with a unary
        Ak.sort_indices()
        with warnings.catch_warnings():   # "sparse CSR support is in beta"
            warnings.simplefilter("ignore", UserWarning)
            A = torch.sparse_csr_tensor(torch.as_tensor(Ak.indptr.astype(np.int64), device=dev),
                                        torch.as_tensor(Ak.indices.astype(np.int64), device=dev),
                                        torch.as_tensor(Ak.data, **f64), size=(k, k))
        b = torch.as_tensor(b_h, **f64)
        x, it, conv = _pcg(A, torch.as_tensor(1.0 / diag, **f64), b, dev)
        if conv:
            xh = x.cpu().numpy()
            res = Ak @ xh - b_h
            info.update(cg_iter=int(it), residual=float(np.abs(res).max()) if k else 0.0,
                        box_violation=float(max(0.0, -xh.min(), xh.max() - 1.0)) if k else 0.0)
            mm = np.full(n, np.nan)
            mm[sel] = np.clip(xh, 0.0, 1.0)
            m = mm
        else:   # not converged: the locked host solve
            m, info2 = G.solve(sub, np.ones(n, bool), w, lam_u, s, lam)
            info.update(info2)
            info["cg_fallback"] = int(it)
    info["ms_solve"] = (time.perf_counter() - t0) * 1e3
    return m, info
# --- T78d — end


def select(views: Sequence[Any], cfg, smap: np.ndarray, *, arm: str, codes: bool = True,
           device=None, rec: Optional[dict] = None, t0: Optional[float] = None, crease=None):
    """`select_target_surface` (keep_debug False) on the GPU. `views` in `CAMERA_ORDER`, `smap` = the (768,)
    attention vector (both checked by the caller). Returns `attention_surface.SurfaceResult`.

    `crease` (T78d, `attention_surface_crease.CreaseParams`, None = unchanged): `cfg` must already carry
    `crease.voxel_m` (`select_target_surface` does that); the ROI edges are cut at concave creases before the solve
    and `crease.reattach_ratio` re-attaches small cut-off components (record ``crease``)."""
    import torch

    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_round as RD
    from AG3S.ag3s.experiments.studies import ifg_run as RN
    from AG3S.ag3s.experiments.studies import ifg_s1 as S1
    from AG3S.ag3s.runtime.attention_surface import TAU, SurfaceResult

    t0 = time.perf_counter() if t0 is None else t0
    dev = cuda_device() if device is None else device
    rec = {"applied": False, "arm": arm, "cameras": [v.name for v in views], "ms": {}} if rec is None else rec
    rec["device"] = "cuda"
    det: dict[str, float] = {}

    def done(res):
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        rec["ms_detail"] = {k: round(v, 3) for k, v in det.items()}
        return res

    t = time.perf_counter()
    try:
        g = build(views, cfg, dev)
    except ValueError as exc:
        rec["why"] = f"no_depth_points ({exc})"
        return done(SurfaceResult(None, rec))
    det.update({f"build_{k}": v for k, v in g.ms.items()})
    rec["ms"]["build"] = round((time.perf_counter() - t) * 1e3, 3)
    rec["n_nodes"] = int(g.N)
    rec["n_support_nodes"] = int(g.support.sum())
    rec["n_robot_px"] = int(g.n_px_robot)
    # ---- seeds (`ifg_round_ns.eligible(geo, cfg, True)`, `ifg_s1.top_k`)
    t = time.perf_counter()
    nonsup = ~g.support[g.inv]
    vp = (torch.bincount(g.patch_alg[nonsup], minlength=768) >= cfg.min_seed_patch_px).cpu().numpy()
    rec["n_eligible_patches"] = int(vp.sum())
    six = S1.top_k(smap, S1.SEED_K, vp) if g.N else np.zeros(0, np.int64)
    if six.size == 0:
        rec["ms"]["solve"] = round((time.perf_counter() - t) * 1e3, 3)
        rec["why"] = "no_seed"
        return done(SurfaceResult(None, rec))
    sw = smap[six] / smap[six].sum()
    wn = np.asarray(sw, np.float64) / np.max(sw)
    # seed unary (`ifg_graph.seed_unary`): each seed patch's mass split over its nodes by pixel share
    w_patch = np.zeros(768)
    pap = g.patch_alg_px.cpu().numpy().astype(np.float64)
    for k, p in enumerate(six):
        w_patch[p] += wn[k] / max(pap[p], 1e-12)
    w_px = torch.as_tensor(w_patch, device=dev)[g.patch_alg]
    lam_full = torch.zeros(g.N, dtype=torch.float64, device=dev).index_add_(0, g.inv, w_px)
    seed_nodes = torch.nonzero(lam_full > 0).reshape(-1)
    roi = (_roi(g.pos, seed_nodes, cfg.roi_radius, dev) if cfg.roi_radius is not None
           else torch.ones(g.N, dtype=torch.bool, device=dev))
    roi_idx = torch.nonzero(roi).reshape(-1)
    n = int(roi_idx.shape[0])
    normal, loc = _roi_normals(g, roi_idx, dev, eigh_device=crease is not None)   # T78d: device eigh when on
    ek = roi[g.ei] & roi[g.ej]
    # --- T78d — begin: the crease cut (the 1-ring normals above keep the uncut edges, as `build_geometry` does)
    cut_edges = None
    if crease is not None:
        if abs(float(cfg.voxel) - crease.voxel_m) > 1e-12:
            raise ValueError(f"T78d: graph voxel {cfg.voxel} != crease voxel {crease.voxel_m}")
        tc = time.perf_counter()
        keep_e, crec = _crease_cut(g, views, roi_idx, loc, ek, crease, dev)
        all_e = torch.stack([loc[g.ei[ek]], loc[g.ej[ek]]], dim=1)
        cut_edges = all_e[~keep_e].cpu().numpy()
        sub_edges = all_e[keep_e].cpu().numpy()
        det["crease"] = (time.perf_counter() - tc) * 1e3
        rec["crease"] = {**crease.record(), **crec, "ms": round(det["crease"], 3)}
    else:
        sub_edges = torch.stack([loc[g.ei[ek]], loc[g.ej[ek]]], dim=1).cpu().numpy()
    # --- T78d — end
    sub = types.SimpleNamespace(pos=g.pos[roi_idx].cpu().numpy(), normal=normal, edges=sub_edges,
                                support=g.support[roi_idx].cpu().numpy())
    sl = lam_full[roi_idx].cpu().numpy()
    px = g.ncnt[roi_idx].to(torch.float64).cpu().numpy()
    det["solve_gpu"] = (time.perf_counter() - t) * 1e3
    # ---- the locked solve + region on the host (`ifg_round_ns.g_and_top1`, after `subset`)
    t1 = time.perf_counter()
    sup = cfg.kappa_support * sub.support.astype(float)
    lam, s = RN._combine([(sl, np.ones(n)), (sup, np.zeros(n))])
    if crease is None:
        m, info = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, cfg, None), lam, s, cfg.lam)
    else:   # T78d: the 2.5 mm ROI is ~3× the 5 mm one — the linear solve goes to the device (PCG)
        m, info = _solve_device(sub, G.edge_weights(sub, cfg, None), lam, s, cfg.lam, dev)
        rec["crease"]["solve"] = {k: info[k] for k in ("cg_iter", "cg_fallback", "residual") if k in info}
    mt = RD.region_gate(m, sub, px, np.zeros(n), 0.5, sl, top1_only=True)[0]
    det["solve_host"] = (time.perf_counter() - t1) * 1e3
    rec["ms"]["solve"] = round((time.perf_counter() - t) * 1e3, 3)
    rec["seed_patches"] = [int(x) for x in six]
    rec["n_roi"] = int(info.get("n_roi", n))
    rec["n_unknown_nodes"] = int(info.get("n_unknown_nodes", 0))
    rec["n_region_nodes_m05"] = int((np.isfinite(m) & (m >= TAU)).sum())
    sel_sub = np.isfinite(mt) & (mt >= TAU)
    # --- T78d — begin: re-attach small cut-off components (one pass, whole components)
    if crease is not None and crease.reattach_ratio is not None and sel_sub.any():
        from AG3S.ag3s.runtime import attention_surface_crease as CR

        tr = time.perf_counter()
        add, rr = CR.reattach(n, sub_edges, cut_edges, sub.support, sel_sub, sub.pos, float(crease.reattach_ratio))
        rr["ms"] = round((time.perf_counter() - tr) * 1e3, 3)
        rec["crease"]["reattach"] = rr
        sel_sub = sel_sub | add
    # --- T78d — end
    roi_h = roi_idx.cpu().numpy()
    sel = roi_h[sel_sub]
    rec["n_selected_nodes"] = int(len(sel))
    if not len(sel):
        rec["why"] = "no_region"
        return done(SurfaceResult(None, rec))
    sp_ = sub.pos[sel_sub]
    extent = {"selected_centroid": [round(float(x), 4) for x in sp_.mean(0)],
              "selected_extent_m": [round(float(x), 4) for x in (sp_.max(0) - sp_.min(0))]}
    if not codes:
        rec.update({"applied": True, "codes_skipped": True, **extent})
        return done(SurfaceResult(None, rec, selected_nodes=np.sort(sel)))
    # ---- code images from the build's pixel → node map
    t = time.perf_counter()
    rec["pixel_node_keys"] = "build"
    chosen = torch.zeros(g.N, dtype=torch.bool, device=dev)
    chosen[torch.as_tensor(sel, device=dev)] = True
    out: dict[str, np.ndarray] = {}
    n_t, n_l, off = {}, {}, 0
    for c, v in enumerate(views):
        H, W = g.hw[c]
        node = g.node_of_px[off:off + g.sizes[c]]
        off += g.sizes[c]
        ok = node >= 0
        tgt = ok & chosen[torch.where(ok, node, torch.zeros_like(node))]
        val = torch.where(tgt, CODE_TARGET, torch.where(ok, CODE_STATIC, CODE_NONE)).to(torch.uint8)
        img = torch.full((H, W), CODE_NONE, dtype=torch.uint8, device=dev)
        img[g.V[c], g.U[c]] = val
        cnt = torch.stack([tgt.sum(), ok.sum()]).tolist()
        out[v.name] = img.cpu().numpy()
        n_t[v.name], n_l[v.name] = int(cnt[0]), int(cnt[1])
    rec["ms"]["codes"] = round((time.perf_counter() - t) * 1e3, 3)
    rec.update({"applied": True, "n_target_px": n_t, "n_labelled_px": n_l, **extent})
    return done(SurfaceResult(out, rec, selected_nodes=np.sort(sel)))


def warmup(device=None, repeat: int = 2, crease=None) -> float:
    """Compile the jiterator kernels and warm the CUDA paths (allocator, cuBLAS, sort / unique / segment kernels) on a
    synthetic request — a table with a box on it, seen by three 480 × 640 cameras, the box under the attention peak.
    Returns the seconds it took. serve_safe calls it at start-up so that the first request is not the slow one.
    T78d: `crease` given → the crease path is warmed too (radius normals, `torch.linalg.eigh`, the cut)."""
    from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER, SurfaceView, graph_config

    t = time.perf_counter()
    H, W, f = 480, 640, 400.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    T = np.eye(4); T[2, 3] = 1.0
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])   # looking down −z from 1 m: the table is z = 0
    # the box top is kept below the support gate (5 % of the nodes) so it is an object, not a second plane
    d = np.full((H, W), 1.0); d[200:280, 270:370] = 0.92
    rm = np.zeros((H, W), bool); rm[:40, :] = True
    att = np.zeros((16, 16)); att[7:9, 7:9] = 1.0
    views = [SurfaceView(name=n, depth_m=d, K=K, T_base_cam=T, robot_mask=rm, attention=att) for n in CAMERA_ORDER]
    smap = np.concatenate([np.asarray(v.attention, np.float64).reshape(-1) for v in views])
    for _ in range(max(1, int(repeat))):
        res = select(views, graph_config(), smap, arm="warmup", device=device)
    if not res.record.get("applied"):
        raise RuntimeError(f"attention-surface GPU warm-up made no selection: {res.record}")
    # --- T78d — begin
    if crease is not None:
        for _ in range(max(1, int(repeat))):
            res = select(views, crease.graph_config(graph_config()), smap, arm="warmup", device=device, crease=crease)
        if not res.record.get("applied") or "crease" not in res.record:
            raise RuntimeError(f"attention-surface GPU crease warm-up made no selection: {res.record}")
    # --- T78d — end
    return time.perf_counter() - t


__all__ = ["build", "cuda_device", "select", "support_mask", "warmup"]
