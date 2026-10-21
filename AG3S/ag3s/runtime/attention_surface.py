"""IFG-CL1 — the **attention-surface target extent** on the server (`esdf.target_extent = "attention_surface"`,
CLI `--target-extent attention-surface`).

The `oracle_visible` path (`runtime.oracle_segmentation`, `curobo_builder.oracle_visible_split`) removes from the
fine target-free tier only the seeds that came from the target's pixels, given a per-camera **code image**
(0 = none / robot, 1 = target, 2 = static, ≥ 3 = another graspable object) that the client renders from the
simulator's GT segmentation. This module makes the same code image on the server from **attention + depth** — the
method fixed on the IFG dev split (`$RUN/LOCK_heldout.json` → `arms.candidate_extent`, ``top1-only-ns+rm:N``).
Everything after the code image (memory, carving, split, tiers, permission) is the oracle path unchanged; only the
source of the target extent differs.

## What runs per request (the locked study code, imported — not copied)

| step | function (locked `experiments/studies/ifg_*.py`) | value |
|---|---|---|
| surface graph | `ifg_graph.build_geometry` with `ifg_graph.GraphConfig()` | voxel 5 mm, range_max 2.0 m, jump_gate 10 mm, radius ×3, … |
| seed eligibility | `ifg_round_ns.eligible(geo, cfg, True)` | ≥ 60 non-robot, non-support pixels in the patch |
| seed + solve + region | `ifg_round_ns.g_and_top1(geo, cfg, map, eligible)` | noun L10H7 top-5 patches; ROI 0.15 m; the region of `{m ≥ 0.5}` holding the most seed mass |

`build_geometry` also takes the evaluator's GT (`class_map`, `classes`) — they only fill the evaluation arrays
(`eval_target`, `eval_class_frac`) that this path never reads. A dummy (`class_map = 0`, `classes = ("target",)`)
is passed; `tests/ag3s/test_ifg_cl1_attention_surface.py` shows that the real GT gives the same selection.

## Code image

| code | pixels |
|---:|---|
| 1 (`CODE_TARGET`) | valid graph pixels whose node is in the selected region |
| 2 (`CODE_STATIC`) | every other valid, non-robot graph pixel (depth inside `GraphConfig.range_max`) |
| 0 (`CODE_NONE`) | robot (the graph's robot mask) and invalid / out-of-range depth |

A pixel's node is the 5 mm voxel `build_geometry` put it in (same back-projection, same voxel origin); the mapping is
re-derived here and checked against the graph's node count.

## Abstain

No code image (→ the frame's fine target-free tier uses the `estimated` extent, as the oracle path does without a
segmentation) when: a camera of the three is missing (`missing_camera`), a camera has no attention map
(`no_attention`), no seed patch is eligible or the map is zero there (`no_seed`), or the solve leaves no node of the
top region at ≥ 0.5 (`no_region`). The reason is the record's `why`.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.runtime.oracle_segmentation import CODE_NONE, CODE_STATIC, CODE_TARGET

#: Attention token order (`policy_record.CAMERA_BINDINGS`: cam_high 0–255, left wrist 256–511, right wrist
#: 512–767) as `CameraID` values. The graph's patch index is `256 · camera index + row · 16 + col`, so the depth
#: views are given in the same order.
CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
#: The LOCK arm (`LOCK_heldout.json` `arms.candidate_extent.name`) with the render mask; with the sphere mask the
#: same selection is the R5 arm ``top1-only-ns:N``.
ARM_RENDER = "top1-only-ns+rm:N"
ARM_SPHERES = "top1-only-ns:N"
#: The region threshold of `ifg_round.region_gate` (`{m ≥ 0.5}`) — also the selection threshold here.
TAU = 0.5
GRID = 16


@dataclasses.dataclass(frozen=True)
class SurfaceView:
    """One camera of one request: metric depth (H, W), `K`, `T_base_cam` (OpenCV → base), the robot mask the graph
    excludes (H, W) bool, and the camera's raw attention map (16, 16) (None = no map)."""

    name: str
    depth_m: np.ndarray
    K: np.ndarray
    T_base_cam: np.ndarray
    robot_mask: np.ndarray
    attention: Optional[np.ndarray] = None


@dataclasses.dataclass
class SurfaceResult:
    """`codes` = `{camera name: (H, W) uint8}` or None (abstain, `record["why"]`)."""

    codes: Optional[dict[str, np.ndarray]]
    record: dict[str, Any]
    #: node ids (into `geometry.pos`) of the selected region — empty on abstain
    selected_nodes: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, np.int64))
    geometry: Any = None
    #: per camera `(u, v, node id or -1)` of the graph's valid pixels (`keep_debug=True` only)
    pixel_nodes: Optional[list] = None
    #: `(sub, seeds, m, m_top1, info)` of `g_and_top1` (`keep_debug=True` only)
    solve: Any = None


def graph_config():
    """The LOCK graph (`ifg_graph.GraphConfig()` defaults — the LOCK lists exactly these values)."""
    from AG3S.ag3s.experiments.studies import ifg_graph as G

    return G.GraphConfig()


def attention_vector(views: Sequence[SurfaceView]) -> tuple[Optional[np.ndarray], Optional[str]]:
    """`(768,) float64` in `CAMERA_ORDER` token order, or `(None, why)`."""
    out = []
    for v in views:
        if v.attention is None:
            return None, f"no_attention:{v.name}"
        a = np.asarray(v.attention, np.float64)
        if a.size != GRID * GRID:
            return None, f"attention_shape:{v.name}:{tuple(a.shape)}"
        out.append(a.reshape(-1))
    return np.concatenate(out), None


def _pixel_nodes(views, cfg, node_pos: np.ndarray):
    """Per camera `(u, v, node)` of the valid graph pixels — `build_geometry`'s voxel assignment re-derived
    (`_cam_pixels` → `floor(P / voxel) − min`, nodes = sorted unique keys of the non-robot pixels).

    The node keys are read off the node centroids (a centroid of points in one voxel lies in that voxel) and every
    non-robot pixel's key is checked against its node's; if any differs (a centroid rounded onto a voxel face) the
    keys are rebuilt with `np.unique`, exactly as `build_geometry` does. Returns `(per camera, how)`."""
    from AG3S.ag3s.experiments.studies import ifg_graph as G

    P, U, V, ROB = [], [], [], []
    for c, v in enumerate(views):
        p, u, vv, _, rob = G._cam_pixels(np.asarray(v.depth_m), v.K, v.T_base_cam, np.asarray(v.robot_mask, bool),
                                         c, cfg.range_max)
        P.append(p); U.append(u); V.append(vv); ROB.append(np.asarray(rob, bool))
    sizes = [len(u) for u in U]
    Pc = np.concatenate(P)
    keys = np.floor(Pc / cfg.voxel).astype(np.int64)
    kmin = keys.min(0)
    keys -= kmin
    ext = keys.max(0) + 1
    packed = (keys[:, 0] * ext[1] + keys[:, 1]) * ext[2] + keys[:, 2]
    alg = ~np.concatenate(ROB)
    n_nodes = len(node_pos)
    nk = np.floor(np.asarray(node_pos) / cfg.voxel).astype(np.int64) - kmin
    nd_keys = (nk[:, 0] * ext[1] + nk[:, 1]) * ext[2] + nk[:, 2]
    pa = packed[alg]
    idx = np.searchsorted(nd_keys, pa)
    how = "centroid_keys"
    if not (len(nd_keys) and np.all(np.diff(nd_keys) > 0) and np.all(idx < n_nodes)
            and np.array_equal(nd_keys[np.minimum(idx, n_nodes - 1)], pa)):
        nd_keys = np.unique(pa)
        if len(nd_keys) != int(n_nodes):
            raise RuntimeError(f"pixel → node re-derivation found {len(nd_keys)} nodes, the graph has {n_nodes}")
        idx = np.searchsorted(nd_keys, pa)
        how = "unique"
    node = np.full(len(packed), -1, np.int64)
    node[alg] = idx
    out, off = [], 0
    for c, n in enumerate(sizes):
        out.append((U[c], V[c], node[off:off + n]))
        off += n
    return out, how


def select_target_surface(views: Sequence[SurfaceView], *, cfg=None, class_map=None, classes=None,
                          keep_debug: bool = False, arm: str = ARM_RENDER, codes: bool = True) -> SurfaceResult:
    """The LOCK `candidate_extent` on one request (`CAMERA_ORDER` views). See the module docstring.

    `class_map` / `classes` default to the dummy (zeros, `("target",)`); they are evaluator inputs of
    `build_geometry` and do not reach the selection — only tests pass the real GT here.

    `codes=False` (IFG-S2SERVE confidence gate only): the selection is made exactly as above, but the pixel → node
    mapping and the code images are skipped — `codes` stays None while `record["applied"]` is True and
    `record["codes_skipped"]` is True. Default True = unchanged.
    """
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_round_ns as RNS

    t0 = time.perf_counter()
    cfg = graph_config() if cfg is None else cfg
    views = list(views)
    rec: dict[str, Any] = {"applied": False, "arm": arm, "cameras": [v.name for v in views], "ms": {}}
    names = tuple(v.name for v in views)
    if names != CAMERA_ORDER:
        miss = [n for n in CAMERA_ORDER if n not in names]
        rec["why"] = f"missing_camera:{','.join(miss)}" if miss else f"camera_order:{','.join(names)}"
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return SurfaceResult(None, rec)
    smap, why = attention_vector(views)
    if smap is None:
        rec["why"] = why
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return SurfaceResult(None, rec)
    depth = [np.asarray(v.depth_m, np.float64) for v in views]
    rmask = [np.asarray(v.robot_mask, bool) for v in views]
    if class_map is None:
        class_map = [np.zeros(d.shape, np.int64) for d in depth]
        classes = ("target",)
    t = time.perf_counter()
    try:
        geo = G.build_geometry(depth, [np.asarray(v.K, np.float64) for v in views],
                               [np.asarray(v.T_base_cam, np.float64) for v in views], rmask, class_map,
                               list(classes), cfg)
    except ValueError as exc:   # no valid depth pixel at all (`keys.min` of an empty array)
        rec["why"] = f"no_depth_points ({exc})"
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return SurfaceResult(None, rec)
    rec["ms"]["build"] = round((time.perf_counter() - t) * 1e3, 3)
    rec["n_nodes"] = int(geo.stats["n_nodes"])
    rec["n_support_nodes"] = int(geo.stats["n_support_nodes"])
    rec["n_robot_px"] = int(geo.stats["n_px_robot"])
    vp = RNS.eligible(geo, cfg, True)
    rec["n_eligible_patches"] = int(vp.sum())
    t = time.perf_counter()
    res = RNS.g_and_top1(geo, cfg, smap, vp) if len(geo.pos) else None
    rec["ms"]["solve"] = round((time.perf_counter() - t) * 1e3, 3)
    if res is None:
        rec["why"] = "no_seed"
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return SurfaceResult(None, rec, geometry=geo if keep_debug else None)
    sub, six, m, mt, info = res
    rec["seed_patches"] = [int(s) for s in six]
    rec["n_roi"] = int(info.get("n_roi", len(sub.pos)))
    rec["n_unknown_nodes"] = int(info.get("n_unknown_nodes", 0))
    rec["n_region_nodes_m05"] = int((np.isfinite(m) & (m >= TAU)).sum())
    sel_sub = np.isfinite(mt) & (mt >= TAU)
    # sub → graph node ids: `subset` keeps `node_eval` (strictly increasing over the graph's nodes)
    sel = np.searchsorted(geo.node_eval, sub.node_eval[sel_sub])
    rec["n_selected_nodes"] = int(len(sel))
    if not len(sel):
        rec["why"] = "no_region"
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return SurfaceResult(None, rec, geometry=geo if keep_debug else None,
                             solve=res if keep_debug else None)
    if not codes:   # IFG-S2SERVE: selection only (the gate reads the nodes, not the code images)
        sp = geo.pos[sel]
        rec.update({"applied": True, "codes_skipped": True,
                    "selected_centroid": [round(float(x), 4) for x in sp.mean(0)],
                    "selected_extent_m": [round(float(x), 4) for x in (sp.max(0) - sp.min(0))]})
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return SurfaceResult(None, rec, selected_nodes=np.sort(sel), geometry=geo if keep_debug else None,
                             solve=res if keep_debug else None)
    t = time.perf_counter()
    pix, rec["pixel_node_keys"] = _pixel_nodes(views, cfg, geo.pos)
    chosen = np.zeros(len(geo.pos), bool)
    chosen[sel] = True
    codes: dict[str, np.ndarray] = {}
    n_t, n_l = {}, {}
    for v, d, (u, vv, node) in zip(views, depth, pix):
        img = np.full(d.shape, CODE_NONE, np.uint8)
        ok = node >= 0
        img[vv[ok], u[ok]] = CODE_STATIC
        tgt = ok & chosen[np.where(ok, node, 0)]
        img[vv[tgt], u[tgt]] = CODE_TARGET
        codes[v.name] = img
        n_t[v.name] = int(tgt.sum())
        n_l[v.name] = int(ok.sum())
    rec["ms"]["codes"] = round((time.perf_counter() - t) * 1e3, 3)
    sp = geo.pos[sel]
    rec.update({"applied": True, "n_target_px": n_t, "n_labelled_px": n_l,
                "selected_centroid": [round(float(x), 4) for x in sp.mean(0)],
                "selected_extent_m": [round(float(x), 4) for x in (sp.max(0) - sp.min(0))]})
    rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
    return SurfaceResult(codes, rec, selected_nodes=np.sort(sel), geometry=geo if keep_debug else None,
                         pixel_nodes=pix if keep_debug else None, solve=res if keep_debug else None)


__all__ = ["ARM_RENDER", "ARM_SPHERES", "CAMERA_ORDER", "SurfaceResult", "SurfaceView", "TAU",
           "attention_vector", "graph_config", "select_target_surface"]
