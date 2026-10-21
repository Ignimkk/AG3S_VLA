"""IFG-CL2 — the **hybrid target extent** (`esdf.target_extent = "attention_surface_hybrid"`, CLI
`--target-extent attention-surface-hybrid`): erase the `ball_owned` range, protect with the attention surface.

**Why** (IFG-CL1 held-out closed loop, C1aN vs SURF, `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` 2026-10-17):
the attention-surface extent (`"attention_surface"` → the `oracle_visible` split) removes only the seeds next to the
*visible* target surface, and leaves the target's unseen side (TSDF seeds from earlier views / the truncation band)
in front of the fingers as an obstacle; with the GT visible segmentation 4 of the 6 failures still failed. The
`ball_owned` extent (C1aN) removes the target volume but its non-target evidence (the 10 mm fused cloud) protects
neighbours coarsely. The hybrid keeps the C1aN range and uses the attention surface **only to protect**.

## Per request

| step | what | where |
|---|---|---|
| 1 | the LOCK selection (`select_target_surface`, unchanged, `keep_debug=True` for its graph) | `pipeline._attention_surface_hybrid_update` |
| 2 | samples = graph node centroids (5 mm voxels): **target** = selected region · **other** = not selected and not a support node (graph RANSAC) · every node with its code (1 selected / 2 else) for the identity check | `hybrid_samples` |
| 3 | identity guard (below); off → no protection this frame (= C1aN), reason recorded | `identity_guard` |
| 4 | `SurfaceProtectEvidence` → builder: `surface_protect_split` = `ball_owned` split ∧ a second `ball_owned` split with the node samples as evidence (same tie · max_dist · unowned · ambiguous) | `curobo_builder` |

The code images (0 / 1 / 2) are still made and recorded (`attention_surface_codes_<camera>`) — with the request's depth
they rebuild the same graph offline (`experiments/studies/cl2_hybrid_replay.py`). The oracle memory is **not** used:
the protection reads this frame's surface only.

## Identity guard (`esdf.surface_hybrid_identity`)

| rule | protection only when | mirrors |
|---|---|---|
| `instance` | ≥ ½ of the manipulated geometry points that have a node within `radius` (= `timing.fusion_voxel_size`) land on a selected node (and at least one does) | `OracleMemory.instance_of` (the oracle path's `identity_unresolved`), on this frame's nodes |
| `centre` | ‖selected centroid − ball centre‖ ≤ ball radius | task design 3 |
| `both` (default) | both | |

## IFG-CL2b — the target's own surface out of the "other" evidence (`esdf.surface_hybrid_other_exclude_voxels`)

IFG-CL2 replay: 71 % of the protected GT-target seeds had, as nearest "other" node, an **unselected node of the target
itself** (the region covers only part of the fruit). So, once the guard is on, "other" nodes within
`other_exclude_voxels × radius` (radius = `timing.fusion_voxel_size`, 10 mm; default 1.0 → 10 mm, fixed in advance) of the
manipulated geometry points — the same points the identity guard reads — are dropped from the other evidence
(`other_near_geometry`). No geometry points → no exclusion (the IFG-CL2 rule). `0` = IFG-CL2 bit-identical (no record key).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

import numpy as np

#: `esdf.surface_hybrid_identity` choices (`EsdfConfig.SURFACE_HYBRID_IDENTITIES` is the same list).
IDENTITY_RULES = ("both", "instance", "centre")
#: `OracleMemory.instance_of`'s majority.
MIN_FRACTION = 0.5


@dataclasses.dataclass(frozen=True)
class HybridSamples:
    """This frame's graph-node samples (base frame, m). `node_pos` / `node_selected` cover every node (support too)."""

    target: np.ndarray
    other: np.ndarray
    node_pos: np.ndarray
    node_selected: np.ndarray
    n_support_other: int = 0


def hybrid_samples(result) -> Optional[HybridSamples]:
    """`HybridSamples` from a `select_target_surface(..., keep_debug=True)` result, or None (abstain / no graph)."""
    geo = getattr(result, "geometry", None)
    if getattr(result, "codes", None) is None or geo is None:
        return None
    sel = np.zeros(len(geo.pos), bool)
    sel[np.asarray(result.selected_nodes, np.int64)] = True
    sup = np.asarray(geo.support, bool)
    pos = np.asarray(geo.pos, np.float64)
    return HybridSamples(target=pos[sel], other=pos[~sel & ~sup], node_pos=pos, node_selected=sel,
                         n_support_other=int((~sel & sup).sum()))


def other_near_geometry(other, geometry_points, radius: float) -> np.ndarray:
    """IFG-CL2b — `(len(other),) bool`: the "other" node is within `radius` (m, ≤) of a manipulated geometry point.

    All False when `radius <= 0`, no node or no geometry point (→ the IFG-CL2 rule).
    """
    o = np.asarray(other, np.float64).reshape(-1, 3)
    p = np.asarray(geometry_points, np.float64).reshape(-1, 3)
    out = np.zeros(len(o), bool)
    if float(radius) <= 0.0 or not len(o) or not len(p):
        return out
    from scipy.spatial import cKDTree

    d, _ = cKDTree(p).query(o, k=1)
    return d <= float(radius)


def _instance(points: np.ndarray, s: HybridSamples, radius: float) -> tuple[bool, dict]:
    p = np.asarray(points, np.float64).reshape(-1, 3)
    info: dict[str, Any] = {"n_points": int(len(p)), "radius_mm": round(float(radius) * 1e3, 3),
                            "min_fraction": MIN_FRACTION}
    if not len(p):
        info["why"] = "no_points"
        return False, info
    box = np.all((s.node_pos >= p.min(0) - radius) & (s.node_pos <= p.max(0) + radius), axis=1)
    if not box.any():
        info.update(n_labelled=0, n_on_selected=0, why="no_node_near")
        return False, info
    from scipy.spatial import cKDTree

    d, j = cKDTree(s.node_pos[box]).query(p, k=1)
    near = d <= float(radius)
    on = s.node_selected[box][j[near]]
    info["n_labelled"] = int(near.sum())
    info["n_on_selected"] = int(on.sum())
    if not on.any():
        info["why"] = "no_selected_node"
        return False, info
    frac = float(on.sum()) / float(near.sum())
    info["fraction"] = round(frac, 4)
    if frac < MIN_FRACTION:
        info["why"] = "no_majority"
        return False, info
    return True, info


def identity_guard(geometry_points, samples: HybridSamples, ball, *, radius: float,
                   rule: str = "both") -> tuple[bool, dict]:
    """`(protect, record)` — whether the selected region is the manipulated object (see the module docstring).

    Both rules are evaluated and recorded whatever `rule` says (the record shows what the other rule would have done).
    """
    if rule not in IDENTITY_RULES:
        raise ValueError(f"surface_hybrid_identity 는 {IDENTITY_RULES} 중 하나여야 합니다: {rule!r}")
    ok_i, inst = _instance(geometry_points, samples, radius)
    c = samples.target.mean(0)
    dist = float(np.linalg.norm(c - np.asarray(ball.centre, np.float64).reshape(3)))
    ok_c = dist <= float(ball.radius)
    rec: dict[str, Any] = {"rule": rule, "instance": {"ok": bool(ok_i), **inst},
                           "centre": {"ok": bool(ok_c), "dist_mm": round(dist * 1e3, 3),
                                      "ball_radius_mm": round(float(ball.radius) * 1e3, 3),
                                      "selected_centroid": [round(float(x), 4) for x in c]}}
    ok = {"both": ok_i and ok_c, "instance": ok_i, "centre": ok_c}[rule]
    if not ok:
        why = []
        if rule in ("both", "instance") and not ok_i:
            why.append(f"instance:{inst.get('why')}")
        if rule in ("both", "centre") and not ok_c:
            why.append("centre:outside_ball")
        rec["why"] = ",".join(why)
    return bool(ok), rec


def hybrid_evidence(samples: Optional[HybridSamples], surface_record: Optional[dict], geometry_points, ball, *,
                    radius: float, rule: str = "both", other_exclude_voxels: float = 0.0):
    """`(curobo_builder.SurfaceProtectEvidence | None, record)` for this frame's fine target-free tier.

    None (→ the `ball_owned` split alone, i.e. C1aN) when the frame abstained (`attention_surface_abstained (<why>)`),
    there is no ball (`no_ball`) or the identity guard is off (`identity_off (<why>)`). The record always has
    `applied` · `why` (when not applied) · `identity` (when evaluated) · `ms`.

    IFG-CL2b: `other_exclude_voxels > 0` drops the other nodes within `other_exclude_voxels × radius` of
    `geometry_points` (`other_near_geometry`) from the evidence once the guard is on; the record then has
    `other_exclude` {`voxels`, `radius_mm`, `n_before`, `n_excluded`[, `why: no_geometry`]} and `n_other_points` is the
    count after the exclusion. `0` (the IFG-CL2 call) leaves evidence and record unchanged.
    """
    from AG3S.ag3s.fields.curobo_builder import SurfaceProtectEvidence

    t0 = time.perf_counter()
    rec: dict[str, Any] = {"applied": False}

    def done(ev=None):
        rec["ms"] = round((time.perf_counter() - t0) * 1e3, 3)
        return ev, rec

    if samples is None:
        rec["why"] = f"attention_surface_abstained ({(surface_record or {}).get('why')})"
        return done()
    if ball is None:
        rec["why"] = "no_ball"
        return done()
    ok, ident = identity_guard(geometry_points, samples, ball, radius=radius, rule=rule)
    rec["identity"] = ident
    rec["n_target_points"] = int(len(samples.target))
    rec["n_other_points"] = int(len(samples.other))
    rec["n_support_points_excluded"] = int(samples.n_support_other)
    if not ok:
        rec["why"] = f"identity_off ({ident.get('why')})"
        return done()
    other = samples.other
    info: dict[str, Any] = {"identity_rule": rule}
    v = float(other_exclude_voxels)
    if v > 0.0:   # IFG-CL2b
        geo = np.asarray(geometry_points, np.float64).reshape(-1, 3)
        drop = other_near_geometry(other, geo, v * float(radius))
        ox: dict[str, Any] = {"voxels": v, "radius_mm": round(v * float(radius) * 1e3, 3),
                              "n_before": int(len(other)), "n_excluded": int(drop.sum())}
        if not len(geo):
            ox["why"] = "no_geometry"
        other = other[~drop]
        rec["other_exclude"] = ox
        rec["n_other_points"] = int(len(other))
        info["other_exclude_voxels"] = v
    rec["applied"] = True
    return done(SurfaceProtectEvidence(target_points=samples.target, other_points=other, info=info))


__all__ = ["HybridSamples", "IDENTITY_RULES", "hybrid_evidence", "hybrid_samples", "identity_guard",
           "other_near_geometry"]
