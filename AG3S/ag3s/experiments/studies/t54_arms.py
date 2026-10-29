"""T54 — grounding arms: L8H2 / noun (N) / noun choice over L8H2 candidates (A) / noun seed + L8H2 split (B).

`T54.task.md` "사전 고정 1" fixes the arms; this module is their only definition. Every arm runs the
**serving grounding code unchanged** (`AG3S.process_multi_debug` → `ground_target`). What an arm changes
is which attention map feeds which of three roles:

=========  ==============================  ==================================  ==========================
role       what reads it in serving        how T54 feeds it                     arms that change it
=========  ==============================  ==================================  ==========================
primary    lifting, fused attention,       the `attention=` maps of             N, B (noun), L12H4
           seeds, growth, DBSCAN, peak,    `wire.unpack_camera_observations`    (`last:L12H4`)
           split component order, the      — the serving input, no hook
           default `target_score`
split      the attention-split **ratio     `ratio_attention` of                 B (L8H2 while primary is
           gate** (`attention_split.       `target_grounding._split_groups`     noun)
           split_by_attention`             (hook): the split map lifted onto
           `ratio_attention`)              the same fused cloud, raw if
                                           `clustering.attention_split_ratio_
                                           source == "raw"` else normalised
choice     the ranking that names the      `ClusterInfo.target_score` (hook on  A, B, A-L12H4, A-k3
           target (`clusters.sort` by      `target_grounding._score_cluster`):  (noun mass)
           `target_score`, then            the **noun mass** = sum of the noun
           `TargetConfirm.select`)         map's normalised fused attention
                                           over the cluster's points
=========  ==============================  ==================================  ==========================

The hooks are **study-side**: `arm_hooks` swaps three module attributes for the duration of one
`process_multi_debug` call and restores them (`pipeline.fuse_observations` — recorded, untouched;
`target_grounding._split_groups` / `_score_cluster` — wrapped). No serving file is edited. An arm whose
split and choice roles equal its primary (L8H2, N, L12H4, N-k3) goes through pass-through wrappers that
call the originals with the original arguments (bit-identical; `tests/ag3s/test_t54_arms.py`).

The secondary map is lifted with the production calls on the **same** per-camera clouds the frame was
fused from (`attention_lifting.lift` per camera exactly as `multiview.process_observation`, then
`multiview.fuse`), and the fused points are checked equal to the frame's — so the split / choice
values are aligned point for point with the primary cloud, and a secondary map equal to the primary
reproduces `fusion.attention` / `fusion.raw_attention` bit for bit.

Interactions documented, not hidden:

* A / B: `target_score` becomes a mass (a sum, typically ≫ 1). `TargetConfirm.select` compares
  `target_score` with `clustering.switch_min_score` for a challenger frame (with-hold condition
  only); with a mass that floor is effectively never binding. Nothing else reads `target_score` for a
  decision (`target_score_threshold` = 0 by default).
* B: the split component order (which component is the "target" of a split) is still by the primary
  (noun) attention mass — `split_by_attention` reads `ratio_attention` for the ratio gate only. That is
  what "the split ratio reads L8H2" means in serving code.
* With `attention_split_ratio_source = normalized` (the dev / T52 profile) a split map is passed as
  normalised `ratio_attention`; the frame's split record then says ``ratio_source: raw`` (the label
  `split_by_attention` gives any `ratio_attention`). The C1a test profile is ``raw`` — no such label
  difference there.

CPU, no MuJoCo, no jax.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import time
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

STEP = "T54"

# ------------------------------------------------------------------------------- sources

#: Map sources (raw, `(F, 3, 16, 16)` in `POLICY_CAMERA_NAMES` order). `T54.task.md` 사전 고정 1.
SOURCES: dict[str, dict] = {
    "l8h2": {"kind": "action", "denoise": 0, "agg": "last", "layer": 8, "head": 2, "sigma": 0.0,
             "label": "serving last:L8H2 (d0, last, raw σ0)"},
    "l12h4": {"kind": "action", "denoise": 0, "agg": "last", "layer": 12, "head": 4, "sigma": 0.0,
              "label": "ablation last:L12H4 (d0, last, raw σ0)"},
    "noun": {"kind": "noun", "query": "target", "cells": ((10, 7),), "sigma": 0.0,
             "label": "noun_target:L10H7 raw σ0 k1 (GT-selected on dev data; not label-free)"},
    "noun_k3": {"kind": "noun", "query": "target", "cells": ((8, 6), (5, 4), (4, 2)), "sigma": 1.0,
                "label": "T49 primary noun_gt_k3_s1: L8H6 + L5H4 + L4H2, σ1"},
}


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    """One arm: which source feeds the primary / split / choice role (None = the primary)."""

    name: str
    primary: str
    split: Optional[str] = None
    choice: Optional[str] = None
    role: str = ""

    @property
    def split_source(self) -> str:
        return self.split or self.primary

    @property
    def choice_source(self) -> str:
        return self.choice or self.primary

    @property
    def hooked(self) -> bool:
        """True when a role reads a map other than the primary (a secondary lift is needed)."""
        return self.split_source != self.primary or self.choice_source != self.primary

    def secondary_sources(self) -> tuple[str, ...]:
        out = []
        for s in (self.split, self.choice):
            if s is not None and s != self.primary and s not in out:
                out.append(s)
        return tuple(out)

    def extract_kinds(self) -> tuple[str, ...]:
        """Which attention extractions the arm needs at inference (for the e2e time)."""
        kinds = []
        for s in (self.primary, self.split_source, self.choice_source):
            k = s if SOURCES[s]["kind"] == "noun" else "action"
            if k not in kinds:
                kinds.append(k)
        return tuple(kinds)

    def as_dict(self) -> dict:
        return {"name": self.name, "primary": self.primary, "split": self.split_source,
                "choice": self.choice_source,
                "choice_score": ("default target_score (serving)" if self.choice is None
                                 else f"mass of {self.choice} (normalised fused attention summed "
                                      "over the cluster's points)"),
                "role": self.role, "extract_kinds": list(self.extract_kinds())}


#: `T54.task.md` 사전 고정 1 table, in its order.
ARMS: dict[str, ArmSpec] = {a.name: a for a in (
    ArmSpec("L8H2", "l8h2", role="baseline (serving)"),
    ArmSpec("N", "noun", role="comparison (noun alone)"),
    ArmSpec("A", "l8h2", split="l8h2", choice="noun", role="main experiment (adoption candidate)"),
    ArmSpec("B", "noun", split="l8h2", choice="noun", role="cause check (not adoption evidence)"),
    ArmSpec("L12H4", "l12h4", role="ablation"),
    ArmSpec("A-L12H4", "l12h4", split="l12h4", choice="noun", role="ablation"),
    ArmSpec("N-k3", "noun_k3", role="auxiliary (T49 primary)"),
    ArmSpec("A-k3", "l8h2", split="l8h2", choice="noun_k3", role="auxiliary (T49 primary)"),
)}
BASELINE = "L8H2"
MAIN_ARM = "A"


def resolve_arms(names: Optional[Sequence[str]]) -> list[ArmSpec]:
    names = list(names or ARMS)
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {list(ARMS)}")
    return [ARMS[n] for n in names]


def sources_needed(arms: Sequence[ArmSpec]) -> list[str]:
    out: list[str] = []
    for a in arms:
        for s in (a.primary, a.split_source, a.choice_source):
            if s not in out:
                out.append(s)
    return out


def action_map(att: np.ndarray, denoise: Sequence[int], aggs: Sequence[str], spec: dict) -> np.ndarray:
    """``(F, 3, 16, 16)`` float32 of one action cell from a T47-format npz (`load_attention` axes
    ``(F, D, A, L, H, C, 16, 16)``) — the T52 replay's expression (float64 then float32)."""
    if spec.get("sigma", 0.0):
        raise ValueError("action sources are raw (σ0) in T54")
    d = list(denoise).index(int(spec["denoise"]))
    a = list(aggs).index(str(spec["agg"]))
    m = np.asarray(att[:, d, a, int(spec["layer"]), int(spec["head"])], np.float64)
    return m.astype(np.float32)


def noun_map(block: np.ndarray, spec: dict) -> np.ndarray:
    """``(F, 3, 16, 16)`` float32 of a noun source from a T49 query block ``(F, C, L, H, 16, 16)``
    (`t49_noun_query.block_fc`) — `t49_noun_query_core.combined`, the T52 expression."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

    m = core.combined(block, [list(c) for c in spec["cells"]], float(spec["sigma"]))
    return np.asarray(m, np.float64).astype(np.float32)


# ---------------------------------------------------------------------- secondary lifting


@dataclasses.dataclass
class Lifted:
    """A secondary map on the frame's fused cloud (aligned with `fusion.cloud.points`)."""

    attention: np.ndarray  # (N,) float32, normalised after fusion (as `fusion.attention`)
    raw: np.ndarray  # (N,) float32, fused raw max (as `fusion.raw_attention`)


def lift_on_fusion(fusion, observations: Sequence[Any], maps_by_camera: Mapping[Any, Any], config,
                   attention_adapter: Any = None) -> Lifted:
    """Lift another attention map onto the per-camera clouds `fusion` was built from and fuse it
    with the production `multiview.fuse` — the same calls `multiview.fuse_observations` /
    `process_observation` make, only the map differs. `maps_by_camera`: `CameraID` → raw map; a
    camera without one (or whose observation had no primary attention) contributes zeros, exactly
    as an observation without a map does in serving."""
    from AG3S.ag3s.runtime.multiview import fuse
    from AG3S.ag3s.stages.attention_lifting import lift, make_adapter

    if len(fusion.per_camera) != len(observations):
        raise ValueError(f"{len(fusion.per_camera)} camera results for {len(observations)} observations")
    results = []
    for obs, r in zip(observations, fusion.per_camera):
        if obs.camera_id != r.camera_id:
            raise ValueError(f"camera order differs: {obs.camera_id} vs {r.camera_id}")
        m = maps_by_camera.get(r.camera_id)
        if m is not None and obs.has_attention and not r.cloud.is_empty:
            adapter = attention_adapter or make_adapter(m, config.attention)
            ac = lift(r.cloud, m, config.attention, adapter=adapter, image_hw=obs.image_hw,
                      camera_intrinsics=obs.camera_intrinsics, T_base_cam=r.T_base_cam)
            att = np.asarray(ac.attention, np.float32)
            raw = att if ac.raw_attention is None else np.asarray(ac.raw_attention, np.float32)
        else:
            att = np.zeros(len(r.cloud), np.float32)
            raw = att
        results.append(dataclasses.replace(r, attention=att, raw_attention=raw))
    fused, attention, raw_attention = fuse(
        results, voxel_size=config.timing.fusion_voxel_size, frame_id=config.frame_id,
        attention_config=config.attention)
    if (fused.points.shape != fusion.cloud.points.shape
            or not np.array_equal(fused.points, fusion.cloud.points)):
        raise RuntimeError("secondary fusion produced a different cloud — refusing to misalign maps")
    return Lifted(attention=np.asarray(attention, np.float32),
                  raw=np.asarray(raw_attention, np.float32))


# ---------------------------------------------------------------------------------- hooks


@dataclasses.dataclass
class CallRecord:
    """What one hooked `process_multi_debug` call saw (filled by `arm_hooks`)."""

    arm: str = ""
    fusion: Any = None
    observations: Any = None
    config: Any = None
    secondary: dict = dataclasses.field(default_factory=dict)
    ms_secondary_lift: float = 0.0
    pre_split: Optional[list] = None  # grown groups, the order `_split_groups` got them
    post_split: Optional[list] = None  # members after the split, same order (before scoring)
    ratio_attention: Optional[np.ndarray] = None  # what the split ratio gate read (None = normalised primary)
    ratio_source: Optional[str] = None
    choice_mass: dict = dataclasses.field(default_factory=dict)  # tuple(indices) digest -> mass
    n_split_calls: int = 0
    n_score_calls: int = 0


def _digest(arrays: Optional[Sequence[np.ndarray]]) -> Optional[str]:
    if arrays is None:
        return None
    h = hashlib.sha1()
    for a in arrays:
        b = np.asarray(a, np.int64).reshape(-1)
        h.update(np.int64(b.size).tobytes())
        h.update(b.tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def arm_hooks(arm: ArmSpec, secondary_maps: Mapping[str, Mapping[Any, Any]],
              record: CallRecord) -> Iterator[CallRecord]:
    """Hook one `process_multi_debug` call for `arm` (module doc). `secondary_maps`: source name →
    `{CameraID: raw map}` for every `arm.secondary_sources()`. Not thread-safe (module attributes)."""
    import AG3S.ag3s.runtime.pipeline as pl
    import AG3S.ag3s.stages.target_grounding as tg

    missing = [s for s in arm.secondary_sources() if s not in secondary_maps]
    if missing:
        raise ValueError(f"arm {arm.name}: no maps for secondary source(s) {missing}")
    record.arm = arm.name
    orig_fuse, orig_split, orig_score = pl.fuse_observations, tg._split_groups, tg._score_cluster

    def fuse_wrap(observations, config, **kw):
        res = orig_fuse(observations, config, **kw)
        record.fusion, record.observations, record.config = res, list(observations), config
        if arm.secondary_sources():
            t0 = time.perf_counter()
            for src in arm.secondary_sources():
                record.secondary[src] = lift_on_fusion(res, record.observations,
                                                       secondary_maps[src], config,
                                                       kw.get("attention_adapter"))
            record.ms_secondary_lift = (time.perf_counter() - t0) * 1000.0
        return res

    def split_wrap(groups, points, attention, cfg, confirm, *, enabled, ratio_attention=None):
        record.n_split_calls += 1
        record.pre_split = [np.asarray(g, np.int64).copy() for g in groups]
        ra = ratio_attention
        record.ratio_source = (None if ratio_attention is None else "primary_raw")
        if arm.split_source != arm.primary:
            lifted = record.secondary[arm.split_source]
            ra = lifted.raw if cfg.attention_split_ratio_source == "raw" else lifted.attention
            if len(ra) != len(attention):
                raise RuntimeError(f"split map has {len(ra)} values for {len(attention)} points")
            record.ratio_source = (f"{arm.split_source}_raw" if cfg.attention_split_ratio_source == "raw"
                                   else f"{arm.split_source}_normalized")
        record.ratio_attention = None if ra is None else np.asarray(ra)
        out, decisions = orig_split(groups, points, attention, cfg, confirm, enabled=enabled,
                                    ratio_attention=ra)
        record.post_split = [np.asarray(idx, np.int64).copy() for idx, _ in out]
        return out, decisions

    def score_wrap(cluster_id, indices, points, attention, peak_point, config):
        record.n_score_calls += 1
        ci = orig_score(cluster_id, indices, points, attention, peak_point, config)
        if arm.choice_source != arm.primary:
            att = record.secondary[arm.choice_source].attention
            if len(att) != len(attention):
                raise RuntimeError(f"choice map has {len(att)} values for {len(attention)} points")
            mass = float(np.asarray(att, np.float64)[np.asarray(indices, np.int64)].sum())
            record.choice_mass[_digest([indices])] = mass
            ci = dataclasses.replace(ci, target_score=mass)
        return ci

    pl.fuse_observations, tg._split_groups, tg._score_cluster = fuse_wrap, split_wrap, score_wrap
    try:
        yield record
    finally:
        pl.fuse_observations, tg._split_groups, tg._score_cluster = orig_fuse, orig_split, orig_score


def posthoc_lift_ms(record: CallRecord, maps_by_camera: Mapping[Any, Any]) -> float:
    """Wall time of lifting + fusing one map onto this call's per-camera clouds (the cost the
    serving front end pays for the primary map inside `scene_reconstruction`), measured after the
    call with the same code as the secondary lift. NaN when the call never fused."""
    if record.fusion is None:
        return float("nan")
    t0 = time.perf_counter()
    lift_on_fusion(record.fusion, record.observations, maps_by_camera, record.config)
    return (time.perf_counter() - t0) * 1000.0


# ----------------------------------------------------------------------- per-call row fields


def cluster_rows(grounding, labels: np.ndarray, primary_attention: Optional[np.ndarray],
                 choice_attention: Optional[np.ndarray], admissibility) -> list[dict]:
    """Per candidate cluster (`grounding.clusters` order = after scoring, before the choice):
    size, GT class counts, strict plurality, scores, masses, admissibility."""
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    out = []
    if grounding is None:
        return out
    n = len(labels)
    adm = list(admissibility) if admissibility is not None else None
    if adm is not None and len(adm) != len(grounding.clusters):
        adm = None
    for rank, c in enumerate(grounding.clusters):
        idx = np.asarray(c.point_indices, np.int64)
        idx = idx[(idx >= 0) & (idx < n)]
        counts = bm.class_counts(labels[idx]) if idx.size else bm.class_counts(np.zeros(0, int))
        out.append({
            "rank": rank, "n": int(len(c.point_indices)), "first_index": int(idx.min()) if idx.size else -1,
            "plurality": bm.plurality(counts),
            "n_target": counts["target"], "n_other_object": counts["other_object"],
            "n_destination": counts["destination"], "n_table": counts["table"],
            "target_score": float(c.target_score), "mean_attention": float(c.mean_attention),
            "primary_mass": (float(np.asarray(primary_attention, np.float64)[idx].sum())
                             if primary_attention is not None and idx.size else None),
            "choice_mass": (float(np.asarray(choice_attention, np.float64)[idx].sum())
                            if choice_attention is not None and idx.size else None),
            "split": None if c.split is None else {k: c.split.get(k) for k in ("role", "reason")},
            "admissible": None if adm is None else bool(adm[rank].admissible),
            "admissibility_reason": None if adm is None else str(adm[rank].reason),
        })
    return out


def candidate_fields(clusters: Sequence[dict]) -> dict:
    """`target_candidate_present` (T54 사전 고정 1): a candidate cluster just before the choice whose
    strict plurality class is the target. Also over admissible candidates only (when known)."""
    present = any(c["plurality"] == "target" for c in clusters)
    known = clusters and all(c["admissible"] is not None for c in clusters)
    return {
        "n_candidates": len(clusters),
        "target_candidate_present": bool(present),
        "target_candidate_present_admissible": (
            bool(any(c["plurality"] == "target" and c["admissible"] for c in clusters))
            if known else (False if not clusters else None)),
        "n_candidates_admissible": (sum(1 for c in clusters if c["admissible"]) if known
                                    else (0 if not clusters else None)),
    }


def chosen_rank(grounding, cs) -> Optional[int]:
    """Rank (in `grounding.clusters`) of the cluster named this frame; None when nothing was named or
    the target is a stand-in geometry (occluded / kept anchor: empty `point_indices`)."""
    if grounding is None or not bool(cs.has_target):
        return None
    idx = np.asarray(cs.target.point_indices, np.int64)
    if idx.size == 0:
        return None
    for r, c in enumerate(grounding.clusters):
        ci = np.asarray(c.point_indices, np.int64)
        if ci.size == idx.size and np.array_equal(ci, idx):
            return r
    return None


def hook_fields(record: CallRecord) -> dict:
    """Digests of the cluster lists (A vs L8H2 bit-identity check per frame) and hook bookkeeping."""
    return {
        "pre_split_digest": _digest(record.pre_split),
        "post_split_digest": _digest(record.post_split),
        "n_pre_split": None if record.pre_split is None else len(record.pre_split),
        "n_post_split": None if record.post_split is None else len(record.post_split),
        "ratio_source": record.ratio_source,
        "ms_secondary_lift": float(record.ms_secondary_lift),
        "secondary_sources": sorted(record.secondary),
    }


__all__ = [
    "ARMS", "ArmSpec", "BASELINE", "CallRecord", "Lifted", "MAIN_ARM", "SOURCES", "STEP",
    "action_map", "arm_hooks", "candidate_fields", "chosen_rank", "cluster_rows", "hook_fields",
    "lift_on_fusion", "noun_map", "posthoc_lift_ms", "resolve_arms", "sources_needed",
]
