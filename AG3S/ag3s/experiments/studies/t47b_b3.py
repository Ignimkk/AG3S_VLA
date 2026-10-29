"""T47b B3 — does anything between lifting and grounding compare the **raw** attention amplitude?

Summing three heads (T47) instead of averaging them multiplies the map by a constant; summing
mass-normalised heads (`t47b_aggregation`) changes the relative weight of the heads and of the
cameras. Which of those the pipeline can feel is a property of the code, so it is tabulated here as
data (`AUDIT`, file:line + verdict) and checked numerically (`snapshot` / `compare_snapshots`; the
verifier runs them on real frames through `t47b_replay.run_b3_record`, the unit tests on synthetic
clouds through `lift_and_ground` / `fuse_and_ground`).

Three transformations of the attention input are distinguished:

``x k``           every camera's map multiplied by the same k > 0 (sum vs mean of a fixed head set).
``per-camera``    camera c multiplied by its own k_c (what `mass_norm_per_camera` does to the
                  camera-to-camera ratio, and what any change of head weighting does when the heads
                  split their mass differently over the cameras).
``head reweight`` the map itself changes shape (a different weighting of the heads) — everything that
                  reads values is then allowed to move; the column records *how* it reads them.

Floating point: x k is exact in IEEE arithmetic only for k a power of two. The pipeline lifts the
16x16 map with float32 bilinear weights (`attention_lifting.py:115-117`) and normalises
``(a - lo) / (hi - lo)`` in float32 (`:283`); for k = 3 the products round differently, so
normalised values may differ in the last bits and a percentile cut can move a tie. The numeric check
therefore reports **decision identity** (seeds, peak, clusters, split, status, chosen cluster) and
**bit identity** (additionally the raw ratio, normalised values, scores, centroid) separately, and
the CLI runs k = 4 next to k = 3 as an exact-arithmetic control.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------ the audit


@dataclasses.dataclass(frozen=True)
class AuditPoint:
    id: str
    file: str  # repo-relative
    line: int  # where `code` was on 2026-10-12 (working tree incl. T46's uncommitted edits)
    code: str  # a substring of that line; `locate` re-finds it if the file moved
    what: str
    reads: str  # raw | normalized | score | presence | config
    absolute: str  # the absolute constant compared against, or "none"
    default_active: bool
    x_k: str  # invariant | variant | invariant (float rounding) | variant (degenerate only)
    per_camera: str
    head_reweight: str
    why: str


AUDIT: tuple[AuditPoint, ...] = (
    AuditPoint("L1", "AG3S/ag3s/stages/attention_lifting.py", 115,
               "top = src[r0, c0] * (1.0 - wx) + src[r0, c1] * wx",
               "bilinear 16x16 -> pixel resampling of the map (float32)", "raw", "none", True,
               "invariant (float rounding)", "invariant (within a camera)", "variant",
               "linear in the map: k*interp(m) = interp(k*m) in reals; float32 products round "
               "differently unless k is a power of two"),
    AuditPoint("L2", "AG3S/ag3s/stages/attention_lifting.py", 272,
               "lo, hi = np.percentile(a, config.percentile_range)",
               "percentile normalisation band (5, 99) of one camera's lifted points", "raw", "none",
               True, "invariant (float rounding)", "invariant (per camera)", "variant",
               "percentiles are scale-equivariant; the result is only used as a ratio (L4)"),
    AuditPoint("L3", "AG3S/ag3s/stages/attention_lifting.py", 278, "if hi - lo <= _EPS:",
               "degenerate-band guard: span <= 1e-12 -> all zeros (=> NO_ATTENTION downstream)",
               "raw", "1e-12 (_EPS, attention_lifting.py:33)", True,
               "variant (degenerate only)", "variant (degenerate only)", "variant (degenerate only)",
               "an absolute constant on the raw span; flips only for spans in (1e-12/k, 1e-12] — "
               "real spans are softmax-mass scale (>= 1e-6 per 16x16 map); a constant map is zeros "
               "at any scale"),
    AuditPoint("L4", "AG3S/ag3s/stages/attention_lifting.py", 283,
               "return np.clip((a - lo) / (hi - lo), 0.0, 1.0)",
               "normalised value = clip((a - lo) / (hi - lo))", "raw", "none", True,
               "invariant (float rounding)", "invariant (per camera)", "variant",
               "a ratio of differences; k cancels (exactly for powers of two)"),
    AuditPoint("L5", "AG3S/ag3s/stages/attention_lifting.py", 262,
               "z = (a - a.max()) / float(config.temperature)",
               "softmax normalisation (non-default)", "raw", "temperature 1.0 (config.py:106)",
               False, "variant", "variant", "variant",
               "exp((a - max)/T) is not scale-free: x k is a temperature change. Off by default "
               "(normalization='percentile', config.py:104)"),
    AuditPoint("L6", "AG3S/ag3s/stages/attention_lifting.py", 259, 'if mode == "none":',
               "normalisation 'none' passes raw values to every downstream threshold (non-default)",
               "raw", "all of G1/G3/G6/G7/G8 then read raw", False, "variant", "variant", "variant",
               "with 'none' the flat-map guard, seed_threshold, score and switch/destination minimum "
               "scores compare raw mass. Off by default"),
    AuditPoint("M1", "AG3S/ag3s/types.py", 449, "return self.attention_map is not None",
               "has_attention: a camera with a map vs without", "presence", "none", True,
               "invariant", "invariant", "invariant", "presence only, no value"),
    AuditPoint("M2", "AG3S/ag3s/runtime/multiview.py", 384,
               "raw_attention = np.maximum.reduceat(obs_raw[order], starts)",
               "multi-view fusion: per fused point, max of the raw lifted values of all cameras",
               "raw", "none", True, "invariant", "variant", "variant",
               "max commutes with a common positive k; with per-camera k_c the camera whose raw "
               "value wins changes — this is where camera-relative amplitude matters"),
    AuditPoint("M3", "AG3S/ag3s/runtime/multiview.py", 399,
               "attention = normalize_attention(raw_attention, attention_config)",
               "fused normalisation (F2): one percentile band over the fused raw values", "raw",
               "none (its degenerate guard is L3)", True, "invariant (float rounding)", "variant",
               "variant",
               "same function as L2-L4 (incl. the L3 1e-12 guard) on the fused raw values; under "
               "per-camera k_c the fused distribution (M2) changes, so the band does"),
    AuditPoint("G1", "AG3S/ag3s/stages/target_grounding.py", 1662,
               "if float(attention.max()) - float(attention.min()) <= _EPS:",
               "flat-map guard on the fused attention -> NO_ATTENTION", "normalized",
               "1e-9 (_EPS, target_grounding.py:120)", True, "invariant", "invariant (after M3)",
               "variant (degenerate only)",
               "reads the normalised [0, 1] array; with normalization='none' it would be raw (L6)"),
    AuditPoint("G2", "AG3S/ag3s/stages/target_grounding.py", 1672,
               "peak_index = int(np.argmax(attention_cloud.peak_signal))",
               "attention peak = argmax of the fused RAW values (peak_signal)", "raw", "none", True,
               "invariant (float rounding)", "variant", "variant",
               "argmax commutes with a common k (float ties possible for k not a power of two); "
               "per-camera k_c changes which camera's raw value is the global max (via M2); "
               "the peak feeds every cluster's peak_proximity (G5)"),
    AuditPoint("G3", "AG3S/ag3s/stages/target_grounding.py", 288,
               "cut = float(np.percentile(att, seed_percentile))",
               "seeds = normalised fused attention >= its 95th percentile", "normalized", "none",
               True, "invariant", "invariant (after M3)", "variant",
               "a percentile of the normalised array; rank-based"),
    AuditPoint("G3b", "AG3S/ag3s/stages/target_grounding.py", 286,
               "keep = np.nonzero(att >= float(seed_threshold))[0]",
               "absolute seed threshold (non-default, seed_threshold=None, config.py:108)",
               "normalized", "seed_threshold (unset)", False, "invariant", "invariant (after M3)",
               "variant", "compares the normalised array, so x k cancels; raw only under L6"),
    AuditPoint("G4", "AG3S/ag3s/stages/target_grounding.py", 291,
               "if keep.size > config.max_seed_points:",
               "seed cap 4000 strongest (argsort of normalised attention)", "normalized", "none",
               True, "invariant", "invariant (after M3)", "variant", "rank-based"),
    AuditPoint("G5", "AG3S/ag3s/stages/target_grounding.py", 1590,
               "score = float((w_a * attention_score + w_g * spatial_consistency) / (w_a + w_g))",
               "cluster score = 0.7 mean normalised attention + 0.3 geometry (peak from G2)",
               "normalized", "none (weights config.py:361-362)", True,
               "invariant (float rounding)", "variant", "variant",
               "mean of normalised values; the absolute level of the score is scale-free under "
               "x k but moves with the map's shape (it is compared in G7/G8/D1)"),
    AuditPoint("G6", "AG3S/ag3s/stages/target_grounding.py", 1729,
               "ratio_attention=(attention_cloud.raw_attention",
               "split ratio source: raw only when clustering.attention_split_ratio_source='raw' "
               "(default 'normalized', config.py:252)", "config", "none", True, "invariant",
               "variant", "variant", "selects which array S1/S2 read"),
    AuditPoint("S1", "AG3S/ag3s/stages/attention_split.py", 276,
               "reason = SPLIT if ratio >= float(min_ratio) else ATTENTION_NOT_DISCRIMINATIVE",
               "split gate: mean(target comp.) / max mean(other comps.) >= 2.0", "normalized",
               "2.0 (attention_split_ratio, config.py:244) — a RATIO", True,
               "invariant (float rounding)", "variant", "variant",
               "a ratio of two means of the same array; k cancels (raw source: also cancels)"),
    AuditPoint("S2", "AG3S/ag3s/stages/attention_split.py", 270, "if top <= _EPS:",
               "split zero guards top/rest <= 1e-12", "normalized",
               "1e-12 (_EPS, attention_split.py:82)", True, "invariant",
               "invariant (after M3)", "variant (degenerate only)",
               "on the normalised array by default; on raw only with ratio source 'raw' (G6), where "
               "it is an absolute constant on raw mass that flips only for means in "
               "(1e-12/k, 1e-12]"),
    AuditPoint("S3", "AG3S/ag3s/stages/attention_split.py", 267,
               "comps.sort(key=lambda c: (-c.attention_mass, int(c.indices.min())))",
               "split target candidate = largest normalised attention mass", "normalized", "none",
               True, "invariant", "invariant (after M3)", "variant", "rank of sums"),
    AuditPoint("G7", "AG3S/ag3s/stages/target_grounding.py", 1747,
               "if best.target_score < cfg.target_score_threshold:",
               "score gate (default 0.0 = never reject, config.py:166)", "score",
               "target_score_threshold 0.0", True, "invariant", "invariant (after M3)", "variant",
               "compares G5 (normalised) with an absolute 0.0; inert at the default"),
    AuditPoint("G8", "AG3S/ag3s/stages/target_grounding.py", 1198,
               "and float(challenger.target_score) >= self.switch_min_score)",
               "TargetConfirm: a challenger frame counts only at score >= 0.1", "score",
               "0.1 (target_switch_min_score, config.py:186)", True, "invariant",
               "variant", "variant",
               "absolute 0.1 on G5, which is built from NORMALISED attention: x k cancels; a "
               "different head weighting changes the normalised distribution and so the score "
               "level (an absolute threshold on a shape-dependent quantity, not on raw mass)"),
    AuditPoint("D1", "AG3S/ag3s/stages/admissibility.py", 312,
               "if graspable[0] is False and float(leader.target_score) >= self.min_score:",
               "destination registration: non-graspable leader at score >= 0.1", "score",
               "0.1 (= target_switch_min_score)", True, "invariant", "variant", "variant",
               "same as G8"),
    AuditPoint("P1", "AG3S/ag3s/runtime/pipeline.py", 3185,
               "peak_signal = attention_cloud.peak_signal[indices]",
               "seed_camera provenance: raw argmax over the target, then per-camera normalised argmax",
               "raw", "none", True, "invariant (float rounding)", "variant", "variant",
               "record field only (no decision)"),
    AuditPoint("P2", "AG3S/trajopt/grasp_latch.py", 364,
               "confident = score >= self._cfg.score_ratio * max(runner_up, 1e-9)",
               "GraspLatch confidence: score >= 1.3 x runner-up (live only, not in offline e2e)",
               "score", "1e-9 floor on the runner-up score", True, "invariant", "variant",
               "variant", "a ratio of G5 scores; not exercised by the offline replay (no latch)"),
)


def locate(point: AuditPoint, root: Optional[pathlib.Path] = None) -> dict[str, Any]:
    """Where `point.code` is now: the occurrence nearest the recorded line (None when gone)."""
    if root is None:
        from common.workspace import PROJECT_ROOT

        root = pathlib.Path(PROJECT_ROOT)
    path = pathlib.Path(root) / point.file
    if not path.is_file():
        return {"found": False, "line_now": None, "n_matches": 0}
    lines = path.read_text().splitlines()
    hits = [i + 1 for i, ln in enumerate(lines) if point.code in ln]
    if not hits:
        return {"found": False, "line_now": None, "n_matches": 0}
    now = min(hits, key=lambda h: abs(h - point.line))
    return {"found": True, "line_now": now, "n_matches": len(hits), "moved": now != point.line}


def audit_table(root: Optional[pathlib.Path] = None) -> list[dict[str, Any]]:
    """`AUDIT` as JSON-ready rows, each with its current location (`locate`)."""
    return [{**dataclasses.asdict(p), **locate(p, root)} for p in AUDIT]


# ------------------------------------------------------------------- numeric comparison


def snapshot(attention_cloud, grounding, target=None) -> dict[str, Any]:
    """Everything grounding decided on one frame, as plain arrays (for `compare_snapshots`)."""
    raw = attention_cloud.raw_attention
    att = np.asarray(attention_cloud.attention, np.float32)
    clusters = [(np.asarray(c.point_indices, np.int64), float(c.target_score),
                 float(c.mean_attention), float(c.attention_mass)) for c in grounding.clusters]
    splits = [(str(d.reason), None if d.ratio is None else float(d.ratio))
              for d in getattr(grounding, "splits", ()) or ()]
    tgt = target if target is not None else grounding.target
    return {
        "points": np.asarray(attention_cloud.points, np.float64),
        "raw": att if raw is None else np.asarray(raw, np.float32),
        "norm": att,
        "seeds": np.asarray(grounding.seed_indices, np.int64),
        "peak": int(grounding.attention_peak_index),
        "status": str(getattr(grounding.status, "value", grounding.status)),
        "clusters": clusters,
        "splits": splits,
        "has_target": tgt is not None,
        "chosen": (np.zeros(0, np.int64) if tgt is None
                   else np.asarray(tgt.point_indices, np.int64)),
        "centroid": None if tgt is None else np.asarray(tgt.centroid, np.float64),
    }


def snapshot_from_debug(constraint_set, debug: dict) -> dict[str, Any]:
    """`snapshot` of an `AG3S.process_multi_debug` result (attention cloud + grounding + target)."""
    return snapshot(debug["attention_cloud"], debug["grounding"], constraint_set.target)


#: Pipeline order of the decision checks; `first_difference` names the first that fails.
DECISION_ORDER = ("cloud", "raw_order", "normalized", "seeds", "peak", "clusters", "split",
                  "scores", "status", "chosen", "centroid")


def compare_snapshots(a: dict, b: dict, factor: float) -> dict[str, Any]:
    """`b` is `a`'s frame with the attention input multiplied by `factor`. Scalars only.

    ``decision_identical`` — same cloud, seeds, peak, cluster partition (in order), split reasons,
    status, has_target and chosen cluster. ``bit_identical`` — additionally ``b.raw ==
    float32(factor * a.raw)`` everywhere, normalised values, cluster scores and centroid equal bit for
    bit. ``first_difference`` — the first failing item of `DECISION_ORDER` (None when all equal).
    """
    k = float(factor)
    out: dict[str, Any] = {"factor": k}
    same_cloud = (a["points"].shape == b["points"].shape
                  and np.array_equal(a["points"], b["points"]))
    out["cloud_equal"] = bool(same_cloud)
    out["n_points"] = int(len(a["points"]))
    ra, rb = np.asarray(a["raw"], np.float32), np.asarray(b["raw"], np.float32)
    if same_cloud and ra.shape == rb.shape and ra.size:
        expect = ra.astype(np.float64) * k
        scale = float(np.max(np.abs(expect))) or 1.0
        out["raw_max_rel_err"] = float(np.max(np.abs(rb.astype(np.float64) - expect)) / scale)
        out["raw_scaled_bit_equal"] = bool(np.array_equal(rb, (ra * np.float32(k)).astype(np.float32)))
        out["raw_order_equal"] = bool(np.array_equal(np.argsort(ra, kind="stable"),
                                                     np.argsort(rb, kind="stable")))
        na, nb = np.asarray(a["norm"], np.float32), np.asarray(b["norm"], np.float32)
        diff = na != nb
        out["norm_bit_equal"] = bool(not diff.any())
        out["norm_n_diff"] = int(diff.sum())
        out["norm_max_abs_diff"] = float(np.max(np.abs(na.astype(np.float64) - nb)))
    else:
        out.update({"raw_max_rel_err": float("nan"), "raw_scaled_bit_equal": False,
                    "raw_order_equal": False, "norm_bit_equal": False, "norm_n_diff": None,
                    "norm_max_abs_diff": float("nan")})
    out["seeds_equal"] = bool(np.array_equal(a["seeds"], b["seeds"]))
    out["n_seeds_a"], out["n_seeds_b"] = int(a["seeds"].size), int(b["seeds"].size)
    out["n_seeds_symdiff"] = int(np.setxor1d(a["seeds"], b["seeds"]).size)
    out["peak_equal"] = a["peak"] == b["peak"]
    ca, cb = a["clusters"], b["clusters"]
    out["n_clusters_a"], out["n_clusters_b"] = len(ca), len(cb)
    out["clusters_equal"] = (len(ca) == len(cb)
                             and all(np.array_equal(x[0], y[0]) for x, y in zip(ca, cb)))
    if len(ca) == len(cb) and ca:
        sd = [abs(x[1] - y[1]) for x, y in zip(ca, cb)]
        out["scores_bit_equal"] = all(x[1] == y[1] for x, y in zip(ca, cb))
        out["scores_max_abs_diff"] = float(max(sd))
    else:
        out["scores_bit_equal"] = len(ca) == len(cb)
        out["scores_max_abs_diff"] = 0.0 if len(ca) == len(cb) else float("nan")
    sa, sb = a["splits"], b["splits"]
    out["split_reasons_equal"] = [s[0] for s in sa] == [s[0] for s in sb]
    ratios = [(x[1], y[1]) for x, y in zip(sa, sb) if x[1] is not None and y[1] is not None
              and np.isfinite(x[1]) and np.isfinite(y[1])]
    out["split_ratio_max_abs_diff"] = (float(max(abs(x - y) for x, y in ratios))
                                       if ratios else 0.0)
    out["status_a"], out["status_b"] = a["status"], b["status"]
    out["status_equal"] = a["status"] == b["status"]
    out["has_target_equal"] = a["has_target"] == b["has_target"]
    out["chosen_equal"] = bool(np.array_equal(a["chosen"], b["chosen"]))
    if a["centroid"] is None or b["centroid"] is None:
        out["centroid_bit_equal"] = a["centroid"] is None and b["centroid"] is None
        out["centroid_max_abs_diff_m"] = 0.0 if out["centroid_bit_equal"] else float("nan")
    else:
        out["centroid_bit_equal"] = bool(np.array_equal(a["centroid"], b["centroid"]))
        out["centroid_max_abs_diff_m"] = float(np.max(np.abs(a["centroid"] - b["centroid"])))
    checks = {
        "cloud": out["cloud_equal"], "raw_order": out["raw_order_equal"],
        "normalized": out["norm_bit_equal"], "seeds": out["seeds_equal"],
        "peak": out["peak_equal"], "clusters": out["clusters_equal"],
        "split": out["split_reasons_equal"], "scores": out["scores_bit_equal"],
        "status": out["status_equal"] and out["has_target_equal"],
        "chosen": out["chosen_equal"], "centroid": out["centroid_bit_equal"],
    }
    out["decision_identical"] = bool(all(checks[c] for c in (
        "cloud", "seeds", "peak", "clusters", "split", "status", "chosen")))
    out["bit_identical"] = bool(out["decision_identical"] and out["raw_scaled_bit_equal"]
                                and all(checks.values()))
    out["first_difference"] = next((c for c in DECISION_ORDER if not checks[c]), None)
    return out


def summarize_b3(rows: Sequence[dict]) -> dict[str, Any]:
    """Per (source, factor): frames compared, decision-identical, bit-identical, first differences."""
    out: dict[str, Any] = {}
    for key in sorted({(r["source"], float(r["factor"])) for r in rows}):
        rs = [r for r in rows if (r["source"], float(r["factor"])) == key]
        firsts: dict[str, int] = {}
        for r in rs:
            if r.get("first_difference"):
                firsts[r["first_difference"]] = firsts.get(r["first_difference"], 0) + 1
        out[f"{key[0]}@x{key[1]:g}"] = {
            "source": key[0], "factor": key[1], "n_frames": len(rs),
            "decision_identical": sum(bool(r["decision_identical"]) for r in rs),
            "bit_identical": sum(bool(r["bit_identical"]) for r in rs),
            "norm_bit_equal": sum(bool(r.get("norm_bit_equal")) for r in rs),
            "raw_scaled_bit_equal": sum(bool(r.get("raw_scaled_bit_equal")) for r in rs),
            "first_difference_counts": firsts,
            "frames_not_decision_identical": [
                {"record": r.get("record"), "frame": r.get("frame"),
                 "first_difference": r.get("first_difference"),
                 "status_a": r.get("status_a"), "status_b": r.get("status_b")}
                for r in rs if not r["decision_identical"]],
            "max_raw_rel_err": max((float(r["raw_max_rel_err"]) for r in rs
                                    if np.isfinite(float(r["raw_max_rel_err"]))), default=float("nan")),
            "max_norm_abs_diff": max((float(r["norm_max_abs_diff"]) for r in rs
                                      if np.isfinite(float(r["norm_max_abs_diff"]))),
                                     default=float("nan")),
        }
    return out


# ------------------------------------------------------------------ synthetic front ends


def lift_and_ground(cloud, attention_map, *, image_hw, config=None, exclude_mask=None):
    """One camera: `attention_lifting.lift` + `target_grounding.ground_target` at `config`'s values
    (default `AG3SConfig()`), no cross-frame confirm. Returns ``(attention_cloud, grounding)``."""
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.stages.attention_lifting import lift
    from AG3S.ag3s.stages.target_grounding import ground_target

    cfg = config or AG3SConfig()
    ac = lift(cloud, np.asarray(attention_map, np.float32), cfg.attention, image_hw=image_hw)
    g = ground_target(ac, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                      seed_threshold=cfg.attention.seed_threshold, exclude_mask=exclude_mask,
                      min_radius=cfg.geometry.min_radius, timestamp=0.0)
    return ac, g


def fuse_and_ground(camera_clouds: Sequence, attention_maps: Sequence, *, image_hw, voxel_size=0.01,
                    config=None):
    """Several cameras: lift each (as `multiview.process_observation` does), `multiview.fuse`
    (raw max + fused normalisation), then `ground_target`. ``(attention_cloud, grounding, fused)``."""
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.multiview import CameraResult, fuse
    from AG3S.ag3s.stages.attention_lifting import lift
    from AG3S.ag3s.stages.target_grounding import ground_target
    from AG3S.ag3s.types import AttentionPointCloud, CameraID

    cfg = config or AG3SConfig()
    ids = list(CameraID)
    results = []
    for i, (cloud, m) in enumerate(zip(camera_clouds, attention_maps)):
        ac = lift(cloud, np.asarray(m, np.float32), cfg.attention, image_hw=image_hw)
        results.append(CameraResult(camera_id=ids[i], cloud=cloud, attention=ac.attention,
                                    raw_attention=ac.raw_attention, timestamp=0.0,
                                    T_base_cam=np.eye(4), had_attention=True))
    fused, att, raw = fuse(results, voxel_size=voxel_size, attention_config=cfg.attention)
    ac = AttentionPointCloud(fused.as_pointcloud(), att, raw)
    g = ground_target(ac, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                      seed_threshold=cfg.attention.seed_threshold,
                      min_radius=cfg.geometry.min_radius, timestamp=0.0)
    return ac, g, fused


__all__ = ["AUDIT", "AuditPoint", "DECISION_ORDER", "audit_table", "compare_snapshots",
           "fuse_and_ground", "lift_and_ground", "locate", "snapshot", "snapshot_from_debug",
           "summarize_b3"]
