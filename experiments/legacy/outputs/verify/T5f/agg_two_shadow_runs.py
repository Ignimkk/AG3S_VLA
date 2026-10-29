"""T5f — 두 shadow 기록(threshold 0.25 / 1 등+confirm 3)을 같은 규칙으로 집계한다.

`outputs/verify/T5c/agg_shadow_record.py` 가 하던 것을 **두 기록에 동시에** 적용한 것이다.
기록만 읽는다 — `benchmark/**` 를 건드리지 않고, 새로 촬영하지 않는다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/agg_two_shadow_runs.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
from collections import Counter

RUNS = {
    "before_threshold_0p25": "outputs/live_test/20260925_t5shadow/shadow_ep1807",
    "after_top1_confirm3": "outputs/live_test/20260925_t5e_shadow/shadow_ep1807",
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f/shadow_agg_two_runs.json'))
CAPTURE_ORDER = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * q / 100.0
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def stats(xs):
    if not xs:
        return None
    return {"n": len(xs), "min": round(min(xs), 2), "p50": round(pct(xs, 50), 2),
            "p95": round(pct(xs, 95), 2), "max": round(max(xs), 2),
            "mean": round(sum(xs) / len(xs), 2)}


def agg(root: str) -> dict:
    p = pathlib.Path(_ag3s_workspace_path('')) / root / "frames.jsonl"
    recs = [json.loads(l) for l in p.open()]
    obs = [r for r in recs if r["kind"] == "observation"]
    plan = sorted((r for r in recs if r["kind"] == "planning"), key=lambda r: r["seq"])
    ctrl = [r for r in recs if r["kind"] == "control"]

    # --- observation: ag3s status / reason codes / camera span ---------------------------
    by_t_obs = {r["t_step"]: r for r in obs}
    spans, chunks = [], []
    for r in plan:
        t = r["t_step"]
        o = by_t_obs.get(t, {})
        cs = o.get("camera_stamps") or {}
        span_ms = (max(cs.values()) - min(cs.values())) * 1000.0 if cs else None
        if span_ms is not None:
            spans.append(span_ms)
        v = r.get("verdict") or {}
        tiers = (r.get("field") or {}).get("tiers") or []
        chunks.append({
            "seq": r["seq"],
            "t_step": t,
            "safe": bool(r.get("safe")),
            "ipc": r.get("ipc"),
            "hold_reason": r.get("hold_reason", ""),
            "executed_chunk": r.get("executed_chunk"),
            "chunk_shape": r.get("chunk_shape"),
            "ag3s_status": v.get("ag3s_status"),
            "geometry_certified": v.get("geometry_certified"),
            "trajopt_status": v.get("trajopt_status"),
            "max_violation_m": v.get("max_violation_m"),
            "verdict_notes": v.get("notes") or [],
            "n_tiers": len(tiers),
            "tier_voxels_m": [t2.get("voxel_size_m") for t2 in tiers],
            "field_state": (r.get("field") or {}).get("state"),
            "field_backend": (r.get("field") or {}).get("backend"),
            "obs_ag3s_status": o.get("ag3s_status"),
            "obs_grounding_status": o.get("grounding_status"),
            "obs_validity": o.get("validity"),
            "obs_reason_codes": o.get("ag3s_reason_codes") or [],
            "obs_notes": o.get("notes") or [],
            "camera_skew_sec": o.get("camera_skew_sec"),
            "camera_span_ms": None if span_ms is None else round(span_ms, 2),
            "timing_ms": v.get("timing_ms"),
        })

    def seqs(pred):
        return [c["seq"] for c in chunks if pred(c)]

    gaps = {f"{a}->{b}": [] for a, b in zip(CAPTURE_ORDER, CAPTURE_ORDER[1:])}
    for r in obs:
        cs = r.get("camera_stamps") or {}
        for a, b in zip(CAPTURE_ORDER, CAPTURE_ORDER[1:]):
            if a in cs and b in cs:
                gaps[f"{a}->{b}"].append((cs[b] - cs[a]) * 1000.0)

    viol = [c for c in chunks if c["trajopt_status"] == "violated"]
    return {
        "record": root,
        "n_chunks": len(chunks),
        "n_observation": len(obs),
        "n_control": len(ctrl),
        "counts": {
            "safe_true": len(seqs(lambda c: c["safe"])),
            "safe_false": len(seqs(lambda c: not c["safe"])),
            "ag3s_status": dict(Counter(c["ag3s_status"] for c in chunks)),
            "obs_ag3s_status": dict(Counter(c["obs_ag3s_status"] for c in chunks)),
            "obs_grounding_status": dict(Counter(c["obs_grounding_status"] for c in chunks)),
            "obs_validity": dict(Counter(c["obs_validity"] for c in chunks)),
            "trajopt_status": dict(Counter(c["trajopt_status"] for c in chunks)),
            "geometry_certified_true": len(seqs(lambda c: c["geometry_certified"] is True)),
            "n_tiers": dict(Counter(c["n_tiers"] for c in chunks)),
            "reason_code_combos": dict(Counter(",".join(c["obs_reason_codes"]) or "(none)"
                                               for c in chunks)),
            "executed_chunk": dict(Counter(c["executed_chunk"] for c in chunks)),
            "ipc": dict(Counter(c["ipc"] for c in chunks)),
        },
        "seqs": {
            "safe_false": seqs(lambda c: not c["safe"]),
            "ag3s_no_target": seqs(lambda c: c["ag3s_status"] == "no_target"),
            "ag3s_degraded": seqs(lambda c: c["ag3s_status"] == "degraded"),
            "trajopt_violated": seqs(lambda c: c["trajopt_status"] == "violated"),
            "grounding_low_score": seqs(lambda c: c["obs_grounding_status"] == "low_score"),
            "two_tiers": seqs(lambda c: c["n_tiers"] == 2),
            "one_tier": seqs(lambda c: c["n_tiers"] == 1),
            "with_reason_codes": seqs(lambda c: bool(c["obs_reason_codes"])),
            "camera_span_at_or_over_100ms": seqs(
                lambda c: c["camera_span_ms"] is not None and c["camera_span_ms"] >= 100.0),
        },
        "violated_detail": [
            {"seq": c["seq"], "t_step": c["t_step"],
             "max_violation_m": c["max_violation_m"],
             "max_violation_mm": None if c["max_violation_m"] is None
             else round(c["max_violation_m"] * 1000.0, 4),
             "ag3s_status": c["ag3s_status"],
             "geometry_certified": c["geometry_certified"],
             "n_tiers": c["n_tiers"],
             "notes": c["verdict_notes"]} for c in viol],
        "camera_span_ms": stats(spans),
        "n_span_at_or_over_100ms": sum(1 for s in spans if s >= 100.0),
        "n_span_at_or_over_90ms": sum(1 for s in spans if s >= 90.0),
        "mean_gap_ms": {k: (round(sum(v) / len(v), 2) if v else None) for k, v in gaps.items()},
        "capture_order": list(CAPTURE_ORDER),
        "chunks": chunks,
    }


def main() -> None:
    out = {"runs": {k: agg(v) for k, v in RUNS.items()},
           "note_camera_span": "span = max(camera_stamps) - min(camera_stamps) per observation, ms",
           "seq_to_t_step": "t_step = (seq - 1) * 8"}
    a, b = out["runs"]["before_threshold_0p25"], out["runs"]["after_top1_confirm3"]
    out["side_by_side"] = {
        "safe": [a["counts"]["safe_true"], b["counts"]["safe_true"]],
        "unsafe": [a["counts"]["safe_false"], b["counts"]["safe_false"]],
        "ag3s_no_target": [len(a["seqs"]["ag3s_no_target"]), len(b["seqs"]["ag3s_no_target"])],
        "ag3s_degraded": [len(a["seqs"]["ag3s_degraded"]), len(b["seqs"]["ag3s_degraded"])],
        "trajopt_violated": [len(a["seqs"]["trajopt_violated"]),
                             len(b["seqs"]["trajopt_violated"])],
        "two_tier_chunks": [len(a["seqs"]["two_tiers"]), len(b["seqs"]["two_tiers"])],
        "camera_span_ms": [a["camera_span_ms"], b["camera_span_ms"]],
        "n_span_ge_100ms": [a["n_span_at_or_over_100ms"], b["n_span_at_or_over_100ms"]],
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["side_by_side"], indent=1, ensure_ascii=False))
    for k, r in out["runs"].items():
        print(f"\n== {k}: {r['record']}")
        print("  counts:", json.dumps(r["counts"], ensure_ascii=False))
        print("  violated:", json.dumps(r["violated_detail"], ensure_ascii=False)[:2000])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
