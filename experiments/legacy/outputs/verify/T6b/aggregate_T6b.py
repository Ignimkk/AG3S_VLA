"""T6b — aggregate the per-chunk verdicts of three live runs side by side.

Reads only the recorded ``frames.jsonl`` / ``completeness.json`` / ``manifest.json`` of the three
runs. Nothing is simulated, nothing is replayed: the T6 execute run diverged from the shadow runs
the moment it started to hold, so replaying the reference episode would not visit the poses T6
actually visited.

Writes ``chunks_T6b.json`` (per-chunk rows, three runs) and ``aggregate_T6b.json`` (the counts and
percentiles that the verify.json quotes).
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import os
from statistics import median

ROOT = _ag3s_workspace_path('')
OUT = os.path.join(ROOT, "outputs/verify/T6b")

RUNS = [
    ("shadow_threshold", "outputs/live_test/20260925_t5shadow/shadow_ep1807"),
    ("shadow_rank1", "outputs/live_test/20260925_t5e_shadow/shadow_ep1807"),
    ("execute", "outputs/live_test/20260926_t6/execute_ep1807"),
]


def pct(values, q):
    """Nearest-rank percentile on a sorted copy; q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = int(round((q / 100.0) * (len(s) - 1)))
    return s[idx]


def load(run_dir):
    planning, observation, control = [], [], []
    with open(os.path.join(ROOT, run_dir, "frames.jsonl")) as fh:
        for line in fh:
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "planning":
                planning.append(rec)
            elif kind == "observation":
                observation.append(rec)
            elif kind == "control":
                control.append(rec)
    completeness = json.load(open(os.path.join(ROOT, run_dir, "completeness.json")))
    manifest = json.load(open(os.path.join(ROOT, run_dir, "manifest.json")))
    return planning, observation, control, completeness, manifest


def run_ranges(flags):
    """Contiguous seq ranges where flags[seq] is True. flags is a dict seq -> bool."""
    out = []
    for seq in sorted(flags):
        if not flags[seq]:
            continue
        if out and out[-1][1] == seq - 1:
            out[-1][1] = seq
        else:
            out.append([seq, seq])
    return [[a, b] for a, b in out]


def main():
    per_run = {}
    chunk_rows = {}

    for name, run_dir in RUNS:
        planning, observation, control, completeness, manifest = load(run_dir)
        obs_by_step = {o["t_step"]: o for o in observation}

        rows = []
        for rec in sorted(planning, key=lambda r: r["seq"]):
            verdict = rec.get("verdict") or {}
            field = rec.get("field") or {}
            tiers = field.get("tiers") or []
            timing = verdict.get("timing_ms") or {}
            obs = obs_by_step.get(rec["t_step"]) or {}
            rows.append({
                "seq": rec["seq"],
                "t_step": rec["t_step"],
                "safe": bool(rec.get("safe")),
                "ipc": rec.get("ipc"),
                "ag3s_status": verdict.get("ag3s_status"),
                "geometry_certified": verdict.get("geometry_certified"),
                "trajopt_status": verdict.get("trajopt_status"),
                "max_violation_m": verdict.get("max_violation_m"),
                "ag3s_reason_codes": rec.get("ag3s_reason_codes") or [],
                "n_tiers": len(tiers),
                "tier_voxel_m": [t.get("voxel_size_m") for t in tiers],
                "field_state": field.get("state"),
                "field_backend": field.get("backend"),
                "hold_reason": rec.get("hold_reason") or "",
                "shadow": rec.get("shadow"),
                "executed_chunk": rec.get("executed_chunk"),
                "timing_infer_ms": timing.get("infer"),
                "timing_ag3s_ms": timing.get("ag3s"),
                "timing_trajopt_ms": timing.get("trajopt"),
                "timing_total_ms": timing.get("total"),
                "policy_infer_ms": (rec.get("timing_ms") or {}).get("policy_infer"),
                "camera_skew_sec": obs.get("camera_skew_sec"),
                "obs_ag3s_status": obs.get("ag3s_status"),
                "obs_validity": obs.get("validity"),
            })
        chunk_rows[name] = rows

        safe_seqs = [r["seq"] for r in rows if r["safe"]]
        hold_seqs = [r["seq"] for r in rows if not r["safe"]]

        # hold classification by (ag3s_status, trajopt_status)
        hold_classes = {}
        for r in rows:
            if r["safe"]:
                continue
            key = "(%s, %s)" % (r["ag3s_status"], r["trajopt_status"])
            hold_classes[key] = hold_classes.get(key, 0) + 1

        safe_classes = {}
        for r in rows:
            if not r["safe"]:
                continue
            key = "(%s, %s)" % (r["ag3s_status"], r["trajopt_status"])
            safe_classes[key] = safe_classes.get(key, 0) + 1

        # violation magnitudes, three populations
        def mm_stats(vals):
            if not vals:
                return {"n": 0}
            return {
                "n": len(vals),
                "min_mm": min(vals),
                "p50_mm": median(vals),
                "p95_mm": pct(vals, 95),
                "max_mm": max(vals),
            }

        all_mm = [r["max_violation_m"] * 1e3 for r in rows if r["max_violation_m"] is not None]
        hold_mm = [r["max_violation_m"] * 1e3 for r in rows
                   if not r["safe"] and r["max_violation_m"] is not None]
        hold_mm_nonzero = [v for v in hold_mm if v > 0.0]

        # timing percentiles per stage
        stages = {}
        for key, col in [("infer", "timing_infer_ms"), ("ag3s", "timing_ag3s_ms"),
                         ("trajopt", "timing_trajopt_ms"), ("total", "timing_total_ms"),
                         ("policy_infer_client", "policy_infer_ms")]:
            vals = [r[col] for r in rows if r[col] is not None]
            vals_no_first = [r[col] for r in rows if r[col] is not None and r["seq"] != 1]
            stages[key] = {
                "n": len(vals),
                "p50_ms": pct(vals, 50),
                "p95_ms": pct(vals, 95),
                "max_ms": max(vals) if vals else None,
                "seq1_ms": next((r[col] for r in rows if r["seq"] == 1), None),
                "p50_ms_excl_seq1": pct(vals_no_first, 50),
                "p95_ms_excl_seq1": pct(vals_no_first, 95),
                "max_ms_excl_seq1": max(vals_no_first) if vals_no_first else None,
            }

        skew = [o["camera_skew_sec"] * 1e3 for o in observation
                if o.get("camera_skew_sec") is not None]
        skew_stats = {
            "n": len(skew),
            "p50_ms": pct(skew, 50),
            "p95_ms": pct(skew, 95),
            "max_ms": max(skew) if skew else None,
            "min_ms": min(skew) if skew else None,
            "over_100ms_frames": sum(1 for v in skew if v > 100.0),
        }

        reason_counts = {}
        for r in rows:
            for code in r["ag3s_reason_codes"]:
                reason_counts[code] = reason_counts.get(code, 0) + 1

        tier_counts = {}
        for r in rows:
            tier_counts[str(r["n_tiers"])] = tier_counts.get(str(r["n_tiers"]), 0) + 1

        ag3s_status_counts = {}
        trajopt_status_counts = {}
        for r in rows:
            ag3s_status_counts[str(r["ag3s_status"])] = ag3s_status_counts.get(str(r["ag3s_status"]), 0) + 1
            trajopt_status_counts[str(r["trajopt_status"])] = trajopt_status_counts.get(str(r["trajopt_status"]), 0) + 1

        # cross-tab: hold class against whether a penetration figure was reported at all
        hold_xtab = {}
        for r in rows:
            if r["safe"]:
                continue
            mm = (r["max_violation_m"] or 0.0) * 1e3
            key = "%s / max_violation %s" % (r["ag3s_status"], "> 0" if mm > 0 else "== 0")
            hold_xtab.setdefault(key, {"n": 0, "seqs": []})
            hold_xtab[key]["n"] += 1
            hold_xtab[key]["seqs"].append(r["seq"])

        hold_flags = {r["seq"]: (not r["safe"]) for r in rows}
        safe_flags = {r["seq"]: r["safe"] for r in rows}

        per_run[name] = {
            "record_dir": run_dir,
            "safe_shadow": (manifest.get("policy") or {}).get("safe_shadow", False),
            "argv_has_safe_shadow": "--safe-shadow" in ((manifest.get("extra") or {}).get("argv") or []),
            "n_chunks": len(rows),
            "n_safe": len(safe_seqs),
            "n_hold": len(hold_seqs),
            "completeness_pass": completeness.get("pass"),
            "client_ipc_stats": completeness.get("client_ipc_stats"),
            "field_state_counts": completeness.get("field_state_counts"),
            "safe_seqs": safe_seqs,
            "hold_seqs": hold_seqs,
            "safe_seq_ranges": run_ranges(safe_flags),
            "hold_seq_ranges": run_ranges(hold_flags),
            "safe_seq_min": min(safe_seqs) if safe_seqs else None,
            "safe_seq_max": max(safe_seqs) if safe_seqs else None,
            "hold_seq_min": min(hold_seqs) if hold_seqs else None,
            "first_unbroken_hold_seq": next(
                (r["seq"] for r in rows
                 if not r["safe"] and all(not q["safe"] for q in rows if q["seq"] >= r["seq"])),
                None),
            "hold_classes": hold_classes,
            "hold_class_by_violation": hold_xtab,
            "safe_classes": safe_classes,
            "ag3s_status_counts": ag3s_status_counts,
            "trajopt_status_counts": trajopt_status_counts,
            "ag3s_reason_code_counts": reason_counts,
            "n_tiers_counts": tier_counts,
            "max_violation_all_chunks": mm_stats(all_mm),
            "max_violation_hold_chunks": mm_stats(hold_mm),
            "max_violation_hold_chunks_nonzero": mm_stats(hold_mm_nonzero),
            "max_violation_zero_hold_chunks": sum(1 for v in hold_mm if v == 0.0),
            "timing_ms": stages,
            "camera_skew_ms": skew_stats,
            "control_records": len(control),
            "control_executed_true": sum(1 for c in control if c.get("executed")),
            "control_executed_false": sum(1 for c in control if not c.get("executed")),
        }

    # ---- violation trend across the hold region of the execute run -------------------------
    ex = chunk_rows["execute"]
    hold_onset = per_run["execute"]["first_unbroken_hold_seq"]
    trend = {}
    if hold_onset is not None:
        before = [r["max_violation_m"] * 1e3 for r in ex
                  if r["seq"] < hold_onset and r["max_violation_m"] is not None]
        after = [r["max_violation_m"] * 1e3 for r in ex
                 if r["seq"] >= hold_onset and r["max_violation_m"] is not None]
        seq_after = [r["seq"] for r in ex if r["seq"] >= hold_onset and r["max_violation_m"] is not None]
        # least squares slope of mm vs seq over the unbroken hold region
        slope = None
        if len(after) >= 2:
            n = len(after)
            mx = sum(seq_after) / n
            my = sum(after) / n
            den = sum((x - mx) ** 2 for x in seq_after)
            if den:
                slope = sum((x - mx) * (y - my) for x, y in zip(seq_after, after)) / den
        first10 = after[:10]
        last10 = after[-10:]
        trend = {
            "hold_onset_seq": hold_onset,
            "before_onset": {"n": len(before),
                             "p50_mm": median(before) if before else None,
                             "max_mm": max(before) if before else None},
            "from_onset": {"n": len(after),
                           "p50_mm": median(after) if after else None,
                           "min_mm": min(after) if after else None,
                           "max_mm": max(after) if after else None},
            "slope_mm_per_chunk_from_onset": slope,
            "first10_from_onset_p50_mm": median(first10) if first10 else None,
            "last10_p50_mm": median(last10) if last10 else None,
            "argmax_seq": max((r for r in ex if r["max_violation_m"] is not None),
                              key=lambda r: r["max_violation_m"])["seq"],
        }

    # ---- task-phase overlay (from T5d.verify.json, shadow rank1 record) ---------------------
    t5d = json.load(open(os.path.join(ROOT, "AG3S/ag3s/docs/handoff/T5d.verify.json")))
    phase_rule = t5d["numbers"]["task_phase_rule"]
    overlay = {
        "source": "AG3S/ag3s/docs/handoff/T5d.verify.json numbers.task_phase_rule",
        "measured_on": "offline replay of the reference episode (outputs/live_test/20260925_ep1807/run_0000)",
        "first_closed_seq": phase_rule["first_closed_seq"],
        "first_reopen_seq": phase_rule["first_reopen_seq"],
        "execute_hold_onset_seq": hold_onset,
        "execute_hold_onset_vs_first_reopen": (hold_onset - phase_rule["first_reopen_seq"])
        if hold_onset is not None else None,
        "execute_safe_seq_max": per_run["execute"]["safe_seq_max"],
        "execute_n_safe_in_seq_1_37": sum(1 for r in ex if r["safe"] and 1 <= r["seq"] <= 37),
        "execute_n_chunks_seq_1_37": sum(1 for r in ex if 1 <= r["seq"] <= 37),
        "execute_n_hold_in_seq_1_37": sum(1 for r in ex if not r["safe"] and 1 <= r["seq"] <= 37),
    }

    json.dump(chunk_rows, open(os.path.join(OUT, "chunks_T6b.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump({"per_run": per_run, "violation_trend_execute": trend,
               "task_phase_overlay": overlay},
              open(os.path.join(OUT, "aggregate_T6b.json"), "w"), ensure_ascii=False, indent=1)

    for name, _ in RUNS:
        p = per_run[name]
        print("== %s  safe %d / hold %d  (completeness pass=%s)"
              % (name, p["n_safe"], p["n_hold"], p["completeness_pass"]))
        print("   hold_classes:", p["hold_classes"])
        print("   hold ranges:", p["hold_seq_ranges"])
        print("   safe range: %s..%s" % (p["safe_seq_min"], p["safe_seq_max"]))
        print("   max_violation hold:", p["max_violation_hold_chunks"])
        print("   max_violation hold nonzero:", p["max_violation_hold_chunks_nonzero"])
        print("   reason codes:", p["ag3s_reason_code_counts"])
        print("   tiers:", p["n_tiers_counts"])
        print("   total p50/p95/max ms:", p["timing_ms"]["total"]["p50_ms"],
              p["timing_ms"]["total"]["p95_ms"], p["timing_ms"]["total"]["max_ms"])
        print("   camera skew ms:", p["camera_skew_ms"])
    print("trend:", json.dumps(trend, indent=1))
    print("overlay:", json.dumps(overlay, indent=1))


if __name__ == "__main__":
    main()
