"""Aggregate d2_<run>.json -> d2_summary.json (counts per run; per-chunk compact table)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, collections
import numpy as np
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d2'))
RUNS = ["E3a_ep1807_r1", "E3a_ep1807_r3", "E3b_ep1807_r1", "E3b_ep1807_r2"]
VAR = ("replay", "no_collision", "no_limits", "no_continuity", "no_smooth", "replay_3iter")
out = {"source": "d2_<run>.json (d2_rows_ablation.py)", "runs": {}}
for r in RUNS:
    J = json.loads((D / f"d2_{r}.json").read_text())
    C = J["chunks"]
    cnt = lambda f: dict(collections.Counter(f(c) for c in C))
    def lk(c, tn):
        w = c[tn]["worst"] or {}
        l = w.get("link") or ""
        return "held_point" if l.startswith("attached") else l
    table = []
    for c in C:
        rec = c["recorded"]; cand = rec.get("candidates") or []
        table.append({"t": c["t"], "executed": c["executed"], "n_control": c["n_control"],
                      "recorded_dev_deg": round(c["recorded_dev_deg"]["max"], 2),
                      "argmax_joint": max(c["recorded_dev_deg"]["per_joint"], key=c["recorded_dev_deg"]["per_joint"].get),
                      "returned": rec["returned"], "sqp_iterations": rec["sqp_iterations"], "time_budget_hit": rec["time_budget_hit"],
                      "step_size_rad": [x.get("step_size_rad") for x in cand], "trust_radius_rad": [x.get("trust_radius_rad") for x in cand],
                      "linearize_ms": (rec.get("timing_ms") or {}).get("linearize"),
                      "initial_violation_mm": None if rec["initial_violation_m"] is None else round(1000 * rec["initial_violation_m"], 2),
                      "final_clearance_mm": None if rec["clearance_m"] is None else round(1000 * rec["clearance_m"], 2),
                      "initial_limit_overshoot_position_rad": (rec.get("initial_limit_overshoot") or {}).get("position"),
                      "ref_worst": {k: c["reference"]["worst"].get(k) for k in ("link", "tier", "attached", "gt_nearest", "gt_nearest_mm", "gt_nearest_excl_apple", "gt_nearest_excl_apple_mm")} | {"clearance_mm": round(c["reference"]["worst_clearance_mm"], 2)},
                      "final_worst": {k: c["refined"]["worst"].get(k) for k in ("link", "tier", "attached", "gt_nearest", "gt_nearest_mm")} | {"clearance_mm": round(c["refined"]["worst_clearance_mm"], 2)},
                      "ref_n_neg_rows": c["reference"]["n_neg_rows"], "ref_neg_by_link": c["reference"]["neg_by_link"], "ref_neg_by_tier": c["reference"]["neg_by_tier"],
                      "ref_beyond_TO_limits": {k: round(v["over_rad"], 4) for k, v in c["reference_beyond_TO_position_limits"].items()},
                      "ref_beyond_mujoco_range": c["reference_beyond_mujoco_range_rad"],
                      "ablation_dev_deg": {v: (round(c["ablation"][v]["dev_max_deg"], 2) if "dev_max_deg" in c["ablation"].get(v, {}) else None) for v in VAR},
                      "ablation_returned": {v: c["ablation"].get(v, {}).get("returned") for v in VAR},
                      "replay_vs_recorded_max_deg": c["ablation"]["replay"].get("vs_recorded_refined_max_deg"),
                      "fidelity": c["fidelity"]})
    devs = np.array([x["recorded_dev_deg"] for x in table])
    ab = {v: [x["ablation_dev_deg"][v] for x in table if x["ablation_dev_deg"][v] is not None] for v in VAR}
    out["runs"][r] = {
        "attached_reconstruction": J["attached_reconstruction"], "window_t": J["window"], "n_chunks": len(C),
        "fidelity_max_abs_diff_mm": max(x["fidelity"]["refined_worst_abs_diff_mm"] for x in table if x["fidelity"]["refined_worst_abs_diff_mm"] is not None),
        "fidelity_same_row": sum(x["fidelity"]["refined_same_row"] for x in table),
        "n_recorded_dev_ge_8deg": int((devs >= 8.0).sum()), "n_recorded_dev_eq_0": int((devs == 0).sum()),
        "recorded_dev_deg_median": float(np.median(devs)),
        "returned_counts": cnt(lambda c: c["recorded"]["returned"]),
        "sqp_iterations_counts": cnt(lambda c: str(c["recorded"]["sqp_iterations"])),
        "ref_worst_link_counts": cnt(lambda c: lk(c, "reference")),
        "ref_worst_tier_counts": cnt(lambda c: c["reference"]["worst"]["tier"]),
        "ref_worst_gt_nearest_counts": cnt(lambda c: c["reference"]["worst"]["gt_nearest"]),
        "ref_worst_attached_counts": cnt(lambda c: str(c["reference"]["worst"]["attached"])),
        "final_worst_link_counts": cnt(lambda c: lk(c, "refined")),
        "final_worst_tier_counts": cnt(lambda c: c["refined"]["worst"]["tier"]),
        "final_worst_gt_nearest_counts": cnt(lambda c: c["refined"]["worst"]["gt_nearest"]),
        "n_chunks_ref_beyond_TO_position_limits": sum(1 for x in table if x["ref_beyond_TO_limits"]),
        "TO_limit_joints": dict(collections.Counter(j for x in table for j in x["ref_beyond_TO_limits"])),
        "max_ref_beyond_TO_limit_rad": max([v for x in table for v in x["ref_beyond_TO_limits"].values()] or [0.0]),
        "n_chunks_ref_beyond_mujoco_range": sum(1 for x in table if x["ref_beyond_mujoco_range"]),
        "ablation_dev_deg_median": {v: float(np.median(a)) for v, a in ab.items()},
        "ablation_n_ge_8deg": {v: int(sum(1 for y in a if y >= 8.0)) for v, a in ab.items()},
        "ablation_n_eq_0": {v: int(sum(1 for y in a if y == 0.0)) for v, a in ab.items()},
        "replay_vs_recorded_max_deg_median": float(np.median([x["replay_vs_recorded_max_deg"] for x in table])),
        "chunks": table}
    R = out["runs"][r]
    print(r, R["n_chunks"], "dev>=8", R["n_recorded_dev_ge_8deg"], "ret", R["returned_counts"], "it", R["sqp_iterations_counts"], "\n  link", R["ref_worst_link_counts"], "tier", R["ref_worst_tier_counts"], "gt", R["ref_worst_gt_nearest_counts"],
          "\n  final link", R["final_worst_link_counts"], R["final_worst_tier_counts"], R["final_worst_gt_nearest_counts"], "\n  lim", R["n_chunks_ref_beyond_TO_position_limits"], R["TO_limit_joints"], R["max_ref_beyond_TO_limit_rad"], "mj", R["n_chunks_ref_beyond_mujoco_range"],
          "\n  abl med", {k: round(v, 2) for k, v in R["ablation_dev_deg_median"].items()}, "n>=8", R["ablation_n_ge_8deg"], "replay_vs_rec", round(R["replay_vs_recorded_max_deg_median"], 3))
(D / "d2_summary.json").write_text(json.dumps(out, indent=1, default=float))
