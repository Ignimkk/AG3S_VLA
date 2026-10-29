"""Condense score.json + cubes.npz into summary.json (numbers only).

Adds: 14D verdict-rule outcome per (phase, scope, view, cell) exactly as attention_report.py:528-536
(beats head-average and uniform at every beta, peak margin >= 0.15 over max(baselines), lift > 2),
per-episode top-1 and top-10 overlap, frame counts behind the rates.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import itertools
import json
import pathlib

import numpy as np

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/attn16d'))
S = json.loads((D / "score.json").read_text())
C = np.load(D / "cubes.npz")
EPS = S["config"]["episodes"]
DEN, AGG = S["config"]["denoise"], S["config"]["aggs"]
BETAS = ["0", "0.5", "1"]
CELLS = {"L8h2_d0_last": (0, "last", 8, 2), "L7h0_d0_last": (0, "last", 7, 0)}


def cube(phase, scope, view, m):
    k = f"{phase}|{scope}|{view}|{m}"
    return C[k] if k in C.files else None


def rule(cell_metrics, ha, un):
    beats = all(cell_metrics[f"hit_b{b}"] >= max(ha[f"hit_b{b}"], un[f"hit_b{b}"]) - 1e-9 for b in BETAS)
    margin = cell_metrics["peak"] - max(ha["peak"], un["peak"])
    return {"beats_both_baselines_all_beta": bool(beats),
            "failing_betas": [b for b in BETAS if cell_metrics[f"hit_b{b}"] < max(ha[f"hit_b{b}"], un[f"hit_b{b}"]) - 1e-9],
            "peak_margin_vs_max_baseline": float(margin), "lift": float(cell_metrics["lift"]),
            "rule_14d_satisfied": bool(beats and margin >= 0.15 and cell_metrics["lift"] > 2.0)}


def cell_label(r):
    return f"d{r['denoise']}_{r['agg']}_L{r['layer']}h{r['head']}"


out = {}
for phase in ("reach", "carry"):
    P = S["phases"][phase]
    out[phase] = {}
    for view in ("cam_high", "cam_left_wrist", "cam_right_wrist", "any"):
        V = {}
        for scope in ["pooled"] + EPS:
            e = P["scopes"][scope][view]
            if e is None:
                V[scope] = None
                continue
            n = e["n_frames"]
            ha, un = e["baseline_head_average_d0_last"], e["baseline_uniform"]
            ent = {"n_frames": n, "top1": cell_label(e["top15"][0]),
                   "top1_metrics": {m: e["top15"][0][m] for m in ("peak", "hit_b0", "hit_b0.5", "hit_b1", "lift")},
                   "top10": [cell_label(r) for r in e["top15"][:10]],
                   "n_cells_at_max_peak": e["n_cells_at_max_peak"],
                   "baseline_head_average_d0_last": ha, "baseline_uniform": un}
            for cname, cm in e["key_cells"].items():
                ent[cname] = {**cm, "peak_frames": int(round(cm["peak"] * n)),
                              "gap_to_top1_peak": e["top15"][0]["peak"] - cm["peak"],
                              **rule(cm, ha, un)}
            ent["top1_rule"] = rule(e["top15"][0], ha, un)
            V[scope] = ent
        # stability across episodes
        per = {s: V[s] for s in EPS if V.get(s)}
        tops = {s: v["top1"] for s, v in per.items()}
        sets = {s: set(v["top10"]) for s, v in per.items()}
        pairs = list(itertools.combinations(sets, 2))
        V["stability"] = {
            "top1_per_episode": tops,
            "L8h2_rank_per_episode": {s: v["L8h2_d0_last"]["rank"] for s, v in per.items()},
            "L7h0_rank_per_episode": {s: v["L7h0_d0_last"]["rank"] for s, v in per.items()},
            "L8h2_in_top10_episodes": sum(1 for v in per.values() if v["L8h2_d0_last"]["rank"] <= 10),
            "top10_jaccard_pairs": {f"{a}~{b}": len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs},
            "cells_in_every_episode_top10": sorted(set.intersection(*sets.values())) if sets else [],
        }
        # pooled rank of every per-episode top1 cell
        pk = cube(phase, "pooled", view, "peak")
        if pk is not None:
            V["pooled_peak_d0_last_layer_head"] = pk[DEN.index(0), AGG.index("last")].tolist()
        out[phase][view] = V

out["provenance_counts"] = {n: S["episodes"][n] for n in S["episodes"]}
out["kept"] = {p: S["phases"][p]["kept"] for p in ("reach", "carry")}
out["crosscheck_cov640_pooled"] = {p: S["phases"][p]["crosscheck_cov640_pooled"] for p in ("reach", "carry")}
(D / "summary.json").write_text(json.dumps(out, indent=1))
for phase in ("reach", "carry"):
    for view in ("cam_high", "cam_left_wrist", "any", "cam_right_wrist"):
        V = out[phase][view]
        p = V["pooled"]
        if not p:
            continue
        a = p["L8h2_d0_last"]
        print(f"{phase:5s} {view:15s} n={p['n_frames']:2d} top1={p['top1']:22s} peak={p['top1_metrics']['peak']:.3f} | "
              f"L8h2 rank={a['rank']:4d} peak={a['peak']:.3f} ({a['peak_frames']}/{p['n_frames']}) rule14D={a['rule_14d_satisfied']} fail={a['failing_betas']} margin={a['peak_margin_vs_max_baseline']:+.3f} | "
              f"L7h0 rank={p['L7h0_d0_last']['rank']} | HA={p['baseline_head_average_d0_last']['peak']:.3f}")
        print("      stability", json.dumps(V["stability"]))
