"""T21 verifier — manipulated geometry per chunk from the working-tree TargetConfirm (T20),
replayed over the recorded server clouds (same inputs as outputs/verify/T20/replay_t14.py).

Configs: `default` = AG3SConfig() clustering defaults (what serve_safe builds today);
`rec` = target_switch_min_score 0.1, target_lost_frames 20 (T20.verify recommendation).
Output: manip_<run>.npz (points per chunk/config) + manip_<run>.json (records).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
from pathlib import Path
import numpy as np
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.types import PointCloud, AttentionPointCloud
from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm

ROOT = Path(_ag3s_workspace_path(''))
OUT = ROOT / "outputs/verify/t21"
RUNS = {"t14": ("20260928_t14", "execute_ep1807"), "t17": ("20260928_t17", "shadow_ep1807")}
cfg = AG3SConfig()
CONFIGS = {"default": (cfg.clustering.target_switch_min_score, cfg.clustering.target_lost_frames),
           "rec": (0.1, 20)}

for run, (d, fr) in RUNS.items():
    rec_dir = ROOT / f"outputs/live_test/{d}/server_constraints/run_0000"
    files = sorted(rec_dir.glob("chunk_*.npz"))
    apple = None
    for line in open(ROOT / f"outputs/live_test/{d}/{fr}/frames.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "planning":
            apple = np.asarray(r["object_poses"]["apple"]["pos"], float); break
    arrays, records = {}, []
    for name, (ms, lost) in CONFIGS.items():
        confirm = TargetConfirm(cfg.clustering, switch_min_score=ms, lost_frames=lost)
        for f in files:
            z = np.load(f)
            k = int(z["chunk_index"])
            apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)),
                                      z["attention_values"], raw_attention=z["attention_raw"])
            g = ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                              seed_threshold=cfg.attention.seed_threshold,
                              exclude_mask=z["support_mask"], min_radius=cfg.geometry.min_radius,
                              timestamp=float(z["t_step"]), confirm=confirm)
            m = confirm.manipulated
            rec = None if m is None else m.record()
            pts = np.zeros((0, 3)) if (m is None or not m.usable) else m.points
            arrays[f"{name}_{k}"] = pts
            records.append(dict(config=name, min_score=ms, lost_frames=lost, chunk=k,
                                usable=bool(m is not None and m.usable), manipulated=rec,
                                to_apple_mm=(None if rec is None else
                                             float(np.linalg.norm(np.asarray(rec["centroid"]) - apple) * 1000)),
                                n_points=int(len(pts)),
                                live_target_n=(int(z["target_points"].shape[0]) if "target_points" in z.files else 0)))
            print(run, name, k, rec and rec["state"], rec and rec["id"], len(pts),
                  records[-1]["to_apple_mm"] and round(records[-1]["to_apple_mm"], 1), flush=True)
    np.savez_compressed(OUT / f"manip_{run}.npz", **arrays)
    json.dump(records, open(OUT / f"manip_{run}.json", "w"), indent=1)
