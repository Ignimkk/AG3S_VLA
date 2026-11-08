#!/usr/bin/env python3
"""SL-GTTO16 (A) failure analysis (2026-11-08): what is REALLY next to the obstacle-object row that refused the pass-through?

For every call of an observe / closed-loop run whose pass-through was refused with reason 'obstacle' (A), the worst obstacle
row's point (`to.passthrough_clear.obstacle_rows.worst.point_m`, base frame) is taken; the occupied fine-tier voxels within
`--radius` of it (`d < 0.5 voxel`, the same rule as gtto_held_voxel_origin) and, if none, the coarse tier's, are moved to world
and classified against the MuJoCo collision geoms of that frame (`panda_fixed_input_ablation.classify_points`: target ·
support · destination · robot (hand / arm) · other:<body> · ghost). The nearest voxel's class is the row's owner.
Needs `--record-constraints full` and geom snapshots (`panda_geom_snapshot.py`).
usage: gtto_obstacle_row_origin.py --run-dir RUN --snap-dir DIR --out DIR [--radius 0.04]"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "segmentation"))
from AG3S.trajopt.experiments import panda_fixed_input_ablation as fa  # noqa: E402
from gtto_held_voxel_origin import classify, occupied_near  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=pathlib.Path, action="append")
    ap.add_argument("--snap-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--radius", type=float, default=0.04)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in a.run_dir:
        eps = {(e["task_id"], e["episode_idx"]): e for e in
               (json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip())}
        for p in sorted((run / "episodes").glob("*_calls.jsonl")):
            stem = p.name[:-len("_calls.jsonl")]
            snap_path = a.snap_dir / (stem + "_geomsnap.npz")
            if not snap_path.is_file():
                continue
            calls = fa.load_episode_calls(run / "episodes", stem)
            fa.attach_chunks(calls, run / "server_constraints" / "run_0000")
            t, e = stem.split("_t")[1].split("_e")
            ep = eps.get((int(t), int(e)), {})
            for c in calls:
                pc = ((c.row.get("to") or {}).get("passthrough_clear") or {})
                if pc.get("reason") != "obstacle" or c.chunk_path is None:
                    continue
                o = pc.get("obstacle_rows") or {}
                w = o.get("worst") or {}
                if w.get("point_m") is None:
                    continue
                snap = fa.load_geom_snapshot(snap_path, c.obs_index)
                if snap is None:
                    continue
                tb = [str(x) for x in snap.get("target_bodies", [])] if snap.get("target_bodies") is not None else []
                z = np.load(c.chunk_path, allow_pickle=False)
                pt = np.asarray(w["point_m"], np.float64)[None]
                owner, tier, n = None, None, 0
                for tier_name, key in (("fine", "esdf"), ("coarse", "esdf_coarse")):
                    if f"{key}_distance" not in z.files:
                        continue
                    vox = occupied_near(z[f"{key}_distance"], np.asarray(z[f"{key}_origin"]), float(z[f"{key}_voxel_size"]),
                                        pt, a.radius)
                    if len(vox):
                        T = np.asarray(c.T_world_base, np.float64)
                        world = (T[:3, :3] @ vox.T).T + T[:3, 3]
                        labels = classify(world, snap, tb)
                        dmin = np.linalg.norm(vox - pt, axis=1)
                        owner = labels[int(np.argmin(dmin))]
                        tier, n = tier_name, len(vox)
                        cnt = collections.Counter(labels)
                        break
                rows.append({"run": run.name[16:60], "stem": stem, "success": bool(ep.get("official_success")),
                             "obs": c.obs_index, "latch": (c.row.get("grasp") or {}).get("state"),
                             "link": w.get("link"), "violation_mm": round(1000 * float(o.get("violation_m") or 0), 1),
                             "owner": owner, "tier": tier, "n_voxels": n,
                             "classes": dict(cnt.most_common(4)) if owner else {}})
    (a.out / "obstacle_row_origin.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    L = ["# A 의 '장애물 행' 거절 — 그 행 옆의 실제 물체 (MuJoCo geom 기준)", "",
         "| episode | VLA 성공 | obs | latch | link | 모델상 위반 mm | 가장 가까운 voxel 의 실제 주인 | 계층 | 주변 voxel class |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['stem']} | {'✓' if r['success'] else '✗'} | {r['obs']} | {r['latch']} | {r['link']} | {r['violation_mm']} | "
                 f"{r['owner']} | {r['tier']} | {r['classes']} |")
    own = collections.Counter(str(r["owner"]).split(":")[0] for r in rows)
    L += ["", "주인 집계: " + ", ".join(f"{k} {v}" for k, v in own.most_common())]
    (a.out / "obstacle_row_origin표.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
