#!/usr/bin/env python3
"""Split the real-ATS comparison by whether the GT target has moved (>= 20 mm from its first evaluated frame; fixed before
looking at the split). Replay uses no-grasp state, so the moved stratum is where latched identity hold would matter."""
import argparse, json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Baseline.paper_baselines import lift as L
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes, seg_labels
import combine_real_ats as C

THRESH_M = 0.020


def main():
    ap = argparse.ArgumentParser()
    for n in ('root', 'real', 'source', 'out'):
        ap.add_argument('--'+n, type=Path, required=True)
    a = ap.parse_args()
    meta = {m['stem']: m for m in load_episodes(a.source)}
    frames = [json.loads(l) for l in (a.root/'all_frames.jsonl').open()]
    tasks = {r['episode_id']: r['task_id'] for r in frames}
    first, moved, shift = {}, {}, {}
    for stem in sorted(tasks):
        ep = Episode(meta[stem], cameras=('head', 'left_wrist'), scene_dir=a.root/'scene')
        ph = np.load(a.root/'phase'/f'phase_{stem}.npz')
        for t in sorted(r['obs_index'] for r in frames if r['episode_id'] == stem):
            fr = ep.frame(t); pi = int(np.flatnonzero(ph['call_frames'] == t)[0]); pts = []
            for c in ('head', 'left_wrist'):
                P, u, v = L.backproject_pixels(fr['depth_m'][c], fr['K'][c], fr['T'][c])
                lab = np.isin(ph[f'segP/{c}'][pi], ep.target_geoms)
                pts.append(P[lab[v, u]])
            pts = np.concatenate(pts)
            cen = np.median(pts, 0) if len(pts) else None
            if stem not in first and cen is not None:
                first[stem] = cen
            d = float(np.linalg.norm(cen-first[stem])) if cen is not None else float('nan')
            shift[(stem, t)] = d; moved[(stem, t)] = bool(d >= THRESH_M)
        print(stem, flush=True)
    rows = C.collect(a.root, a.real)
    out = {'threshold_m': THRESH_M, 'frames': {'static': sum(not v for v in moved.values()), 'moved': sum(moved.values())},
           'methods': {}}
    for name, r in rows.items():
        out['methods'][name] = {}
        for stratum in ('static', 'moved'):
            keys = [k for k in sorted(r) if moved[k] == (stratum == 'moved')]
            ag = C.aggregate(r, tasks, keys)
            out['methods'][name][stratum] = {m: {'mean': ag[m]['mean'], 'n_tasks': len(ag[m]['task'])} for m in ag}
    (a.out).mkdir(parents=True, exist_ok=True)
    (a.out/'stratified.json').write_text(json.dumps(out, indent=1)+'\n')
    for s in ('static', 'moved'):
        print(s, out['frames'][s])
        for n, v in sorted(out['methods'].items(), key=lambda kv: -kv[1][s]['iou']['mean']):
            x = v[s]; print(f"  {n:34s} iou {x['iou']['mean']:.3f} rec {x['recall']['mean']:.3f} id {x['identity']['mean']:.3f} tasks {x['iou']['n_tasks']}")


if __name__ == '__main__':
    main()
