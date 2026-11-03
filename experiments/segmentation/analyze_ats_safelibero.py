#!/usr/bin/env python3
"""Diagnostics of the SafeLIBERO ATS replays: which class the target cluster / erased extent lands on, identity by GT target motion
(>= 20 mm from its first call frame, a GT-only label for analysis), and how the gripper-physics signals line up with that motion."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safelibero_scoring as S
from Baseline.paper_baselines import lift as L

CLS = ('target', 'destination', 'obstacle', 'other_object', 'fixture', 'support', 'robot', 'background')


def target_centroids(gt_dirs):
    out = {}
    for stem, d in gt_dirs.items():
        ep = S.Episode(d, stem); first = None
        for t in ep.call_frames:
            fr = ep.frame(int(t)); pts = []
            for c in S.CAMERAS:
                P, u, v = L.backproject_pixels(fr['depth_m'][c], fr['K'][c], fr['T'][c])
                pts.append(P[(fr['labels'][c] == S.ci('target'))[v, u]])
            pts = np.concatenate(pts)
            cen = np.median(pts, 0) if len(pts) else None
            if first is None and cen is not None:
                first = cen
            out[(stem, int(t))] = None if cen is None or first is None else float(np.linalg.norm(cen-first))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    ap.add_argument('--ats', nargs='+', required=True, help='name=run_dir')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    shift = target_centroids(gt_dirs)
    res = {'moved_threshold_m': 0.02}
    for spec in a.ats:
        name, path = spec.split('=', 1)
        rows = [json.loads(l) for l in (Path(path)/'frames.jsonl').open()]
        r = {}
        for which in ('target_cluster', 'erased_extent'):
            dom = defaultdict(int); byz = defaultdict(lambda: defaultdict(list))
            for x in rows:
                m = x[which]; moved = (shift.get((x['stem'], x['t'])) or 0) >= 0.02
                key = 'empty' if m['pred_voxels'] == 0 else max(CLS, key=lambda c: m['TP'] if c == 'target' else m.get(c+'_fp', 0) or 0)
                dom[key] += 1
                byz['moved' if moved else 'static']['identity'].append(float(m['precision'] is not None and m['precision'] >= 0.5))
                byz['moved' if moved else 'static']['iou'].append(m['iou'] or 0.0)
            r[which] = {'dominant_class_frames': dict(dom),
                        **{k: {m: float(np.mean(v)) for m, v in d.items()} | {'n': len(d['iou'])} for k, d in byz.items()}}
        ph = defaultdict(int)
        for x in rows:
            moved = (shift.get((x['stem'], x['t'])) or 0) >= 0.02
            ph[('moved' if moved else 'static', 'closed_cmd' if x['physical']['closed_cmd'] else 'open_cmd',
                'holding' if x['physical']['holding'] else 'not_holding')] += 1
        r['physical_vs_gt_motion'] = {'/'.join(k): v for k, v in sorted(ph.items())}
        res[name] = r
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'ats_diagnostics.json').write_text(json.dumps(res, indent=1)+'\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
