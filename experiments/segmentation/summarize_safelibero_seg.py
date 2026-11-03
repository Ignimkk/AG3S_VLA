#!/usr/bin/env python3
"""Aggregate SafeLIBERO segmentation scores (external baselines: scores.jsonl; ATS: frames.jsonl) into one table.

frame -> episode -> task -> equal mean over the 4 tasks. 3D metrics are the class-aware soft-voxel metrics of
safelibero_scoring. Strata: target-obstacle nearest distance of the frame (GT lifted pixels), fixed at 30 / 60 mm.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

BUCKETS = (('<30mm', 0.0, 0.030), ('30-60mm', 0.030, 0.060), ('>=60mm', 0.060, 9.0))


def rows_external(path):
    for line in path.open():
        r = json.loads(line)
        yield r, r['3d'], r['2d']


def rows_ats(path, which):
    for line in path.open():
        r = json.loads(line)
        yield r, r[which], None


TOTALS = {}
ONSET = {}      # stem -> first obs index where the official collision rule fires (from obstacle_motion_safelibero.py)
PRE_STEPS = 20


def frame_values(r, m3, m2):
    pv, tv = m3['pred_voxels'], m3['target_voxels']
    v = {'iou3d': m3['iou'], 'recall3d': m3['recall'], 'precision3d': m3['precision'],
         'identity': None if tv <= 0 else float(m3['precision'] is not None and m3['precision'] >= 0.5),
         'dominant_target': None if tv <= 0 else float(pv > 0 and m3['TP'] >= max((m3.get(c+'_fp') or 0.0) for c in
                              ('obstacle', 'other_object', 'destination', 'fixture', 'support', 'robot', 'background'))),
         'empty': float(pv == 0), 'neighbor_fp': m3['neighbor_fp'], 'obstacle_fp': m3['obstacle_fp'],
         'other_object_fp': m3['other_object_fp'], 'destination_fp': m3['destination_fp'],
         'obstacle_in_pred': float(m3['obstacle_fp'] >= 0.5),
         'obstacle_frac_pred': (m3['obstacle_fp']/pv) if pv else None}
    tot = TOTALS.get('%s:%d' % (r['stem'], r['t']))
    if tot:
        # erase / keep fractions: share of each GT class's voxel mass the arm's voxel set covers (= erases from the distance field)
        if tot['target'] > 0:
            v['target_erased'] = m3['TP']/tot['target']
            v['target_kept'] = 1.0-m3['TP']/tot['target']
        for c in ('obstacle', 'other_object', 'destination', 'fixture', 'support'):
            if tot[c] > 0:
                v[c+'_erased'] = (m3.get(c+'_fp') or 0.0)/tot[c]
    if m2:
        for c in ('head', 'left_wrist'):
            d = m2[c]
            v['iou2d_'+c] = d['iou']; v['recall2d_'+c] = d['recall']
            v['obstacle_px_'+c] = float(d['pred_on']['obstacle'])
            v['obstacle_frac2d_'+c] = d['pred_on']['obstacle']/d['pred_px'] if d['pred_px'] else None
            v['invisible_nonempty_'+c] = float(d['pred_px'] > 0) if d['gt_px'] == 0 else None
    return v


def aggregate(frames):
    """frames: list of (task, episode, bucket, values) -> {metric: {'mean', 'task'}}; Nones skipped."""
    out = {}
    metrics = sorted({k for _, _, _, v, *_t in frames for k in v})
    for m in metrics:
        per = defaultdict(lambda: defaultdict(list))
        for task, ep, _, v, *_t in frames:
            x = v.get(m)
            if x is not None:
                per[task][ep].append(float(x))
        task = {t: float(np.mean([np.mean(x) for x in eps.values()])) for t, eps in per.items()}
        out[m] = {'mean': float(np.mean(list(task.values()))) if task else None, 'task': task,
                  'n_frames': sum(len(x) for eps in per.values() for x in eps.values())}
    return out


def summarize(source):
    frames = []
    for r, m3, m2 in source:
        d = r['meta']['target_obstacle_min_dist_m']
        bucket = None if d is None else next(n for n, lo, hi in BUCKETS if lo <= d < hi)
        frames.append((r['task'], r['stem'], bucket, frame_values(r, m3, m2), r['t']))
    res = {'all': aggregate(frames), 'n_frames': len(frames)}
    if ONSET:
        # PRE_STEPS observations before the first official collision of the episodes that collided (the VLA's own trajectory)
        sub = [f for f in frames if ONSET.get(f[1]) is not None and ONSET[f[1]]-PRE_STEPS <= f[4] < ONSET[f[1]]]
        res['pre_collision'] = aggregate(sub) if sub else None
        if sub:
            res['pre_collision']['n_frames_total'] = len(sub)
    for name, _, _ in BUCKETS:
        sub = [f for f in frames if f[2] == name]
        res[name] = aggregate(sub) if sub else None
        if sub:
            res[name]['n_frames_total'] = len(sub)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baselines-root', type=Path, required=True)
    ap.add_argument('--ats', nargs='*', default=[], help='name=path of ATS run dirs (frames.jsonl)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gt-totals', type=Path, default=None)
    ap.add_argument('--motion', type=Path, default=None, help='obstacle_motion_safelibero.py output dir')
    a = ap.parse_args()
    if a.gt_totals:
        TOTALS.update(json.loads(a.gt_totals.read_text()))
    if a.motion:
        for pth in a.motion.glob('*_motion.json'):
            m = json.loads(pth.read_text()); ONSET[m['stem']] = m['first_collision_obs']
    result = {}
    for d in sorted(p for p in a.baselines_root.iterdir() if (p/'scores.jsonl').exists()):
        result[d.name] = summarize(rows_external(d/'scores.jsonl'))
    for spec in a.ats:
        name, path = spec.split('=', 1)
        for which in ('target_cluster', 'erased_extent'):
            result['ATS_%s__%s' % (name, which)] = summarize(rows_ats(Path(path)/'frames.jsonl', which))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(result, indent=1)+'\n')
    cols = ['iou3d', 'recall3d', 'precision3d', 'identity', 'dominant_target', 'empty', 'target_erased', 'obstacle_erased', 'other_object_erased', 'obstacle_in_pred']
    lines = ['| arm | n | ' + ' | '.join(cols) + ' |', '|---|---:|' + '---:|'*len(cols)]
    for name, r in sorted(result.items(), key=lambda kv: -(kv[1]['all']['iou3d']['mean'] or 0)):
        lines.append('| %s | %d | ' % (name, r['n_frames']) + ' | '.join('%.3f' % r['all'][c]['mean'] if r['all'][c]['mean'] is not None else '-' for c in cols) + ' |')
    (a.out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
