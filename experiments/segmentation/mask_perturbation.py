#!/usr/bin/env python3
"""Deterministic offline dev pilot: inject 2D GT mask errors and measure observed 3D voxel errors.

This probes the mask -> depth-lift/voxel link only. It does not run a controller or measure contact.
Frame choice and corruption sizes are fixed in advance; simulator GT is privileged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from Baseline.paper_baselines import lift as L
from Baseline.paper_baselines.eval import metrics as M
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes

CAMERAS = ('head', 'left_wrist')
RADII = (2, 4, 8)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x*x+y*y <= radius*radius


def scores(fr, masks):
    back = {c: L.backproject_pixels(fr['depth_m'][c], fr['K'][c], fr['T'][c]) for c in CAMERAS}
    grid = M.build_eval_grid([{'depth_m': fr['depth_m'][c], 'K': fr['K'][c], 'T_base_cam': fr['T'][c],
                              'labels': fr['labels'][c], 'backprojected': back[c]} for c in CAMERAS])
    output = {}
    for name, ms in masks.items():
        keys = []
        for c in CAMERAS:
            points, u, v = back[c]
            keys.append(L.voxel_keys(points[ms[c][v,u]]))
        metric = M.voxel_metrics(grid, np.concatenate(keys))
        output[name] = {key: metric[key] for key in ('iou', 'precision', 'recall', 'TP', 'FP', 'FN',
                                                      'target_voxels', 'pred_voxels', 'neighbor_fp',
                                                      'fixture_fp', 'support_fp', 'robot_fp')}
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit-episodes', type=int, default=0)
    ap.add_argument('--frames-per-episode', type=int, default=5)
    args = ap.parse_args()
    if args.frames_per_episode < 1: raise ValueError('frames-per-episode must be positive')
    metadata = [r for r in load_episodes(args.source) if r['has_geom'] and r['has_npz']]
    metadata.sort(key=lambda r: (int(r['task_id']), int(r['episode_idx'])))
    if args.limit_episodes: metadata = metadata[:args.limit_episodes]
    if not metadata: raise ValueError('no input episodes')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'kind': 'dev offline GT-mask perturbation; no policy or collision evaluation',
                'source': str(args.source.resolve()), 'scene': str(args.scene.resolve()),
                'source_episodes_sha256': sha(args.source / 'episodes.jsonl'),
                'script_sha256': sha(Path(__file__)), 'cameras': CAMERAS, 'radii_px': RADII,
                'frame_selection': '5 evenly spaced recorded evaluation indices per episode, inclusive endpoints',
                'frames_per_episode': args.frames_per_episode, 'episode_ids': [r['stem'] for r in metadata],
                'mask_errors': ['oracle'] + [f'{op}_{r}px' for op in ('dilate','erode') for r in RADII] +
                               ['drop_head', 'drop_left_wrist'],
                'gt_and_limitation': 'same recorded segmentation and depth as prior BL-4; image-state time alignment may differ',
                'voxel_m': M.VOXEL_M, 'seed': None}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    started=time.monotonic()
    with (args.out/'frames.jsonl').open('x') as out:
        for n, meta in enumerate(metadata,1):
            ep = Episode(meta, cameras=CAMERAS, scene_dir=args.scene)
            frames = ep.eval_frames()
            indices = np.unique(np.linspace(0,len(frames)-1,min(args.frames_per_episode,len(frames)), dtype=int))
            for idx in indices:
                t=frames[int(idx)]
                fr=ep.frame(t)
                gt={c:fr['labels'][c] == M.CLASSES.index('target') for c in CAMERAS}
                masks={'oracle': gt}
                for radius in RADII:
                    shape=disk(radius)
                    masks[f'dilate_{radius}px']={c:binary_dilation(gt[c],structure=shape) for c in CAMERAS}
                    masks[f'erode_{radius}px']={c:binary_erosion(gt[c],structure=shape) for c in CAMERAS}
                for cam in CAMERAS:
                    masks[f'drop_{cam}']={c:np.zeros_like(gt[c]) if c==cam else gt[c] for c in CAMERAS}
                row={'episode_id':ep.stem, 'task_id':int(meta['task_id']), 'episode_idx':int(meta['episode_idx']),
                     'obs_index':t, 'gt_pixels':{c:int(gt[c].sum()) for c in CAMERAS}, 'metrics':scores(fr,masks)}
                out.write(json.dumps(row,allow_nan=False)+'\n')
                out.flush()
            print(f'{n}/{len(metadata)} {ep.stem} elapsed={time.monotonic()-started:.1f}s',flush=True)
    print('complete', flush=True)

if __name__ == '__main__': main()
