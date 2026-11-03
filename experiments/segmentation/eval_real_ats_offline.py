#!/usr/bin/env python3
"""Offline replay of the *real* Panda ATS (AG3S final configuration) on recorded LIBERO frames, scored with the same
image-instant GT and 3D soft-voxel metric as the external segmenters.

The AG3S object is built from the final-configuration server flags (`profile.version_server_flags`), so the target
cluster / ball-owned extent / split partition are the served code, not the earlier research adapter. The attention
grid is the served `noun_target:L<l>H<h>` slice, read from the stored noun rows of the same policy requests. No
simulator GT enters the AG3S call; GT is used by the scorer only. Grasp / latch signals are the no-grasp values
(`perception_replay.apply_frame_state`), so latched identity hold is not exercised (recorded in the manifest).
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmark/runners'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Baseline.paper_baselines import lift as L
from Baseline.paper_baselines.eval import metrics as M
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes, seg_labels
import perception_replay as PR

CAMERAS = PR.CAMERAS
KEEP = ('iou', 'precision', 'recall', 'TP', 'FP', 'FN', 'target_voxels', 'pred_voxels', 'neighbor_fp',
        'fixture_fp', 'support_fp', 'robot_fp')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def eval_grid(fr):
    back = {c: L.backproject_pixels(fr['depth_m'][c], fr['K'][c], fr['T'][c]) for c in CAMERAS}
    return M.build_eval_grid([{'depth_m': fr['depth_m'][c], 'K': fr['K'][c], 'T_base_cam': fr['T'][c],
                               'labels': fr['labels'][c], 'backprojected': back[c]} for c in CAMERAS])


def score_points(grid, points):
    keys = (np.zeros((0, 3), np.int64) if points is None or len(points) == 0
            else L.voxel_keys(np.asarray(points, np.float64)))
    metric = M.voxel_metrics(grid, keys)           # an abstention is scored as the empty prediction, as for the baselines
    return {k: metric[k] for k in KEEP}


def build_ag3s(version, head):
    from AG3S.adapters.panda import profile as pf
    flags = pf.version_server_flags(version, attention_source=f'noun_target:L{head[0]}H{head[1]}')
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(version) + flags)
    kw = pf.v11ab_build_kwargs(args)
    ag3s, info = pf.build_panda_ag3s(kw=kw)
    return ag3s, info, flags


def run(a):
    if os.environ.get('AG3S_PROJECT_ROOT') != str(ROOT):
        raise RuntimeError('source scripts/setup/env.sh of this checkout first')
    selected = defaultdict(list)
    for line in a.frames.open():
        r = json.loads(line)
        selected[r['episode_id']].append(int(r['obs_index']))
    metadata = sorted((m for m in load_episodes(a.source) if m['stem'] in selected),
                      key=lambda m: (m['task_id'], m['episode_idx']))
    if len(metadata) != len(selected):
        raise ValueError('frame list and episode metadata differ')
    if a.limit_episodes:
        metadata = metadata[:a.limit_episodes]
    head = tuple(int(x) for x in a.head.split(','))
    ag3s, info, flags = build_ag3s(a.version, head)
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {'kind': 'offline real-ATS replay (perception only; no policy action, no collision, no controller)',
                'version': a.version, 'server_flags': flags, 'ag3s_info': PR._jsonable(info),
                'attention': f'noun_target:L{head[0]}H{head[1]} from stored noun rows',
                'source': str(a.source.resolve()), 'source_episodes_sha256': sha(a.source/'episodes.jsonl'),
                'noun_run': str(a.noun_run.resolve()), 'scene': str(a.scene.resolve()), 'phase': str(a.phase.resolve()),
                'frames': str(a.frames.resolve()), 'frames_sha256': sha(a.frames), 'script_sha256': sha(__file__),
                'protocol': str(a.protocol.resolve()), 'protocol_sha256': sha(a.protocol),
                'grasp_state': 'no-grasp values every call (perception_replay.apply_frame_state); latch not exercised',
                'state_carry': 'AG3S.reset() at each episode start; frames of an episode in obs_index order',
                'episodes': [m['stem'] for m in metadata], 'limit_frames': a.limit_frames, 'test': False}
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    phase_name = __import__('AG3S.adapters.panda.profile', fromlist=['x']).manifest_entry()['client_phase']
    opts = PR.FrameOptions(phase=phase_name, save_clouds=True, stage_probe=False, grounding_only=False)
    (a.out/'clouds').mkdir()
    start = time.monotonic()
    with (a.out/'frames.jsonl').open('x') as out:
        for n, meta in enumerate(metadata, 1):
            ep = Episode(meta, cameras=CAMERAS, scene_dir=a.scene)
            with np.load(meta['geom_path'], allow_pickle=False) as z:
                geom = {k: z[k] for k in z.files}
            with np.load(a.noun_run/'noun_rows'/f'{ep.stem}_noun.npz', allow_pickle=False) as z:
                noun = {k: z[k] for k in ('noun_image', 'obs_index', 'layers', 'noun_ok')}
            with np.load(a.phase/f'phase_{ep.stem}.npz', allow_pickle=False) as z:
                phase = {k: z[k] for k in ('call_frames', 'segP/head', 'segP/left_wrist')}
            layer_idx = np.flatnonzero(noun['layers'] == head[0])
            if len(layer_idx) != 1:
                raise ValueError('noun layer missing')
            ag3s.reset()
            times = sorted(selected[ep.stem])
            if a.limit_frames:
                times = times[:a.limit_frames]
            kept, kept_ext = [], []
            for t in times:
                pi = np.flatnonzero(phase['call_frames'] == t)
                ni = np.flatnonzero(noun['obs_index'] == t)
                if len(pi) != 1 or len(ni) != 1:
                    raise ValueError(f'frame missing: {ep.stem}/{t}')
                if not noun['noun_ok'][ni[0]]:
                    raise ValueError(f'noun token missing: {ep.stem}/{t}')
                v = noun['noun_image'][ni[0], layer_idx[0], head[1]]
                grids = np.stack([v[j*256:(j+1)*256].reshape(16, 16) for j in range(2)])
                fr = ep.frame(t)
                for c in CAMERAS:
                    fr['labels'][c] = seg_labels(phase[f'segP/{c}'][pi[0]], ep.lut)
                observations, fingers, rs9, _ = PR.build_observations(geom, fr['geom_row'], grids, 'unmirror',
                                                                      ag3s.robot_model)
                t0 = time.perf_counter()
                PR.apply_frame_state(ag3s, fingers, rs9)
                cs, debug = ag3s.process_multi_debug(observations, phase=opts.phase, active_manipulators=['left'],
                                                     static_geometry=None)
                wall = time.perf_counter()-t0
                grounding = debug['grounding']
                target = grounding.target
                grid = eval_grid(fr)
                pts = None if target is None else np.asarray(target.points, np.float64)
                exclusion = cs.metrics.get('exclusion')
                trace = getattr(cs.esdf, 'exclusion_trace', None) or {}
                ext_pts = None
                if len(trace.get('removed_ijk', ())):       # observed surface voxels the target-free field actually erased
                    ext_pts = (np.asarray(trace['origin'], np.float64)
                               + np.asarray(trace['removed_ijk'], np.float64)*float(trace['voxel_size']))
                row = {'episode_id': ep.stem, 'task_id': int(meta['task_id']), 'obs_index': t,
                       'status': grounding.status.value, 'has_target': target is not None,
                       'n_target_points': 0 if target is None else int(len(target.points)),
                       'centroid': None if target is None else PR._jsonable(target.centroid),
                       'seed_camera': None if target is None or not target.seed_camera else target.seed_camera.value,
                       'wall_s': wall, 'metrics': score_points(grid, pts), 'extent_metrics': score_points(grid, ext_pts),
                       'n_removed_voxels': 0 if ext_pts is None else int(len(ext_pts)), 'exclusion': PR._jsonable(exclusion),
                       'gt_target_voxels': int(grid.target.sum()) if hasattr(grid, 'target') else None}
                out.write(json.dumps(row, allow_nan=False, default=str)+'\n')
                out.flush()
                kept.append(np.zeros((0, 3), np.float32) if pts is None else pts.astype(np.float32))
                kept_ext.append(np.zeros((0, 3), np.float32) if ext_pts is None else ext_pts.astype(np.float32))
            np.savez_compressed(a.out/'clouds'/f'{ep.stem}.npz', obs_index=np.asarray(times),
                                offsets=np.cumsum([0]+[len(k) for k in kept]), target_points=np.concatenate(kept),
                                ext_offsets=np.cumsum([0]+[len(k) for k in kept_ext]), ext_points=np.concatenate(kept_ext))
            print(f'{n}/{len(metadata)} {ep.stem} elapsed={time.monotonic()-start:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'scene', 'phase', 'frames', 'protocol', 'noun_run', 'out'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--version', default='AG3S-final-panda')
    p.add_argument('--head', default='8,6')
    p.add_argument('--limit-episodes', type=int, default=0)
    p.add_argument('--limit-frames', type=int, default=0)
    run(p.parse_args())


if __name__ == '__main__':
    main()
