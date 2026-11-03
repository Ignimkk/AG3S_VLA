#!/usr/bin/env python3
"""Paired development diagnostic of target cue x Panda robot self-filter.

Image-time simulator segmentations are used for evaluation and the explicitly
privileged seed / robot conditions only. The FK condition reads joint states,
depth and calibration, never simulator segmentation. No action is executed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from AG3S.adapters.panda import model as PM
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER, SurfaceView, select_target_surface
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes, seg_labels
from Baseline.paper_baselines.eval import metrics as M
from mask_perturbation import CAMERAS, scores


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(a):
    selected = defaultdict(list)
    for line in a.frames.open():
        r = json.loads(line)
        selected[r['episode_id']].append(int(r['obs_index']))
    metadata = sorted((m for m in load_episodes(a.source) if m['stem'] in selected),
                      key=lambda m: (m['task_id'], m['episode_idx']))
    if len(metadata) != len(selected):
        raise ValueError('frame list and episode metadata differ')
    if a.task_ids is not None:
        metadata = [m for m in metadata if int(m['task_id']) in a.task_ids]
        if {int(m['task_id']) for m in metadata} != set(a.task_ids):
            raise ValueError('requested task id missing')
    if a.limit_episodes:
        metadata = metadata[:a.limit_episodes]
    a.out.mkdir(parents=True, exist_ok=False)
    model = PM.build_panda_self_filter_model()
    cfg = PointCloudConfig()
    if cfg.self_filter_inflation != 0 or cfg.self_filter_target_guard_radius != 0:
        raise ValueError('experiment expects zero inflation and target guard')
    manifest = {'source': str(a.source.resolve()), 'source_episodes_sha256': sha(a.source/'episodes.jsonl'),
                'scene': str(a.scene.resolve()), 'phase': str(a.phase.resolve()),
                'noun_run': str(a.noun_run.resolve()), 'frames_sha256': sha(a.frames),
                'script_sha256': sha(Path(__file__)), 'protocol': str(a.protocol.resolve()),
                'protocol_sha256': sha(a.protocol), 'head': 'noun L8H6',
                'seed_conditions': ['noun', 'gt_patch_privileged'],
                'robot_conditions': ['none', 'fk_spheres', 'gt_image_privileged'],
                'fk': {'model': 'Panda default confined_hand_mount self-filter', 'inflation_m': 0,
                       'target_guard_m': 0, 'backend': 'cpu', 'state': 'image-time 9 joint positions',
                       'model_manifest': PM.manifest_entry()},
                'camera_adapter': 'two cameras plus fully robot-masked dummy right wrist',
                'selection': 'all fixed policy-call frames; abstention scored as empty',
                'episodes': [m['stem'] for m in metadata], 'test': False}
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    (a.out/'masks').mkdir()
    start = time.monotonic()
    with (a.out/'frames.jsonl').open('x') as out:
        for n, meta in enumerate(metadata, 1):
            ep = Episode(meta, cameras=CAMERAS, scene_dir=a.scene)
            with np.load(meta['geom_path'], allow_pickle=False) as z:
                state = {c: z[f'robot_state/{c}'] for c in CAMERAS}
            with np.load(a.noun_run/'noun_rows'/f'{ep.stem}_noun.npz', allow_pickle=False) as z:
                noun = {k: z[k] for k in ('noun_image', 'obs_index', 'layers', 'noun_ok')}
            with np.load(a.phase/f'phase_{ep.stem}.npz', allow_pickle=False) as z:
                phase = {k: z[k] for k in ('call_frames', 'segP/head', 'segP/left_wrist')}
            layer_idx = np.flatnonzero(noun['layers'] == 8)
            if len(layer_idx) != 1:
                raise ValueError('noun layer missing')
            times = selected[ep.stem][:a.limit_frames] if a.limit_frames else selected[ep.stem]
            for t in times:
                ni = np.flatnonzero(noun['obs_index'] == t)
                pi = np.flatnonzero(phase['call_frames'] == t)
                if len(ni) != 1 or len(pi) != 1 or not noun['noun_ok'][ni[0]]:
                    raise ValueError(f'frame join or noun missing: {ep.stem}/{t}')
                v = noun['noun_image'][ni[0], layer_idx[0], 6]
                fr = ep.frame(t)
                r = fr['geom_row']
                if not np.array_equal(state['head'][r], state['left_wrist'][r]):
                    raise ValueError(f'camera robot states differ: {ep.stem}/{t}')
                target, gt_robot, fk_robot, seeds = {}, {}, {}, {'noun': {}, 'gt_patch_privileged': {}}
                robot_stats = {}
                for j, c in enumerate(CAMERAS):
                    seg = phase[f'segP/{c}'][pi[0]]
                    fr['labels'][c] = seg_labels(seg, ep.lut)
                    target[c] = np.isin(seg, ep.target_geoms)
                    gt_robot[c] = np.isin(seg, ep.robot_geoms)
                    fk_robot[c] = PM.panda_depth_robot_mask(fr['depth_m'][c], fr['K'][c], fr['T'][c],
                                                           state[c][r], model, cfg, device='cpu').mask
                    seeds['noun'][c] = np.ascontiguousarray(v[j*256:(j+1)*256].reshape(16, 16)[:, ::-1])
                    seeds['gt_patch_privileged'][c] = target[c].reshape(16, 16, 16, 16).mean(axis=(1, 3))
                    pred, true = fk_robot[c], gt_robot[c]
                    robot_stats[c] = {'tp': int((pred & true).sum()), 'fp': int((pred & ~true).sum()),
                                      'fn': int((~pred & true).sum()), 'gt_robot_px': int(true.sum()),
                                      'target_px': int(target[c].sum()),
                                      'target_deleted_px': int((pred & target[c]).sum())}
                robot_masks = {'none': {c: np.zeros_like(gt_robot[c]) for c in CAMERAS},
                               'fk_spheres': fk_robot, 'gt_image_privileged': gt_robot}
                masks, records = {}, {}
                for seed_name, maps in seeds.items():
                    for robot_name, rm in robot_masks.items():
                        arm = seed_name+'__'+robot_name
                        views = [SurfaceView(c, fr['depth_m'][c], fr['K'][c], fr['T'][c], rm[c], maps[c])
                                 for c in CAMERAS]
                        dummy = np.zeros_like(fr['depth_m']['head']); dummy[0, 0] = 1
                        views.append(SurfaceView(CAMERA_ORDER[2], dummy, fr['K']['head'], fr['T']['head'],
                                                 np.ones_like(gt_robot['head']), np.zeros((16, 16))))
                        result = select_target_surface(views, arm=arm)
                        if result.codes is not None and np.any(result.codes['right_wrist']):
                            raise ValueError('dummy camera emitted graph pixels')
                        masks[arm] = {c: result.codes[c] == 1 if result.codes is not None
                                      else np.zeros_like(gt_robot[c]) for c in CAMERAS}
                        records[arm] = {'applied': bool(result.record['applied']),
                                        'why': result.record.get('why'), 'selection': result.record}
                metrics = scores(fr, masks)
                packed = {f'{arm}__{c}': np.packbits(masks[arm][c]) for arm in masks for c in CAMERAS}
                np.savez_compressed(a.out/'masks'/f'{ep.stem}_{t:05d}.npz', **packed)
                row = {'episode_id': ep.stem, 'task_id': int(meta['task_id']), 'obs_index': t,
                       'robot_stats': robot_stats, 'arms': {arm: {**records[arm], 'metrics': metrics[arm]}
                                                          for arm in masks}}
                out.write(json.dumps(row, allow_nan=False)+'\n'); out.flush()
            print(f'{n}/{len(metadata)} {ep.stem} elapsed={time.monotonic()-start:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'scene', 'phase', 'noun-run', 'frames', 'protocol', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limit-episodes', type=int, default=0)
    p.add_argument('--limit-frames', type=int, default=0)
    p.add_argument('--task-ids', type=int, nargs='+', help='independent task shard')
    run(p.parse_args())


if __name__ == '__main__':
    main()
