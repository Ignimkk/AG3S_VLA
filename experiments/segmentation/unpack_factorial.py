#!/usr/bin/env python3
"""Expose packed factorial masks to the existing image-instant evaluator."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    expected = {(r['episode_id'], int(r['obs_index'])) for r in map(json.loads, a.frames.open())}
    arm_dirs = {'noun__none': 'noun_none', 'noun__fk_spheres': 'noun_fk',
                'noun__gt_image_privileged': 'noun_gt_robot',
                'gt_patch_privileged__gt_image_privileged': 'gt_patch_gt_robot'}
    a.out.mkdir(parents=True, exist_ok=False)
    for folder in arm_dirs.values(): (a.out/folder/'masks').mkdir(parents=True)
    found = set()
    for shard in a.shards:
        for r in map(json.loads, (shard/'frames.jsonl').open()):
            key = r['episode_id'], r['obs_index']
            if key in found: raise ValueError(f'duplicate {key}')
            found.add(key)
            filename = f'{key[0]}_{key[1]:05d}.npz'
            with np.load(shard/'masks'/filename, allow_pickle=False) as z:
                for arm, folder in arm_dirs.items():
                    masks = {c: np.unpackbits(z[f'{arm}__{c}'], count=256*256).reshape(256,256).astype(bool)
                             for c in ('head','left_wrist')}
                    np.savez_compressed(a.out/folder/'masks'/filename, **masks)
    if found != expected: raise ValueError(f'frame mismatch {len(found)} vs {len(expected)}')
    (a.out/'manifest.json').write_text(json.dumps({'frames': len(found), 'arms': arm_dirs,
                                                  'shards': list(map(str,a.shards))},indent=2)+'\n')
    print('unpacked',len(found),'frames')


if __name__ == '__main__': main()
