#!/usr/bin/env python3
"""Freeze all policy-call frames after verifying image-time RGB re-render parity."""
import argparse
import json
from pathlib import Path
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import load_episodes


def main():
    p = argparse.ArgumentParser()
    for arg in ('source', 'scene', 'phase', 'out'):
        p.add_argument('--'+arg, type=Path, required=True)
    a = p.parse_args()
    metadata = sorted(load_episodes(a.source), key=lambda x:(x['task_id'], x['episode_idx']))
    expected = {(t,e) for t in range(10) for e in (5,6)}
    if {(e['task_id'],e['episode_idx']) for e in metadata} != expected or len(metadata) != 20:
        raise ValueError('rollout episodes differ from fixed protocol')
    phase_report = json.loads((a.phase/'phase_export.json').read_text())
    scene_report = json.loads((a.scene/'scene_export.json').read_text())
    if len(phase_report['episodes']) != 20 or len(scene_report['episodes']) != 20:
        raise ValueError('incomplete exports')
    for item in phase_report['episodes']:
        checks = item['checks']
        if any(checks['rgb_rerender_equal_call_frames'][c] != checks['call_frames']
               for c in ('head','left_wrist')):
            raise ValueError(f'RGB re-render mismatch {item["file"]}')
    count = 0
    with a.out.open('x') as out:
        for ep in metadata:
            stem = ep['stem']
            with np.load(a.scene/f'scene_{stem}.npz',allow_pickle=False) as s, \
                 np.load(a.phase/f'phase_{stem}.npz',allow_pickle=False) as z, \
                 np.load(ep['npz_path'],allow_pickle=False) as src:
                frames = s['frames'].astype(int)
                if not np.array_equal(frames,z['call_frames']) or not np.array_equal(frames,src['chunk_t']):
                    raise ValueError(f'policy-call frame lists disagree: {stem}')
                for t in frames:
                    out.write(json.dumps({'episode_id':stem,'task_id':int(ep['task_id']),
                                          'episode_idx':int(ep['episode_idx']),'obs_index':int(t)})+'\n')
                    count += 1
    print(f'verified RGB parity and policy-call joins for 20 episodes, {count} frames')


if __name__ == '__main__': main()
