#!/usr/bin/env python3
"""Score the masks of one run_safelibero_baseline.py run on the SafeLIBERO image-instant GT (2D + 3D, class-aware)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safelibero_scoring as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    out = a.out or (a.run/'scores.jsonl')
    t0, n = time.time(), 0
    with out.open('x') as f:
        for mp in sorted((a.run/'masks').glob('*.npz')):
            stem = mp.stem
            ep = S.Episode(gt_dirs[stem], stem)
            with np.load(mp, allow_pickle=False) as z:
                calls = z['call_frames']
                masks = {c: np.unpackbits(z[c], axis=-1, count=256).astype(bool) for c in S.CAMERAS}
            for k, t in enumerate(calls):
                fr = ep.frame(int(t)); gb = S.grid_of(fr)
                r = S.score_masks(fr, {c: masks[c][k] for c in S.CAMERAS}, gb)
                row = {'stem': stem, 'task': int(stem.split('_t')[1].split('_')[0]), 'episode': int(stem.rsplit('_e', 1)[1]),
                       't': int(t), **r, 'meta': S.frame_meta(fr, gb[1])}
                f.write(json.dumps(row, allow_nan=False)+'\n'); n += 1
            print(stem, len(calls), 'elapsed %.0fs' % (time.time()-t0), flush=True)
    print('rows', n)


if __name__ == '__main__':
    main()
