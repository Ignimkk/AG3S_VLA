#!/usr/bin/env python3
"""Per-frame GT voxel mass per class on the SafeLIBERO eval grid (denominators for erase / keep fractions)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safelibero_scoring as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    out = {}
    for stem in sorted(gt_dirs):
        ep = S.Episode(gt_dirs[stem], stem)
        for t in ep.call_frames:
            fr = ep.frame(int(t)); grid, _ = S.grid_of(fr)
            out['%s:%d' % (stem, t)] = {c: float(grid.class_frac[:, i].sum()) for i, c in enumerate(S.CLASSES)}
        print(stem, flush=True)
    a.out.write_text(json.dumps(out)+'\n')


if __name__ == '__main__':
    main()
