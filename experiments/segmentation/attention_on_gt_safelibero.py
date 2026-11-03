#!/usr/bin/env python3
"""Stage-1 diagnostic of the noun attention on SafeLIBERO: how much of the (head / wrist) attention grid sits on each GT class."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safelibero_scoring as S


def patch_class_mass(lab):
    """256x256 class labels -> (16,16,C) fraction of each class per 16x16 patch."""
    C = len(S.CLASSES)
    one = np.stack([(lab == i) for i in range(C)], -1).astype(np.float32)          # (256,256,C)
    return one.reshape(16, 16, 16, 16, C).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noun-run', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    ap.add_argument('--cells', default='8,6;10,7;8,2')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cells = [tuple(int(x) for x in c.split(',')) for c in a.cells.split(';')]
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    acc = {c: {'top1': [], 'mass': {k: [] for k in S.CLASSES}, 'top1_class': []} for c in cells}
    per_task = {c: {} for c in cells}
    for stem in sorted(gt_dirs):
        ep = S.Episode(gt_dirs[stem], stem); task = int(stem.split('_t')[1].split('_')[0])
        with np.load(a.noun_run/'noun_rows'/(stem+'_noun.npz'), allow_pickle=False) as z:
            noun, obs, layers = z['noun_image'], z['obs_index'], list(z['layers'])
        for k, t in enumerate(obs):
            fr = ep.frame(int(t)); pm = {cam: patch_class_mass(fr['labels'][cam]) for cam in S.CAMERAS}
            for (L, H) in cells:
                v = noun[k, layers.index(L), H]
                for j, cam in enumerate(S.CAMERAS):
                    g = v[j*256:(j+1)*256].reshape(16, 16)[:, ::-1]                # policy -> OpenCV orientation (as the replay)
                    g = g/max(g.sum(), 1e-12)
                    mass = {c: float((g*pm[cam][..., i]).sum()) for i, c in enumerate(S.CLASSES)}
                    top = np.unravel_index(np.argmax(g), g.shape); cls = S.CLASSES[int(np.argmax(pm[cam][top]))]
                    acc[(L, H)]['top1_class'].append((cam, cls, task)); 
                    for c in S.CLASSES:
                        acc[(L, H)]['mass'][c].append((cam, mass[c], task))
        print(stem, flush=True)
    out = {}
    for c, d in acc.items():
        o = {}
        for cam in S.CAMERAS:
            tc = [x[1] for x in d['top1_class'] if x[0] == cam]
            o[cam] = {'top1_patch_class': {k: tc.count(k)/len(tc) for k in sorted(set(tc))},
                      'mean_mass_on_class': {k: float(np.mean([m for cc, m, _ in d['mass'][k] if cc == cam])) for k in S.CLASSES}}
        out['L%dH%d' % c] = o
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
