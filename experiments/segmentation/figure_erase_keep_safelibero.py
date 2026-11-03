#!/usr/bin/env python3
"""Pixel view of what each arm erases from / keeps in the distance field: GT target pixels (green = erased, red = kept as an obstacle) and
GT obstacle pixels (black = wrongly erased, orange = kept). A pixel is 'erased' when its lifted 5 mm voxel is in the arm's voxel set."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from AG3S.ag3s.experiments.common import figstyle
from Baseline.paper_baselines import lift as L
import safelibero_scoring as S

FRAMES = [('safelibero_spatial_t0_e0', 30), ('safelibero_spatial_t2_e2', 25), ('safelibero_spatial_t3_e2', 30)]
ARMS = [('SAM2 track (instruction)', 'gsam2_track__instruction', 'mask'), ('Grounded-SAM v1 (phrase)', 'gsam_v1__phrase', 'mask'),
        ('ATS erased extent (physical)', 'ATS', 'points')]
RAW = S.RUN/'episodes'


def keyset(points):
    return {tuple(k) for k in L.voxel_keys(np.asarray(points, np.float64))} if len(points) else set()


def main():
    ap = argparse.ArgumentParser()
    for n in ('baselines', 'ats', 'out'):
        ap.add_argument('--'+n, type=Path, required=True)
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    a = ap.parse_args()
    figstyle.use_korean()
    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    fig, axes = plt.subplots(len(FRAMES), len(ARMS), figsize=(6*len(ARMS), 5.6*len(FRAMES))); sidecar = []
    for r, (stem, t) in enumerate(FRAMES):
        ep = S.Episode(gt_dirs[stem], stem); fr = ep.frame(t)
        with np.load(RAW/(stem+'.npz')) as z:
            rgb = z['obs_agentview_image'][t][::-1]
        Pc = {c: L.backproject_pixels(fr['depth_m'][c], fr['K'][c], fr['T'][c]) for c in S.CAMERAS}
        for cidx, (title, name, kind) in enumerate(ARMS):
            if kind == 'mask':
                with np.load(a.baselines/name/'masks'/(stem+'.npz')) as z:
                    k = int(np.flatnonzero(z['call_frames'] == t)[0])
                    masks = {c: np.unpackbits(z[c][k], axis=-1, count=256).astype(bool) for c in S.CAMERAS}
                erased = set()
                for c in S.CAMERAS:
                    P, u, v = Pc[c]; erased |= keyset(P[masks[c][v, u]])
            else:
                with np.load(a.ats/'clouds'/(stem+'.npz')) as z:
                    k = int(np.flatnonzero(z['call_frames'] == t)[0]); erased = keyset(z['ext_points'][z['ext_offsets'][k]:z['ext_offsets'][k+1]])
            P, u, v = Pc['head']; lab = fr['labels']['head']
            keys = [tuple(x) for x in np.floor(P/0.005).astype(np.int64)]
            ov = np.zeros(lab.shape+(4,), np.float32); n = {'t_er': 0, 't_keep': 0, 'o_er': 0, 'o_keep': 0}
            for i in range(len(P)):
                cls = lab[v[i], u[i]]; er = keys[i] in erased
                if cls == S.ci('target'):
                    ov[v[i], u[i]] = (0, .8, 0, .85) if er else (.9, 0, 0, .85); n['t_er' if er else 't_keep'] += 1
                elif cls == S.ci('obstacle'):
                    ov[v[i], u[i]] = (0, 0, 0, .9) if er else (1, .5, 0, .6); n['o_er' if er else 'o_keep'] += 1
            ax = axes[r, cidx]; ax.imshow(rgb); ax.imshow(ov); ax.axis('off')
            ax.set_title('%s | %s t=%d\ntarget erased %d / kept %d px, obstacle wrongly erased %d / kept %d px' % (
                title, stem.replace('safelibero_spatial_', ''), t, n['t_er'], n['t_keep'], n['o_er'], n['o_keep']), fontsize=9)
            sidecar.append({'arm': name, 'stem': stem, 't': t, **n})
    fig.suptitle('Distance-field erase / keep on real frames. green = target erased, red = target kept (blocks the approach), black = obstacle wrongly erased, orange = obstacle kept')
    fig.tight_layout(rect=[0, 0, 1, 0.97]); a.out.mkdir(parents=True, exist_ok=True); fig.savefig(a.out/'fig4_erase_keep_maps.png', dpi=110)
    (a.out/'fig4_erase_keep_maps.json').write_text(json.dumps(sidecar, indent=1)+'\n')


if __name__ == '__main__':
    main()
