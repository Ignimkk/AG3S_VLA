#!/usr/bin/env python3
"""Figures for the SafeLIBERO baseline-segmentation vs ATS comparison: arm bars (segmentation + erase/keep), obstacle-distance strata, real frames."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from AG3S.ag3s.experiments.common import figstyle
import safelibero_scoring as S

PANELS = [('safelibero_spatial_t0_e0', 30), ('safelibero_spatial_t0_e1', 40), ('safelibero_spatial_t1_e0', 40),
          ('safelibero_spatial_t2_e2', 25), ('safelibero_spatial_t3_e0', 25), ('safelibero_spatial_t3_e2', 30)]   # fixed before looking at scores
SHOW = ['gsam2_track__instruction', 'gsam_v1__phrase', 'yoloe005__phrase']
COL = {'gsam2_track__instruction': '#2a6df4', 'gsam_v1__phrase': '#e8a600', 'yoloe005__phrase': '#8a3ffc'}
RAW = Path('/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only/episodes')


def project(points, K, T):
    cam = (np.linalg.inv(T) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    ok = cam[:, 2] > 1e-3
    uv = (K @ cam.T).T
    return uv[:, 0]/np.where(ok, uv[:, 2], 1), uv[:, 1]/np.where(ok, uv[:, 2], 1), ok


def main():
    ap = argparse.ArgumentParser()
    for n in ('summary', 'baselines', 'ats', 'out'):
        ap.add_argument('--'+n, type=Path, required=True)
    ap.add_argument('--dataset', type=Path, nargs='+', required=True)
    a = ap.parse_args()
    figstyle.use_korean()
    res = json.loads((a.summary/'summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    order = sorted(res, key=lambda n: res[n]['all']['iou3d']['mean'])
    isats = lambda n: n.startswith('ATS')
    colors = [figstyle.CATEGORICAL[0] if isats(n) else figstyle.GRID_INK for n in order]
    panels = [('iou3d', '3D soft IoU (target)'), ('dominant_target', 'target is the largest class'), ('target_erased', 'target voxels erased (recall)'),
              ('obstacle_erased', 'obstacle voxels erased (wrong)')]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        vals = [res[n]['all'][m]['mean'] or 0 for n in order]
        ax.barh(order, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v+0.005, i, '%.3f' % v, va='center', fontsize=7)
        ax.set_title(title, fontsize=10); ax.set_xlim(0, max(max(vals)*1.25, 0.1))
    figstyle.style_axes(fig, axes)
    fig.suptitle('SafeLIBERO spatial Level I dev (20 episodes, 796 frames): external segmenters vs real Panda ATS (blue)')
    fig.tight_layout(); fig.savefig(a.out/'fig1_arms.png', dpi=130); plt.close(fig)

    buckets = ['<30mm', '30-60mm', '>=60mm']
    pick = ['gsam2_track__instruction', 'gsam_v1__phrase', 'yoloe005__phrase', 'ATS_physical__target_cluster', 'ATS_physical__erased_extent']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (m, title) in zip(axes, [('iou3d', '3D soft IoU'), ('target_erased', 'target erased'), ('obstacle_erased', 'obstacle erased (wrong)')]):
        w = 0.16
        for i, n in enumerate(pick):
            vals = [(res[n][b][m]['mean'] if res[n].get(b) and res[n][b][m]['mean'] is not None else 0) for b in buckets]
            ax.bar(np.arange(3)+(i-2)*w, vals, w, label=n, color=figstyle.CATEGORICAL[i % len(figstyle.CATEGORICAL)])
        ax.set_xticks(range(3), ['target-obstacle < 30 mm', '30-60 mm', '>= 60 mm']); ax.set_title(title)
    axes[0].legend(fontsize=7)
    figstyle.style_axes(fig, axes); fig.suptitle('Split by the nearest target-obstacle distance of each frame')
    fig.tight_layout(); fig.savefig(a.out/'fig2_obstacle_distance.png', dpi=130); plt.close(fig)

    gt_dirs = {p.name[:-len('_gt.npz')]: p.parent for d in a.dataset for p in d.glob('*_gt.npz')}
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5)); sidecar = []
    for ax, (stem, t) in zip(axes.reshape(-1), PANELS):
        ep = S.Episode(gt_dirs[stem], stem); fr = ep.frame(t)
        with np.load(RAW/(stem+'.npz')) as z:
            rgb = z['obs_agentview_image'][t][::-1]
        ax.imshow(rgb)
        lab = fr['labels']['head']
        for cls, col in (('target', '#00c000'), ('obstacle', '#ff7f00')):
            ax.contour(lab == S.ci(cls), levels=[0.5], colors=[col], linewidths=1.6)
        for n in SHOW:
            with np.load(a.baselines/n/'masks'/(stem+'.npz')) as z:
                k = int(np.flatnonzero(z['call_frames'] == t)[0]); m = np.unpackbits(z['head'][k], axis=-1, count=256).astype(bool)
            ax.contour(m, levels=[0.5], colors=[COL[n]], linewidths=1.4, linestyles='--')
        with np.load(a.ats/'clouds'/(stem+'.npz')) as z:
            k = int(np.flatnonzero(z['call_frames'] == t)[0])
            tp = z['target_points'][z['offsets'][k]:z['offsets'][k+1]]; xp = z['ext_points'][z['ext_offsets'][k]:z['ext_offsets'][k+1]]
        for pts, c, sz in ((xp, '#2fa8a8', 3), (tp, '#e02020', 8)):
            if len(pts):
                u, v, ok = project(pts.astype(float), fr['K']['head'], fr['T']['head'])
                ax.scatter(u[ok], v[ok], s=sz, c=c, alpha=0.6, linewidths=0)
        ax.set_title('%s  t=%d' % (stem.replace('safelibero_spatial_', ''), t), fontsize=9); ax.axis('off')
        sidecar.append({'stem': stem, 't': t})
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([0], [0], color='#00c000', label='GT target'), Line2D([0], [0], color='#ff7f00', label='GT obstacle'),
                               *[Line2D([0], [0], color=COL[n], ls='--', label=n) for n in SHOW],
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='#e02020', label='ATS target cluster'),
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='#2fa8a8', label='ATS erased voxels')], fontsize=7, loc='lower left')
    fig.suptitle('Head camera, real SafeLIBERO frames (fixed panel list, chosen before looking at scores)')
    fig.tight_layout(); fig.savefig(a.out/'fig3_scene_overlays.png', dpi=120); plt.close(fig)
    (a.out/'figures.json').write_text(json.dumps({'arms': {n: {m: res[n]['all'][m]['mean'] for m in ('iou3d', 'dominant_target', 'target_erased', 'obstacle_erased')} for n in res},
                                                  'strata': {n: {b: (res[n][b]['iou3d']['mean'] if res[n].get(b) else None) for b in buckets} for n in pick}, 'panels': sidecar}, indent=1)+'\n')


if __name__ == '__main__':
    main()
