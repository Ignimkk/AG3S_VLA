#!/usr/bin/env python3
"""Figures for the real-ATS vs external-segmenter offline comparison (graph, per-task table, real-scene overlays)."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle
from Baseline.paper_baselines.eval.offline_grounding import Episode, load_episodes, seg_labels

LABEL = {'oracle_image': 'GT oracle (privileged)', 'adapter_noun_L8H6_fk': 'earlier adapter (surface graph, L8H6)',
         'grounded_sam2_frame_phrase': 'Grounded-SAM-2 phrase', 'grounded_sam2_frame_instruction': 'Grounded-SAM-2 instruction',
         'yoloe005_posthoc': 'YOLOE conf 0.05 (post hoc)', 'yoloe025': 'YOLOE conf 0.25',
         'ATS_real_target_cluster': 'real ATS: target cluster', 'ATS_real_erased_extent': 'real ATS: erased extent'}
# fixed positions (task, episode, call index) chosen before looking at any score; t1_e5 was already inspected for debugging
PANELS = [('libero_object_t1_e5', 3), ('libero_object_t1_e5', 20), ('libero_object_t4_e5', 3), ('libero_object_t4_e5', 20),
          ('libero_object_t7_e5', 3), ('libero_object_t7_e5', 20)]


def project(points, K, T):
    cam = (np.linalg.inv(T) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    ok = cam[:, 2] > 1e-3
    uv = (K @ cam.T).T
    return uv[:, 0]/np.where(ok, uv[:, 2], 1), uv[:, 1]/np.where(ok, uv[:, 2], 1), ok


def main():
    ap = argparse.ArgumentParser()
    for n in ('summary', 'real', 'root', 'source', 'out'):
        ap.add_argument('--'+n, type=Path, required=True)
    a = ap.parse_args()
    figstyle.use_korean()
    s = json.loads((a.summary/'summary.json').read_text())
    res = s['results']
    order = sorted(res, key=lambda n: res[n]['iou']['mean'])
    a.out.mkdir(parents=True, exist_ok=True)

    # 1. graph
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (m, title) in zip(axes, [('iou', '3D soft IoU'), ('recall', 'recall'), ('identity', 'target selected (precision >= 0.5)')]):
        vals = [res[n][m]['mean'] for n in order]
        cols = [figstyle.CATEGORICAL[0] if n.startswith('ATS_real') else figstyle.CATEGORICAL[1] if n == 'oracle_image'
                else figstyle.GRID_INK for n in order]
        ax.barh([LABEL[n] for n in order], vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, 1.12)
    figstyle.style_axes(fig, axes)
    fig.suptitle('Object init 5/6 (552 frames, 10-task equal mean): real Panda final-configuration ATS vs external segmenters')
    fig.tight_layout()
    fig.savefig(a.out/'fig1_methods.png', dpi=140)
    plt.close(fig)

    # 2. per-task table
    names = [n for n in order[::-1]]
    mat = np.array([[res[n]['iou']['task'][str(t)] for t in range(10)] for n in names])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(mat, cmap=figstyle.sequential_cmap(), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(10), [f'task {t}' for t in range(10)])
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names])
    for i in range(mat.shape[0]):
        for j in range(10):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if mat[i, j] > 0.55 else figstyle.INK)
    ax.set_title('3D soft IoU per task (each cell: frames -> episodes -> mean)')
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(a.out/'fig2_per_task_iou.png', dpi=140)
    plt.close(fig)

    # 3. real scene overlays (head camera)
    meta = {m['stem']: m for m in load_episodes(a.source)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sidecar = []
    for ax, (stem, k) in zip(axes.T.reshape(-1) if False else axes.reshape(-1), PANELS):
        ep = Episode(meta[stem], cameras=('head', 'left_wrist'), scene_dir=a.root/'scene')
        z = np.load(a.real/'clouds'/f'{stem}.npz')
        t = int(z['obs_index'][k])
        tp = z['target_points'][z['offsets'][k]:z['offsets'][k+1]].astype(float)
        xp = z['ext_points'][z['ext_offsets'][k]:z['ext_offsets'][k+1]].astype(float)
        fr = ep.frame(t)
        ph = np.load(a.root/'phase'/f'phase_{stem}.npz')
        pi = int(np.flatnonzero(ph['call_frames'] == t)[0])
        gt = np.isin(ph['segP/head'][pi], ep.target_geoms)
        rgb = fr['rgb']['head']
        ax.imshow(np.asarray(rgb))
        sc = rgb.shape[0]/gt.shape[0]
        ax.contour(np.kron(gt, np.ones((int(round(sc)) or 1,)*2)) if sc >= 1 else gt, levels=[0.5], colors=['#00c000'], linewidths=1.6)
        for pts, col, lab, sz in ((xp, '#2a6df4', 'erased extent', 4), (tp, '#e02020', 'target cluster', 8)):
            if len(pts):
                u, v, ok = project(pts, fr['K']['head'], fr['T']['head'])
                ax.scatter(u[ok]*sc, v[ok]*sc, s=sz, c=col, alpha=0.65, label=f'{lab} ({len(pts)})', linewidths=0)
        row = next(json.loads(l) for l in (a.real/'frames.jsonl').open()
                   if f'"{stem}"' in l and f'"obs_index": {t},' in l)
        ax.set_title(f"{stem.replace('libero_object_', '')} obs {t}  IoU {row['metrics']['iou']:.2f} / extent {row['extent_metrics']['iou']:.2f}", fontsize=9)
        ax.legend(loc='lower right', fontsize=7)
        ax.axis('off')
        sidecar.append({'episode': stem, 'obs_index': t, 'n_target_points': int(len(tp)), 'n_extent_points': int(len(xp)),
                        'iou_target_cluster': row['metrics']['iou'], 'iou_extent': row['extent_metrics']['iou']})
    fig.suptitle('Head camera, real frames. green: simulator GT target outline (evaluation only), red: ATS target cluster, blue: voxels the target-free field erased')
    fig.tight_layout()
    fig.savefig(a.out/'fig3_scene_overlays.png', dpi=130)
    plt.close(fig)
    # 4. stratified by whether the GT target has moved
    st = json.loads((a.out/'stratified.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    names4 = sorted(st['methods'], key=lambda n: st['methods'][n]['static']['iou']['mean'])
    for ax, stratum in zip(axes, ('static', 'moved')):
        ax.barh([LABEL[n] for n in names4], [st['methods'][n][stratum]['iou']['mean'] for n in names4],
                color=[figstyle.CATEGORICAL[0] if n.startswith('ATS_real') else figstyle.CATEGORICAL[1] if n == 'oracle_image'
                       else figstyle.GRID_INK for n in names4])
        for i, n in enumerate(names4):
            x = st['methods'][n][stratum]
            ax.text(x['iou']['mean']+0.01, i, f"{x['iou']['mean']:.3f}  (selected {x['identity']['mean']:.2f})", va='center', fontsize=8)
        ax.set_xlim(0, 1.3)
        ax.set_title(f"{'target not yet moved' if stratum == 'static' else 'target moved >= 20 mm (no-grasp replay)'}: {st['frames'][stratum]} frames")
    figstyle.style_axes(fig, axes)
    fig.suptitle('3D soft IoU split by GT target motion (threshold fixed before the split)')
    fig.tight_layout()
    fig.savefig(a.out/'fig4_static_vs_moved.png', dpi=140)
    plt.close(fig)
    (a.out/'figures.json').write_text(json.dumps({'fig1_methods': {n: {m: res[n][m]['mean'] for m in ('iou', 'recall', 'identity')} for n in res},
                                                  'fig2_per_task_iou': {n: res[n]['iou']['task'] for n in res},
                                                  'fig3_panels': sidecar, 'fig4_stratified': st['methods'], 'fig4_frames': st['frames']}, indent=1)+'\n')


if __name__ == '__main__':
    main()
