"""Class-aware scoring on the SafeLIBERO image-instant GT (target / destination / obstacle / other object / ...).

Frames come from `capture_safelibero_dataset.py` outputs (geom-id maps) joined with the source run's depth, K, T.
Both mask-valued (2D + lifted 3D) and point-valued (ATS) predictions are scored on the same 5 mm eval grid.
"""
import json
from pathlib import Path
import numpy as np

from Baseline.paper_baselines import lift as L
from Baseline.paper_baselines.eval import metrics as M

RUN = Path('/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only')
CAMERAS = ('head', 'left_wrist')
CLASSES = ('background', 'target', 'destination', 'obstacle', 'other_object', 'fixture', 'support', 'robot', 'mount')
NEIGHBOR = ('other_object', 'destination', 'obstacle')
FIXTURES = ('wooden_cabinet_1', 'flat_stove_1')
KEEP3D = ('iou', 'precision', 'recall', 'TP', 'FP', 'FN', 'target_voxels', 'pred_voxels', 'neighbor_fp', 'obstacle_fp',
          'other_object_fp', 'destination_fp', 'fixture_fp', 'support_fp', 'robot_fp', 'background_fp')


def ci(name):
    return CLASSES.index(name)


class Episode:
    def __init__(self, gt_dir, stem):
        self.stem = stem
        with np.load(Path(gt_dir)/(stem+'_gt.npz'), allow_pickle=False) as z:
            self.obs_index = z['obs_index'].astype(np.int64)
            self.call_frames = z['call_frames'].astype(np.int64)
            self.seg = {'head': z['seg_head'], 'left_wrist': z['seg_left_wrist']}
            self.inst = json.loads(str(z['instances']))
            self.prompt = str(z['prompt'])
        with np.load(RUN/'episodes'/(stem+'_geom.npz'), allow_pickle=False) as g:
            self.geom_obs = g['obs_index'].astype(np.int64)
            self.depth_mm = {c: g['depth_mm/'+c] for c in CAMERAS}
            self.K = {c: g['K/'+c] for c in CAMERAS}
            self.T = {c: g['T_base_cam/'+c] for c in CAMERAS}
            self.robot_state = g['robot_state/head']
            names = [str(b) for b in g['model_body_names']]
            gb = g['model_geom_bodyid']
        lut = np.full(len(gb)+1, ci('background'), np.int64)          # last entry = segmentation -1
        for gi, b in enumerate(np.asarray(gb).tolist()):
            n = names[b]
            if n.startswith(('robot0_', 'gripper0_')) or n == 'eef_marker':
                lut[gi] = ci('robot')
            elif n.startswith('mount0_'):
                lut[gi] = ci('mount')
            elif n == 'table':
                lut[gi] = ci('support')
            elif n.startswith(FIXTURES) or n.startswith('box_base'):
                lut[gi] = ci('fixture')
            elif n not in ('world',):
                lut[gi] = ci('other_object')
        active = set(self.inst['active_obstacles'] or [])
        for name, gids in self.inst['obstacle_geoms'].items():
            if name in active:
                lut[np.asarray(gids, np.int64)] = ci('obstacle')
        lut[np.asarray(self.inst['destination_geoms'], np.int64)] = ci('destination')
        lut[np.asarray(self.inst['target_geoms'], np.int64)] = ci('target')
        self.lut = lut

    def labels(self, cam, row):
        s = self.seg[cam][row].astype(np.int64)
        return self.lut[np.where(s < 0, len(self.lut)-1, s)]

    def frame(self, t):
        gr = int(np.flatnonzero(self.geom_obs == t)[0]); sr = int(np.flatnonzero(self.obs_index == t)[0])
        out = {'t': int(t), 'depth_m': {}, 'K': {}, 'T': {}, 'labels': {}, 'robot_state': self.robot_state[gr]}
        for c in CAMERAS:
            out['depth_m'][c] = self.depth_mm[c][gr].astype(np.float64)/1000.0
            out['K'][c], out['T'][c] = self.K[c][gr], self.T[c][gr]
            out['labels'][c] = self.labels(c, sr)
        return out


def grid_of(fr):
    back = {c: L.backproject_pixels(fr['depth_m'][c], fr['K'][c], fr['T'][c]) for c in CAMERAS}
    grid = M.build_eval_grid([{'depth_m': fr['depth_m'][c], 'K': fr['K'][c], 'T_base_cam': fr['T'][c], 'labels': fr['labels'][c],
                               'backprojected': back[c]} for c in CAMERAS], classes=CLASSES)
    return grid, back


def metrics3d(grid, keys):
    m = M.voxel_metrics(grid, keys, neighbor=NEIGHBOR)
    return {k: m.get(k) for k in KEEP3D}


def score_masks(fr, masks, grid_back=None):
    """Mask per camera -> 2D class composition + 3D voxel metrics (union of the cameras)."""
    grid, back = grid_back or grid_of(fr)
    keys, two_d = [], {}
    for c in CAMERAS:
        pred, lab = masks[c].astype(bool), fr['labels'][c]
        gt = lab == ci('target')
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        comp = {n: int((pred & (lab == i)).sum()) for i, n in enumerate(CLASSES)}
        two_d[c] = {'gt_px': int(gt.sum()), 'pred_px': int(pred.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
                    'iou': tp/(tp+fp+fn) if gt.any() else None, 'recall': tp/(tp+fn) if gt.any() else None,
                    'precision': tp/(tp+fp) if pred.any() and gt.any() else None, 'pred_on': comp,
                    'gt_obstacle_px': int((lab == ci('obstacle')).sum())}
        P, u, v = back[c]
        keys.append(L.voxel_keys(P[pred[v, u]]))
    keys = np.concatenate(keys) if keys else np.zeros((0, 3), np.int64)
    return {'2d': two_d, '3d': metrics3d(grid, keys)}


def score_points(fr, points, grid_back=None):
    grid, _ = grid_back or grid_of(fr)
    keys = np.zeros((0, 3), np.int64) if points is None or len(points) == 0 else L.voxel_keys(np.asarray(points, np.float64))
    return {'3d': metrics3d(grid, keys)}


def frame_meta(fr, back):
    """Arm-independent facts of a frame: GT pixel counts per class and target-obstacle nearest distance (m, from lifted GT pixels)."""
    from scipy.spatial import cKDTree

    pts = {'target': [], 'obstacle': []}
    counts = {}
    for c in CAMERAS:
        lab = fr['labels'][c]
        counts[c] = {n: int((lab == i).sum()) for i, n in enumerate(CLASSES)}
        P, u, v = back[c]
        for name in pts:
            pts[name].append(P[(lab == ci(name))[v, u]])
    t, o = (np.concatenate(pts[k]) if pts[k] else np.zeros((0, 3)) for k in ('target', 'obstacle'))
    dist = float(cKDTree(o).query(t[::max(1, len(t)//400)])[0].min()) if len(t) and len(o) else None
    return {'gt_px': counts, 'target_obstacle_min_dist_m': dist}
