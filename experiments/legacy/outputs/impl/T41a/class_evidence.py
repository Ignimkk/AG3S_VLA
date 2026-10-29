"""T41a: what the obstacle margin would touch in recorded E3b runs (hand rows at q_now).
classes per hand row: target (|d - d_target| <= 5 mm), support (|d - s| <= 10 mm), other."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
A = json.load(open(_ag3s_workspace_path('outputs/verify/T40/G/analysis_all.json')))['runs']
HAND = ('ee_left', 'ee_finger_l1', 'ee_finger_l2')
def interp(g, o, vs, p):
    idx = ((p - o) / vs).T; sh = np.array(g.shape)[:, None]
    ins = np.all((idx >= 0) & (idx <= sh - 1), axis=0); out = np.full(len(p), np.nan)
    if ins.any(): out[ins] = map_coordinates(g.astype(np.float32), idx[:, ins], order=1)
    return out
res = {}
for key, run in A.items():
    if not key.startswith('E3b'): continue
    seg = run['server_segment']; base = seg['first'].rsplit('/', 1)[0]
    i0 = int(seg['first'][-9:-4]); i1 = int(seg['last'][-9:-4])
    per = []
    for i in range(i0, i1 + 1):
        z = np.load(f'{base}/chunk_{i:05d}.npz', allow_pickle=True)
        summ = json.loads(str(z['summary_json'])); names = np.array(summ['sphere_link_names'])
        h = np.isin(names, HAND); c = z['sphere_centres'].astype(float)[h]; r = z['sphere_radii'].astype(float)[h]
        cl = z['filtered_cloud'].astype(float); sup = cl[z['support_mask']]; sup = sup[sup[:, 2] > 0.4]
        mu = sup.mean(0); _, _, vt = np.linalg.svd(sup - mu, full_matrices=False); n = vt[2] * np.sign(vt[2][2])
        s = c @ n - n @ mu
        d = interp(z['esdf_distance'], z['esdf_origin'], float(z['esdf_voxel_size']), c)
        dc = interp(z['esdf_coarse_distance'], z['esdf_coarse_origin'], float(z['esdf_coarse_voxel_size']), c)
        d = np.where(np.isnan(d), dc, np.fmin(d, dc))
        if 'esdf_free_distance' in z.files:
            f = interp(z['esdf_free_distance'], z['esdf_free_origin'], float(z['esdf_free_voxel_size']), c)
            d = np.where(np.isnan(f), d, np.fmax(f, d))   # free >= d (approximation of the composite)
        tp = z['target_points'].astype(float) if 'target_points' in z.files else np.zeros((0, 3))
        dt = cKDTree(tp).query(c)[0] if len(tp) else np.full(len(c), np.inf)
        is_t = np.abs(d - dt) <= 0.005
        is_s = (~is_t) & (np.abs(d - s) <= 0.010)
        oth = ~(is_t | is_s)
        per.append(dict(t=int(z['t_step']), held=bool(json.loads(str(z['summary_json'])).get('grasp', {}) and summ['grasp'].get('state') == 'held'),
                        cl=d - r, is_t=is_t, is_s=is_s, oth=oth))
    def mins(mask_key):
        v = [(p['t'], float(np.min(p['cl'][p[mask_key]]))) for p in per if p[mask_key].any()]
        return v
    o = mins('oth'); s_ = mins('is_s'); t_ = mins('is_t')
    res[key] = dict(success=run['success'], n_chunks=len(per),
        other_min_clearance_mm=(1e3 * min(v for _, v in o)) if o else None,
        n_chunks_other_lt_10mm=sum(v < 0.010 for _, v in o),
        t_of_other_lt_10mm=[t for t, v in o if v < 0.010],
        support_min_clearance_mm=(1e3 * min(v for _, v in s_)) if s_ else None,
        n_chunks_support_lt_10mm=sum(v < 0.010 for _, v in s_),
        target_min_clearance_mm=(1e3 * min(v for _, v in t_)) if t_ else None)
    print(key, json.dumps(res[key]), flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
