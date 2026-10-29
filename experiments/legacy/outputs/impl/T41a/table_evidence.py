"""T41a evidence: how close do the hand's query spheres get to the TABLE (support plane) in E3b runs,
and how well does |d_field - s_plane| <= tol identify the table as the nearest surface."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np
from scipy.ndimage import map_coordinates
A = json.load(open(_ag3s_workspace_path('outputs/verify/T40/G/analysis_all.json')))['runs']
HAND = ('ee_left', 'ee_finger_l1', 'ee_finger_l2')

def interp(grid, origin, vs, p):
    idx = ((p - origin) / vs).T
    shape = np.array(grid.shape)[:, None]
    inside = np.all((idx >= 0) & (idx <= shape - 1), axis=0)
    out = np.full(len(p), np.nan)
    if inside.any():
        out[inside] = map_coordinates(grid.astype(np.float32), idx[:, inside], order=1)
    return out

out = {}
for key, run in A.items():
    if not key.startswith('E3b'):
        continue
    seg = run['server_segment']
    base = seg['first'].rsplit('/', 1)[0]
    i0 = int(seg['first'][-9:-4]); i1 = int(seg['last'][-9:-4])
    rows = []
    for i in range(i0, i1 + 1):
        z = np.load(f'{base}/chunk_{i:05d}.npz', allow_pickle=True)
        summ = json.loads(str(z['summary_json']))
        names = np.array(summ['sphere_link_names'])
        hand = np.isin(names, HAND)
        c = z['sphere_centres'].astype(float)[hand]; r = z['sphere_radii'].astype(float)[hand]
        sup = z['filtered_cloud'][z['support_mask']].astype(float); sup = sup[sup[:, 2] > 0.4]
        if len(sup) < 50:
            continue
        mu = sup.mean(0); _, _, vt = np.linalg.svd(sup - mu, full_matrices=False)
        n = vt[2] * np.sign(vt[2][2]); off = float(n @ mu)
        s = c @ n - off
        d = interp(z['esdf_distance'], z['esdf_origin'], float(z['esdf_voxel_size']), c)
        dc = interp(z['esdf_coarse_distance'], z['esdf_coarse_origin'], float(z['esdf_coarse_voxel_size']), c)
        d = np.where(np.isnan(d), dc, np.minimum(d, np.nan_to_num(dc, nan=9)))
        rows.append(dict(t=int(z['t_step']), s_minus_r=s - r, d_minus_r=d - r, diff=d - s))
    allrow = [(rw['t'], j) for rw in rows for j in range(len(rw['s_minus_r']))]
    smr = np.concatenate([rw['s_minus_r'] for rw in rows]); dmr = np.concatenate([rw['d_minus_r'] for rw in rows])
    diff = np.concatenate([rw['diff'] for rw in rows])
    near = smr < 0.03            # hand sphere surface within 30 mm of the table plane
    tbl = near & (np.abs(diff) <= 0.005)
    per_chunk_min = [float(np.min(rw['s_minus_r'])) for rw in rows]
    out[key] = dict(success=run['success'], grasp_t=run.get('grasp_t'), n_chunks=len(rows),
                    min_table_gap_mm=1e3 * float(smr.min()),
                    n_chunks_hand_within_10mm_of_table=int(sum(m < 0.010 for m in per_chunk_min)),
                    n_chunks_hand_within_5mm_of_table=int(sum(m < 0.005 for m in per_chunk_min)),
                    n_rows_within_30mm=int(near.sum()),
                    frac_rows_within_30mm_where_abs_d_minus_s_le_5mm=float(tbl.sum() / max(near.sum(), 1)),
                    d_minus_s_mm_rows_within_30mm_q05_q50_q95=[float(1e3 * np.nanquantile(diff[near], q)) for q in (0.05, .5, .95)] if near.any() else None)
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
