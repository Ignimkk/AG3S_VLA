from common.workspace import workspace_path as _ag3s_workspace_path
import sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
BOTH = {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}}
seq = 18
off = R.replay('T40B1968', seq)
c = R.check_plan('T40B1968', seq, off['chunk'], overrides=BOTH)
opt, scene = c['optimizer'], c['scene']
lin = opt.linearizer
w = c['path']['worst']; q = w['query']; pt = np.asarray(w['point_m'])
mask = lin._target_free_rows(scene, lin.n_spheres)
print('worst', w, 'target_free row', None if mask is None else bool(mask[q]))
print('tier', lin._esdf_tier(pt.reshape(1, 3), scene, bool(mask is not None and mask[q])), 'label', lin._esdf_label_name(pt, scene))
print('d (composite)', float(scene.esdf.distance(pt.reshape(1,3))[0]), 'd target-free', float(scene.esdf.target_free_distance(pt.reshape(1,3))[0]) if hasattr(scene.esdf,'target_free_distance') else None, 'r', lin.query_radii[q])
tp = scene.manipulated_points
if tp is not None:
    print('target centroid', tp.mean(0), 'dist to nearest target point', np.linalg.norm(tp - pt, axis=1).min())
print('has_target', c['optimizer'] is not None, 'manip margin', None if scene.manipulated_link_margin is None else scene.manipulated_link_margin[q])
# all rows at q_now below 0
cen = opt.path._centres_now
h0 = lin.flat_clearances(cen[None], scene)[0]
bad = np.argsort(h0)[:5]; print('q_now worst rows', [(int(b), lin._query_name(int(b) - (len(h0) - cen.shape[0])), round(h0[b]*1e3,2)) for b in bad])
