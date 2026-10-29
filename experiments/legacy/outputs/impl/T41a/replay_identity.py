"""T41a: HEAD `_esdf_clearance` vs working-tree `_esdf_clearance` on recorded E3b chunks (margin off: must be
bit-identical), and the per-class effect with margin 0.010 (support excluded)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, importlib.util, numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.trajopt import linearize as NEW
from AG3S.trajopt.experiments.fixed_input_ablation import build_field
spec = importlib.util.spec_from_file_location('linearize_head', sys.argv[1]); OLD = importlib.util.module_from_spec(spec); sys.modules['linearize_head'] = OLD; spec.loader.exec_module(OLD)
AUTH = ('ee_left', 'ee_finger_l1', 'ee_finger_l2')
A = json.load(open(_ag3s_workspace_path('outputs/verify/T40/G/analysis_all.json')))['runs']
def plane(z):
    cl = z['filtered_cloud'].astype(float); sup = cl[z['support_mask']]
    out = []
    for sel in (sup[:, 2] > 0.4, sup[:, 2] <= 0.4):
        p = sup[sel]
        if len(p) < 50: continue
        mu = p.mean(0); _, _, vt = np.linalg.svd(p - mu, full_matrices=False); n = vt[2] * np.sign(vt[2][2])
        out.append([*n, float(n @ mu)])
    return np.asarray(out)
def scene(mod, z, field, names, radii, extra):
    tp = z['target_points'].astype(float) if 'target_points' in z.files and len(z['target_points']) else None
    auth = np.isin(names, AUTH)
    m = np.where(auth, 0.02, 0.05) if tp is not None else None
    mask = auth if (tp is not None and 'esdf_free_distance' in z.files) else None
    return mod.SceneSnapshot(candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
        d_safe=np.zeros((radii.size, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=radii, candidate_ids=np.zeros(0, np.int64), esdf=field, esdf_margin=0.0,
        manipulated_points=tp if m is not None else None, manipulated_link_margin=m, target_free_mask=mask, **extra)
def lin(mod, radii):
    l = object.__new__(mod.CollisionLinearizer); l.query_radii = radii; l.n_spheres = radii.size; l._attached_local = None; return l
rep = {'n_chunks': 0, 'n_rows': 0, 'max_abs_diff_off_vs_head': 0.0, 'n_rows_changed_on': 0, 'by_class_on': {c: 0 for c in NEW.OBSTACLE_CLASSES},
       'changed_rows_all_obstacle_class': True, 'change_equals_margin': True, 'per_run': {}}
for key, run in A.items():
    if not key.startswith('E3b'): continue
    seg = run['server_segment']; base = seg['first'].rsplit('/', 1)[0]
    i0 = int(seg['first'][-9:-4]); i1 = int(seg['last'][-9:-4])
    pr = dict(n_chunks=0, hand_rows_obstacle_lt_10mm=0, hand_rows_support_lt_10mm=0, hand_rows_target=0, chunks_with_obstacle_change=[])
    for i in range(i0, i1 + 1, int(sys.argv[3]) if len(sys.argv) > 3 else 1):
        z = np.load(f'{base}/chunk_{i:05d}.npz', allow_pickle=True)
        summ = json.loads(str(z['summary_json'])); names = np.array(summ['sphere_link_names'])
        radii = z['sphere_radii'].astype(np.float64); c = z['sphere_centres'].astype(np.float64)[None]
        field = build_field(z, outside_distance=1.0)
        old = lin(OLD, radii)._esdf_clearance(c, scene(OLD, z, field, names, radii, {}))[..., 0]
        off = lin(NEW, radii)._esdf_clearance(c, scene(NEW, z, field, names, radii, dict(support_planes=plane(z))))[..., 0]
        L = lin(NEW, radii)
        on = L._esdf_clearance(c, scene(NEW, z, field, names, radii, dict(support_planes=plane(z), obstacle_margin=0.010)))[..., 0]
        cls = L._last_obstacle_class
        rep['max_abs_diff_off_vs_head'] = max(rep['max_abs_diff_off_vs_head'], float(np.max(np.abs(old - off))))
        assert np.array_equal(old, off), (key, i)
        ch = on != off
        rep['n_rows_changed_on'] += int(ch.sum()); rep['n_rows'] += off.size; rep['n_chunks'] += 1
        for k, nm in enumerate(NEW.OBSTACLE_CLASSES): rep['by_class_on'][nm] += int((cls == k).sum())
        if ch.any():
            rep['changed_rows_all_obstacle_class'] &= bool(np.all(cls[ch] == 0))
            rep['change_equals_margin'] &= bool(np.allclose(off[ch] - on[ch], 0.010, atol=1e-12))
            pr['chunks_with_obstacle_change'].append(int(z['t_step']))
        hand = np.isin(names, AUTH)[None]
        pr['hand_rows_obstacle_lt_10mm'] += int(((cls == 0) & hand & (off < 0.010)).sum())
        pr['hand_rows_support_lt_10mm'] += int(((cls == 2) & hand & (off < 0.010)).sum())
        pr['hand_rows_target'] += int(((cls == 1) & hand).sum())
        pr['n_chunks'] += 1
    rep['per_run'][key] = pr
    print(key, json.dumps(pr), flush=True)
print(json.dumps({k: v for k, v in rep.items() if k != 'per_run'}, indent=1))
json.dump(rep, open(sys.argv[2], 'w'), indent=1)
