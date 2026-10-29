from common.workspace import workspace_path as _ag3s_workspace_path
import sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
name, seq, om = 'T40C1807', 28, 0.010
for ov_name, ov in (('servo', {'limits': {'servo_model': True}}), ('sweep', {'reduction': {'sweep_check': True}}), ('both', {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}})):
    for it in (3, 6, 10):
        r = R.replay(name, seq, overrides=ov, obstacle_margin=om, max_iterations=it)
        m = r['metrics']
        print(ov_name, it, r['status'], 'wp %.2f' % (r['clearance_m'] * 1e3), 'path %.2f' % (m['path_check']['clearance_m'] * 1e3), 'step0 deg %.1f' % np.degrees(np.abs(r['trajectory'][:, 0] - r['q_now'][r['optimizer'].layout.q_indices]).max()),
              [(round(c['violation_m'] * 1e3, 1), c['accepted'], round(c['trust_radius_rad'], 2)) for c in m['candidates']], flush=True)
