from common.workspace import workspace_path as _ag3s_workspace_path
import sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
from AG3S.trajopt import sweep as SW
for mode in sys.argv[1].split(','):
  SW.PathBlock.SLACK = mode
  for name, seq, om in (('T40C1807', 28, 0.010), ('T40C1807', 28, 0.0), ('T41aC1807', 35, None)):
    for ov_name, ov in (('servo', {'limits': {'servo_model': True}}), ('sweep', {'reduction': {'sweep_check': True}}), ('both', {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}})):
        r = R.replay(name, seq, overrides=ov, obstacle_margin=om, max_iterations=3)
        m = r['metrics']
        print(mode, name, om, ov_name, r['status'], 'wp %.2f' % (r['clearance_m'] * 1e3), 'path %.2f' % (m['path_check']['clearance_m'] * 1e3), 'step0 deg %.1f' % np.degrees(np.abs(r['trajectory'][:, 0] - r['q_now'][r['optimizer'].layout.q_indices]).max()),
              [(round(c['violation_m'] * 1e3, 1), c['accepted'], round(c['trust_radius_rad'], 2)) for c in m['candidates']], 'ms %.0f' % r['result'].solve_time_ms, flush=True)
