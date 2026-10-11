from common.workspace import workspace_path as _ag3s_workspace_path
import sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
import AG3S.trajopt.sqp as SQ
from AG3S.trajopt.sweep import PathRows, append_path_rows
SKIP = int(sys.argv[1])
orig = SQ.TrajectoryOptimizer._append_path
count = {'n': 0}
def patched(self, problem, iterate, scene):
    count['n'] += 1
    if count['n'] <= SKIP:
        H, Rr, nq = self.horizon, self.path.rows_per_step, self.layout.nq_opt
        rows = PathRows(np.full((H, Rr), np.inf), np.zeros((H, Rr, H, nq)), np.zeros((H, Rr), bool), np.zeros((H, Rr), int), np.zeros((H, Rr), int), 0)
        return append_path_rows(problem, rows, iterate, self.path.block, self.block.n_rows, 0.0)
    return orig(self, problem, iterate, scene)
SQ.TrajectoryOptimizer._append_path = patched
for name, seq, om in (('T40C1807', 28, 0.010), ('T40C1807', 28, 0.0), ('T41aC1807', 35, None)):
  for ov_name, ov in (('servo', {'limits': {'servo_model': True}}), ('sweep', {'reduction': {'sweep_check': True}}), ('both', {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}})):
    for it in (3, 5):
        count['n'] = 0
        r = R.replay(name, seq, overrides=ov, obstacle_margin=om, max_iterations=it)
        m = r['metrics']
        print(name, om, ov_name, it, r['status'], 'wp %.2f' % (r['clearance_m'] * 1e3), 'path %.2f' % (m['path_check']['clearance_m'] * 1e3), 'step0 deg %.1f' % np.degrees(np.abs(r['trajectory'][:, 0] - r['q_now'][r['optimizer'].layout.q_indices]).max()),
              [(round(c['violation_m'] * 1e3, 1), c['accepted'], round(c['trust_radius_rad'], 2)) for c in m['candidates']], flush=True)
