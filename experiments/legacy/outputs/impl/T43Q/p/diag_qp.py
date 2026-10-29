from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
import AG3S.trajopt.sqp as SQ
name, seq, om = 'T40C1807', 28, 0.010
SERVO = {'limits': {'servo_model': True}}
orig = SQ.TrajectoryOptimizer._append_path
log = []
def spy(self, problem, iterate, scene):
    out = orig(self, problem, iterate, scene)
    rows = self.path.linearize(iterate, scene)
    log.append((iterate.copy(), rows))
    return out
SQ.TrajectoryOptimizer._append_path = spy
orig_solve = SQ.QpSolver.solve
sols = []
def spy2(self, problem, warm_start=True):
    s = orig_solve(self, problem, warm_start=warm_start); sols.append((problem, s)); return s
SQ.QpSolver.solve = spy2
on = R.replay(name, seq, overrides=SERVO, obstacle_margin=om)
opt = on['optimizer']; scene = on['scene']
for (it, rows), (prob, sol) in zip(log, sols):
    Qc, sl = prob.split(sol.x)
    pred = rows.value + np.einsum('krin,ni->kr', rows.gradient, Qc - it)
    true_w, _ = opt.path.violation(Qc, scene)
    pv = np.where(rows.used, pred, np.inf)
    print('status', sol.status, 'min linearized path (pred) mm %.2f' % (pv.min()*1e3), 'true path worst mm %.2f' % (true_w*1e3),
          'path slack sum', sl[opt.block.n_rows:].sum(), 'main slack', sl[:opt.block.n_rows].sum(), 'max dQ', np.abs(Qc-it).max(), 'used rows per step', rows.used.sum(1))
    print('  per-step min value (it) mm', np.round(np.where(rows.used, rows.value, np.inf).min(1)*1e3, 2), 'pred', np.round(pv.min(1)*1e3, 2))
    print('  grad norms max per step', np.round(np.abs(rows.gradient).sum(axis=(2,3)).max(1), 3))
