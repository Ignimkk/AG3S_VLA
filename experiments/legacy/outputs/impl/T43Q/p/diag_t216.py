from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
name, seq, om = 'T40C1807', 28, 0.010
SERVO = {'limits': {'servo_model': True}}
off = R.replay(name, seq, obstacle_margin=om)
c = R.check_plan(name, seq, off['chunk'], overrides=SERVO, obstacle_margin=om)
print('escape plan (off) waypoint %.2f path %.2f' % (c['waypoint_clearance_m']*1e3, c['path_clearance_m']*1e3), c['path']['worst'], 'floor rows', c['path']['floor_rows'], c['path']['floor_min_m'])
print('servo state', c['servo_state'].record())
opt = c['optimizer']; st = c['servo_state']
traj = c['trajectory']
qi = opt.layout.q_indices
X = opt.servo.rollout(c['q_now'][qi], st.v, traj, st.tau)['q']
print('plan step0 offset deg', np.round(np.degrees(np.abs(traj[:, 0] - c['q_now'][qi])), 2))
print('X row-end offset deg (row0..3)', [np.round(np.degrees(np.abs(X[:, k] - c['q_now'][qi])).max(), 2) for k in range(4)])
# per-step path clearance profile
cfgs = opt.path.configs(traj)
cen, _ = opt.path._states(cfgs.full, cfgs.params, with_jac=False)
flat = opt.lin.flat_clearances(cen, c['scene']) if hasattr(opt, 'lin') else opt.linearizer.flat_clearances(cen, c['scene'])
print('per config min (h - floor) mm', np.round((flat - opt.path._floor).min(axis=1) * 1e3, 2))
print('per config min h mm', np.round(flat.min(axis=1) * 1e3, 2))
h0 = opt.path._floor
on = R.replay(name, seq, overrides=SERVO, obstacle_margin=om)
m = on['metrics']
print('ON', on['status'], m['clearance_m'], m['path_check']['worst'], m['notes'] if 'notes' in m else on['result'].notes)
print('cands', [(c_['merit'], c_['violation_m'], c_['accepted']) for c_ in m['candidates']], 'initial', m['initial_merit'], m['initial_violation_m'])
