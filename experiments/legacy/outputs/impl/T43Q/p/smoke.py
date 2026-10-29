from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
MODES = {'servo': {'limits': {'servo_model': True}}, 'sweep': {'reduction': {'sweep_check': True}},
         'both': {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}}}
for name, seq, om in (('T40C1807', 28, 0.0), ('T40C1807', 28, 0.010), ('T41aC1807', 35, None)):
    rec = R.npz(name, seq)['refined_chunk']
    for mode, ov in MODES.items():
        c = R.check_plan(name, seq, rec, overrides=ov, obstacle_margin=om)
        p = c['path']
        print(name, seq, om, 'recorded plan', mode, 'waypoint mm %.2f' % (c['waypoint_clearance_m'] * 1e3), 'path mm %.2f' % (c['path_clearance_m'] * 1e3),
              'worst', p['worst'], 'pieces', p['pieces_per_step'], 'sweep', p['max_sweep_mm'], 'state', p.get('servo_state', {}).get('source'), flush=True)
    for mode, ov in [('off', None)] + list(MODES.items()):
        t = time.perf_counter()
        r = R.replay(name, seq, overrides=ov, obstacle_margin=om)
        dt = (time.perf_counter() - t) * 1e3
        m = r['metrics']; p = m.get('path_check') or {}
        print(name, seq, om, 'replay', mode, r['status'], 'wp mm %.2f' % (r['clearance_m'] * 1e3), 'path mm', None if not p else p['clearance_m'] and round(p['clearance_m'] * 1e3, 2),
              'solve ms %.1f' % r['result'].solve_time_ms, 'timing', m.get('timing_ms'), 'path timing', p.get('timing_ms'), 'iters', m['sqp_iterations'], flush=True)
