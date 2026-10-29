from common.workspace import workspace_path as _ag3s_workspace_path
import sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
BOTH = {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}}
for seq in (17, 18):
    off = R.replay('T40B1968', seq)
    on = R.replay('T40B1968', seq, overrides=BOTH)
    m = on['metrics']; p = m['path_check']
    print(seq, 'off', off['status'], off['clearance_m'], 'on', on['status'], on['clearance_m'], 'path', p['clearance_m'], p['worst'], 'floor rows', p['floor_rows'], p['floor_min_m'], p['servo_state']['source'], p['servo_state']['v_rad_s_max'])
    print('  start', m.get('initial_source'), [(round(c['violation_m']*1e3,2), c['accepted']) for c in m['candidates']])
    c = R.check_plan('T40B1968', seq, off['chunk'], overrides=BOTH)
    print('  off plan path', c['path_clearance_m'], c['path']['worst'])
    pair = off['metrics']['max_violation_pair']; print('  off worst', pair['link'], pair.get('obstacle'), pair.get('tier'), pair['clearance_m'])
