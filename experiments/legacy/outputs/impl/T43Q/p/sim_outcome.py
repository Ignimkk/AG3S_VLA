"""T43 Q: execute plans in MuJoCo from the bit-exact recorded state (client apply_action, 33 sim steps/row) and
measure the hand's true-mesh distance to the obstacles (GJK) + the client's contact flag.
Cases: T40 G C1807 seq 28 (t=216; recorded contact t=218) at obstacle margin 0 (as recorded) and 10 mm (T41 chain);
T41 a C1807 seq 35 (t=272; recorded contact t=276, margin 10 mm).
Plans: recorded refined (what ran), replay off, servo, sweep, both. Also the path check of each plan.
Writes outputs/impl/T43Q/sim_outcome.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt')); sys.path.insert(0, _ag3s_workspace_path('tests/sim'))
import _t43q_replay as R
import _t43q_exec as E
MODES = {'servo': {'limits': {'servo_model': True}}, 'sweep': {'reduction': {'sweep_check': True}},
         'both': {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}}}
CASES = [('T40C1807', 28, 0.0), ('T40C1807', 28, 0.010), ('T41aC1807', 35, None)]
only = sys.argv[1:] or None
out = {}
for name, seq, om in CASES:
    key = f'{name}_seq{seq}_margin{int(round((om if om is not None else R.RUNS[name][3]) * 1000))}mm'
    if only and key not in only:
        continue
    t0 = int(R.npz(name, seq)['t_step']) and int(R._setup(name)['rows'][seq]['t_step'])
    plans = {'recorded': (np.asarray(R.npz(name, seq)['refined_chunk'], np.float64), None)}
    for mode, ov in [('off', None)] + list(MODES.items()):
        r = R.replay(name, seq, overrides=ov, obstacle_margin=om)
        m = r['metrics']
        plans[mode] = (r['chunk'], {'status': r['status'], 'waypoint_clearance_mm': round(r['clearance_m'] * 1e3, 3),
                                    'path_clearance_mm': None if 'path_check' not in m else round(m['path_check']['clearance_m'] * 1e3, 3),
                                    'initial_source': (m.get('initial_source') or {}).get('source'),
                                    'solve_ms': round(r['result'].solve_time_ms, 1)})
    checks = {}
    for pname, (chunk, _) in plans.items():
        checks[pname] = {}
        for mode, ov in MODES.items():
            c = R.check_plan(name, seq, chunk, overrides=ov, obstacle_margin=om)
            checks[pname][mode] = {'path_clearance_mm': round(c['path_clearance_m'] * 1e3, 3),
                                   'waypoint_clearance_mm': round(c['waypoint_clearance_m'] * 1e3, 3),
                                   'worst': c['path']['worst'], 'max_sweep_mm': c['path']['max_sweep_mm'],
                                   'pieces_per_step': c['path']['pieces_per_step']}
    client, local, run = E.load(R.RUNS[name][1], t0)
    res = {}
    for pname, (chunk, info) in plans.items():
        ex = E.execute(client, local, chunk, every=3)
        ex.pop('joint_trace')
        res[pname] = dict(to=info, path_checks=checks[pname], sim=ex)
        print(key, pname, info, 'contact', ex['contact'], ex['first_contact_row_substep'], 'true min', ex['true_mm_min'], 'rows', ex['true_mm_min_per_row'],
              'path', {k: v['path_clearance_mm'] for k, v in checks[pname].items()}, flush=True)
    out[key] = {'t0': t0, 'plans': res}
p = pathlib.Path(_ag3s_workspace_path('outputs/impl/T43Q/sim_outcome.json'))
old = json.loads(p.read_text()) if p.exists() else {}
old.update(out)
p.write_text(json.dumps(old, indent=1, default=str))
