"""T43 Q: TO solve time per chunk, off vs servo vs sweep vs both (fixed-input replay, recorded SQP iteration count,
no wall-clock budget so every mode runs the same iterations). Each (chunk, mode) solved REPS times interleaved, the
minimum kept (shared pod: the minimum is the least-contended sample). Solve time = TrajOptResult.solve_time_ms
(the whole `sqp.solve`, including the first linearize on the chunk's new field). Also the path-row parts.
Writes outputs/impl/T43Q/timing.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
MODES = {'off': None, 'servo': {'limits': {'servo_model': True}}, 'sweep': {'reduction': {'sweep_check': True}},
         'both': {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}}}
CHUNKS = [('T40C1807', s) for s in range(20, 29)] + [('T41aC1807', s) for s in range(28, 36)] + [('T40B1968', s) for s in range(10, 30, 2)]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
if len(sys.argv) > 2:
    MODES = {k: v for k, v in MODES.items() if k in sys.argv[2].split(',')}
OUT = sys.argv[3] if len(sys.argv) > 3 else _ag3s_workspace_path('outputs/impl/T43Q/timing.json')
# warm-up: one solve per mode (field code, CasADi maps, servo affine table)
for mode, ov in MODES.items():
    R.replay('T40C1807', 20, overrides=ov)
rows = []
for name, seq in CHUNKS:
    rec = {'chunk': f'{name}:{seq}'}
    for mode, ov in MODES.items():
        best = None
        for _ in range(REPS):
            r = R.replay(name, seq, overrides=ov)
            t = r['result'].solve_time_ms
            if best is None or t < best[0]:
                m = r['metrics']; p = m.get('path_check') or {}
                best = (t, dict(m.get('timing_ms') or {}), p.get('timing_ms'), m['sqp_iterations'], r['status'], p.get('n_configs'))
        rec[mode] = {'solve_ms': round(best[0], 1), 'timing_ms': best[1], 'path_timing_ms': best[2], 'iters': best[3], 'status': best[4], 'n_configs': best[5]}
    rows.append(rec); print(json.dumps(rec), flush=True)
summary = {}
for mode in MODES:
    v = np.array([r[mode]['solve_ms'] for r in rows])
    summary[mode] = {'median_ms': float(np.median(v)), 'p90_ms': float(np.quantile(v, 0.9)), 'max_ms': float(v.max()), 'n': len(v)}
    if mode != 'off':
        d = np.array([r[mode]['solve_ms'] - r['off']['solve_ms'] for r in rows])
        summary[mode]['added_median_ms'] = float(np.median(d)); summary[mode]['added_p90_ms'] = float(np.quantile(d, 0.9))
print(json.dumps(summary, indent=1))
json.dump({'rows': rows, 'summary': summary, 'reps': REPS, 'modes': list(MODES)}, open(OUT, 'w'), indent=1)
