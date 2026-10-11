"""T43 Q: obstacle-free (B1968) and C1807 chunks re-solved off vs on (servo+sweep): status, path-check verdict,
how often the stay start is used, executed-window deviation from the reference. Fixed-input replay, no clock."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/trajopt'))
import _t43q_replay as R
name = sys.argv[1]; seqs = range(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
BOTH = {'limits': {'servo_model': True}, 'reduction': {'sweep_check': True}}
rows = []
for seq in seqs:
    if seq not in R._setup(name)['rows']:
        continue
    try:
        off = R.replay(name, seq)
        on = R.replay(name, seq, overrides=BOTH)
    except AssertionError as e:
        print(seq, 'skip', e); continue
    ref = R._setup(name)['layout'].chunk_to_trajectory(np.asarray(R.npz(name, seq)['reference_chunk'], float))[:, :8]
    m = on['metrics']; p = m['path_check']
    row = dict(seq=seq, off=off['status'], on=on['status'], path_mm=round(p['clearance_m'] * 1e3, 2) if p['clearance_m'] is not None else None,
               start=(m.get('initial_source') or {}).get('source'),
               dev_off_deg=round(float(np.degrees(np.abs(off['trajectory'] - ref).max())), 2),
               dev_on_deg=round(float(np.degrees(np.abs(on['trajectory'] - ref).max())), 2),
               max_sweep_mm=p['max_sweep_mm'], pieces=sum(p['pieces_per_step']), ms_off=round(off['result'].solve_time_ms), ms_on=round(on['result'].solve_time_ms))
    rows.append(row); print(row, flush=True)
json.dump(rows, open(f'{_ag3s_workspace_path()}/outputs/impl/T43Q/census_{name}_{sys.argv[2]}_{sys.argv[3]}.json', 'w'), indent=1)
