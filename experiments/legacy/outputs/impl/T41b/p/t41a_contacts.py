"""T41b: in the T41a (margin 10 mm) E3b runs, which chunk was executing at each robot contact, and was it the
first chunk after a full HOLD? step-0 offset from the client planning rows (refined actions vs capture qpos)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, numpy as np
V = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T41a.verify.json')))['numbers']
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T41/a/runs/E3b'))
out = {}
for run, c in V['contact'].items():
    t = (c.get('T41a') or {}).get('contact_t')
    grp, ep = run.split('_'); d = [p for p in (R / grp).glob(f'E3b_{ep}_s*') if p.is_dir()]
    if not d: continue
    rows = [json.loads(l) for l in open(d[0] / 'frames.jsonl') if l.strip()]
    man = json.loads((d[0] / 'manifest.json').read_text())
    plan = sorted((r for r in rows if r['kind'] == 'planning'), key=lambda r: r['t_step'])
    idx = None
    info = []
    for i, p in enumerate(plan):
        fb = p.get('exec_feedback') or {}
        after_hold = bool(fb.get('available')) and int(fb.get('n_exec') or 0) == 0
        g = p.get('gate'); g = g.get('action') if isinstance(g, dict) else g
        info.append(dict(seq=p['seq'], t=p['t_step'], gate=g, after_full_hold=after_hold))
    # step0: refined row 0 arm targets vs capture arm qpos (rby1_16d: cols 0..6 left, 8..14 right)
    out[run] = dict(contact_t=t, client_robot_collision=(c.get('T41a') or {}).get('client_robot_collision'),
                    n_after_full_hold=sum(x['after_full_hold'] for x in info),
                    n_after_full_hold_executed=sum(x['after_full_hold'] and x['gate'] == 'execute' for x in info))
    if t is not None:
        cur = [x for x in info if x['t'] <= t][-1]
        prev = [x for x in info if x['t'] < cur['t']]
        out[run].update(contact_chunk=cur, previous_chunk=prev[-1] if prev else None)
    print(run, out[run])
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T41b/t41a_contacts.json'), 'w'), indent=1)

# --- step-0 offsets of the contact chunks (needs the model: run with .venv-openpi-live)
import sys
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T41b/p'))
import replay_c1807 as RC
sh = RC.setup(); lay = sh['layout']; qadr = sh['qadr']; A = float(np.degrees(sh['limits'].max_step_change[0]))
for run, o in out.items():
    if o.get('contact_t') is None: continue
    grp, ep = run.split('_'); d = [p for p in (R / grp).glob(f'E3b_{ep}_s*') if p.is_dir()][0]
    rows = [json.loads(l) for l in open(d / 'frames.jsonl') if l.strip()]
    p = [r for r in rows if r['kind'] == 'planning' and r['seq'] == o['contact_chunk']['seq']][0]
    q = RC.T.q_now_from_qpos(p['qpos'], qadr)[lay.q_indices]
    tr = lay.chunk_to_trajectory(np.asarray(p['actions'], float)); rf = lay.chunk_to_trajectory(np.asarray(p['actions_reference'], float))
    o['contact_chunk']['refined_step0_offset_deg'] = round(float(np.degrees(np.abs(tr[:, 0] - q).max())), 3)
    o['contact_chunk']['reference_step0_offset_deg'] = round(float(np.degrees(np.abs(rf[:, 0] - q).max())), 3)
    o['contact_chunk']['over_a_dt2'] = bool(o['contact_chunk']['refined_step0_offset_deg'] > A)
    print(run, o['contact_chunk'])
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T41b/t41a_contacts.json'), 'w'), indent=1)
