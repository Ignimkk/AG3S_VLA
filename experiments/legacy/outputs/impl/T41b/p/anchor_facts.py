"""T41b: what bounds plan step 0? per-joint |refined[0]-q_now| vs the anchor max_step, C1807 seq 24-28."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.trajopt.experiments import fixed_input_ablation as T
from AG3S.trajopt.limits import build_limits
from AG3S.trajopt.types import ChunkLayout
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b'))
SRV = G / 'server_8240_20261003T135313'
CH = SRV / 'server_constraints/run_0000'
run = G / 'C/E3b_ep1807_s18071'
args = T.build_parser().parse_args(['--constraints', str(CH), '--frames', str(run / 'frames.jsonl'),
                                    '--server-log', str(SRV / 'serve_safe.log'), '--out', '/dev/null'])
st = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((run / 'manifest.json').read_text()))
robot, qadr, _ = T.build_robot(st)
layout = ChunkLayout.rby1(robot.joint_names)
cfg = T.to_config(st, collision=True, cost=dict(st.cost), sqp={})
lim = build_limits(robot, layout, dt=cfg.horizon.dt, config=cfg.limits)
print('dt', cfg.horizon.dt, 'max_step deg', np.round(np.degrees(lim.max_step), 3))
rows = [json.loads(l) for l in open(run / 'frames.jsonl') if l.strip()]
plan = {r['seq']: r for r in rows if r.get('kind') == 'planning'}
ctrl = [r for r in rows if r.get('kind') == 'control']
out = {}
for seq in range(20, 29):
    z = np.load(CH / f'chunk_{102 + seq:05d}.npz', allow_pickle=True)
    assert int(z['t_step']) == seq
    q_now = T.q_now_from_qpos(plan[seq]['qpos'], qadr)
    qi = layout.q_indices
    ref = layout.chunk_to_trajectory(np.asarray(z['reference_chunk'], float))
    refi = layout.chunk_to_trajectory(np.asarray(z['refined_chunk'], float))
    d_ref = refi[:, 0] - q_now[qi]
    d_pol = ref[:, 0] - q_now[qi]
    sat = np.abs(np.abs(d_ref) - lim.max_step) < 1e-4
    s = json.loads(str(z['summary_json']))
    fb = s.get('exec_feedback', {})
    out[seq] = dict(t_step=plan[seq]['t_step'], prev_n_exec=fb.get('n_exec'), gate=plan[seq]['gate'].get('action') if isinstance(plan[seq].get('gate'), dict) else plan[seq].get('gate'),
                    refined0_minus_qnow_max_deg=round(float(np.degrees(np.abs(d_ref).max())), 3),
                    reference0_minus_qnow_max_deg=round(float(np.degrees(np.abs(d_pol).max())), 3),
                    n_joints_at_anchor_bound=int(sat.sum()), joints_at_bound=[robot.joint_names[i] if False else int(j) for j in np.where(sat)[0]],
                    argmax_joint=int(np.abs(d_ref).argmax()), max_step_deg_at_argmax=round(float(np.degrees(lim.max_step[np.abs(d_ref).argmax()])), 3))
    print(seq, out[seq])
json.dump(dict(max_step_deg=np.degrees(lim.max_step).round(4).tolist(), dt=cfg.horizon.dt, chunks=out),
          open(_ag3s_workspace_path('outputs/impl/T41b/anchor_facts.json'), 'w'), indent=1)

# --- servo response from rest (seq 28) and acceleration bound
print('max_step_change deg', np.round(np.degrees(lim.max_step_change), 3))
seq = 28
z = np.load(CH / f'chunk_{102 + seq:05d}.npz', allow_pickle=True)
qi = layout.q_indices
q_now = T.q_now_from_qpos(plan[seq]['qpos'], qadr)[qi]
refi = layout.chunk_to_trajectory(np.asarray(z['refined_chunk'], float))
cr = [r for r in ctrl if r['chunk_seq'] == seq] if 'chunk_seq' in ctrl[0] else []
print('n control rows seq28', len(cr), [r['t_step'] for r in cr])
nxt = [r for r in ctrl if r['t_step'] in (217, 218, 219)]
for r in cr + [x for x in nxt if x not in cr]:
    q = T.q_now_from_qpos(r['qpos'], qadr)[qi]
    k = r['t_step'] - 216
    print('row', r['t_step'], 'measured-q_now deg', np.round(np.degrees(q - q_now), 2)[[2, 3, 6]],
          'plan[k-1]-q_now', np.round(np.degrees(refi[:, k - 1] - q_now), 2)[[2, 3, 6]] if k >= 1 else '-')
print('plan steps - q_now (j2,j3,j6) deg:')
for k in range(8):
    print(k, np.round(np.degrees(refi[:, k] - q_now), 2)[[2, 3, 6]])
