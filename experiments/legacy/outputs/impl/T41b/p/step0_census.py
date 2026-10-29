"""T41b: step-0 offset |refined[0]-q_now| over all T40 G E3b chunks, split by whether the previous chunk's last step was held."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, glob, pathlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.trajopt.experiments import fixed_input_ablation as T
from AG3S.trajopt.limits import build_limits
from AG3S.trajopt.types import ChunkLayout
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b'))
A = json.load(open(_ag3s_workspace_path('outputs/verify/T40/G/analysis_all.json')))['runs']
setup = None
res = {'after_hold': [], 'after_exec': [], 'first': []}
per = []
for key, run in A.items():
    if not key.startswith('E3b'): continue
    seg = run['server_segment']; base = pathlib.Path(seg['first'].rsplit('/', 1)[0])
    rd = pathlib.Path(run.get('run_dir') or '')
    cand = list(G.glob(f'*/{key.split("/")[-1]}')) if not rd.exists() else [rd]
    if setup is None:
        srv = base.parent.parent
        args = T.build_parser().parse_args(['--constraints', str(base), '--frames', str(cand[0] / 'frames.jsonl'),
                                            '--server-log', str(srv / 'serve_safe.log'), '--out', '/dev/null'])
        st = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((cand[0] / 'manifest.json').read_text()))
        robot, qadr, _ = T.build_robot(st)
        layout = ChunkLayout.rby1(robot.joint_names)
        cfg = T.to_config(st, collision=True, cost=dict(st.cost), sqp={})
        lim = build_limits(robot, layout, dt=cfg.horizon.dt, config=cfg.limits)
        setup = 1
    rows = [json.loads(l) for l in open(cand[0] / 'frames.jsonl') if l.strip()]
    plan = {r['seq']: r for r in rows if r.get('kind') == 'planning'}
    i0 = int(seg['first'][-9:-4]); i1 = int(seg['last'][-9:-4])
    for i in range(i0, i1 + 1):
        z = np.load(base / f'chunk_{i:05d}.npz', allow_pickle=True)
        seq = int(z['t_step']); p = plan.get(seq)
        if p is None: continue
        s = json.loads(str(z['summary_json'])); fb = s.get('exec_feedback') or {}
        q = T.q_now_from_qpos(p['qpos'], qadr)[layout.q_indices]
        refi = layout.chunk_to_trajectory(np.asarray(z['refined_chunk'], float))
        ref = layout.chunk_to_trajectory(np.asarray(z['reference_chunk'], float))
        d = float(np.degrees(np.abs(refi[:, 0] - q).max())); dr = float(np.degrees(np.abs(ref[:, 0] - q).max()))
        sat = bool(np.any(np.abs(np.abs(refi[:, 0] - q) - lim.max_step) < 1e-4))
        if not fb.get('available'): k = 'first'
        elif not (fb.get('executed') or [True])[-1]: k = 'after_hold'
        else: k = 'after_exec'
        res[k].append((d, dr, sat))
        per.append(dict(run=key, seq=seq, cls=k, refined0_deg=round(d, 3), reference0_deg=round(dr, 3), at_anchor_bound=sat,
                        gate=(p.get('gate') or {}).get('action') if isinstance(p.get('gate'), dict) else None))
A_deg = float(np.degrees(lim.max_step_change[0]))
out = {'max_step_change_deg': A_deg, 'max_step_deg': np.degrees(lim.max_step).round(3).tolist()}
for k, v in res.items():
    if not v: continue
    a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v]); s = np.array([x[2] for x in v])
    out[k] = dict(n=len(v), refined0_deg_q50=round(float(np.median(a)), 3), refined0_deg_q90=round(float(np.quantile(a, .9)), 3),
                  refined0_deg_max=round(float(a.max()), 3), n_over_A=int((a > A_deg + 1e-6).sum()),
                  n_at_anchor_bound=int(s.sum()), reference0_deg_q50=round(float(np.median(b)), 3), reference0_deg_max=round(float(b.max()), 3))
    print(k, out[k])
out['chunks'] = per
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T41b/step0_census.json'), 'w'), indent=1)
