"""T41b: re-solve T40 G C1807 seq 24..28 with the recorded inputs, rest_start off / on (tolerance variants)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
from AG3S.trajopt.experiments import fixed_input_ablation as T
from AG3S.trajopt.limits import build_limits
from AG3S.trajopt.types import ChunkLayout
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b'))
SRV = G / 'server_8240_20261003T135313'; CH = SRV / 'server_constraints/run_0000'
RUN = G / 'C/E3b_ep1807_s18071'
OFFSET = 102  # chunk index = seq + 102 for this run

def setup():
    args = T.build_parser().parse_args(['--constraints', str(CH), '--frames', str(RUN / 'frames.jsonl'),
                                        '--server-log', str(SRV / 'serve_safe.log'), '--out', '/dev/null'])
    st = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((RUN / 'manifest.json').read_text()))
    robot, qadr, _ = T.build_robot(st)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(st, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    base = T.to_config(st, collision=True, cost=dict(st.cost), sqp={})
    limits = build_limits(robot, layout, dt=base.horizon.dt, config=base.limits)
    rows = {int(x['seq']): x for x in (json.loads(l) for l in (RUN / 'frames.jsonl').read_text().splitlines() if l.strip()) if x.get('kind') == 'planning'}
    return dict(T=T, st=st, robot=robot, qadr=qadr, layout=layout, outside=maxd if unk == 'free' else -maxd, rows=rows,
                meter=T.Meter(robot, layout, base.horizon.planned, st.hand, limits), fmap=FingerJointMap(robot), limits=limits)

def support_planes(z):
    cl = z['filtered_cloud'].astype(float); sup = cl[z['support_mask']]
    out = []
    for sel in (sup[:, 2] > 0.4, sup[:, 2] <= 0.4):
        p = sup[sel]
        if len(p) < 50: continue
        mu = p.mean(0); n = np.linalg.svd(p - mu, full_matrices=False)[2][2]; n = n * np.sign(n[2])
        out.append([*n, float(n @ mu)])
    return np.asarray(out)

def replay(sh, seq, *, limits=None, at_rest=None, obstacle_margin=0.0):
    T, st, robot, layout = sh['T'], sh['st'], sh['robot'], sh['layout']
    z = np.load(CH / f'chunk_{seq + OFFSET:05d}.npz', allow_pickle=False)
    assert int(z['t_step']) == seq
    s = json.loads(str(z['summary_json'])); tm = (s.get('to') or {}).get('metrics') or {}
    q_now = T.q_now_from_qpos(sh['rows'][seq]['qpos'], sh['qadr'])
    field = T.build_field(z, outside_distance=sh['outside'])
    tp = z['target_points'] if 'target_points' in z.files else None
    margin, mask, _, _, _ = T.contact_policy(st, robot.sphere_link_names, has_target=bool(s.get('has_target')) and tp is not None,
                                             has_free=bool(field.target_free_layers), attached=False)
    lin = sh['meter'].lin
    scene = T.make_scene(field, lin.robot_radii, st.esdf_margin, target_points=tp, margin=margin, mask=mask)
    if obstacle_margin > 0.0:
        import dataclasses
        scene = dataclasses.replace(scene, obstacle_margin=float(obstacle_margin), support_planes=support_planes(z))
    refc = np.asarray(z['reference_chunk'], np.float64)
    meas = (s.get('finger_joints') or {}).get('opening_norm') or {}
    path, _ = sh['fmap'].path(meas, {'left': refc[:, 7], 'right': refc[:, 15]}, lin.horizon, default=robot.param_vector())
    cfg = T.to_config(st, collision=True, cost=dict(st.cost),
                      sqp={'time_budget_ms': T.NO_CLOCK_MS, 'max_iterations': max(1, int(tm.get('sqp_iterations') or 1))})
    if limits:
        cfg = cfg.with_overrides({'limits': limits})
    runner = T.Instrumented(robot, layout, cfg, {'value': (scene, q_now, s.get('status') == 'ok')})
    runner.refiner.optimizer.linearizer.set_joint_parameter_path(path)
    fb = s.get('exec_feedback') or {}
    ctx = {'t_step': seq}
    cont = s.get('exec_continuity') or {}
    if fb.get('available') and int(fb.get('n_exec') or 0) > 0:
        zp = np.load(CH / f'chunk_{seq + OFFSET - 1:05d}.npz', allow_pickle=False)
        ctx['previous_physical_chunk'] = np.asarray(zp['refined_chunk'], np.float64)
        ctx['previous_executed_steps'] = int(fb['n_exec'])
    elif fb.get('available'):
        ctx['previous_physical_chunk'] = None
    if at_rest:
        ctx['start_at_rest'] = True
    result, _ = runner.run(refc, ctx)
    qi = layout.q_indices
    return dict(result=result, q_now=q_now, z=z, summary=s, step0_deg=float(np.degrees(np.abs(result.trajectory[:, 0] - q_now[qi]).max())),
                recorded_step0_deg=float(np.degrees(np.abs(layout.chunk_to_trajectory(np.asarray(z['refined_chunk'], float))[:, 0] - q_now[qi]).max())),
                recorded_max_diff=float(np.abs(result.chunk - np.asarray(z['refined_chunk'], float)).max()))

if __name__ == '__main__':
    sh = setup()
    out = {}
    for seq in (24, 25, 26, 27, 28):
        rec = {}
        for name, lim, rest in (('off', None, False), ('on_default', {'rest_start': True}, True),
                                ('on_tol0', {'rest_start': True, 'rest_start_tolerance': 0.0}, True)):
            r = replay(sh, seq, limits=lim, at_rest=rest)
            m = r['result'].metrics
            rec[name] = dict(status=r['result'].status.value, step0_deg=round(r['step0_deg'], 3), clearance_mm=round(m['clearance_m'] * 1e3, 3),
                             returned=m.get('returned'), recorded_step0_deg=round(r['recorded_step0_deg'], 3),
                             max_abs_diff_vs_recorded=r['recorded_max_diff'], rest_start=m.get('rest_start'),
                             worst=(m.get('max_violation_pair') or {}).get('link'))
            print(seq, name, {k: v for k, v in rec[name].items() if k != 'rest_start'}, flush=True)
        out[seq] = rec
    json.dump(out, open(_ag3s_workspace_path('outputs/impl/T41b/replay_c1807.json'), 'w'), indent=1, default=float)
