"""T41b: from the bit-exact C1807 t=216 state (after the 3-chunk HOLD), execute an 8-step plan and measure
how far the executed hand leaves the plan (row k+1 measured vs plan target k, left-hand bodies) and contact.
Plans: recorded refined (what ran), replayed off, replayed rest_start on (default e0 = a_max*dt^2)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import copy, json, sys, pathlib, numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/sim')); sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T41b/p'))
import _hold_replay as H
from _gjk import make_support, gjk_dist
import replay_c1807 as R
T0 = 216; SEQ = 28
BODIES = ('EE_BODY_L', 'ee_finger_l1', 'ee_finger_l2')

def hand_geoms(m, ids):
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in ids and (m.geom_contype[g] or m.geom_conaffinity[g])]

def true_dist(m, d, geoms, bar):
    sb = make_support(m, d, bar)
    return round(min(gjk_dist(make_support(m, d, g), sb, d.geom_xpos[g] - d.geom_xpos[bar]) for g in geoms) * 1e3, 2)

def execute(client, local, chunk, n=8):
    m, d0, act = local['m'], local['d'], local['act']
    d = mujoco.MjData(m); mujoco.mj_copyData(d, m, d0)
    mgr = local.get('obstacle_manager')
    if mgr is not None:
        mgr = copy.copy(mgr); mgr.data = d
        mgr.robot_collision = mgr.payload_collision = False
        mgr.contact_pairs, mgr.contact_steps, mgr.min_robot_clearance = set(), 0, float('inf')
    ids = [H._body(m, b) for b in BODIES]
    scratch = mujoco.MjData(m)
    q_start = d.qpos.copy()
    dev, clear, first = [], [], None
    plan_true, exec_true = [], []
    bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'pick_obstacle_hurdle_0_collision_bar')
    exec_true.append(true_dist(m, d, hand_geoms(m, ids), bar))  # q_now
    for k in range(n):
        client.apply_action('rby1_16d', chunk[k], d, local['idx'], act)
        row_clear = np.inf
        for s in range(local['steps_per_action']):
            mujoco.mj_step(m, d)
            if mgr is not None:
                mgr.observe_contacts()
                if mgr.robot_collision and first is None: first = (k, s)
                if s % 5 == 0: row_clear = min(row_clear, float(mgr.robot_clearance()))
        # plan pose k: arm targets of row k on the current qpos (other joints as measured)
        mujoco.mj_copyData(scratch, m, d)  # mocap (obstacle) poses too
        for i in range(H.N):
            scratch.qpos[local['idx']['left_q'][i]] = chunk[k][i]
            scratch.qpos[local['idx']['right_q'][i]] = chunk[k][H.N + 1 + i]
        mujoco.mj_kinematics(m, scratch); mujoco.mj_kinematics(m, d)
        dev.append(float(max(np.linalg.norm(d.xpos[b] - scratch.xpos[b]) for b in ids) * 1e3))
        plan_true.append(true_dist(m, scratch, hand_geoms(m, ids), bar))
        exec_true.append(true_dist(m, d, hand_geoms(m, ids), bar))
        clear.append(row_clear * 1e3 if np.isfinite(row_clear) else None)
    return dict(true_hand_to_bar_mm_at_plan_targets=plan_true, true_hand_to_bar_mm_executed_q_now_then_rows=exec_true,
                hand_dev_from_plan_mm_per_row=[round(v, 2) for v in dev], max_hand_dev_mm=round(max(dev), 2),
                robot_clearance_min_mm_per_row=[None if v is None else round(v, 2) for v in clear],
                contact=bool(mgr.robot_collision) if mgr is not None else None, first_contact_row_substep=first,
                contact_pairs=None if mgr is None else sorted(mgr.contact_pairs))

if __name__ == '__main__':
    sh = R.setup()
    rec = np.asarray(np.load(R.CH / f'chunk_{SEQ + R.OFFSET:05d}.npz')['refined_chunk'], np.float64)
    plans = {'recorded_refined': (rec, None)}
    for mg in (0.0, 0.010):
        for name, lim, rest in (('off', None, False), ('rest_start', {'rest_start': True}, True)):
            r = R.replay(sh, SEQ, limits=lim, at_rest=rest, obstacle_margin=mg)
            m = r['result'].metrics
            plans[f'replay_{name}_margin{int(mg*1000)}mm'] = (r['result'].chunk, dict(
                status=r['result'].status.value, step0_deg=round(r['step0_deg'], 3), TO_clearance_mm=round(m['clearance_m'] * 1e3, 3)))
    client = H.load_client()
    run = H.load_run(R.RUN)
    local = H.reset(client, run)
    H.replay_to(client, local, run['control'], T0)
    out = {}
    for name, (ch, info) in plans.items():
        out[name] = dict(to=info, **execute(client, local, ch))
        o = out[name]
        print(name, info, 'plan_true', o['true_hand_to_bar_mm_at_plan_targets'], 'exec_true', o['true_hand_to_bar_mm_executed_q_now_then_rows'],
              'dev', o['hand_dev_from_plan_mm_per_row'], 'contact', o['contact'], o['first_contact_row_substep'], flush=True)
    json.dump(out, open(_ag3s_workspace_path('outputs/impl/T41b/sim_track_c1807_t216.json'), 'w'), indent=1, default=str)
