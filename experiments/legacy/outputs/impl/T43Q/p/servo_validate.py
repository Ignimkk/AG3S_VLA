"""T43 Q: the servo model (AG3S/trajopt/servo.py) against MuJoCo, from bit-exact replays of recorded runs.

For each run: replay every recorded control row (applied_ctrl, 33 sim steps), assert the recorded qpos, record
qpos at every sim sub-step and qvel / qfrc_bias at every row start. Then, from each planning capture, predict the
next 8 rows with the model and compare (left+right arm, 14 joints):
  oracle   : v0 = true qvel, tau = true qfrc_bias at the capture (held constant)
  observer : ServoObserver fed only what the server gets (capture qpos + applied rows of the previous chunk)
  rest0    : v0 = 0, tau = 0 (what a model without an observer would assume)
Writes outputs/impl/T43Q/servo_validate.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/sim'))
import _hold_replay as H
from AG3S.trajopt.servo import ServoModel, ServoObserver, ServoParams

RUNS = sys.argv[1:] or ['C/E3b_ep1807_s18071', 'B/E3b_ep1968_s19681']
JOINTS = [f'left_arm_{i}' for i in range(7)] + [f'right_arm_{i}' for i in range(7)]
out = {}
client = H.load_client()
for rname in RUNS:
    run = H.load_run(H.RUNS / rname)
    local = H.reset(client, run)
    m, d = local['m'], local['d']
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
    qa = np.array([m.jnt_qposadr[j] for j in jid]); da = np.array([m.jnt_dofadr[j] for j in jid])
    params = ServoParams.from_mj_model(m, JOINTS, control_hz=client.CTRL_HZ)
    params_here = None
    S = local['steps_per_action']
    ctl = run['control']
    T = len(ctl)
    sub_q = np.zeros((T, S, 14)); start_q = np.zeros((T, 14)); start_v = np.zeros((T, 14)); start_b = np.zeros((T, 14))
    U = np.zeros((T, 14)); diverged = None
    for t, row in enumerate(ctl):
        if diverged is None and not np.array_equal(np.asarray(row['qpos']), d.qpos):
            diverged = t
            break
        start_q[t] = d.qpos[qa]; start_v[t] = d.qvel[da]; start_b[t] = d.qfrc_bias[da]
        if t == 216 and params_here is None:
            params_here = ServoParams.from_mj_model(m, JOINTS, control_hz=client.CTRL_HZ, qpos=d.qpos.copy())
        H.apply_row(client, local, row)
        U[t] = d.ctrl[[local['act']['left_a'][i] for i in range(7)] + [local['act']['right_a'][i] for i in range(7)]]
        for s in range(S):
            mujoco.mj_step(m, d)
            sub_q[t, s] = d.qpos[qa]
    T = diverged or T
    model = ServoModel(params)
    obs = ServoObserver(model)
    caps = list(range(0, T - 8, 8))
    res = {k: [] for k in ('oracle', 'observer', 'rest0')}
    res_rows = {k: [] for k in res}
    v_err, tau_est = [], []
    prev = None
    for c in caps:
        fb = None if prev is None else {'available': True, 'seq': prev, 'applied_arm': U[c - 8:c]}
        st = obs.update(start_q[c], fb, c)
        if prev is not None:
            v_err.append(float(np.abs(st.v - start_v[c]).max()))
            tau_est.append((st.tau, start_b[c]))
        prev = c
        cmds = U[c:c + 8].T
        truth = sub_q[c:c + 8]  # (8, S, 14)
        for name, v0, tau in (('oracle', start_v[c], -start_b[c]), ('observer', st.v, st.tau), ('rest0', np.zeros(14), np.zeros(14))):
            # note: model tau is the torque subtracted: I a = kp(u-q) - ... - tau  -> tau = qfrc_bias
            tau_m = start_b[c] if name == 'oracle' else tau
            pred = model.rollout(start_q[c], v0, cmds, tau_m, samples=list(range(1, S + 1)))['q_samples']  # (14, 8, S)
            err = np.abs(np.transpose(pred, (1, 2, 0)) - truth)  # (8, S, 14)
            res[name].append(float(err.max())); res_rows[name].append(err.max(axis=(1, 2)).tolist())
    r = {k: {'n': len(v), 'max_err_deg_q50': float(np.degrees(np.median(v))), 'max_err_deg_q90': float(np.degrees(np.quantile(v, 0.9))),
             'max_err_deg_max': float(np.degrees(np.max(v))),
             'per_row_err_deg_q90': [float(np.degrees(x)) for x in np.quantile(np.asarray(res_rows[k]), 0.9, axis=0)]} for k, v in res.items()}
    r['observer_v_err_rad_s'] = {'q50': float(np.median(v_err)), 'q90': float(np.quantile(v_err, 0.9)), 'max': float(np.max(v_err))}
    te = np.asarray([np.abs(a - b).max() for a, b in tau_est])
    r['observer_tau_minus_bias_nm'] = {'q50': float(np.median(te)), 'q90': float(np.quantile(te, 0.9)), 'max': float(te.max())}
    r['params_qpos0'] = params.summary(); r['params_t216'] = None if params_here is None else params_here.summary()
    r['replayed_rows'] = int(T); r['diverged_at'] = diverged
    r['max_true_speed_rad_s'] = float(np.abs(start_v[:T]).max())
    out[rname] = r
    print(rname, json.dumps({k: r[k] for k in ('oracle', 'observer', 'rest0', 'observer_v_err_rad_s', 'observer_tau_minus_bias_nm', 'replayed_rows')}, indent=0), flush=True)
pathlib.Path(_ag3s_workspace_path('outputs/impl/T43Q')).mkdir(parents=True, exist_ok=True)
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T43Q/servo_validate.json'), 'w'), indent=1)
