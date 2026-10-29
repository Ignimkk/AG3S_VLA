"""T43 Q: servo parameters of the left/right arm from the compiled client model, and how well a decoupled
linear PD model predicts MuJoCo from a bit-exact replayed state (C1807 t=216 and a moving state)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/sim'))
import _hold_replay as H
client = H.load_client()
run = H.load_run(H.RUNS / 'C/E3b_ep1807_s18071')
local = H.reset(client, run)
m, d, act, idx = local['m'], local['d'], local['act'], local['idx']
print('integrator', m.opt.integrator, 'timestep', m.opt.timestep, 'spa', local['steps_per_action'])
la = act['left_a'][:7]; lq = idx['left_q'][:7]
for a in la:
    j = m.actuator_trnid[a, 0]; dof = m.jnt_dofadr[j]
    print(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a), 'gain', m.actuator_gainprm[a, :3], 'bias', m.actuator_biasprm[a, :3],
          'arm', m.dof_armature[dof], 'damp', m.dof_damping[dof], 'fl', m.dof_frictionloss[dof], 'qadr', m.jnt_qposadr[j], 'dof', dof)
print('idx left_q', lq)
H.replay_to(client, local, run['control'], 216)
M = np.zeros((m.nv, m.nv))
for i in range(m.nv):
    e = np.zeros(m.nv); e[i] = 1.0; out = np.zeros(m.nv); mujoco.mj_mulM(m, d, out, e); M[:, i] = out
dofs = [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in la]
print('M diag (link only + armature)', np.round(np.diag(M)[dofs], 3))
print('M offdiag max within arm', np.round(np.abs(M[np.ix_(dofs, dofs)] - np.diag(np.diag(M)[dofs])).max(), 3))
print('qvel at t216', np.round(d.qvel[dofs], 4))
print('qfrc_bias', np.round(d.qfrc_bias[dofs], 3), ' /kp rad', np.round(d.qfrc_bias[dofs] / m.actuator_gainprm[la, 0], 4))
print('ctrl - qpos', np.round(np.array([d.ctrl[a] for a in la]) - d.qpos[[m.jnt_qposadr[m.actuator_trnid[a,0]] for a in la]], 4))
