from common.workspace import workspace_path as _ag3s_workspace_path
import sys; sys.path.insert(0,_ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import fkmodel as F, numpy as np, json, inspect
sc,qm=F.model('gripper')
print(qm.n_spheres, qm.joint_names, getattr(qm,'param_joint_names',None))
print(inspect.signature(qm.sphere_centers_numeric))
a=json.load(open(_ag3s_workspace_path('outputs/verify/T44/CL/analysis/C1b_ep12_s122.json')))
rows=[json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T44/CL/runs/C1b/R/C1b_ep12_s122/frames.jsonl'))]
P={r['seq']:r for r in rows if r['kind']=='planning'}
c=[x for x in a['server_chunks'] if x['seq']==48][0]; z=np.load(c['server_file'],allow_pickle=True)
qpos=P[48]['qpos']; q=F.q_from_qpos(sc,qm,qpos); pr=F.params_from_qpos(sc,qm,qpos)
s=F.spheres(qm,q,pr); print(s.shape, np.abs(s-z['sphere_centres']).max())
ref=z['reference_chunk']; Q=F.chunk_q(sc,qm,qpos,ref); s0=F.spheres(qm,Q[0],pr); print('row0', np.abs(s0-z['sphere_centres']).max())
