from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json; sys.path.insert(0,_ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import numpy as np, fkmodel as F, field as FD
sc,qm=F.model('gripper')
a=json.load(open(_ag3s_workspace_path('outputs/verify/T44/CL/analysis/C1b_ep12_s122.json')))
rows=[json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T44/CL/runs/C1b/R/C1b_ep12_s122/frames.jsonl'))]
P={r['seq']:r for r in rows if r['kind']=='planning'}
for seq in (48,49,52,16):
  c=[x for x in a['server_chunks'] if x['seq']==seq][0]; z=np.load(c['server_file'],allow_pickle=True); s=json.loads(str(z['summary_json']))
  mvp=eval(s['to']['metrics']['max_violation_pair']); qpos=P[seq]['qpos']; pr=F.params_from_qpos(sc,qm,qpos)
  Q=F.chunk_q(sc,qm,qpos,np.asarray(z['refined_chunk'],np.float64)[:16]); S=F.spheres(qm,Q[mvp['step']],pr)
  i=mvp['query']; mask=np.asarray(s['target_permission']['target_field_exclude'],bool)
  r=FD.query(z,S,mask)
  print(seq, mvp['link'], mvp['tier'], 'rec clr',round(mvp['clearance_m']*1e3,2),'pt',np.round(S[i],4), mvp['point_m'], 'on',round((r['d_on'][i]-qm.radii[i])*1e3,2),r['tier_on'][i],'off',round((r['d_off'][i]-qm.radii[i])*1e3,2),r['tier_off'][i],'main',round((r['d_main'][i]-qm.radii[i])*1e3,2))
