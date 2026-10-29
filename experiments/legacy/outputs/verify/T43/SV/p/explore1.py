from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib, mujoco
T=pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
XML='${AG3S_PROJECT_ROOT}-t43v8/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
def bid(n): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
def fk_series(rd, side):
    rows=[json.loads(l) for l in open(rd/'frames.jsonl') if l.strip()]
    C=sorted([r for r in rows if r['kind']=='control'], key=lambda r:r['t_step'])
    fl='l' if side=='left' else 'r'; b1,b2=bid(f'ee_finger_{fl}1'),bid(f'ee_finger_{fl}2'); ba=bid('apple')
    site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'{side}_ee')
    tcp=[];fm=[];ap=[]
    for r in C:
        d.qpos[:]=r['qpos']; mujoco.mj_kinematics(m,d)
        tcp.append(d.site_xpos[site].copy()); fm.append(0.5*(d.xpos[b1]+d.xpos[b2])); ap.append(d.xpos[ba].copy())
    return np.array(tcp),np.array(fm),np.array(ap)
def run(A9,A8,a9,a8):
  for k9,r9 in sorted(A9.items()):
    if not k9.startswith(a9+'/'): continue
    k8=k9.replace(a9,a8); r8=A8.get(k8)
    if r8 is None: continue
    side=r9['side']
    t9,f9,p9=fk_series(pathlib.Path(r9['run_dir']),side); t8,f8,p8=fk_series(pathlib.Path(r8['run_dir']),side)
    n=min(len(t9),len(t8))
    d9=np.linalg.norm(f9-p9,axis=1)*1e3; d8=np.linalg.norm(f8-p8,axis=1)*1e3
    dt=np.linalg.norm(t9[:n]-t8[:n],axis=1)*1e3
    fd10=int(np.argmax(dt>10)) if (dt>10).any() else None; fd30=int(np.argmax(dt>30)) if (dt>30).any() else None
    def gs(r):
        c=r.get('server_chunks') or []; fc=next((x['t'] for x in c if x['grasp_state']=='closing'),None); fa=next((x['t'] for x in c if x['attached']),None); return fc,fa
    c9=gs(r9); c8=gs(r8)
    print(k9.split('/')[1],k9.split('_',1)[1], 'G',int(r9['grasp']),int(r8['grasp']),'close/att',c9,c8,'div10',fd10,'div30',fd30,
          'dmin %.0f@%d'%(d9[:200].min(),d9[:200].argmin()),'| %.0f@%d'%(d8[:200].min(),d8[:200].argmin()),
          'd@close', None if c9[0] is None else round(d9[c9[0]]), None if c8[0] is None else round(d8[c8[0]]))
A9=json.load(open(T/'V9/analysis/analysis.json'))['runs']; A8=json.load(open(T/'V8/analysis/analysis.json'))['runs']
run(A9,A8,'V9','V8')
W=json.load(open(T/'W/analysis/analysis.json'))['runs']
print('---- V3 vs V4'); run(W,W,'V3','V4')
