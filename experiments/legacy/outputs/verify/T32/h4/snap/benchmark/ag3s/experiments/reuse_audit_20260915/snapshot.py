from common.workspace import workspace_path as _ag3s_workspace_path
import hashlib,json,subprocess,datetime
from pathlib import Path
ROOT=Path(_ag3s_workspace_path('')); OUT=ROOT/'AG3S/ag3s/docs/figures/reuse-audit-20260915'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
files={}
for prefix in ['AG3S/ag3s','AG3S/trajopt','tests/ag3s','tests/trajopt','third_party/curobo/curobo/_src/perception','third_party/curobo/curobo/_src/collision','third_party/curobo/curobo/_src/solver','third_party/curobo/curobo/_src/rollout']:
 for p in (ROOT/prefix).rglob('*'):
  if p.is_file() and p.suffix in ['.py','.yaml','.yml','.md'] and not any(s in str(p) for s in ['reuse_audit_20260915','reuse-audit-20260915','__pycache__']):files[str(p.relative_to(ROOT))]=sha(p)
for pat in ['run_0004/*','run_0005/*','attention_step1_run0004.npz','attention_step1_run0005.npz']:
 for p in ROOT.glob(pat):
  if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
repos={}
for name in ['benchmark','third_party/curobo','pi05_TO_hybrid']:
 def git(*args):return subprocess.run(['git','-C',str(ROOT/name),*args],capture_output=True,text=True).stdout
 repos[name]={'head':git('rev-parse','HEAD').strip(),'status':git('status','--porcelain'),'diff_stat':git('diff','--stat')}
 (OUT/(name+'-baseline.patch')).write_text(git('diff','--binary'))
(OUT/'baseline.json').write_text(json.dumps({'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repos':repos,'sha256':files},indent=2))
print('baseline files',len(files)); print({n:v['head'] for n,v in repos.items()})
