#!/usr/bin/env python3
"""Read-only audit of IFG records; writes paper-owned provenance/plots with --register.
Separate from the legacy T39 trajectory schema: IFG stores per-request NPZ files.
No model, simulator, or serving code is run or modified.
"""
import argparse, ast, hashlib, json, subprocess, zipfile
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path('/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba')
DUMP = Path('/mnt/dev/AG3S_VLA-storage/results/development/20261015T150325+0900_ddc7a00060c4_d857a08dcb63/phase_dump')
MAN = ROOT/'paper/experiments/manifests/ifg/EXP002_IFG_development.json'
OUT = ROOT/'paper/experiments/summaries/EXP002_IFG_audit.json'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(Path(p).read_text())
def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def shapes(p):
    out = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            with z.open(n) as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                out[n] = list(shape)
    return out

def audit():
    assets = set()
    def load(p):
        assets.add(p); return read(p)
    source = load(RUN/'manifest.json')
    lock=load(RUN/'LOCK_heldout.json')
    drift = [k for k,v in source['code_sha256'].items() if sha(ROOT/k)!=v]
    amendments=[]
    for k in drift:
        amendment=next((a for a in lock.get('amendments',[]) if a['file']==k and a['new_sha256']==sha(ROOT/k)),None)
        assert amendment is not None, f'Undocumented source drift: {k}'
        old=subprocess.check_output(['git','show','4f36ff51:'+k],cwd=ROOT)
        assert hashlib.sha256(old).hexdigest()==source['code_sha256'][k]
        def executable_tree(code):
            tree=ast.parse(code)
            if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
                tree.body=tree.body[1:]
            return ast.dump(tree,include_attributes=False)
        assert executable_tree(old)==executable_tree((ROOT/k).read_bytes()), f'Executable source changed: {k}'
        amendments.append(dict(file=k,old_sha256=source['code_sha256'][k],new_sha256=sha(ROOT/k),check='Only module docstring/comments differ; executable AST identical'))
    assets.update(ROOT/k for k in source['code_sha256'])
    assets.add(RUN/'split_manifest.jsonl')
    assets.add(RUN/'s0/provenance_checks.jsonl')
    valid = {x['frame_id']:x['valid'] for x in rows(RUN/'s0/provenance_checks.jsonl')}
    frames=[]; eps=set(); episode_metadata={}
    for p in sorted(DUMP.glob('*/done.json')):
        d=load(p); fid=f"{d['ds']}_{d['key']}_f{d['frame']:03d}"
        assert valid[fid]
        si,so=shapes(p.parent/'inputs.npz'), shapes(p.parent/'outputs.npz')
        assert si['raw_state.npy']==[16]
        assert so['actions.npy']==[50,16] and so['recorded_server_actions.npy'][-1]==16
        assets.add(p.parent/'outputs.npz')
        # Bind exact small raw state values without hashing large RGB payloads.
        with np.load(p.parent/'inputs.npz') as z: state_hash=hashlib.sha256(z['raw_state'].tobytes()).hexdigest()
        frames.append(dict(frame_id=fid,done=str(p),state_shape=si['raw_state.npy'],action_shape=so['actions.npy'],raw_state_sha256=state_hash))
        ep=f"{d['ds']}_{d['key']}"
        eps.add(ep)
        if ep not in episode_metadata:
            base=Path(source['inputs']['records'][d['ds']])
            hits=list(base.glob(f"runs/C1a/R/C1a_{d['key']}_s*/ag3s/run_0000/meta.json"))
            assert len(hits)==1
            meta=load(hits[0])
            assert meta['policy_model']=='rby1_randomized_pick_place_16d'
            episode_metadata[ep]=dict(source=str(hits[0]),policy_model=meta['policy_model'],obstacle_profile=meta['obstacle_profile'],prompt=meta['prompt'])
    assert len(frames)==180 and len(eps)==32
    r9=load(RUN/'rounds/R9/paired_summary.json')
    # Reaggregate each published arm directly from per-frame rows.
    dirs={'T54region:N':'baseline','T54region+rm:N':'baseline_rm','T54ball+rm:N':'baseline_rm',
          'top1-only-ns:N':'rounds/R5','top1-only-ns+rm:N':'rounds/R6'}
    episode_values={}; counts={}
    for arm,folder in dirs.items():
        acc=defaultdict(lambda:defaultdict(list)); ids=set()
        for p in sorted((RUN/folder/'rows').glob('*.jsonl')):
            assets.add(p)
            for row in rows(p):
                if row['arm_id']!=arm or row.get('threshold')!=.5 or row.get('level')!='voxel': continue
                assert row['valid'] and row['frame_id'] not in ids
                ids.add(row['frame_id'])
                for m in ('iou','precision','recall'):
                    if row[m] is not None: acc[row['episode_id']][m].append(row[m])
        assert len(ids)==180 and len(acc)==32, (arm,len(ids),len(acc))
        episode_values[arm]={e:{m:float(np.mean(v)) for m,v in d.items()} for e,d in acc.items()}
        counts[arm]={m:sum(len(v[m]) for v in acc.values()) for m in ('iou','precision','recall')}
        for m in ('iou','precision','recall'):
            val=np.mean([v[m] for v in episode_values[arm].values()])
            assert np.isclose(val,r9['arms'][arm][m],atol=1e-12), (arm,m,val)
    for c in r9['comparisons']:
        a,b=c['a'][0],c['b'][0]; m=c['metric']
        if a not in episode_values or b not in episode_values: continue
        es=sorted(eps); d=np.array([episode_values[a][e][m]-episode_values[b][e][m] for e in es])
        boot=d[np.random.default_rng(20261015).integers(0,32,(2000,32))].mean(1)
        assert np.isclose(d.mean(),c['diff']) and np.allclose(np.percentile(boot,[2.5,97.5]),c['ci95'])
    memory=load(RUN/'rounds/R10b/paired_summary.json')
    memory_raw=load(RUN/'rounds/R10/paired_summary.json')
    memory_arms=['R6','Mem-first:prefix_L8:d0.05']
    mem_acc={a:defaultdict(list) for a in memory_arms}; mem_ids={a:set() for a in memory_arms}
    for p in sorted((RUN/'rounds/R10b/rows').glob('*.jsonl')):
        assets.add(p)
        for row in rows(p):
            a=row['arm_id']
            if a not in mem_acc:continue
            assert row['frame_id'] not in mem_ids[a]
            mem_ids[a].add(row['frame_id'])
            assert row['iou'] is not None
            mem_acc[a][row['episode_id']].append(row['iou'])
    mem_ep={a:{e:float(np.mean(v)) for e,v in es.items()} for a,es in mem_acc.items()}
    assert all(len(mem_ids[a])==180 and len(mem_ep[a])==32 for a in memory_arms)
    for a in memory_arms: assert np.isclose(np.mean(list(mem_ep[a].values())),memory['arms'][a]['iou'])
    d=np.array([mem_ep[memory_arms[1]][e]-mem_ep['R6'][e] for e in sorted(eps)])
    boot=d[np.random.default_rng(20261015).integers(0,32,(2000,32))].mean(1)
    mc=memory['comparisons'][memory_arms[1]]['iou']
    assert np.isclose(d.mean(),mc['diff']) and np.allclose(np.percentile(boot,[2.5,97.5]),mc['ci95'])
    for name in ['FINAL_REPORT_KO.md','METHODOLOGY_EVOLUTION_KO.md']:
        assets.add(ROOT/'doc/active/reviews/vla_dump_20261015'/name)
    for p in (ROOT/'AG3S/ag3s/docs/figures/ifg/exp').glob('*'):
        if p.suffix in ('.json','.png'):assets.add(p)
    r7=load(RUN/'rounds/R7/paired_summary.json')
    rootcause=load(RUN/'rounds/R9_rootcause/rootcause_summary.json')
    r12=load(RUN/'rounds/R12/summary.json'); assets.add(RUN/'rounds/R12/features.jsonl')
    assert r12['n_frames']==180
    conf=rows(RUN/'rounds/R11b/conf.jsonl');assets.add(RUN/'rounds/R11b/conf.jsonl')
    r11=load(RUN/'rounds/R11b/summary.json'); load(RUN/'LOCK_heldout.json')
    cohort=[c for c in conf if c['target_voxels']>0 and c.get('q_mem_chain:prefix_L8') is not None and c.get('c_mass') is not None]
    wrong=np.array([(c['r6_iou'] or 0)<.1 for c in cohort]); matched={}
    for key in ['c_mass','q_mem_chain:prefix_L8']:
        sc=np.array([c[key] for c in cohort]); pos=-sc[wrong];neg=-sc[~wrong]
        auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
        matched[key]=dict(n=len(cohort),n_wrong=int(wrong.sum()),auroc=auc)
    assert matched['q_mem_chain:prefix_L8']['n']==147
    assert np.isclose(matched['q_mem_chain:prefix_L8']['auroc'],r11['q_mem_chain:prefix_L8']['auroc'])
    result=dict(experiment_id='EXP002',status='PASS',scope='OFFLINE_DEVELOPMENT_ONLY',n_frames=180,n_episodes=32,
        metric_valid_frames=counts,frames=frames,episode_metadata=episode_metadata,source_code_drift=[],verified_source_amendments=amendments,adopted_method=lock['adopted_method'],memory=memory,memory_raw=memory_raw,extent=r9,feature=r7['arms'],rootcause=rootcause,
        confidence_matched=matched,confidence_reported=r11['q_mem_chain:prefix_L8'],phase=r12,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sorted(assets)])
    return result

def plots(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    arms=a['extent']['arms'];fig,axs=plt.subplots(1,2,figsize=(7.0,2.35),layout='constrained')
    for x,ks in enumerate([['T54region:N','top1-only-ns:N'],['T54region+rm:N','top1-only-ns+rm:N']]):
        for j,k in enumerate(ks):
            v=arms[k]['iou']; axs[0].bar(x+(j-.5)*.33,v,width=.31,color=['#9aa6b1','#147d92'][j],label=['Radius + clustering','Attention + surface graph'][j] if x==0 else None)
            axs[0].text(x+(j-.5)*.33,v+.02,f'{v:.3f}',ha='center',fontsize=8)
    axs[0].set(xticks=[0,1],xticklabels=['Volume self-filter','Rendered surface mask'],ylim=(0,1.08),ylabel='Episode-mean 3D IoU',title='(a) Separate mask and extent effects')
    axs[0].legend(loc='lower right',fontsize=7,frameon=False)
    cats=['Geometry','Feature\nedges','Feature\nunary','Edges +\nunary','Feature\ngrowing']
    keys=['G-Q+top1','FGe-Q:prefix_L8+top1','FGu-Q:prefix_L8+top1','FG-Q:prefix_L8+top1','FG-R:prefix_L8+top1']
    vals=[a['feature'][k]['iou'] for k in keys]
    axs[1].bar(range(5),vals,color=['#147d92']+['#b4bdc6']*4,width=.65)
    for x,v in enumerate(vals):axs[1].text(x,v+.02,f'{v:.3f}',ha='center',fontsize=8)
    axs[1].set(xticks=range(5),xticklabels=cats,ylim=(0,1.08),title='(b) More feature fusion is not better')
    axs[1].tick_params(axis='x',labelsize=7)
    fig.savefig(ROOT/'paper/figures/ifg_results/results.pdf');plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');args=ap.parse_args()
    a=audit()
    if args.register:
        OUT.write_text(json.dumps(a,indent=2)+'\n')
        man=dict(schema_version='IFG_OFFLINE_1',experiment_id='EXP002',status='REGISTERED',
          evidence_scope='OFFLINE_DEVELOPMENT_GROUNDING_AND_SUPERVISED_PROBING',state_dimension=16,action_dimension=16,
          validator='paper/tools/analysis/audit_ifg_evidence.py',audit=str(OUT.relative_to(ROOT)),audit_sha256=sha(OUT),
          limitations=['All 32 episodes used for development; no held-out claims.','Simulated replay depth; 180 selected key frames, not continuous video.',
          'Surface FP is a proxy, not executed contact or permission-builder output.','Phase accuracy is pooled frames; paired differences are episode means.',
          'Confidence has only six wrong-choice events and a history-dependent 147-frame cohort.',
          'Option B is adopted; the switching effect is inconclusive and combined switching/gating/phase performance is unmeasured.',
          'Phase height is centroid z minus a floor-dominated support median, not table-relative height.','No new rollout, policy inference, or simulation is run in this audit.'])
        MAN.write_text(json.dumps(man,indent=2)+'\n');plots(a)
    else:
        man=read(MAN);assert man['status']=='REGISTERED' and sha(OUT)==man['audit_sha256']
        assert a==read(OUT), 'Source/aggregation drift: review before re-registering'
    print(json.dumps(dict(status='PASS',frames=a['n_frames'],episodes=a['n_episodes'],valid=a['metric_valid_frames'],matched_confidence=a['confidence_matched'])))
if __name__=='__main__':main()
