#!/usr/bin/env python3
"""Inspect graph internals for deterministic GT-seeded no-region cases; no tuning."""
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from Baseline.paper_baselines.eval.offline_grounding import Episode,load_episodes
from AG3S.ag3s.runtime.attention_surface import SurfaceView,select_target_surface,CAMERA_ORDER
from AG3S.ag3s.experiments.studies import ifg_graph as G


def main():
    ap=argparse.ArgumentParser()
    for arg in ('source','scene','phase','candidates','out'):ap.add_argument('--'+arg,type=Path,required=True)
    ap.add_argument('--per-task',type=int,default=3)
    a=ap.parse_args()
    metadata={e['stem']:e for e in load_episodes(a.source)}
    rows=[json.loads(x) for x in (a.candidates/'frames.jsonl').open()]
    selected=[];counts=Counter()
    for r in rows:
        if r['why']!='no_region' or counts[r['task_id']]>=a.per_task:continue
        counts[r['task_id']]+=1;selected.append(r)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'selection.json').write_text(json.dumps({'rule':f'first {a.per_task} no_region requests per task',
        'source_results':str(a.candidates.resolve()),'selected':[(r['episode_id'],r['obs_index']) for r in selected],
        'truth':'image-time target for diagnosis and privileged seed; return-time GT robot mask','n':len(selected)},indent=2)+'\n')
    with (a.out/'rows.jsonl').open('x') as f:
        for r in selected:
            stem,t=r['episode_id'],r['obs_index'];ep=Episode(metadata[stem],scene_dir=a.scene);fr=ep.frame(t)
            with np.load(a.phase/f'phase_{stem}.npz') as z:
                k=int(np.flatnonzero(z['call_frames']==t)[0]); gt={c:np.isin(z[f'segP/{c}'][k],ep.target_geoms) for c in ('head','left_wrist')}
            views=[];classes=[]
            for c in ('head','left_wrist'):
                agrid=gt[c].reshape(16,16,16,16).mean(axis=(1,3)).astype(np.float32)
                views.append(SurfaceView(c,fr['depth_m'][c],fr['K'][c],fr['T'][c],fr['robot'][c],agrid))
                classes.append(gt[c].astype(np.int64))
            dummy=np.zeros_like(fr['depth_m']['head']);dummy[0,0]=1
            views.append(SurfaceView(CAMERA_ORDER[2],dummy,fr['K']['head'],fr['T']['head'],np.ones_like(fr['robot']['head']),np.zeros((16,16))))
            classes.append(np.zeros_like(fr['robot']['head'],np.int64))
            out=select_target_surface(views,class_map=classes,classes=('other','target'),keep_debug=True,arm='diagnostic')
            if out.record.get('why')!='no_region':raise ValueError(f'not reproducible: {stem}/{t}: {out.record}')
            sub,seeds,m,mt,info=out.solve;geo=out.geometry
            # target fractions are used only for diagnosis; selector uses geometry and attention, not these labels.
            tgt=geo.eval_target[geo.node_eval]; tmask=tgt>=0.5
            support_target=int((geo.support & tmask).sum()); target_nodes=int(tmask.sum())
            finite=m[np.isfinite(m)]
            row={'episode_id':stem,'task_id':r['task_id'],'obs_index':t,'n_gt_target_nodes':target_nodes,
                 'target_nodes_support':support_target,'target_support_fraction':support_target/target_nodes if target_nodes else None,
                 'max_m':float(finite.max()) if len(finite) else None,'n_roi':len(sub.pos),
                 'seed_patches':list(map(int,seeds)),'record':out.record}
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    print('diagnosed',len(selected),'balanced no-region frames',flush=True)

if __name__=='__main__':main()
