"""Fixed single-frame prototype diagnostic. GT is used only by evaluators.
No threshold/layer selection, training, or production changes. Replay depth is synthetic.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def normalize(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dump',type=Path,required=True);ap.add_argument('--t53',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    out=args.out;out.mkdir(exist_ok=True,parents=True);d=args.dump/'frame_022'
    record=args.t53/'runs/C1a/R/C1a_ep1839_s18391/ag3s/run_0000'
    with np.load(d/'derived.npz') as z:a=z['serving_L8H2_last_d0'].astype(float);n=z['noun_target_L10H7'].astype(float)
    score=a*n;seeds=np.argsort(score)[-5:];sw=score[seeds]/score[seeds].sum()
    maps={'action_last':a,'noun_orange':n,'product':score}
    with np.load(d/'suffix_s00.npz') as z:maps['action_first8']=z['L_attn_probs'][8,2,:8,:768].astype(float).mean(0)
    # Fixed representations, all reported; no oracle seed or label-based negative seed.
    features={}
    cams=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']
    features['siglip_encoded']=np.concatenate([np.load(d/f'siglip_{c}.npz')['encoded'] for c in cams])
    with np.load(d/'embed.npz') as z:features['projector']=z['image_tokens']
    for l in [4,8,12]:
        with np.load(d/f'prefix_L{l:02d}.npz') as z:features[f'prefix_L{l}_out']=z['resid_out'][:768]
    for name,f in features.items():
        f=normalize(f);mu=normalize(sw@f[seeds]);maps[name]=f@mu
        for seed_name,seed_score in [('actionseed',a),('nounseed',n)]:
            seed_idx=np.argsort(seed_score)[-5:];seed_weight=seed_score[seed_idx]/seed_score[seed_idx].sum()
            maps[seed_name+'_'+name]=f@normalize(seed_weight@f[seed_idx])
    # Evaluation begins only after scores are constructed.
    from AG3S.ag3s.experiments.sources.policy_record import load_run,record_fingerprint
    run=load_run(record);step=run.steps[22];assert step.t_step==176
    with np.load(d/'inputs.npz') as z:assert np.allclose(z['raw_state'],step.state,atol=1e-6);images=[z['image_'+c] for c in cams]
    with np.load(args.t53/'gt/ep1839.npz') as z:
        assert str(z['source_record_fingerprint'])==record_fingerprint(run)
        idx=int(np.flatnonzero(z['t_step']==176)[0]);classes=z['classes'].tolist();cov=z['class_cov'][idx].astype(float);labels=z['label_small'][idx]
    flat=cov.transpose(0,2,3,1).reshape(768,-1);truth=flat[:,0]
    result=dict(scope='one diagnostic frame, not held-out evaluation; top-K patch masks, not tuned thresholds',seed_rule='top5 action_last*noun, weighted normalized feature mean',seed_indices=seeds.tolist(),seed_pixel_class_fraction=dict(zip(classes,(flat[seeds].mean(0)).tolist())),seed_score_weighted_class_fraction=dict(zip(classes,(sw@flat[seeds]).tolist())),source_record=str(record),record_fingerprint=record_fingerprint(run),t_step=176,open_loop_horizon=run.meta['open_loop_horizon'],methods={})
    def metrics(s,k):
        ix=np.argsort(s)[-k:];tp=truth[ix].sum()
        return dict(precision=float(tp/k),recall=float(tp/truth.sum()),iou=float(tp/(k+truth.sum()-tp)),class_fraction=dict(zip(classes,flat[ix].mean(0).tolist())))
    for name,s in maps.items():result['methods'][name]={str(k):metrics(s,k) for k in [5,10,20,40]}
    ix=np.argsort(-truth);tp=np.cumsum(truth[ix]);ks=np.arange(1,769);iou=tp/(ks+truth.sum()-tp);best=int(np.argmax(iou))
    result['oracle_whole_patch_mask_upper_bound']=dict(iou=float(iou[best]),patch_count=best+1,note='Uses GT to show 16x16 resolution ceiling only; not an algorithm result.')
    with np.load(args.t53/'gt/ep1839.grasp.npz') as z:
        assert str(z['source_record_fingerprint'])==record_fingerprint(run)
        result['existing_grasp_labels']={k:z[k][idx].tolist() for k in ['grasp_state','phys_state','finger_contact_lr','used_gripper','closed_loop_gt']}
    if args.replay:
        import mujoco
        from AG3S.ag3s.experiments.sources.policy_record import replay_scene,pose_scene,patch_coverage,CAMERA_BINDINGS
        from AG3S.ag3s.experiments.studies.t47_replay import body_class_table,class_map,render_policy_rgb,POLICY_SOURCE_HW
        scene=replay_scene(run);pose_scene(scene,step);lut,_=body_class_table(scene.model,'orange');rgb_renderer=mujoco.Renderer(scene.model,*POLICY_SOURCE_HW)
        replay=[]
        try:
            for ci,(pol,(mjcam,_,_)) in enumerate(CAMERA_BINDINGS.items()):
                frame=scene.capture(mjcam);cls=class_map(frame.body_ids,lut);cv,_=patch_coverage(cls,list(range(len(classes))))
                rgb=render_policy_rgb(scene.model,scene.data,rgb_renderer,mjcam);mad=float(np.abs(rgb.astype(float)-step.images[pol]).mean())
                dep=frame.depth;h,w=dep.shape;yy,xx=np.indices(dep.shape);K=frame.camera_intrinsics;T=frame.T_base_cam
                pc=np.stack([(xx-K[0,2])*dep/K[0,0],(yy-K[1,2])*dep/K[1,1],dep],axis=-1)@T[:3,:3].T+T[:3,3]
                patches=(yy*16//h)*16+xx*16//w;valid=(dep>.05)&(dep<3)&np.isfinite(dep)
                details=[]
                for seed in seeds:
                    if seed//256!=ci:continue
                    hit=(patches==seed%256)&valid;byclass={}
                    for j,name in enumerate(classes):
                        sel=hit&(cls==j)
                        if sel.any():byclass[name]=dict(pixels=int(sel.sum()),depth_median_m=float(np.median(dep[sel])),centroid_base_m=pc[sel].mean(0).tolist())
                    details.append(dict(seed=int(seed),classes=byclass))
                replay.append(dict(camera=mjcam,rgb_mad=mad,gt_patch_maxabs=float(np.max(np.abs(cv-cov[ci]))),valid_points=int(valid.sum()),seed_patch_geometry=details))
            result['replay_depth_diagnostic']=replay
        finally:rgb_renderer.close();scene.close()
    (out/'grounding_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['action_last','action_first8','noun_orange','product','actionseed_siglip_encoded','siglip_encoded','nounseed_siglip_encoded','nounseed_projector','nounseed_prefix_L8_out']
    fig,axes=plt.subplots(2,5,figsize=(15,6.4),layout='constrained')
    for ax,name in zip(axes.flat,names):
        ax.imshow(np.clip((images[2]+1)/2,0,1));selected=np.zeros(768);selected[np.argsort(maps[name])[-20:]]=1
        ax.imshow(np.ma.masked_where(selected[512:].reshape(16,16)==0,selected[512:].reshape(16,16)),extent=(-.5,223.5,223.5,-.5),alpha=.45,cmap='autumn',vmin=0,vmax=1)
        ax.set_title(name+' (top20)',fontsize=10);ax.axis('off')
    axes.flat[-1].imshow(labels[2]==0,cmap='gray');axes.flat[-1].set_title('Visible target GT (evaluation only)',fontsize=9);axes.flat[-1].axis('off')
    fig.savefig(out/'grounding_top20.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ['methods','replay_depth_diagnostic']},indent=2))
    for name in names:print(name,result['methods'][name]['20'])
    if args.replay:print('replay',result['replay_depth_diagnostic'])

if __name__=='__main__':main()
