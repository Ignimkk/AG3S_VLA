"""Read-only single-frame diagnostics. No model inference or causal claims.
Run: OPENBLAS_NUM_THREADS=1 .venv-ag3s/bin/python .../analyze.py DUMP_ROOT OUTPUT_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / max(float(np.linalg.norm(x)), 1e-30)


def prob(x):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0)
    return x / max(float(x.sum()), 1e-30)


def js(a, b):
    a, b = prob(a), prob(b)
    m = (a + b) / 2
    return float(sum(np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0])) for p in (a, b)) / 2)


def describe(x):
    p = prob(x)
    return dict(mass=float(np.sum(x)), entropy_nats=float(-np.sum(p[p > 0]*np.log(p[p > 0]))),
                effective_patches=float(np.exp(-np.sum(p[p > 0]*np.log(p[p > 0])))),
                camera_fraction=p.reshape(3,256).sum(1).tolist(), argmax=int(p.argmax()),
                top10_mass=float(np.sort(p)[-10:].sum()))


def overlap(a,b,k=20):
    return len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:])) / k


def cosine_rows(x, y):
    x, y = np.asarray(x,np.float64), np.asarray(y,np.float64)
    return (x @ y) / np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(y),1e-30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('dump',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args()
    d=args.dump/'frame_022'; out=args.out;out.mkdir(parents=True,exist_ok=True)
    with np.load(d/'inputs.npz') as z:
        mask=z['token_mask'];pieces=z['token_pieces'];images=[z['image_'+c] for c in z['image_order']]
        meta=dict(prompt=str(z['prompt']),valid_text=int(mask.sum()),image_order=z['image_order'].tolist(),
                  tokens=pieces[mask].tolist())
    with np.load(d/'derived.npz') as z:
        a=z['serving_L8H2_last_d0'].astype(float); n=z['noun_target_L10H7'].astype(float)
        meta.update(noun_row=int(z['noun_prefix_row']),hsr_p=z['hsr_p'].tolist())
    with np.load(d/'outputs.npz') as z:
        x0=z['x0_normalized'];actions=z['actions'];recorded=z['recorded_server_actions']
    result=dict(source=str(d),scope='one observation, one noise seed; descriptive, not causal',metadata=meta,
                parity=json.loads((args.dump/'parity.json').read_text()),
                original_dump_script_sha256=hashlib.sha256((args.dump/'vla_full_dump.py').read_bytes()).hexdigest())
    result['maps']={k:describe(v) for k,v in dict(action=a,noun=n,product=prob(a)*prob(n),mixture=(prob(a)+prob(n))/2).items()}
    result['map_agreement']=dict(js_nats=js(a,n),cosine=float(unit(a)@unit(n)),top20_overlap=overlap(a,n))
    with np.load(d/'embed.npz') as z:
        e=z['image_tokens']; t=z['text_embeddings']; pmask=z['prefix_mask']
    result['raw_image_text_cosine']={str(i):dict(piece=str(pieces[i]),min=float(cosine_rows(e,t[i]).min()),max=float(cosine_rows(e,t[i]).max()),argmax=int(cosine_rows(e,t[i]).argmax())) for i in [5,8]}
    # In a fixed head's 256D value space, token c_i = attention_i * value_i.
    # This is exact local algebra on stored tensors, before output projection/gates.
    with np.load(d/'prefix_L08.npz') as z:
        pv=z['v'][:,0].astype(np.float64)
    rows=[];maps=[];vel=[];path=[];gate_rows=[]
    for s in range(10):
        with np.load(d/f'suffix_s{s:02d}.npz') as z:
            att=z['L_attn_probs'][8,2].astype(np.float64)
            sv=z['L_v'][8,:,0].astype(np.float64)
            values=np.concatenate([pv,sv]);weights=att[-1]
            c=weights[:,None]*values;y=c.sum(0)
            cn=np.linalg.norm(c,axis=1);projection=c@unit(y)
            image_norm=cn[:768]
            im=att[-1,:768]; maps.append(im)
            vp=prob(im);mu=vp@pv[:768];centered=pv[:768]-mu
            cov=(centered.T*vp)@centered
            eig=np.maximum(np.linalg.eigvalsh(cov),0)
            gram=c@c.T
            row=dict(step=s,time=float(z['time']),image_map=describe(im),
                     first_query_image_map=describe(att[0,:768]),
                     first8_queries_image_map=describe(att[:8,:768].mean(0)),
                     mean_query_image_map=describe(att[:,:768].mean(0)),
                     weighted_value_covariance_trace=float(eig.sum()),
                     weighted_value_covariance_participation_ratio=float(eig.sum()**2/max((eig**2).sum(),1e-30)),
                     gram_diagonal_sum=float(np.trace(gram)),gram_off_diagonal_sum=float(gram.sum()-np.trace(gram)),
                     gram_sum_vs_y_squared_error=float(abs(gram.sum()-y@y)),
                     full_attention_mass=float(weights.sum()),text_mass=float(weights[768:968].sum()),
                     task_text_mass=float(weights[768:778].sum()),state_section_mass=float(weights[778:836].sum()),
                     action_header_mass=float(weights[836:840].sum()),padding_mass=float(weights[840:968].sum()),
                     suffix_mass=float(weights[968:].sum()),
                     value_norm_map=describe(image_norm),
                     attention_vs_value_norm_top20_overlap=overlap(im,image_norm),
                     attention_vs_value_norm_cosine=float(unit(im)@unit(image_norm)),
                     image_negative_projection_fraction=float(np.mean(projection[:768]<0)),
                     negative_projection_image_attention_mass_fraction=float(im[projection[:768]<0].sum()/im.sum()),
                     all_tokens_cancellation_ratio=float(np.linalg.norm(y)/max(cn.sum(),1e-30)),
                     image_tokens_cancellation_ratio=float(np.linalg.norm(c[:768].sum(0))/max(cn[:768].sum(),1e-30)),
                     first_vs_last_query_js=js(att[0,:768],im),
                     mean_vs_last_query_js=js(att[:,:768].mean(0),im),
                     noun_js=js(n,im),
                     velocity_vs_original_rmse=float(np.sqrt(np.mean((z['v_t']-z['v_t_orig'])**2))))
            rows.append(row);vel.append(z['v_t'].astype(float));path.append(z['x_t'].astype(float))
            rin=z['L_resid_in'].astype(float);rm=z['L_resid_mid'].astype(float);ro=z['L_resid_out'].astype(float)
            for l in range(18):
                da=rm[l]-rin[l];dm=ro[l]-rm[l]
                gate_rows.append(dict(step=s,layer=l,attn_relative_norm=float(np.linalg.norm(da)/np.linalg.norm(rin[l])),
                                      mlp_relative_norm=float(np.linalg.norm(dm)/np.linalg.norm(rm[l])),
                                      attn_mlp_cosine=float(unit(da.ravel())@unit(dm.ravel()))))
    path.append(x0.astype(float));path=np.stack(path);vel=np.stack(vel)
    result['suffix_L8H2_last']=rows;result['suffix_gated_updates']=gate_rows
    result['denoise']={}
    for label,sl in [('all32',slice(None)),('active16_normalized',slice(0,16))]:
        v=vel[:,:,sl].reshape(10,-1); x=path[:,:,sl].reshape(11,-1)
        dis=np.linalg.norm(np.diff(x,axis=0),axis=1)
        result['denoise'][label]=dict(velocity_norm=np.linalg.norm(v,axis=1).tolist(),
            adjacent_velocity_cosine=[float(unit(v[i])@unit(v[i+1])) for i in range(9)],
            path_length=float(dis.sum()),endpoint_distance=float(np.linalg.norm(x[-1]-x[0])),
            path_length_over_displacement=float(dis.sum()/np.linalg.norm(x[-1]-x[0])),
            distance_to_final=np.linalg.norm(x-x[-1],axis=1).tolist(),
            euler_maxabs=float(np.max(np.abs(np.diff(x,axis=0)+.1*v))))
    result['denoise']['successive_map_js']=[js(maps[i],maps[i+1]) for i in range(9)]
    result['denoise']['first_last_map_js']=js(maps[0],maps[-1])
    result['replay_vs_recorded_actions']=dict(rmse=float(np.sqrt(np.mean((actions-recorded)**2))),maxabs=float(np.max(np.abs(actions-recorded))),note='No same-noise or serving parity asserted.')
    # Track changes in noun representations and residual branch magnitudes.
    pre=[]
    valid=pmask.astype(bool)
    for l in range(18):
        with np.load(d/f'prefix_L{l:02d}.npz') as z:
            ri=z['resid_in'].astype(float)[valid];rm=z['resid_mid'].astype(float)[valid];ro=z['resid_out'].astype(float)[valid]
            da=rm-ri;dm=ro-rm
            p=z['attn_probs'][7]
            pre.append(dict(layer=l,attn_relative_norm=float(np.linalg.norm(da)/np.linalg.norm(ri)),
                            mlp_relative_norm=float(np.linalg.norm(dm)/np.linalg.norm(rm)),
                            attn_mlp_cosine=float(unit(da.ravel())@unit(dm.ravel())),
                            orange_basket_map_js=js(p[773,:768],p[776,:768])))
    result['prefix_layers']=pre
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    # Shareable plot, using a fixed color scale within each map across cameras.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(11,12),layout='constrained')
    plotmaps=[prob(a),prob(n),prob(prob(a)*prob(n)),prob(a*np.linalg.norm(pv[:768],axis=1))]
    titles=['Action attention: L8 H2, step 0, last query','Orange noun: prefix L10 H7','Product fusion (heuristic)','Attention x value norm (before output projection)']
    for r in range(4):
        maximum=plotmaps[r].max()
        for c in range(3):
            ax=axes[r,c];ax.imshow(np.clip((images[c]+1)/2,0,1));ax.imshow(plotmaps[r].reshape(3,16,16)[c],extent=(-.5,223.5,223.5,-.5),cmap='magma',alpha=.6,vmin=0,vmax=maximum);ax.axis('off');ax.set_title(meta['image_order'][c] if r==0 else '',pad=24,fontsize=10)
            if c==0:ax.text(0,1.03,titles[r],transform=ax.transAxes,fontsize=9)
    fig.savefig(out/'maps.png',dpi=160);plt.close(fig)
    with np.load(d/'suffix_s00.npz') as z: query_att=z['L_attn_probs'][8,2,:,:768].astype(float)
    fig,axes=plt.subplots(3,3,figsize=(10,10),layout='constrained')
    for r,(label,qmap) in enumerate([('First query',query_att[0]),('First 8 queries (illustrative window)',query_att[:8].mean(0)),('Last query',query_att[-1])]):
        qmap=prob(qmap)
        for c in range(3):
            ax=axes[r,c];ax.imshow(np.clip((images[c]+1)/2,0,1));ax.imshow(qmap.reshape(3,16,16)[c],extent=(-.5,223.5,223.5,-.5),cmap='magma',alpha=.6,vmin=0,vmax=qmap.max());ax.axis('off')
            ax.set_title(meta['image_order'][c] if r==0 else '',pad=24,fontsize=10)
            if c==0:ax.text(0,1.03,label,transform=ax.transAxes,fontsize=9)
    fig.savefig(out/'queries.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    axes[0].plot(range(10),result['denoise']['all32']['velocity_norm'],'o-');axes[0].set_title('Flow velocity norm (32 normalized dims)')
    axes[1].plot(range(9),result['denoise']['all32']['adjacent_velocity_cosine'],'o-');axes[1].set_title('Adjacent flow direction cosine')
    cf=np.array([r['image_map']['camera_fraction'] for r in rows]);axes[2].stackplot(range(10),cf.T,labels=meta['image_order']);axes[2].legend(fontsize=7);axes[2].set_title('Camera mass within image attention')
    for ax in axes:ax.set_xlabel('Denoise step');ax.grid(alpha=.2)
    fig.savefig(out/'denoise.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['metadata','map_agreement','denoise','replay_vs_recorded_actions']},indent=2))

if __name__=='__main__':main()
