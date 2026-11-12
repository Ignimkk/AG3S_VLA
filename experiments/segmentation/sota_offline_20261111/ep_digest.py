# ep_digest.py <glob> [<glob>...]: one block per non-safe-success episode -- task, call timeline, ground-truth object motion
# (sim_state of *_geom.npz), when the SafeLIBERO obstacle first moved and what the server was doing at that call.
import glob, json, os, re, sys
import numpy as np
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
P05=json.load(open("table10_A.json"))["safelibero"].get("pi0.5 alone",{}) if os.path.isfile("table10_A.json") else {}
def objs(g):
    names=[str(n) for n in g["model_joint_names"]]; S=g["sim_state"]; adr=1; out={}
    for n in names:
        if n.endswith("_joint0"): out[n[:-7]]=S[:,adr:adr+3].copy(); adr+=7
        else: adr+=1
    return out
for pat in sys.argv[1:]:
  for d in sorted(glob.glob(B+pat)):
    if not os.path.isfile(d+"/episodes.jsonl"): continue
    m=re.search(r"_ms-[^-]+-(\w+)-(LI+)-i",d); cell=f"{m.group(1)} L-{m.group(2)[1:]}"
    for l in open(d+"/episodes.jsonl"):
        e=json.loads(l); sl=e.get("safelibero") or {}; t,i=int(e["task_id"]),int(e["init_state_index"])
        ok=bool(e["official_success"]); col=bool(sl.get("collided"))
        if ok and not col: continue
        f=glob.glob(d+f"/episodes/*_t{t}_e{i}_calls.jsonl")[0]
        cs=[json.loads(x) for x in open(f)]
        g=np.load(f.replace("_calls.jsonl","_geom.npz"),allow_pickle=True); O=objs(g); oi=[int(x) for x in g["obs_index"]]
        z=np.load(f.replace("_calls.jsonl","_calls.npz"),allow_pickle=True); ap=z["actions_policy"]; ae=z["actions_executed"]; n=ae.shape[1]
        tl=""; rows=[]
        for c in cs:
            qp=((c.get("conversion") or {}).get("ee_action_qp") or {}); st=(c.get("grasp") or {}).get("state")
            ch='H' if c.get("hold") else ('Q' if qp.get("used") else 'P'); tl+=ch.lower() if st=="held" else ch
            w=((((c.get("to") or {}).get("passthrough_clear") or {}).get("obstacle_rows") or {}).get("worst") or {})
            s=c["seq"]; da=float(np.linalg.norm((ae[s,:n,:3]-ap[s,:n,:3]).sum(0))) if s<len(ae) else 0.0
            rows.append((ch,st,str(w.get("link")).replace("panda_",""),None if w.get("clearance_m") is None else round(1000*w["clearance_m"],1),round(da,2),float(ap[s,:n,6].min()) if s<len(ap) else 0))
        q=P05.get(cell,{}).get(str(t)); 
        print(f"\n=== {cell} t{t} e{i}: {'S' if ok else 'F'}{'+C' if col else ''} obstacle {1000*float(sl.get('max_displacement_m') or 0):.0f} mm | pi0.5 alone {q['label'] if q else '?'} | calls {len(cs)} | {e.get('task_description')}")
        print("   "+tl)
        k_of=lambda fr: max([j for j,c in enumerate(cs) if c['obs_index'] in oi and oi.index(c['obs_index'])<=fr] or [0])
        for name,p in O.items():
            rise=p[:,2]-p[0,2]; dxy=np.linalg.norm(p[:,:2]-p[0,:2],axis=1); mv=np.linalg.norm(p-p[0],axis=1)
            if "obstacle" in name:
                if mv.max()>0.002 and p[0,2]<2:   # parked obstacles sit far away
                    fr=int(np.argmax(mv>0.001)); k=k_of(fr); r=rows[k]
                    print(f"   OBSTACLE {name}: moved {1000*mv.max():.0f} mm, first >1 mm at call {k} ({r[0]} state {r[1]}, worst {r[2]} {r[3]} mm, QP |Da| {r[4]}); calls around: {tl[max(0,k-3):k+4]}")
                continue
            if rise.max()>0.02 or dxy.max()>0.02:
                fr=int(np.argmax((rise>0.02)|(dxy>0.02))); above=np.where(p[:,2]-p[-1,2]>0.02)[0]
                print(f"   moved {name}: rise max {100*rise.max():.1f} cm, final dxy {100*dxy[-1]:.1f} dz {100*rise[-1]:.1f} cm, starts at call {k_of(fr)}, last airborne call {k_of(int(above[-1])) if len(above) else '-'}")
        opens=[c['seq'] for c,r in zip(cs,rows) if r[1]=='held' and r[5]<0]
        qs=[j for j,r in enumerate(rows) if r[0] in 'Qq']
        links={} 
        for j in qs: links[rows[j][2]]=links.get(rows[j][2],0)+1
        print(f"   QP calls {len(qs)} by worst link {dict(sorted(links.items(), key=lambda x:-x[1]))}; HOLD {sum(r[0] in 'Hh' for r in rows)}; policy opens while held at calls {opens[:6]}")
