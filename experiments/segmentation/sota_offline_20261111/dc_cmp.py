import glob, json, os, sys, collections
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
RUNS=[("ms","goal-LI-i17_18"),("ms","spatial-LII-i17_18"),("ms","long-LII-i17_18"),("ms","spatial-LI-i15_16_17_18"),("ms","spatial-LII-i15_16"),("ms","long-LI-i15_16"),("sub","e1a-spatial-LI-t3")]
def find(kind,tag,name):
    if kind=="ms":
        suite,lvl,ii=name.split("-"); inits=[int(x) for x in ii[1:].split("_")]
        ds=sorted(glob.glob(B+f"2*_ms-{tag}-{suite}-{lvl}-i*"))
        return ds,inits,None
    ds=sorted(glob.glob(B+f"2*_sub-{tag}-e1a-spatial-LI-*")) if tag=="pkgFrssdm" else sorted(glob.glob(B+f"2*_sub-{tag}-{name}-*"))
    return ds,list(range(5,15)),3
def load(ds,inits,task):
    out={}
    for d in ds:
        if not os.path.isfile(d+"/episodes.jsonl"): continue
        for l in open(d+"/episodes.jsonl"):
            if not l.strip(): continue
            e=json.loads(l); s=e.get("safelibero") or {}
            t,i=int(e["task_id"]),int(e["init_state_index"])
            if i not in inits or (task is not None and t!=task): continue
            cj=glob.glob(d+f"/episodes/*_t{t}_e{i}_calls.jsonl")
            holds=0; dead=0; note=0; reasons=collections.Counter(); last_hold=False; tail=0
            if cj:
                recs=[json.loads(x) for x in open(cj[0]) if x.strip()]
                for r in recs:
                    txt=json.dumps(r)
                    if r.get("hold"):
                        holds+=1
                        for v in r.get("verdict_reasons") or []: reasons[v if isinstance(v,str) else json.dumps(v)[:60]]+=1
                        if "esdf_dead_camera" in txt: dead+=1
                    if "esdf_camera_outside_workspace" in txt: note+=1
                tail=0
                for r in reversed(recs):
                    if r.get("hold"): tail+=1
                    else: break
            out[(t,i)]=dict(s=bool(e["official_success"]),c=bool(s.get("collided")),d=round(1000*float(s.get("max_displacement_m") or 0),1),n=int(e.get("policy_calls") or 0),hold=holds,dead=dead,note=note,tail=tail,reasons=dict(reasons))
    return out
def lab(x): return "SS" if x["s"] and not x["c"] else "S+C" if x["s"] else "F+C" if x["c"] else "F"
def cnt(ep): return [len(ep),sum(x["s"] for x in ep.values()),sum(not x["c"] for x in ep.values()),sum(x["s"] and not x["c"] for x in ep.values()),sum(x["d"]>50 for x in ep.values()),sum(x["hold"] for x in ep.values()),sum(x["dead"] for x in ep.values()),sum(x["note"] for x in ep.values())]
res={}; tot={"frs":[0]*8,"dc":[0]*8}
for kind,name in RUNS:
    a=load(*find(kind,"pkgFrssdm",name)); b=load(*find(kind,"pkgFdcsdm",name))
    ca,cb=cnt(a),cnt(b)
    print(f"== {name}: pkgFrs n{ca[0]} {ca[1]}·{ca[2]}·{ca[3]} >50:{ca[4]} hold:{ca[5]} dead:{ca[6]} | -dc n{cb[0]} {cb[1]}·{cb[2]}·{cb[3]} >50:{cb[4]} hold:{cb[5]} dead:{cb[6]} note:{cb[7]}")
    if kind=="ms":
        tot["frs"]=[x+y for x,y in zip(tot["frs"],ca)]; tot["dc"]=[x+y for x,y in zip(tot["dc"],cb)]
    diff=[]
    for k in sorted(a):
        if k not in b: print("  missing in dc",k); continue
        x,y=a[k],b[k]
        same=(lab(x)==lab(y) and x["d"]==y["d"] and x["n"]==y["n"])
        if not same or x["hold"] or y["hold"] or y["note"]:
            print(f"  t{k[0]}e{k[1]}: {lab(x):3s} d{x['d']:6.1f} n{x['n']:3d} hold{x['hold']:3d}(tail{x['tail']}) -> {lab(y):3s} d{y['d']:6.1f} n{y['n']:3d} hold{y['hold']:3d}(tail{y['tail']}) dead{y['dead']} note{y['note']} {y['reasons'] if y['hold'] else ''}{'' if not same else '  [same outcome/disp/calls]'}")
    res[name]={"pkgFrs":{f"t{k[0]}e{k[1]}":dict(v,label=lab(v)) for k,v in sorted(a.items())},"dc":{f"t{k[0]}e{k[1]}":dict(v,label=lab(v)) for k,v in sorted(b.items())},"cnt_pkgFrs":ca,"cnt_dc":cb}
print("STAGE1 total pkgFrs",tot["frs"],"-dc",tot["dc"])
res["stage1_total"]={"pkgFrs":tot["frs"],"dc":tot["dc"],"cols":["n","TSR","CAR","SSR",">50mm","hold_calls","dead_camera_hold_calls","outside_workspace_notes"]}
json.dump(res,open(sys.argv[1],"w"),indent=1,ensure_ascii=False)
