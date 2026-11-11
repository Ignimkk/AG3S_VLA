import json,glob,sys
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
tags=sys.argv[1:]
def load(tag):
    r={}
    for t in range(4):
        ds=sorted(glob.glob(B+f"2*_{tag}-e1a-spatial-LI-t{t}-*"))
        if not ds: continue
        for l in open(ds[-1]+"/episodes.jsonl"):
            e=json.loads(l); sl=e.get("safelibero") or {}
            s=bool(e.get("official_success")); c=bool(sl.get("collided"))
            r[(t,e["init_state_index"])]=(("SS" if s and not c else "S+C" if s else "F+C" if c else "F"),1000*float(sl.get("max_displacement_m") or 0))
    return r
R=[load(t) for t in tags]
keys=sorted(set().union(*[set(r) for r in R]))
for k in keys:
    print(f"t{k[0]} e{k[1]:2d} "+"  ".join(f"{r.get(k,('-',0))[0]:4s}{r.get(k,('-',0))[1]:7.1f}" for r in R))
for i,r in enumerate(R):
    print(tags[i], len(r), sum(v[0] in("SS","S+C") for v in r.values()), sum(v[0] in("SS","F") for v in r.values()), sum(v[0]=="SS" for v in r.values()))
if len(R)>=2:
    a,b=R[-2],R[-1]
    print("gain",[k for k in keys if a.get(k,("",))[0]!="SS" and b.get(k,("",))[0]=="SS"]); print("loss",[k for k in keys if a.get(k,("",))[0]=="SS" and b.get(k,("",))[0]!="SS"])
    print("union SS",sum(any(r.get(k,("",))[0]=="SS" for r in R) for k in keys))
