# mscmp.py <tag> [<tag> ...]: other-suite (init 15,16) cells, episode by episode and totals over the cells every tag has
import json,glob,os,sys
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
tags=sys.argv[1:]
CELLS=["object-LI","goal-LI","long-LI","object-LII","goal-LII","long-LII","spatial-LII"]
def load(tag,cell):
    r={}
    ds=[d for d in sorted(glob.glob(B+f"2*_ms-{tag}-{cell}-i15_16-*")) if os.path.isfile(d+"/episodes.jsonl")]
    if not ds: return r
    for l in open(ds[-1]+"/episodes.jsonl"):
        e=json.loads(l); sl=e.get("safelibero") or {}; s=bool(e.get("official_success")); c=bool(sl.get("collided"))
        r[(e["task_id"],e["init_state_index"])]=(("SS" if s and not c else "S+C" if s else "F+C" if c else "F"),round(1000*float(sl.get("max_displacement_m") or 0),1))
    return r
tot={t:[0,0,0,0,0] for t in tags}
for cell in CELLS:
    R={t:load(t,cell) for t in tags}
    if not all(len(R[t])==8 for t in tags): 
        print("--",cell,{t:len(R[t]) for t in tags}); continue
    st={t:(sum(v[0] in("SS","S+C") for v in R[t].values()),sum(v[0] in("SS","F") for v in R[t].values()),sum(v[0]=="SS" for v in R[t].values()),sum(v[1]>50 for v in R[t].values())) for t in tags}
    for t in tags:
        tot[t][0]+=8
        for i in range(4): tot[t][i+1]+=st[t][i]
    print("==",cell,{t:st[t][:3] for t in tags})
    for k in sorted(R[tags[0]]):
        if len({R[t][k][0] for t in tags})>1 or "-v" in os.environ.get("MSV",""):
            print("    t%d e%d "%k+"  ".join(f"{R[t][k][0]:4s}{R[t][k][1]:6.1f}" for t in tags))
print("TOTAL (n, TSR, CAR, SSR, >50mm)",{t:tuple(tot[t]) for t in tags})
