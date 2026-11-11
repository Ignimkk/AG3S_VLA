import json,glob,sys
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
for tag in sys.argv[1:]:
    print("==",tag)
    tot=0
    for s in (7,8,9):
        ds=sorted(glob.glob(B+f"2*_gtto-{tag}-libero9s{s}-*"))
        if not ds: continue
        E=[json.loads(l) for l in open(ds[-1]+"/episodes.jsonl")]
        ok=[e for e in E if e.get("official_success")]
        tot+=len(ok)
        print(f" libero s{s}: {len(ok)}/{len(E)} fail tasks", [e.get("task_id") for e in E if not e.get("official_success")])
    print(" LIBERO 27 =",tot)
    n=T=C=SS=big=0
    for ds in sorted(glob.glob(B+f"2*_gtto-{tag}-safelibero11-*")):
        for l in open(ds+"/episodes.jsonl"):
            e=json.loads(l); sl=e.get("safelibero") or {}
            s=bool(e.get("official_success")); c=bool(sl.get("collided")); d=1000*float(sl.get("max_displacement_m") or 0)
            n+=1;T+=s;C+=(not c);SS+=(s and not c);big+=(d>50)
            print(f"  t{e.get('task_id')} e{e.get('init_state_index')} {'S' if s else 'F'}{'+C' if c else ''} {d:.1f}")
    print(f" SafeLIBERO {n}: TSR {T} CAR {C} SSR {SS} >50mm {big}")
    for ds in sorted(glob.glob(B+f"2*_gtto-{tag}-ls10-*")):
        E=[json.loads(l) for l in open(ds+"/episodes.jsonl")]
        print("  LS",ds.split("-ls10-")[1].split("-L0")[0],len(E),"succ",sum(bool(e.get("official_success")) for e in E),[ (e.get("task_id"),bool(e.get("official_success")),{k:v for k,v in e.items() if "viol" in k or "cost" in k or "safety" in k.lower()}) for e in E][:12])
