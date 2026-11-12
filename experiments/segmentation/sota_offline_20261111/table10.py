#!/usr/bin/env python3
"""table10.py OUT.json: the 1/10 table run (plan §9.30) in the layout of the paper tables.
Tab. 1 = LIBERO SR | SafeLIBERO L-I TSR/SSR/CAR | L-II TSR/SSR/CAR (suite-equal-weight mean), rows pi0.5 alone and AG3S-VLA (pkgFrs).
Tab. 2 = LIBERO-Safety L2 AAG / TSA / FSHOA success and violations. Episodes: SafeLIBERO task 0-3 x init 19; LIBERO task 0,2,4,6,8 x init 19;
LIBERO-Safety task 0,1 x init 19. pi0.5 alone: sdm-pi05-sl-* (Spatial/Object/Goal) and tab10sdm-pi05-* (the rest)."""
import glob, json, os, sys, collections
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
SUITES=("spatial","object","goal","long"); LB=("spatial","object","goal","10"); LS=("affordance","obstacle_avoidance","obstacle_avoidance_human")
def eps(pats, init=19):
    out={}
    for pat in pats:
        for d in sorted(glob.glob(B+pat)):
            if "INVALID" in os.path.basename(d) or not os.path.isfile(d+"/episodes.jsonl"): continue
            for l in open(d+"/episodes.jsonl"):
                if not l.strip(): continue
                e=json.loads(l)
                if int(e["init_state_index"])!=init: continue
                sl=e.get("safelibero") or {}
                out[int(e["task_id"])]=dict(s=bool(e.get("official_success")), c=bool(sl.get("collided")), d=round(1000*float(sl.get("max_displacement_m") or 0),1),
                                            n=int(e.get("policy_calls") or 0), status=e.get("status"), cost=e.get("cost") or e.get("official_cost") or e.get("violation"), run=os.path.basename(d))
    return out
def lab(x): return "SS" if x["s"] and not x["c"] else "S+C" if x["s"] else "F+C" if x["c"] else "F"
res={"safelibero":{}, "libero":{}, "libero_safety":{}}
for arm,pat in (("AG3S-VLA (pkgFrs)", ["2*_ms-tab10sdm-{s}-L{l}-i19-*"]), ("pi0.5 alone", ["2*_sdm-pi05-sl-{s}-L{l}-L{l}-vla_only","2*_tab10sdm-pi05-sl-{s}-L{l}-L{l}-vla_only"])):
    for s in SUITES:
        for l in ("I","II"):
            res["safelibero"].setdefault(arm,{})[f"{s} L-{l}"]={t:dict(v,label=lab(v)) for t,v in eps([p.format(s=s,l=l) for p in pat]).items()}
for arm,pat in (("AG3S-VLA (pkgFrs)","2*_tab10sdm-lb-{s}-*"),("pi0.5 alone","2*_tab10sdm-pi05-lb-{s}-*")):
    for s in LB: res["libero"].setdefault(arm,{})[s]=eps([pat.format(s=s)])
for arm,pat in (("AG3S-VLA (pkgFrs-lsnative, official ckpt)","2*_tab10sdm-lsb-{s}-L2-pi05_libero_safety-vla_v11ab-*"),("pi0.5 alone (pi05_libero_safety, official)","2*_tab10sdm-pi05-ls-{s}-L2-pi05_libero_safety-*")):
    for s in LS: res["libero_safety"].setdefault(arm,{})[s]=eps([pat.format(s=s)])
pct=lambda a,b: "  -  " if not b else f"{100*a/b:5.1f}"
print("== Tab. 1 layout (1/10: SafeLIBERO 32 ep, LIBERO 20 ep; % with episode counts; SafeLIBERO = mean over the 4 suites)")
print(f"{'arm':22s} | LIBERO SR      | L-I TSR  SSR   CAR   (n) | L-II TSR  SSR   CAR   (n)")
tab1={}
for arm in res["safelibero"]:
    lb=res["libero"].get(arm,{}); n=sum(len(v) for v in lb.values()); k=sum(x["s"] for v in lb.values() for x in v.values())
    row=[f"{pct(k,n)} ({k}/{n})"]; tab1[arm]={"libero":[k,n]}
    for l in ("I","II"):
        cells=[res["safelibero"][arm][f"{s} L-{l}"] for s in SUITES]; m=[c for c in cells if c]
        f=lambda key: sum(sum(key(x) for x in c.values())/len(c) for c in m)/len(m) if m else float("nan")
        nn=sum(len(c) for c in m)
        t,ss,ca=f(lambda x:x["s"]),f(lambda x:x["s"] and not x["c"]),f(lambda x:not x["c"])
        cnt=[sum(x["s"] for c in m for x in c.values()),sum(x["s"] and not x["c"] for c in m for x in c.values()),sum(not x["c"] for c in m for x in c.values()),nn]
        row.append(f"{100*t:5.1f} {100*ss:5.1f} {100*ca:5.1f} ({cnt[0]}·{cnt[1]}·{cnt[2]}/{nn})"); tab1[arm]["L-"+l]=cnt
    print(f"{arm:22s} | "+" | ".join(row))
print("\n== SafeLIBERO per cell: TSR·SSR·CAR of 4, then task 0-3 label (obstacle displacement mm)")
for arm in res["safelibero"]:
    print(arm)
    for cell,v in res["safelibero"][arm].items():
        if not v: print(f"   {cell:13s} -"); continue
        print(f"   {cell:13s} {sum(x['s'] for x in v.values())}·{sum(x['s'] and not x['c'] for x in v.values())}·{sum(not x['c'] for x in v.values())}/{len(v)}  "+"  ".join(f"t{t}:{x['label']}({x['d']:.0f})" for t,x in sorted(v.items())))
print("\n== LIBERO per suite (success / n; failed tasks)")
for arm,v in res["libero"].items():
    print(f"   {arm:22s} "+"  ".join(f"{s}: {sum(x['s'] for x in c.values())}/{len(c)} fail{[t for t,x in sorted(c.items()) if not x['s']]}" for s,c in v.items()))
print("\n== Tab. 2 layout: LIBERO-Safety L2 (success / n per suite; status list)")
for arm,v in res["libero_safety"].items():
    print(f"   {arm:44s} "+"  ".join(f"{s}: {sum(x['s'] for x in c.values())}/{len(c)} {[x['status'] for t,x in sorted(c.items())]}" for s,c in v.items()))
a,p=res["safelibero"].get("AG3S-VLA (pkgFrs)",{}),res["safelibero"].get("pi0.5 alone",{})
x=collections.Counter()
for cell in a:
    for t,e in a[cell].items():
        q=p.get(cell,{}).get(t)
        x[(e["label"], "pi0.5 "+("?" if q is None else "success" if q["s"] else "fail"))]+=1
print("\n== SafeLIBERO pairs (AG3S label, pi0.5 alone task success):", dict(x))
if len(sys.argv)>1: json.dump({"tab1_counts":tab1,**res}, open(sys.argv[1],"w"), indent=1, ensure_ascii=False)
