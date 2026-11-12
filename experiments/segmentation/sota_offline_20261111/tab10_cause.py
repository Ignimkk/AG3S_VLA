# tab10_cause.py OUT.json: per episode of the 1/10 table run -- outcome, pi0.5-alone outcome, the share of pre-grasp calls whose
# recognised target is a task object (target_truth.py logs; top-3 owners, so a lower bound), and the bodies that blocked the
# rejected calls (surface_owner_task.py). Manipulated objects per task are written out by hand from the task sentences.
import ast, glob, json, os, re, sys, collections
T=json.load(open("table10_A.json"))["safelibero"]; ours=T["AG3S-VLA (pkgFrs)"]; pi=T["pi0.5 alone"]
OBJ={"spatial":{t:["akita_black_bowl_1"] for t in range(4)},
     "object":{0:["orange_juice_1"],1:["chocolate_pudding_1"],2:["milk_1"],3:["bbq_sauce_1"]},
     "goal-LI":{0:["akita_black_bowl_1"],1:["akita_black_bowl_1"],2:["akita_black_bowl_1"],3:["akita_black_bowl_1","wooden_cabinet_1_cabinet_top"]},
     "goal-LII":{0:["akita_black_bowl_1"],1:["akita_black_bowl_1"],2:["akita_black_bowl_1"],3:["cream_cheese_1"]},
     "long":{0:["alphabet_soup_1","cream_cheese_1"],1:["alphabet_soup_1","tomato_sauce_1"],2:["porcelain_mug_1","white_yellow_mug_1"],3:["porcelain_mug_1","chocolate_pudding_1"]}}
rows=[]
for f in sorted(glob.glob("tt_tab10/*.log")):
    c=os.path.basename(f)[:-4]; suite,lvl=c.split("-L"); cell=f"{suite} L-{lvl}"
    for l in open(f):
        m=re.match(r"safelibero_\w+_t(\d)_e19 goal (\[.*?\]) pre-grasp calls with target (\d+) top owner == goal \d+ (\[.*\])",l)
        if not m: continue
        t=int(m.group(1)); n=int(m.group(3)); own=ast.literal_eval(m.group(4))
        objs=OBJ.get(c.replace("-L","-L") if c in OBJ else suite, OBJ.get(suite)) if c not in OBJ else OBJ[c]
        objs=(OBJ[c] if c in OBJ else OBJ[suite])[t]
        good=sum(k for o,k in own if o in objs)
        e=ours[cell][str(t)]; p=pi.get(cell,{}).get(str(t))
        blk=collections.Counter()
        fo=f"owner_tab10/{suite}-L{lvl}_t{t}_e19.json"
        if os.path.isfile(fo):
            for call in json.load(open(fo)):
                if call.get("applied") is False and call.get("owners"):
                    o=call["owners"][0][0]
                    blk["task object" if o in objs else "SafeLIBERO obstacle" if "obstacle" in o else "robot itself" if o.startswith("gripper0") or o=="eef_marker" else "other scene object"]+=1
        rows.append(dict(cell=cell,task=t,label=e["label"],disp_mm=e["d"],calls=e["n"],pi05=None if p is None else p["label"],target_calls=n,
                         target_on_task_object=good,share=round(good/n,3) if n else None,recognised=own,task_objects=objs,blocked_by=dict(blk)))
rows.sort(key=lambda r:(r["cell"],r["task"]))
print(f"{'cell':13s} t out  pi    target-ok  recognised (top)                                   rejected calls by blocker")
for r in rows:
    print(f"{r['cell']:13s} {r['task']} {r['label']:4s} {str(r['pi05']):4s} {r['target_on_task_object']:3d}/{r['target_calls']:<3d} {100*r['share']:3.0f}%  {str(r['recognised'][:2])[:52]:52s} {r['blocked_by'] if r['label']!='SS' else ''}")
lo=[r for r in rows if r["share"]<0.5]; hi=[r for r in rows if r["share"]>=0.5]
ss=lambda v: sum(r["label"]=="SS" for r in v)
print(f"\nrecognised target is a task object in < 50 % of pre-grasp calls: {len(lo)} ep, safe success {ss(lo)}; >= 50 %: {len(hi)} ep, safe success {ss(hi)}")
tot=collections.Counter()
for r in rows:
    if r["label"]!="SS":
        for k,v in r["blocked_by"].items(): tot[k]+=v
print("rejected calls in the 20 non-safe-success episodes by blocker:", dict(tot), "sum", sum(tot.values()))
json.dump({"episodes":rows,"summary":{"low_share":[len(lo),ss(lo)],"high_share":[len(hi),ss(hi)],"blocked_calls":dict(tot)}},open(sys.argv[1],"w"),indent=1,ensure_ascii=False)
