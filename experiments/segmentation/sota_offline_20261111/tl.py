# generic timeline: tl.py <run-dir glob (under results/benchmark)>  -> per episode P/Q/H (lowercase = held), worst links, owners unknown
import json,glob,sys,ast,collections,os
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
for d in sorted(glob.glob(B+sys.argv[1])):
    if not os.path.isfile(d+"/episodes.jsonl"): continue
    print("##", os.path.basename(d)[:70])
    for l in open(d+"/episodes.jsonl"):
        e=json.loads(l); sl=e.get("safelibero") or {}
        res=('S' if e.get('official_success') else 'F')+('+C' if sl.get('collided') else '')
        fs=glob.glob(f"{d}/episodes/*_t{e.get('task_id')}_e{e['episode_idx']}_calls.jsonl")
        if not fs: print(e.get('task_id'), e['episode_idx'], res, "no calls"); continue
        cs=[json.loads(x) for x in open(fs[0])]
        tl=""; links=collections.Counter(); why=collections.Counter(); das=[]
        for c in cs:
            to=c.get("to") or {}; pc=lit(to.get("passthrough_clear")); qp=lit(to.get("ee_action_qp"))
            g=(c.get("grasp") or {}).get("state")
            ch='H' if c.get("hold") else ('Q' if qp.get("status") else 'P')
            if g=="held": ch=ch.lower()
            tl+=ch
            if ch in "Qq":
                why[pc.get("reason")]+=1; das.append(qp.get("max_abs_delta_action") or 0)
                w=lit(lit(pc.get("obstacle_rows")).get("worst")) or lit(pc.get("reference_worst"))
                links[(w.get("link") or "?").replace("panda_","")[:10]]+=1
        import statistics
        print(f"t{e.get('task_id')} e{e['episode_idx']} {res:4s} n{len(cs):3d} disp {1000*float(sl.get('max_displacement_m') or 0):6.1f} why {dict(why)} |Da|med {statistics.median(das) if das else 0:.2f} {dict(links.most_common(3))}\n      {tl}")
