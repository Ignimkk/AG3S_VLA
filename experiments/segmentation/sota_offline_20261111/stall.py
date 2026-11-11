# stall.py <tag> "t:e t:e ..." : per stalled episode, statistics of the QP-corrected calls
import json,glob,sys,ast,collections
import numpy as np
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
tag=sys.argv[1]; out={}
for te in sys.argv[2].split():
    t,i=map(int,te.split(":"))
    d=sorted(glob.glob(B+f"2*_{tag}-e1a-spatial-LI-t{t}-*"))[-1]
    e=[x for x in map(json.loads,open(d+"/episodes.jsonl")) if x["init_state_index"]==i][0]
    cs=[json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_calls.jsonl")]
    links=collections.Counter(); cl=[]; rel=0; q=0; steps=[]; tip=[]; pts=[]; prog=collections.Counter(); held=0; dact=[]; last_q=None
    for c in cs:
        to=c.get("to") or {}; pc=lit(to.get("passthrough_clear")); qp=lit(to.get("ee_action_qp"))
        if not qp.get("status"): continue
        q+=1; last_q=c["seq"]
        orow=lit(pc.get("obstacle_rows")); w=lit(orow.get("worst")); pat=lit(orow.get("patience"))
        if pc.get("reason")!="obstacle": links["("+str(pc.get("reason"))+")"]+=1; continue
        links[str(w.get("link","")).replace("panda_","").split("[")[0]]+=1
        cl.append(1000*(w.get("clearance_m") or 0)); rel+=bool(pat.get("relaxed")); steps.append(w.get("step") or 0)
        if w.get("point_m"): pts.append(w["point_m"])
        g=c.get("grasp") or {}; held+=(g.get("state")=="held")
        now=lit(lit(g.get("tip_alignment")).get("now")); 
        if now.get("tip_m"): tip.append(now["tip_m"])
        prog[str(lit(qp.get("progress")).get("mode"))]+=1; dact.append(qp.get("max_abs_delta_action") or 0)
    n=len(cl)
    r=dict(calls=len(cs),qp_calls=q,last_qp_seq=last_q,obstacle_blocked=n,held=held,relaxed=rel,links=dict(links.most_common(4)),
           clearance_mm_median=round(float(np.median(cl)),1) if n else None, clearance_mm_min=round(float(np.min(cl)),1) if n else None,
           worst_step_median=float(np.median(steps)) if n else None,
           point_median_m=[round(float(x),3) for x in np.median(np.array(pts),0)] if pts else None,
           point_spread_mm=round(float(1000*np.linalg.norm(np.std(np.array(pts),0))),0) if pts else None,
           tip_last_m=[round(float(x),3) for x in tip[-1]] if tip else None, tip_travel_last20_mm=round(float(1000*np.linalg.norm(np.ptp(np.array(tip[-20:]),0))),0) if tip else None,
           progress=dict(prog), dact_median=round(float(np.median(dact)),2) if dact else None,
           status=e.get("status"), disp_mm=round(1000*float((e.get("safelibero") or {}).get("max_displacement_m") or 0),1))
    out[f"t{t}_e{i}"]=r
    print(f"t{t} e{i:2d} calls {r['calls']:3d} qp {q:3d} blocked {n:3d} held {held:3d} relaxed {rel:3d} links {r['links']} cl med/min {r['clearance_mm_median']}/{r['clearance_mm_min']} step {r['worst_step_median']} pt {r['point_median_m']} ±{r['point_spread_mm']} tip {r['tip_last_m']} travel20 {r['tip_travel_last20_mm']} prog {dict(prog)} dAct {r['dact_median']} {r['status']} {r['disp_mm']}")
if len(sys.argv)>3: json.dump(out,open(sys.argv[3],"w"),ensure_ascii=False,indent=1)
