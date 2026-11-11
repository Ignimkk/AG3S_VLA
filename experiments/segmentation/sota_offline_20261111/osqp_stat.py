import json,glob,sys,ast
import numpy as np
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
for tag in sys.argv[1:]:
    nsol=ncap=ncall=ncallcap=0; da_cap=[]; da_ok=[]; ms=[]; slack=[]
    for t in range(4):
        ds=sorted(glob.glob(B+f"2*_{tag}-e1a-spatial-LI-t{t}-*"))
        if not ds: continue
        for f in glob.glob(ds[-1]+"/episodes/*_calls.jsonl"):
            for l in open(f):
                c=json.loads(l); to=c.get("to") or {}; qp=lit(to.get("ee_action_qp"))
                if not qp or not qp.get("status"): continue
                st=qp["status"]; cap=[("maximum" in s) for s in st]
                nsol+=len(st); ncap+=sum(cap); ncall+=1; ncallcap+=cap[-1]
                (da_cap if cap[-1] else da_ok).append(qp.get("max_abs_delta_action") or 0)
                ms.append(float(to.get("solve_ms") or 0)); slack.append(qp.get("slack_max_m") or 0)
    print(f"{tag}: QP calls {ncall}, last solve at OSQP cap {ncallcap} ({ncallcap/max(ncall,1):.2f}); solves {nsol}, cap {ncap} ({ncap/max(nsol,1):.2f}); |Da| median cap {np.median(da_cap) if da_cap else 0:.2f} ok {np.median(da_ok) if da_ok else 0:.2f}; solve_ms median {np.median(ms):.0f} p90 {np.percentile(ms,90):.0f}; slack>1mm {np.mean(np.array(slack)>1e-3):.2f}")
