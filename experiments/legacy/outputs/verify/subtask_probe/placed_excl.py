import glob, json, collections, numpy as np, sys
roots = ["outputs/verify/T37/E3a_rec/server_8204/server_constraints","outputs/verify/T37/E3b_rec/server_8206/server_constraints",
         "outputs/verify/T34/E3a_rec/server_8204/server_constraints","outputs/verify/T34/E3b_rec/server_8206/server_constraints"]
f0 = sorted(glob.glob(roots[0]+"/run_*/chunk_*.npz"))[40]
s0 = json.loads(str(np.load(f0)["summary_json"]))
print("keys:", list(s0.keys())[:40]); print("exclusion:", json.dumps(s0.get("exclusion"))[:300])
cnt = collections.Counter(); rows=[]
for root in roots:
    for run in sorted(glob.glob(f"{root}/run_*")):
        prev=None; placed=False; switched=False; ep=0
        for f in sorted(glob.glob(f"{run}/chunk_*.npz")):
            s=json.loads(str(np.load(f)["summary_json"])); t=int(s["t_step"])
            if prev is not None and t<=prev: ep+=1; placed=False; switched=False
            prev=t
            st=(s.get("grasp") or {}).get("state")
            if st=="placed": placed=True
            if not placed: continue
            mode=((s.get("admissibility") or {}).get("decision") or {}).get("mode")
            if mode=="switch": switched=True
            ex=s.get("exclusion") or {}
            key=("after_switch" if switched else "before_switch", bool(ex.get("active")), ex.get("source"))
            cnt[key]+=1
for k,v in sorted(cnt.items(), key=str): print(k, v)
