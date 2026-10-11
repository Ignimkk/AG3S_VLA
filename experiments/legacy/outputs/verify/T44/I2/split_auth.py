from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, collections
R=_ag3s_workspace_path('outputs/verify/T44/I2/raw')
tot=collections.Counter(); per=[]
nonauth_links=collections.Counter(); auth_links=collections.Counter()
for p in sorted(glob.glob(R+"/wt_V11AB*.json")):
    d=json.load(open(p))
    for c in d["chunks"]:
        if "skipped" in c: continue
        allowed=set(c["contact_rows_links"] or [])
        row={"run":p.split("wt_")[1][:-5],"seq":c["seq"],"held":c["held"]}
        for tag,arm in (("off","swR"),("strict","swS")):
            L=c["arms"][arm]["path_target_links"] or {}
            a=sum(v for k,v in L.items() if k in allowed)
            n=sum(v for k,v in L.items() if k not in allowed)
            row[tag+"_auth"]=a; row[tag+"_nonauth"]=n
            tot[tag+"_auth"]+=a; tot[tag+"_nonauth"]+=n
            if tag=="off":
                for k,v in L.items():
                    (auth_links if k in allowed else nonauth_links)[k]+=v
        # probe (same trajectory)
        pr=c.get("probes",{})
        if "path_off" in pr:
            for tag in ("off","strict"):
                L=pr["path_"+tag]["excluded_links"]
                a=sum(v for k,v in L.items() if k in allowed); n=sum(v for k,v in L.items() if k not in allowed)
                row["p_"+tag+"_auth"]=a; row["p_"+tag+"_nonauth"]=n
                tot["p_"+tag+"_auth"]+=a; tot["p_"+tag+"_nonauth"]+=n
        per.append(row)
print(dict(tot)); print("nonauth links off",dict(nonauth_links)); print("auth links off",dict(auth_links))
print("chunks off nonauth>0",sum(1 for r in per if r["off_nonauth"]>0),"strict nonauth>0",sum(1 for r in per if r["strict_nonauth"]>0))
print("chunks p off nonauth>0",sum(1 for r in per if r.get("p_off_nonauth",0)>0),"p strict nonauth>0",sum(1 for r in per if r.get("p_strict_nonauth",0)>0))
print("auth changed solve",sum(1 for r in per if r["off_auth"]!=r["strict_auth"]), "auth changed same traj",sum(1 for r in per if r.get("p_off_auth",0)!=r.get("p_strict_auth",0)))
print("held chunks with nonauth off", sum(1 for r in per if r['held'] and r['off_nonauth']>0), "nonheld", sum(1 for r in per if not r['held'] and r['off_nonauth']>0))
json.dump({"totals":dict(tot),"per_chunk":per,"nonauth_links_off":dict(nonauth_links),"auth_links_off":dict(auth_links)},open(_ag3s_workspace_path('outputs/verify/T44/I2/split_auth.json'),"w"))
