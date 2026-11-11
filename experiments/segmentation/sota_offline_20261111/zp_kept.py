# zp_kept.py <tag> : per episode, calls with kept palm rows by latch state (SL-SOTA13 record to.grasp_contact.zone_kept_palm_rows)
import json,glob,sys,ast,collections
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
tag=sys.argv[1]; out={}
for t in (0,1,2,3):
    d=sorted(glob.glob(B+f"2*_{tag}-e1a-spatial-LI-t{t}-*"))[-1]
    for l in open(d+"/episodes.jsonl"):
        e=json.loads(l); sl=e.get("safelibero") or {}
        cs=[json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_calls.jsonl")]
        by=collections.Counter(); rows=[]; first=None
        for c in cs:
            gc=lit((c.get("to") or {}).get("grasp_contact")); k=gc.get("zone_kept_palm_rows") or 0
            if k:
                by[(c.get("grasp") or {}).get("state")]+=1; rows.append(k)
                if first is None: first=c["seq"]
        ok=bool(e.get("official_success")); col=bool(sl.get("collided"))
        r='SS' if ok and not col else 'S+C' if ok else 'F+C' if col else 'F'
        out[f"t{t}e{e['init_state_index']}"]={"result":r,"calls":len(cs),"kept_calls":dict(by),"kept_rows_max":max(rows) if rows else 0,"first":first}
        print(f"t{t} e{e['init_state_index']:2d} {r:4s} calls {len(cs):2d} kept-calls {dict(by)} max rows {max(rows) if rows else 0} first@{first}")
json.dump(out,open(sys.argv[2],"w"),indent=1) if len(sys.argv)>2 else None
