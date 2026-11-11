# scr.py <tag> <task>: per-episode latch / zone / QP summary of a screen run (E1a sub runs)
import json,glob,sys,ast,collections
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
tag,t=sys.argv[1],int(sys.argv[2])
d=sorted(glob.glob(B+f"2*_{tag}-e1a-spatial-LI-t{t}-*"))[-1]
for l in open(d+"/episodes.jsonl"):
    e=json.loads(l); sl=e.get("safelibero") or {}
    cs=[json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_calls.jsonl")]
    st="".join({"searching":"s","latched":"l","closing":"c","held":"H","placed":"p"}.get((c.get("grasp") or {}).get("state"),"?") for c in cs)
    q=sum(1 for c in cs if lit((c.get("to") or {}).get("ee_action_qp")).get("status")); h=sum(1 for c in cs if c.get("hold"))
    used=[c["seq"] for c in cs if lit(lit((c.get("grasp") or {}).get("evidence")).get("reach_sticky")).get("used")]
    att=[c["seq"] for c in cs if (c.get("grasp") or {}).get("attach")]
    zr=[lit((c.get("to") or {}).get("grasp_contact")).get("zone_release_rows") for c in cs]; zr=[z for z in zr if z]
    ok=bool(e.get("official_success")); col=bool(sl.get("collided"))
    print(f"t{t} e{e['init_state_index']:2d} {'SS' if ok and not col else 'S+C' if ok else 'F+C' if col else 'F':4s}{1000*float(sl.get('max_displacement_m') or 0):6.1f} calls {len(cs):2d} QP {q:2d} HOLD {h:2d} attach@{att} sticky-used@{used[:3]} zone-release rows med {sorted(zr)[len(zr)//2] if zr else 0} | {st}")
