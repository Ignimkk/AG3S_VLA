import json,glob,sys,ast
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
tag,t,init=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); lo=int(sys.argv[4]) if len(sys.argv)>4 else 0; hi=int(sys.argv[5]) if len(sys.argv)>5 else 999
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
d=sorted(glob.glob(B+f"2*{tag}-e1a-spatial-LI-t{t}-*"))[-1]
for l in open(d+"/episodes.jsonl"):
    e=json.loads(l)
    if e["init_state_index"]!=init: continue
    cs=[json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_calls.jsonl")]
    for c in cs:
        if not lo<=c["seq"]<=hi: continue
        to=c.get("to") or {}; pc=lit(to.get("passthrough_clear")); qp=lit((c.get("conversion") or {}).get("ee_action_qp"))
        g=c.get("grasp") or {}; ev=lit(g.get("evidence")); ta=lit(g.get("tip_alignment")); now=lit(ta.get("now"))
        orow=lit(pc.get("obstacle_rows")); w=lit(orow.get("worst")); pat=lit(orow.get("patience")); pr=lit(qp.get("progress"))
        print(f"{c['seq']:2d} {'H' if c.get('hold') else ('Q' if qp.get('used') else 'P')} {str(g.get('state'))[:7]:7s} open {ev.get('opening_after') if ev.get('opening_after') is None else round(ev.get('opening_after'),3)} tipz {now.get('tip_m',[0,0,0])[2]:.3f} dxy {now.get('dxy_mm')} man {ta.get('manipulated_state')} | pass {pc.get('reason')} oviol {round(1000*(orow.get('violation_m') or 0),1)} {str(w.get('link','')).replace('panda_','')[:14]} cl {round(1000*(w.get('clearance_m') or 0),1)} pat {pat.get('blocked_calls')}/{pat.get('relaxed')} | dAct {round(c.get('max_abs_action_change_executed') or 0,3)} prog {pr.get('why') or pr.get('gain')} {str(c.get('verdict_reasons'))[:60]}")
