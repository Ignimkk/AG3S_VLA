# shape.py <glob> [<glob>...]: classify every non-safe-success episode by the call timeline (P / Q / H, held or not)
import json,glob,sys,ast,collections,os,re
B="/mnt/dev/AG3S_VLA-storage/results/benchmark/"
def lit(x):
    if isinstance(x,str):
        try: return ast.literal_eval(x)
        except Exception: return {}
    return x or {}
tot=collections.Counter(); ex=collections.defaultdict(list); n=ss=0
OUT=None
args=sys.argv[1:]
if args and args[0].startswith("--out="): OUT=args.pop(0)[6:]
for pat in args:
  for d in sorted(glob.glob(B+pat)):
    if "INVALID" in d or not os.path.isfile(d+"/episodes.jsonl"): continue
    m=re.search(r"_(ms|sub)-[^-]+-(.*?)-(?:i\d|LI|LII)",os.path.basename(d)); cell=(m.group(2) if m else "?")+("-"+("LII" if "-LII-" in d else "LI") if "ms-" in d else "")
    for l in open(d+"/episodes.jsonl"):
        e=json.loads(l); sl=e.get("safelibero") or {}; n+=1
        ok=bool(e.get("official_success")); col=bool(sl.get("collided")); disp=1000*float(sl.get("max_displacement_m") or 0)
        if ok and not col: ss+=1; continue
        fs=glob.glob(f"{d}/episodes/*_t{e.get('task_id')}_e{e['episode_idx']}_calls.jsonl")
        cs=[json.loads(x) for x in open(fs[0])]
        tl=""; links=collections.Counter()
        for c in cs:
            to=c.get("to") or {}; pc=lit(to.get("passthrough_clear")); qp=lit(to.get("ee_action_qp"))
            g=(c.get("grasp") or {}).get("state")
            ch='H' if c.get("hold") else ('Q' if qp.get("status") else 'P')
            if g=="held": ch=ch.lower()
            tl+=ch
            if ch in "Qq":
                w=lit(lit(pc.get("obstacle_rows")).get("worst")) or lit(pc.get("reference_worst"))
                lk=str(w.get("link") or "?")
                links[("q:" if ch=="q" else "Q:")+("arm" if re.search(r"link[1-7]$",lk) else "held" if lk.startswith("attached") else "hand")]+=1
        ever=any(ch.islower() for ch in tl); last=tl[-15:]
        if ok: cls="성공 + 충돌"
        elif tl.count("H")+tl.count("h")>=10 and (last.count("H")+last.count("h"))>=8: cls="끝까지 HOLD"
        elif not ever: cls="못 쥠: "+("쥐기 전 QP 지속" if tl.count("Q")>=0.3*len(tl) else "QP 적음 (policy 가 못 쥠)")
        elif sum(ch.islower() for ch in last)>=10: cls="쥔 채 끝남: "+("QP 지속" if sum(ch=="q" for ch in tl[-30:])>=10 else "QP 적음")
        else: cls="쥐었다 놓은 뒤 미완: "+("뒤에 QP 지속" if last.count("Q")>=6 else "통과만 이어짐")
        if cls.startswith("못") or "QP 지속" in cls:
            sub=links.most_common(1)[0][0] if links else "-"
            cls+=" ["+{"Q:arm":"팔 link","Q:hand":"손 · 손가락","Q:held":"쥔 물체 행","q:arm":"팔 link","q:hand":"손 · 손가락","q:held":"쥔 물체 행","-":"-"}[sub]+"]"
        tot[cls]+=1; ex[cls].append(f"{cell} t{e.get('task_id')} e{e.get('init_state_index',e['episode_idx'])}"+("" if not col else f" ({disp:.0f} mm)"))
print("n",n,"SS",ss,"non-SS",n-ss)
for k,v in sorted(tot.items(), key=lambda kv:-kv[1]):
    print(f"{v:3d}  {k}\n       "+", ".join(ex[k]))

if OUT: json.dump({"n":n,"safe_success":ss,"classes":dict(tot),"episodes":dict(ex)},open(OUT,"w"),ensure_ascii=False,indent=1)
