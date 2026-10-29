import json, glob, sys, collections, numpy as np
files = sorted(glob.glob(sys.argv[1] + "/*.json"))
rows = []; att_n = 0; att_ok = 0; att_rec = []
for f in files:
    d = json.load(open(f)); man = json.load(open(d["run"] + "/manifest.json"))
    fruit = man["scene"].get("episode_target_fruit") or "apple"
    for A in d["attaches"]:
        if "error" in A: print("ERR", d["run"], A["error"]); continue
        att_n += 1; att_ok += bool(A["off_matches_record"] and A["fit_dc_mm"] < 1e-3)
        if not (A["off_matches_record"] and A["fit_dc_mm"] < 1e-3): att_rec.append((d["run"], A["attach"], A["fit_dc_mm"], A["off_matches_record"]))
        for c in A["chunks"]:
            rows.append(dict(run=d["run"].split("verify/")[1], fruit=fruit, hand=A["hand"], mode=A["body"].get("mode"), seq=c["seq"], near=c["near"],
                             gap=c["gap_mm"], gapnow=c.get("gap_now_mm"), row=c.get("near_row"), old=min(c["old_mm"]), new=min(c["new_mm"]), last=c["last_held"], first=c["seq"] == A["attach"]))
print("files", len(files), "attaches", att_n, "off rebuild == record", att_ok, "chunks", len(rows))
for x in att_rec: print("   mismatch", x)
print("hands", collections.Counter(r["hand"] for r in rows), "modes", collections.Counter(r["mode"] for r in rows), "fruit", collections.Counter(r["fruit"] for r in rows))
def phase(r):
    if r["last"]: return "release(last held)"
    if r["near"]: return "near support (lifted)"
    return "away (unlifted)"
tab = collections.defaultdict(collections.Counter)
for r in rows:
    k = "new<0<=old" if r["new"] < 0 <= r["old"] else ("both<0" if r["new"] < 0 and r["old"] < 0 else ("old<0<=new" if r["old"] < 0 <= r["new"] else "both>=0"))
    tab[phase(r)][k] += 1
for p, c in tab.items(): print(f"{p:24s}", dict(c))
print("by fruit/mode new<0<=old:")
cc = collections.Counter((r["fruit"], r["mode"], phase(r)) for r in rows if r["new"] < 0 <= r["old"]); 
for k, v in sorted(cc.items()): print("  ", k, v)
print("chunks per fruit/mode:", collections.Counter((r["fruit"], r["mode"]) for r in rows))
print("near chunks: new-old min diff", min((r["new"] - r["old"]) for r in rows if r["near"]))
for r in rows:
    if r["near"] and r["new"] < 0 <= r["old"]: print("  NEAR new viol", r)
if "-v" in sys.argv[2:]:
    for r in rows:
        if r["new"] < 0 <= r["old"] or (r["last"] and r["new"] < 0):
            print(f"  {r['run'][-34:]:34s} {r['fruit']:6s} {r['hand'][0]} {r['mode'][:3]} seq {r['seq']:3d} {'L' if r['last'] else ' '} near {str(r['near'])[0]} row {r['row']} gap {r['gap']} now {r['gapnow']} old {r['old']:6.1f} new {r['new']:6.1f}")
