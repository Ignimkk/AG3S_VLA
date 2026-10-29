"""T44b D-merge: per candidate, merged frames where the chosen component is mostly NOT target (precision < 0.5 after a split) — the
neighbour would be carved as the target (worse than staying merged). usage: wrongpick.py merge_gt.json merge_vla.json"""
import json, sys
out = {}
for f in sys.argv[1:]:
    R = json.load(open(f)); src = R[0]["attention_source"]
    M = [r for r in R if r["e2_merged"]]; N = [r for r in R if not r["e2_merged"]]
    o = {}
    for n in M[0]["candidates"]:
        o[n] = {s: {"merged_wrong_pick": sum(r["candidates"][n][s]["precision"] < .5 and r["candidates"][n][s].get("split_taken") for r in M),
                    "nonmerged_wrong_pick": sum(r["candidates"][n][s]["precision"] < .5 and r["candidates"][n][s].get("split_taken") for r in N),
                    "merged_unsplit": sum(r["candidates"][n][s].get("split_taken") is False for r in M)}
                for s in ("att", "mean_raw")}
    out[src] = o
json.dump(out, open("wrongpick.json", "w"), indent=1)
for s, o in out.items():
    print("==", s)
    for n, v in o.items():
        print(f"  {n:42s} att wrong {v['att']['merged_wrong_pick']:2d} unsplit {v['att']['merged_unsplit']:2d} nonm-wrong {v['att']['nonmerged_wrong_pick']} | mean_raw wrong {v['mean_raw']['merged_wrong_pick']:2d} unsplit {v['mean_raw']['merged_unsplit']:2d}")
