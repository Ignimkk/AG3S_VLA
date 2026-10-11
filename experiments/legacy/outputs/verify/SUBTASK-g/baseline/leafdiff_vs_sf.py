"""SUBTASK-g: leaf-level diff of baseline/{legacy,curobo}.json vs SUBTASK-f baseline (frames[*] every leaf). Output: leafdiff_vs_sf.json"""
import json
def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, f"{pre}{k}.")
    else: yield pre[:-1], x
out = {}
for b in ("legacy", "curobo"):
    A = json.load(open(f"${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-f/baseline/{b}.json")); B = json.load(open(f"{b}.json"))
    fa, fb = A["frames"], B["frames"]; diff = {}
    for i, (x, y) in enumerate(zip(fa, fb)):
        dx, dy = dict(flat(x)), dict(flat(y))
        for k in set(dx) | set(dy):
            if json.dumps(dx.get(k), sort_keys=True) != json.dumps(dy.get(k), sort_keys=True):
                diff.setdefault(k, []).append(i)
    top = {k for k in set(A) | set(B) if k != "frames" and json.dumps(A.get(k), sort_keys=True) != json.dumps(B.get(k), sort_keys=True)}
    out[b] = {"n_frames": [len(fa), len(fb)], "frame_leaf_diff": {k: {"n_frames": len(v), "frames": v} for k, v in sorted(diff.items())},
              "top_level_keys_differing": sorted(top)}
json.dump(out, open("leafdiff_vs_sf.json", "w"), indent=1)
for b in out: print(b, out[b]["n_frames"], {k: v["n_frames"] for k, v in out[b]["frame_leaf_diff"].items()}, out[b]["top_level_keys_differing"])
