"""Collect every recorded apple-target closed-loop result (VLA-only and AG3S arms) per episode. Read only."""
import json, collections, sys
W = "/mnt/dev/AG3S_VLA-storage/results/workspace/verify"
D = "/mnt/dev/AG3S_VLA-storage/results/development"
rec = collections.defaultdict(lambda: {"arm_side": None, "split": None, "results": []})
def add(ep, src, cond, seed, ok, extra=None, side=None, split=None):
    r = rec[int(ep)]
    if side: r["arm_side"] = side
    if split: r["split"] = split
    r["results"].append({"src": src, "cond": cond, "seed": seed, "ok": ok, **(extra or {})})
# T42-1 (test split, no obstacle; E0 = VLA only, E3b = AG3S of that time; apple-left rep 1-2 are T39 runs)
t = json.load(open(f"{W}/T42/1/summary.json"))
for ep, v in t["table"].items():
    if v["fruit"] != "apple": continue
    for cond in ("E0", "E3b"):
        for rep, x in (v.get(cond) or {}).items():
            if x is None or x.get("success") is None: continue
            add(ep, "T42-1" if x.get("source") != "T39" else "T39", f"{cond}", x["seed"], bool(x["success"]), {"failure_mode": x.get("failure_mode")}, v["arm"], v["split"])
# T43 LB (train split, no obstacle, VLA only screening)
lb = json.load(open(f"{W}/T43/LB/agg_final.json"))
for ep, v in lb["episodes"].items():
    if v["fruit"] != "apple": continue
    for s in v["seeds"]:
        add(ep, "T43-LB", "E0", s["seed"], bool(s["success"]), {"failure_mode": s.get("failure_mode"), "robust": s.get("robust_B")}, v["arm"], v["split"])
# T44 LB2 (train, crowded, VLA only, pass = both seeds success + robust)
p = json.load(open(f"{W}/T44/LB2/passed.json"))
for x in p["passed"]:
    if x["target_fruit"] != "apple": continue
    for s in x["seeds"]:
        add(x["episode"], "T44-LB2", "E0", s, True, {"note": "passed list (both seeds ok)"}, x["used_arm"], "train")
# T43 V11 hurdle (all apples) contact-inclusive table
ct = json.load(open(f"{W}/T43/V11/ct_V11AB.json"))
for k, v in ct["runs"].items():
    add(v["episode"], "T43-V11", f"{v['arm']}+hurdle", v["seed"], bool(v["success"]))
# IFG hurdle (C1a / C1aN / SURF with hurdle)
h = json.load(open(f"{D}/20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE/agg.json"))
for it in h["items"]:
    for arm, x in it["arms"].items():
        if x: add(it["episode"], "IFG-HURDLE", f"{arm}+hurdle", it["seed"], bool(x.get("success")), None, it.get("side"))
# IFG-CL1 (held-out, C1a successes) and IFG-CL3 (unselected)
for tag, path in (("IFG-CL1", f"{D}/20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1/agg.json"),
                  ("IFG-CL3", f"{D}/20261018T135457+0900_f15e587290f6_2cd4b0b6acad/verify/IFG-CL3/agg.json")):
    a = json.load(open(path))
    for r in a["rows"]:
        if r.get("target_fruit") != "apple": continue
        if tag == "IFG-CL1": add(r["episode"], tag, f"C1a ({r['stratum']})", r["seed"], True, {"note": "selected as C1a success"}, r.get("side"))
        for arm, x in r["arms"].items():
            if x and x.get("success") is not None:
                add(r["episode"], tag, f"{arm} ({r['stratum']})", r["seed"], bool(x.get("primary") if x.get("primary") is not None else x["success"]), {"failure_mode": x.get("failure_mode")}, r.get("side"))
out = []
for ep, r in sorted(rec.items()):
    e0 = [x for x in r["results"] if x["cond"] == "E0"]
    ag = [x for x in r["results"] if x["cond"] != "E0" and "hurdle" not in x["cond"]]
    hu = [x for x in r["results"] if "hurdle" in x["cond"]]
    f = lambda xs: f"{sum(x['ok'] for x in xs)}/{len(xs)}" if xs else "-"
    out.append({"episode": ep, "side": r["arm_side"], "split": r["split"], "vla_only": f(e0), "ag3s_no_hurdle": f(ag), "ag3s_hurdle": f(hu),
                "sources": sorted({x["src"] for x in r["results"]}), "results": r["results"]})
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{'ep':>5} {'side':>5} {'split':>5} {'VLA':>5} {'AG3S':>6} {'hurdle':>7}  sources")
for o in out:
    print(f"{o['episode']:>5} {str(o['side']):>5} {str(o['split']):>5} {o['vla_only']:>5} {o['ag3s_no_hurdle']:>6} {o['ag3s_hurdle']:>7}  {','.join(o['sources'])}")
