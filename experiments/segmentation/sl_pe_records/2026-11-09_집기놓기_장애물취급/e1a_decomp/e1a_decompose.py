"""(2026-11-09 12:1x dev-4a) E1a 40 failure decomposition per episode: outcome class (safe success / success+collision /
fail+collision / fail no collision), when the collision happened relative to the grasp (pre-held / held / after release),
call paths (HOLD, verdict kinds, action QP used). Arms by run-id glob. usage: python3 e1a_decompose.py <tag> [<tag> ...]"""
import collections
import glob
import json
import sys

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"


def episodes(tag):
    out = {}
    for d in sorted(glob.glob(B + f"2*qe-{tag}-e1a-spatial-LI-t*")):
        if "INVALID" in d:
            continue
        for l in open(d + "/episodes.jsonl"):
            e = json.loads(l)
            out[(int(e["task_id"]), int(e["episode_idx"]))] = (d, e)
    return out


def calls_of(d, t, ep):
    try:
        return [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{ep}_calls.jsonl")]
    except FileNotFoundError:
        return []


def classify(e):
    s, c = bool(e["official_success"]), bool(e["safelibero"]["collided"])
    return {(True, False): "SS", (True, True): "S+C", (False, True): "F+C", (False, False): "F"}[(s, c)]


def row(d, e):
    t, ep = int(e["task_id"]), int(e["episode_idx"])
    cs = calls_of(d, t, ep)
    states = [((c.get("grasp") or {}).get("state"), c["obs_index"]) for c in cs]
    first_held = next((o for s, o in states if s == "held"), None)
    last_held = max((o for s, o in states if s == "held"), default=None)
    ct = e["safelibero"]["collide_t"]
    phase = "-"
    if ct is not None:
        phase = "pre" if first_held is None or ct < first_held else ("held" if last_held is None or ct <= last_held + 10 else "post")
    kinds = collections.Counter(r.get("kind") for c in cs for r in (c.get("verdict_reasons") or []))
    return dict(task=t, ep=ep, cls=classify(e), obst=e["safelibero"]["obstacle"].replace("_obstacle_1", ""),
                disp_mm=round(1000 * e["safelibero"]["max_displacement_m"], 1), collide_t=ct, phase=phase,
                steps=e["env_steps_excluding_wait"], n_calls=len(cs), held=first_held is not None, first_held=first_held,
                hold=sum(bool(c["hold"]) for c in cs), qp=sum((c.get("max_abs_action_change_executed") or 0) > 0 for c in cs),
                kinds=dict(kinds))


def main():
    tags = sys.argv[1:]
    data = {tg: {k: row(*v) for k, v in episodes(tg).items()} for tg in tags}
    for tg in tags:
        cnt = collections.Counter(r["cls"] for r in data[tg].values())
        ph = collections.Counter((r["cls"], r["phase"]) for r in data[tg].values() if r["phase"] != "-")
        ob = collections.Counter((r["obst"], r["cls"]) for r in data[tg].values())
        print(f"== {tg}: n={len(data[tg])} {dict(cnt)} collision phase {dict(ph)}")
        print("   per obstacle:", dict(sorted(ob.items())))
        print("   held reached in failures:", sum(r["held"] for r in data[tg].values() if r["cls"] in ("F", "F+C")),
              "/", sum(r["cls"] in ("F", "F+C") for r in data[tg].values()))
    keys = sorted(set().union(*[set(v) for v in data.values()]))
    print("task ep | " + " | ".join(tags))
    for k in keys:
        cells = []
        for tg in tags:
            r = data[tg].get(k)
            cells.append("-" if r is None else f"{r['cls']}:{r['obst'][:10]}:{r['phase']}:{r['disp_mm']}mm:h{r['hold']}:q{r['qp']}")
        print(f"t{k[0]} e{k[1]:2d} | " + " | ".join(cells))
    json.dump({tg: {f"t{k[0]}e{k[1]}": v for k, v in d.items()} for tg, d in data.items()},
              open("/mnt/dev/AG3S_VLA/outputs/2026-11-09_집기놓기_장애물취급/e1a_decomp/e1a_decomp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
