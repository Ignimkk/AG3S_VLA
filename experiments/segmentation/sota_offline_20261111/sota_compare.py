"""(2026-11-11 dev-57) Paired comparison of AG3S candidates on the same episodes (plan §9.2).
Reads the latest non-INVALID run per run-id from the benchmark store.
usage: python sota_compare.py e1a <base_tag> <tag> [<tag> ...]      E1a 40: run-ids qe-<tag>-e1a-spatial-LI-t<k>
       python sota_compare.py ms  <tag> [<tag> ...]                 other-suite set: run-ids ms-<tag>-<suite>-L<lvl>
       python sota_compare.py sub <base_tag> <tag> [<tag> ...]      episode subset: run-ids sub-<tag>-e1a-spatial-LI-t<k>
Per arm: TSR / CAR / SSR counts, per-task (or per-cell) counts, discordant SSR pairs vs the base + exact McNemar p, and the
episodes whose outcome class differs from the base (SS safe success, S+C, F+C, F)."""
import collections, glob, json, math, sys

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-11_SOTA_개선/"


def load(pat):
    by = {}
    for d in sorted(glob.glob(B + pat)):
        name = d.split("/")[-1]
        if "INVALID" in name:
            continue
        by[name.split("_", 2)[2]] = d
    out = {}
    for d in by.values():
        try:
            lines = open(d + "/episodes.jsonl").read().splitlines()
        except FileNotFoundError:
            continue
        for l in lines:
            if not l.strip():
                continue
            e = json.loads(l)
            s = e.get("safelibero") or {}
            ok, col = bool(e["official_success"]), bool(s.get("collided"))
            cls = "SS" if ok and not col else "S+C" if ok else "F+C" if col else "F"
            key = (e["suite"].replace("safelibero_", ""), e.get("safety_level"), int(e["task_id"]), int(e["episode_idx"]))
            out[key] = dict(ok=ok, col=col, cls=cls, disp=1000 * float(s.get("max_displacement_m") or 0.0),
                            obst=str(s.get("obstacle", "")).replace("_obstacle_1", ""), status=e.get("status"),
                            steps=e.get("env_steps_excluding_wait"))
    return out


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


def summary(eps, keys):
    n = len(keys)
    return n, sum(eps[k]["ok"] for k in keys), sum(not eps[k]["col"] for k in keys), sum(eps[k]["cls"] == "SS" for k in keys)


def report(arms, group):
    base_tag, base = arms[0]
    res = {}
    for tag, eps in arms:
        keys = sorted(set(eps) & set(base))
        n, t, c, s = summary(eps, keys)
        b = sum(eps[k]["cls"] == "SS" and base[k]["cls"] != "SS" for k in keys)
        cc = sum(eps[k]["cls"] != "SS" and base[k]["cls"] == "SS" for k in keys)
        big = sum(eps[k]["disp"] > 50 for k in keys)
        print(f"== {tag}: n={n} TSR {t} · CAR {c} · SSR {s}  ({100*t/max(n,1):.1f} / {100*c/max(n,1):.1f} / {100*s/max(n,1):.1f} %)"
              f"  >50mm {big}  vs {base_tag}: SSR +{b} / -{cc} (McNemar p={mcnemar(b, cc):.3g})")
        cells = collections.OrderedDict()
        for k in keys:
            cells.setdefault(group(k), []).append(k)
        print("   " + " | ".join(f"{g}: {'·'.join(str(x) for x in summary(eps, ks))}" for g, ks in cells.items()) + "   (n·TSR·CAR·SSR)")
        res[tag] = dict(n=n, TSR=t, CAR=c, SSR=s, big=big, gained=b, lost=cc,
                        cells={str(g): summary(eps, ks) for g, ks in cells.items()})
        if tag != base_tag:
            for k in keys:
                if eps[k]["cls"] != base[k]["cls"]:
                    print(f"     {k[0]} L{k[1]} t{k[2]} e{k[3]:2d} {eps[k]['obst']:18s} {base[k]['cls']:3s} ({base[k]['disp']:.1f} mm) -> "
                          f"{eps[k]['cls']:3s} ({eps[k]['disp']:.1f} mm, {eps[k]['status']})")
    return res


def main():
    mode, tags = sys.argv[1], sys.argv[2:]
    if mode == "e1a":
        arms = [(t, load(f"2*_qe-{t}-e1a-spatial-LI-t*")) for t in tags]
        res = report(arms, lambda k: f"t{k[2]}")
    elif mode == "sub":   # episode subset (sota_sub_queue.sh): base = its full E1a run cut to the first subset's episodes
        subs = [(t, load(f"2*_sub-{t}-e1a-spatial-LI-t*")) for t in tags[1:]]
        full = load(f"2*_qe-{tags[0]}-e1a-spatial-LI-t*")
        keys = set().union(*[set(e) for _, e in subs])
        arms = [(tags[0], {k: v for k, v in full.items() if k in keys})] + subs
        res = report(arms, lambda k: f"t{k[2]}")
    else:
        arms = [(t, load(f"2*_ms-{t}-*")) for t in tags]
        res = report(arms, lambda k: f"{k[0]} L{k[1]}")
    json.dump(res, open(OUT + f"compare_{mode}_{'_'.join(tags)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
