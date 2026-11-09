"""(2026-11-09 dev-4a) LIBERO-Safety 10 (L0 seed 7: human_safety 4, obstacle_avoidance 4, affordance 2) per version tag:
success, official violation, safe success. usage: python ls_summary.py <tag> [<tag> ...]"""
import glob, json, sys
for tag in sys.argv[1:]:
    n = ok = viol = safe = 0
    for suite in ("human_safety", "obstacle_avoidance", "affordance"):
        runs = sorted(p for p in glob.glob(f"/mnt/dev/AG3S_VLA-storage/results/benchmark/2*gtto-{tag}-ls10-{suite}-*")
                      if "INVALID" not in p)
        if not runs:
            continue
        for l in open(runs[-1] + "/episodes.jsonl"):
            e = json.loads(l)
            ls = e.get("libero_safety") or {}
            if isinstance(ls, str):
                ls = {}
            n += 1
            ok += bool(e["official_success"])
            viol += bool(ls.get("official_violation"))
            safe += bool(ls.get("safe_success"))
    print(f"{tag:12s} n {n} success {ok} violation {viol} safe-success {safe}")
