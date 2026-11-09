"""pe123458-tro vs pe123458 (dev-4a, 2026-11-09 15:2x KST): SafeLIBERO 11 dev, LIBERO-Safety 10, E1a 40 (init 5-14, dev-used).
Reads episodes.jsonl of the latest non-INVALID run per run-id; paired McNemar (exact) on safe success."""
import glob, json, math, os, sys
B = "/mnt/dev/AG3S_VLA-storage/results/benchmark"

def runs(pattern):
    out = {}
    for d in sorted(glob.glob(f"{B}/2026*_{pattern}")):
        rid = d.split("_", 2)[2].split("-vla_v11ab")[0]
        out[rid] = d                       # latest wins (sorted by timestamp)
    return out

def eps(pattern):
    rec = {}
    for rid, d in runs(pattern).items():
        for line in open(f"{d}/episodes.jsonl"):
            r = json.loads(line)
            sl = r.get("libero_safety") or r.get("safelibero") or {}
            key = (r.get("level") or r["suite"], r["task_id"], r["episode_idx"])
            rec[key] = dict(succ=bool(r["official_success"]), coll=bool(sl.get("collided") or sl.get("official_violation")),
                            safe=bool(sl.get("safe_success")), disp=sl.get("max_displacement_m"), status=r["status"])
    return rec

def mcnemar(a, b):
    n01 = sum(1 for k in a if a[k] and not b[k]); n10 = sum(1 for k in a if b[k] and not a[k]); n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, p

res = {}
for name, tro_pat, base_pat in [("SafeLIBERO 11", "gtto-pe123458tro-safelibero11-t*", "gtto-pe123458-safelibero11-t*"),
                                ("LIBERO-Safety 10", "gtto-pe123458tro-ls10-*", "gtto-pe123458-ls10-*"),
                                ("E1a 40 (init 5-14)", "qe-pe123458tro-e1a-spatial-LI-t*", "qe-pe123458-e1a-spatial-LI-t*")]:
    t, b = eps(tro_pat), eps(base_pat)
    keys = sorted(set(t) & set(b))
    row = {}
    for arm, r in (("tro", t), ("base", b)):
        row[arm] = dict(n=len(keys), TSR=sum(r[k]["succ"] for k in keys), CAR=sum(not r[k]["coll"] for k in keys),
                        SSR=sum(r[k]["safe"] for k in keys),
                        per_task_SSR={tk: sum(r[k]["safe"] for k in keys if k[1] == tk) for tk in sorted({k[1] for k in keys})})
    n01, n10, p = mcnemar({k: t[k]["safe"] for k in keys}, {k: b[k]["safe"] for k in keys})
    row["SSR_tro_only"], row["SSR_base_only"], row["mcnemar_p"] = n01, n10, round(p, 4)
    row["changed"] = [dict(key=list(k), tro=t[k], base=b[k]) for k in keys if (t[k]["safe"], t[k]["succ"]) != (b[k]["safe"], b[k]["succ"])]
    res[name] = row
    print(f"{name}: n={len(keys)}  tro TSR/CAR/SSR {row['tro']['TSR']}/{row['tro']['CAR']}/{row['tro']['SSR']}  "
          f"base {row['base']['TSR']}/{row['base']['CAR']}/{row['base']['SSR']}  SSR tro-only {n01} base-only {n10} p={p:.4f}  "
          f"per-task SSR tro {row['tro']['per_task_SSR']} base {row['base']['per_task_SSR']}")
json.dump(res, open(os.path.join(os.path.dirname(__file__), "tro_compare.json"), "w"), indent=1, default=str)
