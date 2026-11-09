"""PE11 mechanism check (dev-4a, 2026-11-09 15:4x KST): pe1234578911 vs pe12345789 vs pe123458 on gt24att-failed tasks
(LIBERO t3,4,7,9 x seeds 9,7,8 + SafeLIBERO t3 dev ep 0,2,3,4, seed 7)."""
import glob, json, os
B = "/mnt/dev/AG3S_VLA-storage/results/benchmark"
def latest(rid):
    ds = [d for d in sorted(glob.glob(f"{B}/2026*_{rid}-*")) if "INVALID" not in d.split("/")[-1]]
    return ds[-1] if ds else None
out = {}
for tag in ("pe1234578911", "pe12345789", "pe123458"):
    row = {"libero": {}, "safelibero_t3": {}}
    for s in (9, 7, 8):
        d = latest(f"gtto-{tag}-libero4s{s}")
        if d:
            for l in open(d + "/episodes.jsonl"):
                r = json.loads(l); row["libero"][f"s{s}t{r['task_id']}"] = bool(r["official_success"])
    d = latest(f"gtto-{tag}-safelibero11-t3")
    if d:
        for l in open(d + "/episodes.jsonl"):
            r = json.loads(l); sl = r["safelibero"]
            if r["episode_idx"] in (0, 2, 3, 4):
                row["safelibero_t3"][f"e{r['episode_idx']}"] = dict(succ=bool(r["official_success"]), safe=bool(sl["safe_success"]),
                    coll_mm=round(1000 * sl["max_displacement_m"]) if sl.get("collided") else 0, obstacle=sl.get("obstacle"), run=os.path.basename(d)[:40])
    out[tag] = row
    lib = row["libero"]; st = row["safelibero_t3"]
    print(f"{tag}: LIBERO {sum(lib.values())}/{len(lib)} fails={[k for k,v in lib.items() if not v]}  "
          f"SafeLIBERO t3 succ {sum(v['succ'] for v in st.values())} safe {sum(v['safe'] for v in st.values())}/{len(st)} "
          f"coll={ {k:(v['coll_mm'],v['obstacle']) for k,v in st.items() if v['coll_mm']} }")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pe11_mech.json"), "w"), indent=1)
