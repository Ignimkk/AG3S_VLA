"""SUBTASK-extra M1 — after latch PLACED, before first switch: chunk count and exclusion.active count.
Also per-episode PLACED / switch / new-target classification (same rule as
AG3S/ag3s/experiments/tools/scan_placed_switch.py: xy distance to destination centroid < 0.15 m).
Read-only over T34/T37 E3a/E3b server_constraints."""
import glob, json, collections, sys
import numpy as np

ROOTS = {
    "T37/E3a": "outputs/verify/T37/E3a_rec/server_8204/server_constraints",
    "T37/E3b": "outputs/verify/T37/E3b_rec/server_8206/server_constraints",
    "T34/E3a": "outputs/verify/T34/E3a_rec/server_8204/server_constraints",
    "T34/E3b": "outputs/verify/T34/E3b_rec/server_8206/server_constraints",
}

def main():
    out = dict(per_root={}, episodes=[])
    tot = collections.Counter()
    for name, root in ROOTS.items():
        c = collections.Counter()
        for run in sorted(glob.glob(f"{root}/run_*")):
            files = sorted(glob.glob(f"{run}/chunk_*.npz"))
            prev = None; ep = 0; cur = None
            def new_ep(i):
                return dict(root=name, run=run.split('/')[-1], ep=i, n_chunks=0, first_chunk_index=None,
                            t_placed=None, before_switch=0, before_switch_excl_active=0,
                            first_switch=None, after_switch=0, after_switch_excl_active=0,
                            after_switch_excl=collections.Counter())
            for f in files:
                with np.load(f, allow_pickle=False) as z:
                    s = json.loads(str(z["summary_json"]))
                t = int(s["t_step"])
                if prev is None or t <= prev:
                    if cur is not None: out["episodes"].append(cur)
                    cur = new_ep(ep if prev is None else cur["ep"] + 1)
                    cur["first_chunk_index"] = int(s["chunk_index"])
                prev = t
                cur["n_chunks"] += 1
                if (s.get("grasp") or {}).get("state") == "placed" and cur["t_placed"] is None:
                    cur["t_placed"] = t
                if cur["t_placed"] is None:
                    continue
                dec = (s.get("admissibility") or {}).get("decision") or {}
                ex = s.get("exclusion") or {}
                if dec.get("mode") == "switch" and cur["first_switch"] is None:
                    m = s.get("manipulated") or {}; d = s.get("destination") or {}
                    cm = np.asarray(m.get("centroid"), float)
                    dc = np.asarray(d.get("centroid") or [np.nan]*3, float)
                    dxy = float(np.linalg.norm(cm[:2]-dc[:2]))
                    cur["first_switch"] = dict(t=t, chunk_index=int(s["chunk_index"]), dt=t-cur["t_placed"],
                        id=m.get("id"), switched_from=m.get("switched_from"),
                        centroid=np.round(cm, 4).tolist(), destination_centroid=np.round(dc, 4).tolist(),
                        xy_to_destination_m=round(dxy, 4),
                        kind="in_destination" if dxy < 0.15 else "other_object")
                if cur["first_switch"] is None:
                    cur["before_switch"] += 1
                    cur["before_switch_excl_active"] += bool(ex.get("active"))
                    c[("before_switch", bool(ex.get("active")), ex.get("source"))] += 1
                else:
                    cur["after_switch"] += 1
                    cur["after_switch_excl_active"] += bool(ex.get("active"))
                    key = f'{ex.get("source")}|{bool(ex.get("active"))}|{"+".join(ex.get("mechanism") or ())}'
                    cur["after_switch_excl"][key] += 1
                    c[("after_switch", bool(ex.get("active")), ex.get("source"))] += 1
            if cur is not None: out["episodes"].append(cur)
        out["per_root"][name] = {"|".join(map(str, k)): v for k, v in sorted(c.items(), key=str)}
    eps = out["episodes"]
    for e in eps: e["after_switch_excl"] = dict(e["after_switch_excl"])
    placed = [e for e in eps if e["t_placed"] is not None]
    sw = [e for e in placed if e["first_switch"] is not None]
    agg_ex = collections.Counter()
    for e in sw:
        agg_ex.update(e["after_switch_excl"])
    out["totals"] = dict(
        episodes=len(eps), placed_episodes=len(placed), switched_after_placed=len(sw),
        kinds=dict(collections.Counter(e["first_switch"]["kind"] for e in sw)),
        before_switch_chunks=sum(e["before_switch"] for e in placed),
        before_switch_exclusion_active=sum(e["before_switch_excl_active"] for e in placed),
        after_switch_chunks=sum(e["after_switch"] for e in sw),
        after_switch_exclusion_active=sum(e["after_switch_excl_active"] for e in sw),
        after_switch_exclusion_keys=dict(agg_ex),
        per_root_placed={n: sum(1 for e in placed if e["root"] == n) for n in ROOTS},
        per_root_switched={n: sum(1 for e in sw if e["root"] == n) for n in ROOTS},
        per_root_episodes={n: sum(1 for e in eps if e["root"] == n) for n in ROOTS},
    )
    json.dump(out, open("outputs/verify/SUBTASK-extra/m1_placed.json", "w"), indent=1)
    print(json.dumps(out["totals"], indent=1))
    print(json.dumps(out["per_root"], indent=1))

if __name__ == "__main__":
    main()
