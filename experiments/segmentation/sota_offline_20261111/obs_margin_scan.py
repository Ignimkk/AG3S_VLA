"""(2026-11-11 dev-57, offline) Obstacle-row clearance what-if on recorded pe123458 calls.
For every recorded call: passthrough_clear.applied, obstacle_rows.violation_m and worst.clearance_m.
Prints, per bench, the number of pass-through calls that a stricter rule would send to the action QP, and for the E1a
collision episodes the smallest recorded obstacle-row value before the collision call.
usage: python obs_margin_scan.py <tag>   (run-ids gtto-<tag>-*, qe-<tag>-e1a-*)"""
import ast, collections, glob, json, sys
import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"


def latest(pat):
    by = {}
    for d in sorted(glob.glob(B + pat)):
        if "INVALID" in d:
            continue
        rid = d.split("/")[-1].split("_", 2)[2]
        by[rid] = d
    return sorted(by.values())


def calls(d):
    out = []
    for c in sorted(glob.glob(d + "/server_constraints/run_*/chunk_*.npz")):
        sm = json.loads(str(np.load(c)["summary_json"]))
        m = ((sm.get("to") or {}).get("metrics") or {})
        pc = m.get("passthrough_clear")
        if isinstance(pc, str):
            try:
                pc = ast.literal_eval(pc)
            except Exception:
                pc = {}
        pc = pc or {}
        orow = pc.get("obstacle_rows") or {}
        w = orow.get("worst") or {}
        out.append(dict(file=c, applied=bool(pc.get("applied")), reason=pc.get("reason"),
                        viol=1000 * float(orow.get("violation_m") or 0.0),
                        clr=None if w.get("clearance_m") is None else 1000 * float(w["clearance_m"]),
                        link=w.get("link"), step=w.get("step"), nrows=w.get("n_obstacle_rows")))
    return out


def main(tag):
    benches = {"LIBERO27": f"*_gtto-{tag}-libero9s[789]-*", "SafeLIBERO11": f"*_gtto-{tag}-safelibero11-t*",
               "LS10": f"*_gtto-{tag}-ls10-*", "E1a40": f"*_qe-{tag}-e1a-spatial-LI-t*"}
    res = {}
    for b, pat in benches.items():
        tot = collections.Counter(); clrs = []
        for d in latest(pat):
            for c in calls(d):
                tot["calls"] += 1
                tot["reason:" + str(c["reason"])] += 1
                if not c["applied"]:
                    continue
                tot["pass"] += 1
                for x in (8, 5, 2, 0):
                    tot[f"viol>{x}"] += c["viol"] > x
                if c["clr"] is not None:
                    clrs.append(c["clr"])
        res[b] = dict(tot)
        q = np.percentile(clrs, [0, 1, 5, 10, 25, 50]) if clrs else []
        print(b, dict(tot)); print("   pass-through worst.clearance_m [mm] n", len(clrs), "pct 0/1/5/10/25/50:", np.round(q, 1))
        if clrs:
            a = np.array(clrs)
            print("   pass calls with clearance <", {x: int((a < x).sum()) for x in (0, 5, 10, 15, 20, 30)})
    json.dump(res, open(f"/mnt/dev/AG3S_VLA/outputs/2026-11-11_SOTA_개선/offline/obs_margin_scan_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
