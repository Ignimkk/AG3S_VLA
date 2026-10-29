"""T43 TP — aggregate tp_timing JSONs: per setting wall (server-equivalent) median / p90 / max, per-stage
median and mean (exclusive ms), paired ratio vs V11AB, bit-identity of TPfe / V8fe, pod load.

usage: timing_agg.py OUT_JSON LOAD_FILE GLOB [GLOB ...]
First chunk of every run excluded (as the server-record stats: no warm start, OSQP setup).
"""
import glob
import json
import sys

import numpy as np

STAGES = ("fk", "esdf", "clearance", "linearize", "qp_build", "osqp", "objective", "other")


def st(a, nd=1):
    a = np.asarray([x for x in a if x is not None], float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median": round(float(np.median(a)), nd),
            "p90": round(float(np.quantile(a, .9)), nd), "p99": round(float(np.quantile(a, .99)), nd),
            "max": round(float(a.max()), nd), "mean": round(float(a.mean()), nd)}


def main():
    out_path, load_file = sys.argv[1], sys.argv[2]
    files = sorted({f for g in sys.argv[3:] for f in glob.glob(g)})
    runs = [json.load(open(f)) for f in files]
    names = list(runs[0]["variants"])
    acc = {v: {"wall": [], "cpu": [], "raw": [], "st": {k: [] for k in STAGES}, "calls": {},
               "sha_same_v11": 0, "sha_same_v8": 0, "n": 0, "ratio": [], "gate_open": [],
               "status": {}, "qp_it": []} for v in names}
    heavy = {v: [] for v in names}            # chunks where V11AB's look-ahead gate was open
    for r in runs:
        chunks = [c for c in r["chunks"] if c.get("variants")][1:]
        for c in chunks:
            V = c["variants"]
            for v in names:
                x = V[v]
                a = acc[v]
                w = float(np.median(x["wall_ms"]))
                a["wall"].append(w)
                a["cpu"].append(float(np.median(x["cpu_ms"])))
                a["raw"].append(float(np.median(x["wall_raw_ms"])))
                for k in STAGES:
                    a["st"][k].append(float(x["stages"][0].get(k, 0.0)))
                for k, n in x["calls"][0].items():
                    a["calls"].setdefault(k, []).append(n)
                a["n"] += 1
                a["sha_same_v11"] += x["sha1"] == V.get("V11AB", {}).get("sha1")
                a["sha_same_v8"] += x["sha1"] == V.get("V8", {}).get("sha1")
                if "V11AB" in V:
                    a["ratio"].append(w / float(np.median(V["V11AB"]["wall_ms"])))
                a["gate_open"].append(1.0 if x.get("gate") == "open" else 0.0)
                a["status"][x["status"]] = a["status"].get(x["status"], 0) + 1
                a["qp_it"].append(x.get("qp_iterations"))
                if V.get("V11AB", {}).get("gate") == "open":
                    heavy[v].append(w)
    loads = []
    try:
        for line in open(load_file):
            p = line.split()
            kv = dict(t.split("=") for t in p[4:])
            loads.append((int(p[0]), float(p[1]), int(kv.get("nr_periods", 0)),
                          int(kv.get("nr_throttled", 0)), int(kv.get("usage_usec", 0))))
    except OSError:
        pass
    load = {}
    if len(loads) > 1:
        L = np.asarray(loads, float)
        dt = L[-1, 0] - L[0, 0]
        load = {"n_samples": len(L), "seconds": dt, "load1": st(L[:, 1], 2),
                "throttled_share_of_periods": round(float((L[-1, 3] - L[0, 3]) /
                                                          max(1.0, L[-1, 2] - L[0, 2])), 4),
                "cgroup_cores_used_mean": round(float((L[-1, 4] - L[0, 4]) / 1e6 / max(dt, 1)), 2)}
    table = {}
    for v in names:
        a = acc[v]
        table[v] = {"n_chunks": a["n"], "wall_ms": st(a["wall"]), "cpu_ms": st(a["cpu"]),
                    "wall_raw_ms": st(a["raw"]),
                    "wall_ms_gate_open_chunks_of_V11AB": st(heavy[v]),
                    "stage_median_ms": {k: round(float(np.median(a["st"][k])), 2) for k in STAGES},
                    "stage_mean_ms": {k: round(float(np.mean(a["st"][k])), 2) for k in STAGES},
                    "stage_p90_ms": {k: round(float(np.quantile(a["st"][k], .9)), 2) for k in STAGES},
                    "calls_median": {k: float(np.median(n)) for k, n in a["calls"].items()},
                    "ratio_vs_V11AB": st(a["ratio"], 3),
                    "bit_identical_to_V11AB": a["sha_same_v11"], "bit_identical_to_V8": a["sha_same_v8"],
                    "gate_open_share": round(float(np.mean(a["gate_open"])), 4),
                    "status": a["status"], "qp_iterations": st(a["qp_it"], 0)}
    res = {"files": files, "n_runs": len(runs), "field": runs[0].get("field"), "mode": runs[0].get("mode"),
           "load": load, "load_before_after_per_run": [(r["load_before"].get("loadavg"),
                                                        r["load_after"].get("loadavg")) for r in runs],
           "table": table}
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"runs {len(runs)} field {res['field']} load {load}")
    print(f"{'setting':24s} {'med':>6} {'p90':>6} {'p99':>6} {'max':>6} | {'xV11':>5} {'heavy p90':>9} | " +
          " ".join(f"{k[:6]:>6}" for k in STAGES) + " | ident")
    for v in names:
        t = table[v]
        print(f"{v:24s} {t['wall_ms']['median']:6.1f} {t['wall_ms']['p90']:6.1f} {t['wall_ms']['p99']:6.1f} "
              f"{t['wall_ms']['max']:6.1f} | {t['ratio_vs_V11AB'].get('median', 0):5.2f} "
              f"{t['wall_ms_gate_open_chunks_of_V11AB'].get('p90', 0):9.1f} | " +
              " ".join(f"{t['stage_mean_ms'][k]:6.1f}" for k in STAGES) +
              f" | {t['bit_identical_to_V11AB']}/{t['n_chunks']}")


if __name__ == "__main__":
    main()
