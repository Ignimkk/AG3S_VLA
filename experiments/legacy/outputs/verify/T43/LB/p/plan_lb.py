"""T43 LB run planner (verifier, 2026-10-09). Deterministic from ../candidates_lb.json + the finished E0 runs.
seed = 10 * ep + rep (T42 rule). A run is 'ok' = E0 success (analyze_t42: grasp and place) AND robust grasp (B; grasp_robust.py).
usage: plan_lb.py PHASE ARG NCHAINS
  s LO:HI : rep 1 of the candidates ranked LO..HI-1 (0-based) of every fruit x arm that is not yet adopted-2 (adopted = rep 1-3 all ok)
  c2      : rep 2 for every candidate whose rep 1 is ok and rep 2 is not done
  c3      : rep 3 for every candidate whose rep 1 and rep 2 are ok and rep 3 is not done
  status  : print the table
writes lists/<PHASE>_<i>.txt (items EP:SEED, fruits round robin, split over NCHAINS)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/LB/p'))
import analyze_t42 as AN  # noqa: E402
import grasp_robust as GR  # noqa: E402
LB = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/LB'))
RUNS = LB / "runs" / "E0"
CAND = json.loads((LB / "candidates_lb.json").read_text())
FRUITS = ("apple", "orange", "pear", "banana"); ARMS = ("left", "right")
_c = {}

def result(ep, seed):
    name = f"E0_ep{ep}_s{seed}"; d = RUNS / name
    if not (RUNS / f"{name}.done").exists():
        return None
    if name not in _c:
        rj = RUNS / f"{name}.robust.json"
        if rj.exists() and rj.stat().st_mtime >= (d / "frames.jsonl").stat().st_mtime:
            r = json.loads(rj.read_text())
        else:
            r = GR.robust(d); r.pop("series", None); rj.write_text(json.dumps(r, indent=1, default=float))
        a = AN.analyze(d)
        _c[name] = {"seed": seed, "success": bool(a["success"]), "grasp": bool(a["grasp"]), "place": bool(a["place"]),
                    "robust": bool(r["robust_grasp"]), "robust_strict": bool(r["robust_grasp_strict"]),
                    "ok": bool(a["success"] and r["robust_grasp"]), "failure_mode": GR.failure_mode(r)}
    return _c[name]

def reps(ep):
    return [result(ep, 10 * ep + k) for k in (1, 2, 3)]

def adopted(f, a, eps):
    return [e for e in eps if all(x is not None and x["ok"] for x in reps(e))]

def groups():
    return {(f, a): CAND["candidates"][f"{f}_{a}"]["ranked"] for f in FRUITS for a in ARMS}

def items(phase, arg):
    by = {f: [] for f in FRUITS}
    import os
    only = [x for x in os.environ.get("ONLY_GROUPS", "").split(",") if x]
    for (f, a), ranked in groups().items():
        if only and f"{f}_{a}" not in only:
            continue
        if phase == "s":
            lo, hi = map(int, arg.split(":"))
            if len(adopted(f, a, ranked)) >= 2:
                continue
            by[f] += [f"{e}:{10*e+1}" for e in ranked[lo:hi] if result(e, 10*e+1) is None]
        else:
            for e in ranked:
                r = reps(e)
                if phase == "c2" and r[0] and r[0]["ok"] and r[1] is None:
                    by[f].append(f"{e}:{10*e+2}")
                if phase == "c3" and r[0] and r[0]["ok"] and r[1] and r[1]["ok"] and r[2] is None:
                    by[f].append(f"{e}:{10*e+3}")
    seq = []; k = 0
    while any(by.values()):
        f = FRUITS[k % 4]; k += 1
        if by[f]:
            seq.append(by[f].pop(0))
    return seq

def status():
    for (f, a), ranked in groups().items():
        row = []
        for i, e in enumerate(ranked):
            r = reps(e)
            if all(x is None for x in r):
                continue
            row.append(f"{e}:" + "".join("." if x is None else ("S" if x["ok"] else "x") for x in r))
        print(f"{f}_{a}", "adopted", adopted(f, a, ranked), "|", " ".join(row))

if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "status":
        status(); sys.exit(0)
    arg = sys.argv[2] if phase == "s" else ""
    n = int(sys.argv[3] if phase == "s" else sys.argv[2])
    seq = items(phase, arg); D = LB / "p" / "lists"
    tag = phase + (arg.replace(":", "_") if arg else "")
    for i in range(n):
        part = seq[i::n]
        (D / f"{tag}_{i}.txt").write_text(" ".join(part) + "\n")
        print(tag, i, len(part), " ".join(part))
