"""T43 LB3 run planner (verifier, 2026-10-09). Same judgement as plan_lb.py (run ok = analyze_t42 success AND grasp_robust robust_grasp = definition B,
seed = 10*ep + rep). Groups: banana_left, banana_right, pear_left only. Results are read from LB3/runs/E0 and, for episodes LB already ran, from LB/runs/E0.
Order per group (candidates_lb3.json): banana = ranks 11-30 then random order (seed in json); pear_left = ranks 11-40. Cap: LB tried + LB3 tried <= 40.
Stop: a group with >= 2 episodes 3/3 (LB + LB3 combined) gets no more rep1 / rep2 / rep3 items.
usage: plan_lb3.py s N | c2 | c3 | status   (writes lists/<tag>_<i>.txt, 2 chains)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/LB/p'))
import analyze_t42 as AN  # noqa: E402
import grasp_robust as GR  # noqa: E402
LB = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/LB')); L3 = LB / "LB3"
DIRS = [L3 / "runs" / "E0", LB / "runs" / "E0"]
C = json.loads((L3 / "candidates_lb3.json").read_text())
GROUPS = ("banana_left", "banana_right", "pear_left")
CAP = 40
_c = {}

def rundir(ep, seed):
    for D in DIRS:
        if (D / f"E0_ep{ep}_s{seed}.done").exists():
            return D
    return None

def result(ep, seed):
    D = rundir(ep, seed)
    if D is None:
        return None
    name = f"E0_ep{ep}_s{seed}"; d = D / name
    if name not in _c:
        rj = D / f"{name}.robust.json"
        if rj.exists() and rj.stat().st_mtime >= (d / "frames.jsonl").stat().st_mtime:
            r = json.loads(rj.read_text())
        else:
            r = GR.robust(d); r.pop("series", None); rj.write_text(json.dumps(r, indent=1, default=float))
        a = AN.analyze(d)
        _c[name] = {"seed": seed, "success": bool(a["success"]), "grasp": bool(a["grasp"]), "place": bool(a["place"]), "robust": bool(r["robust_grasp"]),
                    "robust_strict": bool(r["robust_grasp_strict"]), "ok": bool(a["success"] and r["robust_grasp"]), "failure_mode": GR.failure_mode(r),
                    "source": "LB" if D == DIRS[1] else "LB3", "run_dir": str(d)}
    return _c[name]

def reps(ep):
    return [result(ep, 10 * ep + k) for k in (1, 2, 3)]

def order(g):
    v = C["groups"][g]
    return v["lb3_ranked_11_40"][:20] + v["random_order_pool_minus_ranks_1_30"][:10] if g.startswith("banana") else v["lb3_ranked_11_40"]

def lb_tried(g):
    return [e for e in C["groups"][g]["ranked_all"] if reps(e)[0] is not None and rundir(e, 10 * e + 1) == DIRS[1]]

def all_tried(g):
    return [e for e in C["groups"][g]["ranked_all"] if reps(e)[0] is not None]

def three(g):
    return [e for e in all_tried(g) if all(x is not None and x["ok"] for x in reps(e))]

def items(phase, n):
    by = {}
    for g in GROUPS:
        by[g] = []
        if len(three(g)) >= 2:
            continue
        if phase == "s":
            room = CAP - len(all_tried(g))
            new = [e for e in order(g) if reps(e)[0] is None]
            by[g] = [f"{e}:{10*e+1}" for e in new[:min(n, room)]]
        else:
            for e in all_tried(g):
                r = reps(e)
                if phase == "c2" and r[0]["ok"] and r[1] is None:
                    by[g].append(f"{e}:{10*e+2}")
                if phase == "c3" and r[0]["ok"] and r[1] and r[1]["ok"] and r[2] is None:
                    by[g].append(f"{e}:{10*e+3}")
    seq = []; k = 0
    while any(by.values()):
        g = GROUPS[k % 3]; k += 1
        if by[g]:
            seq.append(by[g].pop(0))
    return seq

def status():
    for g in GROUPS:
        row = []
        for e in all_tried(g):
            r = reps(e)
            row.append(f"{e}{'*' if rundir(e,10*e+1)==DIRS[1] else ''}:" + "".join("." if x is None else ("S" if x["ok"] else "x") for x in r))
        print(g, "LB-tried", len(lb_tried(g)), "tried", len(all_tried(g)), "3/3", three(g), "|", " ".join(row))

if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "status":
        status(); sys.exit(0)
    n = int(sys.argv[2]) if phase == "s" else 0
    tag = sys.argv[3] if len(sys.argv) > 3 else (phase + (str(n) if n else ""))
    seq = items(phase, n); D = L3 / "p" / "lists"
    for i in range(2):
        part = seq[i::2]
        (D / f"{tag}_{i}.txt").write_text(" ".join(part) + "\n")
        print(tag, i, len(part), " ".join(part))
