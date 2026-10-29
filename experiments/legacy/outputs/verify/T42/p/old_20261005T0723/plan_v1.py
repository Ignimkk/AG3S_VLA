"""T42-1 run planner (verifier, 2026-10-05). Deterministic from candidates.json + the finished runs; prints chain lists.
usage: plan.py PHASE NCHAINS   -> writes p/lists/<PHASE>_<i>.txt (i = 0..NCHAINS-1), items EP:SEED:noobs, prints a summary.
seed = 10 * ep + rep.
  p1  : E3b screening. Candidates per (fruit, arm) from candidates.json 'selected'; rep 1, except apple + left (T39 episodes
        with E0 and E3b success on T39 seeds rep 1 and 2) -> rep 3. Plus the T39 reproduction pair E3b ep1828 s18281.
  p2a : E3b rep 2 for every non-apple-left screening episode whose rep-1 run succeeded.
  p2b : E3b rep 3 for those whose rep 1 and rep 2 succeeded.
  p3  : E0 on the selected library episodes (see select()), rep 1-3 (apple + left: rep 3 only, rep 1-2 are T39), plus the
        T39 reproduction run E0 ep1828 s18281.
select(): per (fruit, arm), episodes with E3b 3/3 (apple + left: T39 rep 1-2 + this rep 3), best crowding score first;
          per fruit keep up to 2: the best of each arm, if one arm has none the next best of the other arm.
Items are interleaved across chains by fruit so every fruit gets early results."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402

T42 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42'))
RUNS = T42 / "1" / "runs"
CAND = json.loads((T42 / "1" / "candidates.json").read_text())
T39M = json.loads(pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T39.verify.json')).read_text())["numbers"]["matrix"]
FRUITS = ("apple", "orange", "banana", "pear"); ARMS = ("left", "right")
REPRO = (1828, 18281)
_cache = {}


def result(cond, ep, seed):
    """True / False for a finished run (check_run passed), None if not done."""
    d = RUNS / cond / "noobs"; name = f"{cond}_ep{ep}_s{seed}"
    if not (d / f"{name}.done").exists():
        return None
    if (cond, ep, seed) not in _cache:
        _cache[(cond, ep, seed)] = bool(AN.analyze(d / name)["success"])
    return _cache[(cond, ep, seed)]


def t39(cond, ep, seed):
    for m in T39M:
        if m["episode"] == ep and m["key"] == seed:
            return bool(m[cond]["success"])
    return None


def e3b_seeds(fruit, arm, ep):
    """[(seed, success or None, source)] for rep 1..3."""
    out = []
    for rep in (1, 2, 3):
        s = 10 * ep + rep
        if (fruit, arm) == ("apple", "left") and rep < 3:
            out.append((s, t39("E3b", ep, s), "T39"))
        else:
            out.append((s, result("E3b", ep, s), "T42"))
    return out


def screening():
    return {(f, a): CAND["candidates"][f"{f}_{a}"]["selected"] for f in FRUITS for a in ARMS}


def score(ep):
    return CAND["episodes"][str(ep)]["score_m"]


def library():
    lib = {}
    for (f, a), eps in screening().items():
        lib[(f, a)] = sorted([e for e in eps if all(x[1] is True for x in e3b_seeds(f, a, e))], key=lambda e: (-score(e), e))
    return lib


def select():
    lib = library(); sel = {}
    for f in FRUITS:
        pick = [lib[(f, a)][0] for a in ARMS if lib[(f, a)]]
        if len(pick) < 2:
            rest = sorted([e for a in ARMS for e in lib[(f, a)] if e not in pick], key=lambda e: (-score(e), e))
            pick += rest[:2 - len(pick)]
        sel[f] = pick
    return sel


def arm_of(ep):
    return CAND["episodes"][str(ep)]["used_arm"]


def items(phase):
    sc = screening(); by_fruit = {f: [] for f in FRUITS}
    if phase == "p1":
        for (f, a), eps in sc.items():
            for e in eps:
                by_fruit[f].append(f"{e}:{10 * e + (3 if (f, a) == ('apple', 'left') else 1)}")
        by_fruit["apple"].append(f"{REPRO[0]}:{REPRO[1]}")
    elif phase in ("p2a", "p2b"):
        for (f, a), eps in sc.items():
            if (f, a) == ("apple", "left"):
                continue
            for e in eps:
                r = [x[1] for x in e3b_seeds(f, a, e)]
                if phase == "p2a" and r[0] is True:
                    by_fruit[f].append(f"{e}:{10 * e + 2}")
                if phase == "p2b" and r[0] is True and r[1] is True:
                    by_fruit[f].append(f"{e}:{10 * e + 3}")
    elif phase == "p3":
        for f, eps in select().items():
            for e in eps:
                reps = (3,) if (f, arm_of(e)) == ("apple", "left") else (1, 2, 3)
                by_fruit[f] += [f"{e}:{10 * e + r}" for r in reps]
        by_fruit["apple"].append(f"{REPRO[0]}:{REPRO[1]}")
    else:
        raise SystemExit(f"unknown phase {phase}")
    # interleave fruits
    seq = []; k = 0
    while any(by_fruit.values()):
        f = FRUITS[k % 4]; k += 1
        if by_fruit[f]:
            seq.append(by_fruit[f].pop(0))
    return seq


if __name__ == "__main__":
    phase, n = sys.argv[1], int(sys.argv[2])
    seq = items(phase)
    D = T42 / "p" / "lists"; D.mkdir(exist_ok=True)
    for i in range(n):
        part = [f"{x}:noobs" for x in seq[i::n]]
        (D / f"{phase}_{i}.txt").write_text(" ".join(part) + "\n")
        print(phase, i, len(part), " ".join(part))
    if phase == "p3":
        print("selected", select())
