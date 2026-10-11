"""T42-1 run planner v2 (verifier, 2026-10-05; v1 phases p1 / p2a were run, p2b superseded by the user's robust-grasp
instruction). Deterministic from candidates.json + candidates_ext.json + the finished runs; prints chain lists.
usage: plan.py PHASE NCHAINS   -> writes p/lists/<PHASE>_<i>.txt (items EP:SEED:noobs), prints a summary.   seed = 10 * ep + rep.
"ok" seed = E3b success (analyze_t42) AND robust grasp (grasp_robust.py: grasp, no slip, no regrasp).
Candidates = candidates.json 'selected' (test split) + candidates_ext.json 'selected' (train / validation; apple + left T39 ep 1939).
apple + left (T39 episodes): rep 1-2 = the T39 runs (success from T39.verify.json, robust from grasp_robust on the T39 run dir);
rep 3 here.
  p1, p2a : v1 phases (kept for the record; p1 = test-split screening rep 1 (apple + left rep 3) + T39 reproduction E3b ep1828
            s18281; p2a = rep 2 for v1-successful rep 1)
  p2b     : superseded (exits 1; stops the v1 launcher)
  x1      : extension screening: rep 1 of the candidates_ext episodes (apple + left 1939: rep 3)
  x2      : rep 2 for every non-apple-left candidate whose rep 1 is ok and whose rep 2 is not done
  x3      : rep 3 for every non-apple-left candidate whose rep 1 and rep 2 are ok; plus rep 1 of extension episodes added after x1
            (banana 4 -> 7 per arm); their rep 2-3 need a later x2 / x3 pass
  e0      : E0 on the selected episodes, rep 1-3 (apple + left: rep 3 only), plus the T39 reproduction run E0 ep1828 s18281
library(): per (fruit, arm), episodes with rep 1-3 all ok, best crowding score first.
select(): per fruit up to 2: the best of each arm; if one arm has none, the next best of the other arm; if the total is < 5,
          further library episodes (best score first, round robin over fruits) until 5."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402
import grasp_robust as GR  # noqa: E402

T42 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42'))
RUNS = T42 / "1" / "runs"
CAND = json.loads((T42 / "1" / "candidates.json").read_text())
EXTC = json.loads((T42 / "1" / "candidates_ext.json").read_text())
T39M = json.loads(pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T39.verify.json')).read_text())["numbers"]["matrix"]
T39R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs'))
FRUITS = ("apple", "orange", "banana", "pear"); ARMS = ("left", "right")
REPRO = (1828, 18281)
_cache = {}


def robust_cached(rd):
    rd = pathlib.Path(rd); f = rd.parent / f"{rd.name}.robust.json"
    if f.exists() and f.stat().st_mtime >= (rd / "frames.jsonl").stat().st_mtime:
        return json.loads(f.read_text())
    r = GR.robust(rd); r.pop("series", None)
    f.write_text(json.dumps(r, indent=1, default=float))
    return r


def result(cond, ep, seed):
    """dict {success, robust} for a finished run here (check_run passed), None if not done."""
    d = RUNS / cond / "noobs"; name = f"{cond}_ep{ep}_s{seed}"
    if not (d / f"{name}.done").exists():
        return None
    if (cond, ep, seed) not in _cache:
        a = AN.analyze(d / name); r = robust_cached(d / name)
        _cache[(cond, ep, seed)] = {"success": bool(a["success"]), "grasp": bool(a["grasp"]), "robust": bool(r["robust_grasp"]), "source": "T42"}
    return _cache[(cond, ep, seed)]


def t39(cond, ep, seed):
    name = f"{cond}_ep{ep}_s{seed}"; rd = T39R / cond / name
    if not (T39R / cond / f"{name}.done").exists():
        return None
    if ("T39", cond, ep, seed) not in _cache:
        a = AN.analyze(rd); r = robust_cached(rd)
        _cache[("T39", cond, ep, seed)] = {"success": bool(a["success"]), "grasp": bool(a["grasp"]), "robust": bool(r["robust_grasp"]), "source": "T39"}
    return _cache[("T39", cond, ep, seed)]


def ok(r):
    return r is not None and r["success"] and r["robust"]


def screening():
    out = {(f, a): list(CAND["candidates"][f"{f}_{a}"]["selected"]) for f in FRUITS for a in ARMS}
    for k, v in EXTC["candidates"].items():
        f, a = k.split("_")
        out[(f, a)] += [e for e in v["selected"] if e not in out[(f, a)]]
    return out


def ext_eps():
    return {e for v in EXTC["candidates"].values() for e in v["selected"]}


def info(ep):
    return CAND["episodes"].get(str(ep)) or EXTC["episodes"].get(str(ep)) or AN.meta(ep)


def score(ep):
    i = info(ep); return i.get("score_m", 0.0)


def arm_of(ep):
    return AN.meta(ep)["used_arm"]


def split_of(ep):
    return AN.meta(ep)["split"]


def is_t39(f, a, ep):
    return (f, a) == ("apple", "left")


def seeds(cond, f, a, ep):
    """[(seed, result or None)] for rep 1..3."""
    out = []
    for rep in (1, 2, 3):
        s = 10 * ep + rep
        out.append((s, t39(cond, ep, s) if (is_t39(f, a, ep) and rep < 3) else result(cond, ep, s)))
    return out


def library():
    lib = {}
    for (f, a), eps in screening().items():
        lib[(f, a)] = sorted([e for e in eps if all(ok(x[1]) for x in seeds("E3b", f, a, e))], key=lambda e: (-score(e), e))
    return lib


def select():
    lib = library(); sel = {}
    for f in FRUITS:
        pick = [lib[(f, a)][0] for a in ARMS if lib[(f, a)]]
        if len(pick) < 2:
            rest = sorted([e for a in ARMS for e in lib[(f, a)] if e not in pick], key=lambda e: (-score(e), e))
            pick += rest[:2 - len(pick)]
        sel[f] = pick
    # total >= 5 (task): if short, add the next library episodes, best crowding score first, round robin over fruits
    rest = {f: sorted([e for a in ARMS for e in lib[(f, a)] if e not in sel[f]], key=lambda e: (-score(e), e)) for f in FRUITS}
    while sum(len(v) for v in sel.values()) < 5 and any(rest.values()):
        for f in FRUITS:
            if rest[f] and sum(len(v) for v in sel.values()) < 5:
                sel[f].append(rest[f].pop(0))
    return sel


def items(phase):
    sc = screening(); by_fruit = {f: [] for f in FRUITS}
    if phase == "p2b":
        raise SystemExit("p2b superseded by plan v2 (robust grasp); run launch_t42b.sh (x1, x2, x3, e0)")
    if phase == "x1":
        for (f, a), eps in sc.items():
            for e in eps:
                if e in ext_eps():
                    rep = 3 if is_t39(f, a, e) else 1
                    by_fruit[f].append(f"{e}:{10 * e + rep}")
    elif phase in ("x2", "x3"):
        for (f, a), eps in sc.items():
            if (f, a) == ("apple", "left"):
                continue
            for e in eps:
                r = [x[1] for x in seeds("E3b", f, a, e)]
                if phase == "x2" and ok(r[0]) and r[1] is None:
                    by_fruit[f].append(f"{e}:{10 * e + 2}")
                if phase == "x3" and ok(r[0]) and ok(r[1]) and r[2] is None:
                    by_fruit[f].append(f"{e}:{10 * e + 3}")
                if phase == "x3" and e in ext_eps() and r[0] is None:   # late extension (banana 4 -> 7 per arm): rep 1 here
                    by_fruit[f].append(f"{e}:{10 * e + 1}")
    elif phase == "e0":
        for f, eps in select().items():
            for e in eps:
                reps = (3,) if is_t39(f, arm_of(e), e) else (1, 2, 3)
                by_fruit[f] += [f"{e}:{10 * e + r}" for r in reps]
        by_fruit["apple"].append(f"{REPRO[0]}:{REPRO[1]}")
    else:
        raise SystemExit(f"unknown phase {phase}")
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
    if phase == "e0":
        print("selected", select())
