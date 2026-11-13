#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) Failure causes of the frozen version counted ACROSS scenarios (user 2026-11-13 11:0x: fix causes common to
many scenarios, not single episodes). Episodes: SafeLIBERO other suites + Spatial L-I init 15-18 (ms-pkgFrssdm), init 19
(ms-tab10sdm), E1a 40 (sub-pkgFrssdm, Spatial L-I init 5-14). For every episode that is not a safe success:
  target share  = pre-grasp calls whose recognised target's top owner is a task object (target_truth.py; BDDL obj_of_interest =
                  target + destination), / pre-grasp calls with a target;
  blockers      = rejected calls (applied False) by the body nearest the worst row (surface_owner_task.py): task object / SafeLIBERO
                  obstacle / robot itself / other scene object;
  pi0.5 alone   = task success of the same (suite, level, task, init) with pi0.5 alone, where measured.
One primary cause per episode, in this order:
  C1 collision        - the obstacle moved > 1 mm (safety failure; the filter let the contact through)
  C2 wrong target     - target share < 0.5
  C3 task object      - most rejected calls blocked by a task object (target / destination) although the target is right
  C4 robot itself     - most rejected calls blocked by the robot's own body
  C5 other object     - most rejected calls blocked by a non-task, non-obstacle body (support, distractor)
  C6 obstacle         - most rejected calls blocked by a SafeLIBERO obstacle, no collision (the filter held the arm back)
  C7 policy           - few rejections (< 5) and no collision: the policy itself did not finish
usage: .venv-ag3s/bin/python common_causes.py OUT.json  (reads the replay outputs listed in DIRS)"""
import ast, glob, json, os, re, sys, collections
B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
S = "/tmp/claude-0/-mnt-dev/a2a2d792-941a-400f-8ace-60815fa2c777/scratchpad/"
TT = [S + "common/tt", S + "tt_tab10"]
OWN = [S + "common/owner", S + "owner_tab10"]


def lab(e):
    s = e.get("safelibero") or {}
    return ("SS" if e["official_success"] and not s.get("collided") else "S+C" if e["official_success"] else
            "F+C" if s.get("collided") else "F"), 1000 * float(s.get("max_displacement_m") or 0)


def runs():
    for pat, kind in (("2*_ms-pkgFrssdm-*", "ms"), ("2*_ms-tab10sdm-*-i19-*", "ms"), ("2*_sub-pkgFrssdm-e1a-spatial-LI-t*", "e1a")):
        for d in sorted(glob.glob(B + pat)):
            if not os.path.isfile(d + "/episodes.jsonl") or os.path.getsize(d + "/episodes.jsonl") == 0:
                continue
            m = re.search(r"_ms-[^-]+-(\w+)-(LI+)-i", d)
            suite, lvl = (m.group(1), m.group(2)) if m else ("spatial", "LI")
            yield d, suite, lvl


def pi05_table():
    out = {}
    for suite in ("spatial", "object", "goal", "long"):
        for lvl in ("LI", "LII"):
            for p in (f"2*_e1a-none-{suite}-{lvl}-{lvl}-none", f"2*_blt1-none-{suite}-{lvl}-{lvl}-none",
                      f"2*_sdm-pi05-sl-{suite}-{lvl}-{lvl}-vla_only", f"2*_tab10sdm-pi05-sl-{suite}-{lvl}-{lvl}-vla_only"):
                for d in sorted(glob.glob(B + p)):
                    if not os.path.isfile(d + "/episodes.jsonl"):
                        continue
                    for l in open(d + "/episodes.jsonl"):
                        if l.strip():
                            e = json.loads(l)
                            out[(suite, lvl, int(e["task_id"]), int(e["init_state_index"]))] = bool(e["official_success"])
    return out


def tt_index():
    """(suite, lvl, task, init) -> (task-object calls, calls with a target, goal list) from target_truth logs."""
    out = {}
    for dd in TT:
        for f in glob.glob(dd + "/*.log"):
            name = os.path.basename(f)
            m = re.match(r"(?:ms-[^-]+-)?(\w+)-(LI+)", name.replace("sub-pkgFrssdm-e1a-spatial-LI", "spatial-LI"))
            if not m:
                continue
            suite, lvl = m.group(1), m.group(2)
            for l in open(f, errors="ignore"):
                mm = re.match(r"safelibero_(\w+?)_t(\d)_e(\d+) goal (\[.*?\]) pre-grasp calls with target (\d+) top owner == goal (\d+)", l)
                if mm:
                    out[(suite, lvl, int(mm.group(2)), int(mm.group(3)))] = (int(mm.group(6)), int(mm.group(5)), ast.literal_eval(mm.group(4)))
    return out


def owner_file(run_name, suite, lvl, t, i):
    cands = [S + f"common/owner/{run_name}_t{t}_e{i}.json", S + f"owner_tab10/{suite}-{lvl}_t{t}_e{i}.json"]
    for c in cands:
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            return c
    return None


def blockers(path, goal):
    cnt = collections.Counter()
    for c in json.load(open(path)):
        if c.get("applied") is False and c.get("owners"):
            o = c["owners"][0][0]
            k = ("task object" if o in goal else "SafeLIBERO obstacle" if "obstacle" in o else
                 "robot itself" if o.startswith(("gripper0", "robot0", "eef")) else "other object")
            cnt[k] += 1
    return cnt


def main():
    pi, tt = pi05_table(), tt_index()
    rows = []
    for d, suite, lvl in runs():
        run_name = re.sub(r"^[0-9T]+_[0-9a-f]+_", "", os.path.basename(d)).split("-vla_v11ab")[0]
        for l in open(d + "/episodes.jsonl"):
            if not l.strip():
                continue
            e = json.loads(l)
            t, i = int(e["task_id"]), int(e["init_state_index"])
            lb, disp = lab(e)
            key = (suite, lvl, t, i)
            r = {"suite": suite, "level": lvl, "task": t, "init": i, "label": lb, "disp_mm": round(disp, 1),
                 "pi05_success": pi.get(key), "run": run_name}
            if key in tt:
                good, n, goal = tt[key]
                r["target_share"] = round(good / n, 3) if n else None
            else:
                goal = []
            r["goal_objects"] = goal
            if lb != "SS":
                of = owner_file(run_name, suite, lvl, t, i)
                b = blockers(of, goal) if of else None
                r["blockers"] = dict(b) if b is not None else None
                n_rej = sum(b.values()) if b else 0
                share = r.get("target_share")
                if disp > 1.0:
                    cause = "C1 collision"
                elif share is not None and share < 0.5:
                    cause = "C2 wrong target"
                elif b is None:
                    cause = "? no replay"
                elif n_rej < 5:
                    cause = "C7 policy"
                else:
                    top = b.most_common(1)[0][0]
                    cause = {"task object": "C3 task object", "robot itself": "C4 robot itself", "other object": "C5 other object",
                             "SafeLIBERO obstacle": "C6 obstacle"}[top]
                r["cause"] = cause
            rows.append(r)
    # dedupe (suite, lvl, task, init): the newest run wins (init 19 appears once)
    seen = {}
    for r in rows:
        seen[(r["suite"], r["level"], r["task"], r["init"])] = r
    rows = list(seen.values())
    bad = [r for r in rows if r["label"] != "SS"]
    by = collections.defaultdict(list)
    for r in bad:
        by[r["cause"]].append(r)
    print(f"episodes {len(rows)}  safe success {len(rows) - len(bad)}  not {len(bad)}")
    print(f"{'cause':20s} {'ep':>4s} {'scen':>5s} {'suites':>7s} {'cells':>6s}  {'pi0.5 succ':>10s}  examples")
    summary = {}
    for c, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        scen = {(r["suite"], r["level"], r["task"]) for r in v}
        cells = {(r["suite"], r["level"]) for r in v}
        suites = {r["suite"] for r in v}
        ps = sum(1 for r in v if r["pi05_success"])
        pk = sum(1 for r in v if r["pi05_success"] is not None)
        summary[c] = {"episodes": len(v), "scenarios": len(scen), "cells": len(cells), "suites": sorted(suites),
                      "pi05_success": [ps, pk]}
        ex = ", ".join(f"{r['suite'][:4]}{r['level'][1:]}t{r['task']}e{r['init']}" for r in v[:5])
        print(f"{c:20s} {len(v):4d} {len(scen):5d} {len(suites):7d} {len(cells):6d}  {ps:4d} / {pk:<4d}  {ex}")
    json.dump({"rule": __doc__, "summary": summary, "episodes": rows}, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
