"""T42-2 step 2: one pick-phase and one place-phase obstacle per episode from raw/search_<EP>_<SLOT>.json; write configs; client
dry runs. usage: select.py EP:SEED [EP:SEED ...]   (order = assignment order)
Place-phase only unless T42_PICK=1 (lead 2026-10-05). Slot assignment: pick slots rotate post_0 -> fence_0 -> overbar_0, place slots rotate hurdle_0 -> crate_post_0 over the episodes in
the given order (episode i starts at rotation index i); if the assigned slot has no feasible candidate, the next slot in the
rotation is taken (recorded as 'fallback').
Tiers within a slot (margin m = min(reset robot, reset object, t0 minimum, other-phase minimum); room = search.py room_m):
  T1: m >= 0.010 and room >= 0.03;  T2: m >= 0.005;  T3: feasible.
Within the best non-empty tier: largest room (None last), ties -> larger m.
Configs: configs/t42_2_ep<EP>.json with profiles t42_2_ep<EP>_pick, t42_2_ep<EP>_place, t42_2_ep<EP>_both (both entries).
Cross-checks: the both-profile on the captured scene (plib, exact activate): reset robot / object clearance, t0 minimum; client dry
run (dryrun.py) for each of the three profiles."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p'))
import plib as L, search as S  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2/raw')); CFG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2/configs')); CFG.mkdir(exist_ok=True)
PY = _ag3s_workspace_path('legacy/.venv-openpi-live/bin/python')
PICK = ["post_0", "fence_0", "overbar_0"]; PLACE = ["hurdle_0", "crate_post_0"]
import os
WITH_PICK = os.environ.get("T42_PICK", "0") == "1"   # lead 2026-10-05: place-phase obstacles first; pick-phase optional, later


def tier(e):
    r = e.get("room_m")
    if e["margin_m"] >= 0.010 and r is not None and r >= 0.03:
        return 1
    return 2 if e["margin_m"] >= 0.005 else 3


def best_of(ep, slot):
    f = RAW / f"search_{ep}_{slot}.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    pool = d["feasible"]
    for e in pool:
        e["tier"] = tier(e)
    for t in (1, 2, 3):
        P = [e for e in pool if e["tier"] == t]
        if P:
            return max(P, key=lambda e: (-1 if e.get("room_m") is None else round(e["room_m"], 4), e["margin_m"])), d
    return None, d


def dry(ep, path, prof):
    dj = RAW / f"dryrun_{prof}.json"
    subprocess.run([PY, _ag3s_workspace_path('outputs/verify/T42/2/p/dryrun.py'), str(ep), str(path), prof, str(dj)], capture_output=True, text=True,
                   cwd="${AG3S_PROJECT_ROOT}-t42c/pi05_TO_hybrid",
                   env={"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin",
                        "PYTHONPATH": "${AG3S_PROJECT_ROOT}-t42c"})
    return json.loads(dj.read_text()) if dj.exists() else None


def main(items):
    res = {"definitions": __doc__, "episodes": {}}
    for i, it in enumerate(items):
        ep, seed = (int(x) for x in it.split(":"))
        r = {"seed": seed}
        for kind, rot in ((("pick", PICK),) if WITH_PICK else ()) + (("place", PLACE),):
            order = [rot[(i + k) % len(rot)] for k in range(len(rot))]
            chosen = None; tried = []
            for slot in order:
                b, d = best_of(ep, slot)
                tried.append({"slot": slot, "n_feasible": None if d is None else d["n_feasible"], "counts": None if d is None else d["counts"]})
                if b is not None:
                    chosen = (slot, b); break
            r[kind] = {"assigned": order[0], "tried": tried, "fallback": chosen is not None and chosen[0] != order[0],
                       "slot": None if chosen is None else chosen[0], "chosen": None if chosen is None else chosen[1]}
        prof = {}
        ents = {}
        for kind in ("pick", "place"):
            if kind not in r:
                continue
            ch = r[kind]["chosen"]
            if ch is None:
                continue
            ents[kind] = S.entry(ch)
            ents[kind]["position"] = [round(float(v), 4) for v in ents[kind]["position"]]
            ents[kind]["yaw"] = round(float(ents[kind]["yaw"]), 6)
            prof[f"t42_2_ep{ep}_{kind}"] = (f"T42-2 {kind}-phase {ents[kind]['slot']} (tier {ch['tier']}, margin {ch['margin_m']:.4f} m, room "
                                          f"{'-' if ch.get('room_m') is None else round(ch['room_m'], 4)} m)", [ents[kind]])
        if len(ents) == 2:
            prof[f"t42_2_ep{ep}_both"] = ("T42-2 pick + place obstacles together", [ents["pick"], ents["place"]])
        path = CFG / f"t42_2_ep{ep}.json"
        L.write_config(path, prof)
        r["config"] = str(path); r["profiles"] = list(prof)
        if len(ents) == 2:
            c = L.Ctx(ep, seed); c.place([ents["pick"], ents["place"]], exact=True)
            rr, ro = c.reset_check()
            t0 = {k: min(float(v.min()) for v in c.series(k, [0], {"robot": c.ROBOT, "target": c.TARGET, "fruits": c.FRUITS, "crate": c.CRATE}, cap=0.1).values()) for k in c.runs}
            r["both_check"] = {"reset_robot": rr, "reset_object": ro, "t0_min": t0}
            c.restore()
        r["dryrun"] = {p: dry(ep, path, p) for p in prof}
        res["episodes"][str(ep)] = r
        print(ep, {k: (r[k]["slot"], None if r[k]["chosen"] is None else (r[k]["chosen"]["tier"], round(r[k]["chosen"]["margin_m"], 4),
                                                                          r[k]["chosen"].get("room_m"))) for k in ("pick", "place") if k in r},
              {p: (v or {}).get("outcome") for p, v in r["dryrun"].items()}, flush=True)
    used = {r[k]["slot"] for r in res["episodes"].values() for k in ("pick", "place") if k in r}
    res["slots_used"] = sorted(s for s in used if s); res["all_slots_used"] = set((PICK if WITH_PICK else []) + PLACE) <= used
    (RAW / "select.json").write_text(json.dumps(res, indent=1, default=float))
    print("slots used", res["slots_used"], res["all_slots_used"])


if __name__ == "__main__":
    main(sys.argv[1:])
