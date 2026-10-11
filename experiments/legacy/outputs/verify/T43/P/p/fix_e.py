"""T43 P: fill free_height_over_bar (criterion e, psearch definition) for candidates passing a', b, d and (c or c_relaxed) that have
none (files written while psearch computed e only for c_relaxed passers), recompute pass_e and the counts.  Idempotent.
usage: fix_e.py FILE..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F
from fsearch import WS
for fp in sys.argv[1:]:
    fp = pathlib.Path(fp); d = json.loads(fp.read_text()); C = d["candidates"]
    todo = [e for e in C if e["pass_a"] and e["pass_b"] and e["pass_d"] and (e["pass_c"] or e["pass_c_relaxed"]) and e.get("free_height_over_bar_m") is None]
    if todo:
        c = F.Ctx(d["tag"]); ws = WS(c); carry = {}
        for cond, R in c.runs.items():
            carry[cond] = R["Q"][(R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"])]
        off = ws.offset(np.vstack(list(carry.values()))); cache = {}
        for e in todo:
            key = (round(e["x"], 3), round(e["y"], 3))
            if key not in cache:
                zs = [z for z in (ws.max_tcp_z(np.array(key), Q) for Q in carry.values()) if z is not None]
                cache[key] = max(zs) if zs else None
            tz = cache[key]; e["tcp_z_max_above_table_m"] = tz; e["tcp_minus_apple_bottom_median_m"] = off
            e["free_height_over_bar_m"] = None if tz is None else tz - off - (e["h"] + 0.012)
            e["e_filled_by_fix_e"] = True
        c.restore()
    for e in C:
        e["pass_e"] = e.get("free_height_over_bar_m") is not None and e["free_height_over_bar_m"] >= 0.05
    d["counts"]["pass_e"] = sum(e["pass_e"] for e in C)
    d["counts"]["all_abcde"] = sum(1 for e in C if e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"])
    d["counts"]["all_ab_crelaxed_de"] = sum(1 for e in C if e["pass_a"] and e["pass_b"] and e["pass_c_relaxed"] and e["pass_d"] and e["pass_e"])
    d["fix_e"] = {"n_filled": len(todo)}
    fp.write_text(json.dumps(d, default=float)); print(fp.name, len(todo), d["counts"], flush=True)
