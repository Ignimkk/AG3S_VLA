"""T43 P local pass: around seeds from the coarse pass (raw/psearch_<TAG>.json) evaluate u, v +-{0, 0.005, 0.01} m, dyaw +-{0, 5} deg,
all bar heights, with the same psearch evaluation/criteria -> raw/psearch_<TAG>_refine.json.
Seeds (distinct u, v, dyaw): top 4 by margin = min(a_min, d_min) among coarse candidates passing b and c; top 3 among b and c_relaxed;
top 3 among a', b, c, d, e when present.  usage: refine.py TAG..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, sys, time
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import psearch as S


def seeds(cands):
    for e in cands:
        e["margin"] = min(e["a_min"], e["d_min"])
    out, seen = [], set()
    groups = [([e for e in cands if e["pass_b"] and e["pass_c"]], 4), ([e for e in cands if e["pass_b"] and e["pass_c_relaxed"]], 3),
              ([e for e in cands if e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"]], 3)]
    for pool, k in groups:
        n = 0
        for e in sorted(pool, key=lambda e: -e["margin"]):
            key = (e["u"], e["v"], e["dyaw_deg"])
            if key in seen:
                continue
            seen.add(key); out.append(key); n += 1
            if n >= k:
                break
    return out


def main(tag):
    t0 = time.time()
    coarse = json.load(open(S.RAW / f"psearch_{tag}.json"))["candidates"]
    sd = seeds(coarse); P = S.PCtx(tag)
    keys = set()
    for u0, v0, y0 in sd:
        for du in (-0.01, -0.005, 0, 0.005, 0.01):
            for dv in (-0.01, -0.005, 0, 0.005, 0.01):
                for dy in (-5, 0, 5):
                    keys.add((round(u0 + du, 4), round(v0 + dv, 4), y0 + dy))
    cands = []
    for u, v, dy in sorted(keys):
        x, y, yaw = P.to_world(u, v, dy)
        if not (0.38 <= x <= 0.85 and -0.35 <= y <= 0.35):
            continue
        for h in S.HS:
            cands.append({"u": u, "v": v, "dyaw_deg": dy, "x": x, "y": y, "yaw": yaw, "h": float(h)})
    print(tag, "seeds", sd, "candidates", len(cands), flush=True)
    S.run(tag, P, cands, "_refine", t0, extra={"refine_seeds": sd, "refine_definitions": __doc__})


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
