"""T42-2 diagnostic: for place-slot candidates passing (a) and (b), the pick-window minimum distance (ROBOT, TARGET, FRUITS; min
over E0 and E3b) and whether (c) holds, to see how far criterion (f) (>= 30 mm) is from reachable.
usage: diag_f.py EP SEED SLOT -> raw/diag/diag_f_<EP>_<SLOT>.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p'))
import plib as L, search as S  # noqa: E402
ep, seed, slot = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
c = L.Ctx(ep, seed); out = []
for e in S.candidates(c, slot):
    x, y = e["position"][:2]
    if not (0.38 <= x <= 0.85 and -0.35 <= y <= 0.35):
        continue
    c.place([S.entry(e)]); rr, ro = c.reset_check()
    if not (rr > 0 and ro > 0):
        continue
    if any(c.first_le0(k, [0], np.r_[c.ROBOT, c.TARGET, c.FRUITS, c.CRATE]) is not None for k in c.runs):
        continue
    if not all(c.first_le0(k, c.rows(k, "carry"), np.r_[c.ROBOT, c.TARGET]) is not None for k in c.runs):
        continue
    pm = {}
    for k in c.runs:
        s, pr = c.series(k, c.rows(k, "pick"), {"all": np.r_[c.ROBOT, c.TARGET, c.FRUITS]}, cap=0.1, want_pair=True)
        j = int(np.argmin(s["all"])); pm[k] = {"min": float(s["all"][j]), "t": int(c.runs[k]["t"][c.rows(k, "pick")[j]]), "pair": pr["all"][j]}
    e.update(reset_robot=rr, reset_object=ro, pick_min=pm, pick_min_both=min(v["min"] for v in pm.values()))
    out.append(e)
out.sort(key=lambda e: -e["pick_min_both"])
pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T42/2/raw/diag/diag_f_{ep}_{slot}.json").write_text(json.dumps({"definitions": __doc__, "n": len(out), "top": out[:50]}, default=float))
print(ep, slot, "n(a,b,c)", len(out), "top", [(round(e["pick_min_both"], 4), [round(v, 3) for v in e["position"][:2]], e.get("bar_height", e.get("height")),
                                              {k: (v["t"], v["pair"]) for k, v in e["pick_min"].items()}) for e in out[:5]])
