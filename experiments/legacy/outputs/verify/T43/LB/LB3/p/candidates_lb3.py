"""T43 LB3 candidate lists (verifier, 2026-10-09). CPU only, dataset metadata only. Same pool + crowd score as LB candidates_lb.py
(train split, place_one, success, grasp_attempts 1, retries 0; score = min(nn_target, gap_target_crate, gap_crate_others), ties by episode),
but the FULL ranking per fruit x arm (200) is kept. LB tried ranks 1-10 (checked against runs/E0 and candidates_lb.json[:20]).
LB3 banana plan: ranks 11-30 (20 ranked) then a random sample (RNG seed below) from pool minus ranks 1-30, total <= 40 incl. LB ranks 1-10.
LB3 pear_left plan: ranks 11-40.  usage: candidates_lb3.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.argv_out = sys.argv[1]
src = open(_ag3s_workspace_path('outputs/verify/T43/LB/p/candidates_lb.py')).read()
# reuse the LB pool code verbatim up to the 'cands = {}' line
head = src.split("cands = {}")[0]
head = head.replace("import json, sys, pathlib\nimport numpy as np\n", "")
exec(head)
RNG_SEED = 20261009
old = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/LB/candidates_lb.json')).read_text())
out = {"definitions": __doc__, "rng_seed": RNG_SEED, "split_bounds_from_records": {k: {"first": v[0], "last": v[1], "n": v[2]} for k, v in bounds.items()},
       "pool_total": len(pool), "groups": {}, "episodes": {}}
for f, a in (("banana", "left"), ("banana", "right"), ("pear", "left")):
    g = f"{f}_{a}"
    ranked = sorted([e for e, v in eps.items() if v["target_fruit"] == f and v["used_arm"] == a], key=lambda e: (-eps[e]["score_m"], e))
    assert ranked[:20] == old["candidates"][g]["ranked"], g
    rest = [e for e in ranked[30:]]
    rng = np.random.default_rng(RNG_SEED + (0 if g == "banana_left" else 1))
    perm = [int(x) for x in rng.permutation(rest)]
    out["groups"][g] = {"pool_size": len(ranked), "ranked_all": ranked, "lb_tried_ranks_1_10": ranked[:10],
                        "lb3_ranked_11_40": ranked[10:40], "random_order_pool_minus_ranks_1_30": perm[:30] if f == "banana" else []}
    for e in ranked:
        out["episodes"][str(e)] = dict(eps[e], rank_in_pool=ranked.index(e) + 1)
pathlib.Path(sys.argv_out).write_text(json.dumps(out, indent=1))
for g, v in out["groups"].items():
    print(g, v["pool_size"], "rank11-40 scores(mm):", [round(out["episodes"][str(e)]["score_m"]*1000) for e in v["lb3_ranked_11_40"]][::5], "rand10", v["random_order_pool_minus_ranks_1_30"][:10])
