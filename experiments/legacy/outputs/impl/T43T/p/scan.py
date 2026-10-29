"""T43 T — per-chunk record scan of the permanent-HOLD runs (records only, no recompute)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, ast, pathlib
import numpy as np
S = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
IX = {**json.load(open(S / "chunk_index_t42.json")), **json.load(open(S / "chunk_index_t40t41.json"))}
RUNS = {
 "T42_R1908": _ag3s_workspace_path('outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1908_s19082'),
 "T42_L1992": _ag3s_workspace_path('outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1992_s19922'),
 "T41c_C1982": _ag3s_workspace_path('outputs/verify/T41/c/runs/E3b/C/E3b_ep1982_s19822'),
 "T41c_D1982": _ag3s_workspace_path('outputs/verify/T41/c/runs/E3b/D/E3b_ep1982_s19822'),
}
def lit(v):
    if isinstance(v, str):
        try: return ast.literal_eval(v)
        except Exception: return v
    return v
def row(s):
    to = s.get("to") or {}; m = to.get("metrics") or {}
    fb = s.get("exec_feedback") or {}
    v = s.get("verdict") or {}
    reasons = v.get("reasons") or []
    cls = v.get("classification") or {}
    g = s.get("grasp") or {}
    pair = lit(m.get("max_violation_pair"))
    cands = lit(m.get("candidates")) or []
    man = s.get("manipulated") or {}
    out = dict(
        t=s.get("t_step"), status=s.get("status"), validity=s.get("validity"), gstat=s.get("grounding_status"),
        has_target=s.get("has_target"), man_state=man.get("state"), man_age=man.get("age_frames"),
        grasp=g.get("state"), attach=g.get("attach"), detach=g.get("detach"), revoked=g.get("attach_revoked"),
        fb_seq=fb.get("seq"), n_exec=fb.get("n_exec"), hold_reason=(fb.get("hold_reason") or "")[:160],
        kinds=[r.get("kind") for r in reasons],
        reason_detail=[(r.get("kind"), (r.get("detail") or "")[:140]) for r in reasons],
        to_status=to.get("status"), iters=to.get("sqp_iterations"), returned=m.get("returned"),
        init_viol_mm=None if m.get("initial_violation_m") is None else 1e3*m["initial_violation_m"],
        clearance_mm=None if m.get("clearance_m") is None else 1e3*m["clearance_m"],
        cand_viol_mm=[round(1e3*c["violation_m"],1) for c in cands if isinstance(c, dict) and "violation_m" in c],
        initial_source=(lit(m.get("selection")) or {}).get("initial_source") if isinstance(lit(m.get("selection")), dict) else m.get("initial_source"),
        worst=None if not isinstance(pair, dict) else {k: pair.get(k) for k in ("block","link","step","tier","clearance_m","point_m")},
        step_min_mm=[round(x,1) for x in (s.get("clearance_min_per_step_mm") or [])],
        n_coll=cls.get("n_collision"), n_unres=cls.get("n_unresolved"), n_allowed=cls.get("n_allowed"),
        coll_by_why=cls.get("collision_by_why"), coll_worst=cls.get("collision_worst"), unres_worst=cls.get("unresolved_worst"),
        notes=[n[:200] for n in (s.get("notes") or []) if any(w in n for w in ("invariant","uncert","lost","not certified","degraded","incomplete","no target"))],
        invariant=s.get("invariant_violation"),
        attached=bool(g.get("attached")),
    )
    return out
res = {}
for name, rd in RUNS.items():
    ix = IX[rd]; chunks = {int(k): v["npz"] for k, v in ix["chunks"].items()}
    rows = {}
    for seq in sorted(chunks):
        with np.load(chunks[seq], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
        rows[seq] = row(s)
    res[name] = {"run": rd, "server": ix["server"], "rows": rows}
pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=str))
