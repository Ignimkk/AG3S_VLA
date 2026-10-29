"""T34 per run: attach / detach / attach_revoked chunks from the server records with evidence.hold_window / hold_checks /
revoke / attach_revoked, and the held.support record at each attach. usage: revoke_detail.py COND -> p/<COND>_revoke_detail.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
COND = sys.argv[1]
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37'))
A = json.loads((V / "p" / f"{COND}_analysis.json").read_text())["runs"]
out = {"doc": __doc__, "runs": {}}
for run in sorted(A):
    ev = []
    for c in A[run]["chunks"]:
        if not c.get("server_chunk"):
            continue
        s = json.loads(str(np.load(V / f"{COND}_rec" / c["server_chunk"], allow_pickle=False)["summary_json"]))
        g = s.get("grasp") or {}; e = g.get("evidence") or {}
        if g.get("attach") or g.get("detach") or (e.get("hold_window") or {}).get("active"):
            held = ((g.get("attached") or {}).get("held") or {}) if g.get("attach") else {}
            sup = held.get("support") or {}
            ev.append({"t": c["t"], "seq": c["seq"], "state": g.get("state"), "attach": g.get("attach"), "detach": g.get("detach"),
                       "note": g.get("note"), "hold_window": e.get("hold_window"), "hold_checks": e.get("hold_checks"), "revoke": e.get("revoke"),
                       "attach_revoked": g.get("attach_revoked"), "opening_after": e.get("opening_after"), "last_command": e.get("last_command"),
                       "support": None if not g.get("attach") else {k: sup.get(k) for k in ("applied", "plane", "clearance_source", "clearance_mm", "max_lift_mm")} | {
                           "band_q10_q50_q90_mm": (sup.get("band") or {}).get("band_q10_q50_q90_mm"), "n_table_only": (sup.get("band") or {}).get("n_table_only"),
                           "spheres": sup.get("spheres")},
                       "radii_mm": [x["radius_mm"] for x in held.get("spheres_parent_frame") or []],
                       "fit": held.get("fit"), "hand_shift_since_capture_mm": held.get("hand_shift_since_capture_mm"),
                       "first_free_spheres": held.get("first_free_spheres")})
    out["runs"][run] = ev
    print(run, [(x["t"], "A" if x["attach"] else ("R" if x["detach"] and "attach_revoked" in str(x["note"]) else ("D" if x["detach"] else "w")), x["hold_checks"]) for x in ev][:12])
(V / "p" / f"{COND}_revoke_detail.json").write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False))
