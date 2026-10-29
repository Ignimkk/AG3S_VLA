"""T43GD contact-inclusive rule (copy of T43/CT/p/agg_ct.py run_contacts + success_ct, arms V7 / V7g, replays in T43/GD/replay_ct (CT/p/replay_ct.py unchanged)).
usage: gd_ct.py OUT_JSON      (CPU)  -> per H-stage run: pre_release_contact, post_release_contact, obstacle_stop, success_ct, contacts list.
Phase of a contact row t:  before_grasp: t < grasp_t (grasp false: every row);  carry: grasp_t <= t <= release_t (release_t none: to the run's end);  after_release: t > release_t   (release_t = Z4 release rule, GD/z4.json).
pre_release_contact = >= 1 robot (hand / robot_other) or apple contact row with the hurdle with phase != after_release;  post_release_contact = >= 1 such row with phase after_release.
success_ct = success (old) and success_rel (Z4) and not pre_release_contact.  Stage B (no hurdle): success_ct = success_rel (no hurdle to touch).  obstacle_stop = '[SAFETY] obstacle stop' in the client log (check.json)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); GD = T / "GD"
ROOTS = {"V7": GD / "runs/V7", "V7g": T / "V7g/runs/V7g"}


def phase(t, grasp, grasp_t, rel_t):
    if not grasp or grasp_t is None or t < grasp_t:
        return "before_grasp"
    if rel_t is None or t <= rel_t:
        return "carry"
    return "after_release"


Z = json.load(open(GD / "z4.json"))["runs"]; out = {}
for key, o in sorted(Z.items()):
    arm, st, name = key.split("/")
    if st != "H":
        out[key] = {"stage": "B", "pre_release_contact": False, "post_release_contact": False, "success_ct": bool(o["success"] and o["success_rel"]), "contacts": []}
        continue
    j = json.load(open(GD / "replay_ct" / f"{arm}_{st}_{name}.json"))
    rel_t, gt, grasp = o["release_t"], o["grasp_t"], o["grasp"]; rec = {}
    for r in j["obstacle_contact_rows"]:
        for g, v in r["geoms"].items():
            gname, body, kind, sub, mind, maxf, npos = v; ph = phase(r["t"], grasp, gt, rel_t)
            e = rec.setdefault((body, kind, ph), {"body": body, "kind": kind, "phase": ph, "geoms": set(), "first_t": r["t"], "last_t": r["t"], "n_rows": 0, "substeps": 0, "max_pen_mm": 0.0, "min_dist_mm": 1e9, "max_force_N": 0.0})
            e["geoms"].add(gname); e["first_t"] = min(e["first_t"], r["t"]); e["last_t"] = max(e["last_t"], r["t"]); e["n_rows"] += 1; e["substeps"] += sub
            e["max_pen_mm"] = max(e["max_pen_mm"], max(0.0, -1e3 * mind)); e["min_dist_mm"] = min(e["min_dist_mm"], 1e3 * mind); e["max_force_N"] = max(e["max_force_N"], maxf)
    for e in rec.values():
        e["geoms"] = sorted(e["geoms"])
    cons = sorted(rec.values(), key=lambda e: (e["first_t"], e["body"]))
    chk = json.load(open(ROOTS[arm] / "H" / f"{name}.check.json"))["checks"]
    pre = [c for c in cons if c["phase"] != "after_release"]; post = [c for c in cons if c["phase"] == "after_release"]
    out[key] = {"stage": "H", "pre_release_contact": bool(pre), "post_release_contact": bool(post), "obstacle_stop": bool(chk["obstacle_stop_in_log"]),
                "pre_release_kinds": sorted({c["kind"] for c in pre}), "post_release_kinds": sorted({c["kind"] for c in post}),
                "success_ct": bool(o["success"] and o["success_rel"] and not pre), "release_t": rel_t, "grasp_t": gt, "n_rows": j["n_rows"], "qpos_mismatch_rows": j["qpos_mismatch_rows"],
                "contacts": cons, "check_contact_steps": chk["npz_obstacle_safety"]["contact_steps"]}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
for a in ("V7", "V7g"):
    L = [v for k, v in out.items() if k.startswith(a + "/") and v["stage"] == "H"]
    print(a, {"n": len(L), "S_ct": sum(v["success_ct"] for v in L), "pre": sum(v["pre_release_contact"] for v in L), "post": sum(v["post_release_contact"] for v in L), "stop": sum(v["obstacle_stop"] for v in L)})
