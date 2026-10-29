"""List paired server chunks of the 4 E3 ep1807 runs with grasp/attach/field facts (T30 pairing in p/<COND>_analysis.json)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, ast
import numpy as np
T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
RUNS = {"E3a_ep1807_r1": "E3a", "E3a_ep1807_r2": "E3a", "E3a_ep1807_r3": "E3a", "E3b_ep1807_r1": "E3b", "E3b_ep1807_r2": "E3b"}
out = {}
for run, cond in RUNS.items():
    A = json.load(open(T30 / "p" / f"{cond}_analysis.json"))["runs"][run]
    rows = []
    for c in A["chunks"]:
        f = T30 / f"{cond}_rec" / c["server_chunk"]
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        st = s.get("esdf_stats") or {}
        g = s.get("grasp") or {}
        man = s.get("manipulated") or {}
        to = s.get("to") or {}; met = to.get("metrics") or {}
        rows.append({"seq": c["seq"], "t": c["t"], "file": str(f), "keys_free": "esdf_free_distance" in z.files,
                     "n_candidates": s.get("n_candidates"), "labels": st.get("label_names"), "n_attached_points": st.get("n_attached_points"),
                     "grasp_state": g.get("state"), "attach": g.get("attach"), "detach": g.get("detach"),
                     "attached": g.get("attached"), "man_radius": man.get("radius"), "man_n_points": man.get("n_points"),
                     "target_points_n": int(z["target_points"].shape[0]) if "target_points" in z.files else None, "sphere_n": int(z["sphere_radii"].shape[0]),
                     "dest": (s.get("destination") or {}).get("source"), "returned": met.get("returned"),
                     "sqp_it": to.get("sqp_iterations"), "executed": c.get("n_executed"), "phase": c.get("phase")})
    out[run] = rows
    for r in rows:
        if r["attach"] or r["detach"] or r["t"] in (A["summary"]["grasp_t"],):
            print(run, r["seq"], r["t"], r["grasp_state"], "attach", r["attach"], "detach", r["detach"], "n_att", r["n_attached_points"], r["attached"] and r["attached"].get("n_points"), "man_r", r["man_radius"], "tp", r["target_points_n"], r["file"].split("/")[-1])
    held = [r["t"] for r in rows if r["n_attached_points"]]
    print(run, "attached t range", held[:1], held[-1:], "n", len(held), "free-layer chunks", sum(r["keys_free"] for r in rows), "ncand set", set(r["n_candidates"] for r in rows), "labels", set(str(r["labels"]) for r in rows))
pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d2/scan_chunks.json')).write_text(json.dumps(out, indent=1, default=str))
