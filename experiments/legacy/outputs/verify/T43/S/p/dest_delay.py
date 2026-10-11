"""T43 S (4) — when is the destination (crate) registered, relative to attach, per arm (E3b T42-1, summary records only).
Per run: first seq with grasp.attach, first seq with summary.destination.centroid, destination.source / registered_frame,
and the number of held chunks without a destination. usage: dest_delay.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
RUNS = json.loads((RAW / "t42_runs_frozen.json").read_text()); IDX = json.loads((RAW / "chunk_index_t42.json").read_text())
res = {}
for key, r in sorted(RUNS.items()):
    if not key.startswith("E3b/"): continue
    ix = IDX[f"${AG3S_PROJECT_ROOT}/outputs/verify/T42/1/runs/E3b/noobs/{key.split('/')[1]}"]
    att = None; dest = None; src = None; held_nodest = 0; held = 0; subtask = {}
    for s in sorted(ix["chunks"], key=int):
        with np.load(ix["chunks"][s]["npz"]) as z: summ = json.loads(str(z["summary_json"]))
        g = summ.get("grasp") or {}; de = summ.get("destination") or {}
        if g.get("attach") and att is None: att = int(s)
        if de.get("centroid") is not None and dest is None: dest = int(s); src = de.get("source")
        if g.get("state") == "held":
            held += 1; held_nodest += int(de.get("centroid") is None)
            lab = str((summ.get("subtask") or {}).get("label")); subtask[lab] = subtask.get(lab, 0) + 1
    res[key] = {"arm": r["used_arm"], "fruit": r["target_fruit"], "grasp": r["grasp"], "success": r["success"], "first_attach_seq": att,
                "first_destination_seq": dest, "destination_source": src, "held_chunks": held, "held_chunks_without_destination": held_nodest,
                "subtask_label_in_held_chunks": subtask}
agg = {}
for v in res.values():
    a = agg.setdefault(v["arm"], {"runs": 0, "runs_dest_before_or_at_attach": 0, "runs_dest_after_attach": 0, "runs_no_dest": 0, "held": 0, "held_nodest": 0,
                                  "dest_minus_attach_chunks": []})
    a["runs"] += 1; a["held"] += v["held_chunks"]; a["held_nodest"] += v["held_chunks_without_destination"]
    if v["first_destination_seq"] is None: a["runs_no_dest"] += 1
    elif v["first_attach_seq"] is not None and v["first_destination_seq"] > v["first_attach_seq"]:
        a["runs_dest_after_attach"] += 1; a["dest_minus_attach_chunks"].append(v["first_destination_seq"] - v["first_attach_seq"])
    else: a["runs_dest_before_or_at_attach"] += 1
(RAW / "dest_delay.json").write_text(json.dumps({"definitions": __doc__, "by_arm": agg, "runs": res}, indent=0))
print(json.dumps(agg, indent=0))
for k, v in res.items():
    print(k.split('/')[1], v["arm"][0], v["fruit"][:3], "G" if v["grasp"] else "-", "S" if v["success"] else "-", v["first_attach_seq"], v["first_destination_seq"], v["destination_source"], v["held_chunks"], v["held_chunks_without_destination"])
