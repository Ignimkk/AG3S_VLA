"""T43 T 3(c) evidence — for every adoption (decision first/switch) in the recorded runs, the adopted
geometry's lowest / highest point above the table plane and its extents. Records only."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np
S = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T43/S/raw'))
IX = {**json.load(open(S / "chunk_index_t42.json")), **json.load(open(S / "chunk_index_t40t41.json"))}
TABLE = 0.8216
rows = []
for rd, ix in IX.items():
    if "chunks" not in ix:
        continue
    man = json.loads((pathlib.Path(rd) / "manifest.json").read_text())
    plan = {}
    for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines():
        if '"planning"' in l[:40]:
            r = json.loads(l); plan[r["seq"]] = r.get("object_poses")
    for seq, v in sorted(ix["chunks"].items(), key=lambda kv: int(kv[0])):
        with np.load(v["npz"], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
            dec = ((s.get("admissibility") or {}).get("decision") or {})
            if dec.get("mode") not in ("first", "switch"):
                continue
            tp = np.asarray(z["target_points"], float).reshape(-1, 3)
            names = s.get("sphere_link_names") or []
            keep = np.array([n.startswith("ee_") for n in names], bool)
            hc = np.asarray(z["sphere_centres"], float)[keep]; hr = np.asarray(z["sphere_radii"], float)[keep]
        if not len(tp):
            continue
        m = s.get("manipulated") or {}
        c = np.asarray(m.get("centroid") or tp.mean(0))
        poses = plan.get(int(seq)) or {}
        near = min(((k, float(np.linalg.norm(np.asarray(p["pos"])[:2] - c[:2]))) for k, p in poses.items()),
                   key=lambda kv: kv[1], default=(None, None))
        rows.append({"run": rd.split("/runs/")[-1], "seq": int(seq), "mode": dec["mode"], "id": m.get("id"),
                     "target_fruit": man["scene"].get("episode_target_fruit"),
                     "obstacle": man["scene"].get("obstacle_profile"),
                     "bottom_mm": round(1e3 * (tp[:, 2].min() - TABLE), 1), "top_mm": round(1e3 * (tp[:, 2].max() - TABLE), 1),
                     "centroid": [round(x, 3) for x in c], "extents_mm": (m.get("admissibility") or {}).get("extents_mm"),
                     "hand_gap_mm": None if not len(tp) or not len(hc) else round(1e3 * float((np.linalg.norm(hc[:, None] - tp[None], axis=2) - hr[:, None]).min()), 1),
                     "grasp": (s.get("grasp") or {}).get("state"),
                     "nearest_object": near[0], "nearest_xy_mm": None if near[1] is None else round(1e3 * near[1], 1)})
pathlib.Path(sys.argv[1]).write_text(json.dumps(rows, indent=0))
print(len(rows))
