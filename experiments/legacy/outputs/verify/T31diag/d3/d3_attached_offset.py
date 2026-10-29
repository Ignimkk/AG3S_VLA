"""Held-apple geometry vs GT (window chunks): |centroid(reconstructed attached points, base) - GT apple centre (base)|,
min distance from GT apple centre to the attached points, and GT apple centre -> nearest left finger sphere centre (mm).
Also the recorded summary.grasp.alignment.hand_shift_mm at the attach chunk. usage as d3_records.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d4'))
import fig_common as FC
out = {"definitions": __doc__, "runs": {}}
D1 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')).read_text())
for run in ["E3a_ep1807_r1", "E3a_ep1807_r3", "E3b_ep1807_r1", "E3b_ep1807_r2"]:
    R = FC.Run(run); g, rel = D1["runs"][run]["grasp_t"], D1["runs"][run]["release_t"]
    rows = []
    att_chunk = None
    for c in R.chunks:
        z = __import__("numpy").load(FC.T30 / f"{R.cond}_rec" / c["server_chunk"], allow_pickle=False)
        s = json.loads(str(z["summary_json"]))
        if (s.get("grasp") or {}).get("attach"):
            att_chunk = {"t": c["t"], "alignment": (s.get("grasp") or {}).get("alignment")}
        if not (g <= c["t"] <= rel):
            continue
        ch = R.load(c["t"])
        if not ch.attached:
            continue
        cen = ch.traj["reference"]["centres"][0]
        hp = cen[R.S:]; fg = cen[R.finger_idx]
        rows.append({"t": c["t"], "attached_centroid_to_gt_apple_mm": 1000 * float(np.linalg.norm(hp.mean(0) - ch.apple_centre)),
                     "gt_apple_to_nearest_attached_point_mm": 1000 * float(np.linalg.norm(hp - ch.apple_centre, axis=1).min()),
                     "gt_apple_to_nearest_finger_sphere_centre_mm": 1000 * float(np.linalg.norm(fg - ch.apple_centre, axis=1).min()),
                     "attached_centroid_minus_gt_apple_base_mm": (1000 * (hp.mean(0) - ch.apple_centre)).tolist()})
    a = np.array([r["attached_centroid_to_gt_apple_mm"] for r in rows])
    out["runs"][run] = {"attach": att_chunk, "n": len(rows), "attached_centroid_to_gt_apple_mm_median": float(np.median(a)),
                        "min": float(a.min()), "max": float(a.max()), "chunks": rows}
    print(run, att_chunk, len(rows), np.round(a, 1).tolist(), flush=True)
pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d3/d3_attached_offset.json')).write_text(json.dumps(out, indent=1, default=float))
