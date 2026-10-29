"""Post-process off_0ab.json: split by grasp phase (GT apple dz from the local planning row; pre-grasp = before the
first row with apple dz >= 50 mm), and separate 'manipulated is a different object' (visible state, nearest centre != apple,
or id switched) from 'manipulated is the apple's stale last-observed geometry' (occluded, id 0)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np
R = json.load(open(_ag3s_workspace_path('outputs/verify/T28/p3/off_0ab.json')))
FR = {"t14": _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl'),
      "t17": _ag3s_workspace_path('outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl')}
out = {"definitions": __doc__}
for key, v in R["runs"].items():
    run = key[:3]
    fr = {}
    for l in open(FR[run]):
        r = json.loads(l)
        if r["kind"] == "planning": fr[r["seq"]] = r
    z0 = fr[1]["object_poses"]["apple"]["pos"][2]
    C = [e for e in v["chunks"] if e["has_local_row"]]
    dz = {e["chunk"]: 1000 * (fr[e["chunk"] + 1]["object_poses"]["apple"]["pos"][2] - z0) for e in C}
    grasp = next((e["chunk"] for e in C if dz[e["chunk"]] >= 50), None)
    pre = [e for e in C if grasp is None or e["chunk"] < grasp]
    post = [e for e in C if grasp is not None and e["chunk"] >= grasp]
    def cnt(L):
        return {"n": len(L),
                "manipulated_nearest_not_apple": sum(1 for e in L if e["manipulated"] and e["manipulated"]["nearest_centre"] != "apple"),
                "manipulated_id_not_first(switched)": sum(1 for e in L if e["manipulated"] and e["manipulated"]["id"] != 0),
                "visible_and_nearest_not_apple": sum(1 for e in L if e["manipulated"] and e["manipulated"]["state"] == "visible" and e["manipulated"]["nearest_centre"] != "apple"),
                "occluded_stale_id0_nearest_not_apple": sum(1 for e in L if e["manipulated"] and e["manipulated"]["state"] != "visible" and e["manipulated"]["id"] == 0 and e["manipulated"]["nearest_centre"] != "apple"),
                "no_manipulated": sum(1 for e in L if e["manipulated"] is None),
                "state_counts": {s: sum(1 for e in L if e["manipulated"] and e["manipulated"]["state"] == s) for s in ("visible", "occluded", "lost")},
                "to_apple_mm_max": max((e["manipulated"]["to_apple_mm"] for e in L if e["manipulated"]), default=None)}
    sw = [{"chunk": e["chunk"], "t": 8 * e["chunk"], "apple_dz_mm": dz[e["chunk"]], "new_id": e["manipulated"]["id"],
           "nearest_centre": e["manipulated"]["nearest_centre"], "to_apple_mm": e["manipulated"]["to_apple_mm"],
           "admissibility": e["manipulated"]["admissibility"]} for e in C if e["decision"] and e["decision"]["mode"] == "switch"]
    dest = next(({"chunk": e["chunk"], "t": 8 * e["chunk"], "apple_dz_mm": dz[e["chunk"]], **e["destination"]} for e in C if e["destination"]), None)
    out[key] = {"grasp_chunk_gt_dz50": grasp, "pre_grasp": cnt(pre), "post_grasp": cnt(post), "switches": sw,
                "destination_registered": dest, "summary_from_off_0ab": {k: x for k, x in v["summary"].items() if k != "apple_cluster_narrowest_extent_mm"},
                "apple_cluster_narrowest_extent_mm": v["summary"]["apple_cluster_narrowest_extent_mm"]}
    print(key, "grasp", grasp, "pre", cnt(pre), "post", cnt(post), "switches", [(s["chunk"], s["new_id"], s["nearest_centre"], round(s["to_apple_mm"])) for s in sw],
          "dest", dest and (dest["chunk"], dest["id"], dest["extents_mm"]))
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T28/p3/off_0ab_summary.json'), "w"), indent=1)
