"""T43 FC — aggregate offline/fc_*.json into the (a) / (b) tables (implementer).

usage: summarize.py OUT_JSON
(a) one row per CT finger-hurdle contact event; 'sees' = recorded-field clearance d - r < 0 (margin 0 in every arm here).
(b) one row per closing / attach chunk; HOLD-relevant = TO plan rows 0..7 (refined chunk) with gap < esdf_margin; 'now' = q_now
    row (not a TO row, informational).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import collections, glob, json, sys
import numpy as np

runs = {}
for f in sorted(glob.glob(_ag3s_workspace_path('outputs/impl/T43FC/offline/fc_*.json'))):
    runs.update(json.load(open(f))["runs"])

A = []
for key, r in runs.items():
    for e in r["contacts"]:
        fid = [x.get("post_fidelity_mm") for x in e["rows"] if x.get("post_fidelity_mm") is not None]
        A.append({"key": key, "arm": r["arm"], "phase": e["phase"], "mjcf_body": e["mjcf_body"], "urdf_link": e["urdf_link"],
                  "t": e["first_t"], "ct_pen_mm": round(e["max_pen_mm"], 2), "mesh_mm": e["mesh_mm_min"],
                  "geo_old_mm": e["geo_old_mm_min"], "geo_new_mm": e["geo_new_mm_min"],
                  "esdf_old_mm": e["esdf_old_mm_min"], "esdf_new_mm": e["esdf_new_mm_min"],
                  "post_fidelity_mm": max(fid) if fid else None})


def agg_a(rows):
    def n(f):
        return sum(1 for x in rows if f(x))
    with_esdf = [x for x in rows if x["esdf_new_mm"] is not None]
    return {"events": len(rows), "with_server_field": len(with_esdf),
            "mesh_lt0": n(lambda x: x["mesh_mm"] is not None and x["mesh_mm"] < 0),
            "geo_old_lt0": n(lambda x: x["geo_old_mm"] < 0), "geo_new_lt0": n(lambda x: x["geo_new_mm"] < 0),
            "geo_new_le_mesh": n(lambda x: x["mesh_mm"] is not None and x["geo_new_mm"] <= x["mesh_mm"] + 1e-6),
            "esdf_old_lt0": n(lambda x: x["esdf_old_mm"] is not None and x["esdf_old_mm"] < 0),
            "esdf_new_lt0": n(lambda x: x["esdf_new_mm"] is not None and x["esdf_new_mm"] < 0),
            "esdf_new_only": n(lambda x: x["esdf_new_mm"] is not None and x["esdf_new_mm"] < 0 <= x["esdf_old_mm"]),
            "esdf_old_only": n(lambda x: x["esdf_old_mm"] is not None and x["esdf_old_mm"] < 0 <= x["esdf_new_mm"]),
            "median_esdf_old_mm": float(np.median([x["esdf_old_mm"] for x in with_esdf])) if with_esdf else None,
            "median_esdf_new_mm": float(np.median([x["esdf_new_mm"] for x in with_esdf])) if with_esdf else None,
            "median_geo_old_mm": float(np.median([x["geo_old_mm"] for x in rows])) if rows else None,
            "median_geo_new_mm": float(np.median([x["geo_new_mm"] for x in rows])) if rows else None}


summary_a = {}
for (ph, body), rows in sorted(collections.defaultdict(list, {}).items()):
    pass
groups = collections.defaultdict(list)
for x in A:
    groups[(x["phase"], x["mjcf_body"])].append(x)
    groups[(x["phase"], "*")].append(x)
for k, rows in sorted(groups.items()):
    summary_a[f"{k[0]}|{k[1]}"] = agg_a(rows)

B = []
for key, r in runs.items():
    for c in r["grasp_chunks"]:
        kind = "attach" if c["attach"] else "closing"
        B.append({"key": key, "arm": r["arm"], "stage": r["stage"], "t": c["t"], "kind": kind,
                  "server_hold": c["held_by_server"],
                  "old_plan_viol": c["old_viol_rows_plan"], "new_plan_viol": c["new_viol_rows_plan"],
                  "old_now_viol": c["old_viol_rows_now"], "new_now_viol": c["new_viol_rows_now"],
                  "old_min_plan_mm": c["old_min_gap_plan_mm"], "new_min_plan_mm": c["new_min_gap_plan_mm"],
                  "old_min_now_mm": c["old_min_gap_now_mm"], "new_min_now_mm": c["new_min_gap_now_mm"],
                  "old_nonallowed_plan": c.get("old_nonallowed_rows_plan"), "new_nonallowed_plan": c.get("new_nonallowed_rows_plan"),
                  "old_lt10": c.get("old_rows_plan_lt10"), "new_lt10": c.get("new_rows_plan_lt10"),
                  "old_lt20": c.get("old_rows_plan_lt20"), "new_lt20": c.get("new_rows_plan_lt20"),
                  "layer": c.get("finger_layer"),
                  "mask": c.get("new_mask"), "fidelity_mm": c.get("old_fidelity_mm"),
                  "new_nearest": c.get("new_viol_nearest"), "old_nearest": c.get("old_viol_nearest")})


def agg_b(rows):
    lab = collections.Counter(); lab_plan = collections.Counter(); olab = collections.Counter()
    for x in rows:
        for k, v in (x.get("old_nearest") or {}).items():
            olab[k] += v["rows_lt10"]
    for x in rows:
        for k, v in (x["new_nearest"] or {}).items():
            lab[k] += v["rows_lt10"]; lab_plan[k] += v["rows_lt_margin"]
    uniq = {(x["key"].split("/")[1], x["key"].split("_", 1)[1], x["t"], x["kind"], x["old_min_plan_mm"], x["new_min_plan_mm"]) for x in rows}
    def ch(f):
        return sum(1 for x in rows if f(x))
    return {"chunks": len(rows), "unique_chunks": len(uniq),
            "nonallowed_old_chunks": ch(lambda x: (x["old_nonallowed_plan"] or 0) > 0),
            "nonallowed_new_chunks": ch(lambda x: (x["new_nonallowed_plan"] or 0) > 0),
            "nonallowed_new_only_chunks": ch(lambda x: (x["new_nonallowed_plan"] or 0) > 0 and not (x["old_nonallowed_plan"] or 0)),
            "lt10_old_chunks": ch(lambda x: (x["old_lt10"] or 0) > 0), "lt10_new_chunks": ch(lambda x: (x["new_lt10"] or 0) > 0),
            "lt10_new_only_chunks": ch(lambda x: (x["new_lt10"] or 0) > 0 and not (x["old_lt10"] or 0)),
            "lt20_old_chunks": ch(lambda x: (x["old_lt20"] or 0) > 0), "lt20_new_chunks": ch(lambda x: (x["new_lt20"] or 0) > 0),
            "lt20_new_only_chunks": ch(lambda x: (x["new_lt20"] or 0) > 0 and not (x["old_lt20"] or 0)),
            "layer_counts": dict(collections.Counter(json.dumps(x["layer"], sort_keys=True) for x in rows)),
            "plan_viol_old_chunks": sum(x["old_plan_viol"] > 0 for x in rows),
            "plan_viol_new_chunks": sum(x["new_plan_viol"] > 0 for x in rows),
            "plan_viol_new_only_chunks": sum(x["new_plan_viol"] > 0 and x["old_plan_viol"] == 0 for x in rows),
            "plan_viol_rows_old": sum(x["old_plan_viol"] for x in rows),
            "plan_viol_rows_new": sum(x["new_plan_viol"] for x in rows),
            "now_viol_old_chunks": sum(x["old_now_viol"] > 0 for x in rows),
            "now_viol_new_chunks": sum(x["new_now_viol"] > 0 for x in rows),
            "now_viol_new_only_chunks": sum(x["new_now_viol"] > 0 and x["old_now_viol"] == 0 for x in rows),
            "median_min_plan_old_mm": float(np.median([x["old_min_plan_mm"] for x in rows])) if rows else None,
            "median_min_plan_new_mm": float(np.median([x["new_min_plan_mm"] for x in rows])) if rows else None,
            "min_plan_new_minus_old_mm": [float(np.min([x["new_min_plan_mm"] - x["old_min_plan_mm"] for x in rows])),
                                          float(np.max([x["new_min_plan_mm"] - x["old_min_plan_mm"] for x in rows]))] if rows else None,
            "server_hold_chunks": sum(x["server_hold"] for x in rows),
            "new_rows_lt10_explained_by": dict(lab), "old_rows_lt10_explained_by": dict(olab), "new_rows_lt_margin_explained_by": dict(lab_plan),
            "fidelity_max_mm": max([x["fidelity_mm"] for x in rows if x["fidelity_mm"] is not None and x["fidelity_mm"] == x["fidelity_mm"]], default=None),
            "fidelity_hist_mm": dict(collections.Counter(str(round(x["fidelity_mm"], 1)) for x in rows if x["fidelity_mm"] is not None))}


summary_b = {k: agg_b([x for x in B if x["kind"] == k]) for k in ("closing", "attach")}
summary_b["all"] = agg_b(B)
by_arm = {arm: agg_b([x for x in B if x["arm"] == arm]) for arm in sorted({x["arm"] for x in B})}
out = {"definitions": __doc__, "n_runs": len(runs), "a_summary": summary_a, "a_events": A,
       "b_summary": summary_b, "b_by_arm": by_arm, "b_chunks": B}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({"a": summary_a, "b": summary_b}, indent=1))
for x in B:
    if (x["new_lt10"] or 0) > 0 or (x["old_lt10"] or 0) > 0:
        print("B", x["key"], x["t"], x["kind"], "viol old/new", x["old_plan_viol"], x["new_plan_viol"], "nonallowed", x["old_nonallowed_plan"], x["new_nonallowed_plan"],
              "<10", x["old_lt10"], x["new_lt10"], "<20", x["old_lt20"], x["new_lt20"], "min", x["old_min_plan_mm"], x["new_min_plan_mm"], x["new_nearest"], "hold", x["server_hold"], "fid", x["fidelity_mm"])
for x in A:
    if x["esdf_new_mm"] is not None and x["esdf_new_mm"] >= 0:
        print("A not seen", x)
