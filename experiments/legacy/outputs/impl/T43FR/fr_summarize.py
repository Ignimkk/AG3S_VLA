"""T43 FR — aggregate offline/fr_*.json into per-option (a) / (b) rows (FC's definitions, one column set per option).

(a) CT finger-hurdle contact events (57): true geometry 'sees' = min over rows/states of (sphere-to-hurdle distance - r) < 0;
    recorded field 'sees' = d - r < 0 (events with a server field, 44).
(b) closing / attach chunks (1254): margin-0 = TO plan rows (refined 0..7) with gap < esdf_margin that the verdict would not
    allow (FC's 'non-allowed'); 10 mm band = plan rows with gap < margin + 10 mm (HOLD iff that row's label is obstacle).
    'new only' = chunk flagged with the option but not with the axial chain (FC's comparison); 'not held' = of those, chunks the
    server did not already HOLD (the new-HOLD upper bound for executed chunks).
usage: fr_summarize.py OUT_JSON
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, sys

runs, options = {}, None
for f in sorted(glob.glob(_ag3s_workspace_path('outputs/impl/T43FR/offline/fr_*.json'))):
    d = json.load(open(f))
    runs.update(d["runs"]); options = d["options"]
names = list(options)
A = [e for r in runs.values() for e in r["contacts"]]
B = [dict(c, kind="attach" if c["attach"] else "closing") for r in runs.values() for c in r["grasp_chunks"]]
out = {"definitions": __doc__, "n_runs": len(runs), "n_events": len(A), "n_chunks": len(B), "options": {}}


def a_row(nm):
    geo = [e[f"geo_{nm}_mm_min"] for e in A]
    es = [e[f"esdf_{nm}_mm_min"] for e in A if e.get(f"esdf_{nm}_mm_min") is not None]
    le_mesh = sum(1 for e in A if e["mesh_mm_min"] is not None and e[f"geo_{nm}_mm_min"] <= e["mesh_mm_min"] + 1e-6)
    return {"geo_sees": sum(g < 0 for g in geo), "geo_le_mesh": le_mesh, "events": len(geo),
            "esdf_sees": sum(x < 0 for x in es), "esdf_events": len(es),
            "geo_missed": [(e["first_t"], e["phase"], e["urdf_link"], e[f"geo_{nm}_mm_min"], e["mesh_mm_min"]) for e in A if e[f"geo_{nm}_mm_min"] >= 0]}


def b_row(nm):
    def na(c, m):
        return (c.get(f"{m}_nonallowed_rows_plan") or 0) > 0

    def lt10(c, m):
        return (c.get(f"{m}_rows_plan_lt10") or 0) > 0
    r = {}
    for kind in ("closing", "attach", "all"):
        rows = [c for c in B if kind == "all" or c["kind"] == kind]
        r[kind] = {
            "chunks": len(rows),
            "margin0": sum(na(c, nm) for c in rows),
            "margin0_new_vs_chain": sum(na(c, nm) and not na(c, "old") for c in rows),
            "margin0_new_vs_chain_not_held": sum(na(c, nm) and not na(c, "old") and not c["held_by_server"] for c in rows),
            "margin0_new_vs_fc25": sum(na(c, nm) and not na(c, "fc25") for c in rows),
            "band10": sum(lt10(c, nm) for c in rows),
            "band10_new_vs_chain": sum(lt10(c, nm) and not lt10(c, "old") for c in rows),
            "band10_new_vs_chain_not_held": sum(lt10(c, nm) and not lt10(c, "old") and not c["held_by_server"] for c in rows),
            "band10_new_vs_fc25_not_held": sum(lt10(c, nm) and not lt10(c, "fc25") and not c["held_by_server"] for c in rows),
        }
    return r


for nm in ["old"] + names:
    out["options"][nm] = {"a": a_row(nm), "b": b_row(nm)}
    a, b = out["options"][nm]["a"], out["options"][nm]["b"]["all"]
    print(f"{nm:11s} (a) geo {a['geo_sees']}/{a['events']} esdf {a['esdf_sees']}/{a['esdf_events']} | (b) m0 {b['margin0']} "
          f"new {b['margin0_new_vs_chain']} (not held {b['margin0_new_vs_chain_not_held']}) vs fc25 {b['margin0_new_vs_fc25']} | "
          f"b10 {b['band10']} new {b['band10_new_vs_chain']} (not held {b['band10_new_vs_chain_not_held']}) vs fc25 nh {b['band10_new_vs_fc25_not_held']}",
          flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
