"""T44 smoke: replay one V11AB run with the ORIGINAL replay_ct.replay and with replay_contacts.replay (same process, same snapshot);
compare every legacy class (cnt / mind / maxf / pairs per control row) exactly, and against the stored V11/replay json; write replay_smoke.json.
usage: smoke_replay.py RUN_DIR OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib, os, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/p'))
import replay_contacts as RC   # sets T43W_SNAP from the manifest
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/CT/p'))
import replay_ct as OLD
rd = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
cl = RC.HW.load_client()
Rn = RC.replay(cl, rd); Rn["run_dir"] = str(rd)
Ro = OLD.replay(cl, rd); Ro["run_dir"] = str(rd)
scratch = out.parent / "smoke_raw"; scratch.mkdir(exist_ok=True)
js = RC.save(Rn, scratch, "new_" + rd.name)
# 1) in-process comparison old vs new, legacy classes, row by row
diff = {c: 0 for c in RC.LEGACY}; n = len(Ro["C"]); assert n == len(Rn["C"])
for a, b in zip(Ro["C"], Rn["C"]):
    assert a["t"] == b["t"] and a["seq"] == b["seq"]
    for c in RC.LEGACY:
        if a["cnt"][c] != b["cnt"][c] or a["mind"][c] != b["mind"][c] or a["maxf"][c] != b["maxf"][c] or a["pairs"].get(c, []) != b["pairs"].get(c, []):
            diff[c] += 1
state_diff = {k: float(np.max(np.abs(np.array([s[k] for s in Ro["S"]]) - np.array([s[k] for s in Rn["S"]])))) for k in Ro["S"][0]}
# 2) stored V11 replay json (made by the original replay_ct.py in an earlier session)
name = rd.parent.parent.name + "_" + rd.parent.name + "_" + rd.name
stored_p = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11/replay')) / f"{name}.json"
stored = json.load(open(stored_p)) if stored_p.exists() else None
stored_cmp = None
if stored:
    stored_cmp = {"file": str(stored_p), "contact_rows_equal": {c: stored["contact_rows"][c] == js["contact_rows"][c] for c in RC.LEGACY},
                  "contact_events_equal": {c: json.dumps(stored["contact_events"][c], sort_keys=True) == json.dumps(js["contact_events"][c], sort_keys=True) for c in RC.LEGACY},
                  "qpos_mismatch_rows": [stored["qpos_mismatch_rows"], js["qpos_mismatch_rows"]]}
# 3) seq / chunk mapping sanity: control chunk_seq == plan_index + 1 for server runs
ms = [(x["seq"], x["plan_index"] + 1) for x in Rn["C"]]
seq_map_ok = all(a == b for a, b in ms)
res = dict(run_dir=str(rd), episode=Rn["ep"], seed=Rn["seed"], profile=Rn["prof"], side=Rn["side"], target_fruit=Rn["target_fruit"], snapshot=str(RC.HW.HY),
           n_control_rows=n, qpos_mismatch_rows=Rn["qpos_mismatch_rows"],
           legacy_vs_original_replay_ct=dict(rows_differing_per_class=diff, all_equal=all(v == 0 for v in diff.values()), state_max_abs_diff=state_diff),
           legacy_vs_stored_V11_json=stored_cmp,
           legacy_contact_rows=Ro["C"] and {c: int(sum(1 for x in Ro["C"] if x["cnt"][c])) for c in RC.LEGACY},
           new_class_contact_rows={c: js["contact_rows"][c] for c in RC.NEW}, new_class_contact_substeps={c: js["contact_substeps"][c] for c in RC.NEW},
           new_class_chunks={c: [(e["seq_local"], e["server_chunk_k"], e["rows"], e["substeps"], e["min_dist_m"], e["max_force_N"]) for e in js["contact_chunks"][c]] for c in RC.NEW},
           new_class_geom_detail={c: js["geom_detail"][c] for c in RC.NEW},
           legacy_geom_detail_top={c: js["geom_detail"][c][:5] for c in RC.LEGACY if js["geom_detail"][c]},
           hand_geoms=js["geom_sets"]["hand"], inactive_hand_geoms=js["geom_sets"]["inactive_hand"], n_robot_other_geoms=len(js["geom_sets"]["robot_other"]),
           control_chunk_seq_equals_plan_index_plus_1=seq_map_ok, server_chunk_alignment=js["server_chunk_alignment"], full_json=str(scratch / f"new_{rd.name}.json"))
out.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in ("legacy_vs_original_replay_ct", "new_class_contact_rows", "new_class_contact_substeps", "control_chunk_seq_equals_plan_index_plus_1")}, default=float))
