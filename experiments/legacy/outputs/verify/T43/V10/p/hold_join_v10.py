"""T43 V10: join hold_cause.json (binding row + nearest surface) with agg.json HOLD records (agg phase, V10-only vs V8 at the same ep:seed and planning step t) -> hold_join.json
(numbers only). new = V10 HOLD chunk whose (stage, ep:seed, t) has no HOLD chunk in the paired V8 run.
row_type: 'held' (binding query >= n_robot) | 'finger' (binding link ee_finger_*) | 'robot_other' (any other robot link). Left-finger names: URDF names as recorded by the server (MJCF ee_finger_l2 = URDF ee_finger_l1, MJCF ee_finger_l1 = URDF ee_finger_l2).
usage: hold_join_v10.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V10'))
hc = json.load(open(V / "hold_cause.json"))["runs"]; ag = json.load(open(V / "agg.json"))
v10 = ag["hold"]["V10"]["per_run"]; v8 = ag["hold"]["V8"]["per_run"]
out = {"chunks": [], "definitions": __doc__}
for key, r in sorted(hc.items()):
    st, nm = key.split("/")[1], key.split("/")[2]; rk = f"{st}:{nm.split('_', 1)[1].replace('ep', '')}"
    rk = f"{st}:{nm.split('_ep')[1].split('_s')[0]}_s{nm.split('_s')[-1]}"
    a = {c["t"]: c for c in v10[rk]["chunks"]}; b = {c["t"] for c in v8.get(rk, {"chunks": []})["chunks"]}
    for c in r["chunks"]:
        s = c.get("surface") or {}; w = c["worst_row"]; ac = a[c["t"]]
        rt = "held" if c["is_held_row"] else ("finger" if str(w["link"]).startswith("ee_finger") else "robot_other")
        out["chunks"].append({"run": rk, "t": c["t"], "new_vs_V8": c["t"] not in b, "agg_phase": ac["phase"], "hm_phase_apple_within_80mm_of_crate": c["phase"], "attached": c["attached"], "row_type": rt, "binding_link": w["link"],
                              "row_class": w["class"], "query": w["query"], "step": w["step"], "clearance_mm": w["clearance_mm"], "d_minus_r_mm": s.get("d_minus_r_mm"), "target_label": w["obstacle_label"],
                              "surface_identity": s.get("identity"), "surface_nearest_group": s.get("gt_group_nearest"), "surface_nearest_mm": s.get("gt_nearest_mm"), "self_cap_mm": s.get("self_cap_mm"),
                              "apple_gt_dist_mm": (s.get("gt_dist_by_group_mm") or {}).get("apple"), "hurdle_gt_dist_mm": (s.get("gt_dist_by_group_mm") or {}).get("hurdle"),
                              "crate_gt_dist_mm": (s.get("gt_dist_by_group_mm") or {}).get("crate"), "table_gt_dist_mm": (s.get("gt_dist_by_group_mm") or {}).get("table")})
def cnt(L, f): return dict(sorted(collections.Counter(str(f(x)) for x in L).items()))
summ = {}
for lab, L in (("all_V10_HOLD", out["chunks"]), ("new_V10_HOLD", [x for x in out["chunks"] if x["new_vs_V8"]]), ("not_new", [x for x in out["chunks"] if not x["new_vs_V8"]])):
    summ[lab] = {"n": len(L), "by_row_type": cnt(L, lambda x: x["row_type"]), "by_row_class": cnt(L, lambda x: x["row_class"]), "by_row_type_class": cnt(L, lambda x: f"{x['row_type']}|{x['row_class']}"),
                 "by_surface_identity": cnt(L, lambda x: x["surface_identity"]), "by_agg_phase": cnt(L, lambda x: x["agg_phase"]), "by_run": cnt(L, lambda x: x["run"]),
                 "by_row_type_identity": cnt(L, lambda x: f"{x['row_type']}|{x['surface_identity']}"), "by_binding_link": cnt(L, lambda x: x["binding_link"]), "by_target_label": cnt(L, lambda x: x["target_label"])}
out["summary"] = summ
held = [x for x in out["chunks"] if x["row_type"] == "held"]
out["held_row_binding_chunks"] = held
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
for k, v in summ.items(): print(k, json.dumps(v))
print("held-row binding HOLD chunks:", [(x["run"], x["t"], x["new_vs_V8"], x["row_class"], x["surface_identity"], round(x["surface_nearest_mm"], 1), x["target_label"]) for x in held])
