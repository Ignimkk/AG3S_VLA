"""T43 U — lead's V2 / V3 question: closing-phase chunks of T43 V (V0-V3 hurdle, V2 no obstacle), side by side.
near-fruit closing chunk = grasp state closing, not attached, fingertip midpoint <= 60 mm from the fruit centre (planning qpos).
binding row = smaller of waypoint (max_violation_pair) and path_check clearance; category by query index (base < 122 <= gripper_cover
< n_robot <= held); point nearest = GT surface nearest the recorded row point (no hands). Writes raw/v_closing.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, statistics as st
from collections import Counter
U = _ag3s_workspace_path('outputs/verify/T43/U/raw')
out = {"doc": __doc__}
for tag, pat in (("V0 hurdle", "T43__V__H__V0_*"), ("V1 hurdle", "T43__V__H__V1_*"), ("V2 hurdle", "T43__V__H__V2_*"),
                 ("V3 hurdle", "T43__V__H__V3_*"), ("V2 clear", "T43__V__B__V2_*")):
    A = {"runs": 0, "outcomes": Counter(), "closing_all": 0, "closing_near": 0, "near_man_not_fruit": 0, "near_not_carved": 0,
         "near_remnant_ge25": 0, "near_binding": Counter(), "near_binding_point": Counter(), "near_binding_path_point_fruit": 0,
         "near_path_target_rows_excluded": 0, "near_fruit_rows_obstacle_class_chunks": 0, "near_defl": [], "near_radial": [],
         "near_binding_cl": [], "all_closing_binding": Counter(), "all_closing_binding_point": Counter(), "all_path_target_rows_excluded": 0,
         "held_rows_binding_preattach": 0, "switch_away_before_attach": 0}
    for f in sorted(glob.glob(f"{U}/runs/{pat}.json")):
        r = json.load(open(f)); A["runs"] += 1
        S = json.load(open(f"{U}/summary_u.json"))["runs"][r["key"]]
        A["outcomes"][S["outcome"]] += 1; A["switch_away_before_attach"] += bool(S.get("switch_away_before_attach"))
        rows = {d["seq"]: d for d in r["rows"] if "error" not in d}
        for e in r["timeline"]:
            b = e["to"][e["to"]["binding"]] or {}
            A["all_path_target_rows_excluded"] += (e["to"].get("path") or {}).get("target_rows_excluded") or 0
            if not e["attached"] and b.get("cat") == "held":
                A["held_rows_binding_preattach"] += 1
            if e["grasp_state"] != "closing" or e["attached"]:
                continue
            A["closing_all"] += 1
            A["all_closing_binding"][f'{e["to"]["binding"]}:{b.get("cat")}'] += 1
            A["all_closing_binding_point"][b.get("point_nearest_nohand")] += 1
            d = rows.get(e["seq"])
            if d is None or (d.get("tip_now_to_fruit_mm") or 1e9) > 60:
                continue
            A["closing_near"] += 1
            A["near_man_not_fruit"] += bool(e["man_centroid"] and e["man_centroid"]["nearest"] != "fruit")
            A["near_not_carved"] += d["carve"].get("carved") is False
            A["near_remnant_ge25"] += (d["carve"].get("remnant_frac_free") or 0) >= 0.25
            A["near_binding"][f'{e["to"]["binding"]}:{b.get("cat")}'] += 1
            A["near_binding_point"][b.get("point_nearest_nohand")] += 1
            A["near_binding_path_point_fruit"] += e["to"]["binding"] == "path" and b.get("point_nearest_nohand") == "fruit"
            A["near_path_target_rows_excluded"] += (e["to"].get("path") or {}).get("target_rows_excluded") or 0
            A["near_fruit_rows_obstacle_class_chunks"] += d["fruit_rows_obstacle_margin_applied"] > 0
            A["near_defl"].append((d["deflection"] or {}).get("defl_mm")); A["near_radial"].append((d["deflection"] or {}).get("radial_mm"))
            A["near_binding_cl"].append(b.get("clearance_mm"))
    for k in ("near_defl", "near_radial", "near_binding_cl"):
        v = [x for x in A[k] if x is not None]; A[k + "_median"] = None if not v else st.median(v); del A[k]
    out[tag] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in A.items()}
json.dump(out, open(f"{U}/v_closing.json", "w"), indent=1, default=str)
for k, v in out.items():
    if k != "doc": print(k, json.dumps(v))
