"""T35-diag A2 — per close cycle stats from a/a_runs.json timelines.
cycle = maximal run of control rows with grip_cmd < 0.5 (applied, left). Per cycle: t_start, t_end, n rows,
min open_meas (and t), at that row: d_l1/d_l2 (GT finger-apple, mm), mid_rel_apple (witness midpoint - apple centre, world mm),
d_ref (policy-commanded arm pose at that row), max apple dz / xy displacement during the cycle (mm, from run start),
and apple xy displacement from cycle start to cycle end (mm).
usage: cycles.py -> a/cycles.json"""
import json, numpy as np
R = json.load(open("a/a_runs.json"))["runs"]
out = {"definitions": __doc__, "runs": {}}
for name, r in R.items():
    tl = r["timeline"]; cyc = []; i = 0
    while i < len(tl):
        if tl[i]["grip_cmd"] < 0.5:
            j = i
            while j + 1 < len(tl) and tl[j + 1]["grip_cmd"] < 0.5:
                j += 1
            seg = tl[i:j + 1]
            km = int(np.argmin([x["open_meas"] for x in seg])); e = seg[km]
            cyc.append({"t_start": seg[0]["t"], "t_end": seg[-1]["t"], "n": len(seg), "min_open": e["open_meas"], "t_min_open": e["t"],
                        "d_l1_mm": e["d_l1_mm"], "d_l2_mm": e["d_l2_mm"], "d_ref_mm": e["d_ref_mm"], "mid_rel_apple_mm": e["mid_rel_apple_mm"],
                        "mid_rel_apple_ref_mm": e["mid_rel_apple_ref_mm"],
                        "max_dz_mm": max(x["dz_mm"] for x in seg), "max_disp_xy_mm": max(x["disp_xy_mm"] for x in seg),
                        "xy_shift_in_cycle_mm": float(1000 * np.linalg.norm(np.array(seg[-1]["apple"][:2]) - np.array(seg[0]["apple"][:2]))),
                        "min_l1_mm": min(x["d_l1_mm"] for x in seg), "min_l2_mm": min(x["d_l2_mm"] for x in seg)})
            i = j + 1
        else:
            i += 1
    out["runs"][name] = {"success": r["success"], "grasp_t": r["grasp_t"], "cycles": cyc}
json.dump(out, open("a/cycles.json", "w"), indent=1)
for n, v in out["runs"].items():
    print(n, "S" if v["success"] else "F", v["grasp_t"])
    for c in v["cycles"][:6]:
        print("   t %3d-%3d n%3d minopen %.2f@%d l1 %6.1f l2 %6.1f ref %5.1f mid %s dz %5.1f xy %5.1f shift %5.1f" % (
            c["t_start"], c["t_end"], c["n"], c["min_open"], c["t_min_open"], c["d_l1_mm"], c["d_l2_mm"], c["d_ref_mm"],
            np.round(c["mid_rel_apple_mm"], 1), c["max_dz_mm"], c["max_disp_xy_mm"], c["xy_shift_in_cycle_mm"]))
