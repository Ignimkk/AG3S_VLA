"""T43 V11 hurdle-crossing clearance table, commanded (C: executed_actions on the measured state) vs measured (M: post state of the row), paired by ep:seed. numbers only.
usage: V11_ARM=V11A crossing_pairs_v11.py OUT_JSON      (reads XM/raw/cross_V8.json and V11/raw/cross_<ARM>.json; xm_cross.py method, no new computation of geometry)
Per run, carry crossing row of the apple (s(t-1) < 0 <= s(t), apple in hand) and of the fingertip midpoint ('tip'):
  apple_bottom_minus_bar_top_mm (C, M) at the apple crossing row; geo_bar_mm (mj_geomDistance to the bar, mm) of apple / fingers at that row; z_minus_bar_top_mm of the tip (C, M) at the tip crossing row;
  fingers_bar_mm (C, M) at the tip crossing; min over rows crossing-8 .. crossing+8 of the fingers' distance to the whole hurdle (C, M).
Return crossing (after the apple left): tip crossing s(t-1) > 0 >= s(t): fingers' bar distance and min hurdle distance +-8 rows (C, M)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, collections
import numpy as np
ARM = os.environ.get("V11_ARM", "V11A"); T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
R = {"V8": json.load(open(T / "XM/raw/cross_V8.json"))["runs"], ARM: json.load(open(T / f"V11/raw/cross_{ARM}.json"))["runs"]}


def stat(v):
    v = [x for x in v if x is not None]
    return {"n": 0} if not v else {"n": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "mean": float(np.mean(v))}


def row(arm, key):
    r = R[arm].get(f"{arm}/H/{arm}_ep{key}")
    if not r or "error" in r:
        return None
    cr = r["crossing"]; o = {}
    for st in ("C", "M"):
        a = cr.get(f"carry_apple_{st}"); tp = cr.get(f"carry_tip_{st}"); rt = cr.get(f"return_tip_{st}")
        o[f"apple_bottom_minus_bar_top_{st}"] = a["apple_bottom_minus_bar_top_mm"] if a and a["inhand"] else None
        o[f"apple_bar_geo_{st}"] = a["geo_bar_mm"]["apple"] if a and a["inhand"] else None
        o[f"tip_z_minus_bar_top_{st}"] = tp["z_minus_bar_top_mm"]["tip"] if tp else None
        o[f"tip_fingers_bar_{st}"] = tp["geo_bar_mm"]["fingers"] if tp else None
        o[f"tip_inhand_{st}"] = tp["inhand"] if tp else None
        o[f"tip_fingers_min_hurdle_pm8_{st}"] = tp["min_geoH_pm8_mm"]["fingers"] if tp else None
        o[f"return_fingers_bar_{st}"] = rt["geo_bar_mm"]["fingers"] if rt else None
        o[f"return_fingers_min_hurdle_pm8_{st}"] = rt["min_geoH_pm8_mm"]["fingers"] if rt else None
    return o


keys = sorted({k.split("_ep")[1] for arm in R for k in R[arm]})
out = {"definitions": __doc__, "arm_new": ARM, "per_run": {}, "arms": {}, "paired": {}}
for k in keys:
    out["per_run"][k] = {arm: row(arm, k) for arm in ("V8", ARM)}
fields = sorted({f for v in out["per_run"].values() for a_ in v.values() if a_ for f in a_ if not f.endswith("inhand_C") and not f.endswith("inhand_M")})
for arm in ("V8", ARM):
    out["arms"][arm] = {f: stat([v[arm][f] for v in out["per_run"].values() if v[arm]]) for f in fields}
for f in fields:
    d = [(v[ARM][f] - v["V8"][f]) for v in out["per_run"].values() if v["V8"] and v[ARM] and v["V8"][f] is not None and v[ARM][f] is not None]
    out["paired"][f] = {"n_pairs": len(d), "diff_new_minus_base": stat(d), "n_new_higher": int(sum(x > 0 for x in d))}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
for f in ("apple_bottom_minus_bar_top_M", "apple_bottom_minus_bar_top_C", "tip_fingers_bar_M", "tip_fingers_bar_C", "tip_z_minus_bar_top_M", "tip_z_minus_bar_top_C", "tip_fingers_min_hurdle_pm8_M", "tip_fingers_min_hurdle_pm8_C", "return_fingers_bar_M", "return_fingers_bar_C", "return_fingers_min_hurdle_pm8_M", "return_fingers_min_hurdle_pm8_C"):
    print(f, {a: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out["arms"][a][f].items()} for a in ("V8", ARM)}, "paired", out["paired"][f]["n_pairs"], {k: round(v, 1) for k, v in (out["paired"][f]["diff_new_minus_base"] or {}).items() if k in ("median", "min", "max")}, out["paired"][f]["n_new_higher"])
