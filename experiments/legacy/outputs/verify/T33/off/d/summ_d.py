"""T33 offline (d) summary: hand gap distribution vs the 30 mm S3 reach, pre-grasp non-frozen chunks of T28+T30 E3 replays.
gap = min over hand spheres, anchor points of |p-c| - r, read BEFORE the frame's decision (what S3 reads).
pipe = AG3S._hand_for_grounding hand (constraint model at q_now, record finger joints); npz = recorded sphere_centres palm+finger rows.
hidden = manipulated state occluded or reason subset_kept_anchor (T32b classification); switches = decision mode 'switch'."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, collections, pathlib
import numpy as np
P = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/off/d/parts'))
out = {"definitions": __doc__, "variants": {}}
for var in ("pipe", "npz"):
    cls = collections.defaultdict(list); sw = []; diff = []; supp = 0; n = 0; band = collections.Counter()
    hist_edges = [0, 10, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200, 1e9]
    for f in sorted(P.glob(f"{var}__*.json")):
        d = json.load(open(f)); rows = d["rows"]
        for i, e in enumerate(rows):
            v = e[var]
            if v.get("mode") == "switch":
                prev = [rows[j].get(f"hand_gap_anchor_pre_{var}_mm") for j in range(max(0, i - 2), i + 1)]
                sw.append({"run": d["label"], "t": e["t"], "new_nearest": v.get("nearest"), "gaps_last3_mm": prev, "frozen": e["frozen"]})
            if e["frozen"] or not e["phase"].startswith("pre"):
                continue
            n += 1
            gp, gn = e.get("hand_gap_anchor_pre_pipe_mm"), e.get("hand_gap_anchor_pre_npz_mm")
            if gp is not None and gn is not None:
                diff.append(gp - gn)
            g = e.get(f"hand_gap_anchor_pre_{var}_mm")
            if (v.get("hand") or {}).get("occluded_by_hand"):
                supp += 1
            if g is None:
                cls["none"].append(0); continue
            hidden = v.get("state") == "occluded" or v.get("reason") == "subset_kept_anchor"
            cls["hidden" if hidden else "visible"].append(g)
            if 25 <= g <= 35:
                band["hidden" if hidden else "visible"] += 1
    q = lambda a: None if not len(a) else {"n": len(a), "q": dict(zip(["0", "5", "25", "50", "75", "95", "100"], np.quantile(a, [0, .05, .25, .5, .75, .95, 1]).round(1).tolist())),
                                           "n_le_30": int((np.asarray(a) <= 30).sum()), "hist_mm": np.histogram(a, bins=hist_edges)[0].tolist()}
    out["variants"][var] = {"n_pre_nonfrozen_rows": n, "hidden": q(cls["hidden"]), "visible": q(cls["visible"]), "n_no_gap": len(cls["none"]),
                            "n_rows_occluded_by_hand_suppressed": supp, "n_rows_gap_25_35mm": dict(band), "hist_edges_mm": hist_edges,
                            "switches": sw, "n_switches": len(sw), "n_switches_non_frozen": sum(1 for s in sw if not s["frozen"])}
    if var == "pipe":
        dd = np.asarray(diff)
        out["pipe_minus_npz_gap_mm"] = {"n": len(dd), "median": float(np.median(dd)), "q05": float(np.quantile(dd, .05)), "q95": float(np.quantile(dd, .95)),
                                        "min": float(dd.min()), "max": float(dd.max()), "n_abs_gt_5mm": int((np.abs(dd) > 5).sum())}
    print(var, json.dumps({k: v for k, v in out["variants"][var].items() if k != "switches"}))
print(out.get("pipe_minus_npz_gap_mm"))
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T33/off/d/summ_d.json'), "w"), indent=1, default=float)
