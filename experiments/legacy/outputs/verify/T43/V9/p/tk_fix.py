"""T43 V9 TK correction (verifier, CPU, recorded data only). Recompute, for every T43TK contacts[] entry whose body is a finger,
clear_row_k_link_mm (record 'clearance' at row k = refined step, min over the spheres of the TO link) and
esdf_link_minus_geom_P_mm = clear_row_k_link_mm - bar_P_mm, with the TO link = the URDF name of the contacting MJCF body.
MJCF -> URDF finger map is DERIVED from OC raw/namemap_V8.json (nearest pairing of URDF-link sphere-centre mean to MJCF finger geom centre mean, run medians);
the old (MJCF-name) value is reproduced first as a check (must equal the TK value).
usage: tk_fix.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import dx_common as C
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
C.ANALYSES["V8"] = T / "V8/analysis/analysis.json"; C.ARM_SRC["V8"] = "V8"
TK = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43TK.verify.json')))
nm = json.load(open(T / "OC/raw/namemap_V8.json"))["runs"]
# derive map
mp = {}; evidence = {}
for key, v in nm.items():
    med = v["median_mm"]
    for k, x in med.items():                       # "urdf:ee_finger_l1->mjcf:ee_finger_l2_mm"
        u, mj = k[:-3].split("->"); u = u.split(":")[1]; mj = mj.split(":")[1]
        evidence.setdefault(mj, {})[u] = x
for mj, d in evidence.items():
    mp[mj] = min(d, key=d.get)
print("MJCF->URDF map (nearest):", mp, evidence)
out = {"definitions": __doc__, "mjcf_to_urdf": mp, "mjcf_to_urdf_distance_mm": evidence, "contacts": []}
for c in TK["numbers"]["contacts"]:
    r = {"arm": c["arm"], "key": c["key"], "body": c["body"], "kind": c["kind"], "phase_ct": c["phase_ct"], "t": c["t"], "k": c["k"], "seq": c["seq"],
         "bar_P_mm": c["bar_P_mm"], "old_clear_row_k_link_mm": c.get("clear_row_k_link_mm"), "old_esdf_link_minus_geom_P_mm": c.get("esdf_link_minus_geom_P_mm")}
    if c["body"] in mp and c.get("clear_row_k_link_mm") is not None:
        a = C.run_entry(c["key"]); ch = C.segment_chunks(a["server_segment"])
        with np.load(ch[c["seq"]], allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"])); links = C.lit(s.get("sphere_link_names")) or []; cl = np.asarray(z["clearance"], float)
        k = c["k"]
        f = lambda name: 1e3 * float(cl[k][np.asarray([l == name for l in links])].min())
        old = f(c["body"]); urdf = mp[c["body"]]; new = f(urdf)
        r.update({"urdf_link": urdf, "old_reproduced_mm": old, "old_matches_TK": abs(old - c["clear_row_k_link_mm"]) < 1e-3,
                  "new_clear_row_k_link_mm": new, "new_esdf_link_minus_geom_P_mm": new - c["bar_P_mm"], "n_spheres_link": int(sum(l == urdf for l in links))})
    out["contacts"].append(r)
# group stats (same st() as tk_agg)
def st(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0: return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 4), "p90": round(float(np.percentile(x, 90)), 4), "max": round(float(x.max()), 4)}
grp = {}
for r in out["contacts"]:
    g = f'{r["arm"]}|{r["kind"]}|{r["phase_ct"]}'
    grp.setdefault(g, []).append(r)
out["groups"] = {g: {"n_contacts": len(v),
                     "clear_row_k_link_mm": st([r.get("new_clear_row_k_link_mm") for r in v]),
                     "esdf_link_minus_geom_P_mm": st([r.get("new_esdf_link_minus_geom_P_mm") for r in v]),
                     "old_clear_row_k_link_mm": st([r["old_clear_row_k_link_mm"] for r in v]),
                     "old_esdf_link_minus_geom_P_mm": st([r["old_esdf_link_minus_geom_P_mm"] for r in v])} for g, v in grp.items()}
out["all_old_match"] = all(r.get("old_matches_TK", True) for r in out["contacts"])
# cross-check vs OC summary per-contact
oc = json.load(open(T / "OC/raw/oc_summary.json"))["contacts"]
xc = []
for o in oc:
    for r in out["contacts"]:
        if r["key"].split("/")[-1] == o.get("key", "").split("/")[-1] and r["t"] == o.get("t") and "new_esdf_link_minus_geom_P_mm" in r:
            xc.append({"key": r["key"], "mine": r["new_esdf_link_minus_geom_P_mm"], "oc": o.get("esdf_minus_geomP_urdf_mm")})
out["oc_crosscheck"] = xc
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
for g, v in out["groups"].items():
    print(g, v["n_contacts"], "new esdf-geom", v["esdf_link_minus_geom_P_mm"], "old", v["old_esdf_link_minus_geom_P_mm"])
print("all old reproduced:", out["all_old_match"]); print("oc xc max abs diff", max([abs(x["mine"] - x["oc"]) for x in xc if x["oc"] is not None] or [None]), len(xc))
