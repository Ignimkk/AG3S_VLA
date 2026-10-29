"""T43 CX — reversal boundaries split at the run's end-of-task row (verifier, numbers only). usage: cx_reversal_split.py OUT_JSON
Reads raw/cx_reversal.json (boundaries), V11/analysis/analysis_<ARM>.json (success, grasp_t, place_t), raw/cx_cases.json (t_sep of the 7 cases).
t_end per run: success -> place_t; the 7 CX cases in V11ABC (and V11AB 1967 s19672 H) -> drop row t_sep, or for no-grasp the first detach chunk t
(T43V11ABC.verify.json fail_stage detach_slip_chunks[0].t); other failed runs -> analysis place_t if any else grasp_t else none (all 'before').
before = t0 < t_end, after = t0 >= t_end. Per arm: boundaries / reversals before and after, in the 7 CX case runs and in the other runs."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX/raw')); V11 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11'))
R = json.load(open(X / "cx_reversal.json")); C = json.load(open(X / "cx_cases.json"))["cases"]
FS = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11ABC.verify.json')))["numbers"]["fail_stage"]["runs"]
cases = {f"{ck.split(':')[0]}/{ck.split(':')[1]}" for ck in C}
out = {"definitions": __doc__, "arms": {}}
for arm in ("V11AB", "V11ABC"):
    A = json.load(open(V11 / f"analysis/analysis_{arm}.json"))["runs"]; agg = {}
    per = {}
    for key, L in R["arms"][arm]["per_run"].items():
        a = A[key]; stg = key.split("/")[1]; nm = key.split("_ep")[1]; ck = f"{stg}:{nm.replace('_s', '_s')}"
        ck = f"{stg}:{nm}"
        tend = None; src = None
        if a.get("success"):
            tend, src = a.get("place_t"), "place_t"
        elif ck in C and C[ck]["arms"][arm]["timeline"].get("t_sep") is not None and C[ck]["fail_stage"][arm]["stage"] == "dropped":
            tend, src = C[ck]["arms"][arm]["timeline"]["t_sep"], "t_sep"
        elif ck in FS and FS[ck][arm]["detach_slip_chunks"]:
            tend, src = FS[ck][arm]["detach_slip_chunks"][0]["t"], "first_detach"
        else:
            tend, src = (a.get("place_t") or a.get("grasp_t")), "place_or_grasp_t"
        grp = "cx_case" if ck in C else "other"
        b = [x for x in L if tend is None or x["t0"] < tend]; f = [x for x in L if tend is not None and x["t0"] >= tend]
        per[key] = {"t_end": tend, "t_end_source": src, "group": grp, "before": [len(b), sum(x["reversal"] for x in b)], "after": [len(f), sum(x["reversal"] for x in f)]}
        for side, S in (("before", b), ("after", f)):
            k = f"{grp}_{side}"; agg.setdefault(k, [0, 0]); agg[k][0] += len(S); agg[k][1] += sum(x["reversal"] for x in S)
            for x in S:
                k2 = f"{grp}_{side}_state:{x['state']}"; agg.setdefault(k2, [0, 0]); agg[k2][0] += 1; agg[k2][1] += int(x["reversal"])
                if x["reversal"]:
                    k3 = f"{grp}_{side}_rev_vmin_mm"; agg.setdefault(k3, []); agg[k3].append(round(min(x["v_mm"]), 2))
    out["arms"][arm] = {"agg_n_boundaries_n_reversal": agg, "per_run": per}
    print(arm, agg)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
