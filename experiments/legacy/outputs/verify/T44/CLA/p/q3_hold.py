"""T44dA verifier Q3: C0 vs C1a HOLD chunks / runs by cause. For every HOLD chunk whose worst collision row is an authorized row with
why == answer_excludes_target: distance (mm) from the violating sphere centre to the nearest recorded target-free fine seed of each class
(npz target_free_ambiguous_ijk / target_free_nontarget_kept_ijk / target_free_unowned_kept_ijk, trace origin + voxel; centre = origin + ijk*voxel)
-> which kept class sits under the finger. -> CLA/q3_hold.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, collections
import numpy as np
C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA')); CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL'))
cg = json.load(open(C / "close_geom.json")); out = {}
for arm in ("C0", "C1a", "C1b"):
    runs = collections.defaultdict(set); chunks = collections.Counter(); det = []
    for f in sorted(glob.glob(str(C / "chunks" / f"{arm}_*.jsonl"))):
        n = pathlib.Path(f).stem; sc = (cg.get(n) or {}).get("seq_c")
        a = json.load(open(CL / "analysis" / f"{n}.json")); files = {c["seq"]: c["server_file"] for c in a["server_chunks"]}
        for r in map(json.loads, open(f)):
            if not r["hold"]: continue
            cw = r.get("collision_worst") or {}; kinds = sorted({q["kind"] for q in r["reasons"] if q["action"] != "execute"})
            if "uncertified" in kinds and r["grasp_state"] == "placed": cause = "uncertified_after_place"
            elif cw.get("why") == "answer_excludes_target":
                cause = "authorized_target_free_" + ("close" if sc is not None and sc <= r["seq"] <= sc + 2 else ("pre_close" if sc is not None and r["seq"] < sc else "post_close"))
            elif cw.get("why") == "unauthorized_link": cause = "strict_unauthorized_link"
            elif cw.get("why") == "attached_point": cause = "held_object_rows"
            else: cause = "other:" + "+".join(kinds) + ":" + str(cw.get("why"))
            chunks[cause] += 1; runs[cause].add(n)
            if cw.get("why") == "answer_excludes_target":
                z = np.load(files[r["seq"]], allow_pickle=True); p = np.asarray(cw["point_m"])
                o = np.asarray(z["target_free_trace_origin"]); v = float(z["target_free_trace_voxel_size"])
                dd = {}
                for k in ("ambiguous", "nontarget_kept", "unowned_kept", "removed"):
                    ijk = z[f"target_free_{k}_ijk"]
                    dd[k] = float(np.linalg.norm(o + ijk * v - p, axis=1).min() * 1e3) if len(ijk) else None
                det.append({"name": n, "seq": r["seq"], "cause": cause, "link": cw.get("link"), "label": cw.get("label"), "tier": cw.get("tier"),
                            "clearance_mm": 1e3 * cw["clearance_m"], "n_ambiguous": r["n_ambiguous"], "n_kept_nontarget": r["n_kept_nontarget"],
                            "nearest_seed_mm": dd, "split_n_split": r["split_n_split"]})
    out[arm] = {"hold_chunks_by_cause": dict(chunks), "hold_runs_by_cause": {k: sorted(v) for k, v in runs.items()},
                "n_hold_runs": len(set().union(*runs.values())) if runs else 0, "detail_answer_excludes_target": det}
    print(arm, out[arm]["n_hold_runs"], {k: (chunks[k], len(runs[k])) for k in chunks})
    for d in det:
        if d["cause"].endswith("close"): print("   ", d["name"], d["seq"], d["link"], d["label"], d["tier"], round(d["clearance_mm"], 1), {k: (None if v is None else round(v, 1)) for k, v in d["nearest_seed_mm"].items()})
json.dump(out, open(C / "q3_hold.json", "w"), indent=1)
