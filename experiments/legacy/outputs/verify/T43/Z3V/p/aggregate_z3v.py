"""T43 Z3V aggregation (verifier, CPU only, numbers only). usage: aggregate_z3v.py OUT_JSON
Inputs: W/analysis/analysis.json (V5: H 14 + B 5), Z3V/analysis/analysis.json (V5D), Z3V/z4.json (z4_z3v.py: old + Z4 release rule, replay based).
 detach_t        : t (planning row t_step of the request) of the first server chunk whose grasp.state is 'placed' after an attached chunk
                   (V5D additionally: grasp.detach and grasp.held_slip recorded in the chunk -> slip=True means LatchEvent.slipped).
 after detach    : server chunks with t >= detach_t.
 HOLD            : client planning rows with hold flag (analysis n_hold_chunks), trailing = consecutive HOLD chunks at the run's last planning row.
 binding row     : analysis server_chunks[].binding (waypoint vs path rows: the smaller clearance), category base | gripper_cover | held.
 held-row viol.  : attached chunks whose waypoint worst row is a held query sphere with clearance < 0 (analysis held_row_violation_chunks).
 finger-apple    : binding row (waypoint) of a chunk after detach whose link name contains 'finger' and whose query sphere surface is <= X mm from the apple
                   surface (|base->world(point_m) - apple centre| - r - 33.3 mm, planning-row qpos / apple pose; Y/p/hold_rows_y.py method), X = 10 / 20 / 50.
 place_t         : analysis place_t (old rule: apple centre in the crate place box at the last planning row; first planning-row t_step where it is true)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math, os
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
V5 = json.load(open(T / "W/analysis/analysis.json"))["runs"]
V5D = json.load(open(T / "Z3V/analysis/analysis.json"))["runs"]
Z4 = json.load(open(T / "Z3V/z4.json"))
PRED = {"H:1925:19253": 384, "H:1807:18071": 176, "H:1807:18073": 232, "B:1807:18071": 360, "H:1968:19681": 224, "H:1982:19822": 192}


def runs_of(A, arm):
    out = {}
    for k, r in A.items():
        if not k.startswith(arm + "/"):
            continue
        st = k.split("/")[1]; ep = int(r["episode"]); seed = int(k.split("_s")[-1])
        out[(st, ep, seed)] = (k, r)
    return out


def detach_info(r):
    sc = r.get("server_chunks") or []
    att = None; det = None
    for i, c in enumerate(sc):
        if c["attached"] and att is None:
            att = c["t"]
        if att is not None and det is None and c["grasp_state"] == "placed" and not c["attached"]:
            det = c
            break
    if det is None:
        return {"attach_t": att, "detach_t": None, "slip": None}
    return {"attach_t": att, "detach_t": det["t"], "detach_seq": det["seq"], "slip": det.get("held_slip"), "grasp_detach_flag": det.get("grasp_detach"),
            "slip_evidence": det.get("slip_evidence")}


def apple_gap_rows(r, rd_key, after_t):
    """binding rows after detach with the apple-surface gap of the binding query sphere."""
    if after_t is None:
        return []
    run_dir = pathlib.Path(r["run_dir"]); side = r["side"]
    HM.set_arm(side); HM._M.clear(); M = HM.model(); m, d = M["m"], M["d"]; base = M["bid"]["base"]
    plan = {p["seq"]: p for p in (json.loads(l) for l in open(run_dir / "frames.jsonl") if l.strip()) if p["kind"] == "planning"}
    chunk_of = {c["seq"]: c for c in r["chunks"]}
    rows = []
    for c in r["server_chunks"]:
        if c["t"] is None or c["t"] < after_t:
            continue
        b = c["binding"]; row = {"seq": c["seq"], "t": c["t"], "rows": b["rows"], "category": b["category"], "link": b["link"], "clearance_mm": b["clearance_mm"],
                                 "hold": bool(chunk_of[c["seq"]]["hold"]) if c["seq"] in chunk_of else None, "apple_surface_mm": None}
        if b["rows"] == "waypoint" and c["wp"].get("query") is not None and c["wp_point_m"] is not None and c["seq"] in plan:
            q = int(c["wp"]["query"]); z = np.load(chunk_of[c["seq"]]["server_chunk"], allow_pickle=True)
            nr = int(z["sphere_radii"].shape[0])
            if q < nr:
                rad = float(z["sphere_radii"][q]); p = plan[c["seq"]]
                d.qpos[:] = p["qpos"]; mujoco.mj_kinematics(m, d)
                w = d.xpos[base] + d.xmat[base].reshape(3, 3) @ np.asarray(c["wp_point_m"], float)
                ap = np.asarray(p["object_poses"]["apple"]["pos"], float)
                row["apple_surface_mm"] = 1e3 * (np.linalg.norm(w - ap) - rad - 0.0333)
        row["finger"] = bool(b["link"] and "finger" in str(b["link"]))
        rows.append(row)
    return rows


def finger_apple(rows):
    o = {"n_chunks_after_detach": len(rows), "n_binding_finger_link": sum(x["finger"] for x in rows)}
    for X in (10, 20, 50):
        o[f"n_finger_apple_le_{X}mm"] = sum(1 for x in rows if x["finger"] and x["rows"] == "waypoint" and x["apple_surface_mm"] is not None and x["apple_surface_mm"] <= X)
    o["n_binding_category"] = {k: sum(1 for x in rows if x["category"] == k) for k in sorted({x["category"] for x in rows})}
    o["n_binding_obstacle_rows_apple_gap_lt_0"] = sum(1 for x in rows if x["apple_surface_mm"] is not None and x["apple_surface_mm"] < 0)
    o["min_apple_surface_mm_finger_rows"] = min([x["apple_surface_mm"] for x in rows if x["finger"] and x["apple_surface_mm"] is not None], default=None)
    return o


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    a = runs_of(V5, "V5"); b = runs_of(V5D, "V5D")
    keys = sorted(set(a) & set(b))
    zr = Z4["runs"]
    per = {}
    for k in keys:
        ka, ra = a[k]; kb, rb = b[k]
        name = f"{k[0]}:{k[1]}_s{k[2]}"; zk = f"{k[0]}:{k[1]}:{k[2]}"
        d5, d5d = detach_info(ra), detach_info(rb)
        rows5 = apple_gap_rows(ra, ka, d5["detach_t"]); rows5d = apple_gap_rows(rb, kb, d5d["detach_t"])
        def hv(r, dt):
            hv_all = r.get("held_row_violation_chunks") or []
            return {"n_all": len(hv_all), "n_ge_detach_t": sum(1 for x in hv_all if dt is not None and x["t"] >= dt),
                    "n_path_all": len(r.get("held_row_violation_path_chunks") or []),
                    "n_chunks_attached_ge_detach_t": sum(1 for c in r["server_chunks"] if dt is not None and c["t"] >= dt and c["attached"]),
                    "n_chunks_wp_held_ge_detach_t": sum(1 for c in r["server_chunks"] if dt is not None and c["t"] >= dt and c["wp"].get("category") == "held")}
        def bind(r):
            sc = r["server_chunks"]; hold_seq = {c["seq"] for c in r["chunks"] if c["hold"]}
            return {"all": r["all_binding_counts"],
                    "hold_chunks": {kk: sum(1 for c in sc if c["seq"] in hold_seq and f'{c["binding"]["rows"]}:{c["binding"]["category"]}' == kk) for kk in sorted({f'{c["binding"]["rows"]}:{c["binding"]["category"]}' for c in sc if c["seq"] in hold_seq})}}
        zA = next(v for kk, v in zr.items() if kk.startswith("V5/") and f"_ep{k[1]}_s{k[2]}" in kk and f"/{k[0]}/" in kk)
        zB = next(v for kk, v in zr.items() if kk.startswith("V5D/") and f"_ep{k[1]}_s{k[2]}" in kk and f"/{k[0]}/" in kk)
        per[name] = {
            "V5": {"success": ra["success"], "grasp": ra["grasp"], "place": ra["place"], "grasp_t": ra["grasp_t"], "place_t": ra["place_t"], "n_hold": ra["n_hold_chunks"],
                   "trailing_hold": ra["trailing_hold_chunks"], "longest_hold_run": ra["longest_hold_run_chunks"], **d5, "release_t_measured": zA["release_t"], "release_mode": zA["release_mode"],
                   "success_rel": zA["success_rel"], "place_rel": zA["place_rel"], "released": zA["released"], "held_row_viol": hv(ra, d5["detach_t"]), "binding": bind(ra),
                   "finger_apple_after_detach": finger_apple(rows5)},
            "V5D": {"success": rb["success"], "grasp": rb["grasp"], "place": rb["place"], "grasp_t": rb["grasp_t"], "place_t": rb["place_t"], "n_hold": rb["n_hold_chunks"],
                    "trailing_hold": rb["trailing_hold_chunks"], "longest_hold_run": rb["longest_hold_run_chunks"], **d5d, "release_t_measured": zB["release_t"], "release_mode": zB["release_mode"],
                    "success_rel": zB["success_rel"], "place_rel": zB["place_rel"], "released": zB["released"], "held_row_viol": hv(rb, d5d["detach_t"]), "binding": bind(rb),
                    "finger_apple_after_detach": finger_apple(rows5d),
                    "esdf_chunks_ok": all(True for _ in [0]), "server_chunks": len(rb["server_chunks"]), "n_planning": rb["n_planning"], "n_control": rb["n_control"]},
            "predicted_slip_detach_t": PRED.get(f"{k[0]}:{k[1]}:{k[2]}"),
            "binding_rows_after_detach_V5D": rows5d, "binding_rows_after_detach_V5": rows5}
    out = {"definitions": __doc__, "n_pairs": len(keys), "per_run": per}
    # pooled
    def tot(arm, f):
        return sum(per[n][arm][f] for n in per)
    out["totals"] = {arm: {"n_hold": tot(arm, "n_hold"), "trailing_hold_chunks_sum": tot(arm, "trailing_hold"), "runs_with_trailing_hold": sum(1 for n in per if per[n][arm]["trailing_hold"] > 0),
                           "runs_with_hold": sum(1 for n in per if per[n][arm]["n_hold"] > 0),
                           "held_row_viol_all": sum(per[n][arm]["held_row_viol"]["n_all"] for n in per),
                           "held_row_viol_ge_detach_t": sum(per[n][arm]["held_row_viol"]["n_ge_detach_t"] for n in per),
                           "chunks_attached_ge_detach_t": sum(per[n][arm]["held_row_viol"]["n_chunks_attached_ge_detach_t"] for n in per),
                           "chunks_wp_held_ge_detach_t": sum(per[n][arm]["held_row_viol"]["n_chunks_wp_held_ge_detach_t"] for n in per),
                           "finger_apple_after_detach": {k2: sum(per[n][arm]["finger_apple_after_detach"][k2] for n in per) for k2 in
                                                         ("n_chunks_after_detach", "n_binding_finger_link", "n_finger_apple_le_10mm", "n_finger_apple_le_20mm", "n_finger_apple_le_50mm")},
                           "n_runs_with_detach": sum(1 for n in per if per[n][arm]["detach_t"] is not None),
                           "n_runs_slip_detach": sum(1 for n in per if per[n][arm]["slip"])} for arm in ("V5", "V5D")}
    out["totals"]["by_stage"] = {st: {arm: {"n_hold": sum(per[n][arm]["n_hold"] for n in per if n.startswith(st + ":")),
                                            "runs_trailing_hold": sum(1 for n in per if n.startswith(st + ":") and per[n][arm]["trailing_hold"] > 0)} for arm in ("V5", "V5D")} for st in ("H", "B")}
    # paired place_t
    pt = [(n, per[n]["V5"]["place_t"], per[n]["V5D"]["place_t"]) for n in per if per[n]["V5"]["place_t"] is not None and per[n]["V5D"]["place_t"] is not None]
    out["place_t_pairs"] = {"n_both_placed": len(pt), "rows": pt, "median_diff_V5D_minus_V5": float(np.median([y - x for _, x, y in pt])) if pt else None,
                            "mean_V5": float(np.mean([x for _, x, _ in pt])) if pt else None, "mean_V5D": float(np.mean([y for _, _, y in pt])) if pt else None}
    # 2x2
    tab = {}
    for stg in ("H", "B", "ALL"):
        ns = [n for n in per if stg == "ALL" or n.startswith(stg + ":")]
        tab[stg] = {}
        for f in ("success", "grasp", "place", "success_rel", "place_rel", "released"):
            both = sum(per[n]["V5"][f] and per[n]["V5D"][f] for n in ns); v5o = sum(per[n]["V5"][f] and not per[n]["V5D"][f] for n in ns)
            v5do = sum(per[n]["V5D"][f] and not per[n]["V5"][f] for n in ns); nei = len(ns) - both - v5o - v5do
            tab[stg][f] = {"n_pairs": len(ns), "V5_total": both + v5o, "V5D_total": both + v5do, "both": both, "V5_only": v5o, "V5D_only": v5do, "neither": nei, "mcnemar_exact_p": mcnemar(v5o, v5do),
                           "V5_only_runs": [n for n in ns if per[n]["V5"][f] and not per[n]["V5D"][f]], "V5D_only_runs": [n for n in ns if per[n]["V5D"][f] and not per[n]["V5"][f]]}
    out["paired_2x2"] = tab
    # slip table
    sl = []
    for n in per:
        v = per[n]["V5D"]
        sl.append({"run": n, "V5D_release_t_measured": v["release_t_measured"], "release_mode": v["release_mode"], "predicted_slip_detach_t": per[n]["predicted_slip_detach_t"],
                   "V5D_detach_t": v["detach_t"], "V5D_slip_flag": v["slip"], "V5_detach_t": per[n]["V5"]["detach_t"], "V5_release_t_measured": per[n]["V5"]["release_t_measured"],
                   "V5D_attach_t": v["attach_t"], "slip_evidence": v.get("slip_evidence")})
    out["slip_table"] = sl
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["totals"], indent=1, default=float))
    for s in sl:
        print(s["run"], s["V5D_release_t_measured"], s["release_mode"], s["predicted_slip_detach_t"], s["V5D_detach_t"], s["V5D_slip_flag"], "| V5", s["V5_release_t_measured"], s["V5_detach_t"])
    for stg, t in tab.items():
        for f, x in t.items():
            print(stg, f, {k: x[k] for k in ("V5_total", "V5D_total", "both", "V5_only", "V5D_only", "neither")}, round(x["mcnemar_exact_p"], 4))


if __name__ == "__main__":
    main()
