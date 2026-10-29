"""t16_reasons.json → 집계 (chunk 별 kind, execute 전환 수, collision 행 신원·물체 거리)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco
sys.path.insert(0, _ag3s_workspace_path(''))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T23/m1'))
RUN = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16'))
d = json.loads((OUT / "t16_reasons.json").read_text())
rows = [json.loads(l) for l in (RUN / "execute_ep1807/frames.jsonl").read_text().splitlines() if l.strip()]
plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
m = mujoco.MjModel.from_xml_path(_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'))
dd = mujoco.MjData(m)
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
GRASP = range(72, 177)


def objects_base(seq):
    r = plan[seq]
    dd.qpos[:] = np.asarray(r["qpos"]); mujoco.mj_kinematics(m, dd)
    R, p = dd.xmat[bid].reshape(3, 3), dd.xpos[bid]
    return {k: R.T @ (np.asarray(v["pos"]) - p) for k, v in r["object_poses"].items()}


vnames = list(d["chunks"][0]["variants"])
summary = {"n_chunks": len(d["chunks"]), "chunks": [], "per_variant": {}}
for v in vnames:
    kinds_count, exec_chunks, coll_chunks, unv_only = {}, [], [], []
    for c in d["chunks"]:
        e = c["variants"][v]
        for k in set(e["kinds"]):
            kinds_count[k] = kinds_count.get(k, 0) + 1
        if e["safe"]:
            exec_chunks.append(c["chunk_index"])
        if "collision" in e["kinds"]:
            coll_chunks.append(c["chunk_index"])
        if set(e["kinds"]) & {"collision", "uncertified", "comms"} == set() and "unverified" in e["kinds"]:
            unv_only.append(c["chunk_index"])
    g = [c for c in d["chunks"] if c["t_step"] in GRASP]
    summary["per_variant"][v] = {
        "chunks_with_kind": kinds_count, "n_execute": len(exec_chunks), "execute_chunks": exec_chunks,
        "n_hold": len(d["chunks"]) - len(exec_chunks),
        "collision_chunks": coll_chunks, "n_collision_chunks": len(coll_chunks),
        "unverified_only_chunks": unv_only, "n_unverified_only": len(unv_only),
        "grasp_window_t72_176": {"n": len(g), "n_execute": sum(1 for c in g if c["variants"][v]["safe"])},
    }
for c in d["chunks"]:
    objs = objects_base(c["seq"])
    e = c["variants"]["main|allow_unresolved=False"]
    coll = []
    for r in c.get("rows_main", []):
        if r["class"] != "collision":
            continue
        p = np.asarray(r["point_m"])
        dist = {k: float(np.linalg.norm(p - v) * 1000) for k, v in objs.items()}
        coll.append({"link": r["link"], "why": r["why"], "tier": r["tier"], "step": r["step"],
                     "clearance_mm": r["clearance_m"] * 1000, "query_to_object_centre_mm": dist,
                     "nearest_object_centre": min(dist, key=dist.get)})
    groups = {}
    for r in coll:
        key = f"{r['link']}|{r['why']}|{r['tier']}"
        gg = groups.setdefault(key, {"n": 0, "worst_clearance_mm": 0.0, "nearest_object_centre": {}, "min_to_apple_mm": 1e9, "min_to_orange_mm": 1e9, "min_to_crate_mm": 1e9})
        gg["n"] += 1
        gg["worst_clearance_mm"] = min(gg["worst_clearance_mm"], r["clearance_mm"])
        gg["nearest_object_centre"][r["nearest_object_centre"]] = gg["nearest_object_centre"].get(r["nearest_object_centre"], 0) + 1
        for o in ("apple", "orange", "crate"):
            gg[f"min_to_{o}_mm"] = min(gg[f"min_to_{o}_mm"], r["query_to_object_centre_mm"][o])
    cls = e["classification"]
    summary["chunks"].append({
        "chunk_index": c["chunk_index"], "seq": c["seq"], "t_step": c["t_step"], "attached": c["attached"],
        "old": {"safe": c["recorded"]["safe"], "trajopt_status": c["recorded"]["trajopt_status"],
                "clearance_mm": (c["recorded"]["clearance_m"] or 0) * 1000,
                "worst_link": (c["recorded"]["max_violation_pair"] or {}).get("link")},
        "kinds": {v: c["variants"][v]["kinds"] for v in vnames},
        "safe": {v: c["variants"][v]["safe"] for v in vnames},
        "main_counts": {k: cls.get(k) for k in ("n_violating", "n_allowed", "n_unresolved", "n_collision", "collision_by_why", "residual_violation_m")},
        "main_unresolved_by_tier": {t: sum(1 for r in c.get("rows_main", []) if r["class"] == "unresolved" and r["tier"] == t) for t in ("coarse", "window_edge", "outside")},
        "main_collision_groups": groups,
        "attached_recorded_row": e.get("attached_recorded_row"),
        "fidelity_record_clearance_max_abs_diff_mm": (c.get("fidelity") or {}).get("record_clearance", {}).get("max_abs_diff_mm"),
        "fidelity_to_view_abs_diff_mm": (c.get("fidelity") or {}).get("to_view", {}).get("abs_diff_mm"),
        "apple_base": c["apple_base"],
    })
(OUT / "t16_reasons_summary.json").write_text(json.dumps(summary, indent=1))
for v, s in summary["per_variant"].items():
    print(v, json.dumps(s))
for c in summary["chunks"]:
    print(c["chunk_index"], c["t_step"], c["kinds"]["main|allow_unresolved=False"], c["main_counts"]["n_unresolved"], c["main_unresolved_by_tier"], {k: (g["n"], round(g["worst_clearance_mm"], 1), g["nearest_object_centre"], round(g["min_to_apple_mm"]), round(g["min_to_crate_mm"])) for k, g in c["main_collision_groups"].items()})
