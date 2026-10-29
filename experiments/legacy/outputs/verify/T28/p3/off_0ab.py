"""T28 Phase 3 offline 0a / 0b — T26 grounding identity replayed on every recorded chunk of T14 (31) and T17 (75).

Input per chunk: server record chunk_k.npz (attention_cloud, attention_values, attention_raw, support_mask, t_step),
paired with local planning row seq k+1 (qpos, object_poses) — T14.audit §1.
Call (working tree, T26+T29):
  confirm = TargetConfirm(cfg.clustering, max_opening=0.07146, overlap_distance=0.005)    # T26.impl "offline replay API"
  ground_target(AttentionPointCloud(PointCloud(attention_cloud), attention_values, raw_attention=attention_raw), cfg.clustering,
                seed_percentile, seed_threshold, exclude_mask=support_mask, min_radius, timestamp=t_step, confirm=confirm)
  cfg = AG3SConfig() defaults (target_switch_min_score 0.1, target_lost_frames None).
Variants: A = never frozen; B = confirm.freeze(True) from the first chunk whose policy chunk (local actions_reference rows 0..7,
col 7) commands the left gripper < 0.5 (P1-2 'grasp attempt') to the end of the run (no latch in these records).
World frame: base-frame centroids -> MuJoCo world with the base body pose from that row's qpos (P1-2 helper).
GT apple distance = |manipulated centroid(world) - apple body pos (object_poses)|.
nearest body = nearest object centre among apple / banana / orange / pear / crate.
0b (offline part): exclusion gate replicated from AG3S._exclusion_gate (pipeline.py:1114): admissibility present & admissible &
narrowest <= max opening & overlap(manipulated.points, destination, 5 mm) < destination_overlap_fraction; ball radius =
quantile(|p - centroid|, esdf.target_ball_quantile) + target_ball_pad_voxels x fine voxel (0.005) — pipeline._target_ball;
z_min (support plane) is not replayed (no support surfaces in the record).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.types import PointCloud, AttentionPointCloud
from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm
from AG3S.ag3s.stages.admissibility import destination_overlap, principal_extents

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p3'))
RUNS = {"t14": (_ag3s_workspace_path('outputs/live_test/20260928_t14'), "execute_ep1807"),
        "t17": (_ag3s_workspace_path('outputs/live_test/20260928_t17'), "shadow_ep1807")}
MAX_OPENING = 0.07146
FINE = 0.005
cfg = AG3SConfig()
CENTRES = ("apple", "banana", "orange", "pear", "crate")


def load(run):
    d, loc = RUNS[run]
    rows = [json.loads(l) for l in open(f"{d}/{loc}/frames.jsonl") if '"planning"' in l]
    plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
    files = sorted(pathlib.Path(f"{d}/server_constraints/run_0000").glob("chunk_*.npz"))
    return plan, files


def to_world(m, d, p):
    return P.base_to_world(m, d, p)


def near(pw, op):
    cd = {b: float(1000 * np.linalg.norm(pw - np.asarray(op[b]["pos"]))) for b in CENTRES if b in op}
    k = min(cd, key=cd.get)
    return k, cd[k], cd


def replay(run, variant, m, d):
    plan, files = load(run)
    confirm = TargetConfirm(cfg.clustering, max_opening=MAX_OPENING, overlap_distance=FINE)
    attempt = None
    for f in files:
        k = int(f.stem.split("_")[1]); pr = plan.get(k + 1)
        if pr is not None and np.asarray(pr["actions_reference"])[:8, 7].min() < 0.5:
            attempt = k; break
    out = []
    for f in files:
        k = int(f.stem.split("_")[1]); pr = plan.get(k + 1)
        z = np.load(f, allow_pickle=True)
        if variant == "B_freeze_from_grasp_attempt" and attempt is not None and k >= attempt:
            confirm.freeze(True)
        apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)), z["attention_values"],
                                  raw_attention=z["attention_raw"])
        g = ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                          seed_threshold=cfg.attention.seed_threshold, exclude_mask=z["support_mask"],
                          min_radius=cfg.geometry.min_radius, timestamp=float(z["t_step"]), confirm=confirm)
        e = {"chunk": k, "t_step": int(z["t_step"]), "has_local_row": pr is not None,
             "grounding_status": str(g.status.value), "frozen": bool(confirm.frozen),
             "n_clusters": len(g.clusters or ())}
        dec = confirm.last
        e["decision"] = None if dec is None else {"mode": dec.mode, "rank": dec.rank, "reason": dec.reason,
                                                   "n_admissible": dec.n_admissible, "frozen": dec.frozen,
                                                   "streak": dec.streak}
        man = confirm.manipulated
        dest = confirm.destination
        e["destination"] = None if dest is None else dest.record()
        if pr is not None:
            P.set_q(m, d, pr["qpos"]); op = pr["object_poses"]
        # clusters (score order) with admissibility
        cl = []
        adm_list = confirm.last_admissibility or []
        for i, c in enumerate(g.clusters or ()):
            a = adm_list[i].record() if i < len(adm_list) else None
            row = {"rank": i, "score": float(c.target_score), "n": int(c.point_count), "admissibility": a}
            if pr is not None:
                cw = to_world(m, d, c.centroid); b, dm, _ = near(cw, op)
                row.update({"nearest_centre": b, "nearest_centre_mm": dm,
                            "to_apple_mm": float(1000 * np.linalg.norm(cw - np.asarray(op["apple"]["pos"])))})
            cl.append(row)
        e["clusters"] = cl
        if man is None:
            e["manipulated"] = None
        else:
            rec = man.record()
            mw = to_world(m, d, man.centroid) if pr is not None else None
            e["manipulated"] = {"id": rec["id"], "state": rec["state"], "age_frames": rec["age_frames"],
                                "switched_from": rec.get("switched_from"), "admissibility": rec.get("admissibility")}
            if mw is not None:
                b, dm, cd = near(mw, op)
                e["manipulated"].update({"nearest_centre": b, "to_apple_mm": cd["apple"], "to_crate_mm": cd.get("crate"),
                                         "centre_dist_mm": cd})
            # 0b gate + ball
            adm = man.admissibility
            pts = None
            try:
                pts = np.asarray(man.points, np.float64).reshape(-1, 3)
            except Exception:
                pts = None
            problem = None; ov = None
            if not man.usable:
                problem = "not usable"
            elif adm is None:
                problem = "no admissibility evidence"
            elif not adm.admissible:
                problem = f"inadmissible ({adm.reason})"
            elif adm.narrowest_extent_m is None or adm.narrowest_extent_m > MAX_OPENING + 1e-12:
                problem = "extent > max opening / unknown"
            else:
                ov = destination_overlap(pts, dest, FINE) if pts is not None else None
                if dest is not None and ov is not None and ov >= cfg.clustering.destination_overlap_fraction:
                    problem = f"overlaps destination {dest.id} ({ov:.2f})"
            gate = {"passed": problem is None and man.usable, "problem": problem, "overlap_now": ov}
            if problem is None and pts is not None and len(pts):
                c = pts.mean(0); spread = np.linalg.norm(pts - c, axis=1)
                r = float(np.quantile(spread, cfg.esdf.target_ball_quantile)) + cfg.esdf.target_ball_pad_voxels * FINE
                gate["ball_radius_mm"] = 1000 * r
                gate["ball_n_points"] = int(len(pts))
                if pr is not None:
                    cw = to_world(m, d, c); b, dm, cd = near(cw, op)
                    gate.update({"ball_centre_nearest": b, "ball_centre_to_apple_mm": cd["apple"]})
            e["gate"] = gate
        out.append(e)
    return {"grasp_attempt_chunk": attempt, "chunks": out}


def summarise(run, res, apple_true):
    L = [e for e in res["chunks"] if e["has_local_row"]]
    man_not_apple = [e["chunk"] for e in L if e["manipulated"] is not None and e["manipulated"]["nearest_centre"] != "apple"]
    no_man = [e["chunk"] for e in L if e["manipulated"] is None]
    apple_clusters = [c for e in L for c in e["clusters"] if c.get("nearest_centre") == "apple" and c["admissibility"]]
    ne = np.array([c["admissibility"]["narrowest_extent_mm"] for c in apple_clusters if c["admissibility"]["narrowest_extent_mm"] is not None])
    man_ext = np.array([e["manipulated"]["admissibility"]["narrowest_extent_mm"] for e in L
                        if e["manipulated"] and e["manipulated"].get("admissibility") and e["manipulated"]["admissibility"].get("narrowest_extent_mm") is not None])
    gates = [e["gate"] for e in L if e.get("manipulated")]
    balls = [g for g in gates if "ball_radius_mm" in g]
    plate = [(e["chunk"], c["rank"], c["admissibility"]["extents_mm"]) for e in L for c in e["clusters"]
             if c["admissibility"] and c["admissibility"]["extents_mm"][1] > 1000 * MAX_OPENING and c["admissibility"]["admissible"]]
    modes = {}
    for e in L:
        k = e["decision"]["mode"] if e["decision"] else None
        modes[k] = modes.get(k, 0) + 1
    reasons = {}
    for e in L:
        for c in e["clusters"]:
            if c["admissibility"]:
                key = f'{c.get("nearest_centre")}:{c["admissibility"]["reason"]}'
                reasons[key] = reasons.get(key, 0) + 1
    dest = [(e["chunk"], e["destination"]["id"], e["destination"]["source"]) for e in L if e["destination"]]
    dfirst = dest[0] if dest else None
    dest_ids = sorted({x[1] for x in dest})
    dest_near = None
    return {
        "n_chunks": len(L), "grasp_attempt_chunk": res["grasp_attempt_chunk"],
        "n_manipulated_not_apple": len(man_not_apple), "chunks_manipulated_not_apple": man_not_apple,
        "n_no_manipulated": len(no_man), "chunks_no_manipulated": no_man,
        "manipulated_ids": sorted({e["manipulated"]["id"] for e in L if e["manipulated"]}),
        "manipulated_state_counts": {s: sum(1 for e in L if e["manipulated"] and e["manipulated"]["state"] == s) for s in ("visible", "occluded", "lost")},
        "decision_mode_counts": {str(k): v for k, v in modes.items()},
        "manipulated_to_apple_mm": {"min": float(min(e["manipulated"]["to_apple_mm"] for e in L if e["manipulated"])),
                                    "median": float(np.median([e["manipulated"]["to_apple_mm"] for e in L if e["manipulated"]])),
                                    "max": float(max(e["manipulated"]["to_apple_mm"] for e in L if e["manipulated"]))} if any(e["manipulated"] for e in L) else None,
        "apple_cluster_narrowest_extent_mm": {"n": int(ne.size), "min": float(ne.min()), "p50": float(np.median(ne)), "p95": float(np.percentile(ne, 95)),
                                              "max": float(ne.max()), "min_margin_to_max_opening_mm": float(1000 * MAX_OPENING - ne.max()),
                                              "n_over_max_opening": int((ne > 1000 * MAX_OPENING).sum()), "values": ne.tolist()} if ne.size else None,
        "manipulated_narrowest_extent_mm": {"min": float(man_ext.min()), "max": float(man_ext.max()),
                                            "min_margin_to_max_opening_mm": float(1000 * MAX_OPENING - man_ext.max())} if man_ext.size else None,
        "cluster_reason_counts_by_nearest_centre": reasons,
        "destination_first": dfirst, "destination_ids": dest_ids, "n_chunks_with_destination": len(dest),
        "admissible_plate_like_clusters(2nd extent > max opening)": plate,
        "0b_gate_passed": sum(1 for g in gates if g["passed"]), "0b_gate_refused(invariant_violation)": sum(1 for g in gates if not g["passed"]),
        "0b_gate_problems": sorted({g["problem"] for g in gates if g["problem"]}),
        "0b_ball_radius_mm": {"n": len(balls), "min": min(b["ball_radius_mm"] for b in balls), "median": float(np.median([b["ball_radius_mm"] for b in balls])),
                              "max": max(b["ball_radius_mm"] for b in balls)} if balls else None,
        "0b_ball_centre_nearest_counts": {k: sum(1 for b in balls if b.get("ball_centre_nearest") == k) for k in {b.get("ball_centre_nearest") for b in balls}},
        "0b_ball_centre_to_apple_mm_max": max((b.get("ball_centre_to_apple_mm", 0) for b in balls), default=None),
        "apple_true_extents_mm": apple_true,
    }


def apple_true_extents(m):
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    V = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            V.append(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]])
    if not V:
        return None
    V = np.vstack(V).astype(float)
    return {"aabb_mm": (1000 * (V.max(0) - V.min(0))).round(2).tolist(), "pca_mm": (1000 * principal_extents(V)).round(2).tolist()}


if __name__ == "__main__":
    m, d, probe, obj, fing = P.build()
    at = apple_true_extents(m)
    res = {"definitions": __doc__, "max_opening_m": MAX_OPENING, "config": {
        "target_switch_min_score": cfg.clustering.target_switch_min_score, "target_lost_frames": cfg.clustering.target_lost_frames,
        "target_confirm_frames": cfg.clustering.target_confirm_frames, "destination_overlap_fraction": cfg.clustering.destination_overlap_fraction,
        "target_ball_quantile": cfg.esdf.target_ball_quantile, "target_ball_pad_voxels": cfg.esdf.target_ball_pad_voxels},
        "runs": {}}
    for run in RUNS:
        for variant in ("A_never_frozen", "B_freeze_from_grasp_attempt"):
            r = replay(run, variant, m, d)
            s = summarise(run, r, at)
            res["runs"][f"{run}_{variant}"] = {"summary": s, "chunks": r["chunks"]}
            print(run, variant, json.dumps({k: v for k, v in s.items() if k not in ("apple_cluster_narrowest_extent_mm",)}, default=str)[:1500], flush=True)
            print("  apple narrowest", {k: v for k, v in (s["apple_cluster_narrowest_extent_mm"] or {}).items() if k != "values"}, flush=True)
    (OUT / "off_0ab.json").write_text(json.dumps(res, indent=1, default=float))
