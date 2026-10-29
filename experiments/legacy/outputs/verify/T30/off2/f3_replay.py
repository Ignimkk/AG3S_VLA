"""T30 #2 — F3 (association + pre-grasp anchor) replayed on recorded grounding inputs: E3a/E3b (T28 Phase 3, 12 runs) + T14 + T17.

Input per chunk: server record chunk_k.npz (attention_cloud, attention_values, attention_raw, support_mask, t_step),
paired with the local planning row (E3: T28 p3/<COND>_analysis.json 'server_chunk'; T14/T17: seq k+1, T14.audit §1).
Call (working tree = T30):
  confirm = TargetConfirm(cfg.clustering, max_opening=0.07146, overlap_distance=0.005)            # 'new' (T30 defaults:
            manipulated_extent_ratio 2.5, manipulated_anchor_coverage 0.9)
  confirm = TargetConfirm(..., extent_ratio=None, anchor_coverage=None)                             # 'old' (T26 behaviour) — the
            fidelity check: its manipulated id/state/centroid must reproduce the server record (E3 only)
  ground_target(AttentionPointCloud(...), cfg.clustering, seed_percentile, seed_threshold, exclude_mask=support_mask,
                min_radius, timestamp=t_step, confirm=confirm)
Freeze: E3 -> confirm.freeze(summary.exclusion.grasp_active) of that chunk (the recorded latch; T30b note);
        T14/T17 (no latch record) -> freeze from the first chunk whose policy chunk (actions_reference rows 0..7, col 7) < 0.5
        to the end (T28 off_0ab variant B).
Exclusion gate + ball: pipeline._exclusion_gate (usable, admissible, narrowest <= 71.46 mm, destination overlap < fraction) and
pipeline._target_ball: centre = mean(manipulated.points), radius = quantile(|p - c|, esdf.target_ball_quantile) +
target_ball_pad_voxels x 5 mm; z_min not applied (support planes are not in the record).
GT apple: MuJoCo body 'apple' pos from the planning row object_poses, brought into the robot base frame with the base pose of
that row's qpos. apple radius R_APPLE = 34 mm (T30b / lead definition).
  centre_inside = |c_ball - apple| <= r_ball ;  whole_inside = |c_ball - apple| + 34 mm <= r_ball ;  no ball -> neither.
Occupancy behind an uncovered apple (E3 only, whole_inside False): voxels of the RECORDED fine layer (esdf_distance, with the apple,
5 mm) with distance <= 0 whose centre is within 34 + 15 mm of the GT apple centre, split into inside / outside the replayed ball.
Voxels outside the ball are the ones the target-free layer would still carry (the target-free layer = the fine layer with the
ball's seeds excluded). Validation of that reading on the ORIGINAL records: recorded target-free layer (esdf_free_distance)
occupied voxels within apple+15 mm vs recorded-fine occupied voxels within apple+15 mm and outside the RECORDED ball.
Groups (coordinator note): pre_grasp_nonfrozen (main), frozen (latch closing/held -> grasp_active), after_switch (any chunk after the
first T26 'switch' decision in that run).
usage: f3_replay.py
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, collections
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa: E402
from AG3S.ag3s.config import AG3SConfig  # noqa: E402
from AG3S.ag3s.types import PointCloud, AttentionPointCloud  # noqa: E402
from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm  # noqa: E402
from AG3S.ag3s.stages.admissibility import destination_overlap  # noqa: E402

T28 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30/off2'))
MAX_OPENING = 0.07146
FINE = 0.005
R_APPLE = 0.034
PAD_OCC = 0.015
cfg = AG3SConfig()
CENTRES = ("apple", "banana", "orange", "pear", "crate")
LIVE = {"t14": (_ag3s_workspace_path('outputs/live_test/20260928_t14'), "execute_ep1807"),
        "t17": (_ag3s_workspace_path('outputs/live_test/20260928_t17'), "shadow_ep1807")}


def world_to_base(m, d, pw):
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = d.xmat[bid].reshape(3, 3)
    return R.T @ (np.asarray(pw, float) - d.xpos[bid])


def gate_and_ball(confirm):
    man = confirm.manipulated
    if man is None or not man.usable:
        return None, "not usable" if man is not None else "none"
    adm = man.admissibility
    if adm is None:
        return None, "no admissibility"
    if not adm.admissible:
        return None, f"inadmissible ({adm.reason})"
    if adm.narrowest_extent_m is None or adm.narrowest_extent_m > MAX_OPENING + 1e-12:
        return None, "extent"
    dest = confirm.destination
    ov = destination_overlap(man.points, dest, FINE)
    if dest is not None and ov >= cfg.clustering.destination_overlap_fraction:
        return None, f"overlaps destination {dest.id}"
    pts = man.points
    c = pts.mean(0)
    r = float(np.quantile(np.linalg.norm(pts - c, axis=1), cfg.esdf.target_ball_quantile)) + cfg.esdf.target_ball_pad_voxels * FINE
    return (c, r), None


def occupied_near(grid_d, origin, voxel, apple_b, radius):
    """indices -> centres of voxels with d <= 0 within `radius` of apple_b (base)."""
    g = np.asarray(grid_d, np.float32)
    o = np.asarray(origin, np.float64)
    lo = np.floor((apple_b - radius - o) / voxel).astype(int)
    hi = np.ceil((apple_b + radius - o) / voxel).astype(int) + 1
    lo = np.clip(lo, 0, np.array(g.shape)); hi = np.clip(hi, 0, np.array(g.shape))
    if np.any(hi <= lo):
        return np.zeros((0, 3))
    sub = g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    idx = np.argwhere(np.isfinite(sub) & (sub <= 0.0)) + lo
    cen = o + idx * voxel
    keep = np.linalg.norm(cen - apple_b, axis=1) <= radius
    return cen[keep]


def replay_run(label, chunks, m, d, is_e3, attempt=None):
    """chunks: list of (npz path, planning row, meta). Returns per-chunk rows for both rules."""
    confirms = {"new": TargetConfirm(cfg.clustering, max_opening=MAX_OPENING, overlap_distance=FINE)}
    if is_e3:
        confirms["old"] = TargetConfirm(cfg.clustering, max_opening=MAX_OPENING, overlap_distance=FINE,
                                        extent_ratio=None, anchor_coverage=None)
    rows = []
    switched = {k: False for k in confirms}
    for i, (f, pr, meta) in enumerate(chunks):
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        if is_e3:
            frozen = bool((s.get("exclusion") or {}).get("grasp_active"))
        else:
            frozen = attempt is not None and meta["k"] >= attempt
        P.set_q(m, d, pr["qpos"])
        op = pr["object_poses"]
        apple_b = world_to_base(m, d, op["apple"]["pos"])
        others_b = {b: world_to_base(m, d, op[b]["pos"]) for b in CENTRES if b in op}
        e = {"run": label, "t": int(pr["t_step"]), "seq": int(pr["seq"]), "file": str(f), "frozen": frozen,
             "latch": (s.get("grasp") or {}).get("state"), "phase": meta.get("phase"), "apple_z": float(op["apple"]["pos"][2])}
        rec_man = s.get("manipulated")
        e["recorded_manipulated"] = None if not rec_man else {"id": rec_man["id"], "state": rec_man["state"]}
        tf = (s.get("esdf_stats") or {}).get("target_free")
        if isinstance(tf, str):
            import ast
            tf = ast.literal_eval(tf)
        e["recorded_ball"] = (tf or {}).get("ball") if isinstance(tf, dict) else None
        for rule, confirm in confirms.items():
            confirm.freeze(frozen)
            apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)), z["attention_values"],
                                      raw_attention=z["attention_raw"])
            ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                          seed_threshold=cfg.attention.seed_threshold, exclude_mask=z["support_mask"],
                          min_radius=cfg.geometry.min_radius, timestamp=float(z["t_step"]), confirm=confirm)
            dec = confirm.last
            if dec is not None and dec.mode == "switch":
                switched[rule] = True
            man = confirm.manipulated
            v = {"mode": None if dec is None else dec.mode, "reason": None if dec is None else dec.reason,
                 "after_switch": switched[rule]}
            if man is not None:
                rec = man.record()
                dd = {b: float(1000 * np.linalg.norm(others_b[b] - man.centroid)) for b in others_b}
                v.update({"id": rec["id"], "state": rec["state"], "nearest_centre": min(dd, key=dd.get), "to_apple_mm": dd["apple"],
                          "association": rec.get("association"), "anchor": rec.get("anchor")})
                if rule == "old" and rec_man:
                    v["centroid_vs_record_mm"] = float(1000 * np.linalg.norm(np.asarray(rec_man["centroid"]) - man.centroid))
                    v["id_state_match_record"] = (rec["id"] == rec_man["id"] and rec["state"] == rec_man["state"])
            ball, problem = gate_and_ball(confirm)
            v["gate_problem"] = problem
            if ball is not None:
                c, r = ball
                da = float(np.linalg.norm(apple_b - c))
                v["ball"] = {"centre_b": c.tolist(), "radius_mm": 1000 * r, "apple_centre_dist_mm": 1000 * da,
                             "centre_inside": bool(da <= r), "whole_inside": bool(da + R_APPLE <= r),
                             "neighbour_centres_inside": [b for b in others_b if b not in ("apple", "crate") and np.linalg.norm(others_b[b] - c) <= r]}
            else:
                v["ball"] = None
            # occupancy behind an uncovered apple (E3 only)
            if is_e3 and "esdf_distance" in z.files and not (v["ball"] and v["ball"]["whole_inside"]):
                occ = occupied_near(z["esdf_distance"], z["esdf_origin"], float(z["esdf_voxel_size"]), apple_b, R_APPLE + PAD_OCC)
                if ball is not None:
                    out_ball = np.linalg.norm(occ - ball[0], axis=1) > ball[1]
                else:
                    out_ball = np.ones(len(occ), bool)
                v["occ_fine_near_apple"] = int(len(occ))
                v["occ_fine_near_apple_outside_ball"] = int(out_ball.sum())
            v["_ball_tuple"] = ball
            e[rule] = v
        # validation of the "target-free = fine minus ball" reading, original record
        if is_e3 and "esdf_free_distance" in z.files and e["recorded_ball"]:
            rb = e["recorded_ball"]
            occ_f = occupied_near(z["esdf_distance"], z["esdf_origin"], float(z["esdf_voxel_size"]), apple_b, R_APPLE + PAD_OCC)
            occ_tf = occupied_near(z["esdf_free_distance"], z["esdf_free_origin"], float(z["esdf_free_voxel_size"]), apple_b, R_APPLE + PAD_OCC)
            outside = np.linalg.norm(occ_f - np.asarray(rb["centre_m"]), axis=1) > rb["radius_mm"] / 1000.0
            below = occ_f[:, 2] < float(rb.get("z_min_m", -1e9))
            e["validation_recorded"] = {"fine_occ_near_apple": int(len(occ_f)), "fine_occ_outside_recorded_ball_or_below_zmin": int((outside | below).sum()),
                                        "fine_occ_outside_recorded_ball": int(outside.sum()),
                                        "target_free_occ_near_apple": int(len(occ_tf))}
        for rule in confirms:
            e[rule].pop("_ball_tuple", None)
        rows.append(e)
    return rows


def main():
    m, d, probe, objg, _ = P.build()
    res = {"definitions": __doc__, "runs": {}}
    for cond in ("E3a", "E3b"):
        an = json.loads((T28 / f"p3/{cond}_analysis.json").read_text())
        for rn, r in an["runs"].items():
            fr = [json.loads(l) for l in (T28 / f"{cond}_rec/{rn}/frames.jsonl").read_text().splitlines() if l.strip()]
            plan = {x["seq"]: x for x in fr if x["kind"] == "planning"}
            chunks = [(T28 / f"{cond}_rec" / c["server_chunk"], plan[c["seq"]], {"phase": c["phase"]}) for c in r["chunks"] if c.get("server_chunk")]
            res["runs"][rn] = replay_run(rn, chunks, m, d, True)
            print(rn, len(res["runs"][rn]), flush=True)
    for label, (root, loc) in LIVE.items():
        fr = [json.loads(l) for l in open(f"{root}/{loc}/frames.jsonl") if l.strip()]
        plan = {int(x["seq"]): x for x in fr if x["kind"] == "planning"}
        files = sorted(pathlib.Path(f"{root}/server_constraints/run_0000").glob("chunk_*.npz"))
        attempt = None
        for f in files:
            k = int(f.stem.split("_")[1]); pr = plan.get(k + 1)
            if pr is not None and np.asarray(pr["actions_reference"])[:8, 7].min() < 0.5:
                attempt = k; break
        chunks = [(f, plan[int(f.stem.split("_")[1]) + 1], {"k": int(f.stem.split("_")[1]), "phase": None})
                  for f in files if int(f.stem.split("_")[1]) + 1 in plan]
        res["runs"][label] = replay_run(label, chunks, m, d, False, attempt=attempt)
        res.setdefault("live_grasp_attempt_chunk", {})[label] = attempt
        print(label, len(res["runs"][label]), flush=True)
    (OUT / "f3_replay.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
