"""T33 offline (a) — H1 self-filter on recorded T30 clouds: GT held-apple points left in the cloud while held.

Runs: T30 E3b ep1807 r1/r2, E3a ep1807 r1/r3 (server chunk <-> local planning row pairing from T30/p/<COND>_analysis.json).
H2 spheres (rebuilt with the tree's own functions, as AG3S.attach/_held_primitives does):
  geometry  = target_points (base) of the chunk BEFORE the first grasp.attach chunk (the manipulated geometry attach used),
  snapshot  = planning-row q of that chunk (T32 H2 "manipulated last seen" capture pose),
  trim_outliers(20 mm, 3) -> fit_surface_sphere(max_radius = 71.46/2 mm) -> covering_spheres(pad = 5 mm fine voxel, max_spheres 4),
  parent    = palm ee_left (URDF link pose of the constraint robot model, T24 tool build_robot); spheres stored in the palm frame.
Self-filter held spheres = covering radius + HELD_FILTER_SLIP_M (20 mm), inflation 0 (T32 H1 _prepare_frame_filter).
Cloud     = recorded raw_cloud (base) = fused cloud AFTER the T30 robot self-filter (no held spheres) — applying the held spheres
            on top of it is the H1 filter (robot spheres unchanged; sim frozen, capture q = planning-row q).
GT apple point = raw_cloud point whose mj_geomDistance to the apple geoms is <= 5 mm and smaller than to every other object
            body and to the left-finger geoms (candidates: within 90 mm of the GT apple centre).  Also reported at <= 10 mm.
Per chunk from the attach chunk to the detach chunk (or the last chunk); held_frame = False for the detach chunk itself
(safe_policy advances the latch before AG3S processes that request, so the detach frame is filtered WITHOUT held spheres).
in_hand = GT apple centre offset in the palm (ee_left) frame within 30 mm of its offset at the attach chunk (T32a h2_eval rule).
Sums (grasp->release windows) use held frames that are in_hand only; 'last3' = the last 3 in-hand held chunks (= the approach to
the release over the crate): n apple pts in raw_cloud, n left after H1 (slip 20 mm),
n left with query spheres only (slip 0), n non-apple pts removed by the held spheres (by nearest body).
Windows: grasp_t / place_t / release_t from T31diag d1 (same rules)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d2'))
import p12_records as P
import d2_rows_ablation as D2
import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.ag3s.robot_models.held_object import fit_surface_sphere, covering_spheres, HELD_FILTER_SLIP_M
from AG3S.ag3s.constraints.attached import trim_outliers

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/off/a'))
RUNS = [("E3b", "E3b_ep1807_r1"), ("E3b", "E3b_ep1807_r2"), ("E3a", "E3a_ep1807_r1"), ("E3a", "E3a_ep1807_r3")]
D1 = json.load(open(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')))["runs"]
PAD = 0.005; MAXR = 0.07146 / 2


def main():
    m, d, probe, obj, fing = P.build()
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    res = {"definitions": __doc__, "slip_m": HELD_FILTER_SLIP_M, "pad_m": PAD, "runs": {}}
    for cond, run in RUNS:
        A = json.load(open(V / "p" / f"{cond}_analysis.json"))["runs"][run]
        chunks = [c for c in A["chunks"] if c.get("server_chunk")]
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in open(V / f"{cond}_rec/{run}/frames.jsonl") if l.strip()) if x.get("kind") == "planning"}
        summ = [json.loads(str(np.load(V / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)["summary_json"])) for c in chunks]
        ia = next(i for i, s in enumerate(summ) if (s.get("grasp") or {}).get("attach"))
        idet = next((i for i, s in enumerate(summ) if i > ia and (s.get("grasp") or {}).get("detach")), len(chunks) - 1)
        port = chunks[0]["server_chunk"].split("/")[0].split("_")[1]
        sh = D2.setup(cond, port, V / f"{cond}_rec" / run)
        robot, qadr = sh["robot"], sh["qadr"]
        zg = np.load(V / f"{cond}_rec" / chunks[ia - 1]["server_chunk"], allow_pickle=False)
        tp = np.asarray(zg["target_points"], np.float64)
        keep = trim_outliers(tp, 0.020, 3)
        pts = tp[keep] if int(keep.sum()) >= 4 else tp
        fit = fit_surface_sphere(pts, max_radius=MAXR)
        spheres, rec = covering_spheres(pts, fit, pad=PAD, max_spheres=4)
        qg = T.q_now_from_qpos(rows[chunks[ia - 1]["seq"]]["qpos"], qadr)
        Tg = robot.link_pose(qg, "ee_left")
        loc = [((c - Tg[:3, 3]) @ Tg[:3, :3], r) for c, r in spheres]
        w = D1[run]
        e = {"cond": cond, "t_attach": chunks[ia]["t"], "t_detach_or_last": chunks[idet]["t"], "detached": bool((summ[idet].get("grasp") or {}).get("detach")),
             "geometry_chunk_t": chunks[ia - 1]["t"], "n_target_points": int(len(tp)), "n_after_trim": int(len(pts)), "fit": fit.record(), "covering": rec,
             "held_query_radii_mm": [1000 * r for _, r in spheres], "held_filter_radii_mm": [1000 * (r + HELD_FILTER_SLIP_M) for _, r in spheres],
             "grasp_t": w["grasp_t"], "place_t": w["place_t"], "release_t": w["release_t"], "chunks": []}
        off0 = None
        for i in range(ia, idet + 1):
            c = chunks[i]; row = rows[c["seq"]]
            z = np.load(V / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
            cloud = np.asarray(z["raw_cloud"], np.float64)
            q = T.q_now_from_qpos(row["qpos"], qadr); Tp = robot.link_pose(q, "ee_left")
            C = np.array([Tp[:3, :3] @ lc + Tp[:3, 3] for lc, _ in loc]); R = np.array([r for _, r in loc])
            P.set_q(m, d, row["qpos"]); Rb = d.xmat[base].reshape(3, 3).copy(); pb = d.xpos[base].copy()
            cw = cloud @ Rb.T + pb
            ap = np.asarray(row["object_poses"]["apple"]["pos"])
            cand = np.flatnonzero(np.linalg.norm(cw - ap, axis=1) <= 0.09)
            lab5 = np.zeros(len(cloud), bool); lab10 = np.zeros(len(cloud), bool); nearest_body = {}
            for j in cand:
                body, cat, per = P.nearest(m, d, probe, obj, cw[j])
                fd = min(P.gdist(m, d, probe, g) for g in fing)
                nearest_body[j] = body if fd > per[body] else "left_finger"
                if body == "apple" and per["apple"] < fd:
                    lab10[j] = per["apple"] <= 0.010; lab5[j] = per["apple"] <= 0.005
            P.set_q(m, d, row["qpos"])
            gap = (np.linalg.norm(cloud[:, None, :] - C[None], axis=2) - R[None]).min(axis=1)
            inside_f = gap <= HELD_FILTER_SLIP_M          # within query radius + slip
            inside_q = gap <= 0.0
            removed_non_apple = collections.Counter()
            for j in np.flatnonzero(inside_f & ~lab10):
                removed_non_apple[nearest_body.get(j, "far_from_apple(>90mm)")] += 1
            t = c["t"]
            win = "pre" if t < w["grasp_t"] else ("transport" if t < w["place_t"] else ("place" if (w["release_t"] is None or t <= w["release_t"]) else "after_release"))
            off = Tp[:3, :3].T @ ((ap - pb) @ Rb - Tp[:3, 3])
            if off0 is None:
                off0 = off
            drift = float(1000 * np.linalg.norm(off - off0))
            held_frame = not (i == idet and e["detached"])     # safe_policy: detach applies to that frame's processing
            e["chunks"].append({"t": t, "seq": c["seq"], "window": win, "held_frame": held_frame, "apple_offset_drift_palm_mm": drift, "in_hand": drift <= 30.0, "n_cloud": int(len(cloud)),
                                "n_apple_5mm": int(lab5.sum()), "n_apple_5mm_left_H1": int((lab5 & ~inside_f).sum()),
                                "n_apple_5mm_left_query_only": int((lab5 & ~inside_q).sum()),
                                "n_apple_10mm": int(lab10.sum()), "n_apple_10mm_left_H1": int((lab10 & ~inside_f).sum()),
                                "max_gap_apple_pts_mm": float(1000 * gap[lab5].max()) if lab5.any() else None,
                                "n_removed_by_held_filter": int(inside_f.sum()), "removed_non_apple_by_nearest": dict(removed_non_apple),
                                "palm_to_gt_apple_mm": float(1000 * np.linalg.norm((ap - pb) @ Rb - Tp[:3, 3]))})
            x = e["chunks"][-1]
            print(run, t, win, "apple", x["n_apple_5mm"], "left", x["n_apple_5mm_left_H1"], "q-only", x["n_apple_5mm_left_query_only"],
                  "removed", x["n_removed_by_held_filter"], x["removed_non_apple_by_nearest"], flush=True)
        ch = [x for x in e["chunks"] if x["window"] in ("transport", "place") and x["held_frame"] and x["in_hand"]]
        e["sum_grasp_to_release"] = {"n_chunks": len(ch), "apple_5mm": sum(x["n_apple_5mm"] for x in ch),
                                     "apple_5mm_left_H1": sum(x["n_apple_5mm_left_H1"] for x in ch),
                                     "apple_5mm_left_query_only": sum(x["n_apple_5mm_left_query_only"] for x in ch),
                                     "apple_10mm": sum(x["n_apple_10mm"] for x in ch), "apple_10mm_left_H1": sum(x["n_apple_10mm_left_H1"] for x in ch),
                                     "chunks_with_any_left_5mm": sum(1 for x in ch if x["n_apple_5mm_left_H1"] > 0),
                                     "max_left_5mm_per_chunk": max((x["n_apple_5mm_left_H1"] for x in ch), default=0)}
        chp = [x for x in e["chunks"] if x["held_frame"] and x["in_hand"]][-3:]
        e["sum_last3_in_hand_chunks"] = {"t": [x["t"] for x in chp], "n_chunks": len(chp), "apple_5mm": sum(x["n_apple_5mm"] for x in chp),
                                 "apple_5mm_left_H1": sum(x["n_apple_5mm_left_H1"] for x in chp),
                                 "apple_10mm_left_H1": sum(x["n_apple_10mm_left_H1"] for x in chp)}
        res["runs"][run] = e
        print(run, e["sum_grasp_to_release"], e["sum_last3_in_hand_chunks"], flush=True)
    (OUT / "offline_a.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
