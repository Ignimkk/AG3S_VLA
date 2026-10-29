"""T44b D-sign verifier measurement (verifier-owned; benchmark/ is imported read-only, nothing there is edited).

Re-uses outputs/verify/T44/E2/e2_measure.py (helpers, frame lists) unchanged. Writes only under outputs/verify/T44/E2b/.

Per frame: ONE AG3S pass (same settings as V-E2), then for each identity mode (gt, gt_label) the same integrator builds
the target-free fine tier with
  ball | bo_d2 | bo_d3 | bo_d4 (ball_owned, unowned=remove, tie 1.0, owned_max_dist_voxels 2/3/4) | oracle | oracle_prime
  + diagnostics: none (only="fine" rebuild, no exclusion) | ball_noforce | bo_d2_noforce (sign correction disabled)
Fixed before measurement (2026-10-10, from T44b.task.md / T44.task.md): margin 10 mm, criteria (1)-(4) as V-E2,
query model V11AB (gripper links, finger cover both), same frame lists.

Unowned analysis (per d): GT-distractor seeds classified unowned -> nearest distance to (raw depth pixels, non-robot),
(per-camera clouds before fusion), (fused filtered cloud), (non-target evidence = fused - support - target),
(GT distractor mesh surface); waterfall cause per seed.

Sign diagnosis: voxel centres (fine lattice) inside each distractor convex hull within ball centre +- (r + 60 mm):
values of main fine layer / coarse layer / main composite / every free tier (raw) / composed target_free_distance per
free tier, answer tier, fine+coarse TSDF state (unallocated / weight<=min / observed sdf), in-ball, depth below surface.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, dataclasses, json, pathlib, sys, time
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2'))
sys.path.insert(0, _ag3s_workspace_path(''))
import e2_measure as M  # noqa: E402
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

IDENTITIES = ("gt", "gt_label")
DISTS = (2.0, 3.0, 4.0)
BO = tuple(f"bo_d{int(d)}" for d in DISTS)
METHODS = ("ball",) + BO + ("oracle", "oracle_prime")
DIAG = ("none", "ball_noforce", "bo_d2_noforce")
MARGIN = M.MARGIN
GRID = M.GRID


def tsdf_lookup(itg, points):
    """Read-only TSDF read at world points (same indexing as curobo_builder._free_tsdf_points).
    Returns state (0 unallocated block, 1 allocated weight<=min, 2 observed) and sdf (nan unless observed)."""
    ts = itg.tsdf
    data = ts.data if hasattr(ts, "data") else ts
    bd = data.block_data
    p = np.asarray(points, np.float64).reshape(-1, 3)
    state = np.zeros(len(p), np.int8)
    sdf = np.full(len(p), np.nan)
    wout = np.zeros(len(p))
    n_alloc = int(data.num_allocated.detach().cpu().reshape(-1)[0])
    if n_alloc <= 0 or not len(p):
        return state, sdf, wout
    bs = int(round(bd.shape[1] ** (1.0 / 3.0)))
    D, Hh, W = (int(v) for v in itg.grid_shape)
    vs = float(itg.voxel_size)
    origin = itg.origin.detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    coords = data.block_coords.detach().cpu().numpy().reshape(-1, 3)[:n_alloc].astype(np.int64)
    slots = data.block_to_hash_slot.detach().cpu().numpy()[:n_alloc]
    live = np.flatnonzero(slots >= 0)
    grid = np.array([W, Hh, D], np.int64)
    offsets = ((grid + bs - 1) // bs) // 2
    g = np.floor((p - origin) / vs + 0.5 * grid).astype(np.int64)
    key = g // bs - offsets
    loc = g % bs

    def pack(k):
        k = k + 4096
        return (k[:, 0] << 26) | (k[:, 1] << 13) | k[:, 2]
    live_keys = pack(coords[live])
    order = np.argsort(live_keys)
    sk = live_keys[order]
    want = pack(key)
    pos = np.clip(np.searchsorted(sk, want), 0, len(sk) - 1)
    hit = sk[pos] == want
    if not hit.any():
        return state, sdf, wout
    import torch
    pool = live[order[pos[hit]]]
    li = loc[hit, 2] * bs * bs + loc[hit, 1] * bs + loc[hit, 0]
    tp = torch.as_tensor(pool, device=bd.device, dtype=torch.long)
    tl = torch.as_tensor(li, device=bd.device, dtype=torch.long)
    w = bd[tp, tl, 1].float().cpu().numpy()
    s = bd[tp, tl, 0].float().cpu().numpy()
    minw = float(getattr(itg.config, "minimum_tsdf_weight", 0.1))
    hi = np.flatnonzero(hit)
    obs = w > minw
    state[hi] = np.where(obs, 2, 1)
    sdf[hi[obs]] = s[obs] / w[obs]
    wout[hi] = w
    return state, sdf, wout


def build_tier(builder, ball, *, owned=None, exclusion=None, noforce=False, no_ball=False):
    import torch
    itg = builder._mapper.integrator
    itg_fine = None if builder._fine_mapper is None else builder._fine_mapper.integrator
    dev = builder._mapper._device if hasattr(builder._mapper, "_device") else "cuda:0"
    if exclusion is not None:
        builder._ball_exclusion = exclusion
    if noforce:
        builder._force_sign_on_pure_voxels = lambda values, att_idx, vs, threshold_voxels=1.5: {
            "n_attached_voxels": 0 if att_idx is None else int(len(att_idx)), "n_sign_forced": 0,
            "noforce": True}
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        (tier,) = builder._build_tiers(itg, None, dev, torch, only="fine", tier_name="fine_no_target",
                                       ball=None if no_ball else ball, itg_fine=itg_fine, owned=owned)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
    finally:
        if exclusion is not None:
            del builder._ball_exclusion
        if noforce:
            del builder._force_sign_on_pure_voxels
    return tier, ms


def reseed_cand(builder, ball):
    """Fine-lattice in-ball candidates and their seed mask, with seeds freshly re-planted (read only)."""
    itg = (builder._fine_mapper or builder._mapper).integrator
    origin = itg._last_esdf_origin.clone()
    itg._seed_esdf_impl(origin, itg._esdf_voxel_size)
    vs = float(itg._esdf_voxel_size[0])
    shape = tuple(int(v) for v in itg._site_index.shape)
    org = builder._origin_of(itg, shape, vs)
    cand = builder._ball_candidates(ball, org, vs, shape)
    if cand is None:
        return None, None, org, vs
    return cand, builder._seed_mask(itg, cand), org, vs


def nearest(tree_pts, q):
    from scipy.spatial import cKDTree
    if tree_pts is None or not len(tree_pts) or not len(q):
        return np.full(len(q), np.inf)
    d, _ = cKDTree(tree_pts).query(q, k=1)
    return np.asarray(d, np.float64)


def tier_vals(tier, ijk):
    import torch
    t = torch.as_tensor(ijk, device=tier.values.device, dtype=torch.long)
    return tier.values[t[:, 0], t[:, 1], t[:, 2]].float().cpu().numpy().astype(np.float64)


def run_one(ctx, item, args):
    import torch
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence, _ball_owned_split
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField
    from AG3S.ag3s.fields.device_field import layer_from_device
    from AG3S.trajopt import wire

    scene, ag3s, target = ctx["scene"], ctx["ag3s"], ctx["target"]
    arm = item["arm"]
    load0 = M.pod_load()
    pose_err = item["pose"](scene)
    gt_centre = H.body_in_base(scene, target)
    frames = {c: scene.capture(c) for c in args.cameras}
    attention = H.gt_attention(scene, frames, target, gt_centre)
    req = wire.pack_request({}, cameras=args.cameras, depth={c: f.depth for c, f in frames.items()},
                            intrinsics={c: f.camera_intrinsics for c, f in frames.items()},
                            extrinsics={c: f.T_base_cam for c, f in frames.items()},
                            robot_state={c: f.robot_state for c, f in frames.items()},
                            stamps={c: float(item["seq"]) for c in args.cameras}, phase="approach",
                            active_manipulators=[arm], seq=int(item["seq"]))
    _, sc = wire.strip_request(req)
    set_finger_joints_from_scene(scene, ag3s.robot_model)
    if ag3s.constraint_robot_model is not None:
        set_finger_joints_from_scene(scene, ag3s.constraint_robot_model)
    obs = wire.unpack_camera_observations(sc, robot_model=ag3s.robot_model, attention=attention)
    t0 = time.perf_counter()
    cs, debug = ag3s.process_multi_debug(obs, phase="approach", active_manipulators=[arm])
    ag3s_ms = (time.perf_counter() - t0) * 1e3
    support_mask = debug.get("support_mask")
    surfaces = list(cs.support_surfaces)
    gt_depth_pts, gt_depth_lab = H.gt_points(frames.values(), ctx["classes"])
    label_of = H.point_labeler(gt_depth_pts, gt_depth_lab, radius=0.003)
    mesh_pts, mesh_lab, hulls = H.gt_mesh_points(scene, ctx["classes"])
    gt_pts, gt_lab = np.vstack([gt_depth_pts, mesh_pts]), np.concatenate([gt_depth_lab, mesh_lab])
    others = [n for n in H.FRUITS if n != target]
    builder = ag3s._esdf_builder
    itg_c = builder._mapper.integrator
    itg_f = (builder._fine_mapper or builder._mapper).integrator
    fusion = debug.get("fusion")
    per_cam = {r.camera_id.value: np.asarray(r.cloud.points, np.float64) for r in fusion.per_camera}
    per_cam_vox = {r.camera_id.value: r.stats.get("final_voxel_size") for r in fusion.per_camera}
    fused = np.asarray(debug["filtered_cloud"].points, np.float64)
    raw_nonrobot = gt_depth_pts[gt_depth_lab != H.LABELS.index("robot")]
    rec = {"key": item["key"], "seq": int(item["seq"]), "source": item["meta"], "set": item["set"],
           "bin": item["bin"], "target": target, "arm": arm, "attention_cameras": sorted(attention),
           "ag3s_ms": round(ag3s_ms, 2), "status": cs.status.value, "object_pose_err_mm": round(pose_err * 1e3, 3),
           "nearest_distractor_3d_mm": min(((n, round(float(np.linalg.norm(H.body_in_base(scene, n) - gt_centre)) * 1e3, 1))
                                            for n in others), key=lambda x: x[1]),
           "settings": {"fusion_voxel_size_m": float(ag3s.config.timing.fusion_voxel_size),
                        "fusion_metric_fusion_voxel_size_m": (fusion.metrics or {}).get("fusion_voxel_size_m"),
                        "pointcloud_voxel_size_m": float(getattr(ag3s.config.pointcloud, "voxel_size", float("nan"))),
                        "per_camera_final_voxel_size_m": per_cam_vox,
                        "n_points_per_camera": {k: int(len(v)) for k, v in per_cam.items()},
                        "n_points_fused": int(len(fused)),
                        "fine_voxel_m": float(builder.fine_voxel), "coarse_voxel_m": float(builder.coarse_voxel),
                        "fine_truncation_m": float(builder.fine_truncation),
                        "coarse_truncation_m": float(builder.coarse_truncation),
                        "minimum_tsdf_weight": float(getattr(itg_f.config, "minimum_tsdf_weight", float("nan"))),
                        "adjacent_skip_steps": float(getattr(itg_f.config, "adjacent_skip_steps", float("nan"))),
                        "seeding_method": str(getattr(itg_f.config, "seeding_method", "?")),
                        "tsdf_voxel_m": float(itg_f.voxel_size),
                        "sign_threshold_voxels": float(getattr(builder.config, "attached_sign_threshold_voxels", 1.5))},
           "load_before": load0, "identities": {}}
    qm = ctx["query_model"]
    set_finger_joints_from_scene(scene, qm)
    q = np.asarray(frames[args.cameras[0]].robot_state, np.float64)
    c, r = qm.sphere_centers_numeric(q)
    names = np.asarray([str(n) for n in qm.sphere_link_names])
    auth = np.isin(names, list(DEFAULT_CONTACT_LINKS[arm]))
    ca, ra = c[auth], r[auth]
    fin = np.char.find(names[auth].astype(str), "finger") >= 0
    r_finger = float(np.median(ra[fin])) if fin.any() else float(np.median(ra))
    rec["finger_sphere_radius_m"] = {"median": r_finger, "min": float(ra[fin].min()) if fin.any() else None,
                                     "max": float(ra[fin].max()) if fin.any() else None, "n": int(fin.sum())}
    gc = (ca[:, None, :] + GRID[None]).reshape(-1, 3)
    gr = np.repeat(ra, len(GRID))
    fine_vs = float(ag3s.config.esdf.fine_voxel_size)
    main = cs.esdf
    npz = {"auth_centres": ca.astype(np.float32), "auth_radii": ra.astype(np.float32), "auth_links": names[auth]}
    for mode in IDENTITIES:
        geom, idm, id_rec = H.choose_identity(ag3s, debug, gt_centre, mode, support_mask=support_mask, label_of=label_of)
        ir = {"identity_mode_resolved": idm, "identity": id_rec}
        rec["identities"][mode] = ir
        if geom is None:
            ir["skipped"] = "no target geometry"; continue
        ball = ag3s._target_ball(geom.points, surfaces, voxel=fine_vs)
        if ball is None:
            ir["skipped"] = "no ball"; continue
        nontarget = ag3s._nontarget_points(debug.get("filtered_cloud"), support_mask, geom)
        ntp = nontarget["points"]
        ir["ball"] = ball.summary()
        ir["nontarget"] = {k: v for k, v in nontarget.items() if k != "points"}
        owned = {f"bo_d{int(d)}": OwnedEvidence(target_points=geom.points, nontarget_points=ntp, tie_voxels=1.0,
                                                max_dist_voxels=d, unowned="remove") for d in DISTS}
        excl = {"oracle": H.make_oracle(builder, ball, gt_pts, gt_lab, 0.020, hulls=hulls),
                "oracle_prime": M.make_oracle_prime(builder, ball, gt_pts, gt_lab, 0.020, hulls)}
        build_tier(builder, ball)                       # warm-up
        build_tier(builder, ball, owned=owned["bo_d2"])
        tiers, ms = {}, {}
        for m in ("ball",) + BO:
            tiers[m], ms[m] = build_tier(builder, ball, owned=owned.get(m))
        for m in ("oracle", "oracle_prime"):
            tiers[m], _ = build_tier(builder, ball, exclusion=excl[m])
        tiers["none"], _ = build_tier(builder, ball, no_ball=True)
        tiers["ball_noforce"], _ = build_tier(builder, ball, noforce=True)
        tiers["bo_d2_noforce"], _ = build_tier(builder, ball, owned=owned["bo_d2"], noforce=True)
        # ---- seed metrics (as V-E2)
        seeds, org, vs = H.roi_seeds(builder, ball, 0.020)
        labels = H.voxel_labels(gt_pts, gt_lab, org, vs, seeds, hulls=hulls)
        centres = org + seeds * vs
        in_ball = (np.linalg.norm(centres - ball.centre, axis=1) <= ball.radius) & (centres[:, 2] >= ball.z_min)
        key = lambda a: (np.asarray(a, np.int64)[:, 0] << 42) | (np.asarray(a, np.int64)[:, 1] << 21) | np.asarray(a, np.int64)[:, 2]  # noqa: E731
        sk = key(seeds) if len(seeds) else np.zeros(0, np.int64)
        L = {n: labels == H.LABELS.index(n) for n in H.LABELS}
        met = {}
        for m in METHODS:
            tier = tiers[m]
            tr = tier.extra.get("exclusion_trace") or {}
            rem = tr.get("removed_ijk", np.zeros((0, 3), np.int64))
            removed = np.isin(sk, key(rem)) if len(rem) else np.zeros(len(seeds), bool)
            npz[f"{mode}__processed_{m}"] = removed.astype(np.int8)
            ex = tier.extra.get("exclusion") or {}
            dis_ball = L["distractor"] & in_ball
            met[m] = {
                "n_target_seeds_roi": int(L["target"].sum()),
                "target_recall": round(float((removed & L["target"]).sum()) / L["target"].sum(), 4) if L["target"].any() else None,
                "distractor_seeds_roi": int(L["distractor"].sum()), "distractor_seeds_in_ball": int(dis_ball.sum()),
                "distractor_seeds_lost": int((removed & L["distractor"]).sum()),
                "distractor_seeds_lost_in_ball": int((removed & dis_ball).sum()),
                "obstacle_retention_roi": round(float((~removed & ~L["target"]).sum()) / max(1, (~L["target"]).sum()), 4),
                "unlabelled_seeds_roi": int(L["unlabelled"].sum()), "unlabelled_removed": int((removed & L["unlabelled"]).sum()),
                "support_seeds_lost": int((removed & L["support"]).sum()),
                "destination_seeds_lost": int((removed & L["destination"]).sum()),
                "n_removed_total": int(ex.get("n_removed", 0)), "n_seeds_in_ball": int(ex.get("n_seeds_in_ball", 0)),
                "n_ambiguous": int(ex.get("n_ambiguous", 0)), "n_kept_nontarget": int(ex.get("n_kept_nontarget", 0)),
                "n_target_owned": int(ex.get("n_target_owned", 0)),
                "n_unowned_removed": int(ex.get("n_unowned_removed", 0)),
                "n_sign_forced": int(tier.extra.get("n_sign_forced", 0)),
                "classify_ms": ex.get("classify_ms"), "build_ms_once": round(ms.get(m, float("nan")), 3),
            }
        ir["metrics"] = met
        ir["n_roi_seeds"] = int(len(seeds))
        ir["roi_label_counts"] = {n: int(v.sum()) for n, v in L.items() if v.any()}
        # ---- unowned analysis (per d)
        cand, seedm, org2, vs2 = reseed_cand(builder, ball)
        un = {}
        if cand is not None:
            cc = org2 + cand * vs2
            clab = H.voxel_labels(gt_pts, gt_lab, org2, vs2, cand, hulls=hulls)
            d_raw = nearest(raw_nonrobot, cc)
            d_pc = {k: nearest(v, cc) for k, v in per_cam.items()}
            d_pcmin = np.min(np.stack(list(d_pc.values())), axis=0) if d_pc else np.full(len(cc), np.inf)
            d_fused = nearest(fused, cc)
            d_ev = nearest(ntp, cc)
            d_mesh_dis = nearest(mesh_pts[mesh_lab == H.LABELS.index("distractor")], cc)
            in_dhull = np.zeros(len(cc), bool)
            for hull, cls in hulls:
                if cls == H.LABELS.index("distractor"):
                    in_dhull |= hull.find_simplex(cc) >= 0
            for d in DISTS:
                m = f"bo_d{int(d)}"
                sp = _ball_owned_split(cc, seedm, ball, owned[m], vs2)
                thr = d * vs2
                u = {"n_removed_split": int(sp["removed"].sum()), "n_removed_tier": met[m]["n_removed_total"],
                     "unowned_by_label": {n: int((sp["unowned"] & (clab == H.LABELS.index(n))).sum()) for n in H.LABELS
                                          if (sp["unowned"] & (clab == H.LABELS.index(n))).any()},
                     "target_owned_by_label": {n: int((sp["target"] & (clab == H.LABELS.index(n))).sum()) for n in H.LABELS
                                               if (sp["target"] & (clab == H.LABELS.index(n))).any()}}
                sel = sp["unowned"] & (clab == H.LABELS.index("distractor"))
                u["n_unowned_distractor"] = int(sel.sum())
                if sel.any():
                    w = {"raw_depth": d_raw[sel], "per_camera_min": d_pcmin[sel], "fused": d_fused[sel],
                         "nontarget_evidence": d_ev[sel], "d_t": sp["d_t"][sel], "distractor_mesh": d_mesh_dis[sel]}
                    cause = np.where(d_raw[sel] > thr, "raw_far",
                             np.where(d_pcmin[sel] > thr, "percam_far_raw_near",
                              np.where(d_fused[sel] > thr, "fused_far_percam_near",
                               np.where(d_ev[sel] > thr, "evidence_far_fused_near", "evidence_near?"))))
                    u["cause_counts"] = {k: int((cause == k).sum()) for k in np.unique(cause)}
                    u["inside_distractor_hull"] = int(in_dhull[sel].sum())
                    u["dist_mm"] = {k: {"median": round(float(np.median(v)) * 1e3, 2), "p10": round(float(np.percentile(v, 10)) * 1e3, 2),
                                        "p90": round(float(np.percentile(v, 90)) * 1e3, 2)} for k, v in w.items()}
                    for k, v in w.items():
                        npz[f"{mode}__un{m}__{k}"] = v.astype(np.float32)
                    npz[f"{mode}__un{m}__per_camera"] = np.stack([d_pc[k][sel] for k in sorted(d_pc)]).astype(np.float32)
                    npz[f"{mode}__un{m}__inhull"] = in_dhull[sel]
                    npz[f"{mode}__un{m}__ijk"] = cand[sel].astype(np.int16)
                un[m] = u
            npz[f"{mode}__per_camera_names"] = np.asarray(sorted(d_pc))
        ir["unowned"] = un
        # ---- queries (as V-E2)
        layers = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in tiers.items()}
        qrec = {}
        for qname, cq, rq in (("q", ca, ra), ("grid", gc, gr)):
            blk = H.query_block(main, {m: layers[m] for m in METHODS}, cq, rq, MARGIN)
            dd = {m: np.asarray(blk["methods"][m]["d_free_mm"]) / 1000 for m in METHODS}
            roi = (np.linalg.norm(cq - ball.centre, axis=1) - rq) <= float(ball.radius) + 0.020
            npz[f"{mode}__{qname}_roi"] = roi
            npz[f"{mode}__{qname}_dmain"] = (np.asarray(blk["d_main_mm"]) / 1000).astype(np.float32)
            for m in METHODS:
                npz[f"{mode}__{qname}_d_{m}"] = dd[m].astype(np.float32)
            sub = {}
            for m in METHODS:
                if m == "oracle":
                    continue
                for sname, sel in (("all", np.ones(len(cq), bool)), ("roi", roi)):
                    sub[f"{m}|vs_oracle|{sname}"] = M.qstats(dd[m], dd["oracle"], dd[m] - rq, dd["oracle"] - rq, sel)
            qrec[qname] = sub
        ir["queries"] = qrec
        # ---- sign diagnosis
        tf = tiers["none"]
        torg, tvs = np.asarray(tf.origin, np.float64), float(tf.voxel_size)
        shp = np.array(tf.values.shape)
        R = float(ball.radius) + 0.06
        lo = np.maximum(np.floor((ball.centre - R - torg) / tvs).astype(int), 0)
        hi = np.minimum(np.ceil((ball.centre + R - torg) / tvs).astype(int) + 1, shp)
        sgn = {"n_hulls": 0}
        if np.all(hi > lo):
            ii = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
            cen = torg + ii * tvs
            hull_id = np.full(len(ii), -1, np.int16)
            for k_, (hull, cls) in enumerate(hulls):
                if cls != H.LABELS.index("distractor"):
                    continue
                inside = (hull.find_simplex(cen) >= 0) & (hull_id < 0)
                hull_id[inside] = k_
            # (e) none tier vs main fine layer over the WHOLE box
            fine_layer = main.layers[-1]
            box_none = tier_vals(tf, ii)
            box_main = np.asarray(fine_layer.distance(cen), np.float64)
            fcov = CuroboEsdfField._covers(fine_layer.grid, cen)
            sgn["e_none_vs_main_fine_box"] = {
                "n": int(fcov.sum()), "max_abs_diff_mm": round(float(np.max(np.abs(box_none - box_main)[fcov])) * 1e3, 4) if fcov.any() else None,
                "n_sign_diff": int(((box_none < 0) != (box_main < 0))[fcov].sum()),
                "same_origin": bool(np.allclose(np.asarray(fine_layer.grid.origin), torg)),
                "same_shape": [int(v) for v in fine_layer.grid.shape] == [int(v) for v in shp]}
            sel = hull_id >= 0
            sgn["n_hulls"] = int(len(np.unique(hull_id[sel])))
            if sel.any():
                P, IJ = cen[sel], ii[sel]
                dep = nearest(mesh_pts[mesh_lab == H.LABELS.index("distractor")], P)
                vals = {m: tier_vals(tiers[m], IJ) for m in METHODS + DIAG}
                vals["main_fine_layer"] = np.asarray(fine_layer.distance(P), np.float64)
                vals["coarse_layer"] = np.asarray(main.layers[0].distance(P), np.float64)
                vals["main_composite"] = np.asarray(main.distance(P), np.float64)
                comp, ans = {}, {}
                for m in METHODS + DIAG:
                    f = CuroboEsdfField(tuple(main.layers), outside_distance=main.outside_distance,
                                        target_free_layers=(layers[m],), stats={})
                    f.static_shapes = tuple(getattr(main, "static_shapes", ()) or ())
                    comp[m] = np.asarray(f.target_free_distance(P), np.float64)
                    ans[m] = np.asarray(f.answer_tier(P, target_free=True))
                fs, fsdf, fw = tsdf_lookup(itg_f, P)
                cs_, csdf, cw = tsdf_lookup(itg_c, P)
                inb = (np.linalg.norm(P - ball.centre, axis=1) <= ball.radius) & (P[:, 2] >= ball.z_min)
                bnd = CuroboEsdfField._boundary_distance(layers["none"].grid, P)
                for k_, v in vals.items():
                    npz[f"{mode}__sg_val_{k_}"] = v.astype(np.float32)
                for k_, v in comp.items():
                    npz[f"{mode}__sg_comp_{k_}"] = v.astype(np.float32)
                    npz[f"{mode}__sg_ans_{k_}"] = v_ans = ans[k_]
                npz.update({f"{mode}__sg_ijk": IJ.astype(np.int16), f"{mode}__sg_hull": hull_id[sel],
                            f"{mode}__sg_depth": dep.astype(np.float32), f"{mode}__sg_inball": inb,
                            f"{mode}__sg_fine_state": fs, f"{mode}__sg_fine_sdf": fsdf.astype(np.float32),
                            f"{mode}__sg_fine_w": fw.astype(np.float32),
                            f"{mode}__sg_coarse_state": cs_, f"{mode}__sg_coarse_sdf": csdf.astype(np.float32),
                            f"{mode}__sg_bound": bnd.astype(np.float32), f"{mode}__sg_origin": torg,
                            f"{mode}__sg_vs": np.float64(tvs), f"{mode}__sg_ball": np.r_[ball.centre, ball.radius, ball.z_min]})
                fneg = lambda x: round(float((x < 0).mean()), 4)  # noqa: E731
                sgn["n_points"] = int(sel.sum())
                sgn["frac_negative"] = {k_: fneg(v) for k_, v in vals.items()}
                sgn["frac_negative_composed_target_free"] = {k_: fneg(v) for k_, v in comp.items()}
                pred_neg = (fs == 2) & (fsdf < 0)
                sgn["fine_tsdf_state_counts"] = {"unallocated": int((fs == 0).sum()), "low_weight": int((fs == 1).sum()),
                                                 "observed_neg": int(((fs == 2) & (fsdf < 0)).sum()),
                                                 "observed_pos": int(((fs == 2) & (fsdf >= 0)).sum())}
                sgn["coarse_tsdf_state_counts"] = {"unallocated": int((cs_ == 0).sum()), "low_weight": int((cs_ == 1).sum()),
                                                   "observed_neg": int(((cs_ == 2) & (csdf < 0)).sum()),
                                                   "observed_pos": int(((cs_ == 2) & (csdf >= 0)).sum())}
                sgn["none_sign_agrees_with_fine_tsdf_sign"] = round(float(((vals["none"] < 0) == pred_neg).mean()), 4)
                sgn["main_fine_sign_agrees_with_fine_tsdf_sign"] = round(float(((vals["main_fine_layer"] < 0) == pred_neg).mean()), 4)
                for m in ("ball", "bo_d2"):
                    nf = vals[f"{m}_noforce"]
                    v = vals[m]
                    sgn[f"{m}_flips"] = {"neg_in_none": int((vals["none"] < 0).sum()),
                                         "neg_after_removal_before_force": int((nf < 0).sum()),
                                         "neg_after_force": int((v < 0).sum()),
                                         "flipped_by_removal": int(((vals["none"] < 0) & (nf >= 0)).sum()),
                                         "flipped_by_force": int(((nf < 0) & (v >= 0)).sum()),
                                         "flipped_by_force_in_ball": int(((nf < 0) & (v >= 0) & inb).sum())}
                for m in ("ball", "bo_d2", "none"):
                    sgn[f"answer_tier_{m}"] = {k_: int((ans[m] == k_).sum()) for k_ in np.unique(ans[m])}
                # safety: authorized (finger) sphere centre at interior points
                saf = {"r_finger_m": r_finger}
                for m in ("main_composite",) + tuple(METHODS) + ("none",):
                    dv = vals[m] if m == "main_composite" else comp[m]
                    saf[m] = {"n_positive": int((dv > 0).sum()),
                              "n_clear_ge_margin": int((dv - r_finger >= MARGIN).sum()),
                              "n_positive_depth_ge_5mm": int(((dv > 0) & (dep >= 0.005)).sum()),
                              "n_positive_depth_ge_10mm": int(((dv > 0) & (dep >= 0.010)).sum())}
                saf["n_depth_ge_5mm"] = int((dep >= 0.005).sum())
                saf["n_depth_ge_10mm"] = int((dep >= 0.010).sum())
                sgn["safety"] = saf
        ir["sign"] = sgn
        npz.update({f"{mode}__seed_ijk": seeds.astype(np.int16), f"{mode}__seed_label": labels,
                    f"{mode}__in_ball": in_ball, f"{mode}__origin": org, f"{mode}__voxel_size": np.float64(vs),
                    f"{mode}__ball": np.r_[ball.centre, ball.radius, ball.z_min],
                    f"{mode}__target_points": geom.points.astype(np.float32)})
        del tiers, layers
        torch.cuda.empty_cache()
    rec["load_after"] = M.pod_load()
    npz["gt_centre"] = gt_centre
    npz["mesh_pts"] = mesh_pts.astype(np.float32); npz["mesh_lab"] = mesh_lab
    return rec, npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posed-frames"); ap.add_argument("--lb-frames")
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--history", type=int, default=0); ap.add_argument("--only", default=None)
    args = ap.parse_args()
    args.phase = "approach"
    import torch
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.trajopt import wire
    from AG3S.trajopt.serve_safe import build_ag3s, constraint_link_filter
    assert torch.cuda.is_available()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    items = M.build_items(args)
    k, n = (int(v) for v in args.shard.split("/"))
    items = items[k::n]
    if args.only:
        items = [it for it in items if it["key"] in args.only.split(",")]
    scene_cache = {}
    for item in items:
        fo = out / f"{item['key']}.json"
        if fo.exists():
            print("skip (exists)", item["key"], flush=True); continue
        xml = str(resolve_asset(item["model_xml"] or "robots/rby1/description/models/rby1a/mujoco/model_transport.xml",
                                what="transport model"))
        if xml not in scene_cache:
            for s in scene_cache.values():
                s[0].close()
            scene_cache.clear()
            sc = TransportScene(xml, settle_steps=0)
            qm = build_constraint_robot_model(sc, link_filter=constraint_link_filter("gripper"),
                                              sphere_options={"capsule_radius_scale": 0.05, "sphere_spacing": 0.4,
                                                              "max_spheres_per_capsule": 32},
                                              finger_cover={"hands": "both"})
            scene_cache[xml] = (sc, qm)
        scene, qm = scene_cache[xml]
        args.cameras = tuple(item.get("cameras") or wire.DEFAULT_CAMERAS)
        ag3s = build_ag3s(xml, voxel=0.020, range_max=2.0, links="gripper", backend="curobo", fine_voxel=0.005,
                          tsdf_voxel=0.005, target_field_policy="exclude-authorized",
                          target_exclusion={"owned_tie_voxels": 1.0, "owned_max_dist_voxels": 2.0})
        ctx = {"scene": scene, "ag3s": ag3s, "target": item["target"], "classes": H.label_classes(scene, item["target"]),
               "query_model": qm}
        for h in item["history"]:
            H._history_step(ctx, h, item, args)
        t0 = time.time()
        rec, npz = run_one(ctx, item, args)
        rec["wall_s"] = round(time.time() - t0, 2)
        rec["history"] = len(item["history"])
        np.savez_compressed(out / f"{item['key']}.npz", **npz)
        fo.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=H._json_default))
        s = " | ".join(f"{m}: " + " ".join(f"{x}:{rec['identities'][m]['metrics'][x]['target_recall']}/{rec['identities'][m]['metrics'][x]['distractor_seeds_lost']}"
                                         for x in ("ball",) + BO)
                       + f" neg none/mainfine/comp {rec['identities'][m]['sign'].get('frac_negative', {}).get('none')}/"
                         f"{rec['identities'][m]['sign'].get('frac_negative', {}).get('main_fine_layer')}/"
                         f"{rec['identities'][m]['sign'].get('frac_negative', {}).get('main_composite')}"
                       for m in IDENTITIES if "metrics" in rec["identities"][m])
        print(item["key"], rec["wall_s"], s, flush=True)
        del ag3s, ctx
        torch.cuda.empty_cache()
    for s in scene_cache.values():
        s[0].close()


if __name__ == "__main__":
    main()
