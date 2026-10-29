"""T44 V-E2 verifier measurement (verifier-owned; the harness `AG3S/ag3s/experiments/live/exclusion_e2.py`
is imported read-only, nothing in benchmark/ is edited).

Per frame: ONE AG3S pass (reconstruction -> self-filter -> support -> grounding -> main ESDF), then for each identity
mode in (gt, gt_label) the same integrator builds the target-free fine tier with
  ball | ball_owned_remove | ball_owned_keep | oracle | oracle_prime
(oracle = harness make_oracle: GT target seeds in ROI removed; oracle_prime = oracle that also removes seeds labelled
`unlabelled` in ROI). Timing: 3 repeats per method, interleaved (rep-major) to spread load drift.

Fixed before measurement (2026-10-10):
  * query model = V11AB constraint model: links gripper, sphere options capsule_radius_scale 0.05 / sphere_spacing 0.4 /
    max_spheres_per_capsule 32, finger cover both (T43 FC). Authorized = DEFAULT_CONTACT_LINKS[used arm].
  * margin = 0.010 m (V11AB --obstacle-margin 0.010, --esdf-margin 0); clearance = d - r.
  * queries: (q) every authorized sphere at the frame's q; (grid) every authorized sphere centre + offsets
    {-20,-10,0,10,20} mm per axis (125 per sphere). Subset `roi`: |c - ball centre| - r <= ball radius + 20 mm.
  * falsely feasible: clearance_oracle < margin and clearance_method >= margin.
  * merge record: the grounding cluster holding the most GT-target cloud points: its target / distractor fraction and
    the share of GT-target cloud points it holds; plus the pipeline geometry's fractions.
  * sign check: values of the tier at voxel centres inside each distractor's convex hull (fraction < 0), and |grad d|
    (central differences) at voxels 10-20 mm from the distractor mesh and outside every fruit hull, d > 0.
usage: e2_measure.py --posed-frames F.json | --lb-frames e2_frames.json  --out DIR [--shard i/n] [--history H]
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, dataclasses, json, os, pathlib, subprocess, sys, time
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

IDENTITIES = ("gt", "gt_label")
METHODS = ("ball", "ball_owned_remove", "ball_owned_keep", "oracle", "oracle_prime")
TIMED = ("ball", "ball_owned_remove", "ball_owned_keep")
MARGIN = 0.010
GRID = np.array([(a, b, c) for a in (-.02, -.01, 0, .01, .02) for b in (-.02, -.01, 0, .01, .02)
                 for c in (-.02, -.01, 0, .01, .02)], np.float64)


def pod_load():
    out = {"t": time.time(), "loadavg": open("/proc/loadavg").read().split()[:3]}
    st = dict(l.split() for l in open("/sys/fs/cgroup/cpu.stat"))
    out.update({k: int(st[k]) for k in ("nr_periods", "nr_throttled", "throttled_usec", "usage_usec")})
    mem = int(open("/sys/fs/cgroup/memory.current").read())
    inact = next(int(l.split()[1]) for l in open("/sys/fs/cgroup/memory.stat") if l.startswith("inactive_file "))
    out["wset_gib"] = round((mem - inact) / 2**30, 2)
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        out["gpu_util_pct"], out["gpu_mem_used_mib"] = int(s[0]), int(s[1])
    except Exception as e:  # noqa: BLE001
        out["gpu_err"] = str(e)
    return out


def make_oracle_prime(builder, ball, gt_pts, gt_lab, shell, hulls):
    """harness make_oracle with removed = seed & label in {target, unlabelled}."""
    from AG3S.ag3s.fields.curobo_builder import _nearest
    t_idx, u_idx = H.LABELS.index("target"), H.LABELS.index("unlabelled")

    def oracle(itg, _ball, origin, vs, shape, *, owned=None):
        big = dataclasses.replace(ball, radius=float(ball.radius) + shell, z_min=-np.inf)
        cand = builder._ball_candidates(big, origin, vs, shape)
        empty = np.zeros((0, 3), np.int64)
        counts = {"exclusion_method": "oracle_prime", "n_seeds_in_ball": 0, "n_removed": 0}
        trace = {"removed_ijk": empty}
        if cand is None:
            return 0, None, {"counts": counts, "trace": trace}
        seed = builder._seed_mask(itg, cand)
        lab = H.voxel_labels(gt_pts, gt_lab, origin, vs, cand, hulls=hulls)
        rm = seed & ((lab == t_idx) | (lab == u_idx))
        removed = cand[rm]
        n = 0
        if len(removed):
            n, _ = builder._clear_seeds(itg, removed, shape)
        centres = origin + cand * vs
        tp = gt_pts[gt_lab == t_idx]
        op = gt_pts[(gt_lab != t_idx) & (gt_lab != H.LABELS.index("robot"))]
        near = np.linalg.norm(op - ball.centre, axis=1) <= float(ball.radius) * 3 + shell
        d_t, d_o = _nearest(tp, centres), _nearest(op[near], centres)
        sign = rm | (~seed & (d_t < d_o - vs))
        counts.update({"n_seeds_in_ball": int(seed.sum()), "n_removed": int(n)})
        trace["removed_ijk"] = removed
        si = cand[sign]
        return n, (si if len(si) else None), {"counts": counts, "trace": trace}
    return oracle


def merge_record(debug, label_of, support_mask, ident_geom):
    t_idx, d_idx = H.LABELS.index("target"), H.LABELS.index("distractor")
    pts = np.asarray(debug["filtered_cloud"].points, np.float64)
    lab = label_of(pts)
    gt_t = lab == t_idx
    if support_mask is not None and len(support_mask) == len(pts):
        gt_t &= ~np.asarray(support_mask, bool)
    n_t = int(gt_t.sum())
    clusters = list(getattr(debug.get("grounding"), "clusters", ()) or ())
    best = None
    for k, c in enumerate(clusters):
        idx = np.asarray(c.point_indices, np.int64)
        if not len(idx):
            continue
        nt = int(gt_t[idx].sum())
        if best is None or nt > best["n_gt_target_points"]:
            l = lab[idx]
            best = {"cluster": k, "n_points": int(len(idx)), "n_gt_target_points": nt,
                    "target_fraction": round(float((l == t_idx).mean()), 4),
                    "distractor_fraction": round(float((l == d_idx).mean()), 4),
                    "share_of_gt_target": round(nt / n_t, 4) if n_t else None,
                    "distractor_names_present": None}
    rec = {"n_clusters": len(clusters), "n_gt_target_cloud_points": n_t, "target_cluster": best}
    if ident_geom is not None:
        l = label_of(ident_geom.points)
        rec["chosen_geometry"] = {"n_points": int(len(l)), "target_fraction": round(float((l == t_idx).mean()), 4),
                                  "distractor_fraction": round(float((l == d_idx).mean()), 4)}
    return rec


def sign_check(tier, mesh_pts, mesh_lab, hulls, ball, main=None):
    from scipy.spatial import cKDTree
    v = tier.values.detach().float().cpu().numpy()
    org, vs = np.asarray(tier.origin, np.float64), float(tier.voxel_size)
    shp = np.array(v.shape)
    d_idx = H.LABELS.index("distractor")
    out = {"distractor_interior": [], "eikonal": None}
    # restrict to ROI-ish box: ball centre +- (radius + 60 mm)
    R = float(ball.radius) + 0.06
    lo = np.maximum(np.floor((ball.centre - R - org) / vs).astype(int), 0)
    hi = np.minimum(np.ceil((ball.centre + R - org) / vs).astype(int) + 1, shp)
    if np.any(hi <= lo):
        return out
    ii = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
    cen = org + ii * vs
    val = v[ii[:, 0], ii[:, 1], ii[:, 2]]
    in_any = np.zeros(len(ii), bool)
    for hull, cls in hulls:
        inside = hull.find_simplex(cen) >= 0
        in_any |= inside
        if cls == d_idx and inside.any():
            x = val[inside]
            out["distractor_interior"].append({"n": int(inside.sum()), "n_negative": int((x < 0).sum()),
                                               "frac_negative": round(float((x < 0).mean()), 4),
                                               "median_mm": round(float(np.median(x)) * 1000, 3),
                                               "max_mm": round(float(x.max()) * 1000, 3)})
            if main is not None:
                xm = np.asarray(main.distance(cen[inside]), np.float64)
                out["distractor_interior"][-1]["main_layer"] = {
                    "n_negative": int((xm < 0).sum()), "frac_negative": round(float((xm < 0).mean()), 4),
                    "median_mm": round(float(np.median(xm)) * 1000, 3)}
    dm = mesh_pts[mesh_lab == d_idx]
    if len(dm):
        dd, _ = cKDTree(dm).query(cen)
        band = (dd >= 2 * vs) & (dd <= 4 * vs) & ~in_any & (val > 0)
        ok = band & np.all(ii >= 1, 1) & np.all(ii < shp - 1, 1)
        if ok.any():
            j = ii[ok]
            g = np.zeros((len(j), 3))
            for a in range(3):
                e = np.zeros(3, int); e[a] = 1
                p, m = j + e, j - e
                g[:, a] = (v[p[:, 0], p[:, 1], p[:, 2]] - v[m[:, 0], m[:, 1], m[:, 2]]) / (2 * vs)
            gn = np.linalg.norm(g, axis=1)
            out["eikonal"] = {"n": int(len(gn)), "median": round(float(np.median(gn)), 4),
                              "p05": round(float(np.percentile(gn, 5)), 4),
                              "p95": round(float(np.percentile(gn, 95)), 4),
                              "frac_in_0.8_1.2": round(float(((gn >= .8) & (gn <= 1.2)).mean()), 4)}
            if main is not None:
                pc = org + j * vs
                gm = np.zeros((len(pc), 3))
                for a in range(3):
                    e = np.zeros(3); e[a] = vs
                    gm[:, a] = (np.asarray(main.distance(pc + e)) - np.asarray(main.distance(pc - e))) / (2 * vs)
                gmn = np.linalg.norm(gm, axis=1)
                out["eikonal_main_layer"] = {"n": int(len(gmn)), "median": round(float(np.median(gmn)), 4),
                                             "frac_in_0.8_1.2": round(float(((gmn >= .8) & (gmn <= 1.2)).mean()), 4)}
    return out


def qstats(d, ref, cl, cl_ref, sel):
    if not sel.any():
        return {"n": 0}
    over = (d - ref)[sel] * 1000
    return {"n": int(sel.sum()), "over_median_mm": round(float(np.median(over)), 3),
            "over_p95_mm": round(float(np.percentile(over, 95)), 3), "over_max_mm": round(float(over.max()), 3),
            "n_over_gt_1mm": int((over > 1).sum()),
            "n_falsely_feasible": int(((cl >= MARGIN) & (cl_ref < MARGIN))[sel].sum()),
            "n_falsely_infeasible": int(((cl < MARGIN) & (cl_ref >= MARGIN))[sel].sum()),
            "n_ref_infeasible": int((cl_ref < MARGIN)[sel].sum())}


def run_one(ctx, item, args):
    import torch
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence
    from AG3S.ag3s.fields.device_field import layer_from_device
    from AG3S.trajopt import wire

    scene, ag3s, target = ctx["scene"], ctx["ag3s"], ctx["target"]
    arm = item["arm"]
    load0 = pod_load()
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
    rec = {"key": item["key"], "seq": int(item["seq"]), "source": item["meta"], "set": item["set"],
           "bin": item["bin"], "target": target, "arm": arm, "attention_cameras": sorted(attention),
           "ag3s_ms": round(ag3s_ms, 2), "status": cs.status.value, "object_pose_err_mm": round(pose_err * 1e3, 3),
           "nearest_distractor_3d_mm": min(((n, round(float(np.linalg.norm(H.body_in_base(scene, n) - gt_centre)) * 1e3, 1))
                                            for n in others), key=lambda x: x[1]),
           "load_before": load0, "identities": {}}
    # query model at q
    qm = ctx["query_model"]
    set_finger_joints_from_scene(scene, qm)
    q = np.asarray(frames[args.cameras[0]].robot_state, np.float64)
    c, r = qm.sphere_centers_numeric(q)
    names = np.asarray([str(n) for n in qm.sphere_link_names])
    auth = np.isin(names, list(DEFAULT_CONTACT_LINKS[arm]))
    ca, ra = c[auth], r[auth]
    gc = (ca[:, None, :] + GRID[None]).reshape(-1, 3)
    gr = np.repeat(ra, len(GRID))
    builder = ag3s._esdf_builder
    fine_vs = float(ag3s.config.esdf.fine_voxel_size)
    cfgE = ag3s.config.esdf
    npz = {"auth_centres": ca.astype(np.float32), "auth_radii": ra.astype(np.float32),
           "auth_links": names[auth]}
    for mode in IDENTITIES:
        geom, idm, id_rec = H.choose_identity(ag3s, debug, gt_centre, mode, support_mask=support_mask,
                                              label_of=label_of)
        ir = {"identity_mode_resolved": idm, "identity": id_rec,
              "merge": merge_record(debug, label_of, support_mask, geom)}
        rec["identities"][mode] = ir
        if geom is None:
            ir["skipped"] = "no target geometry"; continue
        ball = ag3s._target_ball(geom.points, surfaces, voxel=fine_vs)
        if ball is None:
            ir["skipped"] = "no ball"; continue
        tn = time.perf_counter()
        nontarget = ag3s._nontarget_points(debug.get("filtered_cloud"), support_mask, geom)
        ir["nontarget_points_ms"] = round((time.perf_counter() - tn) * 1e3, 3)
        ir["ball"] = ball.summary()
        ir["nontarget"] = {k: v for k, v in nontarget.items() if k != "points"}
        owned = {m: OwnedEvidence(target_points=geom.points, nontarget_points=nontarget["points"],
                                  tie_voxels=cfgE.owned_tie_voxels, max_dist_voxels=cfgE.owned_max_dist_voxels,
                                  unowned=("keep" if m.endswith("keep") else "remove"))
                 for m in ("ball_owned_remove", "ball_owned_keep")}
        excl = {"oracle": H.make_oracle(builder, ball, gt_pts, gt_lab, 0.020, hulls=hulls),
                "oracle_prime": make_oracle_prime(builder, ball, gt_pts, gt_lab, 0.020, hulls)}
        H.build_free_tier(builder, ball)                      # warm-up (JIT)
        H.build_free_tier(builder, ball, owned=owned["ball_owned_remove"])
        runs = {m: [] for m in TIMED}
        tiers = {}
        for rep in range(3):
            for m in TIMED:
                tier, t = H.build_free_tier(builder, ball, owned=owned.get(m))
                runs[m].append(round(t, 3))
                tiers[m] = tier
        for m in ("oracle", "oracle_prime"):
            tiers[m], _ = H.build_free_tier(builder, ball, exclusion=excl[m])
        seeds, org, vs = H.roi_seeds(builder, ball, 0.020)
        labels = H.voxel_labels(gt_pts, gt_lab, org, vs, seeds, hulls=hulls)
        centres = org + seeds * vs
        in_ball = (np.linalg.norm(centres - ball.centre, axis=1) <= ball.radius) & (centres[:, 2] >= ball.z_min)
        key = lambda a: (np.asarray(a, np.int64)[:, 0] << 42) | (np.asarray(a, np.int64)[:, 1] << 21) | np.asarray(a, np.int64)[:, 2]  # noqa: E731
        sk = key(seeds) if len(seeds) else np.zeros(0, np.int64)
        L = {n: labels == H.LABELS.index(n) for n in H.LABELS}
        met = {}
        for m, tier in tiers.items():
            tr = tier.extra.get("exclusion_trace") or {}
            rem = tr.get("removed_ijk", np.zeros((0, 3), np.int64))
            removed = np.isin(sk, key(rem)) if len(rem) else np.zeros(len(seeds), bool)
            npz[f"{mode}__processed_{m}"] = removed.astype(np.int8)
            ex = tier.extra.get("exclusion") or {}
            dis_ball = L["distractor"] & in_ball
            met[m] = {
                "n_target_seeds_roi": int(L["target"].sum()),
                "target_recall": round(float((removed & L["target"]).sum()) / L["target"].sum(), 4) if L["target"].any() else None,
                "distractor_seeds_roi": int(L["distractor"].sum()),
                "distractor_seeds_in_ball": int(dis_ball.sum()),
                "distractor_seeds_lost": int((removed & L["distractor"]).sum()),
                "distractor_seeds_lost_in_ball": int((removed & dis_ball).sum()),
                "obstacle_seeds_roi": int((~L["target"]).sum()),
                "obstacle_retention_roi": round(float((~removed & ~L["target"]).sum()) / max(1, (~L["target"]).sum()), 4),
                "nontarget_labelled_seeds_roi": int((~L["target"] & ~L["unlabelled"]).sum()),
                "nontarget_labelled_retention_roi": round(float((~removed & ~L["target"] & ~L["unlabelled"]).sum()) / max(1, (~L["target"] & ~L["unlabelled"]).sum()), 4),
                "unlabelled_seeds_roi": int(L["unlabelled"].sum()),
                "unlabelled_removed": int((removed & L["unlabelled"]).sum()),
                "shared_seeds_roi": int(L["shared"].sum()), "shared_removed": int((removed & L["shared"]).sum()),
                "support_seeds_lost": int((removed & L["support"]).sum()),
                "destination_seeds_lost": int((removed & L["destination"]).sum()),
                "n_removed_total": int(ex.get("n_removed", 0)),
                "n_removed_outside_roi_seed_list": int(len(rem) - int(removed.sum())),
                "n_seeds_in_ball": int(ex.get("n_seeds_in_ball", 0)),
                "n_ambiguous": int(ex.get("n_ambiguous", 0)), "n_kept_nontarget": int(ex.get("n_kept_nontarget", 0)),
                "n_unowned_removed": int(ex.get("n_unowned_removed", 0)), "n_unowned_kept": int(ex.get("n_unowned_kept", 0)),
                "n_sign_forced": int(tier.extra.get("n_sign_forced", 0)),
                "classify_ms": ex.get("classify_ms"),
                "build_ms_runs": runs.get(m),
            }
        ir["metrics"] = met
        ir["n_roi_seeds"] = int(len(seeds))
        ir["roi_label_counts"] = {n: int(v.sum()) for n, v in L.items() if v.any()}
        # queries
        layers = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in tiers.items()}
        main = cs.esdf
        qrec = {}
        for qname, cc, rr in (("q", ca, ra), ("grid", gc, gr)):
            blk = H.query_block(main, layers, cc, rr, MARGIN)
            d = {m: np.asarray(blk["methods"][m]["d_free_mm"]) / 1000 for m in METHODS}
            dmain = np.asarray(blk["d_main_mm"]) / 1000
            roi = (np.linalg.norm(cc - ball.centre, axis=1) - rr) <= float(ball.radius) + 0.020
            npz[f"{mode}__{qname}_roi"] = roi
            npz[f"{mode}__{qname}_dmain"] = dmain.astype(np.float32)
            for m in METHODS:
                npz[f"{mode}__{qname}_d_{m}"] = d[m].astype(np.float32)
            sub = {}
            for ref in ("oracle", "oracle_prime"):
                for m in METHODS:
                    if m == ref:
                        continue
                    for sname, sel in (("all", np.ones(len(cc), bool)), ("roi", roi)):
                        sub[f"{m}|vs_{ref}|{sname}"] = qstats(d[m], d[ref], d[m] - rr, d[ref] - rr, sel)
            qrec[qname] = sub
            qrec[qname + "_n"] = int(len(cc)); qrec[qname + "_n_roi"] = int(roi.sum())
        ir["queries"] = qrec
        if args.sign_check:
            ir["sign_check"] = {m: sign_check(tiers[m], mesh_pts, mesh_lab, hulls, ball, main if m == 'ball' else None) for m in METHODS}
        npz.update({f"{mode}__seed_ijk": seeds.astype(np.int16), f"{mode}__seed_label": labels,
                    f"{mode}__in_ball": in_ball, f"{mode}__origin": org, f"{mode}__voxel_size": np.float64(vs),
                    f"{mode}__ball": np.r_[ball.centre, ball.radius, ball.z_min],
                    f"{mode}__target_points": geom.points.astype(np.float32)})
        del tiers, layers
        torch.cuda.empty_cache()
    rec["load_after"] = pod_load()
    npz["gt_centre"] = gt_centre
    npz["mesh_pts"] = mesh_pts.astype(np.float32); npz["mesh_lab"] = mesh_lab
    return rec, npz


def build_items(args):
    items = []
    if args.posed_frames:
        for pf in json.loads(pathlib.Path(args.posed_frames).read_text()):
            ep, fr = int(pf["episode_index"]), int(pf["frame"])
            items.append({"key": f"demo_ep{ep:04d}_f{fr:03d}", "seq": fr, "arm": pf["used_arm"], "target": pf["target_fruit"],
                          "set": "demo", "bin": pf["bin"], "model_xml": None, "history": [],
                          "pose": (lambda sc, p=pf: H.pose_posed(sc, p)),
                          "meta": {"episode_index": ep, "frame": fr, "split": pf.get("split"),
                                   "nearest_other_fruit_xy_m": round(H.nearest_other_xy(pf), 5)}})
    else:
        sel = json.loads(pathlib.Path(args.lb_frames).read_text())["fallback_for_lead_decision"]["selected"]
        for e in sel:
            run = pathlib.Path(e["run_dir"])
            man = json.loads((run / "manifest.json").read_text())
            rows = H.load_planning_rows(run)
            by = {int(r["seq"]): i for i, r in enumerate(rows)}
            for f in e["frames"]:
                s = int(f["seq_in_E0_frames_jsonl"]); i = by[s]
                items.append({"key": f"{run.name}_s{s:03d}", "seq": s, "arm": man["scene"]["episode_used_arm"],
                              "target": man["scene"]["episode_target_fruit"], "set": "lb", "bin": "lb",
                              "model_xml": man["scene"]["model_xml"], "cameras": man.get("cameras"),
                              "pose": (lambda sc, r=rows[i]: H.pose(sc, r)),
                              "history": [(lambda sc, r=h: H.pose(sc, r)) for h in rows[max(0, i - args.history):i]],
                              "meta": {"run": str(run), "episode_index": e["episode"], "split": e["split"],
                                       "planning_index": f["planning_index"],
                                       "nearest_other_fruit_xy_m": f["nearest_other_xy_m"]}})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posed-frames"); ap.add_argument("--lb-frames")
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--history", type=int, default=0); ap.add_argument("--sign-check", action="store_true")
    ap.add_argument("--only", default=None)
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
    items = build_items(args)
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
        rec["query_model"] = {"n_spheres": int(len(qm.sphere_link_names)), "margin_m": MARGIN}
        np.savez_compressed(out / f"{item['key']}.npz", **npz)
        fo.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=H._json_default))
        s = " | ".join(f"{m}: " + " ".join(f"{x}:{rec['identities'][m]['metrics'][x]['target_recall']}/{rec['identities'][m]['metrics'][x]['distractor_seeds_lost']}"
                                         for x in ("ball", "ball_owned_remove", "ball_owned_keep"))
                       for m in IDENTITIES if "metrics" in rec["identities"][m])
        print(item["key"], rec["wall_s"], s, flush=True)
        del ag3s, ctx
        torch.cuda.empty_cache()
    for s in scene_cache.values():
        s[0].close()


if __name__ == "__main__":
    main()
