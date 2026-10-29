"""T44c D-sign2 verifier measurement (verifier-owned; benchmark/ imported read-only, nothing there is edited).

Re-uses outputs/verify/T44/E2/e2_measure.py and outputs/verify/T44/E2b/e2b_measure.py helpers unchanged.
Writes only under outputs/verify/T44/E2c/.

Per frame: ONE AG3S pass (same settings as T44E2b, config owned_max_dist_voxels 3.0), then per identity (gt, gt_label):
  fine target-free tiers   : ball | bo_d3 (ball_owned remove, tie 1.0, d=3) | oracle (GT target seeds only)
  coarse target-free tiers : ball_cf | bo_d3_cf (harness build_coarse_free_tier, non-target points for shared check)
                             ball_cfns | bo_d3_cfns (diagnostic: same, shared-voxel preservation disabled = no non-target pts)
                             oracle_cf (diagnostic: GT-target-labelled coarse seeds only removed, majority >=80 %,
                                        candidates = ball + 20 mm shell, sign set as harness make_oracle)
  composition : off = min(f, max(d,b))  (old)   on = with coarse layer (new, T44c)   via harness compose_free_field
Point sets:
  D  distractor interior  (T44E2b definition: fine-lattice centres inside GT distractor hull, ball centre +- (r+60mm))
  T  target interior      (same lattice, inside GT target hull)
  q  authorized spheres (DEFAULT_CONTACT_LINKS[arm], V11AB query model)   grid = q + 125 offsets (+-20 mm)
Gradient check at points where the on-composition answered `target_free_coarse` (D, T, grid): analytic
target_free_gradient vs central finite difference (h = 1 mm) of target_free_distance; probe answer tiers recorded.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, dataclasses, json, pathlib, sys, time
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2b'))
sys.path.insert(0, _ag3s_workspace_path(''))
import e2_measure as M  # noqa: E402
import e2b_measure as B  # noqa: E402
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

IDENTITIES = ("gt", "gt_label")
MARGIN = M.MARGIN
GRID = M.GRID
FD_H = 1e-3
TIER_CODES = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")
MAX_GRAD_PTS = 4000


def tcode(a):
    a = np.asarray(a).astype(str)
    out = np.full(len(a), -1, np.int8)
    for i, n in enumerate(TIER_CODES):
        out[a == n] = i
    return out


def make_oracle_coarse(builder, ball, gt_pts, gt_lab, hulls):
    from AG3S.ag3s.fields.curobo_builder import coarse_exclusion_params
    base = H.make_oracle(builder, ball, gt_pts, gt_lab, 0.020, hulls=hulls)

    def f(itg, ball_, origin, vs, shape, spec):
        n, sign_idx, ex = base(itg, ball_, origin, vs, shape)
        params = coarse_exclusion_params(ball_, vs, spec)
        ex["params"] = params
        ex["counts"]["params"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}
        ex["trace"]["shared_kept_ijk"] = np.zeros((0, 3), np.int64)
        return n, sign_idx, ex
    return f


def build_coarse(builder, ball, *, owned=None, nontarget=None, oracle=None):
    if oracle is not None:
        builder._coarse_ball_exclusion = oracle
    try:
        return H.build_coarse_free_tier(builder, ball, owned=owned, nontarget=nontarget)
    finally:
        if oracle is not None:
            del builder._coarse_ball_exclusion


def coarse_roi(builder, ball, shell=0.020):
    """coarse-lattice seeds within ball+shell (z unbounded), freshly re-seeded (read only)."""
    import torch
    itg = builder._mapper.integrator
    vs = float(builder.coarse_voxel)
    itg._esdf_voxel_size.copy_(torch.tensor([vs], device=itg._esdf_voxel_size.device, dtype=torch.float32))
    origin = itg._origin
    itg._last_esdf_origin.copy_(origin)
    itg._seed_esdf_impl(origin, itg._esdf_voxel_size)
    shape = tuple(int(v) for v in itg._site_index.shape)
    org = builder._origin_of(itg, shape, vs)
    big = dataclasses.replace(ball, radius=float(ball.radius) + shell, z_min=-np.inf)
    cand = builder._ball_candidates(big, org, vs, shape)
    if cand is None:
        return np.zeros((0, 3), np.int64), org, vs
    return cand[builder._seed_mask(itg, cand)], org, vs


def key3(a):
    a = np.asarray(a, np.int64).reshape(-1, 3)
    return (a[:, 0] << 42) | (a[:, 1] << 21) | a[:, 2]


def signed_mesh_dist(P, pts, hulls_of_cls):
    d = B.nearest(pts, P)
    inside = np.zeros(len(P), bool)
    for h in hulls_of_cls:
        inside |= h.find_simplex(P) >= 0
    return np.where(inside, -d, d), inside


def _layer_fd(layer, P):
    fd = np.zeros_like(P)
    for ax in range(3):
        e = np.zeros(3); e[ax] = FD_H
        fd[:, ax] = (np.asarray(layer.distance(P + e), np.float64) - np.asarray(layer.distance(P - e), np.float64)) / (2 * FD_H)
    return fd


def _cos(a, b):
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def grad_check(field, P, coarse_layer=None, main_coarse=None):
    """analytic gradient (target_free_gradient) vs central FD of target_free_distance; probe tiers.
    + branch identity: composed gradient == coarse target-free layer gradient (max abs diff)
    + reference: the same analytic-vs-FD measure on the coarse target-free layer itself and on the main coarse layer."""
    g, tier = field.target_free_gradient(P, return_tier=True)
    extra = {}
    if coarse_layer is not None:
        gl = np.asarray(coarse_layer.gradient(P), np.float64)
        extra["grad_vs_layer_absdiff"] = np.abs(g - gl).max(axis=1)
        extra["layer_cos"] = _cos(gl, _layer_fd(coarse_layer, P))
        extra["layer_value"] = np.asarray(coarse_layer.distance(P), np.float64)
    if main_coarse is not None:
        gm = np.asarray(main_coarse.gradient(P), np.float64)
        extra["maincoarse_cos"] = _cos(gm, _layer_fd(main_coarse, P))
        extra["maincoarse_norm_fd"] = np.linalg.norm(_layer_fd(main_coarse, P), axis=1)
    v_ans = field._answer(P, True, want_gradient=False)[0]
    v = field.target_free_distance(P)
    fd = np.zeros_like(P)
    same = np.ones(len(P), bool)
    for ax in range(3):
        e = np.zeros(3); e[ax] = FD_H
        vp, vm = field.target_free_distance(P + e), field.target_free_distance(P - e)
        fd[:, ax] = (vp - vm) / (2 * FD_H)
        same &= (field.answer_tier(P + e, target_free=True) == tier) & (field.answer_tier(P - e, target_free=True) == tier)
    ng, nf = np.linalg.norm(g, axis=1), np.linalg.norm(fd, axis=1)
    cos = np.sum(g * fd, axis=1) / np.maximum(ng * nf, 1e-12)
    return {"cos": cos, "norm_an": ng, "norm_fd": nf, "same_branch": same, "value": v,
            "value_answer_diff": np.abs(v_ans - v), "abs_err": np.linalg.norm(g - fd, axis=1), **extra}


def run_one(ctx, item, args):
    import torch
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence
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
    cs, debug = ag3s.process_multi_debug(obs, phase="approach", active_manipulators=[arm])
    support_mask = debug.get("support_mask")
    surfaces = list(cs.support_surfaces)
    gt_depth_pts, gt_depth_lab = H.gt_points(frames.values(), ctx["classes"])
    label_of = H.point_labeler(gt_depth_pts, gt_depth_lab, radius=0.003)
    mesh_pts, mesh_lab, hulls = H.gt_mesh_points(scene, ctx["classes"])
    gt_pts, gt_lab = np.vstack([gt_depth_pts, mesh_pts]), np.concatenate([gt_depth_lab, mesh_lab])
    LT, LD = H.LABELS.index("target"), H.LABELS.index("distractor")
    t_mesh, d_mesh = mesh_pts[mesh_lab == LT], mesh_pts[mesh_lab == LD]
    t_hulls = [h for h, c in hulls if c == LT]
    d_hulls = [h for h, c in hulls if c == LD]
    nontgt_gt = gt_pts[(gt_lab != LT) & (gt_lab != H.LABELS.index("robot"))]
    builder = ag3s._esdf_builder
    rec = {"key": item["key"], "seq": int(item["seq"]), "source": item["meta"], "set": item["set"], "bin": item["bin"],
           "target": target, "arm": arm, "object_pose_err_mm": round(pose_err * 1e3, 3), "status": cs.status.value,
           "coarse_voxel_m": float(builder.coarse_voxel), "fine_voxel_m": float(builder.fine_voxel),
           "coarse_truncation_m": float(builder.coarse_truncation), "fine_truncation_m": float(builder.fine_truncation),
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
    gc = (ca[:, None, :] + GRID[None]).reshape(-1, 3)
    gr = np.repeat(ra, len(GRID))
    fine_vs = float(ag3s.config.esdf.fine_voxel_size)
    main = cs.esdf
    npz = {"auth_centres": ca.astype(np.float32), "auth_radii": ra.astype(np.float32), "auth_links": names[auth],
           "r_finger": np.float64(r_finger)}
    # GT distances of the authorized spheres (centre-to-surface, signed by hull) and of grid points
    for qn, P in (("q", ca), ("grid", gc)):
        dt, _ = signed_mesh_dist(P, t_mesh, t_hulls)
        npz[f"{qn}_gt_target_dist"] = dt.astype(np.float32)
        npz[f"{qn}_gt_nontarget_dist"] = B.nearest(nontgt_gt, P).astype(np.float32)
    rng = np.random.default_rng(int(item["seq"]) * 1000 + len(item["key"]))
    for mode in IDENTITIES:
        geom, idm, id_rec = H.choose_identity(ag3s, debug, gt_centre, mode, support_mask=support_mask, label_of=label_of)
        ir = {"identity_mode_resolved": idm}
        rec["identities"][mode] = ir
        if geom is None:
            ir["skipped"] = "no target geometry"; continue
        ball = ag3s._target_ball(geom.points, surfaces, voxel=fine_vs)
        if ball is None:
            ir["skipped"] = "no ball"; continue
        nontarget = ag3s._nontarget_points(debug.get("filtered_cloud"), support_mask, geom)
        ntp = nontarget["points"]
        ir["ball"] = ball.summary()
        owned = OwnedEvidence(target_points=geom.points, nontarget_points=ntp, tie_voxels=1.0, max_dist_voxels=3.0,
                              unowned="remove")
        ofine = H.make_oracle(builder, ball, gt_pts, gt_lab, 0.020, hulls=hulls)
        ocoarse = make_oracle_coarse(builder, ball, gt_pts, gt_lab, hulls)
        # ---- fine target-free tiers (warm-up first: JIT)
        B.build_tier(builder, ball)
        B.build_tier(builder, ball, owned=owned)
        F, fms = {}, {}
        F["ball"], fms["ball"] = B.build_tier(builder, ball)
        F["bo_d3"], fms["bo_d3"] = B.build_tier(builder, ball, owned=owned)
        F["oracle"], _ = B.build_tier(builder, ball, exclusion=ofine)
        # ---- coarse target-free tiers
        build_coarse(builder, ball, nontarget=ntp)           # warm-up
        C, cms = {}, {}
        C["ball"], cms["ball"] = build_coarse(builder, ball, nontarget=ntp)
        C["bo_d3"], cms["bo_d3"] = build_coarse(builder, ball, owned=owned, nontarget=ntp)
        C["ball_ns"], _ = build_coarse(builder, ball, nontarget=None)
        C["bo_d3_ns"], _ = build_coarse(builder, ball, owned=owned, nontarget=None)
        C["oracle"], _ = build_coarse(builder, ball, nontarget=ntp, oracle=ocoarse)
        # ---- coarse seed labels (ROI = ball + 20 mm)
        cseeds, corg, cvs = coarse_roi(builder, ball)
        assert np.allclose(corg, np.asarray(C["ball"].origin, np.float64)), (corg, C["ball"].origin)
        clab = H.voxel_labels(gt_pts, gt_lab, corg, cvs, cseeds, hulls=hulls)
        ccent = corg + cseeds * cvs
        cin_ball = (np.linalg.norm(ccent - ball.centre, axis=1) <= ball.radius) & (ccent[:, 2] >= ball.z_min)
        sk = key3(cseeds) if len(cseeds) else np.zeros(0, np.int64)
        cmet = {}
        for m, t in C.items():
            ex = dict(t.extra.get("exclusion") or {})
            ex.pop("params", None)
            tr = t.extra.get("exclusion_trace") or {}
            rem = np.asarray(tr.get("removed_ijk", np.zeros((0, 3))), np.int64).reshape(-1, 3)
            shk = np.asarray(tr.get("shared_kept_ijk", np.zeros((0, 3))), np.int64).reshape(-1, 3)
            removed = np.isin(sk, key3(rem)) if len(rem) else np.zeros(len(cseeds), bool)
            shared = np.isin(sk, key3(shk)) if len(shk) else np.zeros(len(cseeds), bool)
            Lt, Ld = clab == LT, clab == LD
            shk_lab = H.voxel_labels(gt_pts, gt_lab, corg, cvs, shk, hulls=hulls) if len(shk) else np.zeros(0, np.int16)
            cmet[m] = {"counts": ex, "n_sign_forced": int(t.extra.get("n_sign_forced", 0)),
                       "sign_threshold_m": t.extra.get("threshold_m"),
                       "build_ms_once": round(cms.get(m, float("nan")), 3),
                       "n_target_seeds_roi": int(Lt.sum()),
                       "target_recall": round(float((removed & Lt).sum()) / Lt.sum(), 4) if Lt.any() else None,
                       "distractor_seeds_roi": int(Ld.sum()), "distractor_seeds_in_ball": int((Ld & cin_ball).sum()),
                       "distractor_seeds_lost": int((removed & Ld).sum()),
                       "support_seeds_lost": int((removed & (clab == H.LABELS.index("support"))).sum()),
                       "shared_label_seeds_lost": int((removed & (clab == H.LABELS.index("shared"))).sum()),
                       "n_removed_trace": int(len(rem)),
                       "shared_kept_label_counts": {n: int((shk_lab == i).sum()) for i, n in enumerate(H.LABELS)
                                                    if (shk_lab == i).any()}}
            npz[f"{mode}__c_shared_kept_{m}"] = shk.astype(np.int16)
            npz[f"{mode}__c_removed_{m}"] = rem.astype(np.int16)
        ir["coarse"] = cmet
        ir["fine_build_ms_once"] = {k: round(v, 3) for k, v in fms.items()}
        ir["fine_exclusion"] = {m: {k: v for k, v in (F[m].extra.get("exclusion") or {}).items() if k != "params"}
                                for m in F}
        npz.update({f"{mode}__c_seed_ijk": cseeds.astype(np.int16), f"{mode}__c_seed_label": clab,
                    f"{mode}__c_origin": corg, f"{mode}__c_vs": np.float64(cvs),
                    f"{mode}__ball": np.r_[ball.centre, ball.radius, ball.z_min],
                    f"{mode}__target_points": geom.points.astype(np.float32)})
        # ---- fields
        FL = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in F.items()}
        CL = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in C.items()}
        fields = {}
        for m in ("ball", "bo_d3", "oracle"):
            fields[f"{m}|off"] = H.compose_free_field(main, FL[m], None)
            fields[f"{m}|on"] = H.compose_free_field(main, FL[m], CL[m])
        for m in ("ball", "bo_d3"):
            fields[f"{m}|on_ns"] = H.compose_free_field(main, FL[m], CL[f"{m}_ns"])
        # ---- shared-kept voxel centres (coarse) per method for proximity
        shk_c = {m: corg + np.asarray(npz[f"{mode}__c_shared_kept_{m}"], np.float64) * cvs for m in ("ball", "bo_d3")}
        # ---- queries q / grid
        qrec = {}
        for qn, P, R in (("q", ca, ra), ("grid", gc, gr)):
            roi = (np.linalg.norm(P - ball.centre, axis=1) - R) <= float(ball.radius) + 0.020
            npz[f"{mode}__{qn}_roi"] = roi
            npz[f"{mode}__{qn}_dmain"] = np.asarray(main.distance(P), np.float32)
            for fk, f in fields.items():
                npz[f"{mode}__{qn}_d_{fk}"] = np.asarray(f.target_free_distance(P), np.float32)
                if fk.endswith("|on"):
                    npz[f"{mode}__{qn}_tier_{fk}"] = tcode(f.answer_tier(P, target_free=True))
            for m in ("ball", "bo_d3"):
                npz[f"{mode}__{qn}_dist_sharedkept_{m}"] = B.nearest(shk_c[m], P).astype(np.float32) if len(shk_c[m]) \
                    else np.full(len(P), np.inf, np.float32)
            qrec[qn] = {"n": int(len(P)), "n_roi": int(roi.sum())}
        ir["queries"] = qrec
        # ---- interior point sets on the fine target-free lattice (T44E2b definition)
        tf = F["ball"]
        torg, tvs = np.asarray(tf.origin, np.float64), float(tf.voxel_size)
        shp = np.array(tf.values.shape)
        Rb = float(ball.radius) + 0.06
        lo = np.maximum(np.floor((ball.centre - Rb - torg) / tvs).astype(int), 0)
        hi = np.minimum(np.ceil((ball.centre + Rb - torg) / tvs).astype(int) + 1, shp)
        sets = {}
        if np.all(hi > lo):
            ii = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
            cen = torg + ii * tvs
            ind = np.zeros(len(cen), bool)
            for h in d_hulls:
                ind |= h.find_simplex(cen) >= 0
            int_ = np.zeros(len(cen), bool)
            for h in t_hulls:
                int_ |= h.find_simplex(cen) >= 0
            sets["D"] = (cen[ind], B.nearest(d_mesh, cen[ind]))   # T44E2b definition (distractor hulls only)
            sets["T"] = (cen[int_], B.nearest(t_mesh, cen[int_]))
        srec = {}
        for sname, (P, dep) in sets.items():
            if not len(P):
                srec[sname] = {"n": 0}; continue
            npz[f"{mode}__{sname}_pts"] = P.astype(np.float32)
            npz[f"{mode}__{sname}_depth"] = dep.astype(np.float32)
            npz[f"{mode}__{sname}_dmain"] = np.asarray(main.distance(P), np.float32)
            npz[f"{mode}__{sname}_coarse_main"] = np.asarray(main.layers[0].distance(P), np.float32)
            inb = (np.linalg.norm(P - ball.centre, axis=1) <= ball.radius) & (P[:, 2] >= ball.z_min)
            npz[f"{mode}__{sname}_inball"] = inb
            for fk, f in fields.items():
                npz[f"{mode}__{sname}_d_{fk}"] = np.asarray(f.target_free_distance(P), np.float32)
                if fk.endswith("|on"):
                    npz[f"{mode}__{sname}_tier_{fk}"] = tcode(f.answer_tier(P, target_free=True))
            for m in ("ball", "bo_d3", "oracle"):
                npz[f"{mode}__{sname}_cfree_{m}"] = np.asarray(
                    fields[f"{m}|on"]._coarse_free_values(P), np.float32)
            srec[sname] = {"n": int(len(P))}
        ir["sets"] = srec
        # ---- gradient check where new tier answered
        gk = {}
        for m in ("ball", "bo_d3", "oracle"):
            f = fields[f"{m}|on"]
            pools = []
            for sname in ("D", "T"):
                if f"{mode}__{sname}_tier_{m}|on" in npz:
                    sel = npz[f"{mode}__{sname}_tier_{m}|on"] == TIER_CODES.index("target_free_coarse")
                    pools.append((sname, npz[f"{mode}__{sname}_pts"].astype(np.float64)[sel]))
            sel = npz[f"{mode}__grid_tier_{m}|on"] == TIER_CODES.index("target_free_coarse")
            pools.append(("grid", gc[sel]))
            allp = np.vstack([p for _, p in pools]) if pools else np.zeros((0, 3))
            src = np.concatenate([np.full(len(p), i, np.int8) for i, (_, p) in enumerate(pools)]) if pools else np.zeros(0, np.int8)
            if len(allp) > MAX_GRAD_PTS:
                pick = rng.choice(len(allp), MAX_GRAD_PTS, replace=False)
                allp, src = allp[pick], src[pick]
            gk[m] = {"n": int(len(allp)), "n_by_set": {n: int((src == i).sum()) for i, (n, _) in enumerate(pools)}}
            if len(allp):
                res = grad_check(f, allp, CL[m], main.layers[0])
                for k, v in res.items():
                    npz[f"{mode}__gc_{m}_{k}"] = np.asarray(v, np.float32) if v.dtype != bool else v
                npz[f"{mode}__gc_{m}_src"] = src
        ir["grad_check"] = gk
        del F, C, FL, CL, fields
        torch.cuda.empty_cache()
    rec["load_after"] = M.pod_load()
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
                          target_exclusion={"owned_tie_voxels": 1.0, "owned_max_dist_voxels": 3.0})
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
        s = " | ".join(
            f"{m}: kept " + " ".join(f"{x}:{rec['identities'][m]['coarse'][x]['counts'].get('n_shared_kept')}"
                                     for x in ("ball", "bo_d3"))
            + " Dpos off/on(ball) " + (f"{int((npz[f'{m}__D_d_ball|off'] > 0).sum())}/{int((npz[f'{m}__D_d_ball|on'] > 0).sum())}"
                                       if f"{m}__D_d_ball|off" in npz else "-")
            for m in IDENTITIES if "coarse" in rec["identities"][m])
        print(item["key"], rec["wall_s"], s, flush=True)
        del ag3s, ctx
        torch.cuda.empty_cache()
    for s in scene_cache.values():
        s[0].close()


if __name__ == "__main__":
    main()
