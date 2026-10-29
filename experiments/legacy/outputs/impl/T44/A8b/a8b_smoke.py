"""T44e A8b implementer smoke (copied from A8/a8_smoke.py; variants keep | exclude | exclude_target) — coarse target-free tier with / without support (table) seeds, on the D-sign2 demo frames.

Imports verifier helpers read-only (outputs/verify/T44/{E2,E2b,E2c,E2d}); writes only under outputs/impl/T44/A8b/.
Same frames / identity (gt_label) / point-set definitions as E2d (e2d_measure.py). Per frame, methods ball | bo_d3:
  off     = min(f, max(d,b))                                (no coarse tier)
  keep    = coarse tier, sign floor plane, support seeds kept (A7 = C1b)
  exclude = coarse tier, sign floor plane, support seeds excluded (A8, guard lowered to ball z_min)
Point sets D T TB TA q grid as E2d. Plus the T44dA zero level (CLA/p/table_lift.py definition): columns at the ball
centre xy and 8 ring points at the fit radius and at ball radius + 10 mm; z from support-10 mm to support+40 mm step 1 mm;
zero level = lowest z >= support with target-free distance > 0, minus support (mm).
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, contextlib, json, pathlib, sys, time
import numpy as np

sys.dont_write_bytecode = True
for p in (_ag3s_workspace_path('outputs/verify/T44/E2'), _ag3s_workspace_path('outputs/verify/T44/E2b'),
          _ag3s_workspace_path('outputs/verify/T44/E2c'), _ag3s_workspace_path('')):
    sys.path.insert(0, p)
import e2_measure as M  # noqa: E402
import e2b_measure as B  # noqa: E402
import e2c_measure as C0  # noqa: E402
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402
import AG3S.ag3s.fields.curobo_builder as CB  # noqa: E402

METHODS = ("ball", "bo_d3")
VARS = ("off", "keep", "exclude", "exclude_target")
GRID = M.GRID
IDENT = "gt_label"


@contextlib.contextmanager
def keep_guard():
    """excl_g: the support exclusion, but `z_min_m` computed as in keep (A5 guard)."""
    orig = CB.coarse_exclusion_params

    def patched(ball, vs, spec):
        out = orig(ball, vs, spec)
        if getattr(spec, "support", "keep") == "exclude":
            import dataclasses
            k = orig(ball, vs, dataclasses.replace(spec, support="keep", support_points=None,
                                                   support_plane=None))
            out["z_min_m"] = k["z_min_m"]
            if "sign_z_min_m" in k:
                out["sign_z_min_m"] = k["sign_z_min_m"]
        return out

    CB.coarse_exclusion_params = patched
    try:
        yield
    finally:
        CB.coarse_exclusion_params = orig


def zero_levels(field, ball, sz):
    zs = np.arange(sz - 0.010, sz + 0.040 + 1e-9, 0.001)
    R = float(ball.radius)
    fit = (ball.provenance or {}).get("fit") or {}
    Rf = float(fit.get("radius_mm") or (R * 1e3 - 5.0)) / 1e3
    c = np.asarray(ball.centre, np.float64)
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    xy = [c[:2]] + [c[:2] + rr * np.array([np.cos(t), np.sin(t)]) for rr in (Rf, R + 0.010) for t in th]
    P = np.asarray([[x, y, z] for x, y in xy for z in zs])
    V = np.asarray(field.target_free_distance(P), np.float64).reshape(len(xy), -1)
    lv = []
    for col in V:
        k = np.where((zs >= sz - 1e-9) & (col > 0))[0]
        lv.append(1e3 * (zs[k[0]] - sz) if len(k) else np.nan)
    lv = np.asarray(lv)
    return {"centre": float(lv[0]), "ringfit": float(np.nanmedian(lv[1:9])),
            "ringout": float(np.nanmedian(lv[9:])), "all": lv.round(2).tolist()}


def run_one(ctx, item, args):
    import torch
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence
    from AG3S.ag3s.fields.device_field import layer_from_device
    from AG3S.trajopt import wire

    scene, ag3s, target = ctx["scene"], ctx["ag3s"], ctx["target"]
    arm = item["arm"]
    item["pose"](scene)
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
    cloud = debug.get("filtered_cloud")
    gt_depth_pts, gt_depth_lab = H.gt_points(frames.values(), ctx["classes"])
    label_of = H.point_labeler(gt_depth_pts, gt_depth_lab, radius=0.003)
    mesh_pts, mesh_lab, hulls = H.gt_mesh_points(scene, ctx["classes"])
    LT, LD = H.LABELS.index("target"), H.LABELS.index("distractor")
    t_hulls = [h for h, c in hulls if c == LT]
    d_hulls = [h for h, c in hulls if c == LD]
    all_hulls = [h for h, _ in hulls]
    builder = ag3s._esdf_builder
    rec = {"key": item["key"], "seq": int(item["seq"]), "source": item["meta"], "set": item["set"],
           "target": target, "arm": arm, "status": cs.status.value}
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
    fine_vs = float(ag3s.config.esdf.fine_voxel_size)
    main = cs.esdf
    npz = {"auth_centres": ca.astype(np.float32), "auth_radii": ra.astype(np.float32),
           "auth_links": names[auth], "r_finger": np.float64(r_finger)}
    geom, idm, _ = H.choose_identity(ag3s, debug, gt_centre, IDENT, support_mask=support_mask,
                                     label_of=label_of)
    rec["identity_mode_resolved"] = idm
    if geom is None:
        rec["skipped"] = "no target geometry"; return rec, npz
    ball = ag3s._target_ball(geom.points, surfaces, voxel=fine_vs)
    if ball is None:
        rec["skipped"] = "no ball"; return rec, npz
    nontarget = ag3s._nontarget_points(cloud, support_mask, geom)
    ntp = nontarget["points"]
    rec["ball"] = ball.summary()
    sz = rec["ball"].get("support_plane_z_m")
    ev = ag3s._support_seed_evidence(np.asarray(cloud.points), surfaces, ball)
    if ev is not None:   # A8b: as the pipeline does for exclude_target (ignored by exclude)
        ev["inlier_threshold_m"] = float(ag3s.config.support_surface.distance_threshold)
    rec["support_evidence"] = (None if ev is None else
                               {"surface_id": ev["surface_id"], "n_points": int(len(ev["points"])),
                                "plane": [round(float(v), 6) for v in ev["plane"]],
                                "z_at_centre_m": ev["z_at_centre_m"]})
    owned = OwnedEvidence(target_points=geom.points, nontarget_points=ntp, tie_voxels=1.0,
                          max_dist_voxels=3.0, unowned="remove")
    ow = {"ball": None, "bo_d3": owned}
    B.build_tier(builder, ball); B.build_tier(builder, ball, owned=owned)
    F = {m: B.build_tier(builder, ball, owned=ow[m])[0] for m in METHODS}
    H.build_coarse_free_tier(builder, ball, nontarget=ntp, target=geom.points, sign_floor="plane")  # warm-up
    Ct, ccount = {}, {}
    for m in METHODS:
        for v in ("keep", "exclude", "exclude_target"):
            t, ms = H.build_coarse_free_tier(builder, ball, owned=ow[m], nontarget=ntp,
                                             target=geom.points, sign_floor="plane",
                                             support=v, support_evidence=ev)
            Ct[(m, v)] = t
            ex = dict(t.extra.get("exclusion") or {})
            ccount[f"{m}|{v}"] = {**{k: x for k, x in ex.items() if k != "params"},
                                  "params": ex.get("params"),
                                  "n_sign_forced": int(t.extra.get("n_sign_forced", 0)),
                                  "build_ms_once": round(ms, 3)}
    rec["coarse"] = ccount
    FL = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in F.items()}
    CL = {k: layer_from_device(t.values, t.origin, t.voxel_size) for k, t in Ct.items()}
    fields = {}
    for m in METHODS:
        fields[f"{m}|off"] = H.compose_free_field(main, FL[m], None)
        for v in ("keep", "exclude", "exclude_target"):
            fields[f"{m}|{v}"] = H.compose_free_field(main, FL[m], CL[(m, v)])
    # ---- zero level (T44dA definition)
    if sz is not None:
        rec["zero_level_mm"] = {fk: zero_levels(f, ball, float(sz)) for fk, f in fields.items()}
    # ---- query points
    for qn, P in (("q", ca), ("grid", gc)):
        npz[f"{IDENT}__{qn}_dmain"] = np.asarray(main.distance(P), np.float32)
        for fk, f in fields.items():
            npz[f"{IDENT}__{qn}_d_{fk}"] = np.asarray(f.target_free_distance(P), np.float32)
            if not fk.endswith("|off"):
                npz[f"{IDENT}__{qn}_tier_{fk}"] = C0.tcode(f.answer_tier(P, target_free=True))
    # ---- interior point sets (E2d definition)
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
        ind = np.zeros(len(cen), bool); int_ = np.zeros(len(cen), bool); anyh = np.zeros(len(cen), bool)
        for h in d_hulls: ind |= h.find_simplex(cen) >= 0
        for h in t_hulls: int_ |= h.find_simplex(cen) >= 0
        for h in all_hulls: anyh |= h.find_simplex(cen) >= 0
        sets["D"] = cen[ind]; sets["T"] = cen[int_]
        if sz is not None:
            zz = cen[:, 2] - float(sz)
            sets["TB"] = cen[(zz < 0) & (zz >= -0.030)]
            sets["TA"] = cen[(zz > 0) & (zz <= 0.010) & ~anyh]
            # TA wide: table top band out to the whole fine window box (not only ball +- (r+60 mm))
    srec = {}
    for sname, P in sets.items():
        srec[sname] = {"n": int(len(P))}
        if not len(P):
            continue
        npz[f"{IDENT}__{sname}_pts"] = P.astype(np.float32)
        npz[f"{IDENT}__{sname}_dmain"] = np.asarray(main.distance(P), np.float32)
        if sz is not None:
            npz[f"{IDENT}__{sname}_height"] = (P[:, 2] - float(sz)).astype(np.float32)
        npz[f"{IDENT}__{sname}_inball"] = (np.linalg.norm(P - ball.centre, axis=1) <= ball.radius) & (P[:, 2] >= ball.z_min)
        for fk, f in fields.items():
            npz[f"{IDENT}__{sname}_d_{fk}"] = np.asarray(f.target_free_distance(P), np.float32)
            if not fk.endswith("|off"):
                npz[f"{IDENT}__{sname}_tier_{fk}"] = C0.tcode(f.answer_tier(P, target_free=True))
    rec["sets"] = srec
    # slice x-z through the ball centre for the scene figure
    ext = float(ball.radius) + 0.06
    ax = np.arange(-ext, ext + 1e-9, 0.0025)
    a_, b_ = np.meshgrid(ax, ax, indexing="ij")
    cc = np.asarray(ball.centre, np.float64)
    Pxz = np.stack([cc[0] + a_, np.full(a_.shape, cc[1]), cc[2] + b_], -1).reshape(-1, 3)
    npz["sl_ax"] = ax.astype(np.float32); npz["sl_centre"] = cc
    npz["sl_xz_main"] = np.asarray(main.distance(Pxz), np.float32).reshape(a_.shape)
    for fk in ("bo_d3|off", "bo_d3|keep", "bo_d3|exclude", "bo_d3|exclude_target"):
        npz[f"sl_xz_{fk}"] = np.asarray(fields[fk].target_free_distance(Pxz), np.float32).reshape(a_.shape)
    for nm, hl in (("tgt", t_hulls), ("dis", d_hulls)):
        mm = np.zeros(len(Pxz), bool)
        for h in hl:
            mm |= h.find_simplex(Pxz) >= 0
        npz[f"sl_xz_in_{nm}"] = mm.reshape(a_.shape)
    npz["ball"] = np.r_[ball.centre, ball.radius, ball.z_min, float(sz) if sz is not None else np.nan]
    del F, Ct, FL, CL, fields
    torch.cuda.empty_cache()
    return rec, npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posed-frames", default=_ag3s_workspace_path('outputs/verify/T44/E2/e2_demo_posed_frames.json'))
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    args.lb_frames = None; args.history = 0
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
        t0 = time.time()
        rec, npz = run_one(ctx, item, args)
        rec["wall_s"] = round(time.time() - t0, 2)
        np.savez_compressed(out / f"{item['key']}.npz", **npz)
        fo.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=H._json_default))
        print(item["key"], rec["wall_s"], flush=True)
        del ag3s, ctx
        torch.cuda.empty_cache()
    for s in scene_cache.values():
        s[0].close()


if __name__ == "__main__":
    main()
