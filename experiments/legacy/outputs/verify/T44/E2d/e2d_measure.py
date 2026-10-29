"""T44c-A6 verifier measurement (verifier-owned; benchmark/ imported read-only). Writes only under outputs/verify/T44/E2d/.

Re-uses e2c_measure.py helpers and the D-sign2 frame / identity / method definitions unchanged
(frames demo 64 + lb 48, identities gt | gt_label, methods ball | bo_d3 (ball_owned remove tie 1.0 d=3)).

Coarse target-free layer in three compositions per method m:
  off   = H.compose_free_field(main, F_m, None)
  guard = ... CL(sign_floor="guard", target=geom.points)   (A5, D-sign2 re-run)
  plane = ... CL(sign_floor="plane", target=geom.points)   (A6)
Point sets (fine lattice centres of the T44E2b box ball centre +- (r+60 mm)):
  D  inside GT distractor hull          T  inside GT target hull
  TB z in [plane-30 mm, plane)  (table interior band; plane = ball.summary support_plane_z_m)
  TA z in (plane, plane+10 mm], outside every GT hull
  q  authorized spheres (DEFAULT_CONTACT_LINKS[arm]) centres ; grid = q + 125 offsets (+-20 mm)
Also per frame: sign/seed counts of each coarse tier, gradient check (value-gradient) for guard & plane,
and 2-D slices (xy at plane+10 mm, xz & yz through ball centre) for the scene figure (gt_label only).
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, dataclasses, json, pathlib, sys, time
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2c'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2b'))
sys.path.insert(0, _ag3s_workspace_path(''))
import e2_measure as M  # noqa: E402
import e2b_measure as B  # noqa: E402
import e2c_measure as C0  # noqa: E402
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

IDENTITIES = ("gt", "gt_label")
METHODS = ("ball", "bo_d3")
VARS = ("off", "guard", "plane")
GRID = M.GRID
TIER_CODES = C0.TIER_CODES
MAX_GRAD_PTS = 4000
SL_RES = 0.0025


def slices(fields, main, ball, sz, t_hulls, d_hulls):
    ext = float(ball.radius) + 0.06
    ax = np.arange(-ext, ext + 1e-9, SL_RES)
    a, b = np.meshgrid(ax, ax, indexing="ij")
    c = np.asarray(ball.centre, np.float64)
    out = {"sl_ax": ax.astype(np.float32), "sl_centre": c}
    planes = {"xy": (lambda a, b: np.stack([c[0] + a, c[1] + b, np.full(a.shape, (sz if sz is not None else c[2]) + 0.010)], -1)),
              "xz": (lambda a, b: np.stack([c[0] + a, np.full(a.shape, c[1]), c[2] + b], -1)),
              "yz": (lambda a, b: np.stack([np.full(a.shape, c[0]), c[1] + a, c[2] + b], -1))}
    for pn, fn in planes.items():
        P = fn(a, b).reshape(-1, 3)
        out[f"sl_{pn}_main"] = np.asarray(main.distance(P), np.float32).reshape(a.shape)
        for fk, f in fields.items():
            out[f"sl_{pn}_{fk}"] = np.asarray(f.target_free_distance(P), np.float32).reshape(a.shape)
        for nm, hl in (("tgt", t_hulls), ("dis", d_hulls)):
            m = np.zeros(len(P), bool)
            for h in hl:
                m |= h.find_simplex(P) >= 0
            out[f"sl_{pn}_in_{nm}"] = m.reshape(a.shape)
    return out


def run_one(ctx, item, args):
    import torch
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence
    from AG3S.ag3s.fields.device_field import layer_from_device
    from AG3S.trajopt import wire

    scene, ag3s, target = ctx["scene"], ctx["ag3s"], ctx["target"]
    arm = item["arm"]
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
    LT, LD = H.LABELS.index("target"), H.LABELS.index("distractor")
    t_hulls = [h for h, c in hulls if c == LT]
    d_hulls = [h for h, c in hulls if c == LD]
    all_hulls = [h for h, _ in hulls]
    builder = ag3s._esdf_builder
    rec = {"key": item["key"], "seq": int(item["seq"]), "source": item["meta"], "set": item["set"], "bin": item["bin"],
           "target": target, "arm": arm, "object_pose_err_mm": round(pose_err * 1e3, 3), "status": cs.status.value,
           "identities": {}}
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
    npz = {"auth_centres": ca.astype(np.float32), "auth_radii": ra.astype(np.float32), "auth_links": names[auth],
           "r_finger": np.float64(r_finger)}
    rng = np.random.default_rng(int(item["seq"]) * 1000 + len(item["key"]))
    for mode in IDENTITIES:
        geom, idm, _ = H.choose_identity(ag3s, debug, gt_centre, mode, support_mask=support_mask, label_of=label_of)
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
        sz = ir["ball"].get("support_plane_z_m")
        owned = OwnedEvidence(target_points=geom.points, nontarget_points=ntp, tie_voxels=1.0, max_dist_voxels=3.0,
                              unowned="remove")
        ow = {"ball": None, "bo_d3": owned}
        B.build_tier(builder, ball); B.build_tier(builder, ball, owned=owned)
        F = {m: B.build_tier(builder, ball, owned=ow[m])[0] for m in METHODS}
        H.build_coarse_free_tier(builder, ball, nontarget=ntp, target=geom.points, sign_floor="plane")  # warm-up
        Ct, ccount = {}, {}
        for m in METHODS:
            for fl in ("guard", "plane"):
                t, ms = H.build_coarse_free_tier(builder, ball, owned=ow[m], nontarget=ntp, target=geom.points,
                                                 sign_floor=fl)
                Ct[(m, fl)] = t
                ex = dict(t.extra.get("exclusion") or {}); ex.pop("params", None)
                ccount[f"{m}|{fl}"] = {**ex, "n_sign_forced": int(t.extra.get("n_sign_forced", 0)),
                                       "sign_threshold_m": t.extra.get("threshold_m"), "build_ms_once": round(ms, 3)}
        ir["coarse"] = ccount
        FL = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in F.items()}
        CL = {k: layer_from_device(t.values, t.origin, t.voxel_size) for k, t in Ct.items()}
        fields = {}
        for m in METHODS:
            fields[f"{m}|off"] = H.compose_free_field(main, FL[m], None)
            fields[f"{m}|guard"] = H.compose_free_field(main, FL[m], CL[(m, "guard")])
            fields[f"{m}|plane"] = H.compose_free_field(main, FL[m], CL[(m, "plane")])
        # ---- query points
        for qn, P in (("q", ca), ("grid", gc)):
            npz[f"{mode}__{qn}_dmain"] = np.asarray(main.distance(P), np.float32)
            for fk, f in fields.items():
                npz[f"{mode}__{qn}_d_{fk}"] = np.asarray(f.target_free_distance(P), np.float32)
                if not fk.endswith("|off"):
                    npz[f"{mode}__{qn}_tier_{fk}"] = C0.tcode(f.answer_tier(P, target_free=True))
        # ---- interior point sets
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
        srec = {}
        for sname, P in sets.items():
            srec[sname] = {"n": int(len(P))}
            if not len(P):
                continue
            npz[f"{mode}__{sname}_pts"] = P.astype(np.float32)
            npz[f"{mode}__{sname}_dmain"] = np.asarray(main.distance(P), np.float32)
            if sz is not None:
                npz[f"{mode}__{sname}_height"] = (P[:, 2] - float(sz)).astype(np.float32)
            npz[f"{mode}__{sname}_inball"] = (np.linalg.norm(P - ball.centre, axis=1) <= ball.radius) & (P[:, 2] >= ball.z_min)
            for fk, f in fields.items():
                npz[f"{mode}__{sname}_d_{fk}"] = np.asarray(f.target_free_distance(P), np.float32)
                if not fk.endswith("|off"):
                    npz[f"{mode}__{sname}_tier_{fk}"] = C0.tcode(f.answer_tier(P, target_free=True))
        ir["sets"] = srec
        # ---- gradient check where target_free_coarse answered
        gk = {}
        for m in METHODS:
            for fl in ("guard", "plane"):
                f = fields[f"{m}|{fl}"]
                pools = []
                for sname in ("D", "T", "TB", "TA"):
                    if f"{mode}__{sname}_tier_{m}|{fl}" in npz:
                        sel = npz[f"{mode}__{sname}_tier_{m}|{fl}"] == TIER_CODES.index("target_free_coarse")
                        pools.append((sname, npz[f"{mode}__{sname}_pts"].astype(np.float64)[sel]))
                sel = npz[f"{mode}__grid_tier_{m}|{fl}"] == TIER_CODES.index("target_free_coarse")
                pools.append(("grid", gc[sel]))
                allp = np.vstack([p for _, p in pools])
                src = np.concatenate([np.full(len(p), i, np.int8) for i, (_, p) in enumerate(pools)])
                names_p = [n for n, _ in pools]
                if len(allp) > MAX_GRAD_PTS:
                    pick = rng.choice(len(allp), MAX_GRAD_PTS, replace=False)
                    allp, src = allp[pick], src[pick]
                gk[f"{m}|{fl}"] = {"n": int(len(allp)), "set_names": names_p}
                if len(allp):
                    res = C0.grad_check(f, allp, CL[(m, fl)], main.layers[0])
                    for k, v in res.items():
                        npz[f"{mode}__gc_{m}_{fl}_{k}"] = np.asarray(v, np.float32) if v.dtype != bool else v
                    npz[f"{mode}__gc_{m}_{fl}_src"] = src
        ir["grad_check"] = gk
        if mode == "gt_label":
            npz.update({f"{k}": v for k, v in slices(fields, main, ball, sz, t_hulls, d_hulls).items()})
            npz["ball_gt_label"] = np.r_[ball.centre, ball.radius, ball.z_min]
        del F, Ct, FL, CL, fields
        torch.cuda.empty_cache()
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
        print(item["key"], rec["wall_s"], flush=True)
        del ag3s, ctx
        torch.cuda.empty_cache()
    for s in scene_cache.values():
        s[0].close()


if __name__ == "__main__":
    main()
