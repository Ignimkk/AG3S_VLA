"""T44c A6 implementer smoke — compare coarse target-free fixes on the D-sign2 demo frames (gt_label).

Imports verifier helpers read-only (outputs/verify/T44/{E2,E2b,E2c}); writes only under outputs/impl/T44/A6/.
Per frame, identity gt_label, methods ball | bo_d3 (ball_owned remove, tie 1, d=3):
  fine target-free tier F, coarse target-free tiers C_guard (A5) and C_plane (candidate a).
Compositions at point sets:
  off = min(f, max(d,b))                       (compose_free_field without coarse)
  a5  = A5 composition with C_guard            (real field code)
  a   = A5 composition with C_plane            (real field code)
  b   = where(fine TSDF observed at point, off, a5)   (candidate b emulated: coarse only where fine unobserved)
  ab  = where(fine observed, off, a)
Point sets: D distractor interior, T target interior (T44E2b definition), q authorized spheres,
  TB table interior (fine lattice, z in [plane-30mm, plane), xy box of ball +-(r+60mm)),
  TA table top band (z in (plane, plane+10mm], outside every GT hull).
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, json, pathlib, sys, time
import numpy as np

sys.dont_write_bytecode = True
for p in (_ag3s_workspace_path('outputs/verify/T44/E2'), _ag3s_workspace_path('outputs/verify/T44/E2b'),
          _ag3s_workspace_path('outputs/verify/T44/E2c'), _ag3s_workspace_path('')):
    sys.path.insert(0, p)
import e2_measure as M  # noqa: E402
import e2b_measure as B  # noqa: E402
import e2c_measure as E  # noqa: E402
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

MARGIN = 0.010
VARS = ("off", "a5", "a", "b", "ab")
METHODS = ("ball", "bo_d3")


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
    gt_depth_pts, gt_depth_lab = H.gt_points(frames.values(), ctx["classes"])
    label_of = H.point_labeler(gt_depth_pts, gt_depth_lab, radius=0.003)
    mesh_pts, mesh_lab, hulls = H.gt_mesh_points(scene, ctx["classes"])
    LT, LD = H.LABELS.index("target"), H.LABELS.index("distractor")
    t_mesh, d_mesh = mesh_pts[mesh_lab == LT], mesh_pts[mesh_lab == LD]
    t_hulls = [h for h, c in hulls if c == LT]
    d_hulls = [h for h, c in hulls if c == LD]
    all_hulls = [h for h, _ in hulls]
    builder = ag3s._esdf_builder
    rec = {"key": item["key"], "seq": int(item["seq"]), "source": item["meta"], "target": target, "arm": arm}
    qm = ctx["query_model"]
    set_finger_joints_from_scene(scene, qm)
    q = np.asarray(frames[args.cameras[0]].robot_state, np.float64)
    c, r = qm.sphere_centers_numeric(q)
    names = np.asarray([str(n) for n in qm.sphere_link_names])
    auth = np.isin(names, list(DEFAULT_CONTACT_LINKS[arm]))
    ca, ra = c[auth], r[auth]
    fin = np.char.find(names[auth].astype(str), "finger") >= 0
    r_finger = float(np.median(ra[fin])) if fin.any() else float(np.median(ra))
    fine_vs = float(ag3s.config.esdf.fine_voxel_size)
    main = cs.esdf
    geom, idm, _ = H.choose_identity(ag3s, debug, gt_centre, "gt_label", support_mask=support_mask,
                                     label_of=label_of)
    rec["identity"] = idm
    if geom is None:
        rec["skipped"] = "no target geometry"; return rec, {}
    ball = ag3s._target_ball(geom.points, surfaces, voxel=fine_vs)
    if ball is None:
        rec["skipped"] = "no ball"; return rec, {}
    ntp = ag3s._nontarget_points(debug.get("filtered_cloud"), support_mask, geom)["points"]
    rec["ball"] = ball.summary()
    sz = rec["ball"].get("support_plane_z_m")
    owned = OwnedEvidence(target_points=geom.points, nontarget_points=ntp, tie_voxels=1.0, max_dist_voxels=3.0,
                          unowned="remove")
    ow = {"ball": None, "bo_d3": owned}
    B.build_tier(builder, ball)   # warm-up
    F = {m: B.build_tier(builder, ball, owned=ow[m])[0] for m in METHODS}
    H.build_coarse_free_tier(builder, ball, nontarget=ntp, target=geom.points, sign_floor="plane")  # warm-up
    C, cex, cms = {}, {}, {}
    for m in METHODS:
        for fl in ("guard", "plane"):
            t, ms = H.build_coarse_free_tier(builder, ball, owned=ow[m], nontarget=ntp, target=geom.points,
                                             sign_floor=fl)
            C[(m, fl)] = t
            cms[f"{m}|{fl}"] = round(ms, 3)
            ex = dict(t.extra.get("exclusion") or {})
            ex.pop("params", None)
            cex[f"{m}|{fl}"] = {**ex, "n_sign_forced": int(t.extra.get("n_sign_forced", 0))}
    rec["coarse"] = cex
    rec["coarse_build_ms"] = cms
    FL = {m: layer_from_device(t.values, t.origin, t.voxel_size) for m, t in F.items()}
    CL = {k: layer_from_device(t.values, t.origin, t.voxel_size) for k, t in C.items()}
    fields = {}
    for m in METHODS:
        fields[(m, "off")] = H.compose_free_field(main, FL[m], None)
        fields[(m, "a5")] = H.compose_free_field(main, FL[m], CL[(m, "guard")])
        fields[(m, "a")] = H.compose_free_field(main, FL[m], CL[(m, "plane")])
    itg_fine = builder._fine_mapper.integrator
    # ---- point sets
    tf = F["ball"]
    torg, tvs = np.asarray(tf.origin, np.float64), float(tf.voxel_size)
    shp = np.array(tf.values.shape)
    Rb = float(ball.radius) + 0.06
    lo = np.maximum(np.floor((ball.centre - Rb - torg) / tvs).astype(int), 0)
    hi = np.minimum(np.ceil((ball.centre + Rb - torg) / tvs).astype(int) + 1, shp)
    sets = {"q": ca}
    if np.all(hi > lo):
        ii = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
        cen = torg + ii * tvs
        ind = np.zeros(len(cen), bool)
        for h in d_hulls:
            ind |= h.find_simplex(cen) >= 0
        int_ = np.zeros(len(cen), bool)
        for h in t_hulls:
            int_ |= h.find_simplex(cen) >= 0
        anyh = np.zeros(len(cen), bool)
        for h in all_hulls:
            anyh |= h.find_simplex(cen) >= 0
        sets["D"] = cen[ind]
        sets["T"] = cen[int_]
        if sz is not None:
            zz = cen[:, 2] - float(sz)
            sets["TB"] = cen[(zz < 0) & (zz >= -0.030)]
            sets["TA"] = cen[(zz > 0) & (zz <= 0.010) & ~anyh]
    npz = {"r_finger": np.float64(r_finger), "auth_radii": ra.astype(np.float32)}
    for sname, P in sets.items():
        npz[f"{sname}_pts"] = P.astype(np.float32)
        if not len(P):
            continue
        npz[f"{sname}_dmain"] = np.asarray(main.distance(P), np.float32)
        st, _, _ = B.tsdf_lookup(itg_fine, P)
        obs = st == 2
        npz[f"{sname}_fine_obs"] = obs
        if sname == "T" and sz is not None:
            npz["T_height"] = (P[:, 2] - float(sz)).astype(np.float32)
        for m in METHODS:
            v = {k: np.asarray(fields[(m, k)].target_free_distance(P), np.float64) for k in ("off", "a5", "a")}
            v["b"] = np.where(obs, v["off"], v["a5"])
            v["ab"] = np.where(obs, v["off"], v["a"])
            for k in VARS:
                npz[f"{sname}_{m}_{k}"] = v[k].astype(np.float32)
            # value / gradient branch consistency on the real candidate-a field (value from _answer == hot path)
            if sname in ("D", "T"):
                va, tier, _ = fields[(m, "a")]._answer(P, True, want_gradient=False)
                npz[f"{sname}_{m}_a_answer_diff"] = np.abs(va - v["a"]).astype(np.float32)
                npz[f"{sname}_{m}_a_tier"] = E.tcode(tier)
    rec["n_points"] = {k: int(len(v)) for k, v in sets.items()}
    del F, C, FL, CL, fields
    torch.cuda.empty_cache()
    return rec, npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posed-frames", default=_ag3s_workspace_path('outputs/verify/T44/E2/e2_demo_posed_frames.json'))
    ap.add_argument("--out", default=_ag3s_workspace_path('outputs/impl/T44/A6/raw_demo'))
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--only", default=None)
    args = ap.parse_args()
    args.lb_frames = None; args.history = 0; args.phase = "approach"
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
    xml = str(resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml",
                            what="transport model"))
    scene = TransportScene(xml, settle_steps=0)
    qm = build_constraint_robot_model(scene, link_filter=constraint_link_filter("gripper"),
                                      sphere_options={"capsule_radius_scale": 0.05, "sphere_spacing": 0.4,
                                                      "max_spheres_per_capsule": 32},
                                      finger_cover={"hands": "both"})
    for item in items:
        fo = out / f"{item['key']}.json"
        if fo.exists():
            continue
        args.cameras = tuple(item.get("cameras") or wire.DEFAULT_CAMERAS)
        ag3s = build_ag3s(xml, voxel=0.020, range_max=2.0, links="gripper", backend="curobo", fine_voxel=0.005,
                          tsdf_voxel=0.005, target_field_policy="exclude-authorized",
                          target_exclusion={"owned_tie_voxels": 1.0, "owned_max_dist_voxels": 3.0})
        ctx = {"scene": scene, "ag3s": ag3s, "target": item["target"],
               "classes": H.label_classes(scene, item["target"]), "query_model": qm}
        t0 = time.time()
        rec, npz = run_one(ctx, item, args)
        rec["wall_s"] = round(time.time() - t0, 2)
        np.savez_compressed(out / f"{item['key']}.npz", **npz)
        fo.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=H._json_default))
        msg = ""
        if "T_pts" in npz and "T_ball_off" in npz:
            msg = " Tneg off/a5/a/b/ab(ball) " + "/".join(str(int((npz[f"T_ball_{v}"] < 0).sum())) for v in VARS)
        print(item["key"], rec["wall_s"], msg, flush=True)
        del ag3s, ctx
        torch.cuda.empty_cache()
    scene.close()


if __name__ == "__main__":
    main()
