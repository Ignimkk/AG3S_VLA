"""T44b D-merge (verifier-owned): one AG3S pass per E2-demo frame, everything the merge diagnosis needs
saved to npz. benchmark/ is imported read-only.

Per frame (the same 64 posed frames as V-E2, `outputs/verify/T44/E2/e2_demo_posed_frames.json`):
  * pose (harness `pose_posed`), capture 3 cameras (depth + GT body ids), render RGB 640x480 (AG3S geometry
    cameras) and the 3 policy images (299x224 -> 224x224, reflections off = pi05_infer `render_cam`).
  * attention: `--attention gt`  -> harness `gt_attention` (GT Gaussian, only cameras that see the target)
               `--attention vla` -> the 16x16 VLA maps from `--vla-npz` (all 3 cameras, as serve_safe passes them)
  * `ag3s.process_multi_debug` with the V11AB build (same `build_ag3s` call as V-E2).
  * saved: fused cloud, attention / raw attention, support mask, seeds, grounding clusters (label per point,
    split info), the frame's split record (`TargetConfirm._split`), config values that decide the merge,
    per-point GT body id (nearest GT depth pixel), per-point RGB (projection with depth test),
    GT mesh min gap target <-> every other fruit.
usage: dump_frames.py --out DIR [--attention gt|vla] [--vla-npz F] [--shard i/n] [--only key,..]
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, json, pathlib, sys, time
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/manipulation/src'))
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

POSED = _ag3s_workspace_path('outputs/verify/T44/E2/e2_demo_posed_frames.json')
DATASET = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'))
POLICY_CAMS = {"cam_high": "zed_left", "cam_left_wrist": "wrist_cam_l", "cam_right_wrist": "wrist_cam_r"}


def body_labeler(frames):
    from scipy.spatial import cKDTree
    pts, bid = [], []
    for f in frames:
        depth = np.asarray(f.depth, np.float64)
        ids = np.asarray(f.body_ids)
        v, u = np.nonzero((depth > 1e-6) & np.isfinite(depth) & (ids >= 0))
        z = depth[v, u]
        K, T = np.asarray(f.camera_intrinsics), np.asarray(f.T_base_cam)
        cam = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        pts.append(cam @ T[:3, :3].T + T[:3, 3]); bid.append(ids[v, u])
    P, B = np.vstack(pts), np.concatenate(bid)
    tree = cKDTree(P)

    def f(q, radius=0.010):
        d, j = tree.query(np.asarray(q, np.float64).reshape(-1, 3), k=1)
        out = np.where(d <= radius, B[j], -1).astype(np.int32)
        return out, d.astype(np.float32)
    return f


def point_colours(points, frames, rgbs, order=("zed_left", "wrist_cam_l", "wrist_cam_r"), tol=0.01):
    """RGB per cloud point: first camera (in `order`) that sees it (depth test |z - depth| < tol)."""
    col = np.zeros((len(points), 3), np.uint8)
    src = np.full(len(points), -1, np.int8)
    for ci, cam in enumerate(order):
        f = frames[cam]
        T = np.linalg.inv(np.asarray(f.T_base_cam))
        pc = points @ T[:3, :3].T + T[:3, 3]
        K = np.asarray(f.camera_intrinsics)
        z = pc[:, 2]
        ok = z > 1e-3
        u = np.full(len(points), -1, np.int64); v = u.copy()
        u[ok] = np.round(pc[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(np.int64)
        v[ok] = np.round(pc[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(np.int64)
        h, w = f.depth.shape
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dz = np.full(len(points), np.inf)
        dz[ok] = np.abs(np.asarray(f.depth)[v[ok], u[ok]] - z[ok])
        take = ok & (dz < tol) & (src < 0)
        col[take] = rgbs[cam][v[take], u[take], :3]
        src[take] = ci
    return col, src


def mesh_gaps(scene, target):
    from scipy.spatial import cKDTree
    from AG3S.ag3s.experiments.sources.mujoco_source import _body_surface_points
    base = np.linalg.inv(scene.body_pose("base"))
    surf = {}
    for bid, name in scene._body_names.items():
        if name in H.FRUITS:
            local = _body_surface_points(scene.model, bid, 0.002)
            T = base @ scene.body_pose(name)
            surf[name] = local @ T[:3, :3].T + T[:3, 3]
    tt = cKDTree(surf[target])
    out = {}
    for n, p in surf.items():
        if n == target:
            continue
        d, _ = tt.query(p, k=1)
        out[n] = round(float(d.min()) * 1000, 2)
    return out, surf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention", default="gt", choices=("gt", "vla"))
    ap.add_argument("--vla-npz", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    import mujoco
    import torch
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model  # noqa: F401
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.trajopt import wire
    from AG3S.trajopt.serve_safe import build_ag3s
    from rby1_manipulation.data.recording import POLICY_SOURCE_HEIGHT, POLICY_SOURCE_WIDTH, resize_policy_image

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    posed = json.loads(pathlib.Path(POSED).read_text())
    meta = H.load_demo_meta(DATASET)
    vla = None
    if args.attention == "vla":
        z = np.load(args.vla_npz, allow_pickle=True)
        vla = {str(k): i for i, k in enumerate(z["keys"])}
        vla_maps = z["attention"]          # (F, 3, 16, 16) in POLICY_CAMS order
        vla_cams = [str(c) for c in z["cameras"]]
    k, n = (int(v) for v in args.shard.split("/"))
    posed = posed[k::n]
    xml = str(resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml",
                            what="transport model"))
    scene = TransportScene(xml, settle_steps=0)
    pol_renderer = mujoco.Renderer(scene.model, height=POLICY_SOURCE_HEIGHT, width=POLICY_SOURCE_WIDTH)
    cameras = tuple(wire.DEFAULT_CAMERAS)
    for pf in posed:
        key = f"demo_ep{int(pf['episode_index']):04d}_f{int(pf['frame']):03d}"
        if args.only and key not in args.only.split(","):
            continue
        fo = out / f"{key}.npz"
        if fo.exists():
            print("skip", key, flush=True); continue
        t0 = time.time()
        target, arm = pf["target_fruit"], pf["used_arm"]
        ag3s = build_ag3s(xml, voxel=0.020, range_max=2.0, links="gripper", backend="curobo", fine_voxel=0.005,
                          tsdf_voxel=0.005, target_field_policy="exclude-authorized",
                          target_exclusion={"owned_tie_voxels": 1.0, "owned_max_dist_voxels": 2.0})
        H.pose_posed(scene, pf)
        gt_centre = H.body_in_base(scene, target)
        frames = {c: scene.capture(c) for c in cameras}
        rgbs = {}
        r = scene._renderer_for()
        for c in cameras:
            r.update_scene(scene.data, camera=c)
            rgbs[c] = np.asarray(r.render()).copy()
        pol_imgs = {}
        for pk, c in POLICY_CAMS.items():
            pol_renderer.update_scene(scene.data, camera=c)
            pol_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            pol_imgs[pk] = resize_policy_image(np.asarray(pol_renderer.render()))
        if args.attention == "gt":
            attention = H.gt_attention(scene, frames, target, gt_centre)
        else:
            i = vla[key]
            attention = {POLICY_CAMS[pc]: np.asarray(vla_maps[i, j], np.float32) for j, pc in enumerate(vla_cams)}
        req = wire.pack_request({}, cameras=cameras, depth={c: f.depth for c, f in frames.items()},
                                intrinsics={c: f.camera_intrinsics for c, f in frames.items()},
                                extrinsics={c: f.T_base_cam for c, f in frames.items()},
                                robot_state={c: f.robot_state for c, f in frames.items()},
                                stamps={c: float(pf["frame"]) for c in cameras}, phase="approach",
                                active_manipulators=[arm], seq=int(pf["frame"]))
        _, sc = wire.strip_request(req)
        set_finger_joints_from_scene(scene, ag3s.robot_model)
        if ag3s.constraint_robot_model is not None:
            set_finger_joints_from_scene(scene, ag3s.constraint_robot_model)
        obs = wire.unpack_camera_observations(sc, robot_model=ag3s.robot_model, attention=attention)
        cs, debug = ag3s.process_multi_debug(obs, phase="approach", active_manipulators=[arm])
        cloud = debug["filtered_cloud"]
        pts = np.asarray(cloud.points, np.float64)
        ac = debug["attention_cloud"]
        att = np.asarray(ac.attention, np.float32)
        raw = np.asarray(getattr(ac, "raw_attention", att), np.float32) if getattr(ac, "raw_attention", None) is not None else att
        peak = np.asarray(ac.peak_signal, np.float32)
        support = debug.get("support_mask")
        support = np.zeros(len(pts), bool) if support is None else np.asarray(support, bool)
        g = debug["grounding"]
        clusters = list(getattr(g, "clusters", ()) or ())
        lab = np.full(len(pts), -1, np.int32)
        cl_info = []
        for ci, c in enumerate(clusters):
            lab[np.asarray(c.point_indices, np.int64)] = ci
            cl_info.append({"id": int(c.id), "n": int(c.point_count), "score": float(c.target_score),
                            "mean_attention": float(c.mean_attention), "attention_mass": float(c.attention_mass),
                            "extents_mm": [round(float(v) * 1000, 1) for v in np.asarray(c.principal_extents)],
                            "split": c.split})
        conf = ag3s._target_confirm
        split_rec = getattr(conf, "_split", None)
        bl = body_labeler(frames.values())
        bid, bdist = bl(pts)
        col, csrc = point_colours(pts, frames, rgbs)
        gaps, surf = mesh_gaps(scene, target)
        names = {int(b): n for b, n in scene._body_names.items()}
        cfgc = ag3s.config.clustering
        mo = getattr(ag3s, "_max_opening", None)
        man = getattr(ag3s, "_gated_manipulated", None)
        rec = {
            "key": key, "episode": int(pf["episode_index"]), "frame": int(pf["frame"]), "target": target, "arm": arm,
            "bin": pf["bin"], "attention_source": args.attention, "attention_cameras": sorted(attention),
            "status": cs.status.value, "grounding_status": getattr(getattr(g, "status", None), "value", None),
            "gt_centre": gt_centre.tolist(), "mesh_gap_mm": gaps,
            "fruit_centres": {n: H.body_in_base(scene, n).tolist() for n in H.FRUITS},
            "clusters": cl_info, "split_record": split_rec,
            "chosen_cluster_id": None if g.target is None else int(g.target.id),
            "manipulated_id": None if man is None else int(man.id),
            "config": {"method": cfgc.method, "eps": cfgc.eps, "min_points": cfgc.min_points,
                       "region_growing_radius": cfgc.region_growing_radius,
                       "attention_split_radius": cfgc.attention_split_radius,
                       "attention_split_ratio": cfgc.attention_split_ratio,
                       "gripper_max_opening_cfg": cfgc.gripper_max_opening,
                       "max_opening_m": (float(mo.value_m) if mo is not None and mo.known else None),
                       "seed_percentile": ag3s.config.attention.seed_percentile,
                       "seed_threshold": ag3s.config.attention.seed_threshold,
                       "fusion_voxel_size": ag3s.config.timing.fusion_voxel_size,
                       "pointcloud_voxel_size": ag3s.config.pointcloud.voxel_size,
                       "use_3d_connectivity": cfgc.use_3d_connectivity},
            "fusion_metrics": {k2: (v if isinstance(v, (int, float, str)) else str(v))
                               for k2, v in (getattr(debug.get("fusion"), "metrics", {}) or {}).items()},
            "prompt": meta[int(pf["episode_index"])]["prompt"], "state16": pf["state16"],
            "wall_s": None,
        }
        npz = {"points": pts.astype(np.float32), "attention": att, "raw_attention": raw, "peak_signal": peak,
               "support": support, "seeds": np.asarray(g.seed_indices if g.seed_indices is not None else [], np.int64),
               "cluster_label": lab, "body_id": bid, "body_dist": bdist, "colour": col, "colour_src": csrc,
               "body_names": np.asarray(json.dumps(names)),
               "surf_target": surf[target].astype(np.float32)}
        for nme, p in surf.items():
            npz[f"surf_{nme}"] = p.astype(np.float32)
        if not args.no_images:
            for c in cameras:
                npz[f"rgb_{c}"] = rgbs[c]
                npz[f"bodyids_{c}"] = np.asarray(frames[c].body_ids, np.int32)
                npz[f"K_{c}"] = np.asarray(frames[c].camera_intrinsics)
                npz[f"T_{c}"] = np.asarray(frames[c].T_base_cam)
                if c in attention:
                    npz[f"attmap_{c}"] = np.asarray(attention[c], np.float32)
            for pk, im in pol_imgs.items():
                npz[f"pol_{pk}"] = im
        rec["wall_s"] = round(time.time() - t0, 2)
        np.savez_compressed(fo, **npz)
        (out / f"{key}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=H._json_default))
        print(key, rec["wall_s"], "clusters", len(clusters), "gaps", gaps, flush=True)
        del ag3s, cs, debug
        torch.cuda.empty_cache()
    scene.close()


if __name__ == "__main__":
    main()
