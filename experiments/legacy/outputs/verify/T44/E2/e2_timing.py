"""T44 V-E2-3 dedicated timing pass (verifier). Single process, no other verifier GPU job of mine running.
Per frame: AG3S pass, identity gt_label, then target-free fine tier ball / ball_owned_remove / ball_owned_keep built
10x interleaved (after 2 warm-ups), plus `_nontarget_points` time (ball_owned's extra pipeline step). Pod load per frame."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, pathlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2')); sys.path.insert(0, _ag3s_workspace_path(''))
import e2_measure as M
from AG3S.ag3s.experiments.live import exclusion_e2 as H


def main():
    import argparse, torch
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.trajopt import wire
    from AG3S.trajopt.serve_safe import build_ag3s
    ap = argparse.ArgumentParser(); ap.add_argument("--posed-frames"); ap.add_argument("--lb-frames")
    ap.add_argument("--out", required=True); ap.add_argument("--history", type=int, default=0); ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args(); args.phase = "approach"
    items = M.build_items(args)
    out = []; scene = None; cur = None
    for item in items:
        xml = str(resolve_asset(item["model_xml"] or "robots/rby1/description/models/rby1a/mujoco/model_transport.xml", what="m"))
        if xml != cur:
            if scene: scene.close()
            scene = TransportScene(xml, settle_steps=0); cur = xml
        args.cameras = tuple(item.get("cameras") or wire.DEFAULT_CAMERAS)
        ag3s = build_ag3s(xml, voxel=0.020, range_max=2.0, links="gripper", backend="curobo", fine_voxel=0.005, tsdf_voxel=0.005,
                          target_field_policy="exclude-authorized", target_exclusion={"owned_tie_voxels": 1.0, "owned_max_dist_voxels": 2.0})
        ctx = {"scene": scene, "ag3s": ag3s, "target": item["target"], "classes": H.label_classes(scene, item["target"])}
        for h in item["history"]:
            H._history_step(ctx, h, item, args)
        l0 = M.pod_load()
        item["pose"](scene)
        c = H.body_in_base(scene, item["target"])
        frames = {cam: scene.capture(cam) for cam in args.cameras}
        att = H.gt_attention(scene, frames, item["target"], c)
        req = wire.pack_request({}, cameras=args.cameras, depth={k: f.depth for k, f in frames.items()},
                                intrinsics={k: f.camera_intrinsics for k, f in frames.items()},
                                extrinsics={k: f.T_base_cam for k, f in frames.items()},
                                robot_state={k: f.robot_state for k, f in frames.items()}, stamps={k: float(item["seq"]) for k in args.cameras},
                                phase="approach", active_manipulators=[item["arm"]], seq=int(item["seq"]))
        _, sc = wire.strip_request(req)
        set_finger_joints_from_scene(scene, ag3s.robot_model)
        obs = wire.unpack_camera_observations(sc, robot_model=ag3s.robot_model, attention=att)
        cs, debug = ag3s.process_multi_debug(obs, phase="approach", active_manipulators=[item["arm"]])
        gp, gl = H.gt_points(frames.values(), ctx["classes"])
        geom, idm, _ = H.choose_identity(ag3s, debug, c, "gt_label", support_mask=debug.get("support_mask"), label_of=H.point_labeler(gp, gl, 0.003))
        rec = {"key": item["key"], "set": item["set"], "bin": item["bin"]}
        ball = None if geom is None else ag3s._target_ball(geom.points, list(cs.support_surfaces), voxel=0.005)
        if ball is None:
            rec["skipped"] = True; out.append(rec); continue
        nt_ms = []
        for _ in range(5):
            t = time.perf_counter(); nt = ag3s._nontarget_points(debug.get("filtered_cloud"), debug.get("support_mask"), geom); nt_ms.append((time.perf_counter() - t) * 1e3)
        ow = {m: OwnedEvidence(target_points=geom.points, nontarget_points=nt["points"], tie_voxels=1.0, max_dist_voxels=2.0,
                               unowned="keep" if m.endswith("keep") else "remove") for m in ("ball_owned_remove", "ball_owned_keep")}
        b = ag3s._esdf_builder
        for _ in range(2):
            H.build_free_tier(b, ball); H.build_free_tier(b, ball, owned=ow["ball_owned_remove"])
        runs = {m: [] for m in ("ball", "ball_owned_remove", "ball_owned_keep")}
        cls = []
        for _ in range(args.reps):
            for m in runs:
                tier, t = H.build_free_tier(b, ball, owned=ow.get(m)); runs[m].append(round(t, 3))
                if m == "ball_owned_remove": cls.append((tier.extra.get("exclusion") or {}).get("classify_ms"))
        rec.update({"build_ms_runs": runs, "nontarget_points_ms": [round(v, 3) for v in nt_ms], "classify_ms": cls,
                    "load_before": l0, "load_after": M.pod_load()})
        out.append(rec)
        print(item["key"], {m: round(float(np.median(v)), 2) for m, v in runs.items()}, round(float(np.median(nt_ms)), 2), flush=True)
        del ag3s, ctx; torch.cuda.empty_cache()
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=H._json_default))
    scene.close()


if __name__ == "__main__":
    main()
