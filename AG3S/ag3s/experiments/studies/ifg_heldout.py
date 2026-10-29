"""IFG held-out — S0 preparation of the collector's held-out rollouts (request v2, `HELDOUT_COLLECTION_REQUEST_KO.md`).

Not part of the locked method (`LOCK_heldout.json`); it only produces the same per-frame files the locked code reads,
from the held-out rollout records instead of the dev replay:

  index.jsonl              one row per frame = every recorded policy step of every finished primary run
                           (clear `main`, adjacent / place `obstacle`), success or not (failures -> appendix)
  replay/<fid>.npz         depth = the RECORDED rollout depth (uint16 mm -> m; algorithm input), K, T_base_cam,
                           robot_mask = serving FK-sphere self-filter on that depth (as dev s0), qpos, t_step
  labels/<fid>.npz         GT for the evaluator only: MuJoCo replay of the recorded qpos in the run's snapshot scene with the
                           run's obstacle profile re-applied; obstacle slots are labelled `other_object` (neighbour)
  robot_mask_render/<fid>.npz  rendered-robot-depth mask (|recorded depth - robot-only render| <= 10 mm), as R6
  s0.jsonl                 per-frame checks: replay depth vs recorded depth, policy RGB MAD, roundtrip, problems

Frame id: HO_<stratum>_ep<episode>_f<step index:03d>.  grasp_state from the collector's grasp_t / place_t (control steps).

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_heldout --heldout <collector run>/heldout --out <analysis run>/heldout_eval --workers 4
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, pathlib, time

import numpy as np

SUB_DIR = {"main": "clear", "adjacent": "obst_adjacent", "place": "obst_place"}
TOL = 0.010


def primary_runs(heldout: pathlib.Path) -> list[dict]:
    st = json.loads((heldout / "state.json").read_text())
    out = []
    for key, r in st["runs"].items():
        if r.get("status") != "done" or r.get("role") not in ("main", "obstacle"):
            continue
        ep = int(r["ep"]); seed = st["episodes"][str(ep)]["seed"]
        sub = SUB_DIR["main"] if r["role"] == "main" else SUB_DIR[r["stratum"]]
        d = heldout / "runs" / sub / f"C1a_ep{ep}_s{seed}"
        if not (d / "ag3s/run_0000/meta.json").exists() or not pathlib.Path(str(d) + ".done").exists():
            continue
        out.append(dict(key=key, ep=ep, seed=seed, stratum=r["stratum"], role=r["role"], success=bool(r.get("success")),
                        failure_mode=r.get("failure_mode"), grasp_t=r.get("grasp_t"), place_t=r.get("place_t"), run_dir=str(d)))
    return sorted(out, key=lambda x: (x["stratum"], x["ep"]))


def grasp_state(t, g, p):
    if g is None or t < g:
        return "pre"
    if p is None or t < p:
        return "held"
    return "placed"


def _run(args):
    info, out = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    from AG3S.ag3s.config import PointCloudConfig
    from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, is_robot_body, set_finger_joints_from_scene
    from AG3S.ag3s.experiments.sources.policy_record import (CAMERA_BINDINGS, load_run, patch_coverage, pose_scene,
                                                              record_fingerprint)
    from AG3S.ag3s.experiments.studies.t47_replay import (CLASS_NAMES, CLS, POLICY_SOURCE_HW, body_class_table, class_map,
                                                          gripper_body_ids, render_policy_rgb)
    from AG3S.ag3s.stages.reconstruction import backproject
    from AG3S.ag3s.stages.robot_filter import depth_robot_mask
    from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager, load_pick_place_obstacle_config

    out = pathlib.Path(out)
    run_dir = pathlib.Path(info["run_dir"])
    rec = load_run(run_dir / "ag3s/run_0000")
    man = json.loads((run_dir / "manifest.json").read_text())
    argv = man.get("extra", {}).get("argv") or []
    prof = argv[argv.index("--obstacle-profile") + 1] if "--obstacle-profile" in argv else "clear"
    xml = pathlib.Path(rec.meta["model_xml"])
    if prof != "clear":
        xml = xml.with_name("model_transport_pick_place_obstacles.xml")
    scene = TransportScene(xml, height=480, width=640, settle_steps=0)
    m = scene.model
    if int(m.nq) != int(rec.meta["nq"]):
        raise ValueError(f"{xml}: nq {m.nq} != recorded {rec.meta['nq']}")
    obstacle_bodies = set()
    if prof != "clear":
        cfg_path = argv[argv.index("--obstacle-config") + 1]
        mgr = PickPlaceObstacleManager(m, scene.data, load_pick_place_obstacle_config(cfg_path))
        mgr.activate(prof)
        gids = [x() if callable(x) else x for x in (mgr.visual_geom_ids, mgr.collision_geom_ids)]
        obstacle_bodies = {int(m.geom_bodyid[g]) for g in set(gids[0]) | set(gids[1])}
    target = man["scene"]["episode_target_fruit"]
    lut, _ = body_class_table(m, target)
    lut = lut.copy()
    for b in obstacle_bodies:
        lut[b] = CLS["other_object"]
    fp = record_fingerprint(rec)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    grip = gripper_body_ids(m)
    robot_geom = np.array([(i in grip or is_robot_body(names[i])) and m.body_mocapid[i] < 0 for i in range(m.nbody)])[m.geom_bodyid]
    orig_group = m.geom_group.copy()
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[:3] = 1
    depth_r = mujoco.Renderer(m, 480, 640)
    rgb_r = mujoco.Renderer(m, *POLICY_SOURCE_HW)
    pc_cfg = PointCloudConfig()
    rows, idx = [], []
    try:
        for k, step in enumerate(rec.steps):
            t0 = time.perf_counter()
            fid = f"HO_{info['stratum']}_ep{info['ep']}_f{k:03d}"
            gs = grasp_state(int(step.t_step), info["grasp_t"], info["place_t"])
            idx.append(dict(frame_id=fid, episode_id=f"HO_ep{info['ep']}", stratum=info["stratum"], role=info["role"],
                            success=info["success"], failure_mode=info["failure_mode"], frame=k, t_step=int(step.t_step),
                            grasp_state=gs, target=target, prompt=rec.meta["prompt"], obstacle_profile=prof, run_dir=info["run_dir"]))
            if (out / "replay" / f"{fid}.npz").exists():
                continue
            pose_scene(scene, step)
            row = dict(frame_id=fid, problems=[], cameras=[])
            depth, Ks, Ts, bids, clss, covs, rmask_r = [], [], [], [], [], [], []
            m.geom_group[:] = orig_group
            for ci, (pol, (mjcam, _, _)) in enumerate(CAMERA_BINDINGS.items()):
                fr = scene.capture(mjcam)
                dep = np.asarray(step.depth[mjcam], np.float64)
                K, T = step.camera_intrinsics[mjcam], step.T_base_cam[mjcam]
                cls = class_map(fr.body_ids, lut)
                cov, _ = patch_coverage(cls, list(range(len(CLASS_NAMES))))
                rgb = render_policy_rgb(m, scene.data, rgb_r, mjcam)
                v = (dep > 0.05) & (dep < 3.0) & (fr.depth > 0.05) & (fr.depth < 3.0)
                dd = np.abs(dep - fr.depth)[v]
                yy, xx = np.nonzero(v); z = dep[yy, xx]
                pc = np.stack([(xx + 0.5 - K[0, 2]) * z / K[0, 0], (yy + 0.5 - K[1, 2]) * z / K[1, 1], z], -1)
                pb = pc @ T[:3, :3].T + T[:3, 3]; pc2 = (pb - T[:3, 3]) @ T[:3, :3]
                rt = np.hypot(K[0, 0] * pc2[:, 0] / pc2[:, 2] + K[0, 2] - 0.5 - xx, K[1, 1] * pc2[:, 1] / pc2[:, 2] + K[1, 2] - 0.5 - yy)
                c = dict(camera=mjcam, depth_vs_replay_median_mm=float(np.median(dd) * 1e3) if dd.size else None,
                         depth_vs_replay_frac_le_2mm=float((dd <= 2e-3).mean()) if dd.size else None,
                         K_vs_replay_maxabs=float(np.abs(K - fr.camera_intrinsics).max()),
                         T_vs_replay_maxabs=float(np.abs(T - fr.T_base_cam).max()),
                         rgb_mad_u8=float(np.abs(rgb.astype(float) - step.images[pol]).mean()),
                         roundtrip_px_max=float(rt.max()) if rt.size else None,
                         target_px=int((cls == CLS["target"]).sum()))
                row["cameras"].append(c)
                if c["depth_vs_replay_frac_le_2mm"] is not None and c["depth_vs_replay_frac_le_2mm"] < 0.95:
                    row["problems"].append(f"{mjcam}: recorded depth differs from replay")
                if c["T_vs_replay_maxabs"] > 1e-3 or c["K_vs_replay_maxabs"] > 1e-6:  # 1 mm; wrist T differs by <= 0.4 mm while moving
                    row["problems"].append(f"{mjcam}: K/T differ from replay")
                if c["rgb_mad_u8"] > 2.0:
                    row["problems"].append(f"{mjcam}: replay RGB MAD > 2")
                if c["roundtrip_px_max"] is not None and c["roundtrip_px_max"] > 0.5:
                    row["problems"].append(f"{mjcam}: roundtrip > 0.5 px")
                depth.append(dep.astype(np.float32)); Ks.append(K); Ts.append(T)
                bids.append(fr.body_ids.astype(np.int32)); clss.append(cls.astype(np.int16)); covs.append(cov)
            # rendered robot mask (R6), against the recorded depth
            m.geom_group[:] = np.where(robot_geom, orig_group, 5)
            for ci, (pol, (mjcam, _, _)) in enumerate(CAMERA_BINDINGS.items()):
                depth_r.update_scene(scene.data, camera=mjcam, scene_option=opt)
                depth_r.enable_depth_rendering(); rd = np.asarray(depth_r.render(), np.float64); depth_r.disable_depth_rendering()
                rmask_r.append((rd < 0.99 * m.stat.extent * m.vis.map.zfar) & (np.abs(depth[ci] - rd) <= TOL))
            m.geom_group[:] = orig_group
            # serving FK-sphere self-filter on the recorded depth (dev s0 convention)
            robot_model = build_robot_model(scene); set_finger_joints_from_scene(scene, robot_model)
            q = scene.robot_state(); fk = []
            for ci in range(3):
                full = backproject(depth[ci].astype(np.float64), Ks[ci], Ts[ci], pc_cfg)
                fk.append(depth_robot_mask(full, depth[ci].shape, robot_model, q, pc_cfg).mask)
            for d_ in ("replay", "labels", "robot_mask_render"):
                (out / d_).mkdir(parents=True, exist_ok=True)
            np.savez(out / "labels" / f"{fid}.npz", body_ids=np.stack(bids), class_map=np.stack(clss), classes=np.array(CLASS_NAMES),
                     lut=lut, target=target, class_cov=np.stack(covs),
                     grasp=json.dumps(dict(grasp_t=info["grasp_t"], place_t=info["place_t"], state=gs)))
            np.savez_compressed(out / "robot_mask_render" / f"{fid}.npz", robot_mask=np.stack(rmask_r), tol=TOL)
            tmp = out / "replay" / f"{fid}.tmp.npz"
            np.savez(tmp, depth=np.stack(depth), robot_mask=np.stack(fk), K=np.stack(Ks), T_base_cam=np.stack(Ts),
                     cameras=np.array([b[0] for b in CAMERA_BINDINGS.values()]), qpos=step.qpos, t_step=int(step.t_step),
                     record_fingerprint=fp, note="recorded rollout depth (uint16 mm); robot_mask = serving FK-sphere self-filter")
            os.replace(tmp, out / "replay" / f"{fid}.npz")
            row["valid"] = not row["problems"]; row["seconds"] = time.perf_counter() - t0
            rows.append(row)
    finally:
        m.geom_group[:] = orig_group
        depth_r.close(); rgb_r.close(); scene.close()
    return idx, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-runs", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = primary_runs(a.heldout)[: a.limit_runs]
    (a.out / "runs.json").write_text(json.dumps(runs, indent=1))
    print(f"{len(runs)} primary runs", flush=True)
    idx_all, rows_all = [], []
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as p:
        for idx, rows in p.imap_unordered(_run, [(r, str(a.out)) for r in runs]):
            idx_all += idx; rows_all += rows
            print(idx[0]["episode_id"], len(idx), "frames", sum(not r["valid"] for r in rows), "invalid", flush=True)
    idx_all.sort(key=lambda r: r["frame_id"])
    (a.out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in idx_all))
    with open(a.out / "s0.jsonl", "a") as f:
        for r in rows_all:
            f.write(json.dumps(r) + "\n")
    print(f"frames {len(idx_all)}  new {len(rows_all)}  invalid {sum(not r['valid'] for r in rows_all)}")


if __name__ == "__main__":
    main()
