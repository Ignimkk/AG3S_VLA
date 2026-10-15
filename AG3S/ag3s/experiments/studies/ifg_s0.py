"""IFG S0 — provenance / alignment checks of the phase dump, and the depth replay used by S2/S3.

Protocol v1.1 §4. Per completed dump frame:
  * record pairing: dump raw_state vs record step.state, dump images vs record uint8 images,
    collector t_step, GT fingerprints vs `record_fingerprint(run)`;
  * internal tensors: non-finite / float16-saturated members of embed / prefix / suffix / derived,
    and whether the members S1 uses (L4/L8/L12 image rows, encoded, image_tokens, maps) are finite;
    serving map in derived.npz vs `attn_probs[0, 8, 2, -1, :768]`;
  * replay (CPU MuJoCo, osmesa): RGB MAD vs record, patch class coverage vs stored GT,
    depth convention via `mj_ray` (planar z vs radial), back-project -> project roundtrip (px);
  * writes `replay/<frame_id>.npz` (depth, K, T_base_cam: algorithm input — a simulated sensor) and
    `labels/<frame_id>.npz` (body ids, class map: evaluator only).

Usage:
  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_s0 \
      --out $RUN/s0 --workers 4
Frames whose replay npz already exists are skipped unless --force (the dump is still growing).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_reader as R

#: geom groups the default MuJoCo renderer draws (0-2); collision-only groups are invisible to depth.
from AG3S.ag3s.experiments.studies.t47_replay import CLS as R_CLS  # noqa: E402
VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)


def _check_episode(args):
    refs, out, force = args
    from AG3S.ag3s.experiments.sources.policy_record import (CAMERA_BINDINGS, load_run, patch_coverage,
                                                              pose_scene, record_fingerprint, replay_scene)
    from AG3S.ag3s.experiments.studies.t47_replay import (POLICY_SOURCE_HW, body_class_table, class_map,
                                                          render_policy_rgb)
    import mujoco

    out = pathlib.Path(out)
    todo = [r for r in refs if force or not (out / "replay" / f"{r.frame_id}.npz").exists()]
    if not todo:
        return []
    rec_dir = R.record_dir(todo[0])
    run = load_run(rec_dir)
    fp = record_fingerprint(run)
    scene = replay_scene(run)
    rows = []
    try:
        rgb_renderer = mujoco.Renderer(scene.model, *POLICY_SOURCE_HW)
        for ref in todo:
            t0 = time.perf_counter()
            row = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds, key=ref.key, frame=ref.frame,
                       moment=ref.moment, kind=ref.kind, record=str(rec_dir), record_fingerprint=fp, problems=[])
            step = run.steps[ref.frame]
            row["t_step"] = int(step.t_step)
            if int(ref.done["parity"]["t_step"]) != int(step.t_step):
                row["problems"].append("collector t_step != record t_step")
            row["collector_parity"] = {k: v for k, v in ref.done["parity"].items() if k != "hsr_p"}
            fin = R.load_frame(ref)
            row["state_maxabs"] = float(np.abs(fin.raw_state - step.state).max())
            img_u8 = np.stack([step.images[c] for c in R.RECORD_CAMERAS]).astype(np.float64)
            row["image_maxabs_u8"] = float(np.abs((fin.images.astype(np.float64) + 1) / 2 * 255 - img_u8).max())
            row["serving_map_vs_probs_maxabs"] = float(np.abs(fin.maps["A"] - fin.maps["_A_from_probs"]).max())
            row["s1_inputs_finite"] = bool(all(np.isfinite(v).all() for v in fin.features.values())
                                           and all(np.isfinite(v).all() for v in fin.maps.values()))
            row["nonfinite"] = {f: R.npz_nonfinite(ref.path / f"{f}.npz") for f in ("embed", "derived", "suffix")}
            row["nonfinite"]["prefix"] = R.npz_nonfinite(ref.path / "prefix.npz", ["resid_out", "final_hidden"])
            if ref.done.get("target") and ref.done["target"] not in fin.prompt:
                row["problems"].append("target noun not in prompt")
            row["prompt"] = fin.prompt
            row["query_rows"] = fin.query_rows
            del fin
            gt = R.load_gt(ref, int(step.t_step))
            row["gt_fingerprint_ok"] = gt["fingerprint"] == fp and gt["grasp_fingerprint"] == fp
            if not row["gt_fingerprint_ok"]:
                row["problems"].append("GT fingerprint mismatch")
            row["target"] = gt["target"]
            row["grasp"] = gt["grasp"]
            # ---------------------------------------------------------------- replay
            pose_scene(scene, step)
            lut, _ = body_class_table(scene.model, gt["target"])
            depth, Ks, Ts, bids, clss, cams = [], [], [], [], [], []
            per_cam = []
            for ci, (pol, (mjcam, _, _)) in enumerate(CAMERA_BINDINGS.items()):
                fr = scene.capture(mjcam)
                cls = class_map(fr.body_ids, lut)
                cov, _ = patch_coverage(cls, list(range(len(gt["classes"]))))
                rgb = render_policy_rgb(scene.model, scene.data, rgb_renderer, mjcam)
                dep = fr.depth
                K, T = fr.camera_intrinsics, fr.T_base_cam
                valid = (dep > 0.05) & (dep < 3.0) & np.isfinite(dep)
                # mj_ray along 24 deterministic valid pixels: planar-z vs radial and K/T consistency
                vy, vx = np.nonzero(valid)
                pick = np.linspace(0, len(vy) - 1, 24).astype(int) if len(vy) else np.zeros(0, int)
                cid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, mjcam)
                cam_pos = scene.data.cam_xpos[cid].copy()
                Rwc = scene.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
                z_err, r_err = [], []
                for j in pick:
                    u, v = vx[j] + 0.5, vy[j] + 0.5
                    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
                    d_w = Rwc @ (d_cam / np.linalg.norm(d_cam))
                    gid = np.array([-1], np.int32)
                    r = mujoco.mj_ray(scene.model, scene.data, cam_pos, d_w, VISIBLE_GROUPS, 1,
                                     int(scene.model.cam_bodyid[cid]), gid)  # skip the camera housing
                    if r < 0:
                        continue
                    z_ray = r / np.linalg.norm(d_cam)
                    z_err.append(abs(z_ray - dep[vy[j], vx[j]]))
                    r_err.append(abs(r - dep[vy[j], vx[j]]))
                # back-project -> project roundtrip on valid pixels
                yy, xx = np.nonzero(valid)
                z = dep[yy, xx]
                pc = np.stack([(xx + 0.5 - K[0, 2]) * z / K[0, 0], (yy + 0.5 - K[1, 2]) * z / K[1, 1], z], -1)
                pb = pc @ T[:3, :3].T + T[:3, 3]
                pc2 = (pb - T[:3, 3]) @ T[:3, :3]
                u2 = K[0, 0] * pc2[:, 0] / pc2[:, 2] + K[0, 2] - 0.5
                v2 = K[1, 1] * pc2[:, 1] / pc2[:, 2] + K[1, 2] - 0.5
                rt = np.hypot(u2 - xx, v2 - yy)
                per_cam.append(dict(
                    policy_camera=pol, mujoco_camera=mjcam,
                    rgb_mad_u8=float(np.abs(rgb.astype(float) - step.images[pol]).mean()),
                    gt_patch_cov_maxabs=float(np.abs(cov - gt["class_cov"][ci]).max()),
                    valid_fraction=float(valid.mean()),
                    ray_n=len(z_err),
                    ray_planar_z_err_max_m=float(max(z_err)) if z_err else None,
                    ray_planar_z_err_median_m=float(np.median(z_err)) if z_err else None,
                    ray_planar_z_frac_le_1mm=float(np.mean(np.asarray(z_err) <= 1e-3)) if z_err else None,
                    ray_radial_err_max_m=float(max(r_err)) if r_err else None,
                    roundtrip_px_max=float(rt.max()) if rt.size else None))
                depth.append(dep.astype(np.float32)); Ks.append(K); Ts.append(T)
                bids.append(fr.body_ids.astype(np.int32)); clss.append(cls.astype(np.int16)); cams.append(mjcam)
            row["cameras"] = per_cam
            # serving self-filter (FK spheres + covering capsules, fingers at the rendered opening),
            # decided per pixel of the serving back-projection: algorithm input, not GT
            from AG3S.ag3s.config import PointCloudConfig
            from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
            from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
            from AG3S.ag3s.stages.reconstruction import backproject
            from AG3S.ag3s.stages.robot_filter import depth_robot_mask
            robot_model = build_robot_model(scene)
            set_finger_joints_from_scene(scene, robot_model)
            pc_cfg = PointCloudConfig()
            q = scene.robot_state()
            rmask = []
            for ci in range(len(depth)):
                full = backproject(depth[ci].astype(np.float64), Ks[ci], Ts[ci], pc_cfg)
                rm = depth_robot_mask(full, depth[ci].shape, robot_model, q, pc_cfg).mask
                rmask.append(rm)
                dv = (depth[ci] >= pc_cfg.depth_min) & (depth[ci] <= pc_cfg.depth_max) & np.isfinite(depth[ci])
                gt_robot = np.isin(clss[ci], [R_CLS["robot"], R_CLS["robot_gripper"]]) & dv
                per_cam[ci]["fk_robot_mask_px"] = int(rm.sum())
                per_cam[ci]["fk_robot_mask_recall_vs_gt"] = (float((rm & gt_robot).sum() / gt_robot.sum())
                                                             if gt_robot.any() else None)
                per_cam[ci]["fk_robot_mask_nonrobot_px"] = int((rm & ~gt_robot).sum())
                per_cam[ci]["fk_robot_mask_target_px"] = int((rm & (clss[ci] == R_CLS["target"])).sum())
            for c in per_cam:
                if c["gt_patch_cov_maxabs"] > 1e-6:
                    row["problems"].append(f"{c['mujoco_camera']}: replay GT coverage differs")
                if c["rgb_mad_u8"] > 1.0:
                    row["problems"].append(f"{c['mujoco_camera']}: replay RGB MAD > 1")
                # revised 2026-10-15 18:55 KST after 2 frames failed on a single mesh-silhouette ray
                # (banana, 1.4-1.6 mm; other 23 rays at um level): median <= 0.1 mm and >= 90 % rays <= 1 mm
                if c["ray_planar_z_err_median_m"] is not None and (c["ray_planar_z_err_median_m"] > 1e-4
                                                                    or c["ray_planar_z_frac_le_1mm"] < 0.9):
                    row["problems"].append(f"{c['mujoco_camera']}: depth is not planar z within 1 mm")
                if c["roundtrip_px_max"] is not None and c["roundtrip_px_max"] > 0.5:
                    row["problems"].append(f"{c['mujoco_camera']}: roundtrip > 0.5 px")
            if row["state_maxabs"] > 1e-5:
                row["problems"].append("dump raw_state != record state")
            if row["image_maxabs_u8"] > 1.0:
                row["problems"].append("dump image != record image")
            if not row["s1_inputs_finite"]:
                row["problems"].append("S1 feature/map non-finite")
            (out / "replay").mkdir(parents=True, exist_ok=True)
            (out / "labels").mkdir(parents=True, exist_ok=True)
            np.savez(out / "labels" / f"{ref.frame_id}.npz", body_ids=np.stack(bids), class_map=np.stack(clss),
                     classes=np.array(gt["classes"]), lut=lut, target=gt["target"],
                     class_cov=gt["class_cov"], grasp=json.dumps(gt["grasp"]))
            tmp = out / "replay" / f"{ref.frame_id}.tmp.npz"
            np.savez(tmp, depth=np.stack(depth), robot_mask=np.stack(rmask), K=np.stack(Ks), T_base_cam=np.stack(Ts), cameras=np.array(cams),
                     qpos=step.qpos, t_step=int(step.t_step), record_fingerprint=fp,
                     note="simulated depth: MuJoCo replay of the recorded qpos, not a recorded sensor")
            os.replace(tmp, out / "replay" / f"{ref.frame_id}.npz")
            row["valid"] = not row["problems"]
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
        rgb_renderer.close()
    finally:
        scene.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit-episodes", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    refs = R.list_frames()
    eps: dict = {}
    for r in refs:
        eps.setdefault(r.episode_id, []).append(r)
    jobs = [(v, str(a.out), a.force) for _, v in sorted(eps.items())][: a.limit_episodes]
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool:
        for res in pool.imap_unordered(_check_episode, jobs):
            rows.extend(res)
            for r in res:
                print(f"{r['frame_id']} valid={r['valid']} {r['seconds']:.1f}s {r['problems'][:2]}", flush=True)
    path = a.out / "provenance_checks.jsonl"
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} frames checked in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
