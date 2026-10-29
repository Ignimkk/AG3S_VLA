"""IFG R6: model-based robot mask from rendered robot depth (realtime_urdf_filter-style).

Robot geoms = geoms of bodies below the gripper root links or `is_robot_body` (the robot description,
not scene GT; mocap markers excluded). They alone are rendered (other geoms moved to group 5 and
disabled) at the recorded pose; a pixel is robot iff the robot render is valid there and the observed
depth agrees within `tol`. Writes <out>/<frame_id>.npz {robot_mask} and quality rows vs GT (evaluator).

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_render_mask --run $RUN --out $RUN/rounds/R6/robot_mask
"""
import argparse, json, multiprocessing as mp, os, pathlib

import numpy as np

TOL = 0.010


def episode(args):
    refs, run, out = args
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    from AG3S.ag3s.experiments.sources.mujoco_source import is_robot_body
    from AG3S.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS, load_run, pose_scene, replay_scene
    from AG3S.ag3s.experiments.studies import ifg_reader as R
    from AG3S.ag3s.experiments.studies.t47_replay import CLS, gripper_body_ids

    todo = [r for r in refs if not (out / f"{r.frame_id}.npz").exists()]
    if not todo:
        return []
    run_rec = load_run(R.record_dir(todo[0]))
    scene = replay_scene(run_rec)
    m = scene.model
    grip = gripper_body_ids(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    robot_body = np.array([(i in grip or is_robot_body(names[i])) and m.body_mocapid[i] < 0 for i in range(m.nbody)])
    robot_geom = robot_body[m.geom_bodyid]
    orig_group = m.geom_group.copy()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[:3] = 1
    renderer = mujoco.Renderer(m, 480, 640)
    rows = []
    try:
        for ref in todo:
            step = run_rec.steps[ref.frame]
            pose_scene(scene, step)
            with np.load(run / "s0/replay" / f"{ref.frame_id}.npz") as z:
                depth, fk_mask = z["depth"], z["robot_mask"]
            with np.load(run / "s0/labels" / f"{ref.frame_id}.npz") as z:
                cm = z["class_map"]
            masks = []
            m.geom_group[:] = np.where(robot_geom, orig_group, 5)
            for ci, (pol, (mjcam, _, _)) in enumerate(CAMERA_BINDINGS.items()):
                renderer.update_scene(scene.data, camera=mjcam, scene_option=opt)
                renderer.enable_depth_rendering()
                rd = np.asarray(renderer.render(), np.float64)
                renderer.disable_depth_rendering()
                valid_r = rd < 0.99 * m.stat.extent * m.vis.map.zfar
                mk = valid_r & (np.abs(depth[ci] - rd) <= TOL)
                masks.append(mk)
                dv = (depth[ci] >= 0.05) & (depth[ci] <= 3.0)
                gt_rob = np.isin(cm[ci], [CLS["robot"], CLS["robot_gripper"]]) & dv
                tg = (cm[ci] == CLS["target"]) & dv
                rows.append(dict(frame_id=ref.frame_id, moment=ref.moment, camera=mjcam,
                                 robot_recall_render=float((mk & gt_rob).sum() / max(gt_rob.sum(), 1)),
                                 robot_recall_fk=float((fk_mask[ci] & gt_rob).sum() / max(gt_rob.sum(), 1)),
                                 nonrobot_removed_render=int((mk & ~gt_rob & dv).sum()), nonrobot_removed_fk=int((fk_mask[ci] & ~gt_rob & dv).sum()),
                                 target_removed_render=int((mk & tg).sum()), target_removed_fk=int((fk_mask[ci] & tg).sum())))
            m.geom_group[:] = orig_group
            np.savez_compressed(out / f"{ref.frame_id}.npz", robot_mask=np.stack(masks), tol=TOL)
    finally:
        m.geom_group[:] = orig_group
        renderer.close(); scene.close()
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--limit-episodes", type=int, default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from AG3S.ag3s.experiments.studies import ifg_reader as R
    eps = {}
    for r in R.list_frames():
        eps.setdefault(r.episode_id, []).append(r)
    jobs = [(v, a.run, a.out) for _, v in sorted(eps.items())][: a.limit_episodes]
    with mp.get_context("spawn").Pool(4) as p, open(a.out / "quality.jsonl", "a") as f:
        for rows in p.imap_unordered(episode, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print("done")


if __name__ == "__main__":
    main()
