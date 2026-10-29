"""T30 #5 — self-filter leak per robot body on the regression-baseline record run_16d_ep1800 (15 frames x zed_left / wrist_cam_l /
wrist_cam_r), working tree (T30b cover incl. base), GT segmentation. Method = T28 off_0e / T29 (P1-3b): every valid depth pixel
back-projected, filter_robot_points(robot = grounding_report.build_robot_model(scene), q, PointCloudConfig(self_filter_inflation=X)),
finger joints from the scene (set_finger_joints_from_scene). Scene / pose via policy_record.load_run / replay_scene / pose_scene
(the esdf_rollout path). 'leak' = robot-body pixels that pass the filter; 'other_pass' = non-robot, non-apple pixels kept;
also non-robot pixels REMOVED by the filter per body (e.g. crate). usage: base_leak.py INFLATION OUT.jsonl"""
import sys, json
import numpy as np, mujoco
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import filter_robot_points
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
INF = float(sys.argv[1]); OUT = sys.argv[2]
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
run = load_run("run_16d_ep1800", limit=15)
scene = replay_scene(run); m = scene.model
robot = build_robot_model(scene)
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ""
root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
robot_bids = np.array([b for b in range(m.nbody) if m.body_rootid[b] == root])
apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
cfg = PointCloudConfig(self_filter_inflation=INF)
with open(OUT, "w") as fh:
    fh.write(json.dumps({"n_filter_spheres": int(robot.n_spheres), "inflation": INF}) + "\n")
    for i, step in enumerate(run):
        pose_scene(scene, step); set_finger_joints_from_scene(scene, robot); q = scene.robot_state()
        for cam in CAMS:
            f = scene.capture(cam)
            valid = (f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0)
            vs, us = np.nonzero(valid); K = f.camera_intrinsics; z = f.depth[vs, us]
            pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
            pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
            bids = f.body_ids[vs, us]
            kept_cloud, _ = filter_robot_points(PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32)), robot, q, cfg)
            kept = np.zeros(f.depth.shape, bool); kept[kept_cloud.uv[:, 1], kept_cloud.uv[:, 0]] = True
            passed = kept[vs, us]; rb = np.isin(bids, robot_bids); ap = bids == apple
            fh.write(json.dumps(dict(i=i, t=int(step.t_step), camera=cam, robot_vis=int(rb.sum()), robot_pass=int((rb & passed).sum()),
                leak={bname(b): int(((bids == b) & passed).sum()) for b in np.unique(bids[rb & passed])},
                base_vis=int((bids == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")).sum()),
                base_pass=int(((bids == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")) & passed).sum()),
                nonrobot_removed_by_body={bname(b): int(((bids == b) & ~passed).sum()) for b in np.unique(bids[~rb & ~passed])},
                apple_vis=int(ap.sum()), apple_pass=int((ap & passed).sum()),
                other_vis=int((~rb & ~ap).sum()), other_pass=int((~rb & ~ap & passed).sum()))) + "\n")
        print(i, flush=True)
print("done")
