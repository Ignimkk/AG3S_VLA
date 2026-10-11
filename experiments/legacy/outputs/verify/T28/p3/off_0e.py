"""T28 Phase 3 offline 0e — self-filter at inflation 0 with the T29 cover model, every planning row of T14 (30) / T17 (75), ep 1807.
Same method as P1-3b (outputs/verify/T28/P1-3/p13b_selffilter.py): MuJoCo model_transport.xml re-posed with the row qpos,
cameras zed_left / wrist_cam_l / wrist_cam_r, every valid depth pixel back-projected, GT = TransportScene.capture(cam).body_ids.
Changes vs P1-3b (working tree T26+T29):
  filter model = grounding_report.build_robot_model(scene) (T29: arm_5 URDF capsule replaced, 22-body mesh cover, 461 spheres);
  finger joints = mujoco_source.set_finger_joints_from_scene(scene, robot) per row (the row's measured opening, client convention);
  PointCloudConfig(self_filter_inflation=0.0) (T26 default).
Rows appended to off_0e_raw.jsonl (resumable). usage: off_0e.py t14|t17
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, pathlib
import numpy as np, mujoco
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import filter_robot_points
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {"t14": _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl'),
        "t17": _ag3s_workspace_path('outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl')}
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
run = sys.argv[1]
OUT = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T28/p3/off_0e_raw_{run}.jsonl")
m = mujoco.MjModel.from_xml_path(XML)
robot = build_robot_model(TransportScene.attach(m, mujoco.MjData(m)))
d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
robot_bids = np.array([b for b in range(m.nbody) if m.body_rootid[b] == root])
done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        r = json.loads(l); done.add((r["seq"], r["camera"]))
print("n_spheres filter model", robot.n_spheres, "param joints", robot.param_joint_names, "already", len(done), flush=True)
cfg = PointCloudConfig(self_filter_inflation=0.0)
plan = [json.loads(l) for l in open(RUNS[run]) if '"planning"' in l]
z0 = plan[0]["object_poses"]["apple"]["pos"][2]
for fr in plan:
    if all((fr["seq"], c) in done for c in CAMS):
        continue
    d.qpos[:] = np.asarray(fr["qpos"], float); mujoco.mj_forward(m, d)
    q = scene.robot_state()
    fj = set_finger_joints_from_scene(scene, robot)
    for cam in CAMS:
        if (fr["seq"], cam) in done:
            continue
        f = scene.capture(cam)
        valid = (f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0)
        vs, us = np.nonzero(valid); K = f.camera_intrinsics; zz = f.depth[vs, us]
        pc = np.stack([(us - K[0, 2]) * zz / K[0, 0], (vs - K[1, 2]) * zz / K[1, 1], zz], 1)
        pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
        bids = f.body_ids[vs, us]
        ap = bids == apple_bid; rb = np.isin(bids, robot_bids)
        t0 = time.time()
        cloud = PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32))
        kept_cloud, stats = filter_robot_points(cloud, robot, q, cfg)
        kept = np.zeros(f.depth.shape, bool); kept[kept_cloud.uv[:, 1], kept_cloud.uv[:, 0]] = True
        passed = kept[vs, us]
        per = {}
        for b in np.unique(bids[rb]):
            sel = bids == b
            per[bname(b)] = [int(sel.sum()), int((sel & passed).sum())]
        other = ~ap & ~rb
        row = dict(run=run, seq=fr["seq"], t=fr["t_step"], camera=cam, inflation=0.0, finger_joints=fj,
                   apple_dz_mm=1000 * (fr["object_poses"]["apple"]["pos"][2] - z0),
                   apple_px_visible=int(ap.sum()), apple_px_passed=int((ap & passed).sum()),
                   robot_px_visible=int(rb.sum()), robot_px_passed=int((rb & passed).sum()),
                   other_px_visible=int(other.sum()), other_px_passed=int((other & passed).sum()),
                   robot_by_body_visible_passed=per, cloud_total=int(len(pb)), cloud_passed=int(passed.sum()),
                   wall_s=round(time.time() - t0, 3))
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    print(run, fr["seq"], fr["t_step"], flush=True)
print("done")
