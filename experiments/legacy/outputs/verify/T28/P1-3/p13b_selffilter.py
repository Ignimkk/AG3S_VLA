"""T28 P1-3 (b) self-filter inflation 0 vs 0.05 on every planning row of T14 (30) and T17 (75), ep 1807.
Code path: AG3S.ag3s.stages.robot_filter.filter_robot_points (HEAD cee8f84 snapshot), explicit
PointCloudConfig(self_filter_inflation=X) — no new defaults. Filter model = grounding_report.build_robot_model(scene)
on a fresh MjData (same call serve_safe.build_ag3s makes: head joints read at attach time = 0, finger joints pinned 0).
Scene: MuJoCo model_transport.xml re-posed with the planning-row qpos; cameras zed_left, wrist_cam_l, wrist_cam_r;
every valid depth pixel back-projected (no stride); ground truth = TransportScene.capture(cam).body_ids.
Rows appended to p13b_raw.jsonl (resumable).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, os, json, time, pathlib
import numpy as np, mujoco
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import filter_robot_points
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/P1-3/p13b_raw.jsonl'))
RUNS = {"t14": _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl'),
        "t17": _ag3s_workspace_path('outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl')}
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
SETTINGS = (0.0, 0.05)

m = mujoco.MjModel.from_xml_path(XML)
robot = build_robot_model(TransportScene.attach(m, mujoco.MjData(m)))   # fresh data, like the server
d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
robot_bids = np.array([b for b in range(m.nbody) if m.body_rootid[b] == root])
done = set()
if OUT.exists():
    for l in OUT.read_text().splitlines():
        r = json.loads(l); done.add((r["run"], r["seq"], r["camera"], r["inflation"]))
print("n_spheres filter model", robot.n_spheres, "already", len(done), flush=True)
for run, path in RUNS.items():
    plan = [json.loads(l) for l in open(path) if '"planning"' in l]
    for fr in plan:
        if all((run, fr["seq"], c, s) in done for c in CAMS for s in SETTINGS):
            continue
        d.qpos[:] = np.asarray(fr["qpos"], float); mujoco.mj_forward(m, d)
        q = scene.robot_state()
        apple = np.asarray(fr["object_poses"]["apple"]["pos"], float)
        for cam in CAMS:
            f = scene.capture(cam)
            valid = (f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0)
            vs, us = np.nonzero(valid); K = f.camera_intrinsics; z = f.depth[vs, us]
            pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
            pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
            bids = f.body_ids[vs, us]
            ap = bids == apple_bid; rb = np.isin(bids, robot_bids)
            for s in SETTINGS:
                if (run, fr["seq"], cam, s) in done:
                    continue
                t0 = time.time()
                cfg = PointCloudConfig(self_filter_inflation=s)
                cloud = PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32))
                kept_cloud, stats = filter_robot_points(cloud, robot, q, cfg)
                kept = np.zeros(f.depth.shape, bool); kept[kept_cloud.uv[:, 1], kept_cloud.uv[:, 0]] = True
                passed = kept[vs, us]
                per = {}
                for b in np.unique(bids[rb]):
                    sel = bids == b
                    per[bname(b)] = [int(sel.sum()), int((sel & passed).sum())]
                row = dict(run=run, seq=fr["seq"], t=fr["t_step"], camera=cam, inflation=s,
                           apple_px_visible=int(ap.sum()), apple_px_passed=int((ap & passed).sum()),
                           robot_px_visible=int(rb.sum()), robot_px_passed=int((rb & passed).sum()),
                           robot_by_body_visible_passed=per, cloud_total=int(len(pb)), cloud_passed=int(passed.sum()),
                           apple_dist_from_base_m=None, wall_s=round(time.time() - t0, 3))
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
        print(run, fr["seq"], fr["t_step"], flush=True)
print("done")
