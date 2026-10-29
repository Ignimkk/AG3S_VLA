"""Which link's sphere removes each erased apple pixel (B and D). Uses robot_sphere_mask per link subset."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, os, json
sys.path.insert(0, _ag3s_workspace_path('')); os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.stages.robot_filter import robot_sphere_mask, resolve_self_filter_inflation
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, gap_filling_capsules
from AG3S.ag3s.robot_models.urdf_sphere_chain import load_rby1
from AG3S.ag3s.runtime.asset_path import resolve_asset
RUN = _ag3s_workspace_path('outputs/live_test/20260928_t14')
p = resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
m = mujoco.MjModel.from_xml_path(str(p)); d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
robot = load_rby1(extra_capsules=gap_filling_capsules(m), fixed_joint_values={"head_0": 0.0, "head_1": 0.0})
names = np.asarray([str(n) for n in robot.sphere_link_names])
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
frames = {json.loads(l)["seq"]: json.loads(l) for l in open(f"{RUN}/execute_ep1807/frames.jsonl") if json.loads(l).get("kind") == "planning"}
cfgB = PointCloudConfig(self_filter_inflation_by_link={"gripper": 0.010})
inflB = resolve_self_filter_inflation(list(names), cfgB)
out = {}
for seq in [11, 12, 13, 14]:
    fr = frames[seq]; d.qpos[:] = np.asarray(fr["qpos"], float); mujoco.mj_forward(m, d)
    q = scene.robot_state(); C, R = robot.sphere_centers_numeric(q)
    apple = np.asarray(fr["object_poses"]["apple"]["pos"], float)
    for cam in ("zed_left", "wrist_cam_l"):
        f = scene.capture(cam); mask = f.body_ids == apple_bid
        vs, us = np.nonzero(mask); K = f.camera_intrinsics; z = f.depth[vs, us]
        pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
        pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
        erasedB = robot_sphere_mask(pb, C, R, inflB)
        by_link_B = {}; by_link_bare = {}
        for ln in dict.fromkeys(names):
            sel = names == ln
            hitB = robot_sphere_mask(pb, C[sel], R[sel], inflB[sel])
            hit0 = robot_sphere_mask(pb, C[sel], R[sel], 0.0)
            if hitB.any(): by_link_B[ln] = {"erased_px_B": int(hitB.sum()), "inflation_m": float(inflB[sel][0])}
            if hit0.any(): by_link_bare[ln] = int(hit0.sum())
        bare_any = robot_sphere_mask(pb, C, R, 0.0)
        dist_apple = np.linalg.norm(pb - apple, axis=1)
        out[f"seq{seq}_t{fr['t_step']}_{cam}"] = {
            "apple_px": int(mask.sum()), "erased_px_B": int(erasedB.sum()),
            "erased_B_by_link": by_link_B, "inside_bare_sphere_by_link": by_link_bare,
            "inside_any_bare_sphere": int(bare_any.sum()),
            "apple_px_farther_than_60mm_from_apple_pos": int((dist_apple > 0.06).sum()),
            "apple_px_dist_to_apple_pos_mm_min_max": [float(dist_apple.min() * 1000), float(dist_apple.max() * 1000)],
        }
        print(seq, cam, json.dumps(out[f"seq{seq}_t{fr['t_step']}_{cam}"]))
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T19/attribution.json'), "w"), indent=1)
