"""T19 verify: T14 planning poses re-posed in MuJoCo; self-filter decided by the real code path
(AG3S.ag3s.stages.robot_filter.filter_robot_points -> self_filter_mask -> robot_sphere_mask).
Each (setting, seq, camera) row is appended to handoff/T19.verify.partial.json as one JSON line.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, os, json, time
sys.path.insert(0, _ag3s_workspace_path('')); os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import (filter_robot_points, resolve_self_filter_inflation,
                                                inflation_table)
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, gap_filling_capsules
from AG3S.ag3s.robot_models.urdf_sphere_chain import load_rby1
from AG3S.ag3s.runtime.asset_path import resolve_asset

RUN = _ag3s_workspace_path('outputs/live_test/20260928_t14')
PARTIAL = _ag3s_workspace_path('AG3S/ag3s/docs/handoff/T19.verify.partial.json')
RAW = _ag3s_workspace_path('outputs/verify/T19')
SEQS = [10, 11, 12, 13, 14]
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
GUARD_R = 0.06

p = resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
m = mujoco.MjModel.from_xml_path(str(p)); d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
robot = load_rby1(extra_capsules=gap_filling_capsules(m), fixed_joint_values={"head_0": 0.0, "head_1": 0.0})
names = [str(n) for n in robot.sphere_link_names]
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
robot_root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
robot_bids = [b for b in range(m.nbody) if m.body_rootid[b] == robot_root]

frames = {}
for line in open(f"{RUN}/execute_ep1807/frames.jsonl"):
    r = json.loads(line)
    if r.get("kind") == "planning": frames[r["seq"]] = r

def chunk_centroid(k):
    z = np.load(f"{RUN}/server_constraints/run_0000/chunk_{k:05d}.npz", allow_pickle=True)
    return (np.asarray(z["target_centroid"], float) if "target_centroid" in z.files else None,
            json.loads(str(z["summary_json"])).get("target_n_points"))

def cfg_for(setting):
    if setting == "A":
        return PointCloudConfig()
    return PointCloudConfig(self_filter_inflation_by_link={"gripper": 0.010},
                            self_filter_target_guard_radius=(0.0 if setting == "B" else GUARD_R))

def guard_for(setting, seq, apple_pos):
    if setting == "C":   # centroid in the server record chunk_{seq-1} (lead's instruction)
        c, n = chunk_centroid(seq - 1); return c, f"chunk_{seq-1:05d}.target_centroid", n
    if setting == "C2":  # supplementary: chunk_{seq-2} (the value held before this frame's grounding)
        c, n = chunk_centroid(seq - 2); return c, f"chunk_{seq-2:05d}.target_centroid", n
    if setting == "D":
        return apple_pos.copy(), "object_poses.apple.pos (ground truth)", None
    return None, None, None

SETTINGS = ["A", "B", "C", "C2", "D"]
done = set()
if os.path.exists(PARTIAL):
    for line in open(PARTIAL):
        r = json.loads(line); done.add((r["setting"], r["seq"], r["camera"]))
print("already done:", len(done))

# inflation tables
tables = {}
for s in SETTINGS:
    infl = resolve_self_filter_inflation(names, cfg_for(s))
    tables[s] = inflation_table(names, infl)
json.dump({"n_spheres": int(robot.n_spheres), "tables": tables}, open(f"{RAW}/inflation_tables.json", "w"), indent=1)

pix_dump = {}
for seq in SEQS:
    fr = frames[seq]; d.qpos[:] = np.asarray(fr["qpos"], float); mujoco.mj_forward(m, d)
    q = scene.robot_state()
    apple = np.asarray(fr["object_poses"]["apple"]["pos"], float)
    caps = {cam: scene.capture(cam) for cam in CAMS}
    clouds = {}
    for cam, f in caps.items():
        valid = (f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0)
        vs, us = np.nonzero(valid); K = f.camera_intrinsics; z = f.depth[vs, us]
        pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
        pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
        clouds[cam] = (pb, us, vs, f.body_ids[vs, us])
    for s in SETTINGS:
        cfg = cfg_for(s); infl = resolve_self_filter_inflation(names, cfg)
        gc, gsrc, gn = guard_for(s, seq, apple)
        for cam in CAMS:
            if (s, seq, cam) in done: continue
            t0 = time.time()
            pb, us, vs, bids = clouds[cam]
            cloud = PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32))
            kept_cloud, stats = filter_robot_points(cloud, robot, q, cfg, inflation=infl, guard_centre=gc)
            kept = np.zeros((caps[cam].depth.shape[0], caps[cam].depth.shape[1]), bool)
            kept[kept_cloud.uv[:, 1], kept_cloud.uv[:, 0]] = True
            passed = kept[vs, us]
            ap = bids == apple_bid
            rb = np.isin(bids, robot_bids)
            in_guard = (np.linalg.norm(pb - gc, axis=1) <= GUARD_R) if gc is not None and cfg.self_filter_target_guard_radius > 0 else np.zeros(len(pb), bool)
            near = np.linalg.norm(pb - apple, axis=1) < 0.05
            per_body = {}
            for b in np.unique(bids[rb]):
                sel = bids == b
                per_body[bname(b)] = {"visible": int(sel.sum()), "passed": int((sel & passed).sum()),
                                      "passed_in_guard": int((sel & passed & in_guard).sum()),
                                      "passed_out_guard": int((sel & passed & ~in_guard).sum())}
            near_by_body = {}
            for b in np.unique(bids[near & passed]):
                near_by_body[bname(b)] = int(((bids == b) & near & passed).sum())
            row = dict(setting=s, seq=seq, t=fr["t_step"], camera=cam,
                       guard_centre=None if gc is None else [float(x) for x in gc], guard_source=gsrc,
                       guard_centre_to_apple_mm=None if gc is None else float(np.linalg.norm(gc - apple) * 1000),
                       guard_chunk_target_n_points=gn,
                       apple_px_visible=int(ap.sum()), apple_px_passed=int((ap & passed).sum()),
                       robot_px_visible=int(rb.sum()), robot_px_passed=int((rb & passed).sum()),
                       robot_px_passed_in_guard=int((rb & passed & in_guard).sum()),
                       robot_by_body=per_body,
                       cloud_passed_within_50mm_of_apple=int((near & passed).sum()),
                       cloud_passed_within_50mm_by_body=near_by_body,
                       cloud_total=int(len(pb)), cloud_passed=int(passed.sum()),
                       filter_stats={k: v for k, v in stats.items()},
                       wall_s=round(time.time() - t0, 3))
            with open(PARTIAL, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(s, seq, fr["t_step"], cam, "apple", row["apple_px_visible"], row["apple_px_passed"],
                  "robot", row["robot_px_visible"], row["robot_px_passed"], "near50", row["cloud_passed_within_50mm_of_apple"],
                  "prot", stats["n_guard_protected"], flush=True)
            if seq == 13 and cam == "wrist_cam_l":
                np.savez_compressed(f"{RAW}/pix_{s}_seq13_wrist_cam_l.npz", us=us, vs=vs, bids=bids, passed=passed,
                                    guard_centre=np.zeros(3) if gc is None else gc, has_guard=gc is not None and cfg.self_filter_target_guard_radius > 0)
print("done")
