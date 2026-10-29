"""rm-model-compare — self-filter (robot mask) sphere model BEFORE vs NOW on one recorded frame.

Frame: T14 closed-loop 16D record execute_ep1807, planning row seq 13 (t_step 96) — the frame of
figures/t14/t14-selffilter-erases-target.png. MuJoCo model_transport.xml re-posed with the row qpos
(qpos + mj_forward = policy_record.pose_scene; this is a live frames.jsonl, not a RunRecord).
Every valid depth pixel back-projected (no stride); GT = TransportScene.capture(cam).body_ids.
Decision = stages/robot_filter.filter_robot_points (current working tree), guard off.

BEFORE (reconstructed from current code by constructor, cross-checked against the HEAD tree in
dump_head.py): UrdfSphereChain(parse_urdf(RBY1_URDF) with ALL URDF capsules incl. link_*_arm_5,
extra_capsules=gap_filling_capsules(m), fixed_joint_values=head) — no cover capsules; finger joints
left at construction default 0.0 (closed); PointCloudConfig(self_filter_inflation=0.05).
NOW: grounding_report.build_robot_model(scene) ; set_finger_joints_from_scene(scene, robot) ;
PointCloudConfig(self_filter_inflation=0.0).
Writes raw.npz + raw.json in this directory.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, pathlib, dataclasses
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import filter_robot_points, robot_sphere_mask
from AG3S.ag3s.experiments.sources.mujoco_source import (
    TransportScene, set_finger_joints_from_scene, gap_filling_capsules, HEAD_JOINTS)
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

OUTD = pathlib.Path(_ag3s_workspace_path('outputs/verify/rm-model-compare'))
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
REC = _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl')
SEQ = 13
CAMS = ("wrist_cam_l", "zed_left", "wrist_cam_r")

m = mujoco.MjModel.from_xml_path(XML)
scene0 = TransportScene.attach(m, mujoco.MjData(m))      # fresh data, like the server (head = 0)

# ---- BEFORE
urdf = parse_urdf(RBY1_URDF)
head = {n: float(scene0.data.qpos[scene0._qadr[n]]) for n in HEAD_JOINTS if n in scene0._qadr}
before = UrdfSphereChain(urdf, extra_capsules=list(gap_filling_capsules(m)), fixed_joint_values=head)
# ---- NOW
now = build_robot_model(scene0)

d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
fr = [json.loads(l) for l in open(REC) if '"planning"' in l]
fr = [r for r in fr if r["seq"] == SEQ][0]
assert fr["t_step"] == 96, fr["t_step"]
d.qpos[:] = np.asarray(fr["qpos"], float); d.qvel[:] = 0; mujoco.mj_forward(m, d)
q = scene.robot_state()
fj_before = {n: before._param_default[n] for n in before.param_joint_names}   # construction default
fj_now = set_finger_joints_from_scene(scene, now)

MODELS = {"before": (before, 0.05), "now": (now, 0.0)}
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
robot_bids = np.array([b for b in range(m.nbody) if m.body_rootid[b] == root])
T_world_base = scene.body_pose("base")

rend = mujoco.Renderer(m, scene.height, scene.width)
npz = {}; out = {"record": REC, "seq": SEQ, "t_step": fr["t_step"], "xml": XML,
                 "robot_state_q": [float(x) for x in q], "T_world_base": T_world_base.tolist(),
                 "apple_pos_world": fr["object_poses"]["apple"]["pos"], "models": {}, "cameras": {}}
for key, (rm, infl) in MODELS.items():
    c, r = rm.sphere_centers_numeric(q)
    npz[f"{key}_centers"] = c; npz[f"{key}_radii"] = r
    npz[f"{key}_links"] = np.array(rm.sphere_link_names); npz[f"{key}_roles"] = np.array(rm.sphere_roles)
    roles = {}
    for ro in rm.sphere_roles: roles[ro] = roles.get(ro, 0) + 1
    links = {}
    for ln in rm.sphere_link_names: links[ln] = links.get(ln, 0) + 1
    out["models"][key] = dict(n_spheres=int(rm.n_spheres), inflation_m=infl, nq=int(rm.nq),
                              param_joint_names=list(rm.param_joint_names),
                              finger_joints=(fj_before if key == "before" else fj_now),
                              spheres_by_role=roles, spheres_by_link=links,
                              arm5_urdf_spheres=int(sum(1 for ln, ro in zip(rm.sphere_link_names, rm.sphere_roles)
                                                        if ln in ("link_left_arm_5", "link_right_arm_5") and ro != "cover")))
for cam in CAMS:
    f = scene.capture(cam)
    rend.update_scene(d, camera=cam); rgb = np.asarray(rend.render()).copy()
    npz[f"{cam}_rgb"] = rgb; npz[f"{cam}_bids"] = f.body_ids; npz[f"{cam}_depth"] = f.depth
    npz[f"{cam}_K"] = f.camera_intrinsics; npz[f"{cam}_T_base_cam"] = f.T_base_cam
    valid = (f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0)
    vs, us = np.nonzero(valid); K = f.camera_intrinsics; zz = f.depth[vs, us]
    pc = np.stack([(us - K[0, 2]) * zz / K[0, 0], (vs - K[1, 2]) * zz / K[1, 1], zz], 1)
    pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
    bids = f.body_ids[vs, us]
    ap = bids == apple_bid; rb = np.isin(bids, robot_bids); other = ~ap & ~rb
    out["cameras"][cam] = {"apple_px_visible": int(ap.sum()), "robot_px_visible": int(rb.sum()),
                           "other_px_visible": int(other.sum()), "valid_px": int(valid.sum())}
    for key, (rm, infl) in MODELS.items():
        cfg = PointCloudConfig(self_filter_inflation=infl)
        t0 = time.time()
        cloud = PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32))
        kept_cloud, stats = filter_robot_points(cloud, rm, q, cfg)
        kept = np.zeros(f.depth.shape, bool); kept[kept_cloud.uv[:, 1], kept_cloud.uv[:, 0]] = True
        passed = kept[vs, us]
        removed_img = np.zeros(f.depth.shape, bool); removed_img[vs[~passed], us[~passed]] = True
        npz[f"{cam}_{key}_removed"] = removed_img
        # which link's (inflated) spheres contain the erased apple px (overlaps counted per link)
        cen, rad = rm.sphere_centers_numeric(q); lk = np.array(rm.sphere_link_names)
        by_link = {}
        if (ap & ~passed).any():
            P = pb[ap & ~passed]
            for ln in dict.fromkeys(lk):
                sel = lk == ln
                n = int(robot_sphere_mask(P, cen[sel], rad[sel], infl).sum())
                if n: by_link[ln] = n
        leak_by_body = {}
        for b in np.unique(bids[rb & passed]):
            leak_by_body[bname(b)] = int(((bids == b) & passed).sum())
        out["cameras"][cam][key] = dict(
            apple_px_kept=int((ap & passed).sum()), apple_px_erased=int((ap & ~passed).sum()),
            robot_px_masked=int((rb & ~passed).sum()), robot_px_leaked=int((rb & passed).sum()),
            other_px_erased=int((other & ~passed).sum()),
            other_erased_by_body={bname(b): int(((bids == b) & ~passed).sum()) for b in np.unique(bids[other & ~passed])},
            robot_leaked_by_body=leak_by_body, apple_erased_by_link=by_link,
            filter_stats={k: v for k, v in stats.items() if not isinstance(v, np.ndarray)},
            wall_s=round(time.time() - t0, 3))
    print(cam, json.dumps({k: out["cameras"][cam][k] for k in ("apple_px_visible", "robot_px_visible")}),
          {k: {kk: out["cameras"][cam][k][kk] for kk in ("apple_px_kept", "apple_px_erased", "robot_px_leaked", "other_px_erased")} for k in MODELS}, flush=True)

# gripper geometry for the side view (world frame): left gripper MJCF mesh vertices + apple
def body_verts(name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); outv = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        outv.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(outv) if outv else np.zeros((0, 3))
for b in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2", "apple", "link_left_arm_5", "link_left_arm_6",
          "wrist_bracket_l", "d435i_body_l", "FT_SENSOR_L"):
    npz[f"mesh_{b}"] = body_verts(b)
    npz[f"xpos_{b}"] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy()
np.savez_compressed(OUTD / "raw.npz", **npz)
json.dump(out, open(OUTD / "raw.json", "w"), indent=1)
print("before", out["models"]["before"]["n_spheres"], "now", out["models"]["now"]["n_spheres"], "fj_now", fj_now)
