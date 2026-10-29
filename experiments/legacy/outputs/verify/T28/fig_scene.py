"""T28 Phase 1 real-scene figure: T16 t=96 (server chunk 12 <-> local seq 13).
MuJoCo re-posed with the planning-row qpos. Overlays in world coordinates: recorded constraint spheres of
ee_left / ee_finger_l1 / ee_finger_l2 (server record sphere_centres, base->world), actual MuJoCo collision meshes
of the left fingers/palm, apple mesh, worst-row query point, and the zed_left + wrist_cam_l renders with the
sphere centres projected. Also the P1-3b self-filter pixel mask at T14 t=96 wrist_cam_l (inflation 0 vs 0.05)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-3'))
from AG3S.ag3s.experiments.common import figstyle as fs
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import filter_robot_points
from p13a_spheres import body_mesh_world, XML
fs.use_korean(); C = fs.CATEGORICAL
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28'))
RUN = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16'))
K_CHUNK = 12
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
rows = {json.loads(l)["seq"]: json.loads(l) for l in open(RUN / "execute_ep1807/frames.jsonl") if '"planning"' in l}
pr = rows[K_CHUNK + 1]
d.qpos[:] = pr["qpos"]; mujoco.mj_forward(m, d)
z = np.load(RUN / f"server_constraints/run_0000/chunk_{K_CHUNK:05d}.npz", allow_pickle=True)
s = json.loads(str(z["summary_json"])); names = np.array(s["sphere_link_names"])
Tb = scene.body_pose("base")
Cw = z["sphere_centres"].astype(float) @ Tb[:3, :3].T + Tb[:3, 3]; R = z["sphere_radii"].astype(float)
mv = pr["max_violation_pair"]; wp = Tb[:3, :3] @ np.asarray(mv["point_m"]) + Tb[:3, 3]
tc = Tb[:3, :3] @ np.asarray(z["target_centroid"]) + Tb[:3, 3]
apple = body_mesh_world(m, d, "apple", True)
meshes = {"ee_finger_l1": body_mesh_world(m, d, "ee_finger_l1", True), "ee_finger_l2": body_mesh_world(m, d, "ee_finger_l2", True),
          "EE_BODY_L": body_mesh_world(m, d, "EE_BODY_L", True)}
col = {"ee_left": C[2], "ee_finger_l1": C[0], "ee_finger_l2": C[4], "EE_BODY_L": C[2], "apple": C[1]}
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3)
side = {}
for j, (ax_i, (a, b), lab) in enumerate(((gs[0, 0], (0, 1), "top view x-y (world)"), (gs[0, 1], (0, 2), "side view x-z (world)"))):
    ax = fig.add_subplot(ax_i)
    ax.scatter(apple[:, a], apple[:, b], s=2, color=col["apple"], alpha=0.35, label="apple mesh (GT)")
    for n, V in meshes.items():
        ax.scatter(V[::3, a], V[::3, b], s=1, color=col[n], alpha=0.25, label=f"{n} mesh (MuJoCo, open)")
    for L in ("ee_left", "ee_finger_l1", "ee_finger_l2"):
        for c, r in zip(Cw[names == L], R[names == L]):
            ax.add_patch(Circle((c[a], c[b]), r, fill=False, lw=1.0, ec=col[L] if L != "ee_left" else fs.INK_2))
        ax.plot([], [], "o", mfc="none", color=col[L] if L != "ee_left" else fs.INK_2, label=f"{L} constraint spheres (record)")
    ax.plot(wp[a], wp[b], "x", color=C[7], ms=10, mew=2, label=f"worst row {mv['link']} {1000*mv['clearance_m']:.1f} mm")
    ctr = np.vstack([apple, *meshes.values()]).mean(0)
    ax.set_xlim(ctr[a] - 0.12, ctr[a] + 0.12); ax.set_ylim(ctr[b] - 0.12, ctr[b] + 0.12); ax.set_aspect("equal")
    ax.set_title(f"T16 t=96 (chunk 12, HOLD violated): {lab}", fontsize=10); ax.set_xlabel("xyz"[a] + " (m)"); ax.set_ylabel("xyz"[b] + " (m)")
    if j == 0: ax.legend(fontsize=6.5, loc="lower left", frameon=True, markerscale=1.5)
    fs.style_axes(fig, ax)
# camera renders
ren = None
for k, cam in enumerate(("zed_left", "wrist_cam_l")):
    f = scene.capture(cam); H, W = f.depth.shape
    ren = mujoco.Renderer(m, H, W); ren.update_scene(d, camera=cam); img = ren.render(); ren.close()
    Tbc = f.T_base_cam; K = f.camera_intrinsics
    Cb = z["sphere_centres"].astype(float); pc = (Cb - Tbc[:3, 3]) @ Tbc[:3, :3]
    ax = fig.add_subplot(gs[0 if k == 0 else 1, 2] if k == 0 else gs[1, 2])
    ax.imshow(img)
    for L in ("ee_left", "ee_finger_l1", "ee_finger_l2"):
        sel = (names == L) & (pc[:, 2] > 0.01)
        u = K[0, 0] * pc[sel, 0] / pc[sel, 2] + K[0, 2]; v = K[1, 1] * pc[sel, 1] / pc[sel, 2] + K[1, 2]
        rpx = K[0, 0] * R[sel] / pc[sel, 2]
        for uu, vv, rr in zip(u, v, rpx):
            ax.add_patch(Circle((uu, vv), rr, fill=False, lw=0.8, ec=col[L] if L != "ee_left" else "white"))
    pw = (np.asarray(mv["point_m"]) - Tbc[:3, 3]) @ Tbc[:3, :3]
    ax.plot(K[0, 0] * pw[0] / pw[2] + K[0, 2], K[1, 1] * pw[1] / pw[2] + K[1, 2], "x", color=C[7], ms=12, mew=2.5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(f"{cam}: render + recorded constraint spheres + worst-row point (x)", fontsize=9); ax.axis("off")
# P1-3b mask panel: T14 t=96 wrist_cam_l
rows14 = {json.loads(l)["seq"]: json.loads(l) for l in open(_ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl')) if '"planning"' in l}
fr = rows14[13]; d.qpos[:] = fr["qpos"]; mujoco.mj_forward(m, d)
robot_f = build_robot_model(TransportScene.attach(m, mujoco.MjData(m)))
f = scene.capture("wrist_cam_l"); q = scene.robot_state()
valid = (f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0); vs, us = np.nonzero(valid); K = f.camera_intrinsics; zz = f.depth[vs, us]
pcl = np.stack([(us - K[0, 2]) * zz / K[0, 0], (vs - K[1, 2]) * zz / K[1, 1], zz], 1)
pb = (f.T_base_cam[:3, :3] @ pcl.T).T + f.T_base_cam[:3, 3]; bids = f.body_ids[vs, us]
apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")]
isrob = np.isin(bids, [b for b in range(m.nbody) if m.body_rootid[b] == root])
for j, infl in enumerate((0.0, 0.05)):
    kept_cloud, _ = filter_robot_points(PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32)), robot_f, q, PointCloudConfig(self_filter_inflation=infl))
    kept = np.zeros(f.depth.shape, bool); kept[kept_cloud.uv[:, 1], kept_cloud.uv[:, 0]] = True
    img = np.full(f.depth.shape + (3,), 0.93)
    other = valid & ~np.isin(f.body_ids, [apple_bid]) & ~np.isin(f.body_ids, [b for b in range(m.nbody) if m.body_rootid[b] == root])
    img[other & kept] = (0.75, 0.75, 0.72); img[other & ~kept] = (0.45, 0.45, 0.45)
    ap = f.body_ids == apple_bid; rb = np.isin(f.body_ids, [b for b in range(m.nbody) if m.body_rootid[b] == root])
    img[ap & kept] = (0.11, 0.69, 0.48); img[ap & ~kept] = (0.89, 0.29, 0.28)
    img[rb & kept] = (0.91, 0.48, 0.64); img[rb & ~kept] = (0.16, 0.47, 0.84)
    ax = fig.add_subplot(gs[1, j])
    ax.imshow(img); ax.axis("off")
    na, nk = int(ap.sum()), int((ap & kept).sum()); nr = int((rb & kept).sum())
    ax.set_title(f"T14 t=96 wrist_cam_l, self-filter inflation {infl} m\napple kept {nk}/{na} (green kept / red removed), robot leaked {nr} (pink) / removed (blue)", fontsize=9)
    side[f"t14_t96_wrist_cam_l_inflation_{infl}"] = {"apple_visible": na, "apple_kept": nk, "robot_leaked": nr}
fig.tight_layout()
fig.savefig(F / "t28-p1-scene.png", dpi=120); plt.close(fig)
side.update({"t16_t96_chunk12": {"worst_row": {k: mv[k] for k in ("link", "clearance_m", "step", "query")}, "worst_point_world": wp.tolist(),
             "target_centroid_world": tc.tolist(), "target_to_apple_centre_mm": float(1000 * np.linalg.norm(tc - np.asarray(pr["object_poses"]["apple"]["pos"]))),
             "recorded_l1_l2_sphere_centroid_sep_mm": float(1000 * np.linalg.norm(Cw[names == "ee_finger_l1"].mean(0) - Cw[names == "ee_finger_l2"].mean(0))),
             "mujoco_l1_l2_mesh_centroid_sep_mm": float(1000 * np.linalg.norm(meshes["ee_finger_l1"].mean(0) - meshes["ee_finger_l2"].mean(0)))}})
(F / "t28-p1-scene.json").write_text(json.dumps(side, indent=1))
print(json.dumps(side, indent=1))
