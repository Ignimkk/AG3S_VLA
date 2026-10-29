from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T22'))
import replay_t22 as R
from AG3S.ag3s.experiments.common import figstyle as fs
from AG3S.ag3s.constraints.attached import parent_reach
KO = fs.use_korean()
OUT = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t22/t22-grasp-scene')
m = R._m; W, H = 640, 480
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
ren = mujoco.Renderer(m, H, W)


def project(p, cam):
    pos = np.array(cam.pos); f = np.array(cam.forward); up = np.array(cam.up); r = np.cross(f, up)
    v = np.asarray(p) - pos; z = v @ f
    th = np.tan(np.deg2rad(m.vis.global_.fovy) / 2)
    xn, yn = (v @ r) / z, (v @ up) / z
    u = (xn / (th * W / H) + 1) / 2 * W
    vv = (1 - yn / th) / 2 * H
    return u, vv


def scene(run, seq, views, centroid_key=None):
    P, C, S = R.load_run(run)
    q = np.asarray(P[seq]["qpos"], np.float64)
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    Tb = np.eye(4); Tb[:3, :3] = d.xmat[bid].reshape(3, 3); Tb[:3, 3] = d.xpos[bid]
    q20 = q[R.Q20]
    l1_urdf_w = (Tb @ np.r_[R.RM.link_pose(q20, "ee_finger_l1")[:3, 3], 1])[:3]
    bp = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
    apple = np.asarray(P[seq]["object_poses"]["apple"]["pos"])
    srv = S.get(seq - 1, {}).get("centroid")
    srv_w = None if srv is None else (Tb @ np.r_[srv, 1])[:3]
    pts = {"URDF ee_finger_l1 원점": l1_urdf_w, "MJCF ee_finger_l1 body": bp("ee_finger_l1"), "MJCF ee_finger_l2 body": bp("ee_finger_l2"), "사과 (참값)": apple}
    if srv_w is not None and centroid_key:
        pts["서버 target_centroid (직전 프레임)"] = srv_w
    imgs = []
    for (az, el, dist, look) in views:
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look if look is not None else apple; cam.azimuth, cam.elevation, cam.distance = az, el, dist
        ren.update_scene(d, camera=cam); img = ren.render().copy()
        gl = ren.scene.camera[0]
        proj = {k: project(v, gl) for k, v in pts.items()}
        # reach circle (0.12 m) around URDF l1 origin, drawn in the view plane
        f = np.array(gl.forward); up = np.array(gl.up); r = np.cross(f, up)
        circ = [project(l1_urdf_w + 0.12 * (np.cos(a) * r + np.sin(a) * up), gl) for a in np.linspace(0, 2 * np.pi, 120)]
        imgs.append((img, proj, circ, (az, el, dist)))
    info = {k: np.round(v, 4).tolist() for k, v in pts.items()}
    info["reach_urdf_l1_apple_mm"] = round(parent_reach(R.RM, q20, "ee_finger_l1", (np.linalg.inv(Tb) @ np.r_[apple, 1])[:3]) * 1000, 2)
    if srv is not None: info["reach_urdf_l1_server_prev_mm"] = round(parent_reach(R.RM, q20, "ee_finger_l1", srv) * 1000, 2)
    info["opening"] = round(abs(q[R.GL]) / 0.045, 4); info["t_step"] = int(P[seq]["t_step"]); info["seq"] = seq
    return imgs, info


panels = [("t17", 16, [(180, -89.0, 0.55, None), (-90, -8.0, 0.6, None)], False, "T17 t=120 (attach 판정 요청)"),
          ("t14", 26, [(180, -89.0, 1.9, np.array([0.5, 0.0, 0.85])), (-90, -8.0, 1.9, np.array([0.5, 0.0, 0.85]))], True, "T14 t=200 (seq 26, 636 mm phantom 자리)")]
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5)); fs.style_axes(fig, axes)
MK = {"URDF ee_finger_l1 원점": ("o", fs.CATEGORICAL[1]), "MJCF ee_finger_l1 body": ("^", fs.CATEGORICAL[3]), "MJCF ee_finger_l2 body": ("v", fs.CATEGORICAL[4]),
      "사과 (참값)": ("*", fs.CATEGORICAL[7]), "서버 target_centroid (직전 프레임)": ("X", fs.CATEGORICAL[6])}
side = {}
for row, (run, seq, views, withsrv, title) in enumerate(panels):
    imgs, info = scene(run, seq, views, withsrv); side[f"{run}_seq{seq}"] = info
    for col, (img, proj, circ, cv) in enumerate(imgs):
        ax = axes[row, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        cx, cy = zip(*circ); ax.plot(cx, cy, color=fs.CATEGORICAL[2], lw=1.5, ls="--", label="reach 0.12 m (URDF l1 원점 중심, 화면 평면)")
        for k, (u, v) in proj.items():
            mk, c = MK[k]; ax.plot(u, v, mk, color=c, ms=11 if mk == "*" else 8, mec="white", mew=0.8, label=k)
        if "서버 target_centroid (직전 프레임)" in proj:
            a, b = proj["URDF ee_finger_l1 원점"], proj["서버 target_centroid (직전 프레임)"]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=fs.CATEGORICAL[6], lw=1)
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 8, f"{info['reach_urdf_l1_server_prev_mm']:.0f} mm", color="white", fontsize=9,
                    bbox=dict(fc=fs.CATEGORICAL[6], ec="none", pad=1.5))
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        vname = ("top view (위에서)" if col == 0 else "side view (+y 쪽에서)")
        sub = f"개도 {info['opening']:.3f} · 원점↔사과 {info['reach_urdf_l1_apple_mm']:.1f} mm"
        if withsrv: sub += f" · 원점↔서버 centroid {info['reach_urdf_l1_server_prev_mm']:.1f} mm"
        ax.set_title(f"{title} — {vname}\n{sub}", fontsize=9, color=fs.INK, loc="left")
        if row == 0 and col == 0: ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
        if row == 1 and col == 0: ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
fig.suptitle("T22 실제 씬: ee_finger_l1 원점(URDF FK) 과 reach 0.12 m 원 — 위: T17 파지 순간, 아래: T14 옛 attach 순간", fontsize=11, color=fs.INK, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=140)
json.dump(side, open(OUT + ".json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(side, ensure_ascii=False, indent=0))
