"""rm-model-compare figure: BEFORE (top) / NOW (bottom) self-filter sphere model on the same frame.
Reads raw.npz / raw.json / crosscheck_head.json (compute.py, dump_head.py). Writes the png + json sidecar."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib, subprocess
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from AG3S.ag3s.experiments.common import figstyle

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/rm-model-compare'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/ppt-0930/rm-model-before-after.png'))
z = np.load(D / "raw.npz"); raw = json.load(open(D / "raw.json")); xc = json.load(open(D / "crosscheck_head.json"))
ko = figstyle.use_korean(); assert ko, "no CJK font"
C = figstyle.CATEGORICAL
COL_SPH, COL_MASK, COL_KEEP, COL_ERASE, COL_OTHER, COL_LEAK, COL_ARM5 = C[0], C[6], C[2], C[7], C[4], C[3], C[1]
APPLE = 65  # MuJoCo body id not needed: recompute from bids via apple px count check below
CROPS = {"wrist_cam_l": (0, 640, 150, 480), "zed_left": (110, 370, 200, 335)}   # u0,u1,v0,v1
CAM_TITLE = {"wrist_cam_l": "왼손목 카메라 (wrist_cam_l)", "zed_left": "머리 카메라 (zed_left), 왼손 주변 확대"}
MODELS = ("before", "now")
ROWLAB = {"before": "이전: URDF capsule + 여유 50 mm, 손가락 닫힘 고정",
          "now": "현재: mesh 표면에 맞춘 덮개, 여유 0, 측정 손가락 개도"}
T_wb = np.asarray(raw["T_world_base"])

import mujoco
m = mujoco.MjModel.from_xml_path(raw["xml"])
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
apple_bid = bid("apple")
root = m.body_rootid[bid("base")]
is_robot_body = np.array([m.body_rootid[b] == root for b in range(m.nbody)])


def sphere_outline(c, R, K, n=48):
    """Exact image outline of a sphere (camera frame centre c, radius R): the tangent-cone circle."""
    L = np.linalg.norm(c)
    if L <= R * 1.0001:
        return None
    w = c / L
    cc = c * (1 - R * R / L / L); rho = R * np.sqrt(L * L - R * R) / L
    a = np.cross(w, [0, 0, 1.0]); a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0, 0])
    b = np.cross(w, a)
    th = np.linspace(0, 2 * np.pi, n)
    P = cc[None] + rho * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
    if (P[:, 2] <= 1e-3).any():
        return None
    return np.stack([K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]], 1)


def silhouette(cen, R, K, u0, u1, v0, v1):
    us, vs = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    dirs = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    hit = np.zeros(len(dirs), bool)
    for c, r in zip(cen, R):
        t = dirs @ c; L2 = c @ c
        hit |= ((L2 - t * t) < r * r) & ((t > 0) | (L2 < r * r))
    return hit.reshape(us.shape)


side = {"panels": {}}
fig = plt.figure(figsize=(13.2, 6.2)); fig.patch.set_facecolor(figstyle.SURFACE)
# layout (figure fraction)
X0, W_IMG, GAP, W_SIDE = 0.012, 0.345, 0.012, 0.255
ROW_Y = {"before": 0.525, "now": 0.088}; H_IMG = 0.315
for key in MODELS:
    info = raw["models"][key]; infl = info["inflation_m"]
    cen_b, rad = z[f"{key}_centers"], z[f"{key}_radii"]; Reff = rad + infl
    links, roles = z[f"{key}_links"], z[f"{key}_roles"]
    arm5 = np.isin(links, ["link_left_arm_5", "link_right_arm_5"]) & (roles != "self_filter_cover")
    y = ROW_Y[key]
    t1 = fig.text(X0, y + H_IMG + 0.078, ROWLAB[key], fontsize=12.5, color=figstyle.INK, ha="left", va="bottom", fontweight="bold")
    bb = t1.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.012, y + H_IMG + 0.078, f"(구 {info['n_spheres']}개 · 여유 {1000 * infl:.0f} mm)", fontsize=11.5,
             color=figstyle.INK, ha="left", va="bottom")
    for j, cam in enumerate(("wrist_cam_l", "zed_left")):
        ax = fig.add_axes([X0 + j * (W_IMG + GAP), y, W_IMG, H_IMG])
        u0, u1, v0, v1 = CROPS[cam]
        rgb = z[f"{cam}_rgb"].astype(float) / 255.0; bids = z[f"{cam}_bids"]; K = z[f"{cam}_K"]
        removed = z[f"{cam}_{key}_removed"]; depth = z[f"{cam}_depth"]
        valid = (bids >= 0) & np.isfinite(depth) & (depth > 0)
        ap = (bids == apple_bid) & valid
        rb = np.zeros_like(bids, bool); rb[valid] = is_robot_body[bids[valid]]
        other = valid & ~ap & ~rb
        img = rgb[v0:v1, u0:u1].copy()
        Tcb = np.linalg.inv(z[f"{cam}_T_base_cam"])
        cen_c = cen_b @ Tcb[:3, :3].T + Tcb[:3, 3]
        sil = silhouette(cen_c, Reff, K, u0, u1, v0, v1)

        def tint(mask, col, a):
            cc = np.asarray(matplotlib.colors.to_rgb(col)); mm = mask[v0:v1, u0:u1] if mask.shape != sil.shape else mask
            img[mm] = (1 - a) * img[mm] + a * cc
        tint(sil, COL_SPH, 0.16)
        tint(rb & removed, COL_MASK, 0.45)
        tint(rb & ~removed, COL_LEAK, 0.95)
        tint(other & removed, COL_OTHER, 0.85)
        tint(ap & ~removed, COL_KEEP, 0.92)
        tint(ap & removed, COL_ERASE, 0.92)
        ax.imshow(img, extent=(u0, u1, v1, v0), interpolation="nearest")
        n_out = 0
        for i, (c, r) in enumerate(zip(cen_c, Reff)):
            P = sphere_outline(c, r, K)
            if P is None:
                continue
            if P[:, 0].max() < u0 or P[:, 0].min() > u1 or P[:, 1].max() < v0 or P[:, 1].min() > v1:
                continue
            n_out += 1
            ax.plot(P[:, 0], P[:, 1], color=COL_ARM5 if arm5[i] else COL_SPH, lw=0.9 if arm5[i] else 0.35,
                    alpha=0.9 if arm5[i] else 0.45)
        ax.set_xlim(u0, u1); ax.set_ylim(v1, v0); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor(figstyle.GRID_INK)
        n = raw["cameras"][cam]; r_ = n[key]
        ax.set_title(f"{CAM_TITLE[cam]}\n사과 {n['apple_px_visible']:,} px 중 지워짐 {r_['apple_px_erased']:,} · 남음 {r_['apple_px_kept']:,}   "
                     f"로봇 누수 {r_['robot_px_leaked']:,} / {n['robot_px_visible']:,} px   다른 물체 지워짐 {r_['other_px_erased']:,} px",
                     fontsize=8.6, color=figstyle.INK, pad=3)
        side["panels"][f"{key}_{cam}"] = dict(crop_uv=[u0, u1, v0, v1], apple_px_visible=n["apple_px_visible"],
                                              apple_px_erased=r_["apple_px_erased"], apple_px_kept=r_["apple_px_kept"],
                                              robot_px_visible=n["robot_px_visible"], robot_px_masked=r_["robot_px_masked"],
                                              robot_px_leaked=r_["robot_px_leaked"], other_px_erased=r_["other_px_erased"],
                                              other_erased_by_body=r_["other_erased_by_body"],
                                              apple_erased_by_link_overlapping=r_["apple_erased_by_link"],
                                              spheres_outlined_in_crop=n_out,
                                              sphere_silhouette_px_in_crop=int(sil.sum()))
    # ---- side view (orthographic): plane of finger closing axis u and approach axis a
    ax = fig.add_axes([X0 + 2 * (W_IMG + GAP) + 0.012, y, W_SIDE - 0.02, H_IMG])
    f1, f2, palm, apple = z["xpos_ee_finger_l1"], z["xpos_ee_finger_l2"], z["xpos_EE_BODY_L"], np.asarray(raw["apple_pos_world"])
    uax = (f2 - f1) / np.linalg.norm(f2 - f1)
    a = 0.5 * (f1 + f2) - palm; a -= (a @ uax) * uax; a /= np.linalg.norm(a)
    O = apple
    proj = lambda P: np.stack([(P - O) @ uax, -((P - O) @ a)], -1) * 1000.0   # mm; approach axis points down
    cw = cen_b @ T_wb[:3, :3].T + T_wb[:3, 3]
    near = np.linalg.norm(cw - O, axis=1) < 0.22
    for bn in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2", "wrist_bracket_l", "d435i_body_l", "FT_SENSOR_L", "link_left_arm_6", "link_left_arm_5"):
        V = z[f"mesh_{bn}"]
        if len(V):
            p = proj(V[::2]); ax.scatter(p[:, 0], p[:, 1], s=1.0, c="#5e5e5a", lw=0, alpha=0.55, rasterized=True, zorder=3)
    for i in np.nonzero(near)[0]:
        p = proj(cw[i:i + 1])[0]
        ax.add_patch(Circle(p, 1000 * Reff[i], fill=True, fc=COL_ARM5 if arm5[i] else COL_SPH, alpha=0.05, lw=0))
        ax.add_patch(Circle(p, 1000 * Reff[i], fill=False, ec=COL_ARM5 if arm5[i] else COL_SPH,
                            lw=1.0 if arm5[i] else 0.45, alpha=0.9 if arm5[i] else 0.6))
    erased_all = raw["cameras"]["wrist_cam_l"][key]["apple_px_erased"] == raw["cameras"]["wrist_cam_l"]["apple_px_visible"]
    ax.add_patch(Circle((0, 0), 33.3, fc=COL_ERASE if erased_all else COL_KEEP, ec="none", alpha=0.85, zorder=5))
    ax.set_xlim(-135, 135); ax.set_ylim(-55, 150); ax.set_aspect("equal")
    figstyle.style_axes(fig, ax)
    ax.tick_params(labelsize=7); ax.set_xlabel("손가락 벌림 축 (mm)", fontsize=8, labelpad=1)
    ax.set_ylabel("접근 축 (mm)", fontsize=8, labelpad=1)
    ax.set_title(f"옆에서 본 왼손 · 사과 (정사영)\n사과 0.22 m 안 구 {int(near.sum())}개, 원 = 반지름 + 여유", fontsize=8.6, color=figstyle.INK, pad=3)
    side["panels"][f"{key}_side_view"] = dict(spheres_within_0p22m_of_apple=int(near.sum()),
                                              arm5_urdf_spheres_within=int((near & arm5).sum()),
                                              finger_gap_mm_mujoco=float(1000 * np.linalg.norm(f2 - f1)))
fig.text(0.5, 0.992, "로봇 마스크(자기 필터) 구 모델, 이전 vs 현재: 같은 프레임, 잡기 직전 (폐루프 16D 기록 ep1807, t=96)",
         ha="center", va="top", fontsize=13.5, color=figstyle.INK)
H = [Patch(fc=COL_SPH, alpha=0.35, label="구 실루엣 (반지름 + 여유)"), Line2D([], [], color=COL_ARM5, lw=1.4, label="link_left_arm_5 URDF capsule 구"),
     Patch(fc=COL_MASK, alpha=0.6, label="로봇 px, 마스크됨"), Patch(fc=COL_LEAK, label="로봇 px, 새어 나감"),
     Patch(fc=COL_KEEP, label="사과 px, 남음"), Patch(fc=COL_ERASE, label="사과 px, 지워짐"), Patch(fc=COL_OTHER, label="다른 물체 px, 지워짐"),
     Line2D([], [], marker="o", ls="", color="#5e5e5a", ms=3, label="실제 손 mesh (MuJoCo)")]
fig.legend(handles=H, loc="lower center", ncol=8, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.2)
fig.savefig(OUT, dpi=200, facecolor=figstyle.SURFACE)

rev = subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
dirty = subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "status", "--porcelain", "--", "ag3s/experiments/reports/grounding_report.py",
                        "ag3s/experiments/sources/mujoco_source.py", "ag3s/robot_models/urdf_sphere_chain.py", "ag3s/stages/robot_filter.py"],
                       capture_output=True, text=True, timeout=30).stdout.rstrip().splitlines()
side.update({
    "provenance": {"record": raw["record"], "planning_seq": raw["seq"], "t_step": raw["t_step"], "xml": raw["xml"],
                   "pose": "d.qpos = record row qpos, qvel = 0, mj_forward (= policy_record.pose_scene)",
                   "benchmark_git_head": rev, "benchmark_code_used": "working tree (uncommitted changes in: " + ", ".join(l[3:] for l in dirty) + ")",
                   "interpreter": _ag3s_workspace_path('legacy/.venv-openpi-live/bin/python (MUJOCO_GL=osmesa)'),
                   "decision": "stages/robot_filter.filter_robot_points, target guard off; every valid depth px back-projected, no stride",
                   "gt": "TransportScene.capture(cam).body_ids; robot = bodies with rootid of 'base'",
                   "drivers": [str(D / "compute.py"), str(D / "dump_head.py"), str(D / "plot.py")]},
    "models": {
        "before": {"constructor": "UrdfSphereChain(parse_urdf(RBY1_URDF) [all URDF capsules incl. link_*_arm_5], extra_capsules=gap_filling_capsules(model), fixed_joint_values=head@fresh MjData)",
                   "finger_joints": raw["models"]["before"]["finger_joints"], "finger_joints_source": "construction default 0.0 (closed)",
                   "inflation_m": 0.05, "n_spheres": raw["models"]["before"]["n_spheres"],
                   "spheres_by_role": raw["models"]["before"]["spheres_by_role"],
                   "arm5_urdf_spheres": int((np.isin(z["before_links"], ["link_left_arm_5", "link_right_arm_5"]) & (z["before_roles"] != "self_filter_cover")).sum()),
                   "crosscheck_vs_HEAD_tree_build_robot_model": xc},
        "now": {"constructor": "grounding_report.build_robot_model(TransportScene.attach(model, fresh MjData)) + mujoco_source.set_finger_joints_from_scene(scene@t96, robot)",
                "finger_joints": raw["models"]["now"]["finger_joints"], "finger_joints_source": "FingerJointMap(openings_from_mujoco) at the recorded pose",
                "inflation_m": 0.0, "n_spheres": raw["models"]["now"]["n_spheres"],
                "spheres_by_role": raw["models"]["now"]["spheres_by_role"],
                "arm5_urdf_spheres": int((np.isin(z["now_links"], ["link_left_arm_5", "link_right_arm_5"]) & (z["now_roles"] != "self_filter_cover")).sum())}},
    "cameras_full_frame": raw["cameras"],
    "check_vs_earlier_figure": {"figure": "figures/t14/t14-selffilter-erases-target.json",
                                "wrist_cam_l_before_apple_visible_erased": [raw["cameras"]["wrist_cam_l"]["apple_px_visible"], raw["cameras"]["wrist_cam_l"]["before"]["apple_px_erased"]],
                                "zed_left_before_apple_visible_erased": [raw["cameras"]["zed_left"]["apple_px_visible"], raw["cameras"]["zed_left"]["before"]["apple_px_erased"]],
                                "earlier_values": {"wrist_cam_l": [2497, 2497], "zed_left": [378, 378], "self_filter_spheres": 218}},
    "drawing": "sphere outline = exact tangent-cone silhouette of radius (r + inflation); fill = union of ray-sphere hits; side view orthographic in the plane (finger closing axis, approach axis) through the apple centre, spheres with centre < 0.22 m of the apple",
})
json.dump(side, open(OUT.with_suffix(".json"), "w"), indent=1, ensure_ascii=False)
print("wrote", OUT)
