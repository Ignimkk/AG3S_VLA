"""T8f — T8e 가 고른 설정을 실제 씬 위에 겹쳐 본다. 지금 모델(A) 대 확정 설정(B).

T8d 와 **같은 pose·같은 방법**이고 B 만 다르다. T8d 의 B 는 `arm_5` 를 자르고 손가락만 줄인
판이었고, 여기 B 는 **전 link 이 함께 얇아진다** (전역 `capsule_radius_scale 0.2`) 그리고
**구가 훨씬 촘촘하다** (`sphere_spacing 0.4`, `max_spheres_per_capsule 32`).
수치 기대값은 T8e s2 의 `spacing 0.4 x scale 0.2` 칸이다.

A. 지금  : `ARM_LINKS` = link_{l,r}_arm_{0..6} + ee_finger_* , 구 120 개. `link_*_arm_5` 는
           r 81.25 mm 구 5 개 (URDF capsule r 75 · L 250, 축 z ∈ [-225, +25] mm).
B. 확정 : `sphere_spacing 0.4` · 전역 `capsule_radius_scale 0.2` ·
           `max_spheres_per_capsule 32` · `link_*_arm_5` 를 z_cut -100 mm 에서 자르고 ·
           손바닥 `ee_left`/`ee_right` 를 link_filter 에 **추가**.
           자르기는 근사다: capsule 길이를 줄이는 flag 가 없어 구를 빼는 것으로 흉내냈고,
           실제로 짧은 capsule 을 만들면 구 개수가 줄어 반지름도 달라진다.

**그리기 (2026-09-26 재렌더)**: 처음 판은 구가 반투명이고 로봇 mesh 가 불투명이라 B 의
`link_*_arm_5` 구(r 20.44 mm)가 **mesh 안에 파묻혀 안 보였다**. 그래서 뒤집었다 —
mesh 를 옅게(alpha MESH_ALPHA), 구를 진하게. 그리고 아래 줄에 **구만** 그린 칸을 더하고
로봇 실루엣을 윤곽선으로 얹었다. 구가 로봇 안인지 밖인지가 A 와 B 의 차이 전부다.

빨간 점 = `link_*_arm_5` capsule 축 위에서 그 모델의 **어떤 구에도 안 덮이는** 표본점.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import os
import pathlib
import sys

import numpy as np

#: A1 이 `benchmark/**` 를 고치는 중이라 작업 트리가 일시적으로 깨질 수 있다. `AG3S_REPO` 로
#: HEAD pristine worktree 를 가리키면 거기서 돈다. 구 모델이 T8a 와 같은지는 아래에서 확인한다.
REPO = os.environ.get("AG3S_REPO", _ag3s_workspace_path(''))
sys.path.insert(0, REPO)
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SHADOW, load_planning)

OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8f'))
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8f'))
CHUNK = 9
EXPECTED_STEP = 7
Z_CUT_M = -0.10
GLOBAL_SCALE = 0.2
SPHERE_SPACING = 0.4
MAX_SPHERES = 32
ARM5 = ("link_left_arm_5", "link_right_arm_5")
PALM = ("ee_left", "ee_right")
SCENE_ROOTS = ("office", "table", "crate", "apple", "banana", "orange", "pear", "shelf")
VIEWS = [("front", 90.0, -12.0, "t8f-front.png"),
         ("oblique-left", 140.0, -18.0, "t8f-oblique-left.png"),
         ("left", 180.0, -12.0, "t8f-left.png")]
W, H_PX = 800, 720
DIST = 1.75
MESH_ALPHA = 0.22        # 로봇 mesh — 옅게
SPHERE_A_TOP = 0.62      # 겹친 씬 칸의 구
SPHERE_A_ONLY = 0.92     # 구만 그린 칸의 구


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    print("repo root for benchmark.*:", REPO)

    rob_a = build_constraint_robot_model(scene, link_filter=tuple(ARM_LINKS))
    names_a = list(resolve_link_names(rob_a, rob_a.n_spheres))
    rob_b = build_constraint_robot_model(
        scene, link_filter=tuple(ARM_LINKS) + PALM,
        sphere_options={"sphere_spacing": SPHERE_SPACING,
                        "max_spheres_per_capsule": MAX_SPHERES,
                        "capsule_radius_scale": GLOBAL_SCALE})
    names_b = list(resolve_link_names(rob_b, rob_b.n_spheres))
    z_local_b = {i: float(c[2]) for i, (ln, c, _) in enumerate(rob_b._local_spheres)
                 if ln in ARM5}
    keep_b = np.asarray([not (names_b[i] in ARM5 and z_local_b[i] < Z_CUT_M)
                         for i in range(len(names_b))], bool)

    # --- pose: pre-grasp frame -----------------------------------------------------------
    recs = load_planning(SHADOW)
    rec = recs[CHUNK]
    chunk = np.asarray(rec["actions_reference"], np.float64)
    t8a = np.load(_ag3s_workspace_path('outputs/verify/T8a/m0_clearance_tensor.npz'), allow_pickle=True)
    tn = [str(x) for x in t8a["link_names"]]
    ai = [str(b) for b in t8a["bodies"]].index("apple")
    fing = np.asarray([n.startswith("ee_finger_l") for n in tn], bool)
    ap = t8a["clearance_m"][CHUNK, :8][:, fing, ai] * 1000.0
    STEP = int(np.argmin(ap.min(axis=1)))
    fingertip_mm = float(ap.min())
    assert STEP == EXPECTED_STEP, f"pose drifted from T8d: step {STEP} != {EXPECTED_STEP}"
    qpos = np.asarray(rec["qpos"], np.float64)
    d.qpos[: qpos.size] = qpos
    for n_, ci in ACT_COL.items():
        d.qpos[QADR[n_]] = float(chunk[STEP, ci])
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    q = np.asarray([qpos[QADR[n_]] if n_.startswith("torso_") else float(chunk[STEP, ACT_COL[n_]])
                    for n_ in JOINT_ORDER], np.float64)
    print(f"pose: chunk {CHUNK} seq {rec['seq']} t_step {rec['t_step']} plan step {STEP} "
          f"(fingertip-to-apple {fingertip_mm:+.2f} mm) - same as T8d")
    print(f"A: {rob_a.n_spheres} spheres · B: {rob_b.n_spheres} in the model, "
          f"{int(keep_b.sum())} after the cut")

    _same_names = names_a == tn
    _rad_probe = np.asarray(rob_a.sphere_centers_numeric(np.zeros(rob_a.nq))[1],
                            np.float64).reshape(-1) * 1000.0
    _rad_diff = float(np.abs(_rad_probe - np.asarray(t8a["radii_m"], np.float64) * 1000.0).max())
    print(f"model A identical to the T8a tensor model: names {_same_names}, "
          f"max|radius diff| {_rad_diff:.3e} mm")

    cen_a, rad_a = rob_a.sphere_centers_numeric(q)
    cen_a = np.asarray(cen_a, np.float64).reshape(-1, 3)
    rad_a = np.asarray(rad_a, np.float64).reshape(-1)
    cen_b, rad_b = rob_b.sphere_centers_numeric(q)
    cen_b = np.asarray(cen_b, np.float64).reshape(-1, 3)
    rad_b = np.asarray(rad_b, np.float64).reshape(-1)

    # --- link_*_arm_5 축 표본점과 덮개 ------------------------------------------------------
    a5 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_left_arm_5")
    P5 = np.asarray(d.xpos[a5], np.float64)
    M5 = np.asarray(d.xmat[a5], np.float64).reshape(3, 3)
    zs = np.arange(-0.300, 0.1001, 0.010)
    samples = np.asarray([P5 + M5 @ np.array([0.0, 0.0, float(z)]) for z in zs])

    def depth(pts, cen, rad, mask=None):
        dd = np.linalg.norm(pts[:, None, :] - cen[None, :, :], axis=2) - rad[None, :]
        if mask is not None:
            dd = np.where(mask[None, :], dd, np.inf)
        return dd.min(axis=1)

    dep_a = depth(samples, cen_a, rad_a)
    dep_b = depth(samples, cen_b, rad_b, keep_b)
    unc_a = dep_a > 0.0
    unc_b = dep_b > 0.0

    # --- 색: link 별 --------------------------------------------------------------------
    groups = ["link_*_arm_5", "link_*_arm_6", "ee_finger_*", "ee_left/ee_right",
              "link_*_arm_0..4"]
    colour = {"link_*_arm_5": (0.92, 0.41, 0.20),
              "link_*_arm_6": (0.11, 0.69, 0.48),
              "ee_finger_*": (0.93, 0.63, 0.00),
              "ee_left/ee_right": (0.29, 0.23, 0.65),
              "link_*_arm_0..4": (0.16, 0.47, 0.84)}

    def group_of(n: str) -> str:
        if n in ARM5:
            return "link_*_arm_5"
        if n.endswith("arm_6"):
            return "link_*_arm_6"
        if n.startswith("ee_finger"):
            return "ee_finger_*"
        if n in PALM:
            return "ee_left/ee_right"
        return "link_*_arm_0..4"

    # --- 로봇 geom 의 alpha 를 조절할 수 있게 한다 -------------------------------------------
    scene_root_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_ROOTS}
    scene_root_ids.discard(-1)

    def root_of(b: int) -> int:
        while int(m.body_parentid[b]) != 0 and b != 0:
            b = int(m.body_parentid[b])
        return b

    robot_geoms = [g for g in range(m.ngeom)
                   if root_of(int(m.geom_bodyid[g])) not in scene_root_ids]
    other_geoms = [g for g in range(m.ngeom) if g not in set(robot_geoms)]
    saved_rgba = m.geom_rgba.copy()
    saved_matid = m.geom_matid.copy()
    print(f"robot geoms {len(robot_geoms)} · scene geoms {len(other_geoms)}")

    def set_alpha(geoms, a: float) -> None:
        """alpha 를 실제로 듣게 하려면 material 을 떼고 geom_rgba 로 내려야 한다."""
        for g in geoms:
            mid = int(saved_matid[g])
            rgba = (np.asarray(m.mat_rgba[mid], np.float32).copy() if mid >= 0
                    else np.asarray(saved_rgba[g], np.float32).copy())
            rgba[3] = a
            m.geom_rgba[g] = rgba
            m.geom_matid[g] = -1

    def restore() -> None:
        m.geom_rgba[:] = saved_rgba
        m.geom_matid[:] = saved_matid

    renderer = mujoco.Renderer(m, height=H_PX, width=W)
    fovy = float(m.vis.global_.fovy)

    def cam_pose(az, el, look):
        a, e = np.radians(az), np.radians(el)
        f = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        pos = np.asarray(look, np.float64) - DIST * f
        right = np.cross(f, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        return pos, f, right, up

    def project(pts, az, el, look):
        pos, f, right, up = cam_pose(az, el, look)
        v = np.atleast_2d(pts) - pos[None, :]
        zc = v @ f
        xc = v @ right
        yc = v @ up
        t = np.tan(np.radians(fovy) / 2.0)
        u = (W / 2.0) * (1.0 + (xc / np.maximum(zc, 1e-9)) / (t * W / H_PX))
        vv = (H_PX / 2.0) * (1.0 - (yc / np.maximum(zc, 1e-9)) / t)
        return np.stack([u, vv, zc], axis=1)

    palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FT_SENSOR_L")
    ap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    LOOK = 0.5 * (np.asarray(d.xpos[palm_id], np.float64)
                  + np.asarray(d.xpos[ap_id], np.float64)) + np.array([0.0, 0.0, -0.02])

    def render(az, el, spheres, shadow: bool = True):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = LOOK
        cam.distance, cam.azimuth, cam.elevation = DIST, az, el
        renderer.update_scene(d, cam)
        scn = renderer.scene
        if not shadow:
            # 실루엣 mask 를 만들 때 그림자가 차이에 섞이면 탁자 위에 유령 윤곽이 생긴다.
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        eye = np.eye(3).flatten()
        for pos, r, rgba in spheres:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0.0, 0.0]), np.asarray(pos, np.float64), eye,
                                np.asarray(rgba, np.float32))
            scn.ngeom += 1
        return renderer.render()

    def silhouette(az, el):
        """로봇만 불투명 / 씬 투명으로 한 번, 전부 투명으로 한 번 — 차이가 실루엣이다."""
        set_alpha(robot_geoms, 1.0)
        set_alpha(other_geoms, 0.0)
        with_robot = render(az, el, [], shadow=False).astype(np.int16)
        set_alpha(robot_geoms, 0.0)
        empty = render(az, el, [], shadow=False).astype(np.int16)
        restore()
        return (np.abs(with_robot - empty).max(axis=2) > 6).astype(np.float32)

    def spheres_for(model_key, alpha):
        out = []
        if model_key == "A":
            for i, n in enumerate(names_a):
                out.append((cen_a[i], float(rad_a[i]), (*colour[group_of(n)], alpha)))
        else:
            for i, n in enumerate(names_b):
                if not keep_b[i]:
                    continue
                out.append((cen_b[i], float(rad_b[i]), (*colour[group_of(n)], alpha)))
        return out

    figstyle.use_korean()
    written = []
    arm5_r_b = float(rad_b[[i for i, n in enumerate(names_b) if n == "link_left_arm_5"][0]]) * 1000
    fing_r_b = max(float(rad_b[i]) * 1000 for i, n in enumerate(names_b) if n == "ee_finger_l1")
    palm_r_b = max(float(rad_b[i]) * 1000 for i, n in enumerate(names_b) if n in PALM)
    n_palm_drawn = int(sum(1 for i, n in enumerate(names_b) if n in PALM and keep_b[i]))
    a5_kept = int(keep_b[[i for i, n in enumerate(names_b) if n == "link_left_arm_5"]].sum())
    a5_total = sum(1 for n in names_b if n == "link_left_arm_5")

    for vname, az, el, fname in VIEWS:
        sil = silhouette(az, el)
        # 겹친 씬: mesh 옅게, 구 진하게
        set_alpha(robot_geoms, MESH_ALPHA)
        img_scene = {k: render(az, el, spheres_for(k, SPHERE_A_TOP)) for k in ("A", "B")}
        # 구만: 로봇 안 보이게, 씬 물체는 그대로 두어 위치 감각을 남긴다
        set_alpha(robot_geoms, 0.0)
        img_only = {k: render(az, el, spheres_for(k, SPHERE_A_ONLY)) for k in ("A", "B")}
        restore()

        objs = rec["object_poses"]
        marks = {"apple (target)": np.asarray(objs["apple"]["pos"], np.float64),
                 "crate": np.asarray(objs["crate"]["pos"], np.float64),
                 "pear": np.asarray(objs["pear"]["pos"], np.float64),
                 "banana": np.asarray(objs["banana"]["pos"], np.float64)}
        px = {k: project(v[None, :], az, el, LOOK)[0] for k, v in marks.items()}
        sam_px = project(samples, az, el, LOOK)
        lm_world = np.asarray([P5 + M5 @ np.array([0.0, 0.0, z]) for z in
                               (0.0, -0.1087, -0.1548, -0.2278)])
        lm_px = project(lm_world, az, el, LOOK)
        lm_lab = ("link_*_arm_6  z 0", "FT_sensor  -108.7", "ee_left (palm)  -154.8",
                  "finger root  -227.8")

        fig, axes = plt.subplots(2, 2, figsize=(15.4, 13.4))
        figstyle.style_axes(fig, axes)
        titles = {
            "A": (f"A  current constraint model - {rob_a.n_spheres} spheres\n"
                  f"link_*_arm_5: 5 spheres of r 81.25 mm per arm · no ee_left/ee_right"),
            "B": (f"B  settled setting (approximate) - {int(keep_b.sum())} spheres used / "
                  f"{rob_b.n_spheres} in the model\n"
                  f"sphere_spacing {SPHERE_SPACING:g} · global capsule_radius_scale "
                  f"{GLOBAL_SCALE:g} · max_spheres_per_capsule {MAX_SPHERES}\n"
                  f"arm_5 cut at z = {Z_CUT_M*1000:.0f} mm ({a5_kept}/{a5_total} per arm) · "
                  f"{int(keep_b.sum()) - n_palm_drawn} of the {int(keep_b.sum())} "
                  f"are not on the palm\n"
                  f"link_*_arm_5 r {arm5_r_b:.2f} mm · ee_finger_l1 r {fing_r_b:.2f} mm · "
                  f"palm r {palm_r_b:.2f} mm"),
        }
        for col, k in enumerate(("A", "B")):
            for row in (0, 1):
                ax = axes[row, col]
                ax.imshow(img_scene[k] if row == 0 else img_only[k])
                ax.contour(sil, levels=[0.5], colors=[(0.10, 0.10, 0.10)],
                           linewidths=1.0 if row == 0 else 1.4, alpha=0.75)
                unc = unc_a if k == "A" else unc_b
                ax.plot(sam_px[:, 0], sam_px[:, 1], "-", lw=0.9, color=(0.2, 0.2, 0.2, 0.55),
                        zorder=5)
                ax.plot(sam_px[~unc, 0], sam_px[~unc, 1], "o", ms=2.6,
                        color=(1.0, 1.0, 1.0, 0.9), zorder=6,
                        markeredgecolor=(0.2, 0.2, 0.2, 0.7), markeredgewidth=0.4)
                if unc.any():
                    ax.plot(sam_px[unc, 0], sam_px[unc, 1], "o", ms=5.0,
                            color=(0.89, 0.17, 0.17), zorder=7)
                if row == 0:
                    for lab, uv in px.items():
                        if not (0 <= uv[0] < W and 0 <= uv[1] < H_PX and uv[2] > 0):
                            continue
                        dx, dy = (-40.0, 78.0) if lab == "apple (target)" else (0.0, 30.0)
                        ax.annotate(lab, (uv[0], uv[1]), xytext=(uv[0] + dx, uv[1] + dy),
                                    fontsize=8.6, ha="center",
                                    color=figstyle.CATEGORICAL[7] if lab.startswith("apple")
                                    else figstyle.INK,
                                    arrowprops=dict(arrowstyle="->", lw=1.1,
                                                    color=figstyle.CATEGORICAL[7]
                                                    if lab.startswith("apple")
                                                    else figstyle.INK_2),
                                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                              edgecolor="none", alpha=0.75))
                    for (u, v, zc), lab in zip(lm_px, lm_lab):
                        ax.plot([u], [v], marker="|", ms=9, mew=1.6, color=figstyle.INK, zorder=8)
                        ax.annotate(lab, (u, v), xytext=(10, 0), textcoords="offset points",
                                    fontsize=6.9, color=figstyle.INK, va="center", zorder=9,
                                    bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                                              edgecolor="none", alpha=0.7))
                    ax.set_title(titles[k], fontsize=9.8, color=figstyle.INK)
                    ax.annotate(f"robot mesh at alpha {MESH_ALPHA:g}, spheres at "
                                f"{SPHERE_A_TOP:g}\nlink_*_arm_5 axis, 41 samples "
                                f"z -300..+100 mm\nred = covered by no sphere here: "
                                f"{int(unc.sum())}",
                                (0.015, 0.985), xycoords="axes fraction", va="top",
                                fontsize=7.6, color=figstyle.INK,
                                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                          edgecolor=figstyle.GRID_INK, alpha=0.88))
                else:
                    ax.set_title(f"{k}  spheres only - robot mesh hidden, its outline kept",
                                 fontsize=9.6, color=figstyle.INK)
                    ax.annotate("black outline = robot silhouette at this pose\n"
                                "spheres outside it stick out of the robot;\n"
                                "spheres inside it are swallowed by the mesh",
                                (0.015, 0.985), xycoords="axes fraction", va="top",
                                fontsize=7.6, color=figstyle.INK,
                                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                          edgecolor=figstyle.GRID_INK, alpha=0.88))
                ax.set_xlim(0, W)
                ax.set_ylim(H_PX, 0)
                ax.set_axis_off()

        handles = []
        for g in groups:
            ra = sorted({round(float(rad_a[i]) * 1000, 2) for i, n in enumerate(names_a)
                         if group_of(n) == g})
            rb = sorted({round(float(rad_b[i]) * 1000, 2) for i, n in enumerate(names_b)
                         if group_of(n) == g and keep_b[i]})
            na = sum(1 for n in names_a if group_of(n) == g)
            nb = int(sum(1 for i, n in enumerate(names_b) if group_of(n) == g and keep_b[i]))
            if not ra:
                lab = (f"{g}  B only  r {'/'.join(f'{v:.1f}' for v in rb)} mm  ({nb} spheres)")
            else:
                lab = (f"{g}  A r {'/'.join(f'{v:.1f}' for v in ra)} mm ({na})  ->  "
                       f"B r {'/'.join(f'{v:.1f}' for v in rb)} mm ({nb})")
            handles.append(Patch(facecolor=(*colour[g], 0.85), edgecolor="none", label=lab))
        handles.append(Patch(facecolor=(0.89, 0.29, 0.29, 0.95), edgecolor="none",
                             label=f"axis sample point covered by NO sphere in B "
                                   f"({int(unc_b.sum())} of {samples.shape[0]} points on the "
                                   f"link_*_arm_5 axis, z -300..+100 mm; T8d's B was 10)"))
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.2, frameon=False,
                   bbox_to_anchor=(0.5, 0.004))
        fig.suptitle(
            f"T8f - the settled setting drawn over the real scene, view: {vname}\n"
            f"pre-grasp pose from the record: shadow chunk {CHUNK} (seq {rec['seq']}, "
            f"t={rec['t_step']}), plan step {STEP}, fingertip to apple surface "
            f"{fingertip_mm:+.2f} mm",
            fontsize=12.0, color=figstyle.INK, y=0.988)
        fig.text(0.5, 0.118,
                 "B's spheres are smaller than the robot itself - link_*_arm_5 goes "
                 f"81.25 -> {arm5_r_b:.2f} mm against a URDF capsule of r 75 mm - so with an "
                 "opaque mesh they are hidden inside it.\nThat is why the mesh is drawn at alpha "
                 f"{MESH_ALPHA:g} here and why the bottom row shows the spheres alone. "
                 "In A the spheres stick out past the silhouette; in B they sit inside it.",
                 ha="center", fontsize=8.4, color=figstyle.INK_2)
        fig.text(0.5, 0.082,
                 "B reaches 0 exec8 violations only when the apple rows are also removed for "
                 "ee_left: with the apple kept for the palm the same cell\nleaves 5 rows at "
                 "-4.81 mm in exec8. T8e cell spacing_0.4__scale_0.2__palm_in_authorized: "
                 "exec8 0 rows (min slack +3.0363 mm), full50 22 rows / 2 chunks / -6.3714 mm.",
                 ha="center", fontsize=8.2, color=figstyle.INK_2)
        fig.text(0.5, 0.050,
                 "B is still an approximation drawn for inspection, not a built model: there is "
                 "no flag yet to shorten a capsule, so the cut is done by dropping\nthe arm_5 "
                 "spheres whose centre is distal to z = -100 mm; a really shortened capsule "
                 "would also have fewer spheres and therefore a different radius.",
                 ha="center", fontsize=8.2, color=figstyle.INK_2)
        fig.tight_layout(rect=(0, 0.150, 1, 0.945))
        out = OUTDIR / fname
        fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
        plt.close(fig)
        side = {
            "view": vname, "azimuth_deg": az, "elevation_deg": el, "distance_m": DIST,
            "model_xml": MODEL_XML, "repo_root": REPO,
            "render": {"robot_mesh_alpha": MESH_ALPHA,
                       "sphere_alpha_overlay_panel": SPHERE_A_TOP,
                       "sphere_alpha_spheres_only_panel": SPHERE_A_ONLY,
                       "silhouette": "robot geoms opaque / scene transparent minus an "
                                     "all-transparent render, shadows off; contour of the "
                                     "difference mask",
                       "n_robot_geoms": len(robot_geoms), "n_scene_geoms": len(other_geoms),
                       "why_redrawn": "첫 판은 mesh 가 불투명하고 구가 반투명이라 B 의 "
                                      f"link_*_arm_5 구(r {arm5_r_b:.2f} mm)가 URDF capsule "
                                      "r 75 mm 보다 작아 mesh 안에 파묻혀 보이지 않았다."},
            "model_A_identical_to_T8a_tensor": {
                "link_names_match": bool(_same_names), "max_abs_radius_diff_mm": _rad_diff},
            "pose": {"record": SHADOW, "chunk": CHUNK, "seq": int(rec["seq"]),
                     "t_step": int(rec["t_step"]), "plan_step": STEP,
                     "variant": "actions_reference",
                     "fingertip_to_apple_surface_mm": round(fingertip_mm, 4),
                     "same_as_T8d": True},
            "model_A": {
                "link_filter": sorted(set(names_a)), "n_spheres": int(rob_a.n_spheres),
                "radius_mm_by_link": {ln: sorted({round(float(rad_a[i]) * 1000, 4)
                                                  for i, n in enumerate(names_a) if n == ln})
                                      for ln in sorted(set(names_a))},
                "n_spheres_by_link": {ln: sum(1 for n in names_a if n == ln)
                                      for ln in sorted(set(names_a))}},
            "model_B": {
                "approximation": "arm_5 spheres with local z < z_cut removed; no capsule-length "
                                 "flag exists, so this is a drawing approximation. The spacing, "
                                 "the global radius scale and the palm links are real settings.",
                "z_cut_m": Z_CUT_M, "sphere_spacing": SPHERE_SPACING,
                "capsule_radius_scale_global": GLOBAL_SCALE,
                "max_spheres_per_capsule": MAX_SPHERES,
                "expected_from_T8e_s2_cell": {
                    "key": "spacing_0.4__scale_0.2__palm_out",
                    "key_with_palm": "spacing_0.4__scale_0.2__palm_in_authorized",
                    "n_spheres_used_after_cut_without_palm": 218,
                    "n_spheres_used_after_cut_with_palm": 264, "n_spheres_model": 274,
                    "palm_must_be_authorized": "apple rows must also be removed for ee_left; "
                                               "with the apple kept for the palm the same cell "
                                               "leaves 5 exec8 rows at -4.8074 mm",
                    "radius_mm_link_left_arm_5": 20.4426,
                    "radius_mm_ee_finger_l1_distal": 4.72,
                    "radius_mm_palm_max": 8.9236,
                    "exec8_rows_slack_le0": 0, "exec8_min_slack_mm": 3.0363,
                    "full50_rows_slack_le0": 22, "full50_min_slack_mm": -6.37},
                "palm_links_added": list(PALM),
                "n_spheres_drawn": int(keep_b.sum()),
                "n_spheres_before_cut": int(rob_b.n_spheres),
                "urdf_capsule_radius_mm_link_arm_5": 75.0,
                "radius_mm_by_link": {ln: sorted({round(float(rad_b[i]) * 1000, 4)
                                                  for i, n in enumerate(names_b)
                                                  if n == ln and keep_b[i]})
                                      for ln in sorted(set(names_b))},
                "n_spheres_by_link": {ln: int(sum(1 for i, n in enumerate(names_b)
                                                  if n == ln and keep_b[i]))
                                      for ln in sorted(set(names_b))}},
            "arm5_axis_samples": {
                "frame": "link_left_arm_5, x=y=0, z from -300 to +100 mm step 10 mm",
                "n_points": int(samples.shape[0]),
                "n_uncovered_model_A": int(unc_a.sum()),
                "n_uncovered_model_B": int(unc_b.sum()),
                "n_uncovered_model_B_in_T8d": 10,
                "uncovered_z_mm_model_B": [round(float(z) * 1000, 1)
                                           for z, u in zip(zs, unc_b) if u],
                "depth_inside_nearest_sphere_mm_model_A":
                    [round(float(-v) * 1000, 3) for v in dep_a],
                "depth_inside_nearest_sphere_mm_model_B":
                    [round(float(-v) * 1000, 3) for v in dep_b],
                "z_mm": [round(float(z) * 1000, 1) for z in zs]},
            "annotated_object_pixels": {k: [round(float(v[0]), 1), round(float(v[1]), 1)]
                                        for k, v in px.items()},
            "object_poses_m": {k: [round(float(x), 5) for x in v] for k, v in marks.items()},
            "landmarks_z_mm_in_arm5_frame": {"link_left_arm_6": 0.0, "FT_sensor_L": -108.7,
                                             "ee_left": -154.8, "ee_finger_l1_root": -227.8},
        }
        (out.with_suffix(".json")).write_text(json.dumps(side, indent=1, ensure_ascii=False))
        written.append(str(out))
        print("wrote", out)

    (RAW / "t8f_summary.json").write_text(json.dumps({
        "files": written, "chunk": CHUNK, "plan_step": STEP, "pose_same_as_T8d": True,
        "fingertip_to_apple_surface_mm": round(fingertip_mm, 4),
        "z_cut_m": Z_CUT_M, "sphere_spacing": SPHERE_SPACING,
        "capsule_radius_scale_global": GLOBAL_SCALE,
        "max_spheres_per_capsule": MAX_SPHERES,
        "n_spheres_A": int(rob_a.n_spheres),
        "n_spheres_B_model": int(rob_b.n_spheres),
        "n_spheres_B_drawn": int(keep_b.sum()),
        "n_uncovered_axis_samples_A": int(unc_a.sum()),
        "n_uncovered_axis_samples_B": int(unc_b.sum()),
        "n_uncovered_axis_samples_T8d_B": 10,
        "radius_mm_link_left_arm_5_B": round(arm5_r_b, 4),
        "robot_mesh_alpha": MESH_ALPHA,
        "redraw_reason": "B 의 구가 로봇 mesh 안에 파묻혀 보이지 않았다 — mesh 를 옅게 하고 "
                         "구만 그린 칸과 실루엣 윤곽을 더했다",
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
