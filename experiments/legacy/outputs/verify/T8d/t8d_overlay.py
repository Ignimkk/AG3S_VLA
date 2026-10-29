"""T8d — 제약 구 근사를 실제 씬 위에 겹쳐 본다. 지금 모델(A) 대 제안 모델(B).

pose 는 기록의 pre-grasp frame (shadow `actions_reference`, 손끝이 사과 표면을 처음 파고드는
chunk 근처). 구는 MuJoCo scene 에 **실제 3D 로** 넣어 반투명으로 그리므로 로봇 mesh 가
그 아래로 비친다 — 2D 로 덧그린 원이 아니다.

A. 지금  : `ARM_LINKS` = link_{l,r}_arm_{0..6} + ee_finger_* , 구 120 개. `link_*_arm_5` 는
           r 81.25 mm 구 5 개 (URDF capsule r 75 · L 250, 축 z ∈ [-225, +25] mm).
B. 제안  : (1) `link_*_arm_5` 의 구 중 중심 z < z_cut 인 것을 **빼고**
           (2) 손가락 capsule 배율을 낮춰 원위 구를 8 mm 아래로
           (3) 손바닥 `ee_left`/`ee_right` 구를 **추가** (MJCF `EE_BODY_L/R` mesh 에서 측정된
               capsule — 지금 제약 모델에는 link_filter 가 빼고 있다).
           (1) 은 근사다: capsule 길이를 줄이는 flag 가 없어 구를 빼는 것으로 흉내냈고,
           실제로 짧은 capsule 을 만들면 구 개수가 줄어 반지름도 달라진다.

빨간 점 = `link_*_arm_5` capsule 축 위에서 **B 의 어떤 구에도 안 덮이는** 표본점.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SHADOW, load_planning)

OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8d'))
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8d'))
CHUNK = 9
Z_CUT_M = -0.10
FINGER_SCALE = 0.28
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
ARM5 = ("link_left_arm_5", "link_right_arm_5")
PALM = ("ee_left", "ee_right")
VIEWS = [("front", 90.0, -12.0, "t8d-front.png"),
         ("oblique-left", 140.0, -18.0, "t8d-oblique-left.png"),
         ("oblique-right", 40.0, -18.0, "t8d-oblique-right.png")]
W, H_PX = 800, 720
DIST = 1.75
ALPHA = 0.30


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)

    rob_a = build_constraint_robot_model(scene, link_filter=tuple(ARM_LINKS))
    names_a = list(resolve_link_names(rob_a, rob_a.n_spheres))
    rob_b = build_constraint_robot_model(
        scene, link_filter=tuple(ARM_LINKS) + PALM,
        sphere_options={"capsule_radius_scale_by_link": {ln: FINGER_SCALE
                                                         for ln in FINGER_LINKS}})
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
    ap = t8a["clearance_m"][CHUNK, :8][:, fing, ai] * 1000.0       # (8, nfinger)
    STEP = int(np.argmin(ap.min(axis=1)))
    fingertip_mm = float(ap.min())
    qpos = np.asarray(rec["qpos"], np.float64)
    d.qpos[: qpos.size] = qpos
    for n_, ci in ACT_COL.items():
        d.qpos[QADR[n_]] = float(chunk[STEP, ci])
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    q = np.asarray([qpos[QADR[n_]] if n_.startswith("torso_") else float(chunk[STEP, ACT_COL[n_]])
                    for n_ in JOINT_ORDER], np.float64)
    print(f"pose: chunk {CHUNK} seq {rec['seq']} t_step {rec['t_step']} plan step {STEP} "
          f"(fingertip-to-apple {fingertip_mm:+.2f} mm)")

    cen_a, rad_a = rob_a.sphere_centers_numeric(q)
    cen_a = np.asarray(cen_a, np.float64).reshape(-1, 3)
    rad_a = np.asarray(rad_a, np.float64).reshape(-1)
    cen_b, rad_b = rob_b.sphere_centers_numeric(q)
    cen_b = np.asarray(cen_b, np.float64).reshape(-1, 3)
    rad_b = np.asarray(rad_b, np.float64).reshape(-1)

    # --- link_*_arm_5 축 표본점과 B 의 덮개 ------------------------------------------------
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
        """world -> pixel, MuJoCo free-camera convention. (u, v, depth)"""
        pos, f, right, up = cam_pose(az, el, look)
        v = np.atleast_2d(pts) - pos[None, :]
        zc = v @ f
        xc = v @ right
        yc = v @ up
        t = np.tan(np.radians(fovy) / 2.0)
        u = (W / 2.0) * (1.0 + (xc / np.maximum(zc, 1e-9)) / (t * W / H_PX))
        vv = (H_PX / 2.0) * (1.0 - (yc / np.maximum(zc, 1e-9)) / t)
        return np.stack([u, vv, zc], axis=1)

    def render(az, el, spheres):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "FT_SENSOR_L")
        ap_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        look = 0.5 * (np.asarray(d.xpos[palm_id], np.float64)
                      + np.asarray(d.xpos[ap_id], np.float64)) + np.array([0.0, 0.0, -0.02])
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = DIST, az, el
        renderer.update_scene(d, cam)
        scn = renderer.scene
        eye = np.eye(3).flatten()
        for pos, r, rgba in spheres:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, 0.0, 0.0]), np.asarray(pos, np.float64), eye,
                                np.asarray(rgba, np.float32))
            scn.ngeom += 1
        return renderer.render(), np.asarray(look, np.float64)

    def spheres_for(model_key):
        out = []
        if model_key == "A":
            for i, n in enumerate(names_a):
                c = colour[group_of(n)]
                a = 0.48 if n in ARM5 else ALPHA
                out.append((cen_a[i], float(rad_a[i]), (*c, a)))
        else:
            for i, n in enumerate(names_b):
                if not keep_b[i]:
                    continue
                c = colour[group_of(n)]
                a = 0.48 if n in ARM5 else ALPHA
                out.append((cen_b[i], float(rad_b[i]), (*c, a)))
        return out

    figstyle.use_korean()
    written = []
    for vname, az, el, fname in VIEWS:
        rendered = {k: render(az, el, spheres_for(k)) for k in ("A", "B")}
        imgs = {k: v[0] for k, v in rendered.items()}
        look = rendered["A"][1]
        objs = rec["object_poses"]
        marks = {"apple (target)": np.asarray(objs["apple"]["pos"], np.float64),
                 "crate": np.asarray(objs["crate"]["pos"], np.float64),
                 "pear": np.asarray(objs["pear"]["pos"], np.float64),
                 "banana": np.asarray(objs["banana"]["pos"], np.float64)}
        px = {k: project(v[None, :], az, el, look)[0] for k, v in marks.items()}
        sam_px = project(samples, az, el, look)
        lm_world = np.asarray([P5 + M5 @ np.array([0.0, 0.0, z]) for z in
                               (0.0, -0.1087, -0.1548, -0.2278)])
        lm_px = project(lm_world, az, el, look)
        lm_lab = ("link_*_arm_6  z 0", "FT_sensor  -108.7", "ee_left (palm)  -154.8",
                  "finger root  -227.8")
        fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.6))
        figstyle.style_axes(fig, axes)
        titles = {
            "A": (f"A  current constraint model - {rob_a.n_spheres} spheres\n"
                  f"link_*_arm_5: {sum(1 for n in names_a if n in ARM5) // 2} spheres of "
                  f"r {rad_a[[i for i, n in enumerate(names_a) if n == 'link_left_arm_5'][0]]*1000:.2f} mm "
                  f"per arm · no ee_left/ee_right"),
            "B": (f"B  proposal (approximate) - {int(keep_b.sum())} spheres\n"
                  f"link_*_arm_5 cut at z = {Z_CUT_M*1000:.0f} mm "
                  f"({int(keep_b[[i for i, n in enumerate(names_b) if n == 'link_left_arm_5']].sum())}"
                  f"/{sum(1 for n in names_b if n == 'link_left_arm_5')} spheres kept) · "
                  f"fingers x{FINGER_SCALE:g} · ee_left/ee_right added"),
        }
        for k, ax in zip(("A", "B"), axes):
            ax.imshow(imgs[k])
            for lab, uv in px.items():
                if not (0 <= uv[0] < W and 0 <= uv[1] < H_PX and uv[2] > 0):
                    continue
                dx, dy, ha = ((-95.0, 52.0, "center") if lab == "apple (target)"
                              else (0.0, 30.0, "center"))
                ax.annotate(lab, (uv[0], uv[1]), xytext=(uv[0] + dx, uv[1] + dy),
                            fontsize=8.6, ha=ha,
                            color=figstyle.CATEGORICAL[7] if lab.startswith("apple")
                            else figstyle.INK,
                            arrowprops=dict(arrowstyle="->", lw=1.1,
                                            color=figstyle.CATEGORICAL[7]
                                            if lab.startswith("apple") else figstyle.INK_2),
                            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                      edgecolor="none", alpha=0.75))
            unc = unc_a if k == "A" else unc_b
            ax.plot(sam_px[:, 0], sam_px[:, 1], "-", lw=0.9, color=(1, 1, 1, 0.65), zorder=5)
            ax.plot(sam_px[~unc, 0], sam_px[~unc, 1], "o", ms=2.6,
                    color=(1.0, 1.0, 1.0, 0.85), zorder=6)
            if unc.any():
                ax.plot(sam_px[unc, 0], sam_px[unc, 1], "o", ms=5.0,
                        color=(0.89, 0.17, 0.17), zorder=7)
            for (u, v, zc), lab in zip(lm_px, lm_lab):
                ax.plot([u], [v], marker="|", ms=9, mew=1.6, color=figstyle.INK, zorder=8)
                ax.annotate(lab, (u, v), xytext=(10, 0), textcoords="offset points",
                            fontsize=6.9, color=figstyle.INK, va="center", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                                      edgecolor="none", alpha=0.7))
            ax.annotate(f"link_*_arm_5 axis, 41 samples z -300..+100 mm\n"
                        f"red = covered by no sphere in this model: {int(unc.sum())}",
                        (0.015, 0.985), xycoords="axes fraction", va="top", fontsize=7.6,
                        color=figstyle.INK,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor=figstyle.GRID_INK, alpha=0.85))
            ax.set_xlim(0, W)
            ax.set_ylim(H_PX, 0)
            ax.set_axis_off()
            ax.set_title(titles[k], fontsize=10.2, color=figstyle.INK)
        handles = []
        for g in groups:
            if g == "ee_left/ee_right":
                rr = sorted({round(float(rad_b[i]) * 1000, 2)
                             for i, n in enumerate(names_b) if n in PALM})
                lab = f"{g}  r {'/'.join(f'{v:.1f}' for v in rr)} mm  (B only)"
            elif g == "ee_finger_*":
                ra = sorted({round(float(rad_a[i]) * 1000, 2)
                             for i, n in enumerate(names_a) if n.startswith("ee_finger")})
                rb = sorted({round(float(rad_b[i]) * 1000, 2)
                             for i, n in enumerate(names_b) if n.startswith("ee_finger")})
                lab = (f"{g}  A r {'/'.join(f'{v:.1f}' for v in ra)} mm  ->  "
                       f"B r {'/'.join(f'{v:.1f}' for v in rb)} mm")
            elif g == "link_*_arm_5":
                lab = (f"{g}  r 81.25 mm x 5 spheres per arm  "
                       f"(URDF capsule r 75.0 / L 250.0 mm)")
            elif g == "link_*_arm_6":
                rr = sorted({round(float(rad_a[i]) * 1000, 2)
                             for i, n in enumerate(names_a) if n.endswith("arm_6")})
                lab = f"{g}  r {'/'.join(f'{v:.1f}' for v in rr)} mm x 15 per arm"
            else:
                rr = sorted({round(float(rad_a[i]) * 1000, 2) for i, n in enumerate(names_a)
                             if group_of(n) == g})
                lab = f"{g}  r {'/'.join(f'{v:.1f}' for v in rr)} mm"
            handles.append(Patch(facecolor=(*colour[g], 0.55), edgecolor="none", label=lab))
        handles.append(Patch(facecolor=(0.89, 0.29, 0.29, 0.95), edgecolor="none",
                             label=f"axis sample point covered by NO sphere in B "
                                   f"({int(unc_b.sum())} of {samples.shape[0]} points on the "
                                   f"link_*_arm_5 axis, z -300..+100 mm)"))
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.4, frameon=False,
                   bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(
            f"T8d - constraint-sphere approximation over the real scene, view: {vname}\n"
            f"pre-grasp pose from the record: shadow chunk {CHUNK} (seq {rec['seq']}, "
            f"t={rec['t_step']}), plan step {STEP}, fingertip to apple surface "
            f"{fingertip_mm:+.2f} mm",
            fontsize=12.0, color=figstyle.INK, y=0.985)
        fig.text(0.5, 0.135,
                 "B is an approximation drawn for inspection, not a built model: there is no flag "
                 "yet to shorten a capsule, so the cut is done by dropping the arm_5 spheres whose "
                 "centre is distal to z = -100 mm; a really shortened capsule would also have "
                 "fewer spheres and therefore a different radius.",
                 ha="center", fontsize=8.2, color=figstyle.INK_2)
        fig.tight_layout(rect=(0, 0.175, 1, 0.945))
        out = OUTDIR / fname
        fig.savefig(out, dpi=140, facecolor=figstyle.SURFACE)
        plt.close(fig)
        side = {
            "view": vname, "azimuth_deg": az, "elevation_deg": el, "distance_m": DIST,
            "model_xml": MODEL_XML,
            "pose": {"record": SHADOW, "chunk": CHUNK, "seq": int(rec["seq"]),
                     "t_step": int(rec["t_step"]), "plan_step": STEP,
                     "variant": "actions_reference",
                     "fingertip_to_apple_surface_mm": round(fingertip_mm, 4)},
            "model_A": {
                "link_filter": sorted(set(names_a)), "n_spheres": int(rob_a.n_spheres),
                "radius_mm_by_link": {ln: sorted({round(float(rad_a[i]) * 1000, 4)
                                                  for i, n in enumerate(names_a) if n == ln})
                                      for ln in sorted(set(names_a))},
                "n_spheres_by_link": {ln: sum(1 for n in names_a if n == ln)
                                      for ln in sorted(set(names_a))}},
            "model_B": {
                "approximation": "arm_5 spheres with local z < z_cut removed; no capsule-length "
                                 "flag exists, so this is a drawing approximation",
                "z_cut_m": Z_CUT_M, "finger_capsule_radius_scale": FINGER_SCALE,
                "palm_links_added": list(PALM),
                "n_spheres_drawn": int(keep_b.sum()),
                "n_spheres_before_cut": int(rob_b.n_spheres),
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
    (RAW / "t8d_summary.json").write_text(json.dumps({
        "files": written, "chunk": CHUNK, "plan_step": STEP,
        "fingertip_to_apple_surface_mm": round(fingertip_mm, 4),
        "z_cut_m": Z_CUT_M, "finger_capsule_radius_scale": FINGER_SCALE,
        "n_spheres_A": int(rob_a.n_spheres), "n_spheres_B_drawn": int(keep_b.sum()),
        "n_uncovered_axis_samples_A": int(unc_a.sum()),
        "n_uncovered_axis_samples_B": int(unc_b.sum()),
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
