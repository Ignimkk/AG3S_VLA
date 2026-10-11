"""T5d figure 4 (실제 씬) — 다섯 순간의 3인칭 · 세 카메라 · base frame 평면도.

사용자 요청 2026-09-25. seq 15 · 16 · 20 · 30 · 33 다섯만 담는다. 기존
`t5d-scene-attention-handoff.png` 을 덮지 않는다 (새 이름).

apple point 를 **self-filter 전과 후 둘 다** 적는다 — 한 값만 보이면 "원래 안 보였나,
self-filter 가 지웠나" 가 구별되지 않는다. 숫자는 전부 이미 잰 파일에서 온다:
`outputs/verify/T5d/apple_stages_ep1807.json` · `outputs/verify/T5d/aggregate_T5d.json`.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/make_fig4_five_moments.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import measure_items as mi  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T5d'))
from measure_attention_flow_ep1807 import build_body_group  # noqa: E402

RECORDS = "outputs/live_test/20260925_ep1807/run_0000"
ATTENTION = "AG3S/ag3s/asset/data/attention_16d_ep1807.npz"
STEP1 = "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json"
AGG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5d/aggregate_T5d.json'))
STAGES = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5d/apple_stages_ep1807.json'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5d'))
PNG = OUTDIR / "t5d-scene-five-moments.png"
SIDECAR = OUTDIR / "t5d-scene-five-moments.json"

PICK = [15, 16, 20, 30, 33]
WHY = {
    15: "apple point 이 무너지기 시작 · top cluster 는 아직 apple",
    16: "apple point 0 · top cluster 가 crate 로 넘어가는 전환점",
    20: "gripper 가 닫힌다 (first_closed_seq) · latch 가 attach",
    30: "place 국면 한가운데 · crate score 최고점 부근",
    33: "gripper 가 열린다 (first_reopen_seq) · apple point 가 돌아온다",
}
WHY_EN = {
    15: "apple points start collapsing; top cluster still apple",
    16: "apple points reach 0; top cluster flips to crate",
    20: "gripper closes (first_closed_seq); latch fires attach",
    30: "middle of place; crate score near its peak",
    33: "gripper reopens (first_reopen_seq); apple points come back",
}
NAME_SLOT = {"apple": 0, "crate": 1, "table": 2, "pear": 3, "orange": 4,
             "banana": 5, "robot": 6, "unknown": 7}
ROW_LABEL = ["third person\n(free camera)", "zed_left", "wrist_cam_l", "wrist_cam_r",
             "base frame plan · y [m] (base)"]


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)

    korean = figstyle.use_korean()
    agg = json.loads(AGG.read_text())
    table = {r["seq"]: r for r in agg["frame_table_seq_1_51"]}
    stage_by_seq = {f["seq"]: f for f in json.loads(STAGES.read_text())["frames"]}
    closed_below = float(agg["task_phase_rule"]["gripper_closed_below"])
    first_closed = agg["task_phase_rule"]["first_closed_seq"]
    first_reopen = agg["task_phase_rule"]["first_reopen_seq"]

    run = load_run(RECORDS)
    boundaries, _ = phase_boundaries_for_path(RECORDS)
    blob = np.load(ATTENTION, allow_pickle=False)
    cell = json.loads(pathlib.Path(STEP1).read_text())["best"]
    att_block = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    names_in_npz = [str(c) for c in blob["cameras"]]
    cam_index = {mj: names_in_npz.index(pol) for pol, mj in mi.ATTENTION_ALIAS.items()
                 if pol in names_in_npz}

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    group, _names = build_body_group(scene.model)
    apple_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
    renderer = mujoco.Renderer(scene.model, scene.height, scene.width)

    # 이 모델에는 3인칭 카메라 body 가 없다 (ncam 4 = wrist_cam_r/l, zed_left/right).
    # 자유 카메라를 테이블에 맞춰 세운다 — 그 설정을 sidecar 에 그대로 남긴다.
    free = mujoco.MjvCamera()
    free.type = mujoco.mjtCamera.mjCAMERA_FREE
    free.lookat[:] = np.asarray(scene.data.xpos[
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "crate")], np.float64)
    free.distance = 2.2
    free.azimuth = 138.0
    free.elevation = -26.0
    free_cfg = {"type": "free", "distance": free.distance, "azimuth": free.azimuth,
                "elevation": free.elevation, "lookat_body": "crate"}

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)

    fig, axes = plt.subplots(5, len(PICK), figsize=(3.45 * len(PICK), 15.8))
    figstyle.style_axes(fig, axes)
    side = {"records": RECORDS, "picked_seq": PICK,
            "free_camera": free_cfg,
            "count_definitions": {
                "apple_points_after_reconstruct":
                    "세 카메라 각각의 reconstruct 뒤 apple point 를 더한 값 "
                    "(apple_stages_ep1807.json 의 n_after_recon_total)",
                "apple_points_after_self_filter":
                    "같은 합, robot self-filter 뒤 (n_after_filter_total)",
                "apple_points_in_fused_cloud":
                    "fusion 이 voxel 마다 대표점 하나만 남긴 뒤의 apple point "
                    "(attention_flow_ep1807.json 의 apple_cloud_points) — 위 두 값보다 "
                    "작은 것은 같은 voxel 의 관측이 하나로 접히기 때문이다"},
            "panels": []}

    for col, seq in enumerate(PICK):
        i = seq - 1
        step = run.steps[i]
        pose_scene(scene, step)
        gt = {c: scene.capture(c) for c in mi.CAMERAS}
        att = {cam: att_block[i, di, ai, cell["layer"], cell["head"], ci]
               for cam, ci in cam_index.items()}
        observations, obs_name = [], {}
        for c in mi.CAMERAS:
            obs, _f = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                         attention_map=att.get(c))
            observations.append(obs)
            obs_name[obs.camera_id] = c
        cs, debug = ag.process_multi_debug(
            observations, phase=phase_for(step.t_step, boundaries))
        g = debug["grounding"]
        fc = debug["fusion"].cloud
        row = table[seq]
        st = stage_by_seq[seq]
        n_recon = st["n_after_recon_total"]
        n_filt = st["n_after_filter_total"]
        n_fused = row["apple_cloud_points"]
        grip_state = ("closed" if row["gripper_obs_left"] < closed_below else "open")

        # --- row 0: 3인칭 자유 카메라 --------------------------------------------
        free.lookat[:] = np.asarray(scene.data.xpos[
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "crate")], np.float64)
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(scene.data, camera=free)
        rgb3 = np.asarray(renderer.render())
        renderer.enable_segmentation_rendering()
        renderer.update_scene(scene.data, camera=free)
        seg3 = np.asarray(renderer.render())
        renderer.disable_segmentation_rendering()
        is_geom = seg3[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        bid3 = np.full(seg3.shape[:2], -1, np.int32)
        bid3[is_geom] = scene.model.geom_bodyid[seg3[..., 0][is_geom]]
        ax = axes[0, col]
        ax.imshow(rgb3)
        if (bid3 == apple_bid).any():
            ax.contour((bid3 == apple_bid).astype(float), levels=[0.5],
                       colors=[figstyle.CATEGORICAL[0]], linewidths=1.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"seq {seq} · t_step {row['t_step']}\n"
            f"{WHY[seq] if korean else WHY_EN[seq]}\n"
            f"top cluster {row['top_name']} {row['top_score']:.4f}\n"
            f"task phase {row['task_phase']} · gripper {grip_state} "
            f"({row['gripper_obs_left']:.3f})\n"
            f"apple points: before self-filter {n_recon} -> after {n_filt}\n"
            f"(in fused cloud {n_fused})",
            fontsize=8.2, color=figstyle.INK, linespacing=1.5)

        # --- rows 1-3: 세 카메라 실제 이미지 ---------------------------------------
        for ri, cname in enumerate(mi.CAMERAS, start=1):
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(scene.data, camera=cname)
            rgb = np.asarray(renderer.render())
            axc = axes[ri, col]
            axc.imshow(rgb)
            m = gt[cname].body_ids == apple_bid
            if m.any():
                axc.contour(m.astype(float), levels=[0.5],
                            colors=[figstyle.CATEGORICAL[0]], linewidths=1.6)
            axc.set_xticks([]); axc.set_yticks([])
            pc = st["per_camera"][cname]
            axc.set_title(
                f"{cname} · apple GT pixel {pc['n_gt_pixel']} · "
                f"recon {pc['n_after_recon']} → filter {pc['n_after_filter']}",
                fontsize=7.8,
                color=(figstyle.CATEGORICAL[7] if pc["n_after_filter"] == 0
                       and pc["n_after_recon"] > 0 else figstyle.INK_2))

        # --- row 4: base frame 평면도 ---------------------------------------------
        ax2 = axes[4, col]
        rep = np.asarray(fc.representative, np.int64)
        rep_cam = np.asarray(fc.obs_camera, np.int16)[rep]
        rep_uv = np.asarray(fc.obs_uv, np.int32)[rep]
        cams = [obs_name[cid] for cid in fc.cameras]
        pb = np.full(len(fc), -1, np.int64)
        for ci, cn in enumerate(cams):
            sel = np.nonzero(rep_cam == ci)[0]
            if not sel.size:
                continue
            ids = np.asarray(gt[cn].body_ids)
            u = np.clip(rep_uv[sel, 0], 0, ids.shape[1] - 1)
            v = np.clip(rep_uv[sel, 1], 0, ids.shape[0] - 1)
            pb[sel] = ids[v, u]
        pnames = np.array([group.get(int(b), "unknown") for b in pb], dtype=object)
        pts = np.asarray(fc.points, np.float64)
        for nm in ("table", "crate", "robot", "pear", "orange", "banana", "apple"):
            m = pnames == nm
            if not m.any():
                continue
            ax2.scatter(pts[m, 0], pts[m, 1], s=(9.0 if nm == "apple" else 0.7),
                        c=figstyle.CATEGORICAL[NAME_SLOT[nm]], linewidths=0,
                        alpha=(1.0 if nm == "apple" else 0.5),
                        label=nm if col == 0 else None, zorder=(3 if nm == "apple" else 1))
        for cl in g.clusters[:4]:
            c = np.asarray(cl.centroid, np.float64)
            idx = np.asarray(cl.point_indices, np.int64)
            labs, cnts = np.unique(pnames[idx], return_counts=True)
            nm = str(labs[int(np.argmax(cnts))])
            ax2.plot([c[0]], [c[1]], marker="x", ms=8, mew=2.0,
                     color=figstyle.CATEGORICAL[NAME_SLOT.get(nm, 7)], zorder=5)
            ax2.annotate(f"{nm} {cl.target_score:.2f}", (c[0], c[1]),
                         textcoords="offset points", xytext=(6, 5), fontsize=7.2,
                         color=figstyle.INK)
        T_base = np.linalg.inv(scene.body_pose("base"))
        ap_b = T_base[:3, :3] @ np.asarray(scene.data.xpos[apple_bid], np.float64) + T_base[:3, 3]
        ax2.plot([ap_b[0]], [ap_b[1]], marker="P", ms=13,
                 color=figstyle.CATEGORICAL[0], mec=figstyle.INK, mew=0.8, zorder=6)
        ax2.annotate("apple GT", (ap_b[0], ap_b[1]), textcoords="offset points",
                     xytext=(9, -11), fontsize=7.2, color=figstyle.CATEGORICAL[0])
        ax2.set_xlim(-0.1, 1.2); ax2.set_ylim(-0.75, 0.75)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x [m] (base)", fontsize=8)
        if col == 0:
            # y 축 이름은 왼쪽 row label 이 이미 갖고 있다 — 겹치지 않게 여기서는 비운다
            ax2.legend(fontsize=6.5, loc="lower left", frameon=False, markerscale=4)
        ax2.set_title(f"apple points in fused cloud = {n_fused}", fontsize=7.8,
                      color=(figstyle.CATEGORICAL[7] if n_fused == 0 else figstyle.INK_2))

        side["panels"].append({
            "seq": seq, "t_step": row["t_step"],
            "why": WHY_EN[seq],
            "top_cluster_name": row["top_name"], "top_cluster_score": row["top_score"],
            "apple_rank": row["apple_rank"], "apple_score": row["apple_score"],
            "task_phase": row["task_phase"],
            "gripper_obs_left": row["gripper_obs_left"],
            "gripper_state": grip_state,
            "gripper_closed_below": closed_below,
            "apple_points_after_reconstruct": n_recon,
            "apple_points_after_self_filter": n_filt,
            "apple_points_in_fused_cloud": n_fused,
            "per_camera": {c: {"apple_gt_pixel": st["per_camera"][c]["n_gt_pixel"],
                               "after_reconstruct": st["per_camera"][c]["n_after_recon"],
                               "after_self_filter": st["per_camera"][c]["n_after_filter"],
                               "min_apple_to_sphere_surface_mm":
                                   st["per_camera"][c]["min_apple_to_sphere_surface_mm"]}
                           for c in mi.CAMERAS},
            "offline_grounding_status": row["offline_grounding_status"],
            "offline_has_target": row["offline_has_target"],
            "live_has_target": row["live_has_target"],
            "latch_phase_col7": row["latch_phase_col7"],
            "latch_manipulated_col7": row["latch_manipulated_col7"],
            "clusters_top4": [{"score": float(c.target_score),
                               "point_count": int(c.point_count),
                               "centroid": [float(x) for x in np.asarray(c.centroid)]}
                              for c in g.clusters[:4]],
        })

    for ri, lab in enumerate(ROW_LABEL):
        axes[ri, 0].text(-0.085 if ri < 4 else -0.20, 0.5, lab,
                         transform=axes[ri, 0].transAxes, rotation=90,
                         ha="center", va="center", fontsize=9.5, color=figstyle.INK)

    fig.suptitle(
        ("T5d — 다섯 순간의 실제 scene (ep1807 offline replay, threshold 0.25). "
         f"gripper 는 seq {first_closed} 에 닫히고 seq {first_reopen} 에 열린다"
         if korean else
         "T5d - five moments, real scene (ep1807 offline replay, threshold 0.25). "
         f"gripper closes at seq {first_closed}, reopens at seq {first_reopen}"),
        fontsize=13, color=figstyle.INK)
    fig.tight_layout(rect=(0.02, 0, 1, 0.968))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=figstyle.SURFACE)
    SIDECAR.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    renderer.close()
    scene.close()
    print("wrote", PNG)


if __name__ == "__main__":
    main()
