"""T5f figure 1 (실제 씬) — 여섯 chunk 에서 **이전과 이후가 고른 target 이 어느 물체인가**.

위 줄: `zed_left` RGB + 이후가 고른 target cluster 의 점을 그 카메라에 투영(굵은 점) +
       이전이 고른 target cluster centroid 를 O 로. 이전이 `low_score` 면 O 가 없다.
아래 줄: 같은 chunk 의 base frame 평면도(x-y). point 는 MuJoCo 참값 이름으로 색.
       이후 target 의 점을 크게, centroid 를 X 로. apple 참값 자리는 P 로.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/make_fig1_scene.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T5f'))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

import measure_items as mi  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from measure_target_identity_ep1807 import build_body_group  # noqa: E402

RECORDS = "outputs/live_test/20260925_ep1807/run_0000"
ATTENTION = "AG3S/ag3s/asset/data/attention_16d_ep1807.npz"
STEP1 = "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json"
AGG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f/aggregate_T5f.json'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5f'))
PNG = OUTDIR / "t5f-scene-which-object-became-target.png"
SIDECAR = OUTDIR / "t5f-scene-which-object-became-target.json"

PICK = [15, 16, 18, 31, 32, 66]
NAME_SLOT = {"apple": 0, "crate": 1, "table": 2, "pear": 3, "orange": 4,
             "banana": 5, "robot": 6, "unknown": 7}


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
    table = {r["seq"]: r for r in agg["rows"]}

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
    group, _bn = build_body_group(scene.model)
    apple_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
    renderer = mujoco.Renderer(scene.model, scene.height, scene.width)

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)

    fig, axes = plt.subplots(2, len(PICK), figsize=(3.15 * len(PICK), 7.2))
    figstyle.style_axes(fig, axes)
    side = {"records": RECORDS, "picked_seq": PICK,
            "note": ("confirm 은 프레임 순서에 의존하므로 1..75 를 전부 순서대로 돌리고 "
                     "고른 여섯 chunk 만 그린다"),
            "panels": []}

    for i, step in enumerate(run):
        seq = i + 1
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
        if seq not in PICK:
            continue
        col = PICK.index(seq)
        g = debug["grounding"]
        fc = debug["fusion"].cloud
        row = table[seq]

        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(scene.data, camera="zed_left")
        rgb = np.asarray(renderer.render())

        rep = np.asarray(fc.representative, np.int64)
        rep_cam = np.asarray(fc.obs_camera, np.int16)[rep]
        rep_uv = np.asarray(fc.obs_uv, np.int32)[rep]
        cams = [obs_name[cid] for cid in fc.cameras]
        pb = np.full(len(fc), -1, np.int64)
        for ci, cname in enumerate(cams):
            sel = np.nonzero(rep_cam == ci)[0]
            if not sel.size:
                continue
            ids = np.asarray(gt[cname].body_ids)
            u = np.clip(rep_uv[sel, 0], 0, ids.shape[1] - 1)
            v = np.clip(rep_uv[sel, 1], 0, ids.shape[0] - 1)
            pb[sel] = ids[v, u]
        names = np.array([group.get(int(b), "unknown") for b in pb], dtype=object)
        pts = np.asarray(fc.points, np.float64)

        chosen = g.clusters[row["after_rank"] - 1] if row["after_rank"] else None
        after_slot = NAME_SLOT.get(str(row["after_name"]), 7)

        # --- 위: RGB + target 점 투영 ------------------------------------------------------
        ax = axes[0, col]
        ax.imshow(rgb)
        K = gt["zed_left"].camera_intrinsics
        T = np.linalg.inv(np.asarray(gt["zed_left"].T_base_cam, np.float64))
        if chosen is not None:
            tp = pts[np.asarray(chosen.point_indices, np.int64)]
            cam_p = (T[:3, :3] @ tp.T).T + T[:3, 3]
            ok = cam_p[:, 2] > 0.05
            u = K[0, 0] * cam_p[ok, 0] / cam_p[ok, 2] + K[0, 2]
            v = K[1, 1] * cam_p[ok, 1] / cam_p[ok, 2] + K[1, 2]
            ax.scatter(u, v, s=2.0, c=figstyle.CATEGORICAL[after_slot], linewidths=0, alpha=0.9)
        if row["before_name"] is not None:
            bc = np.asarray(g.clusters[0].centroid, np.float64)
            p = T[:3, :3] @ bc + T[:3, 3]
            if p[2] > 0.05:
                ax.plot([K[0, 0] * p[0] / p[2] + K[0, 2]], [K[1, 1] * p[1] / p[2] + K[1, 2]],
                        marker="o", ms=14, mfc="none", mew=2.4, color=figstyle.INK)
        ax.set_xticks([]); ax.set_yticks([])
        bt = row["before_name"] if row["before_name"] else "no target"
        mode_short = {"keep_or_first": "keep", "hold": "HOLD", "switch": "SWITCH",
                      "unobserved": "UNOBS"}[row["after_mode"]]
        ax.set_title(f"seq {seq} · t {row['t_step']} · {row['task_phase']}\n"
                     f"before(0.25): {bt}\n"
                     f"after: {row['after_name']} (rank {row['after_rank']}, {mode_short})",
                     fontsize=8.5, color=figstyle.INK)

        # --- 아래: 평면도 -----------------------------------------------------------------
        ax2 = axes[1, col]
        for nm in ("table", "crate", "robot", "pear", "orange", "banana", "apple"):
            msk = names == nm
            if not msk.any():
                continue
            ax2.scatter(pts[msk, 0], pts[msk, 1], s=0.8,
                        c=figstyle.CATEGORICAL[NAME_SLOT[nm]], linewidths=0, alpha=0.4,
                        label=nm if col == 0 else None, zorder=1)
        if chosen is not None:
            tp = pts[np.asarray(chosen.point_indices, np.int64)]
            ax2.scatter(tp[:, 0], tp[:, 1], s=5.0, c=figstyle.CATEGORICAL[after_slot],
                        linewidths=0, alpha=1.0, zorder=3)
            c3 = np.asarray(chosen.centroid, np.float64)
            ax2.plot([c3[0]], [c3[1]], marker="X", ms=9, color=figstyle.INK, zorder=5,
                     ls="none")
        T_base = np.linalg.inv(scene.body_pose("base"))
        ap = T_base[:3, :3] @ np.asarray(scene.data.xpos[apple_bid], np.float64) + T_base[:3, 3]
        ax2.plot([ap[0]], [ap[1]], marker="P", ms=9, color=figstyle.CATEGORICAL[0],
                 mec=figstyle.INK, mew=0.7, zorder=6, ls="none")
        ax2.set_xlim(0.15, 1.0); ax2.set_ylim(-0.62, 0.62); ax2.set_aspect("equal")
        ax2.set_xlabel("x [m] (base)", fontsize=8)
        if col == 0:
            ax2.set_ylabel("y [m] (base)", fontsize=8)
            handles = [Line2D([], [], ls="none", marker="o", ms=4,
                              color=figstyle.CATEGORICAL[NAME_SLOT[n]], label=n)
                       for n in ("table", "crate", "pear", "orange", "banana", "apple")]
            handles += [Line2D([], [], ls="none", marker="X", ms=7, color=figstyle.INK,
                               label="after target centroid"),
                        Line2D([], [], ls="none", marker="P", ms=7,
                               color=figstyle.CATEGORICAL[0], mec=figstyle.INK, mew=0.7,
                               label="apple GT")]
            ax2.legend(handles=handles, fontsize=6.5, loc="upper left", frameon=False,
                       ncol=4, bbox_to_anchor=(0.0, -0.20), columnspacing=1.1)
        ax2.set_title(f"target {row['after_point_count']} pts · rms "
                      f"{row['after_rms_radius_mm']:.0f} mm · purity "
                      f"{row['after_name_purity']:.2f}\n"
                      f"GT min clearance {row['gt_min_clearance_mm']:+.2f} mm "
                      f"({row['gt_min_pair'][0]} vs {row['gt_min_pair'][1]})",
                      fontsize=7.0, color=figstyle.INK_2)

        side["panels"].append({k: row[k] for k in (
            "seq", "t_step", "task_phase", "rank1_name", "rank1_score", "before_name",
            "before_status", "after_name", "after_rank", "after_score", "after_mode",
            "after_confirm_streak", "after_point_count", "after_rms_radius_mm",
            "after_name_purity", "after_name_counts", "n_clusters",
            "gt_min_clearance_mm", "gt_min_pair")})

    title = ("T5f — 1 등 + confirm 3 이 고른 target 은 어느 물체였나 (ep1807 offline replay)"
             if korean else
             "T5f - which object the rank-1 + confirm-3 target actually was (ep1807 replay)")
    fig.suptitle(title, fontsize=12, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=170, facecolor=figstyle.SURFACE)
    SIDECAR.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    renderer.close()
    scene.close()
    print("wrote", PNG)


if __name__ == "__main__":
    main()
