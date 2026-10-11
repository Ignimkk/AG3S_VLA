"""T5d figure 1 (실제 씬) — 여섯 frame 에서 attention 이 무엇 위에 앉았나.

위 줄: `zed_left` RGB + apple 참값 mask 윤곽 + 1 등 cluster centroid 를 그 카메라에 투영.
아래 줄: 같은 frame 의 base frame 평면도(x-y). point 는 MuJoCo 참값 이름으로 색을 주고,
cluster centroid 를 점수와 함께 찍는다. apple 참값 자리는 X 로.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/make_fig1_scene.py
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
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5d'))
PNG = OUTDIR / "t5d-scene-attention-handoff.png"
SIDECAR = OUTDIR / "t5d-scene-attention-handoff.json"

PICK = [1, 15, 18, 25, 33, 40]
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
    table = {r["seq"]: r for r in agg["frame_table_seq_1_51"]}

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
    group, _body_names = build_body_group(scene.model)
    apple_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "apple")
    renderer = mujoco.Renderer(scene.model, scene.height, scene.width)

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)

    fig, axes = plt.subplots(2, len(PICK), figsize=(3.05 * len(PICK), 6.6))
    figstyle.style_axes(fig, axes)
    side = {"records": RECORDS, "picked_seq": PICK, "panels": []}

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

        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(scene.data, camera="zed_left")
        rgb = np.asarray(renderer.render())

        row = table[seq]
        ax = axes[0, col]
        ax.imshow(rgb)
        ax.contour((gt["zed_left"].body_ids == apple_bid).astype(float), levels=[0.5],
                   colors=[figstyle.CATEGORICAL[0]], linewidths=1.4)
        K = gt["zed_left"].camera_intrinsics
        T = np.linalg.inv(np.asarray(gt["zed_left"].T_base_cam, np.float64))
        proj = []
        for cl in g.clusters[:3]:
            p = T[:3, :3] @ np.asarray(cl.centroid, np.float64) + T[:3, 3]
            if p[2] <= 0.05:
                continue
            u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
            proj.append((float(u), float(v), float(cl.target_score)))
        if proj:
            u, v, s = proj[0]
            ax.plot([u], [v], marker="o", ms=11, mfc="none", mew=2.2,
                    color=figstyle.CATEGORICAL[NAME_SLOT.get(str(row["top_name"]), 7)])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"seq {seq} · t {row['t_step']}\n"
                     f"top {row['top_name']} {row['top_score']:.3f} · {row['task_phase']}",
                     fontsize=9, color=figstyle.INK)

        ax2 = axes[1, col]
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
        for nm in ("table", "crate", "robot", "pear", "orange", "banana", "apple"):
            m = names == nm
            if not m.any():
                continue
            ax2.scatter(pts[m, 0], pts[m, 1], s=(3.5 if nm == "apple" else 0.7),
                        c=figstyle.CATEGORICAL[NAME_SLOT[nm]], linewidths=0,
                        alpha=(1.0 if nm == "apple" else 0.5),
                        label=nm if col == 0 else None, zorder=(3 if nm == "apple" else 1))
        for cl in g.clusters[:3]:
            c = np.asarray(cl.centroid, np.float64)
            ax2.plot([c[0]], [c[1]], marker="x", ms=7, mew=1.8, color=figstyle.INK, zorder=5)
            ax2.annotate(f"{cl.target_score:.2f}", (c[0], c[1]), textcoords="offset points",
                         xytext=(5, 4), fontsize=7, color=figstyle.INK)
        T_base = np.linalg.inv(scene.body_pose("base"))
        ap_w = np.asarray(scene.data.xpos[apple_bid], np.float64)
        ap_b = T_base[:3, :3] @ ap_w + T_base[:3, 3]
        ax2.plot([ap_b[0]], [ap_b[1]], marker="P", ms=9,
                 color=figstyle.CATEGORICAL[0], mec=figstyle.INK, mew=0.7, zorder=6)
        ax2.set_xlim(-0.2, 1.3); ax2.set_ylim(-0.9, 0.9)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x [m] (base)", fontsize=8)
        if col == 0:
            ax2.set_ylabel("y [m] (base)", fontsize=8)
            ax2.legend(fontsize=6, loc="lower left", frameon=False, markerscale=4)
        ax2.set_title(f"apple pts in cloud = {row['apple_cloud_points']}",
                      fontsize=8, color=figstyle.INK_2)

        side["panels"].append({
            "seq": seq, "t_step": row["t_step"], "task_phase": row["task_phase"],
            "top_name": row["top_name"], "top_score": row["top_score"],
            "apple_rank": row["apple_rank"], "apple_score": row["apple_score"],
            "apple_cloud_points": row["apple_cloud_points"],
            "n_clusters": row["n_clusters"],
            "cluster_top3": [{"score": float(c.target_score), "point_count": int(c.point_count),
                              "centroid": [float(x) for x in np.asarray(c.centroid)]}
                             for c in g.clusters[:3]],
            "offline_grounding_status": row["offline_grounding_status"],
            "live_has_target": row["live_has_target"],
        })

    title = ("T5d — attention 이 무엇 위에 앉았나 (ep1807 offline replay, threshold 0.25)"
             if korean else
             "T5d - what attention sits on (ep1807 offline replay, threshold 0.25)")
    fig.suptitle(title, fontsize=12, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=170, facecolor=figstyle.SURFACE)
    SIDECAR.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    renderer.close()
    scene.close()
    print("wrote", PNG)


if __name__ == "__main__":
    main()
