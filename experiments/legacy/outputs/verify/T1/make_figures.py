"""T1 항목 1~7 figure 3종 (규칙 A). 항목 8(diagnostic frame card)은 여기 없다 — A0 판정 대기.

`outputs/verify/T1/items_ep*.json` (항목 1~7 raw) 을 읽고, 실제 씬 그림을 위해서만 각
에피소드에서 target 이 잡힌 첫 프레임 하나를 다시 돌려 점군을 뽑는다 (`benchmark/**` 는
건드리지 않는다 — `measure_items.py` 와 같은 진입점을 그대로 다시 부른다).

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/make_figures.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AG3S.ag3s.experiments.common import figstyle

import measure_items as mi

OUT_DIR = pathlib.Path("AG3S/ag3s/docs/figures/t1")
RAW_DIR = pathlib.Path("outputs/verify/T1")

EPISODES = [
    {"id": "ep1800", "target": "apple",
     "records": "outputs/live_test/20260925_headfix16d/records/run_0000/run_0000",
     "attention": "AG3S/ag3s/asset/data/attention_16d_headfix.npz"},
    {"id": "ep1850", "target": "banana",
     "records": "outputs/live_test/20260925_t1/ep1850/run_0000",
     "attention": "AG3S/ag3s/asset/data/attention_16d_ep1850.npz"},
    {"id": "ep1900", "target": "pear",
     "records": "outputs/live_test/20260925_t1/ep1900/run_0000",
     "attention": "AG3S/ag3s/asset/data/attention_16d_ep1900.npz"},
]


def load_items():
    return {e["id"]: json.loads((RAW_DIR / f"items_{e['id']}.json").read_text()) for e in EPISODES}


def scene_data_for_episode(ep: dict, frame_i: int):
    """`process_multi_debug` 를 그 프레임 하나만 다시 불러 점군을 뽑는다."""
    import mujoco
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)

    run = load_run(ep["records"], limit=frame_i + 1)
    boundaries, _ = phase_boundaries_for_path(ep["records"])
    blob = np.load(ep["attention"], allow_pickle=False)
    cell = json.loads(pathlib.Path(mi.__file__).parent.joinpath(
        "..", "..", "..", "AG3S/ag3s/docs/archive/step-verification-20260904/"
        "step-01-attention.json").resolve().read_text()) if False else json.loads(
        pathlib.Path("AG3S/ag3s/docs/archive/step-verification-20260904/"
                     "step-01-attention.json").read_text())["best"]
    attention_block = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    names_in_npz = [str(c) for c in blob["cameras"]]
    attention_camera_index = {mujoco_name: names_in_npz.index(policy_name)
                              for policy_name, mujoco_name in mi.ATTENTION_ALIAS.items()
                              if policy_name in names_in_npz}

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, ep["target"])
    ag_cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)

    step = run.steps[frame_i]
    pose_scene(scene, step)
    att_by_cam = {cam: attention_block[frame_i, di, ai, cell["layer"], cell["head"], ci]
                  for cam, ci in attention_camera_index.items()}
    observations = [camera_observation(scene, c, filter_robot, timestamp=float(frame_i),
                                       attention_map=att_by_cam.get(c))[0]
                    for c in mi.CAMERAS]
    phase = phase_for(step.t_step, boundaries)
    cs, debug = ag.process_multi_debug(observations, phase=phase)
    cloud = debug["filtered_cloud"]
    gt_center = mi.backproject_body(scene.capture("zed_left"), body_id)
    scene.close()

    horiz = mi.horizontal_surface_heights(cs.support_surfaces, cloud.points)
    table_surf = max(horiz, key=lambda h: h["point_count"], default=None)
    table_points = None
    if table_surf is not None:
        for s in cs.support_surfaces:
            if int(s.id) == table_surf["surface_id"]:
                table_points = cloud.points[np.asarray(s.point_indices, np.int64)]
                break
    return {
        "cloud_points": cloud.points,
        "target_points": (np.asarray(cs.target.points) if cs.target is not None else None),
        "target_centroid": (np.asarray(cs.target.centroid) if cs.target is not None else None),
        "table_points": table_points,
    }


def fig_scene(items):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ep in zip(axes, EPISODES):
        d = items[ep["id"]]
        frame_i = next(f["i"] for f in d["frames"] if f["has_target"])
        sd = scene_data_for_episode(ep, frame_i)
        pts = sd["cloud_points"]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=figstyle.GRID_INK, alpha=0.5,
                  label="filtered cloud")
        if sd["table_points"] is not None:
            ax.scatter(sd["table_points"][:, 0], sd["table_points"][:, 1], s=1.0,
                      c=figstyle.SEQ_BLUE[3], alpha=0.6, label="table surface (inliers)")
        if sd["target_points"] is not None:
            ax.scatter(sd["target_points"][:, 0], sd["target_points"][:, 1], s=6,
                      c=figstyle.CATEGORICAL[0], label=f"{ep['target']} (fused target cluster)")
        ax.scatter(*sd["target_centroid"][:2], marker="x", s=90, c=figstyle.CATEGORICAL[1],
                  linewidths=2.5, label="AG3S target centroid")
        f = next(f for f in d["frames"] if f["i"] == frame_i)
        gt = f["gt_target_center_base"]
        ax.scatter(gt[0], gt[1], marker="*", s=160, c=figstyle.CATEGORICAL[3],
                  edgecolors=figstyle.INK, linewidths=0.5, label="MuJoCo GT surface center")
        ax.set_title(f"{ep['id']} ({ep['target']})  frame {frame_i}\n"
                    f"target err {f['target_center_error_mm']:.1f} mm", fontsize=10)
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_aspect("equal")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("T1 항목 1·2 — 실제 씬: target 중심 · 지지면 (bird's-eye, base frame)")
    figstyle.style_axes(fig, axes)
    fig.tight_layout()
    return fig


def fig_graph(items):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("target_center_error_mm", "target 중심 오차 (mm)", axes[0, 0]),
        ("support_height_error_mm", "지지면 높이 오차 (mm)", axes[0, 1]),
        ("cross_camera_alignment_mm", "카메라 간 표면 정렬 오차 (mm)", axes[1, 0]),
        ("robot_mask_violations", "로봇 마스크 위반 (픽셀 수)", axes[1, 1]),
    ]
    for key, title, ax in metrics:
        for ci, ep in enumerate(EPISODES):
            frames = items[ep["id"]]["frames"]
            xs = [f["i"] for f in frames]
            ys = [f[key] for f in frames]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.2, c=figstyle.CATEGORICAL[ci],
                   label=f"{ep['id']} ({ep['target']})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame i")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("T1 항목 1·2·3·4 — 프레임별 추이 (세 에피소드)")
    figstyle.style_axes(fig, axes.ravel())
    fig.tight_layout()
    return fig


def fig_table(items):
    rows = [
        ("target center error (mm), median", "item1_target_center_error_mm", "median"),
        ("target center error (mm), max|.|", "item1_target_center_error_mm", "max_abs"),
        ("support height error (mm), median", "item2_support_height_error_mm", "median"),
        ("cross-camera alignment (mm), median", "item3_cross_camera_alignment_mm", "median"),
        ("robot mask violations, total (px)", "item4_robot_mask_violations_total", None),
    ]
    cols = [ep["id"] for ep in EPISODES]
    cell = []
    for label, key, sub in rows:
        row = [label]
        for ep in EPISODES:
            s = items[ep["id"]]["summary"][key]
            v = s if sub is None else s[sub]
            row.append("-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v))
        cell.append(row)
    status_rows = []
    for ep in EPISODES:
        c = items[ep["id"]]["summary"]["item6_status_counts"]
        v = items[ep["id"]]["summary"]["item6_validity_counts"]
        g = items[ep["id"]]["summary"]["item6_grounding_status_counts"]
        comp = items[ep["id"]]["summary"]["item7_completeness"]
        status_rows.append([ep["id"], json.dumps(c, ensure_ascii=False),
                           json.dumps(v, ensure_ascii=False), json.dumps(g, ensure_ascii=False),
                           f"{comp['n_processed_frames']}/{comp['n_captured_observation_frames']}"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5),
                                  gridspec_kw={"height_ratios": [1.0, 0.8]})
    for ax in (ax1, ax2):
        ax.axis("off")
    t1 = ax1.table(cellText=cell, colLabels=["항목 1~4"] + cols, loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1, 1.6)
    ax1.set_title("T1 항목 1~4 요약 (median / total)", fontsize=11)

    t2 = ax2.table(cellText=status_rows,
                  colLabels=["episode", "status(6)", "validity(6)", "grounding(6)",
                           "processed/captured(7)"],
                  loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(8)
    t2.scale(1, 2.0)
    ax2.set_title("T1 항목 6·7 — 입력 부족 처리 분포 · completeness", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figstyle.use_korean()
    items = load_items()

    numbers = {ep["id"]: items[ep["id"]]["summary"] for ep in EPISODES}

    fig1 = fig_scene(items)
    fig1.savefig(OUT_DIR / "t1-items-scene.png", dpi=140)
    (OUT_DIR / "t1-items-scene.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False))
    plt.close(fig1)
    print(f"wrote {OUT_DIR / 't1-items-scene.png'}")

    fig2 = fig_graph(items)
    fig2.savefig(OUT_DIR / "t1-items-trend.png", dpi=140)
    (OUT_DIR / "t1-items-trend.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False))
    plt.close(fig2)
    print(f"wrote {OUT_DIR / 't1-items-trend.png'}")

    fig3 = fig_table(items)
    fig3.savefig(OUT_DIR / "t1-items-table.png", dpi=140)
    (OUT_DIR / "t1-items-table.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False))
    plt.close(fig3)
    print(f"wrote {OUT_DIR / 't1-items-table.png'}")


if __name__ == "__main__":
    main()
