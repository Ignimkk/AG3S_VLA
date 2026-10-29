"""T1 항목 8 비용 측정 — diagnostic frame card 한 장을 만드는 데 몇 초 걸리나.

**카드를 45 장 만들지 않는다.** 한 장만 만들어 시간을 재고, A0 판정을 기다린다 (T1.task.md).

카드 재료는 이미 있는 것만 쓴다 — 다시 AG3S 를 돌리거나 MuJoCo 를 재생하지 않는다:
  - 기록 npz 의 `image_cam_high` / `image_cam_left_wrist` / `image_cam_right_wrist` (RGB)
  - `outputs/verify/T1/items_ep<EP>.json` 의 그 프레임 숫자 (항목 1~7)

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/cost_probe_frame_card.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AG3S.ag3s.experiments.common import figstyle

RECORDS = "outputs/live_test/20260925_headfix16d/records/run_0000/run_0000"
ITEMS_JSON = "outputs/verify/T1/items_ep1800.json"
FRAME_I = 0
OUT_PNG = "outputs/verify/T1/cost_probe_frame_card.png"
OUT_JSON = "outputs/verify/T1/cost_probe_frame_card.json"


def main() -> None:
    t_setup0 = time.time()
    figstyle.use_korean()
    items = json.loads(pathlib.Path(ITEMS_JSON).read_text())
    frame = items["frames"][FRAME_I]
    t_setup = time.time() - t_setup0

    # --- 카드 한 장 -------------------------------------------------------------------
    t0 = time.time()
    step_npz = sorted(pathlib.Path(RECORDS).glob("step_*.npz"))[FRAME_I]
    with np.load(step_npz) as d:
        img_high = np.asarray(d["image_cam_high"])
        img_l = np.asarray(d["image_cam_left_wrist"])
        img_r = np.asarray(d["image_cam_right_wrist"])
    t_load_images = time.time() - t0

    t1 = time.time()
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.65])
    ax_high = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, :])
    for ax, img, title in ((ax_high, img_high, "cam_high"),
                           (ax_l, img_l, "cam_left_wrist"),
                           (ax_r, img_r, "cam_right_wrist")):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    ax_text.axis("off")
    lines = [
        f"frame {frame['i']}  t_step={frame['t_step']}  phase={frame['phase']}  "
        f"status={frame['status']}  validity={frame['validity']}  "
        f"grounding={frame['grounding_status']}",
        f"target center error: {frame['target_center_error_mm']!r} mm",
        f"support height error: {frame['support_height_error_mm']!r} mm",
        f"cross-camera alignment: {frame['cross_camera_alignment_mm']!r} mm",
        f"robot mask violations: {frame['robot_mask_violations']}  "
        f"{frame['robot_mask_violations_by_body']}",
        f"low-obstacle ESDF (mm): {frame['low_obstacle_esdf_mm']}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10,
                family="monospace", transform=ax_text.transAxes)
    fig.suptitle(f"T1 diagnostic frame card — ep1800 frame {frame['i']}", fontsize=12)
    figstyle.style_axes(fig, [ax_high, ax_l, ax_r, ax_text])
    fig.tight_layout()
    t_build = time.time() - t1

    t2 = time.time()
    pathlib.Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    t_save = time.time() - t2

    total = t_load_images + t_build + t_save
    result = {
        "records": RECORDS, "items_json": ITEMS_JSON, "frame_i": FRAME_I,
        "t_setup_s": t_setup, "t_load_images_s": t_load_images,
        "t_build_figure_s": t_build, "t_save_s": t_save,
        "t_per_card_s": total,
        "estimate_45_cards_s": total * 45,
        "estimate_45_cards_min": total * 45 / 60.0,
        "note": "재료는 기록 npz 의 RGB + items_ep*.json 의 이미 계산된 숫자뿐이다. "
                "AG3S/MuJoCo 를 다시 돌리지 않는다 — 그 비용은 items_ep*.json 을 낼 때 "
                "이미 치렀다 (wall_s, 프레임당 약 2.3~2.9 s).",
    }
    pathlib.Path(OUT_JSON).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
