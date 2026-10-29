"""T1 항목 8 — diagnostic frame card, 관측 프레임마다 한 장 (45 = 15 x 3 에피소드).

비용은 이미 쟀다 (`cost_probe_frame_card.py` — 카드당 0.22 s, A0 판정 완료). 재료는 그때와
같다: 기록 npz 의 RGB + `items_ep<EP>.json` 의 그 프레임 숫자(항목 1~7). AG3S/MuJoCo 를
다시 돌리지 않는다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/make_frame_cards.py
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

EPISODES = [
    {"id": "ep1800", "target": "apple",
     "records": "outputs/live_test/20260925_headfix16d/records/run_0000/run_0000",
     "items_json": "outputs/verify/T1/items_ep1800.json"},
    {"id": "ep1850", "target": "banana",
     "records": "outputs/live_test/20260925_t1/ep1850/run_0000",
     "items_json": "outputs/verify/T1/items_ep1850.json"},
    {"id": "ep1900", "target": "pear",
     "records": "outputs/live_test/20260925_t1/ep1900/run_0000",
     "items_json": "outputs/verify/T1/items_ep1900.json"},
]

OUT_ROOT = pathlib.Path("AG3S/ag3s/docs/figures/t1/cards")


def make_card(ep: dict, frame: dict, step_npz: pathlib.Path, out_png: pathlib.Path) -> None:
    with np.load(step_npz) as d:
        img_high = np.asarray(d["image_cam_high"])
        img_l = np.asarray(d["image_cam_left_wrist"])
        img_r = np.asarray(d["image_cam_right_wrist"])

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
        f"{ep['id']} ({ep['target']})  frame {frame['i']}  t_step={frame['t_step']}  "
        f"phase={frame['phase']}",
        f"status={frame['status']}  validity={frame['validity']}  "
        f"grounding={frame['grounding_status']}  has_target={frame['has_target']}",
        f"target center error: {frame['target_center_error_mm']!r} mm",
        f"support height error: {frame['support_height_error_mm']!r} mm  "
        f"(n_support_surfaces={frame['n_support_surfaces']})",
        f"cross-camera alignment: {frame['cross_camera_alignment_mm']!r} mm",
        f"robot mask violations: {frame['robot_mask_violations']}  "
        f"{frame['robot_mask_violations_by_body']}",
        f"low-obstacle ESDF (mm): {frame['low_obstacle_esdf_mm']}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10,
                family="monospace", transform=ax_text.transAxes)
    fig.suptitle(f"T1 diagnostic frame card — {ep['id']} frame {frame['i']}", fontsize=12)
    figstyle.style_axes(fig, [ax_high, ax_l, ax_r, ax_text])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    figstyle.use_korean()
    t0 = time.time()
    manifest = []
    n_cards = 0
    for ep in EPISODES:
        items = json.loads(pathlib.Path(ep["items_json"]).read_text())
        step_files = sorted(pathlib.Path(ep["records"]).glob("step_*.npz"))
        out_dir = OUT_ROOT / ep["id"]
        for frame in items["frames"]:
            i = frame["i"]
            out_png = out_dir / f"frame_{i:02d}.png"
            make_card(ep, frame, step_files[i], out_png)
            manifest.append({"episode": ep["id"], "frame_i": i, "path": str(out_png)})
            n_cards += 1
    dt = time.time() - t0

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps({
        "n_cards": n_cards, "wall_s": dt, "cards": manifest,
        "source": "items_ep{1800,1850,1900}.json 의 프레임 숫자 + 기록 npz 의 RGB. "
                  "AG3S/MuJoCo 를 다시 돌리지 않았다",
    }, indent=1, ensure_ascii=False))
    print(f"wrote {n_cards} cards in {dt:.2f} s ({dt/n_cards:.3f} s/card) -> {OUT_ROOT}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
