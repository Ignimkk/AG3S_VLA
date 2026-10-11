"""T1 항목 4·5 — self_filter_inflation 0.02 -> 0.05 전후 비교 표.

`outputs/verify/T1/items_ep*.json` (전) 과 `items_ep*_after_inflation_005.json` (후),
`item4_attr_ep*.json` / `item4_attr_ep*_after_inflation_005.json` 을 읽는다. 해석 없음.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/make_inflation_compare_figure.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AG3S.ag3s.experiments.common import figstyle

EPISODES = ["ep1800", "ep1850", "ep1900"]
OUT_DIR = pathlib.Path("AG3S/ag3s/docs/figures/t1")


def main() -> None:
    figstyle.use_korean()
    items_before = {ep: json.loads(pathlib.Path(f"outputs/verify/T1/items_ep{ep[2:]}.json").read_text())
                    for ep in EPISODES}
    items_after = {ep: json.loads(
        pathlib.Path(f"outputs/verify/T1/items_ep{ep[2:]}_after_inflation_005.json").read_text())
        for ep in EPISODES}
    attr_before = {ep: json.loads(pathlib.Path(f"outputs/verify/T1/item4_attr_{ep}.json").read_text())
                  for ep in EPISODES}
    attr_after = {ep: json.loads(
        pathlib.Path(f"outputs/verify/T1/item4_attr_{ep}_after_inflation_005.json").read_text())
        for ep in EPISODES}

    rows = []
    for ep in EPISODES:
        b, a = items_before[ep], items_after[ep]
        rows.append([ep, "robot_mask_violations_total",
                    str(b["summary"]["item4_robot_mask_violations_total"]),
                    str(a["summary"]["item4_robot_mask_violations_total"])])
        ab_by = attr_before[ep]["summary"]["leaked_px_by_camera_by_body"]
        aa_by = attr_after[ep]["summary"]["leaked_px_by_camera_by_body"]
        base_before = sum(v.get("base", 0) for v in ab_by.values())
        base_after = sum(v.get("base", 0) for v in aa_by.values())
        rows.append([ep, "base leak (px, total)", str(base_before), str(base_after)])
        ee_before = attr_before[ep]["summary"]["esdf_at_leaked_pooled_by_body"]
        ee_after = attr_after[ep]["summary"]["esdf_at_leaked_pooled_by_body"]
        for body in ("EE_BODY_L", "EE_BODY_R"):
            fb = ee_before.get(body, {}).get("frac_occupied_le0_weighted")
            fa = ee_after.get(body, {}).get("frac_occupied_le0_weighted")
            rows.append([ep, f"{body} frac_occupied_le0",
                        "-" if fb is None else f"{fb:.4f}", "-" if fa is None else f"{fa:.4f}"])
        rows.append([ep, "item6 has_target",
                    str(sum(1 for f in b["frames"] if f["has_target"])) + "/15",
                    str(sum(1 for f in a["frames"] if f["has_target"])) + "/15"])
        # item5: 비-target 과일 esdf 거리 (frame0)
        f0b = b["frames"][0]["low_obstacle_esdf_mm"]
        f0a = a["frames"][0]["low_obstacle_esdf_mm"]
        for obj in f0b:
            rows.append([ep, f"item5 {obj} esdf mm (frame0)",
                        f"{f0b[obj]:.2f}", f"{f0a[obj]:.2f}"])

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["episode", "measure", "before (0.02)", "after (0.05)"],
                loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.4)
    ax.set_title("T1 항목 4·5 — self_filter_inflation 0.02 -> 0.05 전후", fontsize=12)
    fig.tight_layout()
    out_png = OUT_DIR / "t1-inflation-005-compare.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    (OUT_DIR / "t1-inflation-005-compare.json").write_text(
        json.dumps({"rows": rows, "columns": ["episode", "measure", "before_0.02", "after_0.05"]},
                  indent=1, ensure_ascii=False))
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
