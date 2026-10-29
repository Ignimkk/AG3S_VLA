"""T8e figure 1 — s_finger 축. z_cut 고정, arm_5 반지름 1.0.

왼쪽 : 남은 위반 행 수 · chunk 수 (exec8 실선 / full50 점선), 손바닥 세 판을 계열로.
가운데: 최소 margin_slack (mm). 0 선이 판정선.
오른쪽: chunk 축 최소 slack 곡선 — s_finger 몇 개를 겹쳐서 (z_cut -100, 손바닥 없음).
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

T8E = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8e/t8e-finger-axis.png'))
SIDE = OUT.with_suffix(".json")
ZCUT = -0.100
VARIANTS = (("out", "palm not in the model"),
            ("in", "palm in, apple kept for ee_left"),
            ("in_authorized", "palm in, apple removed for ee_left"))


def main() -> None:
    d = json.loads((T8E / "s1_cut_plus_finger.json").read_text())
    cur = np.load(T8E / "s1_curves.npz", allow_pickle=True)
    fs = d["finger_scales"]
    rows = d["rows"]

    def get(v, f):
        tag = ("in_authorized" if v == "in_authorized" else v)
        key = f"finger_{f:g}__zcut_100mm__palm_{tag}"
        return [r for r in rows if r["key"] == key][0]

    figstyle.use_korean()
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
    figstyle.style_axes(fig, axes)
    side: dict = {"z_cut_m": ZCUT, "arm5_capsule_radius_scale": 1.0,
                  "finger_scales": fs, "margin_mm": 10.0, "series": []}

    distal_r = [get("out", f)["radius_mm_ee_finger_l1"][-1] for f in fs]

    ax = axes[0]
    for k, (v, lab) in enumerate(VARIANTS):
        for w, ls in (("exec8", "-"), ("full50", "--")):
            y = [get(v, f)["windows"][w]["n_rows_slack_le0"] for f in fs]
            ax.plot(fs, y, ls, marker="o", ms=4, lw=1.6 if w == "exec8" else 1.1,
                    color=figstyle.CATEGORICAL[k], label=f"{lab} · {w}")
            side["series"].append({"variant": v, "window": w, "metric": "n_rows_slack_le0",
                                   "finger_scales": fs, "values": y})
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.set_xlabel("capsule_radius_scale on ee_finger_l1 / ee_finger_l2", fontsize=8.5,
                  color=figstyle.INK_2)
    ax.set_ylabel("remaining rows with slack <= 0  (symlog)", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("rows left after cutting link_*_arm_5 at z = -100 mm\n"
                 "arm_5 radius untouched (scale 1.0), margin 10 mm",
                 fontsize=9.2, color=figstyle.INK)
    ax.legend(fontsize=6.8, loc="lower left", framealpha=0.9)
    ax.invert_xaxis()
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(fs)
    sec.set_xticklabels([f"{v:.1f}" for v in distal_r], fontsize=7.0, color=figstyle.INK_2)
    sec.set_xlabel("distal finger sphere radius  [mm]", fontsize=8.0, color=figstyle.INK_2)

    ax = axes[1]
    for k, (v, lab) in enumerate(VARIANTS):
        for w, ls in (("exec8", "-"), ("full50", "--")):
            y = [get(v, f)["windows"][w]["global_min_slack_mm"] for f in fs]
            ax.plot(fs, y, ls, marker="s", ms=4, lw=1.6 if w == "exec8" else 1.1,
                    color=figstyle.CATEGORICAL[k], label=f"{lab} · {w}")
            side["series"].append({"variant": v, "window": w, "metric": "global_min_slack_mm",
                                   "finger_scales": fs, "values": y})
    ax.axhline(0.0, color=figstyle.INK, lw=1.2)
    ax.set_xlabel("capsule_radius_scale on ee_finger_l1 / ee_finger_l2", fontsize=8.5,
                  color=figstyle.INK_2)
    ax.set_ylabel("min margin_slack over remaining rows  [mm]", fontsize=8.5,
                  color=figstyle.INK_2)
    fe = d["largest_finger_scale_reaching_zero"]["zcut_100mm__palm_out__exec8"]
    ax.set_title(f"min slack - the 0 line is the verdict\n"
                 f"largest finger scale reaching 0 in exec8 (palm out): "
                 f"{'none in the grid' if fe is None else f'x{fe:g}'}",
                 fontsize=9.2, color=figstyle.INK)
    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.9)
    ax.invert_xaxis()

    ax = axes[2]
    for k, f in enumerate((1.0, 0.45, 0.30, 0.22)):
        y = np.asarray(cur[f"finger_{f:g}__zcut_100mm__palm_out__exec8"], np.float64)
        ax.plot(np.arange(y.size), y, lw=1.4, color=figstyle.CATEGORICAL[k % 8],
                label=f"finger x{f:g}  (distal r {get('out', f)['radius_mm_ee_finger_l1'][-1]:.2f} mm)")
        side.setdefault("per_chunk_min_slack_exec8", {})[f"finger_{f:g}"] = [
            round(float(v), 4) for v in y]
    ax.axhline(0.0, color=figstyle.INK, lw=1.2)
    ax.set_xlabel("chunk (shadow, 75 planning records)", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("min margin_slack, exec8  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("per-chunk worst remaining row, exec8\nz_cut -100 mm, palm not in the model",
                 fontsize=9.2, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.9)

    fig.suptitle("T8e - cut link_*_arm_5 at z = -100 mm, then shrink only the fingertips",
                 fontsize=12.5, color=figstyle.INK, y=0.975)
    fig.text(0.5, 0.012,
             "apple rows removed for ee_finger_l1, ee_finger_l2, link_left_arm_5 (and ee_left in "
             "the third series) · the cut is approximated by dropping arm_5 spheres whose "
             "capsule-frame z < -0.100; the remaining spheres keep r = 81.25 mm",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.935))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
