"""T7a figure 2 — 그래프. chunk 축 × link 별 `clearance - required_margin`.

(a),(b) heatmap: 왼팔 link 마다 chunk 별 최소 margin_slack (exec window 8 스텝).
        위 = shadow 가 실행한 reference, 아래 = closed-loop 가 실행한 refined. 같은 색눈금.
(c) chunk 별 **최소** margin_slack 세 곡선 + 기록의 ESDF clearance (`max_violation_pair`).
(d) link_left_arm_5 · ee_finger_l1 · ee_finger_l2 의 chunk 곡선과 손끝-사과 거리.
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

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t7a/t7a-graphs.png'))
SIDE = OUT.with_suffix(".json")
SEL = "no_target__left_spheres"


def main() -> None:
    s1 = json.loads((V / "s1_active_rows.json").read_text())
    s2 = json.loads((V / "s2_joint_axis.json").read_text())
    s3 = json.loads((V / "s3_divergence_context.json").read_text())
    t = np.load(V / "s2_tables.npz", allow_pickle=True)
    links = [str(x) for x in t["link_axis"]]
    left = [ln for ln in links if "left" in ln or ln.startswith("ee_finger_l")]
    li = [links.index(ln) for ln in left]
    ma = t[f"shadow_approach__reference__{SEL}"][:, li]
    mb = t[f"closed_loop_approach__refined__{SEL}"][:, li]
    n = ma.shape[0]

    a = s2["runs"]["shadow_approach"][f"reference__{SEL}"]
    ar = s2["runs"]["shadow_approach"][f"refined__{SEL}"]
    b = s2["runs"]["closed_loop_approach"][f"refined__{SEL}"]
    rec_sh = [c["recorded"]["max_violation_pair_clearance_mm"]
              for c in s1["runs"]["shadow_approach"]["chunks"]]
    rec_cl = [c["recorded"]["max_violation_pair_clearance_mm"]
              for c in s1["runs"]["closed_loop_approach"]["chunks"]]

    figstyle.use_korean()
    fig, axes = plt.subplots(4, 1, figsize=(15.0, 13.6),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.15, 1.15]})
    figstyle.style_axes(fig, axes)
    vmax = 100.0   # 색눈금은 +-100 mm 에서 자른다 (0 근처 구조를 보려고). 실제 범위는 sidecar 에.

    for k, (mat, ttl) in enumerate(((ma, "(a) shadow — executed reference chunk"),
                                    (mb, "(b) closed-loop — executed refined chunk"))):
        ax = axes[k]
        im = ax.imshow(mat.T, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax,
                       interpolation="nearest", origin="upper", extent=(-0.5, n - 0.5,
                                                                        len(left) - 0.5, -0.5))
        ax.set_yticks(range(len(left)))
        ax.set_yticklabels(left, fontsize=7.5)
        ax.set_title(f"{ttl} — min (clearance - required_margin) per link, mm "
                     "(exec window 8 steps, apple excluded; colour clipped at +-100 mm)",
                     fontsize=10, color=figstyle.INK, loc="left")
        ax.set_xlabel("chunk index (planning seq - 1)", fontsize=8.5, color=figstyle.INK_2)
        cb = fig.colorbar(im, ax=ax, pad=0.008, fraction=0.03, extend="both")
        cb.set_label("mm", fontsize=8, color=figstyle.INK_2)
        cb.ax.tick_params(labelsize=7)

    ax = axes[2]
    ax.axhline(0.0, color=figstyle.INK_2, lw=1.0, ls="--")
    ax.plot(a["chunk_min_slack_mm"], color=figstyle.CATEGORICAL[0], lw=1.6,
            label="shadow · executed reference (offline ground truth)")
    ax.plot(ar["chunk_min_slack_mm"], color=figstyle.CATEGORICAL[4], lw=1.0, ls=":",
            label="shadow · TO refined (discarded)")
    ax.plot(b["chunk_min_slack_mm"], color=figstyle.CATEGORICAL[1], lw=1.6,
            label="closed-loop · executed refined (offline ground truth)")
    ax.plot(rec_sh, color=figstyle.CATEGORICAL[0], lw=1.0, marker=".", ms=3, alpha=0.55,
            ls="-.", label="shadow · recorded ESDF max_violation_pair clearance")
    ax.plot(rec_cl, color=figstyle.CATEGORICAL[1], lw=1.0, marker=".", ms=3, alpha=0.55,
            ls="-.", label="closed-loop · recorded ESDF max_violation_pair clearance")
    ax.set_ylim(-90, 130)
    ax.set_ylabel("clearance - required_margin  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_xlabel("chunk index", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("(c) tightest row per chunk — offline ground truth vs the value the run recorded",
                 fontsize=10, color=figstyle.INK, loc="left")
    ax.legend(fontsize=7.6, ncol=2, loc="upper right", framealpha=0.92)

    ax = axes[3]
    ax.axhline(0.0, color=figstyle.INK_2, lw=1.0, ls="--")
    for k, ln in enumerate(("link_left_arm_5", "ee_finger_l1", "ee_finger_l2")):
        j = left.index(ln)
        ax.plot(ma[:, j], color=figstyle.CATEGORICAL[k], lw=1.5, label=f"shadow · {ln}")
        ax.plot(mb[:, j], color=figstyle.CATEGORICAL[k], lw=1.5, ls="--",
                label=f"closed-loop · {ln}")
    ax.plot(s3["shadow_reference_tip_mm"], color=figstyle.INK_2, lw=1.2, alpha=0.8,
            label="shadow · fingertip to apple surface")
    ax.plot(s3["closed_loop_refined_tip_mm"], color=figstyle.INK_2, lw=1.2, ls="--", alpha=0.8,
            label="closed-loop · fingertip to apple surface")
    ax.axvline(s3["first_chunk_shadow_tip_below_0mm"], color=figstyle.CATEGORICAL[5], lw=1.2)
    ax.annotate(f"chunk {s3['first_chunk_shadow_tip_below_0mm']} — shadow fingertip first reaches "
                "the apple surface", (s3["first_chunk_shadow_tip_below_0mm"] + 0.6, 205),
                fontsize=8, color=figstyle.CATEGORICAL[5])
    ax.set_ylim(-90, 230)
    ax.set_ylabel("mm", fontsize=8.5, color=figstyle.INK_2)
    ax.set_xlabel("chunk index", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("(d) the three links that hold the tightest rows, and where the hand ends up",
                 fontsize=10, color=figstyle.INK, loc="left")
    ax.legend(fontsize=7.4, ncol=4, loc="lower right", framealpha=0.92)

    fig.suptitle("T7a — active collision rows per chunk. Violation is 0 in the closed-loop run, "
                 "yet its tightest row sits at +0.76 mm",
                 fontsize=12.5, color=figstyle.INK, y=0.995)
    fig.tight_layout(rect=(0, 0.004, 1, 0.977))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)

    side = {
        "selection": SEL, "exec_window_steps": 8, "required_margin_mm": 50.0,
        "link_axis": left,
        "heatmap_shadow_reference_mm": ma.round(4).tolist(),
        "heatmap_closed_loop_refined_mm": mb.round(4).tolist(),
        "chunk_min_slack_shadow_reference_mm": a["chunk_min_slack_mm"],
        "chunk_min_slack_shadow_refined_mm": ar["chunk_min_slack_mm"],
        "chunk_min_slack_closed_loop_refined_mm": b["chunk_min_slack_mm"],
        "recorded_clearance_shadow_mm": rec_sh,
        "recorded_clearance_closed_loop_mm": rec_cl,
        "fingertip_to_apple_shadow_mm": s3["shadow_reference_tip_mm"],
        "fingertip_to_apple_closed_loop_mm": s3["closed_loop_refined_tip_mm"],
        "first_chunk_shadow_tip_below_0mm": s3["first_chunk_shadow_tip_below_0mm"],
        "comparison": s2["comparison"],
    }
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
