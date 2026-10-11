"""T9b figure 1 — 실제 씬에 **두 손끝 경로를 겹쳐** 그린다.

왼쪽: chunk 14 (손이 가장 가까웠던 chunk, 원본 손끝이 사과 표면 +2.16 mm).
오른쪽: chunk 15 (최근접 link 이 두 변형에서 갈리는 chunk).

위 줄  MuJoCo 렌더 — 그 chunk 의 기록 자세에 원본 실행 창 마지막 step 을 얹은 것.
가운데 base frame **XZ** 투영 · 아래 줄 base frame **XY** (위에서 본 것).
경로는 `ee_finger_l1` body 원점이다. 실행 창 8 step 은 굵게, 나머지 42 step 은 얇게.
TO 는 실행 창만 고치므로 8 step 이후 두 경로는 겹친다.
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
from matplotlib.patches import Circle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from m3_to_refined_vs_reference import ACT_COL, MODEL_XML, QADR  # noqa: E402

REC = _ag3s_workspace_path('outputs/live_test/20260926_t9/execute_ep1807/frames.jsonl')
T9B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9b'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t9b/t9b-scene.png'))
SIDE = OUT.with_suffix(".json")
EXEC_H = 8
APPLE_R = 0.0333
PANELS = [14, 15]


def main() -> None:
    import mujoco

    ana = json.loads((T9B / "n1to5.json").read_text())
    recs = [json.loads(l) for l in open(REC)]
    P = [r for r in recs if r["kind"] == "planning"]
    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")
    tip2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l2")
    apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")

    def path(rec, chunk_arr):
        qpos = np.asarray(rec["qpos"], np.float64)
        p1 = np.empty((chunk_arr.shape[0], 3))
        p2 = np.empty((chunk_arr.shape[0], 3))
        for s in range(chunk_arr.shape[0]):
            d.qpos[: qpos.size] = qpos
            for n, c in ACT_COL.items():
                d.qpos[QADR[n]] = float(chunk_arr[s, c])
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            p1[s] = d.xpos[tip_id]
            p2[s] = d.xpos[tip2_id]
        return p1, p2

    renderer = mujoco.Renderer(m, height=470, width=620)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    figstyle.use_korean()
    fig, axes = plt.subplots(3, 2, figsize=(14.6, 15.2),
                             gridspec_kw={"height_ratios": [1.0, 1.05, 1.05]})
    figstyle.style_axes(fig, axes[1:])
    side: dict = {"exec_window": EXEC_H, "tip_body": "ee_finger_l1", "panels": []}

    for col, ch in enumerate(PANELS):
        rec = P[ch]
        ref = np.asarray(rec["actions_reference"], np.float64)
        ref_o = np.asarray(rec["actions"], np.float64)
        qpos = np.asarray(rec["qpos"], np.float64)
        p_ref, p_ref2 = path(rec, ref)
        p_to, p_to2 = path(rec, ref_o)

        d.qpos[: qpos.size] = qpos
        for n, c in ACT_COL.items():
            d.qpos[QADR[n]] = float(ref[EXEC_H - 1, c])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        apple = np.asarray(d.xpos[apple_bid], np.float64)

        cam.lookat[:] = [float(apple[0]), float(apple[1]), float(apple[2]) + 0.08]
        cam.distance, cam.azimuth, cam.elevation = 0.58, 128.0, -14.0
        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        nl = ana["n1_link_to_apple"]
        near_r = ([x for x in nl["chunks_where_nearest_link_differs"] if x["chunk"] == ch]
                  or [None])[0]
        ax.set_title(f"chunk {ch} (seq {rec['seq']}, t_step {rec['t_step']})  ·  "
                     f"pose = policy reference at exec step {EXEC_H-1}\n"
                     + (f"nearest link differs: reference {near_r['reference_nearest']} "
                        f"{near_r['reference_mm']:.2f} mm  vs  TO {near_r['refined_nearest']} "
                        f"{near_r['refined_mm']:.2f} mm" if near_r else
                        "nearest link is the same for both: "
                        + ana["n1_link_to_apple"]["closest_chunk_detail"]["ranking_reference"][0]
                        ["link"]),
                     fontsize=9.4, color=figstyle.INK)

        for row, (ia, ib, alab, blab) in enumerate(((0, 2, "base x  [m]", "base z  [m]"),
                                                    (0, 1, "base x  [m]", "base y  [m]"))):
            ax = axes[1 + row, col]
            ax.add_patch(Circle((apple[ia], apple[ib]), APPLE_R,
                                facecolor=figstyle.CATEGORICAL[7], edgecolor="none",
                                alpha=0.9, zorder=3))
            ax.add_patch(Circle((apple[ia], apple[ib]), APPLE_R + 0.010, facecolor="none",
                                edgecolor=figstyle.CATEGORICAL[7], lw=1.0, ls="--", zorder=4))
            ax.annotate("apple (+ margin 10 mm)", (apple[ia], apple[ib] + APPLE_R + 0.016),
                        ha="center", fontsize=7.6, color=figstyle.CATEGORICAL[7], zorder=9)
            for p, cname, lab, lw in ((p_ref, figstyle.CATEGORICAL[2],
                                       "policy reference (actions_reference)", 2.0),
                                      (p_to, figstyle.CATEGORICAL[1],
                                       "TO output (actions)", 2.0)):
                ax.plot(p[EXEC_H - 1 :, ia], p[EXEC_H - 1 :, ib], color=cname, lw=0.9,
                        alpha=0.45, zorder=5)
                ax.plot(p[:EXEC_H, ia], p[:EXEC_H, ib], color=cname, lw=lw, zorder=6, label=lab)
                ax.plot([p[0, ia]], [p[0, ib]], marker="o", ms=5, color=cname, zorder=7)
                ax.plot([p[EXEC_H - 1, ia]], [p[EXEC_H - 1, ib]], marker="s", ms=6,
                        color=cname, zorder=7)
            gap = (p_ref[EXEC_H - 1] - p_to[EXEC_H - 1]) * 1000.0
            ax.annotate("", xy=(p_ref[EXEC_H - 1, ia], p_ref[EXEC_H - 1, ib]),
                        xytext=(p_to[EXEC_H - 1, ia], p_to[EXEC_H - 1, ib]),
                        arrowprops=dict(arrowstyle="<->", color=figstyle.INK_2, lw=1.0),
                        zorder=8)
            ax.annotate(f"{np.linalg.norm(gap):.1f} mm at exec step {EXEC_H-1}\n"
                        f"(x {gap[0]:+.1f}, y {gap[1]:+.1f}, z {gap[2]:+.1f})",
                        ((p_ref[EXEC_H - 1, ia] + p_to[EXEC_H - 1, ia]) / 2,
                         (p_ref[EXEC_H - 1, ib] + p_to[EXEC_H - 1, ib]) / 2),
                        textcoords="offset points", xytext=(10, -16), fontsize=7.4,
                        color=figstyle.INK_2, zorder=9)
            lo = np.minimum(p_ref.min(axis=0), p_to.min(axis=0))
            hi = np.maximum(p_ref.max(axis=0), p_to.max(axis=0))
            lo = np.minimum(lo, apple - APPLE_R - 0.03)
            hi = np.maximum(hi, apple + APPLE_R + 0.03)
            cx, cy = (lo[ia] + hi[ia]) / 2, (lo[ib] + hi[ib]) / 2
            half = max(hi[ia] - lo[ia], hi[ib] - lo[ib]) / 2 + 0.02
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal")
            ax.set_xlabel(alab, fontsize=8.6, color=figstyle.INK_2)
            ax.set_ylabel(blab, fontsize=8.6, color=figstyle.INK_2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
            ax.set_title(("side view (XZ)" if row == 0 else "top view (XY)")
                         + "  ·  circles = exec step 0, squares = exec step 7, "
                           "thin tail = steps 8..49 (TO does not touch them)",
                         fontsize=8.6, color=figstyle.INK_2)

        side["panels"].append({
            "chunk": ch, "seq": int(rec["seq"]), "t_step": int(rec["t_step"]),
            "apple_pos_m": [round(float(v), 6) for v in apple],
            "tip_reference_xyz_m": [[round(float(v), 6) for v in p] for p in p_ref],
            "tip_refined_xyz_m": [[round(float(v), 6) for v in p] for p in p_to],
            "gap_at_exec_end_mm": [round(float(v), 4) for v in
                                   (p_ref[EXEC_H - 1] - p_to[EXEC_H - 1]) * 1000.0],
            "gap_norm_at_exec_end_mm": round(float(np.linalg.norm(
                (p_ref[EXEC_H - 1] - p_to[EXEC_H - 1]) * 1000.0)), 4),
            "max_gap_over_exec_mm": round(float(np.linalg.norm(
                (p_ref[:EXEC_H] - p_to[:EXEC_H]) * 1000.0, axis=1).max()), 4),
            "tail_identical_after_exec": bool(np.allclose(p_ref[EXEC_H:], p_to[EXEC_H:],
                                                          atol=1e-9)),
        })

    fig.suptitle("T9b - the two fingertip paths on the real scene: policy reference vs TO output "
                 "(ee_finger_l1 body origin)", fontsize=12.6, color=figstyle.INK, y=0.988)
    fig.text(0.5, 0.008,
             "green = policy reference (actions_reference) · orange = TO output (actions) · "
             "both start from the same recorded qpos · the nearest link to the apple is a "
             "gripper link (ee_finger_l1 / ee_finger_l2 / ee_left) in 75 of 75 chunks for BOTH, "
             "see the table figure",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.018, 1, 0.968))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
