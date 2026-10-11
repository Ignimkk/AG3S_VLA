"""T9a figure 1 — 실제 씬. task 가 지정한 두 자리.

왼쪽 두 칸: **gripper 가 닫힌 순간** (M2, t=112) — 손이 사과의 어느 쪽에 있었나.
오른쪽 두 칸: **하강이 멈춘 자리** (M4, 마지막 기록 t=304) — 무엇이 얼마나 가까웠나.
위 줄 MuJoCo 렌더, 아래 줄 base frame XZ 투영 (제약 구 266 개 모델 그대로).
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
from m3_to_refined_vs_reference import MODEL_XML  # noqa: E402

T9A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9a'))
REC = _ag3s_workspace_path('outputs/live_test/20260926_t8/execute_ep1807/frames.jsonl')
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t9a/t9a-scene.png'))
SIDE = OUT.with_suffix(".json")
MARGIN_MM = 10.0
APPLE_R = 0.0333


def main() -> None:
    import mujoco

    z = np.load(T9A / "m0_state_tensor.npz", allow_pickle=True)
    ana = json.loads((T9A / "m1to5_analysis.json").read_text())
    clear = z["clearance_m"].astype(np.float64) * 1000.0
    centres = z["centres_m"].astype(np.float64)
    bodies = [str(b) for b in z["bodies"]]
    link_names = [str(n) for n in z["link_names"]]
    radii = np.asarray(z["radii_m"], np.float64)
    t = np.asarray(z["t_step"], np.int64)
    kind = np.asarray([str(k) for k in z["kind"]])
    obj = np.asarray(z["obj_pos"], np.float64)
    obj_names = [str(x) for x in z["obj_names"]]
    ee = np.asarray(z["ee_pos"], np.float64)
    grip = np.asarray(z["grip_cmd"], np.float64)
    crate_top = float(z["crate_top_z_m"]) if "crate_top_z_m" in z.files else None
    ai, ci = bodies.index("apple"), bodies.index("crate")
    apple_i = obj_names.index("apple")
    crate_i = obj_names.index("crate")

    recs = [json.loads(l) for l in open(REC)]
    plan_by_seq = {int(r["seq"]): r for r in recs if r["kind"] == "planning"}
    qpos_by_row = []
    for r in recs:
        if r["kind"] in ("control", "planning"):
            qpos_by_row.append((int(r["t_step"]), r["kind"], np.asarray(r["qpos"], np.float64),
                                int(r.get("seq", r.get("chunk_seq", -1)))))
    qpos_by_row.sort(key=lambda x: (x[0], 0 if x[1] == "planning" else 1))

    t_grip = ana["m2_gripper_close"]["left"]["t_step"]
    i_grip = int(np.nonzero(t == t_grip)[0][0])
    i_stop = len(t) - 1
    PANELS = [("A  gripper closes (M2)", i_grip), ("B  descent stops (M4, last record)", i_stop)]

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=470, width=620)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.4),
                             gridspec_kw={"height_ratios": [1.0, 1.12]})
    figstyle.style_axes(fig, axes[1])
    side: dict = {"margin_mm": MARGIN_MM, "panels": []}

    for col, (label, i) in enumerate(PANELS):
        q = qpos_by_row[i][2]
        assert int(qpos_by_row[i][0]) == int(t[i])
        d.qpos[: q.size] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        apple = obj[i, apple_i]
        cam.lookat[:] = [float(apple[0]) if col == 0 else float(obj[i, crate_i][0]),
                         float(apple[1]) if col == 0 else float(obj[i, crate_i][1]),
                         0.95]
        cam.distance = 0.62 if col == 0 else 1.05
        cam.azimuth, cam.elevation = 128.0, -16.0
        renderer.update_scene(d, cam)
        ax = axes[0, col]
        ax.imshow(renderer.render())
        ax.set_axis_off()
        seq = int(z["chunk_seq"][i])
        pr = plan_by_seq.get(seq, {})
        mvp = (pr.get("max_violation_pair") or {})
        ax.set_title(f"{label}\nt_step {int(t[i])} · {kind[i]} · chunk seq {seq} · "
                     f"safe={pr.get('safe')} · {pr.get('verdict', {}).get('trajopt_status')}",
                     fontsize=9.8, color=figstyle.INK)

        ax = axes[1, col]
        ax.add_patch(Circle((apple[0], apple[2]), APPLE_R, facecolor=figstyle.CATEGORICAL[7],
                            edgecolor="none", alpha=0.9, zorder=5))
        ax.annotate("apple (target)", (apple[0], apple[2] + APPLE_R + 0.012), ha="center",
                    fontsize=7.6, color=figstyle.CATEGORICAL[7], zorder=9)
        cr = obj[i, crate_i]
        if crate_top is not None:
            ax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18,
                                       crate_top - cr[2] + 0.004, facecolor="none",
                                       edgecolor=figstyle.INK_2, lw=1.3, zorder=3))
            ax.annotate(f"crate rim z={crate_top*1000:.0f} mm", (cr[0] + 0.095, crate_top),
                        fontsize=7.4, color=figstyle.INK_2, zorder=9)
        tz = float(z["table_top_z_m"]) if "table_top_z_m" in z.files else None
        if tz is not None:
            ax.axhline(tz, color=figstyle.GRID_INK, lw=1.5, zorder=1)
            ax.annotate(f"table top z={tz*1000:.0f} mm", (0.30, tz + 0.006), fontsize=7.4,
                        color=figstyle.INK_2)

        for k, n in enumerate(link_names):
            if n in ("ee_finger_l1", "ee_finger_l2"):
                fc, al, zo = figstyle.CATEGORICAL[2], 0.85, 7
            elif n == "ee_left":
                fc, al, zo = figstyle.CATEGORICAL[1], 0.55, 6
            elif n.startswith("link_left"):
                fc, al, zo = figstyle.SEQ_BLUE[3], 0.22, 2
            elif n in ("ee_right", "ee_finger_r1", "ee_finger_r2") or n.startswith("link_right"):
                fc, al, zo = figstyle.CATEGORICAL[4], 0.16, 2
            else:
                continue
            ax.add_patch(Circle((centres[i, k, 0], centres[i, k, 2]), radii[k],
                                facecolor=fc, edgecolor="none", alpha=al, zorder=zo))

        fin = np.asarray([n in ("ee_finger_l1", "ee_finger_l2") for n in link_names], bool)
        sj = int(np.nonzero(fin)[0][int(np.argmin(clear[i, fin, ai]))])
        ax.add_patch(Circle((centres[i, sj, 0], centres[i, sj, 2]),
                            radii[sj] + MARGIN_MM / 1000.0, facecolor="none",
                            edgecolor=figstyle.CATEGORICAL[6], lw=1.2, ls="--", zorder=8))
        ax.plot([centres[i, sj, 0]], [centres[i, sj, 2]], marker="x", ms=8, mew=2.0,
                color=figstyle.CATEGORICAL[2], zorder=9)
        ax.annotate("", xy=(apple[0], apple[2]), xytext=(centres[i, sj, 0], centres[i, sj, 2]),
                    arrowprops=dict(arrowstyle="<->", color=figstyle.CATEGORICAL[2], lw=1.1),
                    zorder=9)

        off = (centres[i, sj] - apple) * 1000.0
        crate_link = link_names[int(np.argmin(clear[i, :, ci]))]
        txt = (f"t_step {int(t[i])}   chunk seq {seq}\n"
               f"gripper cmd  L {grip[i,0]:.3f}   R {grip[i,1]:.3f}\n"
               f"left finger -> apple surface {clear[i, fin, ai].min():8.2f} mm\n"
               f"  (centre -> surface        {clear[i, fin, ai].min()+radii[sj]*1000:8.2f} mm)\n"
               f"offset sphere - apple centre  [mm]\n"
               f"  x {off[0]:+8.2f}  y {off[1]:+8.2f}  z {off[2]:+8.2f}\n"
               f"EE_BODY_L z {ee[i,0,2]*1000:8.2f} mm\n"
               f"min clearance to crate {clear[i,:,ci].min():8.2f} mm ({crate_link})\n"
               f"apple z displacement {(obj[i,apple_i,2]-obj[0,apple_i,2])*1000:+8.4f} mm\n"
               f"max_violation_pair {mvp.get('link')} / {mvp.get('obstacle')}\n"
               f"  clearance {(mvp.get('clearance_m') or 0)*1000:.2f} mm   margin 10 mm")
        ax.annotate(txt, (0.985, 0.985), xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.4, family="monospace", color=figstyle.INK,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=figstyle.SURFACE,
                              edgecolor=figstyle.GRID_INK, alpha=0.94))
        if col == 0:
            ax.set_xlim(float(apple[0]) - 0.22, float(apple[0]) + 0.22)
            ax.set_ylim(float(apple[2]) - 0.10, float(apple[2]) + 0.34)
        else:
            ax.set_xlim(0.24, 0.92)
            ax.set_ylim(0.78, 1.30)
        ax.set_aspect("equal")
        ax.set_xlabel("base x  [m]", fontsize=8.6, color=figstyle.INK_2)
        if col == 0:
            ax.set_ylabel("base z  [m]", fontsize=8.6, color=figstyle.INK_2)
        side["panels"].append({
            "label": label, "row_index": int(i), "t_step": int(t[i]), "kind": str(kind[i]),
            "chunk_seq": seq, "gripper_cmd_left": round(float(grip[i, 0]), 6),
            "gripper_cmd_right": round(float(grip[i, 1]), 6),
            "left_finger_to_apple_mm": round(float(clear[i, fin, ai].min()), 4),
            "closest_sphere_index": sj, "closest_sphere_link": link_names[sj],
            "closest_sphere_radius_mm": round(float(radii[sj]) * 1000.0, 4),
            "offset_sphere_minus_apple_centre_mm": [round(float(v), 4) for v in off],
            "ee_body_l_z_mm": round(float(ee[i, 0, 2]) * 1000.0, 4),
            "min_clearance_to_crate_mm": round(float(clear[i, :, ci].min()), 4),
            "min_clearance_to_crate_link": crate_link,
            "apple_z_displacement_mm": round(float(obj[i, apple_i, 2] - obj[0, apple_i, 2])
                                             * 1000.0, 6),
            "max_violation_pair": mvp, "safe": pr.get("safe"),
            "trajopt_status": pr.get("verdict", {}).get("trajopt_status")})

    fig.suptitle("T9a - execute_ep1807: the moment the gripper closed, and the place the "
                 "descent stopped", fontsize=12.8, color=figstyle.INK, y=0.983)
    fig.text(0.5, 0.012,
             "green = left fingertip spheres (r 3.6/3.97/4.72 mm) · orange = ee_left palm spheres "
             "(r 8.92 mm, new this run) · blue = rest of left arm · pink = right arm\n"
             "dashed ring = focus sphere radius + margin 10 mm · 266-sphere model rebuilt with "
             "the server settings: spacing 0.4, radius scale 0.2, arm_5 cut at link-frame "
             "z = -100 mm",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.042, 1, 0.960))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
