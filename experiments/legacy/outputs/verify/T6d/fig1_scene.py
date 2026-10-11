"""T6d figure 1 — 실제 씬. 세 실행의 같은 chunk 를 나란히.

위 줄: MuJoCo 렌더 (osmesa) — 그 frame 의 `qpos` 그대로.
아래 줄: 같은 frame 을 base frame XZ 로 투영한 배치도 —
  · 참값 물체(사과·오렌지·배·바나나·상자 상단)
  · 왼팔 제약 구 (arm_5 5개는 굵게, 손끝 22개는 따로)
  · `max_violation_pair.point_m` 과 그 구가 요구하는 자유 반지름 (r 81.25 + margin 50 = 131.25 mm)

고른 chunk: `seq 5` (grasp 실행의 유일한 HOLD 가 난 자리, t=32). 세 실행 모두 같은 seq.
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
from matplotlib.patches import Circle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = [
    ("closed-loop / approach", _ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl')),
    ("shadow / approach", _ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl')),
    ("closed-loop / grasp", _ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl')),
]
SEQ = 5
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6d/t6d-scene.png'))
SIDE = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6d/t6d-scene.json'))
ESDF_MARGIN = 0.050
FRUITS = ("apple", "orange", "pear", "banana")
FRUIT_R = {"apple": 0.0333, "orange": 0.0307, "pear": 0.0432, "banana": 0.0226}


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)
    JOINTS = tuple([f"torso_{i}" for i in range(6)]
                   + [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)])
    QADR = {**{f"torso_{i}": 5 + i for i in range(6)},
            **{f"right_arm_{i}": 11 + i for i in range(7)},
            **{f"left_arm_{i}": 20 + i for i in range(7)}}

    renderer = mujoco.Renderer(m, height=420, width=560)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.52, 0.18, 0.92]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 128.0, -18.0

    ok = figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.4),
                             gridspec_kw={"height_ratios": [1.0, 1.05]})
    figstyle.style_axes(fig, axes[1])
    side: dict = {"seq": SEQ, "esdf_margin_m": ESDF_MARGIN, "runs": {}}

    for col, (label, path) in enumerate(RUNS):
        rec = None
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "planning" and r["seq"] == SEQ:
                    rec = r
                    break
        assert rec is not None
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, cam)
        img = renderer.render()
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_axis_off()
        pair = rec["max_violation_pair"]
        st = rec["verdict"]["trajopt_status"]
        hold = "HOLD" if rec.get("hold_reason") else "no hold"
        ax.set_title(f"{label}\nseq {SEQ} · t={rec['t_step']} · {st} · {hold}",
                     fontsize=10, color=figstyle.INK)

        # ---- 배치도 (base frame XZ) ----
        ax = axes[1, col]
        q = np.asarray([qpos[QADR[n]] for n in JOINTS], np.float64)
        centres = np.asarray(robot.sphere_centers_numeric(q)[0], np.float64).reshape(-1, 3)
        # 물체
        objs = rec["object_poses"]
        for k, name in enumerate(FRUITS):
            p = np.asarray(objs[name]["pos"], np.float64)
            ax.add_patch(Circle((p[0], p[2]), FRUIT_R[name], facecolor=figstyle.CATEGORICAL[k],
                                edgecolor="none", alpha=0.85, zorder=4))
            ax.annotate(name, (p[0], p[2] + FRUIT_R[name] + 0.012), ha="center",
                        fontsize=7.5, color=figstyle.INK_2)
        cr = np.asarray(objs["crate"]["pos"], np.float64)
        ax.add_patch(plt.Rectangle((cr[0] - 0.09, cr[2] - 0.004), 0.18, 0.112,
                                   facecolor="none", edgecolor=figstyle.INK_2, lw=1.0, zorder=3))
        ax.annotate("crate", (cr[0], cr[2] + 0.12), ha="center", fontsize=7.5,
                    color=figstyle.INK_2)
        tb = np.asarray(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "table")], np.float64)
        ax.axhline(tb[2] + 0.02, color=figstyle.GRID_INK, lw=1.4, zorder=1)
        ax.annotate("table top", (0.30, tb[2] + 0.028), fontsize=7.5, color=figstyle.INK_2)

        # 왼팔 구
        for i, n in enumerate(names):
            if "left" not in n and not n.startswith("ee_finger_l"):
                continue
            if n == "link_left_arm_5":
                fc, a, z = figstyle.CATEGORICAL[7], 0.30, 5
            elif n.startswith("ee_finger_l"):
                fc, a, z = figstyle.CATEGORICAL[2], 0.85, 6
            else:
                fc, a, z = figstyle.SEQ_BLUE[3], 0.20, 2
            ax.add_patch(Circle((centres[i, 0], centres[i, 2]), radii[i], facecolor=fc,
                                edgecolor="none", alpha=a, zorder=z))

        # 위반 좌표 + 요구 자유 반지름
        p = np.asarray(pair["point_m"], np.float64)
        rq = float(radii[int(pair["query"])])
        need = rq + ESDF_MARGIN
        ax.add_patch(Circle((p[0], p[2]), need, facecolor="none", edgecolor=figstyle.CATEGORICAL[1],
                            lw=1.6, ls="--", zorder=7))
        ax.plot([p[0]], [p[2]], marker="x", ms=9, mew=2.2,
                color=figstyle.CATEGORICAL[1], zorder=8)
        ap = np.asarray(objs["apple"]["pos"], np.float64)
        ax.plot([p[0], ap[0]], [p[2], ap[2]], color=figstyle.CATEGORICAL[1], lw=0.9,
                ls=":", zorder=7)
        dmm = float(np.linalg.norm(p - ap)) * 1000.0
        ax.annotate(f"{dmm:.0f} mm to apple centre",
                    ((p[0] + ap[0]) / 2 + 0.01, (p[2] + ap[2]) / 2),
                    fontsize=7.5, color=figstyle.CATEGORICAL[1])
        ax.annotate(f"worst pair: {pair['link']}\nclearance {pair['clearance_m']*1000:+.2f} mm\n"
                    f"needs r {rq*1000:.2f} + margin {ESDF_MARGIN*1000:.0f} = {need*1000:.2f} mm free",
                    (0.02, 0.98), xycoords="axes fraction", fontsize=8, color=figstyle.INK,
                    va="top", bbox=dict(boxstyle="round,pad=0.35", facecolor=figstyle.SURFACE,
                                        edgecolor=figstyle.GRID_INK, alpha=0.92))
        ax.set_xlim(0.22, 0.86)
        ax.set_ylim(0.80, 1.26)
        ax.set_aspect("equal")
        ax.set_xlabel("base x  [m]", fontsize=8.5, color=figstyle.INK_2)
        if col == 0:
            ax.set_ylabel("base z  [m]", fontsize=8.5, color=figstyle.INK_2)
        ax.set_title("XZ projection, base frame — spheres at the recorded qpos, "
                     f"x at plan step {pair['step']}", fontsize=9, color=figstyle.INK)

        side["runs"][label] = {
            "t_step": int(rec["t_step"]), "trajopt_status": st,
            "hold_reason": rec.get("hold_reason", ""),
            "worst_link": pair["link"], "query": pair["query"],
            "step_in_chunk": pair["step"],
            "clearance_mm": float(pair["clearance_m"]) * 1000.0,
            "sphere_radius_mm": rq * 1000.0,
            "required_free_radius_mm": need * 1000.0,
            "point_m": [float(v) for v in p],
            "dist_point_to_apple_centre_mm": dmm,
            "apple_pos": [float(v) for v in ap],
        }

    fig.suptitle("T6d — worst-clearance query point against ground-truth objects, "
                 "same chunk (seq 5) in the three runs",
                 fontsize=12.5, color=figstyle.INK, y=0.985)
    legend = ("green = left fingertip spheres · pale orange = link_left_arm_5 spheres (r 81.25 mm) · "
              "blue = other left-arm spheres · x with dashed circle = worst-clearance query point "
              "and the free radius it demands")
    fig.text(0.5, 0.012, legend, ha="center", fontsize=8.5, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("korean font:", ok)
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
