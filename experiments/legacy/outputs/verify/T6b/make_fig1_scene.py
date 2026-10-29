"""T6b figure 1 (real scene) — what the robot sees at the shared start, and at the reference
poses of the seq where the execute run held.

Only seq 1 (t_step 0) is a pose the execute run actually visited: all three runs start from the
same episode-1807 reset state. From its first hold (seq 3) the execute run's trajectory left the
reference, and this record carries no qpos, so the poses for the later columns come from the
reference episode replay (outputs/live_test/20260925_ep1807/run_0000, bare policy, port 8123).
They are drawn with a warning band and must not be read as T6 poses.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T6b/make_fig1_scene.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

RECORDS = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000'))
VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6b'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6b'))
PNG = OUTDIR / "t6b-scene-shared-start-and-hold-seqs.png"
SIDECAR = OUTDIR / "t6b-scene-shared-start-and-hold-seqs.json"

PICK = [1, 3, 22, 38, 49]
CAMS = ["zed_left", "wrist_cam_l"]


def main() -> None:
    import mujoco

    figstyle.use_korean()
    meta = json.loads((RECORDS / "meta.json").read_text())
    chunks = json.loads((VER / "chunks_T6b.json").read_text())
    ex = {r["seq"]: r for r in chunks["execute"]}
    agg = json.loads((VER / "aggregate_T6b.json").read_text())
    onset = agg["task_phase_overlay"]["execute_hold_onset_seq"]

    model = mujoco.MjModel.from_xml_path(meta["model_xml"])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    free = mujoco.MjvCamera()
    free.type = mujoco.mjtCamera.mjCAMERA_FREE
    free.distance = 2.2
    free.azimuth = 138.0
    free.elevation = -26.0
    crate_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate")

    fig, axes = plt.subplots(3, len(PICK), figsize=(3.5 * len(PICK), 8.7))
    figstyle.style_axes(fig, axes)
    side = {"records": str(RECORDS), "model_xml": meta["model_xml"],
            "picked_seq": PICK,
            "free_camera": {"type": "free", "distance": free.distance,
                            "azimuth": free.azimuth, "elevation": free.elevation,
                            "lookat_body": "crate"},
            "pose_provenance": {
                "seq 1": "shared reset state of all three runs (t_step 0)",
                "other seq": "reference episode replay run_0000; the execute run left this "
                             "trajectory at its first hold (seq %d)" % min(
                                 s for s in ex if not ex[s]["safe"])},
            "columns": []}

    for col, seq in enumerate(PICK):
        step = np.load(RECORDS / ("step_%05d.npz" % (seq - 1)))
        t_step = int(step["t_step"])
        data.qpos[:] = np.asarray(step["qpos"], np.float64)
        mujoco.mj_forward(model, data)

        free.lookat[:] = np.asarray(data.xpos[crate_bid], np.float64)
        renderer.update_scene(data, camera=free)
        rgb = np.asarray(renderer.render())

        r = ex[seq]
        mm = (r["max_violation_m"] or 0.0) * 1e3
        shared = (seq == 1)
        head_c = figstyle.CATEGORICAL[2] if shared else figstyle.CATEGORICAL[7]
        head_t = ("pose shared by all three runs" if shared
                  else "pose from the reference episode, not from T6")

        ax = axes[0, col]
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(head_c)
            sp.set_linewidth(2.2)
        ax.set_title(
            "seq %d · t_step %d\n%s\nT6 verdict: %s · ag3s %s · %.2f mm"
            % (seq, t_step, head_t, "safe" if r["safe"] else "hold",
               r["ag3s_status"], mm),
            fontsize=8.6, color=figstyle.INK, linespacing=1.55)

        for ri, cam in enumerate(CAMS, start=1):
            renderer.update_scene(data, camera=cam)
            img = np.asarray(renderer.render())
            axc = axes[ri, col]
            axc.imshow(img)
            axc.set_xticks([]); axc.set_yticks([])
            for sp in axc.spines.values():
                sp.set_visible(True)
                sp.set_color(head_c)
                sp.set_linewidth(1.4)
            if col == 0:
                axc.set_ylabel(cam, fontsize=9, color=figstyle.INK_2)

        side["columns"].append({
            "seq": seq, "t_step": t_step, "pose_is_t6": shared,
            "t6_safe": r["safe"], "t6_ag3s_status": r["ag3s_status"],
            "t6_trajopt_status": r["trajopt_status"],
            "t6_max_violation_mm": mm,
            "t6_ag3s_reason_codes": r["ag3s_reason_codes"]})

    axes[0, 0].set_ylabel("third person (free camera)", fontsize=9, color=figstyle.INK_2)

    fig.suptitle(
        "T6b · episode 1807 scene at the seq the execute run held "
        "(unbroken hold from seq %d)" % onset,
        fontsize=12.5, color=figstyle.INK, x=0.012, ha="left", y=0.995)
    fig.text(0.012, 0.005,
             "Only the first column is a pose the execute run visited. The record carries no "
             "qpos, so the other columns show the reference episode replay at the same t_step; "
             "the verdict line under each column is the execute run's own verdict at that seq.",
             fontsize=8.4, color=figstyle.INK_2)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.975))
    fig.savefig(PNG, dpi=185, facecolor=figstyle.SURFACE)
    SIDECAR.write_text(json.dumps(side, ensure_ascii=False, indent=1))
    print("wrote", PNG)
    print("wrote", SIDECAR)


if __name__ == "__main__":
    main()
