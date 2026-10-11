"""T36-baseline scene dump (for the scene figure only). Replays the esdf_rollout setup (same AG3S/TO
config as the regression command, cameras=all) for 15 frames with ONE backend per process (T0
invariant forbids both in one process), and per frame stores:
  - the worst ESDF row of the reference chunk (step h, sphere s, centre, radius, clearance)
  - a z-slice (x,y) and a y-slice (x,z) of cs.esdf.distance() through that centre (+-0.30 m, 2.5 mm)
  - zed_left depth + K + T for frames in DEPTH_FRAMES
No SQP. Mirrors AG3S/trajopt/experiments/esdf_rollout.py (read-only copy of its setup).
usage (cwd ${AG3S_PROJECT_ROOT}):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} <py> outputs/verify/T36/scene_dump.py legacy|curobo OUT.npz
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
from AG3S.trajopt.types import ChunkLayout
from AG3S.ag3s.experiments.reports.grounding_report import (
    ARM_LINKS, build_constraint_robot_model, build_robot_model)
from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
from AG3S.ag3s.experiments.sources.policy_record import (
    load_run, replay_scene, pose_scene, phase_for, phase_boundaries_for_path)

CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
REC = "run_16d_ep1800"
ATT = "AG3S/ag3s/asset/data/attention_16d_ep1800.npz"
STEP1 = "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json"
DEPTH_FRAMES = (0, 4)
HALF, RES = 0.30, 0.0025


def main(backend, out):
    run = load_run(REC, limit=15)
    boundaries, _ = phase_boundaries_for_path(REC)
    blob = np.load(ATT, allow_pickle=False)
    cell = json.loads(pathlib.Path(STEP1).read_text())["best"]
    att = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    alias = {"cam_high": "zed_left", "cam_left_wrist": "wrist_cam_l", "cam_right_wrist": "wrist_cam_r"}
    names = [str(c) for c in blob["cameras"]]
    cam_idx = {m: names.index(p) for p, m in alias.items() if p in names}

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    esdf = {"voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False, "backend": backend}
    if backend == "curobo":
        esdf.update({"fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005})
    ag = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
                                    "esdf": esdf}),
              robot_model=filter_robot, constraint_robot_model=robot)
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    to_cfg = TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False},
        "limits": {"source": "model_xml", "model_xml": str(run.model_xml)}})
    planned = to_cfg.with_overrides({"horizon": {"horizon": to_cfg.horizon.planned,
                                                 "execution_length": min(to_cfg.horizon.execution_length,
                                                                         to_cfg.horizon.planned),
                                                 "plan_horizon": None}})
    lin = CollisionLinearizer(robot, layout, planned.horizon.horizon)
    link_names = list(robot.sphere_link_names)

    g = np.arange(-HALF, HALF + 1e-9, RES)
    rec = {}
    for i, step in enumerate(run):
        pose_scene(scene, step)
        frames = {c: scene.capture(c) for c in CAMERAS}
        obs = [camera_observation(scene, c, filter_robot, timestamp=float(i),
                                  attention_map=(att[i, di, ai, cell["layer"], cell["head"], cam_idx[c]]
                                                 if c in cam_idx else None))[0] for c in CAMERAS]
        cs = ag.process_multi(obs, phase=phase_for(step.t_step, boundaries))
        snap = scene_from_constraint_set(cs, lin.robot_radii, to_cfg)
        ref = layout.chunk_to_trajectory(step.actions)[:, :planned.horizon.horizon]
        q_now = np.asarray([scene.data.qpos[scene._qadr[j]] for j in DEFAULT_RBY1_JOINTS], float)
        centres = lin.sphere_states(ref, q_now)[0]
        e = lin._esdf_clearance(centres, snap)[..., 0]
        h, s = np.unravel_index(int(np.argmin(e)), e.shape)
        c = centres[h, s]
        full = lin.full_violation(ref, q_now, snap)
        X, Y = np.meshgrid(c[0] + g, c[1] + g, indexing="xy")
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, c[2])], 1)
        XZ, ZZ = np.meshgrid(c[0] + g, c[2] + g, indexing="xy")
        pxz = np.stack([XZ.ravel(), np.full(XZ.size, c[1]), ZZ.ravel()], 1)
        rec[f"slicexz_{i}"] = np.asarray(cs.esdf.distance(pxz), np.float32).reshape(XZ.shape)
        sl = np.asarray(cs.esdf.distance(pts), np.float32).reshape(X.shape)
        rec[f"worst_{i}"] = np.array([h, s, *c, lin.robot_radii[s], e[h, s], full])
        rec[f"link_{i}"] = np.array(link_names[s])
        if hasattr(cs.esdf, "answer_tier"):
            rec[f"tier_{i}"] = np.asarray(cs.esdf.answer_tier(c[None, :]))
        rec[f"slice_{i}"] = sl
        rec[f"centres_{i}"] = centres[:, : lin.robot_radii.shape[0]].astype(np.float32)
        if i in DEPTH_FRAMES:
            hd = frames["zed_left"]
            rec[f"depth_{i}"] = hd.depth.astype(np.float32)
            rec[f"K_{i}"] = np.asarray(hd.camera_intrinsics)
            rec[f"T_{i}"] = np.asarray(hd.T_base_cam)
        print(i, link_names[s], h, np.round(c, 3), f"{e[h, s]*1000:+.4f} mm  full {full*1000:+.4f}", flush=True)
    rec["radii"] = lin.robot_radii
    rec["grid_offsets"] = g
    scene.close()
    np.savez_compressed(out, **rec)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
