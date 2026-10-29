"""T36-baseline probe: AG3S-only replay of the esdf_rollout cuRobo configuration (same config dict,
same 3-camera wiring, same attention cell) that records, per frame, (a) which cuRobo-path stats the
builder reports (tsdf truncation / separate fine TSDF, held_free, target_free, window) and (b) sha1 of
every field tier's distance array. No SQP. Run twice in separate processes and diff (bitwise check at
field level, not only at clearance level).

usage (cwd ${AG3S_PROJECT_ROOT}):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python outputs/verify/T36/probe_fields.py OUT.json
Mirrors AG3S/trajopt/experiments/esdf_rollout.py lines 150-205 and 280-315 (read-only copy).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import hashlib, json, pathlib, sys
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.experiments.reports.grounding_report import (
    ARM_LINKS, build_constraint_robot_model, build_robot_model)
from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
from AG3S.ag3s.experiments.sources.policy_record import (
    load_run, replay_scene, pose_scene, phase_for, phase_boundaries_for_path)

CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
REC = "run_16d_ep1800"
ATT = "AG3S/ag3s/asset/data/attention_16d_ep1800.npz"
STEP1 = "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json"


def sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha1(a.tobytes()).hexdigest()


def main(out):
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
    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False,
                 "backend": "curobo", "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005}})
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=robot)
    rows = []
    for i, step in enumerate(run):
        pose_scene(scene, step)
        obs = [camera_observation(scene, c, filter_robot, timestamp=float(i),
                                  attention_map=(att[i, di, ai, cell["layer"], cell["head"], cam_idx[c]]
                                                 if c in cam_idx else None))[0] for c in CAMERAS]
        cs = ag.process_multi(obs, phase=phase_for(step.t_step, boundaries))
        f = cs.esdf
        st = f.stats
        rows.append({
            "i": i, "has_target": bool(cs.has_target),
            "backend": st.get("backend"),
            "n_occupied": st.get("n_occupied"),
            "tsdf_truncation": st.get("tsdf_truncation"),
            "held_free": st.get("held_free"),
            "target_free": {k: st.get("target_free", {}).get(k) for k in ("n_layers", "mode", "n_target_points")},
            "window": st.get("window"),
            "n_attached_points": st.get("n_attached_points"),
            "tiers": st.get("tiers"),
            "layer_sha1": [sha(l.distance_grid) for l in getattr(f, "layers", ())],
            "target_free_layer_sha1": [sha(l.distance_grid) for l in getattr(f, "target_free_layers", ())],
            "label_sha1": [sha(l.label_grid) if getattr(l, "label_grid", None) is not None else None
                           for l in getattr(f, "layers", ())],
        })
        print(i, rows[-1]["n_occupied"], [s[:10] for s in rows[-1]["layer_sha1"]], flush=True)
    scene.close()
    pathlib.Path(out).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
