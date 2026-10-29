"""T2 ep1807 — 물체 위치·손 궤적 참값. measure_t2_events.py 와 별도로 돈다(참값만, latch 안 건드림)."""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))

from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

FRUITS = ("apple", "banana", "orange", "pear")

def crate_local(scene, world_point):
    T = np.linalg.inv(scene.body_pose("crate"))
    p = T @ np.array([*world_point, 1.0])
    return p[:3]

def main():
    run = load_run("outputs/live_test/20260925_ep1807/run_0000")
    scene = replay_scene(run)
    out = []
    attach_pos = None
    for i, step in enumerate(run):
        pose_scene(scene, step)
        rec = {"i": i, "t_step": int(step.t_step)}
        for f in FRUITS:
            p = scene.body_position(f)
            rec[f + "_world"] = [float(x) for x in p]
            rec[f + "_z"] = float(p[2])
            local = crate_local(scene, p)
            rec[f + "_crate_local_xy_dist"] = float(np.hypot(local[0], local[1]))
        rec["crate_world"] = [float(x) for x in scene.body_position("crate")]
        rec["ee_finger_l1_world"] = [float(x) for x in scene.body_position("ee_finger_l1")]
        out.append(rec)
    scene.close()

    apple0 = np.array(out[0]["apple_world"])
    apple_max_z = max(r["apple_z"] for r in out)
    apple_min_z_after0 = out[0]["apple_z"]
    dist0 = out[0]["apple_crate_local_xy_dist"]
    dist_last = out[-1]["apple_crate_local_xy_dist"]

    result = {
      "records": "outputs/live_test/20260925_ep1807/run_0000",
      "n_frames": len(out),
      "apple_z_frame0": out[0]["apple_z"],
      "apple_z_max": apple_max_z,
      "apple_lift_mm": (apple_max_z - out[0]["apple_z"]) * 1000.0,
      "apple_crate_xy_dist_mm_frame0": dist0 * 1000.0,
      "apple_crate_xy_dist_mm_last": dist_last * 1000.0,
      "other_fruit_drift_mm": {
        f: float(np.linalg.norm(np.array(out[-1][f+"_world"]) - np.array(out[0][f+"_world"]))) * 1000.0
        for f in ("banana", "orange", "pear")
      },
      "frames": out,
    }
    with open("outputs/verify/T2/t2_objects_ep1807.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=1))

if __name__ == "__main__":
    main()
