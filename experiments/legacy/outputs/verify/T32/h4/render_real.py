"""T32 H4 — re-render the depth the client sent for each planning seq of a recorded run.

usage (cwd = outputs/verify/T32/h4/runroot):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python \
      ../render_real.py E3b_ep1807_r1 37
For seq = 1..N: MuJoCo qpos := planning row 'qpos' of that seq; mj_forward; TransportScene.attach(m, d) (480x640, the
client's AG3S capture object) .capture(cam) for cam in manifest cameras (zed_left, wrist_cam_l, wrist_cam_r).
Depth is wire-quantised like trajopt/wire.py (rint(d*1000) clipped to uint16). Robot mask = GT segmentation of robot
bodies (mujoco_source.is_robot_body) -- NOT the server's sphere self-filter (difference documented in verify json).
T_base_cam = frame.T_base_cam (MuJoCo camera pose), K = TransportScene.intrinsics.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, is_robot_body

T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4/real'))
run = sys.argv[1]; nseq = int(sys.argv[2])
cond = run.split("_")[0]
rd = T30 / f"{cond}_rec" / run
man = json.loads((rd / "manifest.json").read_text())
cams = man["cameras"]
rows = {int(x["seq"]): x for x in (json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
m = mujoco.MjModel.from_xml_path(man["scene"]["model_xml"]); d = mujoco.MjData(m)
sc = TransportScene.attach(m, d)
(OUT / f"frames_{run}").mkdir(parents=True, exist_ok=True)
info = {"run": run, "cameras": cams, "seqs": []}
for seq in range(1, nseq + 1):
    row = rows[seq]
    d.qpos[:] = row["qpos"]; mujoco.mj_forward(m, d)
    blob = {}
    for cam in cams:
        fr = sc.capture(cam)
        dm = np.clip(np.rint(np.asarray(fr.depth, np.float64) * 1000.0), 0, 65535).astype(np.uint16)
        names = np.array([fr.body_names.get(int(b), "") if b >= 0 else "" for b in np.unique(fr.body_ids)])
        robot_ids = [int(b) for b in np.unique(fr.body_ids) if b >= 0 and is_robot_body(fr.body_names.get(int(b)))]
        mask = np.isin(fr.body_ids, robot_ids)
        blob[f"depth_mm/{cam}"] = dm; blob[f"mask/{cam}"] = mask
        blob[f"K/{cam}"] = fr.camera_intrinsics; blob[f"T/{cam}"] = fr.T_base_cam
    np.savez_compressed(OUT / f"frames_{run}" / f"seq{seq:03d}.npz", **blob)
    info["seqs"].append({"seq": seq, "t_step": row["t_step"],
                         "mask_px": {c: int(blob[f"mask/{c}"].sum()) for c in cams}})
    print(seq, row["t_step"], info["seqs"][-1]["mask_px"], flush=True)
(OUT / f"frames_{run}.json").write_text(json.dumps(info, indent=1))
