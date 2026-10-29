"""T38 Phase A — re-render the wire request the client sent for each planning seq of a T34 E3b run.

Same method as T32 H4 render_real.py: MuJoCo qpos := planning row 'qpos' of that seq; mj_forward;
TransportScene.attach(m, d).capture(cam) for the 3 cameras (zed_left, wrist_cam_l, wrist_cam_r).
Here the whole SafeRemoteClient._pack payload is rebuilt (depth uint16 mm, K, T_base_cam, robot_state, stamps,
exec_feedback, 16D state) so serve_safe's SafePolicy.infer can be driven unchanged.
Attention = the 16x16 maps the server recorded for that chunk (float16 in the record).
Policy chunk = planning row 'actions_reference' (the policy's original chunk).

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src \
         ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python render_requests.py E3b_ep1807_r2 75
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

REC = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/E3b_rec'))
A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/p/E3b_analysis.json'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38/frames'))
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATT = {"zed_left": "attention_head", "wrist_cam_l": "attention_left_wrist", "wrist_cam_r": "attention_right_wrist"}
RBY1_GRIPPER_OPEN = -0.045   # pi05_infer.py:127

run = sys.argv[1]; nseq = int(sys.argv[2])
rd = REC / run
man = json.loads((rd / "manifest.json").read_text())
rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
obs_rows = [r for r in rows if r["kind"] == "observation"]
chunks = {int(c["seq"]): c for c in json.loads(A.read_text())["runs"][run]["chunks"]}
m = mujoco.MjModel.from_xml_path(man["scene"]["model_xml"]); d = mujoco.MjData(m)
sc = TransportScene.attach(m, d)


def qadr(name):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])


left_q = [qadr(f"left_arm_{i}") for i in range(7)]
right_q = [qadr(f"right_arm_{i}") for i in range(7)]
lg, rg = qadr("gripper_finger_l1"), qadr("gripper_finger_r1")
od = OUT / run
od.mkdir(parents=True, exist_ok=True)
index = {"run": run, "cameras": list(CAMS), "seqs": []}
for i, seq in enumerate(range(1, nseq + 1)):
    row = plan[seq]
    ob = obs_rows[i]
    assert ob["t_step"] == row["t_step"], (seq, ob["t_step"], row["t_step"])
    d.qpos[:] = row["qpos"]; mujoco.mj_forward(m, d)
    blob = {}
    for cam in CAMS:
        fr = sc.capture(cam)
        blob[f"depth_mm/{cam}"] = np.clip(np.rint(np.asarray(fr.depth, np.float64) * 1000.0), 0, 65535).astype(np.uint16)
        blob[f"K/{cam}"] = np.asarray(fr.camera_intrinsics, np.float64)
        blob[f"T/{cam}"] = np.asarray(fr.T_base_cam, np.float64)
        blob[f"robot_state/{cam}"] = np.asarray(fr.robot_state, np.float64)
    state = np.concatenate([[d.qpos[j] for j in left_q], [abs(d.qpos[lg]) / abs(RBY1_GRIPPER_OPEN)],
                            [d.qpos[j] for j in right_q], [abs(d.qpos[rg]) / abs(RBY1_GRIPPER_OPEN)]])
    blob["state"] = state.astype(np.float64)
    blob["actions_reference"] = np.asarray(row["actions_reference"], np.float32)
    c = chunks[seq]
    z = np.load(REC / c["server_chunk"], allow_pickle=False)
    for cam in CAMS:
        blob[f"attention/{cam}"] = z[ATT[cam]].astype(np.float32)
    rec_summary = json.loads(str(z["summary_json"]))
    np.savez_compressed(od / f"seq{seq:03d}.npz", **blob)
    index["seqs"].append({
        "seq": seq, "t_step": row["t_step"], "server_chunk": c["server_chunk"],
        "stamps": ob["camera_stamps"], "render_stamps": ob.get("camera_render_stamps"),
        "stamp_mode": ob.get("stamp_mode"),
        "exec_feedback": row.get("exec_feedback"),
        "phase_label": c.get("phase"),
        "recorded_profile_ms": rec_summary.get("profile_ms"),
        "recorded_to_timing": (rec_summary.get("to") or {}).get("metrics", {}).get("timing_ms"),
        "recorded_grasp_state": (rec_summary.get("grasp") or {}).get("state"),
        "recorded_n_clusters": rec_summary.get("n_clusters"),
        "recorded_server_timing_ms": row["verdict"].get("timing_ms"),
    })
    print(seq, row["t_step"], c.get("phase"), index["seqs"][-1]["recorded_grasp_state"], flush=True)
(OUT / f"{run}.json").write_text(json.dumps(index, indent=1, default=str))
