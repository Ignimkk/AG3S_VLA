"""T43 S (4) — grasp geometry of the TRAINING DEMONSTRATIONS (dataset rby1_randomized_pick_place_16d_v1), same metrics as
measure_pick.geom_at, so the policy's rollouts can be compared with what it was trained on. CPU only, read-only.

Per episode (meta/randomized_episodes.jsonl, success only): used_arm, target_fruit, events.grasp_close_start_frame (onset),
onset + 16 (settled, the rollout metric's row), events.grasp_verified_frame.
qpos: template = T42-1 run control row 0 qpos (base / torso / head at reset; torso sag ~0.005 rad), arm joints and finger joints
from observation.state (cols 0-6 / 8-14 arms; 7 / 15 normalized opening -> gripper_finger_<s>1 = -open*0.045, <s>2 = +open*0.045),
fruits at initial_fruit_poses with the target at randomized_scene.settled_target_pose (the fruit does not move before the
lift: lift_from_table_start_frame >= grasp_verified_frame).
usage: measure_demo.py [N_WORKERS]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import measure_pick as MP

DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
TEMPLATE_RUN = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1834_s18341/frames.jsonl'))


def template():
    for l in TEMPLATE_RUN.open():
        d = json.loads(l)
        if d["kind"] == "control":
            return np.asarray(d["qpos"], float)


Q0 = template()
META = [json.loads(l) for l in (DS / "meta/randomized_episodes.jsonl").read_text().splitlines() if l.strip()]


def one(rec):
    import pandas as pd
    M = MP.model(); J = M["J"]
    ep = rec["episode_index"]; arm = rec["used_arm"]; fruit = rec["target_fruit"]
    ev = rec.get("events") or {}
    on = ev.get("grasp_close_start_frame"); ver = ev.get("grasp_verified_frame")
    if on is None: return {"episode": ep, "error": "no grasp_close_start_frame"}
    df = pd.read_parquet(DS / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=["observation.state", "frame_index"])
    S = np.stack(df["observation.state"].to_numpy())
    out = {"episode": ep, "split": rec["split"], "arm": arm, "fruit": fruit, "onset_frame": on, "verified_frame": ver,
           "lift_start_frame": ev.get("lift_from_table_start_frame"), "rows": {}}
    for nm, f in (("onset", on), ("settled", on + 16), ("verified", ver)):
        if f is None or f >= len(S): continue
        s = S[f]; q = Q0.copy()
        for i in range(7):
            q[J[f"left_arm_{i}"]] = s[i]; q[J[f"right_arm_{i}"]] = s[8 + i]
        for side, col in (("l", 7), ("r", 15)):
            q[J[f"gripper_finger_{side}1"]] = -s[col] * MP.OPEN_M; q[J[f"gripper_finger_{side}2"]] = s[col] * MP.OPEN_M
        for fr, pose in rec["initial_fruit_poses"].items():
            q[MP.QADR[fr]:MP.QADR[fr] + 7] = pose
        q[MP.QADR[fruit]:MP.QADR[fruit] + 7] = rec["randomized_scene"]["settled_target_pose"]
        q[MP.QADR["crate"]:MP.QADR["crate"] + 7] = rec["basket_pose"]
        g = MP.geom_at(q, arm, fruit); g.pop("ee_xmat", None); g.pop("contacts", None)
        out["rows"][nm] = {"frame": int(f), **{k: v for k, v in g.items() if not isinstance(v, np.ndarray)},
                           "finger_mid": g["finger_mid"], "fruit_pos": g["fruit_pos"], "tip_mid": g["tip_mid"]}
    return MP.jl(out)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    recs = [r for r in META if r.get("success")]
    with Pool(nw) as pool:
        res = pool.map(one, recs, chunksize=8)
    (OUT / "demo_grasp.json").write_text(json.dumps({"definitions": __doc__, "episodes": res}, indent=0))
    print("done", len(res))
