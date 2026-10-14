"""T60 step 3a — P / H endpoints of the T47 / T49 / T51 attention records (policy step records).

A step record holds the full ``qpos`` / ``qvel`` only at the inference times (every 8 control rows), so the
frame-resolution `endpoints_sampled` is used (validated by `t60_sampled_check` on 142 closed-loop runs).
Per frame, after `pose_scene` (mj_forward on the recorded state, repo model via `replay_scene`):
    in_crate     target body centre, crate body pose (`analyze_t42` centre rule)
    hand_target  any MuJoCo contact between the used arm's hand (subtree EE_BODY_<S> + both fingers, the
                 `replay_contacts.py` set) and the target fruit's collision geoms
    speed        |linear qvel| of the target free joint
    q_err        max |state − state[0]| over the used arm's 7 joints (16D state)
    tcp_err      |site <side>_ee − its frame-0 position|
Used arm / target = the dataset episode record of the recorded --episode-index.

usage: t60_records.py OUT_DIR      (MUJOCO_GL=osmesa, CPU)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
DS_META = STORE / "datasets/rby1/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl"


def episode_meta() -> dict:
    out = {}
    for line in DS_META.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["episode_index"])] = r
    return out


def subtree(model, root: int) -> set:
    out = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in out:
            out.add(b)
    return out


def record_series(path: pathlib.Path, *, used_arm: str, target: str) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    rec = load_run(path)
    scene = replay_scene(rec, height=48, width=64)
    mj, m, d = scene.mujoco, scene.model, scene.data
    B = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
    sd, fl = ("L", "l") if used_arm == "left" else ("R", "r")
    coll = [bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)]
    hand_b = subtree(m, B(f"EE_BODY_{sd}")) | subtree(m, B(f"ee_finger_{fl}1")) | subtree(m, B(f"ee_finger_{fl}2"))
    hand = {g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in hand_b}
    tgt = {g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) == B(target)}
    assert hand and tgt, (path, len(hand), len(tgt))
    jt = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{target}_free"); dof = int(m.jnt_dofadr[jt])
    site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f"{used_arm}_ee")
    b_t, b_c = B(target), B("crate")
    sl = T.ARM_SLICE[used_arm]
    rows = {k: [] for k in ("t_step", "target_pos", "crate_pos", "crate_quat", "hand_target", "speed", "tcp", "q")}
    for st in rec.steps:
        pose_scene(scene, st)
        contact = any((int(c.geom1) in hand and int(c.geom2) in tgt) or (int(c.geom2) in hand and int(c.geom1) in tgt)
                      for c in d.contact[:d.ncon])
        rows["t_step"].append(int(st.t_step)); rows["target_pos"].append(d.xpos[b_t].copy())
        rows["crate_pos"].append(d.xpos[b_c].copy()); rows["crate_quat"].append(d.xquat[b_c].copy())
        rows["hand_target"].append(contact); rows["speed"].append(float(np.linalg.norm(d.qvel[dof:dof + 3])))
        rows["tcp"].append(d.site_xpos[site].copy()); rows["q"].append(np.asarray(st.state)[sl].copy())
    a = {k: np.asarray(v) for k, v in rows.items()}
    a["in_crate"] = T.in_crate(a["target_pos"], a["crate_pos"], a["crate_quat"])
    a["q_err"] = np.abs(a["q"] - a["q"][0]).max(axis=1)
    a["tcp_err"] = np.linalg.norm(a["tcp"] - a["tcp"][0], axis=1)
    return a


def main(out_dir: pathlib.Path) -> None:
    from AG3S.ag3s.experiments.studies import t51_records as R

    meta = episode_meta()
    p = T.Params()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"protocol": T.PROTOCOL_VERSION, "params": p, "definitions": __doc__, "records": {}}
    for key in R.ALL_KEYS:
        spec = R.hs.RECORDS.get(key)
        rel = spec.record if spec is not None else R.EXTRA_RECORDS[key].record
        role = spec.role if spec is not None else R.EXTRA_ROLE
        ep = int(R.EPISODE_OF[key]); em = meta[ep]
        s = record_series(STORE / rel, used_arm=em["used_arm"], target=em["target_fruit"])
        e = T.endpoints_sampled(s["t_step"], s["in_crate"], s["hand_target"], s["speed"], s["q_err"], s["tcp_err"], p)
        kp, kh = T.keep_frames(s["t_step"], e["t_P"]), T.keep_frames(s["t_step"], e["t_H"])
        np.savez_compressed(out_dir / f"{key}.npz", protocol=T.PROTOCOL_VERSION, t_step=s["t_step"],
                            keep_frames_P=kp, keep_frames_H=kh, in_crate=s["in_crate"], hand_target=s["hand_target"],
                            speed=s["speed"], q_err=s["q_err"], tcp_err=s["tcp_err"],
                            t_P=-1 if e["t_P"] is None else e["t_P"], t_H=-1 if e["t_H"] is None else e["t_H"])
        first_in = np.flatnonzero(s["in_crate"])
        res["records"][key] = {"record": rel, "role": role, "episode": ep, "used_arm": em["used_arm"],
                               "target": em["target_fruit"], "n_frames": len(s["t_step"]),
                               "last_t_step": int(s["t_step"][-1]), **e,
                               "frames_removed_P": int((~kp).sum()), "frames_removed_H": int((~kh).sum()),
                               "first_in_crate_t": None if not len(first_in) else int(s["t_step"][first_in[0]]),
                               "q_err_final": float(s["q_err"][-1]), "tcp_err_final": float(s["tcp_err"][-1])}
        print(key, res["records"][key])
    T.write_json(out_dir / "record_endpoints.json", res)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
