"""T42-1 address check (generalized T39/s20/address_check.py; verifier 2026-10-05). CPU only, MUJOCO_GL=osmesa.

- model: the snapshot client's own model for rby1_randomized_pick_place_16d; qpos addresses resolved by NAME from the
  client's constants: OBJECT_JOINTS[<fruit>] for the four fruits, CRATE_JOINT, GRIPPER_L_JOINT / GRIPPER_R_JOINT.
- per episode (the T42-1 candidate list): the client's reset_randomized_scene(settle_seconds=0) -> qpos[adr:adr+7] equals the
  record's pose for every fruit and the basket; after the client's 1.5 s settle (+ mj_kinematics) the target body xpos equals
  qpos[adr:adr+3] (frames 'object_poses' come from body poses).
- also the record's target_fruit / used_arm, which the analysis and the client's --safe-manipulators use.
usage: address_check.py CANDIDATES_JSON OUT_JSON"""
import json, sys, pathlib, importlib.util, os
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
SCRIPT = pathlib.Path("${AG3S_PROJECT_ROOT}-t42/integration/rby1/bringup/pi05_infer.py")
CAND, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path[0] = str(SCRIPT.parent); sys.argv = [str(SCRIPT), "--headless"]
spec = importlib.util.spec_from_file_location("pi05_infer", SCRIPT)
pi = importlib.util.module_from_spec(spec); spec.loader.exec_module(pi)
import mujoco
xml = pi.MODEL_XML_TRANSPORT
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)


def jadr(name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name); assert j >= 0, name
    return int(m.jnt_qposadr[j]), int(m.jnt_bodyid[j])


fruits = {k: jadr(v) for k, v in pi.OBJECT_JOINTS.items()}
crate_adr, crate_body = jadr(pi.CRATE_JOINT)
grip = {"left": jadr(pi.GRIPPER_L_JOINT)[0], "right": jadr(pi.GRIPPER_R_JOINT)[0]}
res = {"definitions": __doc__, "client_script": str(SCRIPT), "model_xml": xml, "model_xml_resolved": str(pathlib.Path(xml).resolve()),
       "nq": int(m.nq), "joint_names": {**{k: v for k, v in pi.OBJECT_JOINTS.items()}, "crate": pi.CRATE_JOINT,
                                        "grip_left": pi.GRIPPER_L_JOINT, "grip_right": pi.GRIPPER_R_JOINT},
       "qposadr": {**{k: v[0] for k, v in fruits.items()}, "crate": crate_adr, "grip_left": grip["left"], "grip_right": grip["right"]},
       "body_names": {**{k: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, v[1]) for k, v in fruits.items()},
                      "crate": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, crate_body)},
       "applied_ctrl_gripper_index": {"left": 0, "right": 1}, "action16_gripper_col": {"left": 7, "right": 15}}
cand = json.loads(CAND.read_text())["candidates"]
eps_list = sorted({e for v in cand.values() for e in v["selected"]})
eps = {}
for e in eps_list:
    r = pi.load_randomized_episode(e)
    pi.reset_randomized_scene(m, d, r, settle_seconds=0.0)
    pre = {k: bool(np.allclose(d.qpos[a:a + 7], np.asarray(r["initial_fruit_poses"][k]), atol=0, rtol=0)) for k, (a, _) in fruits.items()}
    pre["crate"] = bool(np.array_equal(d.qpos[crate_adr:crate_adr + 7], np.asarray(r["basket_pose"])))
    pi.reset_randomized_scene(m, d, r)
    mujoco.mj_kinematics(m, d)
    tf = r["target_fruit"]; a, b = fruits[tf]
    eps[e] = {"split": r.get("split"), "target_fruit": tf, "used_arm": r.get("used_arm"), "task_type": r.get("task_type"),
              "prompt": r.get("prompt"), "pre_settle_qpos_equals_record": pre,
              "settled_target_body_xpos_minus_qpos_max_abs": float(np.abs(d.xpos[b] - d.qpos[a:a + 3]).max()),
              "settled_target_z": float(d.qpos[a + 2]), "settled_grip_qpos": {k: float(d.qpos[v]) for k, v in grip.items()}}
res["episodes"] = eps
res["all_consistent"] = bool(all(all(v["pre_settle_qpos_equals_record"].values()) and v["split"] == "test"
                                 and v["settled_target_body_xpos_minus_qpos_max_abs"] < 1e-9 for v in eps.values()))
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("episodes", "definitions")}, indent=1))
for e, v in eps.items():
    print(e, v["target_fruit"], v["used_arm"], all(v["pre_settle_qpos_equals_record"].values()), f'{v["settled_target_body_xpos_minus_qpos_max_abs"]:.1e}')
