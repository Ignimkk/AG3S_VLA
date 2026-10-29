"""T39 S2-0: are the hardcoded qpos addresses of T28 analyze_e0.py (APPLE_ADR 38, CRATE_ADR 31, GRIP_L_ADR 27)
the right ones for all 24 apple + left-arm test episodes?
- model: the client's own MODEL_XML_TRANSPORT (snapshot pi05_infer.py, model rby1_randomized_pick_place_16d);
  addresses looked up by NAME from the client's constants (OBJECT_JOINTS['apple'], CRATE_JOINT, GRIPPER_L_JOINT).
- per episode: the client's own reset_randomized_scene(settle_seconds=0) -> qpos[38:45] must equal the record's apple pose,
  qpos[31:38] the basket pose; after the client's 1.5 s settle, the apple body xpos must equal qpos[38:41]
  (frames 'object_poses' come from body poses) and the crate body xpos/xquat qpos[31:38].
- dataset scan: every test episode 1800-1999 with target_fruit apple and used_arm left == the task's list of 24.
usage: address_check.py OUT_JSON  (last argument)"""
import json, sys, pathlib, importlib.util, os
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
SCRIPT = pathlib.Path("${AG3S_PROJECT_ROOT}-o1/integration/rby1/bringup/pi05_infer.py")
sys.path[0] = str(SCRIPT.parent); sys.argv = [str(SCRIPT), "--headless"] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("pi05_infer", SCRIPT)
pi = importlib.util.module_from_spec(spec); spec.loader.exec_module(pi)
import mujoco
OUT = pathlib.Path(sys.argv[-1])
TASK_EPS = [1800, 1807, 1808, 1818, 1819, 1824, 1828, 1836, 1863, 1878, 1887, 1888, 1896, 1914, 1925, 1927,
            1939, 1965, 1967, 1968, 1976, 1982, 1983, 1995]
HARD = {"apple": 38, "crate": 31, "grip_l": 27}
xml = pi.MODELS["rby1_randomized_pick_place_16d"]["model_xml"] if hasattr(pi, "MODELS") else pi.MODEL_XML_TRANSPORT
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
def jadr(name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name); assert j >= 0, name
    return int(m.jnt_qposadr[j]), int(m.jnt_bodyid[j])
apple_adr, apple_body = jadr(pi.OBJECT_JOINTS["apple"]); crate_adr, crate_body = jadr(pi.CRATE_JOINT); gl_adr, _ = jadr(pi.GRIPPER_L_JOINT)
by_name = {"apple": apple_adr, "crate": crate_adr, "grip_l": gl_adr}
res = {"model_xml": xml, "model_xml_resolved": str(pathlib.Path(xml).resolve()), "nq": int(m.nq),
       "joint_names": {"apple": pi.OBJECT_JOINTS["apple"], "crate": pi.CRATE_JOINT, "grip_l": pi.GRIPPER_L_JOINT},
       "body_names": {"apple": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, apple_body),
                      "crate": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, crate_body)},
       "qposadr_by_name": by_name, "qposadr_hardcoded": HARD, "model_addresses_match": by_name == HARD,
       "object_joints": {k: jadr(v)[0] for k, v in pi.OBJECT_JOINTS.items()}}
recs = [json.loads(l) for l in (pi.RANDOMIZED_16D_DATASET / "meta" / "randomized_episodes.jsonl").read_text().splitlines() if l.strip()]
scan = sorted(r["episode_index"] for r in recs if 1800 <= r["episode_index"] <= 1999 and r.get("split") == "test"
              and r.get("target_fruit") == "apple" and r.get("used_arm") == "left")
res["dataset_scan_apple_left_test_1800_1999"] = scan
res["dataset_scan_equals_task_list"] = scan == sorted(TASK_EPS)
res["splits_of_task_eps"] = sorted({pi.randomized_split_of(e) for e in TASK_EPS}, key=str)
eps = {}
for e in TASK_EPS:
    r = pi.load_randomized_episode(e)
    pi.reset_randomized_scene(m, d, r, settle_seconds=0.0)
    a0 = np.asarray(r["initial_fruit_poses"]["apple"]); c0 = np.asarray(r["basket_pose"])
    pre_ok = bool(np.array_equal(d.qpos[38:45], a0) and np.array_equal(d.qpos[31:38], c0))
    pi.reset_randomized_scene(m, d, r)  # client default settle 1.5 s
    mujoco.mj_kinematics(m, d)  # mj_step leaves xpos one integration step behind qpos; recompute at the current qpos
    apple_body_vs_qpos = float(np.abs(d.xpos[apple_body] - d.qpos[38:41]).max())
    crate_body_vs_qpos = float(max(np.abs(d.xpos[crate_body] - d.qpos[31:34]).max(), np.abs(d.xquat[crate_body] - d.qpos[34:38]).max()))
    eps[e] = {"split": r.get("split"), "target_fruit": r.get("target_fruit"), "used_arm": r.get("used_arm"),
              "fruits": sorted(r["initial_fruit_poses"]), "internal_container_body": r.get("internal_container_body"),
              "pre_settle_qpos_equals_record_at_hardcoded_adr": pre_ok,
              "settled_apple_body_xpos_minus_qpos38_max_abs": apple_body_vs_qpos,
              "settled_crate_body_pose_minus_qpos31_max_abs": crate_body_vs_qpos,
              "settled_apple_z": float(d.qpos[40]), "settled_grip_l_qpos27": float(d.qpos[27])}
res["episodes"] = eps
res["all_episodes_consistent"] = bool(res["model_addresses_match"] and all(
    v["pre_settle_qpos_equals_record_at_hardcoded_adr"] and v["target_fruit"] == "apple" and v["used_arm"] == "left"
    and v["split"] == "test" and v["settled_apple_body_xpos_minus_qpos38_max_abs"] < 1e-9 for v in eps.values()))
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=1))
for e, v in eps.items():
    print(e, v["split"], v["target_fruit"], v["used_arm"], v["pre_settle_qpos_equals_record_at_hardcoded_adr"],
          f'{v["settled_apple_body_xpos_minus_qpos38_max_abs"]:.1e}', f'{v["settled_crate_body_pose_minus_qpos31_max_abs"]:.1e}')
