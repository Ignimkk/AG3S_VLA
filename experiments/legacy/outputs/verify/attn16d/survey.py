"""Survey candidate 16D records: head joint qpos, grasp onset, gripper traces."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
from AG3S.ag3s.experiments.sources.policy_record import load_run, grasp_onset, gripper_columns_for_state
RECS = sys.argv[1:]
out = {}
model = None
for r in RECS:
    rec = load_run(r)
    if model is None:
        model = mujoco.MjModel.from_xml_path(rec.model_xml)
    adr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("head_0", "head_1")}
    q = np.stack([s.qpos for s in rec])
    st = np.stack([s.state for s in rec])
    cols = gripper_columns_for_state(st.shape[1])
    on = grasp_onset(rec)
    out[r] = dict(n=len(rec), prompt=rec.prompt,
                  head_0=[float(q[:, adr["head_0"]].min()), float(q[:, adr["head_0"]].max())],
                  head_1=[float(q[:, adr["head_1"]].min()), float(q[:, adr["head_1"]].max())],
                  grasp_onset=on, t_steps=[int(rec.steps[0].t_step), int(rec.steps[-1].t_step)],
                  grip_left=[round(float(x), 3) for x in st[:, cols[0]]],
                  grip_right=[round(float(x), 3) for x in st[:, cols[1]]])
    print(r, json.dumps({k: v for k, v in out[r].items() if not k.startswith("grip")}))
    print("  L", out[r]["grip_left"])
    print("  R", out[r]["grip_right"])
json.dump(out, open(_ag3s_workspace_path('outputs/verify/attn16d/survey.json'), "w"), indent=1)
