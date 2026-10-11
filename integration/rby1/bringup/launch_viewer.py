import pathlib

import mujoco
import mujoco.viewer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = str(REPO_ROOT / "robots" / "rby1" / "description" / "models" / "rby1a" / "mujoco" / "model.xml")

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.viewer.launch(model, data)
