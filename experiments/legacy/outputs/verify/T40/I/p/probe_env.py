from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys
os.environ.setdefault("MUJOCO_GL","osmesa"); os.environ["JAX_PLATFORMS"]="cpu"; os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/p'))
import hurdle_metrics as HM, mujoco, numpy as np
M = HM.model(); m = M["m"]
for g in [1051,1052,1053,1079,1085,1086]:
    print(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])), int(m.geom_type[g]), m.geom_size[g], int(m.geom_dataid[g]))
for b in ["EE_BODY_L","ee_finger_l1","ee_finger_l2","ee_left","base"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b); print(b, bid, [g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and (m.geom_contype[g] or m.geom_conaffinity[g])])
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
print(FIA.__file__)
