from common.workspace import workspace_path as _ag3s_workspace_path
import time, logging, sys, resource
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
from AG3S.trajopt.serve_safe import build_ag3s
M=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
cover = None if sys.argv[1]=="off" else {"hands": sys.argv[1]}
t=time.time()
ag=build_ag3s(M, voxel=0.02, range_max=2.0, links="gripper", backend="legacy", constraint_sphere_options=dict(sphere_spacing=0.4,max_spheres_per_capsule=32,capsule_radius_scale=0.05),
              self_collision=False, target_field_policy="exclude_authorized", plan_horizon_steps=8, rows_per_step=24, gripper_cover=cover)
print("build_ag3s s", round(time.time()-t,1), "constraint spheres", ag.constraint_robot_model.n_spheres, "maxrss GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,2))
b=ag.builder; print("builder rows", getattr(b,"n_constraints",None))
import numpy as np
from AG3S.ag3s.types import ContactPolicyContext
ctx=ContactPolicyContext.make("approach",["left"])
ts=[]
for _ in range(5):
    t=time.perf_counter(); spec=b.build((),(),context=ctx); ts.append((time.perf_counter()-t)*1e3)
print("builder.build ms (no candidates)", [round(x,1) for x in ts])
