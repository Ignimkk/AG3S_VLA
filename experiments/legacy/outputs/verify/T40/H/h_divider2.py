from common.workspace import workspace_path as _ag3s_workspace_path
import sys, importlib.util, numpy as np, mujoco, json
S=_ag3s_workspace_path('outputs/verify/T40/H/')
sys.path.insert(0,_ag3s_workspace_path('integration/rby1/manipulation/src')); sys.path.insert(0,_ag3s_workspace_path(''))
from rby1_manipulation.simulation import pick_place_obstacles as NEW
from tests.sim import _gjk as gjk
spec=importlib.util.spec_from_file_location("oldppo", S+"h_old_ppo.py"); OLD=importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
sys.path.insert(0,S); import replay as RP
cl=RP.load_client()
# reach the policy call with the divider config; the old code raises SystemExit at the reset object check
import pathlib, io, contextlib, inspect
cfgp=_ag3s_workspace_path('outputs/verify/T40/A/configs/1967_19672_divider.json'); prof="t40_ep1967_s19672_divider"
cap={}
def stub(remote):
    cap.update(inspect.currentframe().f_back.f_locals); raise RP._R
cl.load_remote_policy=stub
sys.argv=["x","--model","rby1_randomized_pick_place_16d","--remote","localhost:1","--episode-index","1967","--headless","--warmup-steps","1000","--policy-seed","19672","--obstacle-config",cfgp,"--obstacle-profile",prof]
out=io.StringIO()
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out): cl.main()
except RP._R: print("reached policy call")
except SystemExit as e: print("SystemExit", e)
print([l for l in out.getvalue().splitlines() if "obstacle" in l.lower() or "clearance" in l.lower()][-3:])
m,d,mg=cap["m"],cap["d"],cap["obstacle_manager"]
o=OLD.PickPlaceObstacleManager.__new__(OLD.PickPlaceObstacleManager); o.model,o.data=m,d; o.active_slots=mg.active_slots; o._collision_geoms=mg._collision_geoms
ref=min([0.25]+[gjk.gjk_dist(gjk.make_support(m,d,a),gjk.make_support(m,d,b),d.geom_xpos[b]-d.geom_xpos[a]+1e-9) for a in mg.collision_geom_ids for b in mg.payload_geom_ids])
print("object clearance new %.6f old %.6f gjk %.6f"%(mg.object_clearance(), OLD.PickPlaceObstacleManager._minimum_distance(o, mg.payload_geom_ids, 0.25), ref))
ft=np.zeros(6)
for a in mg.collision_geom_ids:
    for b in mg.payload_geom_ids:
        v=mujoco.mj_geomDistance(m,d,a,b,0.25,ft)
        if v<=0: print("  raw zero", mujoco.mj_id2name(m,8,a), m.geom_type[a], mujoco.mj_id2name(m,8,b), m.geom_type[b], v, "gjk", gjk.gjk_dist(gjk.make_support(m,d,a),gjk.make_support(m,d,b),d.geom_xpos[b]-d.geom_xpos[a]+1e-9))
