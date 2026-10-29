"""T40 H sweep: replay each run from main() reset + recorded applied_ctrl; at every client stop-check state (every 5th sub-step)
and at every recorded control row (kinematic): new robot/object clearance vs independent GJK minimum (tests/sim/_gjk.py), and the
pre-H (98e07d6) raw minimum. Also counts certify refusals."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, importlib.util, numpy as np, mujoco
S=_ag3s_workspace_path('outputs/verify/T40/H/')
sys.path.insert(0,S); import replay as RP
from tests.sim import _gjk as gjk
from rby1_manipulation.simulation import convex_distance as CD
spec=importlib.util.spec_from_file_location("oldppo", S+"h_old_ppo.py"); OLD=importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
CAP=0.25
def refmin(m,d,obst,others,cap=CAP):
    obst=np.array(sorted(obst)); others=np.array(others)
    gap=np.linalg.norm(d.geom_xpos[others][None]-d.geom_xpos[obst][:,None],axis=2)-m.geom_rbound[others][None]-m.geom_rbound[obst][:,None]
    best=cap
    for idx in np.argsort(gap,axis=None):
        if gap.flat[idx]>best: break
        i,j=divmod(int(idx),len(others)); o,g=int(obst[i]),int(others[j])
        best=min(best,gjk.gjk_dist(gjk.make_support(m,d,o),gjk.make_support(m,d,g),d.geom_xpos[g]-d.geom_xpos[o]+1e-9))
    return best
refusals={"n":0}
orig=CD.certify
def counting(*a,**k):
    r=orig(*a,**k)
    if not r: refusals["n"]+=1
    return r
import rby1_manipulation.simulation.pick_place_obstacles as NEWPPO
NEWPPO.certify=counting
def run(name):
    cl=RP.load_client(); rows_out=[]
    oldmg={}
    def check(m,d,mg):
        if "o" not in oldmg:
            cfg=mg.config; o=OLD.PickPlaceObstacleManager.__new__(OLD.PickPlaceObstacleManager)
            # old manager without park/activate side effects: share geometry bookkeeping
            o.model, o.data = m, d; o.active_slots=mg.active_slots; o._collision_geoms=mg._collision_geoms
            oldmg["o"]=o
        o=oldmg["o"]
        t=time.perf_counter(); rn=mg._minimum_distance(mg.robot_geom_ids, CAP); t1=time.perf_counter()
        on=mg._minimum_distance(mg.payload_geom_ids, CAP)
        ro=OLD.PickPlaceObstacleManager._minimum_distance(o, mg.robot_geom_ids, CAP); t2=time.perf_counter()
        return dict(rn=rn,on=on,ro=ro,oo=OLD.PickPlaceObstacleManager._minimum_distance(o, mg.payload_geom_ids, CAP),
                    rr=refmin(m,d,mg.collision_geom_ids,mg.robot_geom_ids), rp=refmin(m,d,mg.collision_geom_ids,mg.payload_geom_ids),
                    t_new_ms=(t1-t)*1e3, t_old_ms=(t2-t1)*1e3)
    R=RP.replay(cl,name,check)
    m,d,mg=R["m"],R["d"],R["mg"]
    states=[dict(kind="substep",t=t,s=s,**info) for (t,s,c,info) in R["checks"]]
    # control rows (kinematic, recorded qpos)
    rows=[json.loads(l) for l in open(RP.G/name/"frames.jsonl") if l.strip()]; rows=[r for r in rows if r["kind"]=="control"]
    for r in rows:
        d.qpos[:]=r["qpos"]; mujoco.mj_kinematics(m,d)
        states.append(dict(kind="row",t=r["t_step"],s=None,**check(m,d,mg)))
    return dict(run=name, qpos_mismatch_rows=R["qpos_mismatch_rows"], nrows=R["nrows"], stopped=R["stop"] is not None, certify_refusals=refusals["n"], states=states)
if __name__=="__main__":
    NAME=sys.argv[1]; out=run(NAME); json.dump(out,open(S+"h_sweep_"+NAME.replace("/","_")+".json","w"))
    st=out["states"]; f=lambda k: [x[k] for x in st]
    err_r=max(abs(a-b) for a,b in zip(f("rn"),f("rr"))); err_p=max(abs(a-b) for a,b in zip(f("on"),f("rp")))
    err_old=max(abs(a-b) for a,b in zip(f("ro"),f("rr")))
    print(NAME, "states",len(st),"rows",out["nrows"],"qpos mismatch",out["qpos_mismatch_rows"],"replay stopped",out["stopped"],"refusals",out["certify_refusals"],
          "max|new-gjk| robot %.2e payload %.2e | old robot max err %.4f n_old_zero %d | t new %.2f old %.2f ms"%(err_r,err_p,err_old,sum(x==0.0 for x in f("ro")),np.mean(f("t_new_ms")),np.mean(f("t_old_ms"))))
