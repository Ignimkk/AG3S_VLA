"""Replay a T40 G run from main() reset + recorded applied_ctrl, sub-step by sub-step, with the client's stop check."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, os, io, inspect, contextlib, importlib.util, pathlib, time
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
HY = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid'))
sys.path.insert(0, str(HY / "rby1_manipulation" / "src")); sys.path.insert(0, _ag3s_workspace_path(''))
from tests.sim import _gjk as gjk
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs')); F = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/configs'))

class _R(Exception): pass

def load_client():
    spec = importlib.util.spec_from_file_location("_cl", HY / "rby1_bringup/pi05_infer.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def reset(cl, ep, seed, prof):
    cap = {}
    def stub(remote):
        cap.update(inspect.currentframe().f_back.f_locals); raise _R
    orig = cl.load_remote_policy; cl.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index", str(ep),
                "--headless", "--warmup-steps", "1000", "--policy-seed", str(seed), "--obstacle-stop-distance", "0",
                "--obstacle-config", str(F / f"{prof}.json"), "--obstacle-profile", prof]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out): cl.main()
    except _R: pass
    finally: cl.load_remote_policy = orig
    return cap

def replay(cl, run, check=None):
    rd = G / run
    man = json.load(open(rd / "manifest.json")); prof = man["scene"]["obstacle_profile"]; ep = man["scene"]["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"])
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; rows = [r for r in rows if r["kind"] == "control"]
    L = reset(cl, ep, seed, prof); m, d, mg, act = L["m"], L["d"], L["obstacle_manager"], L["act"]
    spa = max(1, int(round(1.0 / (cl.CTRL_HZ * m.opt.timestep))))
    for _ in range(1000): mujoco.mj_step(m, d)
    n = 7; mism = 0; stop = None; checks = []
    for r in rows:
        dq = np.max(np.abs(np.asarray(r["qpos"]) - d.qpos))
        if dq != 0.0: mism += 1
        a = r["applied_ctrl"]
        for i in range(n):
            d.ctrl[act["left_a"][i]] = a["arm"][i]; d.ctrl[act["right_a"][i]] = a["arm"][n + i]
        d.ctrl[act["left_grip_a"]] = a["gripper"][0] * cl.RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = a["gripper"][1] * cl.RBY1_GRIPPER_OPEN
        for s in range(spa):
            qprev = d.qpos.copy()
            mujoco.mj_step(m, d); mg.observe_contacts()
            if s % 5 == 0:
                c = mg.robot_clearance()
                info = check(m, d, mg) if check else None
                checks.append((r["t_step"], s, c, info))
                if mg.robot_collision or c <= 0.0:
                    stop = (r["t_step"], s, c, qprev, d.geom_xpos.copy()); break
        if stop: break
    return dict(run=run, prof=prof, ep=ep, seed=seed, spa=spa, nrows=len(rows), last_t=rows[-1]["t_step"], qpos_mismatch_rows=mism,
                stop=stop, checks=checks, m=m, d=d, mg=mg)

RUNS = ["E3b/C/E3b_ep1995_s19952", "E3b/D/E3b_ep1967_s19672", "E3b/D/E3b_ep1982_s19822", "E3b/D/E3b_ep1995_s19952", "E0/D/E0_ep1967_s19672"]
if __name__ == "__main__":
    cl = load_client()
    nm = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}"
    for run in (sys.argv[1:] or RUNS):
        t0 = time.time(); R = replay(cl, run)
        m, d, mg = R["m"], R["d"], R["mg"]
        print(run, R["prof"], "rows", R["nrows"], "last_t", R["last_t"], "spa", R["spa"], "qpos mismatched rows", R["qpos_mismatch_rows"], f"{time.time()-t0:.0f}s")
        if R["stop"]:
            t, s, c, q, v = R["stop"]; print("  STOP at t", t, "substep", s, "clearance", c)
            np.savez(f"{_ag3s_workspace_path()}/outputs/verify/T40/H/stop_{run.replace('/','_')}.npz", qpos_kin=q, geom_xpos=v, t=t, s=s)
            ft = np.zeros(6)
            for o in sorted(mg.collision_geom_ids):
                for g in mg.robot_geom_ids:
                    val = mg.pair_distance(o, g, 0.25, ft)
                    if val <= 0.0:
                        ref = gjk.gjk_dist(gjk.make_support(m, d, o), gjk.make_support(m, d, g), d.geom_xpos[g] - d.geom_xpos[o] + 1e-9)
                        print("   ", nm(m, o), int(m.geom_type[o]), nm(m, g), g, int(m.geom_type[g]), val, "fromto gap", np.linalg.norm(ft[:3]-ft[3:]), "gjk", ref)
        else:
            print("  no stop reproduced")
