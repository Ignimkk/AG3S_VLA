"""T42-2 placement helpers (verifier, 2026-10-05; generalized from T40/F/p/flib.py). CPU only.
Client code = immutable snapshot ${AG3S_PROJECT_ROOT}-t42c (pi05_TO_hybrid fce4af8 = c7c1cc9 black obstacles + avoidance-arrow overlay; slots post_0 / fence_0 /
overbar_0 / crate_post_0 + hurdle_0, T40 H certified pair_distance).

reset state : snapshot pi05_infer.main() for the episode, run up to load_remote_policy (stubbed; no server) with an obstacle
              profile so the obstacle model (model_transport_pick_place_obstacles.xml) is loaded; during the capture the manager's
              activate() only parks (the scene state does not depend on the obstacle: mocap only, no mj_step before the check).
(a) reset   : the snapshot PickPlaceObstacleManager on the captured (m, d), obstacles placed by manager._place (the call activate()
              makes), robot_clearance() / object_clearance() with the client defaults (max_distance 0.25). Client rule (stop
              distance 0): robot <= 0 or object <= 0 -> refuse.
distances   : manager.pair_distance (certified, T40 H) on recorded rows: robot / fruit / crate geom poses from mj_kinematics on the
              recorded control-row qpos (cached), written into d before each call; bounding-sphere prefilter; capped at CAP.
recorded    : T42-1 no-obstacle runs E0 and E3b of the episode's seed rep 1 (apple + left: the T39 runs of seed rep 1); row 0 =
              recorded planning row 0 (t0, after warm-up), then the 600 control rows.
windows     : grasp_t / place_t = analyze_t42 (planning rows); lift_t = first control row after the last row t <= grasp_t whose
              target centre z < z0 + 2 mm (T40 A rule); pick window = rows 0 <= t < lift_t (approach + grasp + lift-off);
              approach window = pick-window rows after the last row whose TCP is > 0.12 m (xy) from the target's reset centre
              (excludes an earlier failed grasp attempt and its empty excursion; = pick window when there is none);
              carry window = lift_t <= t < place_t; transport window = carry rows from the first row whose target centre
              is >= 0.05 m (xy) from its reset centre (the vertical lift-off column, shared by pick and place, excluded).
warmup      : self.wu = the reset state stepped 1000 times (mj_step, ctrl as captured, obstacles parked), rows every 50 steps
              (criterion a', T43 P convention).
geom groups : GRIP = EE_BODY_<arm> + ee_finger_<arm>1/2; ROBOT = every collision geom under body 'base'; TARGET = the target
              fruit; FRUITS = the three other fruits; CRATE = crate geoms. OBST = collision geoms of the placed slots."""
from common.workspace import workspace_path as _ag3s_workspace_path
import copy, json, math, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t42c")
sys.path.insert(0, str(SNAP / "integration" / "rby1" / "manipulation" / "src"))
import numpy as np, mujoco  # noqa: E402
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402
assert PPO.__file__.startswith(str(SNAP)), PPO.__file__
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402

TABLE_Z = 0.82
CAP = 0.30
CAPDIR = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2/raw/capture'))
T42R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/1/runs'))
T39R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs'))


def _client():
    sys.path.insert(0, str(SNAP / "integration" / "rby1" / "bringup"))
    import pi05_infer as P
    assert P.__file__.startswith(str(SNAP)), P.__file__
    return P


class _Reached(Exception):
    pass


def save_capture(ep):
    P = _client(); cap = {}

    class Park(PPO.PickPlaceObstacleManager):
        def activate(self, profile_name):
            self.park_all(); self.reset_metrics(); self.profile_name = profile_name

    def stub(remote):
        import inspect
        cap.update(inspect.currentframe().f_back.f_locals); raise _Reached

    cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "obstacles": []},
           "cap": {"scene": "fruit", "obstacles": [{"slot": "hurdle_0", "position": [0.80, -0.30, TABLE_Z], "yaw": 0.0}]}}}
    tmp = pathlib.Path(f"/tmp/claude-0/t42_2_cap_{os.getpid()}.json"); tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cfg))
    orig = P.PickPlaceObstacleManager, P.load_remote_policy, sys.argv
    P.PickPlaceObstacleManager = Park; P.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index",
                str(ep), "--headless", "--obstacle-config", str(tmp), "--obstacle-profile", "cap"]
    try:
        P.main()
    except _Reached:
        pass
    finally:
        P.PickPlaceObstacleManager, P.load_remote_policy, sys.argv = orig
    m, d = cap["m"], cap["d"]
    CAPDIR.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(CAPDIR / f"{ep}.mjb"), None)
    np.savez(CAPDIR / f"{ep}_reset.npz", qpos=d.qpos, qvel=d.qvel, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat, ctrl=d.ctrl,
             act=d.act, model_xml=np.asarray(cap["model_xml"]))
    return cap["model_xml"]


def run_dir(cond, ep, seed):
    for base, sub in ((T42R / cond / "noobs", ""), (T39R / cond, "")):
        rd = base / f"{cond}_ep{ep}_s{seed}"
        if rd.exists() and (base / f"{cond}_ep{ep}_s{seed}.done").exists():
            return rd
    raise FileNotFoundError(f"{cond} ep{ep} s{seed}")


def windows(rd, a):
    """grasp_t / place_t (analyze_t42) and lift_t (T40 A rule) of one recorded run."""
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
    adr = AN.QADR[a["target_fruit"]]; z0 = plan[0]["object_poses"][a["target_fruit"]]["pos"][2]
    t = np.array([c["t_step"] for c in ctrl]); z = np.array([c["qpos"][adr + 2] for c in ctrl])
    lift_t = None
    if a["grasp_t"] is not None:
        k = np.where((t <= a["grasp_t"]) & (z < z0 + 0.002))[0]
        lift_t = int(t[k[-1] + 1]) if len(k) and k[-1] + 1 < len(t) else None
    return {"grasp_t": a["grasp_t"], "place_t": a["place_t"], "lift_t": lift_t, "success": a["success"]}


class Ctx:
    def __init__(self, ep, seed):
        self.ep, self.seed = ep, seed
        self.m = mujoco.MjModel.from_binary_path(str(CAPDIR / f"{ep}.mjb")); self.d = mujoco.MjData(self.m)
        z = np.load(CAPDIR / f"{ep}_reset.npz"); self.model_xml = str(z["model_xml"])
        d = self.d; d.qpos[:] = z["qpos"]; d.qvel[:] = z["qvel"]; d.mocap_pos[:] = z["mocap_pos"]; d.mocap_quat[:] = z["mocap_quat"]
        d.ctrl[:] = z["ctrl"]
        if self.m.na:
            d.act[:] = z["act"]
        mujoco.mj_forward(self.m, d); m = self.m
        self.q_reset = d.qpos.copy(); self.mocap_reset = (d.mocap_pos.copy(), d.mocap_quat.copy())
        self.cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "obstacles": []}}}
        self.mgr = PPO.PickPlaceObstacleManager(m, d, self.cfg)
        self.compiled = {f: getattr(m, f).copy() for f in ("geom_size", "geom_rbound", "geom_aabb", "bvh_aabb")}
        a0 = AN.analyze(run_dir("E3b", ep, seed))
        self.fruit, self.arm = a0["target_fruit"], a0["used_arm"]
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        gb = lambda names: np.array([g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in {B(n) for n in names}], int)

        def desc(b, root):
            while b > 0:
                if b == root:
                    return True
                b = int(m.body_parentid[b])
            return False
        s = "L" if self.arm == "left" else "R"; s2 = s.lower()
        self.GRIP = gb([f"EE_BODY_{s}", f"ee_finger_{s2}1", f"ee_finger_{s2}2"])
        self.TARGET = gb([self.fruit])
        base = B("base")
        self.ROBOT = np.array([g for g in range(m.ngeom) if coll[g] and desc(int(m.geom_bodyid[g]), base)], int)
        self.FRUITS = gb([f for f in ("apple", "banana", "orange", "pear") if f != self.fruit])
        self.CRATE = gb(["crate"])
        self.ALL = np.unique(np.concatenate([self.ROBOT, self.TARGET, self.FRUITS, self.CRATE]))
        self.pos_in_all = {int(g): i for i, g in enumerate(self.ALL)}
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{self.arm}_ee")
        self.gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
        self.runs = {}
        for cond in ("E0", "E3b"):
            rd = run_dir(cond, ep, seed); a = AN.analyze(rd)
            rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
            ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
            Q = np.vstack([np.array(plan[0]["qpos"])[None], np.array([c["qpos"] for c in ctrl])]); t = np.r_[-1, [c["t_step"] for c in ctrl]]
            xp, xm, tcp = self.pose_cache(Q)
            self.runs[cond] = {"run_dir": str(rd), "t": t, "Q": Q, "xpos": xp, "xmat": xm, "tcp": tcp, **windows(rd, a)}
        # warmup replay (criterion a'): 1000 mj_step from the captured reset state, obstacles parked, every 50 steps (row 0 = reset)
        dw = mujoco.MjData(m); dw.qpos[:] = z["qpos"]; dw.qvel[:] = z["qvel"]; dw.mocap_pos[:] = z["mocap_pos"]; dw.mocap_quat[:] = z["mocap_quat"]
        dw.ctrl[:] = z["ctrl"]
        if m.na:
            dw.act[:] = z["act"]
        mujoco.mj_forward(m, dw); QW = [dw.qpos.copy()]
        for k in range(1, 1001):
            mujoco.mj_step(m, dw)
            if k % 50 == 0:
                QW.append(dw.qpos.copy())
        QW = np.array(QW); xp, xm, tcp = self.pose_cache(QW)
        self.wu = {"t": np.arange(len(QW)) * 50, "Q": QW, "xpos": xp, "xmat": xm, "tcp": tcp}
        self.restore()

    def pose_cache(self, Q):
        m, d = self.m, self.d
        xp = np.zeros((len(Q), len(self.ALL), 3)); xm = np.zeros((len(Q), len(self.ALL), 9)); tcp = np.zeros((len(Q), 3))
        for i, q in enumerate(Q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            xp[i] = d.geom_xpos[self.ALL]; xm[i] = d.geom_xmat[self.ALL]; tcp[i] = d.site_xpos[self.site]
        return xp, xm, tcp

    def restore(self):
        self.mgr.park_all()
        self.d.qpos[:] = self.q_reset; mujoco.mj_forward(self.m, self.d)

    def place(self, obstacles, exact=False):
        """obstacles: list of config entries. exact=True: manager.activate on a profile (the client call)."""
        self.d.qpos[:] = self.q_reset
        if exact:
            self.mgr.config["profiles"]["cand"] = {"scene": "fruit", "obstacles": copy.deepcopy(obstacles)}
            self.mgr.config = PPO.validate_pick_place_obstacle_config(self.mgr.config)
            self.mgr.activate("cand")
        else:
            self.mgr.park_all(); self.d.qpos[:] = self.q_reset
            for ob in obstacles:
                self.mgr._place(ob["slot"], ob["position"], float(ob.get("yaw", 0.0)),
                                sizes={k: ob[k] for k in PPO.SLOT_PARAMS.get(ob["slot"], {}) if k in ob})
            self.mgr.active_slots = tuple(ob["slot"] for ob in obstacles); self.mgr.profile_name = "cand"
            mujoco.mj_kinematics(self.m, self.d)
        self.OBST = np.array(sorted(self.mgr.collision_geom_ids), int)

    def reset_check(self):
        return float(self.mgr.robot_clearance()), float(self.mgr.object_clearance())

    def rows(self, cond, phase):
        R = self.runs[cond]; t = R["t"]
        if phase == "t0":
            return np.array([0])
        if phase == "pick":
            hi = R["lift_t"] if R["lift_t"] is not None else 10 ** 9
            return np.where((t >= 0) & (t < hi))[0]
        if phase == "carry":
            if R["lift_t"] is None or R["place_t"] is None:
                return np.array([], int)
            return np.where((t >= R["lift_t"]) & (t < R["place_t"]))[0]
        if phase == "approach":
            # final approach to the grasp that lifted: rows after the last pick-window row whose TCP is > 0.12 m (xy) from the
            # target's reset centre (an earlier failed attempt and its empty excursion are excluded), up to lift_t
            rows = self.rows(cond, "pick")
            a = AN.QADR[self.fruit]; xy0 = R["Q"][0][a:a + 2]
            far = np.linalg.norm(R["tcp"][rows][:, :2] - xy0, axis=1) > 0.12
            return rows[np.flatnonzero(far)[-1] + 1:] if far.any() else rows
        if phase == "transport":
            # carry rows after the target has moved >= 0.05 m in xy from its reset centre (the lift-off column excluded)
            rows = self.rows(cond, "carry")
            if not len(rows):
                return rows
            a = AN.QADR[self.fruit]; xy0 = R["Q"][0][a:a + 2]
            mv = np.linalg.norm(R["Q"][rows][:, a:a + 2] - xy0, axis=1) >= 0.05
            return rows[np.argmax(mv):] if mv.any() else np.array([], int)
        if phase == "all":
            return np.where(t >= 0)[0]
        raise ValueError(phase)

    def series(self, cond, rows, sets, cap=CAP, want_pair=False):
        R = cond if isinstance(cond, dict) else self.runs[cond]; m, d = self.m, self.d; OB = self.OBST
        hx = d.geom_xpos[OB].copy(); hr = m.geom_rbound[OB]
        out = {k: np.full(len(rows), cap) for k in sets}; pairs = {k: [None] * len(rows) for k in sets}
        idx = {k: np.array([self.pos_in_all[int(g)] for g in v], int) for k, v in sets.items()}
        allidx = np.unique(np.concatenate(list(idx.values())))
        ft = np.zeros(6)
        for j, i in enumerate(rows):
            P_ = R["xpos"][i]
            gap = np.linalg.norm(P_[allidx][:, None, :] - hx[None], axis=2) - m.geom_rbound[self.ALL[allidx]][:, None] - hr[None]
            near = gap.min(axis=1) <= cap
            if not near.any():
                continue
            nidx = allidx[near]
            d.geom_xpos[self.ALL[nidx]] = R["xpos"][i][nidx]; d.geom_xmat[self.ALL[nidx]] = R["xmat"][i][nidx]
            nset = set(nidx.tolist())
            for k, ix in idx.items():
                best, bp = out[k][j], None
                for a in ix:
                    if a not in nset:
                        continue
                    g = int(self.ALL[a])
                    for o in OB:
                        if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[o]) - m.geom_rbound[g] - m.geom_rbound[o] > min(cap, best):
                            continue
                        v = self.mgr.pair_distance(int(o), g, max(min(cap, best), 1e-6), ft)
                        if v < best:
                            best, bp = v, (self.gname(o), self.gname(g))
                out[k][j] = best; pairs[k][j] = bp
        mujoco.mj_kinematics(m, d)   # geom poses back to the placed reset scene
        return (out, pairs) if want_pair else out

    def first_le0(self, cond, rows, geoms, thr=0.0):
        R = cond if isinstance(cond, dict) else self.runs[cond]; m, d = self.m, self.d; OB = self.OBST
        hx = d.geom_xpos[OB].copy(); hr = m.geom_rbound[OB]
        ix = np.array([self.pos_in_all[int(g)] for g in geoms], int); gr = m.geom_rbound[self.ALL[ix]]
        cap = thr + 0.001; ft = np.zeros(6); hit = None
        for j, i in enumerate(rows):
            gap = np.linalg.norm(R["xpos"][i][ix][:, None, :] - hx[None], axis=2) - gr[:, None] - hr[None]
            near = np.argwhere(gap <= thr)
            for a, o in near:
                g = int(self.ALL[ix[a]])
                d.geom_xpos[g] = R["xpos"][i][ix[a]]; d.geom_xmat[g] = R["xmat"][i][ix[a]]
                if self.mgr.pair_distance(int(OB[o]), g, cap, ft) <= thr:
                    hit = j; break
            if hit is not None:
                break
        mujoco.mj_kinematics(m, d)
        return hit


def write_config(path, profiles):
    """profiles: name -> (description, [entries])"""
    cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "description": "no obstacle", "obstacles": []}}}
    for name, (desc, obs) in profiles.items():
        cfg["profiles"][name] = {"scene": "fruit", "description": desc, "obstacles": obs}
    PPO.validate_pick_place_obstacle_config(copy.deepcopy(cfg))
    pathlib.Path(path).write_text(json.dumps(cfg, indent=1))
    return PPO.load_pick_place_obstacle_config(path)
