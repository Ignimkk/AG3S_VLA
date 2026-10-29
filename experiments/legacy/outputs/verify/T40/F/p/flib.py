"""T40 F shared helpers (verifier, CPU only). Hurdle placement checks with the CLIENT's own code from the immutable
snapshot ${AG3S_PROJECT_ROOT}-t40 (pi05_TO_hybrid 98e07d6).

reset state  : pi05_infer.main() of the snapshot, run up to load_remote_policy (stubbed; no server), with an obstacle
               profile so the obstacle XML (model_transport_pick_place_obstacles.xml) is loaded. During this capture the
               manager class is replaced by a subclass whose activate() only parks (so main's reset check passes for any
               placement); the scene state does not depend on the obstacle (mocap only, no mj_step before the check).
(a) reset    : PickPlaceObstacleManager (snapshot) on the captured (m, d): activate(profile) -> robot_clearance() and
               object_clearance() with the client's defaults (max_distance 0.25), the exact calls main() makes
               (pi05_infer.py:1159-1173). Client rule: robot <= stop_distance or object <= 0 -> refuse.
distances    : manager.pair_distance (mj_geomDistance; capsule-box -> capsule_box_distance), same bounding-sphere
               prefilter as manager._minimum_distance, over arbitrary geom sets. Recorded rows: robot/fruit geom poses
               (geom_xpos, geom_xmat) are computed once per row by mj_kinematics on the recorded qpos and written into
               d before the distance calls (mj_geomDistance reads only geom_xpos/geom_xmat + model). Values capped at CAP.
geom sets    : HURDLE = hurdle_0 collision geoms (feet, sleeves, bar, posts); A_GRIP = EE_BODY_L + ee_finger_l1/l2;
               B_APPLE = apple geoms; C_ROBOT = every collision geom under body 'base' minus A_GRIP; ROBOT_ALL = A + C;
               FRUITS = banana/orange/pear; CRATE = crate geoms. (manager.robot_geom_ids = arms only, used by (a).)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import copy, json, math, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t40")
sys.path.insert(0, str(SNAP / "integration" / "rby1" / "manipulation" / "src"))
import numpy as np, mujoco  # noqa: E402
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402
assert PPO.__file__.startswith(str(SNAP)), PPO.__file__


def _client():
    """the snapshot client module (imported only for the reset capture; heavy)."""
    sys.path.insert(0, str(SNAP / "integration" / "rby1" / "bringup"))
    import pi05_infer as P
    assert P.__file__.startswith(str(SNAP)), P.__file__
    return P

T39 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs'))
T40A = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40A.verify.json')))["numbers"]["episodes"]
TAGS = ["1807_18071", "1968_19681", "1982_19822", "1828_18281", "1995_19952", "1967_19672"]
TABLE_Z = 0.82
CAP = 0.30


class _Reached(Exception):
    pass


def capture_reset(ep):
    """(m, d) of pi05_infer.main() for the episode at the point of the obstacle reset check."""
    P = _client()
    cap = {}

    class Park(PPO.PickPlaceObstacleManager):
        def activate(self, profile_name):
            self.park_all(); self.reset_metrics(); self.profile_name = profile_name

    def stub(remote):
        import inspect
        cap.update(inspect.currentframe().f_back.f_locals); raise _Reached

    cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "obstacles": []},
           "cap": {"scene": "fruit", "obstacles": [{"slot": "hurdle_0", "position": [0.80, -0.30, TABLE_Z], "yaw": 0.0}]}}}
    tmp = pathlib.Path(f"/tmp/claude-0/t40f_cap_{os.getpid()}.json"); tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cfg))
    orig_mgr, orig_lrp, argv = P.PickPlaceObstacleManager, P.load_remote_policy, sys.argv
    P.PickPlaceObstacleManager = Park; P.load_remote_policy = stub
    sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index",
                str(ep), "--headless", "--obstacle-config", str(tmp), "--obstacle-profile", "cap"]
    try:
        P.main()
    except _Reached:
        pass
    finally:
        P.PickPlaceObstacleManager, P.load_remote_policy, sys.argv = orig_mgr, orig_lrp, argv
    return cap["m"], cap["d"], cap["model_xml"]


CAPDIR = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/raw/capture'))


def save_capture(tag):
    """run once per episode (imports the client): model binary + reset state of pi05_infer.main() at the reset check."""
    ep = int(tag.split("_")[0])
    m, d, xml = capture_reset(ep)
    CAPDIR.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(CAPDIR / f"{tag}.mjb"), None)
    np.savez(CAPDIR / f"{tag}_reset.npz", qpos=d.qpos, qvel=d.qvel, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat,
             ctrl=d.ctrl, act=d.act, model_xml=np.asarray(xml))
    return xml


class Ctx:
    """one episode: captured reset scene (model binary + state saved by save_capture from the client's main()) + the
    snapshot manager + recorded T39 runs (E0, E3b) with cached geom poses."""

    def __init__(self, tag):
        self.tag = tag; ep, seed = (int(x) for x in tag.split("_")); self.ep, self.seed = ep, seed
        self.m = mujoco.MjModel.from_binary_path(str(CAPDIR / f"{tag}.mjb")); self.d = mujoco.MjData(self.m)
        z = np.load(CAPDIR / f"{tag}_reset.npz"); self.model_xml = str(z["model_xml"])
        self.d.qpos[:] = z["qpos"]; self.d.qvel[:] = z["qvel"]; self.d.mocap_pos[:] = z["mocap_pos"]; self.d.mocap_quat[:] = z["mocap_quat"]
        self.d.ctrl[:] = z["ctrl"]
        if self.m.na:
            self.d.act[:] = z["act"]
        mujoco.mj_forward(self.m, self.d)
        m = self.m
        self.q_reset = self.d.qpos.copy()
        self.cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "obstacles": []},
                    "cand": {"scene": "fruit", "obstacles": [{"slot": "hurdle_0", "position": [0.80, -0.30, TABLE_Z], "yaw": 0.0, "bar_height": 0.26}]}}}
        self.mgr = PPO.PickPlaceObstacleManager(m, self.d, self.cfg)
        self.HURDLE = np.array(sorted(self.mgr._collision_geoms["hurdle_0"]), int)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        gb = lambda names: np.array([g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in {B(n) for n in names}], int)

        def desc(b, root):
            while b > 0:
                if b == root:
                    return True
                b = int(m.body_parentid[b])
            return False
        self.A_GRIP = gb(["EE_BODY_L", "ee_finger_l1", "ee_finger_l2"])
        self.B_APPLE = gb(["apple"])
        base = B("base")
        self.ROBOT_ALL = np.array([g for g in range(m.ngeom) if coll[g] and desc(int(m.geom_bodyid[g]), base)], int)
        sA = set(self.A_GRIP.tolist())
        self.C_ROBOT = np.array([g for g in self.ROBOT_ALL if g not in sA], int)
        self.FRUITS = gb(["banana", "orange", "pear"])
        self.CRATE = gb(["crate"])
        self.ARMS = np.array(sorted(self.mgr.robot_geom_ids), int)
        self.ALL = np.unique(np.concatenate([self.ROBOT_ALL, self.B_APPLE, self.FRUITS, self.CRATE]))
        self.pos_in_all = {int(g): i for i, g in enumerate(self.ALL)}
        self.gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
        self.runs = {}
        for cond in ("E0", "E3b"):
            rd = T39 / cond / f"{cond}_ep{ep}_s{seed}"
            rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
            ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
            Q = np.array([c["qpos"] for c in ctrl]); t = np.array([c["t_step"] for c in ctrl])
            Q = np.vstack([np.array(plan[0]["qpos"])[None], Q])  # row 0 = recorded planning row 0 (t0, after warmup)
            t = np.r_[-1, t]
            xp, xm = self.pose_cache(Q)
            w = T40A[tag]["windows"][cond]
            self.runs[cond] = {"run_dir": str(rd), "t": t, "Q": Q, "xpos": xp, "xmat": xm, "lift_t": w["lift_t"],
                               "grasp_t": w["grasp_t"], "place_t": w["place_t"]}
        self.restore()

    def pose_cache(self, Q):
        m, d = self.m, self.d
        xp = np.zeros((len(Q), len(self.ALL), 3)); xm = np.zeros((len(Q), len(self.ALL), 9))
        for i, q in enumerate(Q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            xp[i] = d.geom_xpos[self.ALL]; xm[i] = d.geom_xmat[self.ALL]
        return xp, xm

    def restore(self):
        self.d.qpos[:] = self.q_reset; mujoco.mj_forward(self.m, self.d)

    def set_hurdle(self, x, y, yaw, h, exact=False):
        """exact=True: manager.activate (parks all, places, mj_forward) = the client call. exact=False (search): the same
        manager._place for hurdle_0 (other slots stay parked from __init__) + mj_kinematics -> identical geom poses."""
        ob = self.mgr.config["profiles"]["cand"]["obstacles"][0]
        ob["position"] = [float(x), float(y), TABLE_Z]; ob["yaw"] = float(yaw); ob["bar_height"] = float(h)
        self.d.qpos[:] = self.q_reset
        if exact:
            self.mgr.activate("cand")
        else:
            self.mgr._place("hurdle_0", ob["position"], ob["yaw"], ob["bar_height"])
            self.mgr.active_slots = ("hurdle_0",); self.mgr.profile_name = "cand"
            mujoco.mj_kinematics(self.m, self.d)

    def reset_check(self):
        """the client's two reset numbers (manager defaults)."""
        return float(self.mgr.robot_clearance()), float(self.mgr.object_clearance())

    def series(self, cond, rows, sets, cap=CAP, want_pair=False):
        """per row in `rows` (indices into run arrays): min distance HURDLE <-> each geom set (dict name -> array)."""
        R = self.runs[cond]; m, d = self.m, self.d
        hx = d.geom_xpos[self.HURDLE].copy(); hr = m.geom_rbound[self.HURDLE]
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
                    for o in self.HURDLE:
                        if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[o]) - m.geom_rbound[g] - m.geom_rbound[o] > min(cap, best):
                            continue
                        v = self.mgr.pair_distance(int(o), g, cap, ft)
                        if v < best:
                            best, bp = v, (self.gname(o), self.gname(g))
                out[k][j] = best; pairs[k][j] = bp
        return (out, pairs) if want_pair else out


def hurdle_xy_segments(x, y, yaw):
    u = np.array([math.cos(yaw), math.sin(yaw)]); c = np.array([x, y])
    return c - 0.12 * u, c + 0.12 * u


def seg_point_dist(a, b, P):
    ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t


def write_config(path, profile, x, y, yaw, h, desc):
    cfg = {"version": 1, "table_surface_z": TABLE_Z,
           "profiles": {"clear": {"scene": "any", "description": "no obstacle", "obstacles": []},
                        profile: {"scene": "fruit", "description": desc,
                                  "obstacles": [{"slot": "hurdle_0", "position": [round(float(x), 4), round(float(y), 4), TABLE_Z],
                                                 "yaw": round(float(yaw), 6), "bar_height": round(float(h), 4)}]}}}
    PPO.validate_pick_place_obstacle_config(copy.deepcopy(cfg))
    pathlib.Path(path).write_text(json.dumps(cfg, indent=1))
    return PPO.load_pick_place_obstacle_config(path)


def first_le0(c, cond, rows, geoms, thr=0.0):
    """early-exit scan: index (into rows) of the first row whose min distance HURDLE <-> geoms is <= thr, else None.
    Same distance code as Ctx.series (manager.pair_distance on cached geom poses), cap thr + 1 mm."""
    R = c.runs[cond]; m, d = c.m, c.d
    hx = d.geom_xpos[c.HURDLE].copy(); hr = m.geom_rbound[c.HURDLE]
    ix = np.array([c.pos_in_all[int(g)] for g in geoms], int); gr = m.geom_rbound[c.ALL[ix]]
    cap = thr + 0.001; ft = np.zeros(6)
    for j, i in enumerate(rows):
        P_ = R["xpos"][i][ix]
        gap = np.linalg.norm(P_[:, None, :] - hx[None], axis=2) - gr[:, None] - hr[None]
        near = np.argwhere(gap <= thr)
        if not len(near):
            continue
        for a, o in near:
            g = int(c.ALL[ix[a]])
            d.geom_xpos[g] = R["xpos"][i][ix[a]]; d.geom_xmat[g] = R["xmat"][i][ix[a]]
            if c.mgr.pair_distance(int(c.HURDLE[o]), g, cap, ft) <= thr:
                return j
    return None
