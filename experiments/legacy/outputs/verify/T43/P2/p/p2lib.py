"""T43 P2 shared helpers (verifier, CPU only). overbar_0 (T42-2 slot) placed ACROSS the crate, inside the handles.

Code      : live pi05_TO_hybrid tree at fce4af8 (clean; md5 of the client / manager / distance / XML files in
            raw/code_md5_start.txt and raw/code_md5_end.txt).  Distances = PickPlaceObstacleManager.pair_distance
            (certified mj_geomDistance + exact GJK fallback, T40 H; overbar geoms are capsule + 2 cylinders).
Reset     : the client's pi05_infer.main() state at the obstacle reset check (capture as T43 P flib.capture_reset, but with
            the fce4af8 client; the manager's activate() is replaced by park-only during the capture).
Runs      : T40 episodes 1807/1968/1982/1995/1967 with their T40 seeds = T39 runs E0 + E3b (as T43 P; T40 A windows);
            T42-1 apple episodes 1925 s19251 (T39 runs, left) and 1834 s18341 (T42-1 runs, right); grasp_t / place_t from
            outputs/verify/T42/1/summary.json (T42-1 rule), lift_t by the T40 A rule (first control row after the last row
            <= grasp_t whose apple centre z < z0 + 2 mm).
Geom sets : as T43 P (ROBOT = every collision geom under 'base'; GRIP = the used arm's EE_BODY + 2 fingers; APPLE; FRUITS;
            CRATE); OB = overbar_0 collision geoms (bar capsule + 2 leg cylinders)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import copy, json, math, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
LIVE = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid'))
sys.path.insert(0, str(LIVE / "rby1_manipulation" / "src"))
import numpy as np, mujoco  # noqa: E402
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402
assert PPO.__file__.startswith(str(LIVE)), PPO.__file__
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import gjk  # noqa: E402  (T40 A independent numpy GJK)
from scipy.optimize import minimize  # noqa: E402

ROOT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P2'))
RAW = ROOT / "raw"; CAPDIR = RAW / "capture"
T39 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs'))
T42R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/1/runs'))
T40A = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40A.verify.json')))["numbers"]["episodes"]
T42S = json.load(open(_ag3s_workspace_path('outputs/verify/T42/1/summary.json')))["runs"]
TAGS = ["1807_18071", "1968_19681", "1982_19822", "1995_19952", "1967_19672", "1925_19251", "1834_18341"]
TABLE_Z = 0.82
SLOT = "overbar_0"
RED = [0.85, 0.10, 0.10, 1.0]
LEG_R = PPO.FRAME_LEG_RADIUS[SLOT]


def run_dir(cond, ep, seed):
    for p in (T39 / cond / f"{cond}_ep{ep}_s{seed}", T42R / cond / "noobs" / f"{cond}_ep{ep}_s{seed}"):
        if (p / "frames.jsonl").exists():
            return p
    raise FileNotFoundError((cond, ep, seed))


def _client():
    sys.path.insert(0, str(LIVE / "rby1_bringup"))
    import pi05_infer as P
    assert P.__file__.startswith(str(LIVE)), P.__file__
    return P


class _Reached(Exception):
    pass


def capture_reset(ep):
    P = _client(); cap = {}

    class Park(PPO.PickPlaceObstacleManager):
        def activate(self, profile_name):
            self.park_all(); self.reset_metrics(); self.profile_name = profile_name

    def stub(remote):
        import inspect
        cap.update(inspect.currentframe().f_back.f_locals); raise _Reached

    cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "obstacles": []},
           "cap": {"scene": "fruit", "obstacles": [{"slot": SLOT, "position": [0.80, -0.30, TABLE_Z], "yaw": 0.0}]}}}
    CAPDIR.mkdir(parents=True, exist_ok=True)
    tmp = CAPDIR / f"cap_cfg_{os.getpid()}.json"; tmp.write_text(json.dumps(cfg))
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
    return cap["m"], cap["d"], cap["model_xml"], cap.get("randomized_episode")


def save_capture(tag):
    ep = int(tag.split("_")[0])
    m, d, xml, rec = capture_reset(ep)
    mujoco.mj_saveModel(m, str(CAPDIR / f"{tag}.mjb"), None)
    np.savez(CAPDIR / f"{tag}_reset.npz", qpos=d.qpos, qvel=d.qvel, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat,
             ctrl=d.ctrl, act=d.act, model_xml=np.asarray(xml))
    (CAPDIR / f"{tag}_episode.json").write_text(json.dumps(rec, default=str))
    return xml


def windows(cond, ep, seed, Q, t):
    tag = f"{ep}_{seed}"
    if tag in T40A:
        w = dict(T40A[tag]["windows"][cond]); w["source"] = "T40A"
        return w
    k = f"{cond}_ep{ep}_s{seed}"; s = T42S.get(k) or T42S.get(f"T39/{k}")
    g, p = s["grasp_t"], s["place_t"]
    z = Q[:, 40]; z0 = z[0]
    ok = np.where((t >= 0) & (t <= g) & (z < z0 + 0.002))[0]
    lift = int(t[ok[-1] + 1])
    return {"lift_t": lift, "grasp_t": g, "place_t": p, "source": "T42-1 summary grasp/place + T40A lift rule"}


class Ctx:
    def __init__(self, tag):
        self.tag = tag; ep, seed = (int(x) for x in tag.split("_")); self.ep, self.seed = ep, seed
        self.m = mujoco.MjModel.from_binary_path(str(CAPDIR / f"{tag}.mjb")); self.d = mujoco.MjData(self.m)
        z = np.load(CAPDIR / f"{tag}_reset.npz"); self.model_xml = str(z["model_xml"])
        self.episode = json.loads((CAPDIR / f"{tag}_episode.json").read_text())
        self.arm = self.episode.get("used_arm", "left")
        self.d.qpos[:] = z["qpos"]; self.d.qvel[:] = z["qvel"]; self.d.mocap_pos[:] = z["mocap_pos"]; self.d.mocap_quat[:] = z["mocap_quat"]
        self.d.ctrl[:] = z["ctrl"]
        if self.m.na:
            self.d.act[:] = z["act"]
        mujoco.mj_forward(self.m, self.d)
        m = self.m
        self.q_reset = self.d.qpos.copy()
        self.cfg = {"version": 1, "table_surface_z": TABLE_Z, "profiles": {"clear": {"scene": "any", "obstacles": []},
                    "cand": {"scene": "fruit", "obstacles": [{"slot": SLOT, "position": [0.80, -0.30, TABLE_Z], "yaw": 0.0}]}}}
        self.mgr = PPO.PickPlaceObstacleManager(m, self.d, self.cfg)
        self.OB = np.array(sorted(self.mgr._collision_geoms[SLOT]), int)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        gb = lambda names: np.array([g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in {B(n) for n in names}], int)

        def desc(b, root):
            while b > 0:
                if b == root:
                    return True
                b = int(m.body_parentid[b])
            return False
        S = "L" if self.arm == "left" else "R"; s = S.lower()
        self.A_GRIP = gb([f"EE_BODY_{S}", f"ee_finger_{s}1", f"ee_finger_{s}2"])
        self.B_APPLE = gb(["apple"])
        base = B("base")
        self.ROBOT_ALL = np.array([g for g in range(m.ngeom) if coll[g] and desc(int(m.geom_bodyid[g]), base)], int)
        self.FRUITS = gb(["banana", "orange", "pear"])
        self.CRATE = gb(["crate"])
        self.ARMS = np.array(sorted(self.mgr.robot_geom_ids), int)
        self.ALL = np.unique(np.concatenate([self.ROBOT_ALL, self.B_APPLE, self.FRUITS, self.CRATE]))
        self.pos_in_all = {int(g): i for i, g in enumerate(self.ALL)}
        self.gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
        self.runs = {}
        for cond in ("E0", "E3b"):
            rd = run_dir(cond, ep, seed)
            rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
            ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
            Q = np.array([c["qpos"] for c in ctrl]); t = np.array([c["t_step"] for c in ctrl])
            Q = np.vstack([np.array(plan[0]["qpos"])[None], Q]); t = np.r_[-1, t]
            w = windows(cond, ep, seed, Q, t)
            xp, xm = self.pose_cache(Q)
            self.runs[cond] = {"run_dir": str(rd), "t": t, "Q": Q, "xpos": xp, "xmat": xm, **w}
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

    def set_overbar(self, x, y, yaw, h, L, r=0.012, exact=False):
        ob = self.mgr.config["profiles"]["cand"]["obstacles"][0]
        ob.update({"position": [float(x), float(y), TABLE_Z], "yaw": float(yaw), "bar_height": float(h), "length": float(L),
                   "radius": float(r)})
        self.d.qpos[:] = self.q_reset
        if exact:
            self.mgr.activate("cand")
        else:
            self.mgr._place(SLOT, ob["position"], ob["yaw"], sizes={"bar_height": float(h), "length": float(L), "radius": float(r)})
            self.mgr.active_slots = (SLOT,); self.mgr.profile_name = "cand"
            mujoco.mj_kinematics(self.m, self.d)

    def reset_check(self):
        return float(self.mgr.robot_clearance()), float(self.mgr.object_clearance())


class WS:
    """kinematic max TCP z of the used arm (copy of T43 P fsearch.WS, arm-generic)."""
    def __init__(self, c):
        self.c = c; m = c.m; s = c.arm
        self.S = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_ee")
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_arm_{i}") for i in range(7)]
        self.ARM = [int(m.jnt_qposadr[j]) for j in jid]
        self.LO, self.HI = m.jnt_range[jid, 0], m.jnt_range[jid, 1]

    def fk(self, qb, x):
        q = qb.copy(); q[self.ARM] = x; d = self.c.d; d.qpos[:] = q; mujoco.mj_kinematics(self.c.m, d)
        return d.site_xpos[self.S].copy(), d.site_xmat[self.S].reshape(3, 3).copy()

    def axis(self, Q):
        M = np.mean([self.fk(q, q[self.ARM])[1] for q in Q], axis=0); k = int(np.argmax(np.abs(M[2, :])))
        return k, M[:, k] / np.linalg.norm(M[:, k])

    def offset(self, Q):
        c = self.c; off = []
        for q in Q:
            p, _ = self.fk(q, q[self.ARM]); s = gjk.make_support(c.m, c.d, int(c.B_APPLE[0]))
            off.append(p[2] - s(np.array([0, 0, -1.0]))[2])
        return float(np.median(off))

    def max_tcp_z(self, xy, Q, ang=15.0, tol=0.005):
        k, a = self.axis(Q); cmin = math.cos(math.radians(ang))
        idx = list(np.linspace(0, len(Q) - 1, 8).astype(int))
        dxy = [np.linalg.norm(self.fk(q, q[self.ARM])[0][:2] - xy) for q in Q]; idx.append(int(np.argmin(dxy)))
        best = None
        for i in idx:
            qb = Q[i].copy()
            cons = [{"type": "ineq", "fun": lambda x: tol ** 2 - np.sum((self.fk(qb, x)[0][:2] - xy) ** 2)},
                    {"type": "ineq", "fun": lambda x: self.fk(qb, x)[1][:, k] @ a - cmin}]
            r = minimize(lambda x: -self.fk(qb, x)[0][2], qb[self.ARM], method="SLSQP", bounds=list(zip(self.LO, self.HI)),
                         constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
            p, Rm = self.fk(qb, r.x)
            if np.linalg.norm(p[:2] - xy) <= tol + 1e-4 and Rm[:, k] @ a >= cmin - 1e-4 and (best is None or p[2] > best):
                best = float(p[2])
        return None if best is None else best - TABLE_Z


def write_config(path, profile, x, y, yaw, h, L, r, desc, rgba=RED):
    ob = {"slot": SLOT, "position": [round(float(x), 4), round(float(y), 4), TABLE_Z], "yaw": round(float(yaw), 6),
          "bar_height": round(float(h), 4), "length": round(float(L), 4), "radius": round(float(r), 4)}
    if rgba is not None:
        ob["rgba"] = list(rgba)
    cfg = {"version": 1, "table_surface_z": TABLE_Z,
           "profiles": {"clear": {"scene": "any", "description": "no obstacle", "obstacles": []},
                        profile: {"scene": "fruit", "description": desc, "obstacles": [ob]}}}
    PPO.validate_pick_place_obstacle_config(copy.deepcopy(cfg))
    pathlib.Path(path).write_text(json.dumps(cfg, indent=1))
    return PPO.load_pick_place_obstacle_config(path)
