"""Drive ``benchmark/runners/libero_safe_client.SafeClient`` with a fake env / geometry / websocket (no simulator, no
policy) and print a digest of everything it produces. **Python 3.8** (``.venv-libero``, the client's interpreter).

    .venv-libero/bin/python Baseline/paper_baselines/closed_loop/tests/client_harness.py \
        --source <commit>|worktree --mode vla_only|vla_v11ab [--emit] [--seg] --out <dir>

Digest (JSON on stdout): per request the ordered key list and a SHA-256 per key of the msgpack (openpi wire) bytes, the
SHA-256 of the whole packed request, the ``save()`` record, the SHA-256 of ``<stem>_calls.jsonl`` and of every array of
``<stem>_calls.npz`` (npz zip timestamps differ between writes, the arrays do not), the parsed SL5c argv. With
``--emit`` the packed requests are also written to ``<out>/requests.msgpack`` (length-prefixed) for the round-trip test.

``--loop chunk|step`` (F2, 2026-11-10; worktree client only): instead, run the **official episode loop**
(``libero_pi05_client.run_episode``) on a fake env with a fake method server, without / with ``--filter-every-step``
(``replan_steps = 1`` for the loop exactly as ``main`` does), and print per request its kind, ``ag3s/seq``, observation
index, the key list and the SHA-256 of the policy-relevant request bytes, plus the executed actions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
CLIENT = "benchmark/runners/libero_safe_client.py"
SL5C_ARGV = ["--run-dir", "/x", "--host", "127.0.0.1", "--port", "8000", "--task-suite-name", "safelibero_spatial",
             "--task-ids", "0,1,2,3", "--episode-ids", "0,1,2,3,4", "--seed", "7", "--safety-level", "I"]
H, W = 24, 32
CAMS = ("head", "left_wrist")


def load_client(source):
    path = ROOT / CLIENT
    if source == "worktree":
        src = path.read_text()
    else:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", "{}:{}".format(source, CLIENT)],
                                      universal_newlines=True)
    spec = importlib.util.spec_from_loader("libero_safe_client_" + source.replace("-", "_")[:12], loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


# --- fake scene (two SafeLIBERO obstacle objects, one active, one parked; one plain object) ----------------------
def fake_scene():
    """Returns (env, ground truth dict). Geom layout as SafeLIBERO: per object a visual mesh (contype 0) + boxes."""
    bodies = ["world", "robot0_base", "moka_pot_obstacle_1_main", "milk_obstacle_1_main", "akita_black_bowl_1_main"]
    joints = ["robot0_joint1", "moka_pot_obstacle_1_joint0", "milk_obstacle_1_joint0", "akita_black_bowl_1_joint0"]
    body_pos = {2: [0.10, 0.05, 0.95], 3: [0.0, -10.0, 0.19], 4: [0.0, 0.2, 0.90]}
    geoms = [  # (name, body, type, size, offset, contype)
        ("robot0_link0", 1, 6, [0.05, 0.05, 0.05], [0, 0, 0], 1),
        ("moka_pot_obstacle_1_g0", 2, 7, [0.063, 0.098, 0.117], [0, 0, 0.01], 0),
        ("moka_pot_obstacle_1_g1", 2, 6, [0.04, 0.04, 0.1065], [0, 0, 0], 1),
        ("moka_pot_obstacle_1_g2", 2, 6, [0.0081, 0.04, 0.1065], [0, 0.042, 0], 1),
        ("milk_obstacle_1_g0", 3, 7, [0.056, 0.055, 0.157], [0, 0, -0.02], 0),
        ("milk_obstacle_1_g1", 3, 6, [0.0525, 0.0525, 0.1095], [0, 0, -0.035], 1),
        ("akita_black_bowl_1_g0", 4, 6, [0.05, 0.05, 0.02], [0, 0, 0], 1)]
    m = types.SimpleNamespace()
    m.joint_names = tuple(joints)
    m.body_names = tuple(bodies)
    m.body_name2id = lambda n: bodies.index(n)
    m.body_id2name = lambda i: bodies[i]
    m.ngeom = len(geoms)
    m.geom_id2name = lambda g: geoms[g][0]
    m.geom_bodyid = np.array([g[1] for g in geoms], np.int32)
    m.geom_type = np.array([g[2] for g in geoms], np.int32)
    m.geom_size = np.array([g[3] for g in geoms], np.float64)
    m.geom_contype = np.array([g[5] for g in geoms], np.int32)
    m.geom_conaffinity = np.array([g[5] for g in geoms], np.int32)
    m.geom_group = np.array([0 if g[5] else 1 for g in geoms], np.int32)
    d = types.SimpleNamespace()
    d.body_xpos = np.array([body_pos.get(i, [0.0, 0.0, 0.0]) for i in range(len(bodies))], np.float64)
    c, s = np.cos(0.3), np.sin(0.3)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    d.body_xquat = np.array([[1, 0, 0, 0]] * len(bodies), np.float64)
    d.body_xquat[2] = [np.cos(0.15), 0, 0, np.sin(0.15)]
    d.geom_xpos = np.array([d.body_xpos[g[1]] + (Rz if g[1] == 2 else np.eye(3)) @ np.array(g[4], float) for g in geoms])
    d.geom_xmat = np.array([(Rz if g[1] == 2 else np.eye(3)).reshape(9) for g in geoms])
    env = types.SimpleNamespace(sim=types.SimpleNamespace(model=m, data=d))
    truth = {"active": ["moka_pot_obstacle_1"], "moka_geom_ids": [1, 2, 3], "moka_collision_ids": [2, 3],
             "geom_xpos": d.geom_xpos.tolist(), "geom_xmat": d.geom_xmat.tolist(), "geom_size": m.geom_size.tolist()}
    return env, truth


T_WORLD_BASE = np.array([[1, 0, 0, -0.66], [0, 1, 0, 0.0], [0, 0, 1, 0.912], [0, 0, 0, 1.0]])


def frame(i, seg):
    rng = np.random.RandomState(1000 + i)
    out = {}
    for k, cam in enumerate(CAMS):
        f = {"depth_m": (0.5 + rng.rand(H, W)).astype(np.float32),
             "K": np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]]),
             "T_base_cam": np.eye(4) + 0.01 * k, "T_world_base": T_WORLD_BASE.copy(),
             "robot_state": np.concatenate([0.1 * np.arange(7) + 0.001 * i, [0.03, -0.03]]), "stamp": 0.05 * i}
        if seg:
            sg = np.full((H, W), -1, np.int16)
            sg[4:12, 6:14] = 2  # moka collision box
            sg[14:20, 20:28] = 6  # bowl
            f["segmentation_geom"] = sg
        out[cam] = f
    return out


class FakeGeometry(object):
    def __init__(self, seg):
        sys.path.insert(0, str(ROOT))
        from benchmark.adapters.libero_panda import wire_pack

        self.camera = types.SimpleNamespace(STAMP_MODE="sim_frozen")
        self.wire_pack = wire_pack
        self.cameras = list(CAMS)
        self.frames = []
        self.seg = seg

    def observe(self, env, obs):
        self.frames.append(frame(len(self.frames), self.seg))


class FakeWS(object):
    def __init__(self, mode, packb):
        self.mode, self.packb, self.packed, self.n = mode, packb, [], 0

    def infer(self, req):
        self.packed.append([(k, self.packb(v)) for k, v in req.items()] + [("__all__", self.packb(req))])
        rng = np.random.RandomState(7 + self.n)
        self.n += 1
        resp = {"actions": rng.uniform(-1, 1, (10, 7)).astype(np.float32), "panda/mode": self.mode,
                "policy_seed": 700000}
        if self.mode == "vla_v11ab":
            resp["safe"] = bool(self.n % 3)
        return resp

    def get_server_metadata(self):
        return {}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def drive(mod, mode, emit, seg, out_dir):
    from openpi_client import msgpack_numpy

    clock = {"t": 0.0}

    def perf_counter():
        clock["t"] += 0.125
        return clock["t"]

    mod.time = types.SimpleNamespace(perf_counter=perf_counter)  # deterministic t_client_ms
    env, truth = fake_scene()
    ws = FakeWS(mode, msgpack_numpy.packb)
    kw = {"emit_privileged": True} if emit else {}
    client = mod.SafeClient(ws, 7, 0, 0, 5, **kw)
    client.collision = types.SimpleNamespace(obstacle="moka_pot_obstacle_1")
    geom = FakeGeometry(seg)
    observe = client.observer(geom)
    rng = np.random.RandomState(3)
    for i in range(26):
        q = rng.randn(4)
        observe(env, {"robot0_eef_pos": rng.randn(3), "robot0_eef_quat": q / np.linalg.norm(q)})
        if i % 5 == 0:
            element = {"observation/image": rng.randint(0, 255, (8, 8, 3)).astype(np.uint8),
                       "observation/wrist_image": rng.randint(0, 255, (8, 8, 3)).astype(np.uint8),
                       "observation/state": rng.randn(8), "prompt": "pick up the black bowl"}
            client.infer(element)
    ep = pathlib.Path(tempfile.mkdtemp(dir=str(out_dir)))
    saved = client.save(ep, "safelibero_spatial_t0_e0")
    with np.load(str(ep / "safelibero_spatial_t0_e0_calls.npz")) as z:
        arrays = {k: [str(z[k].dtype), list(z[k].shape), sha(np.ascontiguousarray(z[k]).tobytes())] for k in z.files}
    if emit:
        with open(str(pathlib.Path(out_dir) / "requests.msgpack"), "wb") as f:
            for req in ws.packed:
                b = dict(req)["__all__"]
                f.write(struct.pack("<Q", len(b)))
                f.write(b)
        (pathlib.Path(out_dir) / "truth.json").write_text(json.dumps(truth))
    return {"requests": [{"keys": [k for k, _ in r if k != "__all__"], "sha": {k: sha(b) for k, b in r}}
                         for r in ws.packed],
            "save": saved, "calls_jsonl_sha": sha((ep / "safelibero_spatial_t0_e0_calls.jsonl").read_bytes()),
            "calls_npz": arrays}


# --- F2: the official loop with and without --filter-every-step ------------------------------------------------------
class LoopWS(object):
    """Fake method server: chunk = f(seq) on a policy request, row k of it + 1e-3 (a 'filtered' row) on a filter step."""

    def __init__(self, packb, step):
        self.packb, self.step, self.log = packb, step, []

    @staticmethod
    def chunk(seq):
        return np.random.RandomState(100 + int(seq)).uniform(-1, 1, (10, 7)).astype(np.float32)

    def infer(self, req):
        k = req.get("ag3s/filter_step")
        seq = int(req["ag3s/seq"])
        policy_keys = [key for key in req if not key.startswith("ag3s/filter_step")]
        self.log.append({"kind": "filter" if (k or 0) > 0 else "policy", "k": k, "seq": seq,
                         "reset": bool(req.get("ag3s/reset")), "policy_seed": req.get("ag3s/policy_seed"),
                         "keys": list(req), "sha_policy_keys": sha(self.packb({key: req[key] for key in policy_keys}))})
        c = self.chunk(seq)
        if k is None or k == 0:
            resp = {"actions": c, "panda/mode": "method:test", "policy_seed": 700000}
            if self.step:
                resp["actions"] = c.copy()
                resp["actions"][0] += 1e-3
            return resp
        return {"actions": c[k:k + 1] + 1e-3, "panda/mode": "method:test"}

    def get_server_metadata(self):
        return {"ag3s_filter_rate": "step"} if self.step else {}


class LoopEnv(object):
    def __init__(self):
        self.n = 0
        self.actions = []
        self.sim = types.SimpleNamespace(model=None, data=None)

    def obs(self):
        rng = np.random.RandomState(self.n)
        q = rng.randn(4)
        return {"agentview_image": rng.randint(0, 255, (16, 16, 3)).astype(np.uint8),
                "robot0_eye_in_hand_image": rng.randint(0, 255, (16, 16, 3)).astype(np.uint8),
                "robot0_eef_pos": rng.randn(3), "robot0_eef_quat": q / np.linalg.norm(q),
                "robot0_joint_pos": rng.randn(7), "robot0_gripper_qpos": rng.randn(2)}

    def reset(self):
        self.n = 0

    def set_init_state(self, s):
        return self.obs()

    def step(self, action):
        self.n += 1
        self.actions.append([float(a) for a in action])
        return self.obs(), 0.0, False, {}


def drive_loop(mod, step):
    import argparse as _ap

    from openpi_client import msgpack_numpy

    env = LoopEnv()
    ws = LoopWS(msgpack_numpy.packb, step)
    client = mod.SafeClient(ws, 7, 0, 0, 5, **({"filter_every_step": True} if step else {}))
    client.collision = None
    geom = FakeGeometry(False)
    args = _ap.Namespace(num_steps_wait=3, replan_steps=5, resize_size=8)
    loop_args = args
    if step:  # = libero_safe_client.main
        loop_args = _ap.Namespace(**vars(args))
        loop_args.replan_steps = 1
    summary, rec, _ = mod.base.run_episode(env, [None], 0, "pick up the black bowl", client, loop_args, 17,
                                           observer=client.observer(geom))
    saved = client.save(pathlib.Path(tempfile.mkdtemp()), "safelibero_spatial_t0_e0")
    return {"requests": ws.log, "executed": env.actions, "summary_status": summary["status"],
            "env_steps": summary["env_steps_including_wait"], "save": saved,
            "client_arrays": {k: list(np.asarray(v).shape) for k, v in client.arrays.items()},
            "actions_executed": [np.asarray(a).round(6).tolist() for a in client.arrays["actions_executed"]]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="git commit of the client, or 'worktree'")
    ap.add_argument("--mode", default="vla_only")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--seg", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loop", choices=("chunk", "step"), default=None, help="F2: drive the official loop (worktree)")
    a = ap.parse_args(argv)
    mod = load_client(a.source)
    if a.loop is not None:
        print(json.dumps(drive_loop(mod, a.loop == "step"), sort_keys=True, default=str))
        return
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = drive(mod, a.mode, a.emit, a.seg, out)
    res["parsed_sl5c_argv"] = vars(mod.build_parser().parse_args(SL5C_ARGV))
    res["python"] = sys.version.split()[0]
    print(json.dumps(res, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
