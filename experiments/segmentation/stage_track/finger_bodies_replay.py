"""P5a (2026-11-10): which bodies touch both Panda fingers at the call frames of SafeLIBERO episodes (replay of sim_state_image).
usage: MUJOCO_GL=egl .venv-libero/bin/python experiments/segmentation/stage_track/finger_bodies_replay.py RUN TASK_ID EP[,EP...]"""
import json, pathlib, sys
sys.argv = [sys.argv[0]] + sys.argv[1:]
RUN = pathlib.Path(sys.argv[1]); TID = int(sys.argv[2]); EPS = [int(x) for x in sys.argv[3].split(",")]
BENCH = pathlib.Path("/mnt/dev/AG3S_VLA")
sys.path.insert(0, str(BENCH)); sys.path.insert(0, str(BENCH / "benchmark/runners"))
from benchmark.adapters.libero_panda import client_env
if client_env.needs_reexec():
    client_env.reexec_with_client_environment()
import numpy as np, collections
import libero_pi05_client as base, libero_safe_client as safe
safe.activate_safelibero()
from libero.libero import benchmark
from AG3S.ag3s.experiments.studies.t91b_safelibero_export import FINGER_BODIES
suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
task = suite.get_task(TID); env, _ = base.make_task_env(task, 7, False); inits = suite.get_task_init_states(TID)
for ei in EPS:
    stem = f"safelibero_spatial_t{TID}_e{ei}"
    base.start_episode(env, inits, ei)
    sim = env.sim; model = sim.model; m, d = model._model, sim.data._data
    bn = [model.body_id2name(i) or "" for i in range(model.nbody)]; gb = np.asarray(model.geom_bodyid)
    side = {}
    for s, names in FINGER_BODIES.items():
        for n in names: side[bn.index(n)] = s
    with np.load(RUN / "episodes" / (stem + ".npz")) as z:
        ct = z["chunk_t"].astype(int); q = z["obs_robot0_gripper_qpos"]
    with np.load(RUN / "episodes" / (stem + "_geom.npz")) as g:
        states = g["sim_state_image"]
    both = collections.Counter(); rows = []
    for k, t in enumerate(ct):
        sim.set_state_from_flattened(np.asarray(states[t]).copy()); sim.forward()
        touch = {"left": set(), "right": set()}
        for c in range(d.ncon):
            g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                s = side.get(int(gb[a]))
                if s and int(gb[b]) not in side:
                    touch[s].add(bn[int(gb[b])])
        common = sorted(touch["left"] & touch["right"])
        for x in common: both[x] += 1
        rows.append((k, int(t), round(float((q[t, 0] - q[t, 1]) * 1e3), 1), common))
    print(json.dumps({"stem": stem, "calls_both_fingers_on": dict(both),
                      "per_call": [r for r in rows if r[3]][:40]}), flush=True)
