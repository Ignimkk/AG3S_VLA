"""T63b — does the serving path feed cuRobo B0 the same world as the offline T63 path? (CUDA)

Two fresh `CuroboChunkSolver`s (world `b0`, timing `exact`), one after the other, the same recorded
chunks in the same order:

| solver | world load | what it stands for |
|---|---|---|
| `offline` | `ComposedSceneCollision.load_chunk(npz)` — float16 record arrays | T63 offline B0 |
| `serving` | `load_layers(serving-style layers)` — the record's grids as **float32 CUDA tensors** (what `DeviceEsdfField.values_device` holds; the recorder rounded exactly those values to float16), rounded on the device | `CuroboB0Backend.solve_chunk` |

Same spec (T63 `robot_spec.json`), same `q_now`, same finger value (`finger_path_plan[:, 0]`), same
torso pose. Pass = every channel's voxel buffers byte-identical after each load, and the cuRobo
trajectories bit-identical chunk by chunk.

    cd $SNAP_CR0 && PYTHONPATH=$SNAP_CR0 $AG3S_PROJECT_ROOT/.venv-openpi-live/bin/python \\
        -m AG3S.trajopt.experiments.t63b_live_parity --inputs <T63 run>/impl/T63/inputs \\
        --episode 126 --n 6 --out <dir>/live_parity.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--episode", type=int, default=126)
    ap.add_argument("--n", type=int, default=6, help="first N chunks of the episode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch

    from AG3S.trajopt.curobo_v2 import spec as S
    from AG3S.trajopt.curobo_v2 import world as W
    from AG3S.trajopt.curobo_v2.solver import CuroboChunkSolver

    inputs = pathlib.Path(args.inputs)
    spec = S.load_spec(inputs / "robot_spec.json")
    ep = np.load(inputs / f"episode_{args.episode}.npz", allow_pickle=False)
    chunks = []
    for k in range(min(args.n, len(ep["chunk_file"]))):
        with np.load(str(ep["chunk_file"][k]), allow_pickle=True) as z:
            z = {key: z[key] for key in z.files if key.startswith("esdf") or key == "reference_chunk"}
        chunks.append((k, z, S.q_now_cspace(ep["q_now"][k], spec),
                       np.asarray(ep["T_base_torso5"][k], np.float64),
                       S.finger_values(spec, ep["finger_path_plan"][k], 0)))

    def run(mode):
        """One solver at a time (T63: several captured solvers alive at once crashed cuRobo)."""
        t0 = time.perf_counter()
        s = CuroboChunkSolver(spec, world="b0", timing="exact")
        build = time.perf_counter() - t0
        out = []
        for k, z, q14, T, fingers in chunks:
            if mode == "offline":
                info = s.set_world(z, T)
            else:
                live = {}
                for name in ("coarse", "main", "free"):
                    if f"{W.LAYER_PREFIXES[name]}_distance" not in z:
                        continue
                    lay = W.layer_arrays(z, name)
                    live[name] = {"distance": torch.as_tensor(np.asarray(lay["distance"], np.float32),
                                                              device="cuda"),
                                  "origin": lay["origin"], "voxel_size": lay["voxel_size"]}
                info = s.scene.load_layers(live, T)
            bufs = {f"{ch}.{a}": getattr(s.scene.channels[ch].data.voxels, a).detach().cpu().clone()
                    for ch in s.scene.channels for a in ("features", "params", "dims", "inv_pose",
                                                         "enable")}
            bufs.update({"box_min_coarse": s.scene.box_min_coarse.detach().cpu().clone(),
                         **{f"box.{i}": v.detach().cpu().clone()
                            for i, v in enumerate(s.scene.boxes["box"])}})
            s.set_fingers(fingers)
            r = s.solve(q14, np.asarray(z["reference_chunk"], np.float64))
            out.append({"info": info, "bufs": bufs, "res": r})
        del s
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return build, out

    build_off, off = run("offline")
    build_srv, srv = run("serving")
    build_s = [build_off, build_srv]
    rows = []
    for (k, *_), a_, b_ in zip(chunks, off, srv):
        same_world = a_["info"] == b_["info"] and all(
            torch.equal(a_["bufs"][key], b_["bufs"][key]) for key in a_["bufs"])
        a, b = a_["res"]["cspace"], b_["res"]["cspace"]
        rows.append({"chunk": k, "box_layer": a_["info"].get("box_layer"),
                     "world_identical": bool(same_world),
                     "trajectory_bit_identical": bool(a is not None and b is not None
                                                      and a.tobytes() == b.tobytes()),
                     "max_abs_diff_rad": (None if a is None or b is None
                                          else float(np.abs(a - b).max())),
                     "success": [a_["res"]["success"], b_["res"]["success"]],
                     "wall_ms": [a_["res"]["wall_ms"], b_["res"]["wall_ms"]],
                     "cold": a_["res"]["cold"]})
        print(json.dumps(rows[-1]), flush=True)
    res = {"tool": "AG3S/trajopt/experiments/t63b_live_parity.py", "inputs": str(inputs),
           "episode": args.episode, "build_s": build_s, "rows": rows,
           "pass": all(r["world_identical"] and r["trajectory_bit_identical"] for r in rows),
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print("PASS" if res["pass"] else "FAIL", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
