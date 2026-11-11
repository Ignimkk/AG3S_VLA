"""SL-GTTO — the client's GT code image on a real LIBERO / SafeLIBERO / LIBERO-Safety scene (CPU, no policy, no GPU).

Runs inside the benchmark's own interpreter (``.venv-libero`` Python 3.8 for LIBERO / SafeLIBERO, ``.venv-libero-safety``
Python 3.10 for LIBERO-Safety); ``tests/benchmark/test_sl_gtto_gt_perception.py`` starts it and reads the JSON it
prints. One env of the task is built **exactly as the client builds it** (``libero_pi05_client.make_task_env`` with the
geometry capture), init state 0 is reset (``start_episode``), the wait steps run with the official dummy action, and
the observation goes through ``camera.capture(..., segmentation=True)`` and the client's ``oracle_coder_for`` /
``OracleCoder.request_keys`` (the request keys the server gets). Checked:

* the code image of every camera has the depth image's shape (uint8), and code 1 == the gt_oracle mask
  (``np.isin(segmentation_geom, task_target.instance_geom_ids(target))``) pixel for pixel;
* robot geoms are 0, the table is 2, the other BDDL objects are >= 3 (one code each);
* **orientation / pixel grid**: the target body's world position, projected with the request's ``K`` / ``T_base_cam``,
  lands on a code-1 pixel; the code-1 pixels back-projected with the request's depth lie at the target (median
  distance to the target body); the same test on the row-flipped code image (the raw OpenGL order) fails.

    MUJOCO_GL=osmesa .venv-libero/bin/python tests/benchmark/libero_gtto_scene_checks.py --bench safelibero --task 0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/runners"))


def _activate(bench):
    if bench == "safelibero":
        import libero_safe_client as sc

        return sc.activate_safelibero()
    if bench == "libero_safety":
        import libero_safety_client as lsc

        # a git worktree has no LIBERO-Safety checkout of its own: the interpreter's editable install points at the
        # main checkout's copy, which this variable names (test-only; the client itself is unchanged)
        if os.environ.get("GTTO_LIBERO_SAFETY_ROOT"):
            lsc.LIBERO_SAFETY_ROOT = pathlib.Path(os.environ["GTTO_LIBERO_SAFETY_ROOT"])
        return lsc.activate_libero_safety()
    import libero_pi05_client as base

    return {"config": str(base.require_libero_config())}


def _suite(bench, suite_name, level, safety_level):
    from libero.libero import benchmark

    if bench == "safelibero":
        return benchmark.get_benchmark_dict()[suite_name](safety_level=safety_level)
    if bench == "libero_safety":
        import libero_safety_client as lsc

        return lsc.LevelSuite(benchmark.get_benchmark_dict()[suite_name](), level)
    return benchmark.get_benchmark_dict()[suite_name]()


def _pixel(points_base, K, T_base_cam):
    T = np.linalg.inv(np.asarray(T_base_cam, np.float64))
    pc = np.asarray(points_base, np.float64) @ T[:3, :3].T + T[:3, 3]
    uv = pc[:, :2] / pc[:, 2:3] @ np.asarray(K)[:2, :2].T + np.asarray(K)[:2, 2]
    return uv, pc[:, 2]


def _backproject(depth, K, T_base_cam, mask):
    v, u = np.nonzero(mask & (np.asarray(depth) > 0))
    z = np.asarray(depth, np.float64)[v, u]
    cam = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    T = np.asarray(T_base_cam, np.float64)
    return cam @ T[:3, :3].T + T[:3, 3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("libero", "safelibero", "libero_safety"), required=True)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--safety-level", default="I")
    ap.add_argument("--wait", type=int, default=None)
    args = ap.parse_args(argv)
    suite_name = args.suite or {"libero": "libero_spatial", "safelibero": "safelibero_spatial",
                                "libero_safety": "affordance"}[args.bench]
    env_rec = _activate(args.bench)
    import libero_pi05_client as base
    import libero_safe_client as sc

    camera = base._geometry_adapter()[0]
    oc, tt = sc._oracle_modules()
    suite = _suite(args.bench, suite_name, args.level, args.safety_level)
    task = suite.get_task(args.task)
    coder, rec = sc.oracle_coder_for(task)
    env, _ = base.make_task_env(task, 7, True)
    obs = base.start_episode(env, suite.get_task_init_states(args.task), 0)
    wait = args.wait if args.wait is not None else {"safelibero": 20, "libero_safety": 10}.get(args.bench, 10)
    for _ in range(wait):
        obs, _, _, _ = env.step(base.LIBERO_DUMMY_ACTION)
    frames = camera.capture(env, obs, segmentation=True)
    model = env.sim.model
    coder.prepare(model)
    keys = coder.request_keys(frames)
    body_names = [model.body_id2name(i) or "" for i in range(model.nbody)]
    gb = np.asarray(model.geom_bodyid)
    target_geoms = tt.instance_geom_ids(coder.target, gb, body_names)
    target_bodies = tt.instance_body_ids(coder.target, body_names)
    main_body = [i for i in target_bodies if body_names[i] == coder.target + "_main"] or target_bodies[:1]
    T_world_base = np.asarray(frames[list(frames)[0]]["T_world_base"], np.float64)
    T_base_world = np.linalg.inv(T_world_base)
    target_world = np.asarray(env.sim.data.body_xpos[main_body[0]], np.float64)
    target_base = T_base_world[:3, :3] @ target_world + T_base_world[:3, 3]
    out = {"bench": args.bench, "suite": suite_name, "task": args.task, "task_name": task.name,
           "python": sys.version.split()[0], "env": env_rec, "target": rec, "legend": coder.legend,
           "table": coder.info, "cameras": {}}
    robot_geoms = [g for g in range(model.ngeom) if oc.is_robot_body(body_names[int(gb[g])])
                   or oc.is_robot_body(body_names[int(model.body_rootid[int(gb[g])])])]
    table_geoms = [g for g in range(model.ngeom) if body_names[int(gb[g])] == "table"]
    other_codes = {}
    for cam, frame in frames.items():
        code = keys["ag3s/oracle_seg/" + cam]
        seg = np.asarray(frame["segmentation_geom"])
        depth = np.asarray(frame["depth_m"], np.float64)
        K, T = np.asarray(frame["K"]), np.asarray(frame["T_base_cam"])
        gt_mask = np.isin(seg, np.asarray(target_geoms, np.int64))
        c = {"shape": list(code.shape), "depth_shape": list(depth.shape), "dtype": str(code.dtype),
             "n_code1": int((code == 1).sum()), "equals_gt_oracle_mask": bool(np.array_equal(code == 1, gt_mask)),
             "robot_pixels": int(np.isin(seg, robot_geoms).sum()),
             "robot_pixels_code0": bool(np.all(code[np.isin(seg, robot_geoms)] == 0)),
             "table_pixels_code2": bool(np.all(code[np.isin(seg, table_geoms)] == 2)) if table_geoms else None,
             "codes_present": sorted(int(x) for x in np.unique(code))}
        uv, z = _pixel(target_base[None], K, T)
        u, v = int(round(float(uv[0, 0]))), int(round(float(uv[0, 1])))
        inside = 0 <= u < code.shape[1] and 0 <= v < code.shape[0]
        c["target_projection"] = {"u": u, "v": v, "z_m": float(z[0]), "inside": bool(inside),
                                  "code_at_pixel": int(code[v, u]) if inside else None,
                                  "code_at_flipped_pixel": int(code[code.shape[0] - 1 - v, u]) if inside else None}
        for flip in (False, True):
            lab = code[::-1] if flip else code
            pts = _backproject(depth, K, T, lab == 1)
            d = np.linalg.norm(pts - target_base, axis=1) if len(pts) else np.zeros(0)
            c["backprojected_flipped" if flip else "backprojected"] = {
                "n": int(len(pts)), "median_dist_to_target_body_m": float(np.median(d)) if len(d) else None,
                "frac_within_0.10m": float(np.mean(d < 0.10)) if len(d) else None}
        for name, cd in (coder.legend or {}).items():
            if cd >= 3:
                other_codes.setdefault(name, 0)
                other_codes[name] += int((code == cd).sum())
        out["cameras"][cam] = c
    out["other_object_pixels"] = other_codes
    out["meta"] = keys["ag3s/oracle_seg_meta"]
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    raise SystemExit(main())
